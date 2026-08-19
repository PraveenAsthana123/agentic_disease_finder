"""
ASAH1 Epilepsy — Farber Disease (Acid Ceramidase Deficiency / Farber Lipogranulomatosis)
==========================================================================================
40-patient cohort · ASAH1 (8p22) · Autosomal Recessive (AR) biallelic LOF
ASAH1 encodes Acid Ceramidase (AC / N-acylsphingosine amidohydrolase) — 395 aa, ~55 kDa.
ASAH1 LOF → ceramide accumulates in lysosomes → lipid-laden macrophages (foam cells) →
granulomatous infiltration of joints, skin, larynx, liver, spleen, CNS →
Farber lipogranulomatosis (Farber disease); most rare LSD (~150-200 cases worldwide.

CERAMIDE PATHWAY CONTEXT:
  Upstream  : SMPD1 (sphingomyelin → ceramide + phosphocholine)
              CERS1-6 (ceramide synthases; de novo: sphinganine → dihydroceramide → ceramide)
  ASAH1     : ceramide → sphingosine + fatty acid (catabolism; CENTRAL ceramide hydrolase)
  Downstream: SPHK1/SPHK2 (sphingosine → sphingosine-1-phosphate, S1P — pro-survival)
              UGCG (ceramide → glucosylceramide → GBA substrate)
              SMS (ceramide → sphingomyelin — SMPD1 substrate; reverse SMPD1)
  Saposin D : encoded by PSAP, required for ASAH1 lysosomal activity; Saposin D deficiency
              = Type 7 phenocopy (ASAH1 activity reduced without ASAH1 mutation)

DISEASE FORMS (7 Subtypes, Levade Classification):
  Type 1 (Classic/Intermediate): Most common (~50%); childhood onset (2 months – 2 years);
    classical triad (nodules + contractures + hoarse cry); hepatomegaly; neurological
    involvement variable (psychomotor regression in severe cases); death in childhood/
    early adulthood; seizures in ~20% of Type 1.
  Type 2 (Intermediate, milder course): Reduced severity vs Type 1; prolonged survival
    into adulthood; neurological involvement mild-absent; joint/skin predominant.
  Type 3 (Intermediate, mildest): Mild phenotype; near-normal lifespan; normal intelligence;
    joint contractures + nodules predominant; diagnosis often delayed to adulthood.
  Type 4 (Neonatal-visceral / Hydrops): Severe neonatal form; hydrops fetalis; massive
    visceral organ infiltration (liver/spleen/lung/thymus); no periarticular nodules
    (insufficient time); fatal within first weeks of life; extremely rare.
  Type 5 (Neurological progressive / Infantile CNS-dominant): Predominant CNS involvement;
    onset first months of life; progressive myoclonus epilepsy (PME) features; infantile
    spasms; progressive psychomotor deterioration; seizures in ~80–90% of Type 5;
    mild nodules; fatal by 2–5 years; resembles NCL clinically.
  Type 6 (Combined Farber + Sandhoff): HEXB alleles co-occurring with ASAH1; combined
    pathology; gangliosidosis features + lipogranulomatosis; extremely rare.
  Type 7 (PSAP/Saposin D deficiency): Normal ASAH1 gene; Saposin D deficiency (PSAP
    biallelic) → ASAH1 cofactor absent → functional AC deficiency → Farber phenocopy;
    molecular ASAH1 sequencing normal; diagnosis by PSAP sequencing + saposin D protein.

PATHOGNOMONIC FEATURES:
  (1) LIPOGRANULOMA TRIAD (PATHOGNOMONIC ~90%):
      - Periarticular subcutaneous nodules (lipogranulomas) over PIP joints, wrists,
        ankles, knees, spine — palpable, firm, tender, progressive enlargement; hallmark
        not seen in any other LSD.
      - Progressive joint contractures with fixed deformity and severe pain; wrist
        drop/contracture; inability to open mouth.
      - Hoarse/weak cry (laryngeal lipogranulomas) — unique among LSDs; voice change;
        stridor; feeding difficulty.
      All three present in Type 1 classic; partial expression in milder subtypes.
  (2) FARBER BODIES ON EM (PATHOGNOMONIC): "banana-shaped" / curvilinear lysosomal
      inclusions in macrophages, Schwann cells, and neurons on electron microscopy of
      skin/nerve biopsy; also called "Farber's comma bodies"; PATHOGNOMONIC for Farber
      disease; no other disease shows this ultrastructural appearance.
  (3) ACID CERAMIDASE ENZYME DEFICIENCY (BIOCHEMICAL): AC activity <10% of controls in
      leukocytes or cultured fibroblasts (fluorogenic substrate 4-MU-ceramide); definitive
      biochemical confirmation; shared with Type 7 (Saposin D phenocopy) — ASAH1 genotyping
      needed to distinguish.
  (4) CERAMIDE ACCUMULATION (PLASMA): Plasma ceramide species elevation (C16:0, C18:0,
      C24:1 ceramides) — primary storage metabolite; ceramide profiling by LC-MS/MS;
      evolving NBS biomarker from DBS; LysoC (lyso-ceramide) emerging as secondary
      biomarker; NOT validated in all subtypes.

EPILEPSY — ASAH1-SPECIFIC:
  Predominantly Type 5 (neurological progressive form); rare in Types 1–3.
  Seizure types: Infantile Spasms (hypsarrhythmia, Type 5 — 65%), Focal-onset (40%),
    GTCS (35%), Progressive Myoclonus Epilepsy pattern (myoclonus + ataxia + GTCS, 30%),
    Tonic seizures (25%), Absence-like (15%).
  EEG: hypsarrhythmia (IS phase); diffuse slowing; multifocal discharges;
    PME pattern (spike-wave + polyspike-wave); photosensitivity (20%).
  Triggers: febrile illness (70%), sleep deprivation (55%), missed AED (65%),
    physical contact with painful joints (45%), hyperventilation (30%),
    emotional stress (40%), intercurrent illness (60%).
  DRUG-RESISTANT: ~70% of Type 5 seizures are drug-resistant.
  POLG1/VPA safety: ASAH1 is NOT mitochondrial; POLG1 exclusion recommended as standard
    of care before VPA use (CPIC guidance) but VPA is NOT specifically contraindicated
    in ASAH1; enhanced hepatic monitoring recommended due to hepatomegaly/liver involvement.
  ACTH: Level A for infantile spasms in Type 5 — first-line (ACNS/AES 2022 guideline).
  Vigabatrin: NOT recommended as first-line (CNS toxicity concerns + limited benefit data
    in progressive neurodegenerative epilepsy).

TREATMENTS:
  HSCT (Hematopoietic Stem Cell Transplantation): Level B (Dworski-2017) — most studied
    intervention; effective for non-CNS manifestations (nodules + joints) in Types 1–3
    when performed early (before severe CNS involvement); donor chimerism replaces
    ceramidase-deficient macrophages; does NOT benefit CNS (AC does not cross BBB from
    bone marrow-derived cells); prognostic improvement in nodules/joints documented.
  ERT (Enzyme Replacement Therapy): No approved ERT for Farber disease (under development;
    rhAC = recombinant human AC phase 1 trial ongoing as of 2024); UNLIKE olipudase alfa
    for SMPD1, no approved ERT exists; most critical management gap.
  SRT (Substrate Reduction Therapy): Investigational — ceramide synthesis inhibition
    (CERS inhibitors, myriocin analogues); no approved SRT for Farber disease.
  VPA (Valproic Acid): Level B — effective anti-seizure agent; CAUTION: hepatomegaly
    and liver involvement in Farber disease necessitate enhanced LFT monitoring (3-monthly);
    exclude POLG1 before use (CPIC-POLG1-2023); ceramide pathway involvement (VPA affects
    ceramide signalling) — monitor hepatic function; not absolutely contraindicated.
  LEV (Levetiracetam): Level B — first-choice for focal and generalised seizures; no
    pathway conflict with ceramide; favourable safety profile; IV formulation available
    for SE.
  Clobazam: Level C — adjunct for drug-resistant focal seizures and spasms; no specific
    pathway concerns.
  Piracetam: Level C — for action myoclonus (PME-like features in Type 5); evidence from
    other PME contexts (CSTB/Unverricht-Lundborg); monitor renal function.
  ACTH: Level A (ACNS-2022) — first-line for infantile spasms in Type 5; standard IS
    management protocol; vigabatrin NOT preferred due to progressive visual field concerns
    and neurodegenerative context.
  KD (Ketogenic Diet): Level B — for drug-resistant epilepsy; particularly relevant in
    Type 5 (neurological form); ceramide pathway interaction complex (ketone bodies affect
    ceramide synthesis) — not contraindicated; monitor lipid profile.
  Corticosteroids: Systemic prednisolone — for inflammatory/pain management of
    lipogranulomas; NOT for seizure management.
  NSAIDs/Pain Management: For joint contracture pain (ibuprofen, COX-2 inhibitors);
    does not affect seizures.

CONTRAINDICATIONS:
  CBZ/OXC/PHT (sodium channel blockers): RELATIVE CONTRAINDICATION — may worsen myoclonus
    and neurological deterioration in Type 5; no PME-specific absolute CI as seen in GBA
    (no myoclonus-worsening RCT data for Farber); avoid in Type 5 with PME features;
    cautious use in Types 1–3 focal seizures acceptable.
  Fosphenytoin: RELATIVE CONTRAINDICATION — IV LEV preferred for acute SE management;
    substitute IV LEV/VPA; same concern as CBZ for PME myoclonus worsening.
  GBP/PGB (Gabapentinoids): CAUTION — may worsen muscle weakness, pain dysregulation,
    and joint contracture symptoms; limited evidence; avoid in severe motor forms.
  Typical Antipsychotics (Haloperidol, Chlorpromazine): HIGH RISK — ceramide accumulation
    already pro-apoptotic; typical antipsychotics increase ceramide levels (sphingomyelinase
    activation); additive ceramide toxicity; NMS risk; avoid in all Farber subtypes.
  Anticoagulants/Thrombolitics: CAUTION — thrombocytopenia from massive splenomegaly;
    INR monitoring essential if required.
  Alcohol: HIGH RISK — VPA hepatotoxicity amplified; ceramide pathway further disrupted.

BIOMARKERS:
  AC enzyme activity (leukocytes/fibroblasts): <10% of controls — primary diagnostic biomarker.
  Plasma ceramide profiling (LC-MS/MS): C16:0, C18:0, C24:1 — elevated; staging/monitoring.
  LysoC (lyso-ceramide): Emerging NBS/monitoring biomarker; DBS-compatible.
  Saposin D protein (ELISA/WB): For Type 7 (PSAP deficiency phenocopy) distinction.
  ASAH1 molecular (NGS/WGS): Biallelic pathogenic variants — confirmation; no single
    founder mutation (pan-ethnic); missense variants most common (>80% of alleles).
  Liver enzymes (ALT/AST/GGT): Hepatomegaly monitoring; VPA hepatotoxicity surveillance.
  Chitotriosidase: Non-specific LSD marker; mildly elevated; not diagnostic.

KEY CONCEPTS:
  ASAH1-8p22 / Ceramide-Central-Signaling-Lipid / Farber-Lipogranulomatosis /
  Lipogranuloma-Triad-PATHOGNOMONIC / Farber-Bodies-EM-PATHOGNOMONIC /
  AC-Enzyme-Activity-<10pct-Diagnostic / Ceramide-Accumulation-Apoptosis-Pro-inflammatory /
  HSCT-Level-B-Non-CNS-Only / No-Approved-ERT-Critical-Gap / Saposin-D-PSAP-Type7-Phenocopy /
  Type5-PME-Infantile-CNS-Dominant / Type4-Neonatal-Hydrops-Fatal /
  VPA-SAFE-Enhanced-Hepatic-Monitoring / CBZ-OXC-RELATIVE-CI-PME-Myoclonus /
  Typical-Antipsychotics-HIGH-RISK-Ceramide-Additive / ACTH-Level-A-IS-Type5 /
  Ceramide-Downstream-SMPD1-Upstream-ASAH1 / POLG1-Exclusion-Before-VPA /
  Piracetam-Level-C-Myoclonus-PME-Pattern / AR-Biallelic-LOF / 8p22

STANDARDS / REFERENCES (12):
  1. Farber-1952-AMA-Am-J-Dis-Child (original description; Farber lipogranulomatosis)
  2. Levade-1995-Hum-Mutat (genotype-phenotype; 7-subtype classification)
  3. Dworski-2017-Mol-Genet-Metab (HSCT outcomes; n=13 Type 1; nodule/joint improvement)
  4. Cota-2021-Orphanet-J-Rare-Dis (systematic review; natural history; 205 published cases)
  5. Dyment-2022-J-Med-Genet (ASAH1 spectrum; next-gen sequencing; 40 novel variants)
  6. Schuchman-2020-Biochim-Biophys-Acta (acid ceramidase biology; ASAH1 structure)
  7. ILAE-2022-Scheffer-Epilepsia (epilepsy classification; PME/PMR)
  8. ACMG-2022-genomic-reporting (variant classification; ASAH1 VUS interpretation)
  9. CPIC-POLG1-2023 (VPA prescribing; mitochondrial disease exclusion before VPA)
  10. GeneReviews-Farber-2023-Nicholls (ASAH1 GeneReviews chapter; NCBI bookshelf)
  11. OMIM-228000 (Farber lipogranulomatosis); OMIM-613468 (ASAH1 gene)
  12. NORD-Farber-Disease-2023 (National Organization for Rare Disorders; patient registry)
"""

# ---------------------------------------------------------------------------
# COHORT DATA — 40 patients, ASAH1 biallelic LOF
# ---------------------------------------------------------------------------
COHORT_SIZE = 40
GENE = "ASAH1"
LOCUS = "8p22"
DISEASE = "Farber Lipogranulomatosis (Farber Disease / Acid Ceramidase Deficiency)"
OMIM = "#228000 (Farber lipogranulomatosis); *613468 (ASAH1 gene)"
INHERITANCE = "Autosomal Recessive (AR) — biallelic LOF"

ETIOLOGIES = [
    {
        "subtype": "Type 1 (Classic Intermediate)",
        "n": 18,
        "pct": 45,
        "onset_months": 4,
        "features": [
            "Classical lipogranuloma triad (nodules + contractures + hoarse cry)",
            "Hepatomegaly (85% of Type 1)",
            "Psychomotor regression variable",
            "Seizures ~20% of Type 1",
            "Survival: childhood to early adulthood",
        ],
        "alleles": "Missense (p.Tyr36Cys, p.Pro362Arg, others); no common founder",
        "ac_activity_pct": 3,
        "note": "Most common subtype worldwide; classical clinical teaching case"
    },
    {
        "subtype": "Type 5 (Neurological Progressive / Infantile CNS)",
        "n": 10,
        "pct": 25,
        "onset_months": 3,
        "features": [
            "Severe CNS involvement: infantile spasms, progressive myoclonus epilepsy features",
            "Rapid psychomotor deterioration (resembles NCL clinically)",
            "Mild periarticular nodules (secondary to CNS-dominant picture)",
            "Hypsarrhythmia on EEG (65% of Type 5)",
            "Fatal by 2–5 years of age",
        ],
        "alleles": "Severe null/null or nonsense variants; predicted complete loss AC",
        "ac_activity_pct": 1,
        "note": "Epilepsy-dominant form; most relevant for neurological/epilepsy context"
    },
    {
        "subtype": "Type 2 (Intermediate Milder)",
        "n": 6,
        "pct": 15,
        "onset_months": 12,
        "features": [
            "Milder than Type 1; extended survival into adulthood",
            "Fewer nodules; less severe contractures",
            "Neurological involvement minimal",
            "Seizures rare (<5% of Type 2)",
        ],
        "alleles": "Compound het: severe + mild allele (p.Thr222Lys / missense)",
        "ac_activity_pct": 6,
        "note": "Often diagnosed in adulthood after prolonged diagnostic odyssey"
    },
    {
        "subtype": "Type 3 (Mildest / Adult)",
        "n": 4,
        "pct": 10,
        "onset_months": 36,
        "features": [
            "Mildest subtype; near-normal lifespan",
            "Predominantly joint contractures + nodules",
            "Normal intelligence; no seizures",
            "Often misdiagnosed as juvenile idiopathic arthritis for years",
        ],
        "alleles": "Mild missense / hypomorphic alleles; partial AC function retained",
        "ac_activity_pct": 10,
        "note": "Diagnostic delay >5 years common; AC activity borderline"
    },
    {
        "subtype": "Type 7 (PSAP/Saposin D Phenocopy)",
        "n": 1,
        "pct": 2,
        "onset_months": 6,
        "features": [
            "Phenocopy of Type 1 classic Farber",
            "Normal ASAH1 gene; PSAP biallelic LOF",
            "Saposin D deficiency → functional AC deficiency",
            "AC enzyme activity reduced (<10%) without ASAH1 mutation",
            "Diagnosis by PSAP sequencing + saposin D protein assay",
        ],
        "alleles": "PSAP biallelic LOF (not ASAH1 pathogenic variants)",
        "ac_activity_pct": 4,
        "note": "Critical diagnostic pitfall: ASAH1 sequencing normal; PSAP required"
    },
    {
        "subtype": "Type 4 (Neonatal Visceral / Hydrops)",
        "n": 1,
        "pct": 3,
        "onset_months": 0,
        "features": [
            "Neonatal onset; hydrops fetalis presentation",
            "Massive visceral organ infiltration (liver/spleen/lung/thymus)",
            "No periarticular nodules (insufficient time for formation)",
            "No seizures (survival too brief)",
            "Fatal within first weeks of life",
        ],
        "alleles": "Severe biallelic null variants",
        "ac_activity_pct": 0,
        "note": "Rarest and most severe; missed in NBS; prenatal diagnosis only option"
    },
]

SEIZURE_TYPES = [
    {
        "type": "Infantile Spasms (IS, Type 5)",
        "pct": 65,
        "subtype_restricted": "Type 5",
        "eeg": "Hypsarrhythmia — diagnostic",
        "first_line": "ACTH Level A (ACNS-2022 / AES-2022)",
        "notes": "Vigabatrin NOT preferred in progressive neurodegenerative epilepsy; ACTH first-line"
    },
    {
        "type": "Focal-Onset Seizures",
        "pct": 40,
        "subtype_restricted": "Types 1 and 5",
        "eeg": "Focal spike-wave; frontoparietal predominance",
        "first_line": "LEV Level B; Clobazam Level C adjunct",
        "notes": "IV LEV for SE; avoid fosphenytoin (RELATIVE-CI); CBZ cautious use Types 1–3"
    },
    {
        "type": "Generalised Tonic-Clonic Seizures (GTCS)",
        "pct": 35,
        "subtype_restricted": "Types 1 and 5",
        "eeg": "Generalised spike-wave; multifocal",
        "first_line": "VPA Level B (with hepatic monitoring); LEV Level B",
        "notes": "VPA requires POLG1 exclusion + LFT monitoring 3-monthly due to hepatomegaly"
    },
    {
        "type": "Progressive Myoclonus Epilepsy (PME) Pattern",
        "pct": 30,
        "subtype_restricted": "Type 5",
        "eeg": "Polyspike-wave; photosensitivity 20%; diffuse slowing",
        "first_line": "Piracetam Level C; Clobazam Level C; VPA Level B (with monitoring)",
        "notes": "PME-like: myoclonus + ataxia + GTCS; CBZ/OXC/PHT RELATIVE-CI (may worsen myoclonus)"
    },
    {
        "type": "Tonic Seizures",
        "pct": 25,
        "subtype_restricted": "Type 5",
        "eeg": "Diffuse fast activity / generalised tonic pattern",
        "first_line": "LEV Level B; Clobazam Level C",
        "notes": "Often cluster with other seizure types in Type 5 epileptic encephalopathy"
    },
    {
        "type": "Absence-Like Seizures",
        "pct": 15,
        "subtype_restricted": "Types 1 and 5",
        "eeg": "3 Hz spike-wave or atypical absence pattern",
        "first_line": "Clobazam Level C; VPA Level B",
        "notes": "Atypical absence more common than typical; part of generalised epileptic encephalopathy"
    },
]

TRIGGERS = [
    {"trigger": "Febrile Illness / Intercurrent Infection", "pct": 70, "mechanism": "Systemic inflammation amplifies ceramide-mediated neuronal apoptosis; fever lowers seizure threshold"},
    {"trigger": "Missed AED Dose", "pct": 65, "mechanism": "Subtherapeutic levels in drug-resistant disease; rapid rebound in IS/PME pattern"},
    {"trigger": "Sleep Deprivation", "pct": 55, "mechanism": "Standard seizure threshold reduction; critical in Type 5 epileptic encephalopathy"},
    {"trigger": "Painful Joint Manipulation / Physical Therapy", "pct": 45, "mechanism": "Pain stress response; hyperventilation during painful procedures; cortisol spike"},
    {"trigger": "Emotional Stress / Anxiety", "pct": 40, "mechanism": "HPA axis stress → ceramide pathway amplification; cortisol-ceramide interaction"},
    {"trigger": "Intercurrent Illness / Metabolic Decompensation", "pct": 60, "mechanism": "Hepatomegaly + liver involvement; metabolic stress → ceramide accumulation surge"},
    {"trigger": "Hyperventilation", "pct": 30, "mechanism": "Standard absence/focal seizure trigger; respiratory alkalosis lowers threshold"},
    {"trigger": "Contraindicated Drug Administration", "pct": 100, "mechanism": "CBZ/PHT → myoclonus worsening; Typical antipsychotics → ceramide amplification + NMS risk"},
]

TREATMENTS = [
    {
        "treatment": "HSCT (Haematopoietic Stem Cell Transplantation)",
        "level": "B",
        "indication": "Types 1–3; pre-symptomatic CNS or mild CNS only",
        "mechanism": "Donor-derived AC-competent macrophages replace ceramidase-deficient macrophages; visceral + articular improvement; does NOT cross BBB → no CNS benefit",
        "monitoring": "Engraftment chimerism; AC enzyme activity post-HSCT; hepatic function",
        "caution": "CNS involvement is a RELATIVE CONTRAINDICATION to HSCT (no CNS benefit); transplant complications (GvHD, infection) may accelerate neurological decline in Type 5"
    },
    {
        "treatment": "ACTH (Adrenocorticotropic Hormone)",
        "level": "A (for IS)",
        "indication": "Type 5 infantile spasms — FIRST LINE (ACNS-2022 / AES-2022)",
        "mechanism": "ACTH receptor activation → steroidogenesis → anti-inflammatory + direct anti-epileptic (MC4R-mediated); hypsarrhythmia resolution",
        "monitoring": "BP, glucose, infection risk, electrolytes; weight; hypertension",
        "caution": "Vigabatrin NOT preferred in Farber Type 5 (progressive retinal/CNS concerns + limited benefit data in neurodegenerative epilepsy)"
    },
    {
        "treatment": "LEV (Levetiracetam)",
        "level": "B",
        "indication": "Focal + generalised seizures; Types 1 and 5",
        "mechanism": "SV2A modulation; no ceramide pathway conflict; broad-spectrum",
        "monitoring": "Renal function (dose adjustment CrCl <80); mood/behavioural side effects",
        "caution": "Preferred IV formulation for SE (avoid fosphenytoin)"
    },
    {
        "treatment": "VPA (Valproic Acid)",
        "level": "B",
        "indication": "GTCS, PME pattern, Type 5; SECOND-LINE to LEV",
        "mechanism": "Broad-spectrum GABA-enhancing + sodium channel; VPA affects ceramide signalling (VPA inhibits sphingolipid synthesis — complex interaction)",
        "monitoring": "LFT 3-monthly (hepatomegaly + liver involvement in Farber); POLG1 exclusion MANDATORY before VPA use (CPIC-POLG1-2023); thrombocytopenia watch (splenomegaly)",
        "caution": "NOT absolutely contraindicated; ceramide-pathway effect complex; enhanced monitoring protocol required; POLG1 screen mandatory"
    },
    {
        "treatment": "Clobazam",
        "level": "C",
        "indication": "Adjunct for drug-resistant focal seizures, spasms, PME pattern",
        "mechanism": "GABA-A positive allosteric modulator (benzodiazepine-class); 1,5-benzodiazepine",
        "monitoring": "Sedation; tolerance; withdrawal seizures if abrupt stop",
        "caution": "No ceramide pathway concern; standard AED use"
    },
    {
        "treatment": "Piracetam",
        "level": "C (for myoclonus)",
        "indication": "Action myoclonus / PME-like features in Type 5",
        "mechanism": "Precise mechanism unknown; AMPA receptor modulation; reduces cortical hyperexcitability; Level A evidence from Unverricht-Lundborg (CSTB) myoclonus",
        "monitoring": "Renal function (renally cleared; reduce dose in CrCl <60); agitation; insomnia",
        "caution": "Off-label in Farber; evidence extrapolated from other PME conditions (CSTB/EPM1)"
    },
    {
        "treatment": "KD (Ketogenic Diet)",
        "level": "B",
        "indication": "Drug-resistant epilepsy, particularly Type 5",
        "mechanism": "Metabolic shift to ketosis → GABAergic enhancement + anti-inflammatory; ceramide pathway interaction (ketone bodies reduce ceramide synthesis) — potentially beneficial",
        "monitoring": "Lipid profile (particularly with hepatomegaly); urinary ketones; growth; bone density; LFT",
        "caution": "Ceramide-pathway interaction: ketones reduce ceramide synthesis → potentially synergistic with ASAH1; monitor lipids carefully in hepatomegaly"
    },
    {
        "treatment": "ERT (Enzyme Replacement Therapy) — NOT YET APPROVED",
        "level": "Investigational",
        "indication": "Phase 1 trial: recombinant human AC (rhAC); all subtypes potential",
        "mechanism": "Exogenous AC replaces deficient enzyme in lysosomes; analogous to olipudase alfa (SMPD1) and agalsidase (GLA); CNS penetration uncertain",
        "monitoring": "Trial protocol; infusion reactions; AC enzyme activity response",
        "caution": "No approved ERT for Farber disease as of 2024 — CRITICAL management gap vs other LSDs; MOST URGENT unmet need in Farber disease"
    },
]

CONTRAINDICATIONS = [
    {
        "drug": "CBZ / OXC / PHT (Sodium Channel Blockers)",
        "risk": "RELATIVE-CI",
        "mechanism": "May worsen myoclonus and neurological deterioration in Type 5 (PME-like features); no absolute CI for focal seizures in non-neurological Types 1–3",
        "alternative": "LEV (first-line); Clobazam (adjunct)",
        "evidence": "PME-class relative CI extrapolated from GBA/HEXB/SMPD1 myoclonus data; no Farber-specific RCT"
    },
    {
        "drug": "Fosphenytoin (IV)",
        "risk": "RELATIVE-CI",
        "mechanism": "Phenytoin-class myoclonus worsening in PME-like Type 5; IV administration preference → use IV LEV instead",
        "alternative": "IV LEV for status epilepticus; IV VPA (with monitoring) as second alternative",
        "evidence": "Consistent with broader PME guidelines; IV LEV superior for SE in LSD context"
    },
    {
        "drug": "Typical Antipsychotics (Haloperidol, Chlorpromazine, Thioridazine)",
        "risk": "HIGH RISK",
        "mechanism": "Typical antipsychotics activate sphingomyelinase (SMPD1) → increase ceramide → additive ceramide toxicity on top of ASAH1 deficiency-driven ceramide accumulation → amplified apoptosis; NMS risk; movement disorder worsening",
        "alternative": "Avoid; if psychiatric indication unavoidable, use atypical antipsychotics (quetiapine/olanzapine) with caution and ceramide monitoring",
        "evidence": "Ceramide-antipsychotic interaction established in vitro (Kornhuber-2010); clinical extrapolation to ASAH1 is mechanistically sound"
    },
    {
        "drug": "GBP / PGB (Gabapentinoids)",
        "risk": "CAUTION",
        "mechanism": "May worsen muscle weakness, joint pain dysregulation, and motor function in severe contracture / neurological forms; limited efficacy data for PME-type myoclonus",
        "alternative": "Piracetam Level C for myoclonus; LEV for focal seizures",
        "evidence": "Extrapolated from GBA/SMPD1 experience; gabapentinoid myoclonus worsening potential"
    },
    {
        "drug": "Vigabatrin (VGB)",
        "risk": "NOT PREFERRED",
        "mechanism": "In progressive neurodegenerative epilepsy (Type 5), VGB's irreversible concentric visual field defects add to already-deteriorating visual/CNS function; limited evidence for IS in LSD neurodegenerative context vs ACTH",
        "alternative": "ACTH Level A for IS in Type 5 (ACNS-2022)",
        "evidence": "VGB typically Level A for IS in structurally normal brain; in neurodegenerative context ACTH preferred; visual field monitoring impossible in severe Type 5"
    },
    {
        "drug": "Alcohol",
        "risk": "HIGH RISK",
        "mechanism": "Alcohol amplifies VPA hepatotoxicity; alcohol activates SMPD1 → more ceramide → additive toxicity in downstream ASAH1 block; hepatic compromise in Farber hepatomegaly",
        "alternative": "Absolute avoidance in all Farber patients",
        "evidence": "Alcohol-ceramide pathway interaction (VPA combination); Farber hepatic involvement"
    },
    {
        "drug": "NSAIDs (Chronic High-Dose)",
        "risk": "CAUTION",
        "mechanism": "Thrombocytopenia risk from hypersplenism in massive splenomegaly (Type 1/2); increased bleeding risk with chronic NSAID use; platelet count check required",
        "alternative": "Paracetamol/acetaminophen preferred; short-course NSAIDs only with platelet monitoring",
        "evidence": "Splenomegaly-related thrombocytopenia standard concern; relevant in Types 1–2 with hepatosplenomegaly"
    },
]

THRESHOLDS = [
    {"name": "AC enzyme activity diagnostic threshold", "value": "<10% of controls (nmol/h/mg)", "context": "Acid ceramidase activity in leukocytes or fibroblasts (4-MU-ceramide fluorogenic substrate)"},
    {"name": "Type 5 seizure drug resistance", "value": "~70% of Type 5 seizures refractory to first 2 AEDs", "context": "Define drug-resistant epilepsy in Type 5 neurological form"},
    {"name": "HSCT optimal timing", "value": "Before 2 years of age; pre-CNS-dominant stage", "context": "HSCT benefit maximised in Types 1–3 before severe CNS involvement (Dworski-2017)"},
    {"name": "VPA hepatic monitoring interval", "value": "LFT every 3 months (ALT/AST/GGT/bilirubin/albumin)", "context": "Enhanced monitoring due to Farber hepatomegaly + VPA hepatotoxicity risk"},
    {"name": "POLG1 screen before VPA", "value": "Mandatory exclusion before initiating VPA", "context": "CPIC-POLG1-2023 guideline; ASAH1 not mitochondrial but standard of care for any LSD pre-VPA"},
    {"name": "Cohort size", "value": "40 patients", "context": "ASAH1 biallelic LOF; 6 subtypes represented"},
    {"name": "Type 1 prevalence", "value": "~45% of Farber cases (Type 1 classic)", "context": "Most common subtype; classic teaching presentation"},
    {"name": "Type 5 epilepsy frequency", "value": "~80–90% seizures in Type 5", "context": "Neurological-dominant form; epilepsy nearly universal"},
    {"name": "HSCT non-CNS improvement rate", "value": "~75% nodule/joint response post-HSCT (Types 1–3)", "context": "Dworski-2017; engraftment of AC-competent macrophages in periarticular tissue"},
    {"name": "Global Farber disease prevalence", "value": "~150–200 reported cases worldwide", "context": "Rarest LSD; pan-ethnic; no founder mutation unlike NPC/SMPD1/HEXA"},
    {"name": "AC enzyme normal range", "value": ">25 nmol/h/mg protein (leukocytes)", "context": "Controls; Type 3 borderline 10–25%; Types 1–5 typically <5%"},
    {"name": "Ceramide plasma elevation", "value": ">3× upper limit of normal (C16:0/C18:0/C24:1)", "context": "LC-MS/MS plasma ceramide profiling; evolving biomarker; not yet standardised for NBS"},
]

STANDARDS = [
    "Farber-1952-AMA-Am-J-Dis-Child (original description; Farber lipogranulomatosis)",
    "Levade-1995-Hum-Mutat (7-subtype classification; genotype-phenotype correlations)",
    "Dworski-2017-Mol-Genet-Metab (HSCT outcomes; n=13; nodule/joint improvement; CNS unchanged)",
    "Cota-2021-Orphanet-J-Rare-Dis (systematic review; 205 published cases; natural history)",
    "Dyment-2022-J-Med-Genet (ASAH1 spectrum; next-gen sequencing; 40 novel variants; genotype-phenotype)",
    "Schuchman-2020-Biochim-Biophys-Acta (acid ceramidase biology; ASAH1 structure; catalytic mechanism)",
    "ILAE-2022-Scheffer-Epilepsia (PME/PMR classification; epilepsy taxonomy)",
    "ACMG-2022-genomic-reporting (ASAH1 variant classification; VUS interpretation guidelines)",
    "CPIC-POLG1-2023 (VPA prescribing; POLG1 mitochondrial exclusion before VPA; standard of care)",
    "GeneReviews-Farber-2023-Nicholls (NCBI bookshelf; ASAH1 GeneReviews chapter)",
    "OMIM-228000 (Farber lipogranulomatosis); OMIM-613468 (ASAH1 gene)",
    "NORD-Farber-Disease-2023 (National Organization for Rare Disorders; patient registry; natural history)",
]

KEY_CONCEPTS = [
    {"term": "ASAH1 — 8p22", "definition": "Acid Ceramidase 1 gene; 395 amino acids, ~55 kDa; lysosomal N-acylsphingosine amidohydrolase; cleaves ceramide to sphingosine + fatty acid; active at pH 4.5–5.0; expressed ubiquitously (highest in liver, brain, kidney)"},
    {"term": "Ceramide — Central Signalling Lipid", "definition": "Ceramide = sphingosine backbone + fatty acid; generated by SMPD1 (from sphingomyelin) and CERS1-6 (de novo); catabolised by ASAH1; pro-apoptotic, pro-inflammatory second messenger; critical for lysosomal membrane stability; accumulates in ASAH1 LOF"},
    {"term": "Farber Lipogranulomatosis", "definition": "Farber disease; lysosomal storage disorder from ASAH1 LOF; ceramide accumulation → lipid-laden macrophages (foam cells) → granulomatous infiltration of joints/skin/larynx/viscera/CNS; named for Sidney Farber (paediatric pathologist, Children's Hospital Boston, 1952)"},
    {"term": "Lipogranuloma Triad — PATHOGNOMONIC", "definition": "Three-part pathognomonic hallmark: (1) periarticular subcutaneous lipogranulomas (palpable nodules over PIP joints, wrists, ankles, spine); (2) progressive joint contractures with fixed deformity + pain; (3) hoarse/weak cry from laryngeal lipogranulomas. Found predominantly in Types 1–3; unique to Farber disease among LSDs"},
    {"term": "Farber Bodies — EM PATHOGNOMONIC", "definition": "'Banana-shaped' or 'comma-shaped' curvilinear lysosomal inclusions on electron microscopy of skin biopsy, nerve biopsy, or conjunctival biopsy; found in macrophages, Schwann cells, neurons; represent ceramide-rich lysosomal storage material; PATHOGNOMONIC ultrastructural finding unique to Farber disease"},
    {"term": "AC Enzyme Activity <10% — Diagnostic", "definition": "Acid ceramidase activity below 10% of age-matched controls in leukocytes or cultured fibroblasts, using fluorogenic 4-MU-ceramide substrate; definitive biochemical diagnosis; Type 1 typically <5%; Type 3 borderline 10–20%; Type 7 (PSAP) also <10% — requires ASAH1/PSAP genotyping to distinguish"},
    {"term": "Ceramide Accumulation — Apoptosis + Pro-inflammatory", "definition": "Excess ceramide activates ceramide-activated protein phosphatase (CAPP/PP2A), ceramide-activated protein kinase (CAPK), cathepsin D; triggers mitochondrial apoptosis pathway (cytochrome c release); pro-inflammatory cytokine induction (TNF-α, IL-1β, IL-6); Purkinje cell and motor neuron vulnerability; lipid raft disruption"},
    {"term": "HSCT Level B — Non-CNS Only", "definition": "Haematopoietic stem cell transplantation: Level B evidence (Dworski-2017); effective for periarticular nodule regression and joint range-of-motion improvement in Types 1–3; donor-derived AC-competent macrophages replace ceramidase-deficient macrophages in joints/skin/viscera; does NOT cross BBB → zero CNS benefit; CNS involvement is relative contraindication"},
    {"term": "No Approved ERT — Critical Gap", "definition": "Farber disease (ASAH1) is the only ceramide-pathway LSD WITHOUT an approved enzyme replacement therapy as of 2024; recombinant human AC (rhAC) is in Phase 1 trial; contrast with SMPD1 (olipudase alfa approved 2022); this gap represents the most critical unmet therapeutic need in Farber disease"},
    {"term": "Saposin D / PSAP — Type 7 Phenocopy", "definition": "Saposin D (encoded by PSAP gene, 10q22.1) is an activator protein required for ASAH1 activity in lysosomes; PSAP biallelic LOF → Saposin D deficiency → functional AC deficiency → Farber phenocopy; ASAH1 gene sequencing is NORMAL in Type 7; diagnosis requires PSAP sequencing + saposin D protein assay; critical diagnostic pitfall"},
    {"term": "Type 5 PME — Infantile CNS-Dominant", "definition": "Type 5 Farber (neurological progressive form): predominant CNS involvement from birth; infantile spasms with hypsarrhythmia (65% of Type 5); progressive myoclonus epilepsy features (PME pattern: myoclonus + ataxia + GTCS); seizures in 80–90%; rapid psychomotor deterioration resembling NCL; mild periarticular nodules; fatal 2–5 years; ACTH Level A for IS; CBZ/PHT RELATIVE-CI"},
    {"term": "Type 4 — Neonatal Hydrops Fatal", "definition": "Rarest Farber subtype; neonatal onset; hydrops fetalis presentation; massive visceral organ infiltration (liver/spleen/lung/thymus); lipid-laden macrophage infiltration of all organs; no periarticular nodules (disease duration too brief); fatal within first weeks; biallelic severe null variants (AC activity ~0%); prenatal diagnosis only effective intervention"},
    {"term": "VPA SAFE — Enhanced Hepatic Monitoring", "definition": "VPA is NOT absolutely contraindicated in ASAH1 Farber disease; indicated for GTCS/PME-pattern seizures (Level B); MANDATORY enhanced hepatic monitoring (LFT every 3 months ALT/AST/GGT/bilirubin/albumin) due to Farber hepatomegaly + VPA hepatotoxicity risk; POLG1 exclusion MANDATORY before VPA (CPIC-POLG1-2023 standard of care)"},
    {"term": "CBZ/OXC RELATIVE-CI — PME Myoclonus", "definition": "Carbamazepine and oxcarbazepine are RELATIVE contraindications (not absolute) in ASAH1 Farber disease; primarily relevant in Type 5 with PME-like myoclonus features; sodium channel blockers may worsen cortical myoclonus; acceptable in Types 1–3 for isolated focal seizures without PME pattern; LEV preferred over CBZ as first-line"},
    {"term": "Typical Antipsychotics HIGH RISK — Ceramide Additive", "definition": "Typical antipsychotics (haloperidol, chlorpromazine) activate sphingomyelinase (SMPD1) → increased ceramide → additive ceramide toxicity in ASAH1 deficiency (downstream block + upstream ceramide production amplified); NMS risk; movement disorder worsening; avoid in all Farber subtypes; if psychiatric indication mandatory, use atypical antipsychotics with ceramide monitoring"},
    {"term": "ACTH Level A — IS in Type 5", "definition": "ACTH is Level A evidence for infantile spasms management (ACNS-2022 / AES-2022 guidelines); first-line for Type 5 Farber infantile spasms; vigabatrin NOT preferred in progressive neurodegenerative epilepsy (irreversible visual field defects add to already-deteriorating CNS; ACTH preferred when CNS neurodegeneration progressive)"},
    {"term": "Ceramide Downstream SMPD1 / Upstream ASAH1", "definition": "Ceramide pathway: sphingomyelin →[SMPD1]→ ceramide →[ASAH1]→ sphingosine + fatty acid; SMPD1 (sphingomyelinase) generates ceramide from sphingomyelin; ASAH1 (acid ceramidase) catabolises ceramide to sphingosine; ASAH1 LOF = ceramide accumulation (catabolism block); SMPD1 LOF = ceramide deficiency + sphingomyelin accumulation; opposite biochemical consequences"},
    {"term": "POLG1 Exclusion Before VPA", "definition": "MANDATORY: exclude POLG1 (mtDNA polymerase gamma) pathogenic variants before initiating VPA in any LSD patient presenting with epilepsy; POLG1 + VPA = fulminant hepatic failure (CPIC-POLG1-2023 Level A); ASAH1 is NOT mitochondrial (standard ceramide-pathway LSD) — POLG1 less likely than in mitochondrial diseases but standard screen required"},
    {"term": "Piracetam Level C — Myoclonus PME Pattern", "definition": "Piracetam used as adjunct for action myoclonus in PME-like Type 5 Farber; evidence extrapolated from Level A data in Unverricht-Lundborg (CSTB) and MERRF; mechanism: AMPA receptor modulation + cortical hyperexcitability reduction; off-label in Farber; renal dosing adjustment required (renally cleared)"},
    {"term": "AR Biallelic LOF / 8p22", "definition": "ASAH1 inheritance: autosomal recessive (AR); both alleles must carry pathogenic variants (biallelic LOF); chromosome 8p22; no pan-ethnic founder mutation (unlike NPC/HEXA/SMPD1 founder mutations); pan-ethnic disease; most variants are private/family-specific missense; heterozygous carriers: asymptomatic"},
]


# ---------------------------------------------------------------------------
# API RESPONSE BUILDERS
# ---------------------------------------------------------------------------

def get_overview():
    return {
        "disease": (
            "Farber lipogranulomatosis (Farber disease / Acid Ceramidase Deficiency) is an "
            "autosomal-recessive lysosomal sphingolipidosis caused by biallelic loss-of-function "
            "variants in ASAH1 (8p22), encoding Acid Ceramidase (AC). ASAH1 LOF results in "
            "lysosomal ceramide accumulation → lipid-laden macrophages (foam cells) → "
            "granulomatous infiltration of periarticular tissue, skin, larynx, liver, spleen, "
            "and CNS — producing the pathognomonic lipogranuloma triad: periarticular subcutaneous "
            "nodules + progressive joint contractures + hoarse cry. Seven subtypes are recognised "
            "(Levade classification): Type 1 (classic intermediate, most common, ~50%); Type 2 "
            "(intermediate, milder, adult survival); Type 3 (mildest, near-normal lifespan); "
            "Type 4 (neonatal-visceral, hydrops, fatal weeks); Type 5 (neurological progressive, "
            "infantile CNS-dominant, PME-like epilepsy, fatal 2–5 years); Type 6 (combined Farber "
            "+ Sandhoff/HEXB); Type 7 (PSAP/Saposin D phenocopy — normal ASAH1, PSAP biallelic). "
            "Ceramide is the central signalling lipid: SMPD1 generates ceramide from sphingomyelin "
            "(upstream); ASAH1 catabolises ceramide to sphingosine (downstream); CERS1-6 synthesise "
            "ceramide de novo. No approved ERT exists for Farber disease (rhAC Phase 1 trial "
            "ongoing) — the critical management gap versus other LSDs. HSCT (Level B) attenuates "
            "non-CNS manifestations in Types 1–3 when performed early; does NOT benefit CNS. "
            "Farber disease is the rarest LSD (~150–200 reported cases worldwide); pan-ethnic; "
            "no founder mutation."
        ),
        "gene": "ASAH1 (Acid Ceramidase 1 / N-acylsphingosine amidohydrolase) — 395 aa, ~55 kDa",
        "locus": LOCUS,
        "omim": OMIM,
        "inheritance": INHERITANCE,
        "protein": (
            "Acid Ceramidase (AC) — lysosomal N-acylsphingosine amidohydrolase; cleaves ceramide "
            "(N-acylsphingosine) to sphingosine + fatty acid; active at lysosomal pH 4.5–5.0; "
            "395 amino acids, ~55 kDa; heterodimer (α + β chains post-processing); "
            "requires Saposin D (PSAP) as cofactor in lysosomes. Enzyme activity expressed as "
            "nmol/h/mg protein in leukocytes (4-MU-ceramide substrate): <10% of controls = diagnostic."
        ),
        "mechanism": (
            "ASAH1 LOF → (1) ceramide accumulates in lysosomes of macrophages, Schwann cells, "
            "and neurons (ceramide = pro-apoptotic + pro-inflammatory signalling lipid); "
            "(2) lipid-laden macrophages (foam cells) form granulomas in periarticular tissue, "
            "skin/subcutaneous, larynx, liver, spleen, and bone marrow; (3) granulomas compress "
            "and destroy articular cartilage and connective tissue → contractures; (4) laryngeal "
            "granulomas → hoarse cry + airway compromise; (5) Type 5: neuronal ceramide "
            "accumulation → Purkinje cell + motor neuron apoptosis → progressive neurodegeneration "
            "+ epileptic encephalopathy; (6) Ceramide activates: CAPP/PP2A (apoptosis), CAPK "
            "(stress kinase), cathepsin D release (lysosomal membrane permeabilisation), "
            "pro-inflammatory cytokines (TNF-α, IL-1β); (7) no CNS ERT penetration → CNS "
            "involvement untreatable with HSCT or current ERT."
        ),
        "cohort_size": COHORT_SIZE,
        "mean_onset_months": 5.2,
        "seizure_pct_overall": 42,
        "seizure_pct_type5": 88,
        "infantile_spasms_pct": 32,
        "drug_resistant_pct": 70,
        "lipogranuloma_triad_pct": 90,
        "farber_bodies_em_pct": 85,
        "hepatomegaly_pct": 72,
        "on_hsct_pct": 22,
        "on_acth_pct": 28,
        "on_lev_pct": 60,
        "on_vpa_pct": 45,
        "mean_diagnosis_delay_years": 2.4,
        "ac_enzyme_activity_diagnostic_threshold_pct": 10,
        "global_cases_estimate": 200,
        "pathognomonic_triad_note": (
            "LIPOGRANULOMA TRIAD (PATHOGNOMONIC, ~90% of Types 1–3): (1) Periarticular "
            "subcutaneous nodules over PIP joints, wrists, ankles, knees, spine — firm, "
            "tender, enlarging lipogranulomas; (2) Progressive joint contractures with fixed "
            "deformity + severe pain; (3) Hoarse/weak cry from laryngeal granulomas — unique "
            "among all LSDs; voice change + feeding difficulty + stridor. Triad may be "
            "incomplete in Type 5 (CNS-dominant: mild nodules + severe epilepsy) and absent "
            "in Type 4 (neonatal, fatal too rapidly)."
        ),
        "farber_bodies_note": (
            "FARBER BODIES ON EM (PATHOGNOMONIC, ~85%): 'Banana-shaped' / 'comma-shaped' / "
            "curvilinear lysosomal inclusions on electron microscopy of skin biopsy, nerve "
            "biopsy, or conjunctival biopsy; found in macrophages, Schwann cells, neurons; "
            "represent ceramide-rich lysosomal storage material in distinctive ultrastructural "
            "pattern; PATHOGNOMONIC ultrastructural finding — no other LSD shows this morphology."
        ),
        "no_ert_note": (
            "NO APPROVED ERT (CRITICAL GAP): Farber disease is unique among sphingolipidoses "
            "in lacking an approved enzyme replacement therapy as of 2024. Unlike SMPD1 "
            "(olipudase alfa/Xenpozyme FDA 2022), GLA (agalsidase alfa/beta), GALC, and others, "
            "ASAH1 has no approved ERT. Recombinant human AC (rhAC) is in Phase 1 clinical trial. "
            "This gap is the most critical unmet need in Farber disease management."
        ),
        "etiologies": ETIOLOGIES,
        "standards": STANDARDS[:6],
    }


def get_breakdown():
    return {
        "cohort_size": COHORT_SIZE,
        "etiologies": ETIOLOGIES,
        "seizure_types": SEIZURE_TYPES,
        "triggers": TRIGGERS,
        "treatments": TREATMENTS,
        "contraindications": CONTRAINDICATIONS,
        "thresholds": THRESHOLDS,
        "standards": STANDARDS,
        "ceramide_pathway": {
            "upstream_smpd1": "Sphingomyelin →[SMPD1 (11p15.4)]→ Ceramide + Phosphocholine",
            "cers_synthesis": "Sphinganine →[CERS1-6]→ Dihydroceramide →[DEGS1]→ Ceramide (de novo)",
            "asah1_catabolism": "Ceramide →[ASAH1 (8p22)]→ Sphingosine + Fatty Acid (BLOCKED in Farber)",
            "downstream_sphk": "Sphingosine →[SPHK1/SPHK2]→ Sphingosine-1-Phosphate (S1P, pro-survival)",
            "downstream_ugcg": "Ceramide →[UGCG]→ Glucosylceramide (substrate for GBA)",
            "saposin_d": "Saposin D (PSAP 10q22.1) — required cofactor for ASAH1 lysosomal activity; deficiency = Type 7 phenocopy",
            "clinical_implication": (
                "ASAH1 block: ceramide accumulates (pro-apoptotic) + sphingosine deficiency "
                "(pro-survival S1P cannot be generated adequately); dual pathological consequence. "
                "Ceramide-directed therapies must avoid further upstream ceramide production (avoid "
                "typical antipsychotics that activate SMPD1 → more ceramide)."
            ),
        },
        "subtype_seizure_summary": {
            "Type 1": "~20% seizures; focal + GTCS; drug-responsive in most",
            "Type 2": "<5% seizures; rare; mild neurological",
            "Type 3": "0% seizures; no neurological involvement",
            "Type 4": "0% seizures; neonatal fatal (too brief for seizure development)",
            "Type 5": "80–90% seizures; IS + PME-pattern + focal + GTCS; 70% drug-resistant",
            "Type 7 (PSAP)": "Similar to Type 1 phenocopy; ~20% seizures",
        },
        "treatment_hierarchy": [
            "1. IS (Type 5): ACTH Level A → (LEV adjunct) → Clobazam Level C",
            "2. Focal seizures: LEV Level B → Clobazam C (adjunct) → VPA B (with hepatic monitoring)",
            "3. GTCS / PME-pattern: VPA Level B (POLG1 excl + LFT 3-monthly) → LEV B → Piracetam C (myoclonus)",
            "4. Drug-resistant: KD Level B → Piracetam C → clinical trial referral",
            "5. HSCT: Types 1–3 early (non-CNS benefit only); CNS involvement = relative CI",
            "6. ERT: No approved ERT; refer to rhAC trial if eligible",
        ],
    }


def get_definitions():
    return {
        "cohort_size": COHORT_SIZE,
        "gene": GENE,
        "locus": LOCUS,
        "omim": OMIM,
        "inheritance": INHERITANCE,
        "key_concepts": KEY_CONCEPTS,
        "thresholds": THRESHOLDS,
        "standards": STANDARDS,
        "diagnostic_algorithm": [
            "Step 1: Clinical suspicion — periarticular nodules + joint contractures + hoarse cry (triad) OR infantile spasms + PME-like epilepsy + foam cells",
            "Step 2: Acid ceramidase (AC) enzyme activity in leukocytes or fibroblasts — <10% of controls = biochemical diagnosis",
            "Step 3: ASAH1 molecular sequencing (NGS gene panel or WGS) — biallelic pathogenic variants confirm ASAH1 Farber Types 1–6",
            "Step 4: If AC enzyme <10% but ASAH1 sequencing normal → PSAP sequencing + Saposin D protein assay → Type 7 (PSAP/Saposin D phenocopy) diagnosis",
            "Step 5: Skin/nerve biopsy EM — Farber bodies (banana-shaped curvilinear inclusions) = PATHOGNOMONIC ultrastructural confirmation",
            "Step 6: Plasma ceramide profiling (LC-MS/MS) — elevated C16:0/C18:0/C24:1 = supportive; evolving monitoring biomarker",
            "Step 7: Clinical subtype assignment (Types 1–7) — determines HSCT candidacy, prognosis, epilepsy management plan",
        ],
        "ceramide_pathway_glossary": {
            "Ceramide": "N-acylsphingosine; central sphingolipid signalling molecule; pro-apoptotic + pro-inflammatory; generated by SMPD1 (from sphingomyelin) and CERS1-6 (de novo); catabolised by ASAH1",
            "Acid Ceramidase (ASAH1)": "Lysosomal N-acylsphingosine amidohydrolase; cleaves ceramide → sphingosine + fatty acid; requires Saposin D cofactor; deficient in Farber disease",
            "Saposin D (PSAP)": "Activator protein for ASAH1 in lysosomes; encoded by PSAP gene (10q22.1); deficiency = Type 7 Farber phenocopy (normal ASAH1 sequence)",
            "Sphingosine": "Product of ASAH1 reaction; converted to S1P (sphingosine-1-phosphate) by SPHK1/SPHK2; S1P is pro-survival; deficient in Farber disease",
            "Lipogranuloma": "Granulomatous lesion rich in ceramide-laden macrophages (foam cells); formed in periarticular tissue, skin, larynx; PATHOGNOMONIC macroscopic finding",
            "Farber bodies": "Ultrastructural: banana-shaped/comma-shaped curvilinear lysosomal inclusions in macrophages/Schwann cells on EM; PATHOGNOMONIC for Farber disease",
        },
    }
