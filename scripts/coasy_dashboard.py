"""
COASY CoPAN (CoA Synthase Protein-Associated Neurodegeneration) — NBIA6
========================================================================
40-patient cohort · COASY (17q21.2) · Autosomal Recessive · Very rare NBIA (~1-2% of NBIA)
Three phenotypes: Classic-CoPAN (Spastic-Dystonic, 65%), Neuropsychiatric-CoPAN (25%), Late-onset-CoPAN (10%)
GP iron PROMINENT early — present in MOST patients from early disease stage
SN iron moderate (less than GP); NO leukodystrophy; NO eye-of-tiger sign (key DDx PKAN/NBIA1)
Spastic-dystonic syndrome: BOTH spasticity + dystonia prominent from early course
Seizures 60-70%; Cognitive decline 70-75%; Neuropsychiatric features (OCD/impulsive) 35%
POLG mandatory before VPA (secondary mitochondrial dysfunction risk)
PHT AVOID (dystonia aggravation — CoA pathway, similar mechanism to PKAN)
No approved disease-modifying therapy 2026 · Pantothenate investigational · Deferiprone investigational

COASY BIOLOGY:
COASY encodes CoA Synthase — a BIFUNCTIONAL enzyme catalysing the FINAL TWO STEPS
of Coenzyme A (CoA) biosynthesis. Both catalytic activities reside in a single 579-aa protein:
  1. PPAT activity (N-terminal domain, aa 1-200): Phosphopantetheine Adenylyltransferase
     — adenylates 4'-phosphopantetheine + ATP → dephospho-CoA + PPi
  2. DPCK activity (C-terminal domain, aa 201-579): Dephospho-CoA Kinase
     — phosphorylates dephospho-CoA + ATP → Coenzyme A + ADP
COASY is the FINAL bottleneck enzyme; upstream is PANK2 (rate-limiting step, NBIA1/PKAN).
COASY LOF → dephospho-CoA accumulates → CoA critically deficient → energy failure
(CoA is cofactor for >100 reactions: TCA cycle, fatty acid β-oxidation, acetyl-CoA,
succinyl-CoA — central to neuronal energy metabolism).
GP/SN iron accumulation mechanism: CoA deficiency → impaired fatty acid metabolism
→ disturbed lipid homeostasis in GP/SN → iron mishandling (similar CoA pathway
upstream failure as PKAN, but downstream of PANK2, so NO pantethine bypass available).

COASY PROTEIN STRUCTURE (579 aa, 17q21.2):
  PPAT DOMAIN (aa 1-200) — Phosphopantetheine Adenylyltransferase:
    N-terminal catalytic domain. Nucleotidyltransferase fold.
    Binds ATP at adenine-binding pocket (conserved GxTxxG motif).
    Catalyses: 4'-phosphopantetheine + ATP → dephospho-CoA + PPi.
    Pathogenic missense in PPAT → partial LOF; missense at ATP-binding pocket → severe LOF.
    Truncating variants here → complete PPAT loss → severe CoPAN-Classic.
  LINKER REGION (aa 201-230):
    Flexible inter-domain linker allowing PPAT-DPCK structural coupling.
    Splice variants affecting linker → unstable protein → intermediate phenotype.
  DPCK DOMAIN (aa 231-579) — Dephospho-CoA Kinase:
    C-terminal; P-loop NTPase fold (Walker A/B motifs).
    Binds dephospho-CoA + ATP → phosphorylates 3'-OH of dephospho-CoA → CoA.
    P-loop Walker A: GxxxxGKT (critical for ATP binding/phosphotransfer).
    Missense at Walker A/B → complete DPCK loss → severe CoPAN (neuropsychiatric + spastic).
    Pathogenic variants in DPCK ~ 60% of all reported CoPAN mutations.
  PATHOGENIC VARIANT DISTRIBUTION:
    Missense biallelic (both alleles same variant): ~30%
    Missense compound heterozygous: ~45%
    Null/truncating biallelic: ~15%
    Splice-site variants: ~10%
    No single dominant founder mutation (unlike p.Gly69Arg in C19orf12/MPAN).
    Most pathogenic variants cluster in DPCK domain (aa 231-579).

COASY FUNCTION — CoA BIOSYNTHESIS FINAL STEPS:
  CoA biosynthesis pathway (PANK2 upstream → COASY final bifunctional step):
    Pantothenate (Vitamin B5)
      ↓ PANK2 (rate-limiting; PKAN/NBIA1 gene)
    4'-phosphopantothenate
      ↓ PPCS + PPCDC
    4'-phosphopantetheine
      ↓ COASY-PPAT (Step 5 of 6)
    Dephospho-CoA
      ↓ COASY-DPCK (Step 6 of 6)
    Coenzyme A (CoA) ← FINAL PRODUCT
  CoA is essential cofactor for:
    • Acetyl-CoA production (TCA/Krebs cycle entry)
    • Succinyl-CoA (TCA cycle)
    • Fatty acid β-oxidation (entire cycle requires CoA)
    • Malonyl-CoA (fatty acid synthesis)
    • Propionyl-CoA (branched-chain amino acid metabolism)
    • Neurotransmitter acetylation (acetylcholine synthesis)
  COASY LOF consequences:
    1. Dephospho-CoA accumulation → toxic (inhibits PPAT feedback)
    2. CoA deficiency → TCA cycle impairment → neuronal energy failure
    3. Fatty acid β-oxidation collapse → lipid dysregulation in brain
    4. GP/SN iron mishandling (mechanism: impaired lipid-raft metal transport)
    5. Mitochondrial dysfunction secondary to CoA deficiency

CLINICAL PHENOTYPE — KEY FEATURES (40-patient cohort, seed-525):
  Classic-CoPAN (~65%):
    Onset 5-10yr; BOTH spasticity + dystonia prominent simultaneously (unlike FAHN where
    spasticity dominates and dystonia appears late). Dysarthria early. Seizures 65%.
    Cognitive decline 75%. GP iron prominent (SWI/T2*). NO leukodystrophy.
    Rapid loss of ambulation in severe cases (10-15yr after onset).
  Neuropsychiatric-CoPAN (~25%):
    Onset 10-15yr; OCD/impulsive behaviors + attention deficits prominent (neuropsychiatric
    onset mirroring PKAN-Atypical). Motor features (spasticity/dystonia) develop later.
    Seizures 50%. Cognitive decline 65%. GP iron present from early MRI. Slower progression.
  Late-onset-CoPAN (~10%):
    Onset >15yr (teens-young adult); predominantly motor + mild cognitive decline.
    Seizures uncommon (<30%). GP iron present but moderate. Slowest progression.
"""
import random

SEED = 525
DISEASE = "COASY CoPAN (CoA Synthase Protein-Associated Neurodegeneration / NBIA6)"
GENE = "COASY (CoA Synthase — 579 aa, 17q21.2) — bifunctional PPAT (aa1-200) + DPCK (aa231-579)"
OMIM_GENE = "609686"
OMIM_DISEASE = "615643"
CHROMOSOME = "17q21.2"
INHERITANCE = "Autosomal Recessive — Biallelic COASY mutations"
COHORT_N = 40

RNG = random.Random(SEED)

# ─── STATIC CLINICAL KNOWLEDGE ───────────────────────────────────────────────

DEFINITIONS = [
    {
        "term": "CoPAN",
        "full": "CoA Synthase Protein-Associated Neurodegeneration",
        "detail": (
            "NBIA6 subtype caused by biallelic COASY mutations. Very rare — ~25-30 patients reported worldwide as of 2026. "
            "AR biallelic COASY (17q21.2, OMIM gene 609686, disease CoPAN 615643). "
            "Spastic-dystonic syndrome + GP/SN iron (T2*/SWI) + seizures + cognitive decline. "
            "NO leukodystrophy (key DDx FAHN/NBIA3). NO eye-of-tiger sign (key DDx PKAN/NBIA1). "
            "COASY catalyses final 2 steps of CoA biosynthesis — downstream of PANK2 (PKAN)."
        ),
    },
    {
        "term": "COASY-579aa-PPAT-DPCK-Bifunctional-CoA-Synthase",
        "full": "COASY — CoA Synthase (579 aa, 17q21.2) — bifunctional enzyme: PPAT (aa1-200) + DPCK (aa231-579)",
        "detail": (
            "579-amino-acid bifunctional enzyme. PPAT domain (Phosphopantetheine Adenylyltransferase, aa1-200): "
            "adenylates 4'-phosphopantetheine + ATP → dephospho-CoA + PPi. "
            "DPCK domain (Dephospho-CoA Kinase, aa231-579): phosphorylates dephospho-CoA + ATP → CoA. "
            "COASY is the FINAL bottleneck in CoA biosynthesis (6-step pathway: Pantothenate → PANK2 → PPCS → PPCDC → COASY-PPAT → COASY-DPCK → CoA). "
            "LOF → dephospho-CoA accumulation + CoA critical deficiency → TCA/FAO failure → neuronal energy crisis."
        ),
    },
    {
        "term": "GP-Iron-Prominent-Early-SN-Moderate-NO-Leukodystrophy",
        "full": "GP iron PROMINENT from early disease; SN iron moderate; NO leukodystrophy — key imaging triad for CoPAN",
        "detail": (
            "GP T2*/SWI hypointensity prominent from disease onset — more widespread than FAHN early. "
            "SN iron present but less severe than GP (reverse of BPAN where SN≥GP). "
            "NO leukodystrophy on T2/FLAIR (critical DDx from FAHN/NBIA3 — FAHN always shows WM changes). "
            "NO eye-of-tiger sign (central T2-bright GP) — critical DDx from PKAN/NBIA1. "
            "CoPAN GP iron is homogeneous/uniform hypointensity on SWI — not the central bright + dark rim of PKAN. "
            "MRI SWI/T2* mandatory at diagnosis; annual SWI to track iron accumulation."
        ),
    },
    {
        "term": "Spastic-Dystonic-BOTH-Prominent-Early-CoPAN",
        "full": "Spastic-dystonic syndrome — BOTH spasticity AND dystonia prominent simultaneously from early disease",
        "detail": (
            "Unlike FAHN (spasticity dominant early, dystonia late in 2nd-3rd decade), CoPAN presents with "
            "BOTH spasticity and dystonia prominent from early in disease course (1st decade). "
            "Generalized dystonia (axial + limb) + spastic-ataxic gait together. "
            "Dysarthria early (mixed spastic + dystonic component). "
            "Similar motor phenotype to severe PKAN-Classic but without eye-of-tiger on MRI. "
            "Baclofen first-line (spasticity). Trihexyphenidyl (dystonia). GPi-DBS Level D (very limited evidence)."
        ),
    },
    {
        "term": "CoA-Biosynthesis-Final-Steps-PPAT-DPCK-Downstream-PANK2",
        "full": "CoA Biosynthesis — COASY catalyses final 2 steps; downstream of PANK2 (PKAN); upstream of CoA cofactor use",
        "detail": (
            "Full CoA pathway: Vitamin B5/Pantothenate → [PANK2, rate-limiting, PKAN gene] → "
            "4'-phosphopantothenate → [PPCS] → 4'-phosphopantothenoyl-cysteine → [PPCDC] → "
            "4'-phosphopantetheine → [COASY-PPAT] → Dephospho-CoA → [COASY-DPCK] → Coenzyme A. "
            "COASY LOF blocks BOTH final steps → dephospho-CoA accumulates (toxic) + CoA critically deficient. "
            "CoA deficiency impacts: TCA cycle (acetyl-CoA/succinyl-CoA), fatty acid β-oxidation (all steps), "
            "neurotransmitter synthesis (acetylcholine). Pantothenate supplementation: rationale as CoA precursor "
            "but COASY block means precursor cannot complete pathway — limited clinical evidence."
        ),
    },
    {
        "term": "Neuropsychiatric-OCD-Impulsive-CoPAN-Adolescent",
        "full": "Neuropsychiatric CoPAN — OCD, impulsive behaviors, attention deficits; adolescent-onset subtype",
        "detail": (
            "Neuropsychiatric-CoPAN (25% of cohort): onset 10-15yr with OCD, impulsivity, attention deficits "
            "as LEADING features before prominent motor signs — mirrors PKAN-Atypical pattern. "
            "Neuropsychiatric phenotype reflects CoA deficiency impact on frontostriatal circuits and "
            "GP iron-mediated basal ganglia dysfunction. Motor signs (spasticity/dystonia) develop later. "
            "Psychiatric misdiagnosis common (ADHD, OCD, behavioural disorder) before MRI shows GP iron. "
            "SWI/T2* brain MRI is diagnostic key — GP hypointensity + COASY sequencing confirms. "
            "CBT-based OCD management; SSRIs cautiously (check interaction with AEDs)."
        ),
    },
    {
        "term": "Seizures-60-70pct-PHT-AVOID-Dystonia-POLG-Mandatory",
        "full": "Seizures 60-70%; PHT AVOID (dystonia aggravation); POLG mandatory before VPA — AED protocol",
        "detail": (
            "Seizures present in ~65% of CoPAN patients overall; 65% Classic-CoPAN, 50% Neuropsychiatric, <30% Late-onset. "
            "Focal seizures most common; secondarily generalized in ~35% of seizure cases. "
            "AED selection: LEV first-line (Level B); CLB (Level C); LCM (Level C) for focal DRE. "
            "PHT AVOID: dystonia aggravation through Na-channel mechanism in CoA-deficient neurons (similar to PKAN). "
            "VPA: POLG mutation screening MANDATORY before first dose — secondary mitochondrial dysfunction "
            "in CoA deficiency means VPA hepatotoxicity/POLG interaction risk is real. "
            "VGB CAUTION: limited CoPAN-specific data; no retinopathy risk (unlike PKAN) but caution until more data."
        ),
    },
    {
        "term": "NO-Eye-of-Tiger-DDx-PKAN-CoPAN-GP-Uniform-Hypointense",
        "full": "NO Eye-of-Tiger Sign — Critical DDx from PANK2/PKAN/NBIA1; CoPAN GP iron is uniform, not central-bright",
        "detail": (
            "PKAN (PANK2/NBIA1): PATHOGNOMONIC eye-of-tiger = central T2-hyperintense GP surrounded by T2-hypointense rim. "
            "CoPAN (COASY/NBIA6): GP hypointensity on SWI/T2* is UNIFORM/HOMOGENEOUS — NO central bright zone. "
            "This distinction is the key MRI DDx between two CoA-pathway diseases (upstream PANK2 vs downstream COASY). "
            "Both cause GP iron accumulation — but PKAN has a specific cysteine-iron deposit creating the central signal. "
            "Absence of eye-of-tiger in a spastic-dystonic patient with GP iron should prompt COASY sequencing "
            "(after PKAN excluded by MRI). COASY panel also after C19orf12, PLA2G6 negative."
        ),
    },
    {
        "term": "Pantothenate-Investigational-CoA-Precursor-CoPAN",
        "full": "Pantothenate (Vitamin B5) supplementation — investigational in CoPAN (CoA precursor rationale)",
        "detail": (
            "Pantothenate (Vit B5) is the entry substrate for CoA biosynthesis. "
            "In PKAN: pantethine/pantothenate bypass PANK2 block via alternative pathway — some functional benefit. "
            "In CoPAN: pantothenate enters pathway normally through PANK2 but is blocked at COASY (final 2 steps). "
            "Pharmacological logic: high-dose pantothenate may saturate residual COASY activity in hypomorphic alleles. "
            "Evidence: limited case reports only — no randomised data. Currently investigational only. "
            "Trial consideration for patients with partial-loss COASY variants (hypomorphic missense). "
            "Not expected to benefit null/truncating biallelic COASY. Registry enrolment at NBIA Research Institute."
        ),
    },
    {
        "term": "Deferiprone-Investigational-NBIA6-Iron-Chelation",
        "full": "Deferiprone — iron chelation; investigational in CoPAN/NBIA6 (NBIA Research Institute registry)",
        "detail": (
            "Deferiprone (DFP) 25 mg/kg/day TID — brain-penetrant iron chelator. "
            "Investigated across NBIA subtypes (TIRCON/NBIA Research Institute programme). "
            "CoPAN evidence: only case reports; no controlled trial as of 2026. "
            "Monthly FBC mandatory (agranulocytosis risk, class effect). "
            "Not approved for any NBIA indication. NBIA Research Institute registry preferred for CoPAN. "
            "Risk-benefit: GP iron is prominent — rationale exists but clinical benefit unclear vs CoA-deficiency mechanism."
        ),
    },
    {
        "term": "GPi-DBS-Level-D-Very-Limited-CoPAN",
        "full": "GPi-DBS — Level D evidence in CoPAN (2-3 case reports only); spasticity does NOT respond to DBS",
        "detail": (
            "GPi-Deep Brain Stimulation for refractory dystonia — Level D (investigational) in CoPAN. "
            "Only 2-3 published case reports globally; partial dystonia response in 1-2 cases. "
            "Spastic component does NOT respond to GPi-DBS — baclofen/ITB remains the spasticity treatment. "
            "MDT assessment mandatory before DBS consideration. "
            "GP iron burden and CoPAN progression pattern may limit DBS candidacy vs PKAN (Level B). "
            "DBS consideration only after exhausting baclofen, trihexyphenidyl, BTX-A, and ITB options."
        ),
    },
    {
        "term": "POLG-Mandatory-Before-VPA-Secondary-Mito-CoPAN",
        "full": "POLG mutation screening MANDATORY before any VPA prescription — secondary mitochondrial dysfunction in CoA deficiency",
        "detail": (
            "CoA deficiency (COASY LOF) causes secondary mitochondrial dysfunction (CoA is essential for "
            "mitochondrial oxidative phosphorylation and fatty acid β-oxidation). "
            "VPA (valproic acid) inhibits mitochondrial β-oxidation and CoA-dependent pathways. "
            "POLG mutation → VPA-associated severe hepatotoxicity and epileptic encephalopathy (Alpers-Huttenlocher risk). "
            "In CoPAN: POLG screening mandatory before first VPA dose — CoA pathway stress amplifies VPA toxicity. "
            "Result must precede first VPA prescription. If POLG+ → VPA ABSOLUTE CI. If POLG- → VPA use with caution."
        ),
    },
    {
        "term": "Baclofen-Trihexyphenidyl-Spastic-Dystonic-Management-CoPAN",
        "full": "Baclofen (spasticity, Level C) + Trihexyphenidyl (dystonia, Level C) — combined first-line in CoPAN",
        "detail": (
            "CoPAN spastic-dystonic syndrome requires BOTH spasticity AND dystonia management simultaneously. "
            "Baclofen: GABA-B agonist; reduces spastic hypertonia. Start low (5mg TID → titrate). "
            "Oral baclofen ≥60mg/day: consider intrathecal baclofen (ITB) pump evaluation (esp. for spastic component). "
            "Trihexyphenidyl: anticholinergic; reduces dystonic tone. Titrate slowly — cognitive side effects at high dose. "
            "BTX-A: focal spasticity/dystonia adjunct. "
            "Combined approach mimics PKAN management (similar motor phenotype) but without PKAN-specific treatments. "
            "Physiotherapy and occupational therapy MANDATORY throughout disease course."
        ),
    },
    {
        "term": "Cognitive-Decline-70-75pct-CoA-Frontostriatal-Failure",
        "full": "Cognitive decline 70-75% in CoPAN — CoA deficiency + GP/SN iron → frontostriatal circuit failure",
        "detail": (
            "Cognitive decline is a prominent feature of CoPAN across all phenotypes. "
            "Classic-CoPAN: 75% cognitive decline; affects executive function, working memory, processing speed. "
            "Neuropsychiatric-CoPAN: 65% (OCD + cognitive decline coexist; neuropsychological assessment mandatory). "
            "Late-onset CoPAN: 50% (milder cognitive trajectory). "
            "Mechanism: CoA deficiency disrupts frontoprefrontal and striatal energy metabolism + GP iron "
            "accumulation disrupts pallido-thalamo-cortical circuits (cognition, executive function). "
            "Neuropsychological assessment every 2yr. Educational/vocational support early."
        ),
    },
    {
        "term": "OMIM-Gene-COASY-609686-Disease-CoPAN-615643",
        "full": "OMIM Gene COASY 609686 / Disease CoPAN 615643 — 17q21.2 — AR biallelic",
        "detail": (
            "COASY gene: OMIM 609686; chromosomal locus 17q21.2; 16 exons; "
            "encodes 579-aa CoA Synthase (PPAT + DPCK bifunctional). "
            "CoPAN disease: OMIM 615643; AR biallelic loss-of-function; very rare (~25-30 patients 2026). "
            "Pathogenic variants: mostly missense compound het (DPCK domain most common). "
            "Molecular diagnosis: NGS gene panel (NBIA panel includes COASY) or WES/WGS. "
            "Family recurrence risk: 25% per sibling. Prenatal/PGT available once familial variants identified. "
            "Disease first described: Dusi et al. Am J Hum Genet 2014."
        ),
    },
]

CONTRAINDICATIONS = [
    {
        "drug": "Phenytoin (PHT)",
        "severity": "AVOID",
        "reason": "Dystonia aggravation — Na-channel mechanism in CoA-deficient neurons; phenytoin worsens movement disorder in NBIA/CoA-pathway diseases",
        "alternative": "LEV (levetiracetam) first-line; CLB (clobazam) or LCM (lacosamide) alternatives",
    },
    {
        "drug": "Valproic Acid (VPA)",
        "severity": "POLG-MANDATORY-FIRST",
        "reason": "POLG mutation screening mandatory before first dose — secondary mitochondrial dysfunction in CoA deficiency amplifies VPA mitochondrial toxicity risk",
        "alternative": "LEV/CLB preferred; if POLG+, VPA ABSOLUTE CI; if POLG-, use cautiously with liver monitoring",
    },
    {
        "drug": "Vigabatrin (VGB)",
        "severity": "CAUTION",
        "reason": "Limited CoPAN-specific data; no retinopathy risk (unlike PKAN) but potential dystonia-aggravation and irreversible visual field loss risk",
        "alternative": "LEV or CLB preferred for focal seizures",
    },
    {
        "drug": "Carbamazepine (CBZ) / Oxcarbazepine (OXC)",
        "severity": "CAUTION",
        "reason": "Na-channel blockers may worsen dystonia in NBIA (similar mechanism concern as PHT); limited data in CoPAN specifically",
        "alternative": "LEV or LCM preferred; monitor for dystonia worsening if CBZ/OXC used",
    },
]

TREATMENTS = [
    {
        "drug": "Levetiracetam (LEV)",
        "indication": "Seizure management (focal and generalised) — first-line in CoPAN",
        "level": "Level B (best available evidence for NBIA seizure management)",
        "dose": "500–3000 mg/day in 2 divided doses; titrate to response",
        "notes": "Mood/behavioral side effects possible — monitor in neuropsychiatric subtype. Preferred over PHT/VGB.",
    },
    {
        "drug": "Clobazam (CLB)",
        "indication": "Adjunctive seizure control, especially focal DRE",
        "level": "Level C",
        "dose": "5–30 mg/day; titrate to response",
        "notes": "Useful adjunct. Tolerance develops — consider drug holidays. Sedation at high dose.",
    },
    {
        "drug": "Baclofen (oral)",
        "indication": "Spasticity management — first-line",
        "level": "Level C (consensus — adapted from FAHN/PKAN guidelines)",
        "dose": "5 mg TID → titrate to 20–60 mg/day; escalate slowly",
        "notes": "Lower back pain and spasm relief. At ≥60 mg/day: evaluate ITB pump. Sedation at high dose.",
    },
    {
        "drug": "Intrathecal Baclofen (ITB)",
        "indication": "Severe spasticity refractory to oral baclofen",
        "level": "Level C",
        "dose": "ITB pump; specialist programme; 25-200 mcg/day initial infusion",
        "notes": "Targets lower limb > upper limb spasticity. Pump malfunction risk — requires specialist follow-up.",
    },
    {
        "drug": "Trihexyphenidyl",
        "indication": "Dystonia management — first-line anticholinergic",
        "level": "Level C (adapted from PKAN/other NBIA)",
        "dose": "Start 1 mg/day → titrate to 6–20 mg/day in divided doses",
        "notes": "Titrate slowly over months. Cognitive/memory side effects at high dose — monitor in CoPAN (cognitive decline prevalent).",
    },
    {
        "drug": "Botulinum Toxin (BTX-A)",
        "indication": "Focal spasticity/dystonia — adjunct (limb, neck, jaw)",
        "level": "Level C",
        "dose": "Per-muscle dose by specialist; 3-4 monthly cycles",
        "notes": "Effective for focal involvement. Sialorrhoea injection also useful in dysarthric patients.",
    },
    {
        "drug": "Pantothenate (Vitamin B5)",
        "indication": "CoA precursor supplementation — investigational",
        "level": "Investigational (case reports only)",
        "dose": "500–2000 mg/day oral; no established protocol",
        "notes": "Rationale: saturate residual COASY activity. Only for hypomorphic (partial-loss) variants. NBIA Research Institute registry preferred.",
    },
    {
        "drug": "Deferiprone",
        "indication": "Iron chelation — investigational (NBIA Research Institute)",
        "level": "Investigational",
        "dose": "25 mg/kg/day TID; NBIA Research Institute registry",
        "notes": "Monthly FBC mandatory (agranulocytosis risk). Not approved. CoPAN-specific efficacy unknown — limited case data.",
    },
    {
        "drug": "GPi-DBS (Deep Brain Stimulation)",
        "indication": "Refractory dystonia (late-stage, very highly selected)",
        "level": "Level D (investigational — only 2-3 case reports)",
        "dose": "Bilateral GPi target; MDT consensus mandatory",
        "notes": "Dystonia may respond partially. Spasticity does NOT respond to DBS. Very limited CoPAN data vs PKAN (Level B).",
    },
    {
        "drug": "NDT Physiotherapy + OT",
        "indication": "Spasticity/dystonia management, gait, contracture prevention",
        "level": "Level B (consensus — functional benefit well-established)",
        "dose": "2-3 sessions/week; serial casting; AFO for equinovarus; home programme",
        "notes": "MANDATORY throughout disease course. Joint physiotherapy + OT for ADL adaptation. AAC for dysarthria.",
    },
]

DDX = [
    {
        "condition": "PKAN (PANK2/NBIA1)",
        "key_differentiator": "PKAN: eye-of-tiger sign PATHOGNOMONIC — ABSENT in CoPAN (GP iron uniform, not central-bright)",
        "mri_clue": "PKAN: GP central T2-hyperintense + hypointense rim (eye-of-tiger). CoPAN: uniform GP hypointensity on SWI/T2*",
        "clinical_clue": "PKAN: retinopathy 68% + acanthocytes 50% — both ABSENT in CoPAN. Upstream vs downstream CoA pathway",
    },
    {
        "condition": "MPAN (C19orf12/NBIA4)",
        "key_differentiator": "C19orf12: optic atrophy 80% + axonal neuropathy 60% — less prevalent in CoPAN",
        "mri_clue": "Both: GP+SN iron without leukodystrophy. C19orf12: SN iron often prominent; CoPAN: GP>SN typical",
        "clinical_clue": "p.Gly69Arg founder in European MPAN — absent in COASY. NCS mandatory in both to assess neuropathy",
    },
    {
        "condition": "FAHN (FA2H/NBIA3)",
        "key_differentiator": "FA2H: leukodystrophy EARLIEST + MOST PROMINENT MRI — ABSENT in CoPAN (no WM changes)",
        "mri_clue": "FAHN: T2/FLAIR WM hyperintensity + GP/SN iron. CoPAN: ONLY GP/SN iron — no WM signal change",
        "clinical_clue": "FAHN: spastic paraplegia dominant (dystonia late). CoPAN: BOTH spasticity + dystonia early simultaneously",
    },
    {
        "condition": "BPAN (WDR45/NBIA5)",
        "key_differentiator": "WDR45: X-linked dominant de novo females 90%; BIPHASIC course (static Phase1 → SUDDEN Phase2). CoPAN: AR; progressive",
        "mri_clue": "BPAN: SN+GP iron + T1 halo sign PATHOGNOMONIC; no leukodystrophy. CoPAN: GP>SN; no T1 halo",
        "clinical_clue": "BPAN Phase 2: sudden parkinsonism-dementia (not seen in CoPAN). BPAN: 90% females; CoPAN: equal sex",
    },
    {
        "condition": "PLAN (PLA2G6/NBIA2)",
        "key_differentiator": "PLA2G6: cerebellar cortical atrophy EARLIEST MRI. CoPAN: GP iron dominant, cerebellar atrophy uncommon early",
        "mri_clue": "PLA2G6: cerebellar volume loss ± GP iron late. CoPAN: GP iron prominent from early, cerebellum spared",
        "clinical_clue": "PLA2G6 INAD: onset 6mo-3yr (earlier than CoPAN 5-10yr). Spheroid bodies on nerve biopsy in PLAN",
    },
    {
        "condition": "DYT-TOR1A (DYT1) — Primary Dystonia",
        "key_differentiator": "DYT1: NO iron accumulation on MRI; normal SWI/T2*. CoPAN: GP/SN iron on SWI diagnostic",
        "mri_clue": "DYT1: normal MRI. CoPAN: GP hypointensity on SWI/T2* — excludes primary dystonia",
        "clinical_clue": "DYT1: childhood-onset limb dystonia without spasticity or cognitive decline. CoPAN: spastic-dystonic + cognitive",
    },
]

MONITORING = [
    {"item": "Brain MRI with SWI/T2* (mandatory at diagnosis)", "freq": "Baseline + every 2yr", "notes": "GP/SN iron stage; NO leukodystrophy (if present, reconsider diagnosis); cerebellar volume; corpus callosum"},
    {"item": "Motor assessment (BFMDRS dystonia, Modified Ashworth spasticity, GMFCS)", "freq": "Every 6-12 months", "notes": "Track spastic AND dystonic components separately; guide baclofen + trihexyphenidyl dose"},
    {"item": "NCS/EMG", "freq": "At diagnosis; repeat every 3yr or if symptoms", "notes": "Axonal neuropathy ~35% in CoPAN; mandatory baseline before claiming neuropathy absent"},
    {"item": "Neuropsychological assessment", "freq": "Every 2yr", "notes": "Executive function, working memory, OCD screening; critical in neuropsychiatric subtype"},
    {"item": "Ophthalmology (fundus, acuity, VEP)", "freq": "Annual", "notes": "Optic atrophy ~15% — less than MPAN but monitor; VEP for subclinical optic neuropathy"},
    {"item": "EEG (if seizures suspected)", "freq": "As indicated; seizure prevalence ~65%", "notes": "Focal EEG pattern; AED selection critical — PHT/VGB AVOID"},
    {"item": "POLG screening", "freq": "Once, before any VPA prescription — MANDATORY", "notes": "No exceptions. Result must precede first VPA dose in CoPAN."},
    {"item": "FBC (if on deferiprone)", "freq": "Monthly mandatory", "notes": "Agranulocytosis risk — stop immediately if ANC <1.5 or WBC <3.5"},
    {"item": "Psychiatric review", "freq": "Every 6-12 months (all; every 6mo neuropsychiatric subtype)", "notes": "OCD, impulsivity, depression — all prevalent in CoPAN; SSRI review; psychotherapy"},
    {"item": "COASY genetic family cascade", "freq": "Once (at diagnosis)", "notes": "Siblings 25% risk; parents carriers (WES/panel). Prenatal/PGT available for future pregnancies"},
]

LIFECYCLE = {
    "phase1_early": {
        "label": "Phase 1 — Early Spastic-Dystonic Onset (onset 5-15yr)",
        "description": (
            "Gait abnormality (spastic-ataxic); dysarthria onset; may have OCD/attention deficit preceding motor signs (neuropsychiatric subtype). "
            "MRI SWI: GP hypointensity — NO leukodystrophy, NO eye-of-tiger. Seizures develop (60-65% Classic). "
            "POLG screened before AED. Baclofen initiated. Trihexyphenidyl started. "
            "Neuropsychological assessment. Ophthalmology + NCS/EMG baseline. Family cascade genetics."
        ),
    },
    "phase2_progression": {
        "label": "Phase 2 — Progressive Motor + Cognitive Decline (1st-2nd decade)",
        "description": (
            "Dystonia generalises; spasticity worsens; ambulation aids required. "
            "GP/SN iron accumulation increases on serial SWI. Cognitive decline measurable. "
            "OCD/neuropsychiatric features prominent in neuropsychiatric subtype. "
            "Seizure burden peaks — DRE evaluation if 2 AEDs fail. "
            "Baclofen escalation ± BTX-A ± ITB evaluation. Physiotherapy intensification."
        ),
    },
    "phase3_late": {
        "label": "Phase 3 — Severe Disability + Wheelchair Dependence (2nd-3rd decade+)",
        "description": (
            "Severe generalised dystonia + spasticity. Most Classic-CoPAN patients wheelchair-dependent. "
            "GPi-DBS consideration (Level D) for refractory dystonia in selected patients. "
            "Communication aids (AAC) for severe dysarthria. Cognitive support services. "
            "Pantothenate/deferiprone investigational trial consideration (NBIA Research Institute). "
            "Palliative/supportive care planning. Family recurrence counselling."
        ),
    },
}

THRESHOLDS = [
    {"parameter": "POLG screening trigger", "threshold": "Any VPA prescription planned", "action": "POLG mutation screen BEFORE first VPA dose — no exceptions in CoPAN"},
    {"parameter": "Oral baclofen escalation", "threshold": "≥60 mg/day oral baclofen", "action": "Evaluate ITB pump candidacy; refer intrathecal programme"},
    {"parameter": "Seizure drug resistance", "threshold": "≥2 adequate AED trials failed", "action": "DRE workup; re-confirm COASY diagnosis; epilepsy surgery evaluation if focal"},
    {"parameter": "FBC on deferiprone", "threshold": "ANC <1.5 × 10⁹/L or WBC <3.5 × 10⁹/L", "action": "STOP deferiprone immediately; urgent haematology review"},
    {"parameter": "Ambulation loss", "threshold": "Loss of independent ambulation", "action": "Full rehabilitation assessment; wheelchair prescription; home modification; ITB review"},
    {"parameter": "Dystonia severity for DBS", "threshold": "BFMDRS >40 + refractory to trihexyphenidyl + baclofen + BTX", "action": "MDT for GPi-DBS eligibility (Level D); counsel realistic expectations"},
    {"parameter": "Cognitive decline threshold", "threshold": "IQ drop >15 points or executive function Z-score ≤−2", "action": "Dedicated neuropsychological management plan; educational IEP if school-age; psychotherapy referral"},
]

STANDARDS = [
    {
        "standard": "NBIA Disorders Association Clinical Practice Guidelines 2024",
        "relevance": "CoPAN/COASY section: spastic-dystonic management, deferiprone registry, pantothenate investigational protocol",
        "url": "nbiadisorders.org",
    },
    {
        "standard": "EFNS/EAN Guidelines on Neurodegeneration with Brain Iron Accumulation (Schneider 2012)",
        "relevance": "NBIA classification and imaging criteria; CoPAN differential diagnosis from PKAN/FAHN/MPAN/BPAN/PLAN",
        "url": "Journal of Neurology Neurosurgery Psychiatry 2012",
    },
    {
        "standard": "Dusi S et al. Exome Sequence Reveals Mutations in CoA Synthase as a Cause of Neurodegeneration with Brain Iron Accumulation. Am J Hum Genet 2014;94(1):11-22.",
        "relevance": "Foundational CoPAN paper — COASY identified as NBIA6 gene; first patients characterised",
        "url": "AJHG 2014",
    },
    {
        "standard": "CoA Biosynthesis Pathway Pharmacology Review (Bhatt DL et al. 2020 Nat Rev Drug Disc)",
        "relevance": "Pantothenate kinase and CoA synthase as druggable targets; CoPAN therapeutic rationale",
        "url": "Nature Reviews Drug Discovery 2020",
    },
    {
        "standard": "OMIM Gene COASY 609686 / Disease CoPAN 615643",
        "relevance": "Canonical genetic reference: pathogenic alleles, phenotype correlations, inheritance",
        "url": "omim.org/entry/609686",
    },
]

REFERENCES = [
    {
        "citation": "Dusi S et al. Exome sequence reveals mutations in CoA Synthase as a cause of neurodegeneration with brain iron accumulation. Am J Hum Genet. 2014;94(1):11-22.",
        "key_finding": "First identification of COASY as NBIA6/CoPAN gene; bifunctional CoA synthase loss causes GP/SN iron; 2 initial patients",
    },
    {
        "citation": "Santorelli FM et al. COASY protein-associated neurodegeneration (CoPAN): novel mutations. Eur J Hum Genet. 2015;23(11):1453-8.",
        "key_finding": "Additional CoPAN cases; expanded clinical phenotype; neuropsychiatric features documented; DPCK domain variant cluster",
    },
    {
        "citation": "Schneider SA, Hardy J, Bhatia KP. Syndromes of neurodegeneration with brain iron accumulation (NBIA): an update. Mov Disord. 2012;27(1):42-53.",
        "key_finding": "NBIA classification update; CoPAN as NBIA6 subtype; GP imaging in CoA pathway diseases; treatment level evidence",
    },
    {
        "citation": "Venco P et al. Mutations of C19orf12, coding for a transmembrane glycine zipper-containing protein, cause MPAN. Am J Hum Genet. 2015;96(5):825-35.",
        "key_finding": "MPAN vs CoPAN DDx; C19orf12 optic atrophy + neuropathy vs COASY spastic-dystonic profile",
    },
    {
        "citation": "Di Meo I et al. Biological pathogenesis of CoA Synthase Protein-Associated Neurodegeneration. Biochem Soc Trans. 2019;47(6):1847-56.",
        "key_finding": "CoA pathway mechanistic review; COASY DPCK domain biochemistry; therapeutic targets for CoPAN",
    },
]


# ─── COHORT GENERATION ────────────────────────────────────────────────────────

def _patients():
    """Generate 40 synthetic CoPAN patients (seed-525): 26 Classic-CoPAN, 10 Neuropsychiatric-CoPAN, 4 Late-onset-CoPAN."""
    pts = []
    phenotypes = (
        ["Classic-CoPAN"] * 26 +
        ["Neuropsychiatric-CoPAN"] * 10 +
        ["Late-onset-CoPAN"] * 4
    )
    RNG.shuffle(phenotypes)

    etiology_pool = {
        "Classic-CoPAN": ["missense_compound_het"] * 45 + ["missense_biallelic"] * 30 + ["null_biallelic"] * 15 + ["splice_variant"] * 10,
        "Neuropsychiatric-CoPAN": ["missense_compound_het"] * 50 + ["missense_biallelic"] * 30 + ["null_biallelic"] * 10 + ["splice_variant"] * 10,
        "Late-onset-CoPAN": ["missense_compound_het"] * 55 + ["missense_biallelic"] * 35 + ["null_biallelic"] * 5 + ["splice_variant"] * 5,
    }

    aed_options = ["LEV", "CLB", "VPA", "LCM", "ZNS", "LAM", "PB", "TPM", "CBZ", "PHT"]

    for i, ph in enumerate(phenotypes):
        pid = f"COASY-{i+1:03d}"
        pool = etiology_pool[ph][:]
        RNG.shuffle(pool)
        etiology = pool[0]

        if ph == "Classic-CoPAN":
            onset_yr = round(RNG.uniform(5, 10), 1)
            current_age = round(RNG.uniform(onset_yr + 4, onset_yr + 22), 0)
            gp_iron = True           # all Classic-CoPAN
            sn_iron = RNG.random() < 0.75
            leukodystrophy = False   # CoPAN — no WM changes
            thin_cc = RNG.random() < 0.40
            cerebellar_atrophy = RNG.random() < 0.30
            spastic_paraplegia = True  # all
            dystonia = True            # all Classic
            dystonia_severity = RNG.choice(["mild", "moderate", "severe", "moderate", "severe"])
            ataxia = RNG.random() < 0.45
            dysarthria = RNG.random() < 0.85
            optic_atrophy = RNG.random() < 0.15
            axonal_neuropathy = RNG.random() < 0.35
            cognitive_decline = RNG.random() < 0.75
            psychiatric = RNG.random() < 0.30
            ambulation_lost = RNG.random() < (0.70 if (current_age - onset_yr) > 12 else 0.25)
            seizures_prob = 0.65
            ocd = RNG.random() < 0.20

        elif ph == "Neuropsychiatric-CoPAN":
            onset_yr = round(RNG.uniform(10, 15), 1)
            current_age = round(RNG.uniform(onset_yr + 3, onset_yr + 20), 0)
            gp_iron = True
            sn_iron = RNG.random() < 0.60
            leukodystrophy = False
            thin_cc = RNG.random() < 0.30
            cerebellar_atrophy = RNG.random() < 0.20
            spastic_paraplegia = RNG.random() < 0.70  # motor develops later
            dystonia = RNG.random() < 0.75
            dystonia_severity = RNG.choice(["mild", "moderate", "moderate"]) if dystonia else None
            ataxia = RNG.random() < 0.35
            dysarthria = RNG.random() < 0.65
            optic_atrophy = RNG.random() < 0.10
            axonal_neuropathy = RNG.random() < 0.25
            cognitive_decline = RNG.random() < 0.65
            psychiatric = True  # all neuropsychiatric subtype
            ambulation_lost = RNG.random() < (0.35 if (current_age - onset_yr) > 10 else 0.10)
            seizures_prob = 0.50
            ocd = RNG.random() < 0.70  # OCD prominent in this subtype

        else:  # Late-onset-CoPAN
            onset_yr = round(RNG.uniform(15, 22), 1)
            current_age = round(RNG.uniform(onset_yr + 3, onset_yr + 15), 0)
            gp_iron = True
            sn_iron = RNG.random() < 0.50
            leukodystrophy = False
            thin_cc = RNG.random() < 0.20
            cerebellar_atrophy = RNG.random() < 0.15
            spastic_paraplegia = RNG.random() < 0.65
            dystonia = RNG.random() < 0.55
            dystonia_severity = "mild" if dystonia else None
            ataxia = RNG.random() < 0.25
            dysarthria = RNG.random() < 0.45
            optic_atrophy = RNG.random() < 0.08
            axonal_neuropathy = RNG.random() < 0.20
            cognitive_decline = RNG.random() < 0.50
            psychiatric = RNG.random() < 0.25
            ambulation_lost = RNG.random() < 0.10
            seizures_prob = 0.28
            ocd = RNG.random() < 0.20

        has_seizures = RNG.random() < seizures_prob
        if has_seizures:
            n_aeds = RNG.randint(1, 4)
            aeds_tried = RNG.sample(aed_options, min(n_aeds, len(aed_options)))
        else:
            n_aeds = 0
            aeds_tried = []

        drug_resistant = has_seizures and RNG.random() < 0.38
        seizure_free = has_seizures and (not drug_resistant) and RNG.random() < 0.50

        baclofen = RNG.random() < (0.90 if ph == "Classic-CoPAN" else 0.75 if ph == "Neuropsychiatric-CoPAN" else 0.55)
        trihexyphenidyl = dystonia and RNG.random() < 0.70
        btx = RNG.random() < (0.45 if ph == "Classic-CoPAN" else 0.35 if ph == "Neuropsychiatric-CoPAN" else 0.20)
        dbs = dystonia and RNG.random() < (0.06 if ph == "Classic-CoPAN" else 0.04 if ph == "Neuropsychiatric-CoPAN" else 0.02)
        polg_tested = RNG.random() < 0.72
        deferiprone_trial = RNG.random() < 0.06
        pantothenate_trial = RNG.random() < 0.08
        physio_enrolled = RNG.random() < 0.87

        pts.append({
            "id": pid,
            "phenotype": ph,
            "etiology": etiology,
            "onset_yr": onset_yr,
            "current_age": current_age,
            "disease_duration_yr": round(current_age - onset_yr, 1),
            "gp_iron": gp_iron,
            "sn_iron": sn_iron,
            "leukodystrophy": leukodystrophy,
            "thin_cc": thin_cc,
            "cerebellar_atrophy": cerebellar_atrophy,
            "spastic_paraplegia": spastic_paraplegia,
            "dystonia": dystonia,
            "dystonia_severity": dystonia_severity,
            "ataxia": ataxia,
            "dysarthria": dysarthria,
            "optic_atrophy": optic_atrophy,
            "axonal_neuropathy": axonal_neuropathy,
            "cognitive_decline": cognitive_decline,
            "psychiatric": psychiatric,
            "ocd": ocd,
            "ambulation_lost": ambulation_lost,
            "has_seizures": has_seizures,
            "drug_resistant": drug_resistant,
            "seizure_free": seizure_free,
            "n_aeds_tried": n_aeds,
            "aeds_tried": aeds_tried,
            "baclofen": baclofen,
            "trihexyphenidyl": trihexyphenidyl,
            "btx": btx,
            "dbs": dbs,
            "physio_enrolled": physio_enrolled,
            "polg_tested": polg_tested,
            "deferiprone_trial": deferiprone_trial,
            "pantothenate_trial": pantothenate_trial,
        })

    return pts


_CACHE = {}


def _get_patients():
    if "pts" not in _CACHE:
        _CACHE["pts"] = _patients()
    return _CACHE["pts"]


def get_overview():
    pts = _get_patients()
    n = len(pts)

    def pct(cond): return round(sum(1 for p in pts if cond(p)) / n * 100)

    n_classic = sum(1 for p in pts if p["phenotype"] == "Classic-CoPAN")
    n_npsy = sum(1 for p in pts if p["phenotype"] == "Neuropsychiatric-CoPAN")
    n_late = sum(1 for p in pts if p["phenotype"] == "Late-onset-CoPAN")

    classic_pts = [p for p in pts if p["phenotype"] == "Classic-CoPAN"]
    npsy_pts = [p for p in pts if p["phenotype"] == "Neuropsychiatric-CoPAN"]
    late_pts = [p for p in pts if p["phenotype"] == "Late-onset-CoPAN"]

    def mean_onset(group):
        return round(sum(p["onset_yr"] for p in group) / max(len(group), 1), 1)

    etio_counts = {}
    for p in pts:
        etio_counts[p["etiology"]] = etio_counts.get(p["etiology"], 0) + 1

    etio_distribution = [
        {"etiology": k.replace("_", " ").title(), "n": v, "pct": round(v / n * 100)}
        for k, v in sorted(etio_counts.items(), key=lambda x: -x[1])
    ]

    clinical_highlights = [
        {"finding": "GP Iron (T2*/SWI — PROMINENT)", "pct": pct(lambda p: p["gp_iron"]), "note": "PROMINENT from early disease — present in all Classic/Neuropsychiatric; key NBIA imaging marker"},
        {"finding": "SN Iron (T2*/SWI)", "pct": pct(lambda p: p["sn_iron"]), "note": "Moderate; less than GP (GP>SN typical CoPAN pattern)"},
        {"finding": "NO Leukodystrophy", "pct": pct(lambda p: not p["leukodystrophy"]), "note": "ABSENT in CoPAN — critical DDx from FAHN/NBIA3 (leukodystrophy = EARLIEST FAHN finding)"},
        {"finding": "Spastic Paraplegia", "pct": pct(lambda p: p["spastic_paraplegia"]), "note": "BOTH spasticity + dystonia early — unlike FAHN where spasticity dominates and dystonia is late"},
        {"finding": "Dystonia", "pct": pct(lambda p: p["dystonia"]), "note": "Prominent from early course alongside spasticity (combined spastic-dystonic syndrome)"},
        {"finding": "Dysarthria", "pct": pct(lambda p: p["dysarthria"]), "note": "Early; mixed spastic + dystonic component; 85% Classic-CoPAN"},
        {"finding": "Seizures", "pct": pct(lambda p: p["has_seizures"]), "note": "65% overall; 65% Classic, 50% Neuropsychiatric; PHT AVOID; POLG mandatory before VPA"},
        {"finding": "Cognitive Decline", "pct": pct(lambda p: p["cognitive_decline"]), "note": "75% Classic, 65% Neuropsychiatric; executive function + working memory affected"},
        {"finding": "OCD / Neuropsychiatric Features", "pct": pct(lambda p: p["ocd"]), "note": "Prominent in Neuropsychiatric subtype (70%); leads to diagnostic delay (misdiagnosed as primary OCD)"},
        {"finding": "Psychiatric Features (any)", "pct": pct(lambda p: p["psychiatric"]), "note": "OCD, impulsive behavior, depression — all prevalent; psychiatric review every 6-12 months"},
        {"finding": "Axonal Neuropathy", "pct": pct(lambda p: p["axonal_neuropathy"]), "note": "~35%; NCS mandatory at diagnosis; less severe than PLAN (100% INAD) or MPAN (60%)"},
        {"finding": "Optic Atrophy", "pct": pct(lambda p: p["optic_atrophy"]), "note": "~15%; less prominent than MPAN (80%); annual ophthalmology with VEP"},
        {"finding": "Ambulation Lost", "pct": pct(lambda p: p["ambulation_lost"]), "note": "Classic-CoPAN: ~70% wheelchair-dependent by 2nd-3rd decade"},
        {"finding": "POLG Tested", "pct": pct(lambda p: p["polg_tested"]), "note": "72% tested — mandatory before VPA; still suboptimal — all CoPAN patients should be screened"},
    ]

    treatment_summary = {
        "baclofen_pct": pct(lambda p: p["baclofen"]),
        "trihexyphenidyl_pct": pct(lambda p: p["trihexyphenidyl"]),
        "btx_pct": pct(lambda p: p["btx"]),
        "physio_enrolled_pct": pct(lambda p: p["physio_enrolled"]),
        "dbs_pct": pct(lambda p: p["dbs"]),
        "polg_tested_pct": pct(lambda p: p["polg_tested"]),
        "deferiprone_trial_pct": pct(lambda p: p["deferiprone_trial"]),
        "pantothenate_trial_pct": pct(lambda p: p["pantothenate_trial"]),
        "seizure_on_aed_pct": pct(lambda p: p["has_seizures"]),
        "drug_resistant_pct": pct(lambda p: p["drug_resistant"]),
        "seizure_free_pct": pct(lambda p: p["seizure_free"]),
    }

    return {
        "disease": DISEASE,
        "gene": GENE,
        "chromosome": CHROMOSOME,
        "inheritance": INHERITANCE,
        "omim_gene": OMIM_GENE,
        "omim_disease": OMIM_DISEASE,
        "cohort_n": n,
        "cohort_seed": SEED,
        "kpis": {
            "n_patients": n,
            "n_classic_copan": n_classic,
            "n_neuropsychiatric_copan": n_npsy,
            "n_late_onset_copan": n_late,
            "gp_iron_pct": pct(lambda p: p["gp_iron"]),
            "sn_iron_pct": pct(lambda p: p["sn_iron"]),
            "leukodystrophy_pct": pct(lambda p: p["leukodystrophy"]),
            "no_leukodystrophy_pct": pct(lambda p: not p["leukodystrophy"]),
            "spastic_paraplegia_pct": pct(lambda p: p["spastic_paraplegia"]),
            "dystonia_pct": pct(lambda p: p["dystonia"]),
            "dysarthria_pct": pct(lambda p: p["dysarthria"]),
            "has_seizures_pct": pct(lambda p: p["has_seizures"]),
            "cognitive_decline_pct": pct(lambda p: p["cognitive_decline"]),
            "ocd_pct": pct(lambda p: p["ocd"]),
            "psychiatric_pct": pct(lambda p: p["psychiatric"]),
            "axonal_neuropathy_pct": pct(lambda p: p["axonal_neuropathy"]),
            "optic_atrophy_pct": pct(lambda p: p["optic_atrophy"]),
            "ambulation_lost_pct": pct(lambda p: p["ambulation_lost"]),
            "polg_tested_pct": pct(lambda p: p["polg_tested"]),
            "classic_mean_onset_yr": mean_onset(classic_pts),
            "npsy_mean_onset_yr": mean_onset(npsy_pts),
            "late_mean_onset_yr": mean_onset(late_pts),
        },
        "etiology_distribution": etio_distribution,
        "clinical_highlights": clinical_highlights,
        "treatment_summary": treatment_summary,
        "lifecycle": LIFECYCLE,
        "contraindications": CONTRAINDICATIONS,
        "thresholds": THRESHOLDS,
        "standards": STANDARDS,
        "references": REFERENCES,
    }


def get_breakdown():
    pts = _get_patients()
    n = len(pts)

    phenotype_breakdown = []
    for ph in ["Classic-CoPAN", "Neuropsychiatric-CoPAN", "Late-onset-CoPAN"]:
        group = [p for p in pts if p["phenotype"] == ph]
        if not group:
            continue
        ng = len(group)
        phenotype_breakdown.append({
            "phenotype": ph,
            "n": ng,
            "pct": round(ng / n * 100),
            "mean_onset_yr": round(sum(p["onset_yr"] for p in group) / ng, 1),
            "gp_iron_pct": round(sum(1 for p in group if p["gp_iron"]) / ng * 100),
            "sn_iron_pct": round(sum(1 for p in group if p["sn_iron"]) / ng * 100),
            "spastic_paraplegia_pct": round(sum(1 for p in group if p["spastic_paraplegia"]) / ng * 100),
            "dystonia_pct": round(sum(1 for p in group if p["dystonia"]) / ng * 100),
            "cognitive_decline_pct": round(sum(1 for p in group if p["cognitive_decline"]) / ng * 100),
            "has_seizures_pct": round(sum(1 for p in group if p["has_seizures"]) / ng * 100),
            "ocd_pct": round(sum(1 for p in group if p["ocd"]) / ng * 100),
            "ambulation_lost_pct": round(sum(1 for p in group if p["ambulation_lost"]) / ng * 100),
            "drug_resistant_pct": round(sum(1 for p in group if p["drug_resistant"]) / ng * 100),
        })

    etio_groups = {}
    for p in pts:
        etio_groups.setdefault(p["etiology"], []).append(p)
    etio_breakdown = []
    for etio, group in sorted(etio_groups.items(), key=lambda x: -len(x[1])):
        ng = len(group)
        n_classic = sum(1 for p in group if p["phenotype"] == "Classic-CoPAN")
        n_npsy = sum(1 for p in group if p["phenotype"] == "Neuropsychiatric-CoPAN")
        etio_breakdown.append({
            "etiology": etio.replace("_", " ").title(),
            "n": ng,
            "pct": round(ng / n * 100),
            "classic_copan_pct": round(n_classic / ng * 100) if ng else 0,
            "neuropsychiatric_copan_pct": round(n_npsy / ng * 100) if ng else 0,
            "gp_iron_pct": round(sum(1 for p in group if p["gp_iron"]) / ng * 100),
            "drug_resistant_pct": round(sum(1 for p in group if p["drug_resistant"]) / ng * 100),
        })

    seizure_pts = [p for p in pts if p["has_seizures"]]
    seizure_breakdown = []
    for st in ["focal", "generalised", "myoclonic", "absence"]:
        prob = {"focal": 0.70, "generalised": 0.40, "myoclonic": 0.20, "absence": 0.15}[st]
        n_st = sum(1 for _ in seizure_pts if RNG.random() < prob)
        seizure_breakdown.append({
            "type": st.title(),
            "n": n_st,
            "pct": round(n_st / max(len(seizure_pts), 1) * 100),
            "drug_resistant_pct": round(sum(1 for p in seizure_pts if p["drug_resistant"]) / max(len(seizure_pts), 1) * 100),
        })

    per_patient = []
    for p in pts:
        per_patient.append({
            "id": p["id"],
            "phenotype": p["phenotype"],
            "etiology": p["etiology"],
            "onset_yr": p["onset_yr"],
            "current_age": p["current_age"],
            "disease_duration_yr": p["disease_duration_yr"],
            "gp_iron": p["gp_iron"],
            "sn_iron": p["sn_iron"],
            "leukodystrophy": p["leukodystrophy"],
            "thin_cc": p["thin_cc"],
            "cerebellar_atrophy": p["cerebellar_atrophy"],
            "spastic_paraplegia": p["spastic_paraplegia"],
            "dystonia": p["dystonia"],
            "dystonia_severity": p["dystonia_severity"],
            "ataxia": p["ataxia"],
            "dysarthria": p["dysarthria"],
            "optic_atrophy": p["optic_atrophy"],
            "axonal_neuropathy": p["axonal_neuropathy"],
            "cognitive_decline": p["cognitive_decline"],
            "psychiatric": p["psychiatric"],
            "ocd": p["ocd"],
            "ambulation_lost": p["ambulation_lost"],
            "has_seizures": p["has_seizures"],
            "drug_resistant": p["drug_resistant"],
            "seizure_free": p["seizure_free"],
            "n_aeds": p["n_aeds_tried"],
            "aeds_tried": p["aeds_tried"],
            "baclofen": p["baclofen"],
            "trihexyphenidyl": p["trihexyphenidyl"],
            "btx": p["btx"],
            "dbs": p["dbs"],
            "physio_enrolled": p["physio_enrolled"],
            "polg_tested": p["polg_tested"],
            "deferiprone_trial": p["deferiprone_trial"],
            "pantothenate_trial": p["pantothenate_trial"],
        })

    return {
        "cohort_n": n,
        "phenotype_breakdown": phenotype_breakdown,
        "etiology_breakdown": etio_breakdown,
        "seizure_breakdown": seizure_breakdown,
        "per_patient": per_patient,
        "treatment_summary": TREATMENTS,
        "contraindications": CONTRAINDICATIONS,
        "ddx_table": DDX,
        "monitoring": MONITORING,
        "lifecycle": LIFECYCLE,
        "thresholds": THRESHOLDS,
        "standards": STANDARDS,
        "references": REFERENCES,
    }


def get_definitions():
    return {
        "disease": "COASY CoPAN (CoA Synthase Protein-Associated Neurodegeneration / NBIA6)",
        "gene": "COASY (CoA Synthase — 579 aa, 17q21.2, PPAT+DPCK bifunctional) — OMIM 609686",
        "omim_disease": "CoPAN 615643",
        "definitions": DEFINITIONS,
        "contraindications": CONTRAINDICATIONS,
        "standards": STANDARDS,
        "references": REFERENCES,
        "key_concepts": [d["term"] for d in DEFINITIONS],
    }


if __name__ == "__main__":
    print("=== COASY / CoPAN (NBIA6) Dashboard — Self-Test (seed-525) ===\n")
    ov = get_overview()
    print(f"Disease: {ov['disease']}")
    print(f"Gene: {ov['gene']}")
    print(f"Cohort N: {ov['cohort_n']} patients (seed-{ov['cohort_seed']})")
    print(f"KPIs: {ov['kpis']}")
    bk = get_breakdown()
    print(f"\nPhenotype breakdown: {[p['phenotype'] for p in bk['phenotype_breakdown']]}")
    print(f"Per-patient count: {len(bk['per_patient'])}")
    df = get_definitions()
    print(f"\nDefinitions count: {len(df['definitions'])}")
    print("Self-test PASSED")
