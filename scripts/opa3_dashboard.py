"""
Costeff Syndrome — OPA3 (3-Methylglutaconic Aciduria Type III)
==============================================================
40-patient cohort · OPA3 (19q13.2-q13.3) · Autosomal Recessive · ~50-100 families worldwide 2026
First described: Costeff et al. 1989 (Am J Dis Child) — Iraqi Jewish pedigrees; MGA Type III nomenclature
Key DDx from MECR/MEPAN: CHOREA dominant (not dystonia); NO GP iron on MRI; lipoic acid pathway INTACT
Shared biomarker with MECR: elevated 3-methylglutaconic acid (3-MGA) — METABOLIC LINK

OPA3 BIOLOGY:
OPA3 (179 amino acids isoform-1, 19q13.2) encodes an outer mitochondrial membrane (OMM) protein.
Two isoforms: OPA3-L (long, 179 aa, main isoform) and OPA3-S (short, 100 aa, brain-enriched).
OPA3 N-terminal transmembrane domain anchors it in the OMM; C-terminal coiled-coil faces cytoplasm.
Function: OMM integrity maintenance; coordinates with OPA1 (IMM fusion) and DRP1/FIS1 (fission);
  OPA3 LOF → mitochondrial fragmentation → OXPHOS inefficiency → ROS accumulation → neuronal loss.
OPA3 is NOT in the lipoic acid synthesis pathway (unlike MECR/mtFAS-II) — different 3-MGA mechanism.
3-MGA generation in OPA3 LOF: mitochondrial dysfunction → HMG-CoA pathway overflow → 3-methylglutaconyl-CoA
  accumulates → not efficiently hydrated by AUH → excreted as 3-methylglutaconic acid.

TWO DISEASE FORMS (DISTINCT GENETICS AND PHENOTYPE):
AR form — Costeff Syndrome (OMIM 258501): biallelic LOF; childhood onset; chorea + optic atrophy.
AD form — Dominant Optic Atrophy Type 3 (OMIM 165300): heterozygous missense; bilateral cataracts +
  adult-onset optic atrophy; NO chorea; NO 3-MGA; COMPLETELY DIFFERENT from AR Costeff.
This dashboard focuses exclusively on the AR Costeff / 3-MGA Type III phenotype.

PROTEIN STRUCTURE (179 aa isoform-1, 19q13.2):
  N-terminal transmembrane domain (aa 1-40): OMM anchor; single-pass; Costeff founder c.313C>T
    (p.Gln105*) causes premature stop upstream of TM domain disrupting full-length protein.
  Intermembrane space loop (aa 41-80): interacts with cytochrome c; ROS-sensitive.
  Cytoplasmic coiled-coil (aa 100-179): DRP1/OPA1 interaction surface; mitochondrial dynamics regulation.
  Isoform-S (100 aa): lacks C-terminal coiled-coil; upregulated in neurons; selectively vulnerable.

PATHOGENIC VARIANT DISTRIBUTION (biallelic LOF mutations, n=40 patients):
  Iraqi Jewish founder p.Gln105* (c.313C>T, exon 2): ~70-80% of alleles in affected families
  Missense (intermembrane loop): ~10% (partial LOF; milder phenotype)
  Frameshift/truncating (non-founder): ~7% (complete null; severe phenotype)
  Splice site variants: ~5% (exon 2 skip; complete LOF)
  Large deletion exon 1-2: ~3% (rare, complete null; European cases)

CLINICAL PHENOTYPE — AR COSTEFF SYNDROME:
  OPTIC ATROPHY (100%):
    Bilateral symmetric; infantile onset (6 months to 2 years, median 12 months).
    Temporal optic disc pallor; reduced VA (6/24-6/60 range); legal blindness uncommon (<10%).
    VEP: prolonged P100 latency + reduced amplitude; ERG NORMAL (retinal photoreceptors spared).
    OCT: RNFL thinning (temporal > nasal); GCL volume reduction.
    Optic atrophy is the EARLIEST feature — precedes chorea by years.
    Key DDx from MECR: in MECR optic atrophy is 80-90% and early; in OPA3 optic atrophy is 100% and
      presents EARLIER (infantile vs childhood in MECR); BUT ERG is normal in OPA3, abnormal in advanced MECR.
  CHOREA (85-90%):
    Generalized choreiform movements; childhood onset (2-10 yr, median 5 yr); HALLMARK feature.
    Arms > trunk > face; brief, irregular, purposeless; superimposed on volitional movement.
    Chorea is DISTINCTIVE and separates OPA3/Costeff from MECR/MEPAN (which is DYSTONIA-dominant).
    Choreic movements increase with stress/anxiety; reduce with sleep.
    Progressive: moderate-severe chorea by adolescence requiring pharmacological management.
    Tetrabenazine/deutetrabenazine: partial response (40-60% reduction in UHDRS choreiform score).
  SPASTIC PARAPLEGIA (50-60%):
    Upper motor neuron signs (hyperreflexia, spastic gait, extensor plantar response).
    Pyramidal tract involvement; gait increasingly spastic by mid-childhood.
    Baclofen first-line for spasticity management; physiotherapy essential.
    MRI: occasionally mild periventricular T2 signal (non-specific); NOT leukodystrophy (DDx FAHN).
  COGNITIVE IMPAIRMENT (50-60%):
    Mild-moderate intellectual disability; language and memory relatively preserved early.
    Executive function and processing speed most affected.
    Distinct from severe cognitive impairment in neonatal MECR (Mixed Severe phenotype).
  SEIZURES (30-40%):
    Focal > generalized; myoclonic component in subset; rarely drug-resistant.
    Less prominent than in MECR (40-50%); NOT a defining feature of Costeff.
    LEV preferred (renal excretion, no mitochondrial interactions).
    VPA: relative caution (not absolute CI as in MECR); monitor ammonia + 3-MGA.
  3-METHYLGLUTACONIC ACIDURIA (100%):
    Present in ALL AR OPA3 patients; Type III 3-MGA-uria classification (Costeff type).
    Level: typically 40-200 mmol/mol creatinine (higher than MECR 20-100 range).
    3-MGA IS the metabolic fingerprint for diagnosis, shared with MECR but different mechanism.
    Key DDx: MECR = 3-MGA Type IV; OPA3 = 3-MGA Type III.
  LACTATE (20-30%):
    Mildly elevated in subset; less prominent than MECR (70-80%).
    Muscle biopsy: mild OXPHOS complex I/III reduction (not severe as in MECR).
  HEARING: Normal (distinguishes from DCAF17/WSS where sensorineural HL is cardinal).
  ENDOCRINE: Normal (distinguishes from DCAF17/WSS where hypogonadism + diabetes are cardinal).
  BRAIN MRI: Usually NORMAL or mild non-specific changes (NO GP iron — KEY DDx from MECR).
    MRI: NO Eye-of-Tiger (PKAN); NO cerebellar atrophy (MECR); NO leukodystrophy (FAHN); NO GP iron.

TREATMENT & PHARMACOGENOMICS:
  Tetrabenazine (VMAT2 inhibitor): CHOREA — Level C
    Depletes pre-synaptic dopamine → reduces dopaminergic drive on basal ganglia → chorea reduction.
    Dose: 12.5-50 mg/day; titrate to response. Monitor: depression, parkinsonism, sedation.
    40-60% choreiform score reduction (UHDRS); partial responders continue; non-responders switch.
    CYP2D6 genotyping mandatory: poor metabolisers at risk of prolonged sedation.
  Deutetrabenazine (VMAT2 inhibitor): CHOREA — Level C (newer, preferred)
    Extended-release VMAT2 inhibitor; fewer side effects than tetrabenazine (lower Cmax).
    Dose: 6-48 mg/day; better tolerability; FDA-approved for HD chorea (extrapolated Costeff).
    CYP2D6 genotyping still recommended.
  Baclofen (GABA-B agonist): SPASTICITY — Level C
    First-line for spastic paraplegia component; 5-20 mg TDS; titrate.
    Monitor: sedation, hypotension, seizure threshold reduction (use cautiously with LEV).
  LEV (Levetiracetam): SEIZURES — PREFERRED FIRST-LINE
    Renal excretion; no hepatic metabolism; no mitochondrial interactions; broad-spectrum.
    Same reasoning as MECR — mitochondrial disease patients benefit from non-hepatic AEDs.
  VPA (Valproate): RELATIVE CAUTION (NOT absolute CI like MECR)
    OPA3 does NOT disrupt the lipoic acid pathway (unlike MECR) → VPA CI mechanism absent.
    However: 3-MGA-uria diseases have mitochondrial dysfunction → VPA may worsen ammonia cycle.
    Monitor: ammonia, LFTs, 3-MGA levels. POLG screening mandatory before VPA in any mitochondrial disease.
    If VPA needed for refractory seizures: use lowest effective dose with close monitoring.
  CBZ/PHT/OXC: AVOID
    Sodium channel blockers can worsen choreic movements (paradoxical increase in involuntary movement).
    CYP3A4 induction (CBZ/PHT): metabolic burden in mitochondrial dysfunction.
    Key trap: CBZ may temporarily improve focal seizures but worsens chorea — avoid in OPA3.
  DHA (Docosahexaenoic acid): SUPPORTIVE (Level D)
    Mitochondrial membrane stabilisation; anecdotal benefit in 3-MGA diseases.
    Dose: 500-1000 mg/day; generally safe; no RCT in OPA3.

DIAGNOSTIC WORKUP:
  Urine organic acids: 3-MGA elevated (Type III pattern; 40-200 mmol/mol creatinine).
  Plasma amino acids: normal (distinguishes from NKH where CSF/plasma glycine elevated).
  Plasma lactate: mildly elevated in subset.
  Ophthalmology: VEP + ERG + OCT (optic atrophy characterisation; ERG normal in OPA3).
  Brain MRI (T2/FLAIR/SWI): usually normal; NO GP iron; NO leukodystrophy.
  OPA3 gene sequencing (WES/targeted panel): biallelic pathogenic variants confirm.
  POLG screening: mandatory before any VPA consideration.
  CYP2D6 genotyping: before tetrabenazine/deutetrabenazine initiation.

DIFFERENTIAL DIAGNOSIS:
  MECR (MEPAN): dystonia (not chorea); GP iron bilateral SWI (ABSENT in OPA3); MGA same; 3-MGA Type IV vs III;
    lipoic acid pathway disrupted (MECR) — NOT in OPA3; Israeli Bedouin vs Iraqi Jewish founder.
  Barth syndrome (TAZ — 3-MGA Type II): dilated cardiomyopathy + neutropenia (ABSENT in OPA3); X-linked males only.
  DNAJC19 / DCMA (3-MGA + cerebellar + cardiomyopathy): cardiomyopathy present; cerebellar atrophy prominent; NO chorea.
  SERAC1 (MEGDEL — 3-MGA + deafness + Leigh-like MRI + liver): liver dysfunction + hearing loss (ABSENT in OPA3).
  ATAD3A (3-MGA + pontocerebellar hypoplasia): neonatal onset; leukomalacia; severe.
  Huntington disease: CAG repeat expansion; family history; no optic atrophy; adult onset typical.
  OPA1 (dominant optic atrophy): adult-onset optic atrophy only; NO chorea; NO 3-MGA.
  OPA3-AD (dominant cataracts): bilateral cataracts + mild optic atrophy; heterozygous; NO chorea; NO 3-MGA.
"""

import random

SEED = 535
DISEASE = "Costeff Syndrome (OPA3 — 3-Methylglutaconic Aciduria Type III)"
GENE = (
    "OPA3 (179 aa isoform-1, 19q13.2-q13.3) — outer mitochondrial membrane protein; "
    "OPA3 LOF → mitochondrial fragmentation + 3-MGA-uria Type III + early optic atrophy + chorea; "
    "AR biallelic LOF = Costeff Syndrome; AD missense = dominant cataracts + optic atrophy (different phenotype)"
)
CHROMOSOME = "19q13.2-q13.3"
OMIM_GENE = "606580"
OMIM_DISEASE = "258501"
PREVALENCE = "~50-100 patients worldwide (AR Costeff); Iraqi Jewish founder p.Gln105* ~1:10,000 Iraqi Jews"
FIRST_DESCRIBED = "Costeff et al. 1989 (Am J Dis Child) — Iraqi Jewish pedigrees; 3-MGA Type III nomenclature Elpeleg 1994"
INHERITANCE = "Autosomal Recessive (AR) — biallelic LOF; Iraqi Jewish founder p.Gln105* (c.313C>T) dominant allele"

RNG = random.Random(SEED)

PHENOTYPES = ["Classic Costeff", "Chorea-Dominant", "Optic-Dominant"]
PHENOTYPE_WEIGHTS = [0.75, 0.15, 0.10]

VARIANTS = [
    "p.Gln105* (c.313C>T) — Iraqi Jewish founder, exon 2 premature stop",
    "p.Cys126Arg (c.376T>C) — intermembrane space loop missense",
    "p.Glu93Lys (c.277G>A) — IMS loop, partial LOF, milder phenotype",
    "c.IVS2+1G>A (splice, exon 2 skip, complete LOF)",
    "p.Ala36Val (c.107C>T) — N-terminal TM domain disruption",
    "Large deletion exon 1-2 (complete null, European cases)",
    "p.Trp147* (c.441G>A) — coiled-coil truncation",
]
VARIANT_WEIGHTS = [0.75, 0.09, 0.07, 0.05, 0.02, 0.01, 0.01]

TREATMENT_OPTIONS = [
    "Tetrabenazine — chorea",
    "Deutetrabenazine — chorea (preferred)",
    "Baclofen + LEV — spasticity + seizures",
    "LEV monotherapy — seizures only",
    "Supportive / no pharmacotherapy",
    "DHA + baclofen — supportive",
]


def _make_patient(i: int, phenotype: str) -> dict:
    rng = RNG

    sex = rng.choice(["Female", "Male"])

    if phenotype == "Classic Costeff":
        optic_onset_mo = rng.randint(6, 24)
        chorea = True
        chorea_onset_yr = rng.randint(2, 8)
        spasticity = rng.random() < 0.58
        cognitive = rng.random() < 0.55
        seizures = rng.random() < 0.38
        lactate_high = rng.random() < 0.28
        mga_mmol_cr = round(rng.uniform(45, 200), 1)
        va_logmar = round(rng.uniform(0.3, 1.0), 2)
        uhdrs_chorea = round(rng.uniform(8, 28), 1)    # UHDRS TMS chorea subscore

    elif phenotype == "Chorea-Dominant":
        optic_onset_mo = rng.randint(12, 36)
        chorea = True
        chorea_onset_yr = rng.randint(2, 6)
        spasticity = rng.random() < 0.25
        cognitive = rng.random() < 0.35
        seizures = rng.random() < 0.20
        lactate_high = rng.random() < 0.18
        mga_mmol_cr = round(rng.uniform(60, 200), 1)
        va_logmar = round(rng.uniform(0.2, 0.7), 2)
        uhdrs_chorea = round(rng.uniform(14, 32), 1)

    else:  # Optic-Dominant
        optic_onset_mo = rng.randint(6, 15)
        chorea = rng.random() < 0.60
        chorea_onset_yr = rng.randint(5, 12)
        spasticity = rng.random() < 0.30
        cognitive = rng.random() < 0.30
        seizures = rng.random() < 0.25
        lactate_high = rng.random() < 0.20
        mga_mmol_cr = round(rng.uniform(40, 120), 1)
        va_logmar = round(rng.uniform(0.5, 1.2), 2)
        uhdrs_chorea = round(rng.uniform(2, 14), 1) if chorea else 0.0

    variant = rng.choices(VARIANTS, weights=VARIANT_WEIGHTS, k=1)[0]
    treatment = rng.choices(
        TREATMENT_OPTIONS,
        weights=[0.30, 0.28, 0.18, 0.12, 0.08, 0.04],
        k=1
    )[0]

    return {
        "id": i + 1,
        "sex": sex,
        "phenotype": phenotype,
        "optic_onset_mo": optic_onset_mo,
        "chorea_onset_yr": chorea_onset_yr if chorea else None,
        "variant": variant,
        "treatment": treatment,
        "optic_atrophy": True,  # 100% in AR OPA3
        "chorea": chorea,
        "spasticity": spasticity,
        "cognitive_impairment": cognitive,
        "seizures": seizures,
        "lactate_elevated": lactate_high,
        "mga_mmol_creatinine": mga_mmol_cr,
        "va_logmar": va_logmar,
        "uhdrs_chorea": uhdrs_chorea,
    }


def _build_patients() -> list:
    patients = []
    n_each = RNG.choices(PHENOTYPES, weights=PHENOTYPE_WEIGHTS, k=40)
    for i, ph in enumerate(n_each):
        patients.append(_make_patient(i, ph))
    return patients


def _pct(pts: list, key: str) -> int:
    if not pts:
        return 0
    return round(sum(1 for p in pts if p.get(key)) / len(pts) * 100)


def _avg(pts: list, key: str) -> float:
    vals = [p[key] for p in pts if p.get(key) is not None and isinstance(p.get(key), (int, float))]
    return round(sum(vals) / len(vals), 1) if vals else 0.0


def get_overview() -> dict:
    pts = _build_patients()
    by_phenotype: dict = {ph: [] for ph in PHENOTYPES}
    for p in pts:
        by_phenotype[p["phenotype"]].append(p)

    phenotype_dist = [
        {"phenotype": ph, "n": len(g), "pct": round(len(g) / 40 * 100)}
        for ph, g in by_phenotype.items()
    ]

    kpis = {
        "n_patients": 40,
        "n_classic": len(by_phenotype["Classic Costeff"]),
        "n_chorea": len(by_phenotype["Chorea-Dominant"]),
        "n_optic": len(by_phenotype["Optic-Dominant"]),
        "optic_atrophy_pct": 100,
        "chorea_pct": _pct(pts, "chorea"),
        "spasticity_pct": _pct(pts, "spasticity"),
        "cognitive_pct": _pct(pts, "cognitive_impairment"),
        "seizures_pct": _pct(pts, "seizures"),
        "lactate_pct": _pct(pts, "lactate_elevated"),
        "mean_optic_onset_mo": _avg(pts, "optic_onset_mo"),
        "mean_mga": _avg(pts, "mga_mmol_creatinine"),
        "mean_uhdrs_chorea": _avg(pts, "uhdrs_chorea"),
        "mean_va_logmar": _avg(pts, "va_logmar"),
    }

    clinical_highlights = [
        {"finding": "Optic atrophy (bilateral)", "pct": 100,
         "note": "100%; EARLIEST feature; infantile onset (median 12 mo); temporal pallor; VEP prolonged; ERG NORMAL (DDx MECR advanced-ERG)"},
        {"finding": "Chorea (generalised)", "pct": kpis["chorea_pct"],
         "note": "85-90%; HALLMARK; childhood onset (median 5yr); generalized choreiform; arms > trunk > face; DOMINANT feature DDx MECR-dystonia"},
        {"finding": "Spastic paraplegia", "pct": kpis["spasticity_pct"],
         "note": "50-60%; pyramidal signs; hyperreflexia; spastic gait; baclofen first-line; MRI may show mild periventricular T2 (NOT leukodystrophy)"},
        {"finding": "Cognitive impairment", "pct": kpis["cognitive_pct"],
         "note": "50-60%; mild-moderate intellectual disability; executive function + processing speed; language spared; less severe than MECR Mixed-Severe"},
        {"finding": "Seizures", "pct": kpis["seizures_pct"],
         "note": "30-40%; focal > generalised; myoclonic subset; LEV preferred; VPA relative caution (not absolute CI like MECR)"},
        {"finding": "Elevated lactate (mild)", "pct": kpis["lactate_pct"],
         "note": "20-30%; mild; secondary mitochondrial dysfunction; NOT as prominent as MECR (70-80%); no PDH/KGDH hypolipoylation in OPA3"},
        {"finding": "3-MGA-uria (100%)", "pct": 100,
         "note": "100%; TYPE III pattern (Costeff type); 40-200 mmol/mol creatinine; shared with MECR but different mechanism (OMM fragmentation not lipoic acid)"},
        {"finding": "Brain MRI — NORMAL", "pct": 75,
         "note": "~75% normal MRI; NO GP iron (DDx MECR/NBIA); NO leukodystrophy (DDx FAHN); occasionally mild periventricular T2 (non-specific)"},
    ]

    contraindications = [
        {
            "drug": "CBZ / OXC (Carbamazepine / Oxcarbazepine)",
            "severity": "AVOID",
            "reason": (
                "Sodium channel blockade paradoxically worsens choreic movements in OPA3/Costeff — "
                "basal ganglia dopaminergic-cholinergic imbalance disrupted by Na-channel inhibition. "
                "CYP3A4 induction (CBZ) adds metabolic burden in mitochondrial dysfunction. "
                "Clinical trap: CBZ may briefly reduce seizure frequency while worsening chorea — net harm. "
                "PHT: same avoidance rationale (Na-channel blockade + CYP2C9/2C19 reactive metabolites)."
            ),
            "alternative": "LEV first-line for seizures; tetrabenazine/deutetrabenazine for chorea separately",
        },
        {
            "drug": "VPA (Valproate)",
            "severity": "RELATIVE CAUTION (NOT absolute CI unlike MECR)",
            "reason": (
                "OPA3 does NOT disrupt the lipoic acid biosynthesis pathway (unlike MECR where PDH/alpha-KGDH "
                "hypolipoylation makes VPA lethal by pushing an already-failing PDH into crisis). "
                "However: any mitochondrial dysfunction disease warrants VPA caution — VPA inhibits "
                "beta-oxidation → ammonia cycle burden → elevated ammonia risk. "
                "Monitor: ammonia, LFTs, 3-MGA level. POLG screen mandatory before VPA initiation. "
                "Use lowest effective dose if VPA required for truly refractory seizures."
            ),
            "alternative": "LEV first-line; CLB second-line; avoid VPA where LEV/CLB sufficient",
        },
        {
            "drug": "Tetrabenazine (high dose)",
            "severity": "MONITOR — depression + parkinsonism risk",
            "reason": (
                "Tetrabenazine (VMAT2 inhibitor) depletes pre-synaptic dopamine — effective for chorea "
                "but dose-dependent risk of depression (15-30%), parkinsonism (15%), and sedation. "
                "Suicidality warning (FDA black box for HD); depression screening mandatory q6M. "
                "CYP2D6 poor metabolisers: drug accumulation → increased side-effect risk. "
                "CYP2D6 genotyping MANDATORY before initiation; dose adjust in poor metabolisers. "
                "Deutetrabenazine preferred (extended-release, lower Cmax, fewer effects)."
            ),
            "alternative": "Deutetrabenazine preferred; lowest effective tetrabenazine dose; depression monitoring mandatory",
        },
        {
            "drug": "LEV (Levetiracetam)",
            "severity": "PREFERRED FIRST-LINE (seizures)",
            "reason": "Renal excretion (66% unchanged); no mitochondrial interactions; broad-spectrum; no CYP450 induction; safe with tetrabenazine co-administration",
            "alternative": None,
        },
        {
            "drug": "Baclofen",
            "severity": "FIRST-LINE (spasticity)",
            "reason": "GABA-B agonist; reduces upper motor neuron spasticity; use cautiously with LEV (additive sedation); monitor respiratory depression at high doses",
            "alternative": None,
        },
    ]

    thresholds = [
        {"metric": "Urine 3-MGA", "threshold": ">5 mmol/mol creatinine (screen)",
         "action": "3-MGA-uria present; Costeff/OPA3 differential active if optic atrophy + chorea; send OPA3 gene sequencing"},
        {"metric": "Urine 3-MGA", "threshold": ">40 mmol/mol creatinine (OPA3 range)",
         "action": "High probability Costeff; confirm OPA3 biallelic variants; initiate ophthalmology + movement disorder referral"},
        {"metric": "VEP P100 latency", "threshold": ">120 ms (abnormal optic conduction)",
         "action": "Optic atrophy functional confirmation; baseline for monitoring; low-vision referral if VA <6/18"},
        {"metric": "Visual acuity", "threshold": "LogMAR >0.5 (6/18 equivalent)",
         "action": "Significant visual loss; low-vision aids; mobility assessment; educational support"},
        {"metric": "UHDRS chorea subscore", "threshold": ">8 (moderate choreiform burden)",
         "action": "Initiate tetrabenazine/deutetrabenazine; CYP2D6 genotype first; monitor depression q6M"},
        {"metric": "Depression screening (PHQ-9)", "threshold": ">10 (moderate depression)",
         "action": "Tetrabenazine dose review; consider dose reduction or switch to deutetrabenazine; psychiatry referral"},
        {"metric": "Plasma ammonia", "threshold": ">80 umol/L (if on VPA)",
         "action": "VPA dose reduction; L-carnitine supplementation; ammonia-scavenger consideration; hepatology review"},
        {"metric": "CYP2D6 genotype", "threshold": "Poor metaboliser (PM) — *4/*4 or *5/*4",
         "action": "Reduce tetrabenazine/deutetrabenazine starting dose by 50%; titrate more slowly; monitor sedation"},
    ]

    return {
        "disease": DISEASE,
        "gene": GENE,
        "chromosome": CHROMOSOME,
        "omim_gene": OMIM_GENE,
        "omim_disease": OMIM_DISEASE,
        "prevalence": PREVALENCE,
        "first_described": FIRST_DESCRIBED,
        "inheritance": INHERITANCE,
        "kpis": kpis,
        "phenotype_distribution": phenotype_dist,
        "clinical_highlights": clinical_highlights,
        "contraindications": contraindications,
        "thresholds": thresholds,
        "seed": SEED,
    }


def get_breakdown() -> dict:
    pts = _build_patients()
    by_phenotype: dict = {ph: [] for ph in PHENOTYPES}
    for p in pts:
        by_phenotype[p["phenotype"]].append(p)

    phenotype_breakdown = []
    for ph, g in by_phenotype.items():
        if not g:
            continue
        phenotype_breakdown.append({
            "phenotype": ph,
            "n": len(g),
            "pct": round(len(g) / 40 * 100),
            "mean_optic_onset_mo": _avg(g, "optic_onset_mo"),
            "chorea_pct": _pct(g, "chorea"),
            "spasticity_pct": _pct(g, "spasticity"),
            "seizures_pct": _pct(g, "seizures"),
            "cognitive_pct": _pct(g, "cognitive_impairment"),
            "lactate_pct": _pct(g, "lactate_elevated"),
            "mean_mga": _avg(g, "mga_mmol_creatinine"),
            "mean_va_logmar": _avg(g, "va_logmar"),
            "mean_uhdrs_chorea": _avg(g, "uhdrs_chorea"),
        })

    vcounts: dict = {}
    for p in pts:
        v = p["variant"]
        vcounts[v] = vcounts.get(v, 0) + 1
    variant_breakdown = sorted(
        [{"variant": v, "n": n, "pct": round(n / 40 * 100)} for v, n in vcounts.items()],
        key=lambda x: -x["n"],
    )

    tcounts: dict = {}
    for p in pts:
        t = p["treatment"]
        tcounts[t] = tcounts.get(t, 0) + 1
    treatment_breakdown = sorted(
        [{"treatment": t, "n": n, "pct": round(n / 40 * 100)} for t, n in tcounts.items()],
        key=lambda x: -x["n"],
    )

    mga_hist = {"<40": 0, "40-100": 0, "100-200": 0, ">200": 0}
    for p in pts:
        v = p["mga_mmol_creatinine"]
        if v < 40:
            mga_hist["<40"] += 1
        elif v < 100:
            mga_hist["40-100"] += 1
        elif v <= 200:
            mga_hist["100-200"] += 1
        else:
            mga_hist[">200"] += 1
    mga_breakdown = [{"range": k, "n": v, "pct": round(v / 40 * 100)} for k, v in mga_hist.items()]

    uhdrs_hist = {"0-8 (mild)": 0, "8-16 (moderate)": 0, "16-28 (severe)": 0, ">28 (very severe)": 0}
    for p in pts:
        u = p["uhdrs_chorea"]
        if u <= 8:
            uhdrs_hist["0-8 (mild)"] += 1
        elif u <= 16:
            uhdrs_hist["8-16 (moderate)"] += 1
        elif u <= 28:
            uhdrs_hist["16-28 (severe)"] += 1
        else:
            uhdrs_hist[">28 (very severe)"] += 1
    uhdrs_breakdown = [{"category": k, "n": v, "pct": round(v / 40 * 100)} for k, v in uhdrs_hist.items()]

    va_hist = {"≤0.3 (good)": 0, "0.3-0.7 (moderate)": 0, ">0.7 (poor)": 0}
    for p in pts:
        va = p["va_logmar"]
        if va <= 0.3:
            va_hist["≤0.3 (good)"] += 1
        elif va <= 0.7:
            va_hist["0.3-0.7 (moderate)"] += 1
        else:
            va_hist[">0.7 (poor)"] += 1
    va_breakdown = [{"category": k, "n": v, "pct": round(v / 40 * 100)} for k, v in va_hist.items()]

    sex_dist = {"Female": 0, "Male": 0}
    for p in pts:
        sex_dist[p["sex"]] += 1

    return {
        "phenotype_breakdown": phenotype_breakdown,
        "variant_breakdown": variant_breakdown,
        "treatment_breakdown": treatment_breakdown,
        "mga_breakdown": mga_breakdown,
        "uhdrs_breakdown": uhdrs_breakdown,
        "va_breakdown": va_breakdown,
        "sex_distribution": [{"sex": k, "n": v, "pct": round(v / 40 * 100)} for k, v in sex_dist.items()],
        "n_patients": 40,
        "seed": SEED,
    }


def get_definitions() -> dict:
    defs = [
        {
            "term": "OPA3-OMM-Fragmentation-Mechanism",
            "full": "OPA3 — outer mitochondrial membrane integrity; OPA3 LOF → fragmentation → OXPHOS failure → 3-MGA + neuronal loss",
            "detail": (
                "OPA3 (179 aa, 19q13.2): single-pass OMM protein; N-terminal TM domain (aa 1-40) anchors in OMM; "
                "C-terminal coiled-coil (aa 100-179) interacts with DRP1, FIS1, and OPA1. "
                "OPA3 opposes excessive mitochondrial fission; loss → fragmented mitochondria → "
                "reduced OXPHOS complex stability → electron leakage → ROS → neuronal apoptosis. "
                "3-MGA mechanism: fragmented mitochondria → HMG-CoA pathway overflow → 3-methylglutaconyl-CoA "
                "not efficiently hydrated → excreted as 3-methylglutaconic acid. "
                "Key distinction from MECR: OPA3 mechanism is OMM fragmentation/fission dysregulation; "
                "MECR mechanism is direct lipoic acid synthesis failure (mtFAS-II step 4 blockade). "
                "Both produce 3-MGA but via entirely different upstream pathways."
            ),
        },
        {
            "term": "3-MGA-Type-III-Costeff-vs-Type-IV-MECR",
            "full": "3-MGA Type III (Costeff/OPA3) vs Type IV (MECR/MEPAN) — shared biomarker, different mechanism and level",
            "detail": (
                "3-methylglutaconic aciduria classification (Wortmann 2013): "
                "Type I (AUH/3-MGA-CoA hydratase): pure 3-MGA-CoA pathway defect; isolated enzyme deficit. "
                "Type II (TAZ/Barth): cardiomyopathy + neutropenia + males; X-linked. "
                "Type III (OPA3/Costeff): optic atrophy + chorea + 3-MGA (40-200 mmol/mol creatinine). "
                "Type IV (secondary — MECR, SERAC1, CLPB, others): heterogeneous mitochondrial diseases with 3-MGA. "
                "OPA3 (Type III) vs MECR (Type IV): "
                "OPA3 3-MGA level HIGHER (40-200 range) than MECR (20-100 range) on average. "
                "OPA3: normal ERG, normal brain MRI, no GP iron. MECR: abnormal ERG advanced, GP iron 75-85%. "
                "Both: 100% penetrance for 3-MGA. OPA3: chorea dominant; MECR: dystonia dominant. "
                "Iraqi Jewish founder (OPA3) vs Israeli Bedouin founder (MECR): distinct ethnic backgrounds."
            ),
        },
        {
            "term": "Iraqi-Jewish-Founder-Gln105stop",
            "full": "p.Gln105* (c.313C>T) — Iraqi Jewish founder; exon 2 premature stop; ~70-80% worldwide OPA3 alleles",
            "detail": (
                "First characterized: Anikster Y et al. 2001 (Am J Hum Genet) — linkage in Iraqi Jewish pedigrees "
                "to 19q13.2-q13.3, OPA3 positional cloning. "
                "p.Gln105* (c.313C>T, NM_025136.3 exon 2): creates a premature stop codon at position 105 "
                "→ truncated protein (104 aa) missing intermembrane space loop + cytoplasmic coiled-coil. "
                "Truncated OPA3 cannot interact with DRP1/OPA1 → complete LOF. "
                "Carrier frequency in Iraqi Jewish population: ~1:100 (heterozygous); disease frequency ~1:10,000. "
                "Haplotype analysis: all carriers share a common haplotype → single founder event. "
                "Estimated founder event: ~30-40 generations ago (Babylonian Jewish community). "
                "Homozygous p.Gln105*: consistent Costeff phenotype — optic atrophy + chorea + 3-MGA. "
                "Compound heterozygotes (p.Gln105* + second variant): slightly more variable phenotype."
            ),
        },
        {
            "term": "Chorea-Dominant-OPA3-vs-Dystonia-MECR",
            "full": "Chorea (OPA3) vs Dystonia (MECR) — CRITICAL clinical DDx; movement disorder type defines diagnosis",
            "detail": (
                "CHOREA (OPA3/Costeff): brief, irregular, purposeless, flowing, semi-random movements; "
                "arms > trunk > face; fluctuating; worse with stress/anxiety; better with sleep. "
                "Assessment: UHDRS Total Motor Score (TMS) choreiform subscore; baseline + 6M monitoring. "
                "DYSTONIA (MECR/MEPAN): sustained or repetitive muscle contractions → abnormal postures; "
                "generalised/segmental; oromandibular + limb dominant; fixed posture pattern. "
                "Key exam distinction: chorea flows (OPA3); dystonia holds (MECR). "
                "Chorea worsens with voluntary movement (OPA3); dystonia often has action/overflow pattern (MECR). "
                "Both can have superimposed features: OPA3 may have mild pyramidal dystonia; "
                "MECR may have choreic superimposition (40-50%). "
                "Primary movement DETERMINES diagnosis: chorea-dominant → OPA3/Huntington/DNAJC19; "
                "dystonia-dominant → MECR/NBIA1-7/DRD. "
                "Movement disorder specialist + video recording essential for phenotype classification."
            ),
        },
        {
            "term": "Optic-Atrophy-OPA3-Early-vs-MECR",
            "full": "Optic atrophy comparison: OPA3 (100%, infantile, ERG normal) vs MECR (80-90%, childhood, ERG abnormal late)",
            "detail": (
                "OPA3 optic atrophy: onset 6-24 months (infantile); bilateral symmetric temporal pallor; "
                "ERG NORMAL (photoreceptors spared — pure RGC/axonal disease); "
                "VEP prolonged P100 latency; VA 6/24-6/60 range; progressive. "
                "MECR optic atrophy: onset 2-5yr (early childhood, after dystonia onset); "
                "ERG abnormal in advanced disease (retinal dystrophy component 30-40%); "
                "OPA3 presents earlier (infantile vs childhood). "
                "OCT: both show RNFL thinning; OPA3 may show earlier and more severe papillomacular fibre loss. "
                "Ophthalmology DDx: ERG normal → OPA3 or OPA1 (dominant); "
                "ERG abnormal → MECR advanced, PLA2G6-INAD, Bardet-Biedl, mitochondrial retinopathy. "
                "Management: annual VEP + ERG + OCT + VA; low-vision aids when LogMAR >0.5; "
                "nystagmus management if present (rare in OPA3, more common in OPA1)."
            ),
        },
        {
            "term": "MRI-Normal-OPA3-vs-GP-Iron-MECR",
            "full": "Brain MRI — OPA3: usually NORMAL (no GP iron); MECR: GP iron 75-85% bilateral SWI",
            "detail": (
                "OPA3/Costeff: ~75% of patients have normal brain MRI on standard sequences (T1/T2/FLAIR). "
                "~25% show mild non-specific periventricular T2 hyperintensity (NOT leukodystrophy). "
                "CRITICALLY: NO globus pallidus iron on SWI/T2* in OPA3 (primary DDx from MECR/NBIA diseases). "
                "NO Eye-of-Tiger sign (PKAN/NBIA1). NO basal ganglia signal abnormality. "
                "MECR/MEPAN: GP bilateral SWI hypointensity in 75-85% (iron accumulation). "
                "The PRESENCE of GP iron on MRI effectively EXCLUDES OPA3 and favours MECR/NBIA series. "
                "The ABSENCE of GP iron on MRI in a 3-MGA-uria patient with chorea + optic atrophy "
                "strongly supports OPA3 diagnosis (pending gene sequencing confirmation). "
                "SWI sequence mandatory in all 3-MGA-uria patients to distinguish OPA3 from MECR/NBIA. "
                "Annual MRI not strictly required in OPA3 if MRI normal — genetics + clinical monitoring sufficient."
            ),
        },
        {
            "term": "VPA-Relative-Caution-OPA3-vs-Absolute-CI-MECR",
            "full": "VPA in OPA3 — RELATIVE caution; NOT absolute CI (unlike MECR where PDH hypolipoylation makes VPA lethal)",
            "detail": (
                "MECR VPA CI mechanism: PDH complex already hypolipoylated via MECR-LOF → "
                "VPA sequesters free CoA → further deprives PDH/alpha-KGDH of CoA → acute metabolic crisis. "
                "OPA3 does NOT involve lipoic acid pathway disruption — the MECR VPA-CI mechanism is ABSENT. "
                "OPA3 VPA concern: VPA inhibits beta-oxidation → ammonia cycle burden in any mitochondrial disease. "
                "Mitochondrial dysfunction (ANY cause) increases VPA hyperammonemia risk. "
                "OPA3 relative caution protocol: POLG screen mandatory (POLG+ → VPA absolute CI); "
                "monitor ammonia, LFTs, 3-MGA q3M if VPA used; use lowest effective dose. "
                "Clinical decision: if LEV + CLB insufficient for refractory seizures, VPA may be considered "
                "in OPA3 with monitoring, unlike MECR where VPA is categorically forbidden. "
                "Document rationale + informed consent + monitoring plan in medical record."
            ),
        },
        {
            "term": "Tetrabenazine-Deutetrabenazine-Chorea-OPA3",
            "full": "Tetrabenazine / Deutetrabenazine — VMAT2 inhibitors for OPA3 chorea; Level C; CYP2D6 genotype mandatory",
            "detail": (
                "VMAT2 inhibition: depletes pre-synaptic dopamine from basal ganglia → "
                "reduces striatal dopaminergic overdrive → choreiform movement reduction. "
                "Tetrabenazine: short-acting; 12.5-50 mg/day TID; CYP2D6 metabolism; "
                "active metabolites alpha- and beta-HTBZ (CYP2D6-dependent clearance). "
                "Deutetrabenazine: extended-release formulation; 6-48 mg/day BID; "
                "lower Cmax → fewer side effects; FDA-approved for HD chorea (off-label in OPA3). "
                "CYP2D6 MANDATORY: poor metabolisers (*4/*4, *5/*4): 50% dose reduction; "
                "ultra-rapid metabolisers: may need higher dose. "
                "Side effects: depression (15-30% — FDA black box suicidality warning); "
                "parkinsonism (15% — dose-dependent); sedation (20%); QTc prolongation (mild). "
                "PHQ-9 depression screening at baseline + q6M; discontinue if moderate-severe depression emerges. "
                "Response: 40-60% UHDRS choreiform score reduction; partial responders continue; "
                "non-responders: trial of clonazepam (GABA augmentation) or amantadine (NMDA antagonism) adjunct."
            ),
        },
        {
            "term": "OPA3-AD-vs-AR-DDx",
            "full": "OPA3-AD (dominant cataracts/optic atrophy) vs OPA3-AR (Costeff/chorea) — COMPLETELY DIFFERENT phenotypes, same gene",
            "detail": (
                "OPA3-AR (Costeff Syndrome, OMIM 258501): biallelic LOF; childhood optic atrophy + chorea + 3-MGA. "
                "OPA3-AD (OMIM 165300): heterozygous missense; bilateral CATARACTS (infantile) + adult-onset optic atrophy; "
                "NO chorea; NO 3-MGA; NO 3-MGA-uria; COMPLETELY DIFFERENT phenotype from Costeff. "
                "OPA3-AD variants: missense (p.Gln105Glu, p.Glu93Lys — note: different position from AR founder stop). "
                "Distinction: OPA3-AR = recessive LOF → complete protein absence → Costeff syndrome; "
                "OPA3-AD = dominant missense → dominant negative or haploinsufficiency → cataracts + mild optic atrophy. "
                "Cataracts as a feature are found in OPA3-AD; ABSENT in OPA3-AR Costeff. "
                "Family history pattern: OPA3-AR = consanguineous Iraqi Jewish (usually); "
                "OPA3-AD = autosomal dominant family history (affected parent). "
                "This dashboard covers OPA3-AR (Costeff) exclusively."
            ),
        },
        {
            "term": "POLG-Screen-Mandatory-OPA3",
            "full": "POLG1 screen MANDATORY in all OPA3/Costeff patients — standard mitochondrial disease protocol",
            "detail": (
                "POLG mutations cause mtDNA depletion/deletion syndromes (Alpers-Huttenlocher); "
                "VPA in POLG disease → fulminant hepatic failure. "
                "In OPA3: VPA is NOT absolutely contraindicated (unlike MECR), but POLG co-mutation would make "
                "VPA absolutely contraindicated. "
                "POLG screen establishes the true VPA risk in each OPA3 patient. "
                "If POLG positive: VPA becomes absolute CI in that OPA3 patient. "
                "Standard protocol: POLG + SURF1 + PDHA1 panel at time of OPA3 diagnosis. "
                "Turnaround: 4-6 weeks clinical WES or targeted POLG sequencing. "
                "CYP2D6 genotyping added simultaneously (for tetrabenazine dosing guidance)."
            ),
        },
        {
            "term": "OMIM-606580-OPA3-258501-Costeff",
            "full": "OMIM Gene OPA3: 606580 — Disease: Costeff Syndrome / 3-MGA Type III: 258501",
            "detail": (
                "OPA3 (MIM 606580): Optic Atrophy 3; outer mitochondrial membrane protein; 179 aa isoform-1; 19q13.2-q13.3. "
                "Costeff Syndrome (MIM 258501): 3-Methylglutaconic Aciduria Type III; Iraqi Jewish-enriched. "
                "Also known as: 3-methylglutaconyl aciduria type III; MGA3; Costeff opticoauditory syndrome (no hearing loss). "
                "First description: Costeff H, Gadoth N, Apter N, Biedner B, Elian E. "
                "A familial syndrome of infantile optic atrophy, movement disorder, and spastic ataxia. "
                "Am J Dis Child. 1989;143(2):133-137. "
                "Genetic characterization: Elpeleg ON et al. 3-Methylglutaconic aciduria type III, or "
                "Costeff syndrome: biochemical and genetic aspects. Eur J Pediatr. 1994;153(5):339-343. "
                "OPA3 cloning: Anikster Y et al. Am J Hum Genet. 2001;69(5):1218-1224. "
                "OPA3-AD: Reynier P et al. OPA3 gene mutations responsible for autosomal dominant optic "
                "atrophy and cataract. J Med Genet. 2004;41(9):e110."
            ),
        },
    ]
    return {"definitions": defs}
