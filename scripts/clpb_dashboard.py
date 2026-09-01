#!/usr/bin/env python3
"""CLPB 3-Methylglutaconic Aciduria Type VII (MGCA7) Dashboard.

MGCA7 = 3-Methylglutaconic Aciduria with Cataracts, Neurologic Involvement,
and Neutropenia (OMIM #616228).

CLPB (Caseinolytic Mitochondrial Matrix Peptidase Chaperone Subunit B, 707 aa,
11q13.1) is the human mitochondrial AAA+ disaggregase. Biallelic LOF → failure
to dissolve mitochondrial protein aggregates → 3-MGA overflow + cataracts +
neutropenia + variable neurological involvement.

KEY FACTS (EXAM / PRESCRIBING HIGHEST-YIELD):
  1. 3-MGA-uria Type VII — CATARACTS are PATHOGNOMONIC; no other 3-MGA type has cataracts
  2. Neutropenia (50-65%) — cyclic or chronic; G-CSF responsive; infection risk
  3. No DCM — KEY DDx from TAZ/Barth and TMEM70 (both have DCM)
  4. No SNHL — KEY DDx from SERAC1/MEGDEL (SNHL 100%)
  5. No hyperammonemia — KEY DDx from TMEM70 (hyperammonemia 90%)
  6. No significant lactic acidosis — KEY DDx from TMEM70 (pH <7.1 neonatal)
  7. Cataract surgery Level A — bilateral, early; prevents deprivation amblyopia
  8. G-CSF for severe neutropenia (ANC <0.5) — Level B; infection prophylaxis
  9. VPA — MODERATE CAUTION (mito disease, not absolute CI; POLG must be ruled out first)
 10. LEV preferred AED — no mito toxicity, no hepatic induction
 11. BAP-domain missense variants → milder phenotype; NBD1/NBD2 null → severe
 12. AR biallelic; both sexes; first described Wortmann 2015 (AJHG)

CLPB BIOLOGY:
CLPB (707 amino acids, 11q13.1) encodes the human mitochondrial AAA+
disaggregase. It forms heptameric rings (confirmed 2018 cryo-EM) that thread
aggregated proteins through the central pore using ATP hydrolysis, resolving
protein aggregates in the mitochondrial matrix — analogous to yeast Hsp104.

Domain architecture (707 aa):
  N-terminal extension / MTS (aa 1-93): mitochondrial targeting sequence;
    cleaved upon import; N-terminal mitochondrial localisation signal
  BAP domain (aa 94-232): Bulk solvent Access Point; substrate engagement;
    most disease variants in BAP → moderate phenotype
  Middle domain (aa 233-325): linker between BAP and AAA+ core;
    couples substrate engagement to ATP hydrolysis
  NBD1 / AAA1 (aa 326-524): first AAA+ ATPase module;
    Walker A (P-loop: GxxxxGK) + Walker B (DExH) motifs;
    pore loop (HBBS aa 495-510) threads substrate
  NBD2 / AAA2 (aa 525-707): second AAA+ ATPase module;
    regulatory; less ATP turnover than NBD1

CLPB function in mitochondria:
  Resolves protein aggregates formed during stress (heat, oxidative)
  Works with CLPP (mitochondrial serine protease) — processivity factor
  Loss → misfolded protein accumulation → respiratory chain dysfunction →
    secondary 3-MGA overflow + neutrophil maturation arrest + lens protein aggregation

Why cataracts?
  Lens crystallins (αA, αB, βγ) are highly aggregation-prone; constitutive
  chaperone demand. Mitochondria-dependent chaperone machinery (CLPB) contributes
  to lens crystallin quality control indirectly. CLPB LOF → crystallin aggregates
  form in lens epithelial cells → opacification → cataracts (infantile onset).
  This is a unique phenotypic feature not seen in other 3-MGA types (which have
  no crystallin involvement).

Why neutropenia?
  Neutrophil maturation requires intense protein synthesis in promyelocytes;
  high proteotoxic stress demand. CLPB LOF → promyelocyte arrest at mitotic spindle
  (protein aggregate-mediated) → maturation block → cyclic or chronic neutropenia.
  Pattern is similar to Barth (TAZ) neutropenia but mechanism differs entirely
  (Barth = cardiolipin remodelling failure → inner membrane stress).

PATHOGENIC VARIANT DISTRIBUTION (biallelic, AR, n=40, seed-545):
  BAP domain missense (aa 94-232): ~35% of alleles → moderate phenotype
    p.Thr268Met (BAP-middle), p.Arg322Gln, p.Leu205Pro — partial disaggregase function
  NBD1 core variants (Walker A/B; pore loop aa 326-524): ~30% of alleles → severe
    p.Arg468His, p.Arg468Cys — ATP hydrolysis impaired; ring assembly defective
  NBD2 missense (aa 525-707): ~20% of alleles → variable
    p.Arg941His equivalents (note: aa numbering varies by isoform)
  Frameshift / nonsense: ~10% of alleles → severe null; early stop → NMD
  Splice-site: ~5% of alleles → partial or complete LOF depending on exon

No single ethnic founder (unlike TMEM70 Roma or Barth X-linked);
de novo component in ~5% (gonadal mosaicism or de novo dominant reported but
most are biallelic recessive).

CLINICAL PHENOTYPE — CLPB MGCA7:
  CATARACTS (80-90%) — CARDINAL UNIQUE FEATURE:
    Bilateral; posterior cortical or nuclear; infantile onset (birth to 12 months)
    Ophthalmology referral at diagnosis mandatory
    Early cataract surgery prevents deprivation amblyopia (Level A)
    Visual rehab (contact lenses / glasses) post-surgery
  3-METHYLGLUTACONIC ACIDURIA (100%):
    Urine 3-MGA: 30-150 mmol/mol Cr (secondary overflow; less severe than TMEM70)
    Acylcarnitine panel: NORMAL (no C4-DC — DDx from Barth/TAZ)
    No lactate elevation (or mild only) — DDx from TMEM70 (severe lactic acidosis)
    No hyperammonemia — DDx from TMEM70 (NH3 50-500 µmol/L)
  NEUTROPENIA (50-65%):
    ANC <1.5 × 10⁹/L (cyclic or chronic)
    Cyclic pattern (10-21 day cycles) in ~30%; chronic in ~35%
    Infection risk: bacterial (Staphylococcal, Pseudomonal) during nadir
    G-CSF (filgrastim): Level B; effective in most; target ANC >1.0 × 10⁹/L
  NEUROLOGICAL (variable):
    Movement disorder (ataxia, dystonia): 50-65%; severity correlates with genotype
    Intellectual disability: mild-moderate 45%; severe 15%; normal 40%
    Seizures: 20-30%; focal or generalised; not the dominant phenotype
    Brain MRI: non-specific T2 changes in basal ganglia/thalamus in ~35%;
      cerebellar atrophy 20%; Leigh-like NOT characteristic (DDx from SERAC1)
    Progressive brain atrophy: 40% in severe cases (NBD1 null genotype)
  METABOLIC CRISES:
    Fever-triggered crises: ~30% of patients; intercurrent illness → 3-MGA spike
    Hypoglycaemia: 15-20%; not as severe or consistent as TMEM70
    Sick-day protocol important but less critical than TMEM70
"""

import random
from datetime import date

SEED = 545  # 40-patient cohort seed


def get_overview() -> dict:
    """CLPB MGCA7 — overview for /api/clpb/overview."""
    return {
        "generated": date.today().isoformat(),
        "disease": "3-Methylglutaconic Aciduria Type VII / MGCA7 / CLPB Deficiency",
        "gene": "CLPB; Caseinolytic Mitochondrial Matrix Peptidase Chaperone Subunit B; SKD3 homologue; AAA+ disaggregase",
        "chromosome": "11q13.1",
        "omim_gene": "616654",
        "omim_disease": "616228",
        "inheritance": "Autosomal Recessive (biallelic); both sexes equally affected; de novo dominant reported rarely (gonadal mosaicism <5%)",
        "prevalence": "Ultra-rare; <200 patients worldwide; no ethnic founder (pan-ethnic AR disease)",
        "protein": "CLPB 707 aa; heptameric AAA+ disaggregase ring; MTS(aa1-93)-BAP(aa94-232)-MD(aa233-325)-NBD1(aa326-524)-NBD2(aa525-707); mitochondrial matrix",
        "category": "Mitochondrial protein quality control / 3-Methylglutaconic Aciduria Type VII",
        "first_described": "Wortmann SB et al., AJHG 2015 — biallelic CLPB mutations in MGCA7",
        "kpis": {
            "cataracts_pct": 85,
            "neutropenia_pct": 60,
            "neurological_pct": 70,
            "three_mga_pct": 100,
            "dcm_pct": 0,
            "lactic_acidosis_severe_pct": 5,
            "hyperammonemia_pct": 0,
            "snhl_pct": 0,
            "vpa_risk": "MODERATE CAUTION (not absolute CI)",
            "c4dc_elevated": "ABSENT (normal acylcarnitine — KEY DDx from Barth/TAZ)",
        },
        "clinical_highlights": [
            "CATARACTS (85%) — PATHOGNOMONIC for 3-MGA Type VII; no other 3-MGA uria type has cataracts; bilateral infantile posterior cortical or nuclear opacification",
            "3-MGA-uria (100%) — secondary overflow from CLPB-driven mitochondrial protein quality control failure; urine organic acids diagnostic",
            "Neutropenia (60%) — cyclic or chronic ANC <1.5 × 10⁹/L; G-CSF responsive (Level B); infection prophylaxis mandatory during nadir",
            "NO DCM — critical DDx from TAZ/Barth (DCM 100%) and TMEM70 (DCM 85%); CLPB does NOT cause cardiomyopathy",
            "NO hyperammonemia — critical DDx from TMEM70 (NH3 50-500 µmol/L); urea cycle intact in CLPB",
            "NO severe lactic acidosis — critical DDx from TMEM70 (pH <7.1); CLPB OXPHOS relatively preserved",
            "NO SNHL — critical DDx from SERAC1/MEGDEL (SNHL 100% PATHOGNOMONIC); CLPB cochlear hair cells unaffected",
            "Normal acylcarnitine (no C4-DC) — critical DDx from TAZ/Barth (C4-DC elevated, PATHOGNOMONIC)",
            "BAP-domain missense → milder phenotype (partial disaggregase); NBD1/NBD2 null → severe with brain atrophy",
            "VPA: MODERATE CAUTION — mito disease risk; NOT absolute CI unlike TMEM70/POLG; rule out POLG first; LEV preferred",
            "Cataract surgery Level A — early bilateral surgery prevents deprivation amblyopia; vision rehab post-surgery essential",
            "G-CSF (filgrastim) Level B — effective in most neutropenic CLPB patients; dose: 5-10 µg/kg/day SC; target ANC >1.0",
        ],
        "contraindications": [
            {
                "drug": "VPA (Valproate)",
                "level": "MODERATE CAUTION — not absolute CI",
                "reason": "Any mitochondrial disease increases VPA hepatotoxicity risk (CoA sequestration, Complex I inhibition). CLPB is NOT direct OXPHOS deficiency so risk is moderate not absolute. Rule out POLG mutations first (POLG = absolute CI). If VPA already on board, monitor LFTs monthly, ammonia, and consider switch to LEV or CLB. Never initiate without metabolic team review.",
            },
            {
                "drug": "POLG — mandatory genetic test before ANY valproate",
                "level": "MANDATORY TEST (not a drug CI)",
                "reason": "POLG mutations in the same patient (or dual diagnosis) would convert VPA risk from moderate to absolute CI. All CLPB patients should have POLG excluded (POLG gene panel or at minimum p.Ala467Thr, p.Trp748Ser, p.Gly848Ser screening) before any AED decision involving valproate.",
            },
            {
                "drug": "Propofol",
                "level": "AVOID — PRIS risk",
                "reason": "Propofol inhibits mitochondrial Complex I; PRIS (propofol infusion syndrome) risk elevated in any mitochondrial disease including CLPB. Prefer ketamine + inhalational agents (sevoflurane/desflurane). Alert anaesthesia team at every procedural encounter.",
            },
            {
                "drug": "Leucine restriction diet",
                "level": "NOT INDICATED — avoid",
                "reason": "Leucine restriction helps 3-MGA Type I (AUH — primary enzyme defect in leucine catabolism). CLPB 3-MGA is secondary overflow from mitochondrial protein QC failure; leucine restriction has no mechanistic basis and risks nutritional harm.",
            },
            {
                "drug": "Fasting / prolonged NPO",
                "level": "CAUTION (milder than TMEM70)",
                "reason": "Fasting can trigger metabolic crisis in CLPB but risk is lower than TMEM70 (which has obligate lactic acidosis on any catabolism). Maintain adequate dextrose during illness; sick-day protocol recommended. IV dextrose if prolonged NPO. Less strict than TMEM70 but still important in NBD1-null severe cases.",
            },
            {
                "drug": "Live vaccines during severe neutropenia",
                "level": "CONTRAINDICATED — timing critical",
                "reason": "ANC <0.5 × 10⁹/L = severe neutropenia; live attenuated vaccines (MMR, varicella, rotavirus) are contraindicated. Give after G-CSF correction and ANC >1.0 × 10⁹/L. Inactivated vaccines can proceed. Immunology co-management recommended.",
            },
        ],
        "thresholds": [
            {"marker": "Urine 3-methylglutaconate", "cutoff": ">20 mmol/mol Cr", "interpretation": "Elevated 3-MGA confirms diagnosis; 30-150 mmol/mol Cr typical for CLPB; >200 suggests TMEM70 or other severe mito disease"},
            {"marker": "Absolute Neutrophil Count (ANC)", "cutoff": "<1.5 × 10⁹/L", "interpretation": "Neutropenia — start infection monitoring; if <0.5 × 10⁹/L (severe) initiate G-CSF (filgrastim 5 µg/kg/day SC); prophylactic antibiotics during nadir"},
            {"marker": "Blood lactate", "cutoff": ">3 mmol/L persistent", "interpretation": "Mild elevation acceptable in CLPB; persistent >3 mmol/L suggests metabolic stress or intercurrent illness; >5 mmol/L requires urgent metabolic review"},
            {"marker": "Blood ammonia (NH3)", "cutoff": ">80 µmol/L", "interpretation": "Should be NORMAL in CLPB; if elevated, reconsider diagnosis (rule out TMEM70, UCD, OTC); simultaneous 3-MGA + elevated NH3 = TMEM70 not CLPB"},
            {"marker": "Serum transaminases (ALT/AST)", "cutoff": ">2× ULN on VPA", "interpretation": "If VPA in use, ALT/AST >2× ULN = stop VPA immediately; risk of VPA-induced hepatotoxicity in any mitochondrial disease; switch to LEV"},
            {"marker": "Lens opacity on slit-lamp", "cutoff": "Any posterior cortical opacity in infant", "interpretation": "Posterior cortical or nuclear cataract in neonate / infant with 3-MGA = CLPB until proven otherwise; urgent ophthalmology referral; cataract surgery before 8 weeks if visually significant"},
        ],
        "ddx_table": [
            {
                "disease": "TAZ — Barth Syndrome (3-MGA Type II)",
                "shared": "3-MGA-uria, neutropenia",
                "distinguishing": "TAZ: DCM 100% (absent in CLPB) + C4-DC elevated (PATHOGNOMONIC, absent in CLPB) + X-linked males only + NO cataracts. CLPB: cataracts 85% + AR + no DCM + no C4-DC.",
            },
            {
                "disease": "TMEM70 — Complex V Deficiency (3-MGA Type VI)",
                "shared": "3-MGA-uria",
                "distinguishing": "TMEM70: lactic acidosis (pH <7.1) + hyperammonemia (NH3 50-500) + DCM 85% + NO cataracts. CLPB: cataracts 85% + NO hyperammonemia + NO lactic acidosis + NO DCM.",
            },
            {
                "disease": "SERAC1 — MEGDEL (3-MGA Type V)",
                "shared": "3-MGA-uria, neurological involvement",
                "distinguishing": "SERAC1: SNHL 100% (PATHOGNOMONIC, absent in CLPB) + Leigh-like MRI (bilateral putamen) + neonatal liver cholestasis. CLPB: cataracts + neutropenia + NO SNHL + NO neonatal liver.",
            },
            {
                "disease": "DNAJC19 — DCMA (3-MGA Type III)",
                "shared": "3-MGA-uria, DCM (NO — CLPB has no DCM)",
                "distinguishing": "DNAJC19: DCM 100% + cerebellar ataxia 95% + male genital anomalies + Hutterite founder + NO cataracts + NO neutropenia. CLPB: cataracts 85% + neutropenia 60% + NO DCM.",
            },
            {
                "disease": "AUH — 3-MGA Type I",
                "shared": "3-MGA-uria (primary, higher in AUH)",
                "distinguishing": "AUH: PRIMARY enzyme defect (3-methylglutaconyl-CoA hydratase) + leucine restriction helps + 3-HMG normal (vs elevated in HMGCL deficiency) + NO cataracts + NO neutropenia. CLPB: secondary 3-MGA + cataracts + neutropenia.",
            },
            {
                "disease": "Primary neutropenia (Kostmann / SCN)",
                "shared": "Neutropenia",
                "distinguishing": "Kostmann/SCN: ELANE/HAX1 mutation + no 3-MGA + no cataracts + bone marrow shows promyelocyte arrest (same as CLPB on biopsy — biopsy alone NOT diagnostic). CLPB: 3-MGA + cataracts + biallelic CLPB mutations on WES.",
            },
            {
                "disease": "Primary cataracts (MYH9, CRYGD, CRYBB2)",
                "shared": "Cataracts",
                "distinguishing": "Primary cataracts: isolated, no 3-MGA, no neutropenia, normal urine organic acids. CLPB: cataracts + 3-MGA + ± neutropenia; urine organic acids diagnostic.",
            },
        ],
    }


# ── breakdown ─────────────────────────────────────────────────────────────────
def get_breakdown() -> dict:
    """CLPB MGCA7 — patient breakdown for /api/clpb/breakdown."""
    rng = random.Random(SEED)
    n = 40

    # Phenotype groups
    phenotype_groups = [
        ("Classic: cataracts + 3-MGA + neutropenia + neurological", 22),
        ("Moderate: cataracts + 3-MGA + neutropenia, mild neuro", 12),
        ("Attenuated: cataracts + 3-MGA, no significant neutropenia, normal cognition", 6),
    ]
    assert sum(c for _, c in phenotype_groups) == n

    # Variant distribution (biallelic AR)
    variant_dist = [
        {"variant": "BAP domain missense (aa 94-232) — substrate engagement impaired", "n_alleles": 28, "pct": 35, "effect": "Partial disaggregase function; moderate phenotype; cataracts + mild-moderate neuro; less severe neutropenia; BAP domain variants cluster around substrate binding loops"},
        {"variant": "NBD1 Walker A/B missense (pore loop aa 326-524) — ATP hydrolysis impaired", "n_alleles": 24, "pct": 30, "effect": "Severe phenotype; ring assembly defective; complete loss of disaggregase function; brain atrophy 65%; severe neutropenia; cataracts early onset"},
        {"variant": "NBD2 missense (aa 525-707) — regulatory domain disrupted", "n_alleles": 16, "pct": 20, "effect": "Variable; NBD2 modulates NBD1 activity; moderate-severe phenotype depending on affected residue; NBD2 null slightly milder than NBD1 null"},
        {"variant": "Frameshift / nonsense (premature stop → NMD)", "n_alleles": 8, "pct": 10, "effect": "Complete null; NMD-mediated; severe neonatal presentation; cataracts at birth; severe neutropenia from birth; worst neurological outcome"},
        {"variant": "Splice-site variants (intronic; variable exon skipping)", "n_alleles": 4, "pct": 5, "effect": "Variable LOF depending on residual splicing; partial or complete disaggregase loss; prognosis correlates with residual CLPB protein on western blot"},
    ]

    # Treatment distribution
    treatment_dist = [
        {"treatment": "Cataract surgery (bilateral)", "n": 34, "pct": 85, "indication": "Level A — visually significant cataracts; bilateral; surgery before 8-12 weeks prevents deprivation amblyopia; contact lens / aphakic glasses post-surgery"},
        {"treatment": "Visual rehabilitation (aphakic glasses / CL)", "n": 34, "pct": 85, "indication": "Level A — post-cataract-surgery correction mandatory; aphakic glasses (age <2yr) then contact lenses; patching therapy for amblyopia if asymmetric"},
        {"treatment": "G-CSF (filgrastim)", "n": 24, "pct": 60, "indication": "Level B — severe neutropenia (ANC <0.5 × 10⁹/L); 5-10 µg/kg/day SC; target ANC >1.0 × 10⁹/L; prophylactic antibiotics during nadir cycles"},
        {"treatment": "Prophylactic antibiotics (neutropenic nadir)", "n": 20, "pct": 50, "indication": "Level B — co-trimoxazole or ciprofloxacin during ANC nadir <0.5 × 10⁹/L; antifungal prophylaxis (fluconazole) in severe sustained neutropenia"},
        {"treatment": "LEV (levetiracetam)", "n": 12, "pct": 30, "indication": "Level B — seizure management; preferred AED in mitochondrial disease; renal excretion; no mito toxicity; no hepatic induction; seizures in 20-30% of CLPB"},
        {"treatment": "Occupational and physiotherapy", "n": 28, "pct": 70, "indication": "Level A — movement disorder (ataxia/dystonia) + developmental delay; early intervention; adaptive equipment; PT from diagnosis in severe cases"},
        {"treatment": "CoQ10 empirical", "n": 14, "pct": 35, "indication": "Level D — empirical mitochondrial cofactor; no controlled evidence in CLPB; sometimes tried given secondary OXPHOS dysfunction; generally safe"},
        {"treatment": "Riboflavin (B2) empirical", "n": 10, "pct": 25, "indication": "Level D — empirical; no specific evidence in CLPB; some centres use for mito support; monitor urine colour as dose guide"},
        {"treatment": "Intravenous immunoglobulin (IVIG)", "n": 6, "pct": 15, "indication": "Level C — for recurrent bacterial infections during neutropenic episodes unresponsive to G-CSF; provides passive immunity; not routine"},
        {"treatment": "Special educational support", "n": 18, "pct": 45, "indication": "Level A — intellectual disability (mild-moderate in 45%; severe in 15%); IEP from school entry; speech-language therapy; AAC devices if needed"},
        {"treatment": "Bone marrow transplant (HSCT)", "n": 3, "pct": 8, "indication": "Level D (case reports) — severe refractory neutropenia in selected severe cases; HSCT can correct neutropenia; neurological outcomes post-HSCT variable; centre-based decision"},
        {"treatment": "Sick-day protocol", "n": 40, "pct": 100, "indication": "Level B — intercurrent fever/illness triggers 3-MGA crisis; oral glucose drinks; hospital threshold: vomiting >2 episodes, any fever >38°C; IV dextrose if unable to feed"},
    ]

    # Biomarker summary
    biomarker_summary = {
        "three_mga_mean_mmol_cr": 75,
        "three_mga_range_mmol_cr": "30-150",
        "three_mga_pct": 100,
        "cataracts_pct": 85,
        "neutropenia_pct": 60,
        "severe_neutropenia_pct": 30,
        "lactic_acidosis_severe_pct": 5,
        "hyperammonemia_pct": 0,
        "dcm_pct": 0,
        "snhl_pct": 0,
        "c4dc_elevated_pct": 0,
        "movement_disorder_pct": 60,
        "id_mild_moderate_pct": 45,
        "id_severe_pct": 15,
        "normal_cognition_pct": 40,
        "brain_atrophy_pct": 40,
        "seizures_pct": 25,
    }

    # Neutropenia profile
    neutropenia_profile = [
        {"type": "Severe cyclic ANC <0.5 (10-21 day cycles)", "n": 12, "pct": 30, "g_csf_response": "90% respond; ANC corrects to >1.0 within 48h of G-CSF"},
        {"type": "Chronic moderate ANC 0.5-1.5 (persistent)", "n": 12, "pct": 30, "g_csf_response": "70% on G-CSF maintenance; target ANC >1.0; prophylactic antibiotics"},
        {"type": "No significant neutropenia (ANC >1.5)", "n": 16, "pct": 40, "g_csf_response": "G-CSF not required; monitor CBC every 3 months in first 2 years"},
    ]

    # Neurological features
    neuro_features = [
        {"feature": "Cerebellar ataxia", "n": 20, "pct": 50, "notes": "Gait ataxia predominant; mild-severe; correlates with NBD1/NBD2 genotype"},
        {"feature": "Dystonia", "n": 12, "pct": 30, "notes": "Focal or segmental; arms > legs; GPi-DBS anecdotally reported in 1 case"},
        {"feature": "Intellectual disability (mild-moderate)", "n": 18, "pct": 45, "notes": "IQ 50-70 range in moderate; educational support mandatory"},
        {"feature": "Intellectual disability (severe)", "n": 6, "pct": 15, "notes": "NBD1 null genotype; brain atrophy + developmental regression"},
        {"feature": "Normal cognition", "n": 16, "pct": 40, "notes": "BAP domain variants predominantly; near-normal IQ; school difficulties mild"},
        {"feature": "Epilepsy", "n": 10, "pct": 25, "notes": "Focal or generalised; not the dominant phenotype; responds to LEV in 75%"},
        {"feature": "Progressive brain atrophy on MRI", "n": 16, "pct": 40, "notes": "Cerebral and cerebellar; worse in NBD1/NBD2 null; not Leigh-like (DDx SERAC1)"},
        {"feature": "Spasticity", "n": 8, "pct": 20, "notes": "Pyramidal signs; baclofen +/- intrathecal in severe; physiotherapy mandatory"},
    ]

    return {
        "generated": date.today().isoformat(),
        "cohort": n,
        "seed": SEED,
        "phenotype_groups": [{"group": g, "n": c, "pct": round(c / n * 100)} for g, c in phenotype_groups],
        "variant_distribution": variant_dist,
        "treatment_distribution": treatment_dist,
        "neutropenia_profile": neutropenia_profile,
        "neurological_features": neuro_features,
        "biomarker_summary": biomarker_summary,
        "outcomes": {
            "cataract_surgery_pct": 85,
            "normal_vision_post_surgery_pct": 65,
            "seizure_free_on_aed_pct": 70,
            "independent_ambulation_pct": 60,
            "id_any_severity_pct": 60,
            "neutropenia_g_csf_controlled_pct": 80,
            "brain_atrophy_progressive_pct": 40,
            "median_age_diagnosis_months": 8,
            "median_age_cataract_surgery_months": 4,
        },
    }


# ── definitions ───────────────────────────────────────────────────────────────
def get_definitions() -> dict:
    """CLPB MGCA7 — definitions for /api/clpb/definitions."""
    return {
        "generated": date.today().isoformat(),
        "disease": "3-Methylglutaconic Aciduria Type VII / MGCA7 / CLPB Deficiency",
        "gene": "CLPB",
        "omim_gene": "616654",
        "omim_disease": "616228",
        "definitions": [
            {
                "term": "CLPB — Mitochondrial AAA+ Disaggregase",
                "definition": "CLPB (Caseinolytic Mitochondrial Matrix Peptidase Chaperone Subunit B; 707 amino acids; 11q13.1) is the human mitochondrial AAA+ disaggregase, homologous to yeast Hsp104. It forms heptameric rings that thread misfolded/aggregated proteins through the central pore using ATP hydrolysis energy, solubilising protein aggregates in the mitochondrial matrix. CLPB cooperates with CLPP (mitochondrial serine protease) as its processivity factor. LOF → accumulation of protein aggregates in the mitochondrial matrix → downstream 3-MGA overflow, lens crystallin aggregation (cataracts), and neutrophil maturation arrest.",
                "relevance": "CLPB protein can be assessed on western blot from fibroblasts or lymphoblasts — absent or severely reduced in nonsense/frameshift variants; reduced but present in missense hypomorphs. Fibroblast disaggregase activity can be measured using luciferase aggregation assays (research setting). Muscle biopsy: non-specific mitochondrial changes; not diagnostic. WES/WGS with CLPB panel sequencing is the standard diagnostic test.",
            },
            {
                "term": "Why Cataracts Are Pathognomonic for 3-MGA Type VII (CLPB)",
                "definition": "Infantile bilateral cataracts are the hallmark feature that distinguishes 3-MGA Type VII (CLPB) from ALL other 3-MGA uria types. Lens crystallins (αA-crystallin, αB-crystallin, βγ-crystallins) are constitutively aggregation-prone proteins that require chaperone machinery for lifelong quality control. In the lens epithelium, CLPB contributes to crystallin disaggregation. CLPB LOF → crystallin aggregates in the lens epithelial cells → progressive opacification → infantile cataracts. This mechanism is unique among the 3-MGA diseases because it involves extramitochondrial crystallin biology as an indirect downstream effect.",
                "relevance": "Cataracts on ophthalmological examination in any infant with 3-MGA-uria on urine organic acids = CLPB until proven otherwise. Ophthalmology should be part of every metabolic screen for 3-MGA diseases. Cataract surgery must be early (before 8-12 weeks for visually significant opacity) to prevent irreversible deprivation amblyopia. Contact lenses or aphakic glasses are needed post-surgery. This is the clinical clue that makes CLPB distinct: cataracts + elevated 3-MGA + ± neutropenia = CLPB — not other 3-MGA types.",
            },
            {
                "term": "CLPB Neutropenia: Mechanism vs TAZ/Barth",
                "definition": "Neutropenia in CLPB results from promyelocyte maturation arrest in the bone marrow. During rapid neutrophil production, promyelocytes undergo intense protein synthesis with high proteotoxic stress. CLPB LOF → protein aggregate accumulation during mitotic spindle assembly → promyelocyte arrest → cyclic or chronic neutropenia. This mechanism is fundamentally different from Barth syndrome (TAZ), where neutropenia results from cardiolipin remodelling failure and inner mitochondrial membrane stress — but the outcome (neutrophil maturation arrest on bone marrow biopsy) is similar. KEY DISTINGUISHING POINT: CLPB neutropenia occurs with cataracts and without DCM or C4-DC elevation, whereas Barth neutropenia occurs with DCM and C4-DC elevation and without cataracts.",
                "relevance": "Bone marrow biopsy alone cannot distinguish CLPB neutropenia from Kostmann syndrome (ELANE mutations) or Barth syndrome — both show promyelocyte arrest. The full clinical picture (3-MGA on organic acids, cataracts, absence of DCM, absence of C4-DC) is essential. G-CSF (filgrastim 5-10 µg/kg/day SC) is effective in most CLPB neutropenia — similar efficacy to Kostmann syndrome. Target ANC >1.0 × 10⁹/L. Live vaccines must be withheld during ANC <0.5 × 10⁹/L.",
            },
            {
                "term": "3-MGA-uria Type VII vs Type VI (CLPB vs TMEM70) — The Critical Prescribing DDx",
                "definition": "CLPB (Type VII) and TMEM70 (Type VI) are the two most recently characterised 3-MGA uria types and share the designation of 'mitochondrial quality control disease'. They differ fundamentally in mechanism and severity: TMEM70 = ATP synthase assembly failure → ATP production collapses → lactic acidosis + hyperammonemia + DCM. CLPB = protein disaggregase failure → protein aggregates → 3-MGA + cataracts + neutropenia. KEY PRESCRIBING DIFFERENCES: (1) Ketogenic diet: CONTRAINDICATED in TMEM70 (Complex V absent) but NOT in CLPB (can try for seizures if appropriate); (2) VPA: ABSOLUTE CI in TMEM70 (Complex I + ammonia) vs MODERATE CAUTION in CLPB (mito risk but not absolute); (3) Ammonia scavengers: mandatory in TMEM70, NOT needed in CLPB; (4) IV glucose emergency: mandatory first-line in TMEM70, supportive only in CLPB.",
                "relevance": "A neonate with 3-MGA + cataracts = CLPB. A neonate with 3-MGA + lactic acidosis + hyperammonemia = TMEM70. These two diseases share a designation (3-MGA uria) but have almost opposite management priorities for cataracts, neutropenia, ammonia, ketogenic diet, and VPA. Never apply TMEM70 management protocols to CLPB or vice versa.",
            },
            {
                "term": "CLPB Genotype-Phenotype Correlation",
                "definition": "Genotype-phenotype correlation in CLPB is moderate. BAP-domain missense variants (aa 94-232) produce partial disaggregase function (hypomorphic) and are associated with milder neurological outcomes — normal or near-normal cognition in ~50% of BAP-only cases, milder neutropenia. NBD1 Walker A/B missense or null variants (ATP hydrolysis abolished) produce severe phenotype — brain atrophy in 65%, severe ID in 30%, severe cyclic neutropenia in 70%. NBD2 variants are intermediate. Nonsense/frameshift = complete null = most severe. However, significant inter-familial variability exists even within the same variant class, suggesting modifier gene effects or stochastic mitochondrial aggregate thresholds.",
                "relevance": "BAP-domain homozygous missense patients may be functionally near-normal on school entry and easily misattributed to isolated cataracts if organic acids are not checked. NBD1 null / frameshift patients have progressive brain atrophy and need intensive neurodevelopmental support. Genotype should be reported to the family for prognosis but should not be used as sole basis for treatment decisions — clinical monitoring remains primary.",
            },
            {
                "term": "VPA Risk Stratification in CLPB vs Absolute CI in TMEM70/POLG",
                "definition": "VPA hepatotoxicity risk in mitochondrial disease is mechanism-dependent. POLG mutations and TMEM70 create specific absolute contraindications: POLG because VPA inhibits mtDNA replication (POLG-dependent) directly, causing mtDNA depletion; TMEM70 because VPA raises ammonia AND inhibits Complex I in a patient already hyperammonemic and ATP-depleted. CLPB does not have either of these direct mechanisms — CLPB deficiency does not directly impair mtDNA replication or Complex I, and CLPB patients do not have hyperammonemia. Therefore VPA is MODERATE CAUTION (not absolute CI) in CLPB. POLG must always be excluded first (POLG panel sequencing before ANY valproate in any mitochondrial disease). If VPA is used, monthly LFTs, ammonia, and drug levels are mandatory.",
                "relevance": "This distinction matters in practice: a child with CLPB epilepsy who responds only to VPA can be managed on VPA with careful monitoring, whereas a child with TMEM70 or POLG mutation must NEVER receive VPA. Metabolic team must approve VPA initiation in any mitochondrial disease. LEV is the preferred first-line AED regardless: no hepatic metabolism, no ammonia effect, no mito toxicity.",
            },
            {
                "term": "Diagnosis Pathway: 3-MGA + Cataracts → CLPB",
                "definition": "The diagnostic pathway for CLPB is triggered by: (1) Elevated 3-methylglutaconate + 3-methylglutarate on urine organic acids (30-150 mmol/mol Cr); AND (2) Bilateral infantile cataracts on ophthalmological examination. Confirmation: WES with CLPB gene sequencing (biallelic pathogenic variants). Supportive: CLPB protein western blot from fibroblasts (absent or severely reduced in nulls; reduced in hypomorphs). Ancillary: CBC with differential (neutropenia 60%); brain MRI (progressive atrophy in 40%); electroencephalogram (epilepsy 25%); developmental assessment. Do not delay cataract surgery awaiting genetic confirmation — visual threat is immediate.",
                "relevance": "When a paediatric ophthalmologist identifies infantile cataracts and the metabolic team detects 3-MGA on routine newborn or organic acid screening, the combination is virtually diagnostic of CLPB — proceed to WES and CLPB western blot while managing cataracts surgically. Neutrophil count monitoring from diagnosis. Neuroimaging baseline MRI at diagnosis and 12-monthly intervals in NBD1-null genotypes (progressive atrophy monitoring).",
            },
            {
                "term": "Sick-Day Protocol and Emergency Management in CLPB",
                "definition": "CLPB metabolic crises are triggered by intercurrent illness (fever, vomiting, surgery) causing catabolism, which spikes 3-MGA and can transiently impair respiratory chain function. The severity is substantially milder than TMEM70 (which has obligate lactic acidosis with any catabolism). Sick-day protocol for CLPB: (1) maintain oral glucose drinks (high-carbohydrate; Lucozade equivalent); (2) hospital attendance threshold: vomiting >2 episodes + fever >38°C; (3) IV 10% dextrose if unable to feed orally; (4) monitor CBC — illness frequently exacerbates neutropenia; (5) blood lactate on admission (>3 mmol/L persistent = metabolic team review). Emergency letter for local ED should document: CLPB, 3-MGA uria Type VII, dextrose requirement, neutropenia risk, no VPA, no propofol.",
                "relevance": "CLPB families need a sick-day action card but the protocols are less strict than TMEM70. The biggest sick-day risk in CLPB is not lactic acidosis crisis but neutropenic sepsis (fever + ANC <0.5 = emergency G-CSF dose + IV antibiotics). G-CSF sick-day dose (double maintenance dose at first sign of fever during nadir) is a key family-held management strategy. All patients should have a metabolic emergency contact number.",
            },
        ],
    }
