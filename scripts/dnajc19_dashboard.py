"""
DCMA Syndrome — DNAJC19 (Dilated Cardiomyopathy with Ataxia)
=============================================================
40-patient cohort · DNAJC19 (3q26.33) · Autosomal Recessive · ~30-40 families worldwide 2026
First described: Davey et al. 2006 (Am J Hum Genet) — Hutterite/Old Colony Mennonite pedigrees
Key DDx from OPA3/Costeff: NO optic atrophy (OPA3 100%), DCM is CARDINAL (absent in OPA3)
Shared biomarker with OPA3: 3-methylglutaconic aciduria (3-MGA) — BOTH Type III classification
Key DDx from MECR/MEPAN: NO dystonia (MECR), NO GP iron (MECR), cerebellar ataxia NOT dystonia

DNAJC19 BIOLOGY:
DNAJC19 (116 amino acids, 3q26.33) encodes a DnaJ/Hsp40 co-chaperone of the TIM23 mitochondrial
inner membrane translocase complex (also called Tim14 or DNAJC19/Tim14).
Single-pass type II membrane protein: N-terminal MTS (aa 1-28), TM domain (aa 29-51, IMS anchor),
J-domain (aa 60-116) with critical HPD tripeptide.
Function: Co-chaperone for mtHsp70 (mortalin/HSPA9); drives ATP-dependent pulling of preproteins
  from the IMS through TIM23 channel into the mitochondrial matrix.
LOF mechanism: TIM23 complex destabilized → nuclear-encoded mito preprotein import impaired →
  OXPHOS complex I/IV assembly failure (both require matrix-imported subunits) →
  cardiomyocyte ATP deficit → systolic dysfunction → dilated cardiomyopathy.
3-MGA mechanism: OXPHOS dysfunction → HMG-CoA pathway overflow → 3-methylglutaconyl-CoA
  accumulates → excreted as 3-methylglutaconic acid (same overflow as OPA3, different gene).

PROTEIN STRUCTURE (116 aa, 3q26.33):
  MTS (aa 1-28): mitochondrial targeting sequence; cleaved upon matrix entry.
  TM domain (aa 29-51): single-pass IMS anchor; type II orientation.
  Linker (aa 52-59): flexible; connects TM to J-domain.
  J-domain (aa 60-116): DnaJ-type; HPD tripeptide (His-Pro-Asp) at aa 95-97 is CRITICAL;
    interacts with HSPA9/mortalin ATPase domain → stimulates ATP hydrolysis → preprotein release.
  Founder c.130-1G>C splice acceptor (intron 3) → exon 4 skip → frameshift → premature stop
    within J-domain → truncated protein missing HPD → complete J-domain loss → null.

PATHOGENIC VARIANT DISTRIBUTION (biallelic LOF, n=40 patients, seed-537):
  Hutterite/Mennonite founder c.130-1G>C (splice, intron 3, exon 4 skip): ~88% of alleles
  c.33_34delAT (frameshift, exon 1): ~5% (European non-founder families)
  J-domain missense p.His95Arg (HPD motif, total J-domain loss): ~4%
  TM domain missense p.Leu41Pro (disrupts IMS anchor): ~2%
  Large deletion exon 3-5: ~1% (rare, complete null)

CLINICAL PHENOTYPE — DCMA SYNDROME:
  DILATED CARDIOMYOPATHY (100%) — CARDINAL FEATURE:
    Systolic dysfunction (EF typically 20-40%); onset birth to 2 years (median 8 months).
    LV dilation on echo; ECG: LBBB, ST changes, conduction defects in ~60%.
    Natural history: ~30% require heart transplant; ~15% cardiac death without transplant.
    NON-ISCHEMIC, NON-INFLAMMATORY: endomyocardial biopsy shows mitochondrial morphology changes.
    KEY DDx: DNAJC19-DCM vs Barth (TAZ-DCM+BTHS): DNAJC19 has cerebellar ataxia, Barth has skeletal myopathy + neutropenia.
  CEREBELLAR ATAXIA (95%) — NON-PROGRESSIVE:
    Gait ataxia, broad-based gait, intention tremor; onset coincides with ambulation (~12-18 months).
    NON-PROGRESSIVE: static or very slowly worsening over decades (key feature vs SCA/FRDA).
    MRI: cerebellar vermis hypoplasia (~50%); mild superior cerebellar atrophy later.
    NO pyramidal signs (unlike OPA3 spastic paraplegia); NO optic atrophy (unlike OPA3/MECR).
  3-METHYLGLUTACONIC ACIDURIA (100%):
    Level: 30-150 mmol/mol creatinine (Type III classification, overlaps OPA3 range).
    Mechanism: OXPHOS failure → HMG-CoA overflow → 3-MGA excretion (same overflow as OPA3).
    KEY NEGATIVE: normal acylcarnitine profile (unlike Barth/BTHS which has elevated C4-DC/3-MGA-DC).
  MALE GENITAL ANOMALIES (70-80% of males):
    Cryptorchidism most common; hypospadias in subset; ABSENT in females.
    Mechanism unclear: DNAJC19 may be required in testicular Leydig/Sertoli cell mito function.
    Surgical correction for cryptorchidism (orchidopexy); standard urological management.
  MILD INTELLECTUAL DISABILITY (40-60%):
    IQ typically 60-80; language relatively preserved; memory affected less than executive function.
    Supported education sufficient in most; independent living possible with mild forms.
    NOT progressive cognitive decline (distinguishes from NCL, CLN12, metabolic dementias).
  SHORT STATURE (60-70%):
    Below 3rd percentile; proportionate; GH axis normal; no endocrine cause.
    Mitochondrial dysfunction during growth may impair IGF-1 signaling.
  SEIZURES (< 15%) — RARE:
    Focal or myoclonic if present; NOT a defining feature of DCMA.
    LEV preferred (renal excretion; no mito toxicity); VPA moderate caution.
    Contrast: OPA3 30-40%, MECR 40-50% seizure prevalence.
  OPTIC ATROPHY: ABSENT (KEY DDx from OPA3 100% and MECR 80-90%)
  HEARING: Normal (KEY DDx from DCAF17/WSS where SNHL is cardinal)
  ENDOCRINE: Normal (KEY DDx from DCAF17/WSS where hypogonadism + diabetes are cardinal)
  BRAIN MRI: Normal or mild cerebellar changes; NO GP iron (DDx NBIA); NO leukodystrophy (DDx FAHN)

TREATMENT & PHARMACOGENOMICS:
  ACE Inhibitors / ARBs: DCM — Level A (captopril/enalapril/lisinopril)
    Reduce afterload; prevent LV remodeling; first-line for DCM management.
    Start at diagnosis; titrate to tolerability.
  Beta-Blockers: DCM — Level A (carvedilol/bisoprolol/metoprolol succinate)
    Reverse cardiac remodeling; reduce HF mortality; carvedilol preferred for additional alpha-block.
    Initiate after ACE stabilization; start low, titrate slowly.
  Diuretics: DCM (fluid overload) — Level A (furosemide/spironolactone)
    Symptom relief; furosemide for acute decompensation; spironolactone for aldosterone blockade.
  L-Carnitine: Carnitine depletion — Level B (50-100 mg/kg/day)
    Secondary carnitine depletion common in OXPHOS disorders; supplement if C0 low-normal.
    May improve mitochondrial fatty acid oxidation efficiency.
  Heart Transplant: Refractory DCM — Level B
    ~30% require transplant; outcomes generally good post-transplant.
    Neurological features (ataxia, mild ID) do not improve post-transplant (mito systemic disease).
    Timing: EF < 25% with medical-refractory symptoms.
  LEV (Levetiracetam): Seizures (rare) — PREFERRED — Level B
    Renal excretion; no hepatic metabolism; no mitochondrial interactions.
    Same preference as OPA3, MECR, DCAF17 — mito disease patients benefit from non-hepatic AEDs.
  VPA (Valproate): MODERATE CAUTION (NOT absolute CI)
    Unlike MECR (absolute CI due to lipoic acid pathway), DNAJC19 does not disrupt lipoic acid.
    However: mito dysfunction + VPA → risk of hyperammonemia; monitor NH3 + LFTs + 3-MGA.
    Use only if seizures are VPA-responsive and LEV fails; informed consent required.
  PHT/CBZ: AVOID IF POSSIBLE
    CYP450 induction may affect cardiac drug levels (warfarin, amiodarone, beta-blockers).
    Cardiac medication interactions primary concern in DCMA — not the same as OPA3 reason.
  Tetrabenazine/Deutetrabenazine: NOT indicated (no chorea — DDx from OPA3 where chorea is 85-90%)
  Baclofen: NOT indicated (no spastic paraplegia — DDx from OPA3 where spasticity is 50-60%)
  Genetic counseling: AR; sibling risk 25%; carrier testing founder c.130-1G>C in at-risk populations.
  Cardiac monitoring: Echo + ECG every 6-12 months; Holter if palpitations; refer to pediatric cardiology.
"""

import random
import math
from datetime import date, timedelta

SEED = 537
RNG = random.Random(SEED)

# ── helpers ──────────────────────────────────────────────────────────────────
def _date(n_days_ago: int) -> str:
    return (date.today() - timedelta(days=n_days_ago)).isoformat()

def _rng_ages(n: int, mu: float, sigma: float, lo: float, hi: float) -> list[float]:
    out = []
    for _ in range(n):
        v = RNG.gauss(mu, sigma)
        out.append(round(max(lo, min(hi, v)), 1))
    return out

# ── overview ──────────────────────────────────────────────────────────────────
def get_overview() -> dict:
    """DCMA Syndrome (DNAJC19) — overview payload for /api/dnajc19/overview."""
    n = 40
    # DCM onset ages (birth-2yr, median 8 months ≈ 0.67 yr)
    dcm_onset = _rng_ages(n, mu=0.67, sigma=0.3, lo=0.0, hi=2.0)
    # Ataxia onset (12-18 months ≈ 1.25 yr)
    atx_onset = _rng_ages(n, mu=1.25, sigma=0.35, lo=0.5, hi=3.0)
    # EF at diagnosis
    ef_vals = [round(max(15, min(45, RNG.gauss(28, 8))), 1) for _ in range(n)]
    # 3-MGA levels (mmol/mol creatinine, 30-150 range)
    mga_vals = [round(max(30, min(150, RNG.gauss(75, 25))), 1) for _ in range(n)]

    male_n = 22  # ~55% male
    cryptorchidism_n = round(male_n * 0.76)
    hypospadias_n = round(male_n * 0.23)

    return {
        "generated": date.today().isoformat(),
        "cohort": n,
        "seed": SEED,
        "disease": "DCMA Syndrome (Dilated Cardiomyopathy with Ataxia)",
        "gene": "DNAJC19; Tim14; DnaJ Heat Shock Protein Family Member C19",
        "protein": "DNAJC19 — 116 aa, TIM23 translocase co-chaperone (J-domain/Hsp40 family), single-pass IMS anchor",
        "chromosome": "3q26.33",
        "omim_gene": "608977",
        "omim_disease": "610198",
        "inheritance": "Autosomal Recessive; biallelic LOF; AR",
        "prevalence": "~30-40 families worldwide (2026); primarily Hutterite & Old Colony Mennonite",
        "first_described": "Davey et al. 2006 (Am J Hum Genet) — Hutterite/Mennonite pedigrees; 3-MGA + DCM + ataxia",
        "founder_mutation": "c.130-1G>C (splice acceptor, intron 3, exon 4 skip → null) — ~88% of alleles",
        "category": "NBIA-adjacent / 3-MGA-uria / Mitochondrial-TIM23",
        "kpis": {
            "n_patients": n,
            "dcm_pct": 100,
            "cerebellar_ataxia_pct": 95,
            "mga_pct": 100,
            "male_genital_anomalies_pct": round(cryptorchidism_n / male_n * 100),
            "mild_id_pct": 50,
            "short_stature_pct": 65,
            "optic_atrophy_pct": 0,
            "seizure_pct": 13,
            "transplant_pct": 30,
            "mean_ef_at_dx": round(sum(ef_vals) / n, 1),
            "mean_mga_mmol": round(sum(mga_vals) / n, 1),
            "mean_dcm_onset_yr": round(sum(dcm_onset) / n, 2),
            "mean_ataxia_onset_yr": round(sum(atx_onset) / n, 2),
        },
        "phenotype_summary": {
            "dcm_100pct": "Dilated cardiomyopathy — 100%; CARDINAL feature; systolic EF 20-40% at dx; LV dilation; onset birth-2yr",
            "ataxia_95pct": "Non-progressive cerebellar ataxia — 95%; gait ataxia, intention tremor; onset 12-18 months",
            "mga_100pct": "3-Methylglutaconic aciduria — 100%; Type III classification (same as OPA3/Costeff); 30-150 mmol/mol Cr",
            "genital_anomalies": f"Male genital anomalies — ~75% of males; cryptorchidism {cryptorchidism_n}/{male_n} males; hypospadias {hypospadias_n}/{male_n}",
            "mild_id": "Mild intellectual disability — 40-60%; IQ 60-80; language preserved; non-progressive",
            "short_stature": "Short stature — 60-70%; below 3rd percentile; proportionate; no endocrine cause",
            "key_negatives": "NO optic atrophy (DDx OPA3/MECR); NO chorea (DDx OPA3); NO dystonia (DDx MECR); NO hearing loss (DDx DCAF17); NO GP iron on MRI (DDx NBIA)",
        },
        "clinical_highlights": [
            "DCM is the DOMINANT feature — cardiomyopathy-first, ataxia-second (opposite to most ataxias)",
            "NON-PROGRESSIVE ataxia distinguishes DCMA from Friedreich ataxia, SCA, FRDA",
            "3-MGA Type III (same class as OPA3/Costeff) — metabolic link despite different gene/mechanism",
            "Male genital anomalies (cryptorchidism) in ~75% of males — unique among 3-MGA-uria diseases",
            "NO optic atrophy: key negative DDx from OPA3 (100%) and MECR (80-90%)",
            "NO chorea: key negative DDx from OPA3 (chorea 85-90%)",
            "Founder c.130-1G>C in intron 3 accounts for ~88% of alleles — targeted sequencing efficient",
            "~30% require heart transplant; ataxia/ID persist post-transplant (systemic mito disease)",
            "Baclofen NOT indicated (no spastic paraplegia); Tetrabenazine NOT indicated (no chorea)",
            "LEV preferred if rare seizures; VPA moderate caution (not absolute CI unlike MECR)",
        ],
        "contraindications": [
            {"drug": "Tetrabenazine/Deutetrabenazine", "reason": "NOT indicated — no chorea in DCMA (OPA3 has chorea; DNAJC19 does not)"},
            {"drug": "Baclofen", "reason": "NOT indicated — no spastic paraplegia in DCMA (OPA3 has spasticity 50-60%)"},
            {"drug": "VPA", "reason": "MODERATE CAUTION — mito dysfunction risk hyperammonemia; NOT absolute CI (lipoic acid pathway intact)"},
            {"drug": "PHT/CBZ", "reason": "AVOID IF POSSIBLE — CYP450 induction affects cardiac drug levels (beta-blockers, amiodarone)"},
            {"drug": "Strong CYP3A4/2D6 inducers", "reason": "Risk of sub-therapeutic cardiac medication levels — review all drug interactions"},
        ],
        "thresholds": [
            {"parameter": "3-MGA urine", "threshold": "> 20 mmol/mol Cr", "action": "Diagnostic of 3-MGA-uria; confirm by GC-MS urine OA"},
            {"parameter": "EF < 40%", "threshold": "Systolic dysfunction threshold", "action": "Start ACE inhibitor + beta-blocker"},
            {"parameter": "EF < 25%", "threshold": "Advanced heart failure", "action": "Transplant evaluation; LVAD consideration"},
            {"parameter": "Carnitine C0 < 25 µmol/L", "threshold": "Secondary depletion", "action": "L-carnitine supplement 50-100 mg/kg/day"},
            {"parameter": "NH3 > 80 µmol/L", "threshold": "Hyperammonemia on VPA", "action": "Stop VPA; switch to LEV; lactulose if severe"},
        ],
        "gene_biology": {
            "protein_length": 116,
            "domains": [
                {"domain": "MTS", "residues": "aa 1-28", "function": "Mitochondrial targeting; cleaved upon import"},
                {"domain": "TM anchor", "residues": "aa 29-51", "function": "Single-pass IMS anchor; type II orientation"},
                {"domain": "Linker", "residues": "aa 52-59", "function": "Flexible linker; TM-J-domain connection"},
                {"domain": "J-domain (Hsp40)", "residues": "aa 60-116", "function": "HPD tripeptide (His95-Pro96-Asp97); activates HSPA9/mortalin ATPase; preprotein release"},
            ],
            "complex": "TIM23 translocase (inner mitochondrial membrane)",
            "partner": "HSPA9/mortalin (mtHsp70); TIM44; TIM17A/B; TIM23",
            "pathway": "Nuclear-encoded mitochondrial protein import (preprotein translocation, matrix targeting)",
            "lof_consequence": "TIM23 destabilization → OXPHOS complex I/IV import failure → ATP deficit → DCM + neuronal dysfunction",
        },
        "ddx_table": [
            {"feature": "3-MGA elevated", "dnajc19_dcma": "✅ 100%", "opa3_costeff": "✅ 100%", "mecr_mepan": "✅ 100%", "barth_taz": "✅ 100%"},
            {"feature": "Dilated Cardiomyopathy", "dnajc19_dcma": "✅ 100%", "opa3_costeff": "❌ Absent", "mecr_mepan": "❌ Absent", "barth_taz": "✅ 100%"},
            {"feature": "Optic Atrophy", "dnajc19_dcma": "❌ Absent", "opa3_costeff": "✅ 100%", "mecr_mepan": "✅ 80-90%", "barth_taz": "❌ Absent"},
            {"feature": "Chorea", "dnajc19_dcma": "❌ Absent", "opa3_costeff": "✅ 85-90%", "mecr_mepan": "❌ Absent (dystonia)", "barth_taz": "❌ Absent"},
            {"feature": "Cerebellar Ataxia", "dnajc19_dcma": "✅ 95% (nonprog)", "opa3_costeff": "❌ Rare", "mecr_mepan": "✅ 60-70%", "barth_taz": "❌ Absent"},
            {"feature": "GP Iron on MRI", "dnajc19_dcma": "❌ Absent", "opa3_costeff": "❌ Absent", "mecr_mepan": "✅ Bilateral GP", "barth_taz": "❌ Absent"},
            {"feature": "Neutropenia", "dnajc19_dcma": "❌ Absent", "opa3_costeff": "❌ Absent", "mecr_mepan": "❌ Absent", "barth_taz": "✅ 95%"},
            {"feature": "Male genital anomalies", "dnajc19_dcma": "✅ 75% males", "opa3_costeff": "❌ Absent", "mecr_mepan": "❌ Absent", "barth_taz": "❌ Absent"},
            {"feature": "Hearing loss", "dnajc19_dcma": "❌ Absent", "opa3_costeff": "❌ Absent", "mecr_mepan": "❌ Absent", "barth_taz": "❌ Absent"},
            {"feature": "Founder mutation", "dnajc19_dcma": "Hutterite c.130-1G>C", "opa3_costeff": "Iraqi-Jewish p.Gln105*", "mecr_mepan": "Bedouin p.Tyr200His", "barth_taz": "X-linked; various"},
        ],
    }


# ── breakdown ─────────────────────────────────────────────────────────────────
def get_breakdown() -> dict:
    """DCMA Syndrome (DNAJC19) — breakdown payload for /api/dnajc19/breakdown."""
    n = 40
    RNG2 = random.Random(SEED + 1)

    # Phenotype distribution
    phenotype_groups = [
        ("Classic DCMA (DCM+ataxia+3-MGA+genital)", 20),
        ("DCM+ataxia only (no genital anomalies, female/mild)", 10),
        ("DCM+ataxia+ID (more severe cogn)", 6),
        ("DCM+ataxia+seizures (rare phenotype)", 4),
    ]

    # Variant distribution
    variant_dist = [
        {"variant": "c.130-1G>C (splice, intron 3 — Hutterite founder)", "n": 35, "pct": 88, "effect": "Exon 4 skip → frameshift → null; complete J-domain loss"},
        {"variant": "c.33_34delAT (frameshift, exon 1)", "n": 2, "pct": 5, "effect": "European non-founder; premature stop codon; null"},
        {"variant": "p.His95Arg (J-domain HPD motif missense)", "n": 2, "pct": 4, "effect": "HPD → RPD; HSPA9 interaction abolished; complete LOF"},
        {"variant": "p.Leu41Pro (TM domain missense)", "n": 1, "pct": 2, "effect": "Disrupts IMS anchor; protein mislocalised; LOF"},
        {"variant": "Large deletion exon 3-5", "n": 1, "pct": 1, "effect": "Complete null; European; rare"},
    ]

    # Treatment distribution
    treatment_dist = [
        {"treatment": "ACE Inhibitor (captopril/enalapril)", "n": 40, "pct": 100, "indication": "DCM — Level A"},
        {"treatment": "Beta-Blocker (carvedilol/bisoprolol)", "n": 37, "pct": 93, "indication": "DCM — Level A"},
        {"treatment": "Diuretics (furosemide/spironolactone)", "n": 30, "pct": 75, "indication": "DCM fluid overload — Level A"},
        {"treatment": "L-Carnitine", "n": 28, "pct": 70, "indication": "Secondary carnitine depletion — Level B"},
        {"treatment": "Heart Transplant (completed)", "n": 12, "pct": 30, "indication": "Refractory DCM EF<25% — Level B"},
        {"treatment": "LEV (for rare seizures)", "n": 5, "pct": 13, "indication": "Seizures (rare) — Level B"},
        {"treatment": "Orchidopexy (males, cryptorchidism)", "n": 14, "pct": 64, "indication": "Male genital anomaly — Level A surgical"},
        {"treatment": "Special education / developmental support", "n": 22, "pct": 55, "indication": "Mild ID support — Level A"},
    ]

    # 3-MGA level by phenotype
    mga_by_pheno = [
        {"phenotype": "Classic DCMA", "mean_mga": 82, "range": "45-148", "n": 20},
        {"phenotype": "DCM+ataxia (no genital)", "mean_mga": 71, "range": "32-130", "n": 10},
        {"phenotype": "DCM+ataxia+severe ID", "mean_mga": 95, "range": "60-148", "n": 6},
        {"phenotype": "DCM+ataxia+seizures", "mean_mga": 88, "range": "50-145", "n": 4},
    ]

    # EF by age group
    ef_by_age = [
        {"age_group": "< 1 yr (n=15)", "mean_ef": 24, "range": "18-35", "transplant_n": 3},
        {"age_group": "1-5 yr (n=16)", "mean_ef": 28, "range": "20-40", "transplant_n": 7},
        {"age_group": "5-18 yr (n=9)", "mean_ef": 33, "range": "22-42", "transplant_n": 2},
    ]

    return {
        "generated": date.today().isoformat(),
        "cohort": n,
        "seed": SEED,
        "phenotype_groups": [{"group": g, "n": c, "pct": round(c / n * 100)} for g, c in phenotype_groups],
        "variant_distribution": variant_dist,
        "treatment_distribution": treatment_dist,
        "mga_by_phenotype": mga_by_pheno,
        "ef_by_age_group": ef_by_age,
        "cardiac_outcomes": {
            "transplant_rate_pct": 30,
            "cardiac_death_no_transplant_pct": 15,
            "stable_medical_mgmt_pct": 55,
            "lv_dilation_pct": 100,
            "lbbb_ecg_pct": 58,
            "conduction_defect_pct": 60,
        },
        "neurological_outcomes": {
            "cerebellar_ataxia_pct": 95,
            "nonprogressive_pct": 92,
            "mild_id_pct": 50,
            "independent_ambulation_pct": 80,
            "seizure_pct": 13,
            "seizure_controlled_pct": 100,
        },
        "biomarker_summary": {
            "mga_range_mmol_cr": "30-150",
            "mga_mean": 82,
            "ef_range_pct": "15-45",
            "ef_mean": 28,
            "c0_carnitine_low_pct": 70,
            "lactate_mild_elevation_pct": 35,
        },
        "sex_specific": {
            "male_n": 22,
            "female_n": 18,
            "cryptorchidism_n": 17,
            "cryptorchidism_pct_males": 77,
            "hypospadias_n": 5,
            "hypospadias_pct_males": 23,
        },
    }


# ── definitions ───────────────────────────────────────────────────────────────
def get_definitions() -> dict:
    """DCMA Syndrome (DNAJC19) — definitions for /api/dnajc19/definitions."""
    return {
        "generated": date.today().isoformat(),
        "disease": "DCMA Syndrome (Dilated Cardiomyopathy with Ataxia)",
        "gene": "DNAJC19",
        "omim_gene": "608977",
        "omim_disease": "610198",
        "definitions": [
            {
                "term": "DNAJC19 / Tim14",
                "definition": "DnaJ/Hsp40 co-chaperone of the TIM23 mitochondrial inner membrane translocase; 116 amino acids; single-pass type II IMS anchor; J-domain with HPD tripeptide activates mtHsp70/HSPA9 ATPase to pull preproteins into matrix.",
                "relevance": "LOF → TIM23 destabilisation → OXPHOS complex I/IV import failure → ATP deficit → DCM + cerebellar neuronal dysfunction + 3-MGA overflow",
            },
            {
                "term": "TIM23 Translocase",
                "definition": "Mitochondrial inner membrane protein complex (Tim23, Tim17A/B, Tim44, Tim14/DNAJC19, mtHsp70); translocates nuclear-encoded preproteins with N-terminal matrix targeting sequences from IMS into the mitochondrial matrix.",
                "relevance": "DNAJC19 LOF disables the J-domain co-chaperone arm → mtHsp70 cannot pull preproteins → OXPHOS subunits stranded in IMS → complex assembly failure.",
            },
            {
                "term": "3-Methylglutaconic Aciduria Type III (3-MGA-uria III)",
                "definition": "Elevated urinary 3-methylglutaconic acid (>20 mmol/mol Cr) due to mitochondrial OXPHOS dysfunction causing HMG-CoA pathway overflow → 3-methylglutaconyl-CoA accumulates → excreted as 3-MGA. Type III = Costeff (OPA3) and DCMA (DNAJC19) share this classification.",
                "relevance": "Shared metabolic fingerprint between DNAJC19 and OPA3; different gene/mechanism but same overflow pathway. Barth (TAZ) is also 3-MGA-uria II but different acylcarnitine profile (C4-DC elevated).",
            },
            {
                "term": "Hutterite/Mennonite Founder Mutation (c.130-1G>C)",
                "definition": "Splice acceptor site mutation in intron 3 of DNAJC19; causes exon 4 skipping → frameshift → premature stop within J-domain → null allele. Found in ~88% of DCMA alleles; Hutterite/Old Colony Mennonite population isolate.",
                "relevance": "Enables targeted sequencing in at-risk populations; if both alleles are c.130-1G>C and clinical triad present (DCM+ataxia+3-MGA), diagnosis is confirmed without WES.",
            },
            {
                "term": "Non-progressive Cerebellar Ataxia",
                "definition": "Cerebellar ataxia that is static or very slowly worsening over decades; contrasts with progressive ataxias (FRDA, SCA, ARCA). DCMA cerebellar ataxia is non-progressive in ~92% of patients.",
                "relevance": "Key prognostic and DDx feature: non-progression distinguishes DCMA from degenerative ataxias; life expectancy and ambulation limited by cardiac not neurological course.",
            },
            {
                "term": "DCM in DNAJC19 vs Barth Syndrome (TAZ)",
                "definition": "Both DNAJC19-DCMA and Barth syndrome (TAZ/tafazzin) present with DCM + 3-MGA-uria. Key DDx: Barth = X-linked, neutropenia (95%), skeletal myopathy, C4-DC acylcarnitine elevated; DNAJC19 = AR, cerebellar ataxia (95%), male genital anomalies, normal acylcarnitine profile, no neutropenia.",
                "relevance": "Acylcarnitine profile and neutrophil count are the fastest DDx tests; WES confirms.",
            },
            {
                "term": "Male Genital Anomalies in DCMA",
                "definition": "Cryptorchidism (~77% of males) and hypospadias (~23% of males) in DNAJC19-DCMA; absent in all other 3-MGA-uria diseases. Mechanism unclear: DNAJC19 may be required in Leydig/Sertoli cell mitochondrial function for testicular descent.",
                "relevance": "Presence of cryptorchidism in a male infant with DCM + 3-MGA-uria is PATHOGNOMONIC for DNAJC19; guides genetic testing priority.",
            },
            {
                "term": "ACE Inhibitor + Beta-Blocker First-Line",
                "definition": "Standard HF pharmacotherapy applied to DNAJC19-DCM: ACE inhibitors (captopril/enalapril) reduce afterload + prevent LV remodeling; beta-blockers (carvedilol) reverse remodeling + reduce mortality. Both Level A in standard DCM guidelines.",
                "relevance": "DCMA-DCM responds to standard HF therapy; start ACE inhibitor at diagnosis regardless of symptoms; add beta-blocker after stabilisation. Monitor: hypotension, renal function, electrolytes.",
            },
            {
                "term": "VPA Moderate Caution (NOT Absolute CI)",
                "definition": "Unlike MECR/MEPAN where VPA is ABSOLUTE CI (CoA sequestration + PDH/lipoic acid collapse), in DNAJC19 the lipoic acid pathway is intact → VPA CI mechanism absent. MODERATE CAUTION: OXPHOS dysfunction may impair urea cycle → hyperammonemia risk with VPA; monitor NH3 + LFTs.",
                "relevance": "Critical prescribing distinction: MECR = absolute VPA CI; DNAJC19 = moderate caution (same as OPA3). Use LEV first; if VPA needed, monitor closely.",
            },
            {
                "term": "Heart Transplant in DNAJC19-DCMA",
                "definition": "~30% of DCMA patients require heart transplant for refractory DCM (EF<25%); transplant outcomes are generally good. Important caveat: ataxia, mild ID, and male genital anomalies are systemic mitochondrial manifestations — they DO NOT improve post-transplant.",
                "relevance": "Transplant counselling must address that cardiac cure ≠ neurological cure; multidisciplinary team (neurology + cardiology + genetics) required for transplant decision.",
            },
        ],
    }
