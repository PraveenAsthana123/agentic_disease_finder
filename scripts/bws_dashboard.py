#!/usr/bin/env python3
"""Beckwith-Wiedemann Syndrome (BWS) Dashboard.

Beckwith-Wiedemann Syndrome is the OVERGROWTH opposite of Silver-Russell Syndrome (SRS)
at the SAME 11p15.5 locus — the canonical same-locus, opposite-phenotype imprinting pair.
  Principal genes: IGF2 (Insulin-Like Growth Factor 2, paternally expressed, growth driver)
                   H19 (lncRNA, maternally expressed, ICR1 methylation anchor — suppressed in BWS)
                   CDKN1C (p57KIP2, maternally expressed, growth inhibitor — LOST in BWS)
                   KCNQ1OT1 (lncRNA, ICR2 control — hypomethylated in BWS)
  Mechanism: MATERNAL LOF at 11p15.5 → EXCESS paternal IGF2 (biallelic) + loss of CDKN1C → OVERGROWTH
  Most common cause: ICR2 (KCNQ1OT1/LIT1) hypomethylation on maternal allele (~50%) → CDKN1C silenced
  OMIM Disease: #130650 · Genes: IGF2 *147470 · H19 *103280 · CDKN1C *600856
  Prevalence: ~1:10,500–1:13,700

IMPRINTING MECHANISM — WHY MATERNAL LOF CAUSES BWS:
  ICR1 (H19/IGF2 imprinting centre 1, 11p15.5):
    — Normal: ICR1 METHYLATED on paternal allele → IGF2 expressed paternally only
    — In BWS ICR1 hypermethylation (maternal allele also methylated, ~5%): IGF2 expressed BIALLELICALLY
    — Result: 2× IGF2 → maximal overgrowth; highest Wilms tumor risk (~29%)
  ICR2 (KCNQ1OT1/LIT1 imprinting centre 2, 11p15.5):
    — Normal: ICR2 UNMETHYLATED on paternal allele → KCNQ1OT1 expressed (silences CDKN1C paternally)
    — Normal: ICR2 METHYLATED on maternal allele → maternal CDKN1C active (growth brake)
    — In BWS ICR2 hypomethylation (~50%): maternal ICR2 unmethylated → KCNQ1OT1 expressed BIALLELICALLY
    — Consequence: CDKN1C silenced on BOTH alleles → NO growth brake → overgrowth
  Paternal UPD11p15 (upd(11p15)pat, ~20%):
    — Two paternal copies of 11p15 → 2× paternal IGF2 + 0× maternal CDKN1C → dual overgrowth signal
    — HIGHEST hyperinsulinism risk (focal: 60%); Wilms tumor risk ~12%
  CDKN1C LOF mutation (maternal allele, ~5%):
    — Loss of the cell-cycle brake (p57KIP2 cyclin-dependent kinase inhibitor 1C)
    — Autosomal dominant; 50% recurrence from carrier mother; familial BWS ~25% CDKN1C
  ICR1 hypermethylation (maternal allele, ~5%):
    — Both alleles behave as paternal → IGF2 expressed biallelically → highest growth excess
    — Wilms tumor risk ~29% (highest of all mechanisms)

§57.7 — All cohort data are synthetic (seeded RNG, no real patients).
"""
import random
from datetime import datetime

SEED = 297
rng = random.Random(SEED)

# ── Genetic mechanisms ──────────────────────────────────────────────────────
MECHANISMS = [
    {
        "id": "icr2_hypo",
        "label": "ICR2 (KCNQ1OT1) hypomethylation — maternal allele",
        "pct": 50, "n": 20,
        "igf2_status": "Normal-to-mildly-elevated (CDKN1C silenced; IGF2 single-allele)",
        "cdkn1c_status": "SILENCED (both alleles — maternal ICR2 unmethylated)",
        "wilms_risk_pct": 2,
        "hepatoblastoma_risk_pct": 1,
        "hyperinsulinism_rate": 0.30,
        "hemihypertrophy_rate": 0.30,
        "omphalocele_rate": 0.25,
        "macroglossia_rate": 0.95,
        "recurrence_pct": "<1 (de novo; ART 10× risk)",
        "notes": "Most common mechanism (50%); isolated ICR2 phenotype; lower Wilms vs ICR1; CDKN1C normal sequencing",
    },
    {
        "id": "upd11p15pat",
        "label": "Paternal UPD11p15 (upd(11p15)pat)",
        "pct": 20, "n": 8,
        "igf2_status": "BIALLELIC OVEREXPRESSION (two paternal copies)",
        "cdkn1c_status": "ABSENT (no maternal copy)",
        "wilms_risk_pct": 12,
        "hepatoblastoma_risk_pct": 2,
        "hyperinsulinism_rate": 0.60,
        "hemihypertrophy_rate": 0.60,
        "omphalocele_rate": 0.10,
        "macroglossia_rate": 0.90,
        "recurrence_pct": "<1 (somatic mosaicism)",
        "notes": "Somatic mosaic — severity varies with % affected cells; focal hyperinsulinism common; highest hemihypertrophy; SNP array required",
    },
    {
        "id": "cdkn1c_lof",
        "label": "CDKN1C LOF mutation — maternal allele",
        "pct": 10, "n": 4,
        "igf2_status": "Normal (ICR1 intact)",
        "cdkn1c_status": "LOSS-OF-FUNCTION (pathogenic variant, maternal)",
        "wilms_risk_pct": 2,
        "hepatoblastoma_risk_pct": 1,
        "hyperinsulinism_rate": 0.15,
        "hemihypertrophy_rate": 0.20,
        "omphalocele_rate": 0.40,
        "macroglossia_rate": 0.80,
        "recurrence_pct": "50 (AD from carrier mother)",
        "notes": "Familial BWS ~25% CDKN1C; exomphalos common; AD inheritance from carrier mother; confirm maternal origin",
    },
    {
        "id": "icr1_hypermeth",
        "label": "ICR1 (H19/IGF2) hypermethylation — maternal allele",
        "pct": 5, "n": 2,
        "igf2_status": "BIALLELIC OVEREXPRESSION (maternal ICR1 also methylated)",
        "cdkn1c_status": "Normal (ICR2 intact)",
        "wilms_risk_pct": 29,
        "hepatoblastoma_risk_pct": 2,
        "hyperinsulinism_rate": 0.20,
        "hemihypertrophy_rate": 0.55,
        "omphalocele_rate": 0.10,
        "macroglossia_rate": 0.90,
        "recurrence_pct": "<1 (de novo)",
        "notes": "HIGHEST Wilms tumor risk (~29%); maximal overgrowth; biallelic IGF2 — diagnostic by methylation (both alleles behave paternally at ICR1)",
    },
    {
        "id": "other_unknown",
        "label": "Duplication / copy number / unknown mechanism",
        "pct": 15, "n": 6,
        "igf2_status": "Variable",
        "cdkn1c_status": "Variable",
        "wilms_risk_pct": 4,
        "hepatoblastoma_risk_pct": 1,
        "hyperinsulinism_rate": 0.20,
        "hemihypertrophy_rate": 0.25,
        "omphalocele_rate": 0.15,
        "macroglossia_rate": 0.85,
        "recurrence_pct": "Variable",
        "notes": "11p15.5 duplications (paternal), mosaic variants, clinically BWS with normal methylation",
    },
]

# ── Phenotype groups ─────────────────────────────────────────────────────────
PHENOTYPE_GROUPS = [
    {"label": "Classic BWS", "n": 20, "description": "Macrosomia + macroglossia + omphalocele or organomegaly"},
    {"label": "Mild BWS", "n": 12, "description": "Macroglossia ± mild overgrowth; no omphalocele; may lack hemihypertrophy"},
    {"label": "Hemihypertrophy-Prominent", "n": 8, "description": "Isolated hemihypertrophy + macroglossia; UPD or ICR1 mechanism"},
]

# ── Variant registry ─────────────────────────────────────────────────────────
VARIANTS = [
    {"type": "ICR2 methylation loss", "mechanism": "Epimutation (maternal ICR2 unmethylated)", "frequency": "50%", "detectable_by": "Methylation analysis (MS-MLPA / bisulfite pyrosequencing)"},
    {"type": "Paternal UPD11p15", "mechanism": "Somatic mosaicism — two paternal 11p15", "frequency": "20%", "detectable_by": "SNP array (LOH 11p15.5) + methylation"},
    {"type": "CDKN1C pathogenic variant", "mechanism": "LOF (frameshift, nonsense, splice); maternal", "frequency": "5-10% (25% familial)", "detectable_by": "CDKN1C NGS sequencing (confirm maternal)"},
    {"type": "ICR1 hypermethylation", "mechanism": "Maternal ICR1 also methylated → biallelic IGF2", "frequency": "5%", "detectable_by": "Methylation (ICR1 methylation >85% = hypermethylation)"},
    {"type": "11p15.5 paternal duplication", "mechanism": "Extra paternal 11p15 region (de novo or inherited)", "frequency": "~2%", "detectable_by": "Chromosomal microarray / FISH"},
]

# ── AED guide (if seizures — mostly hypoglycemic neonatal) ───────────────────
AEDS = [
    {"drug": "Phenobarbital", "class": "Barbiturate", "bws_evidence": "Level C — acceptable for neonatal seizures if source corrected; NOT first-line",
     "key_alert": "TREAT UNDERLYING HYPERINSULINISM FIRST — seizures usually reflect hypoglycemia, not primary epilepsy",
     "ci_in_bws": "No absolute CI (unlike SRS where feeding difficulties make it problematic); use if truly needed"},
    {"drug": "Levetiracetam (LEV)", "class": "SV2A modulator", "bws_evidence": "Level B — acceptable; weight-neutral; no metabolic risk",
     "key_alert": "Safe in BWS; does not worsen hyperinsulinism or tumor risk"},
    {"drug": "Lamotrigine (LTG)", "class": "Na-channel blocker", "bws_evidence": "Level B — acceptable; no significant interaction",
     "key_alert": "Titrate slowly; acceptable safety profile in BWS"},
    {"drug": "Valproate (VPA)", "class": "Broad-spectrum", "bws_evidence": "Level C — MODERATE CAUTION in BWS",
     "key_alert": "VPA hepatotoxicity risk overlaps with hepatoblastoma surveillance (AFP elevation confusion); monitor AFP + LFTs closely; NOT absolute CI but prefer alternative AEDs in high-risk patients"},
    {"drug": "Diazoxide", "class": "KATP channel opener (NOT AED)", "bws_evidence": "Level A — FIRST-LINE for BWS hyperinsulinism",
     "key_alert": "NOT an antiepileptic — treats the CAUSE of hypoglycemic seizures; must use before assuming primary epilepsy"},
    {"drug": "Octreotide", "class": "Somatostatin analog (NOT AED)", "bws_evidence": "Level B — second-line for diazoxide-resistant hyperinsulinism",
     "key_alert": "Inhibits insulin secretion; used for diffuse hyperinsulinism unresponsive to diazoxide"},
]

# ── Management protocols ──────────────────────────────────────────────────────
MANAGEMENTS = [
    {"domain": "Neonatal Hypoglycemia", "level": "DANGER", "protocol": "HYPERINSULINISM mechanism — diazoxide 5-15 mg/kg/day FIRST LINE (NOT cornstarch/fasting restriction as in SRS)", "monitoring": "Continuous glucose monitoring; blood glucose q1-2h neonatal period"},
    {"domain": "Tumor Surveillance", "level": "MANDATORY", "protocol": "Abdominal ultrasound q3months until age 8 + serum AFP q3months until age 4 (hepatoblastoma). CONTINUE ultrasound q6months age 4-8 (Wilms)", "monitoring": "AFP rise >2× baseline → urgent imaging; new flank mass → immediate workup"},
    {"domain": "Macroglossia", "level": "MANAGEMENT", "protocol": "Speech therapy + feeding support (NG tube if severe); consider tongue-reduction surgery if airway compromise, severe drooling, dental/occlusal issues, speech impairment", "monitoring": "ENT + oral surgery referral; growth trajectory"},
    {"domain": "Omphalocele", "level": "SURGICAL", "protocol": "Staged surgical repair; preoperative respiratory assessment; NEC prevention in premature", "monitoring": "Post-repair hernia check; pulmonary function if hepatic herniation"},
    {"domain": "Hemihypertrophy", "level": "MONITOR", "protocol": "Annual leg-length discrepancy measurement; shoe lift if >1 cm; surgical equalization if >2 cm. WILMS RISK HIGHEST in UPD/ICR1 subtypes with hemihypertrophy", "monitoring": "Orthopedic annual; reinforce tumor surveillance (hemihypertrophy = higher risk)"},
    {"domain": "GH Therapy", "level": "CONTRAINDICATED", "protocol": "GH IS CONTRAINDICATED IN BWS — patients already overgrown; GH may increase tumor risk (IGF-1↑) in already IGF2-excess state", "monitoring": "N/A — do NOT prescribe GH for short-stature concerns in BWS without specialist review"},
    {"domain": "Ketogenic Diet", "level": "CAUTION", "protocol": "KD not routinely used in BWS; if needed for refractory epilepsy, proceed with extreme monitoring given underlying hypoglycemia predisposition in neonatal period (most children outgrow by age 2)", "monitoring": "Continuous glucose; avoid in those with active hyperinsulinism"},
    {"domain": "ART / Reproductive", "level": "COUNSEL", "protocol": "IVF/ICSI increases ICR2 hypomethylation risk ~6× above background. BWS parents considering ART: genetic counselling mandatory", "monitoring": "Prenatal methylation testing offered in at-risk families"},
]

# ── Differential diagnoses ────────────────────────────────────────────────────
DIFFERENTIALS = [
    {"disease": "Silver-Russell Syndrome (SRS)", "locus": "11p15.5 (SAME locus)", "key_contrast": "SRS: Paternal LOF → NO IGF2 → GROWTH RESTRICTION; BWS: Maternal LOF → 2× IGF2 → OVERGROWTH. Same genes, opposite parents, opposite phenotypes"},
    {"disease": "Simpson-Golabi-Behmel Syndrome", "locus": "Xq26 (GPC3)", "key_contrast": "X-linked; macrosomia + macroglossia + organomegaly similar to BWS; extra nipples; coarse facies; GPC3 negative methylation"},
    {"disease": "Perlman Syndrome", "locus": "2q37 (DIS3L2)", "key_contrast": "Renal hamartomas/nephroblastomatosis; facial dysmorphia; AR inheritance; high neonatal mortality; DIS3L2 sequencing"},
    {"disease": "Sotos Syndrome", "locus": "5q35 (NSD1)", "key_contrast": "Cerebral overgrowth + tall stature + macrocephaly; NSD1 LOF; DIFFERENT locus; no hyperinsulinism; no Wilms tumor"},
    {"disease": "Costello Syndrome", "locus": "11p15.5 (HRAS)", "key_contrast": "HRAS GOF; short stature (paradoxically); cardiomyopathy; papillomata; rhabdomyosarcoma (not Wilms); distinct facies"},
    {"disease": "Weaver Syndrome", "locus": "7q36 (EZH2)", "key_contrast": "EZH2 GOF; generalized overgrowth; camptodactyly; broad thumbs; no Wilms; epigenetic (PRC2 histone H3K27me3 deficiency)"},
    {"disease": "Congenital hyperinsulinism (CHI)", "locus": "Various (ABCC8, KCNJ11, GCK, HADH)", "key_contrast": "Isolated hyperinsulinism; NO macrosomia, macroglossia, omphalocele, hemihypertrophy; BWS must be excluded first in LGA + hypoglycemia"},
]


# ── Cohort generator ─────────────────────────────────────────────────────────
def _make_cohort():
    pts = []
    pid = 1
    mech_pool = []
    for m in MECHANISMS:
        mech_pool.extend([m] * m["n"])
    rng.shuffle(mech_pool)
    pheno_labels = []
    for pg in PHENOTYPE_GROUPS:
        pheno_labels.extend([pg["label"]] * pg["n"])
    rng.shuffle(pheno_labels)
    for i, (mech, pheno) in enumerate(zip(mech_pool, pheno_labels)):
        age_diag = round(rng.uniform(0.0, 1.5), 1)  # mostly neonatal/infant diagnosis
        birth_weight_sds = round(rng.uniform(1.5, 4.0), 1)   # LGA — opposite of SRS
        birth_length_sds = round(rng.uniform(1.0, 3.0), 1)
        current_height_sds = round(rng.uniform(0.5, 2.5), 1)  # tall
        hyperinsulinism = rng.random() < mech["hyperinsulinism_rate"]
        hemihypertrophy = rng.random() < mech["hemihypertrophy_rate"]
        omphalocele = rng.random() < mech["omphalocele_rate"]
        macroglossia = rng.random() < mech["macroglossia_rate"]
        wilms = rng.random() < (mech["wilms_risk_pct"] / 100)
        hepatoblastoma = rng.random() < (mech["hepatoblastoma_risk_pct"] / 100)
        epilepsy = rng.random() < 0.08  # rare — mostly hypoglycemic seizures, not primary
        neonatal_seizures = hyperinsulinism and rng.random() < 0.35
        pts.append({
            "id": f"BWS-{pid:03d}",
            "mechanism": mech["label"],
            "phenotype_group": pheno,
            "age_diagnosis_y": age_diag,
            "sex": rng.choice(["M", "F"]),
            "birth_weight_sds": birth_weight_sds,
            "birth_length_sds": birth_length_sds,
            "current_height_sds": current_height_sds,
            "macroglossia": macroglossia,
            "omphalocele": omphalocele,
            "hemihypertrophy": hemihypertrophy,
            "neonatal_hyperinsulinism": hyperinsulinism,
            "neonatal_seizures_hypoglycemic": neonatal_seizures,
            "wilms_tumor": wilms,
            "hepatoblastoma": hepatoblastoma,
            "diazoxide_used": hyperinsulinism and rng.random() < 0.90,
            "gh_therapy": False,  # always False — CONTRAINDICATED in BWS
            "tongue_reduction_surgery": macroglossia and rng.random() < 0.35,
            "primary_epilepsy": epilepsy and not neonatal_seizures,
            "aed_if_epilepsy": rng.choice(["LEV", "LTG"]) if (epilepsy and not neonatal_seizures) else None,
            "on_tumor_surveillance": True,  # ALL BWS patients: mandatory
        })
        pid += 1
    return pts


_COHORT = _make_cohort()


def get_overview():
    cohort = _COHORT
    n = len(cohort)
    return {
        "disease": "Beckwith-Wiedemann Syndrome (BWS)",
        "omim": "#130650",
        "genes": ["IGF2 (*147470)", "H19 (*103280)", "CDKN1C (*600856)", "KCNQ1OT1 (ICR2 control)"],
        "locus": "11p15.5 (ICR1: H19/IGF2 + ICR2: KCNQ1OT1/CDKN1C) — SAME LOCUS AS SRS",
        "mechanism": "Maternal LOF at 11p15.5 → Excess/biallelic IGF2 + Loss of CDKN1C → OVERGROWTH",
        "imprinting_class": "Genomic Imprinting — Maternal LOF (11p15.5)",
        "same_locus_opposite": "SRS (Silver-Russell Syndrome) — Paternal LOF same locus → GROWTH RESTRICTION (OPPOSITE phenotype)",
        "prevalence": "~1:10,500–1:13,700",
        "cohort_size": n,
        "seed": SEED,
        "kpis": {
            "total_patients": n,
            "icr2_hypo_pct": 50,
            "upd11p15pat_pct": 20,
            "macroglossia_pct": round(sum(1 for p in cohort if p["macroglossia"]) / n * 100),
            "omphalocele_pct": round(sum(1 for p in cohort if p["omphalocele"]) / n * 100),
            "hemihypertrophy_pct": round(sum(1 for p in cohort if p["hemihypertrophy"]) / n * 100),
            "hyperinsulinism_pct": round(sum(1 for p in cohort if p["neonatal_hyperinsulinism"]) / n * 100),
            "wilms_pct": round(sum(1 for p in cohort if p["wilms_tumor"]) / n * 100),
            "hepatoblastoma_pct": round(sum(1 for p in cohort if p["hepatoblastoma"]) / n * 100),
            "primary_epilepsy_pct": round(sum(1 for p in cohort if p["primary_epilepsy"]) / n * 100),
            "mean_birth_weight_sds": round(sum(p["birth_weight_sds"] for p in cohort) / n, 1),
            "mean_age_diagnosis_y": round(sum(p["age_diagnosis_y"] for p in cohort) / n, 1),
            "tumor_surveillance_pct": 100,
        },
        "mechanism_breakdown": {m["label"]: m["n"] for m in MECHANISMS},
        "phenotype_breakdown": {pg["label"]: pg["n"] for pg in PHENOTYPE_GROUPS},
        "cardinal_features": [
            {"feature": "Macroglossia (enlarged tongue)", "prevalence": "~97%", "notes": "Pathognomonic hallmark; may cause airway obstruction, feeding difficulty, speech delay"},
            {"feature": "Macrosomia / LGA at birth", "prevalence": "~60%", "notes": "Birth weight >90th percentile; mean birth weight SDS +2.5; OPPOSITE of SRS"},
            {"feature": "Abdominal wall defects (omphalocele/hernia)", "prevalence": "~30%", "notes": "Omphalocele more common with CDKN1C LOF; umbilical hernia common in all mechanisms"},
            {"feature": "Organomegaly", "prevalence": "~50%", "notes": "Nephromegaly, hepatomegaly, splenomegaly, cardiomegaly; predisposes to Wilms + hepatoblastoma"},
            {"feature": "Hemihypertrophy (lateralised overgrowth)", "prevalence": "~35-40%", "notes": "ONE side LARGER — OPPOSITE to SRS hemihypotrophy; UPD highest rate; WILMS RISK marker"},
            {"feature": "Neonatal hyperinsulinism", "prevalence": "~30-50%", "notes": "Hypoglycemia from EXCESS insulin (not fasting-type like SRS); diazoxide first-line"},
            {"feature": "Ear creases / posterior helical pits", "prevalence": "~60-75%", "notes": "Posterior linear ear creases; anterior ear pits; minor BWS criterion"},
            {"feature": "Facial nevus flammeus (capillary malformation)", "prevalence": "~60%", "notes": "Glabellar or forehead port-wine stain; fades over years"},
        ],
        "tumor_risks": [
            {"tumor": "Wilms tumor (nephroblastoma)", "overall_bws_risk": "7-7.5%", "highest_subtype": "ICR1 hypermethylation (~29%); UPD (~12%)", "surveillance": "Abdominal US q3mo until age 8; AFP q3mo until age 4"},
            {"tumor": "Hepatoblastoma", "overall_bws_risk": "~1.3%", "highest_subtype": "All mechanisms roughly equal", "surveillance": "AFP q3mo until age 4 (serum)"},
            {"tumor": "Adrenocortical carcinoma", "overall_bws_risk": "~1%", "highest_subtype": "ICR2 mechanism", "surveillance": "Abdominal US (includes adrenals)"},
            {"tumor": "Rhabdomyosarcoma", "overall_bws_risk": "<1%", "highest_subtype": "CDKN1C LOF", "surveillance": "Clinical; imaging if symptomatic"},
            {"tumor": "Neuroblastoma", "overall_bws_risk": "<1%", "highest_subtype": "Variable", "surveillance": "Urine catecholamines annually until age 4"},
        ],
        "key_alerts": [
            {"level": "DANGER", "msg": "TUMOR SURVEILLANCE MANDATORY — Wilms (7%) + hepatoblastoma (1.3%): ultrasound + AFP every 3 months until age 4-8"},
            {"level": "DANGER", "msg": "ICR1 HYPERMETHYLATION: Wilms risk ~29% — highest; intensify surveillance"},
            {"level": "DANGER", "msg": "HYPERINSULINISM: treat with DIAZOXIDE (NOT cornstarch — different mechanism than SRS hypoglycemia)"},
            {"level": "DANGER", "msg": "GH THERAPY CONTRAINDICATED — patients already overgrown; GH may amplify IGF2-excess + tumor risk"},
            {"level": "WARN", "msg": "VPA: MODERATE CAUTION — hepatotoxicity + AFP elevation may confound hepatoblastoma surveillance"},
            {"level": "WARN", "msg": "Hemihypertrophy = HIGHEST Wilms risk marker — reinforce surveillance if asymmetry present"},
            {"level": "INFO", "msg": "Macroglossia surgery: consider if airway compromise, feeding failure, speech delay, or dental occlusion issues"},
            {"level": "INFO", "msg": "Same locus as SRS — 11p15.5; opposite mechanism (maternal LOF vs paternal LOF in SRS)"},
            {"level": "INFO", "msg": "ART (IVF/ICSI) increases ICR2 hypomethylation risk ~6×; genetic counselling before ART in BWS families"},
        ],
        "diagnostic_pathway": [
            {"step": 1, "test": "Clinical: Aziz score / BWS clinical scoring (macroglossia + macrosomia + omphalocele + ear creases)", "threshold": "≥2 major OR ≥1 major +2 minor = test", "detects": "Clinical suspicion"},
            {"step": 2, "test": "MS-MLPA 11p15.5 (ICR1 + ICR2 methylation, copy number)", "yield": "~80%", "detects": "ICR2 hypo (50%), ICR1 hypermeth (5%), copy number changes (7%)"},
            {"step": 3, "test": "SNP array / chromosomal microarray", "yield": "+5-10%", "detects": "Paternal UPD11p15, deletions, duplications"},
            {"step": 4, "test": "CDKN1C sequencing (NGS — exome or targeted)", "yield": "+5-10% (25% familial)", "detects": "CDKN1C pathogenic variants (MUST confirm maternal origin)"},
            {"step": 5, "test": "Clinical diagnosis (clinically BWS + fully negative molecular)", "yield": "~5-10% unsolved", "detects": "Clinical BWS; continue surveillance regardless"},
        ],
        "updated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def get_breakdown():
    cohort = _COHORT
    return {
        "patients": cohort,
        "mechanism_details": MECHANISMS,
        "phenotype_groups": PHENOTYPE_GROUPS,
        "variants": VARIANTS,
        "aed_guide": AEDS,
        "management_protocols": MANAGEMENTS,
        "differentials": DIFFERENTIALS,
        "biomarker_thresholds": {
            "birth_weight_sds_bws": ">+2.0 SDS (often +2.5 to +4.0 SDS)",
            "birth_length_sds_bws": ">+1.5 SDS",
            "blood_glucose_neonatal_danger": "<2.6 mmol/L (47 mg/dL) — critical hyperinsulinism threshold",
            "insulin_glucose_ratio": ">0.3 (µU/mL)/(mg/dL) = pathological in BWS hyperinsulinism",
            "afp_hepatoblastoma_cutoff": ">1000 ng/mL after age 6 months OR rising >2× in serial measurements",
            "afp_normal_neonatal": "Up to 100,000 ng/mL (physiologically elevated at birth; must age-adjust)",
            "afp_surveillance_age_stop": "4 years (then continue ultrasound for Wilms until 8 years)",
            "icr2_methylation_normal": "~50% (maternal allele methylated)",
            "icr2_methylation_bws_threshold": "<35% on maternal allele = pathological hypomethylation",
            "icr1_methylation_bws_hypermeth": ">85% = both alleles behaving paternally",
            "leg_length_discrepancy_threshold": ">0.5 cm (hemihypertrophy marker; shoe lift); >2 cm (orthopaedic referral)",
        },
        "compared_with_srs": {
            "locus": "SAME: 11p15.5",
            "parent_of_origin": "BWS: Maternal LOF | SRS: Paternal LOF",
            "icr_involved": "BWS: ICR2 maternal hypo (50%) OR ICR1 maternal hypermeth (5%) | SRS: ICR1 paternal hypo (45%)",
            "igf2": "BWS: EXCESS (biallelic or near-biallelic expression) | SRS: ABSENT (biallelically silenced)",
            "cdkn1c": "BWS: LOST (silenced or mutated) | SRS: GAIN-OF-FUNCTION (rare mechanism) or intact",
            "growth": "BWS: MACROSOMIA, overgrowth, LGA | SRS: PROFOUND RESTRICTION, SGA, birth weight -3.5 SDS",
            "asymmetry": "BWS: HEMIHYPERTROPHY (one side LARGER) | SRS: HEMIHYPOTROPHY (one side smaller)",
            "hypoglycemia": "BWS: HYPERINSULINISM (excess insulin) → diazoxide | SRS: FASTING (low IGF2) → cornstarch",
            "gh_therapy": "BWS: CONTRAINDICATED | SRS: Level A (FDA 2003) — MANDATORY",
            "tumor_risk": "BWS: Wilms 7-29%, hepatoblastoma 1.3% — MANDATORY surveillance | SRS: minimal tumor risk",
            "macroglossia": "BWS: ~97% (pathognomonic) | SRS: NOT a feature",
            "omphalocele": "BWS: ~30% | SRS: NOT a feature",
            "cognition": "BWS: NORMAL (same as SRS) | SRS: NORMAL",
            "principle": "SAME LOCUS — OPPOSITE PARENT — OPPOSITE PHENOTYPE (canonical imprinting proof)",
        },
        "compared_with_temple": {
            "locus": "BWS: 11p15.5 | Temple: 14q32.3",
            "mechanism": "BWS: maternal LOF 11p15.5 | Temple: paternal LOF 14q32.3",
            "growth": "BWS: OVERGROWTH (macrosomia) | Temple: SGA + short stature",
            "asymmetry": "BWS: hemihypertrophy (larger) | Temple: NO asymmetry",
            "tumor_risk": "BWS: HIGH (Wilms, hepatoblastoma) — surveillance mandatory | Temple: minimal",
            "cognition": "BWS: NORMAL | Temple: 80% mild-moderate ID",
            "cpp": "BWS: NOT a feature | Temple: 50-60% (Level A GnRH analog)",
            "obesity": "BWS: NOT primary (macrosomia resolves; no hyperphagia) | Temple: truncal obesity 60-70%",
        },
        "tumor_surveillance_protocol": {
            "wilms": {
                "imaging": "Abdominal ultrasound",
                "frequency_age_0_4": "Every 3 months",
                "frequency_age_4_8": "Every 6 months",
                "stop_age": "8 years (Wilms rare after 8)",
                "notes": "Stop earlier if molecular subtype is ICR2-only with no hemihypertrophy (low risk) — discuss with oncology",
            },
            "hepatoblastoma": {
                "biomarker": "Serum AFP (age-adjusted interpretation)",
                "frequency_age_0_4": "Every 3 months",
                "stop_age": "4 years (hepatoblastoma almost exclusively age 0-4)",
                "notes": "AFP physiologically elevated at birth — interpret against age norms; rising trend is the key signal",
            },
            "combined_clinic": "Dedicated BWS surveillance clinic (endocrine + oncology + radiology) recommended at specialist centre",
        },
    }


def get_definitions():
    return {
        "disease": "Beckwith-Wiedemann Syndrome (BWS)",
        "omim": "#130650",
        "gene_definitions": [
            {"gene": "IGF2", "omim": "*147470", "protein": "Insulin-Like Growth Factor 2 (70 aa precursor → 7.5 kDa mature)",
             "location": "11p15.5", "expression": "PATERNALLY expressed only (normal); in BWS → BIALLELICALLY expressed (maternal allele also active)",
             "function": "Principal fetal growth factor; binds IGF1R (growth) and M6PR/IGF2R (clearance); stimulates cell proliferation, differentiation, organ growth, tumour progression",
             "in_bws": "OVEREXPRESSED (1.5–2× normal in ICR2 mechanism; 2× in ICR1 hypermethylation/UPD) → macrosomia + organomegaly + tumour predisposition",
             "contrast_srs": "In SRS: ABSENT biallelically (ICR1 paternal hypomethylation) → profound growth restriction; BWS is the exact opposite"},
            {"gene": "H19", "omim": "*103280", "protein": "H19 lncRNA (non-coding RNA; maternally expressed growth repressor)",
             "location": "11p15.5 (adjacent to IGF2; reciprocal ICR1 regulation)", "expression": "MATERNALLY expressed only (normal); in BWS ICR1 hypermethylation → H19 SILENCED biallelically",
             "function": "Growth repressor lncRNA; sequesters miR-675; upregulates IGFBP3; tumour suppressor",
             "in_bws": "SILENCED when ICR1 is hypermethylated (maternal allele also methylated) → loss of H19 growth suppression; contrast with ICR2 mechanism where H19 remains active",
             "contrast_srs": "In SRS: H19 overexpressed biallelically (ICR1 unmethylated on both alleles) — biallelic H19 suppresses growth"},
            {"gene": "CDKN1C", "omim": "*600856", "protein": "p57KIP2 (316 aa) — cyclin-dependent kinase inhibitor 1C",
             "location": "11p15.5 ICR2 region (KCNQ1 locus)", "expression": "MATERNALLY expressed (paternal CDKN1C silenced by KCNQ1OT1); in BWS → LOST from maternal allele",
             "function": "Cell-cycle brake (G1 arrest via CDK2/cyclin E inhibition); growth inhibitor; tumour suppressor; placental development",
             "in_bws": "ABSENT (ICR2 hypomethylation silences maternal CDKN1C, or CDKN1C LOF mutation destroys function, or UPD removes maternal copy) → cell cycle unchecked → overgrowth",
             "contrast_srs": "In SRS CDKN1C gain-of-function (rare mechanism): excessive cell-cycle braking → growth restriction"},
            {"gene": "KCNQ1OT1 (LIT1)", "omim": "*604115", "protein": "KCNQ1 Opposite Strand/Antisense Transcript 1 (lncRNA)",
             "location": "11p15.5 ICR2 region", "expression": "PATERNALLY expressed lncRNA (maternal KCNQ1OT1 normally silenced by ICR2 methylation)",
             "function": "Silences maternally expressed genes in the ICR2 domain in cis (including CDKN1C, KCNQ1, PHLDA2); chromatin-level repression",
             "in_bws": "BIALLELICALLY expressed when ICR2 maternal is unmethylated → CDKN1C silenced on both alleles → growth unrestricted",
             "contrast_srs": "Not directly involved in SRS (SRS is ICR1/IGF2-H19 mechanism)"},
        ],
        "imprinting_concepts": [
            {"term": "ICR2 (KCNQ1OT1/LIT1 Imprinting Control Region 2)", "definition": "CpG island within KCNQ1 intron 10. UNMETHYLATED on paternal allele → KCNQ1OT1 expressed paternally (silences ICR2 genes paternally). METHYLATED on maternal allele → CDKN1C and other ICR2 genes expressed maternally. In BWS: maternal ICR2 UNMETHYLATED → CDKN1C silenced → overgrowth"},
            {"term": "ICR1 (H19/IGF2 Imprinting Control Region 1)", "definition": "CpG island at 11p15.5 controlling H19 and IGF2 reciprocally. METHYLATED on paternal allele = IGF2 active. UNMETHYLATED on maternal allele = H19 active (CTCF blocks IGF2). In BWS ICR1 hypermethylation: maternal ICR1 also methylated → H19 suppressed, IGF2 active biallelically → maximum overgrowth, highest Wilms risk (29%)"},
            {"term": "Paternal UPD11p15 (upd(11p15)pat)", "definition": "Two paternal copies of 11p15; no maternal copy. Both ICR1 and ICR2 now have paternal pattern → biallelic IGF2 expression + complete CDKN1C loss + KCNQ1OT1 biallelic. Most severe form. Somatic mosaicism is usual (postzygotic); extent of mosaicism predicts severity. SNP array required for detection"},
            {"term": "Genomic imprinting", "definition": "Epigenetic marking of genes based on parental origin; imprinted genes expressed from only one parental allele. Failure: if the wrong parent's allele is silenced (or if the expressed allele is lost), disease results. BWS and SRS are the canonical same-locus pair demonstrating this principle"},
            {"term": "Hemihypertrophy (lateralised overgrowth, LO)", "definition": "One body side larger than the other. In BWS reflects asymmetric mosaicism for UPD or epimutation. KEY CONTRAST WITH SRS: SRS has hemihypotrophy (one side smaller). Hemihypertrophy = HIGHER Wilms tumor risk → intensify surveillance on enlarged side; abdominal ultrasound essential"},
            {"term": "Hyperinsulinism (CHI-BWS)", "definition": "Excess insulin secretion in BWS neonates (especially UPD mechanism: focal hyperinsulinism from mosaic biallelic ABCC8/KCNJ11 inactivation). DIFFERENT from SRS hypoglycemia: SRS = fasting (low IGF2/GH, cortisol response intact), BWS = HYPERINSULINISM (suppress glucose inappropriately). Treatment: diazoxide for diffuse; focal pancreatectomy if 18F-DOPA PET identifies focal lesion"},
            {"term": "Diazoxide (treatment)", "definition": "KATP channel opener — inhibits insulin secretion by keeping KATP channels open → less calcium influx → less insulin release. First-line for BWS hyperinsulinism. Dose 5-15 mg/kg/day. Contraindicated in cardiac disease. Hirsutism and fluid retention are side effects. MUST NOT confuse with SRS management (cornstarch is for fasting hypoglycemia, not hyperinsulinism)"},
            {"term": "AFP (alpha-fetoprotein)", "definition": "Oncofetal protein elevated physiologically at birth (up to 100,000 ng/mL). Drops to adult levels by age 6-12 months. In BWS: serial AFP every 3 months for hepatoblastoma surveillance until age 4. An AFP RISE (not just absolute value) is the key alert. VPA hepatotoxicity can also elevate AFP — confirm with liver panel and imaging"},
        ],
        "drug_classes": [
            {"drug": "Diazoxide", "examples": "Proglycem", "mechanism": "KATP channel opener; inhibits beta-cell insulin secretion; reduces hypoglycemia from hyperinsulinism",
             "evidence_bws": "Level A — FIRST-LINE for BWS hyperinsulinism. Treats the primary cause of neonatal hypoglycemic seizures",
             "monitoring": "Daily weight (fluid retention); ECG if cardiac concern; blood glucose diary; glucose q4-6h during titration"},
            {"drug": "Octreotide", "examples": "Sandostatin", "mechanism": "Somatostatin analog; inhibits GH/insulin/glucagon secretion; reduces hyperinsulinism",
             "evidence_bws": "Level B — second-line for diazoxide-resistant or -intolerant BWS hyperinsulinism",
             "monitoring": "Growth velocity; gallstones (USS annually); subcutaneous injection; thyroid function"},
            {"drug": "Levetiracetam (LEV)", "examples": "Keppra", "mechanism": "SV2A synaptic vesicle protein modulator; broad-spectrum AED",
             "evidence_bws": "Level B — preferred AED if true primary epilepsy develops in BWS (rare); weight-neutral; no hepatic interaction",
             "monitoring": "Behavioural side effects (LEV-rage); CBC at baseline"},
            {"drug": "Valproate (VPA)", "examples": "Depakote, Epilim", "mechanism": "Multiple: GABA potentiation, Na-channel, HDAC inhibition; broad-spectrum AED",
             "evidence_bws": "Level C — USE WITH CAUTION in BWS. VPA hepatotoxicity raises AFP (confounds hepatoblastoma surveillance). Prefer LEV/LTG as first AED option",
             "monitoring": "AFP + LFTs monthly if VPA initiated; strict dose minimisation; avoid in first 2 years of life"},
        ],
        "key_facts_exam": [
            "BWS = MATERNAL LOF at 11p15.5 → EXCESS IGF2 + LOSS of CDKN1C → OVERGROWTH (OPPOSITE of SRS paternal LOF → growth restriction)",
            "SAME LOCUS AS SRS (11p15.5) — OPPOSITE PARENT — OPPOSITE PHENOTYPE: the canonical imprinting pair",
            "ICR2 HYPOMETHYLATION = most common mechanism (50%) — maternal ICR2 unmethylated → biallelic CDKN1C silencing",
            "ICR1 HYPERMETHYLATION (~5%) = HIGHEST Wilms tumor risk (~29%); biallelic IGF2 overexpression",
            "MACROGLOSSIA ~97% = pathognomonic hallmark of BWS",
            "HEMIHYPERTROPHY = one side LARGER (BWS) vs HEMIHYPOTROPHY = one side smaller (SRS)",
            "HYPERINSULINISM = BWS hypoglycemia (diazoxide) ≠ FASTING HYPOGLYCEMIA = SRS (cornstarch) — DO NOT CONFUSE",
            "TUMOR SURVEILLANCE MANDATORY: ultrasound q3mo + AFP q3mo from birth to age 4-8",
            "GH THERAPY CONTRAINDICATED in BWS (patients overgrown; GH may amplify IGF2-excess + tumor risk)",
            "VPA CAUTION in BWS — hepatotoxicity raises AFP, confounding hepatoblastoma surveillance; prefer LEV/LTG",
            "CDKN1C LOF = AD 50% recurrence from carrier mother; familial BWS = check CDKN1C first",
            "ART (IVF/ICSI) increases ICR2 hypomethylation risk ~6× — reproductive history essential",
            "WILMS RISK by mechanism: ICR1 hypermeth ~29% > UPD ~12% > ICR2 hypo ~2% > CDKN1C LOF ~2%",
            "DIAGNOSIS: MS-MLPA 11p15.5 (ICR1+ICR2) first → SNP array (UPD) → CDKN1C sequencing",
        ],
        "updated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


if __name__ == "__main__":
    import json
    ov = get_overview()
    br = get_breakdown()
    df = get_definitions()
    print(f"Overview KPIs: {json.dumps(ov['kpis'], indent=2)}")
    print(f"Cohort size: {len(br['patients'])}")
    print(f"Definitions gene count: {len(df['gene_definitions'])}")
    print("BWS dashboard OK")
