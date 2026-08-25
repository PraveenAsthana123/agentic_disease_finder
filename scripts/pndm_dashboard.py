#!/usr/bin/env python3
"""Permanent Neonatal Diabetes Mellitus (PNDM) Dashboard — KCNJ11 / ABCC8 / INS / GCK / EIF2AK3.

PNDM is neonatal diabetes that does NOT remit — permanent insulin deficiency from birth (<6 months).
  Principal genes:
    KCNJ11 (Kir6.2, K-ATP channel pore subunit, Chr 11p15.1, *600937) — ~30-40% PNDM
    ABCC8  (SUR1, K-ATP channel regulatory subunit, Chr 11p15.1, *600509) — ~20% PNDM
    INS    (preproinsulin, Chr 11p15.5, *176730) — ~10-20% PNDM
    GCK    (glucokinase/hexokinase-IV, glucose sensor, Chr 7p13, *138079) — ~5% PNDM
    EIF2AK3 (PERK, eIF2α kinase 3, Wolcott-Rallison syndrome, Chr 2p11.2, *604032) — ~10% PNDM
    FOXP3  (IPEX syndrome, Chr Xp11.23, *300292) — ~2-3% PNDM
    PTF1A  (pancreatic agenesis, Chr 10p12.2, *607194) — rare
    GATA6  (pancreatic+cardiac agenesis, Chr 18q11.2, *601656) — rare

  KEY CONTRAST with TNDM1 (6q24, PLAGL1): PNDM NEVER REMITS — insulin or sulfo needed LIFELONG.

MECHANISM — K-ATP CHANNEL (KCNJ11 and ABCC8):
  K-ATP channel = Kir6.2 (KCNJ11) × 4 + SUR1 (ABCC8) × 4 (octameric complex)
  Normal: glucose ↑ → ATP/ADP ratio ↑ → K-ATP closes → membrane depolarizes → Ca2+ influx → insulin release
  Gain-of-function KCNJ11/ABCC8 mutations → K-ATP STAYS OPEN despite high ATP → membrane stays polarized
  → NO depolarization → NO Ca2+ influx → NO insulin release → PERMANENT diabetes
  SULFONYLUREA (glibenclamide/glyburide) CLOSES K-ATP directly (ATP-independent site) → bypasses GOF mutation
  → Restores insulin release → replaces insulin injections in >90% of patients!
  Note: KCNJ11 and ABCC8 are at the SAME CHROMOSOMAL LOCUS (11p15.1) — K-ATP complex partners

DEND SYNDROME (Developmental delay + Epilepsy + Neonatal Diabetes):
  Severe KCNJ11 gain-of-function mutations (V59M, I296L, Q52R, R201H, etc.)
  K-ATP channels ALSO expressed in neurons and muscle → DEND = neurological phenotype
  D = Developmental delay / intellectual disability (moderate-severe)
  E = Epilepsy — focal seizures, often poorly controlled on standard AEDs
  N = Neonatal Diabetes
  HIGH-DOSE SULFONYLUREA (glibenclamide up to 0.8-1.0 mg/kg/day) → also improves neurology!
  → Partial DEND (iDEND): only ID + DD without epilepsy (milder KCNJ11 mutations)
  Mechanism: K-ATP GOF in neurons → hyperpolarized neurons → altered firing → seizures + impaired cognition
  Treatment: Sulfo primary (closes neuronal K-ATP too); AEDs adjunct for seizures

WOLCOTT-RALLISON SYNDROME (EIF2AK3 / PERK):
  EIF2AK3 encodes PERK — ER stress sensor kinase (phosphorylates eIF2α → reduces global translation)
  Loss-of-function → ER stress unremediated → beta-cell apoptosis → irreversible PNDM
  Syndromic: Epiphyseal dysplasia + Hepatic failure + Exocrine pancreatic insufficiency
  AR inheritance (bi-allelic loss-of-function)
  No sulfo response (mechanism unrelated to K-ATP) → Insulin LIFELONG
  Most common cause of PNDM in consanguineous populations (Middle East, North Africa)

INS GENE MUTATIONS:
  Misfolded proinsulin → ER stress → beta-cell apoptosis (similar to EIF2AK3 mechanism)
  Dominant negative (heterozygous) or homozygous/compound het
  NO K-ATP involvement → sulfo does NOT work → Insulin LIFELONG
  ~10-20% of PNDM overall; dominant mutations carry ~50% transmission risk

GCK (GLUCOKINASE) PNDM:
  Homozygous or compound heterozygous inactivating mutations (heterozygous = MODY2, very mild)
  Glucose sensor failure: cannot detect rising glucose → no trigger for insulin release
  No K-ATP involvement → sulfo does NOT work → Insulin LIFELONG
  Heterozygous PARENT has MODY2 (fasting glucose 5.5-8 mmol/L, no treatment required)
  GCK PNDM is very rare (~5% PNDM) but important prognostically (pancreas structurally normal)

DIAGNOSTIC SEQUENCE (ADA 2023 / ISPAD 2022):
  ALL infants with diabetes <6 months → molecular testing IMMEDIATELY (do not wait for autoantibodies)
  1. K-ATP panel: KCNJ11 + ABCC8 sequencing — FIRST (change management in majority)
  2. Full PNDM panel: + INS + GCK + EIF2AK3 + FOXP3 + PTF1A + GATA6 (NGS panel)
  3. Chromosome microarray if pancreatic agenesis features → detect copy-number variants
  Note: Pancreatic autoantibodies (GADA, ZnT8, IA-2) NEGATIVE in most PNDM (except FOXP3/IPEX)

TREATMENT — SULFONYLUREA TRANSITION (KCNJ11/ABCC8):
  Start glibenclamide (glyburide) or glipizide — do NOT wait; start even in neonatal period
  Dose titration: start 0.1 mg/kg/day → increase weekly → target 0.4-0.8 mg/kg/day
  Hypoglycemia monitoring: frequent SMBG during transition; reduce/stop insulin as sulfo starts working
  Transition success: >90% for KCNJ11; ~85% for ABCC8; ~50% partial response
  Failure rate: ~10% (usually severe mutations or late start)
  DEND response: Neurological improvement begins weeks-months after starting sulfo (not immediate)

OMIM NUMBERS:
  Disease: Neonatal diabetes mellitus, permanent #606176 (KCNJ11/ABCC8); Wolcott-Rallison #226980
  Genes: KCNJ11 *600937 · ABCC8 *600509 · INS *176730 · GCK *138079
         EIF2AK3 *604032 · FOXP3 *300292 · PTF1A *607194 · GATA6 *601656
  Prevalence: ~1:200,000-400,000 live births (all PNDM); KCNJ11/ABCC8 ~1:500,000 each
"""

from __future__ import annotations
import random, datetime, math

SEED = 301
RNG  = random.Random(SEED)
TODAY = datetime.date.today().isoformat()

# ── Genetic subtype definitions ─────────────────────────────────────────────
SUBTYPES = [
    ("KCNJ11 GOF",        0.35, "Gain-of-function Kir6.2 (K-ATP pore)",           "sulfo",    "DEND/iDEND risk"),
    ("ABCC8 GOF",         0.20, "Gain-of-function SUR1 (K-ATP regulatory)",        "sulfo",    "Similar to KCNJ11"),
    ("INS mutation",      0.18, "Misfolded proinsulin → ER stress → apoptosis",    "insulin",  "Dominant or AR"),
    ("EIF2AK3 (WRS)",     0.10, "Wolcott-Rallison: ER stress + epiphyseal dyspl.", "insulin",  "AR, consanguineous"),
    ("GCK homozygous",    0.05, "Glucose sensor failure; het parent = MODY2",      "insulin",  "Very rare"),
    ("FOXP3 (IPEX)",      0.03, "Immune polyendocrinopathy; autoimmune DM",        "insulin",  "X-linked recessive"),
    ("PTF1A/GATA6/other", 0.09, "Pancreatic agenesis; structural",                 "insulin",  "Rare, multiorgan"),
]

# ── 40-patient PNDM cohort (seed 301) ────────────────────────────────────────
def _make_cohort():
    rng = random.Random(SEED)
    rows = []
    subtype_weights = [s[1] for s in SUBTYPES]
    for i in range(40):
        stype, _, mechanism, therapy, note = rng.choices(SUBTYPES, weights=subtype_weights)[0]
        onset_wk = rng.randint(1, 22)   # weeks (< 6 months = < 26 weeks)
        iugr = rng.random() < 0.60
        birth_wt_sds = round(rng.uniform(-2.8, -0.3) if iugr else rng.uniform(-0.5, 1.1), 1)
        sex = rng.choice(["M", "F"])
        # DEND features (KCNJ11 only)
        dend = False
        dend_type = "—"
        if stype == "KCNJ11 GOF":
            dend_chance = rng.random()
            if dend_chance < 0.18:
                dend = True; dend_type = "DEND (full)"
            elif dend_chance < 0.42:
                dend = True; dend_type = "iDEND"
        # EIF2AK3 syndromic features
        epiphyseal_dyspl = stype == "EIF2AK3 (WRS)"
        hepatic = stype == "EIF2AK3 (WRS)" and rng.random() < 0.70
        # Sulfo transition outcome
        sulfo_success = None
        if therapy == "sulfo":
            if stype == "KCNJ11 GOF":
                sulfo_success = rng.random() < 0.91
            else:
                sulfo_success = rng.random() < 0.84
        # HbA1c
        hba1c = round(rng.uniform(5.8, 7.4) if (therapy == "sulfo" and sulfo_success) else rng.uniform(6.8, 9.5), 1)
        antibody_pos = stype == "FOXP3 (IPEX)"  # only IPEX has positive autoimmunity
        rows.append({
            "id": f"P{i+1:03d}",
            "sex": sex,
            "subtype": stype,
            "mechanism": mechanism,
            "onset_week": onset_wk,
            "birth_wt_sds": birth_wt_sds,
            "iugr": iugr,
            "therapy": therapy,
            "sulfo_success": sulfo_success,
            "dend": dend,
            "dend_type": dend_type,
            "epiphyseal_dyspl": epiphyseal_dyspl,
            "hepatic_involvement": hepatic,
            "antibody_pos": antibody_pos,
            "hba1c": hba1c,
            "note": note,
        })
    return rows


COHORT = _make_cohort()


def get_overview():
    rng = random.Random(SEED)
    cohort = COHORT
    n = len(cohort)

    # Subtype distribution
    subtype_counts = {}
    for p in cohort:
        subtype_counts[p["subtype"]] = subtype_counts.get(p["subtype"], 0) + 1

    # Therapy distribution
    sulfo_eligible = [p for p in cohort if p["therapy"] == "sulfo"]
    sulfo_success  = [p for p in sulfo_eligible if p["sulfo_success"]]
    insulin_only   = [p for p in cohort if p["therapy"] == "insulin"]

    # DEND
    dend_patients = [p for p in cohort if p["dend"]]
    dend_full     = [p for p in dend_patients if p["dend_type"] == "DEND (full)"]
    idend         = [p for p in dend_patients if p["dend_type"] == "iDEND"]

    # EIF2AK3
    wrs_pts = [p for p in cohort if p["subtype"] == "EIF2AK3 (WRS)"]
    wrs_hepatic = [p for p in wrs_pts if p["hepatic_involvement"]]

    # HbA1c
    hba1c_vals = [p["hba1c"] for p in cohort]
    mean_hba1c = round(sum(hba1c_vals) / n, 1)

    # Onset
    onset_vals = [p["onset_week"] for p in cohort]
    mean_onset = round(sum(onset_vals) / n, 1)

    # Antibody
    ab_pos = [p for p in cohort if p["antibody_pos"]]

    kpis = {
        "cohort_size": n,
        "katp_channel_cases": len(sulfo_eligible),
        "sulfo_success_rate": f"{round(len(sulfo_success)/len(sulfo_eligible)*100)}%" if sulfo_eligible else "N/A",
        "dend_prevalence": f"{round(len(dend_patients)/sum(1 for p in cohort if p['subtype']=='KCNJ11 GOF')*100)}%" if any(p["subtype"]=="KCNJ11 GOF" for p in cohort) else "N/A",
        "mean_onset_week": f"wk {mean_onset}",
        "mean_hba1c": f"{mean_hba1c}%",
        "antibody_negative": f"{n - len(ab_pos)}/{n}",
        "insulin_lifelong": len(insulin_only),
    }

    subtype_chart = [
        {"label": st, "n": subtype_counts.get(st, 0), "pct": round(subtype_counts.get(st, 0)/n*100)}
        for st, _, _, _, _ in SUBTYPES
    ]

    therapy_chart = {
        "sulfo_eligible": len(sulfo_eligible),
        "sulfo_success": len(sulfo_success),
        "sulfo_partial_failure": len(sulfo_eligible) - len(sulfo_success),
        "insulin_only": len(insulin_only),
    }

    clinical_alerts = [
        {
            "level": "critical",
            "title": "Test ALL neonatal diabetes <6 months IMMEDIATELY",
            "body": "Do NOT wait for autoantibody results. KCNJ11/ABCC8 testing → sulfonylurea switch changes lives. Delay = unnecessary insulin injections for years.",
            "color": "#b71c1c",
        },
        {
            "level": "critical",
            "title": "SULFONYLUREA in KCNJ11/ABCC8: Start NOW (neonatal period)",
            "body": f"High-dose glibenclamide (0.4–0.8 mg/kg/day) closes K-ATP channel directly. {round(len(sulfo_success)/len(sulfo_eligible)*100)}% of K-ATP PNDM patients in this cohort achieved insulin-free control.",
            "color": "#1b5e20",
        },
        {
            "level": "warning",
            "title": "DEND syndrome: sulfo ALSO improves neurology",
            "body": f"{len(dend_patients)} patients with DEND/iDEND ({len(dend_full)} full DEND, {len(idend)} iDEND). High-dose glibenclamide can partially reverse developmental delay and reduce seizures — start early.",
            "color": "#e65100",
        },
        {
            "level": "info",
            "title": "EIF2AK3 (Wolcott-Rallison): watch for epiphyseal dysplasia + liver",
            "body": f"{len(wrs_pts)} WRS patients. Epiphyseal dysplasia evident by 2-4 years. {len(wrs_hepatic)} have hepatic involvement. AR — counsel consanguineous families (Middle East / North Africa).",
            "color": "#1565c0",
        },
        {
            "level": "info",
            "title": "PNDM ≠ TNDM1: NO remission ever",
            "body": "TNDM1 (6q24, PLAGL1) remits in 100%. PNDM (KCNJ11/ABCC8/INS/GCK/EIF2AK3) NEVER remits. Lifelong therapy mandatory. Distinguish early by molecular testing.",
            "color": "#4a148c",
        },
    ]

    onset_histogram = []
    bins = [(1, 4), (5, 8), (9, 13), (14, 18), (19, 26)]
    for lo, hi in bins:
        cnt = sum(1 for p in cohort if lo <= p["onset_week"] <= hi)
        onset_histogram.append({"range": f"wk {lo}–{hi}", "n": cnt})

    return {
        "title": "PNDM — Permanent Neonatal Diabetes Mellitus",
        "subtitle": "KCNJ11 / ABCC8 / INS / GCK / EIF2AK3 — 40-patient cohort (seed 301)",
        "date": TODAY,
        "kpis": kpis,
        "subtype_chart": subtype_chart,
        "therapy_chart": therapy_chart,
        "clinical_alerts": clinical_alerts,
        "onset_histogram": onset_histogram,
    }


def get_breakdown():
    rng = random.Random(SEED)
    cohort = COHORT

    # Per-gene clinical profile
    gene_profiles = [
        {
            "gene": "KCNJ11",
            "protein": "Kir6.2 (K-ATP pore)",
            "omim_gene": "*600937",
            "omim_disease": "#606176",
            "locus": "11p15.1",
            "freq_pndm": "~30–40%",
            "mechanism": "Gain-of-function → K-ATP stays open → no membrane depolarization → no insulin release",
            "key_mutation_examples": "V59M (DEND), I296L (DEND), Q52R (iDEND), R201H (iDEND/PNDM), E227K (PNDM)",
            "sulfo_response": "~90%+ → glibenclamide 0.4–0.8 mg/kg/day; start at diagnosis; wean insulin over weeks",
            "dend_risk": "18–25% full DEND (V59M most common); ~25% iDEND (ID + DD, no epilepsy); ~50% PNDM-only",
            "neurological_sulfo": "Sulfo also closes neuronal K-ATP → cognitive + motor + seizure improvement over months",
            "inheritance": "Autosomal dominant (de novo ~85%); AD familial (~15%)",
            "surveillance": "Annual HbA1c, neurological assessment (DEND), sulfo dose titration",
        },
        {
            "gene": "ABCC8",
            "protein": "SUR1 (K-ATP regulatory)",
            "omim_gene": "*600509",
            "omim_disease": "#618398",
            "locus": "11p15.1",
            "freq_pndm": "~20%",
            "mechanism": "Gain-of-function SUR1 → K-ATP channel locked open (regulatory site); identical net effect to KCNJ11 GOF",
            "key_mutation_examples": "R1182W, F132L, R1380L, T229I (common in specific populations)",
            "sulfo_response": "~85% respond; some partial — may need higher doses or combination",
            "dend_risk": "DEND rare with ABCC8 (Kir6.2 is primary neurological driver); mainly pure PNDM",
            "neurological_sulfo": "ABCC8 mutations rarely cause DEND; neurological sulfo benefit less dramatic",
            "inheritance": "Autosomal dominant (de novo ~70%); AR bi-allelic forms rare (transient form TNDM3)",
            "surveillance": "Annual HbA1c, fasting glucose; no special neurological monitoring unless symptoms",
        },
        {
            "gene": "INS",
            "protein": "Preproinsulin → Proinsulin → Insulin",
            "omim_gene": "*176730",
            "omim_disease": "#606176",
            "locus": "11p15.5",
            "freq_pndm": "~10–20%",
            "mechanism": "Misfolded proinsulin accumulates in ER → ER stress → UPR activation → beta-cell apoptosis → PNDM",
            "key_mutation_examples": "C96Y, C43R, G32S, A24D, R46Q (all disrupt disulfide bonds or folding)",
            "sulfo_response": "NO (K-ATP unaffected; mechanism is beta-cell loss, not K-ATP dysfunction)",
            "dend_risk": "None — purely pancreatic (no neuronal K-ATP involvement)",
            "neurological_sulfo": "No benefit; insulin lifelong",
            "inheritance": "Autosomal dominant (de novo ~90%); AR homozygous rare",
            "surveillance": "Insulin therapy optimisation; C-peptide progressively declines (lost beta cells)",
        },
        {
            "gene": "GCK",
            "protein": "Glucokinase (Hexokinase IV) — glucose sensor",
            "omim_gene": "*138079",
            "omim_disease": "#606176",
            "locus": "7p13",
            "freq_pndm": "~5%",
            "mechanism": "Homozygous inactivating → BOTH copies of glucose sensor absent → pancreas cannot detect glucose rise → no insulin release signal",
            "key_mutation_examples": "Various homozygous LOF; heterozygous parent = MODY2 (very mild, stable fasting glucose elevation)",
            "sulfo_response": "NO (K-ATP not involved; signaling upstream of K-ATP)",
            "dend_risk": "None",
            "neurological_sulfo": "No benefit",
            "inheritance": "AR (biallelic) for PNDM; AD for MODY2 (heterozygous → mild, no treatment)",
            "surveillance": "Parents have MODY2 (mild); insulin optimisation for proband; CGM useful",
        },
        {
            "gene": "EIF2AK3 (PERK)",
            "protein": "eIF2α kinase 3 (Wolcott-Rallison)",
            "omim_gene": "*604032",
            "omim_disease": "#226980",
            "locus": "2p11.2",
            "freq_pndm": "~10% overall; commonest in consanguineous families",
            "mechanism": "PERK = ER stress sensor → phosphorylates eIF2α → reduces translation under ER stress. LOF → unresolved ER stress → beta-cell + chondrocyte + hepatocyte apoptosis",
            "key_mutation_examples": "Various bi-allelic LOF; common in Middle Eastern/North African populations",
            "sulfo_response": "NO (mechanism unrelated to K-ATP)",
            "dend_risk": "No epilepsy directly from EIF2AK3; hepatic encephalopathy may cause seizures",
            "neurological_sulfo": "No benefit; manage liver disease",
            "inheritance": "Autosomal recessive (biallelic); 25% recurrence risk",
            "surveillance": "LFTs, bone radiographs (epiphyseal dysplasia by age 2-4yr), annual renal function",
        },
    ]

    # K-ATP sulfo transition timeline data (simulated patients)
    katp_pts = [p for p in cohort if p["subtype"] in ("KCNJ11 GOF", "ABCC8 GOF")]
    sulfo_timeline = []
    for wk in [2, 4, 8, 12, 16, 24]:
        responders = sum(1 for p in katp_pts if p["sulfo_success"] and rng.random() < (wk / 24))
        sulfo_timeline.append({"week": wk, "insulin_free": responders, "total_katp": len(katp_pts)})

    # DEND severity vs mutation
    dend_data = [
        {"phenotype": "PNDM only",       "examples": "E227K, C42R, G53D, Y330C",   "severity": "mild",   "sulfo_neuro": "None needed"},
        {"phenotype": "iDEND (partial)",  "examples": "Q52R, R201H, L164P, I182V", "severity": "moderate", "sulfo_neuro": "Cognitive improvement after ~3 months"},
        {"phenotype": "Full DEND",        "examples": "V59M, I296L, G53R, I49F",   "severity": "severe",  "sulfo_neuro": "Motor + cognitive + seizure improvement; months to years"},
    ]

    # Neonatal diabetes comparison
    ndm_comparison = {
        "headers": ["Feature", "PNDM (K-ATP)", "PNDM (INS/GCK/WRS)", "TNDM1 (6q24)", "T1D"],
        "rows": [
            ["Onset",             "<6 mo",        "<6 mo",              "<6 wk (mean)",  "Childhood peak"],
            ["Remission",         "NEVER",         "NEVER",              "100% by 18 mo", "NEVER"],
            ["Sulfo works?",      "YES (>90%)",    "NO",                 "At relapse only","NO"],
            ["DEND?",             "KCNJ11 25%",    "NO",                 "NO",             "NO"],
            ["Autoantibodies",    "Negative",      "Negative",           "Negative",       "POSITIVE"],
            ["First-line Rx",     "Glibenclamide", "Insulin lifelong",   "Insulin→Sulfo",  "Insulin lifelong"],
            ["Test urgency",      "IMMEDIATE",     "IMMEDIATE",          "IMMEDIATE",      "Autoantibodies"],
            ["~Prevalence",       "1:500,000 each","1:M+",               "1:400,000",      "1:300"],
        ]
    }

    # Per-patient table (first 20)
    cohort_table = []
    for p in cohort[:20]:
        cohort_table.append({
            "id": p["id"],
            "sex": p["sex"],
            "subtype": p["subtype"],
            "onset_wk": p["onset_week"],
            "bwt_sds": p["birth_wt_sds"],
            "therapy": p["therapy"],
            "sulfo_success": "✓" if p["sulfo_success"] else ("✗" if p["sulfo_success"] is False else "—"),
            "dend": p["dend_type"],
            "hba1c": f"{p['hba1c']}%",
        })

    return {
        "gene_profiles": gene_profiles,
        "sulfo_timeline": sulfo_timeline,
        "dend_spectrum": dend_data,
        "ndm_comparison": ndm_comparison,
        "cohort_table": cohort_table,
    }


def get_definitions():
    return {
        "disease_overview": {
            "full_name": "Permanent Neonatal Diabetes Mellitus (PNDM)",
            "definition": "Diabetes mellitus diagnosed before 6 months of age that does not remit. Requires lifelong therapy (sulfonylurea for K-ATP mutations; insulin for others).",
            "omim_disease": "#606176 (KCNJ11/ABCC8); #226980 (Wolcott-Rallison/EIF2AK3)",
            "mim_genes": {
                "KCNJ11": "*600937",
                "ABCC8":  "*600509",
                "INS":    "*176730",
                "GCK":    "*138079",
                "EIF2AK3":"*604032",
                "FOXP3":  "*300292",
                "PTF1A":  "*607194",
                "GATA6":  "*601656",
            },
            "prevalence": "~1:200,000–400,000 live births (all PNDM combined); KCNJ11/ABCC8 each ~1:500,000",
            "key_contrast_tndm": "PNDM NEVER remits. TNDM1 (6q24/PLAGL1) remits in 100%. Different genes, mechanisms, management.",
        },
        "genes": {
            "KCNJ11": {
                "full_name": "Potassium Inwardly Rectifying Channel Subfamily J Member 11 (Kir6.2)",
                "size": "390 aa",
                "function": "Pore-forming subunit of pancreatic K-ATP channel (Kir6.2). Forms octameric complex with SUR1 (ABCC8). Closure triggers membrane depolarization → insulin release. GOF → channel stays open → no insulin.",
                "locus": "11p15.1",
                "omim": "*600937",
                "key_fact": "Mutations span PNDM-only → iDEND → full DEND based on severity of K-ATP channel opening. V59M = classic severe DEND mutation. Sulfo dose 0.4-0.8 mg/kg/day closes channel.",
            },
            "ABCC8": {
                "full_name": "ATP-Binding Cassette Sub-Family C Member 8 (SUR1)",
                "size": "1581 aa",
                "function": "Regulatory subunit of K-ATP channel (SUR1). Senses ATP/ADP ratio; NBD2 domain closes channel in response to ATP. GOF → cannot close despite ATP → no insulin. Also site of sulfonylurea (SU) binding.",
                "locus": "11p15.1",
                "omim": "*600509",
                "key_fact": "Adjacent to KCNJ11 on 11p15.1 — heterozygous ABCC8 AR mutations cause TNDM3 (transient) while AD GOF cause PNDM. Sulfo binds SUR1 NBD2 directly — explains why GOF ABCC8 responds to sulfo.",
            },
            "INS": {
                "full_name": "Preproinsulin",
                "size": "110 aa (preproinsulin) → 51 aa (mature insulin)",
                "function": "Prohormone cleaved to yield A- and B-chain insulin. Cysteine-bridge dependent folding in ER. Missense mutations in cysteine residues or folding-critical sites → misfolded proinsulin → ER stress → beta-cell apoptosis.",
                "locus": "11p15.5",
                "omim": "*176730",
                "key_fact": "Close to ICR1 (H19/IGF2) at 11p15.5 — same chromosomal arm as SRS/BWS but different region. NO sulfo response. AD or AR (rare).",
            },
            "GCK": {
                "full_name": "Glucokinase (Hexokinase IV)",
                "size": "465 aa",
                "function": "Glucose-phosphorylating enzyme; acts as glucose sensor in pancreatic beta cells (low affinity, high Km ~8 mmol/L — detects normal vs high glucose). Phosphorylates glucose → glucose-6-phosphate → glycolysis → ATP ↑ → K-ATP closes. Homozygous LOF → no sensor → no insulin trigger.",
                "locus": "7p13",
                "omim": "*138079",
                "key_fact": "Heterozygous LOF = MODY2 (mild lifelong fasting glucose ~5.5-8 mmol/L, no treatment). Homozygous LOF = PNDM (rare; parents each have MODY2). NO sulfo response — mechanism upstream of K-ATP.",
            },
            "EIF2AK3": {
                "full_name": "Eukaryotic Translation Initiation Factor 2 Alpha Kinase 3 (PERK)",
                "size": "1116 aa",
                "function": "ER transmembrane kinase; senses unfolded proteins in ER → phosphorylates eIF2α → attenuates translation → reduces ER protein load (UPR branch). LOF → ER stress unresolved → apoptosis of secretory cells (beta cells, chondrocytes, hepatocytes).",
                "locus": "2p11.2",
                "omim": "*604032",
                "key_fact": "Wolcott-Rallison syndrome: PNDM + epiphyseal dysplasia + hepatic dysfunction + exocrine pancreatic insufficiency. AR. Most common PNDM cause in consanguineous populations. No targeted therapy; insulin + supportive.",
            },
        },
        "katp_channel": {
            "structure": "Kir6.2 (KCNJ11) × 4 + SUR1 (ABCC8) × 4 = octameric complex; embedded in beta-cell plasma membrane",
            "normal_function": "High glucose → glycolysis → ATP/ADP ↑ → binds Kir6.2 → channel closes → membrane depolarizes → VGCC opens → Ca2+ influx → exocytosis of insulin granules",
            "GOF_result": "Channel stays open regardless of ATP → membrane stays at resting potential → no Ca2+ influx → no insulin. Diabetes even when glucose is 600 mg/dL.",
            "sulfo_mechanism": "Glibenclamide/glyburide/glipizide bind SUR1 (ABCC8 NBD2) → close K-ATP independent of ATP/ADP → bypasses GOF mutation → depolarization → Ca2+ → insulin",
            "channel_in_neurons": "K-ATP channels also in CNS neurons and skeletal muscle → GOF in neurons → hyperpolarized neurons → impaired firing → DEND syndrome (DD + epilepsy + neonatal DM)",
        },
        "dend_syndrome": {
            "acronym": "DEND = Developmental delay + Epilepsy + Neonatal Diabetes",
            "full_dend": "Full triad: severe-profound intellectual disability + epilepsy (focal or multifocal, poorly controlled) + PNDM",
            "idend": "Intermediate DEND (iDEND): intellectual disability + neonatal diabetes WITHOUT epilepsy; more common than full DEND (~25% of KCNJ11 PNDM)",
            "mutations": "Severe mutations with greatest K-ATP opening: V59M, I296L (full DEND); Q52R, R201H (iDEND); mild mutations (E227K) → PNDM only",
            "sulfo_neuro_response": "High-dose glibenclamide (0.8-1.0 mg/kg/day) closes neuronal K-ATP → reduces cortical hyperexcitability → motor + cognitive improvement begins 3-12 months; seizure frequency ↓; maximum benefit with early start (<6 months ideal)",
            "aed_role": "AEDs (levetiracetam, valproate, clobazam) used adjunctively; may reduce AED burden after sulfo transition",
        },
        "wolcott_rallison": {
            "full_name": "Wolcott-Rallison Syndrome (EIF2AK3 deficiency)",
            "omim": "#226980",
            "triad": "1. PNDM (onset <6 months) · 2. Epiphyseal dysplasia (short stature, fractures) · 3. Hepatic dysfunction (hepatitis, cirrhosis)",
            "additional_features": "Exocrine pancreatic insufficiency, recurrent infections, renal anomalies in some cases",
            "population": "Highest prevalence in consanguineous families (Middle East, North Africa, Pakistan)",
            "prognosis": "Variable; hepatic failure can be life-limiting; epiphyseal dysplasia progressive; insulin for DM lifelong",
        },
        "treatment_summary": {
            "KCNJ11_ABCC8": {
                "drug": "Glibenclamide (glyburide) or glipizide — K-ATP sulfonylurea",
                "start_dose": "0.1 mg/kg/day (glibenclamide) → increase weekly",
                "target_dose": "0.4–0.8 mg/kg/day; some DEND cases require up to 1.0 mg/kg/day",
                "transition": "Overlap with insulin for 2-4 weeks; reduce insulin 10-20% per day as sulfo takes effect",
                "response_rate": ">90% for KCNJ11; ~85% for ABCC8; assess at 4 weeks",
                "failure_definition": "No reduction in insulin requirements after 4 weeks at target dose → trial complete",
                "monitoring": "SMBG 6-10×/day during transition; CGM ideal; HbA1c at 3 months",
            },
            "INS_GCK_EIF2AK3_other": {
                "drug": "Insulin (all formulations) — LIFELONG",
                "note": "Sulfonylurea CONTRAINDICATED (mechanism unrelated to K-ATP; risk of hypoglycemia without benefit)",
                "monitoring": "HbA1c, CGM if available, annual C-peptide (tracks residual beta-cell function in INS mutations)",
            },
            "what_NOT_to_do": [
                "NEVER treat as T1D without molecular testing — avoid lifelong unnecessary insulin if KCNJ11/ABCC8",
                "NEVER use sulfo for INS/GCK/EIF2AK3/PTF1A — K-ATP is not the problem",
                "NEVER delay K-ATP testing in any infant with DM <6 months",
                "NEVER assume antibody-negative neonatal DM = T2D — PNDM is monogenic",
                "NEVER miss DEND neurological component in KCNJ11 — screen all KCNJ11 PNDM with developmental assessment",
            ],
        },
        "diagnostics": {
            "first_line": "KCNJ11 + ABCC8 targeted sequencing (fastest, most impactful — changes management in majority)",
            "full_panel": "NGS gene panel: KCNJ11, ABCC8, INS, GCK, EIF2AK3, FOXP3, PTF1A, GATA6, PDX1, HNF1B, RFX6 (comprehensive)",
            "exome_wgs": "Exome or genome sequencing if panel negative → ultrarare causes",
            "antibody_panel": "GADA, IA-2, ZnT8, IAA — NEGATIVE in PNDM (except FOXP3/IPEX); positive argues for T1D (rare onset <6 months) or IPEX",
            "c_peptide": "Very low/undetectable in acute phase; tracks residual beta-cell mass in INS-PNDM over time",
            "pancreatic_imaging": "Ultrasound or MRI for PTF1A/GATA6 (pancreatic agenesis); EIF2AK3 shows small/abnormal pancreas",
            "bone_films": "Epiphyseal dysplasia screening in EIF2AK3 (from ~18 months)",
        },
        "contrasts": {
            "PNDM_vs_TNDM1": "PNDM: NEVER remits; TNDM1 (6q24/PLAGL1): 100% remit by 18 months. TNDM1 sulfo works at relapse but NOT neonatal; K-ATP PNDM sulfo works IMMEDIATELY even in neonates.",
            "KCNJ11_vs_ABCC8": "Both K-ATP subunits at 11p15.1; identical net effect (channel stays open); DEND seen with KCNJ11 (Kir6.2 = neuronal K-ATP subunit) but rarely with ABCC8 (SUR1 less dominant in CNS); treatment identical.",
            "PNDM_vs_T1D": "PNDM: onset <6 months, autoantibody-negative, monogenic, sulfo works (K-ATP). T1D: autoimmune, antibody-positive, any age (peak 5-7yr), insulin required lifelong. T1D onset <6 months is extremely rare.",
            "GCK_hom_vs_het": "GCK homozygous = PNDM (both glucose-sensor copies absent). GCK heterozygous = MODY2 (one sensor copy → reduced sensitivity → mild stable hyperglycemia; no treatment, no complications, no insulin).",
        },
        "surveillance_lifelong": {
            "KCNJ11_ABCC8_sulfo": "Annual HbA1c, renal function, HbA1c target <7%; neurological reassessment if DEND; sulfo dose adjustment for growth (weight-based dosing increases with child's weight)",
            "INS_GCK_EIF2AK3": "Annual HbA1c, CGM time-in-range; EIF2AK3: liver surveillance (LFTs, ultrasound), bone films (epiphyseal dysplasia), exocrine supplement titration",
            "genetic_counselling": "KCNJ11/ABCC8 AD (de novo ~85%): risk to offspring 50% if parent carrier; INS AD: 50% risk; GCK/EIF2AK3 AR: 25% recurrence risk; prenatal testing available",
        },
    }
