#!/usr/bin/env python3
"""BH4-Atlas — Complete 6-Gene Tetrahydrobiopterin Disorders Atlas
GCH1 · PTS · QDPR · SPR · PCBD1 · DNAJC12
240-patient aggregate cohort (6 × 40, seeds 920–925)

Tetrahydrobiopterin (BH4) Disorders facts:
  - BH4 is an essential cofactor for: PAH (phenylalanine→tyrosine),
    TH (tyrosine→L-DOPA, start of dopamine/noradrenaline/adrenaline synthesis),
    TPH1/2 (tryptophan→5-HTP, start of serotonin synthesis), and NOS.
  - Defects in BH4 synthesis or regeneration → neurotransmitter deficiency
    (dopamine, serotonin) + hyperphenylalaninemia (most cases).
  - KEY TEACHING POINTS:
      ALL BH4 deficiencies (except GCH1 AD/DRD and SPR) present with elevated Phe on NBS
        → MIMIC PKU — but phenylalanine diet restriction alone DOES NOT HELP.
      SPR: NORMAL Phe on NBS → NOT DETECTED by standard screening → delayed diagnosis.
      QDPR: FOLINIC ACID is MANDATORY (cerebral folate deficiency develops from qBH2 inhibiting DHFR).
      BH4 loading test (20 mg/kg BH4): Phe drops >30% within 4-8h = BH4 responsive.
      Pterin panel (urine pterins + CSF neurotransmitters) MANDATORY for any HPA.
      GCH1 AD (DRD): L-DOPA 1-2 mg/kg/day — exquisite sensitivity pathognomonic.
      PTS: Primapterin (7-biopterin) in urine = PATHOGNOMONIC (also PCBD1).
      Neurotransmitter deficiency: LOW CSF HVA (dopamine pathway) + LOW 5-HIAA (serotonin pathway).

COHORT: 6 × 40 = 240 patient slots (seeds 920–925; gene-specific seeds)
"""

import random

SEED_BASE = 920

# ── All 6 BH4 Genes ───────────────────────────────────────────────────────────────
BH4_GENES = [
    # ── GCH1 — GTP Cyclohydrolase I ──────────────────────────────────────────────
    {
        "gene": "GCH1", "alias": "GCH1 — GTPCH-I deficiency: AD Segawa DRD (OMIM #128230) / AR severe BH4 deficiency (OMIM #233910)",
        "aa": "250 aa", "kDa": "29 kDa",
        "gene_class": "BH4 synthesis step 1: GTP → 7,8-dihydroneopterin-3'-triphosphate",
        "bh4_subgroup": "BH4 de novo synthesis pathway (GCH1 · PTS · SPR · PCBD1)",
        "locus": "14q22.2", "omim_gene": 600225,
        "phenotype": "AD: DRD/Segawa — diurnal dystonia, dramatic L-DOPA response 1-2 mg/kg/day, NORMAL Phe; AR: severe HPA + progressive encephalopathy; Neopterin LOW, Biopterin LOW (both AD and AR)",
        "disease": (
            "GCH1 encodes GTP cyclohydrolase I (GTPCH-I, 250aa, 29kDa), the rate-limiting enzyme "
            "catalysing the first step of BH4 biosynthesis: GTP → 7,8-dihydroneopterin-3'-triphosphate "
            "(H2NTP). Two distinct inheritance modes: "
            "AD GCH1 loss (haploinsufficiency) → DOPA-responsive dystonia (DRD, Segawa syndrome, OMIM #128230): "
            "the classic autosomal dominant dystonia of childhood. Females > males (3:1). BH4 partially reduced "
            "(AD haploinsufficiency) → TH activity mildly compromised → dopamine synthesis reduced → striatal "
            "dopamine deficiency → dystonia. Phenylalanine NORMAL (PAH has enough BH4 residual from one "
            "functional allele). Presents age 6-12y (range 1-30y) with lower limb dystonia, gait abnormality, "
            "diurnal fluctuation (worse evening/night, better after sleep), parkinsonian features in some cases. "
            "AR GCH1 biallelic loss → severe BH4 deficiency (OMIM #233910): marked HPA (Phe elevated on NBS) + "
            "progressive encephalopathy, hypotonia, seizures, hypersalivation — same clinical picture as PTS/QDPR. "
            "Incidence: AD DRD ~1/500,000; AR severe ~1/1,000,000. "
            "Pterin profile: Neopterin LOW (GCH1 step 1 blocked; distinguishes from PTS where neopterin HIGH), "
            "Biopterin LOW. "
            "BH4 loading test: 20 mg/kg → Phe drops >30% in AR form (confirms BH4 deficiency). "
            "DRD: exquisite L-DOPA response at low doses (1-2 mg/kg/day) is PATHOGNOMONIC — much lower dose "
            "than idiopathic PD; motor features resolve almost completely within weeks."
        ),
        "inheritance": "AD (DRD/Segawa, ~1/500,000) or AR (severe BH4 deficiency, ~1/1,000,000). 14q22.2. Females:Males = 3:1 in AD DRD (sex-specific penetrance). Missense dominant-negative (GTPCH is a homodecamer — one bad subunit poisons the complex).",
        "hallmark": (
            "GCH1 HALLMARKS: "
            "(1) DIURNAL FLUCTUATION PATHOGNOMONIC FOR DRD: dystonia worse in evening, better after sleep — "
            "catecholamine circadian rhythm + BH4 regeneration during sleep; no other dystonia has this pattern; "
            "(2) L-DOPA EXQUISITE SENSITIVITY IN DRD: 1-2 mg/kg/day carbidopa/levodopa → dramatic improvement "
            "within 1-2 weeks; doses used in DRD are 5-10× lower than idiopathic PD; overdose gives dyskinesia; "
            "(3) NEOPTERIN LOW (unlike PTS where neopterin is very HIGH): because GCH1 step 1 is blocked, "
            "no dihydroneopterin triphosphate produced → neopterin LOW; biopterin LOW; key differentiator; "
            "(4) NORMAL PHE IN DRD (AD form): AD haploinsufficiency → enough residual GTPCH-I for PAH; "
            "NBS Phe NORMAL → DRD missed by NBS; diagnosis only on clinical grounds + L-DOPA trial; "
            "(5) MISDIAGNOSED AS CEREBRAL PALSY FOR YEARS: typical DRD diagnostic delay 5-10y; "
            "all childhood dystonia should have L-DOPA trial; "
            "(6) AR FORM: same clinical as PTS/QDPR; pterin profile needed; "
            "neopterin LOW (vs PTS: neopterin VERY HIGH) = key differentiator; "
            "(7) GTPCH-I IS A HOMODECAMER: 10 subunits; "
            "AD dominant-negative: one mutant subunit destabilises complex activity"
        ),
        "key_ddx": (
            "GCH1 DDx: "
            "(1) PTS deficiency: neopterin VERY HIGH (not low); most common BH4 HPA cause; primapterin in urine; "
            "(2) QDPR deficiency: biopterin HIGH (qBH2 accumulates); neopterin normal; "
            "(3) Cerebral palsy: no diurnal fluctuation; normal pterins; no L-DOPA response; "
            "(4) SPR deficiency: NORMAL Phe; CSF sepiapterin elevated; "
            "(5) Idiopathic PD: adults; much higher L-DOPA doses needed; no diurnal fluctuation; "
            "(6) AR vs AD GCH1: AR — elevated Phe; AD — normal Phe; pterin levels distinguish; "
            "genetic sequencing confirms"
        ),
        "diet_treatment": "DRD (AD): Carbidopa/L-DOPA 1-2 mg/kg/day in 3 divided doses — exquisite response; titrate slowly to avoid dyskinesia; lifelong treatment; excellent prognosis. BH4 supplementation (sapropterin) adjunct in some AD cases. AR severe: BH4 (sapropterin 5-20 mg/kg/day) + L-DOPA/Carbidopa + 5-HTP for serotonin supplementation. Low-Phe diet in AR form. Folinic acid NOT needed (QDPR pathway intact).",
        "gene_therapy_status": "No approved gene therapy. GCH1 (750bp coding sequence) is small enough for AAV delivery. AAV-mediated GCH1 gene therapy preclinical (mouse DRD models, 2020s). Clinical need limited by excellent L-DOPA response in AD DRD. AR form potential candidate for gene therapy given more severe phenotype.",
        "critical_ci": (
            "CRITICAL: (1) Missing DRD — all childhood dystonia deserves L-DOPA trial; "
            "failure to trial L-DOPA in DRD = years of misdiagnosis as cerebral palsy; "
            "(2) Expecting high L-DOPA doses: DRD responds to 1-2 mg/kg/day; "
            "higher doses → unnecessary dyskinesia; "
            "(3) Misinterpreting neopterin as HIGH — GCH1 shows LOW neopterin; "
            "PTS shows HIGH neopterin; confusing these → wrong diagnosis; "
            "(4) Giving folinic acid in GCH1 (not needed; only QDPR requires it); "
            "(5) Diagnosing AR GCH1 without genetic confirmation when pterins resemble PTS"
        ),
        "nbs_marker": "AR form: elevated Phe on NBS (same as PKU). AD form: NORMAL Phe on NBS — missed by standard screening. Pterin panel: neopterin LOW, biopterin LOW (both forms). BH4 loading test (20 mg/kg): Phe drops >30% (AR). Urine pterins: total pterins low. CSF: low HVA and 5-HIAA (both forms if neurotransmitter synthesis affected). L-DOPA trial: dramatic response diagnostic for DRD.",
        "key_biomarker": "Urine pterins: Neopterin LOW, Biopterin LOW, neopterin:biopterin ratio normal (low:low — unlike PTS high:low). BH4 loading test: >30% Phe reduction (AR form). CSF HVA low (dopamine deficit). CSF 5-HIAA low. GTPCH enzyme activity in RBCs (reduced/absent). GCH1 sequencing. Plasma phenylalanine: elevated (AR) or NORMAL (AD/DRD).",
        "severity_spectrum": "AD DRD (haploinsufficiency): childhood-onset dystonia + diurnal fluctuation + excellent L-DOPA response; normal Phe; normal life expectancy with treatment → AR severe (complete biallelic loss): neonatal HPA + progressive encephalopathy + seizures + hypersalivation; requires full BH4 + neurotransmitter therapy.",
        "founder_variant": "No major founder. AD DRD: many missense variants across GCH1. Common AD: p.Arg184His, p.Arg249Gly (GTPCH-I active site/interface). AR: null/null or compound het.",
        "key_variants": [
            "p.Arg184His — AD DRD hotspot; dominant-negative; interferes with homodecamer",
            "p.Arg249Gly — AD DRD; active site adjacent; dominant-negative",
            "p.Lys224Arg — AD DRD; substrate binding",
            "Biallelic null — AR severe BH4 deficiency; HPA + encephalopathy",
            "p.Asp164Glu — AR; reduced catalytic activity",
        ],
        "seed": SEED_BASE + 0,
    },
    # ── PTS — 6-Pyruvoyltetrahydropterin Synthase ─────────────────────────────────
    {
        "gene": "PTS", "alias": "PTS — 6-PTS deficiency: most common BH4 deficiency/HPA (~60% of BH4 cases); Neopterin VERY HIGH, Primapterin pathognomonic",
        "aa": "145 aa", "kDa": "16 kDa",
        "gene_class": "BH4 synthesis step 2: 7,8-dihydroneopterin-3'-triphosphate → 6-pyruvoyltetrahydropterin",
        "bh4_subgroup": "BH4 de novo synthesis pathway (GCH1 · PTS · SPR · PCBD1)",
        "locus": "11q23.1", "omim_gene": 261640,
        "phenotype": "Most common BH4 HPA cause (~60%): Classic/Severe (<30% activity) — progressive encephalopathy, seizures, hypotonia, hypersalivation despite diet alone; Mild (30-75%) — mild neuro risk; Peripheral (>75%) — benign, liver only; Neopterin VERY HIGH, Biopterin LOW, Primapterin in urine PATHOGNOMONIC",
        "disease": (
            "PTS (also written 6-PTPS) encodes 6-pyruvoyltetrahydropterin synthase (145aa, 16kDa, homodimer), "
            "catalysing BH4 synthesis step 2: 7,8-dihydroneopterin-3'-triphosphate → 6-pyruvoyltetrahydropterin "
            "(6-PTP). PTS biallelic loss is the MOST COMMON cause of BH4 deficiency, accounting for ~60% of "
            "all hyperphenylalaninemia (HPA) due to BH4 defects. "
            "Without PTS: H2NTP accumulates → diverted to 7-biopterin (7-BH4 / primapterin) by an alternative "
            "pathway (neopterin accumulates as well). Pterin hallmark: Neopterin VERY HIGH (upstream accumulates), "
            "Biopterin LOW (downstream blocked), Primapterin (7-biopterin) in urine — PATHOGNOMONIC of PTS (or PCBD1). "
            "Three clinical forms based on residual PTS activity: "
            "Classic/Severe (<30% PTS activity, OMIM #261640): profound BH4 + neurotransmitter deficiency → "
            "progressive encephalopathy, hypotonia, drooling/hypersalivation, myoclonic/tonic seizures, "
            "oculomotor disturbances, truncal hypotonia with limb hypertonia, impaired psychomotor development. "
            "CRITICAL: If treated only with low-Phe diet (like PKU), Phe normalises but neurological deterioration "
            "CONTINUES because neurotransmitters (dopamine, serotonin) remain depleted. MUST treat with "
            "BH4 + L-DOPA/Carbidopa + 5-HTP. "
            "Mild form (30-75% activity): mild intellectual disability risk; some require BH4 only. "
            "Peripheral/benign form (>75% activity): HPA confined to liver enzyme; no brain involvement; "
            "no neurotransmitter supplementation needed; BH4 alone normalises Phe. "
            "Incidence: ~1/500,000 (all forms). ~1/1,000,000 classic severe."
        ),
        "inheritance": "Autosomal recessive. 11q23.1. Both sexes equally affected. Compound heterozygotes common. Allele frequency higher in Asian populations (founder effects in East Asia).",
        "hallmark": (
            "PTS HALLMARKS: "
            "(1) NEOPTERIN VERY HIGH — BIOPTERIN LOW: ratio >5:1 pathognomonic for PTS; "
            "GCH1 shows neopterin LOW (not high); QDPR shows biopterin HIGH (not low); "
            "the neopterin:biopterin inversion >5:1 = PTS until proven otherwise; "
            "(2) PRIMAPTERIN (7-BIOPTERIN) IN URINE: PTS block → H2NTP diverts to 7-biopterin "
            "via an alternative isomerisation; primapterin is absent in normal urine; "
            "PATHOGNOMONIC for PTS (also seen in PCBD1 but rarely); "
            "(3) DIET ALONE FAILS IN CLASSIC FORM: Phe normalises on low-Phe diet but "
            "neurological deterioration CONTINUES — neurotransmitters still depleted; "
            "this is the classic clinical trap; "
            "(4) THREE FORMS BASED ON RESIDUAL ACTIVITY: "
            "peripheral benign (>75%): liver enzyme only, no neuro supplementation needed; "
            "mild (30-75%): some neuro risk, BH4 alone may suffice; "
            "classic severe (<30%): full BH4 + L-DOPA/Carbidopa + 5-HTP mandatory; "
            "(5) CSF NEUROTRANSMITTERS MANDATORY: CSF HVA and 5-HIAA must be measured "
            "BEFORE starting L-DOPA (to confirm deficiency) and to monitor treatment; "
            "(6) BH4 LOADING TEST: 20 mg/kg BH4 → Phe drops >30% within 4-8h = BH4-responsive HPA; "
            "distinguishes BH4 deficiency (responsive) from PAH-classic-PKU (non-responsive); "
            "(7) EAST ASIAN FOUNDER EFFECT: PTS p.Asn52Ser and p.Tyr58His enriched in Chinese/Korean/Japanese"
        ),
        "key_ddx": (
            "PTS DDx: "
            "(1) GCH1 AR: neopterin LOW (not very high); "
            "(2) QDPR: biopterin HIGH; neopterin normal; DHPR enzyme assay very low; folinic acid needed; "
            "(3) PCBD1: neopterin normal or mildly elevated; primapterin present; TRANSIENT + BENIGN; "
            "(4) Classic PKU (PAH deficiency): Phe elevated but BH4 loading test NEGATIVE (Phe does not drop); "
            "pterin profile normal; no neurotransmitter deficiency; "
            "(5) SPR: Phe NORMAL; CSF sepiapterin elevated; MISSED on NBS; "
            "(6) DNAJC12: pterin profile NORMAL; BH4 loading test responsive"
        ),
        "diet_treatment": "Classic/Severe: BH4 (sapropterin) 5-20 mg/kg/day + L-DOPA/Carbidopa 10-15 mg/kg/day L-DOPA + 5-HTP 3-8 mg/kg/day + carbidopa to prevent peripheral decarboxylation of 5-HTP. Low-Phe diet adjunct (not sufficient alone in classic form). Monitor CSF HVA and 5-HIAA to guide neurotransmitter doses. Mild form: BH4 alone ± L-DOPA. Peripheral benign form: BH4 or low-Phe diet alone — NO neurotransmitter supplementation needed.",
        "gene_therapy_status": "No approved gene therapy. PTS (435bp coding) is a small gene amenable to AAV delivery. Hepatocyte-directed gene therapy (AAV8-PTS) normalises Phe in mouse models. CNS delivery also being explored. Clinical translation hindered by the availability of effective pharmacological treatment (BH4 + neurotransmitters).",
        "critical_ci": (
            "CRITICAL: (1) Treating classic PTS with low-Phe diet alone — will normalise Phe "
            "but NOT neurotransmitters → progressive neurological deterioration despite 'normal' Phe; "
            "(2) Not measuring CSF neurotransmitters — cannot guide treatment without CSF HVA/5-HIAA; "
            "(3) Prescribing folinic acid in PTS (NOT needed; only QDPR requires folinic acid); "
            "(4) Diagnosing as PAH-PKU without pterin panel — pterin panel MANDATORY in any NBS-detected HPA; "
            "(5) Treating peripheral/benign form with neurotransmitters — not needed; over-treatment risk"
        ),
        "nbs_marker": "Elevated Phe on NBS (same as PKU). Pterin panel MANDATORY: Neopterin VERY HIGH, Biopterin LOW, Primapterin in urine (7-biopterin). BH4 loading test (20 mg/kg): >30% Phe reduction. Urine pterins by HPLC. CSF: HVA and 5-HIAA (neurotransmitters). PTS enzyme activity. PTS sequencing.",
        "key_biomarker": "Urine neopterin: VERY HIGH (>1000 nmol/mmol creatinine; often >100× ULN). Biopterin: LOW. Neopterin:biopterin ratio >5:1 (hallmark). Primapterin (7-biopterin) in urine (PATHOGNOMONIC). CSF HVA: low (<100 nmol/L). CSF 5-HIAA: low (<100 nmol/L). BH4 loading test: >30% Phe drop. PTS enzyme activity: reduced.",
        "severity_spectrum": "Peripheral/Benign (>75% PTS activity): HPA, liver enzyme only, no neuro → Mild (30-75%): HPA + mild neurodevelopmental risk, BH4 ± L-DOPA → Classic/Severe (<30%): HPA + progressive encephalopathy + seizures + hypersalivation; full BH4 + neurotransmitter supplementation mandatory.",
        "founder_variant": "p.Asn52Ser — East Asian (Chinese, Korean, Japanese) founder allele. p.Tyr58His — East Asian. p.Pro87Leu — European. p.Cys54Arg — Mediterranean.",
        "key_variants": [
            "p.Asn52Ser — East Asian founder; classic form; ~30-40% residual activity in some",
            "p.Tyr58His — East Asian; severe classic; <10% residual activity",
            "p.Pro87Leu — European; severe; near-null",
            "p.Cys54Arg — Mediterranean; severe classic",
            "p.Thr67Ile — mild to moderate form; some residual activity",
        ],
        "seed": SEED_BASE + 1,
    },
    # ── QDPR — Dihydropteridine Reductase ─────────────────────────────────────────
    {
        "gene": "QDPR", "alias": "QDPR — DHPR deficiency: BH4 regeneration failure; Biopterin HIGH; FOLINIC ACID MANDATORY (cerebral folate deficiency); ~20% of BH4 HPA cases",
        "aa": "244 aa", "kDa": "26 kDa",
        "gene_class": "BH4 regeneration: quinonoid BH2 (qBH2) + NADH → BH4 + NAD+ (regenerates BH4 after PAH/TH/TPH reactions)",
        "bh4_subgroup": "BH4 regeneration pathway (QDPR · PCBD1)",
        "locus": "4p15.32", "omim_gene": 261630,
        "phenotype": "HPA + severe progressive encephalopathy if untreated; Biopterin HIGH (qBH2 accumulates, measured as biopterin); UNIQUE: cerebral folate deficiency from qBH2 inhibiting DHFR → FOLINIC ACID (5-formyl-THF) MANDATORY in addition to BH4 + neurotransmitters; ~20% of BH4 HPA cases; DHPR dried blood spot enzyme assay DEFINITIVE",
        "disease": (
            "QDPR encodes dihydropteridine reductase (DHPR, 244aa, 26kDa), which regenerates BH4 from "
            "quinonoid dihydrobiopterin (qBH2) after each catalytic cycle of aromatic amino acid hydroxylases "
            "(PAH, TH, TPH). Every time PAH, TH, or TPH uses BH4, it is oxidised to qBH2; QDPR/DHPR "
            "recycles qBH2 back to BH4 using NADH. "
            "Without QDPR: qBH2 accumulates massively → BH4 pool depletes progressively (even if BH4 "
            "synthesis is intact) → PAH, TH, and TPH all fail → HPA + neurotransmitter deficiency. "
            "UNIQUE CRITICAL FEATURE — CEREBRAL FOLATE DEFICIENCY: "
            "qBH2 is a potent inhibitor of dihydrofolate reductase (DHFR). DHFR normally regenerates "
            "dihydrofolate (DHF) to tetrahydrofolate (THF), essential for 1-carbon metabolism and CNS "
            "folate transport. qBH2 accumulation → DHFR inhibited → brain folate transport blocked "
            "→ cerebral folate deficiency. This causes white matter abnormalities and progressive "
            "cognitive decline INDEPENDENT of BH4/neurotransmitter treatment. "
            "THEREFORE: QDPR treatment MUST include FOLINIC ACID (5-formyl-THF, leucovorin, "
            "15-20 mg/day) in addition to BH4 + L-DOPA/Carbidopa + 5-HTP. "
            "Failure to give folinic acid → progressive cognitive deterioration despite otherwise "
            "good neurotransmitter control. "
            "Pterin profile: Biopterin HIGH (qBH2 is measured as biopterin by standard HPLC → "
            "elevated biopterin); Neopterin NORMAL (synthesis intact). "
            "Hallmark test: DHPR enzyme activity in dried blood spots (DBS) — the definitive "
            "diagnostic test; DHPR <1% of normal = QDPR deficiency. "
            "Incidence: ~1/500,000; accounts for ~20% of BH4 deficiency HPA cases."
        ),
        "inheritance": "Autosomal recessive. 4p15.32. Both sexes equally affected. ~20% of BH4 deficiency HPA. Compound heterozygotes common.",
        "hallmark": (
            "QDPR HALLMARKS: "
            "(1) BIOPTERIN HIGH — NEOPTERIN NORMAL: qBH2 accumulates → measured as biopterin by HPLC; "
            "OPPOSITE of PTS (neopterin very high, biopterin low); synthesis intact but regeneration fails; "
            "(2) FOLINIC ACID MANDATORY — UNIQUE TO QDPR: qBH2 inhibits DHFR → cerebral folate deficiency; "
            "folinic acid (5-formyl-THF, leucovorin) 15-20 mg/day MUST be given; "
            "folic acid does NOT work (DHFR blocked by qBH2 → cannot convert folate → THF); "
            "MUST use 5-formyl-THF (folinic acid / leucovorin) which bypasses DHFR; "
            "brain MRI: periventricular white matter changes (folate deficiency); "
            "(3) DHPR ENZYME ASSAY IN DBS: dried blood spot DHPR enzyme activity is the definitive test; "
            "<1% of normal = QDPR deficiency; available in most metabolic labs; rapid; "
            "(4) BH4 SYNTHESIS IS INTACT: pterins upstream of QDPR are normal; "
            "(5) PROGRESSIVE IF UNTREATED OR INCOMPLETELY TREATED: "
            "neurological deterioration even with low-Phe diet + BH4 + neurotransmitters "
            "if folinic acid omitted → cerebral folate deficiency progresses silently; "
            "(6) CSF FOLATE LOW: confirms cerebral folate deficiency; check CSF folate in all QDPR patients; "
            "CSF folate should be monitored on folinic acid treatment; "
            "(7) TREATMENT TRIAD: BH4 (sapropterin) + L-DOPA/Carbidopa + 5-HTP + FOLINIC ACID "
            "— all four components mandatory in classic QDPR"
        ),
        "key_ddx": (
            "QDPR DDx: "
            "(1) PTS: neopterin VERY HIGH (not normal); biopterin LOW (not high); primapterin in urine; "
            "(2) GCH1 AR: neopterin LOW; biopterin LOW (both low); "
            "(3) PAH-PKU: pterin profile NORMAL; DHPR activity NORMAL; "
            "(4) PCBD1: neopterin normal; primapterin in urine; TRANSIENT benign; "
            "(5) SPR: Phe NORMAL (NBS missed); CSF sepiapterin; "
            "(6) Key: DHPR DBS enzyme assay resolves all BH4 HPA cases definitively"
        ),
        "diet_treatment": "Sapropterin (BH4) 5-20 mg/kg/day + L-DOPA/Carbidopa (8-15 mg/kg/day L-DOPA + carbidopa) + 5-HTP 3-5 mg/kg/day + carbidopa + FOLINIC ACID (leucovorin/5-formyl-THF) 15-20 mg/day. Low-Phe diet adjunct. Monitor CSF HVA, 5-HIAA, and CSF folate. Brain MRI for white matter changes (folate deficiency marker). Folic acid is NOT effective (use folinic acid = 5-formyl-THF).",
        "gene_therapy_status": "No approved gene therapy. QDPR (732bp coding) is small for AAV. Gene therapy research ongoing in principle. High medical urgency due to cerebral folate deficiency component. Pharmacological treatment is complex but effective when all 4 components (BH4 + L-DOPA + 5-HTP + folinic acid) are given.",
        "critical_ci": (
            "CRITICAL: (1) Omitting folinic acid — cerebral folate deficiency will develop even with "
            "perfect Phe and neurotransmitter control; folinic acid is NON-NEGOTIABLE in QDPR; "
            "(2) Using folic acid instead of folinic acid — DHFR is blocked by qBH2; "
            "folic acid cannot be converted to THF; MUST use 5-formyl-THF (leucovorin/folinic acid); "
            "(3) Diagnosing as PTS without DHPR enzyme assay — pterin profiles can overlap; "
            "DHPR DBS assay is cheap, fast, definitive; "
            "(4) Failing to check CSF folate — need to monitor adequacy of folinic acid treatment; "
            "(5) Treating QDPR as PTS (same BH4 + NT supplementation but no folinic acid) = incomplete treatment"
        ),
        "nbs_marker": "Elevated Phe on NBS. Pterin panel: Biopterin HIGH, Neopterin NORMAL (key difference from PTS). DHPR enzyme activity in DBS: <1% of normal = QDPR (DEFINITIVE TEST). BH4 loading test: >30% Phe drop. CSF: HVA low, 5-HIAA low, folate LOW (cerebral folate deficiency). Brain MRI: white matter changes (periventricular). QDPR sequencing.",
        "key_biomarker": "DBS DHPR enzyme activity: <1% of normal (DEFINITIVE). Biopterin HIGH (>3 μmol/mmol creatinine). Neopterin NORMAL. CSF HVA low. CSF 5-HIAA low. CSF folate low (cerebral folate deficiency). Plasma Phe elevated (NBS abnormal). BH4 loading test: >30% Phe drop (distinguishes from PAH-PKU).",
        "severity_spectrum": "Classic QDPR (null/null): neonatal HPA + progressive encephalopathy + seizures; requires full 4-drug regimen including folinic acid → Moderate (compound het with partial allele): childhood HPA + slower progression → Mild partial DHPR activity: HPA + milder neuro; BH4 ± neurotransmitters ± folinic acid.",
        "founder_variant": "No major founder. Various missense and splicing variants. Common: p.Arg221Ter (nonsense; null), p.Arg102Gln (catalytic domain), p.Tyr150Cys.",
        "key_variants": [
            "p.Arg221Ter — nonsense; null; complete DHPR loss; classic severe",
            "p.Arg102Gln — catalytic pocket; complete loss; severe",
            "p.Tyr150Cys — active site adjacent; severe",
            "p.Pro27Leu — N-terminal; partial activity; moderate",
            "p.His63Pro — substrate binding; partial; moderate",
        ],
        "seed": SEED_BASE + 2,
    },
    # ── SPR — Sepiapterin Reductase ───────────────────────────────────────────────
    {
        "gene": "SPR", "alias": "SPR — Sepiapterin Reductase deficiency: NORMAL Phe → MISSED by NBS; CSF HVA/5-HIAA LOW + Sepiapterin PATHOGNOMONIC; L-DOPA + 5-HTP + Folinic acid (NO sapropterin needed)",
        "aa": "261 aa", "kDa": "30 kDa",
        "gene_class": "BH4 synthesis final steps: 6-PTP → 6-lactoyl-BH4 → BH4 (SPR catalyses both carbinol reductions)",
        "bh4_subgroup": "BH4 de novo synthesis pathway (GCH1 · PTS · SPR · PCBD1)",
        "locus": "2p14", "omim_gene": 612716,
        "phenotype": "CRITICAL: NORMAL Phe on NBS → NOT detected by standard screening; CSF analysis mandatory for diagnosis: LOW HVA + LOW 5-HIAA + elevated biopterin (peripheral conversion of sepiapterin) + sepiapterin PATHOGNOMONIC in CSF; motor disorder, ataxia, oculomotor abnormalities, cognitive regression; Treatment: L-DOPA/Carbidopa + 5-HTP + Folinic acid (NOT sapropterin — BH4 synthesis is not the issue in periphery; BH4 regeneration intact)",
        "disease": (
            "SPR encodes sepiapterin reductase (261aa, 30kDa), which catalyses the final two reductive steps "
            "of BH4 biosynthesis: (1) 6-pyruvoyltetrahydropterin (6-PTP) → 6-lactoyl-BH4 (SPR aldo-keto "
            "reductase activity); (2) 6-lactoyl-BH4 → BH4 (SPR carbonyl reductase activity). "
            "CRITICAL CLINICAL DISTINCTION: SPR is expressed in the brain AND periphery. "
            "In PERIPHERAL tissues (liver, etc.), when SPR is absent, an alternative enzyme "
            "(aldo-keto reductase AKR1C3) can bypass SPR and make BH4 from 6-PTP. "
            "In the BRAIN, this bypass is absent/limited → severe BH4 deficiency in the brain "
            "but adequate peripheral BH4. "
            "This explains the key clinical feature: PHENYLALANINE IS NORMAL because liver PAH "
            "has enough BH4 (via peripheral bypass). NBS Phe is NORMAL → SPR deficiency is NOT "
            "DETECTED by standard NBS. This is the most important feature for clinical recognition. "
            "Neurological effects: severe brain BH4 deficiency → TH and TPH severely impaired in CNS → "
            "profound dopamine + serotonin deficiency → motor disorder (DOPA-responsive dystonia-like), "
            "ataxia, oculomotor abnormalities (nystagmus, strabismus), cognitive regression. "
            "Diagnosis: CSF analysis: LOW HVA (homovanillic acid), LOW 5-HIAA (5-hydroxyindoleacetic acid), "
            "ELEVATED biopterin (sepiapterin accumulates → converted to biopterin peripherally; in CSF "
            "biopterin elevated as sepiapterin → biopterin conversion), SEPIAPTERIN in CSF PATHOGNOMONIC "
            "(sepiapterin is the substrate that accumulates; not routinely measured but diagnostic when done). "
            "Treatment: L-DOPA/Carbidopa + 5-HTP + Folinic acid. "
            "Sapropterin (BH4) is NOT needed — the problem is not BH4 synthesis/regeneration; "
            "peripheral PAH has enough BH4 via bypass. L-DOPA bypasses TH; 5-HTP bypasses TPH. "
            "Folinic acid given as adjunct (sepiapterin can interfere with folate metabolism). "
            "Incidence: ~1/500,000-1,000,000+."
        ),
        "inheritance": "Autosomal recessive. 2p14. Both sexes equally affected. Rare: <100 cases described. Pan-ethnic.",
        "hallmark": (
            "SPR HALLMARKS: "
            "(1) NORMAL PHE ON NBS — THE DIAGNOSTIC TRAP: PAH in liver has enough BH4 via peripheral bypass; "
            "Phe is NORMAL; standard NBS MISSES SPR deficiency; "
            "any child with unexplained dopa-responsive motor disorder / ataxia / oculomotor abnormalities "
            "must have CSF neurotransmitter analysis even if NBS was normal; "
            "(2) CSF ANALYSIS IS MANDATORY FOR DIAGNOSIS: "
            "CSF HVA low (dopamine deficiency) + 5-HIAA low (serotonin deficiency) = neurotransmitter profile; "
            "urine/plasma pterins may appear normal because peripheral bypass works; "
            "CSF biopterin elevated (sepiapterin → biopterin in CSF); "
            "(3) SEPIAPTERIN IN CSF PATHOGNOMONIC: sepiapterin is the accumulating substrate; "
            "not routinely tested but diagnostic; specialised pteridine HPLC required; "
            "(4) L-DOPA + 5-HTP + FOLINIC ACID (NOT SAPROPTERIN): "
            "sapropterin addresses BH4 deficiency in periphery (not needed — bypass works there); "
            "L-DOPA bypasses TH block in brain; 5-HTP bypasses TPH block; "
            "folinic acid for CNS folate support (sepiapterin-folate competition); "
            "(5) DOPA-RESPONSIVE MOTOR DISORDER: resembles DRD but ataxia + oculomotor features more prominent; "
            "L-DOPA response is good but not as dramatic as GCH1-DRD; "
            "(6) COGNITIVE REGRESSION: seen in untreated/late-diagnosed cases; "
            "early treatment important to prevent cognitive decline"
        ),
        "key_ddx": (
            "SPR DDx: "
            "(1) GCH1 AD DRD: Phe normal (both); DRD responds at lower L-DOPA; no oculomotor/ataxia; "
            "pterin profile: neopterin low in GCH1-DRD; CSF biopterin not elevated; "
            "(2) PTS peripheral/mild: Phe elevated (not normal); primapterin present; neopterin very high; "
            "(3) QDPR: Phe elevated; biopterin high; DHPR DBS assay positive; folinic acid mandatory; "
            "(4) Friedreich ataxia: no CSF HVA/5-HIAA deficiency; GAA repeat expansion; "
            "(5) DNAJC12: Phe may be elevated; pterin profile normal; BH4 loading test responsive; "
            "CSF HVA/5-HIAA low; sapropterin responsive; "
            "(6) Any unexplained childhood dystonia + normal NBS → CSF neurotransmitters mandatory"
        ),
        "diet_treatment": "L-DOPA/Carbidopa 4-10 mg/kg/day L-DOPA (in 4-5 divided doses; cross BBB; L-DOPA crosses, dopamine does not) + 5-HTP 2-5 mg/kg/day (precursor for serotonin synthesis) + carbidopa (prevents peripheral L-DOPA decarboxylation; allows more L-DOPA to reach brain) + Folinic acid 15 mg/day (CNS folate support). Sapropterin (BH4) NOT indicated — Phe is normal; peripheral BH4 synthesis is adequate via bypass. Titrate L-DOPA carefully (dyskinesia risk at high doses). Monitor CSF HVA and 5-HIAA as treatment response markers.",
        "gene_therapy_status": "No approved gene therapy. SPR (783bp coding) is small for AAV. AAV2/9-SPR CNS gene therapy would need to target neurons specifically (unlike liver-directed gene therapy for other BH4 disorders). Proof-of-concept work in murine models (2020s). L-DOPA + 5-HTP treatment is effective when started early; gene therapy for severe/refractory cases.",
        "critical_ci": (
            "CRITICAL: (1) Reassuring family that NBS was normal — SPR is NOT detected by standard Phe-based NBS; "
            "normal NBS does NOT exclude BH4 deficiency; any motor disorder + normal NBS needs CSF analysis; "
            "(2) Prescribing sapropterin (BH4) — does not help (peripheral bypass already provides adequate BH4); "
            "may even paradoxically worsen some sepiapterin-accumulation dynamics; "
            "(3) Not measuring sepiapterin in CSF — standard CSF neurotransmitter panels may not include "
            "sepiapterin; request specifically; "
            "(4) Diagnosing as idiopathic dystonia without CSF — CSF is required for diagnosis; "
            "(5) Omitting folinic acid from treatment regimen"
        ),
        "nbs_marker": "NORMAL Phe on NBS — NOT detected. No standard NBS marker. Diagnosis requires: CSF analysis (HVA low, 5-HIAA low, biopterin elevated, sepiapterin if measured). Urine pterins may appear near-normal. SPR enzyme activity in fibroblasts (reduced). SPR sequencing. High clinical suspicion needed in any child with unexplained DOPA-responsive motor disorder + normal NBS.",
        "key_biomarker": "CSF HVA: LOW (<100 nmol/L). CSF 5-HIAA: LOW (<100 nmol/L). CSF biopterin: ELEVATED (sepiapterin → biopterin conversion in CSF). CSF sepiapterin: elevated (PATHOGNOMONIC if measured). Plasma Phe: NORMAL. Urine pterins: may appear near-normal (peripheral bypass compensates). SPR fibroblast enzyme activity: reduced/absent.",
        "severity_spectrum": "All SPR deficiency is neurologically severe if untreated (brain-specific BH4 deficiency). Severity correlates with residual SPR activity and age of treatment initiation. Early-treated: near-normal cognitive outcome possible. Late-diagnosed: variable cognitive regression, dystonia, ataxia. All forms have Phe NORMAL — diagnostic challenge.",
        "founder_variant": "No founder allele. Very rare. Missense, frameshift, splicing variants. Common reported: p.Asn47Ser, p.Arg150Cys, p.Ser158Asn (SPR aldo-keto reductase domain).",
        "key_variants": [
            "p.Asn47Ser — aldo-keto reductase domain; severe; near-null",
            "p.Arg150Cys — carbonyl reductase domain; severe",
            "p.Ser158Asn — active site; severe",
            "p.Leu209Phe — cofactor binding; partial activity; moderate",
            "Exon 2-3 deletion — null; neonatal presentation",
        ],
        "seed": SEED_BASE + 3,
    },
    # ── PCBD1 — Pterin-4α-Carbinolamine Dehydratase 1 ────────────────────────────
    {
        "gene": "PCBD1", "alias": "PCBD1 — PCD deficiency / Primapterinuria: TRANSIENT BENIGN HPA; also MODY10 (HNF1α/β nuclear cofactor); primapterin (7-biopterin) in urine PATHOGNOMONIC",
        "aa": "104 aa", "kDa": "12 kDa",
        "gene_class": "BH4 regeneration auxiliary: pterin-4α-carbinolamine → quinonoid BH2 (qBH2) → QDPR → BH4; also HNF1α/β nuclear cofactor (dimerisation cofactor of homeobox)",
        "bh4_subgroup": "BH4 regeneration pathway (QDPR · PCBD1)",
        "locus": "10q22.1", "omim_gene": 126090,
        "phenotype": "TRANSIENT + BENIGN: mild HPA normalises spontaneously within months; primapterin (7-biopterin) in urine PATHOGNOMONIC; NO neurotransmitter supplementation usually needed (most benign); also causes MODY10 (HNF1α/β cofactor mutations → maturity-onset diabetes of the young)",
        "disease": (
            "PCBD1 encodes pterin-4α-carbinolamine dehydratase 1 (PCD, 104aa, 12kDa). PCD has two distinct "
            "biological roles: "
            "(1) BH4 REGENERATION AUXILIARY: During the PAH/TH/TPH catalytic cycle, BH4 is first oxidised to "
            "4α-hydroxy-BH4 (pterin-4α-carbinolamine); PCD normally dehydrates this carbinolamine back to "
            "quinonoid BH2 (qBH2), which is then recycled to BH4 by QDPR/DHPR. "
            "Without PCD: 4α-hydroxy-BH4 is not properly dehydrated → shunts to 7-biopterin "
            "(primapterin/7-BH4) via non-enzymatic isomerisation. "
            "Primapterin (7-biopterin) accumulates in urine — PATHOGNOMONIC of PCBD1 deficiency "
            "(also seen in PTS where PTS block also leads to primapterin production). "
            "(2) HNF1α/β NUCLEAR COFACTOR: PCBD1 protein also functions in the nucleus as the "
            "dimerisation cofactor of liver-enriched transcription factor homeobox (DCoH/DCOH1). "
            "PCBD1 mutations affecting the dimerisation cofactor function → impaired HNF1α/HNF1β "
            "transcriptional activity → MODY10 (maturity-onset diabetes of the young, type 10). "
            "CLINICAL: PCBD1/PCD deficiency is THE MOST BENIGN BH4 disorder. "
            "HPA: mild, TRANSIENT — Phe mildly elevated, normalises spontaneously within "
            "6-24 months without treatment (enzyme activity often recovers or compensatory "
            "pathways develop; OR the QDPR pathway handles the qBH2 deficit). "
            "No progressive neurological deterioration. No neurotransmitter deficiency severe enough "
            "to require treatment in most cases. "
            "Some patients may need short courses of BH4 (sapropterin) to normalise Phe during the "
            "transient HPA phase. "
            "Incidence: ~1/500,000-1,000,000."
        ),
        "inheritance": "Autosomal recessive. 10q22.1. Both sexes equally affected. Very rare. MODY10 from the same gene (HNF1 cofactor function).",
        "hallmark": (
            "PCBD1 HALLMARKS: "
            "(1) TRANSIENT HPA — MOST BENIGN BH4 DISORDER: Phe mildly elevated on NBS; "
            "normalises spontaneously within months without treatment; "
            "NO treatment usually needed; reassurance important to avoid over-treatment; "
            "(2) PRIMAPTERIN (7-BIOPTERIN) IN URINE — PATHOGNOMONIC: "
            "primapterin absent in normal urine; present in PCBD1 (and PTS); "
            "differentiates from GCH1 (no primapterin), QDPR (no primapterin), SPR (no primapterin); "
            "(3) NO NEUROTRANSMITTER SUPPLEMENTATION NEEDED: "
            "BH4 regeneration is only transiently impaired; QDPR/DHPR still works; "
            "CSF HVA and 5-HIAA are NORMAL or near-normal; "
            "do NOT give L-DOPA or 5-HTP (inappropriate over-treatment); "
            "(4) DUAL FUNCTION — MODY10: PCBD1 is the DCoH cofactor for HNF1α/HNF1β; "
            "MODY10 mutations primarily affect the cofactor function (not BH4 recycling); "
            "MODY10: young adult onset diabetes, autosomal dominant, pancreatic beta-cell dysfunction; "
            "(5) DIFFERENTIATION FROM PTS: both have primapterin; "
            "PTS has neopterin VERY HIGH; PCBD1 has neopterin NORMAL or mildly elevated; "
            "PTS needs BH4 + neurotransmitters; PCBD1 does NOT; "
            "DHPR DBS assay: NORMAL in PCBD1 (unlike QDPR); "
            "(6) FOLLOW-UP IS IMPORTANT: confirm HPA resolves; check CSF NT if any clinical concerns"
        ),
        "key_ddx": (
            "PCBD1 DDx: "
            "(1) PTS: neopterin VERY HIGH (not normal); primapterin also present; severe if classic form; "
            "needs BH4 + neurotransmitters (unlike PCBD1 which usually needs nothing); "
            "(2) QDPR: biopterin HIGH; DHPR DBS low; folinic acid mandatory; NO primapterin typically; "
            "(3) PAH-PKU: no primapterin; pterin profile normal; BH4 loading test negative; "
            "(4) GCH1 AR: neopterin LOW; biopterin LOW; NO primapterin; "
            "(5) SPR: Phe NORMAL; CSF HVA/5-HIAA low; "
            "(6) MODY differentiation: MODY10 (PCBD1 HNF1 cofactor mutation) presents as adult diabetes; "
            "MODY1 (HNF4A), MODY3 (HNF1A), MODY5 (HNF1B) are different genes in same pathway"
        ),
        "diet_treatment": "Most cases: NO treatment needed. Transient HPA resolves spontaneously within 6-24 months. Short-course BH4 (sapropterin) may be given during HPA phase if Phe significantly elevated (>600 μmol/L). Do NOT give L-DOPA or 5-HTP — neurotransmitters are not deficient. Low-Phe diet rarely needed (transient phase only). Monitor Phe levels until normalisation. CSF neurotransmitters only if clinical neurological concerns. MODY10 treatment: oral hypoglycaemics / sulphonylureas as for other MODY types.",
        "gene_therapy_status": "No gene therapy needed — condition is benign and self-resolving. PCBD1 research interest focuses on the DCoH/HNF1 cofactor function for understanding MODY10. The BH4 phenotype does not require gene therapy.",
        "critical_ci": (
            "CRITICAL: (1) Over-treating with L-DOPA or 5-HTP — NOT indicated; "
            "CSF neurotransmitters are normal; iatrogenic neurotransmitter excess causes dyskinesia/serotonin syndrome; "
            "(2) Prolonged low-Phe diet unnecessarily — HPA is transient; "
            "(3) Confusing with PTS (both have primapterin) — check neopterin level: "
            "PTS = very high neopterin; PCBD1 = normal neopterin; "
            "(4) Not recognising MODY10 connection — adult diabetes may be due to PCBD1 HNF1 cofactor mutation; "
            "(5) Missing diagnosis of PCBD1 as cause of unexpected MODY in family with prior neonatal HPA history"
        ),
        "nbs_marker": "Mildly elevated Phe on NBS. Pterin panel: Primapterin in urine (PATHOGNOMONIC). Neopterin NORMAL or mildly elevated. Biopterin NORMAL. DHPR enzyme activity: NORMAL (distinguishes from QDPR). BH4 loading test: Phe drops. PCD enzyme activity in liver/fibroblasts. PCBD1 sequencing. Phe normalises spontaneously on follow-up.",
        "key_biomarker": "Primapterin (7-biopterin) in urine: PRESENT (PATHOGNOMONIC). Neopterin: NORMAL or mildly elevated. Biopterin: NORMAL. DHPR DBS activity: NORMAL (key difference from QDPR). Plasma Phe: mildly elevated (NBS positive) → normalises spontaneously. CSF HVA: NORMAL. CSF 5-HIAA: NORMAL.",
        "severity_spectrum": "Essentially all PCBD1 BH4 deficiency is benign/transient: neonatal/infant mild HPA → spontaneous Phe normalisation within 6-24 months → no neurological consequences. MODY10 (HNF1 cofactor function): separate phenotype (young adult diabetes); not related to BH4 deficiency severity.",
        "founder_variant": "No founder allele. Very rare overall. PCBD1 and MODY10 mutations affect different functional domains. Common BH4 phenotype mutations affect PCD dehydratase domain. MODY10 mutations affect DCoH dimerisation interface.",
        "key_variants": [
            "p.Trp58Ter — dehydratase domain; null; BH4 phenotype",
            "p.Arg67Gln — active site; BH4 phenotype; partial activity",
            "p.Ile76Thr — DCoH domain; MODY10 phenotype primarily",
            "p.Ser80Arg — DCoH interface; MODY10",
            "p.Arg61Trp — dehydratase; BH4 phenotype",
        ],
        "seed": SEED_BASE + 4,
    },
    # ── DNAJC12 — DnaJ Chaperone Family Member C12 ───────────────────────────────
    {
        "gene": "DNAJC12", "alias": "DNAJC12 — chaperone for PAH/TH/TPH; HPA + neurotransmitter deficiency; NORMAL pterin profile; Sapropterin (BH4) RESPONSIVE; most recently characterised (Bherer 2017)",
        "aa": "198 aa", "kDa": "22 kDa",
        "gene_class": "Molecular chaperone for aromatic amino acid hydroxylases (PAH, TH, TPH1/2): stabilises enzyme folding/activity; NOT directly in BH4 pathway",
        "bh4_subgroup": "BH4 cofactor utilisation (DNAJC12 chaperone group — PAH/TH/TPH stability)",
        "locus": "10q21.3", "omim_gene": 606875,
        "phenotype": "HPA + neurotransmitter deficiency (CSF HVA/5-HIAA LOW); NORMAL pterin profile (neopterin, biopterin both NORMAL) → EASILY MISSED if only pterins checked; BH4 loading test RESPONSIVE (Phe drops >30%); Treatment: Sapropterin (BH4) — often sufficient without additional neurotransmitters; most recently characterised BH4-related disorder (Bherer et al. 2017 AJHG)",
        "disease": (
            "DNAJC12 encodes DnaJ heat shock protein family member C12 (198aa, 22kDa), a co-chaperone of the "
            "HSP40/HSP70 system. First described as causing hyperphenylalaninemia in 2017 (Bherer et al., "
            "American Journal of Human Genetics). "
            "DNAJC12 function: acts as a chaperone specifically for aromatic amino acid hydroxylases: "
            "PAH (phenylalanine hydroxylase), TH (tyrosine hydroxylase), and TPH1/2 (tryptophan hydroxylase). "
            "All three enzymes require BH4 as cofactor AND correct folding/stability for activity. "
            "DNAJC12 is required for proper folding and stabilisation of PAH, TH, and TPH. "
            "Without DNAJC12: PAH, TH, and TPH are unstable and degraded prematurely, despite: "
            "(a) normal BH4 levels, (b) normal BH4 synthesis, (c) normal BH4 regeneration. "
            "Result: HPA (PAH unstable) + neurotransmitter deficiency (TH/TPH unstable) "
            "despite normal BH4 levels → pterin profile is NORMAL. "
            "Key diagnostic trap: pterin profile appears NORMAL → diagnosis MISSED if pterins alone are checked. "
            "BH4 loading test IS RESPONSIVE (>30% Phe reduction within 4-8h): "
            "mechanism: sapropterin (pharmacological BH4 at high dose) → stabilises PAH/TH/TPH by "
            "chaperone-independent pharmacological chaperone effect (BH4 binds to enzyme active site → "
            "prevents misfolding/aggregation → enzyme half-life increases). "
            "Treatment: Sapropterin (BH4) 5-20 mg/kg/day is often sufficient. "
            "Some patients may also need L-DOPA/Carbidopa + 5-HTP if neurotransmitter deficiency severe. "
            "CSF: LOW HVA and LOW 5-HIAA (neurotransmitter deficiency despite normal pterins). "
            "Incidence: ~1/1,000,000+. Very rare — fewer than 50 cases described to 2026."
        ),
        "inheritance": "Autosomal recessive. 10q21.3. Both sexes equally affected. Very rare. Compound heterozygotes described.",
        "hallmark": (
            "DNAJC12 HALLMARKS: "
            "(1) PTERIN PROFILE APPEARS NORMAL — THE DIAGNOSTIC TRAP: "
            "neopterin NORMAL, biopterin NORMAL, no primapterin; "
            "if clinician stops at pterins → diagnosis missed; "
            "MUST also do BH4 loading test AND CSF neurotransmitters; "
            "(2) BH4 LOADING TEST RESPONSIVE (>30% PHE DROP): "
            "sapropterin stabilises PAH by pharmacological chaperone effect (not by providing BH4 per se); "
            "this is what makes it BH4-responsive despite normal BH4; "
            "(3) CSF HVA AND 5-HIAA LOW: neurotransmitter deficiency is present; "
            "TH and TPH are also unstable (chaperone needed for all three hydroxylases); "
            "CSF analysis essential — confirms neurotransmitter deficiency; "
            "(4) SAPROPTERIN OFTEN SUFFICIENT: unlike other BH4 disorders, "
            "sapropterin alone (without L-DOPA + 5-HTP) often controls both Phe AND neurotransmitter production; "
            "mechanism: pharmacological BH4 stabilises TH/TPH in addition to PAH; "
            "(5) MOST RECENTLY CHARACTERISED: Bherer et al. 2017 AJHG; "
            "may be under-recognised due to normal pterin profile; "
            "any BH4-responsive HPA with normal pterins should prompt DNAJC12 sequencing; "
            "(6) CHAPERONE FUNCTION IS UNIQUE: DNAJC12 is the only BH4-related gene that is "
            "NOT directly in the BH4 synthesis or regeneration pathway — it is a cofactor for "
            "the enzymes THAT USE BH4"
        ),
        "key_ddx": (
            "DNAJC12 DDx: "
            "(1) PTS: neopterin VERY HIGH (not normal); primapterin present; "
            "(2) QDPR: biopterin HIGH; DHPR DBS low; folinic acid needed; "
            "(3) PAH-classic-PKU: BH4 loading test NEGATIVE (Phe does NOT drop); "
            "no neurotransmitter deficiency; "
            "(4) PAH-BH4-responsive (mild/moderate PKU with partial PAH): "
            "BH4 loading test also responsive; NO neurotransmitter deficiency (CSF NT normal); "
            "DNAJC12: CSF NT LOW — key differentiator; "
            "(5) GCH1 AR: neopterin LOW; biopterin LOW; different pterin pattern; "
            "(6) SPR: Phe NORMAL; CSF biopterin elevated; "
            "(7) KEY: DNAJC12 = BH4-responsive + normal pterins + low CSF NT"
        ),
        "diet_treatment": "Sapropterin (BH4) 5-20 mg/kg/day — pharmacological chaperone effect stabilises PAH/TH/TPH; often sufficient to control both Phe AND neurotransmitter synthesis. If CSF HVA/5-HIAA remain low on sapropterin alone: add L-DOPA/Carbidopa + 5-HTP. Low-Phe diet adjunct if Phe not fully controlled by sapropterin. Monitor CSF HVA and 5-HIAA periodically. Sapropterin is well-tolerated long term (same as BH4-responsive PKU treatment).",
        "gene_therapy_status": "No approved gene therapy. DNAJC12 (594bp coding) is small. Gene therapy of interest given the multiple enzyme targets (PAH, TH, TPH all depend on DNAJC12). Pharmacological treatment (sapropterin) is effective and well-tolerated. Research interest in understanding HSP40/HSP70 chaperone networks for enzyme stabilisation.",
        "critical_ci": (
            "CRITICAL: (1) Dismissing as BH4-responsive PKU because pterins are normal and Phe drops on BH4 — "
            "must check CSF: CSF HVA and 5-HIAA LOW in DNAJC12 (normal in BH4-responsive PAH-PKU); "
            "(2) Not doing BH4 loading test because pterins are normal — always do BH4 loading test "
            "in any HPA with normal pterins; "
            "(3) Assuming normal pterin profile excludes BH4 pathway — DNAJC12 proves this wrong; "
            "(4) Not sequencing DNAJC12 in unexplained BH4-responsive HPA with normal pterins; "
            "(5) Withholding neurotransmitter supplementation if sapropterin alone fails to normalise CSF NT"
        ),
        "nbs_marker": "Elevated Phe on NBS (HPA). Pterin panel: NORMAL (neopterin normal, biopterin normal) — key distinguishing feature. BH4 loading test (20 mg/kg): >30% Phe drop (BH4 responsive). CSF: HVA low + 5-HIAA low (neurotransmitter deficiency despite normal pterins). DNAJC12 sequencing (required for definitive diagnosis). BH4-responsive HPA with normal pterins = DNAJC12 until proven otherwise.",
        "key_biomarker": "Plasma Phe: elevated (NBS positive). Pterin panel: NORMAL (neopterin normal, biopterin normal) — KEY. BH4 loading test: >30% Phe drop. CSF HVA: LOW. CSF 5-HIAA: LOW. DHPR DBS: NORMAL. DNAJC12 sequencing: biallelic pathogenic variants.",
        "severity_spectrum": "Phenotype range not fully characterised (rare disease). Published cases: moderate to severe HPA + neurotransmitter deficiency. All cases BH4 loading test responsive. Sapropterin alone controls Phe in most; some need additional neurotransmitter supplementation for CSF NT normalisation. Long-term neurological outcome better with early sapropterin treatment.",
        "founder_variant": "No founder allele described. Very rare. All cases compound heterozygotes or homozygotes for DNAJC12 missense/splicing variants. Most variants cluster in J-domain or C-terminal region.",
        "key_variants": [
            "p.Gln22Ter — J-domain nonsense; null; HPA + neurotransmitter deficiency",
            "p.Arg95Trp — HPD-interacting domain; partial chaperone function",
            "p.Ile147Thr — C-terminal structural; moderate",
            "p.Gly64Ser — J-domain core; near-null activity",
            "Exon 3 deletion — null; classic presentation",
        ],
        "seed": SEED_BASE + 5,
    },
]


def _make_patients(gene_dict):
    """Generate 40 synthetic patient records for a given BH4 gene."""
    rng = random.Random(gene_dict["seed"])
    gene = gene_dict["gene"]

    # Phenotypic class probabilities per gene
    PHENO_PROBS = {
        "GCH1":   [0.60, 0.40, 0.00],   # AD DRD / AR severe / (no 3rd class)
        "PTS":    [0.50, 0.30, 0.20],   # Classic severe / Mild / Peripheral benign
        "QDPR":   [0.65, 0.25, 0.10],   # Classic severe / Moderate / Mild
        "SPR":    [0.70, 0.20, 0.10],   # Severe neurological / Moderate / Mild
        "PCBD1":  [0.70, 0.20, 0.10],   # Transient benign / Mild / Asymptomatic
        "DNAJC12":[0.55, 0.35, 0.10],   # Classic HPA+NT deficiency / Moderate / Mild
    }
    CLASS_NAMES = {
        "GCH1":   ["AD DRD Segawa", "AR Severe HPA", "AR Moderate"],
        "PTS":    ["Classic Severe (<30%)", "Mild (30-75%)", "Peripheral Benign (>75%)"],
        "QDPR":   ["Classic Severe", "Moderate", "Mild"],
        "SPR":    ["Severe Neurological", "Moderate", "Mild"],
        "PCBD1":  ["Transient Benign", "Mild Transient", "Asymptomatic"],
        "DNAJC12":["Classic HPA+NT", "Moderate HPA", "Mild HPA"],
    }
    probs = PHENO_PROBS.get(gene, [0.50, 0.35, 0.15])
    classes = CLASS_NAMES.get(gene, ["Severe", "Moderate", "Mild"])

    patients = []
    for i in range(40):
        r = rng.random()
        if r < probs[0]:
            pheno = classes[0]
        elif r < probs[0] + probs[1]:
            pheno = classes[1]
        else:
            pheno = classes[2]

        is_severe = (pheno == classes[0])
        is_mod    = (pheno == classes[1])
        is_mild   = (pheno == classes[2])

        # Age at diagnosis (years)
        if gene == "GCH1":
            if is_severe:
                # AD DRD: childhood onset, often delayed diagnosis
                age_dx = round(rng.uniform(1.0, 20.0), 1)
            else:
                # AR severe: neonatal HPA detected on NBS
                age_dx = round(rng.uniform(0.0, 0.3), 1)
        elif gene == "PTS":
            if is_severe:
                age_dx = round(rng.uniform(0.0, 0.3), 1)
            elif is_mod:
                age_dx = round(rng.uniform(0.0, 1.0), 1)
            else:
                age_dx = round(rng.uniform(0.0, 0.5), 1)
        elif gene == "QDPR":
            if is_severe:
                age_dx = round(rng.uniform(0.0, 0.3), 1)
            elif is_mod:
                age_dx = round(rng.uniform(0.1, 2.0), 1)
            else:
                age_dx = round(rng.uniform(0.3, 5.0), 1)
        elif gene == "SPR":
            # NBS normal; typically diagnosed later after neurological presentation
            if is_severe:
                age_dx = round(rng.uniform(0.5, 5.0), 1)
            elif is_mod:
                age_dx = round(rng.uniform(1.0, 10.0), 1)
            else:
                age_dx = round(rng.uniform(2.0, 15.0), 1)
        elif gene == "PCBD1":
            # Detected on NBS; HPA transient
            age_dx = round(rng.uniform(0.0, 0.2), 1)
        elif gene == "DNAJC12":
            if is_severe:
                age_dx = round(rng.uniform(0.0, 0.5), 1)
            elif is_mod:
                age_dx = round(rng.uniform(0.1, 1.0), 1)
            else:
                age_dx = round(rng.uniform(0.3, 5.0), 1)
        else:
            age_dx = round(rng.uniform(0.0, 10.0), 1)

        # Sex
        if gene == "GCH1" and is_severe:
            # AD DRD: female predominance 3:1
            sex = rng.choice(["F", "F", "F", "M"])
        else:
            sex = rng.choice(["M", "F"])

        # Gene-specific clinical fields
        if gene == "GCH1":
            is_ad_drd = is_severe  # In our model, AD DRD = "severe" class (most common presentation)
            phe_umol = round(rng.uniform(60, 120), 0) if is_ad_drd else round(rng.uniform(300, 1200), 0)
            bh4_response_pct = round(rng.uniform(30, 60), 0) if not is_ad_drd else round(rng.uniform(0, 20), 0)
            neopterin_low = True  # Both AD and AR: neopterin LOW
            nbs_detected = False if is_ad_drd else True  # AD DRD: Phe normal → NBS miss
            presenting_feature = (
                rng.choice(["Diurnal dystonia", "Gait abnormality", "Foot dystonia", "Parkinsonian features"])
                if is_ad_drd else
                rng.choice(["Elevated Phe on NBS", "Hypotonia", "Seizures", "Encephalopathy"])
            )
            outcome_class = (
                rng.choice(["Excellent L-DOPA response", "Good L-DOPA response", "Near-complete resolution"])
                if is_ad_drd else
                rng.choice(["Stable on BH4+NT", "Moderate neuro", "Severe encephalopathy"])
            )
            inheritance_type = "AD" if is_ad_drd else "AR"
            variant_1 = rng.choice(["p.Arg184His", "p.Arg249Gly", "p.Lys224Arg", "c.572+1G>A"])
            variant_2 = None if is_ad_drd else rng.choice(["p.Asp164Glu", "c.1A>G (start lost)", "Exon3del"])
            patients.append({
                "patient_id": f"{gene}-{i+1:03d}", "gene": gene, "sex": sex,
                "phenotypic_class": pheno, "age_dx_y": age_dx,
                "phe_umol_L": phe_umol, "bh4_response_pct": bh4_response_pct,
                "presenting_feature": presenting_feature,
                "inheritance": inheritance_type,
                "variant_1": variant_1, "variant_2": variant_2,
                "neopterin_low": neopterin_low,
                "nbs_detected": nbs_detected,
                "outcome_class": outcome_class,
                "diurnal_fluctuation": is_ad_drd,
                "l_dopa_responsive": is_ad_drd,
            })
        elif gene == "PTS":
            phe_umol = round(rng.uniform(400, 1500), 0) if is_severe else round(rng.uniform(200, 600), 0) if is_mod else round(rng.uniform(120, 350), 0)
            bh4_response_pct = round(rng.uniform(30, 80), 0)  # All PTS forms are BH4-responsive
            neopterin_very_high = True
            primapterin_urine = rng.random() < 0.95
            nbs_detected = True  # All PTS have elevated Phe
            presenting_feature = (
                rng.choice(["Elevated Phe on NBS", "Progressive encephalopathy", "Seizures", "Hypotonia + hypersalivation"])
                if is_severe else
                rng.choice(["Elevated Phe on NBS", "Mild developmental delay", "HPA on screening"])
            )
            outcome_class = (
                rng.choice(["Stable on BH4+L-DOPA+5-HTP", "Moderate neuro", "Severe if diet-only treated"])
                if is_severe else
                rng.choice(["Good on BH4 alone", "Excellent response", "Near-normal development"])
            )
            pts_form = "Classic (<30%)" if is_severe else "Mild (30-75%)" if is_mod else "Peripheral (>75%)"
            variant_1 = rng.choice(["p.Asn52Ser", "p.Tyr58His", "p.Pro87Leu", "p.Cys54Arg"])
            variant_2 = rng.choice(["p.Thr67Ile", "p.Phe72Ser", "p.Arg16Cys", "p.Ile114Val"])
            patients.append({
                "patient_id": f"{gene}-{i+1:03d}", "gene": gene, "sex": sex,
                "phenotypic_class": pheno, "age_dx_y": age_dx,
                "phe_umol_L": phe_umol, "bh4_response_pct": bh4_response_pct,
                "presenting_feature": presenting_feature,
                "inheritance": "AR",
                "variant_1": variant_1, "variant_2": variant_2,
                "neopterin_very_high": neopterin_very_high,
                "primapterin_in_urine": primapterin_urine,
                "nbs_detected": nbs_detected,
                "outcome_class": outcome_class,
                "pts_activity_form": pts_form,
                "neurotransmitter_supplement_needed": is_severe or is_mod,
            })
        elif gene == "QDPR":
            phe_umol = round(rng.uniform(400, 1400), 0) if is_severe else round(rng.uniform(200, 700), 0) if is_mod else round(rng.uniform(150, 400), 0)
            bh4_response_pct = round(rng.uniform(30, 75), 0)
            biopterin_high = True  # qBH2 accumulates → measured as biopterin
            folinic_acid_given = rng.random() < (0.92 if is_severe else 0.80 if is_mod else 0.65)
            dhpr_dbs_low = True  # DHPR activity <1% of normal
            nbs_detected = True
            presenting_feature = (
                rng.choice(["Elevated Phe on NBS", "Progressive encephalopathy", "White matter changes on MRI", "Seizures"])
                if is_severe else
                rng.choice(["Elevated Phe on NBS", "Developmental delay", "HPA on NBS"])
            )
            outcome_class = (
                rng.choice(["Stable on BH4+L-DOPA+5-HTP+folinic", "Moderate neuro despite treatment", "Progressive if folinic omitted"])
                if is_severe else
                rng.choice(["Good response", "Near-normal with complete treatment", "Mild cognitive delay"])
            )
            variant_1 = rng.choice(["p.Arg221Ter", "p.Arg102Gln", "p.Tyr150Cys", "c.68+1G>A"])
            variant_2 = rng.choice(["p.Pro27Leu", "p.His63Pro", "p.Gly63Ser", "p.Phe52Ser"])
            patients.append({
                "patient_id": f"{gene}-{i+1:03d}", "gene": gene, "sex": sex,
                "phenotypic_class": pheno, "age_dx_y": age_dx,
                "phe_umol_L": phe_umol, "bh4_response_pct": bh4_response_pct,
                "presenting_feature": presenting_feature,
                "inheritance": "AR",
                "variant_1": variant_1, "variant_2": variant_2,
                "biopterin_high": biopterin_high,
                "dhpr_dbs_activity_low": dhpr_dbs_low,
                "folinic_acid_prescribed": folinic_acid_given,
                "nbs_detected": nbs_detected,
                "outcome_class": outcome_class,
                "cerebral_folate_deficiency_risk": is_severe or is_mod,
            })
        elif gene == "SPR":
            # SPR: NORMAL Phe → nbs_detected always False
            phe_umol = round(rng.uniform(40, 100), 0)  # Normal range
            bh4_response_pct = 0.0  # BH4 loading test NOT applicable (Phe normal)
            csf_hva_low = rng.random() < (0.97 if is_severe else 0.85 if is_mod else 0.70)
            csf_5hiaa_low = rng.random() < (0.97 if is_severe else 0.85 if is_mod else 0.70)
            sepiapterin_csf = rng.random() < 0.80  # Not always tested but when done, positive
            nbs_detected = False  # ALWAYS False for SPR — NBS misses SPR
            presenting_feature = rng.choice([
                "Motor developmental delay", "Dystonia", "Oculomotor abnormalities",
                "Ataxia", "Cognitive regression", "Dopa-responsive motor disorder"
            ])
            outcome_class = (
                rng.choice(["Good on L-DOPA+5-HTP+folinic", "Moderate improvement", "Delayed diagnosis, residual neuro"])
                if is_severe else
                rng.choice(["Good response", "Near-normal development", "Mild residual ataxia"])
            )
            variant_1 = rng.choice(["p.Asn47Ser", "p.Arg150Cys", "p.Ser158Asn", "p.Leu209Phe"])
            variant_2 = rng.choice(["p.Asn47Ser", "Exon2-3del", "p.Thr61Ala", "p.Gly74Ser"])
            patients.append({
                "patient_id": f"{gene}-{i+1:03d}", "gene": gene, "sex": sex,
                "phenotypic_class": pheno, "age_dx_y": age_dx,
                "phe_umol_L": phe_umol, "bh4_response_pct": bh4_response_pct,
                "presenting_feature": presenting_feature,
                "inheritance": "AR",
                "variant_1": variant_1, "variant_2": variant_2,
                "nbs_detected": nbs_detected,
                "csf_hva_low": csf_hva_low,
                "csf_5hiaa_low": csf_5hiaa_low,
                "sepiapterin_csf_positive": sepiapterin_csf,
                "outcome_class": outcome_class,
                "sapropterin_used": False,  # NOT indicated for SPR
            })
        elif gene == "PCBD1":
            phe_umol = round(rng.uniform(120, 400), 0)  # Mild HPA
            bh4_response_pct = round(rng.uniform(30, 70), 0)
            primapterin_urine = rng.random() < 0.90  # Primapterin pathognomonic
            hpa_resolved_spontaneously = rng.random() < 0.85
            nbs_detected = True
            presenting_feature = rng.choice([
                "Mild HPA on NBS", "HPA on newborn screening", "Elevated Phe NBS",
                "Incidental HPA", "Screening-detected HPA"
            ])
            outcome_class = rng.choice([
                "Phe normalised spontaneously", "Transient benign HPA", "Resolved without treatment",
                "Short BH4 course then normalised", "Asymptomatic"
            ])
            variant_1 = rng.choice(["p.Trp58Ter", "p.Arg67Gln", "p.Arg61Trp", "c.88C>T"])
            variant_2 = rng.choice(["p.Ile76Thr", "p.Ser80Arg", "p.Gly28Asp", "c.161T>C"])
            patients.append({
                "patient_id": f"{gene}-{i+1:03d}", "gene": gene, "sex": sex,
                "phenotypic_class": pheno, "age_dx_y": age_dx,
                "phe_umol_L": phe_umol, "bh4_response_pct": bh4_response_pct,
                "presenting_feature": presenting_feature,
                "inheritance": "AR",
                "variant_1": variant_1, "variant_2": variant_2,
                "primapterin_in_urine": primapterin_urine,
                "hpa_resolved_spontaneously": hpa_resolved_spontaneously,
                "nbs_detected": nbs_detected,
                "outcome_class": outcome_class,
                "treatment_needed": not hpa_resolved_spontaneously,
                "neurotransmitter_deficiency": False,  # NOT in PCBD1 (benign form)
            })
        elif gene == "DNAJC12":
            phe_umol = round(rng.uniform(300, 1200), 0) if is_severe else round(rng.uniform(150, 600), 0) if is_mod else round(rng.uniform(100, 350), 0)
            bh4_response_pct = round(rng.uniform(30, 80), 0)  # BH4-loading test responsive
            pterin_profile_normal = True  # HALLMARK: normal pterins despite HPA
            csf_hva_low = rng.random() < (0.95 if is_severe else 0.80 if is_mod else 0.55)
            csf_5hiaa_low = rng.random() < (0.93 if is_severe else 0.75 if is_mod else 0.50)
            sapropterin_sufficient = rng.random() < 0.70  # Often sapropterin alone is enough
            nbs_detected = True
            presenting_feature = (
                rng.choice(["Elevated Phe on NBS + normal pterins", "HPA with BH4-responsive test", "Unexplained HPA normal pterins"])
                if is_severe else
                rng.choice(["HPA on NBS", "Mild HPA normal pterins", "BH4-responsive HPA"])
            )
            outcome_class = (
                rng.choice(["Controlled on sapropterin", "Good on sapropterin+L-DOPA", "Stable with complete regimen"])
                if is_severe else
                rng.choice(["Excellent sapropterin response", "Near-normal on BH4", "Good outcome"])
            )
            variant_1 = rng.choice(["p.Gln22Ter", "p.Arg95Trp", "p.Ile147Thr", "p.Gly64Ser"])
            variant_2 = rng.choice(["Exon3del", "p.Gln22Ter", "p.Thr89Ala", "c.234+2T>A"])
            patients.append({
                "patient_id": f"{gene}-{i+1:03d}", "gene": gene, "sex": sex,
                "phenotypic_class": pheno, "age_dx_y": age_dx,
                "phe_umol_L": phe_umol, "bh4_response_pct": bh4_response_pct,
                "presenting_feature": presenting_feature,
                "inheritance": "AR",
                "variant_1": variant_1, "variant_2": variant_2,
                "pterin_profile_normal": pterin_profile_normal,
                "csf_hva_low": csf_hva_low,
                "csf_5hiaa_low": csf_5hiaa_low,
                "sapropterin_sufficient_alone": sapropterin_sufficient,
                "nbs_detected": nbs_detected,
                "outcome_class": outcome_class,
            })
        else:
            patients.append({
                "patient_id": f"{gene}-{i+1:03d}", "gene": gene, "sex": sex,
                "phenotypic_class": pheno, "age_dx_y": age_dx,
                "nbs_detected": True,
            })
    return patients


# ── Populate patient cohorts ──────────────────────────────────────────────────────
for _g in BH4_GENES:
    _g["patients"] = _make_patients(_g)
    _g["n_patients"] = len(_g["patients"])

ALL_PATIENTS = [p for g in BH4_GENES for p in g["patients"]]


# ─── API: get_overview ───────────────────────────────────────────────────────────
def get_overview():
    total = len(ALL_PATIENTS)

    gene_summary = []
    for g in BH4_GENES:
        pts = g["patients"]
        gene_summary.append({
            "gene": g["gene"],
            "alias": g["alias"],
            "locus": g["locus"],
            "gene_class": g["gene_class"],
            "bh4_subgroup": g["bh4_subgroup"],
            "n_patients": g["n_patients"],
            "phenotype": g["phenotype"],
            "diet_treatment": g["diet_treatment"],
            "nbs_marker": g["nbs_marker"],
            "key_biomarker": g["key_biomarker"],
            "severity_spectrum": g["severity_spectrum"],
            "founder_variant": g["founder_variant"],
            "mean_age_dx_y": round(sum(p["age_dx_y"] for p in pts) / len(pts), 1),
        })

    # BH4-loading test responsive count
    n_bh4_responsive = sum(
        1 for p in ALL_PATIENTS
        if isinstance(p.get("bh4_response_pct"), (int, float)) and p.get("bh4_response_pct", 0) >= 30
    )

    # NBS miss count (SPR always False; GCH1 AD DRD also False)
    n_nbs_missed = sum(1 for p in ALL_PATIENTS if not p.get("nbs_detected", True))

    return {
        "atlas": "BH4-Atlas — Complete 6-Gene Tetrahydrobiopterin Disorders Atlas",
        "n_genes": len(BH4_GENES),
        "n_patients": total,
        "seeds": [g["seed"] for g in BH4_GENES],
        "genes_covered": [g["gene"] for g in BH4_GENES],
        "gene_subgroups": {
            "BH4 de novo synthesis (GCH1 · PTS · SPR)": ["GCH1", "PTS", "SPR"],
            "BH4 regeneration auxiliary (PCBD1 · QDPR)": ["PCBD1", "QDPR"],
            "BH4 cofactor utilisation / chaperone (DNAJC12)": ["DNAJC12"],
        },
        "n_bh4_loading_responsive": n_bh4_responsive,
        "n_nbs_missed": n_nbs_missed,
        "critical_clinical_rules": [
            "ALL BH4 DEFICIENCIES (except GCH1-AD-DRD and SPR) MIMIC PKU ON NBS: elevated Phe triggers PKU pathway — BUT phenylalanine-restricted diet alone is INSUFFICIENT; neurotransmitter supplementation MANDATORY in classic forms (L-DOPA/Carbidopa + 5-HTP); failing to add neurotransmitters → progressive neurological deterioration despite normal Phe on diet",
            "SPR (Sepiapterin Reductase) — NORMAL PHE ON NBS: peripheral bypass (AKR1C3) provides adequate liver BH4 → Phe NORMAL → NBS MISSES SPR; any unexplained childhood motor disorder / dystonia / ataxia with NORMAL NBS must have CSF neurotransmitter analysis (HVA + 5-HIAA); sapropterin NOT indicated (add L-DOPA + 5-HTP + folinic acid)",
            "QDPR — FOLINIC ACID IS NON-NEGOTIABLE: qBH2 accumulates and inhibits DHFR → cerebral folate deficiency → white matter changes + cognitive decline INDEPENDENT of BH4/NT control; must give 5-formyl-THF (folinic acid/leucovorin) 15-20 mg/day; folic acid (folate) does NOT work (DHFR blocked); failure to give folinic acid = incomplete treatment; check CSF folate",
            "GCH1 AD (DRD/Segawa) — L-DOPA AT 1-2 mg/kg/day: childhood-onset diurnal dystonia worse in evening + improves after sleep is PATHOGNOMONIC; L-DOPA response is exquisite at very low doses (1-2 mg/kg/day = 5-10× lower than Parkinson disease doses); all unexplained childhood dystonia deserves L-DOPA trial; misdiagnosed as cerebral palsy for years is common",
            "BH4 LOADING TEST (20 mg/kg BH4): Phe drops >30% within 4-8h = BH4-responsive HPA (PTS / QDPR / PCBD1 / DNAJC12 / GCH1-AR); does NOT respond = classic PAH-PKU or PAH-non-responsive; MANDATORY in all NBS-detected HPA before labelling as classic PKU",
            "PTERIN PANEL (urine pterins + CSF neurotransmitters) MANDATORY FOR ALL HPA: neopterin VERY HIGH → PTS; neopterin LOW, biopterin LOW → GCH1-AR; biopterin HIGH, neopterin normal → QDPR; all normal + BH4 responsive → DNAJC12; primapterin in urine → PTS or PCBD1; determines which gene, which form, which treatment",
            "PTS PRIMAPTERIN (7-BIOPTERIN) PATHOGNOMONIC: absent in normal urine; present in PTS and PCBD1; neopterin:biopterin ratio >5:1 = PTS hallmark; do not confuse PTS (severe, needs full treatment) with PCBD1 (benign, self-resolving); neopterin level distinguishes: PTS = very high; PCBD1 = normal",
            "DNAJC12 — NORMAL PTERIN PROFILE IS MISLEADING: BH4-responsive HPA + normal pterins = DNAJC12 until proven otherwise; do NOT label as BH4-responsive PAH-PKU without CSF neurotransmitters (CSF HVA and 5-HIAA are LOW in DNAJC12 but NORMAL in BH4-responsive PAH-PKU); sapropterin often controls both Phe and neurotransmitter synthesis",
            "PCBD1 — TRANSIENT AND BENIGN, AVOID OVER-TREATMENT: HPA resolves spontaneously within 6-24 months; do NOT prescribe L-DOPA or 5-HTP (CSF neurotransmitters are normal); short BH4 course if Phe very high initially; dual function as HNF1 cofactor means PCBD1 mutations can also cause MODY10 (adult diabetes) — different clinical context",
            "BH4 NEUROTRANSMITTER PATHWAY: BH4 is cofactor for TH (tyrosine→L-DOPA; start of dopamine pathway) and TPH1/2 (tryptophan→5-HTP; start of serotonin pathway); BH4 deficiency → brain TH and TPH failure → dopamine + serotonin deficiency → progressive movement disorder + cognitive regression even when Phe is normal on diet; neurotransmitter supplementation (L-DOPA + 5-HTP) bypasses TH/TPH blocks directly",
        ],
        "gene_summary": gene_summary,
        "nbs_note": "GCH1-AD-DRD and SPR have NORMAL Phe on NBS — missed by standard screening. PTS (60% of BH4 HPA), QDPR (~20%), GCH1-AR, PCBD1, DNAJC12 have elevated Phe and are detected by NBS Phe screen — but MUST be distinguished from PAH-PKU by pterin panel and BH4 loading test.",
    }


# ─── API: get_breakdown ──────────────────────────────────────────────────────────
def get_breakdown():
    gene_rows = []
    for g in BH4_GENES:
        pts = g["patients"]
        n_nbs_miss = sum(1 for p in pts if not p.get("nbs_detected", True))
        n_bh4_resp = sum(
            1 for p in pts
            if isinstance(p.get("bh4_response_pct"), (int, float)) and p.get("bh4_response_pct", 0) >= 30
        )
        gene_rows.append({
            "gene": g["gene"],
            "alias": g["alias"],
            "aa": g["aa"],
            "kDa": g["kDa"],
            "locus": g["locus"],
            "omim_gene": g["omim_gene"],
            "gene_class": g["gene_class"],
            "bh4_subgroup": g["bh4_subgroup"],
            "n_patients": g["n_patients"],
            "seed": g["seed"],
            "phenotype": g["phenotype"],
            "inheritance": g["inheritance"],
            "hallmark": g["hallmark"],
            "key_ddx": g["key_ddx"],
            "diet_treatment": g["diet_treatment"],
            "gene_therapy_status": g["gene_therapy_status"],
            "critical_ci": g["critical_ci"],
            "nbs_marker": g["nbs_marker"],
            "key_biomarker": g["key_biomarker"],
            "severity_spectrum": g["severity_spectrum"],
            "founder_variant": g["founder_variant"],
            "key_variants": g["key_variants"],
            "mean_age_dx_y": round(sum(p["age_dx_y"] for p in pts) / len(pts), 1),
            "n_nbs_missed": n_nbs_miss,
            "n_bh4_loading_responsive": n_bh4_resp,
        })
    return {
        "genes": gene_rows,
        "total": len(BH4_GENES),
        "total_patients": len(ALL_PATIENTS),
    }


# ─── API: get_definitions ─────────────────────────────────────────────────────
def get_definitions():
    return {
        "atlas": "BH4-Atlas — Complete 6-Gene Tetrahydrobiopterin Disorders Atlas",
        "bh4_overview": {
            "full_name": "Tetrahydrobiopterin (BH4) Disorders — inherited defects in synthesis, regeneration, or cofactor utilisation of BH4, the essential cofactor for aromatic amino acid hydroxylases (PAH, TH, TPH) and nitric oxide synthases (NOS)",
            "genes_in_atlas": 6,
            "collective_incidence": "PTS: ~1/500,000 (most common BH4 HPA, ~60%); QDPR: ~1/500,000 (~20% of BH4 HPA); GCH1-AD-DRD: ~1/500,000; GCH1-AR: ~1/1,000,000; SPR: ~1/500,000+; PCBD1: ~1/500,000-1,000,000; DNAJC12: ~1/1,000,000+",
            "nbs_note": "GCH1-AD-DRD and SPR have NORMAL Phe — MISSED by standard NBS. PTS, QDPR, GCH1-AR, PCBD1, DNAJC12 have elevated Phe and are NBS-detected but must be differentiated from PAH-PKU by pterin panel.",
        },
        "definitions": [
            {
                "term": "BH4 Biosynthesis Pathway — 3 Enzymatic Steps",
                "definition": "BH4 (tetrahydrobiopterin, sapropterin) is synthesised de novo from GTP in three steps: Step 1 (GCH1/GTPCH-I): GTP → 7,8-dihydroneopterin-3'-triphosphate (H2NTP). This is the rate-limiting step. GCH1 is a homodecamer; AD loss causes DRD (haploinsufficiency of the decamer); AR loss causes severe BH4 deficiency with HPA. Step 2 (PTS/6-PTPS): H2NTP → 6-pyruvoyltetrahydropterin (6-PTP). PTS deficiency is the most common BH4 disorder (~60% of BH4 HPA); neopterin accumulates (very high), biopterin is low; primapterin is produced as a shunt product. Step 3 (SPR): 6-PTP → 6-lactoyl-BH4 → BH4 (two SPR-catalysed reductions). SPR is absent in the brain but AKR1C3 provides peripheral bypass → Phe normal (NBS misses SPR) but brain BH4 deficient → neurotransmitter deficiency despite normal Phe.",
            },
            {
                "term": "BH4 Regeneration — PCBD1 and QDPR (DHPR)",
                "definition": "During each catalytic cycle of PAH/TH/TPH, BH4 is oxidised to 4α-hydroxy-BH4 (pterin-4α-carbinolamine). Two enzymes regenerate BH4: PCBD1 (PCD): dehydrates 4α-carbinolamine → quinonoid BH2 (qBH2). Without PCD: carbinolamine shunts to 7-biopterin (primapterin) — explains primapterin in urine in PCBD1 deficiency. This step is largely a salvage that feeds into QDPR. QDPR (DHPR): reduces qBH2 + NADH → BH4 + NAD+. This is the critical regeneration step. Without QDPR: qBH2 accumulates massively → BH4 depleted (even if synthesis is intact) → all BH4-dependent enzymes fail. qBH2 is also a potent DHFR inhibitor → cerebral folate deficiency → folinic acid MANDATORY.",
            },
            {
                "term": "Neurotransmitter Pathway — BH4 as Cofactor for TH and TPH",
                "definition": "BH4 is the obligate cofactor (not substrate — BH4 is regenerated after each cycle) for three aromatic amino acid hydroxylases: (1) PAH (phenylalanine hydroxylase): Phe → Tyr. BH4 deficiency → hyperphenylalaninemia (HPA). (2) TH (tyrosine hydroxylase): Tyr → L-DOPA → dopamine → noradrenaline → adrenaline. BH4 deficiency → L-DOPA (and downstream) deficiency → dopaminergic neurotransmitter deficiency → movement disorder. (3) TPH1/2 (tryptophan hydroxylase 1/2): Trp → 5-HTP → serotonin. BH4 deficiency → 5-HTP (and serotonin) deficiency → serotonin deficiency. Treatment bypasses: L-DOPA bypasses TH (crosses BBB; dopamine does not). 5-HTP bypasses TPH (crosses BBB; serotonin does not). Carbidopa inhibits peripheral decarboxylases — allows more L-DOPA and 5-HTP to reach brain.",
            },
            {
                "term": "BH4 Loading Test — 20 mg/kg Oral BH4",
                "definition": "The BH4 loading test is the central diagnostic investigation for any NBS-detected HPA: give 20 mg/kg sapropterin (or natural BH4) orally, then measure plasma Phe at 4h and 8h. Interpretation: >30% Phe reduction = BH4-responsive HPA. This response occurs in: PTS (most common), QDPR, PCBD1, DNAJC12, GCH1-AR, and BH4-responsive PAH-PKU (mild/moderate PAH deficiency where BH4 stabilises residual PAH). Non-response = classic PAH-PKU or severe PAH deficiency (no residual enzyme to stabilise). Critical: BH4-responsive HPA ≠ BH4-responsive PAH-PKU. Must distinguish using: pterin panel (differentiates PTS/QDPR/GCH1-AR/PCBD1 from PAH-responsive) AND CSF neurotransmitters (low in BH4 gene defects including DNAJC12; normal in BH4-responsive PAH-PKU). DNAJC12 has normal pterins but abnormal CSF NT — only way to distinguish from PAH-responsive.",
            },
            {
                "term": "Urine Pterin Panel — Neopterin, Biopterin, Primapterin",
                "definition": "Urine pterins by HPLC are the cornerstone of BH4 disorder subtyping. Each gene has a specific pterin signature: GCH1 (all forms): Neopterin LOW + Biopterin LOW (total pterins low; step 1 blocked). PTS: Neopterin VERY HIGH + Biopterin LOW (neopterin:biopterin ratio >5:1; step 2 blocked; primapterin in urine PATHOGNOMONIC). QDPR: Biopterin HIGH + Neopterin NORMAL (qBH2 accumulates → measured as biopterin; regeneration blocked). PCBD1: Primapterin in urine + Neopterin normal (shunt to 7-biopterin; step between PCBD1 and QDPR). SPR: Pterins MAY appear normal in urine (peripheral bypass compensates); CSF biopterin elevated (sepiapterin → biopterin conversion in CSF). DNAJC12: ALL PTERINS NORMAL — this is the diagnostic trap; normal urine pterins in BH4-responsive HPA = DNAJC12 until proven otherwise.",
            },
            {
                "term": "CSF Neurotransmitter Analysis — HVA and 5-HIAA in BH4 Disorders",
                "definition": "CSF neurotransmitter metabolites are mandatory for diagnosis and monitoring of all BH4 deficiencies (except PCBD1 benign): HVA (homovanillic acid): final dopamine metabolite; low HVA = dopaminergic deficiency (TH impaired). 5-HIAA (5-hydroxyindoleacetic acid): final serotonin metabolite; low 5-HIAA = serotonergic deficiency (TPH impaired). Expected CSF findings by gene: PTS classic, QDPR, SPR: HVA LOW + 5-HIAA LOW. GCH1 (both AD and AR): HVA LOW; 5-HIAA may be low. DNAJC12: HVA LOW + 5-HIAA LOW despite normal pterins (this finding distinguishes from BH4-responsive PAH-PKU where CSF NT are NORMAL). PCBD1: CSF NT NORMAL (benign). SPR: HVA LOW + 5-HIAA LOW + biopterin elevated + sepiapterin in CSF (pathognomonic). CSF must be taken via lumbar puncture BEFORE starting treatment, then used to guide and monitor therapy.",
            },
            {
                "term": "QDPR/DHPR Cerebral Folate Deficiency — Mechanism and Treatment",
                "definition": "In QDPR deficiency, qBH2 accumulates massively. qBH2 (quinonoid dihydrobiopterin) is a potent competitive inhibitor of DHFR (dihydrofolate reductase; Ki ~1 μM). DHFR normally converts dietary folate (DHF) → THF (tetrahydrofolate), the active form for 1-carbon transfer reactions and CNS methylation. qBH2 inhibition of DHFR → brain folate transport and utilisation blocked → CEREBRAL FOLATE DEFICIENCY. Consequences: impaired myelin synthesis, impaired S-adenosylmethionine (SAM)-dependent methylation → white matter abnormalities on MRI + progressive cognitive decline. This occurs INDEPENDENT of BH4 and neurotransmitter treatment. Treatment: FOLINIC ACID (5-formyl-THF / leucovorin) at 15-20 mg/day. Folinic acid (5-formyl-THF) bypasses the DHFR block by entering the folate cycle as a pre-reduced THF. FOLIC ACID (folate) does NOT work because it requires DHFR for activation → DHFR is blocked by qBH2 → folic acid useless here. Monitor CSF folate on treatment.",
            },
            {
                "term": "Dopa-Responsive Dystonia (DRD / Segawa Syndrome) — GCH1 AD",
                "definition": "DRD (OMIM #128230) is caused by GCH1 haploinsufficiency (AD). The GTPCH-I enzyme is a homodecamer (10 identical subunits). A single mutant subunit (dominant-negative) destabilises the complex → ~50% reduction in GTPCH-I activity → partial BH4 deficiency. In the brain, dopaminergic neurons of the striatum (caudate, putamen) are selectively vulnerable because they have the highest TH activity and therefore the highest BH4 demand. Partial BH4 → partial TH impairment → reduced dopamine synthesis in striatum → dystonia. Clinical features: onset age 1-20y (peak 6-12y); lower limb dystonia → progressive to upper limbs; DIURNAL FLUCTUATION (worse evening/night, better after sleep) PATHOGNOMONIC — due to BH4 pool depletion during day from cumulative TH usage; females affected 3:1 (sex-specific penetrance — oestrogen interacts with GCH1 expression). Treatment: Carbidopa/L-DOPA 1-2 mg/kg/day — EXQUISITE response within 1-4 weeks; much lower dose than idiopathic PD; misdiagnosed as cerebral palsy for average 5-10 years before correct diagnosis. Prognosis: excellent with treatment.",
            },
            {
                "term": "SPR Deficiency — Peripheral Bypass and the NBS Diagnostic Gap",
                "definition": "Sepiapterin reductase (SPR) deficiency illustrates how the tissue-specific distribution of bypass enzymes determines clinical phenotype. SPR catalyses the final two steps of BH4 synthesis from 6-PTP. In peripheral tissues (liver, erythrocytes), an alternative aldo-keto reductase (AKR1C3) can reduce 6-PTP → BH4, bypassing the SPR deficiency. In the brain, AKR1C3 activity is negligible → no bypass → severe BH4 deficiency in neurons → TH and TPH fail → profound dopamine + serotonin deficiency. Since liver PAH has adequate BH4 (via bypass), plasma Phe is NORMAL. NBS Phe screen is NORMAL → SPR not detected. Urine pterins may appear near-normal (peripheral bypass functions). The only way to diagnose: CSF analysis showing LOW HVA + LOW 5-HIAA + elevated biopterin (sepiapterin → biopterin in CSF) + sepiapterin in CSF (if measured). Sapropterin is NOT indicated (peripheral bypass already provides BH4; adding sapropterin adds no benefit for liver PAH and does not easily cross BBB in brain). Correct treatment: L-DOPA + 5-HTP + folinic acid — bypasses TH/TPH blocks at the substrate level.",
            },
            {
                "term": "DNAJC12 — Pharmacological Chaperone Mechanism of Sapropterin",
                "definition": "DNAJC12 (DnaJ/HSP40 family) is a co-chaperone of the HSP70 system. It specifically assists folding and stabilisation of aromatic amino acid hydroxylases (PAH, TH, TPH). Without DNAJC12, all three hydroxylases are unstable — they misfold, aggregate, and are degraded. This happens DESPITE normal BH4 (the cofactor) — so pterin levels are normal. The BH4 loading test is RESPONSIVE by a different mechanism: pharmacological doses of BH4 (sapropterin, 5-20 mg/kg) can directly bind to the enzyme active sites (as a pharmacological chaperone), stabilising the misfolded PAH/TH/TPH independent of the HSP40/HSP70 chaperone system — this is the same mechanism that underlies BH4-responsive PAH-PKU but here applied to multiple enzymes. Sapropterin thus acts as both a cofactor supplement AND a pharmacological chaperone. Treatment with sapropterin often normalises Phe AND improves neurotransmitter synthesis (TH/TPH also stabilised by sapropterin). If CSF HVA/5-HIAA remain low, add L-DOPA/Carbidopa + 5-HTP.",
            },
        ],
    }


if __name__ == "__main__":
    import json
    print("=== BH4 Atlas — Functional Test ===")
    ov = get_overview()
    print(f"Genes: {ov['n_genes']}, Patients: {ov['n_patients']}, Seeds: {ov['seeds']}")
    print(f"Subgroups: {list(ov['gene_subgroups'].keys())}")
    print(f"NBS missed: {ov['n_nbs_missed']}, BH4-loading responsive: {ov['n_bh4_loading_responsive']}")
    bd = get_breakdown()
    print(f"Breakdown genes: {len(bd['genes'])}")
    df = get_definitions()
    print(f"Definitions: {len(df['definitions'])}")
    print("OK")
