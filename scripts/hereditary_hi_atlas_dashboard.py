#!/usr/bin/env python3
"""Hereditary-Hyperinsulinism-Atlas — Complete 8-Gene Hereditary Hyperinsulinism Atlas
ABCC8   (SUR1; 1581 aa; 11p15.1; AR/AD;
         Congenital Hyperinsulinism type 1 — most common CHI 40-50%;
         KATP channel; focal vs diffuse; 18F-DOPA PET mandatory;
         diazoxide-UNRESPONSIVE; octreotide → pancreatectomy; seed SEED_BASE+0) ·
KCNJ11  (Kir6.2; 390 aa; 11p15.1; AR/AD;
         CHI type 2 + neonatal diabetes mellitus (NDM); same KATP channel subunit;
         activating = CHI; deactivating = NDM; sulfonylurea rescues NDM; seed SEED_BASE+1) ·
GLUD1   (glutamate dehydrogenase; 558 aa; 10q23.33; AD GOF;
         Hyperinsulinism-Hyperammonaemia (HI/HA) syndrome;
         leucine-sensitive; ammonia elevated; protein-induced hypoglycaemia;
         diazoxide-RESPONSIVE; protein restriction + diazoxide; seed SEED_BASE+2) ·
GCK     (glucokinase; 465 aa; 7p13; AD GOF activating;
         Activating GCK-HI — persistent mild hypoglycaemia;
         glucose-sensor set-point lowered; diazoxide-RESPONSIVE;
         NOT the same as GCK-MODY2 (LOF); seed SEED_BASE+3) ·
HADH    (3-hydroxyacyl-CoA dehydrogenase; 314 aa; 4q25; AR;
         Protein-sensitive HI — SCHAD deficiency;
         protein-induced hypoglycaemia; 3-OH-glutaric acid PATHOGNOMONIC;
         HADH-GDH interaction disrupted; diazoxide-responsive; seed SEED_BASE+4) ·
SLC16A1 (MCT1 monocarboxylate transporter 1; 494 aa; 1p13.2; AD GOF promoter;
         Exercise-induced HI (EIHI) — ectopic MCT1 in beta-cell;
         pyruvate enters during exercise → K-ATP closes → insulin surge;
         hypoglycaemia 30-60 min post-exercise; diazoxide-INEFFECTIVE; seed SEED_BASE+5) ·
HNF4A   (hepatocyte nuclear factor 4-alpha; 455 aa; 20q13.12; AD;
         Neonatal HI → MODY1 in adolescence; macrosomia;
         spontaneous recovery possible in infancy → monitor glucose;
         diazoxide-RESPONSIVE; anticipatory guidance for MODY1; seed SEED_BASE+6) ·
FOXA2   (forkhead box A2 / HNF3β; 1161 aa; 20p11.21; AD;
         Neonatal HI + Hypopituitarism triad (GH/ACTH/TSH deficiency);
         ABSENT glucagon counter-regulation — PATHOGNOMONIC;
         agenesis of anterior pituitary + pulmonary sequestration;
         pituitary MRI mandatory; seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1774–1781)
"""

import random

SEED_BASE = 1774

HI_GENES = [
    # ── ABCC8 — Congenital Hyperinsulinism type 1 (KATP-CHI) ─────────────────
    {
        "gene": "ABCC8",
        "protein": (
            "ABCC8 — 11p15.1 AR/AD — SUR1-1581aa — "
            "Congenital-Hyperinsulinism-CHI1-Most-Common-40-50pct-CHI — "
            "KATP-Channel-SUR1-Regulatory-Subunit — "
            "Focal-vs-Diffuse-18F-DOPA-PET-CT-MANDATORY — "
            "Diazoxide-UNRESPONSIVE — Octreotide → Focal-Pancreatectomy-Curative / "
            "Diffuse-Near-Total-95pct-Pancreatectomy"
        ),
        "alias": (
            "ABCC8 (ATP-binding cassette subfamily C member 8; SUR1 sulphonylurea receptor 1); "
            "OMIM gene 600509; Congenital Hyperinsulinism type 1 (CHI1; diazoxide-unresponsive) OMIM 256450. "
            "11p15.1; 1581 aa; ~177 kDa; autosomal recessive (biallelic = diffuse; paternally inherited + somatic LOH = focal) "
            "or autosomal dominant (gain-of-function in some cases). "
            "FUNCTION: SUR1 is the regulatory subunit of the pancreatic beta-cell ATP-sensitive potassium (K-ATP) channel. "
            "The K-ATP channel is a hetero-octamer: (SUR1 × 4) + (Kir6.2 × 4). "
            "Under fasting (low ATP/ADP ratio): K-ATP opens → K+ efflux → hyperpolarisation → "
            "voltage-gated Ca2+ channels closed → no insulin secretion. "
            "After glucose meal (high ATP): K-ATP closes → depolarisation → Ca2+ influx → "
            "exocytosis → insulin secretion. "
            "ABCC8 loss-of-function → K-ATP cannot open even during fasting → "
            "membrane remains depolarised → unregulated Ca2+ influx → "
            "uncontrolled insulin secretion regardless of blood glucose (hypoglycaemia). "
            "FOCAL vs DIFFUSE DISTINCTION — CRITICAL: "
            "Diffuse CHI: biallelic (compound heterozygous or homozygous) ABCC8/KCNJ11 mutations — "
            "all beta-cells throughout the pancreas secrete unregulated insulin; "
            "Focal CHI: paternally inherited heterozygous ABCC8/KCNJ11 mutation + "
            "somatic loss of heterozygosity (LOH) of maternal 11p15 imprinting region in a discrete focus of beta-cells; "
            "focal CHI is CURABLE by limited focal pancreatectomy — surgical cure rate 95%+; "
            "18F-DOPA PET-CT pre-operatively to localise the focus is MANDATORY before surgery; "
            "Diffuse CHI: near-total (95-98%) pancreatectomy required — risk of subsequent diabetes. "
            "CLINICAL FEATURES: "
            "Neonatal hypoglycaemia: profound, often plasma glucose <1.5 mmol/L; "
            "Large for gestational age (LGA): foetal hyperinsulinism → macrosomia; "
            "Hypoglycaemia refractory to glucose infusion rates (GIR) >8 mg/kg/min; "
            "Seizures, jitteriness, poor feeding, lethargy; "
            "Hypoketotic hypoglycaemia: insulin suppresses lipolysis and ketogenesis → "
            "inappropriately low/absent ketones despite severe hypoglycaemia. "
            "DIAGNOSTIC BIOCHEMISTRY: "
            "Critical sample at time of hypoglycaemia (<3.0 mmol/L): "
            "insulin detectable (even if low — inappropriate if glucose <3 mmol/L); "
            "C-peptide elevated (endogenous insulin); "
            "Ketones (plasma 3-OHB) low/absent; "
            "Free fatty acids low; "
            "IGFBP1 low (insulin suppresses IGFBP1); "
            "Glucagon stimulation test: glucose response >1.7 mmol/L after 1 mg IV glucagon confirms hyperinsulinism; "
            "Leucine tolerance test: leucine-sensitive in ABCC8 (less specific than GLUD1); "
            "GENETIC TESTING: "
            "ABCC8 + KCNJ11 sequenced together (same chromosome, same channel); "
            "Paternal vs maternal origin of mutation — determines focal vs diffuse risk; "
            "If paternally inherited mutation: 18F-DOPA PET mandatory to detect focal lesion. "
            "TREATMENT: "
            "Diazoxide: K-ATP opener — INEFFECTIVE in ABCC8 biallelic loss-of-function "
            "(channel cannot respond to diazoxide — structural loss of SUR1 binding site); "
            "Octreotide (somatostatin analogue): reduces insulin secretion via SSTR2; "
            "second-line when diazoxide fails; continuous SC infusion or long-acting formulation; "
            "Glucagon: emergency (1 mg IM/IV) for acute hypoglycaemia; "
            "High GIR (glucose infusion rate): maintain euglycaemia while awaiting surgery; "
            "Surgery: 18F-DOPA PET-CT → focal resection (curative) or near-total pancreatectomy (diffuse)."
        ),
        "locus": "11p15.1",
        "aa": 1581,
        "kDa": 177,
        "omim_gene": "600509",
        "omim_disease": "256450",
        "inheritance": "AR (biallelic=diffuse) / paternally inherited het + somatic LOH = focal",
        "gene_class": "KATP channel sulphonylurea receptor — K-ATP regulatory subunit",
        "key_alerts": [
            "ABCC8-18F-DOPA-PET-MANDATORY: All diazoxide-unresponsive CHI MUST have 18F-DOPA PET-CT before pancreatectomy to distinguish focal (curative partial resection) from diffuse (near-total pancreatectomy); 18F-DOPA PET has 80-85% sensitivity for focal CHI localisation",
            "ABCC8-DIAZOXIDE-UNRESPONSIVE: Biallelic ABCC8/KCNJ11 mutations — K-ATP channel structurally absent/non-functional; diazoxide has no target; failure to open the channel; trial of diazoxide is diagnostic (failure = supports KATP-CHI)",
            "ABCC8-FOCAL-CURATIVE: Paternally inherited ABCC8/KCNJ11 + somatic 11p15 LOH = focal CHI; limited focal pancreatectomy is curative in 95%+ without risk of diabetes; parental mutation origin must always be determined",
            "ABCC8-HYPOKETOTIC-HYPOGLYCAEMIA: Insulin suppresses both lipolysis and ketogenesis; absent/low ketones at the time of hypoglycaemia is the hallmark of hyperinsulinism — distinguishes from fatty acid oxidation disorders (high ketones) and GH/cortisol deficiency",
        ],
        "etiologies": [
            {"variant": "p.Arg1420Cys (c.4258C>T)", "type": "missense", "frequency": "common European", "severity": "severe — diffuse"},
            {"variant": "p.Glu1506Lys (c.4516G>A)", "type": "missense", "frequency": "common", "severity": "severe"},
            {"variant": "p.Gly1479Arg (c.4435G>A)", "type": "missense", "frequency": "moderate", "severity": "severe"},
            {"variant": "Large exon deletion", "type": "deletion/rearrangement", "frequency": "10-15% CHI", "severity": "severe diffuse"},
            {"variant": "Frameshift mutations", "type": "loss-of-function", "frequency": "variable", "severity": "severe"},
        ],
        "stats": {
            "proportion_chi": "40-50% of all CHI",
            "diazoxide_response": "Unresponsive (biallelic LOF)",
            "focal_risk": "~50% of ABCC8-CHI cases",
            "surgical_cure_focal": "95%+",
            "diabetes_risk_diffuse_pancreatectomy": "~80% by 20y",
        },
        "dx_delay_distribution": {
            "neonatal_0_7d": 60,
            "neonatal_8_28d": 30,
            "infant_1_6m": 8,
            "late_6m_plus": 2,
        },
    },

    # ── KCNJ11 — CHI type 2 + Neonatal Diabetes Mellitus ──────────────────────
    {
        "gene": "KCNJ11",
        "protein": (
            "KCNJ11 — 11p15.1 AR/AD — Kir6.2-390aa — "
            "Congenital-Hyperinsulinism-CHI2 (AR-LOF) + "
            "Neonatal-Diabetes-NDM (AD-GOF-Activating) — "
            "KATP-Channel-Kir6.2-Pore-Subunit — "
            "NDM: Sulphonylurea-RESCUES-90pct — "
            "CHI: Same-Management-as-ABCC8"
        ),
        "alias": (
            "KCNJ11 (potassium inwardly rectifying channel subfamily J member 11; Kir6.2); "
            "OMIM gene 600937; CHI2 OMIM 601820; Permanent Neonatal Diabetes (PNDM-KCNJ11) OMIM 606176. "
            "11p15.1; 390 aa; ~43 kDa; autosomal recessive (CHI) or autosomal dominant (neonatal DM). "
            "FUNCTION: Kir6.2 is the pore-forming subunit of the K-ATP channel — it forms the actual K+ conducting pore. "
            "Together with SUR1 (ABCC8), it forms the (Kir6.2)4(SUR1)4 hetero-octamer. "
            "DUAL PHENOTYPES — SAME GENE, OPPOSITE MUTATIONS: "
            "LOSS-OF-FUNCTION (biallelic recessive) → K-ATP channel absent/non-functional → "
            "beta-cell membrane perpetually depolarised → uncontrolled insulin secretion → CHI; "
            "GAIN-OF-FUNCTION (heterozygous dominant-activating) → K-ATP channel hyperactivated → "
            "membrane hyperpolarised even after glucose → insulin secretion impossible → neonatal diabetes. "
            "NEONATAL DIABETES (KCNJ11-NDM): "
            "Onset: <6 months of age (by definition of neonatal DM); "
            "Most common cause of permanent neonatal diabetes mellitus (PNDM); "
            "Phenotype spectrum: "
            "Isolated NDM (most common); "
            "DEND syndrome: Developmental delay + Epilepsy + Neonatal Diabetes — severe mutations; "
            "intermediate DEND (iDEND): developmental delay without severe epilepsy; "
            "KEY TREATMENT: Sulphonylureas (glibenclamide/glyburide, tolbutamide) BYPASS the K-ATP channel — "
            "they bind SUR1 directly and close the channel independent of ATP → "
            "restore insulin secretion; sulphonylurea switches >90% of PNDM-KCNJ11 off insulin injections; "
            "neurological improvement also occurs (channel in neurons too); "
            "transition from insulin to sulphonylurea: critical timing — earlier is better for neurology; "
            "CHI (KCNJ11-LOF): "
            "Same clinical picture as ABCC8-CHI; "
            "focal vs diffuse distinction equally important; "
            "18F-DOPA PET if diazoxide-unresponsive and paternally inherited mutation. "
            "CLINICAL CLUE — NDM vs CHI: "
            "Infant with severe hypoglycaemia in first weeks: CHI (KCNJ11-LOF); "
            "Infant with diabetes onset <6 months: NDM (KCNJ11-GOF); "
            "NDM misdiagnosed as type 1 DM historically — critical because sulphonylurea CURES NDM; "
            "all neonatal DM MUST have KCNJ11 + ABCC8 sequencing before committing to insulin therapy."
        ),
        "locus": "11p15.1",
        "aa": 390,
        "kDa": 43,
        "omim_gene": "600937",
        "omim_disease": "601820",
        "inheritance": "AR (CHI-LOF) / AD (NDM-GOF activating)",
        "gene_class": "KATP channel pore-forming subunit — K+ inwardly-rectifying",
        "key_alerts": [
            "KCNJ11-NDM-SULPHONYLUREA-RESCUES: ALL neonatal DM (<6 months onset) must have KCNJ11 sequencing; if GOF mutation confirmed → sulphonylurea trial (glibenclamide 0.05-0.5 mg/kg/day) rescues 90%+ off insulin; neurological benefit in DEND syndrome also documented — start early",
            "KCNJ11-NDM-MISDIAGNOSED-T1DM: Neonatal DM onset <6 months is genetic UNTIL PROVEN OTHERWISE — type 1 DM almost never presents before 6 months; C-peptide may be detectable in NDM (unlike T1DM); GAD/IA2 antibodies NEGATIVE in genetic NDM",
            "KCNJ11-DEND-SYNDROME: Severe KCNJ11 GOF mutations cause DEND — Developmental delay + Epilepsy + Neonatal Diabetes; K-ATP channels also in neurons and muscle; sulphonylurea may improve neurological features if started early — irreversible if delayed",
            "KCNJ11-SAME-KATP-ABCC8: KCNJ11 and ABCC8 always sequenced TOGETHER as they encode the two subunits of the same K-ATP channel; mutations in either cause identical CHI phenotype; 11p15.1 is the chromosomal region for both genes",
        ],
        "etiologies": [
            {"variant": "p.Arg201His (c.602G>A)", "type": "GOF missense", "frequency": "most common NDM", "severity": "NDM ± iDEND"},
            {"variant": "p.Arg201Cys (c.601C>T)", "type": "GOF missense", "frequency": "common NDM", "severity": "NDM"},
            {"variant": "p.Val59Met (c.175G>A)", "type": "GOF missense", "frequency": "DEND", "severity": "DEND severe"},
            {"variant": "p.Lys170Asn (c.510G>C)", "type": "LOF", "frequency": "CHI", "severity": "CHI"},
            {"variant": "Biallelic LOF (various)", "type": "LOF", "frequency": "CHI-KCNJ11", "severity": "CHI diazoxide-unresponsive"},
        ],
        "stats": {
            "proportion_ndm": "30-40% of all PNDM",
            "sulphonylurea_rescue_ndm": ">90%",
            "proportion_chi": "~10% of KATP-CHI",
            "dend_frequency": "~5% of KCNJ11-NDM",
        },
        "dx_delay_distribution": {
            "neonatal_0_7d": 55,
            "neonatal_8_28d": 28,
            "infant_1_6m": 12,
            "late_6m_plus": 5,
        },
    },

    # ── GLUD1 — Hyperinsulinism-Hyperammonaemia (HI/HA) Syndrome ─────────────
    {
        "gene": "GLUD1",
        "protein": (
            "GLUD1 — 10q23.33 AD-GOF — GDH-558aa — "
            "Hyperinsulinism-Hyperammonaemia-HI-HA-Syndrome — "
            "Leucine-Sensitive-Hypoglycaemia — "
            "Ammonia-ELEVATED-60-200-micromol/L-Fasting-AND-Postprandial — "
            "Diazoxide-RESPONSIVE — Protein-Restriction + Diazoxide + Avoid-Leucine-Loads"
        ),
        "alias": (
            "GLUD1 (glutamate dehydrogenase 1); OMIM gene 138130; "
            "Hyperinsulinism-Hyperammonaemia Syndrome (HI/HA) OMIM 606762. "
            "10q23.33; 558 aa; ~61.4 kDa; autosomal dominant gain-of-function. "
            "FUNCTION: GDH (glutamate dehydrogenase) catalyses the reversible reaction: "
            "Glutamate ↔ α-ketoglutarate + NH3 (ammonia). "
            "In the beta-cell, GDH links amino acid (glutamate) metabolism to the TCA cycle, "
            "providing anaplerotic substrate for ATP synthesis and insulin secretion. "
            "GDH is allosterically INHIBITED by GTP and ACTIVATED by ADP and leucine. "
            "GLUD1 GOF mutations → loss of GTP inhibition → GDH constitutively active → "
            "excess glutamate oxidation → excess ATP → K-ATP closes → "
            "Ca2+ influx → insulin secretion even during fasting or protein meals. "
            "LEUCINE SENSITIVITY — THE DIAGNOSTIC CLUE: "
            "Leucine is a potent allosteric activator of GDH; "
            "normal GDH is inhibited by GTP (physiological brake); "
            "GLUD1-GOF: GTP inhibition lost → leucine drives GDH hyperactivity → "
            "protein/leucine meals cause acute insulin surge → hypoglycaemia; "
            "historically called 'leucine-sensitive hypoglycaemia.' "
            "HYPERAMMONAEMIA — THE KEY DIAGNOSTIC MARKER: "
            "GDH hyperactivity in liver (not just pancreas) → excess glutamate deamination → "
            "excess ammonia production from glutamate → hyperammonaemia (60-200 μmol/L); "
            "CRITICAL: hyperammonaemia is PRESENT BOTH FASTING AND POSTPRANDIAL; "
            "this distinguishes HI/HA from other causes of hyperammonaemia (e.g. UCDs where "
            "ammonia is highest postprandial/fasting in the same direction); "
            "ammonia NEVER causes hepatic encephalopathy at these levels in HI/HA "
            "(levels insufficient, unlike urea cycle disorders where ammonia >500 μmol/L); "
            "BUT ammonia must be monitored — levels may occasionally exceed safe thresholds. "
            "EPILEPSY IN HI/HA: "
            "Absence seizures (generalised) occur in ~30% of GLUD1-HI/HA patients; "
            "MECHANISM: GLUD1 GOF in neurons → altered glutamate/GABA balance → "
            "network hyperexcitability; absent ketones may compound seizure risk; "
            "EEG: generalised 3 Hz spike-wave (absence pattern); "
            "TREATMENT: diazoxide + protein restriction; valproate AVOIDED "
            "(inhibits GDH → may worsen or unpredictably alter HI/HA). "
            "DIAGNOSTIC WORKUP: "
            "Ammonia: elevated fasting AND postprandial (60-200 μmol/L) — PATHOGNOMONIC pattern; "
            "Leucine tolerance test: oral leucine 150 mg/kg → glucose nadir within 30-60 min; "
            "GLUD1 sequencing; "
            "TREATMENT: "
            "Diazoxide: K-ATP opener — RESPONSIVE (K-ATP intact in GLUD1-HI/HA); "
            "Protein intake restriction: limit leucine-rich foods; "
            "Avoid leucine load: whey protein supplements, branched-chain amino acid supplements; "
            "Carbamylglutamate (Carbaglu): N-acetylglutamate synthase (NAGS) activator — "
            "may reduce ammonia production but evidence limited in HI/HA; "
            "Diazoxide titrated to euglycaemia monitoring; thiazide (chlorothiazide) co-prescribed "
            "to counteract diazoxide-induced fluid retention."
        ),
        "locus": "10q23.33",
        "aa": 558,
        "kDa": 61,
        "omim_gene": "138130",
        "omim_disease": "606762",
        "inheritance": "AD gain-of-function (de novo ~80%, familial ~20%)",
        "gene_class": "Mitochondrial glutamate dehydrogenase — TCA anaplerosis enzyme",
        "key_alerts": [
            "GLUD1-AMMONIA-PATHOGNOMONIC: Hyperinsulinism + elevated ammonia (60-200 μmol/L) fasting AND postprandial = HI/HA syndrome until proven otherwise; ammonia must be measured at EVERY hypoglycaemia evaluation in neonates and infants — it is the key diagnostic clue",
            "GLUD1-LEUCINE-SENSITIVE: Protein meals (especially leucine-rich: meat, dairy, whey) trigger acute hypoglycaemia; post-prandial glucose nadir 30-60 min after protein load; leucine tolerance test diagnostic; avoid whey protein supplements",
            "GLUD1-VALPROATE-AVOID: Valproate inhibits GDH and may unpredictably alter hyperinsulinism + hyperammonaemia balance in HI/HA; if epilepsy treatment needed, use levetiracetam or ethosuximide instead",
            "GLUD1-DIAZOXIDE-RESPONSIVE: Unlike ABCC8/KCNJ11 (KATP-absent), GLUD1-HI/HA has an intact K-ATP channel; diazoxide opens K-ATP and restores hyperpolarisation — response is usually good; first-line medical treatment",
        ],
        "etiologies": [
            {"variant": "p.Arg221Cys (c.661C>T)", "type": "GOF missense — GTP inhibition site", "frequency": "most common", "severity": "HI/HA + absence epilepsy"},
            {"variant": "p.Arg269His (c.806G>A)", "type": "GOF missense — GTP inhibition site", "frequency": "common", "severity": "HI/HA"},
            {"variant": "p.Ser445Leu (c.1334C>T)", "type": "GOF missense — antenna helix", "frequency": "moderate", "severity": "HI/HA + epilepsy"},
            {"variant": "p.Asp451Val (c.1352A>T)", "type": "GOF missense", "frequency": "less common", "severity": "HI/HA"},
            {"variant": "p.Arg221His (c.662G>A)", "type": "GOF missense", "frequency": "moderate", "severity": "HI/HA"},
        ],
        "stats": {
            "proportion_chi": "~5% of all CHI (second most common non-KATP cause)",
            "diazoxide_response": "Responsive (K-ATP intact)",
            "ammonia_range": "60-200 μmol/L (persistent fasting and postprandial)",
            "epilepsy_prevalence": "~30% absence seizures",
            "leucine_sensitivity": "~95% of GLUD1-HI/HA",
        },
        "dx_delay_distribution": {
            "neonatal_0_28d": 20,
            "infant_1_6m": 35,
            "infant_6_12m": 25,
            "late_1y_plus": 20,
        },
    },

    # ── GCK — Activating Glucokinase Hyperinsulinism ──────────────────────────
    {
        "gene": "GCK",
        "protein": (
            "GCK — 7p13 AD-GOF-Activating — Glucokinase-465aa — "
            "Activating-GCK-HI — Glucose-Sensor-Set-Point-Lowered — "
            "Persistent-Mild-Hypoglycaemia-2.5-3.5-mmol-L — "
            "NOT-GCK-MODY2-Opposite-Mutation — "
            "Diazoxide-RESPONSIVE — Mild-Clinical-Course-Usually"
        ),
        "alias": (
            "GCK (glucokinase; hexokinase IV); OMIM gene 138079; "
            "Hyperinsulinism, Congenital, due to Glucokinase Activating Mutation OMIM 602485. "
            "7p13; 465 aa; ~52 kDa; autosomal dominant gain-of-function (activating mutations). "
            "FUNCTION: Glucokinase (GCK) is the 'glucose sensor' of the pancreatic beta-cell. "
            "GCK phosphorylates glucose → glucose-6-phosphate as the first step of glycolysis. "
            "Unlike other hexokinases, GCK has a HIGH Km (S0.5 ~8 mmol/L) for glucose, "
            "meaning it only becomes active at physiological post-meal glucose concentrations — "
            "it acts as the molecular threshold sensor for insulin secretion. "
            "NORMAL GCK THRESHOLD: ~4.5-5.5 mmol/L glucose triggers 50% maximal GCK activity "
            "→ this defines the normal fasting glucose set-point. "
            "ACTIVATING GCK MUTATIONS (GOF) → LOWER Km for glucose: "
            "Set-point shifted DOWN to 2.0-3.5 mmol/L; "
            "Beta-cells interpret lower glucose as 'post-meal' → secrete insulin → "
            "persistent mild to moderate hypoglycaemia at glucose levels normal people tolerate; "
            "MILD CLINICAL COURSE in most (glucose 2.5-3.5 mmol/L) — asymptomatic in some; "
            "DIAZOXIDE RESPONSIVE — K-ATP channel intact and functional. "
            "CONTRAST WITH GCK-MODY2 (LOF): "
            "GCK LOF mutations → set-point RAISED → mild fasting hyperglycaemia → "
            "MODY2 (maturity-onset diabetes of the young type 2); "
            "SAME GENE, OPPOSITE CLINICAL EFFECT — a critical distinction on examination. "
            "DIAGNOSTIC FEATURES: "
            "Persistent mild fasting hypoglycaemia (2.5-3.5 mmol/L typical); "
            "Hypoketotic or mildly ketotic (variable); "
            "Glucagon stimulation response present; "
            "Glucose variability: wide phenotypic range within families (same mutation, different severity); "
            "GCK sequencing confirms activating mutation; "
            "FUNCTIONAL CHARACTERISATION: activity index (AI) calculated in vitro — "
            "AI >1.0 = activating (GOF); AI <1.0 = inactivating (LOF/MODY2). "
            "TREATMENT: "
            "Diazoxide first-line (responsive); "
            "Often manageable with frequent feeding alone if mild; "
            "Some adults are asymptomatic and require no treatment; "
            "Long-term monitoring for neuroglycopaenia even in mild cases."
        ),
        "locus": "7p13",
        "aa": 465,
        "kDa": 52,
        "omim_gene": "138079",
        "omim_disease": "602485",
        "inheritance": "AD gain-of-function (activating)",
        "gene_class": "Hexokinase IV — pancreatic glucose sensor enzyme",
        "key_alerts": [
            "GCK-ACTIVATING-vs-MODY2-OPPOSITE: GCK GOF (activating) = hypoglycaemia (HI); GCK LOF = hyperglycaemia (MODY2); the same gene causes opposite metabolic phenotypes — always confirm the nature of the mutation (activity index in vitro)",
            "GCK-HI-MILD-COURSE: GCK-HI is typically the mildest form of CHI; glucose 2.5-3.5 mmol/L; diazoxide usually effective; some adults asymptomatic; compare with ABCC8/KCNJ11 (profound neonatal hypoglycaemia requiring surgery)",
            "GCK-HI-DIAZOXIDE-RESPONSIVE: K-ATP channel is intact in GCK-HI; diazoxide works; contrast with KATP-CHI (ABCC8/KCNJ11) where K-ATP is absent",
            "GCK-HI-GENOTYPE-PHENOTYPE: The degree of glucose set-point shift (activity index) predicts clinical severity; AI >5 may require surgery; most mutations AI 1.5-3.0 → medically manageable",
        ],
        "etiologies": [
            {"variant": "p.Val455Met (c.1363G>A)", "type": "GOF activating missense", "frequency": "common", "severity": "mild-moderate"},
            {"variant": "p.Ala456Val (c.1367C>T)", "type": "GOF activating", "frequency": "moderate", "severity": "mild"},
            {"variant": "p.Tyr214Cys (c.641A>G)", "type": "GOF activating", "frequency": "moderate", "severity": "moderate"},
            {"variant": "p.Val62Met (c.184G>A)", "type": "GOF activating", "frequency": "less common", "severity": "moderate"},
            {"variant": "p.Asp205His (c.613G>C)", "type": "GOF activating", "frequency": "rare", "severity": "severe"},
        ],
        "stats": {
            "proportion_chi": "~1-2% of CHI",
            "diazoxide_response": "Usually responsive",
            "glucose_range": "2.5-3.5 mmol/L typical fasting",
            "surgical_rate": "<5%",
        },
        "dx_delay_distribution": {
            "neonatal_0_28d": 15,
            "infant_1_12m": 30,
            "child_1_5y": 30,
            "adult_5y_plus": 25,
        },
    },

    # ── HADH — Protein-Sensitive HI / SCHAD Deficiency ──────────────────────
    {
        "gene": "HADH",
        "protein": (
            "HADH — 4q25 AR — SCHAD-314aa — "
            "Short-Chain-3-Hydroxyacyl-CoA-Dehydrogenase-Deficiency — "
            "Protein-Sensitive-Hypoglycaemia — "
            "3-Hydroxyglutaric-Acid-Urine-PATHOGNOMONIC — "
            "HADH-Inhibits-GDH-Interaction-Disrupted — "
            "Diazoxide-RESPONSIVE — Protein-Restriction"
        ),
        "alias": (
            "HADH (3-hydroxyacyl-CoA dehydrogenase; SCHAD — short-chain 3-hydroxyacyl-CoA dehydrogenase); "
            "OMIM gene 601609; Congenital Hyperinsulinism, HADH-related (CHI-HADH) OMIM 609975. "
            "4q25; 314 aa; ~34 kDa; autosomal recessive. "
            "FUNCTION: HADH encodes SCHAD, a mitochondrial enzyme of the beta-oxidation spiral: "
            "SCHAD catalyses the third step of fatty acid beta-oxidation: "
            "L-3-hydroxyacyl-CoA → 3-ketoacyl-CoA (NAD+ dependent). "
            "In medium and short-chain fatty acid substrates (C4-C6). "
            "MECHANISM OF HYPERINSULINISM — NOVEL: "
            "HADH/SCHAD physically INTERACTS with and INHIBITS GDH (glutamate dehydrogenase) in the beta-cell; "
            "This HADH→GDH inhibitory interaction is the normal brake on GDH activity in the beta-cell. "
            "HADH LOF → loss of GDH inhibition → GDH hyperactivity (similar to GLUD1 GOF) → "
            "excess glutamate oxidation → excess ATP → K-ATP closes → insulin secretion → hypoglycaemia. "
            "PROTEIN SENSITIVITY: "
            "Protein meals provide amino acid substrates (glutamate, glutamine, leucine) → "
            "without HADH to brake GDH → GDH hyperactivated by protein substrate → "
            "insulin surge after protein meals; "
            "similar leucine/protein sensitivity to GLUD1-HI/HA. "
            "KEY METABOLIC DIFFERENCE FROM GLUD1-HI/HA: "
            "HADH: ammonia is NORMAL (GDH is inhibited via HADH but it is a different regulatory node — "
            "net ammonia accumulation is not the issue); "
            "GLUD1: ammonia is ELEVATED 60-200 μmol/L (excess glutamate deamination in liver). "
            "DIAGNOSTIC BIOMARKER: "
            "3-Hydroxyglutaric acid in urine (by GC-MS organic acids): ELEVATED — PATHOGNOMONIC; "
            "3-OH-glutaric acid is a marker of SCHAD deficiency; "
            "plasma acylcarnitine: 3-OH-C4-carnitine (C4-OH) may be elevated; "
            "HADH sequencing confirms. "
            "TREATMENT: "
            "Diazoxide: RESPONSIVE (K-ATP intact, downstream of HADH→GDH→K-ATP axis); "
            "Protein restriction: moderate reduction in protein to avoid GDH substrate overload; "
            "Avoid prolonged fasting; "
            "Emergency glucose (oral/IV) for hypoglycaemia episodes."
        ),
        "locus": "4q25",
        "aa": 314,
        "kDa": 34,
        "omim_gene": "601609",
        "omim_disease": "609975",
        "inheritance": "AR (biallelic loss-of-function)",
        "gene_class": "Short-chain 3-hydroxyacyl-CoA dehydrogenase — beta-oxidation + GDH regulator",
        "key_alerts": [
            "HADH-3-HYDROXYGLUTARIC-ACID-PATHOGNOMONIC: 3-Hydroxyglutaric acid in urine (GC-MS organic acids) is the PATHOGNOMONIC biomarker for SCHAD/HADH deficiency; always request organic acids in any CHI workup",
            "HADH-NORMAL-AMMONIA-KEY-DDx: Unlike GLUD1-HI/HA (ammonia 60-200 μmol/L), HADH-CHI has NORMAL ammonia — this biochemical difference distinguishes them; both are protein-sensitive and diazoxide-responsive",
            "HADH-GDH-INTERACTION: HADH physically inhibits GDH in beta-cells; HADH loss = GDH disinhibition = similar molecular mechanism to GLUD1-GOF but without hepatic ammonia production",
            "HADH-PROTEIN-SENSITIVE: Same protein/leucine sensitivity as GLUD1; protein meals trigger hypoglycaemia; avoid whey supplements and high-protein loads",
        ],
        "etiologies": [
            {"variant": "p.Leu147Arg (c.440T>G)", "type": "LOF missense", "frequency": "Middle Eastern founder", "severity": "severe"},
            {"variant": "p.Arg236Cys (c.706C>T)", "type": "LOF missense", "frequency": "moderate", "severity": "severe"},
            {"variant": "Exon deletions", "type": "deletion", "frequency": "variable", "severity": "severe"},
            {"variant": "p.Thr136Met (c.407C>T)", "type": "LOF missense", "frequency": "less common", "severity": "moderate"},
        ],
        "stats": {
            "proportion_chi": "~1% of CHI",
            "diazoxide_response": "Responsive",
            "ammonia": "Normal (DDx from GLUD1-HI/HA)",
            "biomarker": "3-OH-glutaric acid urine PATHOGNOMONIC",
        },
        "dx_delay_distribution": {
            "neonatal_0_28d": 25,
            "infant_1_6m": 40,
            "infant_6_12m": 25,
            "late_1y_plus": 10,
        },
    },

    # ── SLC16A1 — Exercise-Induced Hyperinsulinism (EIHI) ────────────────────
    {
        "gene": "SLC16A1",
        "protein": (
            "SLC16A1 — 1p13.2 AD-GOF-Promoter — MCT1-494aa — "
            "Exercise-Induced-Hyperinsulinism-EIHI — "
            "Ectopic-MCT1-Expression-in-Beta-Cell — "
            "Pyruvate-Entry-During-Exercise-Triggers-Insulin — "
            "Hypoglycaemia-30-60min-Post-Exercise — "
            "Diazoxide-INEFFECTIVE — Exercise-Avoidance-Carbohydrate-Pre-Load"
        ),
        "alias": (
            "SLC16A1 (solute carrier family 16 member 1; MCT1 — monocarboxylate transporter 1); "
            "OMIM gene 600682; Exercise-Induced Hyperinsulinism (EIHI) OMIM 606517. "
            "1p13.2; 494 aa; ~54 kDa; autosomal dominant (GOF — promoter activation mutations). "
            "FUNCTION: MCT1 is a plasma membrane transporter for monocarboxylates: "
            "pyruvate, lactate, and ketone bodies. "
            "MCT1 is normally ABSENT from pancreatic beta-cells — this is physiologically essential "
            "because pyruvate/lactate entry into beta-cells would trigger insulin secretion; "
            "the absence of MCT1 in beta-cells is a PROTECTIVE design of normal physiology. "
            "EXERCISE PHYSIOLOGY IN NORMAL INDIVIDUALS: "
            "Vigorous exercise → muscle glycogenolysis → elevated blood lactate + pyruvate; "
            "in normal beta-cells: no MCT1 → pyruvate/lactate CANNOT enter → no aberrant insulin; "
            "EIHI (SLC16A1 GOF): "
            "Promoter mutations in SLC16A1 activate MCT1 expression in beta-cells (ectopic); "
            "Exercise → ↑ blood pyruvate/lactate → enters beta-cells via ectopic MCT1 → "
            "pyruvate enters mitochondria → ATP generated → K-ATP closes → "
            "Ca2+ influx → insulin secreted despite falling glucose during exercise; "
            "HYPOGLYCAEMIA PATTERN: 30-60 minutes AFTER exercise (not during); "
            "anaerobic exercise more provocative than aerobic (higher lactate generation). "
            "CLINICAL FEATURES: "
            "Often children/adolescents with sport participation; "
            "Hypoglycaemia symptoms post-exercise (sweating, tremor, confusion, seizure); "
            "Normal fasting glucose; "
            "Normal post-prandial glucose; "
            "Exercise provocative test: cycle ergometer → glucose monitoring post-exercise. "
            "TREATMENT — UNIQUE CHALLENGE: "
            "Diazoxide: INEFFECTIVE (K-ATP is functional — it responds normally to ATP — "
            "the problem is excess ATP from ectopic MCT1, not K-ATP dysfunction; "
            "opening K-ATP further does not prevent pyruvate-driven ATP generation); "
            "Approaches: "
            "Carbohydrate load before exercise (raise glucose before exercise to buffer fall); "
            "Glucose gel immediately post-exercise; "
            "Reduce exercise intensity; "
            "Avoid fasted exercise; "
            "Experimental: partial beta-cell surgical reduction; "
            "Rapamycin (mTOR inhibitor) — research stage."
        ),
        "locus": "1p13.2",
        "aa": 494,
        "kDa": 54,
        "omim_gene": "600682",
        "omim_disease": "606517",
        "inheritance": "AD gain-of-function (promoter mutations — ectopic beta-cell expression)",
        "gene_class": "Monocarboxylate transporter — pyruvate/lactate membrane transport",
        "key_alerts": [
            "SLC16A1-EXERCISE-TRIGGERED: Hypoglycaemia SPECIFICALLY post-exercise (30-60 min after), NOT fasting or postprandial; pattern is pathognomonic for EIHI; standard CHI evaluation may miss it if fasting and post-prandial studies are normal",
            "SLC16A1-DIAZOXIDE-INEFFECTIVE: MCT1-EIHI cannot be treated with diazoxide; the K-ATP channel is structurally normal; diazoxide has no therapeutic benefit; management is carbohydrate loading pre-exercise and glucose supplementation post-exercise",
            "SLC16A1-PROMOTER-MUTATIONS: Mutations are in the SLC16A1 PROMOTER (not the coding region); standard exome sequencing (WES) MISSES promoter mutations; specifically request SLC16A1 promoter sequencing or promoter-inclusive targeted panel",
            "SLC16A1-PYUVATE-MECHANISM: Pyruvate (from muscle during exercise) enters beta-cells via ectopic MCT1; pyruvate → mitochondrial ATP → closes K-ATP → insulin release; it is a post-exercise phenomenon because lactate/pyruvate peak 30-60 min post-exercise",
        ],
        "etiologies": [
            {"variant": "SLC16A1 promoter c.-224C>T", "type": "GOF promoter", "frequency": "most common EIHI", "severity": "moderate"},
            {"variant": "SLC16A1 promoter c.-223A>G", "type": "GOF promoter", "frequency": "common", "severity": "moderate"},
            {"variant": "SLC16A1 promoter c.-71G>A", "type": "GOF promoter", "frequency": "moderate", "severity": "moderate"},
            {"variant": "SLC16A1 promoter complex rearrangements", "type": "promoter duplication/inversion", "frequency": "rare", "severity": "variable"},
        ],
        "stats": {
            "proportion_chi": "<1% of CHI (under-recognised due to WES promoter miss)",
            "diazoxide_response": "Ineffective",
            "trigger": "Exercise — anaerobic > aerobic",
            "timing": "30-60 min post-exercise",
        },
        "dx_delay_distribution": {
            "early_childhood_0_5y": 15,
            "school_age_5_12y": 50,
            "adolescent_12_18y": 25,
            "adult_18y_plus": 10,
        },
    },

    # ── HNF4A — Neonatal HI → MODY1 ──────────────────────────────────────────
    {
        "gene": "HNF4A",
        "protein": (
            "HNF4A — 20q13.12 AD — HNF4α-455aa — "
            "Neonatal-HI-Transient-or-Persistent → MODY1-Adolescence — "
            "Macrosomia-LGA-PATHOGNOMONIC-Clue — "
            "Diazoxide-RESPONSIVE — "
            "ANTICIPATORY-GUIDANCE-MODY1-INEVITABLE — "
            "SU-Monotherapy-When-Diabetes-Established"
        ),
        "alias": (
            "HNF4A (hepatocyte nuclear factor 4-alpha); OMIM gene 600281; "
            "MODY1 (OMIM 125850); HNF4A-related Hyperinsulinism (neonatal). "
            "20q13.12; 455 aa; ~50 kDa; autosomal dominant haploinsufficiency. "
            "FUNCTION: HNF4A is a nuclear receptor transcription factor critical for hepatocyte, "
            "intestinal, kidney, and pancreatic beta-cell function. "
            "In the beta-cell: HNF4A regulates genes controlling glucose-stimulated insulin secretion (GSIS), "
            "mitochondrial function, and ATP synthesis coupling. "
            "DUAL TEMPORAL PHENOTYPE — ONE GENE, TWO LIFE-STAGE DISEASES: "
            "NEONATAL/INFANCY (HI phase): "
            "HNF4A haploinsufficiency → transcriptional dysregulation in the immature beta-cell → "
            "paradoxically EXCESS insulin secretion; "
            "MECHANISM not fully elucidated but involves altered K-ATP channel expression; "
            "Clinically: transient or persistent neonatal hypoglycaemia, often diazoxide-responsive; "
            "MACROSOMIA: foetal hyperinsulinism → excess growth → large for gestational age (LGA) at birth — "
            "this is a CRITICAL clinical clue; any LGA neonate with hypoglycaemia → HNF4A/HNF1A testing; "
            "SPONTANEOUS RESOLUTION: in many cases HI remits in infancy/early childhood as beta-cell maturation occurs; "
            "CHILDHOOD/ADOLESCENCE (MODY1 phase): "
            "Same HNF4A mutation that caused neonatal HI now manifests as MODY1 (diabetes); "
            "MODY1 onset: typically teens to young adulthood (earlier than MODY3/HNF1A in many families); "
            "MODY1 is the SECOND rarest MODY form (~5% MODY); "
            "Sulphonylurea-responsive: HNF4A MODY diabetes responds to low-dose SU (same K-ATP intact); "
            "TRANSITION: child with resolved HI → years of normoglycaemia → "
            "MODY1 diabetes emerges; monitoring essential. "
            "CLINICAL PEARLS: "
            "Macrosomia + neonatal hypoglycaemia + family history of diabetes = HNF4A until excluded; "
            "A parent with diabetes + a macrosomic infant with hypoglycaemia = very high pre-test probability; "
            "HNF4A mutation may be missed if only neonatal HI investigated (genetics not done); "
            "Anticipatory guidance: every child with HNF4A-HI WILL develop MODY1 eventually — "
            "annual fasting glucose monitoring from age 10; glycated haemoglobin. "
            "TREATMENT: "
            "Neonatal HI: diazoxide (responsive); "
            "MODY1: sulphonylurea monotherapy (low dose); GLP1 agonists adjunct; "
            "Monitor renal function (HNF4A also expressed in kidney — renal tubulopathy in some)."
        ),
        "locus": "20q13.12",
        "aa": 455,
        "kDa": 50,
        "omim_gene": "600281",
        "omim_disease": "125850",
        "inheritance": "AD haploinsufficiency",
        "gene_class": "Nuclear receptor transcription factor — hepatocyte/beta-cell gene regulator",
        "key_alerts": [
            "HNF4A-MACROSOMIA-CLUE: Macrosomia (LGA, birthweight >4 kg or >2 SD above mean) + neonatal hypoglycaemia → must test HNF4A and HNF1A; foetal hyperinsulinism drives excess growth; this is the pathognomonic clinical triad",
            "HNF4A-NEONATAL-HI-THEN-MODY1: The same mutation causes neonatal hypoglycaemia (HI in infancy) AND MODY1 (diabetes in adolescence/adulthood); the HI may resolve spontaneously; anticipatory guidance is mandatory — annual glucose monitoring from age 10",
            "HNF4A-DIAZOXIDE-RESPONSIVE: Unlike KATP-CHI (ABCC8/KCNJ11), HNF4A-HI K-ATP channel is intact; diazoxide works; medical management often sufficient without surgery",
            "HNF4A-FAMILY-HISTORY-DIABETES: Always ask for family history of diabetes in CHI families; a parent with early-onset diabetes + macrosomic infant = near-diagnostic for HNF4A/HNF1A; genetic testing of the parent is diagnostic",
        ],
        "etiologies": [
            {"variant": "Various exon 1-3 deletion/truncation", "type": "LOF", "frequency": "common HNF4A-HI", "severity": "HI + MODY1"},
            {"variant": "p.Arg154Ter (c.460C>T)", "type": "nonsense LOF", "frequency": "moderate", "severity": "HI + MODY1"},
            {"variant": "Exon 2 deletion", "type": "large deletion", "frequency": "less common", "severity": "HI + MODY1"},
            {"variant": "Splice site variants intron 1-2", "type": "LOF splice", "frequency": "moderate", "severity": "HI + MODY1"},
            {"variant": "p.Arg245Gln (c.734G>A)", "type": "LOF missense", "frequency": "less common", "severity": "MODY1 dominant"},
        ],
        "stats": {
            "proportion_chi": "~5% of CHI",
            "diazoxide_response": "Usually responsive",
            "macrosomia_prevalence": "~80% of HNF4A-HI neonates",
            "mody1_inevitability": "~95% develop MODY1 by adult age",
            "mody1_su_response": ">80%",
        },
        "dx_delay_distribution": {
            "neonatal_0_28d": 70,
            "infant_1_6m": 20,
            "infant_6_12m": 8,
            "late_1y_plus": 2,
        },
    },

    # ── FOXA2 — Neonatal HI + Hypopituitarism Triad ──────────────────────────
    {
        "gene": "FOXA2",
        "protein": (
            "FOXA2 — 20p11.21 AD — HNF3β-1161aa — "
            "Neonatal-HI + Hypopituitarism-Triad-GH-ACTH-TSH-Deficiency — "
            "ABSENT-Glucagon-Counter-Regulation-PATHOGNOMONIC — "
            "Pituitary-MRI-Mandatory — "
            "Anterior-Pituitary-Aplasia/Hypoplasia — "
            "Pulmonary-Sequestration-Rare-Association"
        ),
        "alias": (
            "FOXA2 (forkhead box protein A2; hepatocyte nuclear factor 3-beta; HNF3β); "
            "OMIM gene 600288; FOXA2-related Hyperinsulinism/Hypopituitarism syndrome. "
            "20p11.21; 1161 aa; ~47 kDa (DNA-binding domain); autosomal dominant. "
            "FUNCTION: FOXA2 is a pioneer transcription factor of the Forkhead (FOX) family: "
            "it opens chromatin at target gene promoters, enabling other TFs to bind. "
            "FOXA2 is essential during embryonic development for: "
            "(1) Pancreatic development: beta-cell differentiation and function — "
            "controls insulin, glucagon, somatostatin gene expression; "
            "(2) Anterior pituitary development: required for anterior pituitary organogenesis — "
            "haploinsufficiency → anterior pituitary aplasia/hypoplasia → "
            "multiple pituitary hormone deficiencies (GH, ACTH, TSH, LH, FSH); "
            "(3) Hepatic gluconeogenesis and lipid metabolism. "
            "CLINICAL PHENOTYPE — THE TRIAD: "
            "1. NEONATAL HYPERINSULINISM: "
            "Profound neonatal hypoglycaemia; "
            "Mechanism: FOXA2 haploinsufficiency → loss of K-ATP channel gene expression regulation → "
            "beta-cell dysregulation → excess insulin secretion; "
            "2. HYPOPITUITARISM (MULTIPLE ANTERIOR PITUITARY HORMONE DEFICIENCY): "
            "Growth hormone (GH) deficiency → short stature + neonatal hypoglycaemia (compound); "
            "ACTH deficiency → secondary adrenal insufficiency → hypocortisolaemia → "
            "hypoglycaemia (compounds the hyperinsulinism); "
            "TSH deficiency → central hypothyroidism → prolonged neonatal jaundice + poor feeding; "
            "LH/FSH deficiency → hypogonadotropic hypogonadism in older patients; "
            "3. ABSENT GLUCAGON COUNTER-REGULATION — PATHOGNOMONIC: "
            "FOXA2 controls the glucagon gene (GCG) in pancreatic alpha-cells; "
            "FOXA2 haploinsufficiency → reduced/absent glucagon secretion during hypoglycaemia; "
            "Glucagon stimulation test: ABSENT or severely blunted glucagon response — "
            "this is the PATHOGNOMONIC finding; "
            "Clinical implication: glucagon emergency injection will be INEFFECTIVE in hypoglycaemia crisis — "
            "must use IV glucose; standard emergency glucagon kit is CONTRAINDICATED. "
            "PULMONARY SEQUESTRATION: rare associated anomaly in some FOXA2 patients "
            "(FOXA2 also expressed in lung development). "
            "PITUITARY MRI: "
            "MANDATORY in all cases of HI + hypopituitarism features; "
            "Shows: anterior pituitary aplasia, hypoplasia, or ectopic posterior pituitary; "
            "Stalk abnormalities. "
            "TREATMENT: "
            "HI: diazoxide (usually responsive) + octreotide; "
            "Hypopituitarism: GH replacement, hydrocortisone (ACTH deficiency), "
            "levothyroxine (TSH deficiency), sex hormone replacement later; "
            "GLUCAGON EMERGENCY: IV glucose is the ONLY safe emergency treatment; "
            "glucagon injection unreliable and often ineffective — "
            "carers and school must be trained in IV glucose administration."
        ),
        "locus": "20p11.21",
        "aa": 1161,
        "kDa": 47,
        "omim_gene": "600288",
        "omim_disease": "None (FOXA2 haploinsufficiency syndrome)",
        "inheritance": "AD haploinsufficiency (de novo dominant)",
        "gene_class": "Pioneer forkhead transcription factor — pancreas/pituitary/liver organogenesis",
        "key_alerts": [
            "FOXA2-ABSENT-GLUCAGON-PATHOGNOMONIC: FOXA2 haploinsufficiency causes absent alpha-cell glucagon production; glucagon stimulation test shows absent/severely blunted response; this is PATHOGNOMONIC — glucagon emergency injection WILL NOT WORK; always use IV glucose for hypoglycaemia crisis",
            "FOXA2-HYPOPITUITARISM-MANDATORY-MRI: All patients with HI + any pituitary hormone deficiency (GH, ACTH, TSH) must have pituitary MRI; FOXA2 causes anterior pituitary aplasia/hypoplasia visible on MRI; cortisol/GH deficiency compounds hypoglycaemia severity",
            "FOXA2-ACTH-DEFICIENCY-HIDDEN: Secondary adrenal insufficiency (ACTH deficient) in FOXA2 patients compounds hypoglycaemia; always test morning cortisol + ACTH stimulation test and replace hydrocortisone if deficient; adrenal crisis risk",
            "FOXA2-GLUCAGON-EMERGENCY-CI: In FOXA2-HI, glucagon kits given to families may be INEFFECTIVE; educate parents/carers that IV dextrose 10% is the ONLY reliable emergency treatment; prescribe Glucogel + written instructions for IV glucose administration",
        ],
        "etiologies": [
            {"variant": "FOXA2 de novo LOF mutations (various)", "type": "haploinsufficiency", "frequency": "rare — reported series <30 patients worldwide", "severity": "severe triad"},
            {"variant": "p.Arg261Trp (c.781C>T)", "type": "LOF missense forkhead domain", "frequency": "reported", "severity": "severe"},
            {"variant": "p.Arg266Gln (c.797G>A)", "type": "LOF missense forkhead domain", "frequency": "reported", "severity": "severe"},
            {"variant": "p.Ser261fs (frameshift)", "type": "frameshift LOF", "frequency": "reported", "severity": "severe"},
        ],
        "stats": {
            "proportion_chi": "Rare (<1%) — possibly under-diagnosed",
            "hypopituitarism": "Multiple anterior pituitary hormone deficiencies ~100% of reported cases",
            "glucagon_response": "Absent or severely blunted — PATHOGNOMONIC",
            "pituitary_mri_abnormal": "~90%",
        },
        "dx_delay_distribution": {
            "neonatal_0_7d": 65,
            "neonatal_8_28d": 25,
            "infant_1_6m": 8,
            "late_6m_plus": 2,
        },
    },
]


def _generate_patients():
    """Generate 40 deterministic synthetic patients per gene using seeded RNG."""
    for idx, gene_data in enumerate(HI_GENES):
        seed = SEED_BASE + idx
        rng = random.Random(seed)
        gene = gene_data["gene"]
        patients = []

        for i in range(40):
            if gene == "ABCC8":
                pattern = rng.choices(
                    ["diffuse_biallelic", "focal_paternal_loh", "severe_persistent"],
                    weights=[50, 42, 8]
                )[0]
                onset_age = rng.uniform(0.0, 0.2)   # days (fraction of years)
                dx_delay = rng.randint(0, 21)        # days
                diazoxide_response = False
                glucose_nadir = rng.uniform(0.6, 2.0)
                gir_required = rng.uniform(8, 18)    # mg/kg/min
                focal = (pattern == "focal_paternal_loh")
                surgical = rng.random() < (0.45 if focal else 0.85)
                patients.append({
                    "patient_id": f"ABCC8-{i+1:03d}",
                    "onset_age_days": round(onset_age * 365, 1),
                    "dx_delay_days": dx_delay,
                    "phenotype": pattern,
                    "focal": focal,
                    "diazoxide_response": diazoxide_response,
                    "glucose_nadir_mmol": round(glucose_nadir, 1),
                    "gir_mg_kg_min": round(gir_required, 1),
                    "surgical": surgical,
                    "gene": gene, "seed": seed,
                })

            elif gene == "KCNJ11":
                phenotype = rng.choices(
                    ["chi_lof", "ndm_isolated", "idend", "dend"],
                    weights=[30, 52, 12, 6]
                )[0]
                onset_age_days = rng.randint(0, 30) if phenotype == "chi_lof" else rng.randint(1, 90)
                su_responsive = phenotype in ("ndm_isolated", "idend", "dend")
                patients.append({
                    "patient_id": f"KCNJ11-{i+1:03d}",
                    "onset_age_days": onset_age_days,
                    "phenotype": phenotype,
                    "su_responsive": su_responsive,
                    "neurological": phenotype in ("idend", "dend"),
                    "epilepsy": phenotype == "dend",
                    "gene": gene, "seed": seed,
                })

            elif gene == "GLUD1":
                onset_age = rng.randint(2, 24)  # months
                dx_delay = rng.randint(6, 48)
                ammonia = rng.uniform(65, 195)
                leucine_sensitive = True
                epilepsy = rng.random() < 0.30
                glucose_nadir = rng.uniform(1.8, 3.0)
                patients.append({
                    "patient_id": f"GLUD1-{i+1:03d}",
                    "onset_age_months": onset_age,
                    "dx_delay_months": dx_delay,
                    "ammonia_umol": round(ammonia, 1),
                    "leucine_sensitive": leucine_sensitive,
                    "absence_epilepsy": epilepsy,
                    "glucose_nadir_mmol": round(glucose_nadir, 1),
                    "diazoxide_response": True,
                    "gene": gene, "seed": seed,
                })

            elif gene == "GCK":
                onset_age = rng.randint(1, 48)  # months (wide range)
                dx_delay = rng.randint(3, 60)
                fasting_glucose = rng.uniform(2.4, 3.6)
                diazoxide_response = rng.random() < 0.80
                activity_index = rng.uniform(1.4, 4.5)
                surgical = rng.random() < 0.05
                patients.append({
                    "patient_id": f"GCK-{i+1:03d}",
                    "onset_age_months": onset_age,
                    "dx_delay_months": dx_delay,
                    "fasting_glucose_mmol": round(fasting_glucose, 1),
                    "activity_index": round(activity_index, 1),
                    "diazoxide_response": diazoxide_response,
                    "surgical": surgical,
                    "gene": gene, "seed": seed,
                })

            elif gene == "HADH":
                onset_age = rng.randint(1, 12)  # months
                dx_delay = rng.randint(6, 36)
                oh_glutaric_urine = rng.random() < 0.98  # pathognomonic
                ammonia_normal = True
                protein_sensitive = True
                patients.append({
                    "patient_id": f"HADH-{i+1:03d}",
                    "onset_age_months": onset_age,
                    "dx_delay_months": dx_delay,
                    "3oh_glutaric_acid_urine": oh_glutaric_urine,
                    "ammonia_normal": ammonia_normal,
                    "protein_sensitive": protein_sensitive,
                    "diazoxide_response": True,
                    "gene": gene, "seed": seed,
                })

            elif gene == "SLC16A1":
                onset_age = rng.randint(5, 17)  # years (school/adolescent age)
                dx_delay = rng.randint(12, 84)  # months
                sport_triggered = True
                post_exercise_nadir_min = rng.randint(25, 75)
                glucose_nadir = rng.uniform(1.6, 2.8)
                promoter_confirmed = rng.random() < 0.85
                patients.append({
                    "patient_id": f"SLC16A1-{i+1:03d}",
                    "onset_age_years": onset_age,
                    "dx_delay_months": dx_delay,
                    "sport_triggered": sport_triggered,
                    "post_exercise_nadir_min": post_exercise_nadir_min,
                    "glucose_nadir_mmol": round(glucose_nadir, 1),
                    "diazoxide_response": False,
                    "promoter_variant_confirmed": promoter_confirmed,
                    "gene": gene, "seed": seed,
                })

            elif gene == "HNF4A":
                macrosomia = rng.random() < 0.82
                onset_age_days = rng.randint(0, 5)
                dx_delay = rng.randint(0, 30)  # days
                resolved_hi = rng.random() < 0.60  # many resolve
                mody1_age = rng.randint(12, 35)
                patients.append({
                    "patient_id": f"HNF4A-{i+1:03d}",
                    "onset_age_days": onset_age_days,
                    "dx_delay_days": dx_delay,
                    "macrosomia": macrosomia,
                    "hi_resolved": resolved_hi,
                    "mody1_onset_age_years": mody1_age,
                    "diazoxide_response": True,
                    "gene": gene, "seed": seed,
                })

            else:  # FOXA2
                onset_age_days = rng.randint(0, 3)
                dx_delay = rng.randint(14, 90)  # days (hypopituitarism delayed dx)
                gh_deficient = True
                acth_deficient = rng.random() < 0.90
                tsh_deficient = rng.random() < 0.85
                glucagon_response = rng.choices(
                    ["absent", "severely_blunted"],
                    weights=[70, 30]
                )[0]
                pituitary_mri = rng.choices(
                    ["aplasia", "hypoplasia", "ectopic_posterior"],
                    weights=[40, 40, 20]
                )[0]
                patients.append({
                    "patient_id": f"FOXA2-{i+1:03d}",
                    "onset_age_days": onset_age_days,
                    "dx_delay_days": dx_delay,
                    "gh_deficient": gh_deficient,
                    "acth_deficient": acth_deficient,
                    "tsh_deficient": tsh_deficient,
                    "glucagon_response": glucagon_response,
                    "pituitary_mri": pituitary_mri,
                    "diazoxide_response": True,
                    "gene": gene, "seed": seed,
                })
        gene_data["patients"] = patients


_generate_patients()


def overview():
    all_genes_info = [
        {
            "gene": g["gene"],
            "locus": g["locus"],
            "aa": g["aa"],
            "n_patients": len(g["patients"]),
        }
        for g in HI_GENES
    ]
    total = sum(len(g["patients"]) for g in HI_GENES)
    return {
        "atlas": "Hereditary Hyperinsulinism Atlas — Complete 8-Gene Reference",
        "subtitle": (
            "ABCC8 (SUR1-CHI1) · KCNJ11 (Kir6.2-CHI2/NDM) · GLUD1 (HI/HA-GDH) · GCK (Activating-HI) · "
            "HADH (SCHAD-Protein-Sensitive) · SLC16A1 (EIHI-Exercise-Induced) · "
            "HNF4A (Neonatal-HI→MODY1) · FOXA2 (HI+Hypopituitarism) — "
            "320 Patients (8×40, Seeds 1774–1781)"
        ),
        "total_patients": total,
        "seed_range": f"{SEED_BASE}–{SEED_BASE + 7}",
        "aggregate_stats": {
            "genes_covered": 8,
            "patients_per_gene": 40,
            "katp_channel_genes": 2,
            "enzyme_gain_of_function_genes": 2,
            "transcription_factor_genes": 2,
            "transporter_gene": 1,
            "dehydrogenase_gene": 1,
        },
        "genes": all_genes_info,
        "top_alerts": [
            "ABCC8-18F-DOPA-PET-MANDATORY: All diazoxide-unresponsive CHI require 18F-DOPA PET-CT before pancreatectomy to distinguish focal (limited curative resection) from diffuse (near-total pancreatectomy); ABCC8 paternal mutation origin must always be determined — paternal het = focal candidate",
            "KCNJ11-NDM-SULPHONYLUREA-RESCUES: Neonatal DM onset <6 months is genetic until proven otherwise; KCNJ11 GOF → sulphonylurea (glibenclamide) rescues >90% off insulin; DEND syndrome neurological features may also improve if started early",
            "GLUD1-AMMONIA-PATHOGNOMONIC: Hyperinsulinism + fasting AND postprandial ammonia 60-200 μmol/L = HI/HA syndrome; GLUD1 sequencing is diagnostic; diazoxide-responsive; avoid valproate",
            "GCK-ACTIVATING-vs-MODY2-OPPOSITE: GCK GOF = hypoglycaemia (HI); GCK LOF = hyperglycaemia (MODY2); confirm mutation type with activity index; mild course often medically managed",
            "HADH-3-OH-GLUTARIC-ACID-PATHOGNOMONIC: Urine organic acids (GC-MS) showing 3-hydroxyglutaric acid elevation = SCHAD/HADH deficiency; normal ammonia distinguishes from GLUD1-HI/HA; diazoxide-responsive",
            "SLC16A1-EXERCISE-TRIGGERED-PROMOTER-MISS: Exercise-induced HI (30-60 min post-exercise); standard WES MISSES promoter mutations — specifically request SLC16A1 promoter analysis; diazoxide INEFFECTIVE; carbohydrate pre-load + post-exercise glucose",
            "HNF4A-MACROSOMIA-THEN-MODY1: Macrosomia + neonatal HI → HNF4A; HI resolves in ~60%; MODY1 inevitable — annual glucose monitoring from age 10; sulphonylurea-responsive when diabetes established",
            "FOXA2-GLUCAGON-ABSENT-IV-GLUCOSE-ONLY: FOXA2 = absent glucagon response; glucagon emergency injection INEFFECTIVE; IV dextrose 10% is the ONLY reliable acute treatment; pituitary MRI mandatory for anterior pituitary aplasia/hypoplasia",
        ],
    }


def breakdown():
    result = []
    for idx, g in enumerate(HI_GENES):
        result.append({
            "gene": g["gene"],
            "protein": g["protein"],
            "alias": g["alias"],
            "locus": g["locus"],
            "aa": g["aa"],
            "kDa": g["kDa"],
            "omim_gene": g["omim_gene"],
            "omim_disease": g["omim_disease"],
            "inheritance": g["inheritance"],
            "gene_class": g["gene_class"],
            "key_alerts": g["key_alerts"],
            "etiologies": g["etiologies"],
            "stats": g["stats"],
            "dx_delay_distribution": g["dx_delay_distribution"],
            "computed": {
                "n_patients": len(g["patients"]),
                "seed": SEED_BASE + idx,
            },
            "sample_patients": g["patients"][:10],
        })
    return result


def definitions():
    return {
        "concepts": {
            "The K-ATP Channel — Molecular Architecture and the Hyperinsulinism Mechanism": (
                "The pancreatic beta-cell ATP-sensitive potassium (K-ATP) channel is the MASTER regulator "
                "of glucose-stimulated insulin secretion. "
                "ARCHITECTURE: The K-ATP channel is a hetero-octamer: (SUR1)4 × (Kir6.2)4. "
                "SUR1 (ABCC8) = regulatory subunit (sulphonylurea receptor, ABC transporter). "
                "Kir6.2 (KCNJ11) = pore-forming subunit (K+ inwardly-rectifying). "
                "Both genes are on chromosome 11p15.1 — sequenced together in all CHI panels. "
                "NORMAL PHYSIOLOGY: "
                "FASTING (low glucose → low ATP/ADP ratio): K-ATP OPEN → K+ efflux → "
                "membrane hyperpolarisation → voltage-gated Ca2+ channels CLOSED → "
                "NO insulin secretion (physiological). "
                "POST-MEAL (high glucose → glycolysis → high ATP): K-ATP CLOSES → "
                "K+ efflux stops → depolarisation → Ca2+ influx → "
                "exocytosis → INSULIN SECRETED. "
                "CHI MECHANISM (ABCC8/KCNJ11 LOF): "
                "K-ATP cannot open even during fasting → permanently depolarised → "
                "Ca2+ always entering → insulin always secreted → hypoglycaemia. "
                "NDM MECHANISM (KCNJ11 GOF): "
                "K-ATP constitutively OPEN → permanently hyperpolarised → "
                "Ca2+ channels never open → NO insulin secretion → diabetes. "
                "DIAZOXIDE MECHANISM: K-ATP opener (binds SUR1 MgADP-binding site) → "
                "promotes channel opening → hyperpolarisation → reduces insulin; "
                "REQUIRES SUR1 (ABCC8) TO BE PRESENT AND FUNCTIONAL — "
                "if ABCC8 biallelic LOF: no SUR1 → diazoxide has no target → INEFFECTIVE. "
                "SULPHONYLUREA MECHANISM: Binds SUR1 → CLOSES K-ATP → depolarisation → insulin; "
                "in KCNJ11-GOF NDM: sulphonylurea closes the constitutively open channel → "
                "restores normal glucose-stimulated secretion → cures NDM."
            ),
            "Focal vs Diffuse CHI — The 11p15 Imprinting Mechanism": (
                "The focal/diffuse distinction in ABCC8/KCNJ11-CHI is determined by the molecular mechanism "
                "of the second hit on chromosome 11p15: "
                "FOCAL CHI MECHANISM: "
                "Step 1: Patient inherits one ABCC8/KCNJ11 mutation from the FATHER "
                "(paternal allele carries the mutation). "
                "Step 2: In a discrete focus of beta-cells, somatic loss of heterozygosity (LOH) occurs — "
                "the MATERNAL 11p15 region is lost/replaced by a duplication of the paternal allele. "
                "Step 3: This focal region now has: (a) the paternal ABCC8/KCNJ11 LOF mutation on both copies, "
                "(b) loss of the maternal IGF2 imprinted region (normally expressed from maternal — wait, "
                "IGF2 is paternally expressed; H19 is maternally expressed); "
                "the key: maternal LOH → biallelic ABCC8/KCNJ11 LOF only in the focus → "
                "focal hyperinsulinism confined to that anatomical area. "
                "DIFFUSE CHI MECHANISM: "
                "Biallelic (compound heterozygous or homozygous) ABCC8/KCNJ11 LOF mutations — "
                "ALL beta-cells throughout the entire pancreas are affected. "
                "18F-DOPA PET-CT: "
                "18F-DOPA (fluorodopa) is taken up by amine-precursor uptake cells (APUD cells including beta-cells); "
                "focal CHI: discrete area of increased uptake; "
                "diffuse CHI: uniform diffuse uptake throughout pancreas; "
                "sensitivity for focal lesion localisation: 80-85%; "
                "guides surgical approach: focal → limited resection (curative); "
                "diffuse → near-total (95-98%) pancreatectomy (not curative, manages disease). "
                "PATERNAL ORIGIN OF MUTATION IS CRITICAL: "
                "If the mutation is paternally inherited → risk of focal CHI → 18F-DOPA PET; "
                "if maternally inherited → diffuse CHI only (no focal risk because LOH of maternal allele "
                "does not create biallelic situation); "
                "ALWAYS determine parental origin before surgery."
            ),
            "The Critical Sample — How to Diagnose Hyperinsulinism During Hypoglycaemia": (
                "CRITICAL SAMPLE: Blood collected at the moment of hypoglycaemia (glucose <3.0 mmol/L, "
                "ideally <2.6 mmol/L). "
                "THIS SAMPLE IS DIAGNOSTIC — do not treat before collecting. "
                "Components of the critical sample: "
                "PLASMA GLUCOSE: confirm hypoglycaemia (POC glucometer underestimates plasma glucose "
                "in neonates — use laboratory plasma glucose to confirm). "
                "INSULIN: detectable insulin (even if 'low normal' value) during hypoglycaemia = "
                "INAPPROPRIATE (normal physiology suppresses insulin <2 mU/L during hypoglycaemia); "
                "note: some immunoassays may give 'undetectable' insulin in true CHI paradoxically — "
                "interpret with C-peptide. "
                "C-PEPTIDE: elevated in hyperinsulinism (endogenous insulin); "
                "suppressed in exogenous insulin administration (factitious hypoglycaemia). "
                "PLASMA 3-HYDROXYBUTYRATE (ketones): HYPOKETOTIC = hyperinsulinism hallmark; "
                "insulin suppresses lipolysis → no FFAs → no ketone bodies; "
                "ketones >1.5 mmol/L at time of hypoglycaemia virtually excludes hyperinsulinism. "
                "FREE FATTY ACIDS: low in hyperinsulinism (insulin anti-lipolytic). "
                "IGFBP1: suppressed by insulin — very low in CHI. "
                "AMMONIA: elevated in GLUD1-HI/HA (always measure). "
                "ACYLCARNITINES (blood spot): 3-OH-C4-carnitine elevated in HADH-CHI. "
                "GLUCAGON STIMULATION TEST (1 mg IV): "
                "glucose rise >1.7 mmol/L within 30 min = positive (confirms hepatic glycogen stores "
                "responsive to glucagon → consistent with hyperinsulinism maintaining glycogen); "
                "ABSENT response = FOXA2 (absent alpha-cell glucagon) or glycogen storage. "
                "AFTER DIAGNOSIS: urine organic acids (3-OH-glutaric acid → HADH); "
                "GLUD1/ABCC8/KCNJ11/GCK/HADH/HNF4A/FOXA2 gene panel; "
                "insulin secretory profile during prolonged fast."
            ),
            "Diazoxide — Pharmacology, Use, and Contraindications": (
                "Diazoxide is a K-ATP channel OPENER (activates the channel). "
                "MECHANISM: Diazoxide binds the MgADP-binding site on SUR1 (ABCC8) → "
                "stabilises the K-ATP open state → K+ efflux → "
                "membrane hyperpolarisation → Ca2+ channels closed → reduced insulin secretion. "
                "REQUIRES FUNCTIONAL SUR1 (ABCC8): "
                "Diazoxide INEFFECTIVE if: "
                "(1) Biallelic ABCC8/KCNJ11 LOF — no functional SUR1 to bind; "
                "(2) SLC16A1-EIHI — K-ATP channel is functional but ATP is generated from "
                "exercise-derived pyruvate; diazoxide-opened K-ATP cannot overcome the high ATP "
                "driven by ectopic MCT1-mediated pyruvate entry. "
                "Diazoxide EFFECTIVE if K-ATP channel is intact: "
                "GLUD1-HI/HA, GCK-HI, HADH-CHI, HNF4A-HI, FOXA2-HI. "
                "SIDE EFFECTS OF DIAZOXIDE: "
                "Fluid retention (oedema, hypertension): common — co-prescribe chlorothiazide "
                "(thiazide diuretic) to counteract fluid retention; "
                "Hypertrichosis: excess facial/body hair (especially in infants) — cosmetically distressing "
                "but benign; "
                "Pulmonary hypertension: rare but serious — monitor with echocardiography; "
                "Tachycardia; "
                "Hyperuricaemia (gout risk in older patients). "
                "DOSING: 5-15 mg/kg/day divided 2-3 times daily; "
                "always prescribe with chlorothiazide 7-10 mg/kg/day. "
                "DIAZOXIDE RESPONSE TRIAL: "
                "Give for 5 days minimum; "
                "Assess: able to fast for age-appropriate period without hypoglycaemia? "
                "Yes = responsive; No = KATP-CHI likely → 18F-DOPA PET."
            ),
        },
        "pharmacological_distinctions": [
            "Diazoxide (K-ATP opener, 5-15 mg/kg/day) — effective in GLUD1-HI/HA, GCK-HI, HADH, HNF4A, FOXA2 (K-ATP channel intact); INEFFECTIVE in ABCC8/KCNJ11 biallelic LOF (no SUR1 target) and SLC16A1-EIHI; always co-prescribe chlorothiazide 7-10 mg/kg/day to counteract fluid retention",
            "Octreotide (somatostatin analogue SC continuous infusion 5-20 μg/kg/day) — second-line for diazoxide-unresponsive CHI; reduces insulin secretion via SSTR2 in beta-cells; risk of necrotising enterocolitis in neonates — avoid as first-line in neonates; long-acting form (octreotide LAR) for older children",
            "Glucagon (1 mg IV/IM) — emergency only for acute hypoglycaemia; CONTRAINDICATED as emergency in FOXA2-HI (absent glucagon response from alpha-cells — will be INEFFECTIVE); safe and effective in ABCC8/KCNJ11 CHI as bridge to glucose infusion",
            "Glibenclamide / Glyburide (sulphonylurea, 0.05-0.5 mg/kg/day) — CURATIVE for KCNJ11-NDM; closes the constitutively open K-ATP channel in Kir6.2-GOF NDM; switches >90% of PNDM-KCNJ11 off insulin; also works in ABCC8-GOF NDM; start early for best neurological outcomes in DEND syndrome",
            "Nifedipine (calcium channel blocker, off-label) — experimental for CHI; blocks voltage-gated Ca2+ channels → reduces Ca2+ influx → reduces insulin; limited evidence; not first-line; occasionally used as adjunct in difficult-to-control CHI",
            "Sirolimus (mTOR inhibitor) — used in diffuse CHI refractory to octreotide as alternative to near-total pancreatectomy; reduces beta-cell mass and insulin secretion; significant immunosuppression risk; case series data only; centres vary in use",
            "Carbohydrate pre-loading (glucose gel, dextrose polymer) — the ONLY reliable management for SLC16A1-EIHI; oral carbohydrates 30 min before exercise buffer the post-exercise glucose fall; post-exercise glucose snack mandatory; diazoxide unhelpful",
            "Chlorothiazide (thiazide diuretic, 7-10 mg/kg/day) — always co-prescribed with diazoxide; counteracts diazoxide-induced fluid retention and oedema; enhances diazoxide effect (synergistic K-ATP effect via different mechanism); reduces risk of pulmonary hypertension",
            "GH replacement (SC, 0.025-0.05 mg/kg/day) — for FOXA2-HI with GH deficiency; compound hypoglycaemia from GH deficiency exacerbates insulin-mediated hypoglycaemia; replacing GH is essential",
            "Hydrocortisone (HC, 8-12 mg/m2/day) — for FOXA2-HI with ACTH deficiency (secondary adrenal insufficiency); ACTH-deficient hypocortisolaemia compounds the hyperinsulinism-driven hypoglycaemia; HC replacement is mandatory + sick day rules",
        ],
        "key_standards": [
            "ISPAD CHI Guideline (2022): critical sample protocol mandatory at glucose <3.0 mmol/L; diazoxide trial 5 days minimum with chlorothiazide; 18F-DOPA PET-CT mandatory before pancreatectomy for diazoxide-unresponsive CHI; parental mutation origin (paternal vs maternal) determines focal vs diffuse risk",
            "EAP/ESPE CHI European Reference Networks Standard: all CHI should be managed at or in consultation with a CHI specialist centre; genetic diagnosis (ABCC8, KCNJ11, GLUD1, GCK, HADH, SLC16A1, HNF4A, HNF1A, FOXA2) standard panel; 18F-DOPA PET in specialised nuclear medicine centre",
            "KCNJ11-NDM Sulphonylurea Transition Protocol: confirm KCNJ11 GOF mutation → start glibenclamide 0.05 mg/kg/day and titrate while overlapping with insulin → reduce insulin as glibenclamide doses increase → aim for complete insulin cessation; monitor HbA1c + CGM + neurological assessment; outcomes best if transition completed by age 6 months",
            "HNF4A-MODY1 Monitoring Standard: any child with HNF4A-HI who achieves euglycaemia in infancy → annual fasting glucose + HbA1c from age 10; MODY1 diagnosis when fasting glucose >7.0 mmol/L on two occasions or HbA1c >48 mmol/mol; sulphonylurea (gliclazide) first-line at low dose",
            "FOXA2 Pituitary Assessment: all FOXA2-CHI → pituitary MRI + morning cortisol + GH stimulation test + TFTs + IGF-1 at diagnosis; replace deficient hormones before diazoxide trial; anticipate absent glucagon response — glucagon stimulation test confirmatory; document FOXA2 glucagon absence on medical alert bracelet",
            "Ammonia Protocol in CHI Workup: ammonia MUST be measured at the time of hypoglycaemia (or at first diagnostic evaluation); fasting ammonia + post-protein ammonia both elevated in GLUD1-HI/HA; urea cycle disorder (UCD) excluded by different clinical context; HI/HA ammonia 60-200 μmol/L (lower than UCD crisis); GLUD1 sequencing confirmatory",
            "SLC16A1-EIHI Diagnosis Protocol: exercise-induced hypoglycaemia workup → cycle ergometer exercise test (15 min moderate intensity, fasting state) → glucose monitoring q10min for 90 min post-exercise → glucose nadir confirms EIHI; SLC16A1 PROMOTER sequencing (not WES alone); family members tested; emergency school plan: carbohydrate snack post-PE mandatory",
            "Near-Total Pancreatectomy — Post-Operative Monitoring: diffuse CHI → 95-98% pancreatectomy → monitor: post-op hypoglycaemia (residual beta-cells), subsequent diabetes (most patients by 20y), exocrine pancreatic insufficiency (EPI) requiring PERT — test faecal elastase annually; DEXA bone density; maintain surveillance into adulthood",
        ],
    }
