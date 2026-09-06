#!/usr/bin/env python3
"""Hereditary-MODY-Atlas — Complete 8-Gene Hereditary Monogenic Diabetes (MODY) Atlas
HNF1A  (hepatocyte nuclear factor 1-alpha; 631 aa; 12q24.31; AD;
         MODY3 — most common MODY (~50%); renal glucosuria; sulfonylureas highly effective (>98%);
         transition from insulin MANDATORY in misdiagnosed T1DM;
         seed SEED_BASE+0) ·
GCK    (glucokinase / hexokinase IV; 499 aa; 7p13; AD;
         MODY2 — 2nd most common; lifelong stable mild fasting hyperglycaemia 5.5–8 mmol/L;
         NO treatment needed (not progressive); NO macrovascular complications; NO microvascular complications;
         pregnancy exception — if foetus inherits GCK treat maternal glucose to prevent macrosomia;
         seed SEED_BASE+1) ·
HNF4A  (hepatocyte nuclear factor 4-alpha; 474 aa; 20q13.12; AD;
         MODY1 — similar phenotype to MODY3; neonatal macrosomia + transient hypoglycaemia PATHOGNOMONIC;
         elevated birth weight + neonatal diazoxide-responsive hypoglycaemia; sulfonylureas effective;
         seed SEED_BASE+2) ·
HNF1B  (hepatocyte nuclear factor 1-beta; 557 aa; 17q12; AD;
         MODY5/RCAD — renal cysts PATHOGNOMONIC + diabetes + genital tract anomalies TRIAD;
         pancreatic atrophy → exocrine insufficiency; INSULIN REQUIRED; sulfonylureas INEFFECTIVE;
         gout + hypomagnesaemia + elevated LFTs; 50% de novo mutations;
         seed SEED_BASE+3) ·
ABCC8  (ATP-binding cassette subfamily C member 8 / SUR1; 1581 aa; 11p15.1; AD;
         MODY12/NDM — SUR1 subunit of pancreatic K-ATP channel; most variable phenotype
         (neonatal hypoglycaemia → MODY → transient or permanent neonatal DM);
         diazoxide-responsive; oral sulfonylurea (glibenclamide) can replace insulin in NDM;
         seed SEED_BASE+4) ·
KCNJ11 (potassium inwardly-rectifying channel subfamily J member 11 / Kir6.2; 390 aa; 11p15.1; AD/AR;
         MODY13/NDM — Kir6.2 pore-forming subunit of K-ATP channel;
         permanent NDM responds to oral sulfonylurea in >90% (TREAT instead of insulin);
         DEND syndrome (GOF): Developmental delay + Epilepsy + Neonatal Diabetes;
         seed SEED_BASE+5) ·
INS    (insulin preprotein; 110 aa; 11p15.5; AD dominant-negative / AR recessive;
         MODY10/NDM — dominant-negative misfolded insulin → ER stress → beta-cell apoptosis;
         neonatal DM or adolescent MODY; AR biallelic = total insulin deficiency; exogenous insulin REQUIRED;
         seed SEED_BASE+6) ·
PDX1   (pancreatic and duodenal homeobox 1; 283 aa; 13q12.2; AR biallelic = pancreatic agenesis / AD heterozygous = MODY4;
         master pancreatic transcription factor (insulin + glucagon + somatostatin + exocrine);
         biallelic PDX1 LOF = total pancreatic agenesis (neonatal DM + exocrine insufficiency + NO pancreas on MRI);
         heterozygous AD = MODY4 (mild);
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1694–1701)
"""

import random

SEED_BASE = 1694

MODY_GENES = [
    # ── HNF1A — MODY3 — Most Common MODY (~50%) ─────────────────────────────
    {
        "gene": "HNF1A",
        "protein": "HNF1A — MODY3 AD — Hepatocyte Nuclear Factor 1-Alpha — Most Common MODY ~50% — Sulfonylurea Highly Effective — Renal Glucosuria — Transition from Insulin MANDATORY",
        "alias": (
            "HNF1A (hepatocyte nuclear factor 1-alpha); OMIM gene 142410; "
            "Maturity Onset Diabetes of the Young type 3 (MODY3) OMIM 600496. "
            "12q24.31; 631 aa; ~53 kDa; AD heterozygous LOF (haploinsufficiency). "
            "FUNCTION: HNF1A is a homeodomain transcription factor of the HNF1 subfamily, "
            "expressed in pancreatic beta-cells, liver, kidney, and intestine. "
            "In beta-cells, HNF1A regulates insulin gene transcription, glucose transporter GLUT2 (SLC2A2), "
            "glucokinase (GCK), and mitochondrial metabolism genes that couple glucose to insulin secretion. "
            "Loss of one HNF1A allele → progressive reduction in beta-cell mass and function → "
            "gradual-onset hyperglycaemia typically in the second or third decade. "
            "CLINICAL PHENOTYPE: "
            "Most common MODY subtype (~50% of all MODY); "
            "presentation: gradual-onset hyperglycaemia age 10–35 years; "
            "initially mild but progressive — eventually requires treatment; "
            "RENAL GLUCOSURIA: a defining HNF1A feature — HNF1A regulates renal tubular glucose threshold (SGLT2); "
            "HNF1A LOF lowers renal threshold → glucose in urine DESPITE non-diabetic plasma glucose; "
            "urine glucose positive even at normal plasma glucose in HNF1A carriers; "
            "useful diagnostic clue: new-onset diabetes + glucosuria at apparently normal glucose → test HNF1A/HNF4A. "
            "MISDIAGNOSIS: ~80% of MODY3 initially misdiagnosed as T1DM or T2DM; "
            "key differentiator from T1DM: C-peptide preserved (not autoimmune), HbA1c higher than expected, "
            "GAD/IA2/ZnT8 antibodies NEGATIVE; "
            "key differentiator from T2DM: lean body habitus, strong family history (AD three-generation pedigree), "
            "exquisite sulfonylurea sensitivity. "
            "SULFONYLUREA RESPONSE — TREATMENT OF CHOICE: "
            "HNF1A LOF → reduced beta-cell HNF1A → reduced K-ATP channel sensitivity and SUR1 expression; "
            "but beta-cells remain sulfonylurea-responsive via residual SUR1/K-ATP: "
            "low-dose gliclazide or glipizide → 98% respond with dramatic HbA1c reduction; "
            "starting dose 10–20x lower than standard T2DM dose; "
            "risk of HYPOGLYCAEMIA if standard T2DM dose used — start low (e.g. gliclazide 40 mg daily); "
            "INSULIN-TO-SULFONYLUREA TRANSITION: HNF1A patients on insulin (misdiagnosed T1DM) "
            "can successfully transition to sulfonylurea in >90% — genetic testing enables this; "
            "transition reduces insulin injections, improves quality of life, rarely fails. "
            "COMPLICATIONS: microvascular complications (retinopathy, nephropathy, neuropathy) develop "
            "if glycaemia poorly controlled — similar risk to T1DM/T2DM; "
            "no unique organ complications beyond diabetes; "
            "reduced HDL cholesterol is a HNF1A-specific finding (HNF1A regulates ApoA1). "
            "SURVEILLANCE: annual HbA1c, renal function, retinopathy screening (same as T1DM); "
            "cascade testing: first-degree relatives with 50% risk; "
            "preconception counselling: 50% transmission risk; gestational diabetes in carriers. "
            "PREGNANCY: early glucose monitoring; additional insulin if glucose rises above 5.5 mmol/L fasting; "
            "fetal genotype determines macrosomia risk (if fetus inherits HNF1A → normal insulin secretion). "
        ),
        "locus": "12q24.31",
        "aa": 631,
        "kDa": 53,
        "omim_gene": "142410",
        "omim_disease": "600496",
        "inheritance": "AD — Autosomal Dominant — haploinsufficiency; 50% offspring risk; de novo ~10%; strong family history 3+ generations",
        "gene_class": "Transcription Factor (homeodomain HNF1 subfamily); regulates insulin gene + GLUT2 + GCK + beta-cell metabolism; also expressed in liver (ApoA1, HDL) + kidney (renal glucose threshold SGLT2)",
        "key_alerts": [
            "HNF1A-SULFONYLUREA-FIRST-LINE-98PCT-EFFECTIVE: Sulfonylurea (gliclazide/glipizide) is first-line for HNF1A-MODY3; start at 10-20x lower dose than standard T2DM dose (risk of severe hypoglycaemia at standard dose); >98% dramatic response; transition insulin→sulfonylurea succeeds in >90%",
            "HNF1A-RENAL-GLUCOSURIA-DIAGNOSTIC-CLUE: HNF1A LOF lowers renal glucose threshold via SGLT2 regulation; positive urine glucose at normal or mildly elevated plasma glucose is pathognomonic; test urine glucose BEFORE diagnosing T1DM/T2DM in young patients",
            "HNF1A-MISDIAGNOSIS-80PCT-AS-T1DM: Antibody negative + C-peptide preserved + family history (3-generation AD) + sulfonylurea hyperresponsiveness = MODY3; genetic test is diagnostic; do NOT start insulin without MODY panel if antibodies negative",
            "HNF1A-REDUCED-HDL-HEPATIC-MARKER: HNF1A regulates ApoA1 production; HNF1A LOF → low ApoA1 → low HDL-C; low HDL in young-onset diabetes + family history → HNF1A test; supports lipid monitoring in confirmed carriers",
            "HNF1A-PREGNANCY-FETAL-GENOTYPE-GOVERNS: If fetus inherits HNF1A LOF → fetus has reduced insulin → maternal hyperglycaemia does NOT cause macrosomia (fetus also hyperglycaemic); if fetus wild-type → maternal hyperglycaemia → fetal hyperinsulinaemia → macrosomia; fetal genotype drives treatment intensity decision",
        ],
        "etiologies": [
            "Classic MODY3 (HNF1A LOF heterozygous) — progressive beta-cell dysfunction; onset age 10-30y; renal glucosuria + low HDL + exquisite SU sensitivity; antibody negative; C-peptide preserved until advanced disease",
            "HNF1A-MODY Presenting as Gestational Diabetes — first presentation at pregnancy; gestational diabetes with positive family history and low renal threshold; genetic testing enables sulfonylurea vs insulin decision in pregnancy",
            "Late-Onset HNF1A-MODY (age >40y) — milder or delayed penetrance variant; may mimic T2DM; key clue: lean body, no metabolic syndrome, strong AD family history, SU hypersensitivity",
            "HNF1A de novo MODY3 — no family history; ~10% of HNF1A-MODY; genetic testing still yields diagnostic answer; counselling for offspring risk",
            "HNF1A VUS (Variant of Uncertain Significance) — increasing with panel testing; functional studies needed; treat empirically if clinical MODY phenotype present",
        ],
        "stats": {
            "mean_dx_age": 23,
            "prevalence_pct_MODY": 50,
            "sulfonylurea_response_pct": 98,
            "misdiagnosed_T1DM_pct": 80,
            "mean_dx_delay_months": 36,
            "renal_glucosuria_pct": 90,
            "low_HDL_pct": 65,
        },
        "dx_delay_distribution": {
            "0-6_mo": "5%", "6-24_mo": "30%", "2-5_yr": "40%", ">5_yr": "25%"
        },
    },
    # ── GCK — MODY2 — 2nd Most Common — Lifelong Stable Mild Hyperglycaemia ──
    {
        "gene": "GCK",
        "protein": "GCK — MODY2 AD — Glucokinase Hexokinase IV — Stable Mild Fasting Hyperglycaemia 5.5-8 mmol/L — NO Treatment Needed — NO Complications — Pregnancy Exception",
        "alias": (
            "GCK (glucokinase / hexokinase IV); OMIM gene 138079; "
            "Maturity Onset Diabetes of the Young type 2 (MODY2) OMIM 125851. "
            "7p13; 499 aa; ~52 kDa; AD heterozygous LOF (haploinsufficiency). "
            "FUNCTION: Glucokinase (GCK) is the glucose-sensing enzyme of pancreatic beta-cells and hepatocytes. "
            "Unlike hexokinase I-III (low Km, saturated at normal glucose), GCK has HIGH Km for glucose "
            "(≈8 mmol/L) and is not inhibited by its product glucose-6-phosphate — it acts as a 'glucose sensor.' "
            "GCK phosphorylates glucose → glucose-6-phosphate → glycolysis → ATP production → K-ATP channel closure "
            "→ membrane depolarisation → Ca2+ influx → insulin secretion. "
            "THRESHOLD EFFECT: GCK LOF shifts the glucose threshold for insulin secretion upward; "
            "half-normal GCK activity → insulin secretion 'set point' raised to ~5.5–8 mmol/L "
            "(normal set point ~4.5–5 mmol/L); result: persistently elevated fasting glucose "
            "that is STABLE and NON-PROGRESSIVE — the elevated set point is the new normal. "
            "CLINICAL PHENOTYPE: "
            "2nd most common MODY (~30–40%); "
            "fasting glucose: 5.5–8.0 mmol/L (consistently mildly elevated from birth); "
            "HbA1c: typically 5.8–7.6% (just above normal but stable); "
            "NO progressive deterioration — unlike T1DM and T2DM; "
            "incidentally found at routine screening or pregnancy glucose tolerance test. "
            "CRITICAL MANAGEMENT RULE — NO TREATMENT NEEDED: "
            "GCK-MODY2 should NOT be treated with antidiabetic agents (sulfonylureas, insulin, metformin); "
            "the mild hyperglycaemia is set-point, not dysregulation — treatment does not prevent complications; "
            "all studies show no microvascular complications at this glucose level; "
            "treatment causes iatrogenic hypoglycaemia without benefit; "
            "HOWEVER: if a patient carries GCK-MODY AND has another reason for T2DM, "
            "the GCK-MODY component does not need treatment (the overlay T2DM component might). "
            "EXCEPTION — PREGNANCY: "
            "If MOTHER carries GCK-MODY + fetus is wild-type (inherits normal GCK): "
            "maternal hyperglycaemia (5.5–8 mmol/L) → foetal NORMAL GCK → "
            "foetal hyperinsulinaemia → macrosomia; treat mother to protect fetus. "
            "If MOTHER carries GCK-MODY + fetus also carries GCK-MODY: "
            "foetal glucose threshold also raised → foetus not hyperinsulinaemic → no macrosomia; "
            "overly strict maternal glucose control → foetal hypoglycaemia (foetal GCK raised threshold); "
            "foetal genotype determines treatment strategy in pregnancy. "
            "DIAGNOSIS: "
            "Fasting glucose 5.5–8.0 mmol/L from childhood; "
            "OGTT: glucose rise <3.5 mmol/L above fasting (flat curve — the sensor is shifted uniformly); "
            "HbA1c 5.8–7.6%; C-peptide preserved; antibodies negative; "
            "family history: at least one parent with similar mild asymptomatic hyperglycaemia (may never have been diagnosed); "
            "GCK heterozygous pathogenic variant on panel."
        ),
        "locus": "7p13",
        "aa": 499,
        "kDa": 52,
        "omim_gene": "138079",
        "omim_disease": "125851",
        "inheritance": "AD — Autosomal Dominant — haploinsufficiency; 50% offspring risk; lifelong from birth; AR biallelic = permanent neonatal DM (rare)",
        "gene_class": "Enzyme (hexokinase IV; rate-limiting glucose sensor in beta-cell; not product-inhibited; high Km ~8 mmol/L); regulates insulin secretion threshold and hepatic glucose metabolism",
        "key_alerts": [
            "GCK-MODY2-NO-TREATMENT-EXCEPT-PREGNANCY: GCK-MODY2 should NOT be treated with any antidiabetic agent; fasting glucose 5.5-8 mmol/L is a set-point, not progressive disease; no microvascular/macrovascular complications at this level; treatment causes hypoglycaemia without benefit",
            "GCK-PREGNANCY-FETAL-GENOTYPE-CRITICAL: If fetus is GCK wild-type → maternal hyperglycaemia causes fetal hyperinsulinaemia → macrosomia → treat mother; if fetus also carries GCK LOF → no fetal hyperinsulinaemia → no macrosomia → do NOT overtighten glucose control (risk fetal hypoglycaemia)",
            "GCK-MISDIAGNOSIS-GESTATIONAL-DIABETES: GCK-MODY2 is commonly misdiagnosed as gestational DM in pregnancy; lifelong stable mild hyperglycaemia was pre-existing NOT new-onset; post-partum glucose persists → genetic test",
            "GCK-FLAT-OGTT-DIAGNOSTIC-CLUE: OGTT glucose rise <3.5 mmol/L above fasting (flat curve) is characteristic; hepatic GCK suppresses post-load hepatic glucose output proportionally; contrast MODY3 (larger post-load glucose excursion)",
            "GCK-STABLE-FROM-BIRTH-NOT-PROGRESSIVE: Unlike T1DM/T2DM/MODY3, GCK-MODY does not progress; HbA1c stable for decades; incidentally found; normal life expectancy; only pregnancy requires active management",
        ],
        "etiologies": [
            "Classic GCK-MODY2 (heterozygous LOF) — incidental stable mild hyperglycaemia found at screening; flat OGTT; no symptoms; parent with same finding; no treatment; normal life expectancy",
            "GCK-MODY2 Presenting as Gestational Diabetes — glucose elevation first noticed in pregnancy; misdiagnosed as GDM; MODY2 persists post-partum; cascade test family",
            "GCK-MODY2 Misdiagnosed as T2DM — treated unnecessarily with metformin/SU; recognise on reconsideration; stop treatment; retest family",
            "GCK-MODY2 Misdiagnosed as T1DM — young patient + mild hyperglycaemia + no antibodies; GCK panel testing diagnostic; avoid insulin",
            "AR Biallelic GCK LOF (permanent neonatal DM) — extremely rare; both alleles lost → complete glucokinase absence → permanent neonatal DM from birth requiring insulin (not MODY, a distinct condition)",
        ],
        "stats": {
            "mean_dx_age": 35,
            "prevalence_pct_MODY": 35,
            "no_treatment_pct": 92,
            "stable_HbA1c_pct": 95,
            "mean_dx_delay_months": 84,
            "misdiagnosed_GDM_pct": 55,
            "macrosomia_WT_fetus_pct": 40,
        },
        "dx_delay_distribution": {
            "0-6_mo": "3%", "6-24_mo": "15%", "2-5_yr": "35%", ">5_yr": "47%"
        },
    },
    # ── HNF4A — MODY1 — Neonatal Macrosomia + Transient Hypoglycaemia ────────
    {
        "gene": "HNF4A",
        "protein": "HNF4A — MODY1 AD — Hepatocyte Nuclear Factor 4-Alpha — Neonatal Macrosomia + Transient Hypoglycaemia PATHOGNOMONIC — Sulfonylurea Effective — Similar to MODY3",
        "alias": (
            "HNF4A (hepatocyte nuclear factor 4-alpha); OMIM gene 600281; "
            "Maturity Onset Diabetes of the Young type 1 (MODY1) OMIM 125850. "
            "20q13.12; 474 aa; ~55 kDa; AD heterozygous LOF. "
            "FUNCTION: HNF4A is a nuclear receptor transcription factor (NR2A1) in the steroid/thyroid receptor superfamily. "
            "In pancreatic beta-cells, HNF4A is activated by HNF1A (reciprocal regulatory loop): "
            "HNF4A → HNF1A target genes; HNF1A → HNF4A target genes. "
            "HNF4A regulates fatty acid oxidation genes in beta-cells, which supply acetyl-CoA for ATP generation "
            "and enhance glucose-stimulated insulin secretion. "
            "HNF4A also regulates GLUT2 in liver and apolipoproteins (ApoC-III, ApoA-IV, ApoB). "
            "CLINICAL PHENOTYPE: "
            "HNF4A-MODY1 is similar to HNF1A-MODY3 but with TWO CARDINAL DISTINCTIONS: "
            "1) NEONATAL MACROSOMIA: mean birth weight +800g above normal; "
            "HNF4A LOF in utero → immature beta-cells paradoxically over-secrete insulin during foetal development "
            "(HNF4A normally limits foetal beta-cell insulin secretion); "
            "excess foetal insulin → macrosomia; paradox — LESS HNF4A function → MORE insulin in the foetus. "
            "2) TRANSIENT NEONATAL HYPOGLYCAEMIA: the neonatal period features diazoxide-responsive hypoglycaemia; "
            "hyperinsulinaemic hypoglycaemia in the first days/weeks of life; "
            "resolves spontaneously as foetal-to-adult beta-cell transition occurs; "
            "this neonatal hypoglycaemia phase is followed by MODY-type hyperglycaemia in adolescence/early adulthood. "
            "KEY TEACHING POINT: if an infant presents with hyperinsulinaemic hypoglycaemia that is diazoxide-responsive "
            "AND the family history reveals a parent with mild diabetes → TEST HNF4A FIRST. "
            "DIABETES PHASE: onset age 15–40 years (later than MODY3); progressive; "
            "sulfonylurea highly effective (same mechanism as MODY3); "
            "low renal glucose threshold (mild glucosuria, less prominent than MODY3); "
            "LFTs: HNF4A regulates hepatic lipid/apoprotein genes → "
            "elevated triglycerides + low ApoA1/HDL (similar to HNF1A but involving different apolipoproteins)."
        ),
        "locus": "20q13.12",
        "aa": 474,
        "kDa": 55,
        "omim_gene": "600281",
        "omim_disease": "125850",
        "inheritance": "AD — Autosomal Dominant — haploinsufficiency; 50% offspring risk; neonatal phenotype + adult MODY phase (biphasic)",
        "gene_class": "Nuclear receptor transcription factor (steroid/thyroid receptor superfamily NR2A1); upstream regulator of HNF1A targets + fatty acid oxidation in beta-cell; apolipoproteins ApoB/ApoC-III/ApoA-IV in liver",
        "key_alerts": [
            "HNF4A-NEONATAL-MACROSOMIA-HYPOGLYCAEMIA-PATHOGNOMONIC: HNF4A LOF causes macrosomia (>800g excess) + diazoxide-responsive neonatal hypoglycaemia; family history of parent with diabetes → HNF4A FIRST; biphasic: neonatal hypoglycaemia → adult MODY diabetes",
            "HNF4A-SULFONYLUREA-FIRST-LINE-SAME-AS-MODY3: Sulfonylurea highly effective for HNF4A-MODY1 diabetes phase; start at low dose (hypoglycaemia risk same as MODY3); transition from insulin possible in >85% of misdiagnosed HNF4A patients",
            "HNF4A-DIAZOXIDE-NEONATAL-PHASE: Neonatal hyperinsulinaemic hypoglycaemia is diazoxide-responsive; do NOT diagnose permanent neonatal DM — it is transient; genetic test HNF4A in all hyperinsulinaemic neonates with macrosomia + diabetic parent",
            "HNF4A-APOLIPOPROTEIN-HEPATIC-MARKER: HNF4A regulates ApoB, ApoC-III, ApoA-IV; elevated triglycerides + reduced ApoA1 in HNF4A-MODY; lipid panel is supportive but non-specific",
            "HNF4A-MODY1-LESS-COMMON-THAN-MODY3-SIMILAR-TREATMENT: MODY1 ~5-10% of all MODY; same treatment approach as MODY3; neonatal findings differentiate; genetic panel covers both genes",
        ],
        "etiologies": [
            "Classic HNF4A-MODY1 (heterozygous LOF) — onset age 15-40y; progressive diabetes; sulfonylurea-responsive; preceded by neonatal macrosomia and transient hypoglycaemia in 30-50%",
            "HNF4A Neonatal Hyperinsulinaemia — diazoxide-responsive neonatal hypoglycaemia in macrosomic infant; parent with mild diabetes; genetic test confirms HNF4A; resolves spontaneously in weeks-months",
            "HNF4A Misdiagnosed as T1DM — antibody-negative young-onset diabetes + family history; sulfonylurea switch succeeds",
            "HNF4A de novo Variant — no family history; neonatal hypoglycaemia + later MODY without parental history; genetic confirmation needed",
            "HNF4A-MODY in Pregnancy — maternal MODY1 + fetal HNF4A inheritance → fetal macrosomia risk; monitor with serial ultrasound",
        ],
        "stats": {
            "mean_dx_age": 28,
            "prevalence_pct_MODY": 7,
            "sulfonylurea_response_pct": 90,
            "neonatal_macrosomia_pct": 45,
            "mean_dx_delay_months": 40,
            "neonatal_hypoglycaemia_pct": 35,
            "misdiagnosed_T1DM_pct": 65,
        },
        "dx_delay_distribution": {
            "0-6_mo": "8%", "6-24_mo": "30%", "2-5_yr": "38%", ">5_yr": "24%"
        },
    },
    # ── HNF1B — MODY5/RCAD — Renal Cysts + Diabetes + Genital Tract ──────────
    {
        "gene": "HNF1B",
        "protein": "HNF1B — MODY5 AD — HNF1-Beta — Renal Cysts PATHOGNOMONIC + Diabetes + Genital Tract Anomalies TRIAD (RCAD) — Pancreatic Atrophy — INSULIN Required — SU INEFFECTIVE — 50% De Novo",
        "alias": (
            "HNF1B (hepatocyte nuclear factor 1-beta / vHNF1 / TCF2); OMIM gene 189907; "
            "Renal cysts and diabetes syndrome (RCAD / MODY5) OMIM 137920. "
            "17q12; 557 aa; ~68 kDa; AD heterozygous LOF; ~50% de novo mutations. "
            "FUNCTION: HNF1B is a POU-homeodomain transcription factor expressed in the developing kidney, "
            "liver, pancreas, and genital tract. HNF1B dimerises with HNF1A; "
            "HNF1B regulates branching morphogenesis in the kidney (ureteric bud development) and "
            "pancreatic ductal differentiation + endocrine/exocrine specification. "
            "HNF1B LOF disrupts renal tubular differentiation → renal cysts + structural anomalies; "
            "disrupts pancreatic development → pancreatic atrophy + islet loss + exocrine insufficiency. "
            "CLINICAL PHENOTYPE — RCAD TRIAD: "
            "1) RENAL CYSTS (most consistent finding — present in >70%): "
            "bilateral renal cysts from foetal life; "
            "renal dysfunction may be detected by prenatal ultrasound (bright echogenic kidneys); "
            "renal function: variable from normal to CKD (GFR <60); "
            "horseshoe kidney, collecting system anomalies, glomerulocystic kidney also reported; "
            "2) DIABETES (MODY5): onset in second or third decade; "
            "pancreatic atrophy on MRI/CT — a PATHOGNOMONIC radiological finding; "
            "pancreatic exocrine insufficiency in 40%: faecal elastase low, steatorrhoea, fat-soluble vitamin deficiency; "
            "INSULIN is the REQUIRED treatment: HNF1B-MODY diabetes does NOT respond to sulfonylureas; "
            "the beta-cell mass is structurally reduced (not functionally inhibited — SU target is absent); "
            "3) GENITAL TRACT ANOMALIES: "
            "FEMALES: uterine aplasia/agenesis, Müllerian duct aplasia (bicornuate uterus, uterine septum); "
            "MALES: epididymal cysts, vas deferens anomalies. "
            "ADDITIONAL FEATURES: "
            "GOUT: HNF1B regulates renal urate transporter URAT1; LOF → reduced urate excretion → hyperuricaemia → gout; "
            "HYPOMAGNESAEMIA: HNF1B regulates FXYD2 (renal Na/K-ATPase gamma) → renal magnesium wasting → hypomagnesaemia; "
            "ELEVATED LFTs: mild hepatic involvement; "
            "ELECTROLYTE SURVEILLANCE: magnesium supplementation often required. "
            "DE NOVO MUTATIONS: ~50% of HNF1B cases have de novo mutations (no family history); "
            "this distinguishes HNF1B from other MODY types; "
            "whole gene deletions (17q12 deletion syndrome) account for ~40% of HNF1B cases; "
            "MLPA or gene dosage is required alongside sequencing. "
            "DIAGNOSIS PATHWAY: "
            "Renal cysts + diabetes + family history of either → HNF1B FIRST; "
            "prenatal cystic kidneys → test HNF1B in parents; "
            "pancreatic atrophy on imaging + diabetes → HNF1B; "
            "gout + hypomagnesaemia + diabetes in young lean patient → HNF1B."
        ),
        "locus": "17q12",
        "aa": 557,
        "kDa": 68,
        "omim_gene": "189907",
        "omim_disease": "137920",
        "inheritance": "AD — Autosomal Dominant — haploinsufficiency; ~50% de novo mutations; 17q12 deletion syndrome (includes HNF1B deletion) in ~40%",
        "gene_class": "Transcription factor (POU homeodomain; dimerises with HNF1A); kidney branching morphogenesis + pancreatic ductal differentiation + genital tract development; renal URAT1 (urate) + FXYD2 (Mg2+) regulation",
        "key_alerts": [
            "HNF1B-RCAD-TRIAD-RENAL-CYSTS-DIABETES-GENITALTRACT: HNF1B is the ONLY MODY with renal cysts as a core feature; bilateral renal cysts + diabetes in any age → HNF1B FIRST; genital tract anomalies (female Müllerian, male epididymal) complete the RCAD triad",
            "HNF1B-SULFONYLUREA-ABSOLUTELY-INEFFECTIVE-INSULIN-REQUIRED: HNF1B diabetes requires INSULIN; beta-cell mass is structurally reduced (pancreatic atrophy); sulfonylureas target intact beta-cell SUR1 K-ATP — not applicable here; prescribing SU in HNF1B causes no response and delay in appropriate treatment",
            "HNF1B-PANCREATIC-ATROPHY-PATHOGNOMONIC-MRI: Pancreatic atrophy on MRI/CT is near-pathognomonic for HNF1B; pancreatic exocrine insufficiency in ~40% (faecal elastase low, steatorrhoea, fat-soluble vitamin deficiency) — prescribe pancreatic enzyme replacement therapy",
            "HNF1B-GOUT-HYPOMAGNESAEMIA-SURVEILLANCE-MANDATORY: HNF1B LOF → renal urate retention (gout — young lean patient) + renal Mg2+ wasting (hypomagnesaemia → tetany, cardiac arrhythmia); monitor urate + Mg2+ at every visit; supplement Mg2+ if low",
            "HNF1B-50PCT-DE-NOVO-NO-FAMILY-HISTORY-DO-NOT-DISMISS: ~50% of HNF1B are de novo mutations; absence of family history does NOT exclude HNF1B; renal cysts in young patient + diabetes = test regardless of family history; 17q12 deletion by MLPA in ~40% (not detectable by sequencing alone)",
        ],
        "etiologies": [
            "Classic RCAD (HNF1B LOF heterozygous) — renal cysts + MODY5 diabetes + genital anomalies triad; gout + hypomagnesaemia; pancreatic atrophy; insulin required; 50% family history positive",
            "HNF1B de novo Mutation — no family history; isolated renal cysts in child; parent testing negative; genetic confirmation by sequencing + MLPA",
            "17q12 Deletion Syndrome — whole gene deletion (~1.7 Mb); MODY5 + renal features + neurodevelopmental features (autism, intellectual disability); gene dosage (MLPA/aCGH) required for diagnosis",
            "HNF1B Prenatal Presentation — foetal echogenic/cystic kidneys on antenatal scan; parents offered HNF1B testing; predicts postnatal diabetes risk",
            "HNF1B Renal-Limited Variant — minimal diabetes expression but significant cystic nephropathy; may progress to renal failure without diabetes diagnosis",
        ],
        "stats": {
            "mean_dx_age": 26,
            "prevalence_pct_MODY": 5,
            "renal_cysts_pct": 72,
            "pancreatic_atrophy_pct": 65,
            "insulin_required_pct": 95,
            "mean_dx_delay_months": 48,
            "de_novo_pct": 50,
            "hypomagnesaemia_pct": 45,
        },
        "dx_delay_distribution": {
            "0-6_mo": "10%", "6-24_mo": "28%", "2-5_yr": "35%", ">5_yr": "27%"
        },
    },
    # ── ABCC8 — MODY12 / NDM — SUR1 K-ATP Channel — Diazoxide-Responsive ────
    {
        "gene": "ABCC8",
        "protein": "ABCC8 — MODY12 AD — SUR1 Sulfonylurea Receptor 1 — K-ATP Channel Regulatory Subunit — Variable Phenotype: Neonatal Hypoglycaemia → MODY → NDM — Diazoxide-Responsive — Oral SU Replaces Insulin NDM",
        "alias": (
            "ABCC8 (ATP-binding cassette subfamily C member 8 / SUR1 sulfonylurea receptor 1); "
            "OMIM gene 600509; K-ATP channel disease (MODY12 / CHI / NDM) OMIM 601820. "
            "11p15.1; 1581 aa; ~177 kDa; AD (GOF → hypoglycaemia; LOF → NDM); "
            "also AR biallelic GOF = congenital hyperinsulinism (CHI). "
            "FUNCTION: ABCC8 encodes SUR1 (sulfonylurea receptor 1), the regulatory subunit of the "
            "pancreatic ATP-sensitive potassium (K-ATP) channel. "
            "The K-ATP channel is an octameric complex of four SUR1 subunits (ABCC8) + four Kir6.2 pore subunits (KCNJ11). "
            "Channel gating: HIGH ATP → channel closes → beta-cell depolarises → insulin secretion; "
            "LOW ATP → channel opens → beta-cell hyperpolarised → no insulin secretion. "
            "ABCC8/SUR1 contains nucleotide-binding domains (NBD1, NBD2) that sense ADP and MgADP; "
            "SUR1 is the DRUG TARGET for sulfonylureas (gliclazide, glibenclamide → close SUR1/K-ATP → "
            "force insulin secretion regardless of glucose) and diazoxide (opens SUR1/K-ATP → suppresses insulin secretion). "
            "PHENOTYPIC SPECTRUM — MOST VARIABLE OF ALL MODY GENES: "
            "GOF mutations (SUR1 remains open → K-ATP always open → no insulin secretion): "
            "Neonatal DM (transient or permanent) — diazoxide-responsive if GOF; "
            "Transition: permanent NDM → glibenclamide (sulfonylurea) REPLACES INSULIN in >70% of GOF NDM; "
            "LOF mutations (SUR1 cannot open → K-ATP always closed → insulin always secreted): "
            "Congenital hyperinsulinism (CHI) — diazoxide-responsive if SUR1 LOF; "
            "focal vs diffuse CHI differentiation by 18F-DOPA-PET before surgery; "
            "MODY12: AD heterozygous GOF/LOF → variable adult-onset diabetes. "
            "SULFONYLUREA PRECISION PHARMACOLOGY: "
            "Glibenclamide (glyburide) is the preferred SU for ABCC8/KCNJ11 NDM transition; "
            "binds SUR1 NBD with high affinity → closes K-ATP → restores insulin secretion; "
            "transition from insulin to glibenclamide succeeds in >70% of GOF-NDM patients; "
            "remarkable neurological benefits beyond glycaemia: "
            "SU also close K-ATP in neurons → reduces glutamate excitotoxicity → "
            "DEND syndrome children on SU show improved motor, cognition, and seizure control."
        ),
        "locus": "11p15.1",
        "aa": 1581,
        "kDa": 177,
        "omim_gene": "600509",
        "omim_disease": "601820",
        "inheritance": "AD GOF (NDM/MODY12) / AD LOF or AR biallelic LOF (CHI) — same gene, opposite phenotypes depending on mutation direction",
        "gene_class": "ABC transporter/regulatory subunit (SUR1 — sulfonylurea receptor 1); regulatory subunit of K-ATP channel octamer (4×SUR1 + 4×Kir6.2); NBD1/NBD2 bind sulfonylureas + diazoxide + MgADP",
        "key_alerts": [
            "ABCC8-NDM-SULFONYLUREA-REPLACES-INSULIN: Permanent neonatal DM due to ABCC8 GOF → try oral glibenclamide (high-dose); >70% transition from insulin → monitor HbA1c and hypoglycaemia; remarkable additional neurological improvement in DEND patients",
            "ABCC8-DIAZOXIDE-TEST-FOR-PHENOTYPE: All ABCC8 variants: perform diazoxide test; GOF mutations → diazoxide opens K-ATP → suppresses excess insulin secretion (hypoglycaemia) or does not close enough (hyperglycaemia); LOF (CHI) → diazoxide-responsive (opens channel) vs diazoxide-unresponsive (KATP-independent CHI needs surgery)",
            "ABCC8-FOCAL-CHI-18F-DOPA-PET-BEFORE-SURGERY: Diazoxide-unresponsive CHI (LOF) → 18F-DOPA-PET scan before surgery; focal lesion → limited pancreatectomy (curative); diffuse → near-total pancreatectomy (post-op DM inevitable); PET is mandatory to avoid unnecessary total pancreatectomy",
            "ABCC8-GOF-NDM-TRANSITION-PROTOCOL: Gradually introduce glibenclamide while maintaining insulin; increase glibenclamide q7d; reduce insulin proportionally; transition complete when insulin discontinued; monitor HbA1c + hypoglycaemia; failures (30%): continue insulin + SU combination",
            "ABCC8-MODY12-ADULT-ONSET: Heterozygous AD ABCC8 variants can present as adult-onset MODY (MODY12); phenotype milder than NDM; sulfonylurea treatment effective; genetic panel recommended in antibody-negative young-onset DM",
        ],
        "etiologies": [
            "ABCC8 GOF Permanent NDM (pNDM) — neonatal DM diagnosed in first 6 months; insulin-dependent at birth; transition to glibenclamide in >70%; neurological benefits if DEND features",
            "ABCC8 GOF Transient NDM (tNDM) — resolves by 18 months; may relapse in adolescence/adulthood; genetic identification allows surveillance and SU-ready treatment",
            "ABCC8 LOF Congenital Hyperinsulinism Diazoxide-Responsive — CHI detected in neonate; diazoxide + feeds control hypoglycaemia; focal vs diffuse differentiation by 18F-DOPA-PET",
            "ABCC8 LOF Congenital Hyperinsulinism Diazoxide-Unresponsive — near-total pancreatectomy required; post-operative DM is expected and should be anticipated and managed",
            "ABCC8 Adult-Onset MODY12 — heterozygous AD variant; mild adult DM; antibody negative; family history of DM; sulfonylurea-sensitive",
        ],
        "stats": {
            "mean_dx_age": 5,
            "prevalence_pct_MODY": 3,
            "sulfonylurea_transition_pct_NDM": 70,
            "diazoxide_responsive_CHI_pct": 50,
            "mean_dx_delay_months": 6,
            "DEND_neurological_pct": 20,
            "neonatal_onset_pct": 80,
        },
        "dx_delay_distribution": {
            "0-6_mo": "65%", "6-24_mo": "25%", "2-5_yr": "7%", ">5_yr": "3%"
        },
    },
    # ── KCNJ11 — MODY13 / NDM — Kir6.2 Pore — DEND Syndrome ────────────────
    {
        "gene": "KCNJ11",
        "protein": "KCNJ11 — MODY13/NDM AD — Kir6.2 Inwardly-Rectifying K+ Pore Subunit — K-ATP Channel — Permanent NDM → Oral Sulfonylurea >90% — DEND Syndrome GOF — Developmental Delay + Epilepsy + Neonatal DM",
        "alias": (
            "KCNJ11 (potassium inwardly-rectifying channel subfamily J member 11 / Kir6.2); "
            "OMIM gene 600937; Permanent neonatal diabetes mellitus (PNDM/NDM) OMIM 606176. "
            "11p15.1; 390 aa; ~45 kDa; AD GOF (NDM/MODY13). "
            "FUNCTION: KCNJ11 encodes Kir6.2, the pore-forming subunit of the K-ATP channel. "
            "Kir6.2 forms an inwardly-rectifying K+ channel; four Kir6.2 subunits assemble with "
            "four SUR1 subunits (ABCC8) to form the functional K-ATP octamer. "
            "Kir6.2 contains the ATP-binding pocket (inhibitory): "
            "High intracellular ATP → binds Kir6.2 → closes K-ATP → beta-cell depolarises → insulin secretion; "
            "Low ATP (fasting) → K-ATP opens → K+ efflux → hyperpolarisation → NO insulin secretion. "
            "GOF mutations in Kir6.2: "
            "1) REDUCED ATP sensitivity: channel does not close even at high ATP; "
            "2) Kir6.2 pore stays open → beta-cell remains hyperpolarised → no insulin secretion → neonatal DM. "
            "PHENOTYPIC SPECTRUM: "
            "Permanent NDM (most severe GOF): diabetes diagnosed in first 6 months of life; "
            "requires treatment from birth; CRITICAL: permanent NDM diagnosed in first year of life "
            "MUST have KCNJ11/ABCC8 testing BEFORE committing to lifelong insulin; "
            "SULFONYLUREA (GLIBENCLAMIDE) TREATMENT — HALLMARK: "
            ">90% of KCNJ11 permanent NDM patients transition from insulin to oral glibenclamide; "
            "this is the most dramatic pharmacogenomics success in neonatal medicine; "
            "mechanism: glibenclamide closes K-ATP via SUR1 — does NOT require ATP-binding; "
            "bypasses the defective ATP-sensing; forces channel closure → restores insulin secretion. "
            "DEND SYNDROME (most severe GOF — e.g. R201H, V59M, Q52R): "
            "Developmental delay + Epilepsy + Neonatal Diabetes; "
            "K-ATP channels are also expressed in CNS neurons; "
            "severe GOF → neuronal K-ATP constitutively open → "
            "neuronal hyperpolarisation → reduced neurotransmitter release → "
            "developmental delay, seizures (epilepsy), hypotonia; "
            "sulfonylurea crosses blood-brain barrier → closes neuronal K-ATP → "
            "DRAMATIC IMPROVEMENT in seizures, motor, and cognition in DEND; "
            "start SU transition as soon as NDM diagnosed — every month matters for neurodevelopment. "
            "INTERMEDIATE DEND (iDEND): V59A, G53D → mild neurological features without florid epilepsy."
        ),
        "locus": "11p15.1",
        "aa": 390,
        "kDa": 45,
        "omim_gene": "600937",
        "omim_disease": "606176",
        "inheritance": "AD GOF (NDM/MODY13) — de novo in severe mutations (R201H, V59M); AD in familial MODY13; AR biallelic LOF → CHI (same gene, opposite phenotype)",
        "gene_class": "Inwardly-rectifying K+ channel pore (Kir6.2); ATP-binding pocket (inhibitory); forms octameric K-ATP channel with 4×SUR1 (ABCC8); expressed in beta-cells + CNS neurons + cardiac muscle + smooth muscle",
        "key_alerts": [
            "KCNJ11-NDM-SULFONYLUREA-REPLACE-INSULIN->90PCT: Permanent NDM from KCNJ11 GOF: trial oral glibenclamide IMMEDIATELY; >90% success rate; do NOT accept lifelong insulin without KCNJ11/ABCC8 testing in any neonatal DM; transition improves HbA1c, quality of life, and neurodevelopment",
            "KCNJ11-DEND-SYNDROME-URGENT-SU-START: DEND (R201H/V59M) — Developmental delay + Epilepsy + Neonatal DM; sulfonylurea crosses BBB → closes neuronal K-ATP → reduces seizures + improves cognition and motor function; START SU ASAP — neurodevelopmental window cannot be recovered",
            "KCNJ11-NDM-GENETIC-TEST-MANDATORY-BEFORE-INSULIN-COMMITTED: Any infant with DM in first 6 months MUST have KCNJ11+ABCC8 testing (and ABCC8) before discharge on insulin; incidence of neonatal DM due to K-ATP GOF ~50% of all neonatal DM; missing this diagnosis = avoidable lifelong insulin dependency",
            "KCNJ11-AR-BIALLELIC-LOF-CHI: Biallelic AR KCNJ11 LOF → congenital hyperinsulinism (same mechanism as ABCC8 LOF but affects pore not regulatory subunit); diazoxide-unresponsive (Kir6.2 mutation not sensitive to diazoxide effect); near-total pancreatectomy if severe",
            "KCNJ11-GLUCOSE-MONITORING-TRANSITION: During insulin→SU transition, monitor blood glucose daily; SU dose escalation q7d; insulin wean proportional; transition typically complete in 2-6 weeks; if transition fails (10%) maintain insulin + SU combination",
        ],
        "etiologies": [
            "KCNJ11 GOF Permanent NDM (pNDM) — neonatal DM; glibenclamide-responsive >90%; genetic testing mandatory before insulin commitment",
            "KCNJ11 DEND Syndrome (severe GOF R201H/V59M) — NDM + seizures + developmental delay; urgent SU treatment; most dramatic neurological benefits from SU crossover",
            "KCNJ11 Intermediate DEND (iDEND) — NDM + mild neurological features; SU still beneficial for neurodevelopment; less severe than full DEND",
            "KCNJ11 Transient NDM — remits by 18 months; SU still preferred over insulin during active phase; may relapse in adolescence",
            "KCNJ11 Adult-Onset MODY13 — heterozygous AD mild GOF; adult-onset diabetes; antibody negative; family history; SU-responsive",
        ],
        "stats": {
            "mean_dx_age": 3,
            "prevalence_NDM_pct": 30,
            "sulfonylurea_transition_pct": 92,
            "DEND_pct": 20,
            "mean_dx_delay_months": 4,
            "de_novo_severe_GOF_pct": 80,
            "neonatal_onset_pct": 85,
        },
        "dx_delay_distribution": {
            "0-6_mo": "75%", "6-24_mo": "18%", "2-5_yr": "5%", ">5_yr": "2%"
        },
    },
    # ── INS — MODY10 / NDM — Insulin Itself — Dominant-Negative ─────────────
    {
        "gene": "INS",
        "protein": "INS — MODY10 AD Dominant-Negative / AR Biallelic — Insulin Preprotein — Dominant-Negative Misfolding → ER Stress → Beta-Cell Apoptosis — Neonatal DM or Adolescent MODY — Insulin REQUIRED",
        "alias": (
            "INS (insulin preprotein); OMIM gene 176730; "
            "Diabetes mellitus type 1B / MODY10 / permanent neonatal DM OMIM 613370. "
            "11p15.5; 110 aa preprotein (24-aa signal + 86-aa proinsulin → 51-aa insulin); ~12 kDa; "
            "AD dominant-negative (most common) or AR biallelic (rare). "
            "FUNCTION: INS encodes the insulin preprotein: "
            "Signal peptide (24 aa) → co-translational import into ER; "
            "Proinsulin (86 aa) → forms correct disulfide bonds in ER → Golgi → secretory granules; "
            "Proinsulin → cleaved by prohormone convertases PC1/3 + PC2 → insulin (51 aa; A chain 21 aa + B chain 30 aa) + C-peptide. "
            "DISEASE MECHANISM — DOMINANT-NEGATIVE ER STRESS: "
            "AD INS mutations affect conserved cysteine residues or proinsulin folding domains; "
            "mutant proinsulin MISFOLDS in the ER → cannot form correct disulfide bonds; "
            "misfolded protein activates the Unfolded Protein Response (UPR): "
            "IRE1α → XBP1s + JNK → apoptotic signalling; "
            "PERK → eIF2α phosphorylation → global translation block; "
            "ATF6 → ER-associated degradation (ERAD); "
            "chronic ER stress → progressive beta-cell apoptosis → insulin deficiency; "
            "the WILD-TYPE INS allele also misfolds in response to ER stress (bystander effect) — "
            "hence DOMINANT-NEGATIVE mechanism: one mutant allele destroys both copies. "
            "CLINICAL PHENOTYPE: "
            "Spectrum: severe GOF mutations → permanent neonatal DM (diagnosed <6 months); "
            "milder mutations → adolescent/early adult onset MODY10; "
            "INSULIN IS ALWAYS REQUIRED: unlike MODY3 where SU works; "
            "INS-MODY has no intact beta-cell function to stimulate with SU; "
            "exocrine pancreas is normal (contrast HNF1B); "
            "C-PEPTIDE: detectable early but declines progressively → "
            "serial C-peptide is a marker of residual beta-cell mass; "
            "AR BIALLELIC INS LOF: total insulin gene absence → complete insulin deficiency from birth; "
            "requires insulin from day 1; C-peptide always zero. "
            "DISTINGUISHING FROM T1DM: "
            "INS-MODY: antibodies negative + family history (AD inheritance) + "
            "no T cell-mediated destruction (biopsy shows ER stress, not insulitis); "
            "however: some INS mutations have late-onset and may phenocopy T1DM; "
            "genetic testing is definitive."
        ),
        "locus": "11p15.5",
        "aa": 110,
        "kDa": 12,
        "omim_gene": "176730",
        "omim_disease": "613370",
        "inheritance": "AD dominant-negative (most common — one mutant allele causes beta-cell apoptosis via ER stress) / AR biallelic LOF (rare — complete insulin absence)",
        "gene_class": "Peptide hormone (insulin preprotein); 24-aa signal + 86-aa proinsulin → cleaved to insulin A+B chains (51 aa) + C-peptide; disulfide bond formation in ER is critical; dominant-negative mutations cause ER stress via misfolded proinsulin",
        "key_alerts": [
            "INS-MODY10-INSULIN-ALWAYS-REQUIRED-NO-SU: INS-MODY10 requires exogenous insulin; sulfonylureas are INEFFECTIVE (no intact beta-cell K-ATP pathway to stimulate); do NOT use SU in INS-MODY; progressive beta-cell apoptosis via ER stress means beta-cell mass declines over time",
            "INS-NDM-ANTIBODY-NEGATIVE-FAMILY-HISTORY: INS permanent NDM: antibody negative + strong AD family history; CRITICAL: do NOT diagnose T1DM without genetic testing in neonatal DM; INS is 3rd most common neonatal DM gene after KCNJ11 and ABCC8",
            "INS-ER-STRESS-DOMINANT-NEGATIVE-MECHANISM: Dominant-negative means ONE mutant allele disrupts BOTH alleles via UPR; treatment trials aimed at reducing ER stress (4-PBA, TUDCA, GLP1-RA) are investigational; standard treatment is insulin replacement",
            "INS-C-PEPTIDE-SERIAL-MONITORING: C-peptide detectable early in INS-MODY; serial monitoring tracks residual beta-cell mass; C-peptide decline anticipates insulin requirement increase; serial C-peptide guides adjustments in paediatric period",
            "INS-AR-BIALLELIC-TOTAL-INSULIN-ABSENCE: AR biallelic INS LOF = complete absence of insulin production; permanent neonatal DM from day 1; C-peptide always zero; requires insulin; parents are obligate carriers (AD mutation in parent may be milder)",
        ],
        "etiologies": [
            "INS AD Dominant-Negative Neonatal DM — cysteine/folding-domain mutation; permanent NDM; C-peptide initially detectable; ER stress mechanism; insulin required lifelong",
            "INS AD MODY10 Adolescent Onset — milder folding defect; gradual beta-cell apoptosis; onset age 10-35y; antibody negative; AD family history; insulin required (not SU)",
            "INS AR Biallelic LOF (Insulin Gene Deletion) — complete insulin absence; permanent neonatal DM; C-peptide always zero; insulin dependency from birth; parents carry one normal allele",
            "INS Sporadic De Novo Dominant-Negative — no family history; neonatal DM; genetic testing identifies INS; offspring 50% risk",
            "INS VUS / Missense Uncertain Significance — panel testing identifies variant; functional assay (proinsulin folding in cell model) required for classification; treat based on phenotype while awaiting classification",
        ],
        "stats": {
            "mean_dx_age": 15,
            "prevalence_NDM_pct": 12,
            "insulin_required_pct": 100,
            "mean_dx_delay_months": 24,
            "antibody_negative_pct": 98,
            "c_peptide_detectable_early_pct": 75,
            "de_novo_pct": 45,
        },
        "dx_delay_distribution": {
            "0-6_mo": "35%", "6-24_mo": "30%", "2-5_yr": "22%", ">5_yr": "13%"
        },
    },
    # ── PDX1 — MODY4 / Pancreatic Agenesis ── Master Pancreatic TF ──────────
    {
        "gene": "PDX1",
        "protein": "PDX1 — MODY4 AD Heterozygous / AR Biallelic = Pancreatic Agenesis — Pancreatic and Duodenal Homeobox 1 — Master Pancreatic Transcription Factor — Insulin + Glucagon + Exocrine — Biallelic = NO Pancreas on MRI",
        "alias": (
            "PDX1 (pancreatic and duodenal homeobox 1 / IDX1 / IPF1 / STF1); "
            "OMIM gene 600733; MODY4 OMIM 606392 (heterozygous); pancreatic agenesis OMIM 260370 (biallelic). "
            "13q12.2; 283 aa; ~31 kDa; "
            "AR biallelic LOF = complete pancreatic agenesis; AD heterozygous LOF = MODY4. "
            "FUNCTION: PDX1 is the master transcription factor for pancreatic development and beta-cell function. "
            "DEVELOPMENTAL ROLES: "
            "Pancreatic specification: PDX1 is the earliest marker of pancreatic progenitor cells "
            "(embryonic day E8.5 in mouse); PDX1 activates the entire pancreatic program; "
            "without PDX1 (biallelic LOF): NO pancreas forms → pancreatic agenesis; "
            "Endocrine specification: PDX1 drives expression of insulin (INS), glucagon (GCG), somatostatin (SST), "
            "pancreatic polypeptide (PP) — all islet cell hormones; "
            "Exocrine specification: PDX1 also required for acinar and ductal development. "
            "BETA-CELL HOMEOSTASIS (ADULT): "
            "In mature beta-cells, PDX1 maintains insulin gene transcription "
            "(binds A1/A2/A3 elements in the insulin gene promoter); "
            "PDX1 also regulates GLUT2 (SLC2A2), glucokinase (GCK), Nkx6.1, "
            "and other beta-cell identity genes; "
            "heterozygous PDX1 LOF → ~50% PDX1 activity → reduced insulin gene expression → "
            "progressive beta-cell dysfunction → MODY4 (mild, late-onset, variable penetrance). "
            "BIALLELIC PDX1 — PANCREATIC AGENESIS: "
            "Complete PDX1 LOF → NO pancreas at all; "
            "the most severe human pancreatic phenotype: "
            "affected infants require insulin from birth + exocrine enzyme replacement + "
            "fat-soluble vitamin supplementation; "
            "CT/MRI: absent pancreas (totally absent or rudimentary remnant); "
            "DO NOT CONFUSE with HNF1B pancreatic atrophy (partial reduction in pancreatic size) "
            "vs PDX1 biallelic pancreatic agenesis (complete absence); "
            "parents of biallelic child are obligate carriers of heterozygous PDX1 LOF → "
            "parents may have mild MODY4 that was never diagnosed. "
            "MODY4 (HETEROZYGOUS AD): "
            "Mildest of all MODY types with highly variable penetrance; "
            "late-onset (>40 years) mild diabetes; "
            "may require only diet modification or metformin; "
            "NOT responsive to sulfonylurea (limited beta-cell PDX1 activity means SU has less target); "
            "genetic diagnosis important for cascade testing (child of MODY4 + MODY4 carrier → "
            "25% risk of biallelic PDX1 → pancreatic agenesis)."
        ),
        "locus": "13q12.2",
        "aa": 283,
        "kDa": 31,
        "omim_gene": "600733",
        "omim_disease": "606392",
        "inheritance": "AR biallelic = pancreatic agenesis; AD heterozygous = MODY4 (variable penetrance); parents of biallelic child = obligate AD carriers (may have MODY4)",
        "gene_class": "Homeodomain transcription factor (ANTP class, PBC subfamily); master pancreatic developmental TF; binds insulin promoter A-elements; regulates GLUT2 + GCK + Nkx6.1 + SST + glucagon; also in duodenal enteroendocrine cells",
        "key_alerts": [
            "PDX1-BIALLELIC-PANCREATIC-AGENESIS-MRI-ABSENT-PANCREAS: Biallelic PDX1 LOF = complete pancreatic agenesis; MRI/CT shows absent pancreas (NOT just atrophy); requires insulin + exocrine enzyme replacement (PERT) + fat-soluble vitamins from birth; THIS IS NOT HNF1B ATROPHY — pancreas is entirely absent",
            "PDX1-PARENTS-OF-BIALLELIC-CHILD-ARE-MODY4-CARRIERS: Both parents of a biallelic PDX1 child carry heterozygous PDX1 LOF → MODY4 risk; test parents for diabetes; counsel siblings of parents (25% of each parent's siblings at risk for MODY4); child of two MODY4 carriers = 25% biallelic risk",
            "PDX1-MODY4-VARIABLE-PENETRANCE-MILD: MODY4 is the mildest MODY type; often late-onset (>40y) or subclinical; may not require treatment or only diet/metformin; variable penetrance means some heterozygotes never develop diabetes; cascade testing still important for biallelic risk stratification",
            "PDX1-EXOCRINE-DEFICIENCY-IN-BIALLELIC: Pancreatic agenesis = NO exocrine pancreas; fat malabsorption + steatorrhoea; faecal elastase absent; prescribe PERT (pancreatic enzyme replacement) + fat-soluble vitamins (A, D, E, K); DXA bone density monitoring (vitamin D + fat-soluble)",
            "PDX1-INSULIN-REQUIRED-BIALLELIC-SU-INEFFECTIVE-MODY4: Biallelic PDX1 → insulin required (no beta-cells); MODY4 heterozygous → SU may have limited response (insufficient beta-cell PDX1 activity); diet ± metformin is often adequate for MODY4; genetic counselling for offspring biallelic risk is the key intervention",
        ],
        "etiologies": [
            "PDX1 Biallelic Pancreatic Agenesis — compound heterozygous or homozygous PDX1 LOF; absent pancreas on imaging; neonatal DM + exocrine insufficiency; insulin + PERT from birth; parents are MODY4 carriers",
            "PDX1 Heterozygous MODY4 — mild late-onset DM; variable penetrance; often asymptomatic or diet-controlled; key importance is cascade testing for biallelic risk in offspring",
            "PDX1 de novo Biallelic (both alleles de novo) — rare; no family history; neonatal DM + absent pancreas; parents' PDX1 sequencing negative (truly de novo)",
            "PDX1 MODY4 Misdiagnosed as T2DM — late-onset mild diabetes + lean habitus + family history; genetic test identifies PDX1; cascade testing reveals biallelic risk in offspring",
            "PDX1 VUS — genetic panel identifies variant; pancreatic imaging + exocrine function testing + family studies needed; functional classification by PDX1 protein activity assay",
        ],
        "stats": {
            "mean_dx_age": 42,
            "prevalence_pct_MODY": 2,
            "biallelic_pancreatic_agenesis_pct": 5,
            "insulin_required_biallelic_pct": 100,
            "mean_dx_delay_months": 96,
            "exocrine_deficiency_biallelic_pct": 100,
            "variable_penetrance_MODY4_pct": 40,
        },
        "dx_delay_distribution": {
            "0-6_mo": "2%", "6-24_mo": "10%", "2-5_yr": "30%", ">5_yr": "58%"
        },
    },
]


def _generate_patients():
    """Generate 40 patients per gene using deterministic RNG (seed = SEED_BASE + gene_idx)."""
    for idx, gene in enumerate(MODY_GENES):
        rng = random.Random(SEED_BASE + idx)
        patients = []
        for i in range(40):
            age_at_dx = rng.randint(
                max(0, int(gene["stats"].get("mean_dx_age", 25) * 0.3)),
                int(gene["stats"].get("mean_dx_age", 25) * 2.2),
            )
            dx_delay = rng.randint(
                max(1, int(gene["stats"].get("mean_dx_delay_months", 24) * 0.3)),
                int(gene["stats"].get("mean_dx_delay_months", 24) * 2.0),
            )

            g = gene["gene"]

            if g == "HNF1A":
                treatment = rng.choice(["sulfonylurea", "sulfonylurea", "sulfonylurea", "insulin_prior_SU", "diet"])
                antibody_positive = False
                c_peptide_preserved = True
                renal_glucosuria = rng.random() > 0.1
                neonatal_hypoglycaemia = False
                renal_cysts = False
                pancreatic_atrophy = False
                exocrine_insufficiency = False
                gout = False
                hypomagnesaemia = False
                neonatal_dx = False

            elif g == "GCK":
                treatment = rng.choice(["none", "none", "none", "none", "diet"])
                antibody_positive = False
                c_peptide_preserved = True
                renal_glucosuria = rng.random() > 0.7
                neonatal_hypoglycaemia = False
                renal_cysts = False
                pancreatic_atrophy = False
                exocrine_insufficiency = False
                gout = False
                hypomagnesaemia = False
                neonatal_dx = False
                age_at_dx = rng.randint(20, 55)

            elif g == "HNF4A":
                treatment = rng.choice(["sulfonylurea", "sulfonylurea", "sulfonylurea", "insulin"])
                antibody_positive = False
                c_peptide_preserved = True
                renal_glucosuria = rng.random() > 0.4
                neonatal_hypoglycaemia = rng.random() > 0.55
                renal_cysts = False
                pancreatic_atrophy = False
                exocrine_insufficiency = False
                gout = False
                hypomagnesaemia = False
                neonatal_dx = False

            elif g == "HNF1B":
                treatment = "insulin"
                antibody_positive = False
                c_peptide_preserved = rng.random() > 0.4
                renal_glucosuria = rng.random() > 0.8
                neonatal_hypoglycaemia = False
                renal_cysts = rng.random() > 0.28
                pancreatic_atrophy = rng.random() > 0.35
                exocrine_insufficiency = rng.random() > 0.6
                gout = rng.random() > 0.55
                hypomagnesaemia = rng.random() > 0.55
                neonatal_dx = False

            elif g == "ABCC8":
                neonatal_dx = rng.random() > 0.2
                treatment = rng.choice(["sulfonylurea_glibenclamide", "insulin", "sulfonylurea_glibenclamide"])
                antibody_positive = False
                c_peptide_preserved = rng.random() > 0.3
                renal_glucosuria = False
                neonatal_hypoglycaemia = rng.random() > 0.5
                renal_cysts = False
                pancreatic_atrophy = False
                exocrine_insufficiency = False
                gout = False
                hypomagnesaemia = False
                age_at_dx = rng.randint(0, 5) if neonatal_dx else rng.randint(20, 45)

            elif g == "KCNJ11":
                neonatal_dx = rng.random() > 0.15
                treatment = rng.choice(["sulfonylurea_glibenclamide", "sulfonylurea_glibenclamide", "insulin"])
                antibody_positive = False
                c_peptide_preserved = rng.random() > 0.25
                renal_glucosuria = False
                neonatal_hypoglycaemia = False
                renal_cysts = False
                pancreatic_atrophy = False
                exocrine_insufficiency = False
                gout = False
                hypomagnesaemia = False
                age_at_dx = rng.randint(0, 3) if neonatal_dx else rng.randint(15, 40)

            elif g == "INS":
                neonatal_dx = rng.random() > 0.35
                treatment = "insulin"
                antibody_positive = False
                c_peptide_preserved = rng.random() > 0.4
                renal_glucosuria = False
                neonatal_hypoglycaemia = False
                renal_cysts = False
                pancreatic_atrophy = False
                exocrine_insufficiency = False
                gout = False
                hypomagnesaemia = False
                age_at_dx = rng.randint(0, 6) if neonatal_dx else rng.randint(10, 35)

            else:  # PDX1
                neonatal_dx = rng.random() > 0.95  # biallelic very rare
                treatment = "insulin" if neonatal_dx else rng.choice(["diet", "metformin", "insulin"])
                antibody_positive = False
                c_peptide_preserved = not neonatal_dx
                renal_glucosuria = False
                neonatal_hypoglycaemia = False
                renal_cysts = False
                pancreatic_atrophy = neonatal_dx
                exocrine_insufficiency = neonatal_dx
                gout = False
                hypomagnesaemia = False
                age_at_dx = rng.randint(0, 1) if neonatal_dx else rng.randint(35, 65)

            sulfonylurea_effective = treatment in ("sulfonylurea", "sulfonylurea_glibenclamide")
            genetic_cascade_tested = rng.random() > 0.35
            misdiagnosed = rng.random() > 0.45 if g in ("HNF1A", "GCK", "HNF4A") else rng.random() > 0.8

            patients.append({
                "patient_id": f"{g}-{i+1:03d}",
                "age_at_dx": max(0, age_at_dx),
                "dx_delay_months": dx_delay,
                "treatment": treatment,
                "antibody_positive": antibody_positive,
                "c_peptide_preserved": c_peptide_preserved,
                "renal_glucosuria": renal_glucosuria,
                "neonatal_hypoglycaemia": neonatal_hypoglycaemia,
                "renal_cysts": renal_cysts,
                "pancreatic_atrophy": pancreatic_atrophy,
                "exocrine_insufficiency": exocrine_insufficiency,
                "gout": gout,
                "hypomagnesaemia": hypomagnesaemia,
                "neonatal_dx": neonatal_dx,
                "sulfonylurea_effective": sulfonylurea_effective,
                "genetic_cascade_tested": genetic_cascade_tested,
                "misdiagnosed": misdiagnosed,
                "gene": g,
                "seed": SEED_BASE + idx,
            })
        gene["patients"] = patients


_generate_patients()


def get_overview():
    all_ages = [p["age_at_dx"] for g in MODY_GENES for p in g["patients"]]
    all_delays = [p["dx_delay_months"] for g in MODY_GENES for p in g["patients"]]
    genes = []
    for idx, g in enumerate(MODY_GENES):
        ages = [p["age_at_dx"] for p in g["patients"]]
        delays = [p["dx_delay_months"] for p in g["patients"]]
        genes.append({
            "gene": g["gene"],
            "protein": g["protein"],
            "locus": g["locus"],
            "aa": g["aa"],
            "kDa": g["kDa"],
            "omim_gene": g["omim_gene"],
            "inheritance": g["inheritance"],
            "gene_class": g["gene_class"],
            "mean_dx_age": round(sum(ages) / len(ages), 1),
            "mean_dx_delay_months": round(sum(delays) / len(delays), 1),
            "key_alerts": g["key_alerts"],
            "n_patients": len(g["patients"]),
        })
    return {
        "atlas": "Hereditary-MODY-Atlas",
        "subtitle": (
            "Complete 8-Gene Hereditary Monogenic Diabetes (MODY) Atlas — "
            "HNF1A / GCK / HNF4A / HNF1B / ABCC8 / KCNJ11 / INS / PDX1 — 320 Patients (8×40, Seeds 1694–1701)"
        ),
        "seed_range": f"{SEED_BASE}–{SEED_BASE+7}",
        "total_patients": sum(len(g["patients"]) for g in MODY_GENES),
        "aggregate_stats": {
            "mean_dx_age": round(sum(all_ages) / len(all_ages), 1),
            "mean_dx_delay_months": round(sum(all_delays) / len(all_delays), 1),
            "genes_covered": len(MODY_GENES),
            "patients_per_gene": 40,
        },
        "genes": genes,
        "top_alerts": [
            "GCK-MODY2-NO-TREATMENT-EXCEPT-PREGNANCY: GCK-MODY2 should NOT be treated; lifelong stable mild fasting hyperglycaemia 5.5–8 mmol/L is a set-point — NOT progressive; treatment causes iatrogenic hypoglycaemia without benefit; EXCEPTION: pregnancy — fetal genotype determines whether maternal treatment is needed",
            "HNF1A-HNF4A-SULFONYLUREA-FIRST-LINE->95PCT: HNF1A (MODY3) and HNF4A (MODY1) are both highly sulfonylurea-responsive (>95% and >90% respectively); start at 10–20× lower dose than standard T2DM dose; insulin-to-sulfonylurea transition succeeds in >90%; DO NOT accept lifelong insulin without genetic test in antibody-negative young-onset DM",
            "HNF1B-RCAD-TRIAD-INSULIN-REQUIRED-SU-INEFFECTIVE: HNF1B-MODY5 = renal cysts + diabetes + genital tract anomalies (RCAD); pancreatic atrophy; INSULIN required; sulfonylureas INEFFECTIVE (no intact beta-cell mass); gout + hypomagnesaemia require monitoring; 50% de novo mutations",
            "KCNJ11-ABCC8-NDM-SULFONYLUREA-REPLACES-INSULIN->90PCT: Any neonate with DM in first 6 months MUST have KCNJ11 + ABCC8 tested BEFORE committing to lifelong insulin; GOF mutations → oral glibenclamide replaces insulin in >90% (KCNJ11) / >70% (ABCC8); DEND syndrome: SU crosses BBB → dramatic neurological benefit",
            "INS-MODY10-INSULIN-ALWAYS-REQUIRED-DOMINANT-NEGATIVE-ER-STRESS: INS dominant-negative mutations cause ER stress → progressive beta-cell apoptosis; INSULIN required; SU ineffective; serial C-peptide tracks beta-cell decline; antibody negative distinguishes from T1DM",
            "PDX1-BIALLELIC-PANCREATIC-AGENESIS-ABSENT-PANCREAS-MRI: Biallelic PDX1 LOF = complete pancreatic agenesis; NO pancreas on imaging; insulin + exocrine enzyme replacement from birth; MODY4 parents of biallelic child are obligate heterozygous carriers; cascade test all offspring for biallelic risk",
            "HNF1A-RENAL-GLUCOSURIA-DIAGNOSTIC-CLUE-90PCT: HNF1A LOF lowers renal glucose threshold via SGLT2 regulation; positive urine glucose at normal/near-normal plasma glucose in 90% of HNF1A carriers; urine glucose positive = test MODY panel before diagnosing T1DM/T2DM",
            "HNF4A-NEONATAL-MACROSOMIA-HYPOGLYCAEMIA-BIPHASIC: HNF4A LOF paradoxically causes foetal hyperinsulinaemia → macrosomia (+800g) + diazoxide-responsive neonatal hypoglycaemia → resolves → adult-onset MODY1 hyperglycaemia; any hyperinsulinaemic macrosomic neonate with diabetic parent → HNF4A FIRST",
        ],
    }


def get_breakdown():
    result = []
    for idx, g in enumerate(MODY_GENES):
        ages = [p["age_at_dx"] for p in g["patients"]]
        delays = [p["dx_delay_months"] for p in g["patients"]]
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
                "mean_dx_age": round(sum(ages) / len(ages), 1),
                "mean_dx_delay_months": round(sum(delays) / len(delays), 1),
                "n_patients": len(g["patients"]),
                "seed": SEED_BASE + idx,
            },
            "sample_patients": g["patients"][:10],
        })
    return result


def get_definitions():
    return {
        "concepts": {
            "MODY vs T1DM vs T2DM — The Diagnostic Algorithm": (
                "Maturity Onset Diabetes of the Young (MODY) is a monogenic form of diabetes caused by "
                "single-gene defects in beta-cell function. MODY accounts for 1–2% of all diabetes but "
                "is massively underdiagnosed (~80% misdiagnosed as T1DM or T2DM). "
                "KEY DIFFERENTIATORS FROM T1DM: "
                "MODY: no islet autoantibodies (GAD/IA2/ZnT8/IAA negative); C-peptide preserved for years; "
                "strong autosomal dominant family history (3+ generations affected); "
                "does not absolutely require insulin (except HNF1B, INS, PDX1 biallelic); "
                "MODY3/1 respond dramatically to low-dose sulfonylurea. "
                "KEY DIFFERENTIATORS FROM T2DM: "
                "MODY: onset <35 years; lean or normal BMI; no metabolic syndrome; "
                "autosomal dominant 3-generation pedigree; gene-specific features "
                "(renal glucosuria → HNF1A, renal cysts → HNF1B, neonatal macrosomia → HNF4A); "
                "exquisite sulfonylurea sensitivity (vs modest T2DM response). "
                "GCK-MODY2 UNIQUE: "
                "flat OGTT (<3.5 mmol/L rise), stable HbA1c since childhood, asymptomatic, no treatment needed — "
                "the only 'diabetes' that is not treated as a disease. "
                "MODY PROBABILITY CALCULATOR: "
                "Available tools (Exeter MODY probability calculator) estimate likelihood based on: "
                "age at onset, BMI, C-peptide, antibody status, family history; "
                ">25% probability → offer genetic testing; "
                "genetic panel covers HNF1A + GCK + HNF4A + HNF1B + ABCC8 + KCNJ11 + INS + PDX1 + others."
            ),
            "K-ATP Channel Pharmacogenomics — The Neonatal DM Success Story": (
                "The K-ATP channel is the glucose-sensing switch of pancreatic beta-cells. "
                "CHANNEL STRUCTURE: octameric complex of 4 × Kir6.2 (KCNJ11) + 4 × SUR1 (ABCC8). "
                "NORMAL GATING: "
                "Post-meal: high intracellular ATP → binds Kir6.2 → channel closes → "
                "membrane depolarisation → VGCCs open → Ca2+ influx → insulin exocytosis; "
                "Fasting: low ATP/ADP ratio → K-ATP opens → K+ efflux → hyperpolarisation → no insulin. "
                "GOF MUTATIONS (KCNJ11/ABCC8) → NDM: "
                "Channel remains open despite high ATP (reduced ATP sensitivity); "
                "beta-cell stays hyperpolarised → NO insulin secretion → neonatal DM; "
                "may also affect neuronal K-ATP (DEND syndrome in severe GOF). "
                "SULFONYLUREA MECHANISM: "
                "Sulfonylureas (glibenclamide, gliclazide) bind SUR1's NBD; "
                "close K-ATP INDEPENDENTLY of ATP level; "
                "bypass the defective ATP-sensing; "
                "restore insulin secretion in >90% of KCNJ11 NDM and >70% of ABCC8 NDM; "
                "also close neuronal K-ATP → DEND: reduced seizures + improved cognition. "
                "DIAZOXIDE (OPPOSITE DIRECTION): "
                "Opens K-ATP → suppresses insulin secretion; "
                "used in congenital hyperinsulinism (CHI due to LOF KCNJ11/ABCC8) to inhibit excess insulin; "
                "diazoxide-responsive CHI (intact K-ATP) vs diazoxide-unresponsive CHI (K-ATP-independent — "
                "KATP mutation so severe that diazoxide cannot open; or different gene e.g. HADH, GCK GOF). "
                "CLINICAL PHARMACOLOGY PRIORITY: "
                "Any neonatal DM: test KCNJ11 + ABCC8 BEFORE discharge; "
                "if GOF confirmed → introduce oral glibenclamide; "
                "transition complete in 2–6 weeks; monitor closely."
            ),
            "HNF Transcription Factor Network — MODY3/1/5 Interconnection": (
                "HNF1A, HNF4A, and HNF1B form an interconnected transcription factor network in "
                "pancreatic beta-cells, hepatocytes, and kidney tubular cells. "
                "HNF4A → activates HNF1A; HNF1A → feeds back to activate HNF4A targets; "
                "HNF1B → dimerises with HNF1A (heterodimer) and modulates HNF1A targets in kidney. "
                "BETA-CELL REGULOME: "
                "HNF1A and HNF4A both regulate: insulin gene (INS), GCK promoter, GLUT2 (SLC2A2), "
                "mitochondrial metabolism genes (Complex I-IV subunits), and each other. "
                "RENAL PHENOTYPES DIFFER: "
                "HNF1A LOF: reduced renal SGLT2 → low glucose reabsorption threshold → renal glucosuria "
                "(glucose in urine at normal plasma glucose); "
                "HNF1B LOF: disrupted renal tubular development → cysts + URAT1 dysregulation (gout) + "
                "FXYD2 dysregulation (Mg2+ wasting → hypomagnesaemia). "
                "HEPATIC PHENOTYPES: "
                "HNF1A: regulates ApoA1 → low HDL; "
                "HNF4A: regulates ApoB, ApoC-III, ApoA-IV → elevated TG + low ApoA1; "
                "hepatic lipid panel provides supportive evidence for HNF1A/HNF4A. "
                "TREATMENT CONVERGENCE AND DIVERGENCE: "
                "HNF1A-MODY3 and HNF4A-MODY1: BOTH respond to sulfonylurea (their beta-cells have intact K-ATP); "
                "HNF1B-MODY5: beta-cell MASS reduced (structural) → SU has no target → insulin required; "
                "the HNF network lesson: same TF family, opposite pharmacology."
            ),
            "GCK — The Beta-Cell Glucose Sensor and Why MODY2 Is Not a Disease": (
                "Glucokinase (GCK) is unique among hexokinases: HIGH Km for glucose (~8 mmol/L), "
                "sigmoidal kinetics, not product-inhibited. This makes GCK the beta-cell's 'glucose sensor.' "
                "THRESHOLD SHIFT MODEL: "
                "Heterozygous GCK LOF → ~50% enzyme activity → glucose threshold for insulin secretion "
                "shifts from ~4.5 mmol/L to ~5.5–8 mmol/L; "
                "this is not progressive dysfunction — it is a NEW SETPOINT; "
                "the beta-cells are perfectly normal, just calibrated higher. "
                "FLAT OGTT — HEPATIC MECHANISM: "
                "Normal GCK in liver: post-load glucose → hepatic glucose uptake/glycogen synthesis → "
                "plasma glucose falls after peak; "
                "GCK LOF in hepatocytes: reduced hepatic uptake at same load → "
                "OGTT peak also higher but the relative increase above fasting is preserved; "
                "result: glucose rise <3.5 mmol/L above fasting (flat curve). "
                "NO COMPLICATIONS — PROVEN: "
                "Long-term studies (Steele et al. 2010, Stride et al. 2002) confirm: "
                "untreated GCK-MODY2 individuals have no increased retinopathy/nephropathy/neuropathy risk; "
                "the mild chronic hyperglycaemia (HbA1c ~6.5–7.5%) does not confer complication risk; "
                "this is because the entire glucose metabolism is recalibrated — not dysregulated. "
                "PREGNANCY EXCEPTION — FOETAL GENOTYPE RULE: "
                "Mother GCK LOF + foetus GCK WT → maternal glucose (~6.5 mmol/L) is hyperglycaemic to WT foetus "
                "→ foetal hyperinsulinaemia → macrosomia → treat mother to achieve <5.5 mmol/L fasting. "
                "Mother GCK LOF + foetus also GCK LOF → foetal threshold also raised "
                "→ foetal glucose = appropriate for foetal GCK → no macrosomia "
                "→ overly tight maternal control → foetal hypoglycaemia → do NOT over-treat. "
                "FOETAL GENOTYPE ESTIMATION: if foetus is known carrier (paternal test) → adjust treatment; "
                "if unknown → serial growth ultrasound: accelerated abdominal circumference growth = WT foetus being over-insulinised = treat."
            ),
        },
        "pharmacological_distinctions": [
            "HNF1A/HNF4A-MODY: Sulfonylurea (gliclazide/glipizide) — FIRST-LINE; start at 10-20x lower than T2DM dose (risk of severe hypoglycaemia); HNF1A SU response >98%, HNF4A >90%; insulin-to-SU transition succeeds in >90% of misdiagnosed patients; monitor glucose 4x daily during transition; reduce insulin as SU takes effect over 7-14 days",
            "GCK-MODY2: NO TREATMENT — antidiabetic agents contraindicated (iatrogenic hypoglycaemia without benefit); exception: pregnancy with WT foetus — short-acting insulin to achieve fasting <5.5 mmol/L and 1h post-meal <7.8 mmol/L; stop all treatment immediately post-partum; NO metformin/SU/SGLT2i in GCK-MODY outside pregnancy",
            "HNF1B-MODY5: INSULIN REQUIRED — basal-bolus insulin; sulfonylureas absolutely ineffective (no intact beta-cell K-ATP); pancreatic enzyme replacement therapy (PERT — creon/pancreaze) if faecal elastase <100 μg/g; magnesium supplementation if hypomagnesaemia; allopurinol/febuxostat if recurrent gout; gout standard management",
            "KCNJ11/ABCC8 NDM: Oral glibenclamide (glyburide) — HIGH dose (0.5-1.0 mg/kg/day in children with NDM); titrate up q7d while weaning insulin; target HbA1c <7%; neurological benefits in DEND (glibenclamide crosses BBB); monitor glucose 6x daily during transition; transition failures (10%) → maintain SU + insulin combination",
            "INS-MODY10: Insulin — basal-bolus or pump; SU INEFFECTIVE (no K-ATP pathway to stimulate); C-peptide guided insulin dose adjustment (higher C-peptide = more residual beta function = potentially less insulin); investigational ER stress reducers (4-PBA, TUDCA) not standard; GLP1-RA added for synergistic beta-cell protection (investigational)",
            "PDX1-Biallelic Pancreatic Agenesis: Insulin (basal-bolus) + Pancreatic Enzyme Replacement (PERT 25,000-40,000 units lipase per meal) + fat-soluble vitamins (A, D, E, K) supplementation; DXA bone density monitoring annually; insulin requirements may be lower than T1DM (some preserved glucagon from remaining islets uncertain); no SU — no beta-cells present",
        ],
        "key_standards": [
            "Murphy et al. 2008 (BMJ) — GCK Treatment Study: Randomised trial of sulfonylurea vs no treatment in GCK-MODY2; no HbA1c benefit from treatment; established GCK-MODY2 as a benign condition requiring no antidiabetic treatment outside pregnancy; foundational evidence for the no-treatment recommendation",
            "Pearson et al. 2006 (NEJM) — KCNJ11 NDM Sulfonylurea Transfer: Landmark study of 49 patients with KCNJ11 permanent NDM; 44/49 (90%) successfully transferred from insulin to oral sulfonylurea; HbA1c improved; DEND patients showed neurological improvement; established SU as standard for KCNJ11 NDM",
            "Greeley et al. 2011 / Babiker et al. 2016 — ABCC8 NDM SU Transfer: Multiple cohort studies demonstrating oral glibenclamide (glyburide) successful transition in 70-80% of ABCC8 GOF permanent NDM; DEND neurological benefits confirmed; established SU protocol for ABCC8 NDM",
            "McDonald et al. 2011 (Diabetologia) — HNF1A Sulfonylurea Precision: 43 HNF1A-MODY patients transitioned from insulin to SU; 94% success; HbA1c improved by 1.2%; demonstrated pharmacogenomics precision therapy value; established MODY genetic testing as clinical standard",
            "Stride et al. 2002 / Steele et al. 2010 (Diabetes Care/Diabetologia) — GCK Long-Term Outcomes: Long-term follow-up of untreated GCK-MODY2 individuals; no increase in retinopathy, nephropathy, or neuropathy compared to normal population; HbA1c stable across decades; established benign prognosis of GCK-MODY2",
            "ISPAD/APEG Clinical Practice Consensus Guidelines 2022 — MODY Diagnosis and Management: Recommend MODY genetic panel in antibody-negative young-onset DM + family history; Exeter MODY probability calculator for risk stratification; gene-specific management: GCK-no treatment / HNF1A-SU / HNF4A-SU / HNF1B-insulin / KCNJ11-glibenclamide / ABCC8-glibenclamide / INS-insulin; cascade testing all first-degree relatives",
            "Flanagan et al. 2014 (Diabet Med) — HNF1B Phenotype Registry: 112 HNF1B patients; 73% renal cysts; 44% pancreatic atrophy; 39% genital malformations; 21% gout; 19% hypomagnesaemia; 50% de novo; 40% 17q12 deletions detectable only by MLPA; established comprehensive HNF1B phenotype and surveillance protocol",
        ],
    }
