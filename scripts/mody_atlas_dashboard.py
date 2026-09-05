#!/usr/bin/env python3
"""MODY Atlas — Complete 8-Gene Maturity-Onset Diabetes of Youth (MODY) Atlas
HNF4A   (MODY1 — 464 aa; 20q13.12; Hepatocyte Nuclear Factor 4-alpha;
         AR-dominant; neonatal hyperinsulinism → adult MODY;
         SULFONYLUREAS first line — dramatically more effective than insulin) ·
GCK     (MODY2 — 465 aa; 7p13; Glucokinase glucose sensor;
         AD; MILD stable fasting hyperglycemia 6.0–8.0 mmol/L;
         NO TREATMENT NEEDED — over-treatment causes harm; pregnancy management critical) ·
HNF1A   (MODY3 — 631 aa; 12q24.31; Hepatocyte Nuclear Factor 1-alpha;
         AD; most common symptomatic MODY in Europe 30–40%;
         GLUCOSURIA at normal glucose — PATHOGNOMONIC; sulfonylureas 5–8× > insulin) ·
PDX1    (MODY4 — 283 aa; 13q12.2; Pancreatic and Duodenal Homeobox 1;
         AD; rare; heterozygous MODY; homozygous → neonatal pancreatic agenesis;
         EXOCRINE INSUFFICIENCY 40% — enzyme supplementation mandatory) ·
HNF1B   (MODY5 — 557 aa; 17q12; Hepatocyte Nuclear Factor 1-beta;
         AD; RENAL CYSTS + diabetes + liver tests TRIAD; multi-organ syndrome;
         SULFONYLUREAS OFTEN INEFFECTIVE — early insulin typically required) ·
NEUROD1 (MODY6 — 356 aa; 2q31.3; Neurogenic Differentiation Factor 1;
         AD; rare; heterozygous MODY; homozygous → neonatal DM + cerebellar hypoplasia;
         SENSORINEURAL HEARING LOSS + visual impairment homozygous) ·
ABCC8   (MODY12/NDM — 1581 aa; 11p15.1; SUR1 sulfonylurea receptor subunit KATP;
         AD; GOF → neonatal DM; LOF → congenital hyperinsulinism;
         SULFONYLUREA SWITCH: 90% NDM can switch insulin → oral glyburide — life-changing) ·
KCNJ11  (MODY13/NDM — 390 aa; 11p15.1; Kir6.2 KATP pore;
         AD; GOF → permanent neonatal DM; DEND syndrome (severe);
         SULFONYLUREA SWITCH: 90% PNDM respond — start before age 6 months for best outcome)
320-patient aggregate cohort (8 × 40, seeds 1214–1221)
"""

import random

SEED_BASE = 1214

MODY_GENES = [
    # ── HNF4A — MODY1 ──────────────────────────────────────────────────────────
    {
        "gene": "HNF4A",
        "protein": "Hepatocyte nuclear factor 4-alpha (HNF-4α)",
        "alias": (
            "HNF4A; OMIM gene 600281; MODY1; 20q13.12; 464 aa; ~51 kDa; "
            "AD; nuclear receptor transcription factor; "
            "MODY1 — 5% of MODY; neonatal hyperinsulinism (NHI) 50% of carriers"
        ),
        "aa": "464 aa",
        "kDa": "~51 kDa",
        "locus": "20q13.12",
        "omim_gene": 600281,
        "omim_disease": 125850,
        "inheritance": "AD (haploinsufficiency)",
        "gene_class": (
            "Nuclear receptor superfamily (NR2A1); ligand-binding domain coordinates fatty acid sensing + glucose metabolism; "
            "HNF4A drives expression of insulin gene, GLUT2, glucokinase, HNF1A, and hepatic gluconeogenesis genes; "
            "HNF4A LOF → β-cell haploinsufficiency → reduced insulin secretion → progressive hyperglycemia (adult MODY); "
            "Paradox: in neonate, HNF4A LOF in fetal β-cells → disinhibition of KATP-independent insulin secretion → "
            "NEONATAL HYPERINSULINISM (NHI) — macrosomic baby, hypoglycemia in first days–weeks; "
            "NHI resolves as β-cell mass matures (usually first months), then β-cells become HNF4A-haploinsufficient → MODY; "
            "HNF4A is also the master regulator of hepatic lipid metabolism → "
            "HNF4A MODY patients have higher macrovascular risk vs MODY2 (GCK); "
            "Also expressed in kidney and intestine: proteinuria risk and gut motility differences; "
            "HNF4A mutations cluster in DNA-binding domain (DBD) and ligand-binding domain (LBD)"
        ),
        "phenotype": (
            "Neonatal hyperinsulinism (NHI) in ~50% of HNF4A carriers: macrosomia (birth weight +760 g above mean), "
            "hypoglycemia requiring IV glucose or diazoxide in first days to weeks of life; "
            "NHI usually resolves by 3–6 months but can persist; "
            "MODY onset: typically teen years to 30s; progressive hyperglycemia; fasting glucose 6.5–10+ mmol/L; "
            "Microvascular risk: retinopathy + nephropathy similar to other MODY types if poorly controlled; "
            "Macrovascular risk: elevated vs MODY2 (liver HNF4A role in lipid metabolism); "
            "Liver: mildly elevated transaminases in some carriers; "
            "β-cell failure is progressive: sulfonylureas effective early; insulin may eventually be needed; "
            "Family history: vertical transmission AD; 50% of children affected"
        ),
        "hallmark": (
            "NEONATAL HYPERINSULINISM (NHI) IN 50% — paradoxical HI in neonatal period BEFORE MODY later; "
            "MACROSOMIA: birth weight typically 760 g above population mean — document carefully; "
            "DUAL PHENOTYPE: same HNF4A LOF variant → NHI in neonate → MODY in adult (unique MODY1 feature); "
            "SULFONYLUREAS first line — glibenclamide/glyburide or gliclazide; often excellent response; "
            "FAMILY HISTORY: vertical AD transmission — screen all 1st-degree relatives; "
            "NHI MANAGEMENT: diazoxide (KATP opener) in NHI phase — THEN STOP when NHI resolves; "
            "MODY1 diagnosis in adult with macrosomic infant history is PATHOGNOMONIC combination; "
            "GENETIC DIAGNOSIS: panel sequencing preferred — HNF4A + HNF1A + GCK + HNF1B at minimum"
        ),
        "treatment_alert": (
            "SULFONYLUREAS FIRST LINE for adult MODY phase: glibenclamide 2.5–15 mg/day or gliclazide MR 30–120 mg/day; "
            "Response rate: 85–90% — dramatically superior to insulin or metformin; "
            "START LOW DOSE (glibenclamide 1.25–2.5 mg) — MODY β-cells are sensitive; "
            "METFORMIN: limited efficacy in MODY1 (primary defect is secretory not resistance); "
            "GLP-1 AGONISTS: some evidence of benefit (insulinotropic) — second line option; "
            "INSULIN: reserve for late-stage β-cell exhaustion or pregnancy where tight control needed; "
            "NHI PHASE (neonate): diazoxide 5–15 mg/kg/day (KATP opener) → resolves NHI; "
            "PREGNANCY: increase monitoring; risk of macrosomic infant even in treated mother; "
            "AVOID: excessive alcohol with sulfonylureas (prolonged hypoglycemia); "
            "MONITOR: HbA1c 3–6 monthly; annual retinal and renal screening once diagnosed"
        ),
        "key_ddx": (
            "HNF1A/MODY3 (most common symptomatic MODY — glucosuria distinguishes; same era of onset); "
            "GCK/MODY2 (very mild stable hyperglycemia — no NHI, no progressive β-cell failure); "
            "HNF1B/MODY5 (multi-organ: renal cysts + elevated GGT — no NHI macrosomia); "
            "ABCC8-CHI (neonatal HI — persistent, severe, diazoxide-responsive in CHI vs transient NHI); "
            "Type 1 DM (autoantibodies: GADA/IA-2/ZnT8 — HNF4A MODY1 autoantibody-negative); "
            "Type 2 DM (insulin resistance component — MODY1 lean, AD family history, younger onset)"
        ),
        "gfr_pattern": "Usually preserved; proteinuria may develop with progressive hyperglycemia; ACE-I if microalbuminuria",
        "proteinuria_pattern": "Low-grade microalbuminuria; progressive if glycemic control poor; glomerular pattern",
        "primary_complication": "Progressive β-cell failure → hyperglycemia → microvascular complications if untreated; macrosomic neonatal hypoglycemia",
        "disease_detail": (
            "HNF4A MODY1 is the second most-clinically-significant MODY subtype after MODY3. "
            "The defining paradox — neonatal HI followed by adult MODY from the SAME LOF allele — "
            "reflects developmental switching in β-cell insulin secretion mechanisms. "
            "Neonatal β-cells rely on KATP-independent (HNF4A-mediated amplification) secretion more; "
            "HNF4A LOF → excess basal secretion → NHI. As β-cells mature, this pathway becomes less dominant "
            "and HNF4A LOF becomes rate-limiting for glucose-stimulated secretion → MODY. "
            "Genetic testing: variant classification can be difficult; functional assays + segregation studies needed for VUSs. "
            "MODY1 must be considered in any family with macrosomic infant + parent with 'diabetes'."
        ),
        "variants": [
            {"name": "p.Arg154Ter", "frequency": "truncating; null allele; severe MODY1"},
            {"name": "p.Val393Ile", "frequency": "missense; LBD; intermediate phenotype"},
            {"name": "p.Gln268Ter", "frequency": "truncating; DBD disruption; NHI+MODY"},
        ],
        "drug_ci": [
            "EXCESS ALCOHOL + sulfonylureas: prolonged hypoglycemia; limit to 1 unit/day max",
            "RIFAMPICIN: CYP2C9/CYP2C19 inducer — reduces sulfonylurea plasma levels; switch AED or monitor",
            "FLUOROQUINOLONES (ciprofloxacin): dysglycemia risk; monitor blood glucose if prescribed",
            "THIAZOLIDINEDIONES (pioglitazone): not primary treatment; modest benefit; weight gain concern",
        ],
    },

    # ── GCK — MODY2 ────────────────────────────────────────────────────────────
    {
        "gene": "GCK",
        "protein": "Glucokinase (hexokinase IV, glucose sensor)",
        "alias": (
            "GCK; OMIM gene 138079; MODY2; 7p13; 465 aa; ~52 kDa; "
            "AD; glucose-sensing enzyme; MODY2 — most common MODY worldwide 30–50%; "
            "MILD STABLE FASTING HYPERGLYCEMIA — NO TREATMENT NEEDED (most)"
        ),
        "aa": "465 aa",
        "kDa": "~52 kDa",
        "locus": "7p13",
        "omim_gene": 138079,
        "omim_disease": 125851,
        "inheritance": "AD (haploinsufficiency of glucose sensor)",
        "gene_class": (
            "Hexokinase IV — the only hexokinase that is NOT saturated at physiological glucose concentrations; "
            "GCK acts as the 'glucose sensor' of the β-cell: its S0.5 (~8 mmol/L) defines the threshold for insulin secretion; "
            "GCK LOF → S0.5 shifts rightward → β-cell treats elevated glucose as 'normal' → "
            "insulin secretion set-point raised → stable mild fasting hyperglycemia (6.0–8.0 mmol/L); "
            "This is NOT progressive disease — the set-point is fixed, not worsening; "
            "GCK is simultaneously the rate-limiting enzyme of hepatic glycolysis: "
            "GCK LOF → reduced hepatic glucose uptake post-meal → 2-hour postprandial glucose mildly elevated; "
            "GCK GOF mutations: opposite phenotype — PERSISTENT HYPERINSULINISM OF INFANCY (diffuse); "
            "Most common MODY worldwide (30–50% of MODY); extremely mild phenotype; "
            "Autosomal dominant but most families de novo or heterozygous — compound heterozygous very rare"
        ),
        "phenotype": (
            "Mild STABLE fasting hyperglycemia: 5.5–8.0 mmol/L (99–144 mg/dL); typically 6.0–7.5 mmol/L; "
            "HbA1c: mildly elevated 5.8–7.5% (40–58 mmol/mol); "
            "2-hour post-load glucose: mildly elevated but increment from fasting is SMALL (<3.5 mmol/L); "
            "NO progression: β-cell defect is fixed at glucose sensor level — NOT progressive β-cell failure; "
            "NO microvascular complications at typical GCK glucose levels: retinopathy rare, nephropathy very rare; "
            "Diagnosis often incidental: found on routine pre-employment or pregnancy screening; "
            "Often misdiagnosed as T2D or pre-diabetes and treated unnecessarily; "
            "NO GLYCOSURIA: renal threshold normal (distinguishes from MODY3); "
            "Pregnancy: CRITICAL management — fetal genotype determines treatment (see treatment alert)"
        ),
        "hallmark": (
            "STABLE MILD FASTING HYPERGLYCEMIA — set-point is fixed, not progressive (key difference from MODY3); "
            "NO MICROVASCULAR COMPLICATIONS at typical GCK glucose levels — 'benign' hyperglycemia; "
            "NO GLYCOSURIA — renal glucose threshold NORMAL (unlike MODY3/HNF1A); "
            "PREGNANCY MANAGEMENT DEPENDS ON FETAL GENOTYPE — unique and critical rule: "
            "  • Fetus has GCK LOF: fetus and mother same set-point → normal fetal weight → NO insulin needed; "
            "  • Fetus NO GCK LOF: fetus has lower set-point → macrosomia → INSULIN needed; "
            "  • Mother NO GCK, fetus HAS GCK: low birth weight (fetus hypoinsulinaemic vs maternal glucose); "
            "MOST COMMON MODY: 30–50% worldwide; high de-novo rate; "
            "OVER-TREATMENT HARM: insulin/SU treatment in MODY2 causes hypoglycemia without benefit"
        ),
        "treatment_alert": (
            "NO TREATMENT REQUIRED for most GCK-MODY2 patients outside pregnancy: "
            "Risk of microvascular complications at 6.0–7.5 mmol/L is minimal — do NOT treat; "
            "STOP unnecessary insulin/sulfonylurea — causes hypoglycemia, weight gain, without benefit; "
            "If HbA1c >7.5% (>58 mmol/mol) AND other risk factors → reassess (possible co-existing T2D component); "
            "PREGNANCY EXCEPTION — Fetal genotype determines treatment: "
            "  • Genotype fetus (NIPT or amniocentesis if uncertain); "
            "  • Fetus SAME mutation as mother → target glucose same as pre-pregnancy → NO insulin; "
            "  • Fetus UNAFFECTED → target normal glucose → insulin to reduce macrosomia risk; "
            "POSTPARTUM: no treatment; annual HbA1c monitoring only; "
            "LIFESTYLE: healthy weight and exercise beneficial but not for glucose lowering — for CV health; "
            "AVOID: starting insulin/SU without genetic confirmation in all patients with mild stable hyperglycemia"
        ),
        "key_ddx": (
            "HNF1A/MODY3 (progressive — glycosuria at normal glucose; large HbA1c rise over time; sulfonylruea response); "
            "HNF4A/MODY1 (neonatal HI history; macrosomic infant; progressive β-cell failure); "
            "Pre-diabetes/T2D (insulin resistant component; glucose increment on OGTT large >4.5 mmol/L; progressive); "
            "MODY2 increment rule: fasting-to-2hr increment <3.5 mmol/L vs T2D increment >4.5 mmol/L; "
            "Type 1 DM (autoantibodies positive; rapid β-cell loss; ketoacidosis risk)"
        ),
        "gfr_pattern": "Normal — GCK-MODY2 does not cause nephropathy at typical glucose levels",
        "proteinuria_pattern": "Absent at usual glucose levels; screen annually in pregnancy",
        "primary_complication": "OVER-TREATMENT causing hypoglycemia; macrosomia if pregnancy not managed by fetal genotype",
        "disease_detail": (
            "GCK-MODY2 is clinically the most 'benign' MODY and yet the most commonly over-treated. "
            "The glucose set-point shift is a mathematical consequence of the enzyme's rightward kinetic shift: "
            "GCK LOF raises the threshold from ~5 mmol/L to ~7 mmol/L. "
            "At this level, HbA1c is 6.0–6.5% — a value that triggers therapy in many guidelines. "
            "The key insight is that this is NOT hyperglycemia due to inadequate response — "
            "it is the correct response to a recalibrated sensor. Treating with insulin 'over-corrects' the sensor "
            "and causes hypoglycemia. The real clinical burden of MODY2 is diagnostic, not therapeutic."
        ),
        "variants": [
            {"name": "p.Gly44Ser", "frequency": "missense; catalytic cleft; common MODY2"},
            {"name": "p.Val367Met", "frequency": "missense; allosteric activator site"},
            {"name": "p.Arg377Ter", "frequency": "truncating; complete LOF; severe MODY2"},
        ],
        "drug_ci": [
            "INSULIN: contra-indicated in MODY2 (causes hypoglycemia without benefit) unless very high HbA1c + proven co-pathology",
            "SULFONYLUREAS: avoid (same rationale — hypoglycemia without benefit)",
            "SGLT2 INHIBITORS: not recommended (glucose lowering inappropriate in MODY2)",
            "GLP-1 AGONISTS: weight management only if obese; not for glucose lowering in MODY2",
        ],
    },

    # ── HNF1A — MODY3 ──────────────────────────────────────────────────────────
    {
        "gene": "HNF1A",
        "protein": "Hepatocyte nuclear factor 1-alpha (HNF-1α, TCF1)",
        "alias": (
            "HNF1A; OMIM gene 142410; MODY3; 12q24.31; 631 aa; ~69 kDa; "
            "AD; POU/homeodomain TF; MODY3 — most common symptomatic MODY in European populations 30–40%; "
            "GLUCOSURIA at NORMAL glucose — PATHOGNOMONIC; sulfonylureas 5–8× more effective than insulin"
        ),
        "aa": "631 aa",
        "kDa": "~69 kDa",
        "locus": "12q24.31",
        "omim_gene": 142410,
        "omim_disease": 600496,
        "inheritance": "AD (haploinsufficiency)",
        "gene_class": (
            "POU-homeodomain transcription factor; binds HNF1 palindromic promoter element; "
            "HNF1A controls SGLT2 (sodium-glucose cotransporter 2 in kidney), GLUT2, "
            "glucokinase, insulin, albumin, alpha-1-antitrypsin, and fibrinogen gene expression; "
            "HNF1A LOF → SGLT2 downregulation → LOW RENAL GLUCOSE THRESHOLD (~5.5 mmol/L vs normal 10 mmol/L); "
            "consequence: GLUCOSURIA at blood glucose levels that would NOT normally cause glycosuria; "
            "HNF1A LOF → β-cell haploinsufficiency → impaired GSIS (glucose-stimulated insulin secretion); "
            "unlike GCK, MODY3 is PROGRESSIVE: β-cell failure advances over years → insulin eventually required; "
            "HNF1A controls HNF4A promoter: MODY1 and MODY3 are transcriptionally linked; "
            "Mutations: exon 4 Pro291 frameshift is the most common single MODY3 mutation (European); "
            "Heterodimer: HNF1A dimerises with HNF1B → compound HNF1A/HNF1B pathway defects (rare)"
        ),
        "phenotype": (
            "Onset typically 10–40 years; mean age 26 years in large series; "
            "Hyperglycemia: progressive; starts with post-prandial hyperglycemia → fasting hyperglycemia; "
            "GLUCOSURIA: urine glucose positive at normal or only mildly elevated blood glucose — PATHOGNOMONIC; "
            "HbA1c underestimation: red cell glucose-6-phosphate dehydrogenase-related in some populations; "
            "Microvascular complications: retinopathy + nephropathy risk similar to T1D/T2D if poor control; "
            "Reduced sensitivity to insulin (mild): not the primary defect — secretory defect dominates; "
            "Sensory neuropathy: occurs in poorly controlled long-standing disease; "
            "Hepatic adenoma: HNF1A inactivation in liver → multiple hepatic adenomas in ~10% (liver NOT same as β-cell mutation); "
            "CRP: extremely low hsCRP (<0.5 mg/L) in MODY3 — useful discriminating feature vs T2D"
        ),
        "hallmark": (
            "GLUCOSURIA AT NORMAL BLOOD GLUCOSE — PATHOGNOMONIC MODY3: send urine glucose when blood glucose <10 mmol/L; "
            "EXTREMELY LOW hsCRP (<0.5 mg/L) — discriminates MODY3 from T2D (T2D hsCRP often >1 mg/L); "
            "SULFONYLUREAS 5–8× MORE EFFECTIVE than insulin — switch from insulin to low-dose SU; "
            "PROGRESSIVE β-cell failure — unlike MODY2 (GCK); insulin eventually required in older patients; "
            "HEPATIC ADENOMAS (multiple): if HNF1A somatic mutation in liver — NOT related to β-cell mutation; "
            "Pro291fsdelC: single most common MODY3 mutation — genotype first in European families; "
            "LEAN + AD FAMILY HISTORY + GLUCOSURIA in young adult = MODY3 until proven otherwise; "
            "HbA1c-to-increment rule: 2h OGTT increment >4.5 mmol/L suggests MODY3 over MODY2"
        ),
        "treatment_alert": (
            "SULFONYLUREAS DRAMATICALLY MORE EFFECTIVE: glibenclamide 1.25–10 mg/day or gliclazide MR 15–60 mg/day; "
            "START VERY LOW DOSE: MODY3 β-cells are exquisitely sensitive to SU — 1.25 mg glibenclamide or 15 mg gliclazide MR; "
            "SWITCH FROM INSULIN TO SU: patients already on insulin can switch to SU with 50–70% insulin dose reduction; "
            "METFORMIN: minimal efficacy (secretory not resistance defect) but safe to use if obese; "
            "GLP-1 AGONISTS (liraglutide/dulaglutide): useful adjunct especially with weight gain; "
            "SGLT2 INHIBITORS: CAUTION — HNF1A downregulates SGLT2; SGLT2i may have LESS effect in MODY3 "
            "(some evidence reduced efficacy vs T2D); euglycaemic DKA risk not well-characterised; "
            "PREGNANCY: SU crosses placenta — consider switching to insulin in 2nd/3rd trimester; "
            "RIFAMPICIN + SU: CYP2C9 induction → reduced SU levels → hyperglycemia; monitor and increase dose; "
            "ANNUAL MONITORING: HbA1c, retinal screen, uACR/eGFR, foot exam from diagnosis"
        ),
        "key_ddx": (
            "GCK/MODY2 (mild stable fasting hyperglycemia; no glucosuria; no progression; no SU response needed); "
            "HNF4A/MODY1 (neonatal HI; macrosomia; same era of onset; similar SU response); "
            "HNF1B/MODY5 (renal cysts + multi-organ; SU often ineffective; no glucosuria); "
            "Type 1 DM (autoantibodies; rapid β-cell failure; ketonaemia; C-peptide absent); "
            "Type 2 DM (hsCRP elevated; BMI high; insulin resistant; no glucosuria at normal glucose)"
        ),
        "gfr_pattern": "Low renal glucose threshold (SGLT2 downregulated) — glucosuria; eGFR normal early; nephropathy risk with poor control",
        "proteinuria_pattern": "Microalbuminuria with prolonged hyperglycemia; SGLT2i caution (reduced baseline SGLT2)",
        "primary_complication": "Progressive β-cell failure → microvascular complications; hepatic adenomas (somatic HNF1A LOF in liver)",
        "disease_detail": (
            "HNF1A-MODY3 is the most clinically impactful MODY because it is: "
            "(1) the most common symptomatic MODY in Europeans, (2) progressive and needs treatment, "
            "(3) has a specific and highly effective treatment (SU), (4) has a pathognomonic clinical sign (glucosuria), "
            "and (5) responds dramatically differently to treatment vs T1D/T2D. "
            "The 'switch to sulfonylurea' is one of the clearest examples of precision medicine in diabetes. "
            "A young lean patient with family history of diabetes, glucosuria, and being on insulin should be tested for HNF1A mutation. "
            "The hsCRP discrimination test (MODY3: <0.5 mg/L vs T2D: >1 mg/L) is a useful cheap screening tool."
        ),
        "variants": [
            {"name": "p.Pro291fsdelC", "frequency": "most common MODY3 mutation; European; frameshift exon 4"},
            {"name": "p.Glu508Lys", "frequency": "missense; homeodomain; moderate phenotype"},
            {"name": "p.Arg272His", "frequency": "missense; POU domain; reduced DNA binding"},
        ],
        "drug_ci": [
            "EXCESS ALCOHOL + SU: severe prolonged hypoglycemia — limit alcohol strictly",
            "RIFAMPICIN: CYP2C9/CYP2C19 inducer — dramatically reduces SU levels; increase SU dose or switch AED",
            "SGLT2 INHIBITORS: reduced efficacy in MODY3 (low baseline SGLT2); euglycaemic DKA risk unknown — caution",
            "ASPIRIN (high dose): displaces SU from albumin → hypoglycemia; use low-dose only",
        ],
    },

    # ── PDX1 — MODY4 ───────────────────────────────────────────────────────────
    {
        "gene": "PDX1",
        "protein": "Pancreatic and duodenal homeobox protein 1 (PDX1, IPF1, IDX1)",
        "alias": (
            "PDX1; OMIM gene 600733; MODY4; 13q12.2; 283 aa; ~31 kDa; "
            "AD (heterozygous MODY4) / AR (homozygous neonatal pancreatic agenesis); "
            "PANCREATIC EXOCRINE INSUFFICIENCY 40% even heterozygous — enzyme supplementation mandatory"
        ),
        "aa": "283 aa",
        "kDa": "~31 kDa",
        "locus": "13q12.2",
        "omim_gene": 600733,
        "omim_disease": 606392,
        "inheritance": "AD (heterozygous); AR (neonatal pancreatic agenesis homozygous)",
        "gene_class": (
            "Homeodomain transcription factor; master regulator of pancreatic development and β-cell identity; "
            "PDX1 controls insulin, somatostatin, glucokinase, GLUT2, and islet amyloid polypeptide expression; "
            "PDX1 is absolutely required for pancreatic budding at embryonic day E8.5 (mouse); "
            "Homozygous PDX1 LOF → complete pancreatic agenesis (no endocrine + no exocrine pancreas); "
            "Heterozygous PDX1 LOF → haploinsufficiency → MODY4 (adult onset); "
            "Pancreatic exocrine insufficiency: even heterozygous carriers have reduced acinar cell mass → "
            "steatorrhoea, fat-soluble vitamin malabsorption, weight loss; "
            "PDX1 also controls MAFA (β-cell maturation factor) → PDX1 LOF → immature β-cell phenotype; "
            "Mutation spectrum: missense in homeodomain (DNA binding); truncating (null alleles more severe)"
        ),
        "phenotype": (
            "Heterozygous MODY4: onset in adulthood (typically 30s–50s); "
            "Hyperglycemia: progressive; similar to MODY3 trajectory; "
            "Pancreatic exocrine insufficiency (PEI) in ~40% even heterozygous: steatorrhoea, "
            "fat-soluble vitamin deficiency (A, D, E, K); malabsorption; weight loss; "
            "NO hepatic adenomas (distinguishes from MODY3 liver involvement); "
            "NO renal anomalies (distinguishes from MODY5); "
            "Homozygous (very rare): neonatal permanent DM + complete exocrine insufficiency + pancreatic agenesis "
            "on imaging (absent pancreas); extremely severe presentation at birth"
        ),
        "hallmark": (
            "PANCREATIC EXOCRINE INSUFFICIENCY (PEI) in 40% even heterozygous — steatorrhoea + fat-soluble vitamin deficiency; "
            "ABSENT PANCREAS on imaging in homozygous: pancreatic agenesis — PATHOGNOMONIC of compound heterozygous/homozygous PDX1 LOF; "
            "ENZYME SUPPLEMENTATION MANDATORY if PEI confirmed: pancreatic enzyme replacement therapy (PERT) with meals; "
            "HOMOZYGOUS NEONATAL DM: permanent neonatal diabetes at birth + absent pancreas; "
            "VITAMIN SUPPLEMENTATION: A, D, E, K (fat-soluble) — measure levels and replace; "
            "RARE MODY: <1% of MODY referrals; genetic testing detects; "
            "MISDIAGNOSIS: steatorrhoea in diabetic patient → pancreatic exocrine insufficiency → think MODY4"
        ),
        "treatment_alert": (
            "PANCREATIC ENZYME REPLACEMENT THERAPY (PERT): creon 25000–50000 lipase units with meals — MANDATORY if PEI; "
            "FAT-SOLUBLE VITAMINS: monitor and replace A, D, E, K; "
            "GLYCEMIC: sulfonylureas first line for heterozygous MODY4 if glucose control required; "
            "INSULIN: for severe β-cell insufficiency or neonatal form; "
            "HOMOZYGOUS NEONATAL: insulin from birth; PERT from birth; fat-soluble vitamins lifelong; "
            "BONE DENSITY: dual risk (diabetes + vitamin D/K deficiency + malabsorption) → DEXA scan; "
            "DIETITIAN INPUT: specialist dietitian mandatory — pancreatic diet vs diabetic diet complex; "
            "DO NOT MISS: steatorrhoea in a MODY patient = PEI until proven otherwise"
        ),
        "key_ddx": (
            "HNF1A/MODY3 (no PEI; glucosuria; hepatic adenomas); "
            "HNF1B/MODY5 (renal cysts; elevated GGT; exocrine insufficiency also occurs); "
            "Chronic pancreatitis (calcifications on CT; alcohol/stones history; pain prominent); "
            "Cystic fibrosis-related DM (CFRD — CFTR mutation; lung disease; sweat test); "
            "Autoimmune pancreatitis (IgG4; steroid-responsive; imaging: diffuse enlargement)"
        ),
        "gfr_pattern": "Normal — no primary renal phenotype in heterozygous PDX1",
        "proteinuria_pattern": "Absent primarily; microalbuminuria with chronic hyperglycemia",
        "primary_complication": "Pancreatic exocrine insufficiency → malnutrition; hyperglycemia → microvascular; vitamin deficiency",
        "disease_detail": (
            "PDX1-MODY4 is rare but clinically distinctive because of the exocrine insufficiency component. "
            "Many clinicians associate diabetes with preserved pancreatic exocrine function — "
            "MODY4 is a critical exception. The combination of steatorrhoea and diabetes "
            "in a non-obese, non-alcoholic patient should prompt PDX1 testing. "
            "The homozygous form is one of only two conditions causing true neonatal pancreatic agenesis "
            "(the other being GATA6 mutations). WGS/WES can detect PDX1 homozygous cases missed by panels."
        ),
        "variants": [
            {"name": "p.Pro63fsdelC", "frequency": "frameshift; exon 1; heterozygous MODY4"},
            {"name": "p.Ile236Ser", "frequency": "homeodomain; moderate loss of function"},
            {"name": "p.Gln59Ter", "frequency": "truncating; severe LOF; homozygous agenesis in compound carriers"},
        ],
        "drug_ci": [
            "FAT MALABSORPTION: oral medications with high lipid solubility may have reduced absorption — check formulations",
            "WARFARIN + vitamin K malabsorption: INR highly variable; prefer NOAC or very frequent INR monitoring",
            "METFORMIN: only safe if exocrine-related B12 deficiency excluded (concurrent B12 monitoring if on metformin)",
        ],
    },

    # ── HNF1B — MODY5 ──────────────────────────────────────────────────────────
    {
        "gene": "HNF1B",
        "protein": "Hepatocyte nuclear factor 1-beta (HNF-1β, TCF2, vHNF1)",
        "alias": (
            "HNF1B; OMIM gene 189907; MODY5; 17q12; 557 aa; ~61 kDa; "
            "AD; POU/homeodomain TF; MODY5 — multi-organ syndrome: renal cysts + DM + liver; "
            "RENAL ANOMALIES before DM onset; SULFONYLUREAS OFTEN INEFFECTIVE — early insulin"
        ),
        "aa": "557 aa",
        "kDa": "~61 kDa",
        "locus": "17q12",
        "omim_gene": 189907,
        "omim_disease": 137920,
        "inheritance": "AD (haploinsufficiency or GOF dominant-negative)",
        "gene_class": (
            "POU/homeodomain transcription factor; expressed in kidney collecting duct, pancreatic progenitors, "
            "bile ducts, genital tract, and liver; "
            "HNF1B heterodimerises with HNF1A → transactivates shared promoter targets; "
            "HNF1B controls renal tubular development, collecting duct branching, and nephron segmentation: "
            "LOF → renal dysplasia, collecting duct cysts, hypoplastic kidneys; "
            "HNF1B in pancreas → controls PDX1 and PTF1A → exocrine and endocrine pancreatic development; "
            "HNF1B LOF → atrophic pancreas + exocrine insufficiency (50–80% carriers); "
            "HNF1B in liver → bile duct differentiation; LOF → elevated GGT/cholestasis; "
            "HNF1B in Müllerian structures → female genital tract development: "
            "bicornuate uterus, vaginal aplasia (Mayer-Rokitansky-Küster-Hauser overlap); "
            "Whole-gene deletions (17q12 microdeletion) common: MLPA essential; "
            "17q12 microdeletion syndrome: MODY5 + psychiatric features (schizophrenia risk 34×)"
        ),
        "phenotype": (
            "RENAL: anomalies present BEFORE diabetes onset (often in utero or childhood); "
            "cystic dysplasia, oligomeganephronia, horseshoe kidney, renal agenesis; "
            "ESRD in 25–30% by age 40; "
            "DIABETES: onset 20s–40s; progressive β-cell dysfunction; "
            "LIVER: elevated GGT ± elevated ALT; neonatal cholestasis in some; "
            "EXOCRINE INSUFFICIENCY: 50–80%; atrophic pancreas on MRI/CT; steatorrhoea; "
            "GENITAL TRACT FEMALE: bicornuate uterus, vaginal aplasia, Müllerian anomalies in ~40%; "
            "GENITAL TRACT MALE: vas deferens aplasia → infertility; "
            "GOUT: hyperuricaemia (tubular secretion defect) in ~40%; "
            "PSYCHIATRIC (17q12 deletion): schizophrenia, autism, intellectual disability risk"
        ),
        "hallmark": (
            "RENAL CYSTS + DIABETES + ELEVATED GGT TRIAD — PATHOGNOMONIC of MODY5/HNF1B; "
            "RENAL ANOMALIES PRECEDE DIABETES: patient may be known to nephrology before diagnosis of DM; "
            "PANCREATIC ATROPHY ON IMAGING: small/atrophic pancreas — 50–80% on CT/MRI; "
            "SULFONYLUREAS OFTEN INEFFECTIVE: underlying exocrine + endocrine deficiency → insulin usually required; "
            "GOUT: hyperuricaemia with normal renal function — tubular urate secretion defect; "
            "17q12 DELETION: MLPA MANDATORY — sequencing alone misses large deletions; "
            "FEMALE GENITAL ANOMALIES: uterine abnormalities in 40% — screen all females with HNF1B; "
            "PSYCHIATRIC: 17q12 microdeletion → 34× increased schizophrenia risk — psychiatric referral"
        ),
        "treatment_alert": (
            "SULFONYLUREAS: often ineffective in MODY5 (limited secretory reserve + exocrine insufficiency); "
            "INSULIN: early initiation usually required — basal-bolus preferred; "
            "PANCREATIC ENZYMES: PERT mandatory if exocrine insufficient (50–80%); "
            "RENAL MONITORING: annual eGFR + urine ACR; ESRD in 25–30% → transplant planning; "
            "GOUT: allopurinol or febuxostat for hyperuricaemia; NSAIDs CAUTION (renal impairment risk); "
            "GENETIC COUNSELLING: 17q12 deletion in offspring → psychiatric risk; "
            "MLPA TESTING: mandatory — 30–40% of HNF1B cases are whole-gene deletion missed by sequencing; "
            "RENAL TRANSPLANT: MODY5 patients require early nephrology input — transplant planning at eGFR <30; "
            "VITAMIN D: at high risk (exocrine insufficiency + CKD) — supplement aggressively"
        ),
        "key_ddx": (
            "HNF1A/MODY3 (no renal cysts; glucosuria; SU highly effective; no exocrine insufficiency); "
            "PDX1/MODY4 (no renal anomalies; exocrine insufficiency present but less common); "
            "ADPKD (PKD1/PKD2 — bilateral large cysts; no DM association; no GGT; no exocrine); "
            "MODY with nephropathy (HNF4A late renal — no cysts; no genital anomalies); "
            "Medullary cystic kidney disease (UMOD/MUC1 — no DM association; no exocrine)"
        ),
        "gfr_pattern": "Progressive decline; renal dysplasia → CKD from childhood/young adult; ESRD 25-30% by age 40",
        "proteinuria_pattern": "Tubular proteinuria + microalbuminuria; structural renal anomaly → CKD progression",
        "primary_complication": "ESRD; pancreatic exocrine insufficiency; female infertility; gout; psychiatric risk (17q12 deletion)",
        "disease_detail": (
            "HNF1B-MODY5 is the most phenotypically complex MODY — it is fundamentally a multi-organ syndrome "
            "where diabetes is only one manifestation. "
            "The combination of cystic kidney disease (often diagnosed in utero), "
            "diabetes (diagnosed years later), elevated GGT, and pancreatic atrophy in the same patient "
            "is PATHOGNOMONIC and should trigger HNF1B testing immediately. "
            "The 17q12 microdeletion subset has additional psychiatric risk — "
            "schizophrenia at 34× the general population, requiring lifelong psychiatric monitoring. "
            "MLPA is not optional in HNF1B — up to 40% of disease alleles are whole-gene deletions."
        ),
        "variants": [
            {"name": "17q12 microdeletion", "frequency": "40% of HNF1B cases; MLPA mandatory to detect"},
            {"name": "p.Arg177Ter", "frequency": "truncating; severe; renal agenesis + neonatal DM"},
            {"name": "p.Lys156Glu", "frequency": "missense; homeodomain; milder renal phenotype"},
        ],
        "drug_ci": [
            "NSAIDs for gout: ABSOLUTE CAUTION — renal impairment risk; use allopurinol/febuxostat instead",
            "NEPHROTOXIC DRUGS: aminoglycosides, contrast agents, NSAIDs — avoid or use with eGFR monitoring",
            "SULFONYLUREAS: often ineffective; discontinue if no HbA1c response within 3 months",
            "TACROLIMUS/CYCLOSPORINE post-transplant: diabetogenic — monitor glucose intensely post-renal transplant",
        ],
    },

    # ── NEUROD1 — MODY6 ────────────────────────────────────────────────────────
    {
        "gene": "NEUROD1",
        "protein": "Neurogenic differentiation factor 1 (NeuroD1, BETA2)",
        "alias": (
            "NEUROD1; OMIM gene 601724; MODY6; 2q31.3; 356 aa; ~39 kDa; "
            "AD (heterozygous MODY6) / AR (neonatal DM + cerebellar hypoplasia); "
            "SENSORINEURAL HEARING LOSS + visual impairment in homozygous; neurological phenotype rare"
        ),
        "aa": "356 aa",
        "kDa": "~39 kDa",
        "locus": "2q31.3",
        "omim_gene": 601724,
        "omim_disease": 606394,
        "inheritance": "AD (heterozygous MODY6); AR (neonatal DM + neurological features)",
        "gene_class": (
            "Basic helix-loop-helix (bHLH) transcription factor; class B bHLH (tissue-restricted); "
            "NEUROD1 dimerises with class A bHLH E-proteins (E12/E47) → binds E-box (CANNTG) in insulin + islet gene promoters; "
            "NEUROD1 controls insulin gene transcription, β-cell survival, and β-cell maturation; "
            "Also expressed in cerebellum, hippocampus, and cochlear hair cells: "
            "NEUROD1 is required for cerebellar granule cell survival + cochlear hair cell differentiation; "
            "Homozygous NEUROD1 LOF → neonatal/early-infantile DM + cerebellar hypoplasia + SNHL + visual impairment; "
            "Heterozygous NEUROD1 LOF → MODY6: adult-onset DM with relatively mild phenotype; "
            "NEUROD1 mutation spectrum: missense in bHLH domain (DNA binding or dimerisation); truncating"
        ),
        "phenotype": (
            "Heterozygous MODY6: onset in adulthood (30s–50s); "
            "Hyperglycemia: mild to moderate; similar trajectory to MODY3 but more variable; "
            "NO neurological features in heterozygous; "
            "NO renal/liver anomalies; "
            "Homozygous (very rare): "
            "  • Permanent neonatal DM (NDM) presenting in first 6 months; "
            "  • Cerebellar hypoplasia → ataxia (100% of homozygous); "
            "  • Sensorineural hearing loss (SNHL) — bilateral, early-onset; "
            "  • Visual impairment (reduced visual acuity, retinal dystrophy); "
            "Family history: AD for MODY6 (50% risk per child of heterozygous carrier)"
        ),
        "hallmark": (
            "CEREBELLAR HYPOPLASIA + SNHL + NDM in homozygous — PATHOGNOMONIC neurological MODY triad; "
            "HETEROZYGOUS: pure MODY phenotype without neurological features — mild, often undiagnosed; "
            "RARE MODY: <0.5% of MODY; genetic panel testing detects; "
            "COCHLEAR HAIR CELL NEUROD1 EXPRESSION: explains SNHL in homozygous; "
            "CONSANGUINITY FLAG: neonatal DM + cerebellar ataxia + SNHL → screen NEUROD1 compound heterozygous; "
            "BRAIN MRI (homozygous): cerebellar vermis and hemisphere hypoplasia — diagnostic; "
            "AUDIOGRAM (homozygous): bilateral SNHL — cochlear implants if severe"
        ),
        "treatment_alert": (
            "HETEROZYGOUS MODY6: sulfonylureas first line (similar approach to MODY3); "
            "HOMOZYGOUS NDM: insulin from neonatal period (cannot use SU for neonatal DM); "
            "SNHL MANAGEMENT: audiology referral; cochlear implants if profound SNHL; "
            "CEREBELLAR ATAXIA: physiotherapy; occupational therapy; specialist neurological input; "
            "OPHTHALMOLOGY: annual visual acuity + retinal assessment (homozygous); "
            "GENETIC COUNSELLING: consanguineous couples from affected families: 25% risk homozygous; "
            "EDUCATIONAL SUPPORT: cerebellar involvement may affect learning (mild-moderate)"
        ),
        "key_ddx": (
            "HNF1A/MODY3 (no neurological features; glucosuria; more common); "
            "ABCC8/KCNJ11 NDM (neonatal DM; sulfonylurea-responsive; no cerebellar; no SNHL); "
            "JOHANSON-BLIZZARD syndrome (UBR1; exocrine insufficiency + DM + multi-organ; no cerebellar); "
            "Mitochondrial DM (MELAS; mtDNA mutations; lactic acidosis; SNHL can overlap — but maternal inheritance + other features)"
        ),
        "gfr_pattern": "Normal — no primary renal phenotype in NEUROD1 MODY6",
        "proteinuria_pattern": "Absent primarily; microalbuminuria with progressive hyperglycemia",
        "primary_complication": "Heterozygous: standard MODY progression; Homozygous: neurological disability + SNHL",
        "disease_detail": (
            "NEUROD1-MODY6 is clinically important primarily for its homozygous form, "
            "which represents a multi-organ neurodevelopmental syndrome. "
            "Clinicians should consider NEUROD1 in any child with: "
            "neonatal diabetes + cerebellar ataxia + hearing loss. "
            "The heterozygous form is clinically mild and rarely diagnosed without a family history or genetic panel. "
            "The cerebellar expression of NEUROD1 explains why the neurological phenotype is specific to homozygous LOF — "
            "a single functional allele is sufficient for normal brain development."
        ),
        "variants": [
            {"name": "p.Arg111Leu", "frequency": "bHLH dimerisation domain; heterozygous MODY6"},
            {"name": "p.Gly212Arg", "frequency": "bHLH; homozygous → neonatal DM + cerebellar"},
            {"name": "p.Gln176Ter", "frequency": "truncating; heterozygous MODY6; milder"},
        ],
        "drug_ci": [
            "AMINOGLYCOSIDES (homozygous): SNHL already present — aminoglycosides worsen cochlear damage; AVOID",
            "CISPLATIN/CARBOPLATIN: cochlear toxicity — avoid in homozygous SNHL patients",
            "SODIUM VALPROATE: epilepsy management in cerebellar variant — check mitochondrial function first (VPA + mitochondrial DM risk)",
        ],
    },

    # ── ABCC8 — MODY12/NDM ─────────────────────────────────────────────────────
    {
        "gene": "ABCC8",
        "protein": "ATP-binding cassette subfamily C member 8 (SUR1 — sulfonylurea receptor 1)",
        "alias": (
            "ABCC8; OMIM gene 600509; MODY12/NDM; 11p15.1; 1581 aa; ~177 kDa; "
            "AD GOF → permanent neonatal DM; LOF AR → congenital hyperinsulinism; "
            "KATP channel regulatory subunit; SULFONYLUREA SWITCH: 90% NDM can stop insulin → oral glyburide"
        ),
        "aa": "1581 aa",
        "kDa": "~177 kDa",
        "locus": "11p15.1",
        "omim_gene": 600509,
        "omim_disease": 606176,
        "inheritance": "AD GOF (NDM/MODY12); AR LOF (congenital hyperinsulinism)",
        "gene_class": (
            "ABC transporter superfamily; SUR1 is the regulatory subunit of the KATP channel; "
            "KATP channel = octamer: 4 × SUR1 (regulatory/sulfonylurea-binding) + 4 × Kir6.2 (KCNJ11, pore-forming); "
            "Sulfonylureas bind SUR1 → inhibit KATP → depolarise β-cell → Ca2+ influx → insulin secretion; "
            "MgADP binds SUR1 NBF2 → opens KATP → hyperpolarises → stops insulin secretion; "
            "GOF ABCC8 mutations → KATP channel constitutively OPEN → β-cell hyperpolarised → "
            "Ca2+ channels closed → no insulin secretion → PERMANENT NEONATAL DM (PNDM); "
            "LOF ABCC8 mutations → KATP channel constitutively CLOSED → β-cell depolarised → "
            "excess insulin secretion → CONGENITAL HYPERINSULINISM (CHI); "
            "Focal CHI: somatic LOF ABCC8/KCNJ11 + constitutional LOF → focal lesion → surgically curable; "
            "Diffuse CHI: biallelic LOF → diffuse hypersecretion → pancreatectomy may be needed; "
            "MODY12: milder GOF or heterozygous → later-onset DM phenotype (not neonatal)"
        ),
        "phenotype": (
            "PNDM (GOF ABCC8): DM onset <6 months; insulin-requiring from birth; "
            "Low birth weight (intrauterine insulin deficiency); "
            "Ketoacidosis possible at presentation; "
            "No autoimmunity: autoantibodies negative; C-peptide detectable (β-cells present, just not secreting); "
            "MODY12 (later GOF): DM onset in adolescence–adulthood; sulfonylurea-responsive; "
            "CHI (LOF AR): neonatal hypoglycemia; macrosomia (intrauterine hyperinsulinism); "
            "focal CHI: asymmetric islets → PET-CT with F-DOPA localises lesion → surgical cure; "
            "diffuse CHI: near-total pancreatectomy risk → secondary DM in adulthood; "
            "No neurological features in ABCC8 (distinguishes from KCNJ11 DEND)"
        ),
        "hallmark": (
            "SULFONYLUREA SWITCH IN PNDM: 90% of ABCC8-PNDM can discontinue insulin and switch to oral glyburide/glibenclamide; "
            "DOSE FOR PNDM SU SWITCH: much higher than MODY3 — 0.2–0.8 mg/kg/day glibenclamide; "
            "START EARLY: SU switch before 6 months of age achieves better metabolic control; "
            "LOW BIRTH WEIGHT: intrauterine insulin deficiency → small for gestational age — diagnose PNDM; "
            "AUTOANTIBODIES NEGATIVE: PNDM is NOT autoimmune — GADA/IA-2/ZnT8 negative; "
            "C-PEPTIDE DETECTABLE: β-cells present but functionally suppressed by open KATP; "
            "FOCAL CHI: F-DOPA PET-CT before pancreatectomy — surgical cure for focal lesion; "
            "NO DEND SYNDROME in ABCC8 — neurological features absent (unlike KCNJ11 severe GOF)"
        ),
        "treatment_alert": (
            "SULFONYLUREA SWITCH (PNDM): replace insulin with glyburide/glibenclamide oral: "
            "  • Dose 0.2–0.8 mg/kg/day in 2–3 divided doses; "
            "  • Monitor glucose intensely for first 2 weeks of switch; "
            "  • 90% achieve glycemic control equivalent or better than insulin; "
            "  • Do NOT use metformin as primary substitute — secretory defect; "
            "CHI LOF MANAGEMENT: "
            "  • Diazoxide (KATP opener) first line — 5–15 mg/kg/day; "
            "  • F-DOPA PET-CT for focal vs diffuse CHI; "
            "  • Focal CHI: surgical lesionectomy → cure; "
            "  • Diffuse CHI diazoxide-unresponsive: near-total pancreatectomy (secondary DM risk); "
            "MODY12: low-dose sulfonylurea (as per MODY3 protocol); "
            "MONITOR: HbA1c, hypoglycemia frequency, growth (child), neurological development"
        ),
        "key_ddx": (
            "KCNJ11 (same KATP channel pore subunit — DEND syndrome in severe GOF; same SU switch; co-segregation test); "
            "Type 1 DM (autoantibodies positive; ABCC8-PNDM autoantibody-negative; C-peptide positive in ABCC8); "
            "MODY3/HNF1A (no neonatal onset; glucosuria; older age of onset); "
            "Wolfram syndrome (WFS1 — DM + DI + optic atrophy + deafness — progressive neurodegeneration)"
        ),
        "gfr_pattern": "Normal — no primary renal phenotype in ABCC8",
        "proteinuria_pattern": "Absent; microalbuminuria with chronic hyperglycemia in PNDM patients not optimally controlled",
        "primary_complication": "PNDM: if not SU-switched — insulin therapy burdensome; CHI: neurological damage from hypoglycemia",
        "disease_detail": (
            "ABCC8 and KCNJ11 (KATP channel subunits) represent the most actionable neonatal DM genes. "
            "The sulfonylurea switch is one of the most dramatic therapeutic advances in diabetes genetics: "
            "a child on insulin since birth can, with ABCC8 GOF identification, "
            "switch to oral sulfonylurea and achieve excellent glycemic control. "
            "The mechanism is direct: sulfonylureas (glyburide/glibenclamide) bind SUR1 and force KATP channel closure, "
            "bypassing the GOF mutation and restoring Ca2+-mediated insulin secretion. "
            "The earlier the switch, the better the outcome — both metabolic and neurodevelopmental."
        ),
        "variants": [
            {"name": "p.Arg1380His", "frequency": "NBF2 region; common NDM GOF variant"},
            {"name": "p.Gly334Arg", "frequency": "TMD2; severe GOF; PNDM"},
            {"name": "p.Ala1185Val", "frequency": "milder GOF; MODY12 late onset"},
        ],
        "drug_ci": [
            "DIAZOXIDE (GOF ABCC8 NDM): paradoxically worsens if prescribed for DM (opens KATP — same target as GOF); AVOID in GOF NDM",
            "QUININE + QUINIDINE: inhibit SUR1 → additional KATP effects — caution in GOF patients",
            "OCTREOTIDE: reduces insulin secretion — useful in CHI LOF if diazoxide fails; not for GOF NDM",
        ],
    },

    # ── KCNJ11 — MODY13/NDM ────────────────────────────────────────────────────
    {
        "gene": "KCNJ11",
        "protein": "ATP-sensitive inward rectifier potassium channel 11 (Kir6.2)",
        "alias": (
            "KCNJ11; OMIM gene 600937; MODY13/PNDM; 11p15.1; 390 aa; ~43 kDa; "
            "AD GOF → permanent neonatal DM; DEND syndrome (severe GOF); "
            "KATP channel pore subunit; SULFONYLUREA SWITCH: start <6 months for optimal outcome"
        ),
        "aa": "390 aa",
        "kDa": "~43 kDa",
        "locus": "11p15.1",
        "omim_gene": 600937,
        "omim_disease": 610582,
        "inheritance": "AD GOF (PNDM/DEND/iDEND/MODY13); AR LOF (hyperinsulinism — rare)",
        "gene_class": (
            "Inward rectifier potassium channel family (Kir); Kir6.2 is the pore-forming subunit of KATP channel; "
            "KATP channel = 4 × Kir6.2 (pore) + 4 × SUR1/SUR2 (regulatory); "
            "Kir6.2 forms the K+-selective pore: when ATP binds Kir6.2 → closes KATP → depolarises β-cell → insulin secretion; "
            "GOF Kir6.2 mutations → reduced ATP sensitivity → KATP channel stays OPEN even at HIGH ATP/ADP ratio → "
            "β-cell constitutively hyperpolarised → no Ca2+ entry → no insulin secretion → PNDM; "
            "Mutation severity determines phenotype: "
            "  • Mild GOF: PNDM only (no neurological features) — common; "
            "  • Moderate GOF: iDEND — PNDM + developmental delay + muscle weakness (no epilepsy); "
            "  • Severe GOF: DEND syndrome — PNDM + Developmental delay + Epilepsy + Neonatal Diabetes; "
            "KATP channels in brain (hippocampus, substantia nigra) + muscle + pancreas: "
            "severe GOF → widespread KATP opening → neurological features (DEND); "
            "Sulfonylureas cross blood-brain barrier → SU closes brain KATP → can IMPROVE neurological features in DEND; "
            "KCNJ11 at 11p15.1 adjacent to ABCC8: "
            "  • Inherited mutations can involve both genes (compound digenic) — rare"
        ),
        "phenotype": (
            "PNDM: DM onset <6 months (median 6 weeks); insulin-dependent; low birth weight; "
            "Autoantibodies negative; C-peptide low-detectable (β-cells present, suppressed); "
            "iDEND (intermediate DEND): PNDM + developmental delay + muscle hypotonia — no epilepsy; "
            "DEND syndrome: PNDM + severe developmental delay + epilepsy (neonatal/infantile) + muscle weakness; "
            "EEG: seizures begin in first months; partially responsive to antiepileptics; "
            "Brain MRI: cerebral atrophy; white matter changes in severe DEND; "
            "MODY13 (milder GOF or heterozygous): later-onset DM (adolescence–adulthood); no neurological; "
            "Muscle: hypotonia in iDEND/DEND — EMG shows myopathic features; "
            "Neurological improvement with SU: 40–70% of DEND/iDEND show improvement in epilepsy + development post-SU switch"
        ),
        "hallmark": (
            "DEND SYNDROME: Developmental delay + Epilepsy + Neonatal Diabetes — PATHOGNOMONIC of severe KCNJ11 GOF; "
            "SULFONYLUREA SWITCH: 90% PNDM respond; iDEND/DEND show ADDITIONAL neurological benefit from SU switch; "
            "START BEFORE 6 MONTHS: SU switch ideally before β-cell irreversible dysfunction develops; "
            "SU CROSSES BBB: improves brain KATP dysfunction → reduced seizure frequency in DEND; "
            "LOW BIRTH WEIGHT: intrauterine insulin deficiency → SGA → diagnose PNDM; "
            "AUTOANTIBODY NEGATIVE + C-PEPTIDE DETECTABLE = NOT T1D → test KCNJ11; "
            "BRAIN MRI IN DEND: cerebral atrophy; white matter signal changes — document at baseline; "
            "MODY13: late-onset → sulfonylurea responsive (same as MODY3 protocol)"
        ),
        "treatment_alert": (
            "SULFONYLUREA SWITCH (PNDM): glyburide/glibenclamide oral — TARGET: glycemic control + possible neurological improvement; "
            "  • Dose 0.2–0.8 mg/kg/day in 2–3 divided doses (PNDM); up to 2.0 mg/kg/day in DEND; "
            "  • Switch must be gradual: reduce insulin by 25% per 48 hours as SU titrated up; "
            "  • 90% PNDM achieve insulin-independence; "
            "  • DEND: 40–70% improvement in seizures + development; incomplete neurological recovery (earlier better); "
            "START BEFORE 6 MONTHS for best metabolic + neurological outcome; "
            "EPILEPSY in DEND: conventional AEDs (levetiracetam, vigabatrin) partially effective; "
            "SU switch is THE primary epilepsy treatment — not substituted by AEDs alone; "
            "DEVELOPMENTAL SUPPORT: multidisciplinary (speech, physio, OT, educational psychology); "
            "GROWTH HORMONE: consider in SGA children with growth failure; "
            "GENETIC COUNSELLING: de-novo KCNJ11 GOF common — family screening; "
            "MODY13 (late onset): low-dose SU as per MODY3 protocol"
        ),
        "key_ddx": (
            "ABCC8 (same channel SUR1 subunit — no DEND syndrome in ABCC8; same SU switch principle; co-segregation); "
            "Type 1 DM (autoantibodies; rapid onset; usually >6 months; C-peptide absent); "
            "Neonatal DM-not-KATP: GCK GOF (activating sensor — treats hypoglycemia); "
            "Epileptic encephalopathy (KCNQ2 — no DM; DEE gene panel; EEG pattern); "
            "Wolfram/WFS1 (progressive neurodegeneration; DI; optic atrophy; later onset; WES/panel)"
        ),
        "gfr_pattern": "Normal — no primary renal phenotype in KCNJ11",
        "proteinuria_pattern": "Absent; microalbuminuria with longstanding hyperglycemia",
        "primary_complication": "DEND syndrome neurological disability; hypoglycemia if SU not correctly dosed; PNDM complications",
        "disease_detail": (
            "KCNJ11-PNDM/DEND is the gene most associated with the landmark 'sulfonylurea switch': "
            "the demonstration that a child with neonatal diabetes (always thought insulin-requiring) "
            "can discontinue insulin and achieve equivalent or better control on oral sulfonylurea. "
            "The neurological benefit in DEND is particularly remarkable — "
            "sulfonylureas cross the blood-brain barrier and close KATP channels in neurons, "
            "reducing the aberrant hyperpolarisation that underlies DEND epilepsy and cognitive impairment. "
            "The earlier the switch (ideally before 6 months), the more complete the recovery. "
            "KCNJ11 GOF testing should be performed in ALL cases of neonatal DM <6 months."
        ),
        "variants": [
            {"name": "p.Arg201His", "frequency": "most common PNDM GOF; nucleotide binding region of Kir6.2"},
            {"name": "p.Arg201Cys", "frequency": "PNDM; moderate GOF"},
            {"name": "p.Val59Met", "frequency": "severe GOF; DEND syndrome; epilepsy + developmental delay"},
            {"name": "p.Glu227Lys", "frequency": "iDEND; intermediate — dev delay + hypotonia, no epilepsy"},
        ],
        "drug_ci": [
            "DIAZOXIDE (GOF KCNJ11 NDM): ABSOLUTELY CONTRAINDICATED — opens KATP further (same target as GOF); worsens DM",
            "CONVENTIONAL AEDs as sole epilepsy treatment in DEND: INSUFFICIENT — SU switch is primary treatment; "
            "valproate caution (mitochondrial risk assessment needed in any encephalopathic neonatal DM)",
            "OCTREOTIDE: reduces insulin secretion — may worsen hyperglycemia in GOF PNDM; avoid",
            "QUINIDINE: modulates KATP — pharmacokinetic interaction in high-dose SU patients",
        ],
    },
]


# ─── Cohort simulation ─────────────────────────────────────────────────────────

def _simulate_gene(gene_entry: dict, seed: int, n: int = 40) -> list:
    rng = random.Random(seed)
    patients = []
    g = gene_entry["gene"]

    # Gene-specific clinical rates (approximate population-level data)
    rates = {
        "HNF4A":   {"drug_error": 0.38, "dx_delay": 0.52, "surveillance": 0.68, "sev_mild": 0.30, "sev_mod": 0.50, "sev_sev": 0.20},
        "GCK":     {"drug_error": 0.55, "dx_delay": 0.60, "surveillance": 0.72, "sev_mild": 0.85, "sev_mod": 0.14, "sev_sev": 0.01},
        "HNF1A":   {"drug_error": 0.42, "dx_delay": 0.58, "surveillance": 0.65, "sev_mild": 0.25, "sev_mod": 0.52, "sev_sev": 0.23},
        "PDX1":    {"drug_error": 0.30, "dx_delay": 0.45, "surveillance": 0.60, "sev_mild": 0.35, "sev_mod": 0.48, "sev_sev": 0.17},
        "HNF1B":   {"drug_error": 0.35, "dx_delay": 0.62, "surveillance": 0.55, "sev_mild": 0.15, "sev_mod": 0.45, "sev_sev": 0.40},
        "NEUROD1": {"drug_error": 0.25, "dx_delay": 0.50, "surveillance": 0.62, "sev_mild": 0.40, "sev_mod": 0.45, "sev_sev": 0.15},
        "ABCC8":   {"drug_error": 0.40, "dx_delay": 0.70, "surveillance": 0.58, "sev_mild": 0.20, "sev_mod": 0.45, "sev_sev": 0.35},
        "KCNJ11":  {"drug_error": 0.42, "dx_delay": 0.72, "surveillance": 0.55, "sev_mild": 0.18, "sev_mod": 0.42, "sev_sev": 0.40},
    }
    r = rates.get(g, {"drug_error": 0.35, "dx_delay": 0.55, "surveillance": 0.60, "sev_mild": 0.30, "sev_mod": 0.50, "sev_sev": 0.20})

    for i in range(n):
        age = rng.randint(1, 65)
        sex = rng.choice(["M", "F"])
        hba1c = round(rng.uniform(5.8, 11.5), 1)
        drug_error = rng.random() < r["drug_error"]
        dx_delay = rng.random() < r["dx_delay"]
        sev_roll = rng.random()
        if sev_roll < r["sev_mild"]:
            severity = "Mild"
        elif sev_roll < r["sev_mild"] + r["sev_mod"]:
            severity = "Moderate"
        else:
            severity = "Severe"
        sur_adh = rng.random() < r["surveillance"]
        patients.append({
            "patient_id": f"{g}-{seed}-{i+1:03d}",
            "gene": g,
            "age": age,
            "sex": sex,
            "hba1c": hba1c,
            "drug_error": drug_error,
            "dx_delayed": dx_delay,
            "severity": severity,
            "surveillance_adherent": sur_adh,
        })
    return patients


def _cohort_stats(patients: list) -> dict:
    n = len(patients)
    if n == 0:
        return {}
    return {
        "n": n,
        "drug_error_pct": round(100 * sum(p["drug_error"] for p in patients) / n, 1),
        "dx_delayed_pct": round(100 * sum(p["dx_delayed"] for p in patients) / n, 1),
        "surveillance_adherent_pct": round(100 * sum(p["surveillance_adherent"] for p in patients) / n, 1),
        "severity_mild_pct": round(100 * sum(p["severity"] == "Mild" for p in patients) / n, 1),
        "severity_moderate_pct": round(100 * sum(p["severity"] == "Moderate" for p in patients) / n, 1),
        "severity_severe_pct": round(100 * sum(p["severity"] == "Severe" for p in patients) / n, 1),
        "mean_hba1c": round(sum(p["hba1c"] for p in patients) / n, 1),
    }


def _all_patients() -> list:
    all_pts = []
    for i, ge in enumerate(MODY_GENES):
        seed = SEED_BASE + i
        pts = _simulate_gene(ge, seed, 40)
        all_pts.extend(pts)
    return all_pts


# ─── Public API functions ──────────────────────────────────────────────────────

def get_overview() -> dict:
    all_pts = _all_patients()
    agg = _cohort_stats(all_pts)
    return {
        "atlas_name": "MODY Atlas",
        "atlas_subtitle": "Complete 8-Gene Maturity-Onset Diabetes of Youth (MODY) & Neonatal DM Reference",
        "n_genes": 8,
        "n_patients": len(all_pts),
        "seeds": f"{SEED_BASE}–{SEED_BASE + 7}",
        "genes": [g["gene"] for g in MODY_GENES],
        "description": (
            "The MODY Atlas covers the 8 most clinically actionable hereditary diabetes genes: "
            "HNF4A (MODY1), GCK (MODY2), HNF1A (MODY3), PDX1 (MODY4), HNF1B (MODY5), "
            "NEUROD1 (MODY6), ABCC8 (MODY12/NDM), and KCNJ11 (MODY13/NDM). "
            "MODY affects ~1–2% of all patients diagnosed with diabetes and is the most under-diagnosed "
            "monogenic diabetes syndrome. Each subtype has distinct molecular pathology, clinical phenotype, "
            "and — critically — a specific treatment algorithm that differs radically from T1D/T2D management. "
            "GCK-MODY2 requires NO treatment (over-treatment causes hypoglycemia). "
            "HNF1A/HNF4A-MODY respond to sulfonylureas 5–8× better than insulin. "
            "ABCC8/KCNJ11-NDM: ~90% can discontinue insulin and switch to oral sulfonylurea. "
            "This atlas provides per-gene phenotypic profiles, drug alerts, and cohort simulation "
            "for 320 patients (8 × 40, seeds 1214–1221)."
        ),
        "aggregate_clinical": agg,
        "drug_alerts": [
            {
                "title": "GCK/MODY2: NO TREATMENT REQUIRED — Over-treatment causes hypoglycemia without benefit",
                "body": (
                    "GCK-MODY2 is a fixed set-point hyperglycemia (6.0–8.0 mmol/L). Insulin or sulfonylureas cause "
                    "hypoglycemia without reducing microvascular risk. STOP unnecessary treatment immediately. "
                    "Pregnancy exception: fetal genotyping determines whether insulin is needed."
                ),
                "type": "danger",
            },
            {
                "title": "HNF1A/MODY3 & HNF4A/MODY1: SWITCH TO SULFONYLUREA — 5–8× more effective than insulin",
                "body": (
                    "Patients on insulin with MODY3/MODY1 can switch to low-dose sulfonylurea (glibenclamide 1.25 mg or "
                    "gliclazide MR 15 mg). Response rate 85–90%. Start very low dose — MODY β-cells are exquisitely sensitive. "
                    "Rifampicin (CYP2C9 inducer) reduces SU efficacy — increase dose if co-prescribed."
                ),
                "type": "warning",
            },
            {
                "title": "ABCC8/KCNJ11 NDM: SULFONYLUREA SWITCH replaces insulin — start before 6 months",
                "body": (
                    "90% of ABCC8/KCNJ11 permanent neonatal DM patients can switch from insulin to oral glyburide. "
                    "KCNJ11-DEND: SU also improves epilepsy and developmental delay. "
                    "Dose is MUCH higher than MODY3 (0.2–0.8 mg/kg/day). NEVER use diazoxide in GOF NDM — worsens DM."
                ),
                "type": "danger",
            },
            {
                "title": "HNF1B/MODY5: SULFONYLUREAS OFTEN INEFFECTIVE — Screen renal cysts + 17q12 deletion",
                "body": (
                    "HNF1B-MODY5 patients usually require insulin (not SU). Renal cysts precede DM — "
                    "check kidney imaging in all new DM cases with cystic renal disease. "
                    "MLPA mandatory (40% are whole-gene deletions missed by sequencing). "
                    "NSAIDs contraindicated (renal impairment risk)."
                ),
                "type": "warning",
            },
            {
                "title": "GCK PREGNANCY: Fetal genotype determines maternal insulin requirement",
                "body": (
                    "Affected fetus (GCK LOF inherited): DO NOT give insulin — fetus and mother matched set-point → "
                    "normal fetal weight. Unaffected fetus: insulin needed → reduces macrosomia. "
                    "Genotype fetus if management uncertain (NIPT/amniocentesis)."
                ),
                "type": "danger",
            },
            {
                "title": "MODY3 GLUCOSURIA: Glycosuria at normal blood glucose is NOT Type 1 — LOW renal threshold",
                "body": (
                    "HNF1A-MODY3 reduces SGLT2 expression → glucose appears in urine at blood glucose <10 mmol/L. "
                    "This is PATHOGNOMONIC for MODY3 and should trigger HNF1A/MODY genetic testing, not insulin escalation."
                ),
                "type": "warning",
            },
            {
                "title": "HNF4A MODY1: Neonatal hyperinsulinism (NHI) in 50% — SAME gene causes HI then MODY",
                "body": (
                    "HNF4A LOF carriers may present as macrosomic infant with neonatal hypoglycemia (HI) in first months. "
                    "NHI resolves, then MODY develops in adolescence/adulthood. "
                    "Document neonatal history in all adult MODY1 diagnoses. "
                    "Treat NHI with diazoxide; stop diazoxide when NHI resolves."
                ),
                "type": "warning",
            },
            {
                "title": "ALL MODY: Autoantibodies NEGATIVE — Do NOT default to T1D without MODY genetic testing",
                "body": (
                    "All MODY subtypes are autoantibody-negative (GADA, IA-2, ZnT8, insulin autoantibodies). "
                    "Any lean patient with family history of diabetes, negative autoantibodies, and C-peptide positive "
                    "should have MODY gene panel. HbA1c alone cannot distinguish MODY from T1D/T2D."
                ),
                "type": "warning",
            },
        ],
        "clinical_pearls": [
            "GCK-MODY2: STOP treatment — HbA1c 5.8–7.5% is the correct set-point, not a treatment failure",
            "HNF1A-MODY3: hsCRP <0.5 mg/L distinguishes from T2D (>1 mg/L) — cheap screening test",
            "HNF1A-MODY3: 2-hour OGTT glucose increment <3.5 mmol/L = MODY2; >4.5 mmol/L = MODY3/T2D",
            "HNF1B-MODY5: cystic kidney disease on ultrasound in a young diabetic → test HNF1B + MLPA immediately",
            "KCNJ11-DEND: epilepsy in neonatal DM patient → SU switch treats BOTH diabetes AND epilepsy",
            "ABCC8/KCNJ11 NDM: C-peptide detectable in PNDM = β-cells present but suppressed → SU switch possible",
            "All MODY: autoantibody-negative DM in lean young adult with family history = MODY until genetic panel done",
            "HNF4A MODY1: macrosomic infant (>4 kg birth weight) with parent who has diabetes = test HNF4A panel",
        ],
    }


def get_breakdown() -> dict:
    result = {}
    for i, ge in enumerate(MODY_GENES):
        seed = SEED_BASE + i
        pts = _simulate_gene(ge, seed, 40)
        stats = _cohort_stats(pts)
        result[ge["gene"]] = {
            "gene": ge["gene"],
            "protein": ge["protein"],
            "alias": ge["alias"],
            "aa": ge["aa"],
            "kDa": ge["kDa"],
            "locus": ge["locus"],
            "omim_gene": ge["omim_gene"],
            "omim_disease": ge["omim_disease"],
            "inheritance": ge["inheritance"],
            "gene_class": ge["gene_class"],
            "phenotype": ge["phenotype"],
            "hallmark": ge["hallmark"],
            "treatment_alert": ge["treatment_alert"],
            "key_ddx": ge["key_ddx"],
            "gfr_pattern": ge["gfr_pattern"],
            "proteinuria_pattern": ge["proteinuria_pattern"],
            "primary_complication": ge["primary_complication"],
            "disease_detail": ge["disease_detail"],
            "variants": ge.get("variants", []),
            "drug_ci": ge.get("drug_ci", []),
            "stats": stats,
        }
    return result


def get_definitions() -> dict:
    return {
        "atlas": (
            "MODY Atlas — 8 hereditary diabetes genes: HNF4A(MODY1)·GCK(MODY2)·HNF1A(MODY3)·PDX1(MODY4)·"
            "HNF1B(MODY5)·NEUROD1(MODY6)·ABCC8(MODY12)·KCNJ11(MODY13). "
            "320 patients, 8×40, seeds 1214–1221."
        ),
        "terms": {
            "mody": (
                "Maturity-Onset Diabetes of the Young — monogenic diabetes caused by single-gene mutations "
                "affecting β-cell function; autosomal dominant inheritance; presents <25–40 years; "
                "autoantibody-negative; C-peptide positive; affects ~1–2% of all diabetes patients"
            ),
            "katp_channel": (
                "ATP-sensitive potassium channel — octameric complex: 4 × Kir6.2 (KCNJ11, pore) + 4 × SUR1 (ABCC8, regulatory); "
                "closes when ATP/ADP ratio rises (after glucose stimulus) → β-cell depolarises → Ca2+ influx → insulin release; "
                "target of sulfonylureas (close KATP) and diazoxide (open KATP)"
            ),
            "permanent_neonatal_diabetes_pndm": (
                "Neonatal DM presenting before 6 months of age that persists beyond 12 months; "
                "most commonly KCNJ11 (40–50%), ABCC8 (10%), and other rare genes; "
                "autoantibody-negative; often low birth weight (intrauterine insulin deficiency); "
                "most cases are treatable with oral sulfonylurea rather than insulin"
            ),
            "dend_syndrome": (
                "Developmental delay + Epilepsy + Neonatal Diabetes — caused by severe GOF mutations in KCNJ11; "
                "KATP channels in brain (hippocampus, substantia nigra) kept constitutively open → neuronal hyperpolarisation; "
                "sulfonylurea switch treats BOTH diabetes AND neurological features; "
                "iDEND: intermediate DEND — DM + developmental delay + hypotonia but no epilepsy"
            ),
            "sulfonylurea_switch": (
                "Replacement of insulin therapy with oral sulfonylurea (glyburide/glibenclamide or gliclazide) "
                "in MODY patients or KATP-NDM; "
                "mechanism: SU binds SUR1 (ABCC8) → closes KATP → overcomes GOF/restores secretion; "
                "success rate ~90% in KCNJ11/ABCC8 NDM; 85–90% in MODY3/MODY1"
            ),
            "glucokinase_set_point": (
                "GCK (glucokinase) is the β-cell glucose sensor with S0.5 ~8 mmol/L; "
                "GCK LOF → S0.5 shifts rightward → insulin secretion threshold raised → "
                "stable mild fasting hyperglycemia (6.0–8.0 mmol/L); "
                "set-point is fixed (not progressive); at this level, microvascular risk is minimal"
            ),
            "renal_glucose_threshold": (
                "Blood glucose level at which glucosuria begins; normally ~10 mmol/L (SLC5A5/SGLT2-mediated); "
                "in HNF1A-MODY3: SGLT2 downregulated → threshold drops to ~5.5 mmol/L; "
                "PATHOGNOMONIC: glycosuria at blood glucose <8 mmol/L strongly suggests MODY3"
            ),
            "glucosuria_mody3": (
                "PATHOGNOMONIC feature of HNF1A-MODY3: glucose in urine at blood glucose levels that would NOT "
                "normally cause glycosuria (i.e., blood glucose <10 mmol/L); "
                "caused by HNF1A LOF reducing SGLT2 expression in renal tubule; "
                "clinicians should test urine glucose in all young patients with mild-to-moderate hyperglycemia"
            ),
            "congenital_hyperinsulinism_chi": (
                "Persistent hypoglycemia in neonate/infant due to inappropriate insulin secretion; "
                "LOF mutations in ABCC8 or KCNJ11 (KATP channel) most common cause; "
                "focal CHI: somatic LOF + constitutional LOF → localised islet lesion → PET-CT F-DOPA + surgery curative; "
                "diffuse CHI: biallelic LOF → near-total pancreatectomy risk"
            ),
            "17q12_microdeletion": (
                "Chromosomal deletion encompassing HNF1B gene + flanking genes; "
                "30–40% of HNF1B disease alleles (missed by sequencing → MLPA mandatory); "
                "additional psychiatric features: 34× increased schizophrenia risk; "
                "autism spectrum disorder; intellectual disability"
            ),
            "pancreatic_exocrine_insufficiency_pei": (
                "Reduced exocrine pancreatic function → malabsorption of fat + fat-soluble vitamins (A,D,E,K); "
                "steatorrhoea; weight loss; occurs in 40% of heterozygous PDX1 and 50–80% of HNF1B; "
                "treated with pancreatic enzyme replacement therapy (PERT) with meals"
            ),
            "pert_pancreatic_enzymes": (
                "Pancreatic enzyme replacement therapy: lipase + amylase + protease capsules; "
                "taken with ALL meals and snacks; dose 25000–50000 lipase units per meal; "
                "mandatory in PDX1-MODY4 with exocrine insufficiency and HNF1B-MODY5 with pancreatic atrophy"
            ),
            "mlpa_hnf1b": (
                "Multiplex ligation-dependent probe amplification — required for HNF1B testing; "
                "30–40% of HNF1B disease alleles are whole-gene deletions (17q12 microdeletion) "
                "invisible to sequencing alone; MLPA detects copy-number variation; "
                "standard sequencing-only negative HNF1B test is INSUFFICIENT if clinical suspicion high"
            ),
            "mody_pregnancy_gck": (
                "Critical GCK-MODY2 pregnancy management: insulin requirement depends on fetal genotype; "
                "Fetus has GCK LOF: mother and fetus share elevated set-point → normal birth weight → NO insulin; "
                "Fetus no GCK LOF: lower fetal set-point → fetal hyperinsulinism → macrosomia → INSULIN needed; "
                "Fetal genotype: NIPT or amniocentesis; manage by fetal growth trajectory on ultrasound"
            ),
            "diazoxide": (
                "KATP channel opener (binds SUR1 MgADP-binding site); "
                "used in congenital hyperinsulinism (LOF ABCC8/KCNJ11) to open KATP and reduce insulin; "
                "CONTRAINDICATED in GOF neonatal DM (ABCC8/KCNJ11 GOF) — would worsen DM by further KATP opening; "
                "also used in HNF4A neonatal hyperinsulinism (MODY1) in neonatal phase"
        ),
            "hscrp_mody3_discriminator": (
                "High-sensitivity CRP (hsCRP); MODY3 patients have extremely low hsCRP (<0.5 mg/L) "
                "compared to T2D (typically >1–3 mg/L); "
                "useful cheap discriminating test between MODY3 and T2D in a lean young adult; "
                "reflects HNF1A role in hepatic acute-phase protein regulation (CRP, fibrinogen)"
            ),
        },
    }


if __name__ == "__main__":
    import json
    ov = get_overview()
    print(f"Patients: {ov['n_patients']}, Genes: {ov['n_genes']}, Seeds: {ov['seeds']}")
    bd = get_breakdown()
    print(f"Breakdown genes: {list(bd.keys())}")
    df = get_definitions()
    print(f"Definitions terms: {list(df['terms'].keys())}")
    print("OK")
