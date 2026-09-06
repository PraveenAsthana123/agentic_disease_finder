#!/usr/bin/env python3
"""Cardiomyopathy Atlas — Complete 8-Gene Hereditary Cardiomyopathy Atlas
MYH7   (Beta-myosin heavy chain; ~1935 aa; 14q12; HCM1 + DCM1S; AD; most common HCM gene 35-40%; mavacamten/aficamten; ICD for SCD) ·
MYBPC3 (Cardiac myosin binding protein C3; ~1274 aa; 11p11.2; HCM4; AD haploinsufficiency; most common HCM worldwide 40-50%; Mavacamten FDA 2022) ·
TNNT2  (Cardiac troponin T2; ~298 aa; 1q32.1; HCM3 + DCM1D; AD; disproportionate SCD risk to wall thickness — MALIGNANT) ·
PKP2   (Plakophilin-2; ~837 aa; 12p11.21; ARVC9; AD; most common ARVC gene 40-50%; epsilon waves PATHOGNOMONIC; sports restriction MANDATORY) ·
DSP    (Desmoplakin; ~2871 aa; 6p24.3; ARVC8 + DCMEP; AD; biventricular ARVC; woolly hair + PPK = Carvajal PATHOGNOMONIC; LGE subepicardial) ·
LMNA   (Lamin A/C; ~664 aa; 1q22; DCM1A + EDMD2; AD; AV block PATHOGNOMONIC; ICD MANDATORY; SCD risk 15-40% without ICD) ·
TTN    (Titin; ~34350 aa; 2q31.2; DCM1G; AD; largest human protein; A-band TTNtv PATHOGENIC; 25% familial DCM; peripartum CM) ·
RBM20  (RNA-binding motif protein 20; ~1228 aa; 10q25.2; DCM1HH; AD; most aggressive DCM; RS-domain hotspot; ICD MANDATORY)
320-patient aggregate cohort (8 × 40, seeds 1102–1109)
"""

import random

SEED_BASE = 1102

CARDIOMYOPATHY_GENES = [
    # ── MYH7 — HCM1 / DCM1S ─────────────────────────────────────────────────
    {
        "gene": "MYH7",
        "protein": "Beta-Myosin Heavy Chain (MYH7)",
        "alias": "MYH7; OMIM gene 160760; 14q12; ~1935 aa; HCM1 (OMIM #192600) + DCM1S (OMIM #613426); AD; 35-40% of HCM families; thick-filament sarcomeric protein; apical HCM association",
        "aa": "~1935 aa",
        "kDa": "~223 kDa",
        "mechanism": (
            "MYH7 encodes the beta isoform of myosin heavy chain, the predominant motor protein of "
            "the cardiac sarcomere thick filament. Beta-myosin drives actin filament sliding and force "
            "generation during systole. "
            "NORMAL FUNCTION: MYH7 forms the head (S1 subfragment) that hydrolyses ATP and engages "
            "actin cross-bridges; it constitutes ~80% of ventricular myosin in adults. "
            "HCM PATHOMECHANISM: missense variants (predominant) → hypercontractile cross-bridge cycling → "
            "increased Ca²⁺ sensitivity of the sarcomere → increased ATP consumption → "
            "myofibrillar disarray → fibrosis → hypertrophic remodelling of the left ventricle. "
            "GAIN-OF-FUNCTION (HCM): actin-activated ATPase rate increased; super-relaxed state (SRX) "
            "destabilised → more cross-bridges recruited → hypercontractility. "
            "DCM1S PATHOMECHANISM: LOF/truncating variants → reduced contractile force → dilated remodelling. "
            "APICAL HCM: MYH7 enriched in apical HCM subtype (ace-of-spades morphology on MRI/echo). "
            "MAVACAMTEN/AFICAMTEN (cardiac myosin inhibitors): allosterically stabilise SRX state → "
            "reduce number of active cross-bridges → reduce LVOTO gradient and symptoms."
        ),
        "disease_type": (
            "Hypertrophic Cardiomyopathy HCM1 (OMIM #192600) + DCM1S (OMIM #613426); AD; "
            "35-40% of HCM; thick filament; missense variants dominant; apical HCM; "
            "mavacamten/aficamten first-in-class; ICD for SCD risk (Maron criteria)"
        ),
        "locus": "14q12",
        "omim_gene": 160760,
        "omim_disease": 192600,
        "inheritance": (
            "AUTOSOMAL DOMINANT: heterozygous pathogenic variants. "
            "Most HCM1 variants are missense (gain-of-function) — truncating variants rare for HCM but "
            "seen in DCM1S. "
            "PENETRANCE: high (~95%) but age-dependent; most clinically manifest by 4th decade. "
            "DE NOVO VARIANTS: ~5-10% of cases. "
            "FAMILY SCREENING: echocardiogram + ECG + genetic testing cascade for all first-degree "
            "relatives; screening every 1-5 years for at-risk relatives aged 10-60. "
            "GENOTYPE-PHENOTYPE: specific variants (e.g., p.Arg403Gln, p.Arg453Cys, p.Arg719Gln) "
            "carry particularly high SCD risk — 'malignant' variants — ICD threshold lowered. "
            "MAVS REGISTRY: international MYH7 variant registry for variant classification."
        ),
        "phenotype": (
            "HYPERTROPHIC CARDIOMYOPATHY (HCM1): "
            "LVH: asymmetric septal hypertrophy (maximal wall thickness ≥15 mm in adults, "
            "or ≥13 mm with family history); LV cavity usually NOT dilated (differentiates from DCM). "
            "LVOTO: left ventricular outflow tract obstruction — resting gradient >30 mmHg in ~1/3; "
            "provocable gradient >50 mmHg on Valsalva in ~70%; SAM (systolic anterior motion of "
            "mitral valve) causes dynamic obstruction and mitral regurgitation. "
            "SYMPTOMS: exertional dyspnoea, chest pain, pre-syncope/syncope (exertion or post-exercise — "
            "PATHOGNOMONIC timing; vasovagal syncope usually non-exertional). "
            "SCD RISK: primary risk in young patients — usually VF; risk stratification mandatory "
            "(Maron 5 risk factors or ESC HCM Risk-SCD calculator). "
            "APICAL HCM: MYH7-associated; ACE-OF-SPADES LV cavity on end-diastole echo/MRI; "
            "deep T-wave inversions in lateral leads (V4-V6) on ECG. "
            "CARDIAC MRI: LGE in interventricular septum and junction of RV insertion — extent of LGE "
            "correlates with SCD risk."
        ),
        "treatment_options": [
            "Beta-blockers first-line (bisoprolol 2.5-10 mg daily or metoprolol succinate): "
            "reduce LVOTO gradient; blunt exertional tachycardia; symptom relief; "
            "titrate to heart rate 50-60 bpm at rest; do NOT abruptly stop — "
            "rebound tachycardia worsens gradient",
            "Mavacamten (CAMZYOS, FDA 2022) or Aficamten: cardiac myosin inhibitor — FIRST new HCM "
            "class therapy; reduce LVOTO gradient >50%; reduce symptoms (NHYA class); "
            "LVEF monitoring mandatory (HOLD if LVEF <55%); AVOID in pregnancy; "
            "CYP2C19 drug interactions (adjust dose per genotype); eligibility: symptomatic HOCM with "
            "LVOTO gradient >30 mmHg resting or >50 mmHg provoked + LVEF ≥55%",
            "Disopyramide (200-400 mg BD): negative inotrope reduces LVOTO gradient; "
            "combine with beta-blocker (prevents atrial conduction acceleration); "
            "anticholinergic side effects (urinary retention, dry mouth, constipation — "
            "caution in men with BPH); QTc monitoring mandatory",
            "Verapamil or diltiazem (rate-limiting calcium channel blocker): ONLY if no severe LVOTO "
            "(systolic dysfunction risk with significant LVOTO + verapamil — avoid if gradient >50 mmHg "
            "or severe obstruction); use for symptom relief when beta-blockers not tolerated",
            "ICD (implantable cardioverter-defibrillator): indicated if ≥1 major SCD risk factor "
            "(prior cardiac arrest/VF/sustained VT; family SCD <50y in close relative; "
            "unexplained syncope; severe LVH ≥30 mm; LVEF <50%; abnormal BP response to exercise; "
            "NSVT on Holter; extensive LGE on CMR); AHA/Maron criteria OR ESC HCM Risk-SCD ≥6%/5y",
            "Septal reduction therapy (SRT): surgical myectomy (Morrow procedure) GOLD STANDARD for "
            "HOCM with gradient >50 mmHg refractory to medical therapy; "
            "alcohol septal ablation (ASA) alternative when myectomy not feasible; "
            "eligibility: symptomatic NYHA ≥3 or recurrent syncope despite maximal medical therapy; "
            "resting LVOTO gradient >50 mmHg (or provoked >70 mmHg)",
            "Atrial fibrillation management: rate control + anticoagulation (CHA₂DS₂-VASc independent — "
            "ALL HCM patients with AF should be anticoagulated given high stroke risk); "
            "rhythm control preferred (maintain sinus rhythm improves haemodynamics in HCM)",
        ],
        "critical_avoid": (
            "MYH7/HCM: AVOID nifedipine/amlodipine (dihydropyridine CCBs — peripheral vasodilation "
            "worsens LVOTO by reducing afterload — CONTRAINDICATED in severe LVOTO). "
            "AVOID vigorous exercise/dehydration (exacerbate LVOTO). "
            "MAVACAMTEN: HOLD if LVEF <55% — systolic depression risk; avoid CYP2C19 inhibitors "
            "without dose adjustment; ABSOLUTELY avoid in pregnancy. "
            "DISOPYRAMIDE alone without beta-blocker: accelerates AV conduction → ventricular response "
            "rate increased in AF — always combine. "
            "Do NOT use as 'athlete's heart' without genetic/CMR workup — MYH7 HCM can be fatal on field."
        ),
        "key_ddx": [
            "MYBPC3-HCM4: clinically identical; MYBPC3 more common (40-50%); haploinsufficiency; age-dependent penetrance",
            "Cardiac amyloidosis: infiltrative LVH; low voltage on ECG despite thick walls; TTR or AL type",
            "Fabry disease (GLA): X-linked; enzyme deficiency; renal/skin/neuropathy; treatable with ERT",
            "Pompe disease (GAA): glycogen storage; multisystem; enzyme deficiency — treatable",
            "Athlete's heart: physiological hypertrophy; regression with detraining; no LVOTO; no fibrosis on CMR",
            "HCM with LVOTO vs AS: aortic stenosis — fixed obstruction; different murmur dynamics",
        ],
        "severity_weights": {"Mild": 0.25, "Moderate": 0.45, "Severe": 0.30},
        "onset_age_range": (15, 45),
        "dx_lag_y": (2, 10),
        "drug_error_rate": 0.18,
        "icd_eligible_rate": 0.35,
        "cardiac_transplant_rate": 0.05,
        "arrhythmia_rate": 0.40,
        "scd_risk_high_rate": 0.30,
        "lvoto_rate": 0.65,
        "progression_rate": 0.72,
        "first_line_drug": "Beta-blocker (bisoprolol/metoprolol); Mavacamten (HOCM); ICD (SCD risk)",
    },
    # ── MYBPC3 — HCM4 ───────────────────────────────────────────────────────
    {
        "gene": "MYBPC3",
        "protein": "Cardiac Myosin Binding Protein C3 (MYBPC3)",
        "alias": "MYBPC3; OMIM gene 600958; 11p11.2; ~1274 aa; HCM4 (OMIM #115197); AD haploinsufficiency; most common HCM gene worldwide 40-50%; Mavacamten FDA 2022",
        "aa": "~1274 aa",
        "kDa": "~150 kDa",
        "mechanism": (
            "MYBPC3 encodes cardiac myosin binding protein C (cMyBP-C), a thick-filament-associated "
            "regulatory protein with 11 immunoglobulin/fibronectin-like domains (C0-C10). "
            "NORMAL FUNCTION: cMyBP-C binds myosin, titin, and actin — modulates cross-bridge cycling; "
            "phosphorylation of MYBPC3 by PKA (cAMP-dependent) activates contractility during "
            "beta-adrenergic stimulation (fight-or-flight); "
            "C10 domain anchors protein to thick filament backbone. "
            "HCM4 PATHOMECHANISM: haploinsufficiency — truncating variants (frameshift, nonsense, "
            "splice-site ~60% of MYBPC3 variants) produce unstable mRNA (NMD — nonsense-mediated decay) → "
            "50% cMyBP-C protein → impaired thick-filament regulation → hypercontractility → HCM. "
            "FOUNDER VARIANTS: p.Arg502Trp — South Asian (India/Pakistan) founder; "
            "c.927-2A>G — Finnish founder; both account for large proportions of HCM in respective populations. "
            "AGE-DEPENDENT PENETRANCE: unlike MYH7, MYBPC3 penetrance increases markedly with age — "
            "~50% by age 40, ~95% by age 60; older-onset HCM more common with MYBPC3 than MYH7."
        ),
        "disease_type": (
            "Hypertrophic Cardiomyopathy HCM4 (OMIM #115197); AD haploinsufficiency; "
            "most common HCM gene worldwide 40-50%; truncating variants 60%; "
            "age-dependent penetrance (50% by 40y, 95% by 60y); founder variants in South Asian and Finnish populations"
        ),
        "locus": "11p11.2",
        "omim_gene": 600958,
        "omim_disease": 115197,
        "inheritance": (
            "AUTOSOMAL DOMINANT: haploinsufficiency — one pathogenic allele sufficient. "
            "TRUNCATING VARIANTS dominant (~60%): frameshift, nonsense, splice-site → NMD → 50% protein. "
            "MISSENSE also pathogenic but less common than MYH7. "
            "PENETRANCE: age-dependent — 50% by age 40, 95% by age 60; "
            "younger carriers may have normal echo (LVH develops later). "
            "FOUNDER VARIANTS: p.Arg502Trp (South Asian) — screen in all South Asian HCM patients; "
            "c.927-2A>G (Finnish). "
            "FAMILY SCREENING: cascade testing + annual echo/ECG up to age 60 in carriers "
            "(LVH can develop in 5th-6th decade — cannot stop screening at 30). "
            "SCD RISK STRATIFICATION: ESC HCM Risk-SCD score or AHA Maron criteria at each visit."
        ),
        "phenotype": (
            "HYPERTROPHIC CARDIOMYOPATHY HCM4 (MYBPC3): "
            "CLINICAL: similar to MYH7-HCM — asymmetric septal LVH, LVOTO (SAM-MR), exertional dyspnoea, "
            "syncope, AF, SCD. "
            "DISTINCTIVE FEATURES COMPARED TO MYH7: "
            "LATER ONSET — mean age of LVH expression ~40-50y vs ~20-30y for MYH7; "
            "MILDER MAXIMUM WALL THICKNESS on average but still at SCD risk; "
            "HIGHER PREVALENCE of late-gadolinium enhancement (LGE) on CMR with age. "
            "LVOTO: similar prevalence to MYH7 (~60-70% of symptomatic HCM). "
            "SCD RISK: MYBPC3 truncating variants carry significant SCD risk — "
            "particularly p.Arg502Trp in South Asian populations (higher SCD event rates reported). "
            "AF BURDEN: high AF prevalence in MYBPC3-HCM (30-40% lifetime risk); "
            "ALL HCM+AF patients anticoagulate regardless of CHA₂DS₂-VASc. "
            "OUTCOME: generally more benign than MYH7 'malignant' missense variants, but not low-risk."
        ),
        "treatment_options": [
            "Mavacamten (CAMZYOS, FDA 2022): FIRST AND ONLY FDA-approved drug specifically for "
            "symptomatic obstructive HCM (HOCM); cardiac myosin inhibitor; "
            "AFFIRM-AHF trial: 37% of patients achieved ≥1 NYHA class improvement; "
            "reduces resting LVOTO gradient, exercise gradient, NT-proBNP; "
            "LVEF monitoring every 4 weeks (echocardiography); HOLD if LVEF <55%; "
            "CYP2C19 metaboliser status guides dosing (5 mg, 10 mg, or 15 mg daily); "
            "CONTRAINDICATED in pregnancy (risk to fetus); "
            "start at lowest dose and titrate based on LVOTO gradient response",
            "Beta-blockers (bisoprolol 2.5-10 mg; metoprolol succinate 25-200 mg): "
            "first-line symptom control; reduce gradient by blunting tachycardia; "
            "target resting HR 55-65 bpm; do not abruptly stop",
            "Disopyramide (100-400 mg BD) + beta-blocker: for refractory LVOTO; "
            "negative inotrope + rate-control combination; QTc monitoring",
            "ICD therapy: indicated when ESC HCM Risk-SCD ≥6%/5y OR ≥1 AHA major risk factor; "
            "primary prevention ICD well-established in high-risk MYBPC3; "
            "subcutaneous ICD (S-ICD) option if no pacing indication",
            "Septal reduction therapy: myectomy (surgical, gold standard) or "
            "alcohol septal ablation (ASA) for refractory HOCM; "
            "referral to HCM Centre of Excellence mandatory for SRT decisions",
            "Annual screening for gene-positive/phenotype-negative relatives: "
            "ECG + echocardiogram every 1-5 years until age 60 "
            "(late-onset LVH possible in 5th-6th decade with MYBPC3)",
            "AF management: cardioversion for first AF; long-term rhythm control preferred "
            "(amiodarone or flecainide + anticoagulation); "
            "ALL HCM+AF anticoagulate regardless of CHA₂DS₂-VASc (stroke risk elevated inherently)",
        ],
        "critical_avoid": (
            "MYBPC3/HCM: AVOID dihydropyridine CCBs (amlodipine, nifedipine) in obstructive HCM — "
            "peripheral vasodilation worsens LVOTO. "
            "MAVACAMTEN: LVEF monitoring mandatory — hold at LVEF <55%; "
            "avoid moderate/strong CYP2C19 inhibitors (fluconazole, omeprazole in high dose) without "
            "dose adjustment; CONTRAINDICATED in pregnancy. "
            "Do NOT stop beta-blocker abruptly. "
            "Age-dependent penetrance: do NOT clear younger MYBPC3 carriers as unaffected — "
            "LVH may not appear until age 50-60; continue annual screening to age 60. "
            "South Asian p.Arg502Trp: HIGH population prevalence — ensure family cascade screening."
        ),
        "key_ddx": [
            "MYH7-HCM1: clinically identical; earlier onset; more malignant missense variants; MYH7 more apical HCM",
            "TNNT2-HCM3: mild LVH but HIGH SCD risk — disproportionate; thin-filament",
            "Cardiac amyloidosis (ATTR): thick walls + low voltage ECG; technetium scintigraphy distinguishes",
            "Hypertensive heart disease: concentric not asymmetric LVH; hypertension history; negative genetics",
            "Anderson-Fabry disease: X-linked enzyme deficiency; posterior wall LVH; renal/skin involvement",
        ],
        "severity_weights": {"Mild": 0.30, "Moderate": 0.45, "Severe": 0.25},
        "onset_age_range": (30, 60),
        "dx_lag_y": (3, 12),
        "drug_error_rate": 0.15,
        "icd_eligible_rate": 0.28,
        "cardiac_transplant_rate": 0.04,
        "arrhythmia_rate": 0.38,
        "scd_risk_high_rate": 0.25,
        "lvoto_rate": 0.62,
        "progression_rate": 0.68,
        "first_line_drug": "Beta-blocker; Mavacamten FDA 2022 (HOCM); ICD (SCD risk); annual echo to age 60",
    },
    # ── TNNT2 — HCM3 / DCM1D ────────────────────────────────────────────────
    {
        "gene": "TNNT2",
        "protein": "Cardiac Troponin T2 (TNNT2)",
        "alias": "TNNT2; OMIM gene 191045; 1q32.1; ~298 aa; HCM3 (OMIM #115195) + DCM1D (OMIM #601494); AD; thin-filament; MALIGNANT SCD risk disproportionate to LVH",
        "aa": "~298 aa",
        "kDa": "~36 kDa",
        "mechanism": (
            "TNNT2 encodes cardiac troponin T, the tropomyosin-binding subunit of the thin-filament "
            "troponin regulatory complex (TnT-TnI-TnC trimer). "
            "NORMAL FUNCTION: TnT anchors the troponin complex to tropomyosin on the actin filament; "
            "coordinates Ca²⁺-dependent regulation of actomyosin interaction; "
            "the T1 and T2 domains wrap around tropomyosin, repositioning it to enable cross-bridge formation. "
            "HCM3 PATHOMECHANISM: missense variants → altered TnT-tropomyosin interaction → "
            "increased Ca²⁺ sensitivity of thin filament → hypercontractility at physiological Ca²⁺ → "
            "myocyte disarray → fibrosis → HCM. "
            "DISTINCTIVE PHENOTYPE: HCM3 causes marked sarcomeric dysfunction WITHOUT proportional "
            "hypertrophic remodelling — wall thickness often MILD (12-15 mm only) but "
            "fibrosis (LGE on CMR) is EXTENSIVE and SCD risk is HIGH. "
            "DCM1D PATHOMECHANISM: LOF variants → reduced thin-filament activation → "
            "impaired systolic function → dilated cardiomyopathy. "
            "TROPONIN T BIOMARKER: acute myocardial injury elevates cardiac TnT in serum — "
            "note this is the chronic genetic protein, not the acute biomarker."
        ),
        "disease_type": (
            "HCM3 (OMIM #115195) + DCM1D (OMIM #601494); AD; thin-filament troponin complex; "
            "MALIGNANT: SCD risk DISPROPORTIONATE to wall thickness — mild LVH does NOT mean low SCD risk; "
            "ICD threshold LOWER than other HCM genes; Holter + ECG mandatory"
        ),
        "locus": "1q32.1",
        "omim_gene": 191045,
        "omim_disease": 115195,
        "inheritance": (
            "AUTOSOMAL DOMINANT: heterozygous pathogenic variants. "
            "PENETRANCE: high for HCM3 missense variants; variable expression. "
            "MALIGNANT HCM3: several TNNT2 variants (p.Arg92Gln, p.Arg92Leu, p.Ile79Asn) classified as "
            "'malignant' — high SCD rates in affected families; "
            "family history of SCD <50y in first-degree relative with TNNT2 = very high risk. "
            "DE NOVO VARIANTS: occur; particularly in sporadic SCD survivors with mild or absent HCM on echo. "
            "FAMILY SCREENING: ECG + Holter + echocardiography + CMR in all first-degree relatives; "
            "cascade genetic testing. "
            "DCM1D: TNNT2 LOF variants cause dilated not hypertrophic CM — "
            "genotype-phenotype correlation important for management."
        ),
        "phenotype": (
            "HCM3 (TNNT2) — MALIGNANT HCM: "
            "WALL THICKNESS: often MILD (12-16 mm) — can be within 'borderline' range; "
            "DO NOT be reassured by mild LVH — TNNT2-HCM SCD risk does NOT scale with wall thickness. "
            "FIBROSIS: extensive myocardial fibrosis (LGE on CMR) disproportionate to hypertrophy — "
            "CMR LGE MANDATORY in all TNNT2 HCM patients. "
            "ECG: mandatory — may show ST-T abnormalities, T-wave inversions, or LVH voltage criteria "
            "despite only borderline echo LVH; "
            "some TNNT2 HCM patients present first with ABORTED SCD. "
            "SYMPTOMS: exertional dyspnoea, palpitations; syncope (particularly exertional) = red flag. "
            "HOLTER: mandatory — NSVT a major risk factor; "
            "even short runs of NSVT in TNNT2 HCM = consider ICD. "
            "DCM1D PHENOTYPE: dilated LV + reduced EF (TNNT2 LOF); "
            "treated as genetic DCM with standard HFrEF therapy."
        ),
        "treatment_options": [
            "ICD: LOWER THRESHOLD than other HCM genes — given disproportionate SCD risk; "
            "if any ONE additional SCD risk factor present (NSVT, syncope, family SCD, "
            "extensive LGE, abnormal BP response) + TNNT2 pathogenic variant → strong ICD indication; "
            "primary prevention ICD — AHA class IIa for TNNT2 with any significant risk modifier; "
            "DO NOT be reassured by mild LVH when deciding ICD",
            "Holter monitoring: 24-48h Holter MANDATORY at diagnosis and annually — "
            "NSVT detection critical; even 3-beat NSVT in TNNT2 patient = high concern; "
            "consider extended event monitoring (30-day) if Holter negative but syncope present",
            "Cardiac MRI with LGE: MANDATORY in all TNNT2 HCM — "
            "quantify extent of LGE (>15% LV mass = high SCD risk even with mild LVH); "
            "CMR provides definitive diagnosis when echo LVH borderline",
            "Beta-blockers: symptom control + reduce exertional tachycardia; "
            "blunt NSVT burden (partial); bisoprolol 2.5-10 mg or metoprolol 25-200 mg",
            "Mavacamten: for TNNT2 HCM with significant LVOTO (gradient >30 mmHg) and symptoms; "
            "same protocol as MYH7/MYBPC3 HOCM — LVEF monitoring mandatory",
            "DCM1D management: standard HFrEF therapy — ACEI/ARB/sacubitril-valsartan + "
            "beta-blocker + mineralocorticoid antagonist; ICD/CRT-D if LVEF <35%",
            "Exercise restriction: competitive sports CONTRAINDICATED in TNNT2 HCM; "
            "low-moderate aerobic exercise only (walking, gentle cycling); "
            "avoid burst exertion (SCD trigger)",
        ],
        "critical_avoid": (
            "TNNT2 HCM: CRITICAL PITFALL — DO NOT reassure patient or family based on MILD LVH. "
            "TNNT2 SCD risk is DISPROPORTIONATE to wall thickness — a patient with 13-15 mm LVH "
            "and TNNT2 pathogenic variant has HIGH SCD risk. "
            "CMR LGE MANDATORY — echo alone INSUFFICIENT for risk stratification. "
            "Holter MANDATORY — NSVT must not be missed. "
            "DO NOT delay ICD referral awaiting 'more LVH to develop.' "
            "Competitive sports ABSOLUTELY CONTRAINDICATED. "
            "Normal echo does NOT exclude TNNT2-HCM SCD risk — genetic result drives management."
        ),
        "key_ddx": [
            "MYH7-HCM: thicker LVH; apical HCM variant; thick filament — different gene panel",
            "MYBPC3-HCM4: similar management but lower SCD risk per unit of LVH than TNNT2",
            "Athlete's heart: physiological LVH; regresses with detraining; no fibrosis on CMR; no NSVT",
            "TNNT2-DCM vs MYH7-DCM: both cause genetic DCM — distinguish by LV morphology and family history",
            "Channelopathy (CPVT, LQTS): arrhythmia without structural heart disease — CMR/echo normal; TNNT2 has subtle LVH",
        ],
        "severity_weights": {"Mild": 0.20, "Moderate": 0.40, "Severe": 0.40},
        "onset_age_range": (15, 45),
        "dx_lag_y": (1, 8),
        "drug_error_rate": 0.25,
        "icd_eligible_rate": 0.50,
        "cardiac_transplant_rate": 0.08,
        "arrhythmia_rate": 0.55,
        "scd_risk_high_rate": 0.48,
        "lvoto_rate": 0.40,
        "progression_rate": 0.78,
        "first_line_drug": "ICD (lower threshold); Holter MANDATORY; CMR LGE MANDATORY; beta-blocker",
    },
    # ── PKP2 — ARVC9 ────────────────────────────────────────────────────────
    {
        "gene": "PKP2",
        "protein": "Plakophilin-2 (PKP2)",
        "alias": "PKP2; OMIM gene 602861; 12p11.21; ~837 aa; ARVC9 (OMIM #609040); AD; most common ARVC gene 40-50%; desmosomal protein; epsilon waves PATHOGNOMONIC; sports restriction MANDATORY",
        "aa": "~837 aa",
        "kDa": "~97 kDa",
        "mechanism": (
            "PKP2 encodes plakophilin-2, an armadillo-repeat desmosomal plaque protein that "
            "connects the desmosomal cadherin complex (desmoglein/desmocollin) to intermediate "
            "filaments (desmin) in cardiac myocytes. "
            "NORMAL FUNCTION: PKP2 stabilises the desmosomal plaque at intercalated discs — "
            "the mechanical junctions between cardiomyocytes that transmit contractile force; "
            "also regulates Na channel (SCN5A) trafficking and Wnt/beta-catenin signalling in "
            "cardiomyocytes (nuclear PKP2 role). "
            "ARVC9 PATHOMECHANISM: PKP2 LOF → desmosomal instability at intercalated discs → "
            "cardiomyocyte detachment under mechanical stress → myocyte death → "
            "FIBRO-FATTY REPLACEMENT of RV myocardium (fibrous + adipose tissue infiltration) → "
            "RV dilation + dysfunction + arrhythmia substrate. "
            "PHYSICAL ACTIVITY DRIVES PROGRESSION: mechanical stress on weakened desmosomes → "
            "accelerated myocyte loss → faster fibrofatty replacement → earlier RV failure and VT. "
            "ARRHYTHMIA MECHANISM: fibrofatty scar in RV → anisotropic conduction → "
            "re-entry circuits → monomorphic VT with LBBB morphology (RV origin)."
        ),
        "disease_type": (
            "ARVC9 — Arrhythmogenic Right Ventricular Cardiomyopathy (OMIM #609040); AD; "
            "most common ARVC gene 40-50%; desmosomal LOF; epsilon waves PATHOGNOMONIC (V1-V3); "
            "LBBB-VT (RV origin); sports restriction MANDATORY even in phenotype-negative gene carriers; "
            "ICD for documented VT/VF; sotalol/amiodarone for VT suppression"
        ),
        "locus": "12p11.21",
        "omim_gene": 602861,
        "omim_disease": 609040,
        "inheritance": (
            "AUTOSOMAL DOMINANT: heterozygous pathogenic variants — predominantly truncating (~80%); "
            "some missense. "
            "PENETRANCE: incomplete and age-dependent; ~50% by age 40; "
            "higher penetrance in males and in those with high physical activity levels. "
            "BIGENIC: ~15% of ARVC patients carry variants in >1 desmosomal gene — "
            "digenic/compound carriers have more severe/earlier disease. "
            "PHYSICAL ACTIVITY GENE-ENVIRONMENT INTERACTION: "
            "endurance sports dramatically increase penetrance and accelerate phenotype — "
            "competitive athletes with PKP2 develop ARVC much earlier than sedentary carriers. "
            "SPORTS RESTRICTION in ALL gene-positive individuals regardless of phenotype — "
            "a gene-positive, phenotype-negative carrier who continues competitive sports is at risk. "
            "FAMILY SCREENING: ECG + Holter + echocardiogram + cardiac MRI + genetic testing "
            "for all first-degree relatives."
        ),
        "phenotype": (
            "ARRHYTHMOGENIC RIGHT VENTRICULAR CARDIOMYOPATHY (ARVC9 — PKP2): "
            "ECG FINDINGS: epsilon waves (terminal notch/deflection after QRS complex in V1-V3) — "
            "PATHOGNOMONIC for ARVC; T-wave inversions V1-V4 (repolarisation abnormalities); "
            "LBBB morphology VT (consistent with RV origin); prolonged S-wave upstroke >55 ms in V1-V3. "
            "STRUCTURAL: RV dilation (RVEDD >42 mm indexed) + RV wall motion abnormalities "
            "(akinesis/dyskinesia of RV free wall, apex, or RVOT) — detected by echo and CMR; "
            "LV involvement in ~50% (biventricular ARVC — worse prognosis). "
            "CMR: fibrofatty infiltration of RV — fat signal on T1 and LGE in RV wall; "
            "LGE in RV insertion points of interventricular septum. "
            "ARRHYTHMIA: frequent PVCs (>500/24h ARVC risk marker); NSVT; "
            "sustained monomorphic VT (LBBB morphology) — from RV scar; "
            "VF and SCD (especially during or after exercise). "
            "DIAGNOSIS: 2010 Revised Task Force Criteria — major/minor criteria across "
            "structure, histology, ECG, arrhythmia, and genetics."
        ),
        "treatment_options": [
            "Sports restriction: MANDATORY in ALL PKP2 carriers regardless of phenotype — "
            "no competitive/endurance sports even if phenotype-negative; "
            "only low-intensity recreational exercise (walking, gentle swimming); "
            "this is the most important disease-modifying intervention to slow progression; "
            "ATHLETES with PKP2 variant must retire from competitive sport immediately",
            "ICD: indicated for documented sustained VT or VF; "
            "primary prevention ICD when: LVEF <35%, NSVT + syncope, "
            "severe RV dilation/dysfunction, or inducible sustained VT at EPS; "
            "subcutaneous ICD (S-ICD) if no pacing indication; "
            "appropriate ICD shocks may be frequent — anti-tachycardia pacing (ATP) programming important",
            "Antiarrhythmic therapy: sotalol (40-160 mg BD) or amiodarone for VT suppression; "
            "sotalol preferred in ARVC for VT (beta-blocking effect + Ks channel block); "
            "amiodarone if sotalol ineffective or contraindicated; "
            "mexiletine as adjunct; flecainide AVOID (proarrhythmic in structural heart disease)",
            "Catheter ablation: for refractory VT despite antiarrhythmic therapy or frequent ICD shocks; "
            "endocardial + epicardial approach often required (fibrofatty substrate in epicardium); "
            "experienced centre mandatory; ARVC ablation reduces VT burden but rarely curative",
            "ACE inhibitor/ARB + beta-blocker: cardioprotective in ARVC with RV or LV dysfunction; "
            "beta-blocker reduces adrenergic arrhythmia triggers; carvedilol preferred",
            "Cardiac MRI surveillance: annual CMR to track RV volume, wall motion, LGE progression; "
            "CMR also detects subclinical LV involvement",
            "Cardiac transplantation: for end-stage biventricular failure refractory to therapy; "
            "ARVC is a recognised indication for transplantation at experienced centres",
        ],
        "critical_avoid": (
            "PKP2 ARVC: SPORTS RESTRICTION MANDATORY — competitive/endurance exercise drives "
            "fibrofatty progression and precipitates SCD; this applies to gene-positive EVEN if phenotype-negative. "
            "FLECAINIDE CONTRAINDICATED in ARVC (proarrhythmic in structural heart disease). "
            "DO NOT use epsilon wave absence to exclude ARVC — epsilon waves are present in only ~30-50% of ARVC; "
            "CMR is the most sensitive structural investigation. "
            "AVOID dehydration — exacerbates arrhythmia risk. "
            "PACEMAKER ALONE IS INSUFFICIENT for high-risk ARVC — VF risk requires ICD capability."
        ),
        "key_ddx": [
            "DSP-ARVC8: biventricular involvement (LV+RV); woolly hair + PPK = Carvajal; LGE subepicardial LV",
            "LMNA-DCM: AV block prominent; LV-dominant DCM; limb-girdle myopathy overlap; different ECG pattern",
            "RV infarction: ischaemic; coronary territory; age/risk factors; no family history; negative genetics",
            "Brugada syndrome (SCN5A): RBBB + ST elevation V1-V3; no structural changes; different ECG pattern",
            "Sarcoidosis: AV block + VT + patchy LGE; systemic sarcoid; negative genetics; biopsy diagnostic",
        ],
        "severity_weights": {"Mild": 0.22, "Moderate": 0.42, "Severe": 0.36},
        "onset_age_range": (20, 45),
        "dx_lag_y": (2, 8),
        "drug_error_rate": 0.20,
        "icd_eligible_rate": 0.55,
        "cardiac_transplant_rate": 0.10,
        "arrhythmia_rate": 0.75,
        "scd_risk_high_rate": 0.45,
        "lvoto_rate": 0.05,
        "progression_rate": 0.80,
        "first_line_drug": "Sports restriction (MANDATORY); ICD (VT/VF); Sotalol/Amiodarone (VT)",
    },
    # ── DSP — ARVC8 / DCMEP ─────────────────────────────────────────────────
    {
        "gene": "DSP",
        "protein": "Desmoplakin (DSP)",
        "alias": "DSP; OMIM gene 125647; 6p24.3; ~2871 aa; ARVC8 (OMIM #607450) + DCMEP (OMIM #605676); AD (haploinsufficiency); biventricular ARVC; Carvajal syndrome (AR biallelic) = woolly hair + PPK PATHOGNOMONIC",
        "aa": "~2871 aa",
        "kDa": "~332 kDa",
        "mechanism": (
            "DSP encodes desmoplakin, the largest and most abundant desmosomal plaque protein. "
            "Desmoplakin bridges the desmosomal core (plakophilin-2, plakoglobin) to cytoplasmic "
            "intermediate filaments (desmin) — essential for desmosome-cytoskeleton linkage in "
            "cardiomyocytes and epithelial cells. "
            "NORMAL FUNCTION: DSP N-terminus binds desmosomal plaque proteins (PKP2, JUP); "
            "C-terminus binds desmin intermediate filaments (three plakin repeat domains: A, B, C); "
            "confers mechanical resilience to intercalated disc under contractile stress. "
            "ARVC8 PATHOMECHANISM (AD haploinsufficiency): truncating DSP variants → "
            "50% DSP protein → weakened desmosome-cytoskeleton link → cardiomyocyte death under "
            "mechanical stress → fibro-fatty replacement — BIVENTRICULAR (LV + RV both affected). "
            "DSP-ARVC DISTINCTIVE FEATURE: LV involvement is the HALLMARK — "
            "LGE on CMR is SUBEPICARDIAL INFEROLATERAL (distinct from ischaemic mid-wall or subendocardial). "
            "CARVAJAL SYNDROME (AR biallelic DSP): homozygous or compound heterozygous DSP truncating → "
            "severe dilated cardiomyopathy + WOOLLY HAIR + PALMOPLANTAR KERATODERMA (PPK) — "
            "cutaneous features are PATHOGNOMONIC clues to the genetic diagnosis."
        ),
        "disease_type": (
            "ARVC8 (OMIM #607450) + DCMEP (OMIM #605676); AD haploinsufficiency (monoallelic); "
            "biventricular ARVC — LV+RV involvement HALLMARK; "
            "LGE subepicardial inferolateral DISTINCTIVE on CMR; "
            "ICD MANDATORY for DSP nonsense/frameshift; "
            "Carvajal syndrome (AR biallelic): woolly hair + PPK PATHOGNOMONIC"
        ),
        "locus": "6p24.3",
        "omim_gene": 125647,
        "omim_disease": 607450,
        "inheritance": (
            "AUTOSOMAL DOMINANT (ARVC8/DCMEP): haploinsufficiency — truncating DSP variants "
            "(frameshift, nonsense, splice-site) most pathogenic; "
            "some missense variants in desmin-binding domain also pathogenic. "
            "CARVAJAL SYNDROME: AUTOSOMAL RECESSIVE — biallelic DSP truncating variants; "
            "severe dilated cardiomyopathy + woolly hair + PPK; allelic to AD ARVC8. "
            "PENETRANCE (AD): incomplete — ~70-80% of carriers develop cardiac phenotype by age 50; "
            "females sometimes less severely affected than males. "
            "SPORTS RESTRICTION: MANDATORY in DSP gene-positive individuals — "
            "same rationale as PKP2 (exercise drives desmosomal stress + progression). "
            "FAMILY SCREENING: ECG + Holter + echo + CMR + genetics for all first-degree relatives. "
            "ARRHYTHMIA RISK IN DSP: malignant even with preserved EF — "
            "ICD indicated for DSP nonsense/frameshift with any arrhythmia phenotype."
        ),
        "phenotype": (
            "DSP-ARVC8 — BIVENTRICULAR ARVC (HALLMARK): "
            "CARDIAC: biventricular dilation and dysfunction (LV + RV) — "
            "LV involvement is more prominent in DSP than in PKP2 (which is RV-dominant). "
            "CMR: subepicardial LGE in the inferolateral LV wall — "
            "DISTINCTIVE PATTERN (subendocardial = ischaemic; mid-wall = LMNA; subepicardial inferolateral = DSP). "
            "ECG: T-wave inversions V1-V6 (biventricular); LBBB or RBBB; epsilon waves less prominent than PKP2. "
            "ARRHYTHMIA: malignant VT/VF even with preserved or mildly reduced LVEF — "
            "DO NOT wait for EF to drop before considering ICD in DSP. "
            "CUTANEOUS FEATURES (Carvajal syndrome — AR biallelic DSP): "
            "WOOLLY HAIR (tight curly hair from birth) PATHOGNOMONIC; "
            "PALMOPLANTAR KERATODERMA (thick, hyperkeratotic skin on palms and soles) PATHOGNOMONIC; "
            "these features with cardiac disease = immediate DSP genetic testing. "
            "DCMEP (AD monoallelic): dilated CM + epidermolytic features (rare variant)."
        ),
        "treatment_options": [
            "ICD MANDATORY for DSP nonsense/frameshift variants: "
            "given malignant arrhythmia risk even with preserved EF; "
            "do not wait for significant RV/LV dysfunction — "
            "ICD threshold lower in DSP than PKP2 because LV involvement worsens VF risk; "
            "primary prevention ICD strongly indicated when ≥1 risk factor present (NSVT, syncope, LGE)",
            "Sports restriction MANDATORY: all DSP gene-positive individuals regardless of phenotype; "
            "competitive sports and high-intensity endurance exercise prohibited; "
            "low-moderate recreational activity only",
            "Cardiac MRI with LGE: mandatory at diagnosis and annually — "
            "subepicardial inferolateral LGE quantification; "
            "CMR superior to echo for detecting LV involvement in DSP-ARVC",
            "Antiarrhythmic therapy: sotalol or amiodarone for VT suppression; "
            "catheter ablation for refractory VT (epicardial approach often required given "
            "subepicardial LGE substrate)",
            "HFrEF therapy for biventricular dysfunction: "
            "ACEi/ARB/sacubitril-valsartan + beta-blocker + MRA; "
            "loop diuretics for fluid overload",
            "Carvajal syndrome (AR): early aggressive cardiac therapy; "
            "dermatology referral (PPK management — emollients, keratolytics); "
            "audiological assessment; cardiac transplantation listing early given severe phenotype",
            "Genetic counselling: Carvajal families — both parents carriers; 25% recurrence; "
            "prenatal testing available; cutaneous features allow early clinical suspicion in offspring",
        ],
        "critical_avoid": (
            "DSP-ARVC8: DO NOT WAIT FOR EF TO DROP before ICD implantation — "
            "DSP VT/VF can occur with preserved EF; malignant arrhythmia threshold is lower. "
            "SPORTS RESTRICTION MANDATORY. "
            "Subepicardial inferolateral LGE on CMR = DSP until proven otherwise — "
            "do not attribute this pattern to ischaemia without coronary workup. "
            "CARVAJAL CLUE: woolly hair + PPK + dilated CM in young patient = "
            "DSP biallelic immediately; do not miss this diagnostic opportunity. "
            "FLECAINIDE CONTRAINDICATED in ARVC (proarrhythmic). "
            "Holter mandatory — NSVT in DSP = high-risk finding demanding ICD referral."
        ),
        "key_ddx": [
            "PKP2-ARVC9: RV-dominant (not biventricular); epsilon waves more prominent; RV-LGE at insertion points",
            "LMNA-DCM: AV block PATHOGNOMONIC; no epsilon waves; no subepicardial LGE; limb-girdle myopathy",
            "Myocarditis: acute onset; troponin elevation; CMR oedema sequences; negative genetics usually",
            "Cardiac sarcoidosis: AV block + VT + basal septal LGE; systemic features; biopsy; ACE level",
            "Dilated cardiomyopathy (TTN, RBM20): no subepicardial LGE; no epsilon waves; different genetic cause",
        ],
        "severity_weights": {"Mild": 0.18, "Moderate": 0.40, "Severe": 0.42},
        "onset_age_range": (15, 40),
        "dx_lag_y": (2, 9),
        "drug_error_rate": 0.22,
        "icd_eligible_rate": 0.60,
        "cardiac_transplant_rate": 0.12,
        "arrhythmia_rate": 0.80,
        "scd_risk_high_rate": 0.52,
        "lvoto_rate": 0.04,
        "progression_rate": 0.85,
        "first_line_drug": "ICD MANDATORY (DSP nonsense/frameshift); sports restriction; CMR annual; sotalol/amiodarone",
    },
    # ── LMNA — DCM1A / EDMD2 ────────────────────────────────────────────────
    {
        "gene": "LMNA",
        "protein": "Lamin A/C (LMNA)",
        "alias": "LMNA; OMIM gene 150330; 1q22; ~664 aa; DCM1A (OMIM #115200) + EDMD2 (OMIM #181350); AD; most common genetic DCM with conduction disease; AV block PATHOGNOMONIC; ICD MANDATORY; SCD 15-40% lifetime",
        "aa": "~664 aa (Lamin A); ~572 aa (Lamin C, alternatively spliced)",
        "kDa": "~74 kDa (Lamin A); ~65 kDa (Lamin C)",
        "mechanism": (
            "LMNA encodes lamins A and C (via alternative splicing), type V intermediate filament proteins "
            "that polymerise to form the nuclear lamina — a meshwork lining the inner nuclear membrane. "
            "NORMAL FUNCTION: lamins A/C maintain nuclear shape and mechanical stability; "
            "regulate chromatin organisation, gene expression, and DNA damage response; "
            "interact with emerin (EMD), SUN proteins, and nesprin — forming the LINC complex "
            "that transmits cytoskeletal forces to the nucleus. "
            "CARDIAC PATHOMECHANISM: LMNA variants → nuclear lamina weakness → "
            "mechanically vulnerable cardiomyocyte nuclei → stress-induced apoptosis → "
            "dilated cardiomyopathy; ALSO → dysregulated signalling (MAPK/ERK pathway hyperactivation) → "
            "fibrosis; AND → cardiomyocyte dropout at conduction tissue (AV node, His bundle) → "
            "AV block (PR prolongation → complete AV block). "
            "AV BLOCK PATHOGENESIS: conduction system cardiomyocytes are particularly vulnerable to "
            "LMNA dysfunction — PR prolongation is often the FIRST manifestation. "
            "HIGH SCD RISK: progressive ventricular fibrosis + conduction disease + DCM → "
            "VT/VF; SCD risk 15-40% lifetime without ICD."
        ),
        "disease_type": (
            "DCM1A (OMIM #115200) + EDMD2 Emery-Dreifuss MD (OMIM #181350); AD; "
            "nuclear lamina intermediate filament; most common GENETIC DCM with conduction disease; "
            "AV BLOCK PATHOGNOMONIC (PR prolongation → complete AV block); "
            "ICD MANDATORY when EF<45% OR NSVT OR AV block ≥2nd degree; SCD risk 15-40% lifetime; "
            "pacemaker alone INSUFFICIENT — VF risk requires ICD capability"
        ),
        "locus": "1q22",
        "omim_gene": 150330,
        "omim_disease": 115200,
        "inheritance": (
            "AUTOSOMAL DOMINANT: heterozygous pathogenic variants. "
            "VARIANT SPECTRUM: missense, truncating, splice-site — all classes reported; "
            "certain variants (p.Arg190Trp, LMNA hotspot) particularly associated with severe phenotype. "
            "PENETRANCE: high — ~80-90% of carriers develop DCM and/or conduction disease by age 60. "
            "PROGRESSION: LMNA DCM is one of the MOST PROGRESSIVE genetic DCMs — "
            "EF decline ~5%/year on average; cardiac transplantation required in ~20% by age 55. "
            "EDMD2 OVERLAP: same LMNA variants cause EDMD2 (Emery-Dreifuss muscular dystrophy type 2) — "
            "AD; skeletal muscle (humero-peroneal distribution) + early joint contractures + "
            "cardiac conduction disease; overlap with DCM1A is common. "
            "FAMILY SCREENING: ECG (PR interval) + Holter + echo + genetic testing cascade; "
            "annual screening — AV block can develop rapidly."
        ),
        "phenotype": (
            "LMNA-DCM (DCM1A) — SIGNATURE PHENOTYPE: "
            "CONDUCTION DISEASE (HALLMARK): PR prolongation → 1st degree AV block → "
            "2nd degree AV block (Mobitz I/II) → 3rd degree (complete) AV block → "
            "pacemaker dependency; sinus node dysfunction also common. "
            "DCM: left ventricular dilation + reduced LVEF (mean LVEF ~35-40% at diagnosis); "
            "progressive — EF worsens over years even with optimal medical therapy. "
            "SCD RISK: HIGH — 15-40% lifetime risk from VT/VF; "
            "SCD can occur BEFORE significant EF reduction; "
            "PACEMAKER ALONE IS INSUFFICIENT for LMNA DCM — must be ICD capable. "
            "SKELETAL MYOPATHY (EDMD2 overlap): limb-girdle pattern weakness; "
            "early contractures (elbow, Achilles, spine); CK mildly elevated (200-500 IU/L). "
            "CMR: diffuse LGE mid-wall + LV free wall; RV involvement in advanced disease. "
            "COURSE: rapid — mean time from diagnosis to transplant/death ~10 years without ICD."
        ),
        "treatment_options": [
            "ICD MANDATORY when ANY of: LVEF <45%; NSVT on Holter; 2nd or 3rd degree AV block; "
            "syncope; family history of SCD in LMNA; "
            "ICD provides defibrillation capability — pacemaker alone leaves patient at VF risk; "
            "CRT-D (resynchronisation + ICD) if LVEF <35% + LBBB QRS >150 ms",
            "Pacemaker therapy: for complete AV block + haemodynamic compromise; "
            "but ALWAYS upgrade to ICD capability (CRT-D or ICD-PM combo) given SCD risk — "
            "NEVER implant pacemaker only in LMNA with known DCM + NSVT",
            "Standard HFrEF therapy: ACEi (ramipril/enalapril) or sacubitril/valsartan + "
            "beta-blocker (carvedilol/bisoprolol) + spironolactone/eplerenone + "
            "SGLT2 inhibitor (dapagliflozin/empagliflozin — HFrEF mortality benefit); "
            "loop diuretics (furosemide) for fluid overload; "
            "target 'quadruple therapy' (ACEI/ARNi + BB + MRA + SGLT2i)",
            "Cardiac transplantation: LMNA DCM progresses rapidly — early referral to transplant "
            "centre recommended; transplantation required in ~20% by age 55; "
            "LMNA is one of the commonest genetic indications for transplantation",
            "Holter monitoring: 24-48h annually + at any symptom change; "
            "detect NSVT (ICD trigger) and AV block progression; "
            "permanent monitoring (implantable loop recorder) in high-risk LMNA carriers",
            "MEK inhibitor research (selumetinib/trametinib): ERK/MAPK hyperactivation in LMNA DCM "
            "is a therapeutic target in preclinical models; clinical trials underway",
            "Orthopaedic/physiotherapy (EDMD2 overlap): contracture management; "
            "ankle-foot orthoses for Achilles contracture; spine bracing for scoliosis; "
            "respiratory assessment (NIV if FVC reduced)",
        ],
        "critical_avoid": (
            "LMNA DCM: PACEMAKER ALONE IS INSUFFICIENT — ICD capability MANDATORY. "
            "DO NOT implant pacemaker-only in any LMNA DCM patient with known NSVT or EF <45% — "
            "this patient requires ICD; pacemaker alone leaves them at VF death risk. "
            "ICD threshold LOWER than other DCM genes: AV block alone + LMNA = ICD (not pacemaker). "
            "NSVT in LMNA DCM = ICD indication (do not wait for sustained VT). "
            "EF can be NORMAL at first presentation — AV block or NSVT alone warrants ICD. "
            "Avoid nephrotoxic drugs — renal function critical for diuretic management in advanced DCM."
        ),
        "key_ddx": [
            "EMD-EDMD1 (emerin): X-linked; Emery-Dreifuss MD XLR; clinically similar conduction disease + skeletal myopathy",
            "TTN-DCM1G: DCM without prominent conduction disease; no AV block; largest DCM gene",
            "RBM20-DCM: most aggressive DCM; biventricular; RS domain hotspot; earlier onset",
            "Cardiac sarcoidosis: AV block + DCM + LGE; systemic sarcoid; biopsy diagnostic; negative LMNA",
            "Myocarditis (viral): acute onset; troponin; CMR oedema; negative genetics",
        ],
        "severity_weights": {"Mild": 0.15, "Moderate": 0.40, "Severe": 0.45},
        "onset_age_range": (25, 55),
        "dx_lag_y": (2, 8),
        "drug_error_rate": 0.28,
        "icd_eligible_rate": 0.70,
        "cardiac_transplant_rate": 0.20,
        "arrhythmia_rate": 0.82,
        "scd_risk_high_rate": 0.55,
        "lvoto_rate": 0.03,
        "progression_rate": 0.90,
        "first_line_drug": "ICD MANDATORY (AV block+NSVT+EF<45%); sacubitril-valsartan; SGLT2i; pacemaker → ICD upgrade",
    },
    # ── TTN — DCM1G ─────────────────────────────────────────────────────────
    {
        "gene": "TTN",
        "protein": "Titin (TTN)",
        "alias": "TTN; OMIM gene 188840; 2q31.2; ~34350 aa; DCM1G (OMIM #604145); AD; largest human protein; A-band TTNtv PATHOGENIC; most common genetic DCM ~25% familial; peripartum CM association",
        "aa": "~34350 aa (N2BA cardiac isoform; largest human protein)",
        "kDa": "~3,700 kDa (3.7 MDa)",
        "mechanism": (
            "TTN encodes titin, the largest protein in the human body (~34,350 amino acids, 3.7 MDa), "
            "a giant elastic molecule that spans the half-sarcomere from Z-disc (N-terminus) to "
            "M-band (C-terminus). "
            "CARDIAC ISOFORMS: N2B (stiffer, adult ventricle) and N2BA (more compliant, adjustable ratio); "
            "titin acts as a 'molecular spring' — generates passive tension during diastole and "
            "provides elastic restoring force during systole. "
            "NORMAL FUNCTION: titin maintains sarcomere structure integrity; "
            "contains myosin-binding C-zone (A-band region); "
            "kinase domain (M-band) signals mechanical stress to nucleus; "
            "anchors thick filament to M-band. "
            "DCM1G PATHOMECHANISM: truncating variants (TTNtv) in the A-band → "
            "haploinsufficiency (NMD of mutant transcript) → 50% titin → "
            "reduced sarcomere elasticity and force generation → dilated cardiomyopathy. "
            "A-BAND vs I-BAND DISTINCTION MANDATORY: A-band TTNtv = PATHOGENIC; "
            "I-band TTNtv = likely benign/VUS (constitutively expressed titin domains must be disrupted). "
            "PERIPARTUM CARDIOMYOPATHY: TTNtv found in ~15% of PPCM — genetic testing mandatory."
        ),
        "disease_type": (
            "DCM1G (OMIM #604145); AD; largest human protein; sarcomeric elastic element; "
            "A-band TTNtv PATHOGENIC (I-band = likely benign/VUS — A-band vs I-band distinction MANDATORY); "
            "25% of familial DCM; 18% sporadic DCM; peripartum CM in 15%; incomplete penetrance ~40% by age 40; "
            "standard HFrEF therapy; LGE rare (poor prognosis if present)"
        ),
        "locus": "2q31.2",
        "omim_gene": 188840,
        "omim_disease": 604145,
        "inheritance": (
            "AUTOSOMAL DOMINANT: heterozygous truncating variants (TTNtv) — frameshift, nonsense, "
            "splice-site. "
            "A-BAND vs I-BAND CURATION MANDATORY: A-band TTNtv = pathogenic (causes DCM); "
            "I-band TTNtv = likely benign or VUS (often found in general population without DCM). "
            "This distinction MUST be made before reporting as pathogenic — "
            "ClinGen TTNtv variant curation framework mandatory for all labs. "
            "PENETRANCE: INCOMPLETE — only ~40% of A-band TTNtv carriers develop DCM by age 40; "
            "penetrance increases to ~60-70% with age and in males. "
            "MODIFIERS: peripartum state dramatically unmasks TTNtv (PPCM); "
            "alcohol excess and viral illness can unmask. "
            "PERIPARTUM CM: TTNtv in ~15% of PPCM — offer genetic testing to ALL PPCM patients. "
            "FAMILY SCREENING: echo + ECG + genetic testing; screening more intensive if LVEF <45% in proband."
        ),
        "phenotype": (
            "TITIN DILATED CARDIOMYOPATHY (DCM1G): "
            "CLINICAL: left ventricular dilation + reduced LVEF (typical LVEF 25-45% at diagnosis); "
            "symptoms of heart failure (dyspnoea, oedema, fatigue, reduced exercise tolerance); "
            "onset typically 3rd-5th decade; males more severely affected than females. "
            "ECG: LBBB common (interventricular conduction delay); AF in ~30%; "
            "AV block less prominent than LMNA (distinguishes the two). "
            "CMR: LGE is RARE in TTN-DCM (unlike LMNA or DSP); "
            "if LGE present = WORSE PROGNOSIS (more fibrosis, more arrhythmia risk). "
            "PERIPARTUM CARDIOMYOPATHY (PPCM): TTNtv in ~15% — presents last month of pregnancy "
            "or within 5 months postpartum; LV recovery possible with standard HFrEF therapy "
            "but ~50% have persistent LV dysfunction; subsequent pregnancy carries risk of relapse. "
            "INCOMPLETE PENETRANCE: many TTNtv carriers have normal echo — genetic counselling complex. "
            "PROGNOSIS: generally intermediate — not as aggressive as LMNA or RBM20 but progressive."
        ),
        "treatment_options": [
            "Standard HFrEF therapy (quadruple therapy): "
            "ACEi (ramipril/lisinopril) or sacubitril/valsartan (entresto) — superior to ACEi alone; "
            "beta-blocker (carvedilol/bisoprolol — target maximum tolerated dose); "
            "mineralocorticoid antagonist (spironolactone 25-50 mg or eplerenone); "
            "SGLT2 inhibitor (dapagliflozin 10 mg or empagliflozin 10 mg — mortality benefit in HFrEF); "
            "combine all four classes unless contraindicated",
            "Device therapy: ICD if LVEF <35% despite ≥3 months optimal medical therapy; "
            "CRT-D if LVEF <35% + LBBB + QRS >150 ms; "
            "TTN-DCM: arrhythmia burden lower than LMNA/RBM20 — device threshold as per standard HFrEF guidelines",
            "AF management: anticoagulate ALL DCM+AF (CHA₂DS₂-VASc independent — "
            "cardioembolic risk high in DCM with AF); "
            "rhythm control preferred (cardioversion, flecainide/sotalol if EF normalised, amiodarone)",
            "Peripartum CM management: standard HFrEF therapy + bromocriptine (2.5 mg BD × 2 weeks "
            "→ 2.5 mg OD × 4 weeks — inhibits prolactin, prolactin drives PPCM pathogenesis); "
            "anticoagulation during peripartum period; "
            "subsequent pregnancy: cardiac surveillance mandatory; genetic counselling re recurrence risk",
            "ClinGen TTNtv curation: MANDATORY before reporting — confirm A-band localisation; "
            "use ClinVar/ClinGen titin variant database; "
            "I-band TTNtv = VUS → counsel family accordingly (not confirmed pathogenic)",
            "Genetic counselling: incomplete penetrance explanation; "
            "at-risk relatives (50% carrier probability) should have echo baseline + genetic testing; "
            "PPCM patients: genetic testing → family cascade if A-band TTNtv confirmed; "
            "AVOID alcohol excess (unmasks DCM in TTNtv carriers)",
            "Cardiac transplantation: for end-stage TTN-DCM refractory to medical/device therapy; "
            "TTN-DCM transplant outcomes generally good",
        ],
        "critical_avoid": (
            "TTN-DCM: A-BAND vs I-BAND DISTINCTION MANDATORY before reporting pathogenic — "
            "reporting an I-band TTNtv as pathogenic is a clinical laboratory error; "
            "ClinGen TTNtv curation framework MANDATORY. "
            "PERIPARTUM CM: TTN testing mandatory — do not miss 15% TTNtv carrier rate; "
            "subsequent pregnancy without genetic counselling is unsafe. "
            "LGE on CMR = worse prognosis — do not assume benign course if LGE detected. "
            "AVOID ALCOHOL EXCESS in TTNtv carriers — major DCM unmasking trigger. "
            "Do not stop HFrEF medications based on LVEF improvement alone — "
            "discontinuation risks relapse (especially in PPCM)."
        ),
        "key_ddx": [
            "LMNA-DCM1A: AV block PATHOGNOMONIC (TTN has no AV block); pacemaker/ICD distinction",
            "RBM20-DCM1HH: more aggressive; RS domain hotspot; biventricular; earlier transplant",
            "Peripartum CM (non-genetic): TTNtv in 15% — genetic workup mandatory in ALL PPCM",
            "Alcoholic CM: heavy alcohol history; partial recovery with abstinence; TTNtv may coexist",
            "Viral myocarditis → DCM: acute onset; troponin; CMR oedema; genetic testing to exclude TTNtv",
        ],
        "severity_weights": {"Mild": 0.28, "Moderate": 0.42, "Severe": 0.30},
        "onset_age_range": (30, 55),
        "dx_lag_y": (2, 10),
        "drug_error_rate": 0.20,
        "icd_eligible_rate": 0.40,
        "cardiac_transplant_rate": 0.12,
        "arrhythmia_rate": 0.42,
        "scd_risk_high_rate": 0.28,
        "lvoto_rate": 0.02,
        "progression_rate": 0.70,
        "first_line_drug": "Sacubitril-valsartan + carvedilol + MRA + SGLT2i; ClinGen A-band curation MANDATORY",
    },
    # ── RBM20 — DCM1HH ──────────────────────────────────────────────────────
    {
        "gene": "RBM20",
        "protein": "RNA-Binding Motif Protein 20 (RBM20)",
        "alias": "RBM20; OMIM gene 613171; 10q25.2; ~1228 aa; DCM1HH (OMIM #613642); AD; most aggressive genetic DCM; RS-domain hotspot (p.Arg634Gln, p.Arg636Ser); ICD MANDATORY; early transplant",
        "aa": "~1228 aa",
        "kDa": "~145 kDa",
        "mechanism": (
            "RBM20 encodes an RNA-binding protein that acts as a master splicing regulator for "
            "cardiac-specific transcripts — particularly TTN (titin) and other sarcomeric genes. "
            "NORMAL FUNCTION: RBM20 binds UCUU motifs in pre-mRNAs and represses inclusion of "
            "certain exons; its primary cardiac target is TTN — RBM20 splicing of TTN determines "
            "the N2B (stiff) vs N2BA (compliant) ratio of titin isoforms in the ventricle; "
            "also regulates splicing of CAMK2D, MYOM1, LDB3, and other cardiac genes. "
            "DCM1HH PATHOMECHANISM: RBM20 mutations → splicing dysregulation of TTN and other "
            "cardiac transcripts → giant N2BA-like titin isoforms (mis-spliced) → "
            "altered sarcomere passive tension + Z-disc disorganisation → dilated cardiomyopathy. "
            "RS-DOMAIN HOTSPOT: pathogenic variants cluster in arginine-serine (RS) rich domain "
            "(p.Arg634Gln, p.Arg636Ser, p.Arg636His) — nuclear localisation signal for RBM20; "
            "hotspot variants → cytoplasmic mislocalisation of RBM20 → loss of nuclear splicing function. "
            "AGGRESSIVE PHENOTYPE: the combination of titin mis-splicing + multiple sarcomeric gene "
            "dysregulation makes RBM20-DCM the most mechanistically disruptive and clinically aggressive "
            "of all genetic DCMs."
        ),
        "disease_type": (
            "DCM1HH (OMIM #613642); AD; splicing factor for titin + sarcomeric genes; "
            "most aggressive genetic DCM; RS-domain hotspot p.Arg634Gln/p.Arg636Ser; "
            "biventricular dilation PATHOGNOMONIC; early onset (30s-40s); HIGH SCD risk; "
            "ICD MANDATORY; cardiac transplant frequently required by 5th decade"
        ),
        "locus": "10q25.2",
        "omim_gene": 613171,
        "omim_disease": 613642,
        "inheritance": (
            "AUTOSOMAL DOMINANT: heterozygous pathogenic variants — predominantly missense in RS domain. "
            "RS-DOMAIN HOTSPOT: p.Arg634Gln, p.Arg636Ser, p.Arg636His — "
            "Waddell-Smith families (Australian founder) and other founder populations; "
            "hotspot variants are FUNCTIONALLY DOMINANT NEGATIVE (not just haploinsufficiency). "
            "PENETRANCE: high — ~90% of RS-domain variant carriers develop DCM; "
            "onset typically 3rd-4th decade; male penetrance higher and earlier than female. "
            "WADDELL-SMITH FAMILIES: Australasian founder effect — cluster of severe RBM20 DCM "
            "families in Australia identified through founder p.Arg634Gln mutation. "
            "FAMILY SCREENING: aggressive cascade — all first-degree relatives; "
            "annual echo + Holter from early adulthood in known RS-domain variant families; "
            "early ICD implantation in high-risk carriers."
        ),
        "phenotype": (
            "RBM20 DILATED CARDIOMYOPATHY (DCM1HH) — MOST AGGRESSIVE GENETIC DCM: "
            "ONSET: typically 30s-40s (earlier than TTN or LMNA). "
            "BIVENTRICULAR DILATION: both LV and RV dilated and dysfunctional — "
            "PATHOGNOMONIC of RBM20-DCM; LV LVEDD often >65 mm at presentation. "
            "LVEF: severely reduced — LVEF <30% common at diagnosis. "
            "SCD RISK: HIGH — VT/VF from massive biventricular dilation and fibrosis; "
            "SCD risk among highest of genetic DCMs. "
            "MALIGNANT VT: fast monomorphic VT; frequent appropriate ICD shocks; "
            "VT storm possible. "
            "CMR: biventricular LGE; RV insertion point LGE; extensive mid-wall fibrosis. "
            "COURSE: rapid progression — LVEF decline of 5-10%/year; "
            "many patients require cardiac transplantation by 5th decade; "
            "median time to transplant or death ~8-12 years from diagnosis without aggressive therapy. "
            "WADDELL-SMITH FAMILIES: several generations of severe DCM + SCD in affected pedigrees."
        ),
        "treatment_options": [
            "ICD MANDATORY in all RBM20 pathogenic variant carriers with ANY of: "
            "LVEF <35%; NSVT; syncope; family SCD; "
            "given aggressive natural history, primary prevention ICD threshold very low; "
            "CRT-D if LVEF <35% + LBBB + QRS >150 ms for resynchronisation benefit; "
            "VT storm management: sedation + amiodarone IV; overdrive pacing; "
            "urgent EP referral for catheter ablation if recurrent VT shocks",
            "Aggressive HFrEF therapy: quadruple therapy MANDATORY from diagnosis — "
            "sacubitril-valsartan (proven superior to ACEi in HFrEF) + carvedilol + "
            "eplerenone/spironolactone + dapagliflozin/empagliflozin; "
            "titrate rapidly to maximum tolerated doses; "
            "hospitalisation for IV diuresis if fluid overload",
            "Early cardiac transplantation listing: RBM20-DCM has aggressive course; "
            "transplant listing recommended earlier than other DCMs; "
            "Waddell-Smith families: transplant frequently required by 5th decade; "
            "discuss transplant early with patient and family when LVEF <25%",
            "Catheter ablation for VT: for recurrent VT/appropriate ICD shocks; "
            "experienced centre required; biventricular substrate — both endocardial and epicardial mapping; "
            "ablation reduces VT burden but rarely curative given extensive fibrosis",
            "Antiarrhythmic therapy: amiodarone for VT suppression in addition to ICD; "
            "sotalol if amiodarone not tolerated; mexiletine as adjunct; "
            "NOT a substitute for ICD",
            "RBM20 genetic counselling: RS-domain variants — explain near-complete penetrance; "
            "Waddell-Smith family history — confirm founder variant; "
            "cascade testing all first-degree relatives urgently; "
            "predictive testing in asymptomatic relatives from age 18-20 with annual echo surveillance",
            "Monitoring frequency: 3-6 monthly echo + Holter (not annual) given rapid progression; "
            "CMR annually for LGE quantification; BNP/NT-proBNP monitoring for decompensation",
        ],
        "critical_avoid": (
            "RBM20-DCM: DO NOT manage as routine DCM — this is the MOST AGGRESSIVE genetic DCM. "
            "ICD MANDATORY: do not wait for sustained VT — prophylactic ICD given high SCD risk. "
            "EARLY TRANSPLANT LISTING: do not delay until end-stage; "
            "earlier referral to transplant centre improves outcomes. "
            "ANNUAL MONITORING INSUFFICIENT — use 3-6 monthly echo given rapid EF decline. "
            "Do NOT use flecainide or propafenone for VT in RBM20-DCM (structural heart disease). "
            "BIVENTRICULAR DILATION + EARLY ONSET DCM = RBM20 until proven otherwise — "
            "ensure genetic panel includes RBM20 (often underpanelled in older DCM gene panels)."
        ),
        "key_ddx": [
            "LMNA-DCM1A: AV block PATHOGNOMONIC (RBM20 has ventricular arrhythmia not AV block); different CMR",
            "TTN-DCM1G: less aggressive; incomplete penetrance; I-band vs A-band; no RS domain hotspot",
            "DSP-ARVC8: subepicardial LGE inferolateral; desmosomal; cutaneous features; different MRI pattern",
            "Viral/inflammatory DCM: acute onset; troponin; CMR oedema; genetic testing negative",
            "Peripartum CM: female; pregnancy-associated; ~15% have TTNtv; RBM20 rare in PPCM",
        ],
        "severity_weights": {"Mild": 0.10, "Moderate": 0.35, "Severe": 0.55},
        "onset_age_range": (25, 50),
        "dx_lag_y": (1, 5),
        "drug_error_rate": 0.18,
        "icd_eligible_rate": 0.85,
        "cardiac_transplant_rate": 0.30,
        "arrhythmia_rate": 0.88,
        "scd_risk_high_rate": 0.65,
        "lvoto_rate": 0.02,
        "progression_rate": 0.95,
        "first_line_drug": "ICD MANDATORY; sacubitril-valsartan + carvedilol + MRA + SGLT2i; early transplant listing",
    },
]


def _gen_patients_for_gene(gene_data: dict, seed: int) -> list:
    rng = random.Random(seed)
    n = 40
    patients = []
    sw = gene_data["severity_weights"]
    severities = list(sw.keys())
    weights = list(sw.values())

    for i in range(n):
        sev = rng.choices(severities, weights=weights, k=1)[0]

        onset_lo, onset_hi = gene_data["onset_age_range"]
        onset = round(rng.uniform(onset_lo, max(onset_lo + 0.01, onset_hi)), 2)
        lag = round(rng.uniform(*gene_data["dx_lag_y"]), 2)
        dx_age = round(onset + lag, 2)

        prog = rng.random() < gene_data["progression_rate"]
        drug_err = rng.random() < gene_data["drug_error_rate"]
        icd_elig = rng.random() < gene_data["icd_eligible_rate"]
        transplant = rng.random() < gene_data["cardiac_transplant_rate"]
        arrhythmia = rng.random() < gene_data["arrhythmia_rate"]
        scd_risk = rng.random() < gene_data["scd_risk_high_rate"]
        lvoto = rng.random() < gene_data["lvoto_rate"]

        # Heart failure severity correlated with overall severity
        hf_hospitalised = sev == "Severe" or (sev == "Moderate" and rng.random() < 0.45)

        patients.append({
            "id": f"CARDIO-{gene_data['gene']}-{seed}-{i + 1:03d}",
            "gene": gene_data["gene"],
            "seed": seed,
            "onset_age_y": onset,
            "diagnosis_age_y": dx_age,
            "severity": sev,
            "icd_eligible": icd_elig,
            "cardiac_transplant": transplant,
            "arrhythmia": arrhythmia,
            "scd_risk_high": scd_risk,
            "lvoto": lvoto,
            "hf_hospitalised": hf_hospitalised,
            "drug_avoid_prescribed_error": drug_err,
            "disease_progression": prog,
        })
    return patients


def _gen_cohort() -> list:
    all_patients = []
    for i, gd in enumerate(CARDIOMYOPATHY_GENES):
        all_patients.extend(_gen_patients_for_gene(gd, SEED_BASE + i))
    return all_patients


def get_overview() -> dict:
    patients = _gen_cohort()
    n = len(patients)

    sev = {"Mild": 0, "Moderate": 0, "Severe": 0}
    for p in patients:
        sev[p["severity"]] += 1

    icd_n = sum(1 for p in patients if p["icd_eligible"])
    transplant_n = sum(1 for p in patients if p["cardiac_transplant"])
    arrhythmia_n = sum(1 for p in patients if p["arrhythmia"])
    scd_n = sum(1 for p in patients if p["scd_risk_high"])
    lvoto_n = sum(1 for p in patients if p["lvoto"])
    hf_hosp_n = sum(1 for p in patients if p["hf_hospitalised"])
    drug_err_n = sum(1 for p in patients if p["drug_avoid_prescribed_error"])
    prog_n = sum(1 for p in patients if p["disease_progression"])

    onsets = [p["onset_age_y"] for p in patients]
    mean_onset = round(sum(onsets) / len(onsets), 2)
    mean_dx = round(sum(p["diagnosis_age_y"] for p in patients) / n, 2)

    # Per-gene clinical features prevalence (% arrhythmia as proxy)
    gene_arrhythmia_pct = {}
    for gd in CARDIOMYOPATHY_GENES:
        gpts = [p for p in patients if p["gene"] == gd["gene"]]
        gene_arrhythmia_pct[gd["gene"]] = round(
            100 * sum(1 for p in gpts if p["arrhythmia"]) / len(gpts), 1
        )

    return {
        "atlas": "Cardiomyopathy-Atlas",
        "full_name": "Complete 8-Gene Hereditary Cardiomyopathy Atlas",
        "subtitle": (
            "MYH7·MYBPC3·TNNT2·PKP2·DSP·LMNA·TTN·RBM20 — "
            "320 patients (8×40, seeds 1102–1109)"
        ),
        "description": (
            "Comprehensive atlas of 8 major genetic hereditary cardiomyopathies encompassing: "
            "HCM1 (MYH7 — AD; 35-40% HCM; thick filament missense; apical HCM; MAVACAMTEN FDA 2022; "
            "ICD for SCD risk; septal reduction therapy for refractory HOCM); "
            "HCM4 (MYBPC3 — AD haploinsufficiency; 40-50% HCM worldwide; truncating ~60%; "
            "age-dependent penetrance 50% by 40y/95% by 60y; South Asian p.Arg502Trp founder; "
            "Mavacamten; ESC SCD risk score); "
            "HCM3/DCM1D (TNNT2 — AD; MALIGNANT — SCD risk DISPROPORTIONATE to LVH; "
            "ICD threshold LOWER; CMR LGE mandatory; Holter mandatory; mild LVH does NOT mean low risk); "
            "ARVC9 (PKP2 — AD; most common ARVC gene 40-50%; desmosomal LOF; epsilon waves PATHOGNOMONIC; "
            "LBBB-VT; SPORTS RESTRICTION MANDATORY even in phenotype-negative gene carriers; ICD for VT/VF); "
            "ARVC8/DCMEP (DSP — AD; biventricular ARVC HALLMARK; LGE subepicardial inferolateral DISTINCTIVE; "
            "ICD MANDATORY for DSP truncating; WOOLLY HAIR + PPK = Carvajal syndrome AR biallelic PATHOGNOMONIC; "
            "sports restriction MANDATORY); "
            "DCM1A/EDMD2 (LMNA — AD; nuclear lamina; AV BLOCK PATHOGNOMONIC; "
            "ICD MANDATORY — pacemaker alone INSUFFICIENT SCD risk 15-40%; EF decline 5%/year; "
            "transplant 20% by age 55; limb-girdle myopathy overlap); "
            "DCM1G (TTN — AD; largest human protein; A-BAND TTNtv PATHOGENIC — I-BAND = VUS; "
            "ClinGen curation MANDATORY; 25% familial DCM; 15% peripartum CM; incomplete penetrance); "
            "DCM1HH (RBM20 — AD; most aggressive genetic DCM; RS-domain hotspot; biventricular dilation; "
            "ICD MANDATORY; early transplant; Waddell-Smith families)."
        ),
        "total_patients": n,
        "genes_covered": len(CARDIOMYOPATHY_GENES),
        "patients_per_gene": 40,
        "seed_range": "1102–1109",
        "gene_list": [g["gene"] for g in CARDIOMYOPATHY_GENES],
        "disease_category_breakdown": {
            "HCM1 (AD MYH7; thick filament; missense; apical HCM; mavacamten; septal reduction; ICD SCD)": ["MYH7"],
            "HCM4 (AD MYBPC3; haploinsufficiency; truncating 60%; South Asian founder; age-dependent penetrance)": ["MYBPC3"],
            "HCM3/DCM1D (AD TNNT2; MALIGNANT — SCD disproportionate to LVH; ICD lower threshold; CMR mandatory)": ["TNNT2"],
            "ARVC9 (AD PKP2; desmosomal LOF; epsilon wave PATHOGNOMONIC; sports restriction MANDATORY; ICD)": ["PKP2"],
            "ARVC8 (AD DSP; biventricular ARVC; subepicardial LGE; ICD MANDATORY; Carvajal AR = woolly hair+PPK)": ["DSP"],
            "DCM1A/EDMD2 (AD LMNA; AV block PATHOGNOMONIC; ICD MANDATORY; pacemaker INSUFFICIENT; SCD 15-40%)": ["LMNA"],
            "DCM1G (AD TTN; A-band TTNtv PATHOGENIC; 25% familial DCM; 15% PPCM; ClinGen curation MANDATORY)": ["TTN"],
            "DCM1HH (AD RBM20; most aggressive DCM; RS-domain hotspot; biventricular; ICD; early transplant)": ["RBM20"],
        },
        "severity": {
            "mild_pct": round(100 * sev["Mild"] / n, 1),
            "moderate_pct": round(100 * sev["Moderate"] / n, 1),
            "severe_pct": round(100 * sev["Severe"] / n, 1),
        },
        "mean_onset_age_y": mean_onset,
        "mean_diagnosis_age_y": mean_dx,
        "kpis": [
            {"label": "Total Patients", "value": n, "color": "#37474f"},
            {"label": "Genes Covered", "value": len(CARDIOMYOPATHY_GENES), "color": "#1a237e"},
            {"label": "Patients/Gene", "value": 40, "color": "#4a148c"},
            {"label": "ICD Eligible", "value": f"{round(100 * icd_n / n, 1)}%", "color": "#b71c1c"},
            {"label": "High SCD Risk", "value": f"{round(100 * scd_n / n, 1)}%", "color": "#e65100"},
            {"label": "Arrhythmia", "value": f"{round(100 * arrhythmia_n / n, 1)}%", "color": "#880e4f"},
            {"label": "Cardiac Transplant", "value": f"{round(100 * transplant_n / n, 1)}%", "color": "#01579b"},
        ],
        "clinical_features_prevalence": gene_arrhythmia_pct,
        "severity_prevalence": {
            "ICD Eligible": round(100 * icd_n / n, 1),
            "Cardiac Transplant Required": round(100 * transplant_n / n, 1),
            "Arrhythmia (VT/VF/AF)": round(100 * arrhythmia_n / n, 1),
            "High SCD Risk": round(100 * scd_n / n, 1),
            "LVOTO (HCM)": round(100 * lvoto_n / n, 1),
            "HF Hospitalisation": round(100 * hf_hosp_n / n, 1),
            "Drug-Prescribing Error Detected": round(100 * drug_err_n / n, 1),
            "Disease Progression": round(100 * prog_n / n, 1),
        },
        "drug_alerts": [
            "MYH7/MYBPC3/TNNT2 HCM: DIHYDROPYRIDINE CCBs (amlodipine, nifedipine) CONTRAINDICATED "
            "in severe LVOTO — peripheral vasodilation worsens outflow tract gradient. "
            "MAVACAMTEN: HOLD if LVEF <55%; AVOID in pregnancy; CYP2C19 drug interactions require "
            "dose adjustment; LVEF monitoring echocardiography every 4 weeks during titration.",
            "TNNT2 HCM — CRITICAL PITFALL: DO NOT REASSURE BASED ON MILD LVH. "
            "SCD risk is DISPROPORTIONATE to wall thickness — a 13-15 mm TNNT2 HCM patient has HIGH SCD risk. "
            "CMR LGE MANDATORY. Holter MANDATORY. ICD threshold LOWER than other HCM genes. "
            "Normal echo does NOT exclude TNNT2-HCM SCD risk.",
            "PKP2/DSP ARVC: SPORTS RESTRICTION MANDATORY in ALL gene-positive individuals — "
            "even phenotype-negative gene carriers MUST stop competitive and endurance sport immediately. "
            "Exercise drives desmosomal stress → accelerated fibrofatty replacement → SCD. "
            "FLECAINIDE CONTRAINDICATED in ARVC (proarrhythmic in structural heart disease).",
            "DSP ARVC8: ICD MANDATORY for truncating DSP variants with ANY arrhythmia phenotype — "
            "malignant VT/VF can occur with PRESERVED LVEF. DO NOT WAIT for EF to drop. "
            "CARVAJAL SYNDROME CLUE: woolly hair + palmoplantar keratoderma + dilated CM = "
            "biallelic DSP — test immediately.",
            "LMNA DCM: PACEMAKER ALONE IS INSUFFICIENT — ICD capability MANDATORY. "
            "SCD from VF in LMNA occurs even when pacemaker is functioning normally. "
            "ICD MANDATORY when: LVEF <45% OR NSVT OR AV block ≥2nd degree OR syncope. "
            "NSVT in LMNA = ICD indication; do NOT wait for sustained VT.",
            "TTN DCM: A-BAND vs I-BAND TTNtv DISTINCTION MANDATORY before reporting pathogenic. "
            "I-band TTNtv = VUS/likely benign — reporting as pathogenic is a laboratory error. "
            "ClinGen TTNtv curation framework MANDATORY. "
            "PERIPARTUM CM: TTNtv in ~15% of PPCM — genetic testing mandatory for ALL PPCM patients.",
            "RBM20 DCM: MOST AGGRESSIVE GENETIC DCM — do not manage as routine DCM. "
            "ICD MANDATORY. Early transplant listing recommended. "
            "Annual monitoring INSUFFICIENT — use 3-6 monthly echo given rapid EF decline. "
            "BIVENTRICULAR DILATION + EARLY ONSET DCM in 30s-40s = RBM20 until proven otherwise.",
            "MYH7/MYBPC3 HCM + AF: ALL HCM patients with AF should receive anticoagulation "
            "REGARDLESS of CHA₂DS₂-VASc score — cardioembolic risk elevated inherently in HCM with AF.",
        ],
        "diagnostic_pearls": [
            "MYH7-HCM: apical HCM (ace-of-spades LV cavity); deep T-wave inversions V4-V6; "
            "specific 'malignant' variants (p.Arg403Gln, p.Arg719Gln) — very high SCD risk; "
            "ICD threshold lowered for these variants",
            "MYBPC3-HCM4: most common HCM gene worldwide; LATER ONSET (50-60y possible); "
            "South Asian p.Arg502Trp founder — always screen; age-dependent penetrance means "
            "annual echo up to age 60 in gene-positive relatives",
            "TNNT2-HCM3: MALIGNANT — SCD risk disproportionate to LVH; CMR LGE extensive "
            "despite mild hypertrophy; ICD even in mild HCM; Holter mandatory",
            "PKP2-ARVC9: epsilon waves V1-V3 PATHOGNOMONIC (~30-50% sensitivity); "
            "LBBB-VT (RV origin); sports restriction MANDATORY even gene-positive/phenotype-negative",
            "DSP-ARVC8: biventricular (LV+RV) involvement — LGE subepicardial inferolateral "
            "DISTINCTIVE; woolly hair + PPK = Carvajal AR biallelic immediately; ICD MANDATORY for truncating",
            "LMNA-DCM1A: AV block (PR prolongation → complete AV block) PATHOGNOMONIC; "
            "pacemaker alone INSUFFICIENT — ICD mandatory; EF decline ~5%/year",
            "TTN-DCM1G: A-band truncating = pathogenic; I-band = VUS — curation mandatory; "
            "15% PPCM have TTNtv; LGE on CMR = poor prognosis",
            "RBM20-DCM1HH: MOST AGGRESSIVE genetic DCM; biventricular dilation at 30s-40s; "
            "RS-domain hotspot (p.Arg634Gln/p.Arg636Ser); early transplant; ICD MANDATORY",
        ],
    }


def get_breakdown() -> list:
    all_patients = _gen_cohort()
    breakdown = []

    for i, gd in enumerate(CARDIOMYOPATHY_GENES):
        gene_pts = [p for p in all_patients if p["gene"] == gd["gene"]]
        n = len(gene_pts)
        sev_counts = {"Mild": 0, "Moderate": 0, "Severe": 0}
        for p in gene_pts:
            sev_counts[p["severity"]] += 1

        breakdown.append({
            "gene": gd["gene"],
            "protein": gd["protein"],
            "locus": gd["locus"],
            "aa": gd["aa"],
            "inheritance": gd["inheritance"].split("AUTOSOMAL")[0].strip() or "Autosomal Dominant",
            "disease_type": gd["disease_type"],
            "mechanism": gd["mechanism"],
            "phenotype": gd["phenotype"],
            "treatment_first_line": gd["first_line_drug"],
            "critical_avoid": gd["critical_avoid"],
            "key_features": gd["key_ddx"],
            "n_patients": n,
            "seed": SEED_BASE + i,
            "mean_onset_age_y": round(sum(p["onset_age_y"] for p in gene_pts) / n, 2),
            "mean_dx_age_y": round(sum(p["diagnosis_age_y"] for p in gene_pts) / n, 2),
            "severity_distribution": {
                "mild_pct": round(100 * sev_counts["Mild"] / n, 1),
                "moderate_pct": round(100 * sev_counts["Moderate"] / n, 1),
                "severe_pct": round(100 * sev_counts["Severe"] / n, 1),
            },
            "icd_eligible_pct": round(100 * sum(1 for p in gene_pts if p["icd_eligible"]) / n, 1),
            "cardiac_transplant_pct": round(100 * sum(1 for p in gene_pts if p["cardiac_transplant"]) / n, 1),
            "arrhythmia_pct": round(100 * sum(1 for p in gene_pts if p["arrhythmia"]) / n, 1),
            "scd_risk_high_pct": round(100 * sum(1 for p in gene_pts if p["scd_risk_high"]) / n, 1),
            "lvoto_pct": round(100 * sum(1 for p in gene_pts if p["lvoto"]) / n, 1),
            "hf_hospitalised_pct": round(100 * sum(1 for p in gene_pts if p["hf_hospitalised"]) / n, 1),
            "drug_error_pct": round(100 * sum(1 for p in gene_pts if p["drug_avoid_prescribed_error"]) / n, 1),
            "progression_pct": round(100 * sum(1 for p in gene_pts if p["disease_progression"]) / n, 1),
        })

    return breakdown


def get_definitions() -> list:
    return [
        {
            "term": "Hypertrophic Cardiomyopathy (HCM)",
            "definition": (
                "A hereditary cardiac muscle disease characterised by unexplained left ventricular hypertrophy "
                "(LVH ≥15 mm in adults, ≥13 mm with family history) NOT explained by loading conditions "
                "(hypertension, aortic stenosis, athlete's heart). "
                "GENETICS: >1,400 pathogenic variants in >20 genes; MYH7 (35-40%) and MYBPC3 (40-50%) "
                "account for ~75-80% of genotype-positive HCM. "
                "PATHOLOGY: myocyte hypertrophy + disarray + interstitial fibrosis; "
                "dynamic LVOTO in ~2/3 with SAM of mitral valve. "
                "PREVALENCE: ~1:500 in general population. "
                "SCD RISK: primary risk in young people — ICD for high-risk individuals; "
                "risk stratification: Maron 5-factor criteria or ESC HCM Risk-SCD score. "
                "TREATMENT: beta-blocker/verapamil for symptoms; mavacamten (FDA 2022) for HOCM; "
                "septal reduction therapy (myectomy/ASA) for refractory HOCM; ICD for SCD prevention."
            ),
            "importance": "Most common inherited cardiomyopathy; leading cause of SCD in young athletes; MYH7/MYBPC3 dominant; mavacamten first disease-specific therapy",
        },
        {
            "term": "Mavacamten (CAMZYOS) — FDA 2022 Cardiac Myosin Inhibitor",
            "definition": (
                "First-in-class allosteric cardiac myosin inhibitor — first new drug class for HCM. "
                "FDA APPROVED April 2022 for adults with symptomatic obstructive HCM (HOCM) NYHA II-III. "
                "MECHANISM: stabilises myosin in the super-relaxed state (SRX) → fewer active cross-bridges → "
                "reduced contractility → LVOTO gradient reduction. "
                "CLINICAL EFFECT (EXPLORER-HCM trial): 37% achieved ≥1 NYHA class improvement; "
                "significant reduction in resting LVOTO gradient; "
                "reduction in NT-proBNP and troponin I (fibrosis markers). "
                "DOSING: 2.5-15 mg daily; CYP2C19 genotype guides dosing (poor metabolisers need lower dose). "
                "MONITORING: echocardiography every 4 weeks during titration; HOLD if LVEF <55%. "
                "CONTRAINDICATIONS: pregnancy (embryotoxic); LVEF <55%; moderate/strong CYP2C19 inhibitors "
                "without dose adjustment. "
                "DRUG INTERACTIONS: fluconazole, omeprazole (high dose), fluvoxamine, ticlopidine "
                "inhibit CYP2C19 → mavacamten accumulation. "
                "Aficamten (next-generation cardiac myosin inhibitor): FDA 2024 — similar mechanism, "
                "shorter half-life, simpler monitoring."
            ),
            "importance": "First FDA-approved HCM-specific therapy; mandatory LVEF monitoring; CYP2C19 dosing; avoid in pregnancy",
        },
        {
            "term": "Arrhythmogenic Right Ventricular Cardiomyopathy (ARVC)",
            "definition": (
                "A hereditary cardiomyopathy characterised by fibrofatty replacement of RV myocardium → "
                "RV dilation + dysfunction + ventricular arrhythmias + SCD. "
                "GENETICS: predominantly desmosomal genes — PKP2 (40-50%), DSP, DSG2, DSC2, JUP; "
                "also non-desmosomal: PLN, LMNA (rarely). "
                "PATHOLOGY: cardiomyocyte death under mechanical stress → fibro-fatty replacement "
                "of RV free wall, apex, and RVOT ('triangle of dysplasia'). "
                "DIAGNOSIS: 2010 Revised Task Force Criteria — major/minor from 6 categories: "
                "structural/functional RV abnormalities; tissue characterisation (histology); "
                "repolarisation abnormalities (T-wave inversions V1-V4); "
                "depolarisation/conduction (epsilon waves, prolonged S-wave upstroke); "
                "arrhythmias (LBBB-morphology VT, frequent PVCs); family history/genetics. "
                "EPSILON WAVE: terminal notch after QRS in V1-V3 — PATHOGNOMONIC; present in ~30-50%. "
                "SPORTS RESTRICTION: MANDATORY — physical activity drives progression and precipitates SCD. "
                "TREATMENT: ICD; sotalol/amiodarone; catheter ablation; sports restriction; ACEi/ARB."
            ),
            "importance": "Leading cause of SCD in young athletes; epsilon wave pathognomonic; sports restriction mandatory even in gene-positive/phenotype-negative carriers",
        },
        {
            "term": "Epsilon Wave (ARVC — PKP2/DSP)",
            "definition": (
                "A low-amplitude terminal deflection (notch) occurring after the QRS complex in the "
                "right precordial leads V1-V3 — a PATHOGNOMONIC feature of ARVC. "
                "MECHANISM: slow conduction through fibrofatty-replaced RV myocardium → "
                "delayed RV depolarisation → small deflection after main QRS complex. "
                "DETECTION: requires high-quality ECG; Fontaine bipolar leads (sensitivity ~30% → ~70%); "
                "standard 12-lead sensitivity ~30-50% in confirmed ARVC. "
                "MAJOR CRITERIA: per 2010 Revised Task Force Criteria, epsilon wave = Major structural criterion. "
                "PITFALL: epsilon wave ABSENCE does not exclude ARVC — only present in minority; "
                "use CMR as primary structural investigation. "
                "DISTINCTION: not the same as an rSr' pattern (partial RBBB) or Brugada ECG pattern — "
                "epsilon wave is post-QRS, Brugada ST-elevation is in J-wave territory."
            ),
            "importance": "Pathognomonic for ARVC but sensitivity only 30-50%; absence does not exclude ARVC; Fontaine leads improve detection",
        },
        {
            "term": "LVOTO — Left Ventricular Outflow Tract Obstruction (HCM)",
            "definition": (
                "Dynamic obstruction of blood flow through the LV outflow tract caused by asymmetric "
                "septal hypertrophy + systolic anterior motion (SAM) of the anterior mitral leaflet "
                "engaging the hypertrophied septum → gradient across LVOT. "
                "MEASUREMENT: Doppler echocardiography; "
                "RESTING gradient: >30 mmHg = significant; measured at rest. "
                "PROVOCABLE gradient: >50 mmHg on Valsalva/exercise = obstructive HCM (HOCM). "
                "HAEMODYNAMIC CONSEQUENCE: LVOTO → LV pressure overload → reduced cardiac output → "
                "dyspnoea, syncope (particularly post-exercise — vasovagal worsened by obstructive physiology). "
                "WORSENING FACTORS: tachycardia (shorter diastole → less filling → smaller LV cavity → "
                "more obstruction); dehydration; vasodilation (dihydropyridine CCBs, nitrates CONTRAINDICATED); "
                "standing (reduced preload). "
                "IMPROVING FACTORS: beta-blocker (slows HR + negative inotropy); "
                "volume loading (squat, leg raise); verapamil (negative inotrope). "
                "TREATMENT: mavacamten (first-line pharmacological for HOCM); disopyramide; "
                "septal reduction therapy (myectomy/ASA) for refractory."
            ),
            "importance": "Dynamic obstruction in ~2/3 of symptomatic HCM; dihydropyridines contraindicated; mavacamten first-in-class therapy; septal reduction for refractory",
        },
        {
            "term": "AV Block in LMNA DCM — Pathognomonic Pattern",
            "definition": (
                "Progressive atrioventricular conduction disease is the PATHOGNOMONIC hallmark of "
                "LMNA-associated cardiomyopathy, distinguishing it from other genetic DCMs. "
                "PROGRESSION: PR prolongation (1st degree AV block) → Mobitz I (Wenckebach) → "
                "Mobitz II → 3rd degree (complete) AV block → pacemaker dependency. "
                "MECHANISM: lamin A/C deficiency → nuclear fragility in conduction tissue cardiomyocytes → "
                "AV node and His-Purkinje cell death → conduction block. "
                "FIRST MANIFESTATION: PR prolongation on ECG may be the FIRST sign of LMNA DCM — "
                "even before LV dilation — annual ECG mandatory in LMNA gene-positive individuals. "
                "MANAGEMENT: pacemaker for symptomatic AV block — BUT MUST BE ICD CAPABLE; "
                "pacemaker alone insufficient given VF risk (15-40% SCD lifetime). "
                "SCD MECHANISM: VF from massive ventricular fibrosis — NOT bradycardia-related; "
                "pacemaker prevents syncope but NOT ventricular fibrillation death. "
                "ICD: MANDATORY when LVEF <45% OR 2nd/3rd degree AV block OR NSVT OR syncope."
            ),
            "importance": "Pathognomonic for LMNA; pacemaker alone insufficient — ICD mandatory; SCD from VF not from bradycardia; annual ECG in LMNA gene-positive",
        },
        {
            "term": "TTNtv A-band vs I-band Distinction (ClinGen Mandatory)",
            "definition": (
                "Truncating variants in the titin gene (TTNtv — frameshift, nonsense, splice-site) "
                "must be classified by sarcomeric location before reporting as pathogenic for DCM. "
                "A-BAND TTNtv = PATHOGENIC for DCM: "
                "The A-band region of titin is constitutively expressed in all cardiac titin isoforms "
                "and forms the myosin-binding C-zone of the thick filament; truncation here "
                "disrupts titin function in all cardiomyocytes → haploinsufficiency → DCM. "
                "I-BAND TTNtv = LIKELY BENIGN or VUS: "
                "The I-band region contains exons that are differentially expressed/spliced across "
                "cardiac isoforms and tissues; I-band truncations are found in the general population "
                "without DCM at frequencies inconsistent with DCM prevalence. "
                "MANDATE: ClinGen TTNtv curation framework MUST be applied before any clinical report; "
                "tool: titin variant browser; cross-reference with gnomAD frequency. "
                "PERIPARTUM CM: 15% of PPCM patients carry A-band TTNtv — "
                "genetic testing mandatory for ALL women with PPCM. "
                "CLINICAL ERROR: reporting an I-band TTNtv as pathogenic for DCM is a laboratory error "
                "that can lead to unnecessary family cascade testing and incorrect risk stratification."
            ),
            "importance": "Critical distinction before reporting; I-band TTNtv = VUS (not pathogenic); ClinGen curation mandatory; 15% PPCM have A-band TTNtv",
        },
        {
            "term": "Carvajal Syndrome (Biallelic DSP) — Woolly Hair + PPK",
            "definition": (
                "Carvajal syndrome is an autosomal RECESSIVE form of desmoplakin-associated "
                "cardiomyopathy caused by biallelic (homozygous or compound heterozygous) "
                "DSP truncating variants. "
                "TRIAD (PATHOGNOMONIC): "
                "1. WOOLLY HAIR — tight curly hair present from birth (distinct from straight/wavy); "
                "2. PALMOPLANTAR KERATODERMA (PPK) — hyperkeratotic, thickened skin on palms and soles; "
                "3. SEVERE DILATED CARDIOMYOPATHY — left-dominant; early onset; aggressive. "
                "CUTANEOUS CLUE: woolly hair + PPK in a young patient with dilated CM = "
                "DSP biallelic IMMEDIATELY — do not miss this diagnostic opportunity. "
                "CONTRAST WITH MONOALLELIC DSP: heterozygous (monoallelic) DSP = ARVC8 (AD); "
                "no cutaneous features; biventricular ARVC. "
                "NAXOS DISEASE: related — biallelic JUP (plakoglobin) → woolly hair + PPK + ARVC "
                "(JUP allelic with Carvajal but RV-dominant). "
                "TREATMENT: aggressive cardiac therapy; early ICD; dermatology (emollients/keratolytics); "
                "early transplant listing."
            ),
            "importance": "Pathognomonic triad: woolly hair + PPK + DCM = biallelic DSP immediately; AR (unlike monoallelic DSP = AD ARVC8); do not miss cutaneous clues",
        },
        {
            "term": "RBM20 RS-Domain Hotspot — Most Aggressive Genetic DCM",
            "definition": (
                "RBM20 encodes a splicing factor for titin and other sarcomeric genes; "
                "pathogenic variants cluster in the arginine-serine (RS) domain (aa 634-638). "
                "HOTSPOT VARIANTS: p.Arg634Gln, p.Arg636Ser, p.Arg636His — all in RS domain. "
                "RS DOMAIN FUNCTION: nuclear localisation signal — "
                "wild-type RBM20 localises to nucleus where it performs splicing; "
                "RS-domain mutations → cytoplasmic mislocalisation → loss of nuclear splicing activity. "
                "DOMINANT NEGATIVE MECHANISM: mutant RBM20 sequesters mRNA in cytoplasmic "
                "granules, preventing splicing — more severe than simple haploinsufficiency. "
                "WADDELL-SMITH FAMILIES: Australasian founder families with p.Arg634Gln — "
                "severe DCM + biventricular dilation + high SCD + early transplant need. "
                "CLINICAL RECOGNITION: biventricular dilation + onset 30s-40s + severe LVEF reduction "
                "+ high arrhythmia burden = RBM20 until proven otherwise. "
                "PANEL COVERAGE: RBM20 is sometimes absent from older/shorter DCM panels — "
                "ensure comprehensive panel includes RBM20 in all early-onset severe DCM."
            ),
            "importance": "Most aggressive genetic DCM; RS-domain hotspot; dominant-negative; biventricular; early transplant; ensure gene panel covers RBM20",
        },
        {
            "term": "Sports Restriction in ARVC (PKP2/DSP) — Mandatory Gene-Positive",
            "definition": (
                "Physical activity restriction is the MOST IMPORTANT disease-modifying intervention "
                "in desmosomal cardiomyopathy (ARVC/ARVC8). "
                "EVIDENCE: ARVC patients who continue competitive/endurance sport develop: "
                "earlier phenotype expression; faster RV fibrofatty replacement; "
                "higher rate of appropriate ICD shocks; higher SCD risk vs sedentary gene carriers. "
                "SCOPE OF RESTRICTION: "
                "PROHIBITED: competitive sports (at any level); endurance sports (marathon, triathlon, "
                "cycling, swimming competitions, football, basketball, tennis tournaments). "
                "PERMITTED: low-intensity recreational activity — walking (≤5 km/h), gentle cycling "
                "(non-competitive), leisure swimming (non-competitive, non-cold). "
                "GENE-POSITIVE / PHENOTYPE-NEGATIVE CARRIERS: "
                "Sports restriction applies EVEN BEFORE phenotype develops — "
                "a PKP2 carrier with completely normal echo, CMR, and ECG MUST stop competitive sport. "
                "ATHLETE MANAGEMENT: competitive athletes identified as PKP2/DSP gene-positive "
                "should retire from competitive sport immediately — counsel sensitively but firmly. "
                "LEGAL/ETHICAL: document advice and patient agreement; arrange psychological support "
                "for athletes facing career-ending restriction."
            ),
            "importance": "Mandatory even in phenotype-negative gene carriers; exercise drives fibrofatty progression and SCD; applies to ALL PKP2 and DSP gene-positive individuals",
        },
        {
            "term": "ICD vs Pacemaker in LMNA DCM — Critical Distinction",
            "definition": (
                "LMNA DCM patients require ICD (implantable cardioverter-defibrillator) capability — "
                "pacemaker alone is INSUFFICIENT and leaves patients at risk of sudden cardiac death from VF. "
                "MECHANISM OF SCD IN LMNA: ventricular fibrillation (VF) from progressive ventricular fibrosis "
                "— NOT bradycardia or complete heart block alone. "
                "WHY PACEMAKER FAILS TO PROTECT: pacemaker prevents syncope from AV block but "
                "cannot detect or terminate VF — the patient with a pacemaker + VF dies from VF. "
                "ICD INDICATIONS in LMNA (MANDATORY when ANY of): "
                "LVEF <45%; NSVT on Holter (even 3-beat runs); 2nd or 3rd degree AV block; "
                "unexplained syncope; family SCD history in LMNA; "
                "extensive LGE on CMR. "
                "CRT-D: if LVEF <35% + LBBB + QRS >150 ms — resynchronisation + defibrillation. "
                "UPGRADE STRATEGY: LMNA patient with pacemaker-only who develops NSVT or EF drop → "
                "upgrade to ICD immediately; do not delay. "
                "LMNA ICD GUIDELINE: ESC 2023 HF guidelines and expert consensus — "
                "LMNA DCM has specific ICD indication (Class I) even with LVEF 35-45% if any risk factor present."
            ),
            "importance": "Pacemaker alone = insufficient in LMNA DCM; ICD mandatory; SCD from VF not bradycardia; upgrade any LMNA patient with pacemaker who develops NSVT",
        },
        {
            "term": "Septal Reduction Therapy (SRT) — Myectomy vs Alcohol Septal Ablation",
            "definition": (
                "Invasive treatment for severely symptomatic obstructive HCM (HOCM) refractory to "
                "maximal medical therapy, aimed at reducing LV outflow tract obstruction. "
                "ELIGIBILITY: NYHA class ≥III symptoms despite maximal medical therapy + "
                "resting LVOTO gradient ≥50 mmHg (or provoked ≥70 mmHg). "
                "SURGICAL MYECTOMY (Morrow procedure): "
                "GOLD STANDARD; cardiopulmonary bypass; resection of septal muscle through aortic incision; "
                "gradient abolition in >95% of experienced centres; "
                "mortality <1% at HCM Centres of Excellence. "
                "ALCOHOL SEPTAL ABLATION (ASA): "
                "Catheter-based; inject ethanol into first septal perforator artery → controlled septal infarct; "
                "gradient reduction in ~90%; suitable when surgical risk high (comorbidities, elderly); "
                "risk: complete AV block (3-20% requiring permanent pacemaker); VT from scar. "
                "CENTRE OF EXCELLENCE: SRT must be performed at HCM CoE (≥20 procedures/year); "
                "outcomes dramatically better at experienced centres. "
                "CHOICE: myectomy preferred for younger patients (<65y), significant MR, anomalous papillary "
                "muscle anatomy, AF requiring surgical ablation; ASA for older/high surgical risk."
            ),
            "importance": "For refractory HOCM (gradient >50 mmHg + NYHA ≥3 on max medical therapy); myectomy gold standard; ASA alternative; HCM CoE mandatory",
        },
    ]
