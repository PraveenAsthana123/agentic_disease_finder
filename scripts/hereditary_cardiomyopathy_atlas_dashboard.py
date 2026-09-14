"""Hereditary Cardiomyopathy Atlas — 8-Gene Reference
MYH7-MYBPC3-TNNT2-TNNI3-TPM1-ACTC1-MYL2-MYL3
320 patients (8 x 40), seeds 2590-2597.
Endpoints: /api/hereditary-cardiomyopathy-atlas/overview|breakdown|definitions
"""
import random

ATLAS_GENES = [
    {
        "gene": "MYH7",
        "protein": (
            "MYH7 -- 14q11.2 AD -- 1935aa -- Beta-Myosin-Heavy-Chain-223kDa-Sarcomere-"
            "Motor-Protein-HCM1-DCM1S-AD -- OMIM-Gene-160760-Disease-HCM1-192600"
        ),
        "locus": "14q11.2",
        "protein_size": "1935 aa / 223 kDa",
        "inheritance": (
            "AD dominant negative (poison polypeptide mechanism); "
            "HCM #1 gene (~35% of genotype-positive HCM); DCM, LVNC also caused by MYH7; "
            "missense variants predominate — mutant myosin incorporated into sarcomere, disrupts cross-bridge kinetics; "
            "p.Arg403Gln — first identified HCM mutation; MALIGNANT phenotype; high SCD risk; "
            "p.Arg719Trp, p.Arg663His — also high-risk; full penetrance; "
            "earlier onset (2nd-4th decade) and greater wall thickness than MYBPC3; "
            "MAVACAMTEN (Camzyos) FDA 2022 — first precision therapy for obstructive HCM"
        ),
        "disease_category": (
            "Hypertrophic cardiomyopathy (HCM) — MYH7 dominant negative; "
            "HCM: asymmetric septal hypertrophy (ASH) most common; LVOTO gradient >30 mmHg = obstructive; "
            "septal thickness typically >15 mm (ICD threshold consideration); "
            "LVNC (left ventricular non-compaction) also seen; "
            "Mavacamten reduces LVOTO gradient and improves NYHA class; "
            "DISOPYRAMIDE for refractory LVOTO; BETA-BLOCKER first-line; "
            "VASODILATORS (nitrates, PDE5 inhibitors) ABSOLUTELY CONTRAINDICATED in obstructive HCM"
        ),
        "disease_pathway": (
            "Beta-myosin heavy chain (MYH7) is the predominant motor protein of the adult ventricular sarcomere. "
            "Missense variants act via dominant negative poison-polypeptide mechanism: "
            "the mutant myosin is incorporated into the sarcomere at ~50% stoichiometry "
            "but disrupts cross-bridge cycling kinetics — increased ATP hydrolysis rate, "
            "prolonged actin attachment, impaired relaxation (diastolic dysfunction). "
            "Net result: hypercontractile sarcomere → myocyte disarray → hypertrophic remodelling → "
            "asymmetric septal hypertrophy → LVOTO (if septal hypertrophy narrows outflow tract). "
            "Mavacamten (allosteric myosin inhibitor) stabilises myosin in super-relaxed state (SRX), "
            "reducing basal ATP turnover and cross-bridge number — directly reverses the primary MYH7 pathomechanism. "
            "LGE on CMR correlates with fibrosis burden and SCD risk."
        ),
        "pathognomonic": (
            "ASYMMETRIC SEPTAL HYPERTROPHY (ASH) on echocardiography — septum:posterior wall ratio >1.3 = HCM; "
            "SYSTOLIC ANTERIOR MOTION (SAM) of mitral valve — creates dynamic LVOTO; "
            "LVOTO GRADIENT >30 mmHg at rest or >50 mmHg with Valsalva = significant obstruction; "
            "GIANT NEGATIVE T-WAVES on ECG — apical variant HCM (ACTC1 > MYH7); "
            "LATE GADOLINIUM ENHANCEMENT (LGE) on CMR — fibrosis; >15% LV mass = high SCD risk; "
            "MOLECULAR TESTING — confirms MYH7 pathogenic variant"
        ),
        "treatment": (
            "BETA-BLOCKERS (metoprolol, bisoprolol) — FIRST LINE symptomatic HCM; reduce HR → prolong diastolic filling; "
            "reduce LVOTO gradient; use titrated to resting HR 50-60 bpm; "
            "VERAPAMIL/DILTIAZEM — alternative if beta-blocker contraindicated; reduce LVOTO; "
            "DISOPYRAMIDE — negative inotrope; reduces LVOTO gradient; requires QTc monitoring (>500ms stop); "
            "anticholinergic side effects (dry mouth, urinary retention, blurred vision); "
            "MAVACAMTEN (Camzyos) — FDA 2022; cardiac myosin inhibitor; reduces LVOTO; approved NYHA II-III + LVOTO >30 mmHg; "
            "CI if EF<55%; CYP2C19 inhibitors increase exposure; REMS program mandatory; "
            "hold before septal reduction procedures; "
            "SEPTAL REDUCTION: SURGICAL MYECTOMY (Morrow procedure) — gold standard; preferred in young patients + concomitant mitral surgery; "
            "ALCOHOL SEPTAL ABLATION (ASA) — CI if conduction disease/LBBB; controversial <40 years; "
            "ICD — SCD prevention; ESC risk calculator ≥4% 5-year; implant if multiple risk factors; "
            "AVOID: nitrates, PDE5 inhibitors, CCB dihydropyridines (amlodipine/nifedipine in obstructive HCM); "
            "EXERCISE RESTRICTION — competitive sports CONTRAINDICATED"
        ),
        "key_features": [
            "MYH7 HCM: dominant negative mechanism — mutant myosin incorporated into sarcomere; full penetrance; earlier onset than MYBPC3",
            "p.Arg403Gln — first identified HCM mutation; MALIGNANT; high SCD risk despite standard ICD criteria",
            "Mavacamten (Camzyos) FDA 2022 — first precision cardiac myosin inhibitor; reduces LVOTO; CI if EF<55%; REMS mandatory",
            "LVOTO management: beta-blocker first → verapamil/disopyramide → mavacamten → septal reduction",
            "Vasodilators (nitrates, sildenafil, nifedipine) ABSOLUTELY CONTRAINDICATED in obstructive HCM — precipitate haemodynamic collapse",
            "Competitive sports CONTRAINDICATED in all HCM; moderate recreational exercise generally OK with cardiologist guidance",
            "Atrial fibrillation in HCM: 20-25% lifetime risk; anticoagulation MANDATORY (high stroke risk in HCM-AF)",
            "Genetic cascade testing: all first-degree relatives; negative gene test in known-variant family = reassuring",
        ],
        "key_ddx": [
            "Athlete's heart — symmetric hypertrophy; normal diastolic function; regresses with detraining; no SAM/LVOTO",
            "Hypertensive heart disease — wall thickness <15 mm usually; posterior wall involved; controlled BP reduces",
            "Fabry disease (GLA) — lysosomal storage; hypertrophic pattern; low alpha-galactosidase A; skin lesions",
            "MYBPC3 HCM — haploinsufficiency; incomplete penetrance; later onset; generally more benign than MYH7",
        ],
        "hcm_pct": 35,
        "lvoto_pct": 65,
        "scd_risk_pct": 8,
        "icd_pct": 30,
        "mavacamten_eligible_pct": 40,
        "af_pct": 22,
        "septal_reduction_pct": 15,
    },
    {
        "gene": "MYBPC3",
        "protein": (
            "MYBPC3 -- 11p11.2 AD -- 1274aa -- Myosin-Binding-Protein-C3-150kDa-Thick-"
            "Filament-HCM4-Haploinsufficiency-AD -- OMIM-Gene-600958-Disease-HCM4-115197"
        ),
        "locus": "11p11.2",
        "protein_size": "1274 aa / 150 kDa",
        "inheritance": (
            "AD haploinsufficiency (truncating variants predominate — nonsense, frameshift, splice-site); "
            "HCM #1 gene (tied with MYH7, ~35% of genotype-positive HCM); DCM, LVNC also; "
            "INCOMPLETE PENETRANCE — 40-50% by age 50; many elderly carriers identified only by cardiac MRI LGE; "
            "South Asian FOUNDER: c.3736+1G>A — prevalence 1-in-500 Indian subcontinent; "
            "truncating variants → mRNA nonsense-mediated decay → 50% cMyBP-C reduction; "
            "phenotype age-dependent — rescreen every 3-5 years in gene-positive phenotype-negative carriers"
        ),
        "disease_category": (
            "Hypertrophic cardiomyopathy (HCM) — MYBPC3 haploinsufficiency; "
            "most common HCM gene (35-40%); incomplete penetrance a key feature vs MYH7; "
            "LVOTO in ~70% of HCM (septal hypertrophy + SAM); "
            "Mavacamten reduces LVOTO; beta-blocker first-line; "
            "South Asian c.3736+1G>A founder — screen ALL South Asian HCM patients for this variant; "
            "AAV9-MYBPC3 gene replacement therapy in Phase I/II trials"
        ),
        "disease_pathway": (
            "cMyBP-C (cardiac myosin-binding protein C) is located in the C-zone of the sarcomere A-band. "
            "Its N-terminal domains (C0-C2) interact with myosin S2 and actin, providing a mechanosensitive "
            "brake on cross-bridge cycling — phosphorylation by PKA/CaMKII releases this brake during sympathetic activation. "
            "MYBPC3 haploinsufficiency (50% reduction) removes this brake at baseline → "
            "hypercontractile sarcomere → myocyte disarray → hypertrophic remodelling. "
            "Unlike MYH7, the mechanism is haploinsufficiency (not poison polypeptide); "
            "truncating variants are degraded by NMD, so the sarcomere runs with only 50% cMyBP-C. "
            "Penetrance is age-dependent and influenced by modifier genes and lifestyle factors — "
            "explains why many elderly MYBPC3 carriers have only subclinical hypertrophy (LGE without wall thickening)."
        ),
        "pathognomonic": (
            "ASYMMETRIC SEPTAL HYPERTROPHY on echocardiography — septum:posterior wall ratio >1.3; "
            "INCOMPLETE PENETRANCE — normal echo does NOT exclude future HCM in MYBPC3; rescreen every 3-5 years; "
            "SOUTH ASIAN FOUNDER c.3736+1G>A — pathognomonic in Indian subcontinent origin; "
            "LGE ON CMR — even without wall thickening; >15% LV mass LGE = high SCD risk; "
            "TRUNCATING VARIANT ON GENETIC PANEL — MYBPC3 frameshift/nonsense/splice = likely pathogenic; "
            "HAPLOINSUFFICIENCY MECHANISM — NMD clears mutant transcript; sarcomere protein quantification"
        ),
        "treatment": (
            "Same pharmacological approach as MYH7-HCM: "
            "BETA-BLOCKERS — first-line symptomatic; "
            "VERAPAMIL — alternative; "
            "DISOPYRAMIDE — refractory LVOTO; QTc monitoring mandatory; "
            "MAVACAMTEN (Camzyos) FDA 2022 — reduces LVOTO; NYHA II-III + LVOTO >30 mmHg; CI if EF<55%; "
            "CYP2C19 interactions; REMS program; hold before septal reduction; "
            "SEPTAL REDUCTION: MYECTOMY or ALCOHOL SEPTAL ABLATION; "
            "ICD — per HCM SCD risk score (ESC ≥4% 5-year) or Mayo criteria; "
            "SURVEILLANCE: gene-positive phenotype-negative → echo + CMR every 3-5 years; "
            "South Asian c.3736+1G>A carriers — targeted cascade screening programme; "
            "EXERCISE RESTRICTION — competitive sports CONTRAINDICATED; "
            "AF management: anticoagulation mandatory (high stroke risk)"
        ),
        "key_features": [
            "MYBPC3: most common HCM gene (35-40%); haploinsufficiency NOT dominant negative (unlike MYH7); truncating variants predominate",
            "Incomplete penetrance 40-50% by age 50 — normal echo at 30 does NOT exclude future HCM; rescreen every 3-5 years",
            "South Asian founder c.3736+1G>A — 1-in-500 Indian subcontinent; screen ALL South Asian HCM patients",
            "Mavacamten FDA 2022 — equally effective for MYBPC3 and MYH7 obstructive HCM; REMS programme mandatory",
            "AAV9-MYBPC3 gene replacement therapy — Phase I/II clinical trials; first gene therapy targeting HCM pathomechanism",
            "LGE on CMR without wall thickening — subclinical MYBPC3-HCM; LGE >15% LV mass = high SCD risk",
            "Cascade genetic testing mandatory — 50% inheritance risk; gene-positive/phenotype-negative: surveillance programme",
            "Competitive sports CONTRAINDICATED; moderate recreational exercise permitted with shared decision-making",
        ],
        "key_ddx": [
            "MYH7 HCM — dominant negative; full penetrance; earlier onset; higher SCD risk per episode; thicker walls",
            "Fabry disease (GLA XLR) — lysosomal storage; short PR interval; low alpha-Gal A; renal/skin involvement",
            "TNNT2 HCM — MALIGNANT SCD risk despite MINIMAL hypertrophy; ICD threshold LOWER; cardiac MRI mandatory",
            "Cardiac amyloidosis — diffuse concentric hypertrophy; low voltage ECG; apple-green birefringence on biopsy",
        ],
        "hcm_pct": 35,
        "lvoto_pct": 60,
        "scd_risk_pct": 5,
        "icd_pct": 25,
        "mavacamten_eligible_pct": 38,
        "af_pct": 20,
        "septal_reduction_pct": 12,
    },
    {
        "gene": "TNNT2",
        "protein": (
            "TNNT2 -- 1q32.1 AD -- 298aa -- Cardiac-Troponin-T2-36kDa-Thin-Filament-"
            "Regulatory-Complex-HCM2-DCM1D-AD -- OMIM-Gene-191045-Disease-HCM2-115195"
        ),
        "locus": "1q32.1",
        "protein_size": "298 aa / 36 kDa",
        "inheritance": (
            "AD; HCM (~5% of genotype-positive); DCM (>15%); LVNC; "
            "MALIGNANT RISK: HIGH SCD despite MILD or NO hypertrophy — phenotype-negative sudden death; "
            "p.Arg92Trp — most studied malignant TNNT2 variant; high penetrance but variable expression; "
            "ICD THRESHOLD LOWER than for MYH7/MYBPC3 — standard HCM risk calculator may UNDERESTIMATE risk; "
            "family history of SCD in TNNT2 family = HIGH RISK; "
            "CMR (cardiac MRI) mandatory — LGE may be present without echo hypertrophy"
        ),
        "disease_category": (
            "Hypertrophic cardiomyopathy (HCM) / dilated cardiomyopathy (DCM) — TNNT2 thin filament; "
            "CRITICAL: TNNT2-HCM is MALIGNANT — high SCD risk despite minimal/no hypertrophy; "
            "standard echo criteria may be falsely reassuring; CMR with LGE mandatory for risk stratification; "
            "ICD implantation threshold LOWER than standard HCM; "
            "family history SCD in TNNT2 kindred = immediate high-risk classification; "
            "TNNT2-DCM: thin filament haploinsufficiency → dilated phenotype"
        ),
        "disease_pathway": (
            "Cardiac troponin T (cTnT, TNNT2) is a component of the cardiac troponin regulatory complex (cTnI-cTnT-cTnC). "
            "cTnT links the troponin complex to tropomyosin on the thin actin filament. "
            "TNNT2 HCM mutations alter Ca2+ sensitivity of actomyosin ATPase: "
            "increased Ca2+ sensitivity → hypercontractility → sarcomere dysfunction → hypertrophy. "
            "The paradox of TNNT2-HCM: MARKED ELECTRICAL INSTABILITY (fibrosis, dispersion of repolarisation) "
            "develops disproportionately to MECHANICAL HYPERTROPHY — "
            "fibrous replacement occurs at myocyte level without gross wall thickening → "
            "LGE on CMR detects this fibrosis BEFORE echo shows hypertrophy → "
            "SCD risk is paradoxically HIGH in patients with 'mild' HCM by echo criteria. "
            "TNNT2-DCM: haploinsufficiency/LOF → thin filament weakness → dilated, hypocontractile phenotype."
        ),
        "pathognomonic": (
            "MINIMAL OR NO HYPERTROPHY ON ECHO DESPITE HIGH SCD RISK — most critical and dangerous TNNT2 feature; "
            "LGE ON CMR without echo wall thickening — fibrosis precedes hypertrophy in TNNT2; "
            "FAMILY HISTORY OF SUDDEN CARDIAC DEATH especially in young relatives — key TNNT2 red flag; "
            "p.Arg92Trp VARIANT — pathognomonic malignant TNNT2 mutation; "
            "TNNT2 GENETIC TEST POSITIVE — immediately triggers CMR + Holter + low-threshold ICD discussion; "
            "DUAL PHENOTYPE: HCM vs DCM from same TNNT2 gene — variant class determines phenotype"
        ),
        "treatment": (
            "BETA-BLOCKERS — first-line symptomatic; critical for rate control in arrhythmia; "
            "ICD — LOWER threshold than standard HCM; "
            "family SCD history = immediate high-risk; do NOT wait for wall thickness criteria; "
            "CMR MANDATORY in all TNNT2 carriers — LGE guides ICD decision more than echo; "
            "MAVACAMTEN — applicable if obstructive TNNT2-HCM (rare in TNNT2); "
            "TNNT2-DCM: standard heart failure therapy (ACE-I/ARB/ARNI, beta-blocker, MRA); "
            "ICD for primary prevention in DCM if LVEF <35%; "
            "GENETIC CASCADE: first-degree relatives must have genetic testing + CMR; "
            "EXERCISE RESTRICTION — competitive sports CONTRAINDICATED; all TNNT2 gene-positive; "
            "AF management: anticoagulation mandatory"
        ),
        "key_features": [
            "TNNT2 HCM is MALIGNANT — SCD risk disproportionate to hypertrophy; normal echo does NOT exclude high SCD risk",
            "CMR with LGE mandatory — fibrosis detected BEFORE wall thickening in TNNT2; LGE pattern guides ICD decision",
            "ICD threshold LOWER than for MYH7/MYBPC3; family SCD history = immediate high-risk, do NOT apply standard criteria",
            "p.Arg92Trp — most characterised malignant TNNT2 variant; multiple family SCD events documented",
            "Dual phenotype: HCM (thin filament gain-of-function) vs DCM (thin filament LOF) from SAME gene",
            "TNNT2-DCM >15% of familial DCM — screened in all familial DCM genetic panels",
            "Gene-positive family members require CMR even with normal echo — subclinical fibrosis without hypertrophy",
            "Mavacamten less applicable in TNNT2-HCM (minimal LVOTO common in this subtype)",
        ],
        "key_ddx": [
            "MYH7 HCM — significant wall thickening; dominant negative; SCD risk better calibrated to wall thickness",
            "Lamin A/C DCM (LMNA) — conduction disease precedes DCM; Padua score ≥4 = ICD regardless LVEF",
            "ARVC (PKP2/DSP) — right ventricular fibrofatty; epsilon wave; exercise drives progression; different distribution",
            "Cardiac sarcoidosis — granulomatous; bilateral hilar adenopathy; AV block; CMR patchy LGE pattern different",
        ],
        "hcm_pct": 5,
        "lvoto_pct": 25,
        "scd_risk_pct": 12,
        "icd_pct": 45,
        "mavacamten_eligible_pct": 15,
        "af_pct": 18,
        "septal_reduction_pct": 5,
    },
    {
        "gene": "TNNI3",
        "protein": (
            "TNNI3 -- 19q13.42 AD -- 210aa -- Cardiac-Troponin-I3-24kDa-Thin-Filament-"
            "Inhibitory-Subunit-HCM7-RCM-DCM-AD-AR -- OMIM-Gene-191044-Disease-HCM7-613690"
        ),
        "locus": "19q13.42",
        "protein_size": "210 aa / 24 kDa",
        "inheritance": (
            "AD (HCM, ~5%; DCM); AR biallelic LOF → Restrictive Cardiomyopathy (RCM) — most severe phenotype; "
            "TNNI3-RCM (biallelic): INFANTILE onset; non-dilated, non-hypertrophied, severely stiff ventricle; "
            "Key DDx: RCM vs constrictive pericarditis — mandatory catheter haemodynamics; "
            "dip-and-plateau (square root sign) in RCM; pressure equalisation in constrictive pericarditis; "
            "TNNI3-RCM: pulmonary hypertension, heart failure, transplant often needed; "
            "AR mechanism: biallelic loss of inhibitory troponin I → constitutive actomyosin activation"
        ),
        "disease_category": (
            "Hypertrophic cardiomyopathy (HCM) — TNNI3 AD; "
            "Restrictive cardiomyopathy (RCM) — TNNI3 biallelic AR — MOST SEVERE; "
            "TNNI3 encodes cardiac troponin I, the inhibitory subunit of the troponin complex; "
            "cTnI inhibits actomyosin ATPase at low Ca2+ — lost in AR-RCM → constitutive activation; "
            "RCM: biatrial dilatation, small/normal ventricular volumes, severely elevated filling pressures; "
            "Key DDx: RCM vs constrictive pericarditis"
        ),
        "disease_pathway": (
            "Cardiac troponin I (cTnI, TNNI3) is the inhibitory subunit of the cardiac troponin complex. "
            "At low intracellular Ca2+ (diastole), cTnI inhibits actin-myosin interaction by binding to actin "
            "and stabilising tropomyosin in the blocked state. "
            "AD TNNI3 HCM mutations (GOF, increased Ca2+ sensitivity): "
            "heightened sensitivity to Ca2+ → hypercontractility at lower Ca2+ concentrations → "
            "prolonged relaxation failure → diastolic dysfunction → hypertrophic remodelling. "
            "AR TNNI3 LOF (biallelic): complete loss of inhibitory function → "
            "constitutive actomyosin activation regardless of Ca2+ → "
            "RCM phenotype: severely impaired myocardial relaxation, non-dilated stiff ventricles, "
            "massively elevated filling pressures → biatrial enlargement → pulmonary hypertension."
        ),
        "pathognomonic": (
            "RCM PATTERN ON ECHO: biatrial enlargement, normal/small ventricular volumes, "
            "severely elevated filling pressures (E/e' >15 typically >25), restrictive mitral inflow; "
            "BIALLELIC TNNI3 VARIANTS — AR compound heterozygous or homozygous LOF = RCM phenotype; "
            "DIP-AND-PLATEAU (SQUARE ROOT SIGN) on ventricular pressure tracing (RHC) — "
            "distinguishes RCM from constrictive pericarditis (which shows pressure equalisation); "
            "INFANTILE PRESENTATION of RCM — when biallelic TNNI3, consider genetic testing immediately; "
            "TNNI3 RCM: NO wall thickening, NO LV dilatation — diagnosis requires invasive haemodynamics"
        ),
        "treatment": (
            "TNNI3-HCM (AD): same as standard HCM pharmacotherapy; "
            "BETA-BLOCKERS — first-line; "
            "MAVACAMTEN if obstructive; "
            "ICD per standard HCM SCD risk score; "
            "TNNI3-RCM (AR biallelic): "
            "DIURETICS — cautious use for elevated filling pressures; "
            "AVOID aggressive diuresis (preload dependent); "
            "CALCIUM CHANNEL BLOCKERS (verapamil) — limited evidence; "
            "ANTICOAGULATION — high AF risk; thrombus risk in enlarged atria; "
            "CARDIAC TRANSPLANTATION — definitive treatment for end-stage TNNI3-RCM; "
            "PRIMARY PREVENTION ICD if sustained VT/VF documented; "
            "INVASIVE HAEMODYNAMICS (RHC) — mandatory to distinguish RCM from constrictive pericarditis; "
            "GENETIC CASCADE — all first-degree relatives; "
            "PULMONARY HYPERTENSION management if secondary PH develops"
        ),
        "key_features": [
            "TNNI3 biallelic AR → RCM (infantile, severe) — most severe TNNI3 phenotype; transplant often needed",
            "RCM vs constrictive pericarditis DDx — MANDATORY invasive RHC; dip-and-plateau vs pressure equalisation",
            "AR TNNI3 LOF: biallelic loss of inhibitory cTnI → constitutive actomyosin activation → stiff ventricle",
            "AD TNNI3 HCM (~5%): standard HCM pharmacotherapy; ICD per standard risk score",
            "TNNI3-RCM: NO hypertrophy, NO dilatation — echo-normal ventricular size but massive diastolic dysfunction",
            "Biatrial enlargement: hallmark of RCM from any cause; thrombus risk; anticoagulate",
            "Infantile RCM presentation: check TNNI3 biallelic before any other diagnosis",
            "Pericardiectomy for constrictive pericarditis is curative — NEVER do this in RCM (will not help)",
        ],
        "key_ddx": [
            "Constrictive pericarditis — RHC: pressure equalisation; pericardial calcification on CT; pericardiectomy curative",
            "Cardiac amyloidosis (TTR/AL) — sparkling myocardium; low voltage ECG; Apple-Green birefringence biopsy",
            "TNNT2 HCM — thin filament but different phenotype; no biallelic RCM pattern; SCD without hypertrophy",
            "MYH7 HCM — dominant negative thick filament; significant wall thickening; LVOTO common",
        ],
        "hcm_pct": 5,
        "lvoto_pct": 20,
        "scd_risk_pct": 7,
        "icd_pct": 30,
        "mavacamten_eligible_pct": 12,
        "af_pct": 30,
        "septal_reduction_pct": 4,
    },
    {
        "gene": "TPM1",
        "protein": (
            "TPM1 -- 15q22.2 AD -- 284aa -- Tropomyosin-Alpha-1-Chain-33kDa-Thin-Filament-"
            "Regulatory-Protein-HCM3-DCM1Y-LVNC-AD -- OMIM-Gene-191010-Disease-HCM3-115196"
        ),
        "locus": "15q22.2",
        "protein_size": "284 aa / 33 kDa",
        "inheritance": (
            "AD; HCM (~2% of genotype-positive); DCM (~5%); LVNC; "
            "SAME GENE → DIFFERENT PHENOTYPES: GOF variants → HCM; LOF variants → DCM; "
            "CALCIUM SENSITISATION MECHANISM: TPM1 HCM variants increase Ca2+ sensitivity of actomyosin; "
            "tropomyosin cable traverses thin filament regulating myosin head access; "
            "variable penetrance; prognosis intermediate between MYH7 and MYBPC3; "
            "TPM1-DCM: tropomyosin destabilisation → sarcomere weakness → dilated phenotype"
        ),
        "disease_category": (
            "Hypertrophic cardiomyopathy (HCM) — TPM1 calcium sensitisation; "
            "Dilated cardiomyopathy (DCM) — TPM1 tropomyosin LOF; "
            "LVNC (left ventricular non-compaction); "
            "Same gene causes HCM OR DCM depending on variant class — critical genetic counselling point; "
            "TPM1 pathogenic variants are rare — require functional validation for VUS classification; "
            "Thin filament mechanism — calcium sensitisation distinguishes from thick filament MYH7/MYBPC3"
        ),
        "disease_pathway": (
            "Tropomyosin alpha-1 (TPM1) forms a coiled-coil cable that runs along the thin actin filament, "
            "blocking or exposing myosin-binding sites in a Ca2+-regulated manner. "
            "At low Ca2+ (diastole): tropomyosin blocks myosin access (blocked state). "
            "At high Ca2+ (systole): cTnI releases, tropomyosin shifts → open state → cross-bridge cycling. "
            "TPM1 HCM mutations: destabilise the blocked state → myosin accesses actin at lower Ca2+ → "
            "CALCIUM SENSITISATION → hypercontractility → diastolic dysfunction → hypertrophy. "
            "TPM1 DCM mutations: destabilise the open state OR reduce tropomyosin stability → "
            "impaired force generation → dilated, hypocontractile phenotype. "
            "The same protein can flip phenotype based on the molecular mechanism of the mutation — "
            "GOF (sensitisation) → HCM; LOF (weakness) → DCM."
        ),
        "pathognomonic": (
            "ECHOCARDIOGRAPHIC HCM PATTERN or DCM PATTERN — depends on variant class; "
            "SAME GENE DUAL PHENOTYPE — critical distinguishing feature; variant class determines outcome; "
            "CALCIUM SENSITISATION ASSAY — functional assay shows increased Ca2+ sensitivity in TPM1-HCM variants; "
            "GENETIC PANEL POSITIVE — TPM1 pathogenic variant; "
            "VUS IN TPM1 — requires functional validation; in-vitro calcium sensitisation assay useful; "
            "LVNC PATTERN ON CMR — non-compaction ratio >2.3 in diastole = LVNC criterion"
        ),
        "treatment": (
            "TPM1-HCM: same pharmacotherapy as other HCM subtypes; "
            "BETA-BLOCKERS — first-line; "
            "VERAPAMIL — alternative; "
            "MAVACAMTEN if obstructive HCM (LVOTO >30 mmHg + NYHA II-III); "
            "ICD — per HCM SCD risk score; "
            "EXERCISE RESTRICTION — competitive sports CONTRAINDICATED; "
            "TPM1-DCM: standard heart failure therapy; "
            "ACE-I/ARB/ARNI + beta-blocker + MRA + SGLT2 inhibitor; "
            "ICD primary prevention if LVEF <35%; "
            "LVNC-TPM1: anticoagulation for thromboembolic risk if LVEF <35% or LV thrombus; "
            "GENETIC CASCADE — first-degree relatives; "
            "VARIANT CLASSIFICATION: functional assay for VUS before counselling family members"
        ),
        "key_features": [
            "TPM1: same gene → HCM (GOF calcium sensitisation) OR DCM (LOF tropomyosin weakness) — variant-specific phenotype",
            "Calcium sensitisation mechanism: TPM1-HCM increases myosin access at lower Ca2+ — differs from thick filament MYH7 mechanism",
            "LVNC associated — left ventricular non-compaction; anticoagulate if LVEF <35%; thromboembolic risk",
            "VUS classification in TPM1 requires functional calcium sensitisation assay — not just computational prediction",
            "TPM1 HCM prognosis intermediate between MYH7 (more severe) and MYBPC3 (more benign); variable penetrance",
            "TPM1-DCM: tropomyosin destabilisation → heart failure; treat as standard HCM-DCM overlap",
            "Competitive sports CONTRAINDICATED regardless of phenotype (HCM or DCM or LVNC)",
            "Genetic cascade: 50% inheritance risk; inform family that SAME variant can give different phenotypes",
        ],
        "key_ddx": [
            "MYH7 HCM — thick filament; dominant negative; no DCM phenotype from same gene",
            "MYBPC3 HCM — thick filament haploinsufficiency; incomplete penetrance; no calcium sensitisation",
            "TNNT2 HCM/DCM — troponin T; MALIGNANT SCD without hypertrophy; different thin filament component",
            "ACTC1 HCM — alpha-cardiac actin; apical HCM variant; giant negative T-waves pathognomonic",
        ],
        "hcm_pct": 2,
        "lvoto_pct": 30,
        "scd_risk_pct": 5,
        "icd_pct": 20,
        "mavacamten_eligible_pct": 18,
        "af_pct": 15,
        "septal_reduction_pct": 6,
    },
    {
        "gene": "ACTC1",
        "protein": (
            "ACTC1 -- 15q14 AD -- 375aa -- Alpha-Cardiac-Actin-42kDa-Thin-Filament-"
            "Core-Polymer-HCM11-DCM1R-LVNC-AD -- OMIM-Gene-102540-Disease-HCM11-612098"
        ),
        "locus": "15q14",
        "protein_size": "375 aa / 42 kDa",
        "inheritance": (
            "AD; HCM (~1% of genotype-positive); DCM; LVNC; "
            "ACTC1-HCM: APICAL VARIANT PROMINENT — differs from septal distribution of MYH7/MYBPC3; "
            "apical HCM: hypertrophy predominantly at LV apex; LVOTO RARE in apical HCM; "
            "GIANT NEGATIVE T-WAVES on ECG — 12-lead ECG finding pathognomonic of apical HCM; "
            "V4-V6 deep inverted T-waves (>10 mm depth); "
            "LVNC pattern frequently co-exists; "
            "DCM from ACTC1 LOF/dominant negative → thin filament weakness"
        ),
        "disease_category": (
            "Hypertrophic cardiomyopathy (HCM) — ACTC1 apical predominant; "
            "Dilated cardiomyopathy (DCM); "
            "Left ventricular non-compaction (LVNC); "
            "ACTC1-HCM: apical form with giant negative T-waves on ECG (V4-V6 deeply inverted, >10 mm); "
            "LVOTO RARE (apical hypertrophy does not obstruct outflow tract); "
            "Spade-shaped LV cavity on ventriculography/CMR — pathognomonic of apical HCM; "
            "Japanese descent: apical HCM more common in Japanese population"
        ),
        "disease_pathway": (
            "Alpha-cardiac actin (ACTC1) is the core structural polymer of the cardiac thin filament. "
            "ACTC1 variants affect the actin-myosin interface, the actin-tropomyosin binding sites, "
            "or actin polymerisation stability. "
            "HCM ACTC1 variants (at myosin S1 interface): impair cross-bridge kinetics → "
            "hypercontractility pathway → hypertrophic remodelling → APICAL PREDOMINANT distribution "
            "(unclear why apical; may relate to regional wall stress differences in the apex). "
            "Apical HCM: hypertrophy confined to LV apex → 'spade-shaped' LV cavity on angiography/CMR; "
            "no LVOTO (apex not in outflow tract); "
            "giant negative T-waves (V4-V6 ≥10 mm) on ECG = most sensitive clinical marker; "
            "apical thrombus risk from apical akinesis/aneurysm — anticoagulate if present; "
            "ACTC1 DCM: actin structural weakness → thin filament collapse → dilated phenotype."
        ),
        "pathognomonic": (
            "GIANT NEGATIVE T-WAVES on 12-lead ECG (V4-V6, ≥10 mm depth) — PATHOGNOMONIC of apical HCM; "
            "SPADE-SHAPED LV CAVITY on left ventriculography or CMR — obliterates at systole at apex = apical HCM; "
            "APICAL HYPERTROPHY ON CMR — hypertrophy confined to apex; wall thickness ≥15 mm at apex; "
            "LVOTO ABSENT — apical HCM has no dynamic obstruction (differs from septal HCM); "
            "APICAL ANEURYSM or THROMBUS — long-term complication; anticoagulate if present; "
            "ACTC1 GENETIC VARIANT — confirms molecular diagnosis"
        ),
        "treatment": (
            "ACTC1-HCM (apical): "
            "BETA-BLOCKERS — first-line for symptoms (palpitations, exertional dyspnoea); "
            "VERAPAMIL — alternative; "
            "MAVACAMTEN — LESS applicable (minimal LVOTO in apical HCM); "
            "ICD — per standard HCM SCD risk score; apical aneurysm = additional risk factor; "
            "ANTICOAGULATION — if apical aneurysm, apical thrombus, or AF; "
            "SEPTAL REDUCTION — NOT indicated (no LVOTO in apical HCM); "
            "EXERCISE RESTRICTION — competitive sports CONTRAINDICATED; "
            "ACTC1-DCM: ACE-I/ARB/ARNI + beta-blocker + MRA + SGLT2 inhibitor; "
            "ICD if LVEF <35%; "
            "GENETIC CASCADE — first-degree relatives"
        ),
        "key_features": [
            "ACTC1-HCM: apical variant PROMINENT — hypertrophy at LV apex NOT septum; LVOTO rare (differs from MYH7/MYBPC3)",
            "Giant negative T-waves (V4-V6, ≥10 mm) on ECG — PATHOGNOMONIC of apical HCM; screen with 12-lead ECG",
            "Spade-shaped LV cavity on CMR/ventriculography — obliterates at systole; diagnostic of apical HCM",
            "Apical aneurysm risk: long-standing apical HCM → apical thinning → aneurysm → thrombus → anticoagulate",
            "Japanese descent: apical HCM more common than septal HCM in Japanese population",
            "Mavacamten LESS applicable — no significant LVOTO in apical HCM (apical location = no outflow obstruction)",
            "ACTC1 DCM: same gene, different phenotype; LOF → thin filament weakness → dilated cardiomyopathy",
            "Competitive sports CONTRAINDICATED; apical aneurysm is additional sports restriction criterion",
        ],
        "key_ddx": [
            "Takotsubo cardiomyopathy — transient apical ballooning; stress trigger; female predominance; reversible",
            "MYH7 HCM (septal) — asymmetric septal hypertrophy; SAM; LVOTO present; T-waves less marked",
            "Cardiac sarcoidosis — patchy LGE different distribution; AV block; granulomas; systemic involvement",
            "Apical LV thrombus from ischaemic cardiomyopathy — coronary artery disease; CAD on angiogram",
        ],
        "hcm_pct": 1,
        "lvoto_pct": 10,
        "scd_risk_pct": 4,
        "icd_pct": 18,
        "mavacamten_eligible_pct": 6,
        "af_pct": 20,
        "septal_reduction_pct": 1,
    },
    {
        "gene": "MYL2",
        "protein": (
            "MYL2 -- 12q24.11 AD -- 166aa -- Myosin-Regulatory-Light-Chain-2-Ventricular-"
            "Slow-19kDa-Thick-Filament-HCM10-AD-AR-NEONATAL-LETHAL -- OMIM-Gene-160781-Disease-HCM10-608758"
        ),
        "locus": "12q24.11",
        "protein_size": "166 aa / 19 kDa",
        "inheritance": (
            "AD for HCM (~2%); AR biallelic → LETHAL NEONATAL HCM; "
            "MYL2 encodes the myosin regulatory light chain 2 (RLC; ventricular/slow isoform); "
            "MID-VENTRICULAR OBSTRUCTION more common than in MYH7/MYBPC3; "
            "mid-cavity gradient distinguishes MYL2-HCM clinically; "
            "biallelic AR MYL2 → infantile/neonatal lethal HCM with generalised myopathy; "
            "RLC regulates myosin cross-bridge kinetics via phosphorylation at Ser14/Ser15"
        ),
        "disease_category": (
            "Hypertrophic cardiomyopathy (HCM) — MYL2 regulatory light chain; "
            "MID-VENTRICULAR OBSTRUCTION more common in MYL2-HCM than septal HCM (MYH7/MYBPC3); "
            "biallelic AR MYL2: LETHAL neonatal HCM + skeletal myopathy; "
            "RLC phosphorylation controls cross-bridge cycling kinetics and myosin filament organisation; "
            "MYL2 variants impair RLC phosphorylation → impaired force generation + abnormal sarcomere assembly"
        ),
        "disease_pathway": (
            "Myosin regulatory light chain 2 (RLC, MYL2) is non-covalently bound to the myosin heavy chain "
            "neck domain, where it acts as a lever arm that amplifies power stroke displacement. "
            "RLC phosphorylation (by myosin light chain kinase, MLCK) at Ser14/Ser15 "
            "increases cross-bridge cycling rate and force production. "
            "AD MYL2-HCM variants: alter RLC–myosin heavy chain interaction → "
            "impaired Ca2+-dependent regulation of cross-bridge kinetics → sarcomere dysfunction → hypertrophy. "
            "MID-VENTRICULAR OBSTRUCTION: hypertrophy occurs preferentially in the mid-cavity papillary muscle region, "
            "creating a mid-cavity gradient (vs. LVOT gradient in septal HCM). "
            "AR biallelic MYL2: complete loss of functional RLC → lethal sarcomere assembly failure → "
            "neonatal cardiomyopathy with generalised skeletal muscle weakness."
        ),
        "pathognomonic": (
            "MID-VENTRICULAR OBSTRUCTION ON ECHO — gradient across mid-cavity (papillary muscle level), "
            "not at the LVOT level; distinguishes MYL2-HCM from septal MYH7/MYBPC3-HCM; "
            "BIALLELIC MYL2 + NEONATAL HCM + SKELETAL MYOPATHY — AR lethal neonatal phenotype; "
            "DOPPLER GRADIENT LOCATION: mid-cavity vs. LVOT level — echocardiographic positioning critical; "
            "MYL2 GENETIC VARIANT — AD heterozygous for HCM; biallelic for lethal neonatal form; "
            "SKELETAL MYOPATHY in biallelic — elevated CK; muscle weakness"
        ),
        "treatment": (
            "MYL2-HCM (AD): "
            "BETA-BLOCKERS — first-line; "
            "VERAPAMIL — alternative; "
            "DISOPYRAMIDE — for mid-ventricular obstruction if drug-refractory; "
            "MAVACAMTEN — applicable if mid-ventricular obstruction with EF ≥55%; "
            "SEPTAL REDUCTION: MYECTOMY (extended to mid-cavity) for drug-refractory mid-ventricular obstruction; "
            "ICD — per standard HCM SCD risk score; "
            "EXERCISE RESTRICTION — competitive sports CONTRAINDICATED; "
            "MYL2-AR NEONATAL (biallelic): "
            "INTENSIVE NEONATAL CARDIAC SUPPORT; "
            "heart transplantation is the only curative option; "
            "outcomes poor — most die in infancy without transplantation; "
            "GENETIC CASCADE: AD inheritance; first-degree relatives at 50% risk"
        ),
        "key_features": [
            "MYL2-HCM: MID-VENTRICULAR obstruction more common than in MYH7/MYBPC3 — gradient at papillary muscle level, not LVOT",
            "Mid-ventricular obstruction: may require extended myectomy reaching mid-cavity; different surgical approach from LVOT myectomy",
            "Biallelic AR MYL2: LETHAL neonatal HCM + skeletal myopathy — homozygous or compound heterozygous LOF; transplant only option",
            "RLC phosphorylation (Ser14/Ser15 MLCK sites) — impaired in MYL2-HCM variants; disrupts cross-bridge kinetics",
            "Mavacamten applicable for mid-ventricular obstruction if EF ≥55%; reduces mid-cavity gradient",
            "Competitive sports CONTRAINDICATED in all MYL2-HCM; moderate recreational exercise with cardiologist guidance",
            "Biallelic AR neonatal HCM: check MYL2 in all neonatal HCM with skeletal myopathy; recurrence risk 25%",
            "Genetic cascade: AD inheritance; 50% inheritance risk; gene-positive/phenotype-negative needs surveillance",
        ],
        "key_ddx": [
            "MYH7 HCM — septal hypertrophy predominant; LVOTO (not mid-ventricular); dominant negative",
            "MYL3 HCM — essential light chain; also mid-cavity obstruction; different gene; Asp94Ala Middle Eastern founder",
            "Fabry disease — lysosomal; pseudo-HCM; alpha-Gal A low; lyso-Gb3 elevated; renal/skin involvement",
            "PRKAG2 HCM — metabolic; glycogen storage; accessory pathway WPW; PRKAG2 mutation",
        ],
        "hcm_pct": 2,
        "lvoto_pct": 35,
        "scd_risk_pct": 5,
        "icd_pct": 22,
        "mavacamten_eligible_pct": 20,
        "af_pct": 15,
        "septal_reduction_pct": 8,
    },
    {
        "gene": "MYL3",
        "protein": (
            "MYL3 -- 3p21.31 AD -- 195aa -- Myosin-Essential-Light-Chain-3-Ventricular-"
            "22kDa-Thick-Filament-HCM8-AD-AR-Middle-East-Founder -- OMIM-Gene-160790-Disease-HCM8-608751"
        ),
        "locus": "3p21.31",
        "protein_size": "195 aa / 22 kDa",
        "inheritance": (
            "AD for HCM (~1%); AR biallelic → EARLY-ONSET severe HCM; "
            "MYL3 encodes the myosin essential light chain 3 (ELC; ventricular isoform); "
            "p.Asp94Ala — FOUNDER MUTATION in Middle Eastern/consanguineous populations; "
            "mid-cavity obstruction similar to MYL2; "
            "ELC stabilises the myosin lever arm; essential for power stroke efficiency; "
            "biallelic MYL3 → early-onset severe HCM; homozygous p.Asp94Ala in consanguineous families"
        ),
        "disease_category": (
            "Hypertrophic cardiomyopathy (HCM) — MYL3 essential light chain; "
            "Mid-cavity obstruction (similar to MYL2); "
            "AR biallelic → early-onset severe HCM; "
            "p.Asp94Ala founder variant in Middle Eastern populations; "
            "ELC stabilises myosin lever arm: mutations impair lever arm function → "
            "reduced force per power stroke → compensatory hypertrophy; "
            "clinical presentation similar to other sarcomeric HCM"
        ),
        "disease_pathway": (
            "Myosin essential light chain 3 (ELC, MYL3) is bound to the myosin lever arm "
            "(between the converter and the light chain binding domain of the heavy chain). "
            "ELC provides structural rigidity to the lever arm — critical for efficient force transmission "
            "from the power stroke to the actin filament. "
            "ELC mutations disrupt lever arm stability → impaired force generation efficiency → "
            "compensatory sarcomere remodelling → hypertrophy. "
            "MYL3-HCM variants: mid-cavity distribution more common (papillary muscle region hypertrophy); "
            "p.Asp94Ala (Asp94Ala): alters a conserved residue at the ELC–EF hand Ca2+-binding region — "
            "may alter Ca2+-sensitivity of ELC → increased cross-bridge stiffness. "
            "AR biallelic (homozygous p.Asp94Ala in consanguineous families): "
            "severely impaired ELC function → early-onset severe HCM from infancy."
        ),
        "pathognomonic": (
            "MID-CAVITY OBSTRUCTION on echocardiography — gradient across mid-cavity, not LVOT; "
            "p.Asp94Ala VARIANT in Middle Eastern/consanguineous patient — founder mutation; screen in relevant populations; "
            "BIALLELIC MYL3 + EARLY-ONSET SEVERE HCM — homozygous or compound heterozygous; consanguinity history; "
            "MYL3 GENETIC VARIANT on cardiomyopathy panel — essential light chain 3 confirmed; "
            "MID-VENTRICULAR DISTRIBUTION — similar to MYL2-HCM; distinguish from LVOTO-predominant subtypes; "
            "EARLY ONSET (infancy/childhood) in AR biallelic forms"
        ),
        "treatment": (
            "MYL3-HCM (AD): "
            "BETA-BLOCKERS — first-line; "
            "VERAPAMIL — alternative; "
            "DISOPYRAMIDE — for refractory mid-ventricular obstruction; "
            "MAVACAMTEN — if mid-ventricular obstruction + EF ≥55%; "
            "EXTENDED MYECTOMY — mid-cavity myectomy for drug-refractory cases; "
            "ICD — per standard HCM SCD risk score; "
            "EXERCISE RESTRICTION — competitive sports CONTRAINDICATED; "
            "MYL3-AR (biallelic, early-onset severe): "
            "INTENSIVE PAEDIATRIC CARDIAC SUPPORT; "
            "CARDIAC TRANSPLANTATION — consider early in progressive disease; "
            "GENETIC COUNSELLING — consanguinity; 25% recurrence in AR; "
            "GENETIC CASCADE — first-degree relatives; population screening for p.Asp94Ala in high-risk ethnic groups"
        ),
        "key_features": [
            "MYL3-HCM: mid-cavity obstruction similar to MYL2; essential light chain stabilises myosin lever arm",
            "p.Asp94Ala: FOUNDER mutation in Middle Eastern/consanguineous populations; screen in relevant ethnic groups",
            "AR biallelic MYL3: early-onset severe HCM from infancy; consanguinity is a key diagnostic clue",
            "Homozygous p.Asp94Ala documented in consanguineous Gulf/Middle Eastern families with severe childhood HCM",
            "Mavacamten applicable for mid-ventricular obstruction if EF ≥55%; REMS programme applies",
            "Extended myectomy (mid-cavity) required for drug-refractory mid-ventricular obstruction; different from LVOT myectomy",
            "Competitive sports CONTRAINDICATED; early-onset AR cases: paediatric cardiology + transplant team involvement",
            "Genetic cascade: AD cases 50% risk; AR cases 25% recurrence; founder variant screening in consanguineous families",
        ],
        "key_ddx": [
            "MYL2 HCM — regulatory light chain; also mid-cavity obstruction; neonatal AR lethal vs early-onset AR MYL3",
            "MYBPC3 HCM — most common HCM; septal distribution; haploinsufficiency; incomplete penetrance",
            "PRKAG2 — metabolic HCM; Wolff-Parkinson-White; glycogen storage; accessory pathways",
            "Fabry disease (GLA) — lysosomal storage; pseudo-HCM; low alpha-Gal A; renal/cornea/skin involvement",
        ],
        "hcm_pct": 1,
        "lvoto_pct": 30,
        "scd_risk_pct": 4,
        "icd_pct": 18,
        "mavacamten_eligible_pct": 15,
        "af_pct": 12,
        "septal_reduction_pct": 7,
    },
]

SEEDS = list(range(2590, 2598))  # 8 seeds for 8 genes


def _rng(seed):
    rng = random.Random(seed)
    return rng


def _simulate_cohort(gene_data, seed):
    rng = _rng(seed)
    n = 40
    patients = []
    lvoto_med = gene_data["lvoto_pct"] / 10.0  # median LVOTO gradient proxy
    scd_med = gene_data["scd_risk_pct"]
    for i in range(n):
        age = rng.randint(25, 72)
        sex = rng.choice(["M", "F"])
        septal_thickness = max(10, round(rng.gauss(18, 4), 1))
        lvoto_gradient = max(0, round(rng.gauss(lvoto_med * 10, lvoto_med * 8), 1)) if rng.random() < (gene_data["lvoto_pct"] / 100) else 0
        ef_percent = max(25, round(rng.gauss(62, 8), 1))
        nyha_class = rng.choices([1, 2, 3, 4], weights=[20, 40, 30, 10])[0]
        scd_risk_5yr = max(0, round(rng.gauss(scd_med, scd_med * 0.5), 1))
        icd_implanted = 1 if (rng.random() < (gene_data["icd_pct"] / 100)) else 0
        af_present = 1 if rng.random() < (gene_data["af_pct"] / 100) else 0
        mavacamten_eligible = 1 if (lvoto_gradient > 30 and ef_percent >= 55 and rng.random() < 0.6) else 0
        treatment_strategy = rng.choice(["beta_blocker", "ccb", "disopyramide", "mavacamten", "septal_reduction"])
        septal_reduction_type = "none"
        if rng.random() < (gene_data["septal_reduction_pct"] / 100):
            septal_reduction_type = rng.choice(["myectomy", "asa"])
        family_scd = 1 if rng.random() < 0.25 else 0
        cardiomyopathy_type = "HCM"
        if gene_data["gene"] in ("TNNI3",) and rng.random() < 0.15:
            cardiomyopathy_type = "RCM"
        elif gene_data["gene"] in ("TNNT2", "TPM1", "ACTC1") and rng.random() < 0.1:
            cardiomyopathy_type = "DCM"
        elif gene_data["gene"] in ("TPM1", "ACTC1", "MYH7", "MYBPC3") and rng.random() < 0.05:
            cardiomyopathy_type = "LVNC"
        exercise_restriction = "competitive"
        if nyha_class >= 3:
            exercise_restriction = "sedentary_only"
        # Gene-specific variants
        variant_map = {
            "MYH7": ["p.Arg403Gln", "p.Arg719Trp", "p.Arg663His", "p.Val606Met"],
            "MYBPC3": ["c.3736+1G>A", "p.Trp792Ter", "p.Gln1233Ter", "p.Lys600Ter"],
            "TNNT2": ["p.Arg92Trp", "p.Arg92Leu", "p.Arg278Cys", "p.Ile79Asn"],
            "TNNI3": ["p.Lys178Glu", "p.Arg162Trp", "p.Arg145Gly", "p.Arg141Gln"],
            "TPM1": ["p.Asp175Asn", "p.Glu180Val", "p.Ala63Val", "p.Lys70Thr"],
            "ACTC1": ["p.Ala295Ser", "p.Tyr166Cys", "p.Met123Val", "p.Glu101Lys"],
            "MYL2": ["p.Arg58Gln", "p.Glu22Lys", "p.Phe18Leu", "p.Ala13Thr"],
            "MYL3": ["p.Asp94Ala", "p.Glu143Lys", "p.Met149Val", "p.Ala57Gly"],
        }
        variant = rng.choice(variant_map.get(gene_data["gene"], ["p.Unknown"]))
        variant_class = rng.choices(["pathogenic", "likely_pathogenic", "vus"], weights=[60, 30, 10])[0]
        patients.append({
            "age": age,
            "sex": sex,
            "cardiomyopathy_type": cardiomyopathy_type,
            "septal_thickness_mm": septal_thickness,
            "lvoto_gradient": lvoto_gradient,
            "ef_percent": ef_percent,
            "nyha_class": nyha_class,
            "scd_risk_5yr": scd_risk_5yr,
            "icd_implanted": icd_implanted,
            "af_present": af_present,
            "mavacamten_eligible": mavacamten_eligible,
            "treatment_strategy": treatment_strategy,
            "septal_reduction_type": septal_reduction_type,
            "genotype_variant": variant,
            "variant_class": variant_class,
            "family_scd_history": family_scd,
            "exercise_restriction": exercise_restriction,
        })
    return patients


def generate_overview():
    gene_summaries = []
    for gene, seed in zip(ATLAS_GENES, SEEDS):
        pts = _simulate_cohort(gene, seed)
        avg_septal = round(sum(p["septal_thickness_mm"] for p in pts) / len(pts), 1)
        avg_ef = round(sum(p["ef_percent"] for p in pts) / len(pts), 1)
        avg_scd = round(sum(p["scd_risk_5yr"] for p in pts) / len(pts), 1)
        icd_pct = round(sum(p["icd_implanted"] for p in pts) / len(pts) * 100, 1)
        af_pct = round(sum(p["af_present"] for p in pts) / len(pts) * 100, 1)
        mavacamten_pct = round(sum(p["mavacamten_eligible"] for p in pts) / len(pts) * 100, 1)
        septal_red_pct = round(sum(1 for p in pts if p["septal_reduction_type"] != "none") / len(pts) * 100, 1)
        family_scd_pct = round(sum(p["family_scd_history"] for p in pts) / len(pts) * 100, 1)
        gene_summaries.append({
            "gene": gene["gene"],
            "locus": gene["locus"],
            "protein_size": gene["protein_size"],
            "inheritance": gene["inheritance"].split(";")[0].strip(),
            "disease_name": gene["disease_category"].split(";")[0].strip()[:80],
            "pathognomonic_short": gene["pathognomonic"].split(";")[0].strip()[:100],
            "hcm_pct": gene["hcm_pct"],
            "lvoto_pct": gene["lvoto_pct"],
            "scd_risk_pct": gene["scd_risk_pct"],
            "icd_pct": gene["icd_pct"],
            "mavacamten_eligible_pct": gene["mavacamten_eligible_pct"],
            "af_pct": gene["af_pct"],
            "septal_reduction_pct": gene["septal_reduction_pct"],
            "avg_septal_thickness_mm": avg_septal,
            "avg_ef_percent": avg_ef,
            "avg_scd_risk_5yr": avg_scd,
            "sim_icd_pct": icd_pct,
            "sim_af_pct": af_pct,
            "sim_mavacamten_pct": mavacamten_pct,
            "sim_septal_reduction_pct": septal_red_pct,
            "sim_family_scd_pct": family_scd_pct,
        })

    all_pts = []
    for gene, seed in zip(ATLAS_GENES, SEEDS):
        all_pts.extend(_simulate_cohort(gene, seed))

    return {
        "title": "Hereditary Cardiomyopathy Atlas",
        "subtitle": (
            "Complete 8-Gene Hereditary Cardiomyopathy Reference — "
            "MYH7-MYBPC3-TNNT2-TNNI3-TPM1-ACTC1-MYL2-MYL3"
        ),
        "genes": [g["gene"] for g in ATLAS_GENES],
        "n_genes": 8,
        "total_patients": 320,
        "seeds": "2590–2597",
        "disease_classes": [
            "MYH7 — HCM #1 (~35%); dominant negative thick filament; Arg403Gln malignant; Mavacamten FDA 2022; full penetrance earlier onset",
            "MYBPC3 — HCM #1 (~35%); haploinsufficiency thick filament; incomplete penetrance 40-50% by 50; South Asian c.3736+1G>A founder",
            "TNNT2 — HCM MALIGNANT (~5%); SCD despite minimal hypertrophy; CMR mandatory; ICD threshold LOWER than standard",
            "TNNI3 — HCM AD (~5%); AR biallelic → RCM INFANTILE most severe; DDx RCM vs constrictive pericarditis mandatory RHC",
            "TPM1 — HCM (~2%); SAME GENE → HCM (GOF calcium sensitisation) OR DCM (LOF); thin filament; LVNC associated",
            "ACTC1 — HCM APICAL variant prominent (~1%); giant negative T-waves V4-V6 pathognomonic; LVOTO rare; spade LV cavity",
            "MYL2 — HCM (~2%); mid-ventricular obstruction more common; AR biallelic → NEONATAL LETHAL HCM + skeletal myopathy",
            "MYL3 — HCM (~1%); mid-cavity obstruction; Asp94Ala Middle Eastern founder; AR biallelic → early-onset severe HCM",
        ],
        "gene_summary": gene_summaries,
        "aggregate_metrics": {
            "avg_septal_thickness_mm": round(sum(p["septal_thickness_mm"] for p in all_pts) / len(all_pts), 1),
            "avg_ef_percent": round(sum(p["ef_percent"] for p in all_pts) / len(all_pts), 1),
            "avg_scd_risk_5yr": round(sum(p["scd_risk_5yr"] for p in all_pts) / len(all_pts), 1),
            "icd_pct": round(sum(p["icd_implanted"] for p in all_pts) / len(all_pts) * 100, 1),
            "af_pct": round(sum(p["af_present"] for p in all_pts) / len(all_pts) * 100, 1),
            "mavacamten_eligible_pct": round(sum(p["mavacamten_eligible"] for p in all_pts) / len(all_pts) * 100, 1),
            "family_scd_pct": round(sum(p["family_scd_history"] for p in all_pts) / len(all_pts) * 100, 1),
        },
        "clinical_pearls": [
            "MAVACAMTEN (Camzyos) FDA 2022 — first cardiac myosin inhibitor; reduces LVOTO; approved NYHA II-III + LVOTO >30 mmHg; "
            "CI if EF<55%; CYP2C19 inhibitors increase exposure; REMS program MANDATORY; hold before septal reduction; "
            "equally effective for MYH7 and MYBPC3-HCM",
            "VASODILATORS ABSOLUTELY CONTRAINDICATED in obstructive HCM: nitrates, PDE5 inhibitors (sildenafil/tadalafil), "
            "dihydropyridine CCBs (amlodipine/nifedipine) — reduce afterload → worsen LVOTO → haemodynamic collapse; "
            "phosphodiesterase inhibitors (sildenafil) used in ED for pulmonary hypertension can be FATAL in obstructive HCM",
            "TNNT2 MALIGNANT RISK: ICD threshold LOWER — family SCD history in TNNT2 = immediate high-risk; "
            "CMR mandatory (LGE without echo hypertrophy); standard HCM risk calculator may UNDERESTIMATE risk in TNNT2",
            "ATRIAL FIBRILLATION: 20-25% lifetime risk in HCM; anticoagulation MANDATORY (HCM-AF = high stroke risk); "
            "direct oral anticoagulants (DOACs) preferred; rhythm vs rate control; "
            "AF ablation outcomes WORSE in HCM (fibrotic substrate)",
            "DISOPYRAMIDE for LVOTO: negative inotrope reduces gradient; QTc monitoring MANDATORY (>500 ms stop drug); "
            "anticholinergic side effects (urinary retention, dry mouth, blurred vision) common; second-line after beta-blocker/CCB",
            "MYBPC3 INCOMPLETE PENETRANCE: normal echo at age 30 does NOT exclude future HCM; "
            "rescreen every 3-5 years lifelong; cardiac MRI LGE detects fibrosis before wall thickening; "
            "South Asian c.3736+1G>A founder: 1-in-500 Indian subcontinent prevalence — screen all South Asian HCM",
            "SEPTAL REDUCTION: MYECTOMY (Morrow) = GOLD STANDARD, preferred young <40 + concomitant mitral surgery; "
            "ALCOHOL SEPTAL ABLATION (ASA) = alternative; CI if pre-existing conduction disease (LBBB, complete AV block); "
            "controversial <40 years; ASA creates scar — theoretical arrhythmia risk; myectomy preferred for young patients",
            "ACTC1 APICAL HCM: giant negative T-waves V4-V6 (≥10 mm) PATHOGNOMONIC — screen with ECG; "
            "LVOTO rare (apex is not outflow tract); spade-shaped LV cavity on CMR/ventriculography; "
            "apical aneurysm → thrombus risk → anticoagulate",
            "COMPETITIVE SPORTS CONTRAINDICATED in ALL HCM genotypes — HCM Sports Task Force guidelines; "
            "shared decision-making for moderate recreational exercise; activity journalling for gene-positive/phenotype-negative; "
            "exercise drives fibrofatty replacement in ARVC but HCM: exercise worsens LVOTO and arrhythmia risk",
            "GENETIC CASCADE TESTING: all first-degree relatives when pathogenic variant identified; "
            "negative genetic test in known-variant family = reassuring (does not need ongoing cardiac surveillance); "
            "gene-positive/phenotype-negative: echo + CMR every 3-5 years; lifestyle guidance",
        ],
    }


def generate_breakdown():
    breakdowns = []
    for gene, seed in zip(ATLAS_GENES, SEEDS):
        pts = _simulate_cohort(gene, seed)
        avg_septal = round(sum(p["septal_thickness_mm"] for p in pts) / len(pts), 1)
        avg_ef = round(sum(p["ef_percent"] for p in pts) / len(pts), 1)
        avg_scd = round(sum(p["scd_risk_5yr"] for p in pts) / len(pts), 1)
        icd_pct = round(sum(p["icd_implanted"] for p in pts) / len(pts) * 100, 1)
        af_pct = round(sum(p["af_present"] for p in pts) / len(pts) * 100, 1)
        mavacamten_pct = round(sum(p["mavacamten_eligible"] for p in pts) / len(pts) * 100, 1)
        septal_red_pct = round(sum(1 for p in pts if p["septal_reduction_type"] != "none") / len(pts) * 100, 1)
        family_scd_pct = round(sum(p["family_scd_history"] for p in pts) / len(pts) * 100, 1)
        lvoto_pct_sim = round(sum(1 for p in pts if p["lvoto_gradient"] > 30) / len(pts) * 100, 1)
        breakdowns.append({
            "gene": gene["gene"],
            "locus": gene["locus"],
            "protein_size": gene["protein_size"],
            "inheritance": gene["inheritance"],
            "disease_category": gene["disease_category"],
            "disease_pathway": gene["disease_pathway"],
            "pathognomonic": gene["pathognomonic"],
            "treatment": gene["treatment"],
            "key_features": gene["key_features"],
            "key_ddx": gene["key_ddx"],
            "n_patients": 40,
            "avg_septal_thickness_mm": avg_septal,
            "avg_ef_percent": avg_ef,
            "avg_scd_risk_5yr": avg_scd,
            "icd_pct": icd_pct,
            "af_pct": af_pct,
            "mavacamten_eligible_pct": mavacamten_pct,
            "septal_reduction_pct": septal_red_pct,
            "family_scd_pct": family_scd_pct,
            "lvoto_gt30_pct": lvoto_pct_sim,
        })
    return {"gene_breakdowns": breakdowns}


def generate_definitions():
    gene_entries = {}
    for gene in ATLAS_GENES:
        gene_entries[gene["gene"]] = {
            "gene": gene["gene"],
            "full_name": gene["gene"],
            "locus": gene["locus"],
            "protein_size": gene["protein_size"],
            "inheritance": gene["inheritance"].split(";")[0].strip(),
            "disease_name": gene["disease_category"],
            "disease_pathway": gene["disease_pathway"],
            "pathognomonic": gene["pathognomonic"],
            "treatment": gene["treatment"][:600],
            "key_features": gene["key_features"],
            "key_ddx": gene["key_ddx"],
            "hcm_pct": gene["hcm_pct"],
            "lvoto_pct": gene["lvoto_pct"],
            "scd_risk_pct": gene["scd_risk_pct"],
            "icd_pct": gene["icd_pct"],
            "mavacamten_eligible_pct": gene["mavacamten_eligible_pct"],
            "af_pct": gene["af_pct"],
            "septal_reduction_pct": gene["septal_reduction_pct"],
        }
    return {
        "gene_entries": gene_entries,
        "cardiomyopathy_glossary": {
            "HCM Phenotypic Classification (Obstructive vs Non-Obstructive vs Apical)": (
                "OBSTRUCTIVE HCM (HOCM): LVOT gradient ≥30 mmHg at rest OR ≥50 mmHg with Valsalva; "
                "SAM (systolic anterior motion) of anterior mitral leaflet creates dynamic obstruction; "
                "prevalence ~70% of HCM patients; treatment: beta-blocker → disopyramide → mavacamten → septal reduction. "
                "NON-OBSTRUCTIVE HCM: LVOT gradient <30 mmHg; symptoms from diastolic dysfunction; "
                "treatment: beta-blocker/verapamil; heart failure drugs if LVEF reduced (dilated phase HCM). "
                "APICAL HCM: hypertrophy confined to LV apex; LVOTO absent; "
                "giant negative T-waves V4-V6 (≥10 mm depth) PATHOGNOMONIC; "
                "spade-shaped LV cavity; ACTC1 (and Japanese population); apical aneurysm complication. "
                "MID-VENTRICULAR OBSTRUCTION: gradient at mid-cavity/papillary muscle level; MYL2, MYL3; "
                "extended myectomy may be needed."
            ),
            "Mavacamten (Camzyos) — Mechanism, Indication, Contraindications, REMS": (
                "MECHANISM: allosteric cardiac myosin inhibitor; stabilises myosin super-relaxed state (SRX); "
                "reduces number of cross-bridges interacting with actin → reduces LVOTO gradient, LV contractility; "
                "reduces dynamic obstruction and improves diastolic filling; "
                "reduces NT-proBNP (cardiac wall stress biomarker). "
                "FDA APPROVAL: April 2022; first disease-specific HCM pharmacotherapy. "
                "INDICATION: symptomatic obstructive HCM (NYHA class II-III) + LVOTO ≥30 mmHg + LVEF ≥55%. "
                "CONTRAINDICATIONS: EF <55% (will reduce further); pregnancy (teratogenic); "
                "concurrent strong/moderate CYP2C19 inhibitors (omeprazole, fluconazole — increase exposure 2-4x). "
                "REMS PROGRAM (Risk Evaluation Mitigation Strategy): MANDATORY; echocardiogram before/during; "
                "EF monitoring every 4-12 weeks; if EF drops to 50-54% → reduce dose; if <50% → discontinue. "
                "HOLD BEFORE SEPTAL REDUCTION: withhold 4 weeks before myectomy or ASA. "
                "Drug interactions: CYP2C19 substrates (clopidogrel — reduced activation; PPIs — increased mavacamten)."
            ),
            "ICD Indications in HCM — Standard vs Low-Threshold (TNNT2)": (
                "STANDARD HCM ICD (ESC 2014 HCM Risk-SCD Calculator): "
                "5-year SCD risk ≥4% → strong recommendation for ICD; "
                "risk factors scored: max wall thickness, LA diameter, family SCD history, NSVT, unexplained syncope, LVOTO gradient, age; "
                "AHA/ACC: Mayo Clinic scoring system alternative (not identical to ESC). "
                "LOW-THRESHOLD ICD — TNNT2: "
                "SCD risk calculator may UNDERESTIMATE risk in TNNT2 (minimal hypertrophy but high fibrosis burden); "
                "family SCD history in TNNT2 family = immediate high-risk classification; "
                "CMR LGE burden guides ICD decision more than wall thickness for TNNT2; "
                "ICD discussion at lower ESC 5yr% than for MYH7/MYBPC3. "
                "GENERAL PRINCIPLES: "
                "ICD preferred over pacemaker (does NOT prevent VF without defibrillation); "
                "subcutaneous ICD (S-ICD) — avoid if pacing required or Holter-driven pacing needed; "
                "wearable defibrillator (WCD) as bridge for high-risk patients awaiting ICD implant."
            ),
            "Septal Reduction Therapy — Myectomy vs Alcohol Septal Ablation (ASA)": (
                "SURGICAL MYECTOMY (Morrow Procedure): "
                "GOLD STANDARD for obstructive HCM refractory to medical therapy; "
                "resects basal interventricular septum → widens LVOT → eliminates/reduces LVOTO; "
                "results durable; perioperative mortality <1% at experienced centres; "
                "PREFERRED: young patients (<40-50 years); concomitant mitral valve repair/replacement; "
                "conduction disease present; complex anatomy. "
                "ALCOHOL SEPTAL ABLATION (ASA): "
                "catheter-based; ethanol injected into septal perforator artery → controlled septal infarct → scar; "
                "reduces septal thickness over 3-6 months; LVOTO gradient reduction similar to myectomy; "
                "CONTRAINDICATIONS: pre-existing LBBB (complete AV block risk requiring PPM); "
                "very thick septum >30mm; complex obstruction requiring mitral work; "
                "CONTROVERSIAL <40 years: theoretical arrhythmia risk from scar; myectomy preferred; "
                "RIGHT BUNDLE BRANCH BLOCK (RBBB) common after ASA — expected finding. "
                "Both require experienced HCM centre (volume threshold); mortality <1% at expert sites."
            ),
            "Disopyramide for LVOTO — Dosing, Monitoring, Side Effects": (
                "MECHANISM: Class IA antiarrhythmic; NEGATIVE INOTROPE; reduces myocardial contractility → "
                "reduces LVOTO gradient; does NOT affect myosin directly (unlike mavacamten); "
                "inhibits fast Na+ channel + anticholinergic properties. "
                "INDICATION: second-line for symptomatic obstructive HCM refractory to beta-blocker/CCB; "
                "often used in combination with beta-blocker. "
                "DOSING: 100-150 mg QID (immediate-release) or 200-300 mg BD (extended-release); "
                "titrate to reduce LVOTO gradient + symptom improvement. "
                "QTc MONITORING MANDATORY: baseline ECG; recheck 48-72 hours after each dose increase; "
                "stop if QTc >500 ms (TdP risk); concurrent QT-prolonging drugs CONTRAINDICATED. "
                "ANTICHOLINERGIC SIDE EFFECTS: dry mouth (most common), urinary retention (especially males with BPH), "
                "constipation, blurred vision; tamsulosin for urinary retention; "
                "METABOLIC: can worsen hypoglycaemia in diabetics; "
                "HEPATOTOXICITY: rare; monitor LFTs. "
                "PIVOTAL: disopyramide does NOT prevent SCD — only reduces LVOTO symptoms."
            ),
            "HCM Genetic Testing — Which Genes, When to Test, Cascade Strategy": (
                "WHOM TO TEST: all patients with confirmed HCM phenotype (LV wall ≥15 mm without secondary cause); "
                "PANEL COMPOSITION: at minimum MYH7, MYBPC3, TNNT2, TNNI3, TPM1, ACTC1, MYL2, MYL3 (8-gene HCM panel); "
                "extended panels: PRKAG2, GLA, LAMP2 (Danon), TTR (amyloid), PLN. "
                "YIELD: 30-60% genotype-positive in sporadic HCM; 50-70% in familial HCM; "
                "PATHOGENIC VARIANT FOUND → cascade testing ALL first-degree relatives (parent, sibling, child); "
                "RELATIVE NEGATIVE FOR PATHOGENIC VARIANT → no cardiac surveillance required (low risk); "
                "GENE-POSITIVE / PHENOTYPE-NEGATIVE RELATIVE → cardiac surveillance (echo + CMR every 3-5 years). "
                "VUS (VARIANT OF UNCERTAIN SIGNIFICANCE) challenge: "
                "do NOT use VUS for clinical decisions; reclassification database ClinVar/ClinGen required; "
                "functional assay (calcium sensitisation, sarcomere motility assay) helps VUS classification. "
                "SOUTH ASIAN MYBPC3 c.3736+1G>A: targeted screening in ALL South Asian HCM patients — "
                "1-in-500 prevalence on Indian subcontinent; significant underdiagnosis in South Asian diaspora populations."
            ),
            "Atrial Fibrillation in HCM — Risk, Anticoagulation, Rhythm vs Rate": (
                "PREVALENCE: 20-25% lifetime risk of AF in HCM (vs. 1-2% in general population); "
                "major contributor to HCM morbidity: stroke, symptom worsening, precipitates decompensation. "
                "STROKE RISK: HCM-AF = HIGH stroke risk regardless of CHA2DS2-VASc score; "
                "ANTICOAGULATION MANDATORY in all HCM patients with AF (warfarin or DOAC); "
                "DOAC preferred (apixaban, rivaroxaban, edoxaban); warfarin acceptable; "
                "do NOT use HAS-BLED to deny anticoagulation. "
                "RHYTHM vs RATE CONTROL: "
                "rhythm control preferred where possible (AF worsens diastolic dysfunction and LVOTO); "
                "beta-blocker/verapamil for rate control; amiodarone for rhythm maintenance; "
                "AF ABLATION: pulmonary vein isolation; outcomes WORSE in HCM vs general AF "
                "(fibrotic atrial substrate, abnormal atrial pressure); recurrence higher; "
                "consider ablation + anticoagulation in refractory symptomatic HCM-AF. "
                "NEW-ONSET AF IN HCM: precipitates LVOTO worsening → acute decompensation; "
                "urgent cardioversion if haemodynamic compromise."
            ),
            "Exercise Restriction in HCM — Competitive vs Recreational": (
                "COMPETITIVE SPORTS CONTRAINDICATED in all patients with HCM (genotype-positive or phenotype-positive); "
                "HCM is the most common cause of SCD in competitive athletes in the USA; "
                "exercise-induced catecholamines → increased LVOTO → haemodynamic compromise + VF; "
                "disqualification from competitive sport applies regardless of ICD implant "
                "(ICD cannot prevent SCD reliably during peak exercise in HCM). "
                "RECREATIONAL EXERCISE: moderate-intensity aerobic exercise (walking, cycling at comfortable pace) "
                "generally PERMITTED with cardiologist guidance; "
                "individualised assessment: rest/provoked LVOTO gradient, exercise treadmill test, holter; "
                "avoid isometric exercise (weightlifting) — increases afterload → worsens LVOTO; "
                "GENE-POSITIVE / PHENOTYPE-NEGATIVE: evolving consensus; "
                "AHA 2023 eligibility criteria: case-by-case; shared decision-making; avoid collision sports; "
                "SHARED DECISION-MAKING DOCUMENT: patient signs informed consent for recreational exercise; "
                "document LVOTO gradient, ICD status, family history, symptom class."
            ),
        },
    }


# ── DB population ──────────────────────────────────────────────────────────────
def populate_db():
    import sqlite3, os
    db_path = os.path.join(os.path.dirname(__file__), "..", "data", "clinical.db")
    db_path = os.path.normpath(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS hereditary_cardiomyopathy_atlas")
    cur.execute("""
        CREATE TABLE hereditary_cardiomyopathy_atlas (
            id INTEGER PRIMARY KEY,
            gene TEXT,
            protein TEXT,
            aa_length INTEGER,
            chromosome TEXT,
            inheritance TEXT,
            cardiomyopathy_type TEXT,
            lvoto_gradient REAL,
            septal_thickness_mm REAL,
            ef_percent REAL,
            nyha_class INTEGER,
            scd_risk_5yr REAL,
            icd_implanted INTEGER,
            af_present INTEGER,
            mavacamten_eligible INTEGER,
            treatment_strategy TEXT,
            septal_reduction_type TEXT,
            genotype_variant TEXT,
            variant_class TEXT,
            family_scd_history INTEGER,
            patient_age INTEGER,
            sex TEXT,
            exercise_restriction TEXT,
            seed INTEGER
        )
    """)
    row_id = 1
    for gene, seed in zip(ATLAS_GENES, SEEDS):
        pts = _simulate_cohort(gene, seed)
        aa_length = int(gene["protein_size"].split(" ")[0])
        for p in pts:
            cur.execute("""
                INSERT INTO hereditary_cardiomyopathy_atlas VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                row_id, gene["gene"], gene["protein"], aa_length, gene["locus"],
                gene["inheritance"].split(";")[0].strip(),
                p["cardiomyopathy_type"],
                p["lvoto_gradient"], p["septal_thickness_mm"], p["ef_percent"],
                p["nyha_class"], p["scd_risk_5yr"],
                p["icd_implanted"], p["af_present"], p["mavacamten_eligible"],
                p["treatment_strategy"], p["septal_reduction_type"],
                p["genotype_variant"], p["variant_class"],
                p["family_scd_history"], p["age"], p["sex"],
                p["exercise_restriction"], seed
            ))
            row_id += 1
    conn.commit()
    conn.close()
    print(f"Populated hereditary_cardiomyopathy_atlas: {row_id - 1} rows in {db_path}")


if __name__ == "__main__":
    populate_db()
    # Quick sanity
    ov = generate_overview()
    print(f"Overview: {ov['total_patients']} patients, {ov['n_genes']} genes, seeds {ov['seeds']}")
    bd = generate_breakdown()
    print(f"Breakdown: {len(bd['gene_breakdowns'])} gene entries")
    df = generate_definitions()
    print(f"Definitions: {len(df['gene_entries'])} gene entries, {len(df['cardiomyopathy_glossary'])} glossary terms")
