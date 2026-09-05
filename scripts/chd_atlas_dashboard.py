#!/usr/bin/env python3
"""Congenital Heart Disease (CHD) Atlas — Complete 8-Gene Hereditary Structural Heart Disease Reference
NKX2-5  (Congenital Heart Disease NKX2-5-related; 324 aa; 5q35.1; AD; ASD + AV block PATHOGNOMONIC;
         most common CHD transcription factor gene; >50 pathogenic variants; AV block progressive —
         MANDATORY lifelong rhythm surveillance even after surgical repair; pacemaker BEFORE high-degree block) ·
GATA4   (Congenital Heart Disease GATA4-related; 442 aa; 8p23.1; AD; ASD 55% + VSD + AVSD;
         GATA4/TBX5 protein-protein interaction required for atrial septation; cardiac GATA4 dysmorphic overlap) ·
TBX5    (Holt-Oram Syndrome; 518 aa; 12q24.21; AD; ASD 85% + conduction defects 75% + RADIAL RAY ANOMALY PATHOGNOMONIC;
         ALL radial ray anomaly children need ECHO; conduction defects independent of structural CHD severity;
         TBX5/GATA4/NKX2-5 cardiac transcription factor network) ·
TBX20   (Congenital Heart Disease TBX20-related; 447 aa; 7p14.2; AD; ASD + valvular defects + DCM late-onset;
         interacts NKX2-5 + GATA4; late-onset DCM in repaired CHD survivors — surveillance mandatory) ·
GATA6   (Congenital Heart Disease GATA6-related; 595 aa; 18q11.2; AD; TOF + ASD + PDA + PANCREATIC AGENESIS PATHOGNOMONIC;
         neonatal diabetes + CHD = GATA6 until proven otherwise; biliary atresia; CFTR-like exocrine pancreatic failure) ·
JAG1    (Alagille Syndrome ALGS1; 1218 aa; 20p12.2; AD LOF; PERIPHERAL PULMONARY STENOSIS 90-97% PATHOGNOMONIC +
         cholestasis (bile duct paucity) + butterfly vertebrae + posterior embryotoxon + Alagille facies;
         liver disease drives morbidity — biliary interventions HIGH RISK in Alagille; Notch signaling ligand) ·
NOTCH1  (Bicuspid Aortic Valve + Calcific Aortic Valve Disease; 2555 aa; 9q34.3; AD;
         BAV 1-2% population; calcific valve disease by age 65 in 50%; TAVR UNCERTAIN in young genetic BAV —
         prefer surgical AVR; aortic root surveillance every 2-3yr; JAG1 is Notch ligand) ·
MYH6    (Familial Atrial Septal Defect ASD3 + DCM1E + Sick Sinus Syndrome; 1939 aa; 14q11.2; AD;
         alpha-myosin heavy chain expressed preferentially in atria; sick sinus + ASD → pacemaker risk post-FONTAN/Glenn;
         late-onset DCM allelic; atrial arrhythmia high risk post-operative)
320-patient aggregate cohort (8 × 40, seeds 1270–1277)
"""

import random

SEED_BASE = 1270

CHD_GENES = [
    # ── NKX2-5 — CHD + AV Block ─────────────────────────────────────────────
    {
        "gene": "NKX2-5",
        "protein": "Homeobox Protein NKX2-5 (NK2 Transcription Factor Related Locus 5)",
        "alias": (
            "NKX2-5; OMIM gene 600584; CHD NKX2-5-related #617912 + #614980 + ASD type 2 #607941; "
            "5q35.1; 324 aa; ~34 kDa; AD (haploinsufficiency); most common single-gene CHD transcription factor; "
            "NKX2-5 is a homeodomain TF essential for cardiac morphogenesis and conduction system development; "
            ">50 pathogenic variants identified; no clear genotype-phenotype correlation for AV block risk; "
            "penetrance high (>95%) but expressivity variable (ASD ± VSD ± AV block); "
            "homozygous LOF lethal in utero (cardiac looping failure); "
            "NKX2-5 co-activates ANF, MLC2V, α-MHC, connexin-40 (Cx40) — Cx40 LOF → AV block"
        ),
        "aa": "324 aa",
        "kDa": "~34 kDa",
        "locus": "5q35.1",
        "omim_gene": 600584,
        "omim_disease": 607941,
        "inheritance": "AD (haploinsufficiency); de novo ~15%; familial ~85% with variable expressivity",
        "gene_class": (
            "NKX2-5 (NK2 homeobox 5) is a cardiac master transcription factor expressed from cardiac progenitor stage; "
            "contains homeodomain (DNA binding) + tinman domain (cardiac-specific); "
            "NKX2-5 activates genes required for: (1) cardiac looping and morphogenesis (HAND1, HAND2); "
            "(2) ventricular septation and valve formation (GATA4 co-factor interaction); "
            "(3) conduction system development — specifically CONNEXIN-40 (Cx40) required for AV node conduction; "
            "NKX2-5 LOF → Cx40 downregulation → progressive AV block (independent of structural CHD repair); "
            "NKX2-5 also required for secondary heart field expansion → right ventricle + outflow tract development; "
            "CRITICAL CLINICAL FEATURE: AV block in NKX2-5 is PROGRESSIVE over decades — "
            "patients who have had ASD repair in childhood develop high-degree AV block (2nd/3rd degree) in adulthood; "
            "AV block progression rate: ~25% by age 40, >40% lifetime — lifelong ECG surveillance mandatory; "
            "Pacemaker implantation BEFORE symptomatic high-degree block — syncope risk; "
            "NKX2-5 variants also identified in sporadic TOF, TGA, DORV — lower penetrance at those alleles"
        ),
        "phenotype": (
            "CARDIAC STRUCTURAL: "
            "ASD (secundum most common 50-60%; primum less common); VSD (perimembranous 20-30%); "
            "AV canal defects (10-15%); tetralogy of Fallot (5-10% of NKX2-5 families); "
            "double-outlet right ventricle; ventricular non-compaction; "
            "CONDUCTION: "
            "AV block — cardinal and PROGRESSIVE: 1st degree → 2nd degree (Mobitz I/II) → 3rd degree; "
            "complete AV block can emerge decades after structural CHD repair — must not assume repair cures AV block risk; "
            "sick sinus syndrome; QRS prolongation; "
            "EXTRACARDIAC: NONE typical — isolated cardiac phenotype (differentiates from TBX5/Holt-Oram); "
            "NBS: not routine — prenatal echo if family history; "
            "SEVERITY: variable — some have ASD alone (mild), others have complex CHD + complete AV block (severe)"
        ),
        "hallmark": (
            "PROGRESSIVE AV BLOCK + ASD (OR ANY CHD) IN SAME PATIENT = NKX2-5 until proven otherwise; "
            "AV BLOCK IS INDEPENDENT OF STRUCTURAL CHD — can worsen even if ASD/VSD repaired; "
            "LIFELONG ECG SURVEILLANCE MANDATORY: annual Holter or ECG from age 20 even if post-repair; "
            "PACEMAKER THRESHOLD LOWER THAN IDIOPATHIC AV BLOCK: implant at 2nd degree Mobitz II or "
            "symptomatic 1st degree in NKX2-5 carriers (do not wait for complete block); "
            "NO EXTRACARDIAC FEATURES — distinguishes NKX2-5 from TBX5 (Holt-Oram radial ray) and GATA6 (pancreatic agenesis); "
            "FAMILY SCREENING: first-degree relatives need ECHO + ECG (AV block/ASD may present silently in relatives)"
        ),
        "treatment_alert": (
            "PACEMAKER BEFORE COMPLETE AV BLOCK: "
            "2nd degree Mobitz II in NKX2-5 → pacemaker without waiting for 3rd degree (syncope risk high); "
            "ANNUAL ECG/HOLTER: mandatory even after ASD repair — AV block emerges post-operatively and in adulthood; "
            "ASD REPAIR: surgical closure or device closure (secundum) — standard approach; "
            "primum ASD: surgical repair required (adjacent to AV valves — device CI); "
            "POST-REPAIR ATRIAL ARRHYTHMIAS: flutter/AF risk increased — anticoagulate appropriately; "
            "FAMILY SCREENING: first-degree relatives → ECHO + 12-lead ECG; "
            "cascade genetic testing in family members; "
            "GENETIC COUNSELLING: 50% transmission risk; de novo ~15%; variable expressivity"
        ),
        "key_ddx": (
            "vs GATA4: GATA4 ASD but NO conduction defects (NKX2-5 has AV block); "
            "vs TBX5: Holt-Oram has radial ray anomaly (NKX2-5 has NO limb anomalies); "
            "vs acquired AV block: isolated AV block without CHD → exclude NKX2-5 in young patients; "
            "vs LMNA: LMNA causes DCM + AV block but NOT ASD; NKX2-5 causes structural CHD + AV block; "
            "vs Kearns-Sayre syndrome: mitochondrial + AV block + ptosis (multisystem); "
            "vs DDD: scoliosis/Brugada in DDD + AV block"
        ),
        "structural_defect": "ASD (secundum/primum) + VSD + TOF (variable)",
        "conduction_defect": "Progressive AV block (1st→3rd degree) — hallmark",
        "extracardiac": "NONE — isolated cardiac phenotype",
    },
    # ── GATA4 — ASD/VSD/AVSD ───────────────────────────────────────────────
    {
        "gene": "GATA4",
        "protein": "GATA Binding Protein 4 (GATA4)",
        "alias": (
            "GATA4; OMIM gene 600576; ASD type 2 #614980; VSD #614964; "
            "8p23.1; 442 aa; ~50 kDa; AD (haploinsufficiency); "
            "zinc finger TF critical for heart and gonad development; "
            "interacts with TBX5 (atrial septation) and NKX2-5 (ventricular development); "
            "GATA4 G296S mutation specifically disrupts GATA4-TBX5 interaction → familial ASD; "
            "Del 8p23.1 (large deletion): GATA4 + other genes → more severe phenotype + psychomotor delay"
        ),
        "aa": "442 aa",
        "kDa": "~50 kDa",
        "locus": "8p23.1",
        "omim_gene": 600576,
        "omim_disease": 614980,
        "inheritance": "AD (haploinsufficiency); de novo common; familial ASD well-described",
        "gene_class": (
            "GATA4 is a zinc finger transcription factor (two C4 zinc fingers) active in heart, gut, lung, and gonads; "
            "CARDIAC DEVELOPMENT: GATA4 is essential for cardiac progenitor specification and morphogenesis; "
            "GATA4/TBX5/NKX2-5 triad constitutes the cardiac transcriptional network for atrial septation; "
            "GATA4 G296S mutation disrupts the GATA4-TBX5 protein interaction → familial ASD; "
            "GATA4 regulates: Nppa (ANF), GATA6, Nkx2-5, Hand1, Hand2, cardiac sarcomere genes; "
            "GATA4 LOF → impaired proepicardial/epicardial development → myocardial thinning + ASD + AVSD; "
            "GATA4 also required for: (1) secondary heart field → right ventricle; (2) atrioventricular canal development; "
            "(3) endocardial cushion formation → AV valve development; "
            "GONADAL: GATA4 is gonadal TF — GATA4 mutations can cause premature ovarian failure and 46XY DSD (rare); "
            "INTERACTION: GATA4 physically binds TBX5 at carboxyl activation domain; G296S residue sits at TBX5 binding interface; "
            "EXPRESSIVITY: ASD most penetrant; VSD and AVSD less common; no consistent conduction defect (unlike NKX2-5)"
        ),
        "phenotype": (
            "CARDIAC STRUCTURAL: "
            "ASD secundum (55-65%); VSD perimembranous (25-30%); AVSD (10-20%); "
            "pulmonary stenosis (10-15%); anomalous pulmonary venous return; "
            "CONDUCTION: mild PR prolongation (some); NO progressive AV block (contrast NKX2-5); "
            "EXTRACARDIAC: mostly isolated cardiac; "
            "Del 8p23.1: GATA4 + INTS8/SOX7 → psychomotor delay + more complex CHD; "
            "GONADAL (rare): premature ovarian failure; "
            "SEVERITY: ASD usually hemodynamically significant (large Qp:Qs); AVSD in some families severe"
        ),
        "hallmark": (
            "ASD + VSD COMBINATION WITHOUT CONDUCTION DEFECTS = classic GATA4 (contrast NKX2-5 with AV block); "
            "GATA4 G296S ALLELE: familial ASD with near-100% penetrance in affected families; "
            "GATA4/TBX5 INTERACTION: mutations disrupting GATA4-TBX5 binding = most penetrant for ASD; "
            "AVSD in GATA4 families: GATA4 is NOT ONLY in Down syndrome — non-DS AVSD → test GATA4; "
            "NO RADIAL RAY ANOMALY (differentiates from TBX5/Holt-Oram); "
            "FAMILY SCREENING: echo all first-degree relatives — penetrance high for ASD"
        ),
        "treatment_alert": (
            "ASD CLOSURE: device closure (Amplatzer/Gore CARDIOFORM) for secundum ASD if rims adequate; "
            "surgical closure for primum ASD + AVSD; "
            "AVSD repair: surgical; mitral valve annuloplasty often required; "
            "POST-REPAIR SURVEILLANCE: annual ECHO for residual shunts + valve function; "
            "PULMONARY HYPERTENSION RISK: large unrepaired ASD → Eisenmenger physiology (ASD rarely if ever closes spontaneously); "
            "repair before PAH onset (<2 yrs if large); "
            "GONADAL: FSH + AMH screening in female carriers if symptoms of premature ovarian failure; "
            "GENETIC COUNSELLING: 50% transmission; variable expressivity; del 8p23.1 → array CGH first"
        ),
        "key_ddx": (
            "vs NKX2-5: NKX2-5 has progressive AV block (GATA4 does not); "
            "vs TBX5: TBX5 has radial ray anomaly + conduction defects (GATA4 does not); "
            "vs Down syndrome AVSD: trisomy 21 → chromosome first; GATA4 if normal chromosomes; "
            "vs MYH6: MYH6-ASD is atrial-specific + sick sinus; GATA4 more complex CHD; "
            "vs GATA6: GATA6 has pancreatic agenesis + PDA; GATA4 is isolated cardiac mostly"
        ),
        "structural_defect": "ASD (secundum/primum) + VSD + AVSD",
        "conduction_defect": "Mild PR prolongation; NO progressive AV block",
        "extracardiac": "Mostly isolated; gonadal (rare); Del 8p23.1 adds neurodevelopmental",
    },
    # ── TBX5 — Holt-Oram Syndrome ───────────────────────────────────────────
    {
        "gene": "TBX5",
        "protein": "T-Box Transcription Factor 5 (TBX5)",
        "alias": (
            "TBX5; OMIM gene 601620; Holt-Oram Syndrome HOS #142900; "
            "12q24.21; 518 aa; ~58 kDa; AD (haploinsufficiency); "
            "T-box TF physically binds GATA4 → atrial septation; also regulates cardiac conduction; "
            "Holt-Oram = RADIAL RAY ANOMALY + CHD (± AV block); radial ray anomaly PATHOGNOMONIC; "
            "penetrance near 100%; expression variable (from triphalangeal thumb to absent radius/ulna)"
        ),
        "aa": "518 aa",
        "kDa": "~58 kDa",
        "locus": "12q24.21",
        "omim_gene": 601620,
        "omim_disease": 142900,
        "inheritance": "AD (haploinsufficiency); penetrance ~100%; expressivity highly variable; de novo ~30-40%",
        "gene_class": (
            "TBX5 is a T-box transcription factor expressed in developing forelimb and heart; "
            "TBX5/GATA4 physical interaction: TBX5 binds GATA4 via T-box domain → co-activate ANF, MLC2V, Connexin-40; "
            "TBX5 also directly activates NKX2-5 in cardiac progenitors — TBX5/NKX2-5/GATA4 cardiac circuit; "
            "LIMB ROLE: TBX5 expressed in forelimb but NOT hindlimb progenitors → explains upper limb-specific anomalies; "
            "TBX5 gradient: low TBX5 → proximal limb; high TBX5 → distal elements → TBX5 gradient required for digit specification; "
            "Radial ray: radius/thumb derived from preaxial/radial side → highest TBX5 expression; "
            "CONDUCTION: TBX5 directly activates Connexin-40 (Cx40) and Connexin-43 (Cx43) → AV conduction; "
            "TBX5 LOF → Cx40 reduction → AV block (similar mechanism to NKX2-5); "
            "STRUCTURAL CHD: ASD secondum most common (85%); AV block common (75%); VSD; "
            "EXPRESSIVITY: triphalangeal thumb (mildest) → hypoplastic/absent thumb → partial radial aplasia → "
            "complete absence radius + ASD + complete AV block (most severe)"
        ),
        "phenotype": (
            "RADIAL RAY ANOMALY (100%, PATHOGNOMONIC): "
            "triphalangeal thumb (mildest) → hypoplastic/absent thumb → carpal anomaly → "
            "radial aplasia/hypoplasia → absent radius → absent ulna (rare) → phocomelia (rare); "
            "ALWAYS BILATERAL but asymmetric; LEFT > RIGHT typically; "
            "UPPER LIMB ONLY — hindlimb/lower limb NORMAL (TBX5 expressed only in forelimb); "
            "CARDIAC (ASD 85%): ASD secundum most common; AV canal; VSD; "
            "CONDUCTION (75%): AV block (1st→3rd degree) — INDEPENDENT of structural CHD; "
            "sick sinus syndrome; left axis deviation; "
            "SEVERITY: ANY upper limb anomaly → MANDATORY ECHO; mild limb → may have severe CHD; "
            "severe limb → ASD may be small (expressivity inversely variable)"
        ),
        "hallmark": (
            "RADIAL RAY ANOMALY + CHD + CONDUCTION DEFECT = HOLT-ORAM SYNDROME (TBX5) PATHOGNOMONIC; "
            "RADIAL RAY ANOMALY IN ANY FORM (even triphalangeal thumb) → MANDATORY ECHO + ECG; "
            "CONDUCTION DEFECTS INDEPENDENT of structural CHD severity — "
            "mild ASD + triphalangeal thumb can still develop complete AV block; "
            "PACEMAKER TIMING: same lower threshold as NKX2-5 — 2nd degree Mobitz II → pacemaker; "
            "LOWER LIMB NORMAL: if lower limb involved → NOT Holt-Oram → look for thrombocytopenia-absent radius (TAR) etc.; "
            "BILATERAL ANOMALY: Holt-Oram is always bilateral (may be highly asymmetric); unilateral → exclude HOS"
        ),
        "treatment_alert": (
            "ECHO MANDATORY FOR ALL RADIAL RAY ANOMALY: "
            "every child with triphalangeal thumb or absent/hypoplastic thumb → ECHO + ECG before any surgery; "
            "PACEMAKER: 2nd degree Mobitz II or symptomatic pause in Holt-Oram → pacemaker; "
            "post-repair AV block can emerge — annual Holter lifelong; "
            "UPPER LIMB SURGERY: work with plastic/orthopedic; radial club hand → pollicization + centralization; "
            "CARDIAC REPAIR: ASD device/surgical + primum repair as appropriate; "
            "GENETIC COUNSELLING: 50% transmission; variable expressivity within families; "
            "prenatal: fetal echo + limb ultrasound; FISH/sequencing for TBX5 prenatally feasible"
        ),
        "key_ddx": (
            "vs NKX2-5: NO radial ray in NKX2-5; NKX2-5 isolated cardiac; "
            "vs GATA4: NO radial ray in GATA4; "
            "vs TAR syndrome (Thrombocytopenia-Absent Radius): TAR — absent radius + THUMBS PRESENT (TBX5 absent radius = thumb absent); "
            "vs Fanconi anaemia: pancytopenia + thumb anomaly → Fanconi (DEB test); "
            "vs VACTERL: multisystem (spine, trachea, esophagus, renal) vs isolated HOS; "
            "vs Roberts syndrome: severe bilateral limb reduction + face + IUGR (different spectrum)"
        ),
        "structural_defect": "ASD (85%) + VSD + AV canal",
        "conduction_defect": "AV block 75% (progressive) + sick sinus — independent of structural CHD",
        "extracardiac": "RADIAL RAY ANOMALY — upper limb ONLY (PATHOGNOMONIC) — bilateral",
    },
    # ── TBX20 — ASD/DCM/Valvular ────────────────────────────────────────────
    {
        "gene": "TBX20",
        "protein": "T-Box Transcription Factor 20 (TBX20)",
        "alias": (
            "TBX20; OMIM gene 606061; Congenital Heart Disease TBXAS1 type #611943; DCM1FF #613286; "
            "7p14.2; 447 aa; ~50 kDa; AD (haploinsufficiency); "
            "T-box TF interacting with NKX2-5, GATA4, GATA5 for atrial and valvular development; "
            "TBX20 LOF → ASD + valvular defects + late-onset DCM in CHD survivors; "
            "less common than NKX2-5/GATA4/TBX5 but well-characterized"
        ),
        "aa": "447 aa",
        "kDa": "~50 kDa",
        "locus": "7p14.2",
        "omim_gene": 606061,
        "omim_disease": 611943,
        "inheritance": "AD (haploinsufficiency); de novo common; variable penetrance",
        "gene_class": (
            "TBX20 is a T-box TF expressed in cardiac progenitors and myocardium throughout development; "
            "TBX20 PROTEIN INTERACTIONS: physically binds NKX2-5, GATA4, GATA5 → synergistic cardiac gene activation; "
            "TBX20 activates: cardiac sarcomere genes (MYH6, MYL2, TNNI3), natriuretic peptides, ion channel genes; "
            "TBX20 represses: Tbx2 and Tbx3 (AV canal boundary formation — chamber vs. AV canal identity); "
            "TBX20 required for: (1) atrial septal formation → ASD; (2) endocardial cushion remodelling → valve formation; "
            "(3) left ventricular compaction → non-compaction; (4) mitral and tricuspid valve development; "
            "LATE EFFECT: TBX20 continues to be expressed in adult myocardium → LOF → progressive DCM in adulthood; "
            "DCM appears years/decades AFTER initial CHD repair → surveillance required in all repaired TBX20 patients; "
            "EXPRESSIVITY: ASD + mitral valve prolapse most common; LVNC in some; DCM in survivors of repaired CHD"
        ),
        "phenotype": (
            "CARDIAC STRUCTURAL: "
            "ASD (secundum 50-65%); mitral valve prolapse (MVP) or regurgitation (40-50%); "
            "tricuspid valve anomalies (20%); LVNC (left ventricular non-compaction, 10-15%); VSD (20-30%); "
            "LATE MANIFESTATION: dilated cardiomyopathy — emerges in adult survivors of repaired CHD; "
            "can present as unexplained DCM without prior CHD history (incomplete penetrance); "
            "CONDUCTION: atrial arrhythmia (AF, flutter) especially post-repair; "
            "NO progressive AV block (contrast NKX2-5/TBX5); "
            "EXTRACARDIAC: NONE — isolated cardiac; "
            "SEVERITY: ASD alone may be mild; DCM in adults can be severe → heart transplant in some"
        ),
        "hallmark": (
            "ASD + MITRAL VALVE PROLAPSE/REGURGITATION COMBINATION = TBX20 suspect; "
            "LATE-ONSET DCM IN REPAIRED CHD SURVIVOR = TBX20 until proven otherwise; "
            "TBX20 DCM: may appear as isolated DCM in adults without known CHD history (incomplete penetrance); "
            "LVNC: left ventricular non-compaction + ASD in same patient → TBX20 or NKX2-5; "
            "VALVULAR DISEASE: mitral + tricuspid anomalies together with ASD are TBX20 signature; "
            "ONGOING SURVEILLANCE: ALL TBX20 carriers need annual ECHO throughout life (DCM emergence)"
        ),
        "treatment_alert": (
            "ANNUAL ECHO LIFELONG: DCM can emerge decades after initial CHD repair — do not stop surveillance; "
            "DCM MANAGEMENT: standard HFrEF therapy (ACEi/ARB/ARNI, beta-blocker, MRA, SGLT2i); "
            "ICD: if LVEF <35% or symptomatic arrhythmia; "
            "HEART TRANSPLANT: end-stage TBX20-DCM — outcomes similar to other genetic DCMs; "
            "MITRAL VALVE: MVP with severe regurgitation → surgical repair/replacement; "
            "LVNC SURVEILLANCE: annual ECHO + cardiac MRI every 2-3 years; anticoagulate if LV thrombus; "
            "ATRIAL ARRHYTHMIA: post-repair AF/flutter → rate control + anticoagulation; "
            "rhythm control (ablation) for persistent AF"
        ),
        "key_ddx": (
            "vs NKX2-5: NKX2-5 has AV block (TBX20 does not); both can cause ASD + LVNC; "
            "vs TBX5: TBX5 has radial ray (TBX20 does not); "
            "vs LMNA: LMNA-DCM has AV block + skeletal myopathy; TBX20-DCM is isolated cardiac; "
            "vs TTN: TTN-DCM does NOT cause ASD; TBX20 causes both; "
            "vs Barth syndrome: X-linked; DCM + LVNC + neutropenia (not TBX20 isolated cardiac)"
        ),
        "structural_defect": "ASD + mitral valve prolapse/regurgitation + LVNC",
        "conduction_defect": "Atrial arrhythmia post-repair; NO AV block",
        "extracardiac": "NONE — isolated cardiac phenotype",
    },
    # ── GATA6 — TOF + Pancreatic Agenesis ───────────────────────────────────
    {
        "gene": "GATA6",
        "protein": "GATA Binding Protein 6 (GATA6)",
        "alias": (
            "GATA6; OMIM gene 601656; CHD GATA6-related #614454; Pancreatic Agenesis PAGEN2 #614849; "
            "18q11.2; 595 aa; ~65 kDa; AD (haploinsufficiency); "
            "zinc finger TF expressed in heart, pancreas, liver, lung, GI tract; "
            "GATA6 LOF → TOF + ASD + AVSD + PDA + PANCREATIC AGENESIS (pathognomonic combination); "
            "neonatal diabetes + CHD = GATA6 until proven otherwise; biliary atresia association"
        ),
        "aa": "595 aa",
        "kDa": "~65 kDa",
        "locus": "18q11.2",
        "omim_gene": 601656,
        "omim_disease": 614454,
        "inheritance": "AD (haploinsufficiency); de novo very common (>70%); familial AD cases described",
        "gene_class": (
            "GATA6 is a zinc finger transcription factor closely related to GATA4 but with broader expression; "
            "GATA6 EXPRESSION: heart (endocardium, smooth muscle, outflow tract), pancreatic progenitors, "
            "hepatoblasts, pulmonary epithelium, GI enteroendocrine cells; "
            "CARDIAC ROLE: GATA6 critical for outflow tract development (TOF, TGA, truncus arteriosus); "
            "endocardial cushion formation (AVSD); ductus arteriosus patency regulation; "
            "GATA6/NKX2-5 interaction required for outflow tract septation → GATA6 LOF → TOF most common; "
            "PANCREATIC ROLE: GATA6 is master regulator of pancreatic progenitor identity; "
            "GATA6 LOF → pancreatic agenesis or severe hypoplasia → neonatal diabetes (CFTR-like exocrine failure too); "
            "exocrine pancreatic insufficiency + endocrine insufficiency (insulin-dependent diabetes from birth); "
            "LIVER/BILIARY: GATA6 expressed in hepatoblasts → biliary atresia/paucity in some GATA6 LOF patients; "
            "LUNG: pulmonary hypoplasia (rare); "
            "GATA6 vs GATA4: GATA4 more ASD/VSD; GATA6 more TOF/outflow tract + extracardiac features"
        ),
        "phenotype": (
            "CARDIAC (outflow tract predominant): "
            "Tetralogy of Fallot (TOF 35-45%); ASD (30-40%); AVSD (15-20%); "
            "persistent ductus arteriosus (PDA, 20-30%); transposition of great arteries (TGA, 10%); "
            "truncus arteriosus (5-10%); ventricular septal defect; "
            "PANCREATIC (pathognomonic combination with CHD): "
            "complete pancreatic agenesis → insulin-dependent diabetes from birth (neonatal) + exocrine pancreatic insufficiency; "
            "partial pancreatic hypoplasia → permanent neonatal diabetes (DEND syndrome-like but GATA6); "
            "exocrine failure → steatorrhea + fat-soluble vitamin deficiency; "
            "HEPATOBILIARY: biliary atresia/paucity (5-10%); cholestasis; gallbladder agenesis; "
            "SEVERITY: neonatal diabetes + complex CHD (TOF/TGA) in same neonate → URGENT investigation"
        ),
        "hallmark": (
            "NEONATAL DIABETES + CHD (especially TOF) = GATA6 until proven otherwise; "
            "PANCREATIC AGENESIS + TOF COMBINATION IS PATHOGNOMONIC for GATA6 (virtually no other single gene); "
            "EXOCRINE PANCREATIC INSUFFICIENCY: steatorrhea + fat malabsorption in a neonate with CHD → GATA6; "
            "INSULIN REQUIREMENTS FROM BIRTH: permanent neonatal diabetes (contrast transient NDM); "
            "PDA + TOF + GATA6: PDA in context of TOF should prompt GATA6 testing; "
            "BILIARY ATRESIA: if GATA6 CHD + neonatal jaundice → hepatobiliary evaluation urgently"
        ),
        "treatment_alert": (
            "PANCREATIC AGENESIS MANAGEMENT: "
            "INSULIN THERAPY FROM BIRTH: intensive insulin regime; CGM strongly recommended; "
            "PANCREATIC ENZYME REPLACEMENT (PERT): CREON or Pancreaze for exocrine insufficiency — start early; "
            "FAT-SOLUBLE VITAMINS: A, D, E, K supplementation mandatory; "
            "TOF REPAIR: standard cardiac surgical repair; timing per hemodynamic status; "
            "PDA: indomethacin or surgical ligation per clinical indication; "
            "BILIARY: hepatoportoenterostomy (Kasai) if biliary atresia — timing critical (<90 days); "
            "MULTIDISCIPLINARY TEAM: cardiology + endocrinology + gastroenterology + genetics from birth; "
            "GENETIC COUNSELLING: >70% de novo — recurrence risk low unless parent is carrier"
        ),
        "key_ddx": (
            "vs KATP-channel NDM (KCNJ11/ABCC8): neonatal diabetes but NO CHD structural defect; "
            "vs Down syndrome: trisomy 21 AVSD but NO pancreatic agenesis; "
            "vs GATA4: GATA4 is ASD/VSD without extracardiac; GATA6 is outflow tract + pancreatic; "
            "vs 22q11.2 deletion: TOF + immune + palate (Di George) but NO pancreatic agenesis; "
            "vs Alagille (JAG1): TOF + liver but pulmonary stenosis pattern different; NO pancreatic agenesis in JAG1"
        ),
        "structural_defect": "TOF + ASD + PDA + AVSD + TGA (outflow tract predominant)",
        "conduction_defect": "Post-operative arrhythmia; NO primary conduction defect",
        "extracardiac": "PANCREATIC AGENESIS (pathognomonic) + biliary atresia + pulmonary hypoplasia",
    },
    # ── JAG1 — Alagille Syndrome ─────────────────────────────────────────────
    {
        "gene": "JAG1",
        "protein": "Jagged Canonical Notch Ligand 1 (JAG1)",
        "alias": (
            "JAG1; OMIM gene 601920; Alagille Syndrome ALGS1 #118450; "
            "20p12.2; 1218 aa; ~134 kDa; AD LOF; "
            "Notch pathway ligand; ALGS1 = CHOLESTASIS + PERIPHERAL PULMONARY STENOSIS (90-97%) + "
            "BUTTERFLY VERTEBRAE + POSTERIOR EMBRYOTOXON + ALAGILLE FACIES; "
            "liver disease drives long-term morbidity; biliary interventions HIGH RISK in Alagille; "
            "NOTCH2 LOF → ALGS2 (similar phenotype, rare)"
        ),
        "aa": "1218 aa",
        "kDa": "~134 kDa",
        "locus": "20p12.2",
        "omim_gene": 601920,
        "omim_disease": 118450,
        "inheritance": "AD LOF; penetrance high (~97%) but expressivity EXTREMELY variable; de novo ~50-60%",
        "gene_class": (
            "JAG1 encodes Jagged-1, a transmembrane Notch signaling ligand activating NOTCH1/2/3 receptors; "
            "JAG1 EXPRESSION: biliary epithelium (bile duct development), pulmonary vasculature, vertebrae, eye, kidney; "
            "BILIARY ROLE: JAG1/NOTCH2 signaling required for bile duct morphogenesis; "
            "JAG1 LOF → intrahepatic bile duct paucity → progressive cholestasis → biliary cirrhosis; "
            "PULMONARY VASCULAR ROLE: JAG1/NOTCH regulates pulmonary artery branching; "
            "JAG1 LOF → peripheral pulmonary artery stenosis (PPS) — diffuse bilateral, multiple levels; "
            "PPS in Alagille: hypoplastic main PA + small peripheral branches (NOT valvar stenosis alone); "
            "VERTEBRAL: JAG1/NOTCH required for vertebral segmentation — butterfly vertebrae (anterior arch cleft); "
            "OCULAR: posterior embryotoxon (prominent Schwalbe ring) — anterior segment anomaly (slit lamp); "
            "RENAL: renal tubular acidosis, renal dysplasia (20-30% of Alagille); "
            "INTRACRANIAL VASCULAR: vascular anomalies (Moyamoya, aneurysms) — risk of intracranial hemorrhage"
        ),
        "phenotype": (
            "PENTAD OF ALAGILLE SYNDROME (≥3 = diagnosis if JAG1+): "
            "1. CHOLESTASIS: neonatal jaundice (bile duct paucity on liver biopsy — NOT biliary atresia); "
            "   conjugated hyperbilirubinemia; pruritus (severe — major QoL issue); xanthomata; "
            "2. PERIPHERAL PULMONARY STENOSIS (90-97%): diffuse bilateral PPS — PATHOGNOMONIC; "
            "   gradient may improve with age; right ventricular hypertrophy; balloon dilation/stenting if severe; "
            "3. BUTTERFLY VERTEBRAE: anterior arch cleft on CXR/spine X-ray; usually asymptomatic; "
            "4. POSTERIOR EMBRYOTOXON: prominent Schwalbe ring on slit lamp (80%); "
            "5. ALAGILLE FACIES: triangular face + deep-set eyes + broad forehead + small chin; "
            "HEPATIC: portal hypertension → varices → GI bleeding; cirrhosis in 15-20%; liver Tx in 15%; "
            "RENAL: RTA, renal dysplasia (20%); "
            "INTRACRANIAL: vascular anomalies → hemorrhage risk; "
            "SEVERITY: extremely variable — mild (incidental posterior embryotoxon) to liver transplant + severe RV failure"
        ),
        "hallmark": (
            "PERIPHERAL PULMONARY STENOSIS (BILATERAL DIFFUSE) + NEONATAL CHOLESTASIS = ALAGILLE (JAG1) until proven otherwise; "
            "BUTTERFLY VERTEBRAE on CXR in a jaundiced neonate → immediate JAG1 testing; "
            "POSTERIOR EMBRYOTOXON (80%): slit-lamp exam MANDATORY in all suspected Alagille; "
            "BILIARY INTERVENTIONS EXTREMELY HIGH RISK: "
            "  Kasai hepatoportoenterostomy in Alagille (if misdiagnosed as biliary atresia) → catastrophic surgical risk; "
            "  ERCP, biliary stenting → high risk of bile duct injury; "
            "  liver biopsy — bile duct paucity (NOT biliary atresia) — KEY HISTOLOGIC DISTINCTION; "
            "JAG1/NOTCH2 PATHWAY: JAG1 is Notch-1 ligand — relevant to NOTCH1 BAV (same pathway); "
            "INTRACRANIAL HEMORRHAGE RISK: screen for intracranial vascular anomalies (MRA) in all patients"
        ),
        "treatment_alert": (
            "DO NOT perform KASAI HEPATOPORTOENTEROSTOMY in Alagille: "
            "  bile duct paucity (JAG1) is NOT biliary atresia — liver biopsy + MRCP distinguish; "
            "  Kasai in Alagille = surgical harm with no benefit; "
            "CHOLESTASIS MANAGEMENT: UDCA (ursodeoxycholic acid); cholestyramine for pruritus; "
            "  odevixibat (ileal bile acid transporter inhibitor) FDA approved 2023 for ALGS pruritus — significant benefit; "
            "LIVER TRANSPLANT: 15-20% of ALGS require LTx; living related donor option; "
            "PULMONARY STENOSIS: balloon dilation/stenting for gradient >50 mmHg or RV dysfunction; "
            "  interventions generally safe but PPS may recur after balloon (diffuse disease); "
            "RENAL: monitor creatinine + urinalysis; RTA → bicarbonate supplementation; "
            "MRA BRAIN: screen for intracranial vascular anomalies at diagnosis; repeat if new neurological symptoms; "
            "PRURITUS (major QoL): cholestyramine + rifampicin + naltrexone → odevixibat now first-line FDA approved"
        ),
        "key_ddx": (
            "vs Biliary atresia (BA): BA — extrahepatic ducts obliterated → Kasai needed URGENTLY; "
            "Alagille — intrahepatic bile duct paucity → Kasai NOT indicated + harmful; "
            "MRCP + liver biopsy + JAG1 testing distinguish; "
            "vs Neonatal hepatitis: less specific histology; normal vertebrae/no PPS; "
            "vs PFIC (progressive familial intrahepatic cholestasis): ABCB11/ATP8B1/ABCB4 — normal GGT in PFIC1/2; "
            "vs ALGS2 (NOTCH2 LOF): same phenotype — renal phenotype more prominent in NOTCH2; "
            "vs Williams syndrome: peripheral PS + elfin facies but NO cholestasis; elastin (ELN) gene"
        ),
        "structural_defect": "Peripheral pulmonary stenosis (90-97%, pathognomonic) + TOF (20%)",
        "conduction_defect": "Post-operative; NO primary conduction defect",
        "extracardiac": "Cholestasis + butterfly vertebrae + posterior embryotoxon + Alagille facies + renal + intracranial",
    },
    # ── NOTCH1 — BAV + Calcific Aortic Valve Disease ────────────────────────
    {
        "gene": "NOTCH1",
        "protein": "Neurogenic Locus Notch Homolog Protein 1 (NOTCH1)",
        "alias": (
            "NOTCH1; OMIM gene 190198; Aortic Valve Disease AVD1 #109730; BAV + CALCIFIC AORTIC STENOSIS; "
            "9q34.3; 2555 aa; ~300 kDa (full length); AD LOF; "
            "Notch receptor — JAG1 is Notch ligand (same pathway); "
            "NOTCH1 LOF → BAV + accelerated calcific valve disease; "
            "BAV 1-2% of population; CALCIFIC AoV by age 65 in 50% of BAV; "
            "TAVR in young GENETIC BAV: uncertain — prefer surgical AVR; Turner syndrome — NOTCH1 enhancer variants"
        ),
        "aa": "2555 aa",
        "kDa": "~300 kDa",
        "locus": "9q34.3",
        "omim_gene": 190198,
        "omim_disease": 109730,
        "inheritance": "AD LOF; penetrance variable for calcific AVD (NOTCH1 is necessary but not sufficient); "
                       "BAV concordance ~25% in MZ twins → other modifier genes/environment",
        "gene_class": (
            "NOTCH1 is a type I transmembrane receptor — largest human protein; "
            "NOTCH1 SIGNALING: JAG1/DLL3/DLL4 ligand binding → γ-secretase cleavage → NICD (intracellular domain) "
            "translocates to nucleus → RBPj-mediated transcription; "
            "VALVE DEVELOPMENT: NOTCH1 required for endocardial-to-mesenchymal transition (EMT) in valve primordia; "
            "NOTCH1/JAG1 signaling axis: JAG1 (Alagille) → NOTCH1 (BAV) — same pathway; "
            "CALCIFICATION MECHANISM: NOTCH1 LOF → de-repression of RUNX2 + BMP2 → osteoblast-like differentiation "
            "of valve interstitial cells → calcium deposition → calcific aortic valve disease (CAVD); "
            "COARCTATION: NOTCH1 variants also found in families with coarctation of aorta; "
            "BICUSPID VALVE FORMATION: NOTCH1 required for cusp septation (tricuspid → bicuspid if LOF); "
            "BAV and associated aortopathy: BAV → turbulent flow + NOTCH1 LOF → accelerated aortic root dilation; "
            "TURNER SYNDROME: 45X0 → BAV; NOTCH1 enhancer variants on X chromosome contribute to Turner CHD"
        ),
        "phenotype": (
            "BICUSPID AORTIC VALVE (BAV): "
            "most common CHD if isolated counted (1-2% population); right-left fusion most common type (75%); "
            "right-noncoronary fusion (20%); anterior-posterior (5%); "
            "BAV often asymptomatic for decades → progressive CALCIFIC AORTIC STENOSIS; "
            "CALCIFIC AORTIC VALVE DISEASE (CAVD): "
            "NOTCH1 LOF BAV → calcification ACCELERATED vs sporadic BAV; "
            "AS presenting in 4th-6th decade (earlier than sporadic CAVD); "
            "AORTIC REGURGITATION: BAV → cusp prolapse → AR; "
            "AORTOPATHY: aortic root/ascending aorta dilation independent of valve function (fibrillin/TGF overlap?); "
            "COARCTATION OF AORTA: 5-10% of NOTCH1 BAV families; "
            "SEVERITY: wide range — some NOTCH1 carriers have normal tAV (incomplete penetrance); "
            "others severe CAVD requiring AVR in 40s"
        ),
        "hallmark": (
            "BICUSPID AORTIC VALVE + FAMILY HISTORY OF CAVD IN 4th-6th DECADE = NOTCH1 testing; "
            "NOTCH1 BAV → CALCIFICATION ACCELERATED vs sporadic BAV: earlier AVR expected; "
            "TAVR IN YOUNG GENETIC BAV IS UNCERTAIN: "
            "  BAV anatomy (elliptical annulus, supra-annular leaflets) → higher TAVR paravalvular leak + device migration risk; "
            "  long-term TAVR durability in young patients unknown; "
            "  CURRENT RECOMMENDATION: surgical AVR preferred in NOTCH1-BAV patients <65 yr; "
            "AORTIC SURVEILLANCE: ECHO every 2-3 years from diagnosis regardless of gradient (aortopathy risk); "
            "COARCTATION SCREEN: MRI/CT aorta in all NOTCH1 carriers; "
            "FAMILY SCREEN: ECHO all first-degree relatives; BAV in siblings → cascade screening"
        ),
        "treatment_alert": (
            "SURGICAL AVR PREFERRED OVER TAVR IN YOUNG NOTCH1-BAV: "
            "  BAV anatomy challenges TAVR device seating; long-term durability data lacking in <65yr genetic BAV; "
            "  surgeon expertise critical — Ross procedure or bioprosthetic AVR with root replacement if aortopathy; "
            "AORTIC ROOT SURGERY THRESHOLD LOWER: "
            "  BAV + aortic dilation ≥5.0 cm (or ≥4.5 cm if risk factors: rapid expansion, family history dissection); "
            "  MRI/CT aorta monitoring every 1-2 years if diameter 4.0-5.0 cm; "
            "COARCTATION: balloon + stenting or surgical repair per gradient; "
            "ANTIBIOTIC PROPHYLAXIS: BAV + prior valve intervention → standard SBE prophylaxis for dental; "
            "LIPID-LOWERING: statins do NOT slow BAV calcification (negative ASTRONOMER trial); "
            "  statin for CV risk reduction only; "
            "PREGNANCY: severe BAV/AS contraindication to pregnancy without prior intervention"
        ),
        "key_ddx": (
            "vs sporadic BAV: sporadic BAV — NOTCH1 not identified; lower calcification rate; "
            "vs Turner syndrome: 45X0 → BAV + coarctation + growth failure — chromosomal; "
            "vs JAG1/Alagille: peripheral PS not aortic stenosis; cholestasis; butterfly vertebrae; "
            "vs FBN1/Marfan: aortic root aneurysm + lens ectopia + tall stature (NOT BAV calcification); "
            "vs Williams syndrome: supravalvar AS (not valvar) + peripheral PS + elfin facies (ELN gene)"
        ),
        "structural_defect": "Bicuspid aortic valve + calcific aortic stenosis + aortopathy",
        "conduction_defect": "Post-AVR heart block if CA extends to AV node",
        "extracardiac": "NONE — isolated cardiac + aortic; Turner syndrome if combined 45X0",
    },
    # ── MYH6 — Familial ASD + DCM + Sick Sinus ──────────────────────────────
    {
        "gene": "MYH6",
        "protein": "Myosin Heavy Chain Alpha (Alpha-MHC; MYH6)",
        "alias": (
            "MYH6; OMIM gene 160710; ASD type 3 (ASDIII) #614089; DCM1EE #613252; "
            "Sick Sinus Syndrome SSS3 #614090; "
            "14q11.2; 1939 aa; ~224 kDa; AD; "
            "alpha-myosin heavy chain preferentially expressed in ATRIA (beta-MHC/MYH7 in ventricles); "
            "MYH6 LOF → atrial-specific sarcomere dysfunction → ASD + sick sinus + DCM allelic; "
            "pacemaker risk after atrial surgery (FONTAN/Glenn) in MYH6 patients"
        ),
        "aa": "1939 aa",
        "kDa": "~224 kDa",
        "locus": "14q11.2",
        "omim_gene": 160710,
        "omim_disease": 614089,
        "inheritance": "AD; penetrance variable; MYH6 ASD lower penetrance than NKX2-5/TBX5",
        "gene_class": (
            "MYH6 encodes alpha-myosin heavy chain (α-MHC) — the dominant myosin isoform in HUMAN ATRIA; "
            "ATRIAL SPECIFICITY: alpha-MHC is preferentially atrial in humans (contrast mice where it is ventricular); "
            "this explains the atrial-predominant phenotype (ASD, atrial arrhythmia, sick sinus); "
            "MYH6 encodes the motor domain of the cardiac myosin molecule; "
            "MYH6/MYH7 ratio: alpha-MHC LOF → compensatory MYH7 (beta-MHC) upregulation → stiffer atrial myosin kinetics; "
            "ATRIAL DEVELOPMENT: MYH6 required for atrial sarcomere organization and contractile function; "
            "atrial sarcomere deficiency → impaired atrial wall mechanical tension → atrial septal defect maintenance; "
            "CONDUCTION: MYH6 expressed in SA node → LOF → sick sinus syndrome; "
            "slow heart rate + pauses → syncope; pacemaker needed; "
            "ALLELIC DISEASE: MYH6 variants also cause DCM (DCM1EE) — phenotypically distinct family members; "
            "HCM-allelic: rare MYH6 GOF variants cause HCM but MYH7 is dominant HCM gene; "
            "POST-OPERATIVE RISK: atrial surgery (maze, FONTAN, Glenn) in MYH6 → high sick sinus recurrence"
        ),
        "phenotype": (
            "ATRIAL SEPTAL DEFECT (ASD type 3): "
            "late-presenting ASD (may not diagnose until adult exercise intolerance); "
            "secundum ASD common; moderate-large size; right heart dilation; "
            "SICK SINUS SYNDROME (SSS): "
            "sinus bradycardia + pauses + chronotropic incompetence; "
            "may precede or follow ASD diagnosis; "
            "post-ASD repair arrhythmia: VERY HIGH risk (atrial flutter + sick sinus); "
            "ATRIAL ARRHYTHMIA: AF/flutter at high rate — especially post-operative; "
            "DILATED CARDIOMYOPATHY (allelic): "
            "DCM in some family members without ASD (variable expressivity); "
            "SEVERITY: ASD alone — mild until Eisenmenger; "
            "sick sinus + post-repair arrhythmia → significant morbidity; "
            "pacemaker often required post-atrial surgery"
        ),
        "hallmark": (
            "ASD + SICK SINUS SYNDROME IN SAME PATIENT = MYH6; "
            "PACEMAKER RISK AFTER ASD REPAIR IS VERY HIGH in MYH6: "
            "  ASD device closure → reduced atrial stretch → sick sinus unmasked; "
            "  surgical closure → maze/extensive incisions → sinus node injury; "
            "  INFORM surgeon pre-operatively: MYH6 carrier → plan pacemaker access intraoperatively; "
            "ATRIAL FLUTTER HIGH RISK POST-FONTAN/GLENN in MYH6: "
            "  FONTAN circulation + sick sinus → haemodynamically unstable flutter; "
            "  cavo-pulmonary connection + arrhythmia = life-threatening; "
            "LATE-ONSET DCM: yearly ECHO even without CHD history (allelic DCM in carriers)"
        ),
        "treatment_alert": (
            "PRE-OPERATIVE COUNSELLING FOR ASD REPAIR: "
            "  MYH6 known prior to ASD repair → discuss pacemaker implantation simultaneously; "
            "  sick sinus risk near-certain post-operatively; "
            "PACEMAKER: dual-chamber (DDD) pacing after sick sinus confirmed; "
            "  CRT-D if DCM develops with sick sinus; "
            "ASD CLOSURE: device closure preferred if anatomy allows (less atrial surgery → less sinus node disruption); "
            "POST-CLOSURE MONITORING: 48h Holter within 1 week; Holter at 1 month + 1 year; "
            "DCM SURVEILLANCE: annual ECHO for all MYH6 carriers (even without known ASD); "
            "FONTAN/SINGLE VENTRICLE: MYH6 → very high arrhythmia risk — consider prophylactic antiarrhythmic + Holter"
        ),
        "key_ddx": (
            "vs NKX2-5: NKX2-5 has AV block (not sick sinus); both cause ASD; "
            "vs TBX5: TBX5 has radial ray; both cause ASD + conduction disease; "
            "vs GATA4: GATA4 no conduction defects; MYH6 has sick sinus; "
            "vs MYH7 (HCM gene): MYH7 = HCM (ventricular hypertrophy); MYH6 = ASD + atrial phenotype; "
            "vs SSS sporadic: isolated sick sinus → exclude MYH6 + HCN4 + SCN5A"
        ),
        "structural_defect": "ASD type 3 (atrial-specific, late-presenting)",
        "conduction_defect": "Sick sinus syndrome + post-operative AF/flutter (HALLMARK)",
        "extracardiac": "NONE — isolated cardiac; DCM allelic",
    },
]


def _make_cohort(gd: dict, seed: int) -> list:
    rng = random.Random(seed)
    g = gd["gene"]
    pts = []
    for i in range(40):
        age = rng.randint(0, 65)
        sex = rng.choice(["M", "F"])
        # Severity weighting per gene
        if g == "NKX2-5":
            sev_weights = [0.25, 0.50, 0.25]  # mild, moderate, severe
        elif g == "TBX5":
            sev_weights = [0.20, 0.45, 0.35]
        elif g == "GATA6":
            sev_weights = [0.10, 0.40, 0.50]
        elif g == "JAG1":
            sev_weights = [0.25, 0.45, 0.30]
        elif g == "NOTCH1":
            sev_weights = [0.35, 0.45, 0.20]
        elif g == "MYH6":
            sev_weights = [0.30, 0.50, 0.20]
        else:
            sev_weights = [0.30, 0.45, 0.25]
        severity = rng.choices(["mild", "moderate", "severe"], weights=sev_weights)[0]
        pt = {
            "id": f"{g}-{seed}-{i+1:03d}",
            "gene": g,
            "age_at_diagnosis": age,
            "sex": sex,
            "severity": severity,
            # Structural CHD
            "asd": rng.random() < (
                0.85 if g == "TBX5" else
                0.65 if g in ("NKX2-5", "GATA4", "TBX20") else
                0.60 if g == "MYH6" else
                0.40 if g in ("GATA6", "JAG1") else
                0.30  # NOTCH1 - not primarily ASD
            ),
            "vsd": rng.random() < (
                0.30 if g in ("NKX2-5", "GATA4") else
                0.20 if g == "TBX5" else
                0.25 if g == "TBX20" else
                0.20 if g == "GATA6" else
                0.10
            ),
            "tof": rng.random() < (
                0.40 if g == "GATA6" else
                0.20 if g == "JAG1" else
                0.08 if g == "NKX2-5" else
                0.05
            ),
            "avsd": rng.random() < (
                0.18 if g == "GATA4" else
                0.15 if g in ("GATA6", "JAG1") else
                0.10 if g == "TBX5" else
                0.05
            ),
            "pda": rng.random() < (
                0.28 if g == "GATA6" else
                0.10 if g == "JAG1" else
                0.05
            ),
            "bav": rng.random() < (
                0.90 if g == "NOTCH1" else
                0.08
            ),
            "pps": rng.random() < (
                0.93 if g == "JAG1" else
                0.05
            ),
            "coarctation": rng.random() < (
                0.08 if g == "NOTCH1" else
                0.03
            ),
            "mvp": rng.random() < (
                0.45 if g == "TBX20" else
                0.10
            ),
            "lvnc": rng.random() < (
                0.12 if g == "TBX20" else
                0.08 if g == "NKX2-5" else
                0.03
            ),
            # Conduction
            "av_block": rng.random() < (
                0.65 if g == "NKX2-5" else
                0.72 if g == "TBX5" else
                0.10 if g == "MYH6" else
                0.05
            ),
            "sick_sinus": rng.random() < (
                0.60 if g == "MYH6" else
                0.15 if g == "NKX2-5" else
                0.08
            ),
            "pacemaker_implanted": rng.random() < (
                0.45 if g == "NKX2-5" else
                0.50 if g == "TBX5" else
                0.55 if g == "MYH6" else
                0.05
            ),
            # Extracardiac
            "radial_ray_anomaly": rng.random() < (0.98 if g == "TBX5" else 0.01),
            "pancreatic_agenesis": rng.random() < (0.50 if g == "GATA6" else 0.01),
            "neonatal_diabetes": rng.random() < (0.48 if g == "GATA6" else 0.01),
            "exocrine_pancreatic_insufficiency": rng.random() < (0.45 if g == "GATA6" else 0.01),
            "cholestasis": rng.random() < (
                0.88 if g == "JAG1" else
                0.05 if g == "GATA6" else
                0.01
            ),
            "butterfly_vertebrae": rng.random() < (0.80 if g == "JAG1" else 0.01),
            "posterior_embryotoxon": rng.random() < (0.78 if g == "JAG1" else 0.02),
            "liver_transplant": rng.random() < (0.15 if g == "JAG1" else 0.01),
            "calcific_avd": rng.random() < (
                0.50 if g == "NOTCH1" else
                0.10
            ),
            "late_dcm": rng.random() < (
                0.20 if g == "TBX20" else
                0.15 if g == "MYH6" else
                0.05
            ),
            "post_repair_arrhythmia": rng.random() < (
                0.65 if g == "MYH6" else
                0.40 if g == "NKX2-5" else
                0.35 if g == "TBX5" else
                0.20
            ),
            # Surgical interventions
            "cardiac_surgery": rng.random() < (
                0.80 if g == "TBX5" else
                0.70 if g in ("NKX2-5", "GATA4", "GATA6") else
                0.65 if g == "JAG1" else
                0.55 if g == "TBX20" else
                0.45 if g == "NOTCH1" else
                0.55  # MYH6
            ),
            "avr_performed": rng.random() < (0.35 if g == "NOTCH1" else 0.03),
        }
        pts.append(pt)
    return pts


# Pre-build cohorts at import time
_ALL_COHORTS: dict = {}
for _idx, _gd in enumerate(CHD_GENES):
    _seed = SEED_BASE + _idx
    _ALL_COHORTS[_gd["gene"]] = _make_cohort(_gd, _seed)


def _pct(pts: list, key: str) -> int:
    return round(100 * sum(1 for p in pts if p.get(key)) / max(len(pts), 1))


def get_overview() -> dict:
    all_pts = [p for pts in _ALL_COHORTS.values() for p in pts]
    genes = [g["gene"] for g in CHD_GENES]
    return {
        "atlas_name": "Congenital Heart Disease (CHD) Atlas",
        "atlas_subtitle": (
            "Complete 8-Gene Hereditary Structural Congenital Heart Disease Reference — "
            "NKX2-5 · GATA4 · TBX5 · TBX20 · GATA6 · JAG1 · NOTCH1 · MYH6"
        ),
        "n_genes": 8,
        "n_patients": len(all_pts),
        "seeds": "1270–1277",
        "genes": genes,
        "description": (
            "This atlas covers eight primary hereditary structural congenital heart disease genes in clinical genetics. "
            "Cardiac transcription factor network: NKX2-5 (ASD + progressive AV block — PATHOGNOMONIC; lifelong ECG mandatory even after repair), "
            "GATA4 (ASD + VSD + AVSD — GATA4/TBX5 protein-protein interaction drives familial ASD), and "
            "TBX5 (Holt-Oram syndrome — ASD + RADIAL RAY ANOMALY PATHOGNOMONIC — all radial ray children need ECHO), "
            "TBX20 (ASD + mitral valve prolapse + late-onset DCM — annual ECHO lifelong required). "
            "Outflow tract + multi-organ: GATA6 (TOF + ASD + PDA + PANCREATIC AGENESIS — "
            "neonatal diabetes + CHD = GATA6 until proven otherwise; exocrine pancreatic insufficiency). "
            "Notch signaling pathway: JAG1 (Alagille syndrome — PERIPHERAL PULMONARY STENOSIS 90-97% + cholestasis + "
            "butterfly vertebrae — biliary interventions HIGH RISK — DO NOT do Kasai if JAG1 bile duct paucity), and "
            "NOTCH1 (BAV + calcific aortic valve disease — TAVR uncertain in young genetic BAV — prefer surgical AVR; "
            "JAG1 is Notch ligand: same pathway). "
            "Atrial sarcomere: MYH6 (familial ASD type 3 + sick sinus syndrome — pacemaker risk near-certain after atrial surgery; "
            "alpha-MHC preferentially atrial; FONTAN/Glenn + MYH6 = high arrhythmia mortality risk)."
        ),
        "aggregate_clinical": {
            "asd_pct": _pct(all_pts, "asd"),
            "vsd_pct": _pct(all_pts, "vsd"),
            "tof_pct": _pct(all_pts, "tof"),
            "avsd_pct": _pct(all_pts, "avsd"),
            "bav_pct": _pct(all_pts, "bav"),
            "pps_pct": _pct(all_pts, "pps"),
            "av_block_pct": _pct(all_pts, "av_block"),
            "sick_sinus_pct": _pct(all_pts, "sick_sinus"),
            "pacemaker_pct": _pct(all_pts, "pacemaker_implanted"),
            "radial_ray_pct": _pct(all_pts, "radial_ray_anomaly"),
            "pancreatic_agenesis_pct": _pct(all_pts, "pancreatic_agenesis"),
            "cholestasis_pct": _pct(all_pts, "cholestasis"),
            "butterfly_vertebrae_pct": _pct(all_pts, "butterfly_vertebrae"),
            "calcific_avd_pct": _pct(all_pts, "calcific_avd"),
            "late_dcm_pct": _pct(all_pts, "late_dcm"),
            "cardiac_surgery_pct": _pct(all_pts, "cardiac_surgery"),
            "post_repair_arrhythmia_pct": _pct(all_pts, "post_repair_arrhythmia"),
        },
        "drug_alerts": [
            {
                "title": "NKX2-5 + TBX5 — AV BLOCK IS PROGRESSIVE; PACEMAKER THRESHOLD LOWER (Mobitz II → pacemaker)",
                "body": (
                    "In NKX2-5 and TBX5 (Holt-Oram) carriers, AV block is progressive and INDEPENDENT of structural CHD repair. "
                    "Patients who had ASD corrected in childhood can develop complete AV block decades later. "
                    "Annual ECG/Holter is mandatory for life even after successful surgical repair. "
                    "PACEMAKER THRESHOLD IS LOWER THAN IDIOPATHIC AV BLOCK: implant at 2nd degree Mobitz II "
                    "(do not wait for complete block — syncope risk is high in genetic CHD-associated AV block)."
                ),
            },
            {
                "title": "TBX5 (Holt-Oram) — RADIAL RAY ANOMALY = MANDATORY ECHO + ECG IN EVERY CHILD",
                "body": (
                    "Any child with a radial ray anomaly (triphalangeal thumb, absent/hypoplastic thumb, radial aplasia) "
                    "MUST receive ECHO + 12-lead ECG BEFORE any orthopaedic or hand surgery. "
                    "Holt-Oram conduction defects are present even when ASD is small or absent — "
                    "do NOT reassure parents that mild limb anomaly means mild cardiac disease. "
                    "Expressivity is inverse: mildest limb → may have most severe AV block."
                ),
            },
            {
                "title": "JAG1 (Alagille) — DO NOT PERFORM KASAI HEPATOPORTOENTEROSTOMY (bile duct paucity ≠ biliary atresia)",
                "body": (
                    "Alagille syndrome causes intrahepatic bile duct PAUCITY (NOT atresia). "
                    "Kasai hepatoportoenterostomy is indicated for biliary atresia (extrahepatic obstruction), "
                    "NOT for Alagille. Performing Kasai in Alagille = surgical harm with no benefit and high complication risk. "
                    "Distinguish: liver biopsy (bile duct paucity) + MRCP (patent extrahepatic ducts) + JAG1 testing. "
                    "All biliary interventions (ERCP, stenting) also carry very high risk in Alagille."
                ),
            },
            {
                "title": "GATA6 — NEONATAL DIABETES + CHD = GATA6; PANCREATIC ENZYME REPLACEMENT MANDATORY FROM BIRTH",
                "body": (
                    "Neonatal diabetes mellitus (NDM) in a neonate with congenital heart disease (especially TOF, PDA, ASD) "
                    "is GATA6 until proven otherwise — start testing immediately. "
                    "GATA6 LOF causes both endocrine (insulin-dependent diabetes) and exocrine pancreatic insufficiency. "
                    "Pancreatic enzyme replacement therapy (PERT: CREON) must be started from birth alongside insulin, "
                    "fat-soluble vitamin supplementation (A, D, E, K), and CGM. "
                    "Do NOT treat as Type 1 DM without ruling out exocrine insufficiency."
                ),
            },
            {
                "title": "NOTCH1 (BAV) — TAVR UNCERTAIN IN YOUNG GENETIC BAV; PREFER SURGICAL AVR",
                "body": (
                    "Bicuspid aortic valve associated with NOTCH1 LOF has abnormal anatomy (elliptical annulus, "
                    "supra-annular leaflets) that increases risk of paravalvular leak and device migration with TAVR. "
                    "Long-term TAVR durability in young patients (<65 yr) with genetic BAV is unknown. "
                    "Current recommendation: surgical AVR preferred in NOTCH1-BAV carriers under 65 years. "
                    "Ross procedure (pulmonary autograft) is an option in experienced centers for young adults."
                ),
            },
            {
                "title": "MYH6 — PACEMAKER RISK NEAR-CERTAIN AFTER ASD REPAIR; INFORM SURGEON PRE-OPERATIVELY",
                "body": (
                    "MYH6 (alpha-MHC) LOF causes sick sinus syndrome — often silent before ASD repair. "
                    "After ASD device or surgical closure, sick sinus is unmasked in the majority of MYH6 carriers. "
                    "Inform the cardiac surgeon pre-operatively that the patient is MYH6 positive — "
                    "plan for simultaneous pacemaker implantation or have the lead access available intraoperatively. "
                    "FONTAN/Glenn circulation + MYH6-related arrhythmia is haemodynamically unstable and life-threatening."
                ),
            },
        ],
        "clinical_pearls": [
            "NKX2-5: ASD + PROGRESSIVE AV BLOCK = NKX2-5; AV block emerges decades after repair → annual ECG mandatory for life.",
            "GATA4: ASD + VSD without conduction defects → GATA4; GATA4/TBX5 interaction site is most penetrant allele for ASD.",
            "TBX5: RADIAL RAY ANOMALY + CHD → Holt-Oram; EVERY radial ray child needs ECHO + ECG BEFORE hand surgery.",
            "TBX20: ASD + MVP + late DCM in repaired CHD survivor → TBX20; annual ECHO lifelong required.",
            "GATA6: neonatal diabetes + CHD (especially TOF) = GATA6; pancreatic agenesis → PERT from birth.",
            "JAG1: peripheral pulmonary stenosis + neonatal cholestasis + butterfly vertebrae = Alagille; DO NOT Kasai.",
            "NOTCH1: BAV + family history of calcific AVD in 40s-50s = NOTCH1; prefer surgical AVR over TAVR in young.",
            "MYH6: ASD + sick sinus syndrome → MYH6; pacemaker near-certain after ASD repair — counsel pre-operatively.",
            "ALL CHD GENES: first-degree relatives MUST have ECHO + ECG cascade screening.",
            "CASCADE TESTING: any CHD gene family → screen siblings + parents before elective surgery.",
        ],
    }


def get_breakdown() -> dict:
    out: dict = {}
    for gd in CHD_GENES:
        pts = _ALL_COHORTS[gd["gene"]]
        out[gd["gene"]] = {
            "gene": gd["gene"],
            "protein": gd["protein"],
            "alias": gd["alias"],
            "aa": gd["aa"],
            "kDa": gd["kDa"],
            "locus": gd["locus"],
            "omim_gene": gd["omim_gene"],
            "omim_disease": gd["omim_disease"],
            "inheritance": gd["inheritance"],
            "gene_class": gd["gene_class"],
            "phenotype": gd["phenotype"],
            "hallmark": gd["hallmark"],
            "treatment_alert": gd["treatment_alert"],
            "key_ddx": gd["key_ddx"],
            "structural_defect": gd["structural_defect"],
            "conduction_defect": gd["conduction_defect"],
            "extracardiac": gd["extracardiac"],
            "cohort_n": len(pts),
            "stats": {
                "asd_pct": _pct(pts, "asd"),
                "vsd_pct": _pct(pts, "vsd"),
                "tof_pct": _pct(pts, "tof"),
                "avsd_pct": _pct(pts, "avsd"),
                "pda_pct": _pct(pts, "pda"),
                "bav_pct": _pct(pts, "bav"),
                "pps_pct": _pct(pts, "pps"),
                "coarctation_pct": _pct(pts, "coarctation"),
                "mvp_pct": _pct(pts, "mvp"),
                "lvnc_pct": _pct(pts, "lvnc"),
                "av_block_pct": _pct(pts, "av_block"),
                "sick_sinus_pct": _pct(pts, "sick_sinus"),
                "pacemaker_pct": _pct(pts, "pacemaker_implanted"),
                "radial_ray_pct": _pct(pts, "radial_ray_anomaly"),
                "pancreatic_agenesis_pct": _pct(pts, "pancreatic_agenesis"),
                "neonatal_diabetes_pct": _pct(pts, "neonatal_diabetes"),
                "exocrine_pi_pct": _pct(pts, "exocrine_pancreatic_insufficiency"),
                "cholestasis_pct": _pct(pts, "cholestasis"),
                "butterfly_vertebrae_pct": _pct(pts, "butterfly_vertebrae"),
                "posterior_embryotoxon_pct": _pct(pts, "posterior_embryotoxon"),
                "liver_transplant_pct": _pct(pts, "liver_transplant"),
                "calcific_avd_pct": _pct(pts, "calcific_avd"),
                "late_dcm_pct": _pct(pts, "late_dcm"),
                "post_repair_arrhythmia_pct": _pct(pts, "post_repair_arrhythmia"),
                "cardiac_surgery_pct": _pct(pts, "cardiac_surgery"),
                "avr_pct": _pct(pts, "avr_performed"),
                "severity_severe_pct": round(100 * sum(1 for p in pts if p["severity"] == "severe") / 40),
                "severity_moderate_pct": round(100 * sum(1 for p in pts if p["severity"] == "moderate") / 40),
            },
        }
    return out


def get_definitions() -> dict:
    return {
        "terms": [
            {
                "term": "NKX2-5 — ASD + Progressive AV Block (Independent of Structural Repair)",
                "definition": (
                    "NKX2-5 is a cardiac master homeobox transcription factor required for cardiac looping, septation, "
                    "and conduction system development. NKX2-5 LOF → Connexin-40 (Cx40) downregulation → impaired AV nodal conduction → "
                    "progressive AV block (1st → 2nd → 3rd degree). CRITICAL INSIGHT: AV block in NKX2-5 is PROGRESSIVE "
                    "and INDEPENDENT of whether ASD or other structural CHD has been surgically repaired. "
                    "Patients corrected in childhood develop high-degree AV block in adulthood. "
                    "PACEMAKER THRESHOLD: lower than idiopathic AV block — implant at 2nd degree Mobitz II "
                    "(do not wait for complete block). Lifelong annual ECG/Holter is mandatory for all NKX2-5 carriers."
                ),
            },
            {
                "term": "TBX5 / Holt-Oram Syndrome — Radial Ray Anomaly + CHD + Conduction Defects",
                "definition": (
                    "TBX5 is a T-box TF expressed in developing forelimb (but NOT hindlimb — explains upper limb specificity) and heart. "
                    "TBX5/GATA4 physical interaction → atrial septation; TBX5 activates Cx40 → AV conduction. "
                    "Holt-Oram syndrome (TBX5 LOF): RADIAL RAY ANOMALY (triphalangeal thumb → absent radius) + "
                    "ASD (85%) + conduction defects (AV block 75%). "
                    "KEY RULE: ANY radial ray anomaly in a child → ECHO + ECG BEFORE any surgery. "
                    "Conduction defects occur INDEPENDENTLY of structural CHD severity — mild limb + severe AV block possible. "
                    "Lower limb NORMAL — if lower limb involved → not Holt-Oram (consider TAR, Fanconi)."
                ),
            },
            {
                "term": "GATA4 — ASD/VSD via TBX5-GATA4 Interaction Site Mutations",
                "definition": (
                    "GATA4 encodes a zinc finger TF required for cardiac septation and gonadal development. "
                    "GATA4 physically binds TBX5 at the G296 interaction interface — G296S mutation disrupts this binding → familial ASD. "
                    "GATA4 mutations cause: ASD (55-65%), VSD (25-30%), AVSD (10-20%). "
                    "NO progressive AV block (contrast NKX2-5/TBX5). NO radial ray anomaly (contrast TBX5). "
                    "Del 8p23.1 (genomic): GATA4 deletion with flanking genes → more complex CHD + neurodevelopmental delay. "
                    "GONADAL: rare premature ovarian failure in females with GATA4 variants."
                ),
            },
            {
                "term": "TBX20 — ASD + Mitral Valve Prolapse + Late-Onset DCM",
                "definition": (
                    "TBX20 is a T-box TF interacting with NKX2-5, GATA4, and GATA5 for atrial and valve development. "
                    "TBX20 LOF → ASD + mitral/tricuspid valve prolapse/regurgitation + LVNC (left ventricular non-compaction). "
                    "CRITICAL LATE EFFECT: TBX20 is expressed in adult myocardium — progressive DCM emerges years/decades after "
                    "initial CHD repair. Annual ECHO is mandatory for ALL TBX20 carriers throughout adult life, "
                    "even if initial CHD successfully repaired. DCM management: standard HFrEF therapy; ICD if LVEF <35%; "
                    "heart transplant if end-stage."
                ),
            },
            {
                "term": "GATA6 — TOF + Pancreatic Agenesis (Neonatal Diabetes + CHD = GATA6)",
                "definition": (
                    "GATA6 is a zinc finger TF expressed in heart (outflow tract), pancreatic progenitors, and hepatoblasts. "
                    "GATA6 LOF → TOF (35-45%) + ASD + PDA + AVSD (outflow tract predominant CHD) + "
                    "PANCREATIC AGENESIS (pathognomonic): complete or partial pancreatic hypoplasia → "
                    "insulin-dependent neonatal diabetes (from birth) + exocrine pancreatic insufficiency (steatorrhea, fat-soluble vitamin deficiency). "
                    "CLINICAL RULE: NEONATAL DIABETES MELLITUS + CHD = GATA6 until proven otherwise. "
                    "Management: insulin from birth + PERT (pancreatic enzyme replacement) + fat-soluble vitamins (A,D,E,K) + CGM. "
                    "Biliary atresia/paucity in 5-10%. >70% de novo — low familial recurrence."
                ),
            },
            {
                "term": "JAG1 / Alagille Syndrome — Peripheral Pulmonary Stenosis + Bile Duct Paucity (NOT Biliary Atresia)",
                "definition": (
                    "JAG1 encodes Jagged-1, a transmembrane Notch signaling ligand. "
                    "Alagille syndrome (ALGS1) pentad: (1) PERIPHERAL PULMONARY STENOSIS (90-97%, PATHOGNOMONIC — diffuse bilateral PPS); "
                    "(2) CHOLESTASIS (bile duct paucity on biopsy); (3) BUTTERFLY VERTEBRAE; "
                    "(4) POSTERIOR EMBRYOTOXON; (5) ALAGILLE FACIES (triangular face). "
                    "CRITICAL RULE: Bile duct PAUCITY ≠ biliary atresia. "
                    "DO NOT perform Kasai hepatoportoenterostomy in Alagille — causes surgical harm with no benefit. "
                    "Distinguish with liver biopsy + MRCP + JAG1 sequencing. "
                    "Pruritus treatment: odevixibat (FDA 2023 for ALGS). Intracranial vascular anomalies → screen MRA."
                ),
            },
            {
                "term": "NOTCH1 — Bicuspid Aortic Valve + Calcific Aortic Valve Disease; TAVR Uncertain in Young",
                "definition": (
                    "NOTCH1 is the Notch receptor (JAG1 is its ligand — same signaling pathway as Alagille syndrome). "
                    "NOTCH1 LOF → BAV (right-left cusp fusion most common) → accelerated calcific aortic valve disease (CAVD). "
                    "Mechanism: NOTCH1 LOF → de-repression of RUNX2/BMP2 → osteoblast-like transdifferentiation of "
                    "valve interstitial cells → calcium deposition. CAVD in NOTCH1 BAV occurs earlier (40s-50s) than sporadic BAV. "
                    "TAVR in genetic BAV under 65 years: BAV anatomy (elliptical annulus) → higher paravalvular leak + migration risk; "
                    "long-term durability unknown. Prefer SURGICAL AVR in NOTCH1 carriers <65yr. "
                    "Aortic root surveillance: ECHO/MRI every 2-3 years."
                ),
            },
            {
                "term": "MYH6 — Familial ASD + Sick Sinus Syndrome; Alpha-MHC Atrial Specificity",
                "definition": (
                    "MYH6 encodes alpha-myosin heavy chain (alpha-MHC) — the dominant myosin isoform in HUMAN ATRIA. "
                    "This atrial specificity explains why MYH6 LOF causes an atrial-predominant phenotype: "
                    "ASD (type 3) + sick sinus syndrome + post-operative atrial arrhythmia + DCM (allelic). "
                    "CRITICAL SURGICAL RULE: MYH6 carriers undergoing ASD repair have near-certain sick sinus unmasking post-closure. "
                    "Inform the surgeon pre-operatively — plan for simultaneous pacemaker lead implantation. "
                    "FONTAN/Glenn + MYH6 arrhythmia: haemodynamically unstable and life-threatening. "
                    "DCM surveillance: annual ECHO for all MYH6 carriers even without structural CHD."
                ),
            },
            {
                "term": "Cardiac Transcription Factor Network — NKX2-5 / GATA4 / TBX5 Triad",
                "definition": (
                    "Cardiac septation requires coordinated action of NKX2-5, GATA4, and TBX5: "
                    "TBX5 physically binds GATA4 → activates ANF, MLC2V, Connexin-40; "
                    "GATA4 binds NKX2-5 → ventricular gene program; "
                    "NKX2-5 + GATA4 → TBX20 expression; all three bind same cardiac enhancers. "
                    "Mutations in ANY of these three genes cause familial ASD — with overlapping but distinguishable features: "
                    "NKX2-5 = ASD + AV block (no limbs); GATA4 = ASD/VSD (no AV block, no limbs); "
                    "TBX5 = ASD + AV block + RADIAL RAY ANOMALY. "
                    "Digenic disease: some families have mutations in two network genes (e.g. NKX2-5 + GATA4) → more severe CHD."
                ),
            },
            {
                "term": "JAG1/NOTCH1 Pathway — Same Signaling Axis; Clinically Distinct Phenotypes",
                "definition": (
                    "JAG1 (Alagille syndrome) encodes the Notch ligand; NOTCH1 encodes the Notch receptor — same signaling axis. "
                    "JAG1 LOF → Notch signaling failure in biliary epithelium → bile duct paucity (Alagille); "
                    "pulmonary vascular → PPS; vertebral → butterfly vertebrae; ocular → posterior embryotoxon. "
                    "NOTCH1 LOF → de-repression of osteogenic pathway in valve cells → BAV + calcific aortic valve disease. "
                    "CLINICAL DISTINCTION: JAG1 = neonatal/childhood presentation (cholestasis, PPS, dysmorphic); "
                    "NOTCH1 = adult presentation (BAV, calcific AS, aortopathy). "
                    "Genetic interaction: some families carry variants in BOTH JAG1 and NOTCH1 → more severe valve phenotype."
                ),
            },
        ]
    }
