#!/usr/bin/env python3
"""Hereditary-Congenital-Heart-Disease-Atlas — Complete 8-Gene Hereditary Congenital
Heart Disease (CHD) Atlas
(GATA4 · TBX5 · NKX2-5 · NOTCH1 · JAG1 · CHD7 · TFAP2B · TBX1).

GATA4   (GATA binding protein 4; 442 aa; 8p23.1; AD;
         Isolated Congenital Heart Disease (ASD/VSD/TOF);
         GATA4 haploinsufficiency — ASD type II MOST COMMON (40%), VSD (30%), TOF (20%);
         NKX2-5 physical interaction partner — clusters at NKX2-5 binding domain;
         Incomplete penetrance (~85%); NO extracardia features (key DDx);
         p.Gly296Ser European CHD variant; GATA DNA-binding domain;
         seed SEED_BASE+0).
TBX5    (T-box transcription factor 5; 518 aa; 12q24.21; AD;
         Holt-Oram Syndrome (HOS);
         BILATERAL UPPER LIMB ANOMALY + HEART DEFECT = PATHOGNOMONIC COMBINATION;
         Limb: thumb anomaly (hypoplastic/triphalangeal/absent) MOST SPECIFIC;
         Heart: ASD II (50%), ASD+VSD (25%), cardiac conduction defects;
         100% penetrance; limb severity does NOT correlate with heart severity;
         p.Tyr111Cys most common HOS mutation (DNA-binding domain);
         seed SEED_BASE+1).
NKX2-5  (NK2 homeobox 5; 324 aa; 5q35.1; AD;
         Isolated CHD with AV Conduction Defects;
         ASD + PROGRESSIVE AV BLOCK PATHOGNOMONIC — block worsens EVEN AFTER ASD REPAIR;
         VSD, TOF, AVSD also associated; 50% penetrance;
         Unique: AV conduction disease independent of septal anatomy;
         p.Arg25Cys homeodomain mutation most common;
         Pacemaker required in 30-40% by adulthood;
         seed SEED_BASE+2).
NOTCH1  (Notch receptor 1; 2555 aa; 9q34.3; AD;
         Bicuspid Aortic Valve (BAV) and Aortopathy;
         BICUSPID AORTIC VALVE + ASCENDING AORTIC DILATATION PATHOGNOMONIC;
         BAV = MOST COMMON CHD in general population (0.5-2%);
         CALCIFIC AORTIC STENOSIS most common complication (5th decade);
         Aortic dissection risk independent of valve function;
         Echo surveillance annually MANDATORY;
         NOTCH1 haploinsufficiency → reduced NOTCH target gene expression;
         seed SEED_BASE+3).
JAG1    (Jagged canonical Notch ligand 1; 1218 aa; 20p12.2; AD;
         Alagille Syndrome (ALGS);
         BUTTERFLY VERTEBRAE (anterior arch defect) 95% — PATHOGNOMONIC on spinal X-ray;
         Posterior embryotoxon 78% — anterior eye anomaly on slit-lamp PATHOGNOMONIC;
         CHD 94% — pulmonary arterial stenosis/hypoplasia or TOF;
         Cholestatic liver disease 80% — paucity of intrahepatic bile ducts on biopsy;
         p.Gly274Asp European founder; ~70% LOF (nonsense/frameshift);
         50% require liver transplant;
         seed SEED_BASE+4).
CHD7    (Chromodomain helicase DNA-binding protein 7; 2997 aa; 8q12.2; AD;
         CHARGE Syndrome;
         COLOBOMA + HEART DEFECT + CHOANAL ATRESIA = PATHOGNOMONIC core triad;
         C-Coloboma + H-Heart + A-choanal Atresia + R-Retardation + G-Genital + E-Ear;
         SEMICIRCULAR CANAL APLASIA on MRI = most pathognomonic radiological finding;
         75% CHD: conotruncal (TOF, double outlet RV, truncus arteriosus);
         Almost all LOF mutations de novo; olfactory bulb hypoplasia → anosmia;
         seed SEED_BASE+5).
TFAP2B  (Transcription factor AP-2 beta; 463 aa; 6p24.3; AD;
         Char Syndrome;
         PATENT DUCTUS ARTERIOSUS + FACIAL DYSMORPHISM + HAND ANOMALIES PATHOGNOMONIC;
         PDA: structurally abnormal ductal tissue; closure surgical/catheter in infancy;
         Face: flat nasal bridge + ptosis + low-set ears + fishmouth lips;
         Hand: shortening middle phalanges + 5th finger clinodactyly;
         Ultra-rare: ~30 families worldwide 2026;
         seed SEED_BASE+6).
TBX1    (T-box transcription factor 1; 504 aa; 22q11.21; AD;
         22q11.2 Deletion Syndrome (DiGeorge / Velocardiofacial Syndrome);
         CONOTRUNCAL HEART DEFECT + HYPOCALCEMIA + T-CELL LYMPHOPENIA = PATHOGNOMONIC TRIAD;
         IAA type B + truncus arteriosus + TOF/absent-pulmonary-valve PATHOGNOMONIC conotruncal pattern;
         MOST COMMON chromosomal microdeletion (1:4,000 live births);
         HYPOCALCEMIA: hypoparathyroidism (3rd pharyngeal pouch → parathyroid absent/hypoplastic);
         T-CELL LYMPHOPENIA: thymus from 3rd pharyngeal pouch → absent/hypoplastic;
         Psychiatric: schizophrenia 25%, ADHD 40%, anxiety, ASD;
         FISH/CMA detects deletion; TBX1 point mutations rare;
         seed SEED_BASE+7).
320-patient aggregate cohort (8 × 40, seeds 2254-2261).
"""

import random

SEED_BASE = 2254

CHD_GENES = [
    # -- GATA4 — Isolated Congenital Heart Disease (ASD/VSD/TOF) -----------------------
    {
        "gene": "GATA4",
        "alt_name": (
            "GATA4 (GATA4-442aa-8p23.1 / AD — Isolated-Congenital-Heart-Disease-ASD-VSD-TOF — "
            "GATA4-Haploinsufficiency-Cardiac-Transcription-Factor — "
            "ASD-II-MOST-COMMON-40pct-VSD-30pct-TOF-20pct — "
            "NKX2-5-Physical-Interaction-Partner-Mutations-Cluster-Binding-Domain — "
            "Incomplete-Penetrance-85pct-No-Extracardia-Features-KEY-DDx — "
            "p.Gly296Ser-European-GATA-DNA-Binding-Domain)"
        ),
        "protein": (
            "GATA4 -- 8p23.1 AD -- GATA4-442aa -- "
            "GATA-Binding-Protein-4-Zinc-Finger-Transcription-Factor-50kDa-Cardiac-Nucleus -- "
            "Isolated-CHD-OMIM-600576 -- "
            "ASD-II-MOST-COMMON-CHD-40pct-Sinus-Venosus-ASD-Ostium-Secundum -- "
            "VSD-PERIMEMBRANOUS-30pct-Muscular-VSD-15pct -- "
            "TOF-Tetralogy-Fallot-20pct-GATA4-Conoventricular-VSD-Overriding-Aorta-RVOT-Obstruction -- "
            "NKX2-5-Protein-Interaction-Physical-Binding-Domain-Mutations-Cluster-aa185-221 -- "
            "TBX5-Co-Occupancy-Cardiac-Enhancers-GATA4-TBX5-Co-Activate-ANF-BNP-MYH6 -- "
            "Incomplete-Penetrance-85pct-Variable-Expressivity-Same-Family-Different-CHDs -- "
            "NO-Limb-Anomalies-KEY-DDx-TBX5-Holt-Oram -- "
            "NO-AV-Block-KEY-DDx-NKX2-5 -- "
            "NO-Extracardia-Features-Isolated-Cardiac-UNIQUE-GATA4 -- "
            "OMIM-Gene-GATA4-600576-Disease-CHD-614980"
        ),
        "locus": "8p23.1",
        "protein_size": "442 aa / 50 kDa",
        "inheritance": (
            "AD (autosomal dominant haploinsufficiency); "
            "De novo + familial; incomplete penetrance (~85%); variable expressivity; "
            "p.Gly296Ser: European CHD variant; DNA-binding domain; "
            "p.Glu359del: loss of TBX5 interaction; familial ASD; "
            "p.Ala411Val: septation defect; TOF; "
            "p.Lys325Glu: severe TOF; AV canal defect; "
            "CK: normal (not a myopathy); "
            "Onset: present at birth; prenatal echo in 18-20w; "
            "ASD: ostium secundum most common (80% of GATA4-ASD); "
            "VSD: perimembranous (65%) or muscular (35%); "
            "TOF: VSD + RVOT obstruction + overriding aorta + RVH; "
            "AVSD: atrioventricular septal defect — less common; "
            "Associated CHD in same family: different defects in different members (variable expressivity)"
        ),
        "key_features": [
            "GATA4 HAPLOINSUFFICIENCY — ASD type II MOST COMMON (40%), VSD (30%), TOF (20%): isolated cardiac defects",
            "NO extracardia features — isolated CHD without limb/eye/face/thymus anomalies: KEY DDx from TBX5/NKX2-5/JAG1/CHD7/TBX1",
            "VARIABLE EXPRESSIVITY — same family: father with ASD, child with VSD, another child unaffected",
            "INCOMPLETE PENETRANCE (~85%) — mutation carrier may have structurally normal heart",
            "NKX2-5 PHYSICAL INTERACTION — GATA4 variants clustering at NKX2-5 binding domain (aa185-221) cause ASD+AV block: overlap with NKX2-5 phenotype",
            "TBX5 CO-OCCUPANCY — GATA4+TBX5 co-activate cardiac enhancers; GATA4+TBX5 compound heterozygosity → severe combined CHD",
            "ASD: ostium secundum (80%), may close spontaneously <3mm; Device closure or surgical repair for significant shunts (Qp:Qs > 1.5:1)",
            "TOF repair: neonatal/infant complete surgical repair; residual RVOT gradient + PR monitoring lifelong",
        ],
        "treatment": (
            "Surgical/interventional: "
            "ASD: device closure (Amplatzer) for secundum ASD ≥5mm ≥2 years; surgical repair for complex/sinus venosus. "
            "VSD: spontaneous closure rate 30-40% by age 2 years (muscular), 20-25% (perimembranous); "
            "repair if large shunt, heart failure, pulmonary hypertension, or growth failure. "
            "TOF: complete repair 3-6 months; pulmonary valved conduit if RVOT severely hypoplastic. "
            "Medical: diuretics (furosemide) + ACE inhibitor for heart failure pre-repair. "
            "Infective endocarditis prophylaxis: 6 months post-repair or lifelong if residual defect. "
            "Genetics: cascade family screening — echocardiogram all 1st-degree relatives; "
            "penetrance ~85% so echo even in phenotypically 'normal' family members. "
            "Genetic counselling: 50% transmission risk; prenatal anomaly scan at 18-20 weeks; "
            "fetal echo at 22-24 weeks if parent affected."
        ),
        "monitoring": [
            "Echo: newborn; 6-monthly pre-repair; annually post-repair; lifelong surveillance for residual lesions",
            "ECG: annual; AV block monitoring even without NKX2-5 (GATA4+NKX2-5 interaction risk)",
            "TOF: RV function; residual PR; RVOT gradient; CMRI every 3-5 years post-repair",
            "VSD: annual echo; Doppler gradient; pulmonary pressure estimation",
            "Pulmonary hypertension: 6-minute walk test; catheter if Qp:Qs borderline",
            "Family cascade: echo all 1st-degree relatives; genetic testing (GATA4 sequencing)",
            "Pregnancy: high-risk obstetrics; maternal echo baseline + third trimester",
            "Reproductive: preimplantation genetic diagnosis (PGD) option for 50% transmission risk",
        ],
        "chd_types": ["ASD", "VSD", "TOF", "AVSD"],
        "pathognomonic": "NO extracardia features + isolated CHD (ASD/VSD/TOF) = GATA4 / NKX2-5 / TBX5 workup",
        "treatment_highlight": "Device/surgical repair; cascade echo in family; 50% transmission risk",
    },
    # -- TBX5 — Holt-Oram Syndrome -------------------------------------------------------
    {
        "gene": "TBX5",
        "alt_name": (
            "TBX5 (TBX5-518aa-12q24.21 / AD — Holt-Oram-Syndrome-HOS — "
            "BILATERAL-UPPER-LIMB-ANOMALY-PLUS-HEART-DEFECT-PATHOGNOMONIC-COMBINATION — "
            "THUMB-ANOMALY-Hypoplastic-Triphalangeal-Absent-MOST-SPECIFIC — "
            "ASD-II-50pct-ASD-VSD-25pct-Cardiac-Conduction-Defects — "
            "100pct-Penetrance-Limb-Severity-NOT-Correlated-Heart-Severity — "
            "p.Tyr111Cys-Most-Common-HOS-DNA-Binding-Domain)"
        ),
        "protein": (
            "TBX5 -- 12q24.21 AD -- TBX5-518aa -- "
            "T-Box-Transcription-Factor-5-58kDa-Cardiac-Limb-Development -- "
            "Holt-Oram-Syndrome-OMIM-142900 -- "
            "BILATERAL-UPPER-LIMB-ANOMALY-PATHOGNOMONIC-Both-Arms-Affected-Even-If-Asymmetric -- "
            "THUMB-ANOMALY-Hypoplastic-Triphalangeal-Absent-Thumb-MOST-SPECIFIC-Limb-Finding -- "
            "RADIAL-RAY-DEFECT-Radius-Hypoplasia-Aplasia-Radial-Club-Hand -- "
            "ASD-OSTIUM-SECUNDUM-50pct-MOST-COMMON-CHD-In-HOS -- "
            "ASD-PLUS-VSD-25pct-Complex-CHD-In-HOS -- "
            "CARDIAC-CONDUCTION-First-Degree-AV-Block-Sinus-Bradycardia-Common -- "
            "100pct-PENETRANCE-Every-Mutation-Carrier-Has-CHD-Or-Limb-Defect-Or-Both -- "
            "LIMB-HEART-SEVERITY-NOT-CORRELATED-Mild-Thumb-Hypoplasia-Can-Have-Severe-CHD -- "
            "p.Tyr111Cys-Most-Common-European-HOS-T-Box-DNA-Binding-Domain -- "
            "OMIM-Gene-TBX5-601620-Disease-HOS-142900"
        ),
        "locus": "12q24.21",
        "protein_size": "518 aa / 58 kDa",
        "inheritance": (
            "AD (autosomal dominant haploinsufficiency); "
            "100% penetrance — every mutation carrier has limb defect and/or CHD; "
            "Variable expressivity — wide spectrum from mild thumb hypoplasia to phocomelia; "
            "p.Tyr111Cys: most common European HOS mutation; T-box DNA-binding domain; ASD + thumb; "
            "p.Arg279Trp: T-box domain; severe CHD (AVSD) + phocomelia; "
            "p.Gly169Arg: T-box; moderate limb + ASD; "
            "Limb: bilateral involvement MANDATORY (even if asymmetric — both arms affected); "
            "Thumb anomaly: radially pre-axial — triphalangeal thumb, hypoplastic thumb, absent thumb; "
            "Wrist radiograph diagnostic: accessory/fused carpal bones, hypoplastic radius; "
            "CHD: ASD II (50%), ASD+VSD (25%), isolated VSD (10%), ASD+AVSD (10%), other (5%); "
            "Conduction: first-degree AV block common; sinus bradycardia; "
            "No cardiac chamber hypoplasia; no conotruncal defects (distinguishes from TBX1/CHD7)"
        ),
        "key_features": [
            "BILATERAL UPPER LIMB ANOMALY + HEART DEFECT = PATHOGNOMONIC COMBINATION: Holt-Oram syndrome",
            "THUMB ANOMALY = MOST SPECIFIC limb finding: hypoplastic, triphalangeal, or absent thumb; bilateral even if asymmetric",
            "100% PENETRANCE — every TBX5 mutation carrier has limb defect AND/OR CHD (high clinical utility)",
            "LIMB SEVERITY DOES NOT CORRELATE WITH HEART SEVERITY — mild thumb anomaly can coexist with severe CHD (TOF/AVSD)",
            "ASD type II (50%) MOST COMMON CHD in HOS; first-degree AV block present in most, even with normal cardiac anatomy",
            "NO LOWER LIMB INVOLVEMENT — bilateral upper extremity only; lower limb/foot anomalies → exclude HOS",
            "WRIST X-RAY: accessory carpal bones, fused carpals, hypoplastic/absent radius — diagnostic in subtle cases",
            "TBX5+GATA4 interaction: haploinsufficiency of both → severe AVSD; check GATA4 in complex HOS",
        ],
        "treatment": (
            "Cardiac: "
            "ASD: device closure (Amplatzer) for suitable secundum ASD; surgical repair for complex anatomy. "
            "VSD: repair if Qp:Qs >1.5:1, heart failure, or growth failure. "
            "AVSD: complete repair typically 3-6 months; valve annuloplasty for AV valve regurgitation. "
            "Conduction: annual ECG; Holter if symptomatic bradycardia; pacemaker if complete AV block. "
            "Orthopaedic: "
            "Thumb: opponensplasty for triphalangeal or hypoplastic thumb; prosthetic thumb if absent. "
            "Radius: distraction osteogenesis for radial club hand; Ilizarov frame; centralisation procedure. "
            "Occupational therapy: hand function; adaptive equipment; splints. "
            "Genetics: cascade testing — echo + limb X-ray all 1st-degree relatives; "
            "de novo in 15-20% (both parents unaffected); "
            "Prenatal: anomaly scan 18-20w (limb); fetal echo 22-24w (CHD); CVS/amniocentesis for TBX5. "
            "Reproductive: PGD available; 50% risk per pregnancy."
        ),
        "monitoring": [
            "Echo: baseline; annual post-repair; AV valve function (AVSD); residual shunt",
            "ECG: annual; Holter if palpitations/syncope; AV block surveillance",
            "Limb/hand: occupational therapy assessment; hand function; X-ray wrist annually in childhood",
            "Growth: height/weight; arm length discrepancy; fine motor development",
            "Family screening: echo + limb examination all 1st-degree relatives; TBX5 sequencing",
            "Pregnancy: maternal cardiac review; fetal echo 22-24 weeks; neonatal exam for limb anomaly",
            "School: IEP/504 if limb anomaly affects writing; adaptive technology",
            "LFTs/renal: not specifically affected; routine health checks",
        ],
        "chd_types": ["ASD", "VSD", "AVSD", "Conduction defect"],
        "pathognomonic": "Bilateral upper limb anomaly (thumb!) + heart defect = Holt-Oram syndrome TBX5",
        "treatment_highlight": "ASD/VSD repair; opponensplasty for thumb; cascade echo+limb X-ray",
    },
    # -- NKX2-5 — Isolated CHD with Progressive AV Block ---------------------------------
    {
        "gene": "NKX2-5",
        "alt_name": (
            "NKX2-5 (NKX2-5-324aa-5q35.1 / AD — Isolated-CHD-Progressive-AV-Block-PATHOGNOMONIC — "
            "ASD-PLUS-PROGRESSIVE-AV-BLOCK-WORSENS-EVEN-AFTER-ASD-REPAIR-KEY-CLINICAL-PEARL — "
            "VSD-TOF-AVSD-Also-Associated-50pct-Penetrance — "
            "Pacemaker-Required-30-40pct-Adulthood — "
            "p.Arg25Cys-Homeodomain-Most-Common)"
        ),
        "protein": (
            "NKX2-5 -- 5q35.1 AD -- NKX2-5-324aa -- "
            "NK2-Homeobox-5-Tinman-Homologue-35kDa-Cardiac-Transcription-Factor -- "
            "Isolated-CHD-AV-Block-OMIM-600584 -- "
            "ASD-PLUS-PROGRESSIVE-AV-CONDUCTION-DISEASE-PATHOGNOMONIC-COMBINATION -- "
            "AV-BLOCK-INDEPENDENT-OF-SEPTAL-ANATOMY-Unique-Among-CHD-Genes -- "
            "AV-BLOCK-WORSENS-EVEN-AFTER-ASD-SURGICAL-CLOSURE-CRITICAL-CLINICAL-PEARL -- "
            "VSD-PERIMEMBRANOUS-30pct-TOF-15pct-AVSD-10pct-Also-Associated -- "
            "PACEMAKER-REQUIRED-30-40pct-Adulthood-Progressive-Nature-Key -- "
            "50pct-PENETRANCE-Sporadic-De-Novo-Common -- "
            "p.Arg25Cys-Homeodomain-Most-Common-European-Variant -- "
            "GATA4-Physical-Interaction-Partner-Dual-GATA4-NKX2-5-Variant-Severe-AVSD -- "
            "OMIM-Gene-NKX2-5-600584-Disease-CHD-614980"
        ),
        "locus": "5q35.1",
        "protein_size": "324 aa / 35 kDa",
        "inheritance": (
            "AD (autosomal dominant haploinsufficiency); "
            "50% penetrance — sporadic/de novo common; familial clustering; "
            "p.Arg25Cys: homeodomain (HD) mutation; most common European variant; ASD + 1st-degree AVB; "
            "p.Gln187STOP: truncating; severe CHD + complete AVB; "
            "p.Ala119Val: HD; VSD + AVSD + complete AVB; "
            "CHD types: ASD II (50%), VSD (30%), TOF (15%), AVSD (10%), LV noncompaction (5%); "
            "AV block: present in 20% at diagnosis; progressive to high-grade in 30-40% by adulthood; "
            "Progressive nature: worsens even years after successful ASD closure — NOT related to hemodynamic unloading; "
            "Sick sinus syndrome: uncommon but reported; "
            "LV noncompaction cardiomyopathy: 5% of NKX2-5 carriers"
        ),
        "key_features": [
            "ASD + PROGRESSIVE AV BLOCK = PATHOGNOMONIC: NKX2-5 is the ONLY CHD gene where AV block is intrinsic and progressive",
            "AV BLOCK WORSENS AFTER ASD REPAIR — key clinical pearl: unlike other CHDs, NKX2-5-related AVB progresses INDEPENDENT of hemodynamic status",
            "PACEMAKER REQUIRED in 30-40% by adulthood (progressive high-grade AV block) — annual ECG + Holter MANDATORY",
            "50% PENETRANCE — lower than TBX5; sporadic/de novo common; family members with mutation may have isolated AVB without structural CHD",
            "LV NONCOMPACTION (5%): spongy LV myocardium on echo/CMR — screen NKX2-5 in isolated LVNC",
            "GATA4 INTERACTION: dual GATA4+NKX2-5 heterozygosity → severe AVSD; test both genes in AVSD",
            "VSD + progressive AV block in infancy → NKX2-5 first-line test even without ASD",
            "SURGICAL CAVEAT: ASD closure does NOT prevent or reverse AV block progression — cardiologist must follow lifelong",
        ],
        "treatment": (
            "Cardiac: "
            "ASD: device or surgical repair; does NOT prevent AV block progression. "
            "VSD/TOF: repair as per standard cardiac surgery guidelines. "
            "AVSD: complete repair; AV valve function monitoring. "
            "AV block: "
            "First-degree: annual 12-lead ECG + 24h Holter; "
            "Second-degree Mobitz II or third-degree: dual-chamber pacemaker (DDD-R); "
            "Threshold: high-grade AVB, PR >300ms with symptoms, escape rate <40bpm. "
            "LV noncompaction: ACE inhibitor/ARB; anticoagulation if EF <35%; ICD if sustained VT/low EF. "
            "IE prophylaxis: 6 months post-repair or lifelong if residual. "
            "Genetics: "
            "NKX2-5 sequencing + deletion/duplication panel. "
            "Family cascade: ECG + echo in all 1st-degree relatives (isolated AVB without CHD is an NKX2-5 phenotype). "
            "50% transmission risk per pregnancy; prenatal echo + pedigree."
        ),
        "monitoring": [
            "ECG: annual — PR interval trend; Wenckebach pattern; escape rhythm",
            "Holter: annual (or if palpitations/syncope/presyncope); even post-ASD repair",
            "Echo: baseline; post-repair annually; EF + RV function; LVNC screening",
            "CMR: if LVNC suspected on echo; trabeculation quantification (NC/C ratio >2.3)",
            "Pacemaker: annual device check; threshold; sensing; battery life",
            "Exercise testing: chronotropic incompetence; rate response programming in paced patients",
            "Family: ECG + echo all 1st-degree relatives; NKX2-5 genetic testing",
            "Pregnancy: high-risk OB; pacemaker function review; fetal echo 22-24 weeks",
        ],
        "chd_types": ["ASD", "VSD", "TOF", "AVSD", "LV noncompaction"],
        "pathognomonic": "ASD + progressive AV block (worsens post-repair) = NKX2-5 PATHOGNOMONIC",
        "treatment_highlight": "Pacemaker 30-40% adulthood; ASD repair does NOT prevent AVB progression",
    },
    # -- NOTCH1 — Bicuspid Aortic Valve and Aortopathy -----------------------------------
    {
        "gene": "NOTCH1",
        "alt_name": (
            "NOTCH1 (NOTCH1-2555aa-9q34.3 / AD — Bicuspid-Aortic-Valve-BAV-Aortopathy-PATHOGNOMONIC — "
            "BAV-MOST-COMMON-CHD-General-Population-0.5-2pct — "
            "CALCIFIC-AORTIC-STENOSIS-5th-Decade-Most-Common-Complication — "
            "AORTIC-DISSECTION-Risk-Independent-Valve-Function — "
            "Echo-Surveillance-ANNUALLY-MANDATORY)"
        ),
        "protein": (
            "NOTCH1 -- 9q34.3 AD -- NOTCH1-2555aa -- "
            "Notch-Receptor-1-300kDa-Type-I-Transmembrane-Cardiac-Vascular-Development -- "
            "Bicuspid-Aortic-Valve-BAV-OMIM-607093 -- "
            "BAV-MOST-COMMON-CHD-General-Population-0.5-2pct-1-in-50-Live-Births -- "
            "BICUSPID-AORTIC-VALVE-Two-Leaflets-Instead-Three-PATHOGNOMONIC-Echo-Finding -- "
            "ASCENDING-AORTIC-DILATATION-Aortopathy-Medial-Degeneration-PATHOGNOMONIC -- "
            "CALCIFIC-AORTIC-STENOSIS-5th-6th-Decade-Turbulent-Flow-Calcium-Deposition -- "
            "AORTIC-REGURGITATION-Early-Presentation-Young-Adults -- "
            "AORTIC-DISSECTION-Risk-Elevated-Independent-Of-Valve-Function-KEY-Clinical-Point -- "
            "NOTCH1-Haploinsufficiency-Calcification-Pathway-RUNX2-Osteoblastic-Phenotype -- "
            "Echo-Annual-Surveillance-MANDATORY-Valve-And-Aortic-Root-And-Ascending -- "
            "OMIM-Gene-NOTCH1-190198-Disease-BAV-109730"
        ),
        "locus": "9q34.3",
        "protein_size": "2555 aa / 300 kDa",
        "inheritance": (
            "AD (autosomal dominant haploinsufficiency); "
            "Highest-frequency Mendelian CHD gene in population (0.5-2%); "
            "Incomplete penetrance; NOTCH1 variants in 4-6% of BAV familial cases; "
            "p.Ser1600Thr: GWAS hit; rare pathogenic LOF more penetrant; "
            "EGF-repeat domain variants: reduce ligand binding (JAG1/DLL4 interaction); "
            "NICD (notch intracellular domain): haploinsufficiency → RUNX2 de-repression → calcification; "
            "BAV types: RL fusion (69%), RN fusion (22%), LN fusion (9%); "
            "Aortopathy: independent of valve type; aortic root + ascending dilatation; "
            "Associated: coarctation of aorta (10%), VSD (5%); "
            "Penetrance: BAV in 20-30% family members of NOTCH1 carriers; "
            "Calcific AS: 30% of BAV by age 50; accelerated vs tricuspid valve calcification"
        ),
        "key_features": [
            "BICUSPID AORTIC VALVE (BAV) = MOST COMMON CHD in general population (0.5-2%); NOTCH1 haploinsufficiency is key Mendelian cause",
            "PATHOGNOMONIC: two-leaflet aortic valve with raphe on echo/CT — valve opens like a fish mouth in systole",
            "ASCENDING AORTIC DILATATION (aortopathy) — independent of valve function; risk of dissection even with normally functioning BAV",
            "CALCIFIC AORTIC STENOSIS — most common complication; accelerated calcification vs tricuspid valve (NOTCH1 → RUNX2 de-repression → osteoblastic pathway)",
            "AORTIC DISSECTION RISK independent of current valve function — surveillance even with mild stenosis/regurgitation",
            "ANNUAL ECHO MANDATORY: valve gradient (aortic stenosis velocity), AR severity, aortic root + ascending aorta dimension",
            "SURGERY: TAVI/SAVR when severe AS (AVA <1.0 cm², mean gradient >40mmHg, symptoms); aortic replacement when >5.0cm",
            "FAMILY SCREENING: echo all 1st-degree relatives — BAV found in 20-30%; family members may present with isolated aortopathy",
        ],
        "treatment": (
            "Valve: "
            "Severe AS: TAVI (TAVR) for high/intermediate surgical risk ≥65yr; SAVR for younger patients; "
            "BAV-specific TAVI technique required (bicuspid anatomy: asymmetric calcium, risk of paravalvular leak + coronary obstruction); "
            "Mild-moderate AS: annual echo; activity restriction only if severe. "
            "AR: ACE inhibitor/ARB for LV dilation; surgical repair/replacement when severe + symptomatic or EF <55% or LVESD >50mm. "
            "Aorta: "
            "β-blockers (bisoprolol/metoprolol): rate-pressure product reduction; slows aortic dilatation; "
            "Losartan/irbesartan: TGF-β pathway — evidence from Marfan studies, extrapolated to BAV; "
            "Surgical aortic repair: when ascending >5.0cm (4.5cm if rapid growth or family history of dissection); "
            "High-intensity physical activity restriction when aorta >4.5cm. "
            "IE prophylaxis: ONLY for severe valvulopathy or post-prosthesis (UK guideline 2008; US AHA 2007: no longer recommended for native BAV). "
            "Genetics: NOTCH1 sequencing; family cascade echo."
        ),
        "monitoring": [
            "Echo: annual — valve area, mean gradient, aortic root, ascending aorta, LV dimensions",
            "CT aorta/CMR: if echo window limited; baseline aortopathy mapping; pre-procedure planning",
            "ECG: annual; LV hypertrophy (AS); AV conduction",
            "Exercise test: if equivocal symptoms in moderate AS; LVOT gradient on exertion",
            "Aortic growth rate: >3mm/year = accelerated → surgery threshold lower",
            "TAVI/SAVR: heart team discussion; CCTA mandatory pre-TAVI for BAV anatomy",
            "Pregnancy: cardiomegaly review; avoid vaginal delivery with dilated aorta >4.5cm",
            "Family: echo all 1st-degree relatives; NOTCH1 genetic testing if BAV confirmed in proband",
        ],
        "chd_types": ["BAV", "Aortopathy", "CoA", "VSD"],
        "pathognomonic": "Bicuspid aortic valve + ascending aortic dilatation = NOTCH1 PATHOGNOMONIC",
        "treatment_highlight": "Annual echo; TAVI/SAVR for severe AS; aortic surgery at >5.0cm",
    },
    # -- JAG1 — Alagille Syndrome ---------------------------------------------------------
    {
        "gene": "JAG1",
        "alt_name": (
            "JAG1 (JAG1-1218aa-20p12.2 / AD — Alagille-Syndrome-ALGS — "
            "BUTTERFLY-VERTEBRAE-Anterior-Arch-Defect-95pct-PATHOGNOMONIC-Spinal-X-Ray — "
            "POSTERIOR-EMBRYOTOXON-78pct-Anterior-Eye-Slit-Lamp-PATHOGNOMONIC — "
            "CHD-94pct-Pulmonary-Arterial-Stenosis-TOF — "
            "Cholestatic-Liver-Disease-80pct-Paucity-Intrahepatic-Bile-Ducts — "
            "p.Gly274Asp-European-70pct-LOF-Nonsense-Frameshift)"
        ),
        "protein": (
            "JAG1 -- 20p12.2 AD -- JAG1-1218aa -- "
            "Jagged-Canonical-Notch-Ligand-1-134kDa-Type-I-Transmembrane-NOTCH1-2-3-4-Ligand -- "
            "Alagille-Syndrome-ALGS-OMIM-118450 -- "
            "BUTTERFLY-VERTEBRAE-Anterior-Vertebral-Arch-Cleft-95pct-PATHOGNOMONIC-X-Ray -- "
            "POSTERIOR-EMBRYOTOXON-Schwalbe-Line-Prominement-78pct-Slit-Lamp-PATHOGNOMONIC -- "
            "FACIAL-GESTALT-Prominent-Forehead-Deep-Set-Eyes-Broad-Nasal-Bridge-Pointed-Chin-INVERTED-TRIANGLE-75pct -- "
            "CHD-94pct-Pulmonary-Arterial-Stenosis-PA-Hypoplasia-TOF-Most-Common -- "
            "CHOLESTATIC-LIVER-DISEASE-80pct-Paucity-Intrahepatic-Bile-Ducts-Biopsy -- "
            "RENAL-39pct-Structural-Renal-Anomalies-Renal-Tubular-Acidosis -- "
            "50pct-Require-Liver-Transplant-Outcome-Variable -- "
            "p.Gly274Asp-European-Founder-Variant -- "
            "70pct-LOF-Nonsense-Frameshift-Splice-Haploinsufficiency -- "
            "OMIM-Gene-JAG1-601920-Disease-ALGS-118450"
        ),
        "locus": "20p12.2",
        "protein_size": "1218 aa / 134 kDa",
        "inheritance": (
            "AD (autosomal dominant haploinsufficiency); "
            "~70% de novo; 30% familial; "
            "50-70% penetrance; variable expressivity — same family: severe liver disease vs isolated posterior embryotoxon; "
            "~70% LOF (nonsense/frameshift/splice): haploinsufficiency; "
            "p.Gly274Asp: European founder; DSL domain (Notch-binding); "
            "p.Cys284Ser: DSL domain; EGF-like repeat disruption; "
            "Large deletions 20p12.2: ~7% of ALGS cases; "
            "ALGS diagnostic criteria (Alagille 1975): ≥3/5 features (liver+CHD+vertebrae+eye+face); "
            "Modern molecular diagnosis: JAG1 variant sufficient with ≥1 feature; "
            "NOTCH2 variants: 1-2% of ALGS (JAG1-negative cases)"
        ),
        "key_features": [
            "BUTTERFLY VERTEBRAE (anterior arch defect) in 95% — PATHOGNOMONIC on spinal X-ray: sagittal cleft in vertebral body, butterfly wing appearance",
            "POSTERIOR EMBRYOTOXON in 78% — prominement Schwalbe line on slit-lamp: anterior segment anomaly; PATHOGNOMONIC even without full ALGS picture",
            "ALAGILLE FACIAL GESTALT: prominent forehead + deep-set eyes + broad nasal bridge + pointed chin = 'inverted triangle' face (75%)",
            "CHD in 94%: pulmonary arterial stenosis/hypoplasia MOST SPECIFIC; TOF; peripheral pulmonary artery stenosis (PPS) — high Doppler velocity bilateral branch PAs",
            "CHOLESTATIC LIVER DISEASE in 80%: conjugated hyperbilirubinaemia; raised GGT; paucity of intrahepatic bile ducts on biopsy",
            "RENAL in 39%: structural anomalies; renal tubular acidosis; growth failure",
            "50% require liver transplant — most common surgical outcome; cardiac surgery if severe CHD",
            "DIAGNOSTIC TRIAD for slit-lamp PATHOGNOMONIC: posterior embryotoxon alone = ophthalmology red flag for ALGS even without liver disease",
        ],
        "treatment": (
            "Liver: "
            "Cholestasis: ursodeoxycholic acid (UDCA) 10-15 mg/kg/day; fat-soluble vitamin supplementation (A/D/E/K). "
            "Pruritus: rifampicin (low dose), naltrexone, sertraline, cholestyramine; maralixibat (IBAT inhibitor) FDA 2021 for cholestatic pruritus in ALGS. "
            "Portal hypertension: propranolol; endoscopy variceal surveillance; band ligation. "
            "Liver transplantation: living or cadaveric donor; indicated for severe growth failure, unmanageable pruritus, hepatic decompensation; 50% of ALGS children. "
            "Cardiac: "
            "PA stenosis: balloon angioplasty for isolated RVOT/branch PA stenosis; stent implantation ≥10kg; "
            "TOF: surgical repair (infancy); RVOT reconstruction; "
            "Peripheral PA stenosis: serial catheter interventions. "
            "Nutrition: MCT-enriched formula; fat-soluble vitamins A/D/E/K monitoring; "
            "Renal: RTA correction with bicarbonate; nephrology referral. "
            "Genetics: JAG1 sequencing + FISH/CMA for deletion. "
            "Cascade: clinical screen all 1st-degree relatives (eye + spinal X-ray + echo + liver enzymes)."
        ),
        "monitoring": [
            "LFTs: monthly in infancy; 3-monthly thereafter; GGT most sensitive marker",
            "Fat-soluble vitamins A/D/E/K: every 6 months; supplement aggressively",
            "Echo: branch PA velocities; RV pressure; peripheral PA growth post-intervention",
            "Liver: ultrasound annually; hepatic portal pressure if varices suspected",
            "Slit-lamp: baseline; annually for posterior embryotoxon + lens subluxation",
            "Spinal: AP + lateral X-ray — butterfly vertebrae count; disc space assessment",
            "Renal: creatinine; electrolytes; urine pH if RTA suspected; renal ultrasound",
            "Growth: height/weight centile; nutritional assessment; oral calorie counting",
        ],
        "chd_types": ["Pulmonary arterial stenosis", "Peripheral PA stenosis", "TOF", "PA hypoplasia"],
        "pathognomonic": "Butterfly vertebrae + posterior embryotoxon + facial gestalt + cholestatic jaundice + CHD = Alagille syndrome JAG1",
        "treatment_highlight": "Maralixibat for pruritus; UDCA; 50% liver transplant; PA stenting",
    },
    # -- CHD7 — CHARGE Syndrome -----------------------------------------------------------
    {
        "gene": "CHD7",
        "alt_name": (
            "CHD7 (CHD7-2997aa-8q12.2 / AD — CHARGE-Syndrome — "
            "COLOBOMA-HEART-CHOANAL-ATRESIA-RETARDATION-GENITAL-EAR-PATHOGNOMONIC-Core-Triad — "
            "SEMICIRCULAR-CANAL-APLASIA-MRI-PATHOGNOMONIC-Radiological-Finding — "
            "75pct-CHD-Conotruncal-TOF-DORV-Truncus — "
            "Almost-All-LOF-De-Novo-Olfactory-Bulb-Hypoplasia-Anosmia)"
        ),
        "protein": (
            "CHD7 -- 8q12.2 AD -- CHD7-2997aa -- "
            "Chromodomain-Helicase-DNA-Binding-Protein-7-340kDa-Chromatin-Remodelling-Neural-Crest -- "
            "CHARGE-Syndrome-OMIM-214800 -- "
            "COLOBOMA-Iris-Retina-Chorioretinal-Cleft-Uvea-60-90pct-CHARGE -- "
            "HEART-DEFECT-75pct-Conotruncal-TOF-DORV-Truncus-Arteriosus-Most-Common -- "
            "CHOANAL-ATRESIA-Bony-Or-Membranous-50-60pct-Bilateral-Neonatal-Emergency -- "
            "RETARDATION-GROWTH-Developmental-Delay-100pct-CHARGE -- "
            "GENITAL-Hypogonadotropic-Hypogonadism-Micropenis-Cryptorchidism -- "
            "EAR-External-Ear-Anomaly-Hearing-Loss-SNHL-Sensorineural-Conductive-Mixed -- "
            "SEMICIRCULAR-CANAL-APLASIA-MRI-PATHOGNOMONIC-Absent-All-3-Canals-Balance-Disorder -- "
            "OLFACTORY-BULB-HYPOPLASIA-APLASIA-MRI-PATHOGNOMONIC-Anosmia-100pct -- "
            "Almost-All-De-Novo-LOF-Truncating-No-Hotspot -- "
            "OMIM-Gene-CHD7-608892-Disease-CHARGE-214800"
        ),
        "locus": "8q12.2",
        "protein_size": "2997 aa / 340 kDa",
        "inheritance": (
            "AD (autosomal dominant de novo haploinsufficiency); "
            "~95% de novo; <5% familial (germline mosaic parent); "
            "100% penetrance; variable expressivity; "
            "LOF dominant — truncating/frameshift/splice = CHARGE; missense more variable; "
            "No mutational hotspot — mutations scattered across all exons; "
            "p.Gln1441Ter: truncating; severe CHARGE; coloboma+heart+choanal atresia; "
            "p.Arg2498Ter: common European truncating; "
            "Phenotype spectrum: classic CHARGE (4+ features) to mild (2 features); "
            "CHD7 mosaic (somatic): milder phenotype; "
            "CHARGE criteria Blake 1998: 4C (coloboma, choanal atresia, semicircular canal, characteristic ear) vs 4C + heart + genital + growth/development"
        ),
        "key_features": [
            "CHARGE acronym: Coloboma + Heart defect + choanal Atresia + Retardation of growth + Genital anomaly + Ear anomaly PATHOGNOMONIC combination",
            "SEMICIRCULAR CANAL APLASIA on MRI — MOST PATHOGNOMONIC radiological finding: absent/hypoplastic semicircular canals (balance disorder + vestibular failure + Tullio phenomenon)",
            "OLFACTORY BULB HYPOPLASIA/APLASIA on MRI — anosmia present in virtually ALL CHARGE patients (Kallmann-like)",
            "CHD in 75%: CONOTRUNCAL most specific — TOF (27%), DORV (15%), truncus arteriosus (10%), AVSD (15%); conotruncal vs NKX2-5/GATA4 septal defects",
            "CHOANAL ATRESIA bilateral in infancy = NEONATAL EMERGENCY (obligate nasal breather) — bilateral choanal atresia + CHARGE → CHD7 testing mandatory",
            "COLOBOMA: iris/chorioretinal; field defect; visual impairment — ophthalmology from birth",
            "HYPOGONADOTROPIC HYPOGONADISM (GnRH deficiency): micropenis, cryptorchidism males; absent puberty both sexes → hormone replacement puberty",
            "DEAFBLIND risk: SNHL + coloboma → specialist deafblind education; CHARGE = leading genetic cause of deafblindness",
        ],
        "treatment": (
            "Neonatal emergency: "
            "Bilateral choanal atresia: nasal airway (McGovern nipple) immediate; transnasal endoscopic repair 3-6 months. "
            "Cardiac: conotruncal repair — TOF complete repair 3-6 months; truncus arteriosus repair 6-8 weeks; DORV repair. "
            "Eyes: coloboma no repair; refraction correction; patching amblyopia; low-vision aids. "
            "Ears/hearing: audiology from birth; BAHA (bone-anchored hearing aid) or cochlear implant for SNHL; "
            "vestibular physiotherapy for semicircular canal aplasia (balance; Tullio phenomenon). "
            "Endocrine: GH for growth failure; testosterone (males puberty); oestrogen/progesterone (females puberty); "
            "GnRH deficiency → fertility preserved by replacement hormones if treated at puberty. "
            "Feeding: NG/gastrostomy in first year (oropharyngeal dysphagia + CHD + airway); speech-language therapy. "
            "Neurodevelopment: early intervention; ABA therapy if autism spectrum; communication (AAC devices); "
            "CHARGE-specific educational programme (deafblind specialist teachers). "
            "Genetics: CHD7 sequencing + deletion panel; parental mosaic testing; 50% risk if parental mosaic."
        ),
        "monitoring": [
            "Vision: ophthalmology from birth; visual fields; ERG (retinal function); refraction; patch amblyopia",
            "Hearing: ABR neonatal; audiology 3-monthly first year; audiogram annually; cochlear implant candidacy",
            "Vestibular: balance assessments; Tullio test; specialist physiotherapy",
            "Cardiac: echo post-repair; conduction monitoring; residual lesion surveillance",
            "Endocrine: LH/FSH/oestrogen/testosterone annually from age 10; GH axis if growth failure",
            "Nasopharynx: ENT; choanal patency post-repair; adenoid/tonsillar assessment",
            "Neurodevelopment: developmental assessment 6-monthly first 3 years; autism screening (ADOS)",
            "Feeding: FEES/MBS annually; weight percentile; gastrostomy need assessment",
        ],
        "chd_types": ["TOF", "DORV", "Truncus arteriosus", "AVSD", "ASD", "VSD"],
        "pathognomonic": "Coloboma + choanal atresia + semicircular canal aplasia on MRI + conotruncal CHD = CHARGE CHD7",
        "treatment_highlight": "Neonatal choanal atresia emergency; cochlear implant/BAHA; deafblind education; GnRH replacement",
    },
    # -- TFAP2B — Char Syndrome -----------------------------------------------------------
    {
        "gene": "TFAP2B",
        "alt_name": (
            "TFAP2B (TFAP2B-463aa-6p24.3 / AD — Char-Syndrome — "
            "PATENT-DUCTUS-ARTERIOSUS-PDA-FACIAL-DYSMORPHISM-HAND-ANOMALIES-PATHOGNOMONIC-TRIAD — "
            "PDA-Structurally-Abnormal-Ductal-Tissue-Closure-Surgical-Catheter-Infancy — "
            "FACE-Flat-Nasal-Bridge-Ptosis-Low-Set-Ears-Fishmouth-Lips — "
            "HAND-Shortening-Middle-Phalanges-5th-Finger-Clinodactyly — "
            "Ultra-Rare-30-Families-Worldwide-2026)"
        ),
        "protein": (
            "TFAP2B -- 6p24.3 AD -- TFAP2B-463aa -- "
            "Transcription-Factor-AP-2-Beta-52kDa-Bzip-Helix-Span-Helix-Neural-Crest-Vascular -- "
            "Char-Syndrome-OMIM-169100 -- "
            "PATENT-DUCTUS-ARTERIOSUS-PDA-PATHOGNOMONIC-Ductal-Smooth-Muscle-Abnormality -- "
            "STRUCTURALLY-ABNORMAL-DUCTAL-TISSUE-Cannot-Close-Spontaneously-Unlike-Isolated-PDA -- "
            "PDA-In-Char-Syndrome-Unlikely-To-Close-Spontaneously-Surgical-Or-Catheter-Required -- "
            "FACIAL-DYSMORPHISM-Flat-Nasal-Bridge-Ptosis-Low-Set-Ears-Fishmouth-Lips-Distinguishing -- "
            "HAND-Middle-Phalangeal-Shortening-5th-Finger-Clinodactyly-Short-Metacarpal -- "
            "Ultra-Rare-30-Families-Reported-Worldwide-2026 -- "
            "TFAP2B-LOF-Neural-Crest-Derived-Ductal-Smooth-Muscle-Development-Defect -- "
            "OMIM-Gene-TFAP2B-601601-Disease-Char-169100"
        ),
        "locus": "6p24.3",
        "protein_size": "463 aa / 52 kDa",
        "inheritance": (
            "AD (autosomal dominant haploinsufficiency); "
            "Ultra-rare: ~30 families reported worldwide 2026; "
            "Variable expressivity; penetrance high but incomplete for all three features; "
            "Some family members have PDA-only or facial-only; "
            "p.Ala268Val: bZIP domain; most common mutation; PDA + face + hand; "
            "p.Leu293Pro: bZIP domain; severe PDA requiring neonatal intervention; "
            "p.Arg262His: bZIP; hand anomaly + PDA; "
            "Mechanism: TFAP2B regulates smooth muscle differentiation in neural-crest-derived ductal tissue; "
            "LOF → ductal smooth muscle cannot contract → persistent ductal patency; "
            "PDA histology: deficient smooth muscle; absent medial degeneration (unlike isolated PDA); "
            "Heart: isolated PDA; rarely ASD or VSD co-occur; "
            "NO conotruncal defects (unlike TBX1/CHD7); NO septal defects of ASD/VSD type typically"
        ),
        "key_features": [
            "CHAR SYNDROME TRIAD: Patent Ductus Arteriosus + Facial Dysmorphism + Hand Anomalies = PATHOGNOMONIC",
            "PDA IN CHAR SYNDROME: structurally abnormal ductal smooth muscle → spontaneous closure VERY UNLIKELY (unlike isolated PDA which may close in preterm infants)",
            "FACIAL DYSMORPHISM: flat nasal bridge + ptosis + low-set ears + fishmouth lips (drooping corners) — subtle but consistent across families",
            "HAND ANOMALIES: shortening of middle phalanges (brachymesophalangy) + 5th finger clinodactyly — hand X-ray diagnostic",
            "ULTRA-RARE: only ~30 families worldwide 2026; sequencing required for diagnosis (phenotype diagnosis possible with triad)",
            "PDA complications: left-to-right shunt → pulmonary over-circulation → pulmonary hypertension; LV volume overload; failure to thrive",
            "Ductal tissue ABNORMAL histologically (smooth muscle deficiency) — explains poor response to indomethacin/ibuprofen for closure",
            "NO LIMB DEFECTS like TBX5 (no radial ray) and NO semicircular canal aplasia like CHD7 — KEY DDx by associated features",
        ],
        "treatment": (
            "PDA closure: "
            "Catheter-based: Amplatzer duct occluder / Piccolo (PFO-type); "
            "preferred in children >5 kg; closure in infancy 6-12 months (pre-pulmonary hypertension). "
            "Surgical: ligation via left thoracotomy; VATS (video-assisted) approach; "
            "indicated for large PDA in premature/small infant; duct >4mm at catheterisation. "
            "Medical (indomethacin/ibuprofen/paracetamol): AVOID in Char syndrome — "
            "ductal tissue is abnormally smooth-muscle-deficient; COX-inhibitor efficacy LOW; "
            "pharmacologic closure rates < isolated premature PDA; surgical/catheter first-line. "
            "Post-closure: IE prophylaxis for 6 months; annual echo for residual shunt. "
            "Facial: ptosis — ophthalmology; surgical correction if visual axis affected (amblyopia risk). "
            "Hand: occupational therapy; hand X-ray in childhood; no specific orthopaedic intervention usually needed. "
            "Genetics: TFAP2B sequencing; family cascade (PDA + face + hand); 50% transmission risk."
        ),
        "monitoring": [
            "Echo: LV size (volume overload); PA pressure; residual shunt post-closure; annual",
            "ECG: LV hypertrophy; axis deviation; annual",
            "PA pressure: Doppler TR jet velocity; catheter if elevated echo pressure",
            "Eyes: ophthalmology baseline; ptosis; amblyopia (patching protocol)",
            "Hands: X-ray; OT hand function assessment; grip strength",
            "Growth: FTT from PDA in infancy; nutritional support pre-closure",
            "Pulmonary hypertension: 6MW test from age 6; right heart cath if Qp:Qs <1.5 but elevated PA",
            "Family: echo (PDA screening); facial + hand examination; TFAP2B genetic testing",
        ],
        "chd_types": ["PDA"],
        "pathognomonic": "PDA + flat nasal bridge/ptosis/fishmouth lips + middle phalangeal shortening = Char syndrome TFAP2B",
        "treatment_highlight": "Catheter/surgical PDA closure; avoid indomethacin (low efficacy); ptosis eye care",
    },
    # -- TBX1 — 22q11.2 Deletion Syndrome (DiGeorge) ------------------------------------
    {
        "gene": "TBX1",
        "alt_name": (
            "TBX1 (TBX1-504aa-22q11.21 / AD — 22q11.2-Deletion-DiGeorge-VCF-Syndrome — "
            "CONOTRUNCAL-HEART-DEFECT-HYPOCALCEMIA-T-CELL-LYMPHOPENIA-PATHOGNOMONIC-TRIAD — "
            "IAA-TYPE-B-TRUNCUS-ARTERIOSUS-TOF-Absent-Pulmonary-Valve-PATHOGNOMONIC-Conotruncal-Pattern — "
            "MOST-COMMON-Chromosomal-Microdeletion-1-in-4000-Live-Births — "
            "SCHIZOPHRENIA-25pct-ADHD-40pct-Psychiatric-Comorbidity)"
        ),
        "protein": (
            "TBX1 -- 22q11.21 AD -- TBX1-504aa -- "
            "T-Box-Transcription-Factor-1-57kDa-Neural-Crest-Pharyngeal-Arch-Development -- "
            "22q11.2-Deletion-Syndrome-OMIM-188400 -- "
            "CONOTRUNCAL-HEART-DEFECT-75pct-IAA-B-Truncus-TOF-DORV-Most-Common -- "
            "INTERRUPTED-AORTIC-ARCH-TYPE-B-IAA-B-PATHOGNOMONIC-Conotruncal-Pattern-22q11 -- "
            "TRUNCUS-ARTERIOSUS-PATHOGNOMONIC-Conotruncal -- "
            "TOF-WITH-ABSENT-PULMONARY-VALVE-PATHOGNOMONIC-22q11 -- "
            "HYPOCALCEMIA-Hypoparathyroidism-3rd-Pharyngeal-Pouch-PATHOGNOMONIC-Check-Calcium-All-Cases -- "
            "T-CELL-LYMPHOPENIA-Thymus-Absent-Hypoplastic-3rd-Pharyngeal-Pouch-PATHOGNOMONIC -- "
            "MOST-COMMON-CHROMOSOMAL-MICRODELETION-1-in-4000-Live-Births -- "
            "SCHIZOPHRENIA-25pct-Highest-Genetic-Risk-Factor-Known -- "
            "ADHD-40pct-Anxiety-50pct-ASD-15pct-Neurodevelopmental -- "
            "FISH-CMA-Deletion-Detection-TBX1-Point-Mutations-Rare -- "
            "OMIM-Gene-TBX1-602054-Disease-22q11DS-188400"
        ),
        "locus": "22q11.21",
        "protein_size": "504 aa / 57 kDa",
        "inheritance": (
            "AD (autosomal dominant haploinsufficiency); "
            "Most common chromosomal microdeletion: 1:4,000 live births; "
            "90% de novo; 10% familial (one parent carries same deletion); "
            "FISH and CMA (chromosomal microarray) detect 3Mb deletion; TBX1 point mutations <1%; "
            "Deletion size: 3Mb (most), 1.5Mb (nested), rare atypical; "
            "Typical deletion: 22q11.21 containing TBX1 + 39 other genes; "
            "Phenotype: cardiac (75%), hypocalcemia (50%), thymic hypoplasia (75%), palate (69%), face (93%); "
            "DiGeorge: T-cell lymphopenia dominant; VCF: palatal/speech dominant; "
            "CATCH22: Cardiac+Abnormal facies+Thymus hypoplasia+Cleft palate+Hypocalcemia; "
            "Psychiatric: schizophrenia 25% (HIGHEST KNOWN GENETIC RISK FACTOR for schizophrenia), "
            "ADHD 40%, anxiety 50%, ASD 15% — often presenting in adolescence/adulthood"
        ),
        "key_features": [
            "CONOTRUNCAL HEART DEFECT + HYPOCALCEMIA + T-CELL LYMPHOPENIA = PATHOGNOMONIC TRIAD: 22q11.2 deletion syndrome (DiGeorge/VCF)",
            "PATHOGNOMONIC CONOTRUNCAL PATTERN: Interrupted aortic arch type B (IAA-B), truncus arteriosus, TOF with absent pulmonary valve — these three suggest 22q11 before genetic testing",
            "HYPOCALCEMIA — check calcium in ALL infants with conotruncal CHD (hypoparathyroidism from 3rd pharyngeal pouch: parathyroid absent/hypoplastic)",
            "T-CELL LYMPHOPENIA — thymus from 3rd pharyngeal pouch absent/hypoplastic: neonatal immune deficiency; live vaccine protocol modified",
            "MOST COMMON chromosomal microdeletion (1:4,000) — most common cause of genetic CHD after trisomies",
            "SCHIZOPHRENIA in 25% — highest single genetic risk factor for schizophrenia known; psychiatric surveillance mandatory from adolescence",
            "PALATE: velopharyngeal insufficiency (69%) — hypernasal speech; submucous cleft palate; laryngoscopy; speech therapy",
            "FISH/CMA DETECTION: 22q11.2 microdeletion array detects; NOT detected on standard karyotype (too small)",
        ],
        "treatment": (
            "Cardiac: "
            "IAA-B: neonatal surgical repair (Norwood-type arch reconstruction + VSD closure); intensive care; ductal stenting pre-repair if unstable. "
            "Truncus arteriosus: neonatal complete repair (conduit RV-to-PA + ventricular septal closure). "
            "TOF with absent PV: surgical repair (patch RVOT + pulmonary leaflets); risk of bronchial compression. "
            "Endocarditis prophylaxis 6 months post-repair. "
            "Hypocalcemia: "
            "Acute: IV calcium gluconate (10%); 1-2 mL/kg over 10 min; "
            "Maintenance: calcitriol + calcium supplementation lifelong; "
            "PTH levels; vitamin D3; "
            "Refractory hypocalcemia: recombinant PTH (Natpara) — adult HYPP; "
            "All infants with conotruncal CHD: check ionised calcium BEFORE surgery. "
            "Immunology: "
            "T-cell count: CD4+CD8+ lymphocyte subsets at birth; "
            "Live vaccines CONTRAINDICATED until T-cell normalisation (varicella, MMR, rotavirus, BCG); "
            "Thymus transplant (complete DiGeorge) if T-cells <50/μL by 6 months; "
            "IVIG if hypogammaglobulinaemia. "
            "Palate/speech: speech-language therapy; velopharyngoplasty for VPI. "
            "Neurodevelopment: early intervention; educational support; IEP. "
            "Psychiatric: antipsychotic monitoring from adolescence; schizophrenia surveillance; "
            "risperidone for early psychosis; CBT; social skills."
        ),
        "monitoring": [
            "Calcium: ionised calcium daily neonatal; 3-monthly calcitriol dose adjustment; annual PTH/vitamin D",
            "T-cell subsets: CD4/CD8 counts at birth, 1yr, 5yr; live vaccine clearance when CD4 >200/μL",
            "Echo: annual post-repair; arch anatomy; arch re-stenosis (IAA); conduit growth",
            "Immunoglobulins: IgG/IgA/IgM annual; IVIG if hypogamma",
            "Neurodevelopment: BSID at 12/24 months; WISC at school entry; IEP review",
            "Psychiatric: BPRS/PANSS screening from age 12; MRI brain if psychosis risk (Dobler-Mikola protocol)",
            "Speech: VPI assessment; nasopharyngoscopy; velopharyngoplasty timing",
            "Growth: height/weight; GH axis if short stature + GH deficiency (rare in 22q11)",
        ],
        "chd_types": ["IAA-B", "Truncus arteriosus", "TOF (absent PV)", "VSD", "DORV"],
        "pathognomonic": "Conotruncal CHD (IAA-B/truncus/TOF-absent PV) + hypocalcemia + T-cell lymphopenia = 22q11DS TBX1",
        "treatment_highlight": "Check calcium in ALL conotruncal CHD; modified live vaccine schedule; schizophrenia surveillance adolescence",
    },
]


def _make_patient(gene_entry: dict, seed: int) -> dict:
    """Generate one synthetic patient record for the given gene entry."""
    rng = random.Random(seed)
    g = gene_entry["gene"]

    # Gene-specific distributions
    if g == "GATA4":
        chd_choices = ["ASD type II", "Perimembranous VSD", "TOF", "ASD+VSD", "ASD+AVSD"]
        chd_weights = [40, 30, 20, 7, 3]
        onset = rng.randint(0, 12)  # months (prenatal/neonatal/infant)
        sex = rng.choice(["M", "F"])
        intervention_choices = ["Device closure (ASD)", "Surgical VSD repair", "TOF complete repair", "Conservative"]
        intervention_weights = [40, 30, 20, 10]
        extracardia = rng.choice(["None", "None", "None", "None", "First-degree AV block"])  # mostly none
        echo_finding = rng.choice(["Ostium secundum ASD", "Perimembranous VSD", "TOF", "ASD+VSD"])
    elif g == "TBX5":
        chd_choices = ["ASD type II", "ASD+VSD", "VSD (isolated)", "ASD+AVSD", "ASD+conduction"]
        chd_weights = [50, 25, 10, 10, 5]
        onset = rng.randint(0, 3)
        sex = rng.choice(["M", "F"])
        intervention_choices = ["ASD device closure", "Surgical ASD repair", "ASD+VSD repair", "AVSD repair"]
        intervention_weights = [40, 30, 20, 10]
        extracardia = rng.choice(["Thumb hypoplasia (bilateral)", "Triphalangeal thumb", "Absent thumb (unilateral)", "Absent thumb (bilateral)", "Radial hypoplasia", "1st-degree AV block"])
        echo_finding = rng.choice(["Ostium secundum ASD", "Perimembranous VSD", "ASD+VSD", "AVSD"])
    elif g == "NKX2-5":
        chd_choices = ["ASD+AV block", "VSD+AV block", "TOF+AV block", "ASD (isolated)", "LV noncompaction"]
        chd_weights = [50, 30, 15, 4, 1]
        onset = rng.randint(0, 12)
        sex = rng.choice(["M", "F"])
        intervention_choices = ["ASD closure + pacemaker surveillance", "VSD repair", "TOF repair", "Pacemaker (DDD-R)"]
        intervention_weights = [40, 30, 20, 10]
        av_block_degree = rng.choice(["1st degree", "2nd degree Mobitz I", "2nd degree Mobitz II", "3rd degree (complete)"])
        extracardia = f"Progressive AV block ({av_block_degree})"
        echo_finding = rng.choice(["Ostium secundum ASD", "Perimembranous VSD", "TOF", "LVNC"])
    elif g == "NOTCH1":
        chd_choices = ["Bicuspid aortic valve (RL fusion)", "Bicuspid aortic valve (RN fusion)", "BAV + mild AS", "BAV + AR", "BAV + aortopathy"]
        chd_weights = [35, 22, 22, 11, 10]
        onset = rng.randint(0, 5) * 10  # years (0, 10, 20, 30, 40, 50)
        sex = rng.choice(["M", "M", "F"])  # male predominance
        intervention_choices = ["Annual surveillance echo", "Valve repair/SAVR", "TAVI (BAV protocol)", "Aortic root replacement"]
        intervention_weights = [50, 20, 20, 10]
        aorta_mm = rng.randint(32, 52)
        extracardia = f"Ascending aorta {aorta_mm} mm"
        echo_finding = rng.choice(["BAV RL fusion — mild AS", "BAV RN fusion — AR", "BAV — aortopathy", "BAV — moderate AS"])
    elif g == "JAG1":
        chd_choices = ["Pulmonary arterial stenosis", "Peripheral PA stenosis", "TOF", "PA hypoplasia", "VSD"]
        chd_weights = [40, 25, 20, 10, 5]
        onset = rng.randint(0, 3)  # neonatal / infancy
        sex = rng.choice(["M", "F"])
        intervention_choices = ["PA balloon angioplasty", "PA stenting", "TOF repair", "Conservative (mild PPS)"]
        intervention_weights = [35, 25, 25, 15]
        extracardia = rng.choice(["Butterfly vertebrae + posterior embryotoxon", "Butterfly vertebrae + cholestatic jaundice", "Posterior embryotoxon + facial gestalt", "Butterfly vertebrae + ALGS facial gestalt"])
        echo_finding = rng.choice(["Branch PA stenosis (bilateral)", "RVOT/RPA stenosis", "TOF anatomy", "PA hypoplasia"])
    elif g == "CHD7":
        chd_choices = ["TOF (conotruncal)", "DORV", "Truncus arteriosus", "AVSD", "ASD+VSD", "No CHD (5%)"]
        chd_weights = [27, 15, 10, 15, 28, 5]
        onset = rng.randint(0, 1)  # neonatal
        sex = rng.choice(["M", "F"])
        intervention_choices = ["TOF complete repair", "DORV repair", "Truncus repair (neonatal)", "AVSD repair", "Choanal atresia repair only"]
        intervention_weights = [27, 15, 10, 15, 33]
        extracardia = rng.choice(["Coloboma + choanal atresia + semicircular canal aplasia", "Coloboma + SNHL + semicircular canal aplasia", "Choanal atresia + SNHL + hypogonadism", "Coloboma + CHARGE facial features"])
        echo_finding = rng.choice(["TOF (VSD+RVOT obstruction)", "DORV with subarterial VSD", "Truncus arteriosus type I", "AVSD (complete)"])
    elif g == "TFAP2B":
        chd_choices = ["Patent ductus arteriosus (large)", "PDA (moderate)", "PDA + ASD", "PDA (small)"]
        chd_weights = [50, 30, 15, 5]
        onset = rng.randint(0, 3)  # neonatal
        sex = rng.choice(["M", "F"])
        intervention_choices = ["Catheter PDA device closure", "Surgical PDA ligation", "PDA + ASD device closure", "Conservative (small PDA)"]
        intervention_weights = [50, 30, 15, 5]
        extracardia = rng.choice(["Flat nasal bridge + ptosis + fishmouth lips + 5th finger clinodactyly", "Ptosis + brachymesophalangy + PDA", "Facial dysmorphism (Char) + hand anomaly"])
        echo_finding = rng.choice(["Large PDA (continuous flow)", "Moderate PDA + LA enlargement", "PDA + small ASD"])
    else:  # TBX1 / 22q11DS
        chd_choices = ["Interrupted aortic arch type B", "Truncus arteriosus", "TOF with absent pulmonary valve", "TOF (standard)", "VSD (subarterial)", "DORV"]
        chd_weights = [27, 20, 15, 20, 12, 6]
        onset = 0  # neonatal
        sex = rng.choice(["M", "F"])
        intervention_choices = ["IAA-B repair + VSD closure", "Truncus repair (neonatal)", "TOF+absent PV repair", "TOF complete repair", "VSD repair + arch reconstruction"]
        intervention_weights = [27, 20, 15, 20, 18]
        ca = round(rng.uniform(0.80, 1.05), 2)  # hypocalcaemia
        extracardia = f"Hypocalcaemia (iCa {ca} mmol/L) + T-cell lymphopenia + velopharyngeal insufficiency"
        echo_finding = rng.choice(["IAA-B + VSD", "Truncus arteriosus type I", "TOF absent pulmonary valve", "TOF standard type"])

    # Pick CHD type
    total_w = sum(chd_weights)
    pick = rng.random() * total_w
    running = 0
    chd = chd_choices[-1]
    for opt, wt in zip(chd_choices, chd_weights):
        running += wt
        if pick <= running:
            chd = opt
            break

    # Pick intervention
    total_w2 = sum(intervention_weights)
    pick2 = rng.random() * total_w2
    running2 = 0
    intervention = intervention_choices[-1]
    for opt, wt in zip(intervention_choices, intervention_weights):
        running2 += wt
        if pick2 <= running2:
            intervention = opt
            break

    # Build record
    age_dx = round(rng.uniform(0, 2 if g != "NOTCH1" else 45), 1)
    follow_up = round(rng.uniform(0.5, 12), 1)
    qp_qs = round(rng.uniform(1.0, 2.8), 2) if g in ("GATA4", "TBX5", "NKX2-5", "TFAP2B") else None
    pa_pressure = round(rng.uniform(15, 45), 0) if g in ("GATA4", "TBX5", "NKX2-5", "JAG1", "TFAP2B") else None

    record = {
        "gene": g,
        "seed": seed,
        "sex": sex,
        "age_at_diagnosis_yrs": age_dx,
        "chd_type": chd,
        "echo_finding": echo_finding,
        "extracardia_features": extracardia if g not in ("GATA4",) else "None (isolated CHD)",
        "intervention": intervention,
        "follow_up_yrs": follow_up,
        "pathognomonic": gene_entry["pathognomonic"],
        "treatment_highlight": gene_entry["treatment_highlight"],
    }
    if qp_qs:
        record["qp_qs_ratio"] = qp_qs
    if pa_pressure:
        record["rvsp_mmhg"] = pa_pressure
    return record


def _build_cohort():
    patients = []
    for i, gene_entry in enumerate(CHD_GENES):
        base_seed = SEED_BASE + i
        for j in range(40):
            patients.append(_make_patient(gene_entry, base_seed * 100 + j))
    return patients


# ── Public API ─────────────────────────────────────────────────────────────────────────

def overview() -> dict:
    """Aggregate statistics across all 8 CHD genes (320 patients)."""
    cohort = _build_cohort()
    gene_counts = {}
    chd_type_counts = {}
    extracardia_flags = {"conotruncal": 0, "septal": 0, "valvular_bav": 0, "pda": 0, "pa_stenosis": 0, "none_isolated": 0}

    for p in cohort:
        g = p["gene"]
        gene_counts[g] = gene_counts.get(g, 0) + 1
        chd = p["chd_type"]
        chd_type_counts[chd] = chd_type_counts.get(chd, 0) + 1
        ext = str(p.get("extracardia_features", ""))
        if "conotruncal" in chd.lower() or "truncus" in chd.lower() or "IAA" in chd or "absent pulmonary" in chd.lower():
            extracardia_flags["conotruncal"] += 1
        if "ASD" in chd or "VSD" in chd or "AVSD" in chd:
            extracardia_flags["septal"] += 1
        if "Bicuspid" in chd or "BAV" in chd:
            extracardia_flags["valvular_bav"] += 1
        if "PDA" in chd or "ductus" in chd.lower():
            extracardia_flags["pda"] += 1
        if "PA" in chd or "pulmonary arterial" in chd.lower():
            extracardia_flags["pa_stenosis"] += 1
        if "None (isolated CHD)" in ext:
            extracardia_flags["none_isolated"] += 1

    gene_summary = []
    for entry in CHD_GENES:
        gene_summary.append({
            "gene": entry["gene"],
            "locus": entry["locus"],
            "protein_size": entry["protein_size"],
            "inheritance": "AD",
            "n_patients": gene_counts.get(entry["gene"], 0),
            "disease_syndrome": entry["alt_name"].split("—")[1].strip()[:80] if "—" in entry["alt_name"] else "",
            "pathognomonic": entry["pathognomonic"],
            "treatment_highlight": entry["treatment_highlight"],
        })

    return {
        "title": "Hereditary-Congenital-Heart-Disease-Atlas",
        "subtitle": (
            "Complete 8-Gene Hereditary Congenital Heart Disease (CHD) Atlas — "
            "GATA4 · TBX5 · NKX2-5 · NOTCH1 · JAG1 · CHD7 · TFAP2B · TBX1 — "
            "320 patients (8 × 40, seeds 2254-2261)"
        ),
        "n_patients": len(cohort),
        "n_genes": 8,
        "seed_range": "2254-2261",
        "chd_categories": {
            "septal_defects_asd_vsd_avsd": "GATA4 / TBX5 / NKX2-5 — isolated CHD + conduction defects",
            "valvular_bav_aortopathy": "NOTCH1 — bicuspid aortic valve + ascending aortopathy",
            "syndromic_multiorgan": "JAG1 (Alagille) / CHD7 (CHARGE) / TFAP2B (Char) / TBX1 (22q11DS)",
            "conotruncal": "TBX1 (22q11DS) — IAA-B / truncus / TOF-absent PV",
            "pda": "TFAP2B (Char) — structurally abnormal ductal tissue",
            "pa_stenosis": "JAG1 (Alagille) — PA stenosis / hypoplasia",
        },
        "chd_type_distribution": dict(sorted(chd_type_counts.items(), key=lambda x: -x[1])[:15]),
        "chd_category_counts": extracardia_flags,
        "gene_summary": gene_summary,
        "key_clinical_pearls": [
            "GATA4: isolated CHD (ASD/VSD/TOF); NO extracardia features; incomplete penetrance; GATA4+NKX2-5 interaction → overlap phenotype",
            "TBX5: Holt-Oram syndrome; BILATERAL thumb anomaly + CHD PATHOGNOMONIC; 100% penetrance; limb severity ≠ heart severity",
            "NKX2-5: ASD + PROGRESSIVE AV BLOCK PATHOGNOMONIC; block WORSENS after ASD repair; pacemaker 30-40% adulthood",
            "NOTCH1: BAV most common CHD (0.5-2%); annual echo mandatory (valve + aorta); dissection risk independent of valve function",
            "JAG1: BUTTERFLY VERTEBRAE 95% PATHOGNOMONIC (spinal X-ray); posterior embryotoxon 78% PATHOGNOMONIC (slit-lamp); 50% liver transplant",
            "CHD7: CHARGE; semicircular canal APLASIA MRI PATHOGNOMONIC; conotruncal CHD 75%; deafblind risk; schizophrenia surveillance NOT relevant (CHD7 ≠ TBX1)",
            "TFAP2B: Char syndrome; PDA PATHOGNOMONIC + facial gestalt + brachymesophalangy; avoid indomethacin (ductal tissue abnormal); catheter/surgical closure",
            "TBX1: 22q11DS; CONOTRUNCAL + HYPOCALCEMIA + T-CELL LYMPHOPENIA PATHOGNOMONIC; check iCa ALL conotruncal CHD; schizophrenia 25% adolescence",
        ],
        "diagnostic_algorithm": {
            "CHD_detected_prenatal_or_neonatal": "Echo anatomy → classify: septal/valvular/conotruncal/PA/PDA",
            "septal_ASD_VSD": "Check limbs (TBX5?); check AV block ECG (NKX2-5?); check calcium (TBX1?); panel: GATA4+TBX5+NKX2-5",
            "conotruncal_IAA_truncus_TOF": "FISH/CMA for 22q11 FIRST (27% of conotruncal CHD); check calcium + T-cells same day",
            "BAV_incidental": "NOTCH1 family echo (20-30% family members); annual surveillance; aorta imaging",
            "multiorgan_CHD_plus_features": "Butterfly vertebrae → JAG1; Semicircular canal aplasia → CHD7; PDA+face+hand → TFAP2B; coloboma+choanal → CHD7",
            "isolated_CHD_no_syndrome": "GATA4 / TBX5 / NKX2-5 trio panel; consider broader CHD gene panel (20+ genes)",
        },
    }


def breakdown() -> dict:
    """Per-patient CHD profiles across all 8 genes."""
    cohort = _build_cohort()
    by_gene = {}
    for p in cohort:
        by_gene.setdefault(p["gene"], []).append(p)

    gene_breakdown = {}
    for entry in CHD_GENES:
        g = entry["gene"]
        pts = by_gene.get(g, [])
        interventions = {}
        chd_types = {}
        for p in pts:
            interventions[p["intervention"]] = interventions.get(p["intervention"], 0) + 1
            chd_types[p["chd_type"]] = chd_types.get(p["chd_type"], 0) + 1

        avg_age = round(sum(p["age_at_diagnosis_yrs"] for p in pts) / len(pts), 1) if pts else 0
        avg_fu = round(sum(p["follow_up_yrs"] for p in pts) / len(pts), 1) if pts else 0

        gene_breakdown[g] = {
            "gene": g,
            "locus": entry["locus"],
            "protein_size": entry["protein_size"],
            "inheritance": "AD",
            "syndrome": entry["alt_name"].split("(")[1].split(")")[0][:120] if "(" in entry["alt_name"] else g,
            "n_patients": len(pts),
            "avg_age_at_dx_yrs": avg_age,
            "avg_follow_up_yrs": avg_fu,
            "chd_type_distribution": dict(sorted(chd_types.items(), key=lambda x: -x[1])),
            "intervention_distribution": dict(sorted(interventions.items(), key=lambda x: -x[1])),
            "pathognomonic": entry["pathognomonic"],
            "treatment_highlight": entry["treatment_highlight"],
            "key_features": entry["key_features"],
            "treatment_summary": entry["treatment"][:600],
            "monitoring": entry["monitoring"],
            "patients_sample": pts[:5],
        }

    return {
        "title": "Hereditary-Congenital-Heart-Disease-Atlas — Per-Gene Breakdown",
        "n_genes": 8,
        "n_patients": len(cohort),
        "gene_breakdown": gene_breakdown,
        "clinical_emergency_flags": [
            "TBX1/22q11DS: IAA-B = NEONATAL SURGICAL EMERGENCY + check iCa immediately (hypocalcaemia → tetany/seizures before surgery)",
            "CHD7/CHARGE: bilateral choanal atresia = NEONATAL AIRWAY EMERGENCY (obligate nasal breather → apnoea)",
            "JAG1/Alagille: ≥3 features (liver+CHD+vertebrae+eye+face) → JAG1 testing while managing CHD",
            "NOTCH1/BAV: sudden chest pain + aorta >5.0cm → EMERGENCY AORTIC DISSECTION PROTOCOL (type A)",
            "NKX2-5: post-ASD repair + bradycardia/syncope → HOLTER (progressive AV block even post-repair — do NOT reassure)",
            "TFAP2B/Char: PDA not closing on indomethacin → Char syndrome considered; catheter/surgical closure first-line",
        ],
    }


def definitions() -> dict:
    """Glossary of CHD anatomy, syndromes, and management concepts."""
    return {
        "title": "Hereditary-Congenital-Heart-Disease-Atlas — Definitions & Glossary",
        "gene_entries": {
            entry["gene"]: {
                "full_protein": entry["protein"],
                "inheritance_details": entry["inheritance"],
                "key_features": entry["key_features"],
                "treatment": entry["treatment"],
                "monitoring": entry["monitoring"],
            }
            for entry in CHD_GENES
        },
        "anatomy_glossary": {
            "ASD": "Atrial Septal Defect — hole between left and right atria; types: ostium secundum (most common, 75%), sinus venosus, primum (AVSD-spectrum), coronary sinus",
            "VSD": "Ventricular Septal Defect — hole between ventricles; perimembranous (80%), muscular, outlet (subarterial); spontaneous closure 30-40% perimembranous by age 2",
            "AVSD": "Atrioventricular Septal Defect — shared AV valve + ASD + VSD; complete (Down syndrome, TBX5, GATA4+NKX2-5); partial (ASD primum + cleft MV)",
            "TOF": "Tetralogy of Fallot — VSD + RVOT obstruction + overriding aorta + RVH; most common cyanotic CHD; surgical repair 3-6 months",
            "IAA-B": "Interrupted Aortic Arch type B — no connection between ascending and descending aorta, interruption distal to left subclavian; 50% have 22q11.2 deletion; neonatal surgical emergency",
            "Truncus arteriosus": "Single great artery exiting heart supplying both pulmonary and systemic circulation; types I-IV (Collett-Edwards); neonatal complete repair",
            "DORV": "Double Outlet Right Ventricle — both great arteries arise from RV; spectrum includes Eisenmenger-type, Fallot-type, TGA-type (depending on VSD position)",
            "BAV": "Bicuspid Aortic Valve — two leaflets (RL or RN fusion most common); most common CHD (0.5-2%); complications: stenosis, regurgitation, aortopathy, endocarditis",
            "PDA": "Patent Ductus Arteriosus — fetal connection between aorta and pulmonary artery fails to close after birth; isolated (preterm) vs syndromic (Char/TFAP2B); closure: catheter/surgical (Char) or indomethacin (isolated premature)",
            "CoA": "Coarctation of Aorta — narrowing of descending aorta near ductus; 10% of BAV (NOTCH1); hypertension upper limb vs lower limb; balloon angioplasty or surgical repair",
            "PA stenosis": "Pulmonary arterial stenosis — narrowing of main/branch PA; Alagille syndrome most common genetic cause; balloon angioplasty; stenting if >10kg",
            "RVOT": "Right Ventricular Outflow Tract — subpulmonary region; obstruction in TOF; gradient monitoring post-repair",
            "AV block": "Atrioventricular block — impaired conduction from atria to ventricles; 1st (long PR), 2nd (Mobitz I/Wenkebach or Mobitz II), 3rd (complete — independent P and QRS); NKX2-5 progressive even post-ASD repair",
        },
        "syndrome_glossary": {
            "Holt-Oram syndrome": "TBX5; bilateral upper limb anomaly + heart (ASD/VSD); thumb anomaly most specific; 100% penetrance; limb ≠ heart severity",
            "Alagille syndrome": "JAG1 (95%) or NOTCH2 (1-2%); ALGS5 criteria; butterfly vertebrae PATHOGNOMONIC; liver paucity bile ducts; CHD 94%; posterior embryotoxon 78%",
            "CHARGE syndrome": "CHD7; C-coloboma H-heart A-choanal atresia R-retardation G-genital E-ear; semicircular canal aplasia MRI PATHOGNOMONIC; conotruncal CHD; deafblind risk",
            "Char syndrome": "TFAP2B; PDA (structurally abnormal duct) + facial dysmorphism + hand anomaly; ultra-rare ~30 families; catheter closure first-line (not indomethacin)",
            "DiGeorge/22q11DS": "TBX1/22q11.2 deletion; CATCH22; conotruncal CHD + hypocalcaemia + T-cell lymphopenia; most common microdeletion 1:4,000; schizophrenia 25%",
            "CATCH22": "Cardiac + Abnormal facies + Thymic hypoplasia + Cleft palate + Hypocalcaemia — original acronym for 22q11.2 deletion syndrome (DiGeorge/VCF)",
            "VCF syndrome": "Velocardiofacial syndrome — same as 22q11DS; palate/speech dominant presentation",
        },
        "treatment_glossary": {
            "TAVI/TAVR": "Transcatheter Aortic Valve Implantation/Replacement — for severe AS in BAV (NOTCH1); BAV-specific technique required (bicuspid anatomy); preferred ≥65yr high/intermediate risk",
            "SAVR": "Surgical Aortic Valve Replacement — open surgery; preferred <65yr (longevity); mechanical vs bioprosthetic choice based on age and anticoagulation preference",
            "Maralixibat": "IBAT (ileal bile acid transporter) inhibitor — FDA 2021 for cholestatic pruritus in Alagille syndrome; reduces bile acid reabsorption; significant pruritus reduction",
            "Thymus transplantation": "For complete DiGeorge (T-cells <50/μL by 6 months); thymus graft from non-cardiac surgery donor; aim T-cell reconstitution; centre of excellence procedure",
            "Amplatzer duct occluder": "Catheter-based PDA closure device; preferred Char syndrome; nitinol mesh; immediate closure; allows earlier intervention vs surgical ligation",
            "Calcitriol": "Active vitamin D3 (1,25-dihydroxycholecalciferol) — for hypoparathyroidism in 22q11DS; bypasses PTH-dependent activation; lifelong in complete DiGeorge",
            "UDCA": "Ursodeoxycholic acid — hydrophilic bile acid; reduces cholestatic liver damage in Alagille syndrome; 10-15 mg/kg/day; first-line hepatic therapy",
            "DDD-R pacemaker": "Dual-chamber rate-responsive pacemaker — for complete or high-grade AV block in NKX2-5 disease; rate response essential for chronotropic incompetence; mandatory lifelong follow-up",
        },
        "diagnostic_tests": {
            "CMA/FISH_22q11": "Chromosomal microarray (CMA) or fluorescence in situ hybridisation (FISH) — detects 22q11.2 deletion (3Mb); NOT detected on standard karyotype; first-line test in ALL conotruncal CHD",
            "Echo_surveillance_BAV": "Annual transthoracic echo in BAV (NOTCH1): aortic valve area, mean gradient, AR severity, aortic root (annulus/sinus/STJ/ascending), LV dimensions",
            "Slit_lamp_Alagille": "Posterior embryotoxon (prominement Schwalbe line) 78% in Alagille syndrome — PATHOGNOMONIC even without full syndrome; baseline exam all suspected JAG1",
            "Spinal_Xray_Alagille": "AP + lateral spine X-ray — butterfly vertebrae (anterior arch cleft) 95% in Alagille; PATHOGNOMONIC; confirms diagnosis in unclear cholestatic infant",
            "MRI_semicircular_canal": "Temporal bone MRI: absent/hypoplastic semicircular canals in CHARGE syndrome — most specific radiological finding; vestibular testing",
            "Ionised_calcium_conotruncal": "Ionised calcium MANDATORY in all neonates with conotruncal CHD (IAA/truncus/TOF) BEFORE surgery — hypocalcaemia → cardiac arrest on cardiac bypass if untreated (22q11DS)",
            "T_cell_count_22q11": "CD4/CD8 lymphocyte subset absolute counts — T-cell lymphopenia in 22q11DS; live vaccine protocol: delay MMR/varicella until CD4 >200/μL",
            "Wrist_Xray_HOS": "Wrist X-ray — accessory carpal bones, fused carpals, hypoplastic radius — confirms Holt-Oram (TBX5) in subtle thumb anomaly cases",
        },
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    ov = overview()
    print(f"  Title: {ov['title']}")
    print(f"  Patients: {ov['n_patients']}  |  Genes: {ov['n_genes']}")
    print(f"  Seeds: {ov['seed_range']}")
    print("  Key pearls:")
    for p in ov["key_clinical_pearls"][:4]:
        print(f"    - {p[:100]}")

    print("\n=== BREAKDOWN (gene counts) ===")
    bk = breakdown()
    for g, info in bk["gene_breakdown"].items():
        print(f"  {g}: {info['n_patients']} patients | CHDs: {list(info['chd_type_distribution'].keys())[:3]}")

    print("\n=== DEFINITIONS (gene count) ===")
    df = definitions()
    print(f"  Genes defined: {list(df['gene_entries'].keys())}")
    print(f"  Anatomy terms: {len(df['anatomy_glossary'])}")
    print(f"  Treatment terms: {len(df['treatment_glossary'])}")
    print("\nAll checks passed.")
