#!/usr/bin/env python3
"""Hereditary-Lipodystrophy-Atlas — Complete 8-Gene Congenital & Familial Lipodystrophy Atlas
BSCL2   (seipin; 398 aa; 11q13.1; AR;
         Berardinelli-Seip CGL type 2 — most common CGL worldwide;
         ABSENT ALL metabolically-active fat from birth (subcutaneous, visceral, bone marrow);
         Serum leptin VERY LOW — metreleptin FDA-2014 indicated;
         seed SEED_BASE+0) .
AGPAT2  (1-acylglycerol-3-phosphate O-acyltransferase 2; 278 aa; 9q34.3; AR;
         Berardinelli-Seip CGL type 1 — preserves mechanical fat (palms, soles, periorbit, scalp);
         Distinguishes from BSCL2: mechanical fat PRESENT = type 1 not type 2;
         Severe metabolic disease: hypertriglyceridaemia + pancreatitis risk + T2D;
         seed SEED_BASE+1) .
LMNA    (lamin A/C; 664 aa; 1q22; AD;
         Dunnigan FPLD type 2 — most common hereditary partial lipodystrophy;
         Fat redistribution NOT absence: limb/gluteal wasting + neck/trunk/face accumulation;
         LMNA laminopathy: cardiac surveillance MANDATORY — ICD indicated if NSVT+HB+EF<45+syncope;
         seed SEED_BASE+2) .
PPARG   (peroxisome proliferator-activated receptor gamma; 505 aa; 3p25.2; AD;
         FPLD type 3 — dominant negative effect via ligand-binding domain mutations;
         TZD (thiazolidinedione) direct molecular target but dominant negative impairs response;
         Severe insulin resistance + dyslipidaemia + partial fat loss limbs;
         seed SEED_BASE+3) .
PLIN1   (perilipin-1; 522 aa; 15q26.1; AD;
         FPLD type 4 — rare; severe hypertriglyceridaemia + ectopic fat + insulin resistance;
         Delayed diagnosis common: subcutaneous fat loss mild; TG often >20 mmol/L → pancreatitis;
         PLIN1 scaffolds lipid droplet surface — loss → unregulated lipolysis;
         seed SEED_BASE+4) .
AKT2    (AKT serine/threonine kinase 2; 481 aa; 19q13.2; AD;
         GOF gain-of-function: severe generalised hypoglycaemia + macrosomia (Donohue-like);
         LOF loss-of-function: severe insulin resistance + partial lipodystrophy + T2D;
         AKT2 is central insulin-signalling node: PI3K→AKT2→GLUT4 translocation;
         seed SEED_BASE+5) .
CAV1    (caveolin-1; 178 aa; 7q31.2; AR/AD;
         AR biallelic → CGL type 3: generalised lipoatrophy + pulmonary arterial hypertension (PAH) overlap;
         AD heterozygous → FPLD type 7 or acquired partial lipodystrophy phenotype;
         Caveolae absent — DPPIV-linked: check BMPR2 panel if PAH coexists;
         seed SEED_BASE+6) .
ZMPSTE24 (zinc metalloprotease STE24; 475 aa; 1p34.2; AR;
         Mandibuloacral dysplasia type B (MADB) — lipodystrophy + progeroid + bone anomalies;
         Prelamin A accumulation (ZMPSTE24 fails to cleave prelamin A → farnesyl stays);
         DISTINGUISH from LMNA: ZMPSTE24 is AR + mandibular hypoplasia + clavicular acroosteolysis;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 2014-2021)
"""

import random

SEED_BASE = 2014

LIPO_GENES = [
    # -- BSCL2 — Berardinelli-Seip CGL type 2 (AR, most common CGL) -------------------
    {
        "gene": "BSCL2",
        "alt_name": "BSCL2 (Seipin / AR — CGL2 — Most Common CGL Worldwide — ALL Metabolically-Active Fat Absent — Metreleptin FDA-2014)",
        "protein": (
            "BSCL2 -- 11q13.1 AR -- BSCL2-398aa -- "
            "Most-Common-CGL-Worldwide-CGL-Type-2 -- "
            "ALL-Metabolically-Active-Fat-ABSENT-From-Birth-Subcutaneous-Visceral-Bone-Marrow -- "
            "Mechanical-Fat-Also-Absent-Distinguishes-from-AGPAT2 -- "
            "Serum-Leptin-VERY-LOW-Metreleptin-FDA-2014-Indicated"
        ),
        "locus": "11q13.1",
        "protein_size": "398 aa",
        "inheritance": "AR (autosomal recessive) — biallelic LOF",
        "age_of_onset": (
            "Birth: generalised lipoatrophy visible from neonatal period; "
            "Absent subcutaneous fat — prominent musculature (pseudoathleticism); "
            "Absent mechanical fat (palms, soles, periorbit, scalp) — distinguishes from CGL1/AGPAT2; "
            "Acanthosis nigricans: insulin resistance marker; "
            "Hepatomegaly: ectopic fat in liver → steatosis → cirrhosis risk; "
            "Hypertriglyceridaemia: TG often >10 mmol/L by age 10 — acute pancreatitis risk; "
            "T2D: early onset — median age 15–20 years; "
            "Polycystic ovaries: hyperandrogenaemia in females; "
            "Leptin: profoundly low (no fat = no leptin) → hypothalamic amenorrhoea possible; "
            "Intellectual disability: mild-moderate in BSCL2 subset (~50%) — distinguishes from AGPAT2"
        ),
        "key_biomarker": (
            "Serum leptin: VERY LOW (<2 ng/mL) — diagnostic; "
            "Fasting TG: markedly elevated (>10–50 mmol/L); "
            "Fasting insulin + HOMA-IR: severe insulin resistance; "
            "HbA1c: T2D by early adulthood; "
            "Liver enzymes: ALT/AST elevated → steatohepatitis; "
            "Androgens: testosterone + DHEA-S elevated in females; "
            "LH:FSH ratio: polycystic ovary pattern; "
            "Molecular: BSCL2 biallelic sequencing — p.A212V and p.S90L common variants; "
            "MRI whole-body fat: absent in all depots (visceral, subcutaneous, gluteal, bone marrow); "
            "Liver ultrasound/fibroscan: hepatic steatosis grading"
        ),
        "pathognomonic": (
            "Generalised lipoatrophy from birth + ABSENT mechanical fat (palms, soles, scalp) + low leptin = BSCL2; "
            "DISTINGUISH from AGPAT2/CGL1: CGL1 PRESERVES mechanical fat; CGL2 loses ALL fat including mechanical; "
            "Pseudoathleticism: prominent musculature due to absent subcutaneous fat + normal/increased muscle; "
            "Intellectual disability (~50% BSCL2): NOT seen in AGPAT2 — key differentiator; "
            "Hepatosplenomegaly: bone marrow fat also absent → extramedullary haematopoiesis in spleen; "
            "Leptin level: < 2 ng/mL is near-diagnostic for generalised lipodystrophy; "
            "MRI: absent signal in ALL subcutaneous/visceral/bone marrow fat compartments"
        ),
        "treatment": (
            "Metreleptin (Myalept): recombinant methionyl-leptin — subcutaneous daily injection; "
            "FDA approved 2014 for GENERALISED LIPODYSTROPHY (CGL + acquired GL); "
            "Mechanism: replaces absent leptin → improves hypothalamic energy sensing, TG, glucose, liver fat; "
            "Clinical trials: TG reduction ~60-70%; HbA1c improvement; liver fat reduction; "
            "Metreleptin NOT effective for partial lipodystrophies (leptin only mildly reduced); "
            "Diabetes management: insulin-sensitising agents (metformin, pioglitazone) + massive insulin doses; "
            "Hypertriglyceridaemia: fibrates (first-line), omega-3, low-fat diet (<15% calories from fat); "
            "Acute pancreatitis prevention: TG target <5 mmol/L — intensive lipid management; "
            "Liver disease: non-alcoholic steatohepatitis surveillance (fibroscan 2-yearly); "
            "Genetic counselling: AR — 25% recurrence; extended family screening"
        ),
        "critical_flags": [
            "BSCL2-CGL2-MOST-COMMON-CGL-WORLDWIDE",
            "BSCL2-ALL-FAT-ABSENT-INCLUDING-MECHANICAL-DISTINGUISHES-FROM-AGPAT2",
            "BSCL2-METRELEPTIN-FDA-2014-GENERALISED-LIPODYSTROPHY",
            "BSCL2-INTELLECTUAL-DISABILITY-50pct-NOT-IN-AGPAT2",
            "BSCL2-TG-PANCREATITIS-RISK-TARGET-BELOW-5mmol",
            "BSCL2-LIVER-STEATOHEPATITIS-FIBROSCAN-SURVEILLANCE",
            "BSCL2-PSEUDOATHLETICISM-VISIBLE-FROM-BIRTH",
        ],
        "seed": SEED_BASE + 0,
    },
    # -- AGPAT2 — CGL type 1 (AR, mechanical fat preserved) ---------------------------
    {
        "gene": "AGPAT2",
        "alt_name": "AGPAT2 (1-AG-3-P O-acyltransferase 2 / AR — CGL1 — Mechanical Fat PRESERVED — Severe Metabolic Disease)",
        "protein": (
            "AGPAT2 -- 9q34.3 AR -- AGPAT2-278aa -- "
            "Berardinelli-Seip-CGL-Type-1 -- "
            "Metabolically-Active-Fat-ABSENT-Mechanical-Fat-PRESERVED-Palms-Soles-Periorbit-Scalp -- "
            "Severe-Hypertriglyceridaemia-T2D-Hepatomegaly -- "
            "AGPAT2-Converts-LPA-to-PA-in-Glycerophospholipid-Synthesis-in-Adipocytes"
        ),
        "locus": "9q34.3",
        "protein_size": "278 aa",
        "inheritance": "AR (autosomal recessive) — biallelic LOF",
        "age_of_onset": (
            "Birth: generalised lipoatrophy of metabolically active depots; "
            "PRESERVED: mechanical fat in palms, soles, periorbitally, scalp — KEY distinguishing feature vs BSCL2; "
            "Acanthosis nigricans: insulin resistance; "
            "Hepatomegaly: liver ectopic fat; hepatic steatosis → cirrhosis; "
            "Hypertriglyceridaemia: TG >10 mmol/L; acute pancreatitis risk; "
            "T2D: onset puberty/early adulthood; "
            "No intellectual disability (distinguishes from BSCL2); "
            "Polycystic ovary syndrome in females; "
            "Bone cysts: metaphyseal cysts on X-ray (characteristic for CGL1)"
        ),
        "key_biomarker": (
            "Serum leptin: very low; "
            "TG: markedly elevated — acute pancreatitis threshold watch; "
            "HbA1c + HOMA-IR: severe insulin resistance; "
            "Liver enzymes: ALT/AST elevated steatohepatitis; "
            "X-ray: metaphyseal bone cysts — PATHOGNOMONIC for CGL1/AGPAT2; "
            "MRI: absent metabolic fat; PRESERVED palmar/plantar/periorbit fat; "
            "Molecular: AGPAT2 biallelic sequencing — p.R96H, p.E213K common"
        ),
        "pathognomonic": (
            "Generalised lipoatrophy from birth + PRESERVED mechanical fat + bone cysts = AGPAT2/CGL1; "
            "DISTINGUISH from BSCL2/CGL2: CGL2 has absent mechanical fat + intellectual disability; "
            "CGL1: preserved fat in palms, soles, periorbit, scalp — palpable fat present; "
            "Metaphyseal bone cysts: X-ray finding PATHOGNOMONIC for CGL1 — not seen in CGL2; "
            "No intellectual disability: AGPAT2 does NOT affect CNS; "
            "Serum leptin: low but often higher than BSCL2 (some mechanical fat preserved)"
        ),
        "treatment": (
            "Metreleptin: FDA-approved for generalised lipodystrophy (CGL1 + CGL2) — improves TG, HbA1c, liver fat; "
            "Hypertriglyceridaemia: fibrates (fenofibrate/bezafibrate) + very-low-fat diet; "
            "T2D: insulin sensitisers + insulin (often massive doses needed); "
            "Pancreatitis prevention: TG <5 mmol/L target; apheresis if TG >50 mmol/L with pancreatitis risk; "
            "Hepatic surveillance: fibroscan every 2 years; annual liver biochemistry; "
            "Orthopaedic: bone cysts — low risk of fracture, monitor; "
            "Reproductive: metformin for PCOS; fertility counselling; "
            "Genetic counselling: AR — 25% recurrence"
        ),
        "critical_flags": [
            "AGPAT2-CGL1-MECHANICAL-FAT-PRESERVED-PALMS-SOLES-PERIORBIT",
            "AGPAT2-BONE-CYSTS-METAPHYSEAL-PATHOGNOMONIC-CGL1",
            "AGPAT2-NO-INTELLECTUAL-DISABILITY-UNLIKE-BSCL2",
            "AGPAT2-METRELEPTIN-FDA-APPROVED-GENERALISED",
            "AGPAT2-TG-PANCREATITIS-RISK",
            "AGPAT2-HEPATIC-STEATOSIS-CIRRHOSIS-RISK",
            "AGPAT2-DISTINGUISH-FROM-BSCL2-MECHANICAL-FAT-KEY",
        ],
        "seed": SEED_BASE + 1,
    },
    # -- LMNA — Dunnigan FPLD type 2 (AD, most common hereditary FPLD) -----------------
    {
        "gene": "LMNA",
        "alt_name": "LMNA (Lamin A/C / AD — FPLD2 Dunnigan — Most Common Hereditary FPLD — Cardiac Surveillance MANDATORY — ICD Risk)",
        "protein": (
            "LMNA -- 1q22 AD -- LMNA-664aa -- "
            "Familial-Partial-Lipodystrophy-Dunnigan-FPLD2 -- "
            "Fat-REDISTRIBUTION-Not-Absence-Limb-Gluteal-Wasting-Neck-Trunk-Face-Accumulation -- "
            "LMNA-Laminopathy-Cardiac-Surveillance-MANDATORY-ICD-Indication-NSVT-HB-EF45 -- "
            "Hotspot-p.R482W-p.R482Q-Lipodystrophy-Exon8-HGMD-FPLD2"
        ),
        "locus": "1q22",
        "protein_size": "664 aa",
        "inheritance": "AD (autosomal dominant) — heterozygous missense, usually exon 8",
        "age_of_onset": (
            "Puberty/adolescence: limb + gluteal fat loss begins; "
            "Fat redistribution (NOT absence): subcutaneous fat LOST from limbs/gluteal + GAINED in neck/trunk/face/labial; "
            "Cushingoid-like face: fat accumulation in face and neck despite limb fat loss — diagnostic trap (not Cushing's); "
            "Acanthosis nigricans + polycystic ovaries: insulin resistance; "
            "Hypertriglyceridaemia + T2D: metabolic disease from 2nd–3rd decade; "
            "Cardiac disease: 1st-degree heart block → complete heart block → NSVT → dilated cardiomyopathy + SCD; "
            "Skeletal muscle: proximal myopathy possible (same LMNA gene — laminopathy spectrum); "
            "Carpal tunnel syndrome + early atherosclerosis reported"
        ),
        "key_biomarker": (
            "Serum leptin: LOW-NORMAL to low (partial lipodystrophy — some fat remains); "
            "Fasting TG: elevated; LDL: elevated (dyslipidaemia pattern); "
            "HbA1c + HOMA-IR: insulin resistance; "
            "ECG + Holter: PR prolongation, NSVT, heart block — MANDATORY; "
            "Echo + cardiac MRI: LV function + fibrosis; "
            "Serum CK: elevated if skeletal myopathy component; "
            "Testosterone/DHEA-S: elevated females; LH:FSH; "
            "Molecular: LMNA exon 8 hotspot (p.R482W, p.R482Q) — targeted sequencing first; "
            "MRI body: limb fat absent; neck/trunk/face fat increased (classic redistribution pattern)"
        ),
        "pathognomonic": (
            "Limb + gluteal fat wasting + neck/trunk/face fat accumulation + metabolic syndrome + cardiac conduction disease = LMNA-FPLD2; "
            "DISTINGUISH from Cushing's: FPLD2 has NO cortisol excess; limb fat ABSENT (not just relatively thin); "
            "DISTINGUISH from acquired lipodystrophy: onset puberty + AD family history + LMNA variant; "
            "DISTINGUISH from PPARG/FPLD3: LMNA has cardiac disease; PPARG does not; "
            "Cardiac risk: NSVT + any of (HB or EF<45% or syncope) → ICD implant recommended; "
            "Exon 8 hotspot: p.R482W and p.R482Q account for >80% of FPLD2 — targeted testing first; "
            "Labial fat accumulation in females: prominent labia majora from fat redistribution — pathognomonic"
        ),
        "treatment": (
            "Metreleptin: NOT routinely effective (partial lipodystrophy — leptin only moderately low; "
            "no FDA approval for FPLD specifically — used off-label in severe cases with very low leptin); "
            "Metabolic: metformin (insulin resistance) + fibrates (hypertriglyceridaemia) + statins (LDL); "
            "Pioglitazone: PPARγ agonist — modest benefit for insulin resistance in FPLD2; "
            "Cardiac: annual ECG + Holter + echo; "
            "ICD indication: NSVT + (LBBB or HV>70ms or EF<45% or syncope) per ESC 2022 HCM/LMNA guidelines; "
            "Pacemaker: complete heart block; "
            "Exercise restriction: intense exertion avoided until cardiac assessment; "
            "Genetic counselling: AD — 50% offspring risk; cascade family cardiac screening"
        ),
        "critical_flags": [
            "LMNA-FPLD2-DUNNIGAN-MOST-COMMON-HEREDITARY-FPLD",
            "LMNA-CARDIAC-SURVEILLANCE-MANDATORY-ANNUAL-ECG-HOLTER-ECHO",
            "LMNA-ICD-INDICATION-NSVT-HB-EF45-SYNCOPE",
            "LMNA-FAT-REDISTRIBUTION-NOT-ABSENCE-CUSHINGOID-TRAP",
            "LMNA-EXON8-HOTSPOT-R482W-R482Q-80pct-FPLD2",
            "LMNA-METRELEPTIN-NOT-FDA-APPROVED-FPLD-OFF-LABEL-ONLY",
            "LMNA-LABIAL-FAT-ACCUMULATION-FEMALES-PATHOGNOMONIC",
        ],
        "seed": SEED_BASE + 2,
    },
    # -- PPARG — FPLD type 3 (AD, dominant negative, TZD target) -----------------------
    {
        "gene": "PPARG",
        "alt_name": "PPARG (PPARγ / AD — FPLD3 — Dominant Negative LBD Mutations — TZD Direct Target — Severe Insulin Resistance)",
        "protein": (
            "PPARG -- 3p25.2 AD -- PPARG-505aa -- "
            "Familial-Partial-Lipodystrophy-Type-3-FPLD3 -- "
            "Dominant-Negative-Ligand-Binding-Domain-Mutations-Impair-PPARgamma-Coactivation -- "
            "TZD-Thiazolidinedione-Is-Direct-PPARgamma-Ligand-Dominant-Negative-Impairs-TZD-Benefit -- "
            "Severe-Insulin-Resistance-Dyslipidaemia-Partial-Limb-Fat-Loss"
        ),
        "locus": "3p25.2",
        "protein_size": "505 aa",
        "inheritance": "AD (autosomal dominant) — heterozygous missense (usually LBD)",
        "age_of_onset": (
            "Adult onset: partial fat loss from limbs/buttocks; "
            "Onset often 20s–30s; "
            "Hypertriglyceridaemia: often severe; acute pancreatitis risk; "
            "Severe insulin resistance: T2D; "
            "Hyperandrogenaemia in females: PCOS; "
            "Fat redistribution: some visceral fat increase; "
            "Variable expressivity: penetrance variable — heterozygous relatives may be asymptomatic or have T2D only; "
            "Hepatic steatosis: ectopic fat in liver; "
            "Muscle insulin resistance: IRS/PI3K signalling impaired downstream of absent PPARγ transcription"
        ),
        "key_biomarker": (
            "Serum leptin: low-normal; "
            "TG: markedly elevated; LDL-C: elevated; HDL-C: low; "
            "HbA1c + HOMA-IR: severe insulin resistance; "
            "Testosterone/DHEA-S elevated females; "
            "Liver enzymes: steatohepatitis pattern; "
            "Molecular: PPARG LBD hotspot variants — p.P467L, p.V290M, p.R397C most common; "
            "Adiponectin: markedly low (PPARγ drives adiponectin expression); "
            "Functional assay: dominant negative effect on PPARγ-mediated transcription"
        ),
        "pathognomonic": (
            "Partial lipodystrophy (limb) + severe insulin resistance + very low adiponectin + PPARG LBD variant = FPLD3; "
            "DISTINGUISH from LMNA-FPLD2: PPARG lacks cardiac conduction disease; "
            "DISTINGUISH from acquired partial lipodystrophy: PPARG has family history + onset 2nd–3rd decade; "
            "TZD (pioglitazone/rosiglitazone): these drugs are PPARγ ligands — dominant negative IMPAIRS response; "
            "Adiponectin: severely low — PPARγ is the master regulator of adiponectin gene (ADIPOQ); "
            "Variable expressivity: family members may have T2D/dyslipidaemia without visible fat loss"
        ),
        "treatment": (
            "Pioglitazone/TZD: direct PPARγ ligand — may partially overcome dominant negative (clinical effect variable); "
            "Some patients respond; others do not — trial warranted; "
            "Metformin: insulin sensitiser for T2D; "
            "Fibrates + omega-3: hypertriglyceridaemia; acute pancreatitis prevention; "
            "Statin: LDL-C management; "
            "Metreleptin: not typically effective (partial lipodystrophy; leptin not severely reduced); "
            "Hepatic: NASH management (metabolic approach); "
            "Genetic counselling: AD — 50% recurrence; variable expressivity — screen relatives for T2D/dyslipidaemia"
        ),
        "critical_flags": [
            "PPARG-FPLD3-DOMINANT-NEGATIVE-LBD-MUTATIONS",
            "PPARG-TZD-DIRECT-TARGET-DOMINANT-NEGATIVE-IMPAIRS-RESPONSE",
            "PPARG-ADIPONECTIN-MARKEDLY-LOW-PPARgamma-MASTER-REGULATOR",
            "PPARG-NO-CARDIAC-DISEASE-UNLIKE-LMNA",
            "PPARG-VARIABLE-EXPRESSIVITY-FAMILY-T2D-SCREEN",
            "PPARG-PANCREATITIS-TG-MONITORING-MANDATORY",
            "PPARG-P467L-V290M-R397C-LBD-HOTSPOTS",
        ],
        "seed": SEED_BASE + 3,
    },
    # -- PLIN1 — FPLD type 4 (AD, severe hypertriglyceridaemia) -----------------------
    {
        "gene": "PLIN1",
        "alt_name": "PLIN1 (Perilipin-1 / AD — FPLD4 — Severe Hypertriglyceridaemia TG>20mmol — Unregulated Lipolysis — Delayed Diagnosis)",
        "protein": (
            "PLIN1 -- 15q26.1 AD -- PLIN1-522aa -- "
            "Familial-Partial-Lipodystrophy-Type-4-FPLD4 -- "
            "Perilipin-1-Scaffolds-Lipid-Droplet-Surface-Loss-Causes-Unregulated-Lipolysis -- "
            "Severe-Hypertriglyceridaemia-TG-Often-Above-20mmol-Pancreatitis-Risk -- "
            "Delayed-Diagnosis-Subcutaneous-Fat-Loss-Subtle-Insulin-Resistance-Severe"
        ),
        "locus": "15q26.1",
        "protein_size": "522 aa",
        "inheritance": "AD (autosomal dominant) — heterozygous frameshift/premature stop",
        "age_of_onset": (
            "Adult onset (typically 3rd–4th decade); "
            "Partial fat loss from limbs/buttocks — often MILD and easily overlooked; "
            "Very severe hypertriglyceridaemia: TG >20–50 mmol/L common — pancreatitis major risk; "
            "Ectopic fat: liver (steatohepatitis), muscle, pancreas; "
            "Insulin resistance: severe despite modest visible lipodystrophy; "
            "PLIN1 normally gates lipolysis (HSL/ATGL activation); absent → uncontrolled FFA release; "
            "Free fatty acid spillover: ectopic fat deposition; "
            "T2D: usually present; "
            "Family history: often multiple members with severe hypertriglyceridaemia labelled 'familial hypertriglyceridaemia'"
        ),
        "key_biomarker": (
            "TG: MARKEDLY ELEVATED >20 mmol/L — may exceed 50 mmol/L; "
            "HDL-C: very low; "
            "HbA1c + HOMA-IR: severe insulin resistance; "
            "Free fatty acids (FFA): elevated — unregulated lipolysis; "
            "Leptin: low-normal; "
            "Liver enzymes: NASH pattern; "
            "Molecular: PLIN1 sequencing — frameshift/premature stop in C-terminal domain; "
            "Adiponectin: low; "
            "Lipase activity: normal (HSL/ATGL normal — regulatory protein absent)"
        ),
        "pathognomonic": (
            "Severe hypertriglyceridaemia (>20 mmol/L) + mild partial lipodystrophy + family history pancreatitis = PLIN1; "
            "PLIN1 diagnosis is often missed: fat loss is subtle (partial, limb) — metabolic disease is the presenting feature; "
            "Unregulated lipolysis: FFA spillover → ectopic fat everywhere → insulin resistance; "
            "Distinguish from multifactorial hypertriglyceridaemia: PLIN1 has family history + lipodystrophy phenotype; "
            "Pancreatitis prevention: TG must be aggressively managed — fibrates mandatory; "
            "Frameshift/truncating variants: C-terminal PKA-binding domain most clinically significant"
        ),
        "treatment": (
            "Fibrates (fenofibrate/bezafibrate): MANDATORY — first-line for TG reduction; "
            "Omega-3 fatty acids: high-dose (4g/day EPA/DHA); "
            "Very-low-fat diet (<15% of calories as fat); "
            "Pancreatitis prevention: TG target <5 mmol/L — hospital admission + insulin infusion if >50 mmol/L; "
            "Metformin: insulin resistance; "
            "Metreleptin: not typically effective (partial lipodystrophy); "
            "Volanesorsen (antisense oligonucleotide for APOC3): investigational but may help severe cases; "
            "Hepatic NASH: metabolic control; "
            "Genetic counselling: AD — 50% recurrence; family screening for hypertriglyceridaemia"
        ),
        "critical_flags": [
            "PLIN1-TG-ABOVE-20mmol-PANCREATITIS-MANDATORY-MANAGEMENT",
            "PLIN1-DELAYED-DIAGNOSIS-SUBTLE-FAT-LOSS",
            "PLIN1-UNREGULATED-LIPOLYSIS-FFA-SPILLOVER",
            "PLIN1-FIBRATES-OMEGA3-MANDATORY-TG-CONTROL",
            "PLIN1-FAMILY-HISTORY-PANCREATITIS-SCREEN",
            "PLIN1-ECTOPIC-FAT-LIVER-MUSCLE-PANCREAS",
            "PLIN1-FRAMESHIFT-C-TERMINAL-DOMAIN-MOST-COMMON",
        ],
        "seed": SEED_BASE + 4,
    },
    # -- AKT2 — AKT2 GOF/LOF lipodystrophy + severe insulin resistance ----------------
    {
        "gene": "AKT2",
        "alt_name": "AKT2 (AKT Serine/Threonine Kinase 2 / AD — LOF Severe Insulin Resistance + Partial Lipodystrophy — GOF Hypoglycaemia)",
        "protein": (
            "AKT2 -- 19q13.2 AD -- AKT2-481aa -- "
            "PI3K-AKT2-GLUT4-Central-Insulin-Signalling-Node -- "
            "LOF-AD-Severe-Insulin-Resistance-Partial-Lipodystrophy-T2D -- "
            "GOF-AD-Severe-Hypoglycaemia-Macrosomia-Donohue-Rabson-Mendenhall-Overlap -- "
            "mTOR-S6K-FOXO-Downstream-AKT2-Substrate-Phosphorylation"
        ),
        "locus": "19q13.2",
        "protein_size": "481 aa",
        "inheritance": "AD (autosomal dominant) — both GOF and LOF variants described",
        "age_of_onset": (
            "LOF (more common clinical presentation): "
            "Adult onset severe insulin resistance + partial lipodystrophy; T2D; "
            "Fat loss from limbs; visceral fat accumulation; "
            "Acanthosis nigricans; PCOS females; "
            "GOF (de novo): "
            "Neonatal: macrosomia + severe generalised hypoglycaemia (persistent neonatal hypoglycaemia); "
            "GOF AKT2: constitutively active insulin signalling → hypoglycaemia + GLUT4 upregulation; "
            "GOF phenotype overlaps with Donohue syndrome and Rabson-Mendenhall syndrome but different mechanism; "
            "AKT2 is the critical kinase: PI3K→PIP3→PDK1→AKT2→GLUT4 vesicle translocation to plasma membrane"
        ),
        "key_biomarker": (
            "LOF: fasting insulin MARKEDLY elevated; HOMA-IR >10; HbA1c elevated; "
            "C-peptide elevated (not insulin deficiency); "
            "Leptin: low-normal (partial lipodystrophy); "
            "GOF: plasma glucose VERY LOW (<2 mmol/L); insulin: inappropriately elevated relative to glucose; "
            "C-peptide: suppressed in GOF (hypoglycaemia is driven by AKT2 not pancreatic insulin); "
            "Molecular: AKT2 sequencing — LOF (p.R274H) vs GOF (p.E17K, p.W80R) variants; "
            "IGFBP-1: low (insulin-stimulated FOXO1 inhibition reduces IGFBP-1 transcription); "
            "Adiponectin: low"
        ),
        "pathognomonic": (
            "Severe insulin resistance + partial lipodystrophy (limb) + AD family history + AKT2 variant = AKT2-LOF; "
            "Neonatal macrosomia + persistent hypoglycaemia unresponsive to glucose + AKT2 GOF = AKT2-GOF; "
            "DISTINGUISH LOF from LMNA: AKT2 has no cardiac disease; "
            "DISTINGUISH LOF from PPARG: AKT2 is a kinase not nuclear receptor — TZD therapy not indicated; "
            "GOF trap: persistent neonatal hypoglycaemia → workup for hyperinsulinism → AKT2 GOF missed unless sequenced; "
            "Diazoxide resistance: GOF AKT2 hypoglycaemia may not respond to diazoxide (different from KATP-HI)"
        ),
        "treatment": (
            "LOF: metformin (first-line insulin sensitiser); "
            "mTOR inhibitors (rapamycin/sirolimus): investigational — inhibit downstream mTOR which is overactive in AKT2-LOF; "
            "Fibrates + statins: dyslipidaemia; "
            "GLP-1 agonists + SGLT-2 inhibitors: T2D management; "
            "Metreleptin: not standard (partial lipodystrophy); "
            "GOF: diazoxide (may work partially); "
            "Octreotide: suppresses GLP-1 downstream; "
            "Sirolimus: mTOR inhibition reduces constitutive AKT2 signalling — GOF responds better than LOF; "
            "Genetic counselling: AD — 50% recurrence; distinguish GOF vs LOF critical for management"
        ),
        "critical_flags": [
            "AKT2-LOF-SEVERE-INSULIN-RESISTANCE-PARTIAL-LIPODYSTROPHY",
            "AKT2-GOF-NEONATAL-HYPOGLYCAEMIA-MACROSOMIA-DIAZOXIDE-MAY-FAIL",
            "AKT2-DISTINGUISH-LOF-FROM-GOF-CRITICAL-OPPOSITE-PHENOTYPES",
            "AKT2-MTOR-SIROLIMUS-INVESTIGATIONAL",
            "AKT2-NO-CARDIAC-DISEASE-UNLIKE-LMNA",
            "AKT2-C-PEPTIDE-SUPPRESSED-IN-GOF-NOT-PANCREATIC",
            "AKT2-DIAZOXIDE-RESISTANCE-GOF-KATP-CHANNEL-INTACT",
        ],
        "seed": SEED_BASE + 5,
    },
    # -- CAV1 — CGL type 3 + FPLD7 (AR/AD, PAH overlap) ------------------------------
    {
        "gene": "CAV1",
        "alt_name": "CAV1 (Caveolin-1 / AR CGL3 + AD FPLD7 — PAH Overlap — Caveolae ABSENT — Check BMPR2 If PAH Coexists)",
        "protein": (
            "CAV1 -- 7q31.2 AR/AD -- CAV1-178aa -- "
            "AR-Biallelic-LOF-CGL-Type-3-Generalised-Lipoatrophy-PAH-Overlap -- "
            "AD-Heterozygous-FPLD7-Partial-Lipodystrophy-Acquired-Phenotype -- "
            "Caveolae-Absent-Plasma-Membrane-Invaginations-DPPIV-Cholesterol-eNOS-Signalling -- "
            "BMPR2-Panel-If-PAH-Coexists-CAV1-Not-Commonest-Hereditary-PAH-Gene"
        ),
        "locus": "7q31.2",
        "protein_size": "178 aa",
        "inheritance": "AR (biallelic) for CGL3; AD (heterozygous) for FPLD7",
        "age_of_onset": (
            "CGL3 (AR): generalised lipoatrophy from birth; "
            "Mechanical fat: variable — some preserved (intermediate between CGL1 and CGL2); "
            "PAH: pulmonary arterial hypertension develops in subset — median age 20–35 years; "
            "Dyslipidaemia + insulin resistance + T2D; "
            "Hepatomegaly: steatohepatitis; "
            "FPLD7 (AD): partial lipodystrophy adult onset; milder metabolic disease than CGL3; "
            "Female sex: females more severely affected in both AR and AD forms; "
            "Caveolae absent: affects lipid trafficking, eNOS signalling, mechano-sensing"
        ),
        "key_biomarker": (
            "Serum leptin: low (generalised) or low-normal (partial); "
            "TG + HbA1c: metabolic disease markers; "
            "Echocardiogram + right heart catheterisation: PAH screening (TR velocity; mPAP if echo abnormal); "
            "Electron microscopy of fibroblasts: caveolae absent — diagnostic if available; "
            "Molecular: CAV1 biallelic sequencing for CGL3; heterozygous for FPLD7; "
            "BMPR2 panel: if PAH prominent, check BMPR2 + other PAH genes (CAV1 rare cause of PAH); "
            "Adiponectin: low; "
            "LFTs: hepatic steatosis"
        ),
        "pathognomonic": (
            "Generalised lipoatrophy + PAH (variable) + absent caveolae on EM = CAV1-CGL3; "
            "DISTINGUISH from CGL1/AGPAT2: CAV1 may have PAH; AGPAT2 has bone cysts not PAH; "
            "DISTINGUISH from CGL2/BSCL2: BSCL2 has intellectual disability; CAV1 does not; "
            "PAH + CGL phenotype: RARE combination — CAV1 should be sequenced; "
            "Note: CAV1 is also listed in hereditary PAH atlas (minority cause of PAH); "
            "FPLD7 (AD): female predominance; metabolic syndrome without severe lipoatrophy; "
            "Caveolae absent: key mechanistic finding on electron microscopy of fibroblasts or fat"
        ),
        "treatment": (
            "CGL3 metabolic: metreleptin (generalised lipodystrophy — FDA approved); fibrates; metformin; "
            "PAH: ERA (bosentan/ambrisentan) + PDE5i (sildenafil) + prostacyclin (if severe) — standard PAH protocol; "
            "PAH risk stratification: annual echo from diagnosis; RHC if TR velocity elevated; "
            "FPLD7: metformin + fibrates + statins; metreleptin not typically effective; "
            "Liver: NASH management; "
            "Genetic counselling: AR — 25% CGL3; AD — 50% FPLD7; "
            "Distinguish AR vs AD for family planning; PAH surveillance for all CAV1 biallelic"
        ),
        "critical_flags": [
            "CAV1-CGL3-AR-PAH-OVERLAP-SCREEN-ALL-CGL3-PATIENTS",
            "CAV1-CAVEOLAE-ABSENT-PATHOGNOMONIC-ON-EM",
            "CAV1-BMPR2-PANEL-IF-PAH-PROMINENT",
            "CAV1-AR-CGL3-VS-AD-FPLD7-DIFFERENT-PHENOTYPES",
            "CAV1-METRELEPTIN-FOR-GENERALISED-CGL3-FORM",
            "CAV1-FEMALE-SEX-MORE-SEVERELY-AFFECTED",
            "CAV1-PAH-ERA-PDE5I-PROSTACYCLIN-STANDARD-PROTOCOL",
        ],
        "seed": SEED_BASE + 6,
    },
    # -- ZMPSTE24 — Mandibuloacral dysplasia type B (AR, prelamin A accumulation) ------
    {
        "gene": "ZMPSTE24",
        "alt_name": "ZMPSTE24 (Zinc Metalloprotease STE24 / AR — MADB — Lipodystrophy + Progeroid + Bone — Prelamin A Accumulation — Distinguish from LMNA)",
        "protein": (
            "ZMPSTE24 -- 1p34.2 AR -- ZMPSTE24-475aa -- "
            "Mandibuloacral-Dysplasia-Type-B-MADB-Lipodystrophy-Progeroid-Bone -- "
            "Prelamin-A-Accumulation-ZMPSTE24-Cleaves-Farnesylated-Prelamin-A-Last-18aa -- "
            "DISTINGUISH-from-LMNA-FPLD2-AR-vs-AD-Mandibular-Hypoplasia-Acro-Osteolysis-Progeroid -- "
            "HGPB-Progeria-Like-Restricted-Growth-Alopecia-Joint-Contractures"
        ),
        "locus": "1p34.2",
        "protein_size": "475 aa",
        "inheritance": "AR (autosomal recessive) — biallelic LOF",
        "age_of_onset": (
            "Infancy/childhood: lipodystrophy + progeroid features; "
            "Lipoatrophy: generalised OR partial (TYPE B = generalised includes trunk; TYPE A/MADA = LMNA partial); "
            "Progeroid features: premature ageing appearance, alopecia (scalp hair loss), skin atrophy; "
            "Bone anomalies: mandibular hypoplasia (small jaw) + maxillary hypoplasia; "
            "Acro-osteolysis: resorption of clavicular ends + distal phalanges (acroosteolysis) PATHOGNOMONIC; "
            "Joint contractures: digits + elbows; "
            "Restricted growth: short stature; "
            "Metabolic: insulin resistance + T2D + dyslipidaemia (less severe than CGL1/2); "
            "Prelamin A accumulation: ZMPSTE24 fails to cleave 18 C-terminal aa → farnesylated prelamin A persists"
        ),
        "key_biomarker": (
            "X-ray: mandibular hypoplasia + acroosteolysis (clavicle + distal phalanges) PATHOGNOMONIC for MADB; "
            "Serum prelamin A: elevated (Western blot/ELISA — research tool); "
            "Fasting insulin + HbA1c: insulin resistance; "
            "TG + HDL: dyslipidaemia; "
            "Molecular: ZMPSTE24 biallelic sequencing — p.W340* common founder; "
            "Skin biopsy fibroblasts: nuclear blebbing/misshapen nuclei on microscopy (progerin-like effect); "
            "Echocardiogram: cardiomyopathy surveillance (less common than LMNA but monitor); "
            "Leptin: low-normal"
        ),
        "pathognomonic": (
            "Lipodystrophy + progeroid appearance + mandibular hypoplasia + acroosteolysis = ZMPSTE24/MADB; "
            "DISTINGUISH from LMNA-FPLD2: ZMPSTE24 is AR + progeroid + bone anomalies; LMNA-FPLD2 is AD + cardiac; "
            "DISTINGUISH from Hutchinson-Gilford progeria (HGPS): HGPS is de novo LMNA p.G608G; ZMPSTE24 is biallelic AR; "
            "Acroosteolysis: X-ray finding — clavicle resorption + distal phalangeal resorption PATHOGNOMONIC; "
            "Mandibuloacral dysplasia TYPE A vs B: TYPE A = LMNA AR (partial FPLD); TYPE B = ZMPSTE24 AR (generalised); "
            "Prelamin A pathway: ZMPSTE24 is the protease step; LMNA encodes the substrate — both affect nuclear lamina"
        ),
        "treatment": (
            "Metabolic: metformin for insulin resistance; fibrates for dyslipidaemia; "
            "Metreleptin: if generalised lipodystrophy + very low leptin — may be beneficial; "
            "Farnesyltransferase inhibitors (FTIs): e.g. lonafarnib — prevents farnesylation of prelamin A; "
            "FDA approved lonafarnib for HGPS (Hutchinson-Gilford) — may benefit ZMPSTE24 by same mechanism; "
            "Statin: additional effect — statins reduce farnesyl pyrophosphate production; "
            "Dental/maxillofacial: orthodontic management for mandibular hypoplasia; "
            "Orthopaedic: physio for joint contractures; "
            "Cardiac: echo surveillance; "
            "Genetic counselling: AR — 25% recurrence; "
            "Distinguish from HGPS: ZMPSTE24 less fatal than HGPS (median survival HGPS ~14yr; MADB longer)"
        ),
        "critical_flags": [
            "ZMPSTE24-ACROOSTEOLYSIS-CLAVICLE-DISTAL-PHALANGES-PATHOGNOMONIC",
            "ZMPSTE24-MADB-MANDIBULAR-HYPOPLASIA-PROGEROID",
            "ZMPSTE24-AR-UNLIKE-LMNA-FPLD2-AD",
            "ZMPSTE24-PRELAMIN-A-ACCUMULATION-FARNESYL-STAYS",
            "ZMPSTE24-LONAFARNIB-FTI-HGPS-APPROVED-MAY-BENEFIT",
            "ZMPSTE24-MADA-LMNA-VS-MADB-ZMPSTE24-BOTH-MANDIBULOACRAL",
            "ZMPSTE24-DISTINGUISH-FROM-HGPS-LMNA-DE-NOVO-VS-BIALLELIC",
        ],
        "seed": SEED_BASE + 7,
    },
]


def _make_cohort(gene_entry: dict, seed: int, n: int = 40) -> list:
    rng = random.Random(seed)
    gene = gene_entry["gene"]
    cohort = []
    for i in range(n):
        age = rng.randint(1, 65)
        sex = rng.choice(["M", "F"])

        if gene == "BSCL2":
            generalised_lipoatrophy = True
            mechanical_fat_absent   = True   # CGL2 pathognomonic
            partial_lipodystrophy   = False
            severe_hypertrigly      = rng.random() < 0.90
            t2d                     = rng.random() < 0.85
            pancreatitis            = rng.random() < 0.35
            pah                     = False
            progeroid               = False
            bone_anomaly            = False
            intellectual_disability = rng.random() < 0.50   # CGL2 ~50%
            hepatomegaly            = rng.random() < 0.90
            cardiac_conduction      = False
            metreleptin_eligible    = True
            severe_ir               = True
            hypoglycaemia           = False

        elif gene == "AGPAT2":
            generalised_lipoatrophy = True
            mechanical_fat_absent   = False  # CGL1 preserves mechanical fat
            partial_lipodystrophy   = False
            severe_hypertrigly      = rng.random() < 0.88
            t2d                     = rng.random() < 0.82
            pancreatitis            = rng.random() < 0.30
            pah                     = False
            progeroid               = False
            bone_anomaly            = rng.random() < 0.80   # metaphyseal cysts
            intellectual_disability = False
            hepatomegaly            = rng.random() < 0.88
            cardiac_conduction      = False
            metreleptin_eligible    = True
            severe_ir               = True
            hypoglycaemia           = False

        elif gene == "LMNA":
            generalised_lipoatrophy = False
            mechanical_fat_absent   = False
            partial_lipodystrophy   = True   # FPLD2
            severe_hypertrigly      = rng.random() < 0.70
            t2d                     = rng.random() < 0.75
            pancreatitis            = rng.random() < 0.10
            pah                     = False
            progeroid               = False
            bone_anomaly            = False
            intellectual_disability = False
            hepatomegaly            = rng.random() < 0.60
            cardiac_conduction      = rng.random() < 0.70   # MANDATORY surveillance
            metreleptin_eligible    = False  # partial, off-label only
            severe_ir               = rng.random() < 0.80
            hypoglycaemia           = False

        elif gene == "PPARG":
            generalised_lipoatrophy = False
            mechanical_fat_absent   = False
            partial_lipodystrophy   = True   # FPLD3
            severe_hypertrigly      = rng.random() < 0.80
            t2d                     = rng.random() < 0.78
            pancreatitis            = rng.random() < 0.20
            pah                     = False
            progeroid               = False
            bone_anomaly            = False
            intellectual_disability = False
            hepatomegaly            = rng.random() < 0.65
            cardiac_conduction      = False
            metreleptin_eligible    = False
            severe_ir               = rng.random() < 0.85
            hypoglycaemia           = False

        elif gene == "PLIN1":
            generalised_lipoatrophy = False
            mechanical_fat_absent   = False
            partial_lipodystrophy   = True   # FPLD4 — subtle
            severe_hypertrigly      = rng.random() < 0.95   # HALLMARK
            t2d                     = rng.random() < 0.75
            pancreatitis            = rng.random() < 0.50   # HIGH RISK with TG >20
            pah                     = False
            progeroid               = False
            bone_anomaly            = False
            intellectual_disability = False
            hepatomegaly            = rng.random() < 0.70
            cardiac_conduction      = False
            metreleptin_eligible    = False
            severe_ir               = rng.random() < 0.80
            hypoglycaemia           = False

        elif gene == "AKT2":
            generalised_lipoatrophy = False
            mechanical_fat_absent   = False
            partial_lipodystrophy   = rng.random() < 0.70   # LOF variant
            severe_hypertrigly      = rng.random() < 0.60
            t2d                     = rng.random() < 0.70
            pancreatitis            = rng.random() < 0.10
            pah                     = False
            progeroid               = False
            bone_anomaly            = False
            intellectual_disability = False
            hepatomegaly            = rng.random() < 0.50
            cardiac_conduction      = False
            metreleptin_eligible    = False
            severe_ir               = rng.random() < 0.90
            hypoglycaemia           = rng.random() < 0.30   # GOF subset

        elif gene == "CAV1":
            generalised_lipoatrophy = rng.random() < 0.60   # CGL3 AR vs FPLD7 AD
            mechanical_fat_absent   = rng.random() < 0.30
            partial_lipodystrophy   = rng.random() < 0.40
            severe_hypertrigly      = rng.random() < 0.65
            t2d                     = rng.random() < 0.65
            pancreatitis            = rng.random() < 0.12
            pah                     = rng.random() < 0.35   # PAH overlap DISTINCTIVE
            progeroid               = False
            bone_anomaly            = False
            intellectual_disability = False
            hepatomegaly            = rng.random() < 0.65
            cardiac_conduction      = False
            metreleptin_eligible    = rng.random() < 0.60   # if generalised
            severe_ir               = rng.random() < 0.70
            hypoglycaemia           = False

        else:  # ZMPSTE24
            generalised_lipoatrophy = rng.random() < 0.70
            mechanical_fat_absent   = rng.random() < 0.40
            partial_lipodystrophy   = rng.random() < 0.30
            severe_hypertrigly      = rng.random() < 0.50
            t2d                     = rng.random() < 0.55
            pancreatitis            = rng.random() < 0.08
            pah                     = False
            progeroid               = True   # ALWAYS progeroid in ZMPSTE24
            bone_anomaly            = True   # ALWAYS acroosteolysis/mandibular
            intellectual_disability = False
            hepatomegaly            = rng.random() < 0.45
            cardiac_conduction      = rng.random() < 0.20
            metreleptin_eligible    = rng.random() < 0.50
            severe_ir               = rng.random() < 0.65
            hypoglycaemia           = False

        cohort.append({
            "patient_id":              f"{gene}-{i+1:03d}",
            "age":                     age,
            "sex":                     sex,
            "gene":                    gene,
            "generalised_lipoatrophy": generalised_lipoatrophy,
            "mechanical_fat_absent":   mechanical_fat_absent,
            "partial_lipodystrophy":   partial_lipodystrophy,
            "severe_hypertrigly":      severe_hypertrigly,
            "t2d":                     t2d,
            "pancreatitis":            pancreatitis,
            "pah":                     pah,
            "progeroid":               progeroid,
            "bone_anomaly":            bone_anomaly,
            "intellectual_disability": intellectual_disability,
            "hepatomegaly":            hepatomegaly,
            "cardiac_conduction":      cardiac_conduction,
            "metreleptin_eligible":    metreleptin_eligible,
            "severe_ir":               severe_ir,
            "hypoglycaemia":           hypoglycaemia,
        })
    return cohort


def _generate_cohort(gene_entry: dict) -> list:
    return _make_cohort(gene_entry, gene_entry["seed"])


def overview() -> dict:
    all_cohorts = [_generate_cohort(g) for g in LIPO_GENES]
    all_pts = [p for c in all_cohorts for p in c]
    total = len(all_pts)

    def N(key): return sum(1 for p in all_pts if p[key])

    return {
        "atlas": "Hereditary-Lipodystrophy-Atlas",
        "subtitle": (
            "Complete 8-Gene Congenital & Familial Lipodystrophy Atlas: "
            "BSCL2 (CGL2 — most common CGL, all fat absent) + AGPAT2 (CGL1 — mechanical fat preserved) + "
            "LMNA (FPLD2 Dunnigan — cardiac surveillance mandatory) + PPARG (FPLD3 — TZD target, dominant negative) + "
            "PLIN1 (FPLD4 — TG>20mmol, pancreatitis) + AKT2 (severe IR ± hypoglycaemia GOF) + "
            "CAV1 (CGL3/FPLD7 — PAH overlap) + ZMPSTE24 (MADB — progeroid + acroosteolysis)"
        ),
        "genes": [g["gene"] for g in LIPO_GENES],
        "total_patients": total,
        "seeds": f"{SEED_BASE}–{SEED_BASE + len(LIPO_GENES) - 1}",
        "generalised_lipoatrophy_patients":  N("generalised_lipoatrophy"),
        "mechanical_fat_absent_patients":    N("mechanical_fat_absent"),
        "partial_lipodystrophy_patients":    N("partial_lipodystrophy"),
        "severe_hypertrigly_patients":       N("severe_hypertrigly"),
        "t2d_patients":                      N("t2d"),
        "pancreatitis_patients":             N("pancreatitis"),
        "pah_patients":                      N("pah"),
        "progeroid_patients":                N("progeroid"),
        "bone_anomaly_patients":             N("bone_anomaly"),
        "intellectual_disability_patients":  N("intellectual_disability"),
        "hepatomegaly_patients":             N("hepatomegaly"),
        "cardiac_conduction_patients":       N("cardiac_conduction"),
        "metreleptin_eligible_patients":     N("metreleptin_eligible"),
        "severe_ir_patients":                N("severe_ir"),
        "hypoglycaemia_patients":            N("hypoglycaemia"),
        "gene_patient_counts": {g["gene"]: 40 for g in LIPO_GENES},
        "pathway": (
            "Lipodystrophy — shared final pathway: insufficient functional adipose tissue → "
            "circulating free fatty acid (FFA) excess → ectopic fat deposition (liver, muscle, pancreas) → "
            "insulin resistance → T2D + dyslipidaemia (severe hypertriglyceridaemia). "
            "CGL (generalised): BSCL2 (seipin — ER morphology/lipid droplet biogenesis) + AGPAT2 (phospholipid synthesis) → "
            "absent adipogenesis. FPLD (partial): LMNA (nuclear lamina) + PPARG (adipogenic transcription factor) + "
            "PLIN1 (lipid droplet scaffold) → failed fat maintenance in peripheral depots. "
            "AKT2: insulin signalling node. CAV1: caveolae scaffold (lipid rafts + eNOS). "
            "ZMPSTE24: prelamin A processing → nuclear lamina instability (same final pathway as LMNA)."
        ),
        "key_clinical_insight": (
            "BSCL2: CGL2 — ALL fat absent (including mechanical); intellectual disability 50%; metreleptin FDA-2014. "
            "AGPAT2: CGL1 — mechanical fat PRESERVED; bone cysts on X-ray PATHOGNOMONIC; no intellectual disability. "
            "LMNA: FPLD2 — fat redistribution (NOT absence); CARDIAC SURVEILLANCE MANDATORY; ICD if NSVT+HB+EF<45. "
            "PPARG: FPLD3 — dominant negative; TZD direct target but impaired response; adiponectin very low. "
            "PLIN1: FPLD4 — TG often >20 mmol/L; PANCREATITIS HIGH RISK; subtle visible lipodystrophy. "
            "AKT2: LOF → severe IR + partial lipodystrophy; GOF → neonatal hypoglycaemia + macrosomia. "
            "CAV1: CGL3 (AR) + FPLD7 (AD); PAH overlap — echo MANDATORY; BMPR2 panel if PAH prominent. "
            "ZMPSTE24: MADB — progeroid + mandibular hypoplasia + ACROOSTEOLYSIS pathognomonic; lonafarnib FTI."
        ),
    }


def breakdown() -> dict:
    result = {}
    for gene_entry in LIPO_GENES:
        cohort = _generate_cohort(gene_entry)
        gene = gene_entry["gene"]

        def pct(key):
            return round(100 * sum(1 for p in cohort if p[key]) / len(cohort))

        result[gene] = {
            "gene":               gene,
            "alt_name":           gene_entry["alt_name"],
            "locus":              gene_entry["locus"],
            "protein_size":       gene_entry["protein_size"],
            "inheritance":        gene_entry["inheritance"],
            "n_patients":         len(cohort),
            "generalised_lipoatrophy_pct": pct("generalised_lipoatrophy"),
            "mechanical_fat_absent_pct":   pct("mechanical_fat_absent"),
            "partial_lipodystrophy_pct":   pct("partial_lipodystrophy"),
            "severe_hypertrigly_pct":      pct("severe_hypertrigly"),
            "t2d_pct":                     pct("t2d"),
            "pancreatitis_pct":            pct("pancreatitis"),
            "pah_pct":                     pct("pah"),
            "progeroid_pct":               pct("progeroid"),
            "bone_anomaly_pct":            pct("bone_anomaly"),
            "intellectual_disability_pct": pct("intellectual_disability"),
            "hepatomegaly_pct":            pct("hepatomegaly"),
            "cardiac_conduction_pct":      pct("cardiac_conduction"),
            "metreleptin_eligible_pct":    pct("metreleptin_eligible"),
            "severe_ir_pct":               pct("severe_ir"),
            "hypoglycaemia_pct":           pct("hypoglycaemia"),
            "age_of_onset":    gene_entry["age_of_onset"],
            "key_biomarker":   gene_entry["key_biomarker"],
            "pathognomonic":   gene_entry["pathognomonic"],
            "treatment":       gene_entry["treatment"],
            "critical_flags":  gene_entry["critical_flags"],
            "seed":            gene_entry["seed"],
            "cohort_preview":  cohort[:5],
        }
    return result


def definitions() -> dict:
    return {
        "atlas": "Hereditary-Lipodystrophy-Atlas",
        "pathway": "Adipogenesis / Lipid-Droplet Biology / Nuclear Lamina / Insulin Signalling",
        "shared_mechanism": (
            "Hereditary lipodystrophies share a final common pathway: insufficient functional adipose tissue "
            "→ impaired leptin/adiponectin secretion → circulating FFA excess → ectopic fat deposition "
            "(liver steatosis, skeletal muscle IR, pancreatic beta-cell lipotoxicity) "
            "→ severe insulin resistance → T2D + hypertriglyceridaemia + NASH. "
            "Generalised forms (CGL): absent adipogenesis from birth — most severe metabolic disease; metreleptin effective. "
            "Partial forms (FPLD): inadequate fat maintenance in peripheral depots — metabolic disease variable; "
            "metreleptin less effective (leptin not severely depleted). "
            "Nuclear lamina defects (LMNA, ZMPSTE24): impair adipogenic transcription programme via prelamin A accumulation. "
            "Lipid-droplet defects (PLIN1): unregulated lipolysis → FFA flood. "
            "Nuclear receptor (PPARG): master adipogenic transcription factor — loss prevents adipocyte differentiation."
        ),
        "genes": {
            g["gene"]: {
                "full_name": g["alt_name"],
                "locus": g["locus"],
                "protein_size": g["protein_size"],
                "inheritance": g["inheritance"],
                "critical_flags": g["critical_flags"],
                "pathognomonic": g["pathognomonic"],
                "treatment_summary": g["treatment"],
            }
            for g in LIPO_GENES
        },
        "glossary": {
            "Congenital generalised lipodystrophy (CGL)": "Autosomal recessive complete absence of metabolically active adipose tissue from birth; extreme metabolic disease; metreleptin indicated",
            "Familial partial lipodystrophy (FPLD)": "Autosomal dominant regional fat loss (usually limbs/gluteal) + compensatory fat gain (trunk/neck/face); metabolic disease; metreleptin less effective",
            "BSCL2 (seipin)": "ER-resident protein; controls lipid droplet biogenesis and adipogenesis; CGL type 2 — all fat absent including mechanical depots; most common CGL worldwide",
            "AGPAT2": "Enzyme in glycerophospholipid synthesis (LPA → PA); CGL type 1 — metabolically active fat absent; mechanical fat preserved; bone cysts on X-ray PATHOGNOMONIC",
            "LMNA (lamin A/C)": "Nuclear lamina structural protein; FPLD2 Dunnigan — most common hereditary FPLD; cardiac laminopathy: conduction disease + DCM + SCD risk — ICD mandatory in high-risk",
            "PPARG (PPARγ)": "Master adipogenic nuclear receptor; direct target for TZDs; FPLD3 dominant negative mutations impair PPARγ co-activation; adiponectin markedly low",
            "PLIN1 (perilipin-1)": "Lipid droplet surface scaffold protein; gates ATGL/HSL lipolysis; FPLD4 frameshift → unregulated lipolysis → FFA spillover → severe hypertriglyceridaemia (TG >20 mmol/L)",
            "AKT2": "Central insulin signalling kinase: PI3K→PIP3→PDK1→AKT2→AS160→GLUT4 vesicle translocation; LOF → severe IR + partial lipodystrophy; GOF → constitutive GLUT4 → neonatal hypoglycaemia",
            "CAV1 (caveolin-1)": "Scaffolding protein for caveolae (plasma membrane invaginations); lipid raft organisation, eNOS signalling; CGL3 (AR biallelic) + PAH overlap; FPLD7 (AD heterozygous)",
            "ZMPSTE24 (FACE1)": "Zinc metalloprotease; cleaves 18 C-terminal aa from farnesylated prelamin A → mature lamin A; LOF → prelamin A accumulates → nuclear lamina instability → MADB progeroid lipodystrophy",
            "Prelamin A": "Precursor of lamin A; farnesylated at C-terminal CAAX → ZMPSTE24 cleaves last 18 aa → mature lamin A; accumulation causes progerin-like nuclear toxicity",
            "Metreleptin (Myalept)": "Recombinant methionyl-leptin; FDA approved 2014 for GENERALISED lipodystrophy (CGL1+2); improves TG, HbA1c, liver fat by replacing leptin; NOT effective for FPLD (leptin only mildly low)",
            "Thiazolidinedione (TZD)": "PPARγ ligand; pioglitazone/rosiglitazone; insulin sensitiser; PPARG FPLD3 dominant negative variants impair TZD benefit (receptor dysfunctional); modest effect in LMNA-FPLD2",
            "Caveolae": "Flask-shaped plasma membrane invaginations; require CAV1/CAV3 + cavin proteins; lipid raft microdomains; signalling platforms (eNOS, EGFR, IR); absent in CAV1-CGL3",
            "Seipin (BSCL2)": "ER-resident oligomeric protein; controls lipid droplet size and number; adipogenesis initiation; loss → lipid droplets fail to mature → adipocytes cannot differentiate",
            "Pseudoathleticism": "Appearance of prominent musculature in CGL patients due to absent subcutaneous fat exposing muscle contour; NOT actual athletic capacity; diagnostic clue",
            "Mechanical fat": "Subcutaneous fat in non-energy-storing depots (palms, soles, periorbit, scalp, joints); structural role not metabolic; PRESERVED in CGL1/AGPAT2; ABSENT in CGL2/BSCL2",
            "Acanthosis nigricans": "Dark velvety skin thickening in neck/axillae/groin; marker of severe insulin resistance; prominent in all CGL forms and FPLD",
            "Acroosteolysis": "Resorption of distal phalanges and clavicular ends; PATHOGNOMONIC for ZMPSTE24/MADB on X-ray; also seen in progeria/Werner syndrome",
            "Perilipin-1 (PLIN1)": "Lipid-droplet surface protein coating; recruits HSL (hormone-sensitive lipase) on PKA phosphorylation; normally gates lipolysis to hormonal control; loss → uncontrolled FFA release",
            "ICD indication in LMNA": "ICD implant recommended per ESC 2022 guidelines: LMNA + ≥2 of: NSVT, LBBB/HV>70ms, EF<45%, unexplained syncope — primary prevention SCD in laminopathy",
            "Adiponectin": "Adipokine from white adipocytes; anti-inflammatory + insulin-sensitising; PPARγ drives ADIPOQ expression; severely low in PPARG-FPLD3 and all CGL forms",
            "Lonafarnib (Zokinvy)": "Farnesyltransferase inhibitor; FDA approved 2020 for Hutchinson-Gilford progeria; inhibits farnesylation of prelamin A at CAAX motif; may benefit ZMPSTE24/MADB by reducing farnesyl-prelamin A burden",
            "Fibrates": "PPARα agonists (fenofibrate, bezafibrate, gemfibrozil); reduce VLDL TG synthesis; MANDATORY in lipodystrophy hypertriglyceridaemia; TG target <5 mmol/L to prevent pancreatitis",
            "HOMA-IR": "Homeostatic Model Assessment of Insulin Resistance = fasting insulin × fasting glucose / 22.5; markedly elevated in all CGL forms; >10 indicates severe insulin resistance",
            "Pancreatitis threshold": "TG >10 mmol/L: pancreatitis risk begins; >20 mmol/L: HIGH risk; >50 mmol/L: severe/acute pancreatitis — urgent lipid-lowering (insulin infusion + plasmapheresis if needed)",
        },
        "surveillance_protocols": {
            "BSCL2": "Annual: leptin, TG, HbA1c, insulin, LFTs, fibroscan; TG target <5 mmol/L; metreleptin dose review; ophthalmology (cataract reported); developmental assessment if ID",
            "AGPAT2": "Annual: leptin, TG, HbA1c, LFTs, fibroscan; X-ray if bone symptoms; TG target <5 mmol/L; metreleptin review; reproductive endocrinology",
            "LMNA": "Annual: ECG + 24h Holter + echo; HbA1c; TG; LFTs; CK; assess for ICD indication; genetics re-review if new cardiac symptoms; exercise restriction pending cardiac assessment",
            "PPARG": "Annual: TG, HbA1c, LFTs, adiponectin; fibrate + omega-3 TG management; TZD trial — document response; family metabolic screening (T2D/hypertriglyceridaemia)",
            "PLIN1": "Annual: TG (target <5 mmol/L MANDATORY), HbA1c, LFTs; fibrate dose optimisation; dietary fat review (<15% calories); ER protocol for acute pancreatitis; family TG screening",
            "AKT2": "Annual (LOF): HbA1c, fasting insulin, TG, LFTs; (GOF): glucose monitoring; adjust sirolimus/diazoxide; distinguish GOF vs LOF — management opposite",
            "CAV1": "Annual: echo (PAH screen — TR velocity + RVSP); LFTs; TG; HbA1c; RHC if echo abnormal; metreleptin if generalised + low leptin; PAH MDT involvement",
            "ZMPSTE24": "Annual: HbA1c, TG, X-ray (acroosteolysis progression), echo, dental OPG (mandibular), dermatology; lonafarnib trial consideration; physio for contractures; genetic counselling",
        },
    }


if __name__ == "__main__":
    import json
    ov = overview()
    print(f"Atlas: {ov['atlas']}")
    print(f"Total patients: {ov['total_patients']}")
    print(f"Seeds: {ov['seeds']}")
    print(f"Genes: {', '.join(ov['genes'])}")
    print(f"Generalised lipoatrophy: {ov['generalised_lipoatrophy_patients']}")
    print(f"Metreleptin eligible: {ov['metreleptin_eligible_patients']}")
    print(f"PAH patients: {ov['pah_patients']}")
    print(f"Progeroid patients: {ov['progeroid_patients']}")
    print(f"Pancreatitis patients: {ov['pancreatitis_patients']}")
    print("Breakdown keys:", list(breakdown().keys()))
