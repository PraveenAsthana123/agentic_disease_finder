#!/usr/bin/env python3
"""Hereditary-Coagulation-Disorder-Atlas — Complete 8-Gene Haemophilia, VWD, and Rare Coagulation Factor Deficiency Atlas
F8      (Factor VIII / Haemophilia A; 2351 aa; Xq28; XLR;
         Most common severe X-linked bleeding disorder -- 1:5,000 male births;
         Severe <1% FVIII, Moderate 1-5%, Mild 5-40% FVIII activity;
         Inhibitor development in 25-30% severe HA (major treatment complication);
         Emicizumab (bispecific antibody) approved for severe HA with/without inhibitors;
         seed SEED_BASE+0) .
F9      (Factor IX / Haemophilia B / Christmas disease; 461 aa; Xq27.1; XLR;
         1:30,000 male births; phenotypically identical to HA; APTT prolonged;
         Haemophilia B Leyden variant: severe in childhood, spontaneously improves post-puberty;
         Gene therapies: Hemgenix (etranacogene dezaparvovec) FDA 2022; Beqvez (fidanacogene elaparvovec) FDA 2024;
         seed SEED_BASE+1) .
VWF     (von Willebrand Factor; 2813 aa; 12p13.31; AD/AR;
         Most common inherited bleeding disorder -- 1:100-1:1,000;
         Type 1 (quantitative partial), Type 2 (qualitative -- 2A/2B/2M/2N), Type 3 (severe AR);
         Type 2B GAIN-OF-FUNCTION: DDAVP CONTRAINDICATED (thrombocytopenia crisis);
         Type 2N mimics mild Haemophilia A -- affects males AND females;
         seed SEED_BASE+2) .
F11     (Factor XI / Haemophilia C; 625 aa; 4q35.2; AR;
         Ashkenazi Jewish prevalence 1:450 heterozygotes; bleeding does NOT correlate with FXI level;
         Surgery/trauma bleeds >> spontaneous (contact activation dependent);
         Two founder mutations: E117X (type II) and F283L (type III);
         seed SEED_BASE+3) .
F7      (Factor VII; 444 aa; 13q34; AR;
         Most common rare AR coagulation factor deficiency -- 1:500,000 worldwide;
         Isolated prolonged PT with NORMAL APTT (extrinsic pathway only);
         Treatment: recombinant activated FVII (NovoSeven) or plasma-derived FVII;
         seed SEED_BASE+4) .
F13A1   (Factor XIII A-subunit; 732 aa; 6p24.3; AR;
         Rare (1:2,000,000) but severe: umbilical cord stump bleeding PATHOGNOMONIC;
         Intracranial haemorrhage in 25-30% if untreated -- prophylaxis mandatory;
         PT and APTT BOTH NORMAL -- only clot solubility in 5M urea detects it;
         seed SEED_BASE+5) .
F10     (Factor X / Stuart-Prower factor; 488 aa; 13q34; AR;
         Both PT and APTT prolonged (FX is extrinsic + intrinsic pathway convergence);
         Acquired severe FX deficiency: AL-amyloidosis adsorbs FX -- check amyloid;
         Treatment: prothrombin complex concentrate (PCC) or plasma-derived FX;
         seed SEED_BASE+6) .
LMAN1   (Lectin Mannose-Binding 1; 525 aa; 18q21.3; AR;
         Combined Factor V and VIII Deficiency (F5F8D): ER-Golgi cargo receptor for FV + FVIII;
         MCFD2 also causes F5F8D -- test BOTH LMAN1 and MCFD2;
         FV NOT in FVIII concentrates -- FFP required for FV replacement alongside DDAVP;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 1982-1989)
"""

import random

SEED_BASE = 1982

COAGULATION_GENES = [
    # -- F8 -- Factor VIII / Haemophilia A -------------------------------------------
    {
        "gene": "F8",
        "alt_name": "F8 (Haemophilia A / Factor VIII Deficiency -- Most Common Severe X-linked Bleeding Disorder)",
        "protein": (
            "F8 -- Xq28 XLR -- F8-2351aa -- "
            "Haemophilia-A-Most-Common-Severe-XL-Bleeding-1-in-5000-Males -- "
            "Severe-Less-Than-1pct-FVIII-Moderate-1-5pct-Mild-5-40pct -- "
            "Inhibitor-Development-25-30pct-Severe-HA-Bethesda-Assay-Screen -- "
            "Emicizumab-Bispecific-Antibody-Subcutaneous-Approved-Inhibitor-And-Non-Inhibitor -- "
            "DDAVP-Raises-Endogenous-FVIII-Mild-HA-Test-Response-First"
        ),
        "locus": "Xq28",
        "protein_size": "2351 aa",
        "inheritance": "XLR (X-linked recessive)",
        "age_of_onset": (
            "Severe HA: spontaneous haemarthroses from first ambulation (6-18 months); "
            "Circumcision bleed or traumatic delivery bleed may be first presentation (neonatal); "
            "Moderate HA: post-traumatic and surgical bleeds; some spontaneous joint bleeds; "
            "Mild HA: post-surgical or post-traumatic bleeds only; often not diagnosed until adulthood; "
            "Intracranial haemorrhage: 3-4% of severe HA neonates (major early mortality risk); "
            "Haemophilic arthropathy: recurrent haemarthroses -> synovitis -> cartilage destruction -> end-stage joint disease; "
            "Males ONLY (X-linked recessive); female carriers USUALLY asymptomatic (FVIII 50%); "
            "Obligate carrier females with skewed X-inactivation may be symptomatic (lyonisation) -- check carrier FVIII levels"
        ),
        "key_biomarker": (
            "APTT: prolonged (intrinsic pathway FVIII dependent); "
            "PT: NORMAL (extrinsic pathway -- factor VII/X/V/II/fibrinogen; FVIII not in extrinsic pathway); "
            "Bleeding time (PFA-100): NORMAL (primary haemostasis platelet plug intact); "
            "FVIII activity (one-stage clotting assay): severely reduced; classify severity by level; "
            "VWF:Ag: NORMAL (VWF antigen present; VWF-FVIII binding intact; rules out Type 2N VWD); "
            "Inhibitor screen (Bethesda assay): titre in Bethesda units (BU); >5 BU = high titre; "
            "Molecular: F8 pathogenic variant (hemizygous males); inversion of intron 22 in ~45% severe HA; intron 1 inversion ~5%; "
            "FVIII recovery/half-life study: before surgery; assesses pharmacokinetics for dosing"
        ),
        "pathognomonic": (
            "Male + deep muscle/joint bleeds + prolonged APTT + normal PT + normal bleeding time + low FVIII = Haemophilia A; "
            "Haemarthrosis in a male child with warm swollen joint + prolonged APTT = Haemophilia A until proven otherwise; "
            "DISTINGUISH from Haemophilia B (F9): clinically IDENTICAL; FVIII normal, FIX low in HB; molecular confirms; "
            "DISTINGUISH from VWD Type 2N: FVIII low but VWF also low; VWF:FVIII binding assay differentiates; both sexes; "
            "DISTINGUISH from acquired haemophilia A: elderly or post-partum; high FVIII inhibitor titre; no family history; "
            "Soft tissue/muscle bleeds (iliopsoas haematoma) with normal skin bleeding time = hallmark of coagulation factor deficiency not platelet disorder"
        ),
        "treatment": (
            "Prophylaxis: recombinant FVIII 25-40 IU/kg three times weekly (standard half-life); "
            "Extended half-life rFVIII (rFVIII-Fc, PEGFVIII): twice-weekly or less frequent prophylaxis; "
            "Emicizumab (Hemlibra): bispecific antibody bridging FIXa and FX; subcutaneous weekly/fortnightly/monthly; "
            "EMA/FDA approved 2017-2018 for severe HA with inhibitors; approved 2018 for severe HA without inhibitors; "
            "Inhibitor patients: bypassing agents -- recombinant activated FVII (rFVIIa/NovoSeven) or activated PCC (aPCC/FEIBA); "
            "Emicizumab preferred over bypassing agents for prophylaxis in inhibitor patients (HAVEN trials); "
            "DDAVP (desmopressin): releases VWF + FVIII from endothelial Weibel-Palade bodies; "
            "effective only in mild HA (FVIII >5%); ALWAYS test response before relying on it for surgery; "
            "AVOID: aspirin, NSAIDs, anticoagulants in all HA patients; "
            "AVOID: IM injections (use subcutaneous or IV routes); "
            "AVOID: concurrent emicizumab + aPCC (thrombotic microangiopathy risk -- HAVEN 1 signal; use rFVIIa if breakthrough needed); "
            "Gene therapy: Fitusiran (RNAi antithrombin), Marstacimab (anti-TFPI): non-factor subcutaneous options; "
            "valoctocogene roxaparvovec (Roctavian) EMA/FDA approved 2022/2023 for severe HA adults"
        ),
        "critical_flags": [
            "F8-INHIBITOR-DEVELOPMENT: 25-30% of severe Haemophilia A patients develop inhibitory antibodies to FVIII (Bethesda assay titre >0.6 BU/mL); screen by Bethesda assay before every major surgery and whenever FVIII dose escalation fails to achieve expected rise; inhibitor patients CANNOT be treated with standard FVIII replacement -- it will be neutralised; use bypassing agents (rFVIIa or aPCC) or emicizumab; failure to identify inhibitors before surgery can be fatal",
            "F8-EMICIZUMAB-APCC-THROMBOTIC-RISK: concurrent use of emicizumab prophylaxis with activated prothrombin complex concentrate (aPCC/FEIBA) for breakthrough bleeds carries a risk of thrombotic microangiopathy (TMA) and thromboembolism, as observed in HAVEN 1 trial; if a patient on emicizumab needs bypassing agents, use recombinant FVIIa (NovoSeven) -- NOT aPCC; this combination restriction must be communicated to all treating teams including emergency departments",
            "F8-DDAVP-TEST-RESPONSE-FIRST: DDAVP (desmopressin) raises endogenous FVIII by releasing VWF-FVIII from Weibel-Palade bodies; it is effective ONLY in mild HA (FVIII >5%); always perform a formal DDAVP test dose with pre/post FVIII levels before relying on it for surgical haemostasis; patients who fail to double their FVIII or do not reach haemostatic levels (>50%) after test dose MUST receive FVIII concentrate for any procedure -- do not assume response without testing",
            "F8-INTRON22-INVERSION-PCR: approximately 45% of severe HA (FVIII <1%) is caused by an inversion at intron 22 of the F8 gene; this large structural rearrangement is NOT detected by standard sequencing or exon-focused NGS panels without specific inversion testing (Southern blot or long-range PCR); always test for intron 22 and intron 1 inversions first in severe HA before reporting 'no variant found'; missing this leads to failure of carrier testing and prenatal diagnosis in families",
            "F8-CARRIER-LYONISATION: obligate female carriers of F8 mutations have FVIII levels ranging from 5% to 150% due to random X-inactivation (lyonisation); approximately 10% of carriers have FVIII <40% and are at risk of bleeding (surgical, post-partum haemorrhage, heavy menstrual bleeding); always measure FVIII in all female carriers -- never assume they are asymptomatic without a documented level",
        ],
        "seed": SEED_BASE + 0,
    },
    # -- F9 -- Factor IX / Haemophilia B / Christmas Disease -------------------------
    {
        "gene": "F9",
        "alt_name": "F9 (Haemophilia B / Christmas Disease / Factor IX Deficiency -- First Haemophilia Gene Therapies Licensed)",
        "protein": (
            "F9 -- Xq27.1 XLR -- F9-461aa -- "
            "Haemophilia-B-1-in-30000-Males-Clinically-Identical-Haemophilia-A -- "
            "Haemophilia-B-Leyden-Androgen-Responsive-Promoter-Improves-Puberty -- "
            "Hemgenix-FDA-2022-Etranacogene-Dezaparvovec-AAV5-Gene-Therapy -- "
            "Beqvez-FDA-2024-Fidanacogene-Elaparvovec-SPK-9001-AAV-Spark -- "
            "Inhibitor-Rate-Only-1-3pct-vs-25-30pct-Haemophilia-A"
        ),
        "locus": "Xq27.1",
        "protein_size": "461 aa",
        "inheritance": "XLR (X-linked recessive)",
        "age_of_onset": (
            "Severe HB (<1% FIX): spontaneous joint and muscle bleeds from infancy -- identical to severe HA; "
            "Haemophilia B Leyden: severe FIX deficiency in childhood due to promoter mutation; "
            "spontaneous improvement at puberty as testosterone transactivates the mutant F9 promoter; "
            "FVIII rises to 40-60% after puberty in Leyden -- may no longer require prophylaxis; "
            "Moderate HB (1-5% FIX): post-traumatic bleeds, occasional joint bleeds; "
            "Mild HB (5-40% FIX): post-surgical or traumatic bleeds only; often undiagnosed until adulthood; "
            "Males ONLY (X-linked recessive); female carriers may have FIX 50% -- rarely symptomatic"
        ),
        "key_biomarker": (
            "APTT: prolonged (FIX is in intrinsic pathway); "
            "PT: NORMAL (FIX is intrinsic pathway only); "
            "FIX activity (one-stage clotting assay or chromogenic): severely reduced; "
            "FVIII activity: NORMAL (distinguishes HB from HA); "
            "Inhibitor (Bethesda assay for FIX): rate only 1-3% (much lower than HA); "
            "AAV5 neutralising antibody titre: relevant before gene therapy eligibility (Hemgenix requires titre <678 AAV5 NAb); "
            "molecular: F9 pathogenic variant (hemizygous males); Leyden variants in 5'UTR/promoter region (exon-focused NGS misses them); "
            "FIX antigen level: distinguishes type I (quantity reduced) from type II (dysfunctional FIX) mutations"
        ),
        "pathognomonic": (
            "Male + deep bleeds (joints, muscle) + prolonged APTT + normal PT + normal FVIII + low FIX = Haemophilia B; "
            "Clinically IDENTICAL to Haemophilia A -- FVIII vs FIX assay is the sole biochemical differentiator; "
            "Haemophilia B Leyden: severe HB in boy + spontaneous dramatic improvement at puberty = Leyden phenotype; "
            "confirm by F9 promoter/5'UTR sequencing -- standard exon panels miss promoter mutations; "
            "DISTINGUISH from HA: HA = FVIII low; HB = FIX low; BOTH have prolonged APTT and normal PT; molecular confirms; "
            "DISTINGUISH from acquired FIX deficiency: nephrotic syndrome (FIX lost in urine), warfarin (all vitamin K-dependent factors low), vitamin K deficiency"
        ),
        "treatment": (
            "Prophylaxis: recombinant FIX 40-60 IU/kg twice weekly (standard half-life; t1/2 ~18-24h); "
            "Extended half-life rFIX (rFIX-Fc/Alprolix, rFIX-albumin/Idelvion): twice-weekly or weekly prophylaxis (t1/2 up to 90h); "
            "Gene therapy: etranacogene dezaparvovec (Hemgenix, CSL Behring): AAV5-FIX-Padua variant; FDA approved Nov 2022; EMA approved 2023; "
            "single IV infusion; sustained FIX expression at 40-50% in pivotal trial (HOPE-B); "
            "Beqvez (fidanacogene elaparvovec/SPK-9001, Pfizer/Spark): FDA approved 2024; AAV-Spark100 vector; "
            "Eligibility screening: AAV5 (or relevant serotype) neutralising antibody titre below threshold; liver function normal; no active HCV; "
            "Inhibitor patients (1-3%): recombinant FVIIa (rFVIIa/NovoSeven) preferred; aPCC (FEIBA) second-line; "
            "AVOID: aspirin, NSAIDs in all HB patients; "
            "AVOID: IM injections; "
            "Haemophilia B Leyden: prophylaxis adjusted as FIX level rises post-puberty; may discontinue; "
            "Immunosuppression on gene therapy: corticosteroid course (prednisolone) if ALT rises >2x ULN post-vector (hepatocyte immune response)"
        ),
        "critical_flags": [
            "F9-GENE-THERAPY-NAB-SCREEN: AAV5 pre-existing neutralising antibodies (NAbs) from prior wild-type AAV5 infection block hepatocyte transduction and negate the therapeutic effect of Hemgenix; patients MUST be screened for AAV5 NAbs before gene therapy; patients with titres above the threshold (>678 for Hemgenix) are ineligible for that specific vector; failure to screen and treat a high-NAb patient wastes a single-use, multi-million dollar treatment and achieves no efficacy",
            "F9-LEYDEN-PROMOTER-NOT-DETECTED-BY-EXON-PANELS: Haemophilia B Leyden is caused by mutations in the F9 promoter and 5'UTR region -- exon-based NGS panels DO NOT cover these regions and will report no variant found; when a male patient has Haemophilia B that improves dramatically at puberty, request specific Leyden promoter sequencing or comprehensive F9 gene including 5' regulatory region; missing this diagnosis prevents accurate prognosis and genetic counselling",
            "F9-INHIBITOR-ANAPHYLAXIS-RISK: unlike Haemophilia A inhibitors (which are common, 25-30%), FIX inhibitors in HB are rare (1-3%) but when they occur, they are associated with a much higher rate of anaphylactic reactions to FIX infusion (due to IgE-mediated mechanisms against FIX protein that is absent in severe HB); any patient with severe HB who develops anaphylaxis during FIX infusion must be tested for FIX inhibitor immediately; subsequent FIX infusion without inhibitor recognition carries life-threatening anaphylaxis risk",
            "F9-GENE-THERAPY-LIVER-MONITORING: post-gene-therapy ALT elevation occurs in a subset of patients (20-30%) due to CD8+ T-cell immune response against AAV capsid antigen on transduced hepatocytes; if ALT rises above 2x upper limit of normal, a tapering course of prednisolone (starting 60mg) is required to protect the liver and preserve FIX expression; delayed or omitted corticosteroid treatment leads to loss of FIX production and treatment failure; monitor ALT weekly for at least 52 weeks post-infusion",
            "F9-VITAMIN-K-DEPENDENT-FACTOR: Factor IX is a vitamin K-dependent serine protease; warfarin, vitamin K antagonists, and significant vitamin K deficiency will lower FIX activity in ALL patients, including Haemophilia B carriers; do not interpret low FIX in a patient on warfarin as Haemophilia B without stopping anticoagulation and re-testing; liver disease also reduces FIX (synthesised in hepatocytes); always rule out acquired causes before diagnosing hereditary FIX deficiency",
        ],
        "seed": SEED_BASE + 1,
    },
    # -- VWF -- von Willebrand Factor / VWD ------------------------------------------
    {
        "gene": "VWF",
        "alt_name": "VWF (von Willebrand Disease Types 1/2/3 -- Most Common Inherited Bleeding Disorder)",
        "protein": (
            "VWF -- 12p13.31 AD/AR -- VWF-2813aa -- "
            "Most-Common-Inherited-Bleeding-1-in-100-to-1-in-1000 -- "
            "Type-1-Quantitative-Partial-AD-80pct -- "
            "Type-2B-GOF-DDAVP-ABSOLUTELY-CONTRAINDICATED-Thrombocytopenia-Crisis -- "
            "Type-2N-Low-FVIII-Mimics-Mild-HA-Males-AND-Females-Affected -- "
            "Type-3-AR-Severe-Less-1pct-VWF-Requires-VWF-Concentrate-No-DDAVP"
        ),
        "locus": "12p13.31",
        "protein_size": "2813 aa",
        "inheritance": "AD (Type 1, 2A, 2B, 2M) or AR (Type 3, Type 2N homozygous)",
        "age_of_onset": (
            "Type 1 VWD: mucocutaneous bleeding from childhood (epistaxis, menorrhagia, easy bruising, post-surgical bleeding); "
            "symptoms often mild; frequently undiagnosed for years; menorrhagia in females is a common presenting symptom; "
            "Type 2 VWD: variable by subtype -- may be childhood or adulthood presentation; "
            "Type 2B: often presents with thrombocytopenia on FBC which triggers investigation; "
            "Type 2N: recurrent haemarthroses and muscle bleeds (mimics mild-moderate HA) -- often misdiagnosed as HA; "
            "Type 3 VWD: severe bleeding from infancy; haemarthroses; mucocutaneous + deep tissue bleeds; "
            "Both sexes affected (12p13.31 -- autosomal); females more symptomatic due to menorrhagia; "
            "Heavy menstrual bleeding (HMB): most common presenting complaint in female VWD -- defined as >80 mL/cycle"
        ),
        "key_biomarker": (
            "VWF:Ag (VWF antigen): quantifies total VWF protein; low in Type 1 and 3; may be normal in Type 2; "
            "VWF:RCo (ristocetin cofactor activity) or VWF:GPIb binding: functional platelet-binding activity; "
            "VWF:RCo/VWF:Ag ratio <0.6 = qualitative defect (Type 2A or 2M); "
            "VWF:CB (collagen binding): sensitive for HMW multimer deficiency (Type 2A); "
            "VWF:FVIIIB (FVIII binding assay): severely reduced in Type 2N -- key distinguishing test; "
            "FVIII activity: low in Type 2N and Type 3 (VWF carries and stabilises FVIII); normal in Type 1/2A/2B; "
            "VWF multimer analysis: gel electrophoresis distinguishes subtypes -- large multimers absent in 2A; "
            "Platelet count: thrombocytopenia in Type 2B (VWF binds platelets spontaneously); "
            "RIPA (ristocetin-induced platelet aggregation): enhanced at low ristocetin doses in Type 2B (GOF)"
        ),
        "pathognomonic": (
            "Thrombocytopenia + low VWF + enhanced RIPA at low ristocetin = Type 2B VWD (GOF mutation); "
            "DDAVP is absolutely contraindicated in Type 2B -- releasing stored VWF causes acute platelet aggregation -> thrombocytopenia crisis; "
            "Low FVIII + low VWF:FVIIIB + normal VWF:Ag + both males and females = Type 2N VWD (Normandy); "
            "DISTINGUISH Type 2N from Haemophilia A: Type 2N = both sexes, VWF reduced, FVIII-VWF binding failed; HA = males only, VWF normal, FVIII intrinsically deficient; "
            "DISTINGUISH Type 1 from Type 3: Type 1 = partial reduction (typically 5-50%); Type 3 = <1% VWF; AR; severe bleeds; "
            "VWF:RCo/VWF:Ag <0.6 = qualitative defect; further subtyping by multimer gel and FVIII binding assay"
        ),
        "treatment": (
            "DDAVP (desmopressin, Stimate nasal spray or IV): first-line for Type 1 and most Type 2A; "
            "releases VWF + FVIII from endothelial Weibel-Palade bodies; ALWAYS test response before surgical use; "
            "CONTRAINDICATED in Type 2B (causes thrombocytopenia crisis); AVOID in Type 3 (no VWF stores); "
            "VWF concentrate (Haemate P, Wilate, Vonvendi/rVWF): for Type 3, Type 2B, Type 2N, DDAVP non-responders; "
            "rVWF (Vonvendi): recombinant VWF; licensed 2015 FDA; preferred for Type 3 (no FVIII contamination); "
            "Antifibrinolytics (tranexamic acid, EACA): adjunct for mucocutaneous bleeds; primary treatment for dental procedures; "
            "Hormonal therapy: combined oral contraceptive pill or levonorgestrel IUS (Mirena) for HMB in women; "
            "Iron supplementation: for HMB-related iron deficiency anaemia; "
            "AVOID: aspirin, NSAIDs, anticoagulants in VWD (impair primary haemostasis further); "
            "Pregnancy: VWF rises naturally during pregnancy (especially Type 1); monitor VWF at 28-32 weeks; "
            "Type 3 pregnancy: requires VWF concentrate supplementation around delivery; "
            "Fitusiran (antithrombin RNAi) in clinical trials for VWD as alternative haemostatic approach"
        ),
        "critical_flags": [
            "VWF-TYPE-2B-DDAVP-ABSOLUTELY-CI: Type 2B VWD is a GAIN-OF-FUNCTION mutation causing VWF to bind platelets spontaneously without shear stress; administering DDAVP releases stored high-molecular-weight VWF multimers, which instantly bind platelets, causing acute severe thrombocytopenia (platelet count can fall below 20 x10^9/L) and paradoxical worsening of bleeding; DDAVP is absolutely contraindicated in Type 2B VWD -- this error has been reported as a cause of haemorrhagic complications in surgical settings",
            "VWF-TYPE-2N-MIMICS-HAEMOPHILIA-A: Type 2N (Normandy) VWD causes VWF to fail binding of FVIII; the net effect is rapid FVIII clearance and low plasma FVIII activity, which mimics mild-moderate Haemophilia A exactly; the critical difference is that Type 2N affects BOTH males and females (autosomal) and FVIII correction requires VWF concentrate (not FVIII concentrate alone); female patients 'diagnosed with Haemophilia A' or families with AD inheritance of apparent HA should trigger VWF:FVIII binding assay testing to exclude Type 2N",
            "VWF-DDAVP-RESPONSE-TEST-MANDATORY: DDAVP response varies widely between individuals and VWD subtypes; approximately 20% of Type 1 VWD patients are non-responders; prescribing DDAVP for surgical haemostasis without a formal response test (measure VWF:Ag and VWF:RCo pre-dose and 60 minutes post-dose) risks intraoperative haemorrhage in non-responders who appear to be Type 1 but achieve sub-haemostatic VWF levels; test BEFORE every surgical case, not just at diagnosis",
            "VWF-TYPE-3-NO-DDAVP-STORES: Type 3 VWD is autosomal recessive with <1% VWF; patients have essentially no endothelial VWF stores to release; DDAVP is ineffective and must NOT be used as primary treatment; Type 3 requires VWF concentrate (plasma-derived or recombinant) for all bleeding and surgical episodes; FVIII is also low in Type 3 (FVIII clearance without VWF carrier) and FVIII supplementation may be needed alongside VWF for major surgery",
            "VWF-BLOOD-GROUP-O-LOWER-LEVELS: Blood group O individuals have VWF levels approximately 25% lower than non-O individuals (group O-specific O-glycosylation reduces VWF half-life); a patient with blood group O and VWF in the range 40-50% may carry a VWF diagnostic borderline; blood group must be recorded and accounted for in VWD classification; reclassification of some group O 'low VWF' patients as VWD Type 1 vs physiological low VWF is a recognised diagnostic challenge",
        ],
        "seed": SEED_BASE + 2,
    },
    # -- F11 -- Factor XI / Haemophilia C / Ashkenazi Jewish -------------------------
    {
        "gene": "F11",
        "alt_name": "F11 (Haemophilia C / Factor XI Deficiency -- Ashkenazi Jewish Prevalent Bleeding Disorder)",
        "protein": (
            "F11 -- 4q35.2 AR -- F11-625aa -- "
            "Haemophilia-C-FXI-Deficiency-Ashkenazi-Jewish-1-in-450-Heterozygotes -- "
            "Bleeding-Severity-Does-NOT-Correlate-FXI-Level -- "
            "Surgery-Trauma-Bleeds-More-Than-Spontaneous-Contact-Activation -- "
            "Two-Founder-Mutations-E117X-Type-II-F283L-Type-III-Ashkenazi -- "
            "Tranexamic-Acid-Antifibrinolytic-First-Line-Minor-Bleeds"
        ),
        "locus": "4q35.2",
        "protein_size": "625 aa",
        "inheritance": "AR (autosomal recessive); heterozygotes often have bleeding symptoms",
        "age_of_onset": (
            "Ashkenazi Jewish prevalence: heterozygotes ~1:450 (carrier); homozygotes ~1:100,000; "
            "non-Ashkenazi prevalence: approximately 1:1,000,000 homozygotes; "
            "Bleeding typically provoked: surgery, trauma, dental extraction, childbirth -- NOT spontaneous haemarthroses; "
            "Surgical sites at high-contact-activation risk: urological (prostate, urinary tract), ENT, obstetric -- bleed most severely; "
            "Post-partum haemorrhage: increased risk especially in homozygotes; "
            "Heterozygotes: may have clinically significant bleeding despite FXI levels 40-60% (half normal); "
            "Both sexes equally affected (AR/4q35.2 autosomal); "
            "Often diagnosed incidentally on pre-operative APTT screen"
        ),
        "key_biomarker": (
            "APTT: prolonged (FXI is in intrinsic pathway -- same as FVIII and FIX); "
            "PT: NORMAL (extrinsic pathway intact); "
            "FXI activity (clotting assay): reduced; severity classification -- severe <20 IU/dL, moderate 20-40%; "
            "BLEEDING SEVERITY DOES NOT CORRELATE WITH FXI LEVEL: unique among coagulation factors; "
            "severe FXI deficiency (<20 IU/dL) may have no spontaneous bleeding; mild-moderate FXI may bleed significantly at surgery; "
            "molecular: F11 pathogenic variant; Ashkenazi founder mutations: E117X (c.349G>T) and F283L (c.849C>A); "
            "Combined genotype: E117X/F283L compound heterozygote = Type II/III compound; "
            "Thrombophilia testing: FXI excess associated with thrombosis risk; FXI inhibitor development in therapy (rare)"
        ),
        "pathognomonic": (
            "Ashkenazi Jewish patient + prolonged APTT + normal PT + surgical bleeding without spontaneous bleeds = FXI deficiency until proven; "
            "APTT prolonged with normal FVIII, FIX, and VWF = FXI deficiency (or contact factor deficiency -- FXII, prekallikrein, HMWK); "
            "DISTINGUISH from FXII deficiency: FXII deficiency prolongs APTT but causes NO BLEEDING; FXI deficiency causes bleeding; "
            "DISTINGUISH from HA/HB: HA/HB have spontaneous haemarthroses; FXI deficiency bleeding is surgery/trauma-provoked; sex ratio equal in FXI (AR) vs males only in HA/HB (XLR); "
            "FXI level-bleeding paradox: E117X homozygotes (severe, <1 IU/dL) may have mild or no bleeding; do not rely on FXI level alone to predict bleeding risk"
        ),
        "treatment": (
            "Antifibrinolytics (tranexamic acid): first-line for mucosal and minor bleeds and dental procedures; "
            "prevents fibrinolysis at site of haemostatic plug; especially effective at high-fibrinolytic sites (urological, ENT); "
            "FFP (fresh frozen plasma): replaces FXI; 10-20 mL/kg raises FXI ~20-25%; used pre-operatively; "
            "Plasma-derived FXI concentrate (BPL/Hemoleven): licensed in Europe; concentrated FXI replacement; "
            "AVOID FXI concentrate if history of thrombosis or myocardial infarction (thrombotic risk); "
            "rFVIIa (NovoSeven): bypasses contact pathway; used in high-risk surgery when FXI concentrate contraindicated; "
            "DDAVP: not effective (FXI not released by desmopressin); "
            "Fitusiran (antithrombin RNAi): clinical trials for haemophilias including FXI deficiency; "
            "Pre-operative planning: assess surgical site risk (high-risk = urological, ENT, obstetric); tailor haemostatic approach; "
            "Menorrhagia: tranexamic acid and/or hormonal therapy; FXI replacement peri-delivery; "
            "Israeli/Ashkenazi carrier testing: E117X and F283L targeted genotyping in at-risk families"
        ),
        "critical_flags": [
            "F11-LEVEL-DOES-NOT-PREDICT-BLEEDING: Factor XI deficiency is UNIQUE among coagulation factor deficiencies in that the plasma FXI level does not reliably predict bleeding severity; a patient with severe FXI deficiency (<1 IU/dL) may bleed minimally at surgery while a heterozygote (40-50 IU/dL) may have significant post-operative haemorrhage; the site of surgery (high vs low contact activation) and personal bleeding history are better predictors than the FXI level; never reassure a patient that they are low-risk based on a moderately low FXI level alone",
            "F11-UROLOGICAL-UROKINASE-HIGH-RISK: the urinary tract is rich in urokinase and tissue-type plasminogen activator (tPA), making it a high-fibrinolytic site; FXI is critically important for haemostasis here because local fibrinolysis rapidly dissolves clots; urological surgery (prostatectomy, cystoscopy, TURP) in FXI-deficient patients carries very high bleeding risk; tranexamic acid prophylaxis and FXI replacement are both required; failure to plan haemostatic cover for urological procedures in FXI deficiency has caused life-threatening haematuria",
            "F11-ASHKENAZI-COMPOUND-FOUNDER: the two founder mutations E117X (type II) and F283L (type III) are common in the Ashkenazi Jewish population; compound heterozygotes (E117X/F283L) have severe deficiency; when an Ashkenazi Jewish patient has a prolonged APTT, request targeted genotyping for BOTH E117X and F283L before full sequencing -- this is faster, cheaper, and covers the vast majority of affected individuals; counsel all Ashkenazi Jewish carriers about 25% recurrence risk in offspring",
            "F11-FXI-CONCENTRATE-THROMBOSIS-CI: plasma-derived FXI concentrate (BPL concentrate, Hemoleven) has a documented thrombosis risk -- both arterial thrombosis and venous thromboembolism have been reported post-infusion, particularly in elderly patients with cardiovascular risk factors; FXI concentrate is CONTRAINDICATED in patients with history of thrombosis, MI, or stroke; use rFVIIa (NovoSeven) or antifibrinolytics instead in these patients; never prescribe FXI concentrate without reviewing thrombotic risk history",
            "F11-FXII-DIFFERENTIATION-CRITICAL: Factor XII (Hageman factor) deficiency also causes markedly prolonged APTT but DOES NOT CAUSE BLEEDING -- Hageman himself, who had FXII deficiency, died of pulmonary embolism not haemorrhage; FXII deficiency is not a bleeding disorder and requires NO TREATMENT; FXI deficiency causes bleeding and requires haemostatic management; differentiating the two by factor assay is essential before any haemostatic treatment is initiated or withheld",
        ],
        "seed": SEED_BASE + 3,
    },
    # -- F7 -- Factor VII / Most Common Rare AR Coagulation Deficiency ----------------
    {
        "gene": "F7",
        "alt_name": "F7 (Factor VII Deficiency -- Most Common Rare AR Coagulation Factor Deficiency; Isolated PT Prolongation)",
        "protein": (
            "F7 -- 13q34 AR -- F7-444aa -- "
            "Most-Common-Rare-AR-Coagulation-Deficiency-1-in-500000 -- "
            "Isolated-Prolonged-PT-Normal-APTT-Extrinsic-Pathway-Only -- "
            "ICH-And-Mucous-Membrane-Bleeds-Predominate-Not-Haemarthrosis -- "
            "Treatment-rFVIIa-NovoSeven-Or-Plasma-Derived-FVII-Concentrate -- "
            "F7-F10-Contiguous-13q34-Combined-Deficiency-Some-Patients"
        ),
        "locus": "13q34",
        "protein_size": "444 aa",
        "inheritance": "AR (autosomal recessive)",
        "age_of_onset": (
            "Severe FVII deficiency (<1%): presentation in neonatal period or early infancy; "
            "intracranial haemorrhage (ICH) is the most feared early presentation -- prophylaxis mandatory in severe cases; "
            "Epistaxis, gingival bleeding, menorrhagia are most common symptoms; "
            "Haemarthroses: less common than in HA/HB but can occur in severe FVII deficiency; "
            "Mild-moderate FVII: post-surgical or traumatic bleeds; often incidental APTT screen (normal) + PT prolonged; "
            "Both sexes equally affected (AR); F7 gene at 13q34 -- autosomal; "
            "Geographic clustering: Iran, India -- consanguinity increases prevalence in some populations; "
            "Some patients with FVII levels 15-20% have near-normal haemostasis -- poor correlation of level with bleeding severity"
        ),
        "key_biomarker": (
            "PT: prolonged (FVII initiates extrinsic pathway via TF-FVIIa complex); "
            "APTT: NORMAL (FVII is NOT in intrinsic pathway -- key diagnostic feature); "
            "FVII activity (one-stage or chromogenic): severely reduced; classify: severe <10%, moderate 10-20%, mild >20%; "
            "FVII antigen level: may be normal in type II (dysfunctional) FVII; "
            "PT/APTT discordance (prolonged PT + normal APTT): most specific screen for isolated FVII deficiency or warfarin early-phase; "
            "Rule out early warfarin effect: PT prolonged first (FVII shortest half-life 4-6h among VKDs); APTT normal; check history; "
            "molecular: F7 biallelic pathogenic variants; Arg304Gln (FVII Padua) and Ala294Val common; "
            "Vitamin K-dependent factors: FVII is vitamin K-dependent; exclude vitamin K deficiency and liver disease"
        ),
        "pathognomonic": (
            "Isolated prolonged PT + normal APTT + normal FVIII/FIX/FXI/VWF = FVII deficiency (or early warfarin / vitamin K deficiency); "
            "Mucous membrane and ICH bleeds in a patient with prolonged PT but normal APTT = FVII deficiency until proven; "
            "DISTINGUISH from early warfarin effect: warfarin causes PT prolongation (FVII first, shortest t1/2); stop warfarin, recheck PT; "
            "DISTINGUISH from vitamin K deficiency: all VK-dependent factors low (II, VII, IX, X, protein C, S); APTT also prolonged; "
            "DISTINGUISH from liver disease: multiple factor deficiencies; albumin low; fibrinogen low; context; "
            "DISTINGUISH from lupus anticoagulant: LAC prolongs APTT (not PT) in most cases; mixing study differentiates"
        ),
        "treatment": (
            "Recombinant activated FVII (rFVIIa/NovoSeven): licensed for FVII deficiency and for haemophilia with inhibitors; "
            "dose 15-30 mcg/kg IV; short half-life requires every 4-6h dosing during active bleeding or peri-operatively; "
            "Plasma-derived FVII concentrate: available in some countries; "
            "FFP: provides FVII + all other coagulation factors; volume limitation problematic; "
            "Prothrombin complex concentrate (PCC): 4-factor PCC contains FVII (factors II, VII, IX, X); used in emergencies; "
            "Prophylaxis in severe FVII (<1%): twice-weekly rFVIIa or FVII concentrate; mandatory to prevent ICH; "
            "Monitoring: FVII trough levels and PT; aim FVII >10-15% for prophylaxis; "
            "Menorrhagia: tranexamic acid + hormonal therapy + FVII concentrate peri-procedure; "
            "Pregnancy: FVII rises in normal pregnancy (VK-dependent); may not rise adequately in FVII deficiency; "
            "monitor FVII level in third trimester; delivery management with FVII concentrate or rFVIIa; "
            "AVOID: long-term rFVIIa without monitoring (very short t1/2 mandates frequent administration for prophylaxis)"
        ),
        "critical_flags": [
            "F7-NORMAL-APTT-DIAGNOSTIC-TRAP: Factor VII deficiency produces an ISOLATED prolonged PT with a completely NORMAL APTT; clinicians unfamiliar with this pattern may dismiss a mild PT prolongation (e.g. PT ratio 1.3) as clinically insignificant, especially if APTT is normal; in any patient with unexplained mucocutaneous bleeding or ICH with prolonged PT and normal APTT, measure FVII activity urgently -- do not attribute the isolated PT prolongation to laboratory artefact without investigation",
            "F7-ICH-PROPHYLAXIS-MANDATORY-SEVERE: intracranial haemorrhage occurs in approximately 16% of severe FVII deficiency patients (<1%) if prophylaxis is not given; prophylactic rFVIIa or FVII concentrate (twice-weekly) is MANDATORY in severe FVII deficiency from diagnosis; delayed prophylaxis or treating only on bleeding is unacceptable practice in severe FVII deficiency; ICH carries a 30% mortality in coagulation-deficient neonates",
            "F7-FVII-LEVEL-BLEEDING-DISCORDANCE: like FXI deficiency, FVII level correlates poorly with clinical bleeding severity; FVII Padua (Arg304Gln) is a dysfunctional variant with low one-stage FVII assay activity but relatively preserved haemostasis in vivo because the chromogenic assay gives higher values (assay discordance); always perform both one-stage clotting and chromogenic FVII assays in FVII deficiency to detect type II (dysfunctional) mutations that carry better prognosis than the one-stage result suggests",
            "F7-WARFARIN-FIRST-AFFECTED: among the vitamin K-dependent coagulation factors (II, VII, IX, X, protein C, S), Factor VII has the shortest plasma half-life (4-6 hours); warfarin causes PT prolongation with initially normal APTT because FVII is depleted first; a patient 24-48h into warfarin therapy with prolonged PT and normal APTT may be mistaken for congenital FVII deficiency; always exclude anticoagulant exposure before diagnosing inherited FVII deficiency",
            "F7-F10-COMBINED-CONTIGUOUS-DELETION: F7 and F10 genes are both located at 13q34 and are only 2.8 Mb apart; large chromosomal deletions at 13q34 can remove both genes simultaneously, causing combined FVII and FX deficiency; if a patient has BOTH prolonged PT and APTT (not explained by liver disease or warfarin) and FVII is low, always check FX activity to detect combined F7/F10 contiguous deletion -- treatment and prognosis differ from isolated FVII deficiency",
        ],
        "seed": SEED_BASE + 4,
    },
    # -- F13A1 -- Factor XIII A-subunit / Umbilical Cord Stump Bleeding ---------------
    {
        "gene": "F13A1",
        "alt_name": "F13A1 (Factor XIII A-Subunit Deficiency -- Umbilical Cord Stump Bleeding; PT/APTT Both Normal)",
        "protein": (
            "F13A1 -- 6p24.3 AR -- F13A1-732aa -- "
            "Rare-1-in-2000000-Severe-Umbilical-Cord-Stump-Bleeding-PATHOGNOMONIC -- "
            "PT-And-APTT-BOTH-NORMAL-Not-Detected-Standard-Coagulation-Screen -- "
            "ICH-25-30pct-If-Untreated-Prophylaxis-MANDATORY -- "
            "Clot-Solubility-5M-Urea-Classic-Screening-Test -- "
            "Monthly-FXIII-Concentrate-Fibrogammin-Or-Tretten-rFXIIIA"
        ),
        "locus": "6p24.3",
        "protein_size": "732 aa",
        "inheritance": "AR (autosomal recessive)",
        "age_of_onset": (
            "Umbilical cord stump bleeding: pathognomonic presentation in neonates (delayed separation and persistent bleeding from cord stump); "
            "intracranial haemorrhage: 25-30% of untreated FXIII-deficient patients -- leading cause of morbidity and mortality; "
            "Delayed wound healing: slow scar formation after surgery or trauma; "
            "Recurrent miscarriage: FXIII essential for fibrin cross-linking in placentation; "
            "Post-partum haemorrhage: FXIII deficiency causes delayed haemorrhage hours after delivery (fibrin cross-links dissolve); "
            "Both sexes equally affected (AR; 6p24.3 autosomal); "
            "Joint and soft-tissue bleeds: occur in severe FXIII deficiency; "
            "Diagnosis often delayed because standard coagulation tests (PT, APTT, fibrinogen) are ALL NORMAL in FXIII deficiency"
        ),
        "key_biomarker": (
            "PT: NORMAL; "
            "APTT: NORMAL; "
            "Fibrinogen: NORMAL; "
            "Bleeding time: NORMAL; "
            "STANDARD COAGULATION SCREEN IS ENTIRELY NORMAL IN FXIII DEFICIENCY -- FXIII cross-links AFTER clot forms; "
            "Clot solubility in 5M urea (or 1% monochloroacetic acid): clot dissolves within 2h = FXIII deficiency (positive screen); "
            "FXIII activity (quantitative): severely reduced; FXIII A-subunit antigen; "
            "FXIII B-subunit: B-subunit acts as carrier; B-subunit deficiency (F13B gene) also causes FXIII deficiency; "
            "molecular: F13A1 biallelic pathogenic variants; Val34Leu polymorphism is common variant (affects FXIII activation rate); "
            "FXIII trough levels: monitor on prophylaxis; aim FXIII >3-5% to prevent ICH"
        ),
        "pathognomonic": (
            "Neonatal umbilical cord stump bleeding + normal PT + normal APTT = FXIII deficiency until proven; "
            "Clot dissolves in 5M urea within 2 hours in FXIII-deficient patients (normal clots are insoluble in urea due to cross-links); "
            "Recurrent miscarriage + normal coagulation screen = FXIII deficiency must be excluded; "
            "Delayed post-partum haemorrhage (hours after delivery) in a woman with normal coagulation screen = FXIII deficiency; "
            "DISTINGUISH from fibrinogen deficiency: fibrinogen deficiency also shows clot lysis but fibrinogen IS low; FXIII deficiency = normal fibrinogen; "
            "DISTINGUISH from acquired FXIII deficiency: hepatic failure, DIC, anti-FXIII autoantibodies; context and normal PT/APTT baseline help"
        ),
        "treatment": (
            "FXIII concentrate (Fibrogammin P, plasma-derived): monthly prophylaxis; "
            "once-monthly infusion maintains FXIII above 3-5% trough for 4 weeks (long half-life ~9-12 days); "
            "Recombinant FXIII (Tretten/catridecacog): FDA approved 2013; EMA approved 2012; monthly subcutaneous injection; "
            "FXIII trough target: >3-5% prevents ICH; >10-15% for major surgery; "
            "Prophylaxis is LIFELONG from diagnosis -- do not treat only on demand in severe deficiency (ICH risk too high); "
            "FFP: provides FXIII but requires large volumes; used in resource-limited settings or emergencies; "
            "Cryoprecipitate: higher FXIII concentration than FFP; "
            "Pregnancy: FXIII levels must be maintained throughout pregnancy (risk of miscarriage and post-partum haemorrhage); "
            "monthly concentrate dosing frequency may need to increase in 3rd trimester; target >10% at delivery; "
            "Surgical cover: pre-operative FXIII concentrate; maintain >50% during procedure and peri-operatively; "
            "AVOID: aspirin, NSAIDs; "
            "F13B deficiency (B-subunit): same treatment (FXIII A2B2 tetramer restored by providing A-subunit concentrate)"
        ),
        "critical_flags": [
            "F13A1-NORMAL-COAGULATION-SCREEN-TRAP: Factor XIII is a transglutaminase that cross-links fibrin AFTER the clot has formed; it has no role in the clotting cascade itself; consequently, PT, APTT, fibrinogen, bleeding time, and platelet count are ALL COMPLETELY NORMAL in severe FXIII deficiency; a neonate with umbilical cord stump bleeding, normal coagulation screen, and 'no coagulation disorder found' may still have FXIII deficiency -- always request a clot solubility test or quantitative FXIII assay specifically",
            "F13A1-ICH-PROPHYLAXIS-NON-NEGOTIABLE: intracranial haemorrhage occurs in 25-30% of untreated severe FXIII-deficient patients and carries high mortality and neurological morbidity; once-monthly FXIII concentrate prophylaxis reduces ICH incidence to near-zero; withholding or delaying prophylaxis to treat only on bleeding is contraindicated in severe FXIII deficiency; every patient diagnosed with severe FXIII deficiency must be started on monthly prophylaxis at diagnosis",
            "F13A1-UMBILICAL-CORD-PATHOGNOMONIC: delayed, persistent bleeding from the umbilical cord stump (normally separates at 1-2 weeks) is highly specific for FXIII deficiency; this presentation is more specific for FXIII deficiency than for any other inherited bleeding disorder; every neonate with cord stump bleeding that does not respond to local measures MUST have FXIII activity measured urgently; first ICH can occur in the neonatal period without prophylaxis",
            "F13A1-MISCARRIAGE-PLACENTATION: fibrin cross-linking by FXIII is essential for trophoblast invasion and placental anchoring; women with FXIII deficiency have a very high rate of recurrent first and second trimester miscarriage; the diagnosis must be considered in any woman with recurrent unexplained pregnancy loss who has a normal coagulation screen and normal uterine anatomy; FXIII concentrate supplementation throughout pregnancy has resulted in successful term deliveries",
            "F13A1-5M-UREA-TEST-NOT-QUANTITATIVE: the clot solubility test in 5M urea is a SCREENING test -- a positive result (clot dissolves) indicates FXIII deficiency but does not quantify severity; FXIII activity assay is required to confirm diagnosis and severity; the urea solubility test detects FXIII activity below approximately 1-2% -- it will NOT detect mild or partial FXIII deficiency; quantitative FXIII assay is mandatory for surgical planning and monitoring of trough levels on prophylaxis",
        ],
        "seed": SEED_BASE + 5,
    },
    # -- F10 -- Factor X / Stuart-Prower / PT+APTT Both Prolonged ---------------------
    {
        "gene": "F10",
        "alt_name": "F10 (Factor X Deficiency / Stuart-Prower Factor -- Both PT and APTT Prolonged; Amyloid-Associated Acquired Form)",
        "protein": (
            "F10 -- 13q34 AR -- F10-488aa -- "
            "Both-PT-And-APTT-Prolonged-FX-Convergence-Extrinsic-Plus-Intrinsic-Pathways -- "
            "Severe-Less-1pct-FX-Haemarthrosis-ICH-Rare-1-in-1000000 -- "
            "AL-Amyloidosis-Acquires-Severe-FX-Deficiency-Via-Amyloid-Adsorption -- "
            "PCC-Prothrombin-Complex-Concentrate-Or-Plasma-FX-Concentrate -- "
            "F7-F10-Both-13q34-Contiguous-Deletion-Combined-Deficiency-Some-Cases"
        ),
        "locus": "13q34",
        "protein_size": "488 aa",
        "inheritance": "AR (autosomal recessive)",
        "age_of_onset": (
            "Severe FX deficiency (<1%): neonatal or early infancy presentation; umbilical cord bleeding; ICH; "
            "haemarthroses and muscle haematomas in severe FX deficiency; "
            "Moderate FX (1-10%): post-traumatic bleeds; menorrhagia; haemarthroses less common; "
            "Mild FX (10-40%): post-surgical or incidental; "
            "Acquired FX deficiency (amyloidosis): typically in adults with AL-amyloidosis (plasma cell dyscrasia); "
            "amyloid fibrils adsorb FX, depleting plasma levels; "
            "Both sexes equally affected (AR; 13q34 autosomal); "
            "Geographic: Iran, southern Europe -- consanguinity-related clustering; "
            "Combined FV + FX deficiency: if both PT and APTT prolonged with FX low, also measure FV; "
            "check LMAN1/MCFD2 for combined FV+FVIII first if both FV and FVIII are low"
        ),
        "key_biomarker": (
            "PT: prolonged (FX in extrinsic pathway via TF-VIIa-FXa); "
            "APTT: prolonged (FX in intrinsic pathway prothrombinase complex); "
            "BOTH PT AND APTT PROLONGED = FX, FV, prothrombin (FII), or fibrinogen deficiency -- or liver disease or combined; "
            "FX activity (one-stage or chromogenic): severely reduced; "
            "FV activity: NORMAL in isolated FX deficiency (differentiates from FV+FVIII combined deficiency); "
            "Thrombin time: NORMAL (fibrinogen intact; thrombin cleaves fibrinogen); "
            "Serum protein electrophoresis + free light chains + bone marrow: in acquired FX deficiency to exclude amyloidosis; "
            "Ecarin clotting time: FX in prothrombinase complex; "
            "molecular: F10 biallelic pathogenic variants; FX antigen may be normal in type II (dysfunctional)"
        ),
        "pathognomonic": (
            "Both PT and APTT prolonged + low FX + normal fibrinogen + normal thrombin time = inherited FX deficiency; "
            "Adult with both PT and APTT prolonged + periorbital purpura/carpal tunnel/macroglossia = AL amyloidosis with acquired FX deficiency; "
            "DISTINGUISH from FV deficiency: FV also causes both PT and APTT prolonged; measure FV and FX separately; "
            "DISTINGUISH from liver disease: multiple factor deficiencies; fibrinogen also low; albumin low; context; "
            "DISTINGUISH from warfarin: PT prolonged first; APTT prolonged later with higher doses; FX AND FII AND FIX AND FVII all low; "
            "FX Stuart (Gln358Lys) and FX Friuli (Val298Met): historical named mutations; type II dysfunctional variants"
        ),
        "treatment": (
            "Prothrombin complex concentrate (PCC, 4-factor): contains FII, FVII, FIX, FX; preferred for acute bleeds and surgery; "
            "Plasma-derived FX concentrate (BPL FX concentrate): specifically FX enriched; used for prophylaxis; "
            "FFP: provides FX + all other factors; volume-limited; alternative in resource-limited settings; "
            "Prophylaxis in severe FX (<1%): twice-weekly PCC or FX concentrate; mandatory to prevent ICH and haemarthroses; "
            "FX trough target: >10% for prophylaxis; >30-50% for surgery/major trauma; "
            "Acquired FX deficiency (amyloidosis): treat underlying amyloidosis (bortezomib, melphalan, daratumumab); "
            "high-dose melphalan + autologous stem cell transplant if eligible; FX replacement for acute bleeds; "
            "rFVIIa (NovoSeven): can partially bypass FX requirement by directly activating FX on TF; used in emergencies; "
            "AVOID: aspirin, NSAIDs; "
            "Pregnancy: FX levels must be monitored; FX does not rise substantially in pregnancy; FX concentrate peri-delivery"
        ),
        "critical_flags": [
            "F10-BOTH-PT-APTT-PROLONGED-DIFFERENTIAL: Factor X sits at the convergence of the extrinsic (TF-VIIa) and intrinsic (IXa-VIIIa) pathways into the prothrombinase complex; its deficiency therefore prolongs BOTH the PT and APTT simultaneously; this pattern (prolonged PT + prolonged APTT) has a broad differential including combined factor deficiency, liver disease, warfarin, DIC, and FV or prothrombin deficiency; always measure FX, FV, FII, and fibrinogen specifically before attributing prolonged PT+APTT to liver disease",
            "F10-AMYLOIDOSIS-ACQUIRED-FX: AL-amyloidosis (plasma cell dyscrasia with lambda or kappa light-chain amyloid fibrils) causes severe acquired FX deficiency by a unique mechanism -- amyloid fibrils adsorb FX from plasma, depleting circulating levels; this is NOT factor inhibition (no Bethesda-positive inhibitor); any elderly adult with new-onset both PT and APTT prolongation, periorbital purpura (pathognomonic), macroglossia, carpal tunnel, or cardiac failure must have serum free light chains and a bone marrow biopsy to exclude AL-amyloidosis with FX adsorption",
            "F10-F7-CONTIGUOUS-DELETION-13q34: the F7 and F10 genes are both located at 13q34 approximately 2.8 Mb apart; large chromosomal deletions at 13q34 simultaneously remove both F7 and F10 causing a combined FVII + FX deficiency; combined FVII/FX deficiency presents with both prolonged PT (FVII) and APTT (FX); this is clinically more severe than either alone; always measure FX in any confirmed FVII-deficient patient and vice versa -- chromosomal microarray or MLPA should be performed when both F7 and F10 are low",
            "F10-PCC-4-FACTOR-PREFERRED: 4-factor PCC (Beriplex, Octaplex, Kcentra) contains FX and is the most rapidly available replacement for acute FX-deficient bleeding; 3-factor PCC (Profilnine, Bebulin) has very low or absent FVII and variable FX content -- do not rely on 3-factor PCC for FX replacement; always confirm whether your local PCC is 3-factor or 4-factor when managing acute FX-deficient haemorrhage, as using 3-factor PCC would inadequately replace FX",
            "F10-PROPHYLAXIS-TROUGH-MONITORING: unlike Haemophilia A where FVIII <1% defines severe disease, Factor X deficiency at <1% causes both haemarthroses AND ICH; factor X trough levels must be monitored on prophylaxis to confirm levels remain above 10%; subtherapeutic troughs (FX <3-5%) will result in breakthrough haemarthroses and ICH risk; the half-life of FX is approximately 34-40 hours (longer than FVII at 4-6h), allowing less frequent prophylaxis dosing than FVII replacement",
        ],
        "seed": SEED_BASE + 6,
    },
    # -- LMAN1 -- Combined FV+FVIII Deficiency / F5F8D --------------------------------
    {
        "gene": "LMAN1",
        "alt_name": "LMAN1 (Combined Factor V and VIII Deficiency / F5F8D -- ER-Golgi Cargo Receptor; Both LMAN1 and MCFD2)",
        "protein": (
            "LMAN1 -- 18q21.3 AR -- LMAN1-525aa -- "
            "Combined-FV-Plus-FVIII-Deficiency-F5F8D-LMAN1-Lectin-Mannose-Binding-1 -- "
            "MCFD2-Also-Causes-F5F8D-Test-BOTH-Genes -- "
            "FV-5-30-IU-dL-Plus-FVIII-5-30-IU-dL-Both-PT-APTT-Mildly-Prolonged -- "
            "FFP-For-FV-Plus-DDAVP-For-FVIII-Because-FV-NOT-In-FVIII-Concentrates -- "
            "Middle-Eastern-Mediterranean-Founder-Mutations-LMAN1"
        ),
        "locus": "18q21.3",
        "protein_size": "525 aa",
        "inheritance": "AR (autosomal recessive)",
        "age_of_onset": (
            "Clinical presentation: mild to moderate bleeding; mucocutaneous bleeds (epistaxis, gingival, menorrhagia); "
            "Post-surgical and post-traumatic haemorrhage; haemarthroses are uncommon (FV+FVIII both ~5-30% -- moderate range); "
            "Post-partum haemorrhage; recurrent miscarriage (FV contributes to haemostatic plug); "
            "Both sexes equally affected (AR; 18q21.3 autosomal); "
            "Middle Eastern and Mediterranean populations: higher prevalence due to founder mutations in LMAN1; "
            "Iranian, Iraqi, Italian, Sephardic Jewish populations: reported LMAN1 founder variants; "
            "MCFD2 mutations: same clinical syndrome; smaller protein (16 kDa); MCFD2 and LMAN1 form a receptor complex in ER-Golgi for FV and FVIII cargo transport; "
            "Combined deficiency: both FV and FVIII mildly reduced (rarely <5%); milder than isolated severe haemophilia"
        ),
        "key_biomarker": (
            "PT: mildly prolonged (FV in extrinsic + common pathway via prothrombinase; FX activation step); "
            "APTT: mildly prolonged (FVIII in intrinsic pathway); "
            "FV activity: 5-30 IU/dL (reduced but rarely severely); "
            "FVIII activity: 5-30 IU/dL (reduced but rarely severely); "
            "COMBINATION: both FV AND FVIII simultaneously reduced = diagnostic hallmark of F5F8D; "
            "VWF:Ag: NORMAL (rules out Type 2N VWD as cause of low FVIII); "
            "FIX, FX, FII, fibrinogen: all NORMAL (confirms isolated FV+FVIII dual reduction); "
            "molecular: LMAN1 biallelic pathogenic variants; MCFD2 biallelic variants; sequence BOTH genes; "
            "LMAN1 Trp357Stop and other nonsense/frameshift: common in Middle Eastern families"
        ),
        "pathognomonic": (
            "Both FV AND FVIII simultaneously reduced to 5-30% + mildly prolonged PT + mildly prolonged APTT + normal fibrinogen = F5F8D (LMAN1 or MCFD2); "
            "DISTINGUISH from combined FV+FVIII reduction in liver disease: liver disease also reduces FV+FVIII but reduces ALL liver-synthesised factors (II, VII, IX, X); isolated dual FV+FVIII reduction with other factors normal = F5F8D; "
            "DISTINGUISH from isolated FV deficiency: isolated FV low = prolonged PT+APTT but FVIII is NORMAL; "
            "DISTINGUISH from isolated mild HA (FVIII 5-30%): FV is NORMAL in mild HA; FV reduced in F5F8D; "
            "LMAN1 vs MCFD2: clinically identical; LMAN1 more common; must sequence both genes for complete diagnosis"
        ),
        "treatment": (
            "FFP (fresh frozen plasma): 10-20 mL/kg for FV replacement; "
            "CRITICAL: FV is NOT present in FVIII concentrates, cryoprecipitate, or PCC; "
            "FV is ONLY available in FFP in most countries (no licensed FV concentrate worldwide); "
            "DDAVP (desmopressin): releases stored VWF + FVIII from endothelial Weibel-Palade bodies; "
            "raises FVIII component by 2-3 fold in F5F8D (FV does NOT respond to DDAVP); "
            "Combined FFP + DDAVP: standard approach for procedures; FFP for FV + DDAVP for FVIII; "
            "FVIII concentrate: provides FVIII only -- still need FFP for FV if FV is symptomatic; "
            "Tranexamic acid: antifibrinolytic; adjunct for mucosal bleeds and dental procedures; "
            "Platelet transfusion: platelets contain FV in alpha granules; contribute to FV at wound site; rarely needed; "
            "Prophylaxis: not usually required (levels rarely <5%; spontaneous severe bleeds rare); "
            "Pre-operative planning: FFP to raise FV to >25% + DDAVP for FVIII; monitor both levels; "
            "Pregnancy: both FV and FVIII must be maintained; FFP peri-delivery standard; "
            "Genetic counselling: AR disorder; 25% recurrence; test LMAN1 and MCFD2 in family members"
        ),
        "critical_flags": [
            "LMAN1-FV-NOT-IN-FVIII-CONCENTRATE: the most dangerous error in managing Combined FV+FVIII Deficiency is treating the FVIII component with FVIII concentrate and assuming FV is also covered; Factor V is NOT present in ANY commercial FVIII concentrate, PCC, cryoprecipitate, or rFVIIa; the ONLY source of FV available in most countries is FFP (plasma); treating a F5F8D patient with FVIII concentrate alone for surgery will correct FVIII but leave FV uncorrected, causing continued haemorrhage; always combine FFP (for FV) with DDAVP or FVIII concentrate (for FVIII)",
            "LMAN1-TEST-BOTH-LMAN1-AND-MCFD2: two genes cause F5F8D -- LMAN1 (18q21.3, ERGIC-53 lectin receptor) and MCFD2 (2p21, multiple coagulation factor deficiency protein 2); LMAN1 is more common but MCFD2 accounts for approximately 30% of cases; both proteins form a receptor complex in the ER-Golgi intermediate compartment that traffics FV and FVIII for secretion; sequencing only LMAN1 and reporting negative results in a patient with both FV and FVIII simultaneously reduced will miss MCFD2-positive F5F8D; always request a panel covering both genes",
            "LMAN1-COMBINED-LOW-FV-FVIII-LIVER-DIFFERENTIAL: liver disease reduces Factor V because it is liver-synthesised; liver disease also reduces FVIII because of reduced VWF (FVIII clearance) and impaired hepatic synthesis; in liver disease, HOWEVER, all liver-synthesised factors (FII, FV, FVII, FIX, FX, fibrinogen) are reduced simultaneously; in F5F8D, ONLY FV and FVIII are reduced with all other factors normal; this differential is critical -- treating 'combined FV+FVIII deficiency in liver disease' with FFP is appropriate while treating it as inherited F5F8D prompts genetic testing and family cascade",
            "LMAN1-DDAVP-RAISES-FVIII-NOT-FV: DDAVP (desmopressin) releases VWF and FVIII from endothelial Weibel-Palade bodies via V2 receptor-cAMP pathway; this can raise FVIII 2-3 fold in F5F8D patients from a baseline of 10-20% to potentially haemostatic levels (>30%); however, FV is stored in platelet alpha-granules and is NOT released by DDAVP; DDAVP does NOT raise FV; always also give FFP for FV replacement when using DDAVP in F5F8D -- DDAVP alone is insufficient haemostatic cover for major surgery",
            "LMAN1-MIDDLE-EASTERN-FOUNDER-POPULATIONS: F5F8D caused by LMAN1 mutations has higher prevalence in Middle Eastern and Mediterranean populations (Iran, Iraq, Turkey, Sephardic Jewish communities) due to founder effects; the global prevalence is approximately 1:1,000,000 but local prevalence in consanguineous Middle Eastern communities may be substantially higher; any patient from these backgrounds with mild-moderate bleeding and both PT and APTT mildly prolonged should have FV and FVIII assayed as the first step, before assuming HA or liver disease",
        ],
        "seed": SEED_BASE + 7,
    },
]


def _generate_cohort(gene_entry: dict) -> list:
    rng = random.Random(gene_entry["seed"])
    gene = gene_entry["gene"]
    pts = []
    for i in range(40):
        is_f8 = gene == "F8"
        is_f9 = gene == "F9"
        is_vwf = gene == "VWF"
        is_f11 = gene == "F11"
        is_f7 = gene == "F7"
        is_f13a1 = gene == "F13A1"
        is_f10 = gene == "F10"
        is_lman1 = gene == "LMAN1"

        # Sex assignment -- XLR genes are male-only patients; others equal
        if is_f8 or is_f9:
            sex = "M"
        elif is_vwf or is_f11 or is_f7 or is_f13a1 or is_f10 or is_lman1:
            sex = rng.choice(["M", "F"])
        else:
            sex = rng.choice(["M", "F"])

        # Age at diagnosis (years)
        if is_f8 or is_f9:
            age_dx = rng.randint(0, 3) if rng.random() < 0.6 else rng.randint(4, 30)
        elif is_f13a1:
            age_dx = 0  # neonatal presentation
        elif is_vwf:
            age_dx = rng.randint(5, 50)
        elif is_f7:
            age_dx = rng.randint(0, 40)
        else:
            age_dx = rng.randint(0, 45)

        # Severity classification
        if is_f8 or is_f9:
            sev_roll = rng.random()
            if sev_roll < 0.50:
                severity = "severe"
                severity_pct = round(rng.uniform(0.1, 0.9), 1)
            elif sev_roll < 0.75:
                severity = "moderate"
                severity_pct = round(rng.uniform(1.0, 5.0), 1)
            else:
                severity = "mild"
                severity_pct = round(rng.uniform(5.1, 40.0), 1)
        elif is_vwf:
            sev_roll = rng.random()
            if sev_roll < 0.55:
                severity = "type_1"
                severity_pct = round(rng.uniform(10.0, 50.0), 1)
            elif sev_roll < 0.80:
                severity = "type_2"
                severity_pct = round(rng.uniform(5.0, 50.0), 1)
            else:
                severity = "type_3"
                severity_pct = round(rng.uniform(0.1, 0.9), 1)
        elif is_f11:
            sev_roll = rng.random()
            if sev_roll < 0.35:
                severity = "severe"
                severity_pct = round(rng.uniform(0.1, 20.0), 1)
            elif sev_roll < 0.65:
                severity = "moderate"
                severity_pct = round(rng.uniform(20.1, 40.0), 1)
            else:
                severity = "mild"
                severity_pct = round(rng.uniform(40.1, 70.0), 1)
        elif is_f7:
            sev_roll = rng.random()
            if sev_roll < 0.30:
                severity = "severe"
                severity_pct = round(rng.uniform(0.1, 10.0), 1)
            elif sev_roll < 0.65:
                severity = "moderate"
                severity_pct = round(rng.uniform(10.1, 20.0), 1)
            else:
                severity = "mild"
                severity_pct = round(rng.uniform(20.1, 40.0), 1)
        elif is_f13a1:
            severity = "severe"
            severity_pct = round(rng.uniform(0.1, 2.0), 1)
        elif is_f10:
            sev_roll = rng.random()
            if sev_roll < 0.35:
                severity = "severe"
                severity_pct = round(rng.uniform(0.1, 1.0), 1)
            elif sev_roll < 0.70:
                severity = "moderate"
                severity_pct = round(rng.uniform(1.1, 10.0), 1)
            else:
                severity = "mild"
                severity_pct = round(rng.uniform(10.1, 40.0), 1)
        elif is_lman1:
            severity = "moderate"
            severity_pct = round(rng.uniform(5.0, 30.0), 1)
        else:
            severity = "moderate"
            severity_pct = round(rng.uniform(5.0, 30.0), 1)

        # -- Inhibitor development (F8 and F9) --
        inhibitor_developed = False
        if is_f8 and severity == "severe":
            inhibitor_developed = rng.random() < 0.28
        elif is_f9:
            inhibitor_developed = rng.random() < 0.025

        # -- Prophylaxis --
        on_prophylaxis = False
        if is_f8 or is_f9:
            on_prophylaxis = severity == "severe" and rng.random() < 0.88
        elif is_f13a1:
            on_prophylaxis = rng.random() < 0.92  # mandatory
        elif is_f7 and severity == "severe":
            on_prophylaxis = rng.random() < 0.75
        elif is_f10 and severity == "severe":
            on_prophylaxis = rng.random() < 0.72

        # -- Emicizumab (F8 only) --
        emicizumab = is_f8 and (inhibitor_developed or severity == "severe") and rng.random() < 0.45

        # -- Gene therapy received (F9 primarily; F8 limited) --
        gene_therapy_received = False
        if is_f9 and severity == "severe" and rng.random() < 0.15:
            gene_therapy_received = True
        elif is_f8 and severity == "severe" and rng.random() < 0.06:
            gene_therapy_received = True

        # -- DDAVP responsive --
        ddavp_responsive = False
        if is_f8 and severity == "mild":
            ddavp_responsive = rng.random() < 0.75
        elif is_vwf:
            if severity == "type_1":
                ddavp_responsive = rng.random() < 0.80
            elif severity == "type_2":
                ddavp_responsive = rng.random() < 0.40
            else:  # type_3
                ddavp_responsive = False
        elif is_lman1:
            ddavp_responsive = rng.random() < 0.60  # raises FVIII component only

        # -- Joint disease (haemophilic arthropathy) --
        joint_disease = False
        if is_f8 or is_f9:
            if severity == "severe":
                joint_disease = rng.random() < 0.65
            elif severity == "moderate":
                joint_disease = rng.random() < 0.25

        # -- Intracranial haemorrhage --
        intracranial_hemorrhage = False
        if is_f13a1 and not on_prophylaxis:
            intracranial_hemorrhage = rng.random() < 0.28
        elif is_f7 and severity == "severe":
            intracranial_hemorrhage = rng.random() < 0.18
        elif (is_f8 or is_f9) and severity == "severe":
            intracranial_hemorrhage = rng.random() < 0.04
        elif is_f10 and severity == "severe":
            intracranial_hemorrhage = rng.random() < 0.12

        # -- Umbilical stump bleeding (F13A1 pathognomonic) --
        umbilical_stump_bleeding = is_f13a1 and rng.random() < 0.78

        # -- Mucocutaneous bleeding --
        mucocutaneous_bleeding = False
        if is_vwf:
            mucocutaneous_bleeding = rng.random() < 0.85
        elif is_f11:
            mucocutaneous_bleeding = rng.random() < 0.55
        elif is_f7:
            mucocutaneous_bleeding = rng.random() < 0.70
        elif is_lman1:
            mucocutaneous_bleeding = rng.random() < 0.60
        elif is_f13a1:
            mucocutaneous_bleeding = rng.random() < 0.30
        elif is_f8 or is_f9:
            mucocutaneous_bleeding = rng.random() < 0.20

        # -- Surgery required --
        surgery_required = rng.random() < 0.35

        # -- Ashkenazi founder mutation (F11) --
        ashkenazi_founder = is_f11 and rng.random() < 0.55

        # -- VWD Type 2B (DDAVP contraindicated) --
        type_2b = is_vwf and severity == "type_2" and rng.random() < 0.28

        # -- VWD Type 2N (mimics mild HA; both sexes) --
        type_2n = is_vwf and severity == "type_2" and not type_2b and rng.random() < 0.20

        # -- Combined FV+FVIII (LMAN1) --
        combined_fv_fviii = is_lman1

        # -- Haemophilia B Leyden (F9) --
        leyden_variant = is_f9 and severity == "severe" and rng.random() < 0.06

        # -- Amyloidosis-associated FX (F10 acquired) --
        acquired_amyloid_fx = is_f10 and age_dx > 50 and rng.random() < 0.10

        pts.append({
            "patient_id": f"{gene}-{i+1:03d}",
            "age_at_diagnosis": age_dx,
            "sex": sex,
            "gene": gene,
            "severity": severity,
            "severity_pct": severity_pct,
            # Inhibitor / treatment
            "inhibitor_developed": inhibitor_developed,
            "on_prophylaxis": on_prophylaxis,
            "emicizumab": emicizumab,
            "gene_therapy_received": gene_therapy_received,
            "ddavp_responsive": ddavp_responsive,
            # Bleeding manifestations
            "joint_disease": joint_disease,
            "intracranial_hemorrhage": intracranial_hemorrhage,
            "umbilical_stump_bleeding": umbilical_stump_bleeding,
            "mucocutaneous_bleeding": mucocutaneous_bleeding,
            "surgery_required": surgery_required,
            # Gene-specific flags
            "ashkenazi_founder": ashkenazi_founder,
            "type_2b": type_2b,
            "type_2n": type_2n,
            "combined_fv_fviii": combined_fv_fviii,
            "leyden_variant": leyden_variant,
            "acquired_amyloid_fx": acquired_amyloid_fx,
        })
    return pts


def overview() -> dict:
    all_cohorts = [_generate_cohort(g) for g in COAGULATION_GENES]
    total = sum(len(c) for c in all_cohorts)
    all_pts = [p for c in all_cohorts for p in c]

    inhibitor_n = sum(1 for p in all_pts if p["inhibitor_developed"])
    on_prophylaxis_n = sum(1 for p in all_pts if p["on_prophylaxis"])
    emicizumab_n = sum(1 for p in all_pts if p["emicizumab"])
    gene_therapy_n = sum(1 for p in all_pts if p["gene_therapy_received"])
    joint_disease_n = sum(1 for p in all_pts if p["joint_disease"])
    ich_n = sum(1 for p in all_pts if p["intracranial_hemorrhage"])
    umbilical_n = sum(1 for p in all_pts if p["umbilical_stump_bleeding"])
    mucocutaneous_n = sum(1 for p in all_pts if p["mucocutaneous_bleeding"])
    type_2b_n = sum(1 for p in all_pts if p["type_2b"])
    type_2n_n = sum(1 for p in all_pts if p["type_2n"])
    ashkenazi_n = sum(1 for p in all_pts if p["ashkenazi_founder"])
    combined_fv_fviii_n = sum(1 for p in all_pts if p["combined_fv_fviii"])
    gene_counts = {g["gene"]: len(_generate_cohort(g)) for g in COAGULATION_GENES}

    return {
        "atlas": "Hereditary-Coagulation-Disorder-Atlas",
        "subtitle": (
            "Complete 8-Gene Haemophilia, von Willebrand Disease, and "
            "Rare Coagulation Factor Deficiency Atlas"
        ),
        "genes": [g["gene"] for g in COAGULATION_GENES],
        "total_patients": total,
        "seeds": f"{SEED_BASE}-{SEED_BASE+7}",
        "inhibitor_patients": inhibitor_n,
        "on_prophylaxis_patients": on_prophylaxis_n,
        "emicizumab_patients": emicizumab_n,
        "gene_therapy_patients": gene_therapy_n,
        "joint_disease_patients": joint_disease_n,
        "intracranial_hemorrhage_patients": ich_n,
        "umbilical_stump_bleeding_patients": umbilical_n,
        "mucocutaneous_bleeding_patients": mucocutaneous_n,
        "type_2b_vwd_patients": type_2b_n,
        "type_2n_vwd_patients": type_2n_n,
        "ashkenazi_founder_mutation_patients": ashkenazi_n,
        "combined_fv_fviii_patients": combined_fv_fviii_n,
        "gene_patient_counts": gene_counts,
        "pathway": (
            "F8 (Haemophilia A): FVIII is cofactor in intrinsic tenase complex (IXa-VIIIa-FX) -> prolongs APTT; PT normal. "
            "F9 (Haemophilia B): FIX serine protease in intrinsic pathway (IXa within tenase) -> prolongs APTT; PT normal. "
            "VWF (VWD): VWF multimers mediate platelet adhesion (GPIb) and carry FVIII; loss causes mucocutaneous bleeding + low FVIII in severe forms. "
            "F11 (Haemophilia C): FXI amplifies thrombin via intrinsic pathway -> APTT prolonged; surgery/fibrinolysis-dependent sites bleed most. "
            "F7 (FVII deficiency): FVII initiates extrinsic pathway (TF-FVIIa complex) -> isolated PT prolongation with normal APTT. "
            "F13A1 (FXIII deficiency): FXIII cross-links fibrin (Gln-Lys isopeptide bonds) AFTER clot formation -> PT/APTT both normal; clot dissolves in urea. "
            "F10 (FX deficiency): FX is convergence point of both pathways into prothrombinase (Xa-Va) -> both PT and APTT prolonged. "
            "LMAN1 (F5F8D): LMAN1-MCFD2 receptor complex traffics FV and FVIII from ER to Golgi for secretion -> both FV and FVIII mildly reduced simultaneously."
        ),
        "key_clinical_insight": (
            "F8 (HA): emicizumab (Hemlibra) approved 2017-2018 for inhibitor and non-inhibitor severe HA; do NOT combine with aPCC (TMA risk). "
            "F9 (HB): Hemgenix gene therapy FDA approved Nov 2022 -- first licensed haemophilia gene therapy; screen AAV5 NAbs before use. "
            "VWF (VWD): DDAVP ABSOLUTELY CONTRAINDICATED in Type 2B -- releases GOF VWF causing acute thrombocytopenia crisis. "
            "F11 (Haemophilia C): bleeding severity does NOT correlate with FXI level -- surgical site risk matters more than the FXI value. "
            "F7 (FVII deficiency): isolated prolonged PT with normal APTT; rFVIIa (NovoSeven) is primary treatment; ICH prophylaxis mandatory in severe deficiency. "
            "F13A1 (FXIII deficiency): PT and APTT BOTH NORMAL -- standard coagulation screen will miss it entirely; cord stump bleed = FXIII until proven. "
            "F10 (FX deficiency): both PT and APTT prolonged; check for AL amyloidosis in adults with new-onset FX deficiency. "
            "LMAN1 (F5F8D): FV is NOT in FVIII concentrates -- FFP required for FV + DDAVP/FVIII concentrate for FVIII; test both LMAN1 and MCFD2."
        ),
    }


def breakdown() -> dict:
    result = {}
    for gene_entry in COAGULATION_GENES:
        cohort = _generate_cohort(gene_entry)
        gene = gene_entry["gene"]

        inhibitor_pct = round(100 * sum(1 for p in cohort if p["inhibitor_developed"]) / len(cohort))
        prophylaxis_pct = round(100 * sum(1 for p in cohort if p["on_prophylaxis"]) / len(cohort))
        emicizumab_pct = round(100 * sum(1 for p in cohort if p["emicizumab"]) / len(cohort))
        gene_therapy_pct = round(100 * sum(1 for p in cohort if p["gene_therapy_received"]) / len(cohort))
        ddavp_pct = round(100 * sum(1 for p in cohort if p["ddavp_responsive"]) / len(cohort))
        joint_disease_pct = round(100 * sum(1 for p in cohort if p["joint_disease"]) / len(cohort))
        ich_pct = round(100 * sum(1 for p in cohort if p["intracranial_hemorrhage"]) / len(cohort))
        umbilical_pct = round(100 * sum(1 for p in cohort if p["umbilical_stump_bleeding"]) / len(cohort))
        mucocutaneous_pct = round(100 * sum(1 for p in cohort if p["mucocutaneous_bleeding"]) / len(cohort))
        ashkenazi_pct = round(100 * sum(1 for p in cohort if p["ashkenazi_founder"]) / len(cohort))
        type_2b_pct = round(100 * sum(1 for p in cohort if p["type_2b"]) / len(cohort))
        type_2n_pct = round(100 * sum(1 for p in cohort if p["type_2n"]) / len(cohort))
        surgery_pct = round(100 * sum(1 for p in cohort if p["surgery_required"]) / len(cohort))

        result[gene] = {
            "gene": gene,
            "alt_name": gene_entry["alt_name"],
            "locus": gene_entry["locus"],
            "protein_size": gene_entry["protein_size"],
            "inheritance": gene_entry["inheritance"],
            "n_patients": len(cohort),
            "inhibitor_pct": inhibitor_pct,
            "on_prophylaxis_pct": prophylaxis_pct,
            "emicizumab_pct": emicizumab_pct,
            "gene_therapy_pct": gene_therapy_pct,
            "ddavp_responsive_pct": ddavp_pct,
            "joint_disease_pct": joint_disease_pct,
            "intracranial_hemorrhage_pct": ich_pct,
            "umbilical_stump_bleeding_pct": umbilical_pct,
            "mucocutaneous_bleeding_pct": mucocutaneous_pct,
            "ashkenazi_founder_pct": ashkenazi_pct,
            "type_2b_pct": type_2b_pct,
            "type_2n_pct": type_2n_pct,
            "surgery_required_pct": surgery_pct,
            "age_of_onset": gene_entry["age_of_onset"],
            "key_biomarker": gene_entry["key_biomarker"],
            "pathognomonic": gene_entry["pathognomonic"],
            "treatment": gene_entry["treatment"],
            "critical_flags": gene_entry["critical_flags"],
            "seed": gene_entry["seed"],
            "cohort_preview": cohort[:5],
        }
    return result


def definitions() -> dict:
    return {
        "atlas": "Hereditary-Coagulation-Disorder-Atlas",
        "gene_definitions": {
            g["gene"]: {
                "protein": g["protein"],
                "locus": g["locus"],
                "protein_size": g["protein_size"],
                "inheritance": g["inheritance"],
            }
            for g in COAGULATION_GENES
        },
        "glossary": {
            "Haemophilia A (F8)": (
                "FVIII deficiency; XLR; 1:5,000 males; most common severe inherited bleeding disorder; "
                "severe <1%, moderate 1-5%, mild 5-40%; prolonged APTT, normal PT; "
                "inhibitors in 25-30% severe -- bypassing agents or emicizumab; "
                "emicizumab (Hemlibra) bispecific antibody EMA/FDA approved 2017-2018; "
                "gene therapy valoctocogene roxaparvovec (Roctavian) EMA/FDA approved 2022-2023"
            ),
            "Haemophilia B (F9 / Christmas Disease)": (
                "FIX deficiency; XLR; 1:30,000 males; clinically identical to HA; "
                "prolonged APTT, normal PT; inhibitor rate only 1-3%; "
                "Haemophilia B Leyden: promoter variant, severe in childhood, improves post-puberty (testosterone); "
                "gene therapies: Hemgenix (etranacogene dezaparvovec) FDA Nov 2022, EMA 2023; "
                "Beqvez (fidanacogene elaparvovec) FDA 2024"
            ),
            "von Willebrand Disease (VWF)": (
                "Most common inherited bleeding disorder (1:100-1:1,000); "
                "Type 1 (AD, partial quantitative deficiency 80%); Type 2 qualitative subtypes (2A, 2B, 2M, 2N); "
                "Type 3 (AR, severe <1% VWF); VWF:RCo/VWF:Ag <0.6 = qualitative defect; "
                "Type 2B GOF: DDAVP ABSOLUTELY CONTRAINDICATED (thrombocytopenia crisis); "
                "Type 2N: VWF fails FVIII binding -> low FVIII mimics mild HA; both sexes affected"
            ),
            "Haemophilia C (F11 / FXI Deficiency)": (
                "FXI deficiency; AR; Ashkenazi Jewish prevalence 1:450 heterozygotes; "
                "bleeding severity does NOT correlate with FXI level -- site-specific (urology, ENT, obstetric); "
                "APTT prolonged, PT normal; antifibrinolytics (tranexamic acid) first-line; "
                "two Ashkenazi founder mutations: E117X and F283L; avoid FXI concentrate if thrombosis history"
            ),
            "FVII Deficiency (F7)": (
                "Most common rare AR coagulation factor deficiency (1:500,000); "
                "ISOLATED prolonged PT with NORMAL APTT -- only factor in extrinsic pathway; "
                "ICH and mucosal bleeds predominate; bleeding severity correlates poorly with FVII level; "
                "treatment: rFVIIa (NovoSeven) or plasma-derived FVII concentrate; "
                "ICH prophylaxis mandatory in severe FVII deficiency (<1%)"
            ),
            "FXIII Deficiency (F13A1)": (
                "Factor XIII A-subunit deficiency; AR; rare (1:2,000,000); "
                "PT and APTT BOTH COMPLETELY NORMAL -- standard coagulation screen misses it entirely; "
                "umbilical cord stump bleeding = pathognomonic; ICH in 25-30% untreated; "
                "monthly FXIII concentrate prophylaxis mandatory (Fibrogammin P or Tretten/rFXIIIA); "
                "clot solubility in 5M urea = screening test"
            ),
            "FX Deficiency (F10 / Stuart-Prower Factor)": (
                "Factor X deficiency; AR; rare (~1:1,000,000); "
                "both PT and APTT prolonged (FX at pathway convergence); "
                "acquired severe FX deficiency: AL-amyloidosis adsorbs FX from plasma; "
                "treatment: 4-factor PCC or plasma-derived FX concentrate; "
                "F7 and F10 both at 13q34 -- combined deficiency with contiguous deletion"
            ),
            "Combined FV+FVIII Deficiency (LMAN1 / F5F8D)": (
                "LMAN1 (or MCFD2) mutation; AR; ER-Golgi cargo trafficking defect for FV and FVIII; "
                "both FV and FVIII mildly reduced (5-30%); both PT and APTT mildly prolonged; "
                "FV is NOT in FVIII concentrates -- FFP required for FV; DDAVP raises FVIII only; "
                "Middle Eastern and Mediterranean founder mutations in LMAN1; test both LMAN1 and MCFD2"
            ),
            "Emicizumab (Hemlibra)": (
                "Bispecific antibody bridging FIXa and FX; mimics cofactor function of FVIII; "
                "subcutaneous weekly, fortnightly, or monthly dosing; "
                "FDA approved Oct 2017 (HA+inhibitors), Aug 2018 (severe HA without inhibitors); "
                "EMA approved 2018 (with and without inhibitors); "
                "AVOID concurrent aPCC (FEIBA) -- TMA and thromboembolism risk; use rFVIIa for breakthrough bleeds"
            ),
            "Bethesda Assay": (
                "Quantitative test for FVIII or FIX inhibitory antibodies in haemophilia; "
                "expressed in Bethesda Units (BU/mL); >5 BU = high titre; 0.6-5 BU = low titre; "
                "Nijmegen modification improves specificity by buffering residual FVIII; "
                "mandatory before any major surgery in HA or HB; if FVIII replacement fails to raise, check inhibitor urgently"
            ),
            "DDAVP (Desmopressin)": (
                "Synthetic V2 receptor agonist releasing stored VWF + FVIII from endothelial Weibel-Palade bodies; "
                "effective in mild HA (FVIII >5%), VWD Type 1, most Type 2A; "
                "CONTRAINDICATED in VWD Type 2B (thrombocytopenia crisis); ineffective in Type 3 VWD (no stores); "
                "ALWAYS perform test dose with pre/post FVIII levels before surgical reliance; response wanes with repeat doses"
            ),
            "Clot Solubility Test (5M Urea)": (
                "Screening test for FXIII deficiency; FXIII cross-links fibrin making clot insoluble; "
                "FXIII-deficient clots dissolve in 5M urea within 2 hours (positive = deficiency); "
                "NOT quantitative -- positive screen requires quantitative FXIII activity assay to confirm and grade severity; "
                "detects FXIII levels below approximately 1-2%; does not detect partial FXIII deficiency"
            ),
            "Bypassing Agents": (
                "rFVIIa (NovoSeven): recombinant activated FVII; directly activates FX via TF and on platelet surface; "
                "aPCC (FEIBA, activated prothrombin complex concentrate): contains activated FII, FVII, FIX, FX; "
                "both bypass FVIII and FIX requirements; used in inhibitor haemophilia A and B; "
                "AVOID aPCC concurrent with emicizumab (TMA risk); use rFVIIa if breakthrough bleed on emicizumab"
            ),
            "Haemophilic Arthropathy": (
                "End-stage joint disease from recurrent haemarthroses (joint bleeds) in severe HA and HB; "
                "synovitis -> cartilage destruction -> subchondral bone erosion -> ankylosis; "
                "target joints (bleed >3 times in 6 months in same joint); "
                "prevention: primary prophylaxis from age 1-2 years before joint damage begins; "
                "emicizumab dramatically reduces haemarthroses as once-subcutaneous dosing improves adherence"
            ),
            "Haemophilia B Leyden": (
                "F9 promoter or 5'UTR pathogenic variant causing severe Haemophilia B in childhood; "
                "spontaneous dramatic improvement post-puberty due to testosterone transactivation of mutant F9 promoter; "
                "FIX rises to 40-60% at puberty -- patient may no longer require prophylaxis; "
                "NOT detected by standard exon-based NGS panels (promoter region not covered); "
                "request targeted F9 5'UTR/promoter sequencing in any male with post-pubertal HA improvement"
            ),
            "AL Amyloidosis with FX Deficiency": (
                "Plasma cell dyscrasia producing light-chain amyloid fibrils that adsorb Factor X from plasma; "
                "severe acquired FX deficiency; NOT factor inhibition (Bethesda negative); "
                "periorbital purpura, macroglossia, carpal tunnel, cardiac failure with FX deficiency = amyloid until proven; "
                "treatment: reduce amyloid load (bortezomib, melphalan, daratumumab, ASCT); FX replacement for acute bleeds"
            ),
            "MCFD2 (Multiple Coagulation Factor Deficiency Protein 2)": (
                "Second causative gene for Combined FV+FVIII Deficiency (F5F8D); located at 2p21; 16 kDa protein; "
                "MCFD2 and LMAN1 form a complex in the ERGIC (ER-Golgi intermediate compartment) for FV and FVIII cargo transport; "
                "accounts for approximately 30% of F5F8D; clinically identical to LMAN1-F5F8D; "
                "always test BOTH LMAN1 and MCFD2 when both FV and FVIII are simultaneously reduced"
            ),
            "VWF:FVIII Binding Assay": (
                "Functional test for VWF Type 2N diagnosis; measures VWF capacity to bind FVIII; "
                "severely reduced in Type 2N (Normandy) VWD; normal in Haemophilia A; "
                "essential test when a patient has low FVIII but both sexes affected or VWF also appears reduced; "
                "distinguishes Type 2N (AR or AD) from Haemophilia A (XLR) -- treatment differs (VWF concentrate vs FVIII concentrate)"
            ),
        },
        "clinical_pearls": [
            "F8 (Haemophilia A): emicizumab (Hemlibra) is the preferred prophylaxis in 2024 for severe HA (with or without inhibitors); do NOT combine emicizumab with aPCC for breakthrough bleeds -- use rFVIIa (NovoSeven) instead to avoid TMA and thromboembolism",
            "F9 (Haemophilia B): Hemgenix (etranacogene dezaparvovec) was the first haemophilia gene therapy licensed (FDA Nov 2022, EMA 2023); screen AAV5 neutralising antibodies before eligibility assessment; Haemophilia B Leyden (promoter mutations) is missed by standard exon-based NGS -- always test F9 promoter in any HB patient showing improvement at puberty",
            "VWF (VWD Type 2B): DDAVP is absolutely contraindicated in Type 2B VWD -- it releases stored high-molecular-weight VWF that instantly aggregates platelets causing acute thrombocytopenia and paradoxical haemorrhage; always document VWD subtype before prescribing DDAVP",
            "VWF (VWD Type 2N): low FVIII + normal VWF:Ag + both sexes affected = Type 2N VWD (not Haemophilia A); treat with VWF concentrate not FVIII concentrate alone; VWF:FVIII binding assay confirms diagnosis",
            "F11 (FXI deficiency): bleeding does NOT correlate with FXI level -- the surgical site (urology, ENT, obstetric) is a better predictor of bleeding risk; avoid FXI concentrate if there is any thrombosis or cardiovascular history (thrombosis risk); tranexamic acid first-line for mucosal sites",
            "F7 (FVII deficiency): the ONLY coagulation factor deficiency with isolated prolonged PT and completely normal APTT; ICH prophylaxis with twice-weekly rFVIIa is mandatory in severe FVII deficiency (<1%); FV-FX combined deficiency at 13q34 must be excluded",
            "F13A1 (FXIII deficiency): the standard coagulation screen (PT, APTT, fibrinogen) is ENTIRELY NORMAL in FXIII deficiency -- a 'normal coagulation screen' does NOT exclude FXIII deficiency; request clot solubility (5M urea) or FXIII activity in any neonate with cord stump bleeding or ICH with normal standard tests",
            "F10 (FX deficiency): both PT and APTT are prolonged; in an adult without family history, new-onset FX deficiency with periorbital purpura or macroglossia must prompt urgent serum free light chains and bone marrow biopsy to exclude AL amyloidosis adsorbing FX; 4-factor PCC is the preferred rapid replacement",
            "LMAN1 (F5F8D): Factor V is NOT present in any FVIII concentrate, cryoprecipitate, or PCC -- always combine FFP (for FV) with DDAVP or FVIII concentrate (for FVIII) in F5F8D; failure to replace FV will cause ongoing haemorrhage despite apparently adequate FVIII replacement; test both LMAN1 and MCFD2",
            "All haemophilias: DDAVP test response MUST be performed before any surgical reliance -- approximately 20% of mild HA patients and a significant proportion of Type 1 VWD patients are non-responders; assuming response without testing risks intraoperative haemorrhage in non-responders",
        ],
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(overview(), indent=2, default=str)[:2000])
    print("\n=== BREAKDOWN (first gene) ===")
    bd = breakdown()
    first_gene = list(bd.keys())[0]
    print(json.dumps(bd[first_gene], indent=2, default=str)[:1500])
    print("\n=== DEFINITIONS (glossary sample) ===")
    df = definitions()
    print(json.dumps(list(df["glossary"].items())[:4], indent=2))
