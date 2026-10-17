#!/usr/bin/env python3
"""Hereditary-Hemophilia-Rare-Coagulation-Factor-Deficiency-Atlas -- Complete 8-Gene Reference
Inherited bleeding disorders: hemophilia A/B, von Willebrand disease, rare factor deficiencies.

F8       (Factor-VIII; 2351aa; Xq28; XLR LOF;
          Hemophilia-A most-common-severe-hereditary-bleeding-disorder 1:5000-males;
          Intron-22-inversion 40-50% severe-HA-PATHOGNOMONIC;
          FVIII-inhibitor-development 25-30% severe;
          Emicizumab-Hemlibra bispecific-Ab subcutaneous-prophylaxis-REGARDLESS-inhibitor;
          seed SEED_BASE+0).
F9       (Factor-IX; 461aa; Xq27.1; XLR LOF;
          Hemophilia-B Christmas-disease 1:25000-males;
          F9-Leyden-variant severity-IMPROVES-post-puberty androgen-response-element PATHOGNOMONIC;
          F9-Padua-R338L extreme-GOF-8x-activity paradoxical-VTE;
          Hemgenix-etranacogene-dezaparvovec FDA2022 gene-therapy;
          seed SEED_BASE+1).
VWF      (von-Willebrand-Factor; 2813aa; 12p13.31; AD AR LOF GOF;
          Most-common-inherited-bleeding-disorder 1:100-1:1000;
          Type-2B-DDAVP-CONTRAINDICATED-ABSOLUTELY GOF-GPIb-platelet-activation-thrombocytopenia;
          Type-2N-mimics-mild-Hemophilia-A FVIII-binding-defect;
          Type-3-AR-complete-deficiency;
          seed SEED_BASE+2).
F11      (Factor-XI; 625aa; 4q35.2; AR LOF;
          FXI-deficiency Hemophilia-C;
          Ashkenazi-Jewish 1/190-carrier 1/450-homozygous HIGHEST-prevalence;
          Bleeding-NOT-proportional-to-FXI-level KEY-RULE-surgical>spontaneous;
          seed SEED_BASE+3).
F7       (Factor-VII; 444aa; 13q34; AR LOF;
          FVII-deficiency most-common-rare-coagulation-factor-deficiency worldwide 1:500k;
          PT-elevated-APTT-NORMAL-PATHOGNOMONIC isolated-PT-prolongation;
          Shortest-half-life-4-6h of-all-coag-factors;
          rFVIIa-NovoSeven specific-licensed-treatment;
          seed SEED_BASE+4).
F13A1    (Factor-XIII-A-subunit; 732aa; 6p24.3; AR LOF;
          FXIII-A-deficiency;
          Umbilical-stump-bleeding-at-birth PATHOGNOMONIC 60-70%;
          Intracranial-bleeding-HIGHEST-of-all-factor-deficiencies 25-30%-lifetime;
          PT-APTT-TT-ALL-NORMAL-PATHOGNOMONIC FXIII-acts-after-fibrin-polymerization;
          Urea-solubility-test-PATHOGNOMONIC diagnostic;
          seed SEED_BASE+5).
SERPINE1 (PAI-1; 402aa; 7q22.1; AR LOF;
          PAI-1-deficiency plasminogen-activator-inhibitor;
          Delayed-wound-bleeding-hours-post-trauma PATHOGNOMONIC;
          Post-partum-hemorrhage-PATHOGNOMONIC;
          Antifibrinolytics-tranexamic-acid-FIRST-LINE no-coagulation-factor-works;
          seed SEED_BASE+6).
ADAMTS13 (ADAMTS13; 1427aa; 9q34.2; AR LOF;
          Congenital-TTP Upshaw-Schulman-syndrome;
          MAHA+thrombocytopenia HALLMARK;
          Neonatal-jaundice+thrombocytopenia-newborn PATHOGNOMONIC-congenital-TTP;
          Plasma-infusion-FFP-CURATIVE replaces-enzyme NO-immunosuppression-needed;
          seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 3686-3693)
"""
import random

SEED_BASE = 3686

ATLAS_GENES = [
    {
        "gene": "F8",
        "protein": (
            "F8 -- Xq28 XLR LOF -- 2351aa -- "
            "Hemophilia-A-most-common-severe-inherited-bleeding-disorder-"
            "1in5000-males-"
            "Intron22-inversion-40-50pct-severe-HA-"
            "Emicizumab-bispecific-Ab-prophylaxis-REGARDLESS-inhibitor-"
            "F8-OMIM-306700"
        ),
        "locus": "Xq28",
        "protein_size": (
            "2351 aa / ~280 kDa / Xq28 F8 Factor VIII molecular context: "
            "STRUCTURE: "
            "  2351 aa / ~280 kDa; coagulation Factor VIII; procofactor for intrinsic tenase complex; "
            "  A1-A2-B-A3-C1-C2 domain architecture (B domain is heavily glycosylated, not required for activity); "
            "  Activated by thrombin at Arg372 (A1/A2) and Arg1689 (A2/B); factor Xa also activates; "
            "  FVIIIa forms tenase complex with FIXa, Ca2+, phospholipid → activates FX (intrinsic pathway); "
            "  APC + Protein S: inactivate FVIIIa by cleavage at Arg336 and Arg562; "
            "  vWF carrier: VWF protects FVIII from premature proteolysis; VWF deficiency → secondary FVIII fall (Type 2N/3); "
            "SEVERITY CLASSIFICATION: "
            "  Severe HA: FVIII activity <1 IU/dL (<1%); spontaneous hemarthroses; life-threatening; "
            "  Moderate HA: FVIII 1-5 IU/dL; hemarthroses with minor trauma; "
            "  Mild HA: FVIII 5-40 IU/dL; significant trauma/surgery only; late presentation common; "
            "GENOTYPE-PHENOTYPE: "
            "  Intron 22 inversion (inv22): 40-50% of severe HA; PATHOGNOMONIC large-scale rearrangement; "
            "  Intron 1 inversion (inv1): 2-5% of severe HA; "
            "  Large deletions: 5-10% severe HA; inversely proportional inhibitor risk (large del → high inhibitor risk); "
            "  Missense (C-domain particularly): mild HA; Arg2150His: mild HA in Hispanic populations; "
            "INHIBITORS (highest clinical impact): "
            "  FVIII inhibitor development: 25-30% of severe HA; 3-13% of non-severe HA; "
            "  Inhibitor titer: low (<5 BU/mL) vs high (>5 BU/mL Bethesda units); "
            "  Low-responder: respond to high-dose FVIII; High-responder: bypass agents required; "
            "  Bypass agents: rFVIIa (NovoSeven; 90 mcg/kg q2-3h) OR activated PCC (FEIBA; 50-100 U/kg q8-12h); "
            "EMICIZUMAB (HEMLIBRA) - GAME-CHANGER: "
            "  Bispecific antibody: simultaneously binds FIXa and FX; mimics FVIIIa cofactor function; "
            "  Subcutaneous once-weekly/biweekly/monthly prophylaxis; effective with OR without inhibitors; "
            "  Indicated: severe HA with inhibitors (FDA 2017) → ALL severe HA (FDA 2018); "
            "  Critical drug interaction: FEIBA + emicizumab → thrombotic microangiopathy (TMA) risk; "
            "  Use rFVIIa (NOT FEIBA) for breakthrough bleeds on emicizumab; "
            "FITUSIRAN: "
            "  Antithrombin siRNA (subcutaneous monthly); reduces AT → increases thrombin generation; "
            "  Works in both HA and HB, with or without inhibitors; phase III approved; "
            "GENE THERAPY: "
            "  Roctavian (valoctocogene roxaparvovec): AAV5-F8; FDA 2023; sustained FVIII expression; "
            "  Targeting: liver hepatocyte FVIII expression; 3-5 year durability current data; "
            "  Exclude: anti-AAV5 pre-existing Abs, liver disease, inhibitor history; "
            "TESTING: "
            "  APTT prolonged (intrinsic pathway); PT NORMAL; "
            "  One-stage FVIII activity (APTT-based): standard; "
            "  Chromogenic FVIII activity: required for non-severe HA (one-stage may overestimate); "
            "  Bethesda assay: inhibitor quantification; Nijmegen modification: improved specificity; "
            "  F8 gene sequencing: confirmatory; carrier testing females; inv22/inv1 specific PCR"
        ),
        "syndrome": "Hemophilia-A / FVIII-deficiency / XLR / Most-common-severe-inherited-bleeding-disorder",
        "inheritance": "XLR LOF (males affected; females carriers with variable FVIII levels 30-200%)",
        "bleeding_risk": "Severe <1% FVIII: spontaneous hemarthroses; Moderate 1-5%: trauma; Mild 5-40%: surgery only",
        "pathognomonic": "Intron-22-inversion in 40-50% severe HA; APTT prolonged + PT NORMAL; FEIBA+emicizumab → TMA RISK",
        "key_rule": "Use rFVIIa (NOT FEIBA) for breakthrough bleeds on emicizumab — FEIBA+emicizumab causes thrombotic microangiopathy",
        "key_avoid": "FEIBA with emicizumab (TMA); IM injections; aspirin/NSAIDs; insufficient prophylaxis in severe HA",
        "surveillance": "Annual FVIII level + inhibitor screen; joint imaging (hemophilia arthropathy); hepatology if prior plasma exposure",
        "targeted_rx": "Emicizumab SC prophylaxis (ALL severe HA ± inhibitors); gene therapy (Roctavian, selected); FVIII concentrate (recombinant preferred); rFVIIa/FEIBA bypass for inhibitors",
    },
    {
        "gene": "F9",
        "protein": (
            "F9 -- Xq27.1 XLR LOF -- 461aa -- "
            "Hemophilia-B-Christmas-disease-"
            "1in25000-males-"
            "F9-Leyden-puberty-improvement-PATHOGNOMONIC-"
            "Hemgenix-FDA2022-gene-therapy-"
            "F9-OMIM-306900"
        ),
        "locus": "Xq27.1",
        "protein_size": (
            "461 aa / ~57 kDa / Xq27.1 F9 Factor IX molecular context: "
            "STRUCTURE: "
            "  461 aa / ~57 kDa; coagulation Factor IX; serine protease zymogen; "
            "  Gla domain (aa 1-46): 12 gamma-carboxylated Glu residues; Ca2+/membrane binding; "
            "  EGF-like domains 1 and 2 (aa 47-133): protein-protein interactions; "
            "  Activation peptide (aa 134-145): removed upon activation; "
            "  Serine protease domain (aa 146-461): His221-Asp269-Ser365 catalytic triad; "
            "  Activated by FXIa (intrinsic) or TF-FVIIa complex (extrinsic bypass); "
            "  FIXa + FVIIIa + Ca2+ + phospholipid → intrinsic tenase complex → activates FX; "
            "F9 LEYDEN VARIANT (unique clinical pearl): "
            "  Severe/moderate HB in childhood; spontaneous IMPROVEMENT to mild/normal AFTER puberty; "
            "  Mechanism: androgen-responsive element (ARE) in F9 promoter; testosterone upregulates F9 expression; "
            "  Leyden variants: mutations in ARE (positions -5 to -20, -26, -29, -33, -40 relative to ATG); "
            "  NOT seen in Hemophilia A (no androgen-responsive element in F8 promoter); "
            "  Clinical implication: PATHOGNOMONIC — any male with HB improving post-puberty = Leyden; "
            "  Management: prophylaxis in childhood; may taper after puberty if FIX levels adequate; "
            "F9 PADUA VARIANT (R338L): "
            "  Gain-of-function: FIX activity 8-fold elevated (800%); paradoxical VTE risk; "
            "  NOT HB — elevated FIX activity → deep vein thrombosis without thrombophilia workup = missed; "
            "  Basis for gene therapy optimization (Padua transgene): higher FIX expression per vector copy; "
            "INHIBITORS IN HB (critical danger): "
            "  FIX inhibitor development: 1-3% of HB (much lower than HA); "
            "  Risk: severe HB with large deletions or null mutations; "
            "  Critical distinction: anaphylaxis risk with FIX concentrate IN PRESENCE of inhibitor; "
            "  Management: rFVIIa (bypass); emicizumab NOT indicated for HB (binds FIXa but FIX mutant may not be present); "
            "  Fitusiran: effective in HB with/without inhibitors; "
            "GENE THERAPY (HB-specific): "
            "  Hemgenix (etranacogene dezaparvovec): AAV5-F9-Padua; FDA Nov 2022 (first gene therapy for HB); "
            "  Single IV infusion; sustained FIX levels (mean 41% at 18 months); 94% achieved normal/near-normal; "
            "  Exclude: anti-AAV5 Abs, liver disease; "
            "  Alhemo (fitusiran): licensed for HB with/without inhibitors; "
            "TESTING: "
            "  APTT prolonged; PT NORMAL; "
            "  One-stage FIX activity (APTT-based): standard; "
            "  F9 gene sequencing: definitive; distinguish null mutations (deletion, nonsense) from missense; "
            "  Leyden: specific promoter sequencing (ARE region)"
        ),
        "syndrome": "Hemophilia-B / Christmas-disease / FIX-deficiency / XLR / F9-Leyden-puberty-spontaneous-improvement",
        "inheritance": "XLR LOF (males affected; female carriers variable FIX 30-150%)",
        "bleeding_risk": "Severe <1% FIX: spontaneous bleeds; Moderate 1-5%: trauma bleeds; Mild 5-40%: surgical bleeding",
        "pathognomonic": "F9-Leyden: HB severity IMPROVES spontaneously post-puberty (androgen-responsive promoter) PATHOGNOMONIC; anaphylaxis with FIX concentrate + HB inhibitor",
        "key_rule": "Anaphylaxis risk when giving FIX concentrate to HB patient WITH inhibitor — use rFVIIa bypass, NOT FIX concentrate",
        "key_avoid": "FIX concentrate in HB with inhibitors (anaphylaxis/nephrotic syndrome risk); underdiagnosis of Leyden variant (improving HB in puberty)",
        "surveillance": "Annual FIX level + inhibitor screen; liver function (HCV exposure pre-1985 plasma); joint surveillance",
        "targeted_rx": "Hemgenix gene therapy (curative intent); FIX concentrate (recombinant EHL: Alprolix, Idelvion); rFVIIa/fitusiran for inhibitors",
    },
    {
        "gene": "VWF",
        "protein": (
            "VWF -- 12p13.31 AD AR LOF GOF -- 2813aa -- "
            "von-Willebrand-disease-most-common-inherited-bleeding-1in100-1in1000-"
            "Type-2B-DDAVP-CONTRAINDICATED-ABSOLUTELY-"
            "Type-2N-mimics-Hemophilia-A-"
            "VWF-OMIM-193400"
        ),
        "locus": "12p13.31",
        "protein_size": (
            "2813 aa / ~360 kDa monomer (multimers up to 20,000 kDa) / 12p13.31 VWF von Willebrand Factor molecular context: "
            "STRUCTURE: "
            "  2813 aa / ~360 kDa monomer; stored in Weibel-Palade bodies (endothelium) and alpha-granules (platelets); "
            "  Multimeric structure: disulfide-linked dimers → polymers; ultra-large VWF (UL-VWF) most hemostatically active; "
            "  ADAMTS13 cleaves UL-VWF multimers at Tyr842-Met843 in A2 domain; "
            "  D1-D2: signal propeptide; D'-D3: FVIII binding (N-terminal); A1: GPIb/GPIbalpha binding; "
            "  A2: ADAMTS13 cleavage site; A3: collagen binding; C1-C4: RGD integrin binding; CK: dimerization; "
            "VWD TYPE CLASSIFICATION (CRITICAL): "
            "  Type 1 (70-80%): Partial quantitative deficiency VWF; AD; DDAVP responsive in most; mild-moderate bleeding; "
            "  Type 2A: Loss of large multimers; A1/A2 domain mutations; increased proteolysis or defective secretion; "
            "  Type 2B (GOF): Enhanced GPIbalpha affinity → spontaneous platelet binding → clearance UL-VWF + thrombocytopenia; "
            "    DDAVP ABSOLUTELY CONTRAINDICATED in 2B (releases more abnormal VWF → worsens thrombocytopenia); "
            "  Type 2M: Qualitative; normal multimer pattern; impaired GPIb binding (A1 mutations); "
            "  Type 2N (NORMANDY): FVIII binding domain (D'-D3) mutations; "
            "    Clinical: APTT prolonged + low FVIII + normal VWF multimers → MISDIAGNOSED as mild Hemophilia A; "
            "    Sex-linked feature absent (autosomal); female patients with apparent mild HA = test VWF FVIII-binding assay; "
            "    DDAVP may work transiently but VWF-FVIII complex unstable; use pdVWF/FVIII concentrate; "
            "  Type 3 (AR biallelic): Complete VWF deficiency; severe bleeding including hemarthroses; secondary FVIII low; "
            "DDAVP (DESMOPRESSIN) DECISION TREE: "
            "  Works: Type 1 (most), some Type 2A, some Type 2M; releases stored VWF from Weibel-Palade bodies; "
            "  CONTRAINDICATED: Type 2B (worsens thrombocytopenia); Type 3 (no stored VWF); cardiovascular disease; "
            "  Tachyphylaxis: exhausts stores after 2-3 doses in 48h; "
            "  Test dose: always perform DDAVP challenge first to confirm response before clinical use; "
            "LABORATORY PITFALL: "
            "  ABO blood group effect: Type O → VWF 25-35% lower than Type A/B → ABO-corrected reference ranges mandatory; "
            "  Thyroid disease: hypothyroidism elevates VWF; hyperthyroidism reduces; "
            "  Estrogen/pregnancy: VWF rises in pregnancy → Type 1 may normalize → bleed POST-delivery (not during); "
            "TREATMENT: "
            "  DDAVP: Type 1 and selected Type 2 (NOT 2B/3); intranasal or IV; "
            "  VWF/FVIII concentrate (Humate-P, Wilate): Type 2B, 3, DDAVP-unresponsive; "
            "  Antifibrinolytics (tranexamic acid): ALL VWD types for mucosal bleeding; "
            "  Vonvendi (recombinant VWF): type 3 and DDAVP-unresponsive; "
            "  Fitusiran: NOT specifically indicated for VWD"
        ),
        "syndrome": "von-Willebrand-disease-Type-1-2A-2B-2M-2N-3 / Most-common-inherited-bleeding-disorder / DDAVP-2B-CONTRAINDICATED-ABSOLUTELY",
        "inheritance": "Type 1/2: AD; Type 2N: AR or compound het; Type 3: AR biallelic",
        "bleeding_risk": "Type 1: mucocutaneous mild; Type 2A/2M/2N: moderate; Type 2B: moderate + thrombocytopenia; Type 3: severe including hemarthroses",
        "pathognomonic": "Type-2B: DDAVP worsens thrombocytopenia PATHOGNOMONIC; Type-2N: females with apparent mild Hemophilia A = test VWF FVIII-binding assay; ABO-O VWF 25-35% lower",
        "key_rule": "ALWAYS perform DDAVP challenge before clinical use to confirm response; NEVER use DDAVP in Type 2B (thrombocytopenia worsens); ABO-correct VWF reference ranges",
        "key_avoid": "DDAVP in Type 2B VWD (thrombocytopenia); diagnosing Type 2N as Hemophilia A (different treatment); missing post-partum hemorrhage in Type 1 (VWF falls post-delivery)",
        "surveillance": "Annual; pregnancy VWF levels at 36 weeks; post-partum plan; family cascade; dental plan (high mucosal bleeding risk)",
        "targeted_rx": "DDAVP (Type 1, selected 2); VWF/FVIII concentrate (Humate-P, Wilate, Vonvendi); tranexamic acid ALL types; pdVWF for Type 2N (FVIII binding unstable with DDAVP)",
    },
    {
        "gene": "F11",
        "protein": (
            "F11 -- 4q35.2 AR LOF -- 625aa -- "
            "FXI-deficiency-Hemophilia-C-"
            "Ashkenazi-Jewish-1in190-carrier-1in450-homozygous-HIGHEST-"
            "Bleeding-NOT-proportional-to-FXI-level-UNIQUE-"
            "F11-OMIM-264900"
        ),
        "locus": "4q35.2",
        "protein_size": (
            "625 aa / ~80 kDa / 4q35.2 F11 Factor XI molecular context: "
            "STRUCTURE: "
            "  625 aa / ~80 kDa; FXI circulates as a homodimer (~160 kDa); "
            "  Apple domain structure: 4 apple domains (A1-A4) + serine protease domain; "
            "  A4 domain: FIX binding (apple 4 critical for intrinsic activation); "
            "  A2 domain: HMWK (high-molecular-weight kininogen) binding; "
            "  Serine protease domain: His413-Asp464-Ser557 catalytic triad; "
            "  Activation: FXIIa (contact pathway) or thrombin (feedback activation) at Arg369-Ile370; "
            "  FXIa: activates FIX (intrinsic pathway); "
            "ASHKENAZI JEWISH PREVALENCE (HIGHEST OF ANY ETHNIC GROUP): "
            "  Ashkenazi Jewish carrier frequency: ~1/8 (carrier) = 11% carriers; "
            "  Homozygous/compound heterozygous: ~1/190 Ashkenazi Jewish (severe FXI deficiency); "
            "  Non-Ashkenazi: 1/1,000,000 rare; "
            "  Two founder mutations: Type II (Glu117stop) and Type III (Phe283Leu) in exons 5 and 9; "
            "UNIQUE CLINICAL FEATURE (FXI ≠ FVIII/FIX): "
            "  FXI level does NOT predict bleeding risk (unique among coagulation factor deficiencies); "
            "  Severe FXI deficiency (<20%) = variable bleeding; some patients never bleed spontaneously; "
            "  Bleeding provoked by HIGH fibrinolytic activity tissue (tonsil, oral cavity, genitourinary, uterus); "
            "  Spontaneous hemarthroses: RARE (unlike hemophilia A/B); "
            "  Surgical/procedural bleeding: high risk in fibrinolytic-rich sites; low risk in fibrinolytic-poor sites; "
            "  Dental extraction, tonsillectomy, urological surgery: HIGH risk without treatment; "
            "  Joint surgery, abdominal: LOW-MODERATE risk; "
            "  Key clinical rule: use antifibrinolytics (tranexamic acid) pre-procedure regardless of FXI level; "
            "VTE PARADOX: "
            "  FXI deficiency does NOT reliably protect against VTE; "
            "  Some FXI-deficient patients develop DVT/PE despite FXI <20%; "
            "  FXI as thrombosis target (abelacimab, osocimab phase 3): AF stroke prevention without bleeding; "
            "TESTING: "
            "  APTT prolonged; PT NORMAL; "
            "  FXI activity assay: defines severity; "
            "  F11 sequencing: required; founder mutation panel for Ashkenazi Jewish population; "
            "TREATMENT: "
            "  Antifibrinolytics (tranexamic acid / epsilon-aminocaproic acid): FIRST-LINE for mucosal/fibrinolytic-rich sites; "
            "  FXI concentrate (Hemoleven, European): 1000 U/vial; "
            "  Fresh frozen plasma (FFP): 15-20 mL/kg; VTE risk with large volumes; "
            "  rFVIIa: off-label option for inhibitor patients; "
            "  DDAVP: NOT effective for FXI deficiency (different mechanism)"
        ),
        "syndrome": "FXI-deficiency / Hemophilia-C / AR-LOF / Ashkenazi-Jewish-1-in-190 / Bleeding-NOT-proportional-to-FXI-level",
        "inheritance": "AR LOF (homozygous/compound het severe; heterozygous partial deficiency usually mild)",
        "bleeding_risk": "UNPREDICTABLE: severe FXI <20% = variable; high risk at fibrinolytic-rich sites (tonsil, GU, uterus); rare spontaneous hemarthroses",
        "pathognomonic": "FXI level does NOT predict bleeding risk (unique among all coagulation factor deficiencies); Ashkenazi Jewish founder mutations Type-II (Glu117stop) Type-III (Phe283Leu)",
        "key_rule": "Tranexamic acid FIRST-LINE pre-procedure regardless of FXI level — bleeding at fibrinolytic-rich sites (tonsil, GU) requires antifibrinolytic prophylaxis",
        "key_avoid": "Assuming FXI level predicts bleeding (it does not); large FFP volumes (VTE risk); DDAVP (ineffective for FXI deficiency)",
        "surveillance": "Procedure-specific bleeding risk assessment; family cascade in Ashkenazi Jewish families; GU procedure planning",
        "targeted_rx": "Tranexamic acid pre-/post-procedure (FIRST-LINE all sites); FXI concentrate (Hemoleven); FFP 15-20 mL/kg; rFVIIa off-label inhibitors",
    },
    {
        "gene": "F7",
        "protein": (
            "F7 -- 13q34 AR LOF -- 444aa -- "
            "FVII-deficiency-most-common-rare-factor-deficiency-1in500k-"
            "PT-elevated-APTT-NORMAL-PATHOGNOMONIC-isolated-PT-"
            "Shortest-half-life-4-6h-all-coag-factors-"
            "rFVIIa-NovoSeven-licensed-"
            "F7-OMIM-227500"
        ),
        "locus": "13q34",
        "protein_size": (
            "444 aa / ~50 kDa / 13q34 F7 Factor VII molecular context: "
            "STRUCTURE: "
            "  444 aa / ~50 kDa; coagulation Factor VII; serine protease zymogen; "
            "  Gla domain (aa 1-38): 10 gamma-carboxylated Glu; membrane/TF binding; "
            "  EGF-like domains 1 and 2 (aa 45-135): structural; Ca2+-binding in EGF1; "
            "  Serine protease domain (aa 136-444): His193-Asp242-Ser339 catalytic triad; "
            "  Activation: autocatalytic or by FXa, FIXa, FXIIa, or thrombin at Arg152-Ile153; "
            "  TF-FVIIa complex: triggers extrinsic pathway → activates FX and FIX; "
            "  TF-FVIIa + FXa: primary hemostatic trigger at injury site; "
            "HALF-LIFE: "
            "  FVII half-life: 4-6 hours → SHORTEST of all coagulation factors; "
            "  Clinically important: FVII falls first in warfarin initiation (PT prolongs before other factors); "
            "  FVII falls first in liver disease (early indicator of hepatic synthetic failure); "
            "  PT-INR primary reflect FVII and the extrinsic pathway; "
            "ISOLATED PT PROLONGATION PATTERN (PATHOGNOMONIC): "
            "  FVII deficiency: PT prolonged; APTT NORMAL; TT NORMAL; "
            "  Differential diagnosis of isolated PT prolongation: FVII def > Vitamin K deficiency (early) > Liver disease (early); "
            "  Combined PT+APTT prolongation: deficiency of common pathway (FX, FV, FII, fibrinogen) or multiple factors; "
            "CLINICAL SPECTRUM: "
            "  Severe: FVII <10%; CNS bleeding (intracranial hemorrhage) most life-threatening; "
            "  Moderate 10-20%: mucocutaneous and post-surgical bleeding; "
            "  Mild >20%: variable, often asymptomatic; "
            "  Paradox: FVII deficiency level does NOT always correlate with bleeding severity; "
            "  Thrombosis: some FVII missense variants can be prothrombotic (FVII levels elevated); "
            "TREATMENT: "
            "  rFVIIa (NovoSeven): 15-30 mcg/kg q4-6h; specific licensed treatment for FVII deficiency; "
            "  Plasma-derived FVII concentrate (BeneFIX-like, limited supply); "
            "  FFP: 15-20 mL/kg; large volumes often needed; "
            "  Prothrombin Complex Concentrate (PCC): contains FVII (4-factor PCC preferred); "
            "  Antifibrinolytics: adjunct for mucosal bleeding; "
            "  Vitamin K: rule out Vit K deficiency BEFORE diagnosing congenital F7 deficiency; "
            "TESTING: "
            "  PT prolonged (INR elevated); APTT NORMAL; PATHOGNOMONIC combination; "
            "  FVII activity assay: one-stage clotting or chromogenic; "
            "  F7 gene sequencing: confirms inherited vs acquired (liver, Vit K deficiency); "
            "  Family studies: parents testing (AR carriers often 50% FVII)"
        ),
        "syndrome": "FVII-deficiency / Most-common-rare-coagulation-factor-deficiency / Isolated-PT-prolongation-APTT-NORMAL-PATHOGNOMONIC",
        "inheritance": "AR LOF (homozygous or compound heterozygous; carriers 50% FVII usually asymptomatic)",
        "bleeding_risk": "Severe <10%: intracranial hemorrhage highest risk; Moderate 10-20%: post-surgical; Mild >20%: variable",
        "pathognomonic": "PT prolonged + APTT NORMAL (isolated PT prolongation) PATHOGNOMONIC for FVII deficiency; FVII shortest half-life 4-6h falls first in warfarin/liver disease",
        "key_rule": "ALWAYS exclude Vitamin K deficiency before diagnosing congenital FVII deficiency — Vit K deficiency causes same isolated PT prolongation; treat with rFVIIa (15-30 mcg/kg) specific indication",
        "key_avoid": "Missing intracranial hemorrhage risk in severe FVII (<10%) — prophylaxis mandatory; diagnosing FVII deficiency without excluding Vit K deficiency",
        "surveillance": "Annual FVII level; intracranial hemorrhage imaging if severe; surgical pre-planning; newborn screen if parents known carriers",
        "targeted_rx": "rFVIIa (NovoSeven) 15-30 mcg/kg (licensed for FVII def); 4-factor PCC; FFP 15-20 mL/kg; tranexamic acid adjunct mucosal",
    },
    {
        "gene": "F13A1",
        "protein": (
            "F13A1 -- 6p24.3 AR LOF -- 732aa -- "
            "FXIII-A-deficiency-"
            "Umbilical-stump-bleeding-PATHOGNOMONIC-60-70pct-"
            "Intracranial-bleeding-HIGHEST-all-factor-deficiencies-25-30pct-lifetime-"
            "PT-APTT-TT-ALL-NORMAL-PATHOGNOMONIC-"
            "Urea-solubility-test-PATHOGNOMONIC-"
            "F13A1-OMIM-134570"
        ),
        "locus": "6p24.3",
        "protein_size": (
            "732 aa / ~83 kDa per subunit / 6p24.3 F13A1 Factor XIII A-subunit molecular context: "
            "STRUCTURE: "
            "  732 aa / ~83 kDa per A-subunit; F13 circulates as heterotetramer A2B2 (~340 kDa); "
            "  A-subunit (F13A1): catalytic transglutaminase subunit; carries active site Cys314; "
            "  B-subunit (F13B, 1q31.3): non-catalytic carrier protein; stabilizes A-subunit in plasma; "
            "  Activation by thrombin at Arg37 + Ca2+: FXIIIa (active transglutaminase); "
            "  FXIIIa function: cross-links fibrin alpha-alpha and gamma-gamma chains; "
            "  Cross-linking: converts soluble fibrin into mechanically stable, lysis-resistant clot; "
            "  Also cross-links alpha-2-antiplasmin to fibrin (lysis resistance); "
            "  Platelet FXIIIa (100% A2 homodimer): different from plasma FXIIIa; "
            "LABORATORY PARADOX (CRITICAL): "
            "  Standard coagulation: PT NORMAL; APTT NORMAL; TT NORMAL; fibrinogen NORMAL; "
            "  PATHOGNOMONIC: complete coagulation workup normal in FXIII deficiency; "
            "  FXIIIa acts AFTER fibrin polymerization (post-clot-formation); standard tests do NOT detect; "
            "  Urea solubility test (5M urea or 1% monochloroacetic acid): clot dissolves → positive for FXIII deficiency; "
            "  FXIII activity assay (quantitative): ammonia release or ELISA; definitive; "
            "  F13A1 sequencing: confirms genotype; "
            "CLINICAL SPECTRUM (UNIQUE FEATURES): "
            "  Umbilical stump bleeding (60-70%): MOST COMMON presentation; bleeds 24-72h after birth; PATHOGNOMONIC; "
            "  Intracranial hemorrhage (ICH): 25-30% lifetime risk; HIGHEST of ALL single-factor deficiencies; "
            "  ICH: spontaneous (no trauma) or minor trauma; prophylaxis MANDATORY to prevent ICH; "
            "  Wound healing defect: impaired cross-linking → poor wound tensile strength; delayed healing; "
            "  Spontaneous miscarriage (habitual): cross-linking of fibrin important for implantation; "
            "  Delayed post-circumcision bleeding: PATHOGNOMONIC in males; "
            "  Hemarthroses: RARE (unlike hemophilia A/B); "
            "PROPHYLAXIS (MANDATORY FOR ICH PREVENTION): "
            "  FXIII concentrate (Corifact, Fibrogammin): every 4-6 weeks; prophylaxis indicated in SEVERE FXIII def; "
            "  Target FXIII trough level: >10-20% for prophylaxis; >50% perioperative; "
            "  Regular FFP or cryoprecipitate: alternative if concentrate unavailable; "
            "  Do NOT wait for ICH to start prophylaxis — 25-30% lifetime risk is too high; "
            "RECURRENT MISCARRIAGE: "
            "  FXIII fibrin cross-linking required for trophoblast invasion/implantation; "
            "  FXIII deficiency → habitual/recurrent miscarriage in females (additional presentation); "
            "  Treatment during pregnancy: FXIII concentrate prophylaxis throughout pregnancy"
        ),
        "syndrome": "FXIII-A-deficiency / Umbilical-stump-bleeding-PATHOGNOMONIC / Intracranial-hemorrhage-HIGHEST-risk / PT-APTT-TT-ALL-NORMAL-PATHOGNOMONIC",
        "inheritance": "AR LOF (F13A1 biallelic; carriers 50% FXIII usually asymptomatic; homozygous/compound het symptomatic)",
        "bleeding_risk": "Severe <3%: umbilical stump PATHOGNOMONIC, ICH 25-30% lifetime HIGHEST, wound healing defect, recurrent miscarriage",
        "pathognomonic": "PT-APTT-TT ALL NORMAL despite severe bleeding (FXIII acts post-fibrin-polymerization); Urea-solubility-test clot dissolution; Umbilical-stump bleeding at birth 60-70%",
        "key_rule": "PROPHYLAXIS MANDATORY in severe FXIII deficiency — ICH risk 25-30% lifetime is highest of all single coagulation factor deficiencies; do NOT wait for first bleed",
        "key_avoid": "Assuming normal coagulation screen rules out coagulation disorder (FXIII deficiency misses PT/APTT/TT); delaying prophylaxis until after first ICH; missing umbilical stump bleed diagnosis",
        "surveillance": "FXIII trough levels monthly; ICH surveillance MRI if symptoms; obstetric plan (FXIII prophylaxis mandatory in pregnancy); wound assessment post-surgery",
        "targeted_rx": "FXIII-A concentrate (Corifact/Fibrogammin) every 4-6 weeks prophylaxis; target trough >10-20%; cryoprecipitate/FFP alternative; FXIII >50% perioperative target",
    },
    {
        "gene": "SERPINE1",
        "protein": (
            "SERPINE1 -- 7q22.1 AR LOF -- 402aa -- "
            "PAI-1-deficiency-plasminogen-activator-inhibitor-"
            "Delayed-wound-bleeding-hours-post-trauma-PATHOGNOMONIC-"
            "Postpartum-hemorrhage-PATHOGNOMONIC-"
            "Antifibrinolytics-tranexamic-acid-FIRST-LINE-NO-coag-factor-works-"
            "SERPINE1-OMIM-613329"
        ),
        "locus": "7q22.1",
        "protein_size": (
            "402 aa / ~52 kDa / 7q22.1 SERPINE1 PAI-1 molecular context: "
            "STRUCTURE: "
            "  402 aa / ~52 kDa; Plasminogen Activator Inhibitor-1 (PAI-1); serine protease inhibitor (serpin); "
            "  Reactive Center Loop (RCL; aa 369-380): Arg346-Met347 as pseudo-substrate for tPA and uPA; "
            "  Conformational mechanism: PAI-1 binds tPA/uPA → covalent inhibitory complex; "
            "  PAI-1 inhibits: tissue plasminogen activator (tPA) AND urokinase plasminogen activator (uPA); "
            "  Net effect: PAI-1 suppresses fibrinolysis; PAI-1 deficiency → HYPERFIBRINOLYSIS; "
            "  Stored in platelet alpha-granules (released during activation) and endothelium; "
            "  Vitronectin binding: stabilizes PAI-1 in active conformation (extends half-life 2h → 6h); "
            "4G/5G PROMOTER POLYMORPHISM (NOT complete deficiency): "
            "  4G/4G genotype: higher PAI-1 expression (less fibrinolysis → potential thrombotic risk); "
            "  5G/5G genotype: lower PAI-1 expression (MORE fibrinolysis); "
            "  Complete PAI-1 deficiency (biallelic null): extremely rare; true hereditary bleeding disorder; "
            "  4G/5G polymorphism: common variant, NOT a true deficiency disorder; "
            "UNIQUE BLEEDING PATTERN (PATHOGNOMONIC): "
            "  Initial hemostasis: NORMAL (platelet plug and fibrin clot form normally); "
            "  Delayed bleeding: 24-48h after wound/procedure as fibrin clot dissolves prematurely; "
            "  Pattern: 'bleed, stop, bleed again hours later' — PATHOGNOMONIC hyperfibrinolytic pattern; "
            "  Primary hemostasis: normal bleeding time, normal platelet function; "
            "  Post-partum hemorrhage: delayed PPH hours after delivery (uterus physiologically high fibrinolysis); "
            "  Post-surgical: bleeding after apparent initial hemostasis (wound reopens); "
            "  Mucosal: menorrhagia, epistaxis, post-dental extraction delayed bleeding; "
            "  Oozing from surgical wounds: hallmark; "
            "LABORATORY: "
            "  PT NORMAL; APTT NORMAL; platelet count NORMAL; fibrinogen NORMAL; "
            "  FXIII normal; D-dimer often elevated (ongoing fibrinolysis); "
            "  PAI-1 activity: undetectable or severely reduced (functional assay); "
            "  Euglobulin clot lysis time (ECLT): markedly shortened (rapid clot dissolution); "
            "  Thromboelastography (TEG/ROTEM): characteristic fibrinolysis pattern (ML >15%); "
            "  SERPINE1 sequencing: confirms biallelic null mutations; "
            "TREATMENT: "
            "  Antifibrinolytics (tranexamic acid, TXA): FIRST-LINE and HIGHLY EFFECTIVE; "
            "    Mechanism: TXA blocks plasminogen lysine-binding sites → inhibits plasminogen activation → reduces fibrinolysis; "
            "    Dose: 15-25 mg/kg TID orally or 1g IV; "
            "    Perioperative: TXA 1g IV at induction + 1g at 3h or continuous infusion; "
            "    Post-partum: TXA within 3h of delivery if PPH; "
            "  No coagulation factor concentrate works (coagulation is NORMAL — fibrinolysis is the problem); "
            "  rFVIIa: marginal benefit; NOT primary therapy; "
            "  OCP: reduces menorrhagia by reducing fibrinolytic activity in endometrium"
        ),
        "syndrome": "PAI-1-deficiency / Hyperfibrinolysis / Delayed-wound-bleeding-PATHOGNOMONIC / Postpartum-hemorrhage-PATHOGNOMONIC / Normal-coag-screen",
        "inheritance": "AR LOF biallelic (complete PAI-1 deficiency; partial = heterozygous may have mild bleeding)",
        "bleeding_risk": "Delayed wound bleeding hours-to-days post-trauma/surgery; post-partum hemorrhage; menorrhagia; normal initial hemostasis THEN re-bleeding",
        "pathognomonic": "Bleed-stop-bleed-again pattern (delayed fibrinolysis); Normal PT/APTT/platelets/fibrinogen; Euglobulin clot lysis time SHORTENED; ROTEM ML >15%",
        "key_rule": "Tranexamic acid FIRST-LINE and HIGHLY EFFECTIVE — antifibrinolytics correct the underlying hyperfibrinolysis; NO coagulation factor concentrate works for PAI-1 deficiency",
        "key_avoid": "Using coagulation factor concentrates (FVIII, FFP) — they do not address hyperfibrinolysis; missing delayed post-surgical bleeding presentation; omitting TXA perioperatively",
        "surveillance": "Pre-surgical antifibrinolytic plan; obstetric plan (TXA mandatory post-partum); menorrhagia management (OCP + TXA); dental plan (TXA oral rinse pre-extraction)",
        "targeted_rx": "Tranexamic acid 15-25 mg/kg TID oral OR 1g IV (FIRST-LINE); epsilon-aminocaproic acid alternative; OCP for menorrhagia; perioperative TXA mandatory",
    },
    {
        "gene": "ADAMTS13",
        "protein": (
            "ADAMTS13 -- 9q34.2 AR LOF -- 1427aa -- "
            "Congenital-TTP-Upshaw-Schulman-syndrome-"
            "MAHA+thrombocytopenia-HALLMARK-"
            "Neonatal-jaundice+thrombocytopenia-PATHOGNOMONIC-congenital-TTP-"
            "Plasma-infusion-CURATIVE-NO-immunosuppression-needed-"
            "ADAMTS13-OMIM-604134"
        ),
        "locus": "9q34.2",
        "protein_size": (
            "1427 aa / ~190 kDa / 9q34.2 ADAMTS13 molecular context: "
            "STRUCTURE: "
            "  1427 aa / ~190 kDa; A Disintegrin And Metalloproteinase with ThromboSpondin type 1 motifs, member 13; "
            "  Metalloproteinase domain (aa 291-554): Zn2+-coordinating HEXXHXXGXXHD motif; "
            "  Disintegrin domain, TSP1 repeat, Cys-rich, spacer domain, 7 TSP1 repeats, 2 CUB domains; "
            "  Substrate: ultra-large VWF (UL-VWF) multimers; "
            "  Cleavage site: Tyr842-Met843 in VWF A2 domain; "
            "  Under shear stress: VWF A2 domain unfolds → exposes Tyr842-Met843 → ADAMTS13 cleaves; "
            "  Result: UL-VWF → smaller hemostatically appropriate multimers; prevents spontaneous platelet aggregation; "
            "ADAMTS13 DEFICIENCY → CONGENITAL TTP: "
            "  Upshaw-Schulman syndrome (USS): ADAMTS13 <5-10% activity; "
            "  Accumulation of UL-VWF multimers → platelet microvascular thrombi; "
            "  End-organ damage: brain (neurological), kidney (AKI), heart; "
            "CLINICAL PRESENTATION: "
            "  Classic TTP pentad (only 5-10% have all 5): "
            "    1. Microangiopathic hemolytic anemia (MAHA): fragmented RBCs (schistocytes) + elevated LDH + low haptoglobin; "
            "    2. Thrombocytopenia: platelets <30 × 10^9/L typically; "
            "    3. Fever; 4. Neurological symptoms (confusion, stroke, seizure); 5. Renal dysfunction; "
            "  Neonatal/early infantile TTP: "
            "    Neonatal jaundice + thrombocytopenia in newborn → PATHOGNOMONIC for congenital TTP; "
            "    Presents at birth or within first months; triggered by infection, surgery, pregnancy; "
            "    Umbilical stump bleeding: NOT a feature (unlike FXIII deficiency); "
            "  Triggers in congenital TTP: pregnancy (MOST COMMON trigger in women), infection, surgery, OCP; "
            "  ADAMTS13 activity <5-10% at baseline, regardless of clinical activity; "
            "DISTINGUISHING CONGENITAL vs ACQUIRED TTP: "
            "  Congenital TTP (USS): "
            "    ADAMTS13 activity <5-10%; NO inhibitor (ADAMTS13 antibody ABSENT); "
            "    Plasma infusion replaces enzyme → CURATIVE for acute episode; "
            "    NO immunosuppression needed; "
            "    Family history; young age of onset; triggers; "
            "  Acquired TTP (immune-mediated): "
            "    ADAMTS13 <5-10% + ADAMTS13 inhibitor antibody PRESENT; "
            "    Caplacizumab (anti-VWF nanobody) + immunosuppression (rituximab, steroids); "
            "    PEX mandatory (removes antibody + replaces ADAMTS13); "
            "TREATMENT OF CONGENITAL TTP: "
            "  Acute episode: FFP infusion (10-15 mL/kg): provides ADAMTS13 enzyme; "
            "    Plasma exchange (PEX): NOT mandatory for congenital TTP (no antibody to remove); "
            "    FFP infusion alone sufficient for acute episodes in USS; "
            "  Prophylaxis: FFP infusion every 2-3 weeks (ADAMTS13 t1/2 ~3 days); "
            "  Recombinant ADAMTS13 (Adzynma, TAK-755): FDA 2023 for congenital TTP; "
            "    IV every 2 weeks; avoids plasma exposure; maintains ADAMTS13 >10%; "
            "  Triggers: avoid OCP (estrogen reduces ADAMTS13); folic acid in pregnancy; "
            "LABORATORY: "
            "  CBC: thrombocytopenia; Blood smear: schistocytes MANDATORY to confirm MAHA; "
            "  LDH markedly elevated; direct Coombs NEGATIVE (MAHA is non-immune hemolysis); "
            "  ADAMTS13 activity: <5-10% (send during acute episode); "
            "  ADAMTS13 inhibitor: ABSENT in congenital TTP (present in acquired); "
            "  Renal function, troponin, neuro imaging as clinically indicated"
        ),
        "syndrome": "Congenital-TTP / Upshaw-Schulman-syndrome / ADAMTS13-deficiency / MAHA+thrombocytopenia-HALLMARK / Plasma-infusion-CURATIVE",
        "inheritance": "AR LOF biallelic (ADAMTS13 <5-10%; carriers 50% ADAMTS13 asymptomatic)",
        "bleeding_risk": "NOT a primary bleeding disorder — THROMBOTIC microangiopathy: MAHA + thrombocytopenia + end-organ damage; neonatal jaundice + thrombocytopenia PATHOGNOMONIC",
        "pathognomonic": "ADAMTS13 <5-10% WITHOUT inhibitor (vs acquired TTP WITH antibody); Neonatal jaundice + thrombocytopenia = congenital TTP PATHOGNOMONIC; Schistocytes + thrombocytopenia + normal Coombs",
        "key_rule": "Plasma infusion (FFP) is CURATIVE for congenital TTP — replaces ADAMTS13 enzyme; NO immunosuppression needed (no antibody); distinguish from acquired TTP (inhibitor present) which needs PEX + rituximab",
        "key_avoid": "Starting immunosuppression (steroids/rituximab) for congenital TTP without checking ADAMTS13 inhibitor; missing neonatal TTP diagnosis; OCP in women with congenital TTP (estrogen reduces ADAMTS13)",
        "surveillance": "ADAMTS13 activity every 2-4 weeks on prophylaxis (target >10%); CBC biweekly if symptomatic; pregnancy ADAMTS13 monitoring; Adzynma infusion scheduling",
        "targeted_rx": "Adzynma (recombinant ADAMTS13, FDA 2023) IV every 2 weeks prophylaxis; FFP 10-15 mL/kg acute/prophylaxis; avoid OCP; pregnancy plan (FFP infusions every 2 weeks throughout pregnancy)",
    },
]


def _rng(seed):
    random.seed(seed)
    return random


def _gen_patient(gene_info, seed):
    rng = _rng(seed)
    gene = gene_info["gene"]

    age = int(rng.gauss(32, 12))
    age = max(0, min(80, age))

    def wchoice(population, weights):
        return rng.choices(population, weights=weights, k=1)[0]

    # Bleeding-specific fields (not thrombosis like VTE atlas)
    if gene in ("F8", "F9"):
        bleed_severity = wchoice(["severe", "moderate", "mild"], [0.5, 0.3, 0.2])
        hemarthrosis = 1 if (bleed_severity == "severe" and rng.random() < 0.7) else (
            1 if (bleed_severity == "moderate" and rng.random() < 0.3) else 0)
        inhibitor = 1 if (gene == "F8" and bleed_severity == "severe" and rng.random() < 0.28) else 0
        on_prophylaxis = 1 if (bleed_severity == "severe" or rng.random() < 0.4) else 0
        gene_therapy = 1 if rng.random() < 0.08 else 0
    elif gene == "VWF":
        bleed_severity = wchoice(["type1", "type2a", "type2b", "type2n", "type3"],
                                  [0.55, 0.15, 0.1, 0.1, 0.1])
        hemarthrosis = 1 if (bleed_severity == "type3" and rng.random() < 0.4) else 0
        inhibitor = 0
        on_prophylaxis = 1 if (bleed_severity == "type3" or rng.random() < 0.2) else 0
        gene_therapy = 0
    elif gene == "F11":
        bleed_severity = wchoice(["severe", "partial", "mild"], [0.25, 0.40, 0.35])
        hemarthrosis = 1 if rng.random() < 0.05 else 0
        inhibitor = 0
        on_prophylaxis = 0
        gene_therapy = 0
    elif gene == "F7":
        bleed_severity = wchoice(["severe", "moderate", "mild"], [0.20, 0.35, 0.45])
        hemarthrosis = 0
        inhibitor = 0
        on_prophylaxis = 1 if (bleed_severity == "severe" and rng.random() < 0.6) else 0
        gene_therapy = 0
    elif gene == "F13A1":
        bleed_severity = wchoice(["severe", "moderate"], [0.7, 0.3])
        hemarthrosis = 1 if rng.random() < 0.05 else 0
        inhibitor = 0
        on_prophylaxis = 1  # Mandatory for severe
        gene_therapy = 0
    elif gene == "SERPINE1":
        bleed_severity = wchoice(["complete", "partial"], [0.6, 0.4])
        hemarthrosis = 0
        inhibitor = 0
        on_prophylaxis = 1 if bleed_severity == "complete" else 0
        gene_therapy = 0
    elif gene == "ADAMTS13":
        bleed_severity = "severe_congenital"
        hemarthrosis = 0
        inhibitor = 0
        on_prophylaxis = 1 if rng.random() < 0.75 else 0
        gene_therapy = 0
    else:
        bleed_severity = "unknown"
        hemarthrosis = 0
        inhibitor = 0
        on_prophylaxis = 0
        gene_therapy = 0

    family_hx = 1 if rng.random() < 0.55 else 0
    icb = 1 if (gene == "F13A1" and rng.random() < 0.28) else (
          1 if (gene in ("F7", "F8") and bleed_severity == "severe" and rng.random() < 0.1) else 0)
    delayed_bleed = 1 if (gene == "SERPINE1" and rng.random() < 0.8) else 0
    umbilical_stump = 1 if (gene == "F13A1" and rng.random() < 0.65) else 0
    maha = 1 if (gene == "ADAMTS13") else 0

    return {
        "gene": gene,
        "age": age,
        "severity": bleed_severity,
        "hemarthrosis": hemarthrosis,
        "inhibitor": inhibitor,
        "on_prophylaxis": on_prophylaxis,
        "gene_therapy": gene_therapy,
        "family_hx": family_hx,
        "intracranial_bleed": icb,
        "delayed_bleed": delayed_bleed,
        "umbilical_stump_bleed": umbilical_stump,
        "maha_thrombocytopenia": maha,
    }


def _gene_summary(gene, patients):
    gp = [p for p in patients if p["gene"] == gene]
    n = len(gp)
    if n == 0:
        return {}
    return {
        "gene": gene,
        "n": n,
        "mean_age": round(sum(p["age"] for p in gp) / n, 1),
        "severe_count": sum(1 for p in gp if p["severity"] in ("severe", "severe_congenital", "complete")),
        "hemarthrosis_count": sum(p["hemarthrosis"] for p in gp),
        "inhibitor_count": sum(p["inhibitor"] for p in gp),
        "on_prophylaxis_count": sum(p["on_prophylaxis"] for p in gp),
        "gene_therapy_count": sum(p["gene_therapy"] for p in gp),
        "family_hx_count": sum(p["family_hx"] for p in gp),
        "intracranial_bleed_count": sum(p["intracranial_bleed"] for p in gp),
        "delayed_bleed_count": sum(p["delayed_bleed"] for p in gp),
        "umbilical_stump_count": sum(p["umbilical_stump_bleed"] for p in gp),
        "maha_count": sum(p["maha_thrombocytopenia"] for p in gp),
    }


def _generate_all_patients():
    patients = []
    for i, gene_info in enumerate(ATLAS_GENES):
        seed = SEED_BASE + i
        for j in range(40):
            patients.append(_gen_patient(gene_info, seed * 1000 + j))
    return patients


def generate_overview():
    patients = _generate_all_patients()
    n = len(patients)
    gene_summaries = [_gene_summary(g["gene"], patients) for g in ATLAS_GENES]
    return {
        "atlas": "Hereditary-Hemophilia-Rare-Coagulation-Factor-Deficiency-Atlas",
        "subtitle": "Complete 8-Gene Reference: F8-F9-VWF-F11-F7-F13A1-SERPINE1-ADAMTS13",
        "genes": [g["gene"] for g in ATLAS_GENES],
        "cohort_n": n,
        "seed_range": f"{SEED_BASE}-{SEED_BASE + len(ATLAS_GENES) - 1}",
        "gene_summaries": gene_summaries,
        "atlas_genes": ATLAS_GENES,
        "total_severe": sum(1 for p in patients if p["severity"] in ("severe", "severe_congenital", "complete")),
        "total_inhibitor": sum(p["inhibitor"] for p in patients),
        "total_prophylaxis": sum(p["on_prophylaxis"] for p in patients),
        "total_hemarthrosis": sum(p["hemarthrosis"] for p in patients),
        "total_icb": sum(p["intracranial_bleed"] for p in patients),
        "total_family_hx": sum(p["family_hx"] for p in patients),
        "total_maha": sum(p["maha_thrombocytopenia"] for p in patients),
        "key_clinical_rules": [
            "F8: FEIBA + emicizumab → thrombotic microangiopathy — use rFVIIa (NOT FEIBA) on emicizumab",
            "F9-Leyden: HB severity IMPROVES spontaneously post-puberty (androgen-responsive promoter) PATHOGNOMONIC",
            "VWF Type-2B: DDAVP CONTRAINDICATED ABSOLUTELY — releases abnormal VWF, worsens thrombocytopenia",
            "VWF Type-2N: mimics Hemophilia A — always test VWF FVIII-binding assay in apparent mild HA, especially females",
            "F11: Bleeding NOT proportional to FXI level — unique among all coagulation factor deficiencies",
            "F7: PT elevated + APTT NORMAL (isolated PT prolongation) PATHOGNOMONIC; exclude Vitamin K deficiency first",
            "F13A1: PT-APTT-TT ALL NORMAL despite severe bleeding (FXIII acts post-fibrin-polymerization); ICH risk HIGHEST 25-30%",
            "SERPINE1: No coagulation factor concentrate works — tranexamic acid FIRST-LINE (hyperfibrinolysis, not coagulation defect)",
            "ADAMTS13: Plasma infusion CURATIVE for congenital TTP — NO immunosuppression needed (no antibody in congenital)",
        ],
    }


def generate_breakdown():
    patients = _generate_all_patients()
    gene_data = {g["gene"]: _gene_summary(g["gene"], patients) for g in ATLAS_GENES}
    return {
        "atlas": "Hereditary-Hemophilia-Rare-Coagulation-Factor-Deficiency-Atlas",
        "breakdown_type": "gene-level-bleeding-disorder-aggregation",
        "gene_data": gene_data,
        "atlas_genes": ATLAS_GENES,
        "per_patient_sample": [
            {k: v for k, v in p.items()}
            for p in patients[:5]
        ],
    }


def generate_definitions():
    return {
        "atlas": "Hereditary-Hemophilia-Rare-Coagulation-Factor-Deficiency-Atlas",
        "definitions": {g["gene"]: g for g in ATLAS_GENES},
        "key_abbreviations": {
            "HA": "Hemophilia A (FVIII deficiency)",
            "HB": "Hemophilia B / Christmas disease (FIX deficiency)",
            "VWD": "von Willebrand disease",
            "FXI-def": "FXI deficiency / Hemophilia C",
            "FVII-def": "Factor VII deficiency",
            "FXIII-def": "Factor XIII A-subunit deficiency",
            "PAI-1-def": "PAI-1 deficiency / hyperfibrinolysis",
            "cTTP": "Congenital TTP / Upshaw-Schulman syndrome",
            "UL-VWF": "Ultra-large VWF multimers",
            "MAHA": "Microangiopathic hemolytic anemia",
            "TMA": "Thrombotic microangiopathy",
            "NPF": "Neonatal purpura fulminans",
            "ICH": "Intracranial hemorrhage",
            "XLR": "X-linked recessive",
            "AR": "Autosomal recessive",
            "AD": "Autosomal dominant",
            "LOF": "Loss of function",
            "GOF": "Gain of function",
            "APTT": "Activated partial thromboplastin time (intrinsic pathway)",
            "PT": "Prothrombin time (extrinsic pathway)",
            "TT": "Thrombin time",
            "BU": "Bethesda unit (inhibitor titer)",
            "PEX": "Plasma exchange",
            "TXA": "Tranexamic acid",
            "DDAVP": "Desmopressin (d-deamino-8-D-arginine vasopressin)",
            "rFVIIa": "Recombinant activated Factor VII (NovoSeven)",
            "FEIBA": "Factor Eight Inhibitor Bypassing Activity (activated PCC)",
            "EHL": "Extended half-life (coagulation factor concentrates)",
            "ARE": "Androgen-responsive element (F9 Leyden promoter)",
            "ECLT": "Euglobulin clot lysis time (fibrinolysis assay)",
            "USS": "Upshaw-Schulman syndrome (congenital TTP)",
        },
        "cascade_testing": {
            "hemophilia_a_b": "F8 or F9 gene sequencing; carrier females (FIX/FVIII level + sequencing); preimplantation genetic testing available",
            "vwd": "VWF antigen + activity + multimers + FVIII + FVIII-binding assay (Type 2N); ABO blood group; DDAVP challenge",
            "fxi_ashkenazi": "FXI activity; founder mutation panel (Type II + III); Ashkenazi Jewish family cascade",
            "fvii": "PT/APTT + FVII activity; exclude Vitamin K deficiency; F7 sequencing; newborn screen if family known",
            "fxiii": "Urea solubility test (screen); FXIII activity (quantitative); F13A1 sequencing; prophylaxis plan if severe",
            "pai1_serpine1": "PAI-1 activity; ECLT; ROTEM ML; SERPINE1 sequencing; distinguish from acquired PAI-1 deficiency",
            "adamts13_cttp": "ADAMTS13 activity + inhibitor; schistocyte smear; neonatal screen if family known; Adzynma prophylaxis plan",
        },
    }


if __name__ == "__main__":
    import json
    ov = generate_overview()
    print(f"Atlas: {ov['atlas']}")
    print(f"Cohort N: {ov['cohort_n']}")
    print(f"Genes: {', '.join(ov['genes'])}")
    print(f"Seeds: {ov['seed_range']}")
    print(f"Total severe: {ov['total_severe']}")
    print(f"Total inhibitors: {ov['total_inhibitor']}")
    print(f"Total ICB: {ov['total_icb']}")
    print(f"Total MAHA: {ov['total_maha']}")
    print("\nGene summaries:")
    for gs in ov["gene_summaries"]:
        print(f"  {gs['gene']}: n={gs['n']} severe={gs['severe_count']} inhibitor={gs.get('inhibitor_count',0)} icb={gs.get('intracranial_bleed_count',0)}")
    print("\nKey rules:")
    for r in ov["key_clinical_rules"]:
        print(f"  - {r}")
