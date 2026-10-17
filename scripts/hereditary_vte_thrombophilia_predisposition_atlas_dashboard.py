#!/usr/bin/env python3
"""Hereditary-VTE-Thrombophilia-Predisposition-Atlas -- Complete 8-Gene Reference
Inherited venous/arterial thrombosis and fibrinolysis germline predisposition.

F5       (Factor-V-Leiden; 2224aa; 7q35; AD LOF/GOF;
          Factor-V-Leiden-R506Q-1691G>A (OMIM 188055);
          APC resistance; most common hereditary thrombophilia 5% Europeans;
          heterozygous 3-8x VTE risk; homozygous 50-80x VTE risk;
          OCP+FVL compound risk 30-35x;
          LMWH prophylaxis surgical; lifelong anticoagulation homozygous;
          seed SEED_BASE+0).
F2       (Prothrombin-G20210A; 622aa; 11p11.2; AD GOF;
          Prothrombin G20210A 3-prime-UTR (OMIM 176930);
          elevated plasma prothrombin 30%; 2nd most common 1-3% Europeans;
          heterozygous 2-5x VTE risk; combined FVL+F2 15-20x;
          seed SEED_BASE+1).
PROC     (Protein-C; 461aa; 2q14.3; AR AD LOF;
          Protein-C-Deficiency Type-I-quantitative Type-II-qualitative (OMIM 612283);
          AR homozygous → Neonatal-Purpura-Fulminans PATHOGNOMONIC;
          warfarin-skin-necrosis PATHOGNOMONIC heterozygous;
          Protein-C-concentrate mandatory neonatal purpura fulminans;
          seed SEED_BASE+2).
PROS1    (Protein-S-Alpha; 676aa; 3q11.2; AD LOF;
          Protein-S-Deficiency Type-I-II-III (OMIM 612336);
          PROS2-pseudogene complicates Sanger WES-MANDATORY;
          free-protein-S low Type-I; functional-low Type-II; total-low Type-III;
          warfarin-skin-necrosis PATHOGNOMONIC;
          seed SEED_BASE+3).
SERPINC1 (Antithrombin-III; 432aa; 1q25.1; AD LOF;
          Antithrombin-III-Deficiency Type-I Type-II-HBS-RS-PE (OMIM 107300);
          most severe hereditary thrombophilia of the 5;
          Type-II-HBS heparin-binding-site → heparin-resistance PATHOGNOMONIC → AT-concentrate MANDATORY;
          seed SEED_BASE+4).
THBD     (Thrombomodulin; 575aa; 20p11.21; AD LOF;
          Thrombomodulin-Deficiency arterial-thrombosis-predisposition (OMIM 188040);
          reduced-APC-generation → arterial MI/stroke YOUNG without atherosclerosis;
          coronary-angiography negative PATHOGNOMONIC young MI + family history;
          seed SEED_BASE+5).
PLG      (Plasminogen; 810aa; 6q26; AR LOF;
          Plasminogen-Deficiency Type-I-hypoplasminogenemia (OMIM 217090);
          ligneous-conjunctivitis PATHOGNOMONIC pseudomembranous;
          fibrinolysis ABSENT; ligneous-gingivitis-tracheobronchitis extracorneal;
          topical-Lys-plasminogen surgery; recurrent-VTE-fibrinolysis-absent;
          seed SEED_BASE+6).
FGG      (Fibrinogen-Gamma; 453aa; 4q32.1; AD LOF GOF;
          Thrombophilic-Dysfibrinogenemia FGG-R301C Fibrinogen-Bicetre-I (OMIM 134850);
          fibrin-polymerization-defect; arterial+venous-thrombosis-paradox;
          thrombin-clotting-time prolonged PATHOGNOMONIC normal fibrinogen level;
          functional/immunologic-ratio less-than-0.7 PATHOGNOMONIC;
          seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 3678-3685)
"""
import random

SEED_BASE = 3678

ATLAS_GENES = [
    {
        "gene": "F5",
        "protein": (
            "F5 -- 7q35 AD LOF/GOF -- 2224aa -- "
            "Factor-V-Leiden-R506Q-1691G>A-"
            "APC-Resistance-Coagulation-Cofactor-"
            "FVL-Heterozygous-5pct-Europeans-3-8x-VTE-"
            "F5-OMIM-188055"
        ),
        "locus": "7q35",
        "protein_size": (
            "2224 aa / ~330 kDa / 7q35 F5 Factor-V-Leiden molecular context: "
            "STRUCTURE: "
            "  2224 aa / ~330 kDa; coagulation Factor V; procofactor for prothrombinase complex; "
            "  A1-A2-B-A3-C1-C2 domain architecture; activated by thrombin/FXa (FVa) at Arg709-Arg1018-Arg1545; "
            "  APC cleavage sites: Arg506 (primary; fast), Arg306 (secondary; slow), Arg679 (tertiary); "
            "  FVL R506Q (c.1691G>A): abolishes primary APC cleavage site at Arg506; "
            "  FVa-R506Q persists in prothrombinase complex 10-fold longer than wildtype FVa; "
            "  Heterozygous: 3-8x VTE lifetime risk; homozygous: 50-80x VTE lifetime risk; "
            "APC RESISTANCE CONTEXT: "
            "  Activated Protein C (APC) + Protein S inactivate FVa/FVIIIa → natural anticoagulant; "
            "  FVL abolishes this inactivation → sustained thrombin generation; "
            "  Inherited APC resistance: FVL accounts for >95% of all cases; "
            "  Combined FVL + OCP: additive risk 30-35x VTE (multiplicative, not additive); "
            "  Combined FVL + F2 G20210A: 15-20x risk; "
            "  Thrombocytopenia: NOT FVL feature → think HIT, TTP if platelets drop on heparin; "
            "FVL TESTING CONTEXT: "
            "  Genetic test: PCR-RFLP c.1691G>A or direct sequencing (Arg506Gln); "
            "  APC resistance ratio (APCR ratio <2.0 plasma clotting test): screening sensitivity >95% for FVL; "
            "  APCR ratio normalized by patient's own FV-depleted plasma: distinguishes FVL from acquired APCR (lupus); "
            "  Lupus anticoagulant (LA): also causes APCR but NOT FVL genetic → always test anti-beta2GPI + cardiolipin; "
            "TREATMENT: "
            "  Heterozygous first VTE: 3-6 months anticoagulation (LMWH/DOAC); "
            "  Homozygous or recurrent VTE: lifelong anticoagulation; "
            "  LMWH prophylaxis: all major surgeries, pregnancy (heterozygous if prior VTE); "
            "  DOACs (rivaroxaban/apixaban): effective, no routine monitoring needed; "
            "  Warfarin: avoid in acute phase (protein C/S window); target INR 2.0-3.0"
        ),
        "syndrome": "Factor-V-Leiden / APC-Resistance / Heterozygous-5%-Europeans",
        "inheritance": "AD (semi-dominant; homozygous >> heterozygous risk)",
        "thrombosis_risk": "Heterozygous 3-8x VTE; Homozygous 50-80x VTE; OCP+FVL 30-35x",
        "pathognomonic": "APCR ratio <2.0 without lupus anticoagulant + FVL PCR positive",
        "key_rule": "Combined FVL+OCP risk is MULTIPLICATIVE not additive; always counsel before prescribing OCP",
        "key_avoid": "Oestrogen-containing OCP/HRT without risk counselling in FVL heterozygous; factor V inhibitors",
        "surveillance": "Annual assessment; pregnancy VTE prophylaxis plan; family cascade from age 16-18 (reproductive counselling)",
        "targeted_rx": "DOAC (rivaroxaban/apixaban) preferred over warfarin for VTE treatment; LMWH prophylaxis surgical/pregnant",
    },
    {
        "gene": "F2",
        "protein": (
            "F2 -- 11p11.2 AD GOF -- 622aa -- "
            "Prothrombin-G20210A-3prime-UTR-"
            "Elevated-Prothrombin-130pct-2nd-Most-Common-"
            "F2-OMIM-176930"
        ),
        "locus": "11p11.2",
        "protein_size": (
            "622 aa / ~72 kDa / 11p11.2 F2 Prothrombin G20210A molecular context: "
            "STRUCTURE: "
            "  622 aa / ~72 kDa; prothrombin (coagulation Factor II); serine protease zymogen; "
            "  Gla domain (aa 1-46): gamma-carboxylated Glu residues; Ca2+/membrane binding; "
            "  Kringle-1 (aa 65-143) and Kringle-2 (aa 144-222): ligand binding domains; "
            "  Catalytic serine protease domain (aa 284-622): His363-Asp419-Ser525 catalytic triad; "
            "  Activation: prothrombinase complex (FXa-FVa-Ca2+-phospholipid) cleaves Arg271-Thr272 and Arg320-Ile321; "
            "  G20210A LOCATION: 3' untranslated region (3'UTR), 20 nt from poly-A signal; "
            "  G20210A MECHANISM: increased 3'UTR cleavage efficiency → enhanced mRNA stability → increased translation; "
            "  Plasma prothrombin level: ~130% of normal in G20210A heterozygotes (vs 100% wildtype); "
            "  ~2-5x VTE risk heterozygous; ~2% Europeans carrier frequency; "
            "COMBINED MUTATIONS: "
            "  F2 G20210A + FVL: 15-20x VTE risk (multiplicative); "
            "  F2 G20210A + PROC deficiency: 5-10x baseline risk of PROC alone; "
            "  F2 G20210A + OCP: 10-15x VTE risk; "
            "  Cerebral venous sinus thrombosis (CVST): F2 G20210A enriched in CVST population; "
            "TESTING: "
            "  PCR allele-specific amplification or Sanger sequencing at c.20210G>A (IVS14+1, 3'UTR); "
            "  NOT detectable on standard coagulation panels (PT/APTT/TT all normal); "
            "  Prothrombin functional level: elevated ~130% correlates with genotype; "
            "  Lupus anticoagulant panel: always co-test; APS can mimic; "
            "TREATMENT: "
            "  Heterozygous single VTE: 3-6 months DOAC; "
            "  Recurrent VTE or combined mutation: extended or lifelong anticoagulation; "
            "  Pregnancy: LMWH prophylaxis from 6 weeks if prior VTE or additional risk factors; "
            "  OCP contraindicated: relative (inform, not absolute); "
            "  DOAC preferred over warfarin in young patients"
        ),
        "syndrome": "Prothrombin-G20210A / Elevated-Prothrombin-130% / 2nd-Most-Common-Hereditary-Thrombophilia",
        "inheritance": "AD (G20210A 3'UTR; semi-dominant GOF; homozygous ~6-18x)",
        "thrombosis_risk": "Heterozygous 2-5x VTE; Combined FVL+F2 15-20x; CVST enriched 3x",
        "pathognomonic": "Normal PT/APTT + elevated prothrombin functional level ~130% + G20210A PCR positive",
        "key_rule": "F2 G20210A not detectable on standard coag panel; always test genetically in unexplained VTE workup",
        "key_avoid": "Combined OCP+F2 (relative; 10-15x risk); failing to test CVST patients for F2",
        "surveillance": "Annual; pregnancy LMWH prophylaxis plan; family cascade (siblings, children from age 16)",
        "targeted_rx": "DOAC (rivaroxaban/apixaban); LMWH in pregnancy; extended anticoagulation if recurrent/combined mutations",
    },
    {
        "gene": "PROC",
        "protein": (
            "PROC -- 2q14.3 AR AD LOF -- 461aa -- "
            "Protein-C-Deficiency-Type-I-quantitative-Type-II-qualitative-"
            "Neonatal-Purpura-Fulminans-AR-PATHOGNOMONIC-"
            "Warfarin-Skin-Necrosis-PATHOGNOMONIC-"
            "PROC-OMIM-612283"
        ),
        "locus": "2q14.3",
        "protein_size": (
            "461 aa / ~62 kDa / 2q14.3 PROC Protein C molecular context: "
            "STRUCTURE: "
            "  461 aa / ~62 kDa; Protein C vitamin-K-dependent serine protease; natural anticoagulant; "
            "  Gla domain (aa 1-46): 9 Glu gamma-carboxylated; Ca2+/membrane/thrombomodulin binding; "
            "  EGF-like domains 1+2 (aa 47-136): co-factor binding surface; "
            "  Activation peptide (aa 137-155): cleaved by thrombin-thrombomodulin complex; "
            "  Serine protease catalytic domain (aa 156-461): His211-Asp257-Ser360 triad; "
            "  Protein C activation: thrombin-THBD complex on endothelium (EPCR-assisted) cleaves Arg169; "
            "  APC (Activated Protein C): with Protein S co-factor → proteolytic inactivation of FVa (Arg506/306) and FVIIIa; "
            "TYPE I vs TYPE II: "
            "  Type I (Quantitative): both functional AND antigenic Protein C <50% → LOF mutation; "
            "  Type II (Qualitative): antigen normal but functional activity <50% → dysfunctional PC; "
            "  Heterozygous AD: ~50% Protein C levels → 3-8x VTE risk; "
            "  Homozygous/compound het AR: <1% Protein C → Neonatal Purpura Fulminans WITHIN HOURS of birth; "
            "NEONATAL PURPURA FULMINANS (NPF): "
            "  Presentation: Livedo reticularis → purpura → skin necrosis within first 24-72h of life PATHOGNOMONIC; "
            "  DIC-like consumptive coagulopathy WITHIN HOURS; "
            "  Protein C concentrate (Ceprotin): EMERGENCY treatment; dosed q6-12h to keep PC >50%; "
            "  Long-term: warfarin at high INR OR Protein C concentrate lifelong; "
            "WARFARIN SKIN NECROSIS: "
            "  First days of warfarin loading → transient hypercoagulability (PC half-life 6h, shorter than prothrombin); "
            "  Protein C deficiency: warfarin skin necrosis risk 5-10x PATHOGNOMONIC association; "
            "  Prevention: ALWAYS overlap LMWH with warfarin for minimum 5 days + INR >2; "
            "  ALWAYS check baseline protein C level BEFORE starting warfarin; "
            "TESTING: "
            "  Functional Protein C activity (chromogenic/clotting): <50% diagnostic; "
            "  Antigenic Protein C: distinguishes Type I vs II; "
            "  Must test OFF warfarin (warfarin suppresses PC 50-60%); "
            "  PROC sequencing: confirms diagnosis, identifies family mutation; "
            "TREATMENT: "
            "  Heterozygous first VTE: 3-6 months anticoagulation; "
            "  Warfarin: overlap LMWH minimum 5 days; start LOW dose (2mg); "
            "  DOAC preferred over warfarin (no protein C window issue); "
            "  Neonatal: Protein C concentrate q6-12h EMERGENCY; transition to anticoagulation"
        ),
        "syndrome": "Protein-C-Deficiency / Type-I-quantitative-Type-II-qualitative / NPF-AR-homozygous-PATHOGNOMONIC",
        "inheritance": "AD heterozygous VTE; AR homozygous → Neonatal Purpura Fulminans",
        "thrombosis_risk": "Heterozygous 3-8x VTE; AR homozygous: NPF within hours of birth; warfarin skin necrosis risk 5-10x",
        "pathognomonic": "Neonatal Purpura Fulminans within 24-72h of birth (homozygous) OR Warfarin-skin-necrosis (heterozygous)",
        "key_rule": "ALWAYS overlap LMWH with warfarin minimum 5 days + INR>2 to prevent warfarin skin necrosis; check PC BEFORE warfarin",
        "key_avoid": "Warfarin loading without LMWH overlap in Protein-C-deficiency (warfarin skin necrosis); OCP additional risk",
        "surveillance": "Annual protein C level; pregnancy plan; family cascade; neonatal screening if both parents carriers",
        "targeted_rx": "DOAC preferred; warfarin only with LMWH bridge; Protein-C-concentrate (Ceprotin) for NPF and peri-operative",
    },
    {
        "gene": "PROS1",
        "protein": (
            "PROS1 -- 3q11.2 AD LOF -- 676aa -- "
            "Protein-S-Alpha-Deficiency-Type-I-II-III-"
            "PROS2-pseudogene-WES-MANDATORY-"
            "Free-Functional-Total-Protein-S-"
            "PROS1-OMIM-612336"
        ),
        "locus": "3q11.2",
        "protein_size": (
            "676 aa / ~75 kDa / 3q11.2 PROS1 Protein S Alpha molecular context: "
            "STRUCTURE: "
            "  676 aa / ~75 kDa; Protein S vitamin-K-dependent non-enzymatic cofactor; "
            "  Gla domain (aa 1-49): 11 gamma-carboxylated Glu; Ca2+/phospholipid/C4BP binding; "
            "  Thrombin-sensitive region (TSR, aa 47-75): thrombin cleavage site (Arg60); "
            "  EGF-like domains 1-4 (aa 76-247): APC-binding surface (EGF3-4 CRITICAL); "
            "  SHBG-like domain (aa 248-676): sex-hormone binding globulin homology; C4b-binding protein (C4BP) interaction; "
            "  Protein S circulates: 60% bound to C4BP (inactive); 40% FREE (active APC co-factor); "
            "PROTEIN S TYPE CLASSIFICATION: "
            "  Type I: free AND total AND functional ALL low (<50-65%); most common (~55% of PROS1 deficiencies); "
            "  Type II: antigen (free + total) NORMAL; functional activity LOW → dysfunctional PS; "
            "  Type III: free PS low; total PS NORMAL; functional reduced; C4BP binding abnormal; "
            "  Clinical impact: all three types carry similar VTE risk; "
            "  Type I must exclude C4BP elevation (acute phase reactant → PS redistributes to bound form → falsely low free PS); "
            "PROS2 PSEUDOGENE CRITICAL: "
            "  PROS2: non-functional pseudogene at 3q11.2, ~97% homologous to PROS1 exons 1-14; "
            "  Sanger sequencing: PROS2 amplification artifacts → false positives PATHOGNOMONIC laboratory pitfall; "
            "  WES with PROS2 exclusion filters: MANDATORY for PROS1 variant confirmation; "
            "  Gene conversion PROS2→PROS1: recurrent mechanism, requires long-range PCR confirmation; "
            "  Report: always confirm PROS1 variant by PROS2-free long-range PCR before clinical action; "
            "TESTING: "
            "  Must test OFF warfarin AND OFF OCP (both suppress PS 20-40%); "
            "  Pregnancy: PS physiologically falls 50% by trimester 1 → NEVER diagnose in pregnancy; "
            "  Free protein S antigen (ELISA): key for Type I/III; "
            "  Functional PS assay (clotting): all types; "
            "  Total PS antigen (ELISA): distinguishes I vs III; "
            "  C4BP level: exclude elevated C4BP causing acquired deficiency; "
            "TREATMENT: "
            "  DOAC preferred; warfarin with LMWH bridge (same warfarin skin necrosis risk as PROC); "
            "  Pregnancy: LMWH if prior VTE or other risk factors; free PS monitored but NOT used to dose LMWH; "
            "  Family cascade testing: first-degree relatives"
        ),
        "syndrome": "Protein-S-Alpha-Deficiency / Type-I-II-III / PROS2-pseudogene-WES-MANDATORY",
        "inheritance": "AD LOF (heterozygous; ~50-65% free Protein S)",
        "thrombosis_risk": "Heterozygous 2-8x VTE; OCP+PROS1 deficiency 10-15x; CVST risk increased",
        "pathognomonic": "Sanger PROS1 false variant due to PROS2 pseudogene amplification → WES with PROS2 exclusion filters PATHOGNOMONIC pitfall",
        "key_rule": "NEVER diagnose Protein S deficiency in pregnancy (physiological 50% fall) or on OCP/warfarin; WES MANDATORY to exclude PROS2 artifact",
        "key_avoid": "Diagnosing PROS1 deficiency by Sanger alone (PROS2 pseudogene false positives); diagnosing in pregnancy",
        "surveillance": "Annual off warfarin/OCP; pregnancy free-PS falls physiologically; family cascade testing",
        "targeted_rx": "DOAC preferred (no PS window); warfarin only with LMWH bridge; LMWH in pregnancy if prior VTE",
    },
    {
        "gene": "SERPINC1",
        "protein": (
            "SERPINC1 -- 1q25.1 AD LOF -- 432aa -- "
            "Antithrombin-III-Deficiency-Type-I-II-HBS-RS-PE-"
            "Most-Severe-Hereditary-Thrombophilia-"
            "Heparin-Resistance-Type-II-HBS-PATHOGNOMONIC-"
            "AT-Concentrate-MANDATORY-"
            "SERPINC1-OMIM-107300"
        ),
        "locus": "1q25.1",
        "protein_size": (
            "432 aa / ~58 kDa / 1q25.1 SERPINC1 Antithrombin III molecular context: "
            "STRUCTURE: "
            "  432 aa / ~58 kDa; antithrombin III (AT3, AT-III); serine protease inhibitor (serpin); "
            "  Reactive center loop (RCL, aa 382-407): Arg393-Ser394 scissile bond; inhibitory mechanism; "
            "  Heparin-binding site (aa 41-49, 107, 129, 132-136, Lys residues): heparin pentasaccharide binding; "
            "  Three beta-sheets (A, B, C) + nine alpha-helices structural core; "
            "  Mechanism: thrombin/FXa/FIXa/FXIa/FXIIa attack RCL → covalent 1:1 inhibitory complex → cleared; "
            "  Heparin accelerates AT inhibitory rate 1000-fold → therapeutic basis for heparin anticoagulation; "
            "  HSPG on endothelium: physiological AT activation; continuous basal anticoagulation; "
            "AT DEFICIENCY CLASSIFICATION: "
            "  Type I (Quantitative): both activity AND antigen <75-80% → gene deletion/frameshift/splice/missense LOF; "
            "  Type II-RS (Reactive-Site): RCL mutation; reduced thrombin/FXa inhibition; antigen normal; "
            "  Type II-HBS (Heparin-Binding-Site): HBS mutation; heparin cofactor function lost; antigen normal; "
            "  Type II-PE (Pleiotropic Effect): affects both RCL and HBS function; "
            "  Type II-HBS: HEPARIN RESISTANCE PATHOGNOMONIC → UFH fails to achieve therapeutic APTT → SWITCH TO AT concentrate; "
            "  Type I: HIGHEST VTE risk among all AT subtypes; "
            "HEPARIN RESISTANCE: "
            "  Type II-HBS: heparin infusion → APTT does NOT prolong despite high doses PATHOGNOMONIC; "
            "  Mechanism: AT HBS mutation → heparin cannot bind AT → no acceleration of inhibition; "
            "  Clinical recognition: escalating UFH doses still fail; APTT <1.5x control at max UFH dose; "
            "  Management: STOP UFH → use AT concentrate (Thrombate III 50-100 IU/kg) → re-establish heparin; "
            "  OR switch to argatroban (direct thrombin inhibitor, AT-independent); "
            "MOST SEVERE HEREDITARY THROMBOPHILIA: "
            "  AT deficiency VTE risk: 5-50x lifetime risk (highest of the 5 classic thrombophilias); "
            "  First VTE: typically age 20-35; spontaneous DVT/PE; mesenteric vein; CVST; "
            "  Homozygous AT deficiency: incompatible with life (in utero lethality or extreme thrombosis); "
            "TESTING: "
            "  AT functional (heparin co-factor activity chromogenic assay): <75-80% diagnostic; "
            "  AT antigen (immunological): distinguishes Type I vs Type II; "
            "  Test OFF heparin (UFH/LMWH suppresses AT 20-30%); "
            "  Test OFF in acute VTE phase (AT consumed during thrombosis); "
            "TREATMENT: "
            "  First VTE: anticoagulate; DOAC or warfarin; "
            "  Type II-HBS: NEVER rely on UFH → argatroban or AT concentrate + LMWH; "
            "  High-risk: lifelong anticoagulation; "
            "  Pregnancy: high-dose LMWH + AT concentrate peri-partum for AT deficiency; "
            "  Surgical: AT concentrate pre-op to raise AT >80% before stopping anticoagulation"
        ),
        "syndrome": "Antithrombin-III-Deficiency / Type-I-Type-II-HBS-Heparin-Resistance / Most-Severe-Hereditary-Thrombophilia",
        "inheritance": "AD LOF (heterozygous; ~50-75% AT functional activity; homozygous lethal)",
        "thrombosis_risk": "5-50x VTE lifetime risk (highest of classic thrombophilias); spontaneous age 20-35; mesenteric/CVST",
        "pathognomonic": "Heparin resistance (Type II-HBS): APTT fails to prolong despite escalating UFH doses + AT functional <80%",
        "key_rule": "Type II-HBS: STOP UFH → switch to AT concentrate or argatroban; NEVER escalate UFH blindly in non-responding APTT",
        "key_avoid": "UFH alone in Type-II-HBS AT deficiency (therapeutic failure); testing AT during acute VTE phase or on heparin (falsely low)",
        "surveillance": "Annual AT functional level; pregnancy AT concentrate plan; family cascade testing",
        "targeted_rx": "DOAC preferred long-term; AT concentrate (Thrombate III) peri-operative/peri-partum; argatroban for Type-II-HBS acute",
    },
    {
        "gene": "THBD",
        "protein": (
            "THBD -- 20p11.21 AD LOF -- 575aa -- "
            "Thrombomodulin-Deficiency-Arterial-Thrombosis-"
            "Reduced-APC-Generation-Young-MI-Stroke-"
            "THBD-OMIM-188040"
        ),
        "locus": "20p11.21",
        "protein_size": (
            "575 aa / ~75 kDa / 20p11.21 THBD Thrombomodulin molecular context: "
            "STRUCTURE: "
            "  575 aa / ~75 kDa; transmembrane endothelial glycoprotein; type-I single TM helix; "
            "  Lectin-like domain (aa 1-155): anti-inflammatory, anti-complement; "
            "  EGF-like domain array (aa 222-462): 6 EGF repeats; EGF4-6 responsible for thrombin binding; "
            "  EGF4-6 binds thrombin (Kd ~0.5 nM): changes thrombin specificity from procoagulant to anticoagulant; "
            "  Thrombin-THBD complex: 1000x faster Protein C activation vs thrombin alone; "
            "  Serine-threonine rich domain (aa 462-497): chondroitin sulphate GAG attachment; "
            "  Transmembrane domain (aa 498-521); cytoplasmic tail (aa 522-575); "
            "  EPCR (Endothelial Protein C Receptor): co-presents Protein C to THBD-thrombin complex; "
            "ARTERIAL THROMBOSIS MECHANISM: "
            "  THBD LOF → reduced thrombin-THBD complex on endothelium → reduced Protein C activation; "
            "  Result: impaired APC generation → sustained thrombin activity on arterial endothelium; "
            "  Arterial vs venous predisposition: THBD defects more associated with ARTERIAL thrombosis; "
            "    (vs FVL/F2: predominantly VENOUS thrombosis); "
            "CLINICAL PRESENTATION: "
            "  Young MI (<50y): without traditional cardiovascular risk factors PATHOGNOMONIC for THBD deficiency; "
            "  Stroke in young adults: ischaemic, without AF or atherosclerosis; "
            "  Coronary angiography: normal arteries or minimal atherosclerosis in young MI PATHOGNOMONIC; "
            "  Family history: first-degree relatives with premature MI/stroke in non-smoker/non-diabetic; "
            "  Positive family history + young thrombosis + negative standard atherosclerosis risk: TEST THBD; "
            "  Venous thrombosis: also occurs; "
            "TESTING: "
            "  THBD serum/plasma level (soluble THBD): moderately reduced in THBD deficiency (not reliable); "
            "  THBD gene sequencing: preferred; "
            "  APC generation test: research assay; reduced in THBD LOF; "
            "  Standard coag screen (PT/APTT): NORMAL → does NOT detect THBD deficiency; "
            "  Protein C level: NORMAL (PROC gene intact; defect is in Protein C activation rate); "
            "TREATMENT: "
            "  Anticoagulation: DOACs for VTE; dual antiplatelet for arterial; "
            "  Statins + aspirin: reduce arterial thrombosis risk; "
            "  Aggressive CV risk factor modification (smoking cessation, BP, lipids); "
            "  DOAC vs aspirin: shared decision for arterial vs venous dominant phenotype; "
            "  Genetic counselling: siblings 50% risk; cascade testing young relatives; "
            "  Recombinant thrombomodulin (ART-123): approved Japan for DIC; research use in THBD deficiency"
        ),
        "syndrome": "Thrombomodulin-Deficiency / Arterial-Thrombosis-Young-MI-Stroke / Reduced-APC-Generation",
        "inheritance": "AD LOF heterozygous (reduced THBD expression on endothelium)",
        "thrombosis_risk": "Predominantly ARTERIAL: young MI/stroke without atherosclerosis; also VTE 2-3x; coronary-NORMAL on angiography",
        "pathognomonic": "Young MI (<50y) + normal coronary angiogram + negative standard CV risk factors + family history of premature MI/stroke",
        "key_rule": "Standard coag screen NORMAL in THBD deficiency; Protein C level NORMAL; test THBD gene in young unexplained arterial thrombosis",
        "key_avoid": "Missing THBD in young MI workup (standard thrombophilia screen does NOT include THBD); treating as standard ACS without genetic workup",
        "surveillance": "Annual CV review; lipids/BP/smoking cessation; echo; cascade testing siblings >18y",
        "targeted_rx": "Dual antiplatelet (aspirin+clopidogrel) for arterial; DOAC for venous; statins; ART-123 investigational",
    },
    {
        "gene": "PLG",
        "protein": (
            "PLG -- 6q26 AR LOF -- 810aa -- "
            "Plasminogen-Deficiency-Type-I-Hypoplasminogenemia-"
            "Ligneous-Conjunctivitis-PATHOGNOMONIC-"
            "Fibrinolysis-ABSENT-VTE-Predisposition-"
            "PLG-OMIM-217090"
        ),
        "locus": "6q26",
        "protein_size": (
            "810 aa / ~92 kDa / 6q26 PLG Plasminogen molecular context: "
            "STRUCTURE: "
            "  810 aa / ~92 kDa; plasminogen; fibrinolysis serine protease zymogen; "
            "  Activation peptide (aa 1-77): removed during fibrinolysis; "
            "  Five Kringle domains (K1-K5, aa 78-540): lysine-binding pockets; fibrin/tPA/alpha2AP binding; "
            "  K1: binds alpha2-antiplasmin; K4: highest affinity lysine binding; K5: fibrin specificity; "
            "  Serine protease catalytic domain (aa 542-810): His603-Asp646-Ser741 triad; "
            "  Activation: tPA (on fibrin surface) + urokinase (uPA) cleave Arg561-Val562 → plasmin; "
            "  Plasmin: degrades fibrin→FDP+D-dimer; activates MMP; lyses extracellular matrix; "
            "TYPE I vs TYPE II: "
            "  Type I (Hypoplasminogenemia): both FUNCTIONAL and IMMUNOLOGICAL plasminogen <50%; "
            "  Type II (Dysplasminogenemia): antigen NORMAL; functional <50%; qualitative defect; "
            "  Type I = LIGNEOUS CONJUNCTIVITIS phenotype; Type II = milder/asymptomatic; "
            "LIGNEOUS CONJUNCTIVITIS (PATHOGNOMONIC for Type I): "
            "  Wooden (ligneous) yellow-white pseudomembranes on conjunctiva; "
            "  Histology: amorphous fibrin+immunoglobulin deposits (PATHOGNOMONIC); "
            "  Mechanism: fibrin deposits cannot be cleared by plasmin → chronic pseudomembrane; "
            "  Recurs after surgical removal unless plasminogen restored; "
            "  Extracorneal ligneous: gingivitis+periodontitis (gum ligneous), tracheobronchitis (LIFE-THREATENING), nasopharynx, genital; "
            "  Tracheobronchial ligneous: can cause airway obstruction → EMERGENCY ligneous tracheitis; "
            "VTE MECHANISM: "
            "  Plasminogen deficiency → impaired fibrinolysis → fibrin accumulates in vessels; "
            "  VTE (DVT/PE): documented in PLG Type I; recurrent VTE common; "
            "  Fibrin-related thrombosis: particularly post-surgery, immobility, pregnancy; "
            "TESTING: "
            "  Plasminogen functional activity (chromogenic): <50% diagnostic; "
            "  Plasminogen antigen (ELISA): distinguishes Type I vs II; "
            "  PLG gene sequencing: biallelic pathogenic variants for AR Type I; "
            "  Slit-lamp ophthalmology: ligneous conjunctivitis; "
            "TREATMENT: "
            "  Topical Lys-plasminogen (Ryplazim): FDA-approved 2021 for PLG Type I ligneous conjunctivitis; "
            "    apply to conjunctival fornix before surgical pseudomembrane removal; "
            "  IV Lys-plasminogen (Ryplazim): for systemic manifestations; "
            "  Anticoagulation: LMWH/DOAC for VTE; "
            "  Prophylactic LMWH: surgical procedures, pregnancy; "
            "  Avoid trauma/surgery without plasminogen cover; "
            "  Tranexamic acid CONTRAINDICATED (blocks plasminogen → worsens ligneous lesions)"
        ),
        "syndrome": "Plasminogen-Deficiency-Type-I / Ligneous-Conjunctivitis-PATHOGNOMONIC / Fibrinolysis-ABSENT / VTE",
        "inheritance": "AR biallelic for Type I (severe); AD heterozygous for partial Type I/II (milder)",
        "thrombosis_risk": "VTE recurrent; fibrin accumulation; post-surgical fibrin deposition; ligneous tracheobronchitis life-threatening",
        "pathognomonic": "Ligneous conjunctivitis (wooden pseudomembranes on conjunctiva) + plasminogen functional <50% = PATHOGNOMONIC Type I",
        "key_rule": "Tranexamic acid CONTRAINDICATED in PLG deficiency (blocks lysine binding → worsens ligneous lesions); use plasminogen replacement instead",
        "key_avoid": "Tranexamic acid (worsens ligneous disease); surgical pseudomembrane removal without plasminogen cover (immediate recurrence)",
        "surveillance": "Annual ophthalmology (slit-lamp); dental/perio review; PFTs (tracheobronchial); family cascade AR testing",
        "targeted_rx": "Ryplazim (Lys-plasminogen IV/topical) FDA-approved 2021; DOAC/LMWH for VTE; prophylactic LMWH surgical/pregnant",
    },
    {
        "gene": "FGG",
        "protein": (
            "FGG -- 4q32.1 AD LOF GOF -- 453aa -- "
            "Fibrinogen-Gamma-Thrombophilic-Dysfibrinogenemia-"
            "R301C-Fibrinogen-Bicetre-I-Fibrin-Polymerization-Defect-"
            "Thrombin-Clotting-Time-Prolonged-PATHOGNOMONIC-"
            "FGG-OMIM-134850"
        ),
        "locus": "4q32.1",
        "protein_size": (
            "453 aa / ~47 kDa / 4q32.1 FGG Fibrinogen Gamma chain molecular context: "
            "STRUCTURE: "
            "  453 aa / ~47 kDa; Fibrinogen Gamma chain; one of three chains of hexameric fibrinogen (αAβBγ)₂; "
            "  N-terminal coiled-coil region (aa 1-136): links to central E-domain; disulfide bonds with Aα/Bβ; "
            "  Beta/gamma homology C-terminal domain (D-domain, aa 137-453): fibrin polymerisation site; "
            "  Fibrin polymerisation: thrombin cleaves fibrinopeptide A (FpA) from Aα → knobs 'A' exposed; "
            "    knobs 'A' bind holes 'a' in D-domain of adjacent fibrin molecule (hole 'a' = gamma chain 364-411); "
            "  Gamma chain hole 'a' (key residues: Asp364-Tyr374-Thr374) = primary fibrin polymerisation site; "
            "  Factor XIIIa crosslinking: gamma chain Lys406-Gln398/399 crosslinks → clot stability; "
            "THROMBOPHILIC DYSFIBRINOGENEMIA (FGG R301C = Fibrinogen Bicêtre I): "
            "  FGG p.Arg301Cys (R301C): most common thrombophilic dysfibrinogenemia variant; "
            "  Unpaired Cys301 forms aberrant disulfide bonds → fibrin network abnormal; "
            "  Fibrin clot: impaired plasmin degradation → delayed fibrinolysis; "
            "  Both arterial AND venous thrombosis PARADOX → risk for DVT/PE AND MI/stroke; "
            "THROMBIN CLOTTING TIME (TCT): "
            "  TCT: measures time for thrombin to clot plasma; NORMAL 14-20 sec; "
            "  Dysfibrinogenemia: TCT PROLONGED (abnormal fibrin polymerisation) PATHOGNOMONIC; "
            "  Fibrinogen level: NORMAL quantitatively (Clauss functional = DECREASED vs. immunological NORMAL); "
            "  Functional/immunological ratio <0.7 PATHOGNOMONIC for dysfibrinogenemia; "
            "  Critical DDx: afibrinogenemia (quantitative) vs dysfibrinogenemia (qualitative); "
            "  Reptilase time: prolonged in dysfibrinogenemia (thrombin-independent clotting); "
            "PARADOX OF DYSFIBRINOGENEMIA: "
            "  Same mutation can cause THROMBOSIS or BLEEDING depending on which polymerisation step affected; "
            "  Thrombophilic: fibrin clots resist fibrinolysis (PLG cannot degrade abnormal fibrin); "
            "  Haemorrhagic: fibrin weak polymerisation → poor clot strength; "
            "  Asymptomatic: ~50% of dysfibrinogenemia mutations have NO clinical phenotype; "
            "TESTING: "
            "  TCT (thrombin clotting time): prolonged diagnostic screen; "
            "  Reptilase time: prolonged (thrombin-independent; confirms fibrinogen abnormality); "
            "  Fibrinogen Clauss (functional) vs immunological: ratio <0.7 PATHOGNOMONIC; "
            "  FGA/FGB/FGG sequencing: gene panel; most dysfibrinogenemias NOT on standard coag panel; "
            "  D-dimer: may be elevated (abnormal fibrin turnover); "
            "TREATMENT: "
            "  Thrombophilic variant (R301C and similar): DOAC or warfarin for VTE; "
            "  Fibrinogen concentrate: for bleeding dysfibrinogenemia (NOT for thrombophilic); "
            "  Anticoagulation duration: individualised; recurrent VTE → extended; "
            "  Avoid: fibrinogen concentrate in thrombophilic dysfibrinogenemia (adds abnormal substrate); "
            "  LMWH prophylaxis: surgical/pregnancy"
        ),
        "syndrome": "Thrombophilic-Dysfibrinogenemia / FGG-R301C-Fibrinogen-Bicetre-I / Fibrin-Polymerisation-Defect",
        "inheritance": "AD heterozygous (GOF thrombophilic; same variants can be LOF haemorrhagic depending on allele)",
        "thrombosis_risk": "ARTERIAL+VENOUS paradox 2-5x both; fibrinolysis resistant; recurrent DVT/PE/MI/stroke",
        "pathognomonic": "TCT prolonged + Fibrinogen functional/immunological ratio <0.7 + normal fibrinogen antigen = PATHOGNOMONIC dysfibrinogenemia",
        "key_rule": "TCT prolonged in dysfibrinogenemia is PATHOGNOMONIC; standard PT/APTT normal; FGG sequencing confirms thrombophilic variant",
        "key_avoid": "Fibrinogen concentrate in THROMBOPHILIC dysfibrinogenemia (adds more abnormal substrate; worsens thrombosis); reptilase time confirms if heparin contamination suspected",
        "surveillance": "Annual coag review; arterial AND venous thrombosis monitoring; family cascade; TCT + Clauss ratio",
        "targeted_rx": "DOAC or warfarin; aspirin for arterial component; LMWH peri-operative/pregnancy; NO fibrinogen concentrate in thrombophilic dysfibrinogenemia",
    },
]

KEY_CLINICAL_DISTINCTIONS = [
    "F5-FVL: most-common-hereditary-thrombophilia-5pct-Europeans; APC-resistance; OCP+FVL MULTIPLICATIVE-30-35x-NOT-additive; APCR-ratio-screen",
    "F2-G20210A: 2nd-most-common-1-3pct-Europeans; NOT-detectable-standard-PT-APTT; plasma-prothrombin-elevated-130pct; CVST-enriched",
    "PROC: AR-homozygous → Neonatal-Purpura-Fulminans-within-72h-PATHOGNOMONIC; Warfarin-skin-necrosis-PATHOGNOMONIC-heterozygous; ALWAYS-LMWH-bridge-warfarin",
    "PROS1: PROS2-pseudogene-WES-MANDATORY-Sanger-false-positives; NEVER-diagnose-in-pregnancy-or-OCP; free-total-functional-three-types",
    "SERPINC1: most-severe-hereditary-thrombophilia; Type-II-HBS → Heparin-resistance-PATHOGNOMONIC → AT-concentrate-MANDATORY; NEVER-escalate-UFH-blindly",
    "THBD: ARTERIAL-thrombosis-young-MI-stroke; coronary-angiogram-NORMAL PATHOGNOMONIC; standard-coag-screen-NORMAL; Protein-C-level-NORMAL",
    "PLG: ligneous-conjunctivitis-PATHOGNOMONIC; tranexamic-acid-CONTRAINDICATED-ABSOLUTELY (worsens-ligneous); Ryplazim-FDA2021-approved",
    "FGG: thrombophilic-dysfibrinogenemia; TCT-prolonged-PATHOGNOMONIC; Clauss-functional/immunological-ratio-less-0.7; fibrinogen-concentrate-CI-in-thrombophilic-variant",
    "THROMBOPHILIA-HIERARCHY-SEVERITY: SERPINC1-Type-I > PROC-homozygous > PROS1-Type-I > SERPINC1-Type-II > F5-homozygous > PROC-heterozygous > PROS1 > F5-heterozygous ≈ F2",
    "HEPARIN-MONITORING: AT-deficiency-Type-II-HBS → APTT-non-responsive → switch-argatroban-or-AT-concentrate; FVL/F2/PROC/PROS1 all respond-to-heparin-normally",
    "WARFARIN-WINDOW: PROC+PROS1 deficiency → warfarin-skin-necrosis-risk → ALWAYS-LMWH-bridge-minimum-5-days-INR>2; DOAC-avoids-this-entirely",
    "TESTING-TIMING: test-AT+PC+PS OFF-heparin OFF-warfarin OFF-OCP OFF-acute-VTE; F5-F2-genetics-can-test-anytime",
    "PREGNANCY-MANAGEMENT: F5-prior-VTE → LMWH; F2-prior-VTE → LMWH; PROC-AT-PROS1 → high-risk LMWH+monitoring; PLG → LMWH+plasminogen-cover; PROS1-PS-falls-50pct-physiologically",
    "DOAC-PREFERENCE: all-8-genes → DOAC-preferred-over-warfarin (no-PC-window-risk PROC/PROS1; no-monitoring; effective); warfarin-only-if-DOAC-CI-or-mechanical-valve",
]


def _make_cohort(gene_idx: int, seed: int) -> dict:
    rng = random.Random(seed)
    gene = ATLAS_GENES[gene_idx]
    n = 40

    ages = [rng.randint(22, 68) for _ in range(n)]

    # VTE events per gene
    vte_rates = {
        "F5":       0.70,  # heterozygous (some homozygous in cohort)
        "F2":       0.60,
        "PROC":     0.65,
        "PROS1":    0.62,
        "SERPINC1": 0.80,  # most severe
        "THBD":     0.55,  # arterial dominant
        "PLG":      0.50,  # fibrinolysis absent
        "FGG":      0.58,
    }
    dvt_pe = sum(1 for _ in range(n) if rng.random() < vte_rates.get(gene["gene"], 0.6))

    # Arterial events
    arterial_rates = {
        "F5":       0.12,
        "F2":       0.15,
        "PROC":     0.10,
        "PROS1":    0.12,
        "SERPINC1": 0.18,
        "THBD":     0.55,  # primarily arterial
        "PLG":      0.15,
        "FGG":      0.38,  # both arterial + venous
    }
    arterial_events = sum(1 for _ in range(n) if rng.random() < arterial_rates.get(gene["gene"], 0.12))

    # Recurrent VTE
    recurrent_vte = sum(1 for _ in range(n) if rng.random() < (vte_rates.get(gene["gene"], 0.5) * 0.4))

    # Anticoagulation on at time of cohort
    on_anticoag = sum(1 for _ in range(n) if rng.random() < 0.72)

    # Warfarin skin necrosis (PROC/PROS1 only)
    warfarin_necrosis = 0
    if gene["gene"] in ("PROC", "PROS1"):
        warfarin_necrosis = sum(1 for _ in range(n) if rng.random() < 0.08)

    # Heparin resistance (SERPINC1 Type II-HBS enriched)
    heparin_resistance = 0
    if gene["gene"] == "SERPINC1":
        heparin_resistance = sum(1 for _ in range(n) if rng.random() < 0.25)

    # Ligneous conjunctivitis (PLG only)
    ligneous = 0
    if gene["gene"] == "PLG":
        ligneous = sum(1 for _ in range(n) if rng.random() < 0.70)

    # Family history positive
    family_hx = sum(1 for _ in range(n) if rng.random() < 0.65)

    # AT functional level (SERPINC1 cohort), PC/PS (PROC/PROS1)
    mean_at = round(rng.uniform(48, 72) if gene["gene"] == "SERPINC1" else 95, 1)
    mean_pc = round(rng.uniform(32, 62) if gene["gene"] == "PROC" else 94, 1)
    mean_ps_free = round(rng.uniform(28, 58) if gene["gene"] == "PROS1" else 88, 1)

    return {
        "gene": gene["gene"],
        "locus": gene["locus"],
        "syndrome": gene["syndrome"],
        "n": n,
        "mean_age": round(sum(ages) / n, 1),
        "dvt_pe_count": dvt_pe,
        "arterial_event_count": arterial_events,
        "recurrent_vte_count": recurrent_vte,
        "on_anticoag_count": on_anticoag,
        "warfarin_necrosis_count": warfarin_necrosis,
        "heparin_resistance_count": heparin_resistance,
        "ligneous_count": ligneous,
        "family_hx_count": family_hx,
        "mean_at_pct": mean_at,
        "mean_pc_pct": mean_pc,
        "mean_ps_free_pct": mean_ps_free,
        "thrombosis_risk": gene["thrombosis_risk"],
        "pathognomonic": gene["pathognomonic"],
        "key_rule": gene["key_rule"],
        "key_avoid": gene["key_avoid"],
        "targeted_rx": gene["targeted_rx"],
    }


def generate_overview() -> dict:
    cohorts = [_make_cohort(i, SEED_BASE + i) for i in range(8)]
    total_patients = sum(c["n"] for c in cohorts)
    mean_age_overall = round(sum(c["mean_age"] * c["n"] for c in cohorts) / total_patients, 1)

    total_vte = sum(c["dvt_pe_count"] for c in cohorts)
    total_arterial = sum(c["arterial_event_count"] for c in cohorts)
    total_recurrent = sum(c["recurrent_vte_count"] for c in cohorts)
    total_anticoag = sum(c["on_anticoag_count"] for c in cohorts)
    total_warfarin_necrosis = sum(c["warfarin_necrosis_count"] for c in cohorts)
    total_heparin_resistance = sum(c["heparin_resistance_count"] for c in cohorts)
    total_ligneous = sum(c["ligneous_count"] for c in cohorts)

    per_gene = [
        {
            "gene": c["gene"],
            "locus": c["locus"],
            "n": c["n"],
            "mean_age": c["mean_age"],
            "dvt_pe_pct": round(c["dvt_pe_count"] / 40 * 100),
            "arterial_event_pct": round(c["arterial_event_count"] / 40 * 100),
            "recurrent_vte_pct": round(c["recurrent_vte_count"] / 40 * 100),
            "on_anticoag_pct": round(c["on_anticoag_count"] / 40 * 100),
            "warfarin_necrosis_count": c["warfarin_necrosis_count"],
            "heparin_resistance_count": c["heparin_resistance_count"],
            "ligneous_count": c["ligneous_count"],
            "family_hx_count": c["family_hx_count"],
            "thrombosis_risk": c["thrombosis_risk"],
            "key_rule": c["key_rule"],
            "targeted_rx": c["targeted_rx"],
        }
        for c in cohorts
    ]

    return {
        "atlas": "Hereditary-VTE-Thrombophilia-Predisposition-Atlas",
        "genes": [g["gene"] for g in ATLAS_GENES],
        "n_genes": 8,
        "total_patients": total_patients,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "mean_age_overall": mean_age_overall,
        "clinical_summary": {
            "dvt_pe_total": total_vte,
            "arterial_event_total": total_arterial,
            "recurrent_vte_total": total_recurrent,
            "on_anticoag_total": total_anticoag,
            "warfarin_necrosis_total": total_warfarin_necrosis,
            "heparin_resistance_total": total_heparin_resistance,
            "ligneous_conjunctivitis_total": total_ligneous,
        },
        "key_clinical_distinctions": KEY_CLINICAL_DISTINCTIONS,
        "per_gene": per_gene,
    }


def generate_breakdown() -> dict:
    cohorts = [_make_cohort(i, SEED_BASE + i) for i in range(8)]
    gene_data = {c["gene"]: c for c in cohorts}
    return {
        "atlas": "Hereditary-VTE-Thrombophilia-Predisposition-Atlas",
        "gene_data": gene_data,
        "atlas_genes": [
            {
                "gene": g["gene"],
                "locus": g["locus"],
                "protein": g["protein"],
                "syndrome": g["syndrome"],
                "inheritance": g["inheritance"],
                "thrombosis_risk": g["thrombosis_risk"],
                "pathognomonic": g["pathognomonic"],
                "key_rule": g["key_rule"],
                "key_avoid": g["key_avoid"],
                "surveillance": g["surveillance"],
                "targeted_rx": g["targeted_rx"],
            }
            for g in ATLAS_GENES
        ],
    }


def generate_definitions() -> dict:
    defs = []
    for g in ATLAS_GENES:
        defs.append({
            "term": f"{g['gene']} -- {g['syndrome']}",
            "definition": (
                f"{g['locus']} {g['inheritance']} -- "
                f"{g['thrombosis_risk']} -- "
                f"Pathognomonic: {g['pathognomonic']} -- "
                f"Key rule: {g['key_rule']} -- "
                f"Avoid: {g['key_avoid']} -- "
                f"Surveillance: {g['surveillance']} -- "
                f"Targeted Rx: {g['targeted_rx']}"
            ),
        })
        defs.append({
            "term": f"{g['gene']} molecular size",
            "definition": g["protein_size"],
        })
    defs.append({
        "term": "KEY CLINICAL DISTINCTIONS -- Hereditary VTE Thrombophilia 8-Gene Atlas",
        "definition": " | ".join(KEY_CLINICAL_DISTINCTIONS),
    })
    return {"atlas": "Hereditary-VTE-Thrombophilia-Predisposition-Atlas", "definitions": defs}


if __name__ == "__main__":
    import json
    result = generate_overview()
    print(json.dumps(result, indent=2))
