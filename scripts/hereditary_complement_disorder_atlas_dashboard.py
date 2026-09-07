#!/usr/bin/env python3
"""Hereditary-Complement-Disorder-Atlas — Complete 8-Gene Atlas (aHUS / C3G)
CFH     (Complement Factor H; 1231 aa; 1q31.3; AD/AR;
         Atypical Haemolytic Uraemic Syndrome type 1 (aHUS1) + C3 Glomerulopathy (C3G) + AMD;
         most common complement gene mutated in aHUS (25–30%); also rare CFH-related CNV/deletion;
         SCR domains 19–20 bind C3b on host-cell surfaces — mutations disrupt self-recognition;
         eculizumab/ravulizumab first-line; plasma exchange bridge; HIGH relapse risk post-Tx;
         seed SEED_BASE+0) .
CFI     (Complement Factor I; 583 aa; 4q25; AD/AR;
         Atypical HUS type 2 (~10% aHUS); cleaves C3b + C4b — master regulator of amplification loop;
         CFI LOF → uncontrolled C3 consumption → C3 depletion → secondary immunodeficiency + aHUS;
         C3G (MPGN type) association; eculizumab; C3 levels LOW at presentation distinguishes from CFH;
         seed SEED_BASE+1) .
C3      (Complement Component 3; 1663 aa; 19p13.3; AD/AR;
         C3 GOF/gain-of-function → C3b deposits on host surfaces → aHUS + C3G;
         C3 LOF → recurrent encapsulated-organism infections (Neisseria, pneumococcus);
         C3G hallmark: C3 dominant deposits on electron microscopy in glomerular basement membrane;
         avacopan (oral C5aR1 inhibitor, FDA2021) for ANCA vasculitis but studied in C3G;
         seed SEED_BASE+2) .
CFB     (Complement Factor B; 739 aa; 6p21.33; AD;
         Gain-of-function CFB mutations → alternative pathway over-activation → aHUS;
         Asp279Gly is a recurrent activating mutation; LOF rare (C3G associated);
         CFB forms Mg²⁺-dependent complex with C3b → C3bBb convertase (alternative pathway amplification);
         eculizumab/ravulizumab; factor B inhibitor iptacopan (FDA2023 for PNH — under investigation for aHUS);
         seed SEED_BASE+3) .
CD46    (Membrane Cofactor Protein / MCP; 347 aa; 1q32.2; AD;
         Atypical HUS type 3 (~15% aHUS); cofactor for CFI-mediated cleavage of C3b/C4b on host cells;
         YOUNGEST onset (median 4–8 yr); BEST renal prognosis post-transplant (no recurrence — transplant kidney has normal CD46);
         penetrance ~50%; heterozygous LOF most common; eculizumab if TMA not resolving;
         plasma exchange INEFFECTIVE (membrane protein, not serum);
         seed SEED_BASE+4) .
THBD    (Thrombomodulin; 557 aa; 20p11.21; AD/AR;
         Rare aHUS gene (~5%); thrombomodulin activates protein C + TAFI + cofactor for CFI on EC surface;
         LOF → pro-coagulant state + complement dysregulation → TMA phenotype overlaps CAPS;
         plasma exchange responsive in acute phase; eculizumab for refractory;
         THBD mutations also found in pulmonary arterial hypertension cohorts;
         seed SEED_BASE+5) .
DGKE    (Diacylglycerol Kinase Epsilon; 520 aa; 17q22; AR;
         COMPLEMENT-INDEPENDENT aHUS — eculizumab DOES NOT WORK — CRITICAL CLINICAL PITFALL;
         biallelic LOF → DAG accumulation → endothelial + platelet activation → TMA;
         onset infantile (<2 yr); complement normal; features: nephrotic-range proteinuria + HUS;
         antiproteinuric therapy + supportive; early-onset HUS + normal complement = SEQUENCE DGKE FIRST;
         seed SEED_BASE+6) .
C5      (Complement Component 5; 1676 aa; 9q33.2; AD;
         Direct eculizumab target (anti-C5 monoclonal antibody); rare LOF = aHUS; heterozygous GOF rare;
         Japanese founder: p.Arg885His — eculizumab non-responsive (blocks C5 cleavage at Arg885);
         C5 p.Arg885His = ECULIZUMAB RESISTANT — switch to ravulizumab (different epitope);
         C5 LOW or undetectable + recurrent Neisseria = C5 deficiency — complement terminal lysis absent;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1926–1933)
"""

import random

SEED_BASE = 1926

COMPLEMENT_GENES = [
    # -- CFH — Complement Factor H --------------------------------------------------
    {
        "gene": "CFH",
        "alt_name": "Complement Factor H",
        "protein": (
            "CFH -- 1q31.3 AD/AR -- FactorH-1231aa -- "
            "aHUS-Type1-Most-Common-25-30pct-Complement-Gene -- "
            "C3-Glomerulopathy-C3G-MPGN-Also-CFH -- "
            "AMD-Risk-CFH-Y402H-Polymorphism-Independent-Risk-Factor -- "
            "SCR-Domains-19-20-Bind-C3b-Host-Surface-Self-Recognition-DISRUPTED"
        ),
        "locus": "1q31.3",
        "protein_size": "1231 aa",
        "inheritance": "AD/AR",
        "age_of_onset": (
            "aHUS: any age; children and young adults most common; "
            "C3G: 2nd–4th decade; AMD: 6th–8th decade (CFH Y402H polymorphism)"
        ),
        "key_biomarker": (
            "C3 LOW (consumption by uncontrolled AP); C4 normal (AP-specific); "
            "CFH level reduced in most heterozygous mutations; "
            "ADAMTS13 >10% (distinguishes from TTP); "
            "STEC stool PCR/culture negative (distinguishes from STEC-HUS); "
            "kidney biopsy: C3 staining + thrombi in glomerular capillaries (aHUS) or mesangial/GBM deposits (C3G); "
            "anti-CFH antibodies (CFHR1-CFHR3 deletion risk); "
            "fundus + OCT angiography if AMD family history"
        ),
        "pathognomonic": (
            "Thrombotic microangiopathy (TMA): haemolytic anaemia (Hb <10) + thrombocytopaenia (plt <150) + "
            "AKI (Cr elevation) WITHOUT bloody diarrhoea preceding illness → ATYPICAL HUS; "
            "ADAMTS13 >10% (not TTP); STEC negative; C3 LOW, C4 normal; "
            "renal biopsy: fibrin-rich thrombi in capillary loops; negative ANCA/ANA/APLA; "
            "family history aHUS OR prior TMA → genetic panel including CFH mandatory"
        ),
        "treatment": (
            "Eculizumab (Soliris, anti-C5 mAb, FDA2011) — FIRST-LINE for aHUS; "
            "ravulizumab (Ultomiris, FDA2019) — longer dosing interval (every 8 wk); "
            "BEFORE eculizumab: meningococcal vaccination ×2 (MenACWY + MenB) + penicillin prophylaxis; "
            "plasma exchange (PE) as BRIDGE only — buys time pre-eculizumab; PE not curative for CFH; "
            "fresh frozen plasma (FFP) if PE unavailable; "
            "HIGH relapse rate post-transplant — eculizumab maintenance mandatory; "
            "AVOID transplant without eculizumab cover — allograft loss >80% without; "
            "CFH Y402H AMD: anti-VEGF injections; complement inhibitors in trials"
        ),
        "critical_flags": [
            "CFH-MOST-COMMON-AHUS-GENE: 25-30% of all complement-mediated aHUS; also C3G + AMD (Y402H polymorphism different from disease mutations)",
            "CFH-HIGH-RELAPSE-TRANSPLANT: >80% allograft loss if eculizumab NOT used peri/post-transplant; lifelong eculizumab required",
            "CFH-PE-BRIDGE-NOT-CURE: plasma exchange buys time; does NOT cure aHUS if CFH structural mutation (vs anti-CFH antibodies where PE + immunosuppression works)",
            "CFH-ANTI-CFH-ANTIBODIES: CFHR1-CFHR3 homozygous deletion → anti-CFH IgG → functional CFH deficiency; rituximab + PE + eculizumab",
            "CFH-C4-NORMAL: C3 low + C4 normal = alternative pathway (complement-mediated aHUS); C3+C4 low = classical pathway → different DDx",
            "CFH-MENINGOCOCCAL-MANDATORY: meningococcal vaccination (MenACWY + MenB) BEFORE eculizumab; 1,000-2,000× meningococcal infection risk on eculizumab",
            "CFH-SCR-19-20-MUTATIONS: most disease-causing CFH mutations cluster in SCR19-20 (C-terminal, host-cell surface binding); AMD polymorphism at SCR7 (Y402H)",
            "CFH-GENE-PANEL: CFH + CFH-Related Proteins (CFHR1-5) gene panel — MLPA essential; large deletions/rearrangements 10-15% CFH aHUS",
        ],
        "alias": (
            "CFH (Complement Factor H; 1231 aa; 1q31.3) encodes the major soluble regulator of the alternative complement pathway. "
            "CFH binds C3b via its 20 short consensus repeat (SCR) domains, acting as a cofactor for CFI-mediated C3b cleavage and "
            "a decay accelerator of the alternative-pathway C3 convertase (C3bBb). "
            "SCR domains 19–20 bind polyanions (heparan sulphate, sialic acid) on host-cell surfaces, "
            "enabling CFH to specifically protect self-surfaces while permitting complement activation on pathogens. "
            "LOF mutations (heterozygous or homozygous) → uncontrolled alternative pathway amplification on endothelium → "
            "TMA in kidney → aHUS1 (OMIM #235400). CFH accounts for 25–30% of complement-gene mutations in aHUS. "
            "Additional CFH-associated conditions: C3 Glomerulopathy (C3G/MPGN type 2); "
            "AMD (CFH Y402H rs1061170 polymorphism — 2.5× AMD risk; independent of disease mutations at SCR19–20). "
            "Anti-CFH antibodies (from CFHR1–CFHR3 deletion) cause functional CFH deficiency → managed with rituximab + PE + eculizumab. "
            "Eculizumab (anti-C5 mAb) + ravulizumab are standard of care; "
            "meningococcal vaccination MANDATORY before C5 inhibition (≥2 doses MenACWY + 1 MenB). "
            "Post-transplant: CFH mutations recur in allograft → lifelong eculizumab required; >80% allograft loss without."
        ),
    },

    # -- CFI — Complement Factor I --------------------------------------------------
    {
        "gene": "CFI",
        "alt_name": "Complement Factor I",
        "protein": (
            "CFI -- 4q25 AD/AR -- FactorI-583aa -- "
            "aHUS-Type2-10pct-Complement-Gene -- "
            "CFI-LOF-Uncontrolled-C3-Consumption-C3-Depletion-Secondary-Immunodeficiency -- "
            "C3G-MPGN-Association -- "
            "Serine-Protease-Cleaves-C3b-C4b-Cofactor-CFH-CD46-CR1"
        ),
        "locus": "4q25",
        "protein_size": "583 aa",
        "inheritance": "AD/AR",
        "age_of_onset": (
            "aHUS: childhood/young adults; "
            "C3 depletion secondary immunodeficiency: recurrent infections in childhood; "
            "C3G: 2nd–4th decade"
        ),
        "key_biomarker": (
            "C3 VERY LOW (consumed without CFI cleavage brake); C4 normal (AP); "
            "CFI functional assay reduced; "
            "C3 depletion → low C3 + low total haemolytic complement (CH50); "
            "ADAMTS13 >10%; STEC negative; "
            "recurrent Neisseria meningitidis / pneumococcal infections (C3 depletion blocks opsonisation); "
            "consider CFI in any child with severe unexplained C3 deficiency + HUS"
        ),
        "pathognomonic": (
            "aHUS (TMA triad) + C3 very low + C4 normal + ADAMTS13 >10% + STEC negative; "
            "OR severe C3 deficiency + recurrent encapsulated-organism infections in childhood → sequence CFI; "
            "CFI LOF homozygous: C3 <10% of normal + near-absent haemolytic complement; "
            "distinguishing from CFH: CFI level directly assayed (functional CFI assay distinguishes CFH vs CFI deficiency)"
        ),
        "treatment": (
            "Eculizumab/ravulizumab for aHUS (same protocol as CFH); "
            "plasma exchange less effective than for CFH (CFI is a serine protease — PE can replace transiently); "
            "FFP as bridge; "
            "secondary C3 deficiency → prophylactic antibiotics (penicillin) + pneumococcal + meningococcal + Hib vaccination; "
            "C3 concentrate (investigational) for severe C3 LOF; "
            "eculizumab maintenance post-transplant (recurrence risk)"
        ),
        "critical_flags": [
            "CFI-C3-VERY-LOW: CFI LOF → no cleavage brake → massive C3 consumption; C3 VERY LOW (vs CFH where C3 moderately low); C4 normal",
            "CFI-SECONDARY-IMMUNODEFICIENCY: homozygous CFI → C3 depletion → opsonisation failure → recurrent encapsulated-organism sepsis (meningococcus, pneumococcus, Hib)",
            "CFI-FUNCTIONAL-ASSAY: serum CFI antigen may be normal; FUNCTIONAL assay required to detect LOF mutations (substrate-based cleavage assay)",
            "CFI-PE-TRANSIENTLY-HELPFUL: CFI is a serum protein — PE replaces CFI transiently; NOT a cure for structural LOF mutations",
            "CFI-C3G-LINK: CFI mutations found in C3G (mesangial C3 deposits without Ig) + MPGN type — check renal biopsy electron microscopy for deposits type",
            "CFI-10pct-AHUS: second most common single-gene aHUS cause (~10%); after CFH (25-30%)",
            "CFI-VACCINE-PROPHYLAXIS: C3 deficiency from CFI LOF → prophylactic penicillin/amoxicillin + ALL encapsulated-organism vaccines",
            "CFI-ECULIZUMAB-RECURRENCE: post-transplant recurrence risk similar to CFH; eculizumab maintenance required",
        ],
        "alias": (
            "CFI (Complement Factor I; 583 aa; 4q25) encodes a serine protease that cleaves and inactivates C3b and C4b "
            "in the fluid phase and on cell surfaces, using CFH, CD46 (MCP), or CR1 as cofactors. "
            "CFI is the master brake on both alternative and classical pathway amplification. "
            "Heterozygous LOF mutations cause aHUS2 (~10% of complement-mediated aHUS; OMIM #235400); "
            "homozygous LOF causes severe C3 deficiency with near-complete consumption → secondary immunodeficiency "
            "resembling primary C3 deficiency (recurrent encapsulated-organism infections: meningococcus, pneumococcus, Hib). "
            "CFI deficiency is distinguished from CFH by direct functional assay (serum CFI cleavage activity). "
            "C3G (C3 glomerulopathy / MPGN type 2) is associated with both CFH and CFI mutations — "
            "electron microscopy of kidney biopsy distinguishes from immune-complex MPGN (GBM/mesangial dense deposits, Ig-negative). "
            "Treatment: eculizumab/ravulizumab for aHUS; penicillin prophylaxis + all encapsulated-organism vaccines for severe C3-depleting LOF. "
            "Plasma exchange transiently replaces functional CFI but does not correct structural LOF. "
            "Post-transplant recurrence risk: moderate; eculizumab maintenance recommended for LOF mutations with prior TMA."
        ),
    },

    # -- C3 — Complement Component 3 ------------------------------------------------
    {
        "gene": "C3",
        "alt_name": "Complement Component C3",
        "protein": (
            "C3 -- 19p13.3 AD/AR -- ComplementC3-1663aa -- "
            "C3-GOF-aHUS-Plus-C3G-C3-Dominant-Deposits-EM-PATHOGNOMONIC -- "
            "C3-LOF-Recurrent-Neisseria-Pneumococcus-Hib-Infections-Encapsulated-Organisms -- "
            "Central-Hub-Alternative-Classical-Lectin-Pathway-Convergence -- "
            "Avacopan-C5aR1-Inhibitor-FDA2021-ANCA-Vasculitis-Under-Investigation-C3G"
        ),
        "locus": "19p13.3",
        "protein_size": "1663 aa",
        "inheritance": "AD/AR",
        "age_of_onset": (
            "C3G (GOF): 2nd–3rd decade; haematuria/proteinuria at presentation; "
            "aHUS (GOF): acute onset any age; "
            "LOF (C3 deficiency): childhood recurrent infections"
        ),
        "key_biomarker": (
            "C3G: C3 staining dominant in glomerular mesangium ± GBM on IF; "
            "C3 convertase stabilisation → C3 LOW (consumption); C4 normal (AP); "
            "C3 nephritic factor (C3NeF) antibody (stabilises C3 convertase, often found in sporadic C3G); "
            "EM: intramembranous osmiophilic deposits (dense deposits) in MPGN type 2 / C3G; "
            "C3 LOF: C3 virtually absent (<10%); CH50 undetectable; AP50 undetectable"
        ),
        "pathognomonic": (
            "C3 Glomerulopathy: renal biopsy — C3c staining ≥2+ with no or trace Ig (IgG, IgA, IgM, C1q, C4d) "
            "on immunofluorescence — PATHOGNOMONIC for C3G; "
            "dense deposits disease (DDD/MPGN type 2): osmiophilic sausage-shaped deposits in GBM on EM — PATHOGNOMONIC; "
            "C3 deficiency: serum C3 <10% + CH50 undetectable + recurrent encapsulated-organism sepsis in childhood"
        ),
        "treatment": (
            "C3G: mycophenolate mofetil (first-line for progressive disease); "
            "eculizumab for rapidly progressive or biopsy-proven severe C3G (off-label); "
            "avacopan (C5aR1 oral inhibitor, FDA2021) under investigation in C3G trials; "
            "pegcetacoplan (C3 inhibitor, FDA2021 for PNH) under investigation in C3G; "
            "AVOID: steroids alone (poor evidence); ACE-i/ARB for proteinuria; "
            "aHUS with C3 GOF: eculizumab/ravulizumab; "
            "C3 LOF/deficiency: prophylactic penicillin + all encapsulated-organism vaccines; "
            "plasma C3 infusion (investigational for severe LOF)"
        ),
        "critical_flags": [
            "C3G-BIOPSY-MANDATORY: C3 dominant IF (≥2+ C3, trace Ig) = C3G; DDD has pathognomonic GBM dense deposits on EM; must distinguish from immune-complex MPGN",
            "C3-GOF-ALSO-AHUS: gain-of-function C3 mutations cause both C3G AND aHUS; aHUS may precede or follow C3G",
            "C3-LOF-ENCAPSULATED-INFECTIONS: C3 deficiency blocks opsonisation → Neisseria meningitidis, S. pneumoniae, H. influenzae sepsis; vaccinate + penicillin prophylaxis",
            "C3NEF-NOT-GENETIC: C3 nephritic factor is an IgG autoantibody (NOT a gene mutation) that stabilises C3 convertase; found in sporadic/acquired C3G; rituximab responsive",
            "C3-AVACOPAN-PIPELINE: avacopan (C5aR1 inhibitor, FDA2021 for ANCA) under C3G RCTs — not yet approved for C3G; monitor trial results",
            "C3-CENTRAL-HUB: all three complement pathways (classical, alternative, lectin) converge on C3; C3 is the most important complement protein to measure",
            "C3-PEGCETACOPLAN-PIPELINE: pegcetacoplan (C3/C3b inhibitor, FDA2021 for PNH) in C3G trials — proximal inhibition preserves some opsonisation",
            "C3G-MPGN-DDX: MPGN type 1 = IC-mediated (C3+Ig deposits); MPGN type 2/DDD = C3G (C3 dominant, dense deposits); C3GN = C3G without dense deposits; always EM",
        ],
        "alias": (
            "C3 (Complement Component 3; 1663 aa; 19p13.3) encodes the central effector of all three complement pathways. "
            "C3 is cleaved by C3 convertases (C4b2a for classical/lectin; C3bBb for alternative) into C3a (anaphylatoxin) "
            "and C3b (opsonin + surface deposition). C3b amplifies the alternative pathway and feeds the terminal pathway. "
            "Rare heterozygous GOF mutations stabilise C3 convertases → continuous C3 consumption → "
            "C3G (C3 glomerulopathy, OMIM #614809; characterised by C3-dominant glomerular deposits without Ig) "
            "and/or aHUS. Homozygous LOF causes profound C3 deficiency (OMIM #237550) — "
            "opsonisation failure → recurrent meningococcal/pneumococcal/Hib sepsis; CH50 undetectable. "
            "C3 Nephritic Factor (C3NeF): acquired IgG autoantibody (not genetic) stabilising the alternative C3 convertase → "
            "acquired C3 consumption → sporadic C3G; managed with rituximab/eculizumab. "
            "Dense Deposit Disease (DDD/MPGN type 2): osmiophilic intramembranous deposits on EM — pathognomonic; "
            "C3G without dense deposits = C3GN. Emerging targeted therapies: avacopan (C5aR1, FDA2021 ANCA), "
            "pegcetacoplan (C3 inhibitor, FDA2021 PNH) — both in C3G RCTs."
        ),
    },

    # -- CFB — Complement Factor B --------------------------------------------------
    {
        "gene": "CFB",
        "alt_name": "Complement Factor B",
        "protein": (
            "CFB -- 6p21.33 AD -- FactorB-739aa -- "
            "GOF-Gain-of-Function-aHUS-Alternative-Pathway-Overactivation -- "
            "CFB-Asp279Gly-Recurrent-Activating-Mutation -- "
            "Serine-Protease-C3bBb-Alternative-C3-Convertase -- "
            "Iptacopan-Oral-Factor-B-Inhibitor-FDA2023-PNH-Investigated-aHUS"
        ),
        "locus": "6p21.33",
        "protein_size": "739 aa",
        "inheritance": "AD",
        "age_of_onset": (
            "aHUS (GOF): any age, often childhood; "
            "C3G (LOF variant association): 2nd–4th decade"
        ),
        "key_biomarker": (
            "C3 LOW (AP amplification); C4 normal; "
            "CFB functional assay or CFB antigen may be elevated (GOF — stabilised convertase); "
            "AP50 very low/undetectable; "
            "no anti-CFH or anti-CFB antibodies (distinguishes from acquired); "
            "genetic CFB testing: Asp279Gly (c.836A>G) most common GOF mutation"
        ),
        "pathognomonic": (
            "aHUS TMA + C3 low + C4 normal + negative STEC + ADAMTS13 >10% + "
            "positive CFB GOF mutation (Asp279Gly) → complement-mediated aHUS via AP over-activation; "
            "complement panel shows hyperactive alternative pathway (AP50 very low with normal CH50 initially); "
            "CFB is on chromosome 6p21.33 (MHC region) — beware copy number variants"
        ),
        "treatment": (
            "Eculizumab/ravulizumab (anti-C5) — standard aHUS first-line; "
            "iptacopan (oral factor B inhibitor, FDA2023 for PNH) — proximal AP inhibitor under investigation for aHUS; "
            "danicopan (oral factor D inhibitor, FDA2024 for PNH) — synergistic with C5 inhibitor; "
            "plasma exchange: less effective than for serum-protein targets (CFB is a serum protein but GOF not correctable); "
            "post-transplant: eculizumab maintenance (recurrence risk moderate)"
        ),
        "critical_flags": [
            "CFB-GOF-NOT-LOF: unlike other complement genes, DISEASE-CAUSING CFB mutations are GAIN-OF-FUNCTION (stabilise C3 convertase); LOF = possible C3G",
            "CFB-ASP279GLY: most common activating CFB mutation; c.836A>G; p.Asp279Gly; creates hyperactive C3bBb convertase resistant to decay",
            "CFB-MHC-REGION: CFB gene at 6p21.33 in MHC locus — CNV and deletion common; MLPA important",
            "CFB-AP50-LOW: alternative pathway-specific test (AP50) very low; CH50 (total complement) may be less affected initially vs CFH",
            "CFB-IPTACOPAN-PIPELINE: iptacopan (LNP023, Novartis) is an oral factor B inhibitor approved for PNH (FDA2023); CFB aHUS trials ongoing — watch results",
            "CFB-ECULIZUMAB-RESPONSIVE: GOF CFB aHUS responds to eculizumab (downstream C5 blockade); proximal inhibition (iptacopan, danicopan) also rational",
            "CFB-PLASMA-EXCHANGE-BRIDGE: FFP + PE as bridge while awaiting eculizumab; not curative for GOF mutations",
            "CFB-RARE-5pct: CFB mutations cause ~5% complement-mediated aHUS; always include in gene panel (6p21.33 MHC region)",
        ],
        "alias": (
            "CFB (Complement Factor B; 739 aa; 6p21.33) encodes a serine protease that forms the catalytic subunit "
            "of the alternative pathway C3 convertase (C3bBb). CFB binds to C3b in a Mg²⁺-dependent manner, "
            "is cleaved by factor D into Bb (active serine protease) and Ba (released); "
            "Bb remains complexed with C3b to form the AP C3 convertase. "
            "Gain-of-function CFB mutations (notably p.Asp279Gly / c.836A>G) stabilise C3bBb, "
            "reducing susceptibility to decay by CFH → uncontrolled AP amplification → aHUS (OMIM #612922). "
            "Unlike most complement aHUS genes, pathogenic CFB variants are activating (GOF), not LOF. "
            "CFB LOF variants are associated with C3G. "
            "CFB is located in the MHC class III region (6p21.33) alongside C4A, C4B, and other complement genes; "
            "MLPA is required to detect large deletions/rearrangements. "
            "Treatment: eculizumab/ravulizumab (anti-C5); "
            "proximal inhibitors (iptacopan, factor B inhibitor, FDA2023 for PNH; danicopan, factor D inhibitor, FDA2024) "
            "are being investigated as alternatives/adjuncts in aHUS. "
            "Post-transplant recurrence moderate; eculizumab maintenance recommended with prior TMA."
        ),
    },

    # -- CD46/MCP — Membrane Cofactor Protein ---------------------------------------
    {
        "gene": "CD46",
        "alt_name": "MCP (Membrane Cofactor Protein)",
        "protein": (
            "CD46 -- 1q32.2 AD -- MCP-347aa -- "
            "aHUS-Type3-15pct-Youngest-Onset-Median-4-8yr -- "
            "BEST-Renal-Prognosis-Post-Transplant-No-Recurrence-Transplant-Kidney-Normal-CD46 -- "
            "Plasma-Exchange-INEFFECTIVE-Membrane-Protein-NOT-Serum -- "
            "Cofactor-CFI-Cleavage-C3b-C4b-Host-Cell-Surface-Protection"
        ),
        "locus": "1q32.2",
        "protein_size": "347 aa",
        "inheritance": "AD",
        "age_of_onset": (
            "YOUNGEST onset of all complement aHUS genes: median 4–8 yr; "
            "triggers: upper respiratory infections, gastroenteritis, vaccinations; "
            "multiple episodes (relapsing TMA) more common than in CFH"
        ),
        "key_biomarker": (
            "C3 low (variable — often less depressed than CFH/CFI); C4 normal; "
            "CD46 expression on granulocytes by flow cytometry (REDUCED in LOF); "
            "ADAMTS13 >10%; STEC negative; "
            "complement panel; "
            "genetic testing: heterozygous LOF most common; penetrance ~50%; "
            "MCP flow cytometry on granulocytes: <50% expression vs normal = CD46 defect"
        ),
        "pathognomonic": (
            "Young child (age 4–8 yr) with relapsing TMA triggered by infections/vaccinations + "
            "C3 low (variable) + C4 normal + ADAMTS13 >10% + STEC negative; "
            "MCP expression on granulocytes REDUCED (<50%); "
            "multiple TMA episodes WITHOUT permanent organ damage in intervals = CD46 phenotype; "
            "family history: ~50% penetrance — parent may be asymptomatic despite carrying LOF allele"
        ),
        "treatment": (
            "Eculizumab/ravulizumab for acute TMA (if not resolving with supportive care); "
            "PLASMA EXCHANGE INEFFECTIVE — CD46 is a MEMBRANE protein NOT a serum protein; "
            "PE replaces serum but does NOT correct membrane CD46 deficiency; "
            "TRANSPLANT: BEST outcome — donor kidney has normal CD46 → TMA does NOT recur; "
            "eculizumab NOT required routinely post-transplant (unlike CFH/CFI/C3); "
            "relapse prophylaxis: monitor and treat triggering infections early; "
            "long-term ESRD uncommon if TMA episodes managed; "
            "genetic counselling: 50% penetrance; siblings may be carriers"
        ),
        "critical_flags": [
            "CD46-BEST-TRANSPLANT-OUTCOME: ONLY complement aHUS gene where transplant kidney has normal CD46 → NO recurrence; eculizumab not required post-Tx routinely",
            "CD46-PE-ABSOLUTELY-INEFFECTIVE: plasma exchange does NOT replace a membrane protein; PE is CONTRAINDICATED as treatment (though FFP given diagnostically sometimes)",
            "CD46-YOUNGEST-ONSET: median 4-8 yr (youngest of all complement aHUS genes); relapsing TMA triggered by infections/vaccines",
            "CD46-FLOW-CYTOMETRY: MCP expression on granulocytes by flow cytometry REDUCED (<50% of normal) in LOF — rapid functional test; complements genetic testing",
            "CD46-50pct-PENETRANCE: incomplete penetrance (~50%); heterozygous parent may have NO history of TMA — test siblings and parents",
            "CD46-RELAPSING-NOT-PERMANENT: multiple TMA episodes but renal function often recovers between episodes; different from CFH where single episode can cause ESRD",
            "CD46-ECULIZUMAB-ACUTE-ONLY: eculizumab for acute severe TMA; long-term prophylaxis not mandatory (unlike CFH); each case individualised",
            "CD46-VACCINATION-TRIGGER: TMA episodes often triggered by vaccinations → prophylactic eculizumab peri-vaccination may be considered in severe/frequent relapsers",
        ],
        "alias": (
            "CD46 (Membrane Cofactor Protein/MCP; 347 aa; 1q32.2) encodes a ubiquitously expressed type I "
            "transmembrane glycoprotein that acts as a cofactor for CFI-mediated cleavage of C3b and C4b "
            "on host cell surfaces. CD46 is present on virtually all nucleated cells, "
            "providing surface-specific complement regulation. "
            "Heterozygous LOF mutations cause aHUS type 3 (OMIM #235400), accounting for ~15% of complement-mediated aHUS; "
            "penetrance is ~50%. Onset is the youngest of all complement aHUS genes (median 4–8 yr); "
            "relapsing TMA triggered by infections and vaccinations is typical. "
            "Critical features: (1) plasma exchange is INEFFECTIVE (membrane protein, not serum); "
            "(2) TRANSPLANT has the BEST outcome of all complement aHUS genes — "
            "donor kidney has normal CD46 → TMA does NOT recur; eculizumab not required post-transplant. "
            "CD46 expression on granulocytes measured by flow cytometry provides a rapid functional readout (<50% = LOF). "
            "Eculizumab/ravulizumab for acute TMA episodes; "
            "long-term prophylaxis individualised based on frequency and severity of relapses."
        ),
    },

    # -- THBD — Thrombomodulin ------------------------------------------------------
    {
        "gene": "THBD",
        "alt_name": "Thrombomodulin",
        "protein": (
            "THBD -- 20p11.21 AD/AR -- Thrombomodulin-557aa -- "
            "Rare-aHUS-5pct-Procoagulant-TMA-CAPS-Like -- "
            "Thrombomodulin-Activates-Protein-C-Plus-TAFI-Plus-CFI-Cofactor -- "
            "PAH-Pulmonary-Arterial-Hypertension-THBD-Mutations-Also -- "
            "Plasma-Exchange-May-Respond-Acute-Eculizumab-Refractory"
        ),
        "locus": "20p11.21",
        "protein_size": "557 aa",
        "inheritance": "AD/AR",
        "age_of_onset": (
            "aHUS: any age; often childhood/young adults; "
            "PAH: 3rd–5th decade; "
            "triggers: infections, pregnancy, surgery"
        ),
        "key_biomarker": (
            "C3 may be low (complement dysregulation component); "
            "thrombomodulin antigen REDUCED in plasma; "
            "d-dimer, fibrinogen, PT/PTT (coagulation overlap — pro-coagulant state); "
            "ADAMTS13 >10%; STEC negative; "
            "anti-phospholipid antibodies negative (DDx CAPS); "
            "THBD genetic testing — heterozygous LOF most common"
        ),
        "pathognomonic": (
            "TMA (HUS triad) + complement dysregulation markers + evidence of coagulation system activation "
            "(THBD bridges complement and coagulation); "
            "overlap with catastrophic antiphospholipid syndrome (CAPS) if APLA positive; "
            "THBD aHUS rare — usually diagnosed after CFH, CFI, C3, CFB, CD46 excluded; "
            "PAH with family history TMA → consider THBD in gene panel"
        ),
        "treatment": (
            "Eculizumab/ravulizumab for complement-mediated TMA component; "
            "plasma exchange responsive in acute phase (THBD is a membrane protein but circulating form exists); "
            "anticoagulation (heparin → warfarin/LMWH) for coagulation activation component; "
            "fresh frozen plasma (FFP): provides both complement factors and THBD; "
            "immunosuppression if overlap with CAPS suspected; "
            "PAH: standard PAH therapy (prostacyclins, PDE5 inhibitors, ERA); "
            "eculizumab post-transplant if prior TMA; "
            "genetic counselling: THBD mutations also found in PAH — cardiac screening mandatory"
        ),
        "critical_flags": [
            "THBD-COMPLEMENT-COAGULATION-BRIDGE: thrombomodulin bridges complement (CFI cofactor for C3b/C4b cleavage on EC surface) AND coagulation (protein C activation + TAFI); TMA has both components",
            "THBD-RARE-5pct: ~5% of complement-mediated aHUS; always include in panel after more common genes excluded",
            "THBD-PAH-OVERLAP: THBD mutations found in pulmonary arterial hypertension (PAH) cohorts — cardiac echo + RHC if THBD mutation found",
            "THBD-APLA-DDX: THBD TMA may phenotypically mimic CAPS (catastrophic APLA syndrome); check APLA antibodies (aCL, anti-β2GPI, lupus anticoagulant) in ALL TMA",
            "THBD-FFP-DUAL-BENEFIT: FFP provides complement regulators AND THBD — useful bridge; more effective than PE for THBD specifically",
            "THBD-PLASMA-EXCHANGE-MODERATE: PE partially helpful (soluble THBD in plasma replaced); less effective than for CFH (serum-dominant protein)",
            "THBD-ECULIZUMAB-COMPLEMENT-COMPONENT: eculizumab addresses complement arm; anticoagulation addresses coagulation arm — BOTH required in acute severe TMA",
            "THBD-POST-TRANSPLANT: recurrence risk moderate; eculizumab peri-transplant recommended; kidney function often better post-Tx than pure complement genes",
        ],
        "alias": (
            "THBD (Thrombomodulin; 557 aa; 20p11.21) encodes a transmembrane glycoprotein expressed on vascular endothelium "
            "that bridges the complement and coagulation systems. "
            "Thrombomodulin functions as: (1) cofactor for thrombin-mediated protein C activation → anticoagulant; "
            "(2) activator of carboxypeptidase B (TAFI) → fibrinolysis regulation; "
            "(3) cofactor for CFI-mediated C3b/C4b cleavage on endothelial surfaces → complement regulation. "
            "LOF mutations → reduced protein C activation + impaired complement regulation → "
            "pro-coagulant state + complement-mediated TMA → aHUS (~5% of complement-mediated aHUS; OMIM #235400). "
            "The THBD aHUS phenotype overlaps clinically with catastrophic antiphospholipid syndrome (CAPS); "
            "APLA antibody testing is mandatory. "
            "THBD mutations are also found in pulmonary arterial hypertension (PAH) cohorts — "
            "cardiac evaluation mandatory when THBD mutation identified. "
            "Treatment combines complement inhibition (eculizumab), anticoagulation, and supportive care; "
            "plasma exchange + FFP as bridge. Post-transplant recurrence risk: moderate."
        ),
    },

    # -- DGKE — Diacylglycerol Kinase Epsilon ---------------------------------------
    {
        "gene": "DGKE",
        "alt_name": "DGK-epsilon (Diacylglycerol Kinase Epsilon)",
        "protein": (
            "DGKE -- 17q22 AR -- DGKepsilon-520aa -- "
            "COMPLEMENT-INDEPENDENT-aHUS-Eculizumab-DOES-NOT-WORK-CRITICAL-PITFALL -- "
            "Biallelic-LOF-DAG-Accumulation-Endothelial-Platelet-Activation-TMA -- "
            "Infantile-Onset-Before-Age-2yr-ALWAYS -- "
            "Nephrotic-Range-Proteinuria-PLUS-HUS-Together-PATHOGNOMONIC"
        ),
        "locus": "17q22",
        "protein_size": "520 aa",
        "inheritance": "AR",
        "age_of_onset": (
            "INFANTILE — onset ALWAYS before age 2 yr (median 6–12 months); "
            "early-onset HUS + nephrotic-range proteinuria together = DGKE until proven otherwise"
        ),
        "key_biomarker": (
            "COMPLEMENT NORMAL — C3, C4, CFH, CFI, CD46 all NORMAL; "
            "ADAMTS13 >10%; STEC negative; "
            "nephrotic-range proteinuria (distinguishes from other aHUS — massive protein loss); "
            "renal biopsy: TMA + focal segmental glomerulosclerosis (FSGS) OR mesangial proliferation; "
            "DGKE genetic testing: biallelic LOF (homozygous or compound heterozygous); "
            "DAG accumulation measurable in platelets (research tool)"
        ),
        "pathognomonic": (
            "INFANTILE TMA (HUS triad) before age 2 yr + "
            "NORMAL complement (C3, C4, complement factors all normal) + "
            "nephrotic-range proteinuria + STEC negative + ADAMTS13 >10%; "
            "complement-independent = DGKE sequencing mandatory BEFORE eculizumab; "
            "if eculizumab started empirically → no response → CHECK DGKE"
        ),
        "treatment": (
            "ECULIZUMAB DOES NOT WORK — COMPLEMENT-INDEPENDENT; DO NOT use eculizumab for DGKE; "
            "antiproteinuric therapy: ACE-i/ARB — reduces proteinuria + renoprotective; "
            "anticoagulation if TMA severe; "
            "plasma exchange: no established benefit; "
            "aspirin: antiplatelet in acute phase; "
            "supportive: manage AKI with renal replacement therapy; "
            "renal transplant: DGKE recurs in transplant (FSGS + TMA) — risk moderate; "
            "immunosuppression post-transplant: standard; "
            "no approved targeted therapy — DAG pathway inhibitors investigational"
        ),
        "critical_flags": [
            "DGKE-ECULIZUMAB-FAILS: COMPLEMENT-INDEPENDENT TMA — eculizumab is USELESS and should NOT be used; this is a life-threatening clinical pitfall",
            "DGKE-INFANTILE-ONSET-ALWAYS: onset ALWAYS before age 2 yr; early-onset HUS (age <2) + normal complement = SEQUENCE DGKE BEFORE ECULIZUMAB",
            "DGKE-COMPLEMENT-NORMAL: ALL complement parameters normal (C3, C4, CFH, CFI, CD46, factor B, ADAMTS13 >10%) — complement-independent mechanism",
            "DGKE-PROTEINURIA-PATHOGNOMONIC: nephrotic-range proteinuria + HUS together in infant = DGKE phenotype (complement-mediated aHUS does NOT have massive proteinuria)",
            "DGKE-DAG-MECHANISM: DGK-epsilon phosphorylates DAG → PA; LOF → DAG accumulation → PKC activation → endothelial + platelet activation → TMA without complement",
            "DGKE-TRANSPLANT-RECURS: DGKE TMA/FSGS recurs in transplanted kidney (DGKE mutation persists in recipient cells); different from CD46 which does NOT recur",
            "DGKE-BIALLELIC-AR: autosomal recessive; BOTH alleles must be mutated; heterozygous carriers unaffected; recurrence risk 25% per pregnancy",
            "DGKE-ACE-I-ARB-MANDATORY: antiproteinuric therapy is the mainstay of chronic management; aggressive ACE-i/ARB titration for proteinuria reduction",
        ],
        "alias": (
            "DGKE (Diacylglycerol Kinase Epsilon; 520 aa; 17q22) encodes a lipid kinase that converts "
            "diacylglycerol (DAG) to phosphatidic acid (PA), thereby attenuating PKC signalling. "
            "Biallelic LOF mutations cause complement-INDEPENDENT aHUS (OMIM #615008) — "
            "the critical clinical distinction from all other aHUS genes. "
            "DGKE LOF → DAG accumulation → sustained PKC activation → endothelial cell activation + platelet hyper-reactivity → "
            "TMA without complement consumption. "
            "Onset is uniformly infantile (age <2 yr); nephrotic-range proteinuria accompanying HUS is pathognomonic "
            "(complement-mediated aHUS does not produce heavy proteinuria). "
            "Complement parameters are ALL NORMAL — C3, C4, CFH, CFI, CD46, AP50, CH50, ADAMTS13. "
            "ECULIZUMAB DOES NOT WORK and must not be used for DGKE aHUS — "
            "this is a life-saving distinction, as empiric eculizumab wastes time and resources. "
            "Management: ACE-i/ARB (antiproteinuric + renoprotective, mainstay of chronic therapy), "
            "supportive care in acute TMA, renal replacement if AKI severe. "
            "Renal transplantation: DGKE TMA/FSGS can recur in the allograft (unlike CD46 where donor kidney is protected). "
            "Autosomal recessive; biallelic mutations required; 25% sibling recurrence risk."
        ),
    },

    # -- C5 — Complement Component 5 ------------------------------------------------
    {
        "gene": "C5",
        "alt_name": "Complement Component C5",
        "protein": (
            "C5 -- 9q33.2 AD -- ComplementC5-1676aa -- "
            "Direct-Eculizumab-Target-Anti-C5-mAb-Blocks-C5-Cleavage-At-Arg885 -- "
            "Japanese-Founder-p.Arg885His-ECULIZUMAB-RESISTANT-Switch-Ravulizumab -- "
            "C5-LOF-Recurrent-Neisseria-Gonorrhoea-Meningitidis-Absent-Lytic-Complex -- "
            "C5a-Anaphylatoxin-C5b-Membrane-Attack-Complex-MAC-C5b-9"
        ),
        "locus": "9q33.2",
        "protein_size": "1676 aa",
        "inheritance": "AD",
        "age_of_onset": (
            "C5 deficiency (LOF): childhood recurrent Neisseria infections; "
            "Rare GOF aHUS: any age; "
            "Eculizumab resistance (p.Arg885His): identified when eculizumab treatment fails to achieve C5 blockade"
        ),
        "key_biomarker": (
            "C5 LOF: C5 antigen absent/low; CH50 very low/undetectable (MAC cannot form); "
            "AP50 relatively preserved; serum bactericidal activity absent; "
            "Eculizumab monitoring: free C5 level <0.5 μg/mL = adequate blockade; "
            "p.Arg885His mutation: C5 antigen NORMAL but eculizumab does NOT block cleavage at His885; "
            "genetic testing for Arg885 codon in Japanese/East Asian patients before eculizumab initiation"
        ),
        "pathognomonic": (
            "Recurrent Neisseria gonorrhoea + Neisseria meningitidis infections (especially post-adolescence) "
            "with undetectable CH50 = complement terminal pathway deficiency → C5/C6/C7/C8/C9 testing; "
            "eculizumab treatment failure (free C5 remains >0.5 μg/mL despite full dosing) + "
            "Japanese/East Asian ancestry → test for C5 p.Arg885His IMMEDIATELY (12% frequency in Japan)"
        ),
        "treatment": (
            "C5 LOF/deficiency: prophylactic meningococcal + gonococcal-aware screening; "
            "penicillin prophylaxis; MenACWY + MenB vaccination; "
            "Eculizumab resistance (p.Arg885His): SWITCH to ravulizumab (LNP003, different epitope avoids Arg885); "
            "alternatively pozelimab (anti-C5, different epitope, Regeneron) or avacopan (C5aR1 inhibitor, bypasses C5 cleavage); "
            "GOF C5 aHUS: eculizumab if Arg885 NOT mutated; ravulizumab alternative; "
            "meningococcal vaccination MANDATORY before ANY C5 inhibitor therapy; "
            "eculizumab monitoring: weekly free C5 levels first month, then periodically"
        ),
        "critical_flags": [
            "C5-ECULIZUMAB-DIRECT-TARGET: eculizumab (Soliris) and ravulizumab (Ultomiris) are anti-C5 mAbs; C5 p.Arg885His = eculizumab RESISTANT",
            "C5-ARG885HIS-JAPANESE-FOUNDER: p.Arg885His (c.2654G>A) at 12% allele frequency in Japan; eculizumab binding site blocked → C5 cleavage proceeds → TMA NOT controlled; switch ravulizumab or pozelimab",
            "C5-TEST-BEFORE-ECULIZUMAB: in Japanese/East Asian patients, genotype C5 codon 885 BEFORE initiating eculizumab to avoid treatment failure in aHUS/PNH",
            "C5-LOF-NEISSERIA: C5 deficiency = absent MAC (C5b-9) → Neisseria dissemination; recurrent DGI (disseminated gonococcal infection) or meningococcal meningitis = test CH50",
            "C5-FREE-C5-MONITORING: therapeutic eculizumab goal = free C5 <0.5 μg/mL; breakthrough TMA at trough = increase dose/frequency; check for Arg885 variant if persistent elevation",
            "C5-RAVULIZUMAB-LONGER: ravulizumab (anti-C5, different epitope) — every 8-week dosing vs eculizumab every 2 weeks; preferred for long-term therapy",
            "C5A-ANAPHYLATOXIN: C5a drives inflammation (C5aR1/CD88 signalling); avacopan blocks C5aR1 (approved for ANCA vasculitis) — complement TMA trials ongoing",
            "C5-MENINGOCOCCAL-MANDATORY: C5 inhibition abolishes MAC lysis → 1,000-2,000× meningococcal risk; 2 doses MenACWY + 1 MenB + penicillin prophylaxis BEFORE starting eculizumab/ravulizumab",
        ],
        "alias": (
            "C5 (Complement Component 5; 1676 aa; 9q33.2) encodes the central effector of the terminal complement pathway. "
            "C5 is cleaved by C5 convertases (C4b2a3b for classical; C3bBb3b for alternative) into "
            "C5a (potent anaphylatoxin → C5aR1/CD88 signalling → neutrophil chemotaxis, mast cell degranulation) and "
            "C5b (initiates MAC/C5b-9 assembly → cell lysis). "
            "C5 is the direct target of eculizumab (Soliris, anti-C5 mAb, FDA2007 for PNH; FDA2011 for aHUS) and "
            "ravulizumab (Ultomiris, anti-C5 mAb, longer half-life, every 8-wk dosing). "
            "CRITICAL: Japanese/East Asian founder p.Arg885His (c.2654G>A; 12% allele frequency in Japan) "
            "lies at the eculizumab epitope → eculizumab fails to block C5 cleavage → treatment failure in aHUS/PNH; "
            "ravulizumab or pozelimab (different epitopes) effective. "
            "C5 LOF causes terminal complement deficiency (OMIM #609536): absent MAC → "
            "inability to lyse Neisseria → recurrent meningococcal + gonococcal infections; CH50 undetectable. "
            "Therapeutic monitoring: free C5 <0.5 μg/mL confirms adequate blockade; "
            "if free C5 persists elevated on full dosing → check for Arg885 variant. "
            "Meningococcal vaccination (MenACWY × 2 + MenB × 1) MANDATORY before initiating C5 inhibitor. "
            "Avacopan (C5aR1 inhibitor, FDA2021 for ANCA vasculitis): targets C5a arm without blocking MAC — "
            "relevant for conditions where inflammation (C5a) rather than lysis (MAC) drives pathology."
        ),
    },
]


def _make_cohort(gene_data: dict, seed: int) -> list:
    rng = random.Random(seed)
    gene = gene_data["gene"]
    cohort = []
    for i in range(40):
        age = rng.randint(2, 65)
        severity = rng.choice(["mild", "moderate", "severe", "severe"])

        if gene == "CFH":
            onset_type = rng.choice(["aHUS", "C3G", "C3G"])
            feature = rng.choice(["AKI + TMA", "haematuria + proteinuria", "relapsing TMA", "AMD family history", "C3 low + C4 normal"])
            therapy = "eculizumab" if onset_type == "aHUS" else rng.choice(["MMF", "eculizumab", "ACE-i"])
        elif gene == "CFI":
            onset_type = rng.choice(["aHUS", "C3G", "C3 deficiency"])
            feature = rng.choice(["AKI + TMA", "C3 very low + C4 normal", "recurrent meningococcal sepsis", "MPGN pattern biopsy"])
            therapy = "eculizumab" if onset_type in ("aHUS", "C3G") else "penicillin prophylaxis + vaccines"
        elif gene == "C3":
            onset_type = rng.choice(["C3G", "aHUS", "C3 deficiency"])
            feature = rng.choice(["C3 dominant biopsy IF", "dense deposits EM", "haematuria + proteinuria", "C3 very low", "encapsulated-org sepsis"])
            therapy = rng.choice(["MMF", "eculizumab (C3G/aHUS)", "avacopan (trial)", "prophylaxis (LOF)"])
        elif gene == "CFB":
            feature = rng.choice(["AKI + TMA", "C3 low C4 normal", "AP50 very low", "Asp279Gly mutation", "aHUS family history"])
            therapy = "eculizumab"
        elif gene == "CD46":
            age = rng.randint(2, 14)  # youngest onset
            feature = rng.choice(["TMA after URI", "TMA after vaccination", "relapsing HUS", "MCP expression 30% on granulocytes"])
            therapy = rng.choice(["eculizumab (acute)", "renal transplant (curative)", "supportive"])
        elif gene == "THBD":
            feature = rng.choice(["AKI + TMA", "coagulation activation + TMA", "PAH family history", "CAPS-like TMA", "post-partum TMA"])
            therapy = rng.choice(["eculizumab + anticoagulation", "plasma exchange + FFP", "eculizumab"])
        elif gene == "DGKE":
            age = rng.randint(0, 2)  # infantile always
            feature = rng.choice(["infantile HUS + heavy proteinuria", "TMA + FSGS on biopsy", "complement all normal + HUS", "nephrotic + AKI"])
            therapy = "ACE-i/ARB + supportive (NOT eculizumab)"
        elif gene == "C5":
            feature = rng.choice(["eculizumab target", "Arg885His resistance", "recurrent Neisseria", "C5 LOF + absent CH50", "aHUS GOF"])
            therapy = rng.choice(["eculizumab/ravulizumab", "ravulizumab (if Arg885His)", "meningococcal prophylaxis (LOF)"])
        else:
            feature = "complement TMA"
            therapy = "eculizumab"

        cohort.append({
            "patient_id": f"{gene}-{seed}-{i+1:03d}",
            "age": age,
            "gene": gene,
            "severity": severity,
            "key_feature": feature,
            "current_therapy": therapy,
        })
    return cohort


# ---------- API endpoint functions -------------------------------------------

def overview() -> dict:
    total = 0
    severe_count = 0
    avg_age_sum = 0
    gene_summary = []
    for idx, g in enumerate(COMPLEMENT_GENES):
        cohort = _make_cohort(g, SEED_BASE + idx)
        total += len(cohort)
        severe_count += sum(1 for p in cohort if p["severity"] == "severe")
        avg_age_sum += sum(p["age"] for p in cohort)
        gene_summary.append({
            "gene": g["gene"],
            "alt_name": g.get("alt_name", ""),
            "locus": g["locus"],
            "protein_size": g["protein_size"],
            "inheritance": g["inheritance"],
            "n_patients": len(cohort),
        })
    avg_age = round(avg_age_sum / total, 1)
    return {
        "atlas": "Hereditary-Complement-Disorder-Atlas",
        "aggregate_stats": {
            "total_patients": total,
            "genes_covered": len(COMPLEMENT_GENES),
            "avg_age_at_diagnosis_yr": avg_age,
            "severe_cases_pct": round(100 * severe_count / total, 1),
            "seed_range": f"{SEED_BASE}–{SEED_BASE + len(COMPLEMENT_GENES) - 1}",
        },
        "gene_summary": gene_summary,
        "key_clinical_distinctions": [
            "DGKE-ECULIZUMAB-FAILS: complement-INDEPENDENT aHUS; ALL complement parameters normal; eculizumab USELESS — DO NOT USE; sequence DGKE in any infant <2yr with HUS + normal complement",
            "CD46-BEST-TRANSPLANT: ONLY gene where donor kidney has normal CD46 → NO post-transplant recurrence; eculizumab NOT required post-Tx",
            "CD46-PE-INEFFECTIVE: plasma exchange DOES NOT WORK for CD46 (membrane protein); PE is CONTRAINDICATED as aHUS treatment",
            "CFH-HIGH-RELAPSE-TRANSPLANT: >80% allograft loss without eculizumab cover; lifelong eculizumab mandatory post-Tx for CFH",
            "C5-ARG885HIS-ECULIZUMAB-RESISTANT: Japanese/East Asian founder variant at eculizumab epitope → treatment FAILS; switch ravulizumab or pozelimab",
            "MENINGOCOCCAL-MANDATORY-BEFORE-ECULIZUMAB: 1,000–2,000× meningococcal risk on C5 inhibitors; MenACWY × 2 + MenB × 1 + penicillin prophylaxis BEFORE starting",
            "CFI-C3-VERY-LOW: CFI LOF → most severe C3 consumption; secondary immunodeficiency (encapsulated-organism sepsis) if homozygous",
            "CFH-ANTI-CFH-ANTIBODIES: CFHR1-CFHR3 deletion → anti-CFH IgG → acquired functional CFH deficiency; rituximab + PE + eculizumab",
            "THBD-CAPS-DDX: THBD TMA overlaps CAPS; APLA antibodies mandatory in all TMA workup",
            "C3G-BIOPSY-MANDATORY: C3-dominant IF (no Ig) = C3G; DDD = dense deposits on EM PATHOGNOMONIC; EM required for C3G classification",
        ],
    }


def breakdown() -> dict:
    result = []
    for idx, g in enumerate(COMPLEMENT_GENES):
        cohort = _make_cohort(g, SEED_BASE + idx)
        severities = {}
        for p in cohort:
            severities[p["severity"]] = severities.get(p["severity"], 0) + 1
        result.append({
            "gene": g["gene"],
            "alt_name": g.get("alt_name", ""),
            "protein": g["protein"],
            "locus": g["locus"],
            "protein_size": g["protein_size"],
            "inheritance": g["inheritance"],
            "age_of_onset": g["age_of_onset"],
            "key_biomarker": g["key_biomarker"],
            "pathognomonic": g["pathognomonic"],
            "treatment": g["treatment"],
            "critical_flags": g["critical_flags"],
            "severity_distribution": severities,
            "n_patients": len(cohort),
            "patients": cohort[:5],
        })
    return {"genes": result, "total_genes": len(COMPLEMENT_GENES)}


def definitions() -> dict:
    return {
        "atlas": "Hereditary-Complement-Disorder-Atlas",
        "genes": [
            {
                "gene": g["gene"],
                "alt_name": g.get("alt_name", ""),
                "definition": g["alias"],
                "locus": g["locus"],
                "protein_size": g["protein_size"],
                "inheritance": g["inheritance"],
                "age_of_onset": g["age_of_onset"],
                "critical_flags": g["critical_flags"],
            }
            for g in COMPLEMENT_GENES
        ],
        "glossary": {
            "aHUS (Atypical Haemolytic Uraemic Syndrome)": (
                "TMA triad: microangiopathic haemolytic anaemia + thrombocytopenia + AKI WITHOUT preceding bloody diarrhoea; "
                "ADAMTS13 >10% (not TTP); STEC negative; due to complement dysregulation (CFH, CFI, C3, CFB, CD46, THBD, C5 GOF) "
                "or complement-independent (DGKE); eculizumab standard of care for complement-mediated"
            ),
            "C3 Glomerulopathy (C3G)": (
                "Glomerular disease with C3-dominant (≥2+) deposits on IF without Ig; "
                "two subtypes: C3GN (mesangial/subendothelial deposits) and DDD/MPGN type 2 (dense intramembranous deposits on EM); "
                "caused by complement dysregulation (CFH, CFI, C3 mutations) or acquired C3NeF autoantibody; "
                "EM mandatory for classification; MMF + eculizumab for progressive disease"
            ),
            "Eculizumab (Soliris)": (
                "Anti-C5 monoclonal antibody; FDA2007 (PNH) + FDA2011 (aHUS); "
                "blocks C5 cleavage → no C5a or MAC; meningococcal vaccination mandatory before use; "
                "fails in C5 p.Arg885His (Japanese founder) → switch ravulizumab; "
                "given IV every 2 weeks (loading then maintenance)"
            ),
            "Ravulizumab (Ultomiris)": (
                "Next-generation anti-C5 mAb; different epitope from eculizumab; "
                "extended half-life → every 8-week dosing; FDA2018 (PNH) + FDA2019 (aHUS); "
                "ACTIVE against C5 p.Arg885His (eculizumab-resistant variant); "
                "preferred for long-term maintenance over eculizumab (less frequent infusions)"
            ),
            "TMA (Thrombotic Microangiopathy)": (
                "Endothelial injury → platelet microthrombi in small vessels → "
                "haemolysis (schistocytes on smear, low haptoglobin, elevated LDH) + thrombocytopenia + organ damage; "
                "three main syndromes: TTP (ADAMTS13 <10%), STEC-HUS (bloody diarrhoea), aHUS (complement/DGKE)"
            ),
            "Complement Alternative Pathway (AP)": (
                "Spontaneous C3 hydrolysis ('tick-over') → C3bBb convertase → amplification loop; "
                "regulated by CFH, CFI, CD46, properdin; "
                "dysregulation (LOF regulators OR GOF activators) → uncontrolled C3 consumption → endothelial TMA; "
                "AP-specific test: AP50 (haemolysis of rabbit RBCs)"
            ),
            "Plasma Exchange (PE)": (
                "Removes pathological proteins (anti-CFH antibodies, abnormal complement proteins) and replaces with FFP; "
                "effective for: anti-CFH antibody aHUS (acquired), TTP; "
                "INEFFECTIVE for: membrane protein mutations (CD46), pure complement structural LOF not associated with serum protein; "
                "bridge therapy while awaiting eculizumab in acute aHUS"
            ),
            "Meningococcal Vaccine Protocol": (
                "BEFORE eculizumab/ravulizumab: MenACWY (quadrivalent) × 2 doses (0 + 2 months) + MenB × 1–2 doses; "
                "penicillin V 250–500 mg twice daily ongoing prophylaxis; "
                "annual MenACWY booster; "
                "if cannot wait for vaccination (acute aHUS): start penicillin + vaccine simultaneously with eculizumab"
            ),
            "C5 p.Arg885His (Eculizumab-Resistant Variant)": (
                "Japanese/East Asian founder variant; c.2654G>A; allele frequency 12% in Japan; "
                "lies within eculizumab epitope → eculizumab does NOT block C5 cleavage at His885 → treatment failure; "
                "ravulizumab (different epitope) OR pozelimab EFFECTIVE; "
                "genotype C5 codon 885 in East Asian patients BEFORE starting eculizumab"
            ),
            "DGKE aHUS": (
                "Complement-INDEPENDENT TMA; AR biallelic DGKE LOF; "
                "DAG accumulation → PKC → endothelial + platelet activation → TMA WITHOUT complement consumption; "
                "all complement parameters NORMAL; onset <2 yr ALWAYS; heavy proteinuria + HUS; "
                "ECULIZUMAB DOES NOT WORK — do NOT use; ACE-i/ARB is mainstay of chronic management"
            ),
            "Dense Deposit Disease (DDD/MPGN type 2)": (
                "C3G subtype; osmiophilic highly electron-dense sausage-shaped deposits in glomerular GBM on EM — PATHOGNOMONIC; "
                "associated with C3NeF (acquired) and CFH mutations (genetic); "
                "C3 dominant on IF, minimal Ig; "
                "EM mandatory (GBM deposits visible only on EM, not LM); "
                "MPGN type 1 (IC deposits) must be excluded"
            ),
            "CFH-Related Proteins (CFHR1–5)": (
                "Five proteins encoded near CFH on 1q31.3 (CFHR1, CFHR2, CFHR3, CFHR4, CFHR5); "
                "CFHR1 and CFHR3 compete with CFH for C3b binding; "
                "CFHR1–CFHR3 homozygous deletion → anti-CFH IgG autoantibodies → acquired functional CFH deficiency; "
                "MLPA of CFH locus mandatory; rearrangements create hybrid CFH/CFHR genes"
            ),
        },
    }


if __name__ == "__main__":
    import json
    print("=== HEREDITARY-COMPLEMENT-DISORDER-ATLAS — OVERVIEW ===")
    print(json.dumps(overview(), indent=2)[:3000])
    print("\n=== BREAKDOWN (DGKE — complement-independent) ===")
    bd = breakdown()
    dgke = next(g for g in bd["genes"] if g["gene"] == "DGKE")
    print(json.dumps(dgke, indent=2)[:2000])
    print("\n=== DEFINITIONS (glossary sample) ===")
    df = definitions()
    print(json.dumps(list(df["glossary"].items())[:5], indent=2))
