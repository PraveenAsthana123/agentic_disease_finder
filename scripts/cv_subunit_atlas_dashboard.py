#!/usr/bin/env python3
"""CV-Subunit-Atlas — Complete 16-Gene Nuclear-Encoded Complex V (F1F0-ATP Synthase) Atlas
5 F1 structural subunits + 8 F0 structural subunits + 3 assembly factors (all nuclear-encoded)
640-patient aggregate cohort (16 × 40, seeds 744–759)

Complex V (F1F0-ATP Synthase) facts:
  - Catalyses ATP synthesis from ADP + Pi driven by the proton gradient across the IMM
  - F1 (matrix, extrinsic): α₃β₃γδε hexameric knob — catalytic β subunits + regulatory α
  - F0 (membrane, intrinsic): a-subunit (MT-ATP6) + c-ring (ATP5MC1/2/3) + peripheral stalk
  - Peripheral stalk (stator): b (ATP5PB) + d (ATP5PD) + OSCP (ATP5PO) + F6 (ATP5MF)
  - Central stalk (rotor): γ (ATP5F1C) + δ (ATP5F1D) + ε (ATP5F1E) — connects c-ring to F1
  - Rotary catalysis: proton translocation rotates c-ring → γ subunit → drives β conformational change
  - 3 catalytic sites on β subunits: Open (ADP+Pi bound) → Loose (substrate bound) → Tight (ATP formed)
  - Coupling efficiency: ~2.7 ATP per full 360° rotation (mammalian c8-ring stoichiometry)
  - Oligomycin ABSOLUTE CI — binds OSCP (ATP5PO) c-ring interface, blocks H⁺ translocation
  - ATPIF1 (ATP inhibitory factor 1) physiologic inhibitor — prevents reverse ATP hydrolysis
  - CII ALWAYS NORMAL in isolated CV deficiency (CII = nuclear-only internal reference)
  - WES MISSES MT-ATP6, MT-ATP8 — both covered in MT-Genome-Atlas + individual dashboards

ATLAS SCOPE (16 nuclear genes):
  F1 Structural Subunits (5 nuclear):
    ATP5F1A (α), ATP5F1B (β catalytic), ATP5F1C (γ rotary stalk),
    ATP5F1D (δ stator), ATP5F1E (ε stator)
  F0 Structural Subunits (8 nuclear):
    ATP5PO (OSCP/d), ATP5PB (b peripheral stalk),
    ATP5MC1 (c1), ATP5MC2 (c2), ATP5MC3 (c3),
    ATP5PD (d lateral), ATP5ME (e), ATP5MF (f)
  Assembly Factors (3 nuclear):
    TMEM70, ATPAF1, ATPAF2

PHENOTYPIC SPECTRUM (CV Deficiency):
  - Neonatal hypertrophic cardiomyopathy + lactic acidosis (ATP5F1A, TMEM70)
  - 3-Methylglutaconic aciduria (3-MGA) Type V — CV biochemical fingerprint (TMEM70, ATPAF2)
  - Leigh syndrome / Leigh-like (ATP5F1B, TMEM70)
  - F1 assembly failure — isolated CV deficiency without c-ring defect (ATPAF1, ATPAF2)
  - Czech-Slovak Roma founder effect (TMEM70 c.317-2A>G, 1-in-75 carrier in Roma)
  - OSCP (ATP5PO) — Down syndrome dosage + CV super-complex stability

BIOCHEMICAL FINGERPRINT (Isolated CV Deficiency):
  - CV (ATP synthase activity, measured by oligomycin-sensitive ATPase assay) markedly reduced
  - CI, CII, CIII, CIV normal in pure structural/AF mutations
  - CII ALWAYS NORMAL — internal reference (no mtDNA-encoded CII subunits)
  - 3-MGA (3-methylglutaconic acid) elevated in urine — distinctive CV biomarker (60-70%)
  - Lactate/pyruvate ratio elevated; elevated ammonia (hyperammonaemia) in TMEM70
  - BN-PAGE: reduced mature F1F0 complex; accumulated F1 sub-complex in TMEM70

COHORT: 16 × 40 = 640 patient slots (seeds 744–759; gene-specific seeds)
"""

import random

SEED = 760
rng  = random.Random(SEED)

# ── All 16 nuclear-encoded CV-related genes — authoritative table ─────────────
# gene_class: "f1_structural" | "f0_structural" | "assembly_factor"
CV_GENES = [
    # ── F1 Structural Subunits ───────────────────────────────────────────────
    {
        "gene": "ATP5F1A", "alias": "ATP5A1", "aa": "553 aa", "kDa": "59.7 kDa",
        "gene_class": "f1_structural", "subunit_series": "F1-α",
        "cv_module": "F1 non-catalytic α subunit (α₃β₃ hexameric ring); regulatory ADP/ATP binding sites on α; pseudo-catalytic Asp residue non-functional unlike β; α3 scaffolding for β₃ catalytic arrangement; overexpressed in HCM heart; LOF → F1 fails to assemble",
        "omim_gene": 164360, "chromosome": "18q21.1", "seed": 744,
        "phenotype": "CV Deficiency — Neonatal/Infantile Hypertrophic Cardiomyopathy + Lactic Acidosis + Leigh-like (COXPD22 / MC5DN1)",
        "disease": "MC5DN1 / CV deficiency — ATP5F1A; HCM 75%; lactic acidosis; Jonckheere 2012 EMBO Mol Med first nuclear structural CV subunit causing disease; 5-20% residual ATP synthase activity; F1 hexamer fails without α",
        "disease_omim": 618120, "inheritance": "AR",
        "hallmark": "ATP5F1A = FIRST nuclear structural CV subunit identified as disease gene (Jonckheere 2012 EMBO Mol Med); HCM 75% highest among F1 structural subunits; non-catalytic α subunit forms α₃β₃ scaffold; LOF → complete F1 hexamer failure; isolated CV (CI/CII/CIII/CIV normal); NO 3-MGA (distinguishes from TMEM70/ATPAF2)",
        "key_ddx": "vs TMEM70: TMEM70 → 3-MGA + Roma founder; ATP5F1A no 3-MGA, no Roma; vs ATP5F1B: β is catalytic, α regulatory — different roles; vs Pompe (GAA): GSD II enzyme vs CV activity; vs NDUFV2 (CI-HCM 80%): check CI vs CV enzymatically",
        "founder_variant": "p.Gln196His (Jonckheere 2012); p.Arg316Cys; p.Phe277Leu",
        "three_mga_rate": 0.10, "hcm_rate": 0.75, "cardiac_rate": 0.75,
        "leigh_mri_rate": 0.40, "hepatopathy_rate": 0.15, "lactic_ac_rate": 0.85,
        "neuropathy_rate": 0.10, "hyperammonemia_rate": 0.05,
        "atp_activity_mean": 15.0, "atp_activity_sd": 5.0, "median_onset_months": 3,
    },
    {
        "gene": "ATP5F1B", "alias": "ATP5B", "aa": "529 aa", "kDa": "56.5 kDa",
        "gene_class": "f1_structural", "subunit_series": "F1-β",
        "cv_module": "F1 catalytic β subunit (3 copies in α₃β₃ ring); houses the catalytic site (Walker motif B: DXXG); alternating conformations (Open/Loose/Tight) driven by γ rotation; ATP4- synthesis in Tight state; most conserved OXPHOS subunit across evolution (>90% identity bacteria to human)",
        "omim_gene": 102910, "chromosome": "12q13.3", "seed": 745,
        "phenotype": "CV Deficiency — Leigh-like + Psychomotor Regression + 3-Methylglutaconic Aciduria (COXPD-ATP5F1B / MC5DN2)",
        "disease": "MC5DN2 / CV deficiency — ATP5F1B; Leigh-like + 3-MGA; 10-25% residual CV activity; most conserved subunit → mutations extremely rare; β Walker motif B (DXXG) disruption → catalysis abolished; lactic acidosis + developmental regression",
        "disease_omim": 618119, "inheritance": "AR",
        "hallmark": "ATP5F1B = MOST CONSERVED OXPHOS protein (>90% identity bacteria→human — reflects catalytic irreplaceability); Walker motif B DXXG in β = ATP hydrolysis/synthesis site; 3-MGA present in ~40% (secondary); severe isolated CV; psychomotor regression + Leigh-like MRI; NO 3-MGA hallmark unlike TMEM70 (40% not 70%)",
        "key_ddx": "vs ATP5F1A: α regulatory vs β catalytic — both give F1 assembly failure but β (ATP5F1B) more severe; vs TMEM70: TMEM70 3-MGA 70% vs ATP5F1B 40%; vs ATPAF2: ATPAF2 = chaperone for α-β assembly, ATP5F1B = structural β itself; vs Leigh DDx (SURF1-CIV, NDUFS4-CI): check complex enzymatically",
        "founder_variant": "p.Gly169Val (Walker motif); p.Arg408Cys; p.Trp126Stop",
        "three_mga_rate": 0.40, "hcm_rate": 0.30, "cardiac_rate": 0.30,
        "leigh_mri_rate": 0.60, "hepatopathy_rate": 0.20, "lactic_ac_rate": 0.85,
        "neuropathy_rate": 0.15, "hyperammonemia_rate": 0.10,
        "atp_activity_mean": 18.0, "atp_activity_sd": 6.0, "median_onset_months": 6,
    },
    {
        "gene": "ATP5F1C", "alias": "ATP5C1", "aa": "298 aa", "kDa": "33.0 kDa",
        "gene_class": "f1_structural", "subunit_series": "F1-γ",
        "cv_module": "Central rotary stalk γ subunit; asymmetric coiled-coil inserts into α₃β₃ ring; γ rotation by 120° driven by c-ring proton translocation sequentially catalyses 3 β conformational states; 'asymmetric cam' mechanism; direct mechanical coupling between F0 proton channel and F1 catalytic sites",
        "omim_gene": 108729, "chromosome": "10q22.2", "seed": 746,
        "phenotype": "CV Deficiency — Developmental Delay + Hypotonia + Lactic Acidosis (MC5DN3)",
        "disease": "MC5DN3 / CV deficiency — ATP5F1C; γ rotary stalk disruption → uncouples proton gradient from catalysis; 15-30% residual CV; mild-moderate phenotype; psychomotor delay dominant; Dautant 2010 insight on γ mechanism",
        "disease_omim": 618121, "inheritance": "AR",
        "hallmark": "ATP5F1C γ = ROTARY STALK LINKER between F0 c-ring rotation and F1 catalysis; asymmetric coiled-coil — unique 'asymmetric cam' driving sequential β conformational change; disruption → proton gradient present but cannot drive ATP synthesis (uncoupled); mild phenotype vs α/β mutations; developmental delay + hypotonia without HCM",
        "key_ddx": "vs ATP5F1A/B: α/β give severe F1 failure; ATP5F1C gives uncoupled (partially functional); vs ATP5F1D/E: δ/ε also central stalk but different contacts; vs TMEM70: TMEM70 → 3-MGA + HCM + Roma; ATP5F1C → mild no 3-MGA; vs MT-ATP6: mtDNA vs nuclear",
        "founder_variant": "p.Gly144Arg (coiled-coil break); p.Leu84Pro (helix break); p.Arg252Cys",
        "three_mga_rate": 0.15, "hcm_rate": 0.10, "cardiac_rate": 0.15,
        "leigh_mri_rate": 0.30, "hepatopathy_rate": 0.10, "lactic_ac_rate": 0.70,
        "neuropathy_rate": 0.20, "hyperammonemia_rate": 0.05,
        "atp_activity_mean": 25.0, "atp_activity_sd": 7.0, "median_onset_months": 12,
    },
    {
        "gene": "ATP5F1D", "alias": "ATP5D", "aa": "168 aa", "kDa": "19.0 kDa",
        "gene_class": "f1_structural", "subunit_series": "F1-δ",
        "cv_module": "Central stalk δ subunit (rotor); binds γ C-terminus + ε N-terminus; forms central rotor assembly with γε; essential for F1-F0 coupling; mediates transmission of c-ring rotation to γ; δ in mitochondrial CV corresponds to OSCP (confusingly — nomenclature differs from E. coli)",
        "omim_gene": 603249, "chromosome": "19p13.3", "seed": 747,
        "phenotype": "CV Deficiency — Neonatal Lactic Acidosis + Hypotonia + Encephalopathy (MC5DN4)",
        "disease": "MC5DN4 / CV deficiency — ATP5F1D; δ disruption → γε complex destabilised → central rotor assembly fails; 10-20% residual; severe neonatal presentation; lactic acidosis with elevated lactate:pyruvate",
        "disease_omim": 618122, "inheritance": "AR",
        "hallmark": "ATP5F1D δ subunit = CENTRAL ROTOR CONNECTOR (γε-complex stabiliser); note: mitochondrial δ ≠ bacterial δ (naming discrepancy — mammalian δ = central stalk, bacterial δ = peripheral stalk/OSCP equivalent); neonatal severe phenotype; pure isolated CV; CI/CII/CIII/CIV normal; NO oligomycin in ICU (direct CV inhibitor)",
        "key_ddx": "vs ATP5F1C (γ): both central stalk — γ is cam, δ is connector; vs ATPAF1: ATPAF1 chaperones α/β NOT γδε; vs ATP5PO (OSCP): OSCP = F0 peripheral stalk (not same as δ despite nomenclature confusion); vs BTBGD: mandatory exclusion in all neonatal Leigh",
        "founder_variant": "p.Arg31Cys; p.Gly119Val; p.Leu130Pro",
        "three_mga_rate": 0.20, "hcm_rate": 0.25, "cardiac_rate": 0.30,
        "leigh_mri_rate": 0.45, "hepatopathy_rate": 0.20, "lactic_ac_rate": 0.90,
        "neuropathy_rate": 0.10, "hyperammonemia_rate": 0.10,
        "atp_activity_mean": 16.0, "atp_activity_sd": 5.5, "median_onset_months": 2,
    },
    {
        "gene": "ATP5F1E", "alias": "ATP5E", "aa": "51 aa", "kDa": "5.7 kDa",
        "gene_class": "f1_structural", "subunit_series": "F1-ε",
        "cv_module": "Smallest F1 subunit; central stalk ε (rotor); contacts γ C-terminus + c-ring via δ; maintains γδε rotor rigidity; role: suppress F1 ATPase activity in absence of F0 (auto-inhibitory loop) — similar to ATPIF1 function; conserved from bacteria; van Rooyen 2013 first disease report",
        "omim_gene": 606153, "chromosome": "20q13.32", "seed": 748,
        "phenotype": "CV Deficiency — Neonatal Lactic Acidosis + 3-Methylglutaconic Aciduria + Mild Hypertrophic Cardiomyopathy (MC5DN5)",
        "disease": "MC5DN5 / CV deficiency — ATP5F1E; van Rooyen 2013 first report; ε auto-inhibitory loop disrupted → futile ATPase; 15-30% residual CV; 3-MGA in ~45%; mild HCM; lactic acidosis; French-Algerian patient in original report",
        "disease_omim": 618123, "inheritance": "AR",
        "hallmark": "ATP5F1E ε = SMALLEST F1 SUBUNIT (51aa / 5.7kDa); auto-inhibitory loop prevents futile ATP hydrolysis when F0 decoupled; van Rooyen 2013 first ATP5F1E disease; 3-MGA 45% (intermediate between TMEM70-70% and non-3-MGA genes); mild HCM 35%; distinguishes: ε 3-MGA + mild HCM vs α (no 3-MGA + severe HCM) vs TMEM70 (Roma, 3-MGA 70%)",
        "key_ddx": "vs TMEM70: TMEM70 Roma founder, 3-MGA 70%, HCM 65%; ATP5F1E 3-MGA 45%, milder, no Roma; vs ATP5F1A: no 3-MGA, HCM 75% severe vs ε mild 35%; vs MT-ATP6 (NARP): mtDNA maternal vs AR nuclear; vs ATPIF1 (physiologic auto-inhibitor): ATPIF1 dysregulation vs ε structural",
        "founder_variant": "p.His25Arg (van Rooyen 2013 French-Algerian); p.Thr35Pro (auto-inhibitory loop); p.Arg38Gln",
        "three_mga_rate": 0.45, "hcm_rate": 0.35, "cardiac_rate": 0.40,
        "leigh_mri_rate": 0.35, "hepatopathy_rate": 0.15, "lactic_ac_rate": 0.80,
        "neuropathy_rate": 0.10, "hyperammonemia_rate": 0.10,
        "atp_activity_mean": 22.0, "atp_activity_sd": 6.0, "median_onset_months": 4,
    },
    # ── F0 Structural Subunits ───────────────────────────────────────────────
    {
        "gene": "ATP5PO", "alias": "ATP5O / OSCP", "aa": "213 aa", "kDa": "23.3 kDa",
        "gene_class": "f0_structural", "subunit_series": "F0-OSCP",
        "cv_module": "Oligomycin Sensitivity Conferring Protein (OSCP) = F0 peripheral stalk top-cap; bridges F0 b-subunit (ATP5PB) stator to F1 α/δ; oligomycin BINDS at OSCP/c-ring interface (hence 'Oligomycin Sensitivity Conferring'); Chromosome 21 (trisomy → Down syndrome dosage effect on CV); CV super-complex stability anchor",
        "omim_gene": 600828, "chromosome": "21q22.11", "seed": 749,
        "phenotype": "CV Deficiency — Down Syndrome Dosage Effect + CV Supercomplex Destabilisation (Trisomy 21 gene) / Rare AR LOF: encephalopathy",
        "disease": "ATP5PO on chromosome 21 — trisomy 21 (Down syndrome) increases OSCP dosage → CV stoichiometric imbalance → reduced CV super-complex assembly; rare AR LOF: encephalopathy + isolated CV deficiency; OSCP = oligomycin binding partner (blocks c-ring proton channel at OSCP interface)",
        "disease_omim": 600828, "inheritance": "AR (rare LOF) / Trisomy 21 dosage",
        "hallmark": "ATP5PO (OSCP) = OLIGOMYCIN BINDING SITE — oligomycin blocks H⁺ translocation at OSCP/c-ring interface (OSCP is why complex has oligomycin sensitivity); chromosome 21 → Down syndrome dosage effect on CV assembly; OSCP bridges peripheral stalk top to F1 α subunit; CV supercomplex (V₂) requires OSCP; trisomy 21 shows 20-30% reduced CV super-complex (Zamponi 2018)",
        "key_ddx": "vs MT-ATP6 (NARP): MT-ATP6 c-subunit/channel (mtDNA) vs OSCP (nuclear peripheral stalk); vs ATP5PB (b-subunit): ATP5PB is the peripheral stalk b-chain, OSCP is the top cap; vs Down syndrome cognitive decline: multiple chromosome 21 loci; vs oligomycin-treated cells: pharmacologic vs genetic",
        "founder_variant": "p.Arg178Cys (peripheral stalk interface); p.Gly51Val; p.Trp103Stop",
        "three_mga_rate": 0.20, "hcm_rate": 0.20, "cardiac_rate": 0.25,
        "leigh_mri_rate": 0.30, "hepatopathy_rate": 0.10, "lactic_ac_rate": 0.65,
        "neuropathy_rate": 0.20, "hyperammonemia_rate": 0.05,
        "atp_activity_mean": 28.0, "atp_activity_sd": 8.0, "median_onset_months": 18,
    },
    {
        "gene": "ATP5PB", "alias": "ATP5F1 / b-subunit", "aa": "256 aa", "kDa": "29.0 kDa",
        "gene_class": "f0_structural", "subunit_series": "F0-b",
        "cv_module": "Peripheral stalk b subunit (dimerisation partner of b'); single TM helix + long IMS-facing coiled-coil; forms the 'leg' of CV stator connecting F0 a-subunit to OSCP at F1; holds F1 against rotor rotation; CV₂ dimer formation requires b-b' coiled-coil interaction",
        "omim_gene": 602224, "chromosome": "1p13.2", "seed": 750,
        "phenotype": "CV Deficiency — Encephalopathy + Leigh-like + Isolated CV (MC5DN-ATP5PB)",
        "disease": "CV deficiency — ATP5PB; peripheral stalk disruption → F1 cannot be held against rotor → futile rotation; 20-35% residual; lactic acidosis + Leigh-like; ATP5PB mutations rare; CV₂ dimerisation impaired → cristae curvature lost",
        "disease_omim": 618128, "inheritance": "AR",
        "hallmark": "ATP5PB = PERIPHERAL STALK 'LEG' connecting F0 a-subunit to OSCP/F1 top; b-b' coiled-coil in CV₂ dimer also shapes crista membrane curvature (loss → balloon-like cristae on EM); rare disease gene; F0 stator disruption → mechanical futility without uncoupling in FCCP sense; ATP5PB LOF → CV₂ dimer absent on BN-PAGE",
        "key_ddx": "vs ATP5PO (OSCP): OSCP is b-chain top cap, ATP5PB is the b-chain itself; vs ATP5MC1/2/3 (c-ring): c-ring = rotor vs b = stator; vs TMEM70: TMEM70 = c-ring assembly AF vs ATP5PB = structural stator; vs CIV (propofol CI): CIV not CV enzyme affected",
        "founder_variant": "p.Arg153Stop; p.Gly190Val (coiled-coil break); p.Leu37Pro (TM helix)",
        "three_mga_rate": 0.15, "hcm_rate": 0.20, "cardiac_rate": 0.25,
        "leigh_mri_rate": 0.40, "hepatopathy_rate": 0.15, "lactic_ac_rate": 0.75,
        "neuropathy_rate": 0.15, "hyperammonemia_rate": 0.05,
        "atp_activity_mean": 27.0, "atp_activity_sd": 7.5, "median_onset_months": 8,
    },
    {
        "gene": "ATP5MC1", "alias": "ATP5G1", "aa": "136 aa", "kDa": "15.2 kDa",
        "gene_class": "f0_structural", "subunit_series": "F0-c1",
        "cv_module": "c-ring subunit isoform 1 (17q24.2); one of 8 identical c-subunits forming the proton-translocating c8-ring rotor; conserved Asp61 (key protonation residue for H⁺ translocation across a-subunit); c-ring rotation drives γ central stalk → ATP synthesis; TMEM70 is c-ring assembly factor (inserts c into IMM)",
        "omim_gene": 108729, "chromosome": "17q24.2", "seed": 751,
        "phenotype": "CV Deficiency — c-ring assembly failure + Encephalopathy + Lactic Acidosis (MC5DN-ATP5MC1)",
        "disease": "CV deficiency — ATP5MC1; c-ring rotor disruption; three c-subunit isoforms (ATP5MC1/2/3) exist but human disease rare from single isoform given redundancy; 25-40% residual; overlaps with TMEM70-related c-ring phenotype; 3-MGA possible",
        "disease_omim": 618129, "inheritance": "AR",
        "hallmark": "ATP5MC1 c-ring isoform 1 of 3 (ATP5MC1/2/3 encode identical c-subunit protein via different promoters — tissue-specific expression); Asp61 ESSENTIAL protonation residue; TMEM70 chaperones c-ring insertion into IMM; c8-ring stoichiometry drives 2.7 ATP/rotation in mammalian CV; ATP5MC1 mutations relatively rare due to MC2/MC3 redundancy",
        "key_ddx": "vs TMEM70: TMEM70 = c-ring AF (external to c-ring); ATP5MC1 = structural c itself; both give c-ring failure; vs ATP5MC2/3: isoform redundancy means compound heterozygous (cross-isoform) needed for full c-ring failure; vs MT-ATP6: MT-ATP6 = a-subunit (proton channel), ATP5MC1 = c-ring rotor",
        "founder_variant": "p.Asp61Asn (protonation site disruption); p.Gly23Val (TM entry); p.Ala68Val",
        "three_mga_rate": 0.35, "hcm_rate": 0.30, "cardiac_rate": 0.35,
        "leigh_mri_rate": 0.40, "hepatopathy_rate": 0.15, "lactic_ac_rate": 0.75,
        "neuropathy_rate": 0.15, "hyperammonemia_rate": 0.15,
        "atp_activity_mean": 30.0, "atp_activity_sd": 8.0, "median_onset_months": 6,
    },
    {
        "gene": "ATP5MC2", "alias": "ATP5G2", "aa": "146 aa", "kDa": "15.8 kDa",
        "gene_class": "f0_structural", "subunit_series": "F0-c2",
        "cv_module": "c-ring subunit isoform 2 (12q13.3); same essential Asp61 protonation residue as ATP5MC1; different mitochondrial targeting sequence (MTS) / N-terminal presequence; differentially expressed — ATP5MC2 enriched in brain/liver; three isoforms contribute to c8-ring pool in IMM",
        "omim_gene": 106435, "chromosome": "12q13.3", "seed": 752,
        "phenotype": "CV Deficiency — Brain/Liver-Enriched c-ring Isoform — Encephalopathy + Hepatopathy + Lactic Acidosis",
        "disease": "CV deficiency — ATP5MC2; brain/liver isoform → encephalopathy + hepatopathy dominant; 25-40% residual; c-ring composition altered; very rare; phenotype overlap with TMEM70 and ATP5MC1 except tissue distribution",
        "disease_omim": 618130, "inheritance": "AR",
        "hallmark": "ATP5MC2 = BRAIN/LIVER-ENRICHED c-ring isoform (tissue distribution distinguishes from ATP5MC1/3); same Asp61 protonation site as MC1; longer MTS (N-terminal presequence); encephalopathy + hepatopathy prominent (tissue distribution mirrors disease target organs); extremely rare — isoform redundancy again; hepatopathy distinguishes from MC1/MC3",
        "key_ddx": "vs ATP5MC1: MC1 is 17q24.2, MC2 is 12q13.3; identical mature protein but different tissue enrichment; vs TMEM70: same c-ring biology; vs DGUOK (hepatic mtDNA depletion): CII/CI/CIII also reduced vs isolated CV; vs ATP5F1B (β on 12q13.3): same chromosome different gene — ATP5F1B is F1 structural",
        "founder_variant": "p.Asp61Gly (protonation residue); p.Pro115Arg (TM helix 2); p.Ile98Thr",
        "three_mga_rate": 0.35, "hcm_rate": 0.20, "cardiac_rate": 0.25,
        "leigh_mri_rate": 0.45, "hepatopathy_rate": 0.40, "lactic_ac_rate": 0.80,
        "neuropathy_rate": 0.15, "hyperammonemia_rate": 0.20,
        "atp_activity_mean": 29.0, "atp_activity_sd": 7.5, "median_onset_months": 5,
    },
    {
        "gene": "ATP5MC3", "alias": "ATP5G3", "aa": "142 aa", "kDa": "15.6 kDa",
        "gene_class": "f0_structural", "subunit_series": "F0-c3",
        "cv_module": "c-ring subunit isoform 3 (2q31.1); heart/muscle-enriched isoform; same Asp61 protonation residue; shortest MTS of the three isoforms; ATP5MC3 predominant in cardiac tissue → mutations may cause HCM-dominant phenotype; c-ring stoichiometry requires all three isoforms to reach 8 c-subunits",
        "omim_gene": 603392, "chromosome": "2q31.1", "seed": 753,
        "phenotype": "CV Deficiency — Heart/Muscle-Enriched c-ring Isoform — HCM + Myopathy + Lactic Acidosis",
        "disease": "CV deficiency — ATP5MC3; heart/muscle isoform → HCM + myopathy dominant; 20-35% residual; HCM 60% (highest of c-ring isoforms due to cardiac expression); very rare; c-ring pool depleted in heart/muscle",
        "disease_omim": 618131, "inheritance": "AR",
        "hallmark": "ATP5MC3 = HEART/MUSCLE-ENRICHED c-ring isoform (HCM 60% highest of 3 c-ring isoforms); 2q31.1 — different chromosome from MC1 (17q) and MC2 (12q); shortest MTS; cardiac-prominent expression → HCM dominant feature; ATP5MC3 + ATP5MC1/2 together form the c8-ring — loss of MC3 creates c-ring subunit deficit preferentially in heart; myopathy + exercise intolerance",
        "key_ddx": "vs SCO2 (CIV-HCM 100%): check CIV vs CV enzymatically; vs ATP5MC1 (non-cardiac dominant): MC3 → HCM 60%; vs TMEM70 (HCM 65%): TMEM70 Roma founder + 3-MGA 70% vs MC3 no Roma; vs Pompe: GSD II enzyme vs CV; vs NDUFV2 (CI-HCM 80%): CI enzyme vs CV",
        "founder_variant": "p.Gly64Val (TM helix 1); p.Asp61Asn (protonation); p.Leu108Pro (helix break)",
        "three_mga_rate": 0.30, "hcm_rate": 0.60, "cardiac_rate": 0.65,
        "leigh_mri_rate": 0.35, "hepatopathy_rate": 0.10, "lactic_ac_rate": 0.80,
        "neuropathy_rate": 0.10, "hyperammonemia_rate": 0.10,
        "atp_activity_mean": 24.0, "atp_activity_sd": 7.0, "median_onset_months": 4,
    },
    {
        "gene": "ATP5PD", "alias": "ATP5H / d-subunit", "aa": "161 aa", "kDa": "18.5 kDa",
        "gene_class": "f0_structural", "subunit_series": "F0-d",
        "cv_module": "F0 lateral/peripheral stalk d subunit; part of the b-d-OSCP peripheral stalk connecting F0 a-subunit to F1; stabilises the peripheral stalk assembly; d contacts OSCP (ATP5PO) + b (ATP5PB) in the stator; CV₂ dimerisation-associated subunit; on chromosome 17q25.1 near ATP5MC1 (17q24.2)",
        "omim_gene": 603249, "chromosome": "17q25.1", "seed": 754,
        "phenotype": "CV Deficiency — Peripheral Stalk Disruption — Lactic Acidosis + Encephalopathy + Leigh-like",
        "disease": "CV deficiency — ATP5PD; peripheral stalk d-subunit loss → b-d-OSCP stator destabilised; 20-35% residual; Leigh-like MRI; lactic acidosis; very rare; BN-PAGE shows loss of mature CV without sub-complex accumulation (unlike TMEM70)",
        "disease_omim": 618132, "inheritance": "AR",
        "hallmark": "ATP5PD d subunit = STATOR PERIPHERAL STALK component (part of b-d-OSCP trimeric stator); distinct from central stalk δ (ATP5F1D) — confusingly named differently (d vs δ); on 17q25.1 close to ATP5MC1 (17q24.2) — same chromosome cluster; peripheral stalk disruption → F1 released from F0 → uncoupled; pure isolated CV; no 3-MGA; no Roma founder",
        "key_ddx": "vs ATP5F1D (δ): ATP5F1D = central stalk δ (rotor), ATP5PD = F0 d (stator) — different subunits with similar names; vs ATP5PO (OSCP): OSCP is the top cap; ATP5PD is the middle stator element; vs ATP5PB (b): both stator but b is the long coiled-coil, d contacts OSCP at top; vs TMEM70: no Roma, no 3-MGA 70%",
        "founder_variant": "p.Ala120Val (OSCP interface); p.Pro59Leu (stator kink); p.Arg136Stop",
        "three_mga_rate": 0.15, "hcm_rate": 0.20, "cardiac_rate": 0.25,
        "leigh_mri_rate": 0.40, "hepatopathy_rate": 0.15, "lactic_ac_rate": 0.75,
        "neuropathy_rate": 0.15, "hyperammonemia_rate": 0.05,
        "atp_activity_mean": 26.0, "atp_activity_sd": 7.0, "median_onset_months": 9,
    },
    {
        "gene": "ATP5ME", "alias": "ATP5I / e-subunit", "aa": "69 aa", "kDa": "7.8 kDa",
        "gene_class": "f0_structural", "subunit_series": "F0-e",
        "cv_module": "F0 e subunit (supernumerary); single TM helix in IMM; required for CV₂ dimer formation and cristae curvature maintenance; ATP5ME + ATP5MF (f) + ATP5MG (g) = dimerisation module of CV; e/f/g absent in bacterial CV (eukaryote-specific innovation); dimerisation → rows of CV₂ dimers at cristae ridges",
        "omim_gene": 618133, "chromosome": "4p16.3", "seed": 755,
        "phenotype": "CV Deficiency — CV₂ Dimer Failure — Cristae Morphology Defect + Lactic Acidosis + Mild Encephalopathy",
        "disease": "CV deficiency — ATP5ME; e-subunit LOF → CV₂ dimerisation abolished → balloon-like cristae on EM (balloons not ridges); 30-50% residual CV; milder clinical phenotype; lactic acidosis + mild encephalopathy; cristae morphology defect on EM is diagnostic pointer",
        "disease_omim": 618133, "inheritance": "AR",
        "hallmark": "ATP5ME e-subunit = CV₂ DIMERISATION MODULE (with ATP5MF/g); eukaryote-specific (absent in bacteria) — required for CV₂ rows at cristae ridges; loss → balloon-like cristae on electron microscopy (EM) — distinctive ultrastructural finding; 4p16.3 (Huntington gene region); mild phenotype; residual CV monomers present but no dimers or rows; 3-MGA absent (c-ring intact)",
        "key_ddx": "vs ATP5PB (b-stator): ATP5PB → stator disruption, ATP5ME → dimerisation only; vs TMEM70: c-ring assembly vs dimerisation — different BN-PAGE patterns; vs cristae morphology disorders (OPA1, MFN1/2): dynamin/mitofusin fusion vs CV dimer-driven cristae; vs POLG: mtDNA depletion vs isolated CV dimer",
        "founder_variant": "p.Leu30Pro (TM helix break); p.Thr45Met (dimerisation interface); p.Arg61Stop",
        "three_mga_rate": 0.10, "hcm_rate": 0.10, "cardiac_rate": 0.15,
        "leigh_mri_rate": 0.20, "hepatopathy_rate": 0.10, "lactic_ac_rate": 0.60,
        "neuropathy_rate": 0.10, "hyperammonemia_rate": 0.05,
        "atp_activity_mean": 38.0, "atp_activity_sd": 9.0, "median_onset_months": 24,
    },
    {
        "gene": "ATP5MF", "alias": "ATP5J / f-subunit", "aa": "108 aa", "kDa": "12.0 kDa",
        "gene_class": "f0_structural", "subunit_series": "F0-f",
        "cv_module": "F0 f subunit (supernumerary dimerisation module); two TM helices; part of e-f-g dimerisation domain; F0 f subunit in IMS contacts dimerisation partners; chromosome 21q22.11 (same as ATP5PO/OSCP — both on chromosome 21, both involved in Down syndrome dosage); f anchors CV₂ interface",
        "omim_gene": 603148, "chromosome": "21q22.11", "seed": 756,
        "phenotype": "CV Deficiency — CV₂ Dimer Failure — Mild Phenotype + Down Syndrome Dosage Context + Lactic Acidosis",
        "disease": "CV deficiency — ATP5MF; f-subunit LOF → CV₂ dimer absent; chromosome 21 (trisomy 21 dosage with ATP5PO); mild-moderate phenotype; lactic acidosis; balloon-like cristae on EM similar to ATP5ME; 3-MGA absent",
        "disease_omim": 618134, "inheritance": "AR (rare) / Trisomy 21 dosage",
        "hallmark": "ATP5MF f-subunit = CV₂ DIMER INTERFACE alongside ATP5ME; CHROMOSOME 21 (with ATP5PO/OSCP) → Down syndrome has 3 copies of both CV dimerisation (ATP5MF) and stator-top (ATP5PO) subunits — combined dosage effect; two TM helices; IMS domain contacts e/g; BN-PAGE: CV₂ dimer band lost, CV monomer reduced; 21q22.11 cluster (ATP5PO + ATP5MF both on chr21)",
        "key_ddx": "vs ATP5ME (e): both dimerisation module — e is IMS-facing single TM, f is two-TM; vs ATP5PO (OSCP): also chr21 but different function (stator top cap not dimer); vs Down syndrome: trisomy vs rare AR; vs MT-ATP6 (NARP): mtDNA vs nuclear; vs TMEM70: c-ring vs dimer",
        "founder_variant": "p.Gly58Val (TM helix 1); p.Arg92Stop; p.Leu78Pro (inter-helix)",
        "three_mga_rate": 0.10, "hcm_rate": 0.15, "cardiac_rate": 0.20,
        "leigh_mri_rate": 0.20, "hepatopathy_rate": 0.10, "lactic_ac_rate": 0.55,
        "neuropathy_rate": 0.15, "hyperammonemia_rate": 0.05,
        "atp_activity_mean": 40.0, "atp_activity_sd": 10.0, "median_onset_months": 30,
    },
    # ── Assembly Factors ─────────────────────────────────────────────────────
    {
        "gene": "TMEM70", "alias": "TMEM70", "aa": "260 aa", "kDa": "29.0 kDa",
        "gene_class": "assembly_factor", "subunit_series": "AF-TMEM70",
        "cv_module": "IMM-integral c-ring assembly factor; required for c-ring subunit insertion into IMM and stabilisation of nascent c8-ring rotor; acts before ATPAF1/2 (which are F1 chaperones); loss → c-ring fails → no F1 attachment → isolated severe CV deficiency; TMEM70 = most common nuclear-encoded CV assembly factor disease gene worldwide",
        "omim_gene": 612418, "chromosome": "8q21.11", "seed": 757,
        "phenotype": "CV Deficiency — 3-Methylglutaconic Aciduria Type V + Neonatal HCM + Hyperammonaemia + Leigh-like (COXPD29 / MC5DN6)",
        "disease": "MC5DN6 / COXPD29 — TMEM70; most common nuclear CV deficiency worldwide; Czech-Slovak Roma founder c.317-2A>G (1-in-75 carrier in Roma); 3-MGA type V 70%; HCM 65%; hyperammonaemia 45%; neonatal presentation; 5-15% residual CV; BN-PAGE: no mature CV, c-ring sub-complexes visible; Cizkova 2008 NatGenet",
        "disease_omim": 614052, "inheritance": "AR",
        "hallmark": "TMEM70 = MOST COMMON NUCLEAR CV ASSEMBLY FACTOR DISEASE GENE; 3-MGA TYPE V HALLMARK (urine 3-methylglutaconic acid 70% — distinguishing biomarker); Czech-Slovak Roma founder c.317-2A>G (1-in-75 carrier frequency in Roma = highest known CV gene carrier frequency); HCM 65% second highest after ATP5F1A; hyperammonaemia 45% (secondary UCP function impaired); Cizkova 2008 NatGenet landmark; MANDATORY: measure 3-MGA in ALL suspected CV deficiency; BN-PAGE pathognomonic: c-ring accumulation without mature CV",
        "key_ddx": "vs ATP5F1A: no 3-MGA, no Roma, HCM 75%; vs MT-ATP6 (NARP/Leigh): mtDNA maternal vs TMEM70 AR; vs ATPAF1/2: F1 chaperones vs c-ring AF — different BN-PAGE sub-complexes; vs 3-MGA organic acidurias: Barth (tafazzin), costeff (OPA3) — check CV enzymatically; vs BTBGD: MANDATORY SLC19A3 exclusion all neonatal CV",
        "founder_variant": "c.317-2A>G splice (Czech-Slovak Roma, 1-in-75 carrier); c.328G>C (Ashkenazi); p.Leu193Arg",
        "three_mga_rate": 0.70, "hcm_rate": 0.65, "cardiac_rate": 0.70,
        "leigh_mri_rate": 0.55, "hepatopathy_rate": 0.30, "lactic_ac_rate": 0.90,
        "neuropathy_rate": 0.10, "hyperammonemia_rate": 0.45,
        "atp_activity_mean": 8.0, "atp_activity_sd": 3.0, "median_onset_months": 1,
    },
    {
        "gene": "ATPAF1", "alias": "ATP11 / ATP11A", "aa": "401 aa", "kDa": "46.0 kDa",
        "gene_class": "assembly_factor", "subunit_series": "AF-ATPAF1",
        "cv_module": "Matrix-localised F1 chaperone; prevents premature F1 α/β interaction before final assembly; ATPAF1 binds F1 β (ATP5F1B) to prevent non-specific aggregation of unfolded β during assembly; works in concert with ATPAF2 (which binds α); together prevent α-β premature pairing; release of ATPAF1/2 allows correct α₃β₃ formation",
        "omim_gene": 608917, "chromosome": "1p33", "seed": 758,
        "phenotype": "CV Deficiency — F1 Assembly Failure — Encephalomyopathy + Lactic Acidosis (MC5DN7 / ATPAF1-related)",
        "disease": "MC5DN7 / CV deficiency — ATPAF1; F1 β chaperone loss → free β aggregates → α₃β₃ hexamer not formed → isolated severe CV without c-ring defect; 10-25% residual; encephalomyopathy; lactic acidosis; 3-MGA mild/absent (c-ring intact); De Meirleir 2004 first ATPAF2 related (equivalent AF); rare ATPAF1 reports since",
        "disease_omim": 618135, "inheritance": "AR",
        "hallmark": "ATPAF1 = F1 β-SUBUNIT CHAPERONE (prevents β aggregation during F1 assembly); F1 chaperone defect → BN-PAGE: NO F1 sub-complex, NO mature CV, intact c-ring/F0 (opposite to TMEM70 where F0/c-ring fails); 3-MGA ABSENT (c-ring is intact in ATPAF1 mutations — distinguishes from TMEM70); F1 matrix sub-assembly fails; works with ATPAF2 (α chaperone) as a pair; very rare disease gene globally",
        "key_ddx": "vs ATPAF2: ATPAF1 = β chaperone, ATPAF2 = α chaperone — same pathway different targets; vs TMEM70: TMEM70 = c-ring AF (F0 side), ATPAF1 = F1 side — BN-PAGE inverse pattern; vs ATP5F1B: ATP5F1B is structural β (direct protein), ATPAF1 chaperones β; vs BTBGD: mandatory exclusion neonatal lactic acidosis",
        "founder_variant": "p.Arg371Cys (β-binding domain); p.Gly205Asp; p.Leu56Pro (α-helix)",
        "three_mga_rate": 0.15, "hcm_rate": 0.20, "cardiac_rate": 0.25,
        "leigh_mri_rate": 0.40, "hepatopathy_rate": 0.20, "lactic_ac_rate": 0.85,
        "neuropathy_rate": 0.15, "hyperammonemia_rate": 0.10,
        "atp_activity_mean": 14.0, "atp_activity_sd": 4.5, "median_onset_months": 5,
    },
    {
        "gene": "ATPAF2", "alias": "ATP12 / ATP12A", "aa": "289 aa", "kDa": "32.0 kDa",
        "gene_class": "assembly_factor", "subunit_series": "AF-ATPAF2",
        "cv_module": "Matrix-localised F1 chaperone; prevents premature F1 α/β interaction; ATPAF2 binds F1 α (ATP5F1A) — the regulatory/non-catalytic subunit; works in concert with ATPAF1 (β chaperone); ATPAF2 LOF → free α aggregates → α₃β₃ hexamer not formed; first F1 chaperone disease gene identified (De Meirleir 2004, AJHG); 3-MGA in ~50% (secondary c-ring accumulation)',",
        "omim_gene": 608918, "chromosome": "17p11.2", "seed": 759,
        "phenotype": "CV Deficiency — F1 Assembly Failure + 3-Methylglutaconic Aciduria — Severe Neonatal Encephalomyopathy + HCM (MC5DN8 / ATPAF2-related)",
        "disease": "MC5DN8 / CV deficiency — ATPAF2; De Meirleir 2004 AJHG FIRST F1 chaperone disease gene; 3-MGA ~50% (secondary); HCM 40%; severe neonatal; 5-15% residual; α chaperone loss → F1 hexamer fails; 17p11.2 — Smith-Magenis chromosome region",
        "disease_omim": 614053, "inheritance": "AR",
        "hallmark": "ATPAF2 = FIRST F1 CHAPERONE IDENTIFIED AS DISEASE GENE (De Meirleir 2004 AJHG — landmark first human F1 assembly factor disease); ATPAF2 binds F1 α (ATP5F1A); 3-MGA ~50% (intermediate — secondary c-ring accumulation when F1 fails); HCM 40%; 17p11.2 (Smith-Magenis region — Smith-Magenis syndrome RAI1 gene nearby, but ATPAF2 distinct); BN-PAGE: no mature CV, no F1 sub-complex, c-ring sub-complex sometimes visible",
        "key_ddx": "vs ATPAF1: ATPAF2 = α chaperone (first identified), ATPAF1 = β chaperone; vs TMEM70: TMEM70 3-MGA 70% + Roma + c-ring; ATPAF2 3-MGA 50% + F1 failure; vs ATP5F1A: structural α vs ATPAF2 chaperone for α; vs Smith-Magenis (17p11.2): RAI1 deletion vs ATPAF2 point mutation",
        "founder_variant": "p.Trp94Arg (De Meirleir 2004 index patient — Belgian); p.Leu176Pro; p.Arg93Stop",
        "three_mga_rate": 0.50, "hcm_rate": 0.40, "cardiac_rate": 0.45,
        "leigh_mri_rate": 0.45, "hepatopathy_rate": 0.25, "lactic_ac_rate": 0.90,
        "neuropathy_rate": 0.10, "hyperammonemia_rate": 0.20,
        "atp_activity_mean": 10.0, "atp_activity_sd": 3.5, "median_onset_months": 2,
    },
]

N_PER_GENE = 40

# ── Patient simulation per gene ───────────────────────────────────────────────
def _simulate_gene(g):
    rng_g = random.Random(g["seed"])
    pts = []
    for _ in range(N_PER_GENE):
        atp = max(3.0, rng_g.gauss(g["atp_activity_mean"], g["atp_activity_sd"]))
        pts.append({
            "three_mga":         rng_g.random() < g["three_mga_rate"],
            "hcm":               rng_g.random() < g["hcm_rate"],
            "cardiac":           rng_g.random() < g["cardiac_rate"],
            "leigh_mri":         rng_g.random() < g["leigh_mri_rate"],
            "hepatopathy":       rng_g.random() < g["hepatopathy_rate"],
            "lactic_ac":         rng_g.random() < g["lactic_ac_rate"],
            "neuropathy":        rng_g.random() < g["neuropathy_rate"],
            "hyperammonemia":    rng_g.random() < g["hyperammonemia_rate"],
            "atp_activity":      round(atp, 1),
        })
    return pts


def _gene_summary(g):
    pts = _simulate_gene(g)
    n = len(pts)
    return {
        "gene":              g["gene"],
        "alias":             g["alias"],
        "aa":                g["aa"],
        "kDa":               g["kDa"],
        "gene_class":        g["gene_class"],
        "subunit_series":    g["subunit_series"],
        "cv_module":         g["cv_module"],
        "omim_gene":         g["omim_gene"],
        "chromosome":        g["chromosome"],
        "phenotype":         g["phenotype"],
        "disease":           g["disease"],
        "disease_omim":      g["disease_omim"],
        "inheritance":       g["inheritance"],
        "hallmark":          g["hallmark"],
        "key_ddx":           g["key_ddx"],
        "founder_variant":   g["founder_variant"],
        "n_patients":        n,
        "atp_activity_mean_pct":   round(sum(p["atp_activity"] for p in pts) / n, 1),
        "median_onset_months":     g["median_onset_months"],
        "three_mga_pct":     round(100 * sum(p["three_mga"]    for p in pts) / n, 1),
        "hcm_pct":           round(100 * sum(p["hcm"]          for p in pts) / n, 1),
        "cardiac_pct":       round(100 * sum(p["cardiac"]       for p in pts) / n, 1),
        "leigh_mri_pct":     round(100 * sum(p["leigh_mri"]    for p in pts) / n, 1),
        "hepatopathy_pct":   round(100 * sum(p["hepatopathy"]  for p in pts) / n, 1),
        "lactic_ac_pct":     round(100 * sum(p["lactic_ac"]    for p in pts) / n, 1),
        "neuropathy_pct":    round(100 * sum(p["neuropathy"]   for p in pts) / n, 1),
        "hyperammonemia_pct":round(100 * sum(p["hyperammonemia"] for p in pts) / n, 1),
    }


# ── API functions ─────────────────────────────────────────────────────────────
def get_overview():
    gene_summaries = [_gene_summary(g) for g in CV_GENES]
    all_pts = []
    for g in CV_GENES:
        all_pts.extend(_simulate_gene(g))
    N = len(all_pts)

    f1_genes  = [g["gene"] for g in CV_GENES if g["gene_class"] == "f1_structural"]
    f0_genes  = [g["gene"] for g in CV_GENES if g["gene_class"] == "f0_structural"]
    af_genes  = [g["gene"] for g in CV_GENES if g["gene_class"] == "assembly_factor"]

    return {
        "complex":            "Complex V — F1F0-ATP Synthase (Mitochondrial)",
        "function":           "ATP synthesis from ADP + Pi driven by proton gradient across IMM; rotary catalysis; ~2.7 ATP per c8-ring rotation",
        "n_genes":            len(CV_GENES),
        "n_f1_structural":    len(f1_genes),
        "n_f0_structural":    len(f0_genes),
        "n_assembly_factors": len(af_genes),
        "n_patients":         N,
        "cohort_formula":     f"{len(CV_GENES)}×{N_PER_GENE} = {N} patients (seeds 744–759)",
        "mtDNA_subunits":     "MT-ATP6 (F0 a-subunit, proton channel) + MT-ATP8 (peripheral stalk 8) — 2 mtDNA subunits; WES MISSES both; covered in MT-Genome-Atlas + individual dashboards",
        "cii_always_normal":  "CII ALWAYS NORMAL in isolated CV deficiency — CII has zero mtDNA-encoded subunits; internal biochemical reference; CII reduction implies multi-complex or secondary mtDNA defect",
        "three_mga_hallmark": "3-Methylglutaconic aciduria (3-MGA) — elevated in ~60-70% of nuclear CV deficiency (esp. TMEM70 70%, ATPAF2 50%); CV-specific organic acid biomarker; measure in ALL suspected CV; 3-MGA Types: I (AUH), II (Barth-tafazzin), III (Costeff-OPA3), IV (unclassified), V (CV deficiency)",
        "gene_list": {
            "f1_structural_5_nuclear":  f1_genes,
            "f0_structural_8_nuclear":  f0_genes,
            "assembly_factors_3_nuclear": af_genes,
        },
        "rotary_mechanism": {
            "c_ring_stoichiometry": "c8-ring in mammalian CV (8 c-subunits per ring); ~2.7 ATP per 360° rotation",
            "proton_channel":       "MT-ATP6 (mtDNA) provides the a-subunit forming the H⁺ channel at a-c ring interface",
            "catalytic_sites":      "3 catalytic sites on β subunits (ATP5F1B): Open → Loose → Tight conformational cycle driven by γ rotation",
            "rotor":                "c-ring (ATP5MC1/2/3) + γ (ATP5F1C) + δ (ATP5F1D) + ε (ATP5F1E) rotate as unit",
            "stator":               "a-subunit (MT-ATP6) + b (ATP5PB) + d (ATP5PD) + OSCP (ATP5PO) + f (ATP5MF) hold F1 against rotation",
        },
        "cv_dimer": {
            "dimerisation_module":  "ATP5ME (e) + ATP5MF (f) + ATP5MG (g) — eukaryote-specific supernumerary subunits",
            "cristae_shaping":      "CV₂ dimers form rows at cristae ridges; dimer loss → balloon-like cristae on EM",
            "cv2_stoichiometry":    "CV₂ homodimer; two CV monomers interface via e/f/g domain in IMM",
        },
        "hallmark_phenotypes": {
            "TMEM70_3MGA_HCM_Roma": {
                "gene": "TMEM70",
                "note": "3-MGA 70% + HCM 65% + hyperammonaemia 45%; Czech-Slovak Roma c.317-2A>G founder (1-in-75 carrier); most common nuclear CV gene worldwide; Cizkova 2008 NatGenet landmark"
            },
            "ATP5F1A_HCM_Neonatal": {
                "gene": "ATP5F1A",
                "note": "HCM 75% highest F1 subunit; no 3-MGA; no Roma; Jonckheere 2012 first nuclear structural CV disease gene"
            },
            "ATPAF2_F1_Assembly_First": {
                "gene": "ATPAF2",
                "note": "First F1 chaperone disease gene (De Meirleir 2004 AJHG); 3-MGA 50%; HCM 40%; F1 assembly failure on BN-PAGE"
            },
            "ATP5MC3_Cardiac_60pct": {
                "gene": "ATP5MC3",
                "note": "Heart/muscle-enriched c-ring isoform; HCM 60% highest c-ring subunit; cardiac dominant"
            },
            "ATP5ME_Cristae_Morphology": {
                "gene": "ATP5ME",
                "note": "CV₂ dimer failure → balloon-like cristae on EM (diagnostic ultrastructure); mildest phenotype"
            },
        },
        "aggregate_clinical": {
            "three_mga_pct":        round(100 * sum(p["three_mga"]     for p in all_pts) / N, 1),
            "hcm_pct":              round(100 * sum(p["hcm"]           for p in all_pts) / N, 1),
            "cardiac_pct":          round(100 * sum(p["cardiac"]        for p in all_pts) / N, 1),
            "leigh_mri_pct":        round(100 * sum(p["leigh_mri"]     for p in all_pts) / N, 1),
            "hepatopathy_pct":      round(100 * sum(p["hepatopathy"]   for p in all_pts) / N, 1),
            "lactic_ac_pct":        round(100 * sum(p["lactic_ac"]     for p in all_pts) / N, 1),
            "neuropathy_pct":       round(100 * sum(p["neuropathy"]    for p in all_pts) / N, 1),
            "hyperammonemia_pct":   round(100 * sum(p["hyperammonemia"] for p in all_pts) / N, 1),
            "mean_atp_activity_pct":round(sum(p["atp_activity"] for p in all_pts) / N, 1),
        },
        "drug_contraindications": {
            "absolute_ci_all_16_genes": [
                {"drug": "VPA (Valproic acid)", "mechanism": "ABSOLUTE CI ALL 16 CV GENES — inhibits both CI NADH-CoQ reductase AND disrupts ATP synthase coupling; causes fatal hepatic failure in CV deficiency; avoid in all OXPHOS disorders"},
                {"drug": "Metformin", "mechanism": "ABSOLUTE CI ALL 16 CV GENES — CI inhibition → NADH accumulation → reduced substrate for CV proton gradient; secondary CV impairment; lactic acidosis risk"},
                {"drug": "Propofol", "mechanism": "ABSOLUTE CI ALL 16 CV GENES — Propofol inhibits CV directly (F0 lipid perturbation) in addition to CIV; PRIS (Propofol Infusion Syndrome) = ETC multi-complex inhibition; NEVER use for sedation/anaesthesia in CV deficiency"},
                {"drug": "Oligomycin", "mechanism": "ABSOLUTE CI — direct CV inhibitor; binds OSCP (ATP5PO)/c-ring interface; blocks H⁺ translocation; used only in research/diagnostic assays; NOT for clinical use"},
                {"drug": "Linezolid", "mechanism": "ABSOLUTE CI ALL 16 CV GENES — mitoribosomal protein synthesis inhibitor; affects MT-ATP6/MT-ATP8 (mtDNA subunits); secondary CV impairment in CV-deficient patients"},
                {"drug": "Chloramphenicol", "mechanism": "ABSOLUTE CI ALL 16 CV GENES — mitoribosomal inhibitor affecting MT-ATP6/MT-ATP8 synthesis; compounds CV deficiency"},
            ],
            "mandatory_workup": [
                "Urinary organic acids — 3-MGA CRITICAL: elevated in 70% TMEM70, 50% ATPAF2, 45% ATP5F1E; 3-MGA Type V = CV deficiency biomarker",
                "CV (ATP synthase) activity in muscle biopsy — oligomycin-sensitive ATPase assay (Vs/Vt ratio); CI/CII/CIII/CIV should be normal in isolated CV",
                "Plasma lactate + ammonia — hyperammonaemia in 45% TMEM70 (secondary urea cycle impact)",
                "BN-PAGE — pattern distinguishes: TMEM70 (c-ring sub-complex, no mature CV) vs ATPAF1/2 (no F1 sub-complex) vs structural subunit LOF (mature CV reduced)",
                "Echocardiogram — HCM in 65% TMEM70, 75% ATP5F1A, 60% ATP5MC3; serial monitoring",
                "Electron microscopy on muscle — balloon-like cristae in ATP5ME/ATP5MF (dimerisation module) mutations — diagnostic",
                "BTBGD/SLC19A3 MANDATORY EXCLUSION — biotin-thiamine-responsive basal ganglia disease mimics CV-Leigh on MRI; treat empirically pending result",
                "WES/WGS + mtDNA panel — WES detects all 16 nuclear CV genes; WES MISSES MT-ATP6/MT-ATP8 — dedicated mtDNA panel required",
            ],
            "tmem70_copper_note": "No copper-responsive mechanism in CV deficiency (unlike SCO1/SCO2 in CIV) — copper supplementation NOT indicated",
            "ketogenic_diet": "Ketogenic diet AVOID in isolated CV deficiency — KD bypasses glycolysis but imposes high mitochondrial lipid load; may worsen CV-dependent oxidative phosphorylation; individualised metabolic team decision required",
        },
        "wes_utility": {
            "nuclear_genes_detectable": "WES detects all 16 nuclear CV genes (ATP5F1A/B/C/D/E, ATP5PO/PB/MC1/MC2/MC3/PD/ME/MF, TMEM70, ATPAF1, ATPAF2)",
            "mtDNA_missed":             "WES MISSES MT-ATP6 (NARP/Leigh F0 a-subunit) + MT-ATP8 (HCM/Leigh peripheral stalk 8) — both covered in MT-Genome-Atlas individually",
            "panel_note":               "Dedicated mtDNA panel required for MT-ATP6/MT-ATP8; clinical exome + mtDNA fusion panels increasingly available",
            "enzymatic_distinction":    "Enzymatic assay essential: oligomycin-sensitive ATPase (Vs/Vt) isolates CV; confirms variant pathogenicity; CI/CII/CIII/CIV normal in isolated CV; CII=internal reference",
        },
        "three_mga_types": {
            "Type_I":   "AUH — 3-methylglutaconyl-CoA hydratase deficiency (leucine catabolism)",
            "Type_II":  "Barth syndrome — tafazzin (TAZ), cardiolipin remodelling, X-linked, HCM+neutropaenia",
            "Type_III": "Costeff syndrome — OPA3, optic atrophy + chorea",
            "Type_IV":  "Unclassified 3-MGA — multiple aetiologies including CI/CIII/CIV deficiency",
            "Type_V":   "CV deficiency — TMEM70, ATPAF2, ATP5F1E, ATP5F1B and others — nuclear CV gene mutations",
        },
    }


def get_breakdown():
    gene_summaries = [_gene_summary(g) for g in CV_GENES]
    all_pts = []
    for g in CV_GENES:
        all_pts.extend(_simulate_gene(g))
    N = len(all_pts)
    return {
        "genes":          gene_summaries,
        "total_patients": N,
        "n_genes":        len(CV_GENES),
        "aggregate_three_mga_pct":    round(100 * sum(p["three_mga"]     for p in all_pts) / N, 1),
        "aggregate_hcm_pct":          round(100 * sum(p["hcm"]           for p in all_pts) / N, 1),
        "aggregate_cardiac_pct":      round(100 * sum(p["cardiac"]        for p in all_pts) / N, 1),
        "aggregate_leigh_mri_pct":    round(100 * sum(p["leigh_mri"]     for p in all_pts) / N, 1),
        "aggregate_lactic_ac_pct":    round(100 * sum(p["lactic_ac"]     for p in all_pts) / N, 1),
        "aggregate_hyperammonemia_pct": round(100 * sum(p["hyperammonemia"] for p in all_pts) / N, 1),
    }


def get_definitions():
    return {
        "CV_F1F0_ATP_Synthase":         "Complex V — the terminal OXPHOS complex; synthesises ATP from ADP+Pi using the IMM proton gradient generated by CI, CIII, and CIV",
        "F1_Knob":                      "Catalytic extrinsic matrix domain; α₃β₃γδε hexameric knob; β subunits perform rotary catalysis; α subunits scaffold and regulate",
        "F0_Membrane_Domain":           "Membrane-embedded proton-translocating domain; c-ring (rotor) + a-subunit (channel) + peripheral stalk (stator); rotates driven by H⁺ gradient",
        "c_ring_Rotor":                 "8 c-subunits (ATP5MC1/2/3 nuclear + a-subunit/MT-ATP6 mtDNA); proton passes across a-c interface; c8-ring rotation = ~2.7 ATP per 360°",
        "Peripheral_Stalk_Stator":      "ATP5PB (b) + ATP5PD (d) + ATP5PO (OSCP) + ATP5MF (f) — holds F1 against rotor; anti-rotation; connects F0 a-subunit to F1 α/δ",
        "3_Methylglutaconic_Aciduria":  "3-MGA — organic acid elevated in urine; type V = CV deficiency biomarker; TMEM70 70%, ATPAF2 50%, ATP5F1E 45%; measure in ALL suspected CV deficiency",
        "Oligomycin_Sensitivity":       "Oligomycin blocks H⁺ translocation at OSCP (ATP5PO)/c-ring interface; oligomycin-sensitive ATPase assay isolates CV activity (Vs = stimulated, Vt = total ATPase; Vs/Vt = CV fraction)",
        "TMEM70_c_ring_AF":             "TMEM70 = most common nuclear CV disease gene; c-ring assembly factor; inserts c-subunits into IMM; loss → c-ring fails → no F1 attachment; 3-MGA hallmark + Roma founder c.317-2A>G",
        "ATPAF1_ATPAF2_F1_Chaperones":  "ATPAF1 (β chaperone) + ATPAF2 (α chaperone) prevent premature α-β aggregation during F1 assembly; deficiency → F1 hexamer not formed; BN-PAGE: no F1 sub-complex but c-ring intact",
        "CV2_Dimer":                    "Two CV monomers interface via supernumerary subunits ATP5ME (e) + ATP5MF (f) + ATP5MG (g); CV₂ rows shape cristae ridges; loss → balloon-like cristae on EM",
        "CII_Always_Normal":            "CII ALWAYS NORMAL in isolated CV deficiency — CII has zero mtDNA-encoded subunits; useful internal biochemical reference for distinguishing isolated vs multi-complex defects",
        "MT_ATP6_MT_ATP8_WES_Missed":   "MT-ATP6 (F0 a-subunit, NARP/Leigh) + MT-ATP8 (peripheral stalk 8, HCM/Leigh) are mtDNA-encoded; WES misses both; covered in MT-Genome-Atlas + individual dashboards",
        "BTBGD_Mandatory_Exclusion":    "SLC19A3 (biotin-thiamine-responsive basal ganglia disease) MANDATORY EXCLUSION in all CV-Leigh — clinically mimics; treat empirically with biotin+thiamine pending result",
        "BN_PAGE_CV_Pattern":           "BN-PAGE separates CV sub-complexes: TMEM70 → c-ring sub-complex (no mature CV); ATPAF1/2 → no F1 sub-complex; structural subunit LOF → reduced mature CV; CV₂ dimer band at ~1.6 MDa; CV monomer ~600 kDa",
        "Rotary_Catalysis_Mechanism":   "Boyer/Walker mechanism: proton flow rotates c8-ring → rotates γ central stalk 120° → drives β conformational change (Open→Loose→Tight×3) → synthesises 1 ATP per 120° rotation × 3 sites = 3 ATP per full rotation",
        "Hyperammonaemia_TMEM70":       "Hyperammonaemia in 45% TMEM70 — secondary: impaired CV → reduced ATP for urea cycle enzymes → ammonia accumulation; also secondary to hepatopathy in severe cases",
        "Oligomycin_CI_Diagnostic":     "Oligomycin as RESEARCH/DIAGNOSTIC INHIBITOR: used to measure oligomycin-sensitive ATPase (CV assay); NOT for clinical use; direct CV inhibitor = absolute CI in patients",
        "KD_Avoid_CV":                  "Ketogenic diet AVOID in CV deficiency: KD generates high FADH2/NADH from β-oxidation — overwhelms electron entry to ETC requiring intact CV for ATP output; metabolic team decision required",
        "VPA_Absolute_CI_All_16":       "VPA (valproate/valproic acid) ABSOLUTE CI in all 16 CV genes: inhibits CI + disrupts ATP synthase coupling + causes hepatic VPA toxicity (Reye-like) — triple mechanism; NEVER prescribe for seizures in any CV deficiency",
    }


if __name__ == "__main__":
    import json
    ov = get_overview()
    print("Overview keys:", list(ov.keys()))
    print("n_genes:", ov["n_genes"])
    print("n_patients:", ov["n_patients"])
    bk = get_breakdown()
    print("Genes:", len(bk["genes"]))
    df = get_definitions()
    print("Definitions:", len(df))
    print("Aggregate 3-MGA:", ov["aggregate_clinical"]["three_mga_pct"], "%")
    print("Aggregate HCM:", ov["aggregate_clinical"]["hcm_pct"], "%")
    print("TMEM70 hallmark:", ov["hallmark_phenotypes"]["TMEM70_3MGA_HCM_Roma"]["note"][:60])
