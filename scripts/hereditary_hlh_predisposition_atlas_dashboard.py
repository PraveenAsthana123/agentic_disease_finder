#!/usr/bin/env python3
"""Hereditary-HLH-Predisposition-Atlas -- Complete 8-Gene Reference
PRF1    (Perforin-1; 555aa; 10q22.1; AR LOF;
         FHL2 familial haemophagocytic lymphohistiocytosis (OMIM 603553);
         most common FHL ~30-40% of familial cases;
         perforin null → CD107a NORMAL but cytotoxicity ABSENT;
         Q8R/A91V founder variants; emapalumab bridge + HSCT;
         seed SEED_BASE+0).
UNC13D  (Munc13-4; 2090aa; 17q25.1; AR LOF;
         FHL3 (OMIM 608898); most common FHL Middle East / Turkish populations;
         CD107a ABSENT (<5%) PATHOGNOMONIC degranulation block;
         HSCT curative; emapalumab FDA2018 bridge;
         seed SEED_BASE+1).
STX11   (Syntaxin-11; 287aa; 6q24.2; AR LOF;
         FHL4 (OMIM 603552); rare; Kurdish / Turkish populations;
         CD107a ABSENT; later onset vs FHL2/FHL3; MHC-II overlap;
         seed SEED_BASE+2).
STXBP2  (Munc18-2; 592aa; 19p13.2; AR LOF;
         FHL5 (OMIM 613101); IBD 40% HALLMARK;
         hearing loss 30%; CD107a ABSENT; hypogammaglobulinaemia;
         seed SEED_BASE+3).
RAB27A  (Ras-related protein Rab-27A; 221aa; 15q21.3; AR LOF;
         Griscelli Syndrome type 2 (OMIM 607624);
         silvery hair PATHOGNOMONIC; HLH WITHOUT neurodegeneration;
         absent melanosomes in keratinocytes; HSCT curative HLH NOT hair;
         seed SEED_BASE+4).
LYST    (Lysosomal trafficking regulator; 3801aa; 1q42.3; AR LOF;
         Chediak-Higashi Syndrome (OMIM 214500);
         giant granules PATHOGNOMONIC peripheral smear;
         progressive neurological phase; HSCT curative HLH NOT neurodegeneration;
         seed SEED_BASE+5).
SH2D1A  (SH2 domain-containing protein 1A / SAP; 128aa; Xq25; XLR LOF;
         XLP1 X-linked lymphoproliferative disease (OMIM 308240);
         absent iNKT cells PATHOGNOMONIC; EBV-triggered HLH;
         agammaglobulinaemia 50%; HSCT MANDATORY <70% fatal without HSCT first EBV;
         seed SEED_BASE+6).
XIAP    (X-linked inhibitor of apoptosis / BIRC4; 497aa; Xq25; XLR LOF;
         XLP2 (OMIM 300635); IBD 50% HALLMARK; splenomegaly PROMINENT;
         recurrent HLH milder; iNKT LOW but present; HSCT curative;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3574-3581)
"""
import random

SEED_BASE = 3574

ATLAS_GENES = [
    {
        "gene": "PRF1",
        "protein": (
            "PRF1 -- 10q22.1 AR LOF -- 555aa -- "
            "Perforin-1-67kDa-Pore-Forming-Cytolytic-"
            "FHL2-Most-Common-FHL-30-40pct-"
            "CD107a-NORMAL-Cytotoxicity-ABSENT-"
            "Q8R-A91V-Founder-Variants-OMIM-603553"
        ),
        "locus": "10q22.1",
        "protein_size": (
            "555 aa / ~67 kDa / 10q22.1 PRF1 HLH molecular context: "
            "STRUCTURE: "
            "  555 aa / ~67 kDa; pore-forming cytolytic protein; "
            "  N-terminal membrane attack complex / perforin-like (MACPF) domain (aa 1-390); "
            "  C-terminal C2 calcium-binding domain (aa 390-555) — Ca²⁺-dependent membrane binding; "
            "  Stored in lytic granules; Ca²⁺-dependent polymerisation → cylindrical pore in target membrane; "
            "  Perforin pore → granzyme B entry → caspase cascade → apoptosis of cytotoxic target; "
            "  FHL2: biallelic PRF1 LOF → lytic granule contents released but no pore formed → immune response UNRESOLVED; "
            "FAMILIAL HLH TYPE 2 — MOLECULAR MECHANISM: "
            "  PRF1 null → NK and CTL cannot kill APC or infected cells → hyperactivated macrophages persist; "
            "  Cytokine storm: IFNγ, TNFα, IL-2, IL-6, IL-18, ferritin >> 500 ng/mL (HLH-2004 threshold ≥500); "
            "  Hallmark: NK cytotoxicity ABSENT on functional assay (chromium-release / DELFIA) despite NORMAL CD107a; "
            "  CD107a degranulation NORMAL = granule docking/fusion intact; cytotoxicity absent = perforin non-functional; "
            "  This distinguishes PRF1 (FHL2) from FHL3-5 (UNC13D, STX11, STXBP2) where BOTH CD107a AND cytotoxicity absent; "
            "KEY VARIANTS: "
            "  PRF1 Q8R (c.22G>A): North American/European founder; hypomorphic allele → atypical/adult-onset HLH; "
            "  PRF1 A91V (c.272C>T): functional SNP; heterozygous carriers: possible HLH susceptibility cofactor; "
            "  PRF1 p.Trp374ter, p.Glu183Lys, del.exon1-3: null alleles → classic early-onset severe FHL2; "
            "TREATMENT: "
            "  HLH-2004 protocol: dexamethasone 10 mg/m²/day (induction wks 1-8) + etoposide 150 mg/m² twice-weekly (wks 1-2) then weekly; "
            "  Emapalumab (Gamifant®; anti-IFNγ mAb; FDA2018 APPROVED paediatric relapsed/refractory primary HLH): "
            "    1 mg/kg IV twice-weekly escalating → 3 mg/kg → 6 mg/kg → 10 mg/kg per response; "
            "    First approved immunotherapy for primary HLH; reduces IFNγ-driven cytokine storm; "
            "  Ruxolitinib 10-20 mg/m²/dose BD: JAK1/2 inhibitor; bridge to HSCT if HLH-2004 fails; "
            "  Anakinra (IL-1Ra): adjunct for macrophage activation syndrome (MAS/sHLH) overlap; "
            "  HSCT MANDATORY: only curative; RIC conditioning preferred (fludarabine/melphalan ± alemtuzumab); "
            "  AVOID busulfan-based conditioning in active hyperinflammation (organ toxicity risk); "
            "SURVEILLANCE WITHOUT HSCT: "
            "  Recurrence rate 60-80% without HSCT; each recurrence ≡ re-trigger of cytokine storm; "
            "  Ferritin, triglycerides, fibrinogen, CBC, liver enzymes: weekly surveillance first 6 months; "
        ),
        "syndrome": "FHL2-Familial-HLH-Perforin-Deficiency",
        "inheritance": "AR LOF biallelic",
        "hlh_risk": "FHL2 most common familial HLH; recurrence 60-80% without HSCT; fatal without HSCT",
        "pathognomonic": (
            "NK cytotoxicity ABSENT + CD107a NORMAL = PRF1 (FHL2) "
            "vs FHL3-5 where BOTH absent; "
            "perforin-null IHC: absence of perforin staining CTLs"
        ),
        "key_avoid": (
            "AVOID busulfan-based HSCT conditioning in active HLH (organ toxicity); "
            "AVOID live vaccines until HSCT reconstituted; "
            "AVOID vigorous immunosuppression monotherapy without HSCT plan"
        ),
        "key_rule": (
            "HSCT MANDATORY PRF1: only curative option; "
            "emapalumab FDA2018 for relapsed/refractory bridge to HSCT; "
            "CD107a NORMAL in PRF1 (distinguish from FHL3-5 where CD107a absent)"
        ),
        "surveillance": (
            "Ferritin + TG + fibrinogen + CBC + LFTs weekly post-treatment; "
            "family cascade MANDATORY (25% biallelic sibling risk); "
            "perforin staining CTLs annually; NK function q3M until HSCT"
        ),
        "targeted_rx": (
            "Emapalumab FDA2018 (anti-IFNγ; 1→10 mg/kg IV twice-weekly); "
            "Ruxolitinib 10-20 mg/m²/dose BD (JAK1/2; bridge to HSCT); "
            "HSCT curative: RIC fludarabine/melphalan"
        ),
    },
    {
        "gene": "UNC13D",
        "protein": (
            "UNC13D -- 17q25.1 AR LOF -- 2090aa -- "
            "Munc13-4-237kDa-Priming-Factor-"
            "Lytic-Granule-Docking-"
            "FHL3-Middle-East-Turkish-Most-Common-"
            "CD107a-ABSENT-PATHOGNOMONIC-OMIM-608898"
        ),
        "locus": "17q25.1",
        "protein_size": (
            "2090 aa / ~237 kDa / 17q25.1 UNC13D HLH molecular context: "
            "STRUCTURE: "
            "  2090 aa / ~237 kDa; Munc13-4 priming factor; "
            "  C1 diacylglycerol-binding domain; MHD1 + MHD2 homology domains; "
            "  C2A-C2B-C2C calcium-sensor domains; Zn²⁺-binding domain; "
            "  Functions in lytic granule PRIMING: tethers granule to pre-fusion docked state at plasma membrane; "
            "  UNC13D LOF: granules translocate to immunological synapse (lytic IS) BUT cannot dock/prime/fuse; "
            "FHL3 MOLECULAR MECHANISM: "
            "  Munc13-4 null → lytic granules arrive at IS but fail to prime for fusion; "
            "  CD107a (LAMP1) degranulation ABSENT = granule fusion blocked (CD107a exposed only on exocytosis); "
            "  This is the PATHOGNOMONIC assay distinction for UNC13D/STX11/STXBP2 (FHL3-5) from PRF1 (FHL2); "
            "  CD107a <5% by flow cytometry after NK stimulation = degranulation defect = FHL3-5 or GS2 or CHS; "
            "POPULATION GENETICS: "
            "  UNC13D FHL3 most common FHL in Middle Eastern and Turkish populations; "
            "  Common founder splice-site variant: IVS-27 A+1G (intron 27) — common Turkish/Kurdish; "
            "  North African founder: UNC13D p.Pro247Ser; "
            "  European: UNC13D p.Gly15Glu; heterogeneous spectrum globally; "
            "TREATMENT: "
            "  HLH-2004 + emapalumab (FDA2018) + HSCT (same as FHL2 — see PRF1 for emapalumab dosing); "
            "  RIC-HSCT with fludarabine/melphalan or fludarabine/treosulfan; "
            "  alemtuzumab (anti-CD52) in conditioning reduces rejection risk; "
        ),
        "syndrome": "FHL3-Familial-HLH-Munc13-4-Priming-Defect",
        "inheritance": "AR LOF biallelic",
        "hlh_risk": "FHL3; CD107a ABSENT PATHOGNOMONIC; most common FHL in Middle East/Turkish; fatal without HSCT",
        "pathognomonic": (
            "CD107a degranulation ABSENT (<5% by flow) = UNC13D/STX11/STXBP2/RAB27A/LYST; "
            "granule transport intact but fusion blocked; "
            "IVS-27 splice-site founder mutation Middle Eastern populations"
        ),
        "key_avoid": (
            "AVOID NK function assay alone without CD107a (may miss FHL3); "
            "AVOID delay to HSCT — each HLH recurrence causes CNS damage; "
            "AVOID sole HLH-2004 without emapalumab for relapsed/refractory disease"
        ),
        "key_rule": (
            "CD107a flow cytometry MANDATORY FHL workup (distinguish FHL2 from FHL3-5); "
            "IVS-27 UNC13D targeted testing Middle Eastern / Turkish families; "
            "HSCT MANDATORY curative; RIC preferred; alemtuzumab conditioning"
        ),
        "surveillance": (
            "CD107a after HSCT reconstitution; "
            "cascade testing all siblings (25% biallelic risk); "
            "CNS MRI at diagnosis (cerebral HLH 30%); "
            "ferritin/TG/fibrinogen CBC weekly first 6M post-treatment"
        ),
        "targeted_rx": (
            "Emapalumab FDA2018 (anti-IFNγ; bridge to HSCT); "
            "Ruxolitinib BD (JAK1/2; refractory bridge); "
            "HSCT curative: RIC fludarabine/melphalan/alemtuzumab"
        ),
    },
    {
        "gene": "STX11",
        "protein": (
            "STX11 -- 6q24.2 AR LOF -- 287aa -- "
            "Syntaxin-11-32kDa-SNARE-t-SNARE-"
            "Lytic-Granule-Docking-Factor-"
            "FHL4-Kurdish-Turkish-Rare-"
            "Later-Onset-CD107a-ABSENT-OMIM-603552"
        ),
        "locus": "6q24.2",
        "protein_size": (
            "287 aa / ~32 kDa / 6q24.2 STX11 HLH molecular context: "
            "STRUCTURE: "
            "  287 aa / ~32 kDa; Syntaxin-11; Q-SNARE (t-SNARE) protein; "
            "  SNARE domain (aa 195-265); transmembrane domain absent (palmitoylated membrane anchor); "
            "  Forms SNARE complex with VAMP7/8 (v-SNARE) + SNAP23 + Munc18-2 (STXBP2); "
            "  STX11 + STXBP2 (Munc18-2) complex essential for lytic granule fusion at IS; "
            "FHL4 MOLECULAR MECHANISM: "
            "  STX11 null → SNARE complex formation defective → lytic granule membrane fusion blocked; "
            "  CD107a degranulation ABSENT (same mechanism as FHL3/UNC13D); "
            "  ATYPICAL: later onset than FHL2/FHL3 (median age 2-6yr vs <1yr); some adult onset; "
            "  LESS SEVERE than FHL2/FHL3 on average; MHC-II-related cytokine dysregulation overlap; "
            "POPULATION GENETICS: "
            "  FHL4 predominantly Kurdish and Turkish populations (founder effect); "
            "  p.Leu58Pro, p.Arg63His, p.Ala163Thr common Kurdish variants; "
            "  Rare outside Middle Eastern/South Asian populations; "
            "TREATMENT: "
            "  HLH-2004 + emapalumab + HSCT (same framework as FHL2/FHL3); "
            "  Later onset allows more bridge therapy time before HSCT in some patients; "
        ),
        "syndrome": "FHL4-Familial-HLH-Syntaxin-11-SNARE-Defect",
        "inheritance": "AR LOF biallelic",
        "hlh_risk": "FHL4; later onset vs FHL2/FHL3; rare; Kurdish/Turkish populations; HSCT curative",
        "pathognomonic": (
            "CD107a ABSENT + STX11 absent on intracellular staining; "
            "later-onset FHL (median 2-6yr) vs FHL2/FHL3 (<1yr); "
            "Kurdish/Turkish family history; p.Leu58Pro founder mutation"
        ),
        "key_avoid": (
            "AVOID misdiagnosis as reactive HLH due to later/milder onset; "
            "AVOID withholding HSCT due to milder initial course (late recurrence fatal); "
            "AVOID genetic testing limited to PRF1/UNC13D without STX11/STXBP2 in Kurdish families"
        ),
        "key_rule": (
            "STX11 + STXBP2 sequencing MANDATORY Kurdish/Turkish HLH families; "
            "CD107a ABSENT = FHL3/4/5 (not FHL2 where CD107a normal); "
            "HSCT MANDATORY despite milder phenotype (recurrence fatal)"
        ),
        "surveillance": (
            "CD107a + intracellular STX11 staining; "
            "sibling cascade testing (25% biallelic risk); "
            "ferritin/TG monitoring q4W maintenance; CNS MRI at HLH diagnosis"
        ),
        "targeted_rx": (
            "Emapalumab FDA2018 (bridge to HSCT); "
            "Ruxolitinib BD (refractory bridge); "
            "HSCT curative: RIC fludarabine/melphalan"
        ),
    },
    {
        "gene": "STXBP2",
        "protein": (
            "STXBP2 -- 19p13.2 AR LOF -- 592aa -- "
            "Munc18-2-67kDa-SM-Protein-SNARE-Chaperoning-"
            "FHL5-IBD-40pct-HALLMARK-"
            "Hearing-Loss-30pct-"
            "CD107a-ABSENT-OMIM-613101"
        ),
        "locus": "19p13.2",
        "protein_size": (
            "592 aa / ~67 kDa / 19p13.2 STXBP2 HLH molecular context: "
            "STRUCTURE: "
            "  592 aa / ~67 kDa; Munc18-2 (SM = Sec1/Munc18 protein); "
            "  Arc-shaped closed conformation binds closed Syntaxin-11 (STX11); "
            "  Domain 1 (aa 1-125) + Domain 2 (aa 126-250) + Domain 3a/b (aa 251-592); "
            "  Munc18-2 chaperones STX11 → stabilises STX11 protein levels; "
            "  STXBP2 null → STX11 protein rapidly degraded (secondary STX11 deficiency); "
            "FHL5 MOLECULAR MECHANISM: "
            "  STXBP2 null → STX11 degraded → SNARE complex assembly failure → lytic granule fusion blocked; "
            "  CD107a degranulation ABSENT (same mechanism as FHL3/FHL4); "
            "  UNIQUE PHENOTYPE: IBD (inflammatory bowel disease) in 40% HALLMARK — colitis in FHL5; "
            "  Sensorineural hearing loss 30% (cochlear sensory cell cytolytic dysfunction); "
            "  Hypogammaglobulinaemia (B-cell lytic dysfunction → impaired class-switch recombination); "
            "  Neonatal presentation frequent; "
            "IBD IN FHL5 — CLINICAL MANAGEMENT DISTINCTION: "
            "  STXBP2-associated IBD ≠ standard Crohn's/UC → does NOT respond to steroids alone; "
            "  HSCT is the definitive treatment for BOTH HLH AND IBD in STXBP2; "
            "  Pre-HSCT bowel rest/TPN + anti-TNF may bridge; steroids alone INSUFFICIENT; "
            "  Audiology MANDATORY: SNHL in 30% requires early intervention (hearing aids / CI pre-HSCT); "
            "TREATMENT: "
            "  HLH-2004 + emapalumab + HSCT (HSCT treats BOTH HLH and IBD in STXBP2); "
        ),
        "syndrome": "FHL5-Familial-HLH-Munc18-2-Deficiency-IBD",
        "inheritance": "AR LOF biallelic",
        "hlh_risk": "FHL5; IBD 40% HALLMARK; SNHL 30%; hypogammaglobulinaemia; neonatal onset frequent; HSCT curative HLH+IBD",
        "pathognomonic": (
            "IBD + HLH in infant/child = STXBP2 until proven otherwise; "
            "CD107a ABSENT + secondary STX11 deficiency on Western blot; "
            "SNHL 30% (cochlear lytic dysfunction); neonatal HLH most severe FHL"
        ),
        "key_avoid": (
            "AVOID treating STXBP2-IBD as standard Crohn's/UC (steroids inadequate — needs HSCT); "
            "AVOID HSCT delay due to IBD misdiagnosis; "
            "AVOID immunosuppression monotherapy without HSCT plan"
        ),
        "key_rule": (
            "IBD + infant HLH → STXBP2 sequencing MANDATORY; "
            "HSCT treats BOTH HLH AND IBD in STXBP2; "
            "Audiology MANDATORY pre-HSCT; SNHL needs early intervention"
        ),
        "surveillance": (
            "CD107a + STX11 intracellular staining; "
            "audiology every 6M; GI endoscopy if IBD symptoms; "
            "immunoglobulins q3M (hypogammaglobulinaemia); CBC/ferritin weekly active"
        ),
        "targeted_rx": (
            "Emapalumab FDA2018 (bridge to HSCT); "
            "Anti-TNF (infliximab): bridge for IBD pre-HSCT; "
            "HSCT curative: RIC fludarabine/melphalan (treats HLH + IBD)"
        ),
    },
    {
        "gene": "RAB27A",
        "protein": (
            "RAB27A -- 15q21.3 AR LOF -- 221aa -- "
            "Rab27A-25kDa-Ras-GTPase-"
            "Lytic-Granule-Tethering-Melanosome-Transport-"
            "Griscelli-Syndrome-Type-2-"
            "Silvery-Hair-PATHOGNOMONIC-HLH-OMIM-607624"
        ),
        "locus": "15q21.3",
        "protein_size": (
            "221 aa / ~25 kDa / 15q21.3 RAB27A HLH molecular context: "
            "STRUCTURE: "
            "  221 aa / ~25 kDa; Rab27A small GTPase (Ras superfamily); "
            "  GTPase domain (aa 8-175); Switch I/II regions GTP-dependent effector binding; "
            "  Geranylgeranyl lipid anchor (C-terminal CAAX motif); "
            "  Rab27A (GTP-bound active) → tethers lytic granules to cortical actin via Slp1-3a effectors; "
            "  Rab27A also required for melanosomes in melanocytes → DUAL FUNCTION explains bimodal phenotype; "
            "GRISCELLI SYNDROME TYPE 2 — THREE GS TYPES: "
            "  GS1 (MYO5A): silvery hair + SEVERE NEUROLOGICAL (hypotonia, seizures) + NO HLH; "
            "  GS2 (RAB27A): silvery hair + HLH + NO NEURODEGENERATION → PATHOGNOMONIC COMBINATION; "
            "  GS3 (MLPH): silvery hair ALONE (melanophilin; no immune/neurological); "
            "  KEY DISTINCTION: GS2 = silvery hair + HLH WITHOUT neurological = RAB27A; "
            "    GS1 = silvery hair + neurological WITHOUT HLH = MYO5A; "
            "    GS3 = silvery hair ALONE = MLPH; "
            "SILVERY HAIR MECHANISM: "
            "  RAB27A null → melanosomes accumulate in melanocytes → NOT transferred to keratinocytes → clumped melanin; "
            "  Microscopy: hair shaft shows large irregular melanin clumps (vs normal even distribution); "
            "  Hair microscopy + skin biopsy: diagnostic confirmation for GS2; "
            "LYTIC GRANULE DEFECT IN NK/CTL: "
            "  RAB27A null → lytic granules translocate to IS but fail to tether at plasma membrane; "
            "  CD107a degranulation ABSENT (tethering failure before fusion); "
            "  DISTINCT from MYO5A (GS1): MYO5A moves granules toward IS (proximal transport); "
            "    RAB27A (GS2): tethers granules at membrane (distal tethering); "
            "TREATMENT: "
            "  HSCT MANDATORY for HLH treatment in GS2 (curative for immunological defect); "
            "  HSCT does NOT restore pigmentation (melanocyte defect persists — donor immune cells only); "
            "  Pre-HSCT: HLH-2004 + emapalumab + ruxolitinib as bridge; "
        ),
        "syndrome": "GS2-Griscelli-Syndrome-Type-2-RAB27A-HLH",
        "inheritance": "AR LOF biallelic",
        "hlh_risk": "GS2; HLH WITHOUT neurodegeneration; silvery hair PATHOGNOMONIC; CD107a ABSENT; HSCT curative HLH (NOT hair)",
        "pathognomonic": (
            "Silvery/grey hair + HLH WITHOUT neurological symptoms = GS2 PATHOGNOMONIC; "
            "hair shaft microscopy: clumped melanin in melanocytes; "
            "CD107a ABSENT (tethering failure); "
            "DISTINGUISH from GS1 (MYO5A = neurological, no HLH)"
        ),
        "key_avoid": (
            "AVOID MYO5A testing first in silvery hair + HLH (GS2/RAB27A, not GS1/MYO5A); "
            "AVOID false reassurance: 'silvery hair only' misses GS2 if HLH not yet triggered; "
            "AVOID HSCT for GS3/MLPH (silvery hair alone — no immune defect, no HLH risk)"
        ),
        "key_rule": (
            "Silvery hair + HLH = RAB27A/GS2 until proven otherwise; "
            "Hair microscopy + GS gene panel: RAB27A/MYO5A/MLPH; "
            "HSCT curative for HLH (NOT pigmentation — donor immune not melanocytes)"
        ),
        "surveillance": (
            "CD107a NK degranulation; hair microscopy; skin biopsy (clumped melanosomes); "
            "ferritin/TG/fibrinogen/CBC weekly active disease; "
            "sibling cascade (25% biallelic risk); GS gene panel family"
        ),
        "targeted_rx": (
            "Emapalumab FDA2018 (anti-IFNγ; bridge to HSCT); "
            "Ruxolitinib BD (bridge); "
            "HSCT curative: RIC fludarabine/melphalan (treats HLH — not hair colour)"
        ),
    },
    {
        "gene": "LYST",
        "protein": (
            "LYST -- 1q42.3 AR LOF -- 3801aa -- "
            "Lysosomal-Trafficking-Regulator-430kDa-"
            "Beige-Locus-Protein-"
            "Chediak-Higashi-Syndrome-"
            "Giant-Granules-PATHOGNOMONIC-OMIM-214500"
        ),
        "locus": "1q42.3",
        "protein_size": (
            "3801 aa / ~430 kDa / 1q42.3 LYST HLH molecular context: "
            "STRUCTURE: "
            "  3801 aa / ~430 kDa; largest known lysosomal trafficking regulator; "
            "  ARM/HEAT repeats (aa 1-2500); BEACH domain (aa 2700-3400) — Beige And CHS; "
            "  C-terminal WD40 repeats (aa 3400-3801) — protein-protein interaction; "
            "  LYST regulates lysosome-related organelle (LRO) size via fission; "
            "  LYST LOF → LROs cannot undergo fission → giant granule formation; "
            "CHEDIAK-HIGASHI SYNDROME MOLECULAR MECHANISM: "
            "  LYST null → all LROs affected: lytic granules, melanosomes, dense granules, azurophilic granules; "
            "  Giant granules PATHOGNOMONIC: "
            "    Peripheral blood smear: giant azurophilic granules in neutrophils/eosinophils; "
            "    Also in platelets (dense granule giant → platelet dysfunction → bleeding tendency); "
            "    In melanocytes → clumped giant melanosomes → silvery/light hair (similar to GS2); "
            "  Lytic granule defect: giant granules cannot dock/fuse at IS → CD107a ABSENT (similar to FHL3-5); "
            "  HLH ACCELERATED PHASE: triggered by viral infection (usually EBV); "
            "    Accelerated phase = lymphoproliferation + HLH; 50-85% CHS patients reach accelerated phase; "
            "PROGRESSIVE NEUROLOGICAL PHASE: "
            "  If HSCT not performed: progressive cerebellar ataxia, peripheral neuropathy (giant granules in Schwann cells); "
            "  HSCT cures haematological/immunological defect (HLH, infections, giant lytic granules); "
            "  HSCT does NOT prevent neurological progression (neurodegeneration continues post-HSCT); "
            "  Nerve biopsy: giant granule inclusions in Schwann cells persists post-HSCT; "
            "DIAGNOSTIC CLUE: "
            "  Peripheral blood smear DIAGNOSTIC: Wright-Giemsa stain → giant peroxidase-positive granules in PMNL; "
            "  DISTINGUISHES CHS from GS2 (silvery hair; no giant granules on smear) and FHL3-5 (normal granule size); "
            "TREATMENT: "
            "  HSCT MANDATORY in HLH accelerated phase (curative for HLH, not neurodegeneration); "
            "  Early HSCT before HLH accelerated phase: prevents HLH but neurodegeneration may still progress; "
            "  Vitamin C (antioxidant, historical): partial effect on giant granule number; NOT standard of care; "
            "  HLH-2004 + emapalumab + HSCT conditioning: same bridge as FHL; "
        ),
        "syndrome": "CHS-Chediak-Higashi-Syndrome-LYST-Giant-Granules",
        "inheritance": "AR LOF biallelic",
        "hlh_risk": "CHS; giant granules PATHOGNOMONIC smear; HLH accelerated phase 50-85%; HSCT curative HLH NOT neurodegeneration",
        "pathognomonic": (
            "Giant azurophilic granules in neutrophils/eosinophils on peripheral smear PATHOGNOMONIC; "
            "giant dense granules in platelets → platelet dysfunction/bleeding; "
            "silvery hair + giant granules = CHS (vs GS2 = silvery hair only without giant granules on smear)"
        ),
        "key_avoid": (
            "AVOID confusing CHS silvery hair with GS2 (peripheral smear giant granules distinguishes CHS); "
            "AVOID post-HSCT false reassurance for neurological symptoms (neurodegeneration continues); "
            "AVOID Vitamin C monotherapy as primary HLH management (historical — inadequate alone)"
        ),
        "key_rule": (
            "Peripheral blood smear FIRST in silvery hair + HLH (giant granules = CHS/LYST); "
            "HSCT MANDATORY accelerated phase (curative HLH); "
            "Inform family: HSCT prevents HLH/infection but NOT neurodegeneration"
        ),
        "surveillance": (
            "Peripheral smear giant granule monitoring; platelet function analyzer (PFA-100); "
            "neurological examination + nerve conduction q6M; ophthalmology (photophobia/nystagmus); "
            "ferritin/TG weekly active disease; ferritin trigger-based ALERT >500 ng/mL"
        ),
        "targeted_rx": (
            "Emapalumab FDA2018 (anti-IFNγ; bridge to HSCT); "
            "Ruxolitinib BD (JAK1/2; bridge); "
            "HSCT curative: RIC fludarabine/melphalan (HLH, NOT neurodegeneration)"
        ),
    },
    {
        "gene": "SH2D1A",
        "protein": (
            "SH2D1A -- Xq25 XLR LOF -- 128aa -- "
            "SAP-SLAM-Associated-Protein-15kDa-"
            "SH2-Adaptor-"
            "XLP1-EBV-Triggered-HLH-"
            "Absent-iNKT-PATHOGNOMONIC-OMIM-308240"
        ),
        "locus": "Xq25",
        "protein_size": (
            "128 aa / ~15 kDa / Xq25 SH2D1A HLH molecular context: "
            "STRUCTURE: "
            "  128 aa / ~15 kDa; SAP (SLAM-associated protein); "
            "  Single SH2 domain (aa 1-103); C-terminal tail (aa 104-128); "
            "  SAP bridges SLAM-family receptors (CD150, CD244, NTB-A, CRACC, Ly9, CD84) to signalling; "
            "  SLAM receptors self-ligand (CD150:CD150 homophilic on NK:target) → SAP→FynT → cytolytic activation; "
            "  SAP also required for iNKT cell development in thymus; "
            "XLP1 MOLECULAR MECHANISM: "
            "  SH2D1A null → SAP absent → SLAM receptor signalling impaired in NK/CTL → EBV-infected B-cell cytolysis FAILED; "
            "  EBV-infected B cells express CD150 (SLAM) → normally eliminated by SAP-SLAM→FynT signalling; "
            "  XLP1: EBV infection triggers uncontrolled B-cell expansion + macrophage activation = HLH; "
            "  iNKT cells (invariant NKT) require SAP for thymic development: "
            "    iNKT ABSENT (<0.01% CD3+CD56+ TCRVα24+Vβ11+) = XLP1 PATHOGNOMONIC; "
            "    Normal iNKT 0.01-0.5% CD3+CD56+ in peripheral blood; "
            "  B-cell aplasia/agammaglobulinaemia 50% of XLP1 (SAP required B:T cooperation); "
            "  Lymphoma risk (Burkitt/DLBCL) 30% if survive HLH; "
            "X-LINKED MALES ONLY: "
            "  Carrier females (heterozygous SH2D1A): reduced NK activity but typically asymptomatic; "
            "  ALL affected = males (hemizygous); prenatal testing if carrier female identified; "
            "CLINICAL SPECTRUM: "
            "  EBV-triggered HLH (70% at first EBV exposure — median age 8yr range 2-40yr); "
            "  Agammaglobulinaemia 50% (pre-EBV; diagnosis may precede HLH); "
            "  Lymphoma 30% (EBV-driven Burkitt/DLBCL usually after HLH); "
            "  Aplastic anaemia rare; "
            "TREATMENT: "
            "  HSCT MANDATORY: 70-90% fatal without HSCT after first EBV-HLH; "
            "  If agammaglobulinaemia diagnosed pre-HSCT: IVIG until HSCT; "
            "  Rituximab (anti-CD20): eliminate EBV-infected B cells pre-HLH or at HLH onset; "
            "  AVOID live viral vaccines (varicella, MMR, BCG) — cannot clear viral infection; "
            "  HLH-2004 + emapalumab + HSCT: same bridge; "
        ),
        "syndrome": "XLP1-X-Linked-Lymphoproliferative-SAP-Deficiency",
        "inheritance": "XLR LOF hemizygous males",
        "hlh_risk": "XLP1 MALES ONLY; EBV-triggered HLH 70% at first EBV; 70-90% fatal without HSCT; absent iNKT PATHOGNOMONIC",
        "pathognomonic": (
            "iNKT absent (<0.01% CD3+CD56+TCRVα24+Vβ11+) = XLP1 PATHOGNOMONIC; "
            "EBV-triggered HLH in young male; "
            "agammaglobulinaemia pre-HLH 50%; "
            "males ONLY (X-linked recessive)"
        ),
        "key_avoid": (
            "AVOID live vaccines XLP1 (varicella, MMR, BCG, yellow fever); "
            "AVOID EBV exposure without HSCT (EBV = lethal trigger without SAP); "
            "AVOID IVIG as definitive treatment (bridges until HSCT; not curative)"
        ),
        "key_rule": (
            "iNKT flow cytometry MANDATORY XLP workup; "
            "HSCT MANDATORY XLP1 — URGENT (70-90% fatal without HSCT first EBV-HLH); "
            "Rituximab bridges EBV-B cell load pre-HSCT; "
            "All males in XLP1 family: SH2D1A sequencing MANDATORY before first EBV exposure"
        ),
        "surveillance": (
            "iNKT flow q3M until HSCT; EBV-PCR monthly high-risk period; "
            "immunoglobulins q3M (agammaglobulinaemia); "
            "cascade testing all males in family; carrier testing females; "
            "ferritin/TG weekly if EBV-exposed"
        ),
        "targeted_rx": (
            "Rituximab (anti-CD20; eliminate EBV-B cells; bridge pre-HSCT); "
            "IVIG (agammaglobulinaemia bridge); "
            "Emapalumab FDA2018 (HLH-triggered; anti-IFNγ); "
            "HSCT URGENT MANDATORY: RIC fludarabine/melphalan"
        ),
    },
    {
        "gene": "XIAP",
        "protein": (
            "XIAP -- Xq25 XLR LOF -- 497aa -- "
            "X-Linked-Inhibitor-Of-Apoptosis-BIRC4-56kDa-"
            "BIR1-BIR2-BIR3-RING-"
            "XLP2-IBD-50pct-HALLMARK-"
            "Splenomegaly-PROMINENT-OMIM-300635"
        ),
        "locus": "Xq25",
        "protein_size": (
            "497 aa / ~56 kDa / Xq25 XIAP HLH molecular context: "
            "STRUCTURE: "
            "  497 aa / ~56 kDa; XIAP / BIRC4 (Baculoviral IAP Repeat containing protein 4); "
            "  BIR1 domain (aa 33-93): NOD2 NF-κB signalling; "
            "  BIR2 domain (aa 163-223): caspase-3/-7 inhibition; "
            "  BIR3 domain (aa 265-340): caspase-9 inhibition; "
            "  RING finger domain (aa 440-497): E3 ubiquitin ligase; auto-ubiquitination; "
            "  XIAP regulates NF-κB via NOD2 (innate immunity) + inhibits caspase-mediated apoptosis; "
            "XLP2 MOLECULAR MECHANISM: "
            "  XIAP null → NOD2-NF-κB signalling impaired → defective macrophage autophagy; "
            "  Macrophage activation dysregulated → HLH episodes (recurrent, often milder than FHL/XLP1); "
            "  iNKT cells REDUCED but NOT absent (XLP2 vs XLP1: iNKT lower but detectable); "
            "  EBV-associated HLH: less tightly linked than XLP1 (other viral triggers: CMV, adenovirus); "
            "CLINICAL SPECTRUM DISTINCTIONS FROM XLP1: "
            "  IBD 50% HALLMARK (Crohn's-like; XIAP-IBD ≠ standard Crohn's → steroids inadequate → HSCT); "
            "  Splenomegaly PROMINENT (often massive; persists between HLH episodes); "
            "  HLH tends to be RECURRENT and MILDER than XLP1 (multiple episodes over years); "
            "  Agammaglobulinaemia LESS COMMON than XLP1 (10-15% vs 50%); "
            "  Liver disease (hepatitis, cirrhosis) common in recurrent HLH; "
            "  Lymphoma risk lower than XLP1; "
            "  Males ONLY (X-linked); occasional milder carrier female phenotype; "
            "TREATMENT: "
            "  HSCT curative for HLH + IBD in XIAP; "
            "  Pre-HSCT IBD: anti-TNF (infliximab) + bowel rest; do NOT give long-term steroids alone; "
            "  Ruxolitinib (JAK1/2): bridge for recurrent HLH in XIAP; "
            "  Splenectomy: AVOID (removes natural splenic red cell reservoir; does not cure underlying defect); "
            "  Anakinra (IL-1Ra): adjunct for XIAP-HLH refractory to steroids; "
        ),
        "syndrome": "XLP2-X-Linked-Lymphoproliferative-XIAP-Deficiency",
        "inheritance": "XLR LOF hemizygous males",
        "hlh_risk": "XLP2 MALES ONLY; IBD 50% HALLMARK; splenomegaly PROMINENT; recurrent milder HLH; HSCT curative HLH+IBD",
        "pathognomonic": (
            "IBD + recurrent HLH in male = XIAP/XLP2 until proven otherwise; "
            "massive splenomegaly between HLH episodes; "
            "iNKT LOW but present (vs absent in XLP1); "
            "males ONLY (X-linked recessive)"
        ),
        "key_avoid": (
            "AVOID treating XIAP-IBD as standard Crohn's (steroids alone inadequate — needs HSCT); "
            "AVOID splenectomy (does not cure underlying defect; risk of overwhelming sepsis); "
            "AVOID confusing XLP2 milder HLH with reactive HLH (XIAP sequencing mandatory recurrent HLH male)"
        ),
        "key_rule": (
            "XIAP protein expression + XIAP gene sequencing: males with IBD + recurrent HLH; "
            "HSCT treats BOTH HLH AND IBD in XIAP (same as STXBP2); "
            "iNKT flow: LOW not absent (XLP2) vs absent (XLP1) — distinguishes XLP1 vs XLP2; "
            "Ruxolitinib bridge for recurrent XIAP-HLH"
        ),
        "surveillance": (
            "XIAP protein expression flow cytometry q6M until HSCT; "
            "iNKT quantification (vs XLP1); spleen ultrasound q3M; "
            "GI endoscopy IBD surveillance; liver function q3M (hepatitis risk); "
            "EBV + CMV PCR monthly active; ferritin weekly HLH episode"
        ),
        "targeted_rx": (
            "Ruxolitinib BD (JAK1/2; recurrent HLH bridge); "
            "Anakinra (IL-1Ra; adjunct refractory XIAP-HLH); "
            "Emapalumab FDA2018 (anti-IFNγ; severe episodes); "
            "HSCT curative: RIC fludarabine/melphalan (HLH + IBD)"
        ),
    },
]


# ---------------------------------------------------------------------------
# Synthetic patient cohort
# ---------------------------------------------------------------------------

def _make_patient(gene: str, seed: int) -> dict:
    rng = random.Random(seed)
    g = next(g for g in ATLAS_GENES if g["gene"] == gene)

    # Gene-specific HLH clinical parameters
    hlh_params = {
        "PRF1":   dict(age_mu=1.5, age_sd=1.5,  male_prob=0.50, ebv_trigger_prob=0.45,
                       cd107a_normal_prob=0.90, ferritin_mu=18000, ferritin_sd=8000,
                       ibd_prob=0.02, snhl_prob=0.02, giant_granules_prob=0.02,
                       silvery_hair_prob=0.02, hsct_done_prob=0.82, recurrence_prob=0.68),
        "UNC13D": dict(age_mu=0.8, age_sd=0.8,  male_prob=0.50, ebv_trigger_prob=0.38,
                       cd107a_normal_prob=0.04, ferritin_mu=22000, ferritin_sd=10000,
                       ibd_prob=0.02, snhl_prob=0.03, giant_granules_prob=0.02,
                       silvery_hair_prob=0.02, hsct_done_prob=0.78, recurrence_prob=0.72),
        "STX11":  dict(age_mu=3.5, age_sd=2.5,  male_prob=0.50, ebv_trigger_prob=0.40,
                       cd107a_normal_prob=0.05, ferritin_mu=14000, ferritin_sd=6000,
                       ibd_prob=0.03, snhl_prob=0.04, giant_granules_prob=0.02,
                       silvery_hair_prob=0.02, hsct_done_prob=0.70, recurrence_prob=0.55),
        "STXBP2": dict(age_mu=0.3, age_sd=0.3,  male_prob=0.50, ebv_trigger_prob=0.30,
                       cd107a_normal_prob=0.04, ferritin_mu=20000, ferritin_sd=9000,
                       ibd_prob=0.40, snhl_prob=0.30, giant_granules_prob=0.02,
                       silvery_hair_prob=0.02, hsct_done_prob=0.72, recurrence_prob=0.65),
        "RAB27A": dict(age_mu=2.5, age_sd=2.0,  male_prob=0.50, ebv_trigger_prob=0.48,
                       cd107a_normal_prob=0.04, ferritin_mu=16000, ferritin_sd=7000,
                       ibd_prob=0.02, snhl_prob=0.02, giant_granules_prob=0.02,
                       silvery_hair_prob=0.95, hsct_done_prob=0.78, recurrence_prob=0.62),
        "LYST":   dict(age_mu=4.5, age_sd=3.0,  male_prob=0.50, ebv_trigger_prob=0.55,
                       cd107a_normal_prob=0.04, ferritin_mu=15000, ferritin_sd=6000,
                       ibd_prob=0.03, snhl_prob=0.05, giant_granules_prob=0.95,
                       silvery_hair_prob=0.85, hsct_done_prob=0.68, recurrence_prob=0.58),
        "SH2D1A": dict(age_mu=9.0, age_sd=6.0,  male_prob=1.00, ebv_trigger_prob=0.82,
                       cd107a_normal_prob=0.40, ferritin_mu=25000, ferritin_sd=12000,
                       ibd_prob=0.04, snhl_prob=0.02, giant_granules_prob=0.02,
                       silvery_hair_prob=0.02, hsct_done_prob=0.72, recurrence_prob=0.78),
        "XIAP":   dict(age_mu=7.0, age_sd=7.0,  male_prob=1.00, ebv_trigger_prob=0.42,
                       cd107a_normal_prob=0.35, ferritin_mu=12000, ferritin_sd=6000,
                       ibd_prob=0.50, snhl_prob=0.03, giant_granules_prob=0.02,
                       silvery_hair_prob=0.02, hsct_done_prob=0.60, recurrence_prob=0.55),
    }
    p = hlh_params[gene]

    # Age at HLH onset
    age_raw = rng.gauss(p["age_mu"], p["age_sd"])
    if gene in ("PRF1", "UNC13D", "STXBP2"):
        age = max(0.1, min(5.0, round(age_raw, 1)))
    elif gene in ("STX11", "RAB27A"):
        age = max(0.3, min(15.0, round(age_raw, 1)))
    elif gene == "LYST":
        age = max(0.5, min(20.0, round(age_raw, 1)))
    elif gene == "SH2D1A":
        age = max(2.0, min(40.0, round(age_raw, 1)))
    else:  # XIAP
        age = max(0.5, min(45.0, round(age_raw, 1)))

    is_male        = rng.random() < p["male_prob"]
    ebv_trigger    = rng.random() < p["ebv_trigger_prob"]
    cd107a_normal  = rng.random() < p["cd107a_normal_prob"]
    has_ibd        = rng.random() < p["ibd_prob"]
    has_snhl       = rng.random() < p["snhl_prob"]
    has_giant_gran = rng.random() < p["giant_granules_prob"]
    has_silvery    = rng.random() < p["silvery_hair_prob"]
    hsct_done      = rng.random() < p["hsct_done_prob"]
    recurrence     = rng.random() < p["recurrence_prob"]

    # Ferritin peak
    ferritin_raw = rng.gauss(p["ferritin_mu"], p["ferritin_sd"])
    ferritin_peak = max(500, round(ferritin_raw, -1))

    # HLH trigger
    triggers = []
    if ebv_trigger:
        triggers.append("EBV")
    cmv_p = 0.22 if gene not in ("SH2D1A",) else 0.10
    if rng.random() < cmv_p:
        triggers.append("CMV")
    if rng.random() < 0.18:
        triggers.append("bacterial-sepsis")
    if rng.random() < 0.15:
        triggers.append("febrile-illness-unknown")
    if not triggers:
        triggers.append("spontaneous")
    trigger = "/".join(triggers[:2])

    # NK cytotoxicity (chromium-release % at E:T 25:1)
    if gene == "PRF1":
        nk_cytotox_pct = round(rng.uniform(0.5, 3.0), 1)   # near-zero
    elif gene in ("UNC13D", "STX11", "STXBP2", "RAB27A", "LYST"):
        nk_cytotox_pct = round(rng.uniform(0.5, 4.0), 1)   # near-zero (granule release blocked)
    elif gene == "SH2D1A":
        nk_cytotox_pct = round(rng.uniform(1.0, 6.0), 1)   # very low (SAP absent)
    else:  # XIAP
        nk_cytotox_pct = round(rng.uniform(2.0, 12.0), 1)  # low but variable

    # CD107a degranulation
    if cd107a_normal:
        cd107a_pct = round(rng.uniform(12.0, 28.0), 1)
    else:
        cd107a_pct = round(rng.uniform(0.5, 4.5), 1)

    # Perforin staining
    if gene == "PRF1":
        perforin_pct = round(rng.uniform(0.5, 3.0), 1)  # near absent
    else:
        perforin_pct = round(rng.uniform(15.0, 35.0), 1)  # normal (perforin protein present)

    # iNKT cells (XLP differentiation)
    if gene == "SH2D1A":
        inkt_pct = round(rng.uniform(0.001, 0.009), 4)  # absent (<0.01%)
    elif gene == "XIAP":
        inkt_pct = round(rng.uniform(0.003, 0.025), 4)  # low but present
    else:
        inkt_pct = round(rng.uniform(0.01, 0.40), 3)    # normal range

    # HScore estimate (>169 = 93% sensitivity for HLH)
    hscore_base = rng.gauss(215, 35)
    hscore = max(100, min(320, round(hscore_base)))

    # Agammaglobulinaemia
    agammaglobulinaemia = False
    if gene == "SH2D1A":
        agammaglobulinaemia = rng.random() < 0.50
    elif gene == "XIAP":
        agammaglobulinaemia = rng.random() < 0.12
    elif gene == "STXBP2":
        agammaglobulinaemia = rng.random() < 0.30

    # Outcome
    if hsct_done:
        outcomes = ["remission-post-HSCT", "remission-post-HSCT", "remission-post-HSCT", "active-monitoring-post-HSCT"]
    elif gene in ("PRF1", "UNC13D"):
        outcomes = ["HLH-recurrence-pending-HSCT", "CR-HLH-2004-temporary", "active-treatment"]
    else:
        outcomes = ["CR-HLH-2004", "active-monitoring", "HLH-recurrence-pending-HSCT"]
    outcome = rng.choice(outcomes)

    sex = "M" if is_male else "F"
    pid = f"{gene}-{seed:04d}"

    return {
        "patient_id":          pid,
        "gene":                gene,
        "syndrome":            g["syndrome"],
        "sex":                 sex,
        "age_at_hlh_onset":    age,
        "hlh_trigger":         trigger,
        "ferritin_peak_ng_mL": ferritin_peak,
        "hscore":              hscore,
        "nk_cytotox_pct":      nk_cytotox_pct,
        "cd107a_pct":          cd107a_pct,
        "cd107a_normal":       cd107a_normal,
        "perforin_stain_pct":  perforin_pct,
        "inkt_pct":            inkt_pct,
        "has_ibd":             has_ibd,
        "has_snhl":            has_snhl,
        "giant_granules_smear":has_giant_gran,
        "silvery_hair":        has_silvery,
        "agammaglobulinaemia": agammaglobulinaemia,
        "hsct_done":           hsct_done,
        "hlh_recurrence":      recurrence,
        "outcome":             outcome,
        "locus":               g["locus"],
        "inheritance":         g["inheritance"],
        "key_rule":            g["key_rule"],
    }


def generate_overview() -> dict:
    patients = []
    for i, gd in enumerate(ATLAS_GENES):
        for j in range(40):
            patients.append(_make_patient(gd["gene"], SEED_BASE + i + j * 8))

    gene_counts = {}
    for p in patients:
        gene_counts.setdefault(p["gene"], 0)
        gene_counts[p["gene"]] += 1

    # Summary statistics
    total     = len(patients)
    hsct_done = sum(1 for p in patients if p["hsct_done"])
    ebv_trig  = sum(1 for p in patients if "EBV" in p["hlh_trigger"])
    has_ibd   = sum(1 for p in patients if p["has_ibd"])
    has_giant = sum(1 for p in patients if p["giant_granules_smear"])
    has_silver= sum(1 for p in patients if p["silvery_hair"])
    males     = sum(1 for p in patients if p["sex"] == "M")
    recur     = sum(1 for p in patients if p["hlh_recurrence"])
    agamma    = sum(1 for p in patients if p["agammaglobulinaemia"])

    avg_ferritin = round(sum(p["ferritin_peak_ng_mL"] for p in patients) / total)
    avg_age      = round(sum(p["age_at_hlh_onset"] for p in patients) / total, 1)
    avg_hscore   = round(sum(p["hscore"] for p in patients) / total)

    # Per-gene mean onset age
    gene_ages = {}
    for gd in ATLAS_GENES:
        pts = [p for p in patients if p["gene"] == gd["gene"]]
        gene_ages[gd["gene"]] = round(sum(p["age_at_hlh_onset"] for p in pts) / len(pts), 1)

    gene_summary = []
    for gd in ATLAS_GENES:
        pts = [p for p in patients if p["gene"] == gd["gene"]]
        gene_summary.append({
            "gene":             gd["gene"],
            "syndrome":         gd["syndrome"],
            "locus":            gd["locus"],
            "inheritance":      gd["inheritance"],
            "n":                len(pts),
            "mean_age_onset":   gene_ages[gd["gene"]],
            "hlh_risk":         gd["hlh_risk"],
            "pathognomonic":    gd["pathognomonic"],
            "key_rule":         gd["key_rule"],
            "key_avoid":        gd["key_avoid"],
            "targeted_rx":      gd["targeted_rx"],
        })

    return {
        "atlas":              "Hereditary-HLH-Predisposition-Atlas",
        "seed_range":         f"{SEED_BASE}-{SEED_BASE + 7}",
        "total_patients":     total,
        "total_genes":        8,
        "gene_counts":        gene_counts,
        "hsct_done":          hsct_done,
        "ebv_triggered":      ebv_trig,
        "has_ibd":            has_ibd,
        "giant_granules":     has_giant,
        "silvery_hair":       has_silver,
        "males":              males,
        "recurrence":         recur,
        "agammaglobulinaemia":agamma,
        "avg_ferritin_peak":  avg_ferritin,
        "avg_age_onset":      avg_age,
        "avg_hscore":         avg_hscore,
        "gene_summary":       gene_summary,
        "clinical_pearl": (
            "HLH-DIAGNOSIS: ferritin >10 000 highly specific primary HLH; "
            "HLH-2004 criteria: 5/8 of (fever, splenomegaly, cytopenias, hyperTG/hypoFIB, "
            "haemophagocytosis BM, NK-cytolysis low, ferritin ≥500, sIL2Rα >2400); "
            "EMAPALUMAB (FDA2018): anti-IFNγ mAb; first approved therapy primary HLH; "
            "HSCT MANDATORY all FHL/XLP/GS2/CHS — only curative option"
        ),
    }


def generate_breakdown() -> dict:
    all_patients = []
    for i, gd in enumerate(ATLAS_GENES):
        for j in range(40):
            all_patients.append(_make_patient(gd["gene"], SEED_BASE + i + j * 8))

    gene_data = {}
    for gd in ATLAS_GENES:
        pts = [p for p in all_patients if p["gene"] == gd["gene"]]

        cd107a_absent = sum(1 for p in pts if not p["cd107a_normal"])
        cd107a_normal = sum(1 for p in pts if p["cd107a_normal"])
        hsct_done     = sum(1 for p in pts if p["hsct_done"])
        ebv_trig      = sum(1 for p in pts if "EBV" in p["hlh_trigger"])
        ibds          = sum(1 for p in pts if p["has_ibd"])
        silver        = sum(1 for p in pts if p["silvery_hair"])
        giant         = sum(1 for p in pts if p["giant_granules_smear"])
        recur         = sum(1 for p in pts if p["hlh_recurrence"])
        males         = sum(1 for p in pts if p["sex"] == "M")
        agamma        = sum(1 for p in pts if p["agammaglobulinaemia"])

        avg_ferritin  = round(sum(p["ferritin_peak_ng_mL"] for p in pts) / len(pts))
        avg_cd107a    = round(sum(p["cd107a_pct"] for p in pts) / len(pts), 1)
        avg_nk_cytotox= round(sum(p["nk_cytotox_pct"] for p in pts) / len(pts), 1)
        avg_hscore    = round(sum(p["hscore"] for p in pts) / len(pts))
        mean_age      = round(sum(p["age_at_hlh_onset"] for p in pts) / len(pts), 1)
        avg_perforin  = round(sum(p["perforin_stain_pct"] for p in pts) / len(pts), 1)
        avg_inkt      = round(sum(p["inkt_pct"] for p in pts) / len(pts), 4)

        patients_list = sorted(pts, key=lambda x: x["age_at_hlh_onset"])

        gene_data[gd["gene"]] = {
            "gene":            gd["gene"],
            "syndrome":        gd["syndrome"],
            "locus":           gd["locus"],
            "inheritance":     gd["inheritance"],
            "n":               len(pts),
            "mean_age_onset":  mean_age,
            "males_n":         males,
            "cd107a_absent_n": cd107a_absent,
            "cd107a_normal_n": cd107a_normal,
            "avg_cd107a_pct":  avg_cd107a,
            "avg_nk_cytotox":  avg_nk_cytotox,
            "avg_perforin_pct":avg_perforin,
            "avg_inkt_pct":    avg_inkt,
            "avg_ferritin_peak":avg_ferritin,
            "avg_hscore":      avg_hscore,
            "ebv_triggered_n": ebv_trig,
            "hsct_done_n":     hsct_done,
            "ibd_n":           ibds,
            "silvery_hair_n":  silver,
            "giant_granules_n":giant,
            "recurrence_n":    recur,
            "agammaglobulinaemia_n": agamma,
            "hlh_risk":        gd["hlh_risk"],
            "pathognomonic":   gd["pathognomonic"],
            "key_rule":        gd["key_rule"],
            "key_avoid":       gd["key_avoid"],
            "targeted_rx":     gd["targeted_rx"],
            "patients":        patients_list[:20],
        }

    return {
        "atlas":      "Hereditary-HLH-Predisposition-Atlas",
        "seed_range": f"{SEED_BASE}-{SEED_BASE + 7}",
        "gene_data":  gene_data,
    }


def generate_definitions() -> dict:
    defs = {}
    for gd in ATLAS_GENES:
        defs[gd["gene"]] = {
            "gene":          gd["gene"],
            "locus":         gd["locus"],
            "protein_size":  gd["protein_size"],
            "syndrome":      gd["syndrome"],
            "inheritance":   gd["inheritance"],
            "hlh_risk":      gd["hlh_risk"],
            "pathognomonic": gd["pathognomonic"],
            "key_avoid":     gd["key_avoid"],
            "key_rule":      gd["key_rule"],
            "surveillance":  gd["surveillance"],
            "targeted_rx":   gd["targeted_rx"],
        }

    key_clinical_distinctions = [
        "CD107a degranulation assay distinguishes PRF1 (FHL2) from FHL3-5/GS2/CHS: PRF1 null → CD107a NORMAL (granule exocytosis intact) but NK cytotoxicity ABSENT (perforin non-functional); UNC13D/STX11/STXBP2/RAB27A/LYST → CD107a ABSENT (granule docking/fusion blocked); CRITICAL: CD107a flow cytometry MANDATORY in all primary HLH workup — without it, FHL2 cannot be distinguished from FHL3-5",
        "Emapalumab (Gamifant® anti-IFNγ) FDA2018 FIRST APPROVED therapy primary HLH: 1 mg/kg IV twice-weekly escalating to 3→6→10 mg/kg per response; targets IFNγ-driven macrophage activation; bridge to HSCT; approved paediatric relapsed/refractory primary HLH; do NOT use in secondary HLH/MAS without genetic diagnosis (IFNγ blockade may worsen infection-driven secondary HLH)",
        "Griscelli Syndrome 3 types — silvery hair differential: GS2 (RAB27A) = silvery hair + HLH WITHOUT neurodegeneration → requires HSCT; GS1 (MYO5A) = silvery hair + SEVERE NEURODEGENERATION without HLH → no HSCT benefit; GS3 (MLPH) = silvery hair ALONE without immune/neurological → no HSCT needed; hair microscopy clumped melanin = all 3 GS; giant granules on blood smear = LYST/CHS only (not GS)",
        "XLP1 (SH2D1A) vs XLP2 (XIAP) distinction: XLP1 → iNKT ABSENT (<0.01%) PATHOGNOMONIC + agammaglobulinaemia 50% + EBV-triggered HLH dominant; XLP2 → iNKT LOW but present + IBD 50% HALLMARK + splenomegaly PROMINENT + recurrent milder HLH; BOTH X-linked males only; BOTH require HSCT curative; rituximab pre-HSCT in XLP1 for EBV-B cell load",
        "IBD in primary HLH — STXBP2 and XIAP: STXBP2 IBD 40% + XIAP IBD 50% = two primary HLH genes with IBD HALLMARK; IBD in these conditions ≠ standard Crohn's/UC → does NOT respond adequately to steroids or conventional IBD biologics alone; HSCT curative for BOTH HLH AND IBD in STXBP2/XIAP; pre-HSCT anti-TNF bridge (infliximab) + bowel rest; AVOID prolonged conventional IBD management without genetic testing in HLH-associated IBD",
        "LYST/Chediak-Higashi giant granules — PATHOGNOMONIC blood smear diagnosis: Wright-Giemsa peripheral smear = giant azurophilic granules in neutrophils/eosinophils/monocytes; also giant dense granules in platelets → platelet dysfunction (PFA-100 closure time prolonged); giant melanosomes → silvery hair; LYST giant granules = diagnostic on smear ALONE (immediate diagnosis without genetic result); HSCT curative for HLH ACCELERATED PHASE but NOT for progressive neurodegeneration (neurological decline continues post-HSCT)",
        "HSCT conditioning choice in primary HLH: RIC (reduced-intensity conditioning) preferred over myeloablative: fludarabine 150-160 mg/m² + melphalan 140 mg/m² ± alemtuzumab; AVOID busulfan-based conditioning in ACTIVE hyperinflammation (hepatotoxicity, organ damage risk); alemtuzumab reduces graft rejection in FHL3-5/XLP; HSCT timing: ideally in remission of active HLH episode before HSCT to reduce TRM; ruxolitinib/emapalumab bridge to achieve remission pre-HSCT",
        "Ruxolitinib as bridge to HSCT in HLH: JAK1/2 inhibitor; 10-20 mg/m²/dose BD; reduces IFNγ/IL-6/IL-18 storm; effective bridge in HLH-2004 refractory primary and secondary HLH; NOT curative without HSCT in primary HLH; use alongside emapalumab for dual-pathway IFNγ+JAK suppression; taper before HSCT conditioning to avoid JAK inhibition masking early post-HSCT rejection signals",
    ]

    defs_array = [{
        "term":       f"{gd['gene']} — {gd['syndrome']}",
        "definition": (
            f"LOCUS: {gd['locus']} | INHERITANCE: {gd['inheritance']}\n"
            f"HLH RISK: {gd['hlh_risk']}\n"
            f"PATHOGNOMONIC: {gd['pathognomonic']}\n"
            f"KEY RULE: {gd['key_rule']}\n"
            f"KEY AVOID: {gd['key_avoid']}\n"
            f"SURVEILLANCE: {gd['surveillance']}\n"
            f"TARGETED Rx: {gd['targeted_rx']}"
        ),
    } for gd in ATLAS_GENES]

    return {
        "atlas":                     "Hereditary-HLH-Predisposition-Atlas",
        "seed_range":                f"{SEED_BASE}-{SEED_BASE + 7}",
        "definitions":               defs_array,
        "gene_defs":                 defs,
        "key_clinical_distinctions": key_clinical_distinctions,
    }
