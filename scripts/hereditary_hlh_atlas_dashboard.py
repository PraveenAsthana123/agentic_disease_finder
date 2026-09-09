#!/usr/bin/env python3
"""Hereditary-HLH-Lymphohistiocytosis-Atlas — Complete 8-Gene Familial HLH & Lymphoproliferative Atlas
(PRF1 · UNC13D · STX11 · STXBP2 · RAB27A · AP3B1 · SH2D1A · XIAP).

PRF1     (Perforin-1; 555 aa; ~67 kDa; 10q22.1; AR;
          Familial Hemophagocytic Lymphohistiocytosis type 2 (FHL2);
          MOST COMMON genetic FHL — ~30% of all FHL cases worldwide;
          NK/CTL ZERO CYTOTOXICITY — perforin pore formation absent → no target cell lysis;
          HLH-2004 protocol: dexamethasone + etoposide (+ ciclosporin) — initiate immediately;
          seed SEED_BASE+0).
UNC13D   (Unc-13 Homolog D / Munc13-4; 2090 aa; ~237 kDa; 17q25.1; AR;
          Familial Hemophagocytic Lymphohistiocytosis type 3 (FHL3);
          SECOND MOST COMMON FHL — ~25% of all FHL; Middle East/Turkish founder variants;
          Munc13-4 primes cytotoxic granules for fusion — NK degranulation ABSENT (CD107a assay);
          MACROPHAGE ACTIVATION SYNDROME (MAS) IN JIA — UNC13D most common FHL gene in MAS;
          seed SEED_BASE+1).
STX11    (Syntaxin-11; 287 aa; ~33 kDa; 6q24.2; AR;
          Familial Hemophagocytic Lymphohistiocytosis type 4 (FHL4);
          Kurdish/TURKISH FOUNDER — pThr265Ile prevalent in consanguineous Middle Eastern families;
          Syntaxin-11 SNARE protein: absent → NK/T granule fusion failure;
          UNIQUE AMONG FHL: impaired NK cytotoxicity BUT normal NK degranulation on CD107a — DDx key;
          seed SEED_BASE+2).
STXBP2   (Syntaxin-Binding Protein 2 / Munc18-2; 593 aa; ~67 kDa; 19p13.2; AR;
          Familial Hemophagocytic Lymphohistiocytosis type 5 (FHL5);
          EARLIEST ONSET FHL — neonatal/prenatal hydrops fetalis forms reported;
          INFLAMMATORY BOWEL DISEASE association PATHOGNOMONIC in FHL5 vs other FHL subtypes;
          Munc18-2 chaperones STX11/STX3 — STXBP2 loss destabilises both STX11 and STX3;
          seed SEED_BASE+3).
RAB27A   (Ras-Related Protein Rab-27A; 221 aa; ~26 kDa; 15q21.3; AR;
          Griscelli Syndrome type 2 (GS2);
          PARTIAL ALBINISM + EPISODIC HLH COMBINATION PATHOGNOMONIC — silver-grey hair + immune crisis;
          Hair shaft: LARGE IRREGULAR MELANIN CLUMPS on polarised microscopy PATHOGNOMONIC (vs GS1);
          Rab27A docks secretory lysosomes to plasma membrane — absent → melanosomes + granules trapped;
          SILVER HAIR PATHOGNOMONIC — present from birth; HLH onset variable (months–years);
          seed SEED_BASE+4).
AP3B1    (Adaptor-Related Protein Complex 3 Subunit Beta-1; 1094 aa; ~123 kDa; 5q14.1; AR;
          Hermansky-Pudlak Syndrome type 2 (HPS-2);
          ALBINISM + NEUTROPENIA + ABSENT PLATELET DENSE GRANULES TRIAD PATHOGNOMONIC;
          AP3B1 directs lysosome-related organelle biogenesis — HPS-2: no dense granules (electron microscopy);
          RECURRENT HLH EPISODES due to NK cytotoxicity defect — rarest HLH-associated albinism syndrome;
          BLEEDING RISK: dense granule absent → ADP release absent → platelet aggregation impaired;
          seed SEED_BASE+5).
SH2D1A   (SH2 Domain-Containing Protein 1A / SAP; 128 aa; ~15 kDa; Xq25; XLR;
          X-Linked Lymphoproliferative Disease type 1 (XLP-1) / Duncan Disease;
          EBV-TRIGGERED HLH — SAP deficiency allows uncontrolled EBV-driven lymphoproliferation;
          BOYS ONLY (XLR); EBV exposure = FULMINANT INFECTIOUS MONONUCLEOSIS + HLH in 60% — LETHAL;
          Dysgammaglobulinaemia (50%): hypogammaglobulinaemia post-EBV → IVIG lifelong;
          Lymphoma risk 30% — EBV-positive B-cell lymphoma PATHOGNOMONIC sequela;
          HSCT CURATIVE and MANDATORY before EBV exposure — screen at birth in families;
          seed SEED_BASE+6).
XIAP     (X-Linked Inhibitor of Apoptosis / BIRC4; 497 aa; ~57 kDa; Xq25; XLR;
          X-Linked Lymphoproliferative Disease type 2 (XLP-2);
          NOT EBV-exclusive (unlike XLP-1) — HLH triggered by various infections and vaccines;
          INFLAMMATORY BOWEL DISEASE (Crohn's-like) + HLH COMBINATION PATHOGNOMONIC for XLP-2 vs XLP-1;
          XIAP suppresses NOD2-signalling apoptosis — XIAP loss → dysregulated NF-kB/RIPK3 necroptosis;
          Splenomegaly PRESENT in >80% — most prominent physical finding;
          HSCT CURATIVE — lower urgency than XLP-1 if EBV-naive but IBD + recurrent HLH = HSCT indication;
          seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 2334-2341).
"""

import random

SEED_BASE = 2334

HLH_GENES = [
    # -- PRF1 — FHL2 (Perforin-1) -------------------------------------------------------
    {
        "gene": "PRF1",
        "alt_name": (
            "PRF1 (PRF1-555aa-10q22.1 / AR — FHL2 — "
            "MOST-COMMON-FHL-30pct-All-FHL — "
            "NK-CTL-ZERO-CYTOTOXICITY-PERFORIN-PORE-ABSENT-PATHOGNOMONIC — "
            "ETOPOSIDE+DEX+CICLOSPORIN-HLH-2004-Protocol-Initiate-IMMEDIATELY — "
            "HSCT-CURATIVE-MANDATORY-FHL2)"
        ),
        "protein": (
            "PRF1 -- 10q22.1 AR -- PRF1-555aa -- "
            "Perforin-1-67kDa-Pore-Forming-Cytotoxic-Protein-MACPF-Domain -- "
            "Stored-In-CTL-NK-Lytic-Granules-Released-On-Target-Cell-Contact -- "
            "Inserts-Into-Target-Cell-Membrane-Forms-Transmembrane-Pores-Allows-Granzyme-Entry -- "
            "PRF1-Deficiency-No-Target-Cell-Lysis-No-Cytotoxic-Killing-NK-T-Cells -- "
            "MACROPHAGE-HYPERACTIVATION-IFNgamma-Storm-Ferritin-MARKEDLY-Elevated -- "
            "NK-CYTOTOXICITY-ASSAY-ZERO-Gold-Standard-Functional-Diagnostic-Test -- "
            "CD107a-DEGRANULATION-ASSAY-IMPAIRED-NK-Cannot-Degranulate -- "
            "OMIM-Gene-170280-Disease-FHL2-267700"
        ),
        "locus": "10q22.1",
        "protein_size": "555 aa / 67 kDa",
        "inheritance": (
            "AR (autosomal recessive); biallelic loss-of-function variants; "
            "missense + nonsense + splice site; compound heterozygosity common; "
            "no obligate carrier phenotype; Asian/African/Southern European enriched"
        ),
        "hlh_category": "FHL2 — Familial HLH type 2 (Perforin-1 deficiency)",
        "hlh_pathway": (
            "Cytotoxic granule exocytosis → perforin pore formation → granzyme delivery → target apoptosis. "
            "PRF1 deficiency: pore cannot form → target cells survive → persistent APC activation → "
            "uncontrolled macrophage activation → IFN-gamma storm → cytokine-driven organ failure."
        ),
        "pathognomonic": (
            "NK CYTOTOXICITY ZERO (51Cr release assay or flow-based); CD107a degranulation impaired; "
            "ferritin >10,000 µg/L + haemophagocytosis on BM biopsy + splenomegaly + cytopenias 2+ lineages + "
            "hyperferritinaemia + hypertriglyceridaemia + low/absent NK activity + elevated sIL-2R. "
            "HLH-2004 diagnostic criteria: 5 of 8 required."
        ),
        "treatment": (
            "HLH-2004 protocol: dexamethasone 10 mg/m²/day + etoposide 150 mg/m² twice-weekly initially. "
            "Ciclosporin A added week 3-4. CNS-directed therapy (intrathecal MTX/dexamethasone) if neurological. "
            "Bridge to HSCT (allogeneic, matched/mismatched): HSCT IS CURATIVE AND MANDATORY for FHL2. "
            "Emapalumab (anti-IFN-gamma) FDA 2018 — approved for refractory/relapsed primary HLH. "
            "Do not delay HSCT evaluation — FHL2 is universally fatal without HSCT."
        ),
        "key_features": [
            "Most common genetic FHL (~30%)",
            "NK cytotoxicity zero (gold standard diagnostic)",
            "HLH-2004: dex + etoposide + ciclosporin",
            "Emapalumab for refractory HLH (FDA 2018)",
            "HSCT mandatory — curative",
            "Asian/African/Southern European founder variants",
        ],
        "key_ddx": "FHL3 (UNC13D): CD107a also impaired but NK degranulation pattern differs. FHL4 (STX11): normal CD107a. Secondary HLH: trigger identifiable (EBV/infection), no biallelic PRF1.",
        "albinism_risk": "None — no pigmentary abnormality in FHL2",
        "bleeding_risk": "Moderate (thrombocytopenia from HLH — not platelet dysfunction)",
        "ebv_risk": "EBV can trigger HLH in FHL2 but not EBV-selective — any infection may precipitate",
        "ibd_risk": "Not associated",
        "onset_age": "Median 2 months (infantile onset typical); neonatal forms reported",
    },
    # -- UNC13D — FHL3 (Munc13-4) -------------------------------------------------------
    {
        "gene": "UNC13D",
        "alt_name": (
            "UNC13D (UNC13D-2090aa-17q25.1 / AR — FHL3 — "
            "SECOND-MOST-COMMON-FHL-25pct — "
            "MUNC13-4-Priming-Failure-CD107a-ABSENT-NK-Degranulation — "
            "MAS-IN-JIA-MOST-COMMON-FHL-GENE-Macrophage-Activation-Syndrome — "
            "HLH-2004-HSCT-CURATIVE)"
        ),
        "protein": (
            "UNC13D -- 17q25.1 AR -- UNC13D-2090aa -- "
            "Munc13-4-237kDa-SNARE-Priming-Protein-C2-Domain-Architecture -- "
            "Required-For-Cytotoxic-Granule-Priming-Before-SNARE-Mediated-Membrane-Fusion -- "
            "Expressed-NK-Cells-CTLs-Platelets-Mast-Cells-Neutrophils -- "
            "UNC13D-Loss-Cytotoxic-Granules-Not-Primed-Cannot-Fuse-With-Target-Membrane -- "
            "CD107a-LYSOSOMAL-MEMBRANE-MARKER-ABSENT-On-NK-Surface-After-Stimulation -- "
            "NK-Degranulation-Assay-ZERO-Most-Reliable-Functional-Diagnostic -- "
            "FOUNDER-VARIANTS-Turkish-Kurdish-Middle-Eastern-c.927delT-p.Ala310fs -- "
            "OMIM-Gene-608897-Disease-FHL3-608898"
        ),
        "locus": "17q25.1",
        "protein_size": "2090 aa / 237 kDa",
        "inheritance": (
            "AR (autosomal recessive); biallelic; "
            "Turkish/Kurdish founder p.Ala310fs (~30% of Middle Eastern FHL3); "
            "compound heterozygous variants common in non-consanguineous populations"
        ),
        "hlh_category": "FHL3 — Familial HLH type 3 (Munc13-4 deficiency)",
        "hlh_pathway": (
            "UNC13D primes secretory lysosomes for SNARE-complex docking and fusion. "
            "UNC13D deficiency: granules are formed and trafficked to immune synapse but cannot be primed → "
            "membrane fusion blocked → no perforin/granzyme release → persistent target cell survival."
        ),
        "pathognomonic": (
            "NK DEGRANULATION (CD107a) ABSENT on flow cytometry after K562 stimulation — "
            "single most discriminating functional test. MAS in systemic JIA most commonly driven by UNC13D. "
            "Clinical: fever >7 days + splenomegaly + cytopenias + ferritin >500 µg/L triggers workup. "
            "Haemophagocytosis on BM/LN biopsy confirms HLH; UNC13D sequencing confirms FHL3."
        ),
        "treatment": (
            "HLH-2004 protocol (same as FHL2): dexamethasone + etoposide ± ciclosporin. "
            "MAS variant: may respond to IL-1 blockade (anakinra) or IL-6 blockade (tocilizumab) first. "
            "Emapalumab for refractory. HSCT curative and mandatory for FHL3. "
            "Ruxolitinib (JAK1/2 inhibitor) bridging therapy reported in severe refractory cases."
        ),
        "key_features": [
            "Second most common FHL (~25%)",
            "CD107a (NK degranulation) absent — key diagnostic",
            "Most common FHL gene in MAS-associated JIA",
            "Turkish/Kurdish founder c.927delT",
            "HSCT mandatory",
            "Etoposide + dex backbone treatment",
        ],
        "key_ddx": "FHL4 (STX11): CD107a normal (degranulation present) but cytotoxicity zero. FHL2 (PRF1): also CD107a impaired but CD107a pattern subtly different. MAS in JIA: treat HLH first, then underlying rheumatic disease.",
        "albinism_risk": "None",
        "bleeding_risk": "Moderate (from thrombocytopenia)",
        "ebv_risk": "EBV can precipitate; not EBV-selective",
        "ibd_risk": "Not associated",
        "onset_age": "Median 6 months; wider range than FHL2; MAS onset may be in adolescence/adulthood",
    },
    # -- STX11 — FHL4 (Syntaxin-11) -------------------------------------------------------
    {
        "gene": "STX11",
        "alt_name": (
            "STX11 (STX11-287aa-6q24.2 / AR — FHL4 — "
            "KURDISH-TURKISH-FOUNDER-pThr265Ile-Consanguineous-Enriched — "
            "NORMAL-CD107a-BUT-ZERO-CYTOTOXICITY-KEY-DDx-FHL3-FHL2 — "
            "SNARE-Granule-Fusion-Failure-HLH-2004-HSCT)"
        ),
        "protein": (
            "STX11 -- 6q24.2 AR -- STX11-287aa -- "
            "Syntaxin-11-33kDa-SNARE-Protein-Qa-Family -- "
            "Partners-With-VAMP8-SNAP23-On-Lytic-Granule-Fusion-Step -- "
            "STX11-Deficiency-Final-Fusion-Step-Fails-Despite-Normal-Granule-Priming -- "
            "UNIQUE-FHL-PHENOTYPE-NK-Degranulation-CD107a-PRESERVED-But-Lysis-Zero -- "
            "Stxbp2-Munc18-2-Chaperones-STX11-Stxbp2-Loss-Destabilises-STX11 -- "
            "FOUNDER-pThr265Ile-Iraqi-Kurdish-Turkish-Consanguineous-Families -- "
            "OMIM-Gene-605014-Disease-FHL4-603553"
        ),
        "locus": "6q24.2",
        "protein_size": "287 aa / 33 kDa",
        "inheritance": (
            "AR (autosomal recessive); pThr265Ile Kurdish/Iraqi founder variant ~80% of FHL4 in Middle East; "
            "consanguineous families enriched; rare in non-consanguineous European populations"
        ),
        "hlh_category": "FHL4 — Familial HLH type 4 (Syntaxin-11 deficiency)",
        "hlh_pathway": (
            "STX11 is the Qa-SNARE on cytotoxic granules required for the final membrane fusion event. "
            "Unlike Munc13-4 (priming), STX11 deficiency allows priming and docking but final fusion fails → "
            "CD107a can still reach the surface (degranulation marker appears normal) but perforin is not discharged."
        ),
        "pathognomonic": (
            "PARADOX: NK CD107a degranulation PRESENT/NORMAL → but 51Cr cytotoxicity assay ZERO. "
            "This distinguishes FHL4 from all other FHL subtypes — the only FHL with this functional pattern. "
            "Clinically identical HLH presentation. Kurdish/Iraqi founder variant pThr265Ile diagnostic clue."
        ),
        "treatment": (
            "HLH-2004 protocol: dexamethasone + etoposide ± ciclosporin (same backbone). "
            "Emapalumab for refractory. HSCT curative and mandatory. "
            "FHL4 may have slightly milder early course than FHL2 in some series — but HSCT still required."
        ),
        "key_features": [
            "Kurdish/Turkish founder pThr265Ile",
            "CD107a NORMAL — unique among FHL (key DDx)",
            "NK cytotoxicity zero despite normal degranulation",
            "Final SNARE fusion step failure",
            "HSCT mandatory",
            "Consanguineous Middle Eastern enriched",
        ],
        "key_ddx": "FHL3 (UNC13D): CD107a absent. FHL2 (PRF1): CD107a also reduced. FHL4 is the ONLY FHL with normal CD107a — must do cytotoxicity assay separately. STXBP2 (FHL5): same fusion pathway, CD107a also normal/reduced.",
        "albinism_risk": "None",
        "bleeding_risk": "Moderate (thrombocytopenia)",
        "ebv_risk": "EBV can precipitate; not selective",
        "ibd_risk": "Not associated",
        "onset_age": "Median 12 months; slightly later than FHL2",
    },
    # -- STXBP2 — FHL5 (Munc18-2) -------------------------------------------------------
    {
        "gene": "STXBP2",
        "alt_name": (
            "STXBP2 (STXBP2-593aa-19p13.2 / AR — FHL5 — "
            "EARLIEST-ONSET-FHL-Neonatal-Prenatal-Hydrops-Reported — "
            "IBD-ASSOCIATION-PATHOGNOMONIC-Crohns-Like-Colitis-FHL5-Not-Other-FHL — "
            "MUNC18-2-STX11-Chaperone-HLH-2004-HSCT)"
        ),
        "protein": (
            "STXBP2 -- 19p13.2 AR -- STXBP2-593aa -- "
            "Syntaxin-Binding-Protein-2-Munc18-2-67kDa-Sec1-Munc18-SM-Protein -- "
            "Chaperones-STX11-And-STX3-Stabilises-SNARE-Complexes-On-Lytic-Granules -- "
            "STXBP2-Loss-Destabilises-STX11-Protein-Level-Falls-Double-Functional-Defect -- "
            "Expressed-NK-Cells-Platelets-Mast-Cells-Mucosal-Epithelium-Intestine -- "
            "EARLIEST-FHL-ONSET-Neonatal-Ascites-Hydrops-Fetalis-Prenatal-Cases -- "
            "INFLAMMATORY-BOWEL-DISEASE-Colitis-Pathognomonic-For-FHL5-Not-FHL2-3-4 -- "
            "OMIM-Gene-601717-Disease-FHL5-613101"
        ),
        "locus": "19p13.2",
        "protein_size": "593 aa / 67 kDa",
        "inheritance": (
            "AR (autosomal recessive); biallelic; some hypomorphic alleles → adult-onset IBD without HLH; "
            "no founder variant; diverse ethnic backgrounds; compound het common in Europeans"
        ),
        "hlh_category": "FHL5 — Familial HLH type 5 (Munc18-2 deficiency)",
        "hlh_pathway": (
            "STXBP2/Munc18-2 stabilises STX11 protein — STXBP2 loss → STX11 protein degraded → "
            "compound defect in both priming and final fusion. Intestinal epithelial expression of STXBP2 "
            "explains IBD association. Platelet STXBP2 expression → abnormal platelet dense granule secretion."
        ),
        "pathognomonic": (
            "EARLIEST FHL ONSET (neonatal, prenatal hydrops). "
            "IBD (Crohn's-like colitis) in FHL5 but NOT in FHL2/FHL3/FHL4 — PATHOGNOMONIC for FHL5. "
            "CD107a variable (may be reduced or normal). 51Cr cytotoxicity zero. "
            "Hypomorphic STXBP2 alleles → adult IBD without overt HLH (spectrum disorder)."
        ),
        "treatment": (
            "HLH-2004 protocol: dexamethasone + etoposide ± ciclosporin. "
            "IBD component: may need separate IBD therapy (infliximab) but HLH takes priority. "
            "Emapalumab for refractory. HSCT curative and mandatory for FHL5. "
            "Hypomorphic adults with IBD only: biologics may suffice — HSCT decision individualised."
        ),
        "key_features": [
            "Earliest FHL onset (neonatal/prenatal)",
            "IBD (Crohn's-like colitis) PATHOGNOMONIC for FHL5",
            "Munc18-2 stabilises STX11 — double defect on loss",
            "Hypomorphic alleles → adult IBD without HLH",
            "HSCT mandatory for HLH phenotype",
            "Platelet dense granule dysfunction",
        ],
        "key_ddx": "FHL4 (STX11): STXBP2 loss destabilises STX11 — check STX11 protein on Western blot. Crohn's disease: consider STXBP2 sequencing in early-onset or familial IBD + HLH history. XIAP (XLP-2): also IBD + HLH but X-linked.",
        "albinism_risk": "None",
        "bleeding_risk": "Moderate — platelet dense granule dysfunction (secondary to STXBP2 in platelets)",
        "ebv_risk": "EBV can trigger; not selective",
        "ibd_risk": "HIGH — IBD (Crohn's-like colitis) PATHOGNOMONIC for FHL5",
        "onset_age": "Earliest FHL: neonatal onset; prenatal hydrops fetalis cases reported",
    },
    # -- RAB27A — GS2 (Griscelli Syndrome type 2) -------------------------------------------------------
    {
        "gene": "RAB27A",
        "alt_name": (
            "RAB27A (RAB27A-221aa-15q21.3 / AR — Griscelli-Syndrome-Type-2 — "
            "PARTIAL-ALBINISM+EPISODIC-HLH-COMBINATION-PATHOGNOMONIC — "
            "SILVER-GREY-HAIR-BIRTH-LARGE-MELANIN-CLUMPS-POLARISED-MICROSCOPY-KEY-DDx-GS1 — "
            "HSCT-CURATIVE-Corrects-HLH-Not-Hair-Colour)"
        ),
        "protein": (
            "RAB27A -- 15q21.3 AR -- RAB27A-221aa -- "
            "Ras-Related-Protein-Rab-27A-26kDa-Small-GTPase-Rab-Family -- "
            "Docks-Secretory-Lysosomes-And-Melanosomes-To-Plasma-Membrane -- "
            "RAB27A-Links-Melanosomes-To-MLPH-Melanophilin-Myosin-Va-Motor-Protein -- "
            "RAB27A-On-Lytic-Granules-Links-To-Munc13-4-For-Docking-Step -- "
            "RAB27A-Loss-Melanosomes-Trapped-In-Melanocyte-Centre-Not-Transferred-To-Keratinocytes -- "
            "RAB27A-Loss-Lytic-Granules-Cannot-Dock-At-Immune-Synapse-No-Degranulation -- "
            "SILVER-GREY-HAIR-From-Birth-Large-Irregular-Melanin-Clumps-On-Hair-Shaft -- "
            "OMIM-Gene-603868-Disease-GS2-607624"
        ),
        "locus": "15q21.3",
        "protein_size": "221 aa / 26 kDa",
        "inheritance": (
            "AR (autosomal recessive); biallelic; consanguineous families enriched; "
            "Middle Eastern, Mediterranean, South Asian; hair colour abnormality present from birth in carriers only if biallelic"
        ),
        "hlh_category": "Griscelli Syndrome type 2 (GS2) — Rab27A-associated HLH with partial albinism",
        "hlh_pathway": (
            "Rab27A docks secretory lysosomes to the plasma membrane (via Munc13-4 interaction) and "
            "simultaneously docks melanosomes for transfer. Loss → trapped melanosomes (silver hair) + "
            "impaired lytic granule docking → NK/CTL degranulation failure → HLH identical to classic FHL."
        ),
        "pathognomonic": (
            "SILVER-GREY HAIR FROM BIRTH (before HLH) + HLH = GS2 PATHOGNOMONIC COMBINATION. "
            "Hair shaft polarised light microscopy: LARGE IRREGULAR MELANIN CLUMPS (vs GS1 RAB27A: same clumps; "
            "GS3 MYO5A: normal clumps). Neurological features ABSENT (DDx GS1 RAB27A/MYO5A). "
            "NK degranulation (CD107a) ABSENT. Perforin levels NORMAL (DDx FHL2)."
        ),
        "treatment": (
            "HLH-2004 protocol for acute HLH crisis (same backbone: dex + etoposide). "
            "HSCT CURATIVE for HLH and immune defect — hair pigmentation does NOT normalise after HSCT. "
            "Emapalumab for refractory. Do not delay HSCT — recurrent HLH without HSCT is fatal. "
            "Genetic counselling: hair phenotype persists post-HSCT (immune defect corrected, melanocytes not replaced)."
        ),
        "key_features": [
            "Silver-grey hair from birth PATHOGNOMONIC",
            "Large irregular melanin clumps on polarised hair microscopy",
            "HLH = immune crises superimposed on albinism",
            "NK degranulation (CD107a) absent",
            "Perforin levels NORMAL (DDx FHL2)",
            "HSCT corrects HLH not hair colour",
        ],
        "key_ddx": "GS1 (MYO5A): same silver hair + neurological features BUT NO HLH (no immune defect). GS3 (MLPH): only pigmentary (hair), no immune phenotype. CHS (LYST): giant granules on neutrophil smear + partial albinism + HLH but granules visible on smear.",
        "albinism_risk": "PRESENT — partial albinism (silver-grey hair) from birth; hypopigmented skin",
        "bleeding_risk": "Mild (thrombocytopenia from HLH only; no primary platelet dysfunction)",
        "ebv_risk": "EBV can trigger HLH; not selective",
        "ibd_risk": "Not associated",
        "onset_age": "Hair: birth; HLH onset: 1st year of life (most), up to early childhood",
    },
    # -- AP3B1 — HPS-2 -------------------------------------------------------
    {
        "gene": "AP3B1",
        "alt_name": (
            "AP3B1 (AP3B1-1094aa-5q14.1 / AR — Hermansky-Pudlak-Syndrome-Type-2 — "
            "ALBINISM+NEUTROPENIA+ABSENT-PLATELET-DENSE-GRANULES-TRIAD-PATHOGNOMONIC — "
            "ELECTRON-MICROSCOPY-DENSE-GRANULE-ABSENT-PATHOGNOMONIC-NOT-SEEN-Other-HPS — "
            "RECURRENT-HLH-NK-Cytotoxicity-Defect-Rarest-HLH-Albinism)"
        ),
        "protein": (
            "AP3B1 -- 5q14.1 AR -- AP3B1-1094aa -- "
            "Adaptor-Protein-Complex-3-Beta-1-Subunit-123kDa-AP3-Complex -- "
            "AP3-Complex-Sorts-Cargo-To-Lysosomes-Melanosomes-Dense-Granules-Lytic-Granules -- "
            "AP3B1-Loss-Melanosome-Biogenesis-Disrupted-Albinism -- "
            "AP3B1-Loss-Platelet-Dense-Granule-Biogenesis-Absent-Platelet-Aggregation-Impaired -- "
            "AP3B1-Loss-Lytic-Granule-Trafficking-Disrupted-NK-Cytotoxicity-Reduced -- "
            "NEUTROPENIA-Unique-To-HPS-2-Not-Other-HPS-Subtypes-AP3-In-Neutrophil-Elastase-Packaging -- "
            "ELECTRON-MICROSCOPY-Platelet-Dense-Granule-ABSENT-Gold-Standard-HPS-Diagnosis -- "
            "OMIM-Gene-603401-Disease-HPS2-608233"
        ),
        "locus": "5q14.1",
        "protein_size": "1094 aa / 123 kDa",
        "inheritance": (
            "AR (autosomal recessive); biallelic; no founder variant; rare — fewer than 50 families reported worldwide; "
            "all ethnicities"
        ),
        "hlh_category": "Hermansky-Pudlak Syndrome type 2 (HPS-2) — AP3B1-associated HLH with albinism",
        "hlh_pathway": (
            "AP3 complex sorts lysosome-related organelle (LRO) cargo. In CTL/NK: lytic granule biogenesis and "
            "trafficking partially impaired → reduced but not absent perforin delivery → "
            "NK cytotoxicity defect (less severe than FHL) but recurrent HLH possible. "
            "Neutrophil elastase packaging also AP3-dependent → neutropenia (unique to HPS-2)."
        ),
        "pathognomonic": (
            "ALBINISM + CONGENITAL NEUTROPENIA + ABSENT PLATELET DENSE GRANULES = HPS-2 TRIAD PATHOGNOMONIC. "
            "Platelet electron microscopy: DENSE GRANULE ABSENT (whole mount EM or transmission EM) — "
            "required to distinguish from other causes of bleeding in albinism. "
            "Neutropenia UNIQUE to HPS-2 — no other HPS subtype causes neutropenia. "
            "HLH episodes recurrent but may be less severe than FHL2."
        ),
        "treatment": (
            "Bleeding: DDAVP pre-procedure, NO aspirin/NSAIDs ever (irreversible platelet deficit). "
            "Neutropenia: G-CSF if severe recurrent infections. "
            "HLH: HLH-2004 protocol for acute episodes. "
            "HSCT considered for severe recurrent HLH. "
            "Pulmonary fibrosis (as in HPS-1) — pirfenidone if present."
        ),
        "key_features": [
            "Albinism + neutropenia + absent platelet dense granules TRIAD",
            "Neutropenia UNIQUE to HPS-2 (not HPS-1/3/etc.)",
            "Platelet EM: dense granule absent — gold standard",
            "Recurrent HLH from NK cytotoxicity defect",
            "NO ASPIRIN ever (dense granule absent)",
            "Rarest HLH-associated albinism syndrome",
        ],
        "key_ddx": "HPS-1 (HPS1): albinism + dense granule absent + pulmonary fibrosis — NO neutropenia. HPS-3/5/6: albinism + dense granule absent — NO neutropenia, NO HLH. Chediak-Higashi (LYST): giant granules on smear + HLH + partial albinism — no neutropenia.",
        "albinism_risk": "PRESENT — oculocutaneous albinism (OCA-like hypopigmentation)",
        "bleeding_risk": "HIGH — dense granule absent → impaired platelet aggregation; NO aspirin/NSAIDs ABSOLUTE",
        "ebv_risk": "Not specifically EBV-selective; infections in general may trigger HLH",
        "ibd_risk": "Not typically associated",
        "onset_age": "Albinism at birth; neutropenia from birth; HLH episodes childhood to adolescence",
    },
    # -- SH2D1A — XLP-1 (Duncan Disease) -------------------------------------------------------
    {
        "gene": "SH2D1A",
        "alt_name": (
            "SH2D1A (SH2D1A-128aa-Xq25 / XLR — XLP-1-Duncan-Disease — "
            "EBV-TRIGGERED-HLH-SAP-DEFICIENCY-BOYS-ONLY — "
            "FULMINANT-INFECTIOUS-MONONUCLEOSIS-60pct-LETHAL-Without-HSCT — "
            "HSCT-MANDATORY-Before-EBV-Exposure-Screen-At-Birth-In-Families)"
        ),
        "protein": (
            "SH2D1A -- Xq25 XLR -- SH2D1A-128aa -- "
            "SH2-Domain-Containing-Protein-1A-SAP-SLAM-Associated-Protein-15kDa -- "
            "Adaptor-Protein-Bridges-SLAM-Family-Receptors-To-Fyn-Kinase-In-T-NKT-Cells -- "
            "SAP-SLAM-Signalling-Required-For-NKT-Cell-Development-And-CTL-CD8-Memory -- "
            "SAP-Deficiency-EBV-Infected-B-Cells-Cannot-Be-Killed-By-CTL -- "
            "UNCONTROLLED-EBV-DRIVEN-LYMPHOPROLIFERATION-HLH-Lymphoma -- "
            "NKT-CELLS-ABSENT-In-XLP-1-Diagnostic-Flow-Cytometry-CD3+CD1d-Tet+ -- "
            "DYSGAMMAGLOBULINAEMIA-50pct-Post-EBV-Hypogammaglobulinaemia-IVIG-Lifelong -- "
            "OMIM-Gene-300490-Disease-XLP-1-308240"
        ),
        "locus": "Xq25",
        "protein_size": "128 aa / 15 kDa",
        "inheritance": (
            "XLR (X-linked recessive); hemizygous males affected; "
            "carrier females: usually asymptomatic (rarely: lymphoma risk); "
            "de novo variants ~10-20%; family history may be absent"
        ),
        "hlh_category": "X-Linked Lymphoproliferative Disease type 1 (XLP-1) / Duncan Disease",
        "hlh_pathway": (
            "SAP bridges SLAM family receptors (SLAMF1/SLAM, SLAMF6/Ly108) to Fyn kinase in T and NKT cells. "
            "SAP deficiency → T/NKT cells cannot kill EBV-infected B cells via SLAM-mediated cytotoxicity → "
            "EBV-driven B-cell proliferation unchecked → IFN-gamma storm → HLH + lymphoma."
        ),
        "pathognomonic": (
            "EBV INFECTION → FULMINANT INFECTIOUS MONONUCLEOSIS (60% of XLP-1 males on first EBV exposure) → "
            "HLH + hepatic failure. Lymphoma (EBV+, extranodal, Burkitt-like) in 30%. "
            "NKT CELLS ABSENT on flow cytometry (CD3+CD1d-tetramer+) — near-pathognomonic. "
            "Dysgammaglobulinaemia post-EBV: agammaglobulinaemia → IVIG required lifelong."
        ),
        "treatment": (
            "EBV-HLH: rituximab (anti-CD20) → depletes EBV reservoir B cells. Then HLH-2004 backbone. "
            "Emapalumab for refractory. HSCT MANDATORY AND CURATIVE — must occur BEFORE EBV exposure ideally. "
            "Dysgammaglobulinaemia post-EBV: IVIG every 3-4 weeks lifelong if not transplanted. "
            "EBV-seronegative patients: irradiated/CMV-safe blood products; avoid EBV-positive donors."
        ),
        "key_features": [
            "EBV-triggered HLH — EBV-SELECTIVE (unlike FHL)",
            "Boys only (XLR); carrier females asymptomatic",
            "Fulminant IM on first EBV exposure (60%)",
            "NKT cells absent on flow cytometry",
            "Lymphoma risk 30% (EBV+ extranodal)",
            "HSCT mandatory BEFORE EBV exposure — screen at birth",
        ],
        "key_ddx": "XLP-2 (XIAP): NOT EBV-selective; IBD prominent; splenomegaly greater; NKT cells may be present. EBV-HLH without immunodeficiency: biallelic SH2D1A excluded by sequencing. Secondary HLH from EBV: XIAP/SH2D1A must be excluded in all males with EBV-HLH.",
        "albinism_risk": "None",
        "bleeding_risk": "Moderate (from thrombocytopenia in HLH)",
        "ebv_risk": "EXTREME — EBV is the primary and near-exclusive trigger in XLP-1",
        "ibd_risk": "Not associated (DDx from XLP-2)",
        "onset_age": "First EBV exposure (typically 5-10 years); EBV-seronegative boys can be protected until HSCT",
    },
    # -- XIAP — XLP-2 -------------------------------------------------------
    {
        "gene": "XIAP",
        "alt_name": (
            "XIAP (XIAP-497aa-Xq25 / XLR — XLP-2-BIRC4 — "
            "NOT-EBV-EXCLUSIVE-Unlike-XLP-1-Multiple-Triggers-Vaccines-Infections — "
            "IBD-CROHNS-LIKE+HLH-COMBINATION-PATHOGNOMONIC-XLP-2-Not-XLP-1 — "
            "SPLENOMEGALY->80pct-Most-Prominent-Finding-HSCT-CURATIVE)"
        ),
        "protein": (
            "XIAP -- Xq25 XLR -- XIAP-497aa -- "
            "X-Linked-Inhibitor-Of-Apoptosis-BIRC4-57kDa-IAP-Family -- "
            "3-BIR-Domains-BIR1-BIR2-BIR3-RING-Domain-E3-Ubiquitin-Ligase -- "
            "Inhibits-Caspase-3-Caspase-7-BIR2-Caspase-9-BIR3 -- "
            "NOD2-XIAP-Axis-Muramyl-Dipeptide-Sensing-NF-kB-XIAP-Stabilises-RIPK2 -- "
            "XIAP-Loss-NOD2-Signalling-Disrupted-Mucosal-Immunity-Intestinal-Barrier -- "
            "XIAP-Loss-TNF-Mediated-RIPK3-Necroptosis-Dysregulated-In-NK-T-Cells -- "
            "SPLENOMEGALY-80pct-Most-Common-Finding-Even-Without-Active-HLH -- "
            "OMIM-Gene-300079-Disease-XLP-2-300635"
        ),
        "locus": "Xq25",
        "protein_size": "497 aa / 57 kDa",
        "inheritance": (
            "XLR (X-linked recessive); hemizygous males affected; "
            "carrier females: usually asymptomatic but some develop IBD or mild lymphoproliferation; "
            "de novo variants reported"
        ),
        "hlh_category": "X-Linked Lymphoproliferative Disease type 2 (XLP-2) — XIAP deficiency",
        "hlh_pathway": (
            "XIAP suppresses caspase-mediated apoptosis and stabilises RIPK2 in NOD2 signalling. "
            "XIAP loss → exaggerated TNF/NOD2-mediated necroptosis in NK and T cells → "
            "dysregulated cytokine responses → recurrent HLH. "
            "Mucosal XIAP loss → NOD2/RIPK2 pathway disrupted → Crohn's-like IBD."
        ),
        "pathognomonic": (
            "IBD (Crohn's-like colitis, early-onset, granulomatous, perianal) + HLH COMBINATION — "
            "PATHOGNOMONIC for XLP-2 vs XLP-1 (no IBD). "
            "SPLENOMEGALY in >80% — even between HLH episodes, without active disease. "
            "NKT cells: normal or only mildly reduced (DDx from XLP-1 where NKT absent). "
            "HLH triggers: EBV, other infections, live vaccines, idiopathic."
        ),
        "treatment": (
            "IBD: first-line anti-TNF (infliximab/adalimumab) — but HSCT corrects both IBD and HLH. "
            "HLH: HLH-2004 protocol; emapalumab for refractory. "
            "HSCT: CURATIVE for both HLH and IBD — indicated for recurrent HLH or severe IBD. "
            "Avoid live vaccines (can trigger HLH). Splenomegaly: monitor but avoid splenectomy (worsens). "
            "XIAP enzyme activity assay available — functional test complements sequencing."
        ),
        "key_features": [
            "IBD (Crohn's-like colitis) + HLH PATHOGNOMONIC for XLP-2",
            "NOT EBV-exclusive (unlike XLP-1) — multiple triggers",
            "Splenomegaly >80% (even without active HLH)",
            "NKT cells normal (DDx from XLP-1)",
            "HSCT corrects BOTH HLH and IBD",
            "Live vaccines can trigger HLH — avoid",
        ],
        "key_ddx": "XLP-1 (SH2D1A): EBV-selective, NKT absent, no IBD. STXBP2/FHL5: also IBD + HLH but AR, both sexes. Crohn's disease alone: check XIAP in early-onset male IBD + HLH family history. LRBA deficiency: IBD + immune dysregulation but not X-linked, CTLA4 trafficking failure.",
        "albinism_risk": "None",
        "bleeding_risk": "Moderate (from thrombocytopenia in HLH)",
        "ebv_risk": "Elevated but NOT exclusive — any infection/vaccine may trigger",
        "ibd_risk": "HIGH — Crohn's-like IBD PATHOGNOMONIC for XLP-2; early-onset, granulomatous",
        "onset_age": "Variable: IBD may precede HLH by years; HLH median onset 4-6 years",
    },
]

PATIENTS_PER_GENE = 40


def _make_cohort(gene_entry: dict, seed: int) -> list[dict]:
    rng = random.Random(seed)
    gene = gene_entry["gene"]
    patients = []
    for i in range(PATIENTS_PER_GENE):
        age_at_dx = round(rng.uniform(0.1, 18.0), 1)
        ferritin = int(rng.uniform(3000, 250000))
        triglycerides = round(rng.uniform(1.8, 12.0), 1)
        fibrinogen = round(rng.uniform(0.5, 1.8), 1)
        sil2r = int(rng.uniform(2400, 120000))
        outcome = rng.choice(["HSCT — alive, disease-free", "HSCT — alive, disease-free",
                               "HSCT — alive, disease-free", "HLH-2004 response, awaiting HSCT",
                               "Refractory — emapalumab bridge", "Deceased — before HSCT"],)
        patients.append({
            "patient_id": f"{gene}-{seed}-{i+1:03d}",
            "age_at_diagnosis_years": age_at_dx,
            "ferritin_peak_ug_L": ferritin,
            "triglycerides_mmol_L": triglycerides,
            "fibrinogen_g_L": fibrinogen,
            "sIL2R_U_mL": sil2r,
            "haemophagocytosis_on_bm": rng.choice(["Yes", "Yes", "Yes", "Borderline", "No"]),
            "nk_cytotoxicity_zero": gene_entry.get("gene") not in ("STX11",) or rng.random() < 0.95,
            "cd107a_absent": gene_entry.get("gene") in ("PRF1", "UNC13D", "RAB27A", "AP3B1", "SH2D1A", "XIAP"),
            "albinism": gene_entry.get("gene") in ("RAB27A", "AP3B1"),
            "ibd_present": gene_entry.get("gene") in ("STXBP2", "XIAP") and rng.random() < 0.55,
            "ebv_trigger": gene_entry.get("gene") in ("SH2D1A",) and rng.random() < 0.85 or rng.random() < 0.25,
            "outcome": outcome,
            "gene": gene,
        })
    return patients


def generate_overview() -> dict:
    all_patients = []
    for idx, entry in enumerate(HLH_GENES):
        seed = SEED_BASE + idx
        all_patients.extend(_make_cohort(entry, seed))

    total = len(all_patients)
    hsct_done = sum(1 for p in all_patients if "HSCT" in p["outcome"] and "awaiting" not in p["outcome"].lower())
    deceased = sum(1 for p in all_patients if "Deceased" in p["outcome"])
    avg_ferritin = sum(p["ferritin_peak_ug_L"] for p in all_patients) / total
    ebv_triggered = sum(1 for p in all_patients if p["ebv_trigger"])
    ibd_present = sum(1 for p in all_patients if p["ibd_present"])
    albinism_present = sum(1 for p in all_patients if p["albinism"])
    haemophagocytosis_yes = sum(1 for p in all_patients if p["haemophagocytosis_on_bm"] == "Yes")

    gene_summary = {}
    for idx, entry in enumerate(HLH_GENES):
        gene = entry["gene"]
        cohort = _make_cohort(entry, SEED_BASE + idx)
        gene_summary[gene] = {
            "gene": gene,
            "alt_name": entry["alt_name"],
            "locus": entry["locus"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"].split(";")[0].strip(),
            "hlh_category": entry["hlh_category"],
            "pathognomonic": entry["pathognomonic"][:300],
            "n_patients": len(cohort),
            "avg_ferritin": round(sum(p["ferritin_peak_ug_L"] for p in cohort) / len(cohort)),
            "hsct_rate_pct": round(100 * sum(1 for p in cohort if "HSCT" in p["outcome"] and "awaiting" not in p["outcome"].lower()) / len(cohort), 1),
            "albinism_pct": round(100 * sum(1 for p in cohort if p["albinism"]) / len(cohort), 1),
            "ibd_pct": round(100 * sum(1 for p in cohort if p["ibd_present"]) / len(cohort), 1),
            "ebv_trigger_pct": round(100 * sum(1 for p in cohort if p["ebv_trigger"]) / len(cohort), 1),
        }

    return {
        "atlas": "Hereditary-HLH-Lymphohistiocytosis-Atlas",
        "subtitle": "Complete 8-Gene Familial HLH & X-Linked Lymphoproliferative Disease Reference",
        "genes_covered": [e["gene"] for e in HLH_GENES],
        "total_patients": total,
        "seeds": f"{SEED_BASE}–{SEED_BASE + 7}",
        "aggregate_metrics": {
            "avg_ferritin_ug_L": round(avg_ferritin),
            "hsct_completed_pct": round(100 * hsct_done / total, 1),
            "deceased_pct": round(100 * deceased / total, 1),
            "ebv_triggered_pct": round(100 * ebv_triggered / total, 1),
            "ibd_present_pct": round(100 * ibd_present / total, 1),
            "albinism_present_pct": round(100 * albinism_present / total, 1),
            "haemophagocytosis_confirmed_pct": round(100 * haemophagocytosis_yes / total, 1),
        },
        "gene_summary": gene_summary,
    }


def generate_breakdown() -> dict:
    breakdown = []
    for idx, entry in enumerate(HLH_GENES):
        seed = SEED_BASE + idx
        cohort = _make_cohort(entry, seed)
        breakdown.append({
            "gene": entry["gene"],
            "alt_name": entry["alt_name"],
            "locus": entry["locus"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"],
            "hlh_category": entry["hlh_category"],
            "hlh_pathway": entry["hlh_pathway"],
            "pathognomonic": entry["pathognomonic"],
            "treatment": entry["treatment"],
            "key_features": entry["key_features"],
            "key_ddx": entry["key_ddx"],
            "albinism_risk": entry["albinism_risk"],
            "bleeding_risk": entry["bleeding_risk"],
            "ebv_risk": entry["ebv_risk"],
            "ibd_risk": entry["ibd_risk"],
            "onset_age": entry["onset_age"],
            "n_patients": len(cohort),
            "avg_ferritin_ug_L": round(sum(p["ferritin_peak_ug_L"] for p in cohort) / len(cohort)),
            "avg_triglycerides": round(sum(p["triglycerides_mmol_L"] for p in cohort) / len(cohort), 1),
            "haemophagocytosis_pct": round(100 * sum(1 for p in cohort if p["haemophagocytosis_on_bm"] == "Yes") / len(cohort), 1),
            "hsct_completed_pct": round(100 * sum(1 for p in cohort if "HSCT" in p["outcome"] and "awaiting" not in p["outcome"].lower()) / len(cohort), 1),
            "albinism_pct": round(100 * sum(1 for p in cohort if p["albinism"]) / len(cohort), 1),
            "ibd_pct": round(100 * sum(1 for p in cohort if p["ibd_present"]) / len(cohort), 1),
            "ebv_trigger_pct": round(100 * sum(1 for p in cohort if p["ebv_trigger"]) / len(cohort), 1),
            "sample_patients": cohort[:3],
        })
    return {"gene_breakdowns": breakdown}


def generate_definitions() -> dict:
    return {
        "gene_entries": {
            entry["gene"]: {
                "gene": entry["gene"],
                "full_name": entry["protein"].split(" --")[0].strip(),
                "locus": entry["locus"],
                "protein_size": entry["protein_size"],
                "inheritance": entry["inheritance"].split(";")[0].strip(),
                "disease_name": entry["hlh_category"],
                "hlh_pathway": entry["hlh_pathway"],
                "pathognomonic": entry["pathognomonic"],
                "treatment": entry["treatment"][:400],
                "key_features": entry["key_features"],
                "key_ddx": entry["key_ddx"],
                "albinism_risk": entry["albinism_risk"],
                "bleeding_risk": entry["bleeding_risk"],
                "ebv_risk": entry["ebv_risk"],
                "ibd_risk": entry["ibd_risk"],
                "onset_age": entry["onset_age"],
            }
            for entry in HLH_GENES
        },
        "hlh_glossary": {
            "Familial HLH (FHL)": (
                "A group of autosomal recessive primary immunodeficiencies causing defective cytotoxic granule exocytosis. "
                "The shared pathomechanism is failure to kill activated antigen-presenting cells → uncontrolled macrophage activation → "
                "IFN-gamma storm → multi-organ failure. The five genetic subtypes are: FHL1 (locus 9q21.3, gene unknown), "
                "FHL2 (PRF1), FHL3 (UNC13D), FHL4 (STX11), FHL5 (STXBP2). "
                "All are universally fatal without HSCT. HLH-2004 protocol is the induction backbone."
            ),
            "HLH-2004 Diagnostic Criteria (5/8 required)": (
                "1. Fever (>38.5°C). 2. Splenomegaly. 3. Cytopenias ≥2 lineages (Hb <90 g/L; plt <100×10⁹/L; neutrophils <1×10⁹/L). "
                "4. Hypertriglyceridaemia (≥3.0 mmol/L) and/or hypofibrinogenaemia (≤1.5 g/L). "
                "5. Haemophagocytosis in BM/spleen/LN. 6. Low/absent NK-cell activity. "
                "7. Ferritin ≥500 µg/L (>10,000 = highly suggestive in paediatric HLH). "
                "8. Elevated sIL-2R (>2400 U/mL)."
            ),
            "HScore": (
                "A clinical scoring tool for reactive HLH probability. "
                "Score >169 = 93% probability of HLH. Factors include: immunosuppression, fever, organomegaly, "
                "cytopenias, ferritin, triglycerides, fibrinogen, haemophagocytosis, AST. "
                "Useful in adults and for distinguishing reactive from familial HLH."
            ),
            "CD107a Degranulation Assay": (
                "Flow cytometric functional test for cytotoxic granule degranulation. "
                "NK cells are stimulated with K562 target cells; CD107a (LAMP-1) normally translocates to NK surface during granule fusion. "
                "ABSENT in FHL3 (UNC13D), partially absent/reduced in FHL2 (PRF1), RAB27A, AP3B1. "
                "PRESERVED in FHL4 (STX11) despite zero cytotoxicity — key DDx point."
            ),
            "NK Cytotoxicity Assay": (
                "Gold standard functional test. NK cells are incubated with 51Cr-labelled K562 target cells; "
                "specific lysis measured. Zero in all FHL subtypes + GS2 (RAB27A) + HPS-2 (AP3B1) + XLP-1 (SH2D1A). "
                "Must be paired with CD107a to distinguish FHL subtypes."
            ),
            "Emapalumab": (
                "Anti-IFN-gamma monoclonal antibody (FDA approved November 2018 — first HLH-specific biologic). "
                "Indicated for primary HLH that is refractory, recurrent, or intolerant to conventional HLH therapy. "
                "Neutralises the key effector cytokine (IFN-gamma) driving macrophage hyperactivation. "
                "Bridge to HSCT — not a substitute. Requires infectious disease screening (TB, fungi) before use."
            ),
            "XLP (X-Linked Lymphoproliferative Disease)": (
                "Two X-linked disorders of lymphocyte regulation: XLP-1 (SH2D1A/SAP deficiency) and XLP-2 (XIAP deficiency). "
                "XLP-1: EBV-selective, NKT absent, lymphoma risk 30%, fulminant IM on EBV exposure. "
                "XLP-2: not EBV-selective, IBD (Crohn's-like) + HLH pathognomonic, splenomegaly prominent, NKT normal. "
                "HSCT curative for both — mandatory before EBV exposure in XLP-1 families."
            ),
            "Griscelli Syndrome (GS)": (
                "Three types of AR partial albinism with variable immune phenotypes. "
                "GS1 (MYO5A): partial albinism + neurological — NO immune defect. "
                "GS2 (RAB27A): partial albinism + HLH — immune crisis in addition to silver hair. "
                "GS3 (MLPH): partial albinism only — no neurological, no immune defect. "
                "Hair shaft polarised microscopy shows LARGE IRREGULAR MELANIN CLUMPS in GS1 and GS2 (both RAB27A/MYO5A pathway)."
            ),
            "HPS-2 vs Other HPS Subtypes": (
                "HPS-2 (AP3B1) is UNIQUE among HPS subtypes: it is the ONLY subtype with congenital neutropenia + HLH risk. "
                "All HPS subtypes share: OCA-like albinism + absent platelet dense granules on EM. "
                "HPS-1 (HPS1): adds pulmonary fibrosis (lethal). "
                "HPS-3/5/6: milder, no fibrosis, no neutropenia, no HLH. "
                "EM platelet dense granule assessment is mandatory to diagnose ALL HPS subtypes."
            ),
        },
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(generate_overview(), indent=2)[:2000])
    print("\n=== BREAKDOWN (gene 0 only) ===")
    bd = generate_breakdown()
    print(json.dumps(bd["gene_breakdowns"][0], indent=2)[:2000])
    print("\n=== DEFINITIONS (first key) ===")
    defn = generate_definitions()
    first_key = next(iter(defn["gene_entries"]))
    print(json.dumps(defn["gene_entries"][first_key], indent=2)[:1500])
