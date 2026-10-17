#!/usr/bin/env python3
"""Hereditary-Immune-Dysregulation-HLH-ALPS-Atlas -- Complete 8-Gene Reference
Diseases of Immune Dysregulation: Familial HLH / Griscelli / Chediak-Higashi / ALPS / IPEX
(PRF1/AR-FHL2-Perforin-Pore-Former-CYTOTOXIC-GRANULE-EXOCYTOSIS-DEFECT,
 UNC13D/AR-FHL3-MUNC13-4-Vesicle-Priming-CD107a-Mobilisation-PATHOGNOMONIC,
 STX11/AR-FHL4-Syntaxin-11-Kurdish-Founder-Vesicle-Docking,
 STXBP2/AR-FHL5-MUNC18-2-Syntaxin-Binding-Protein,
 LYST/AR-Chediak-Higashi-GIANT-GRANULES-PATHOGNOMONIC-Partial-Albinism-NK-CTL-Defect,
 RAB27A/AR-Griscelli-Type2-SILVERY-HAIR-PATHOGNOMONIC-No-Giant-Granules-vs-LYST,
 FAS/AD-AR-ALPS-TypeIa-ELEVATED-DNT-CELLS-PATHOGNOMONIC-Evans-Syndrome,
 FOXP3/XL-IPEX-Neonatal-T1DM-Enteropathy-Eczema-PATHOGNOMONIC-Treg-Absent):

PRF1   (perforin 1; 555aa; 10q22.1; AR;
         FHL type 2 (FHL2) — most common familial haemophagocytic lymphohistiocytosis, 30–40% of all FHL;
         Perforin is a pore-forming protein stored in the lytic granules of cytotoxic T lymphocytes
           (CTLs) and natural killer (NK) cells; upon target-cell recognition, CTL/NK cells
           degranulate → perforin polymerises in the target-cell membrane → cylindrical pores →
           entry of granzymes (A and B) → apoptosis of target cell; perforin is also required for
           efficient killing of antigen-presenting cells (APCs) including macrophages, thereby
           terminating immune responses (the 'APC-killing shutdown' hypothesis);
         Without perforin: CTLs and NK cells cannot kill infected macrophages that are over-activated
           by interferon-γ → macrophages continue to proliferate and phagocytose erythrocytes,
           leucocytes, platelets, and haematopoietic precursors (haemophagocytosis);
           simultaneously, unchecked CTL activation → massive cytokine storm (IFN-γ, TNF, IL-6,
           IL-10, IL-18 — the HLH cytokine signature);
         HLH DIAGNOSTIC CRITERIA (HLH-2004): fever + splenomegaly + bicytopenia + hypertriglyceridaemia
           or hypofibrinogenaemia + haemophagocytosis on BM biopsy + NK-cell activity low/absent +
           ferritin ≥500 μg/L + elevated sCD25 (≥2400 U/mL); 5/8 criteria required;
         CRITICAL FERRITIN THRESHOLD: ferritin >10,000 μg/L — nearly PATHOGNOMONIC for HLH (few
           other conditions reach this level); ferritin >500,000 μg/L = extremely high HLH likelihood;
         NK-CELL DEGRANULATION ASSAY: CD107a surface mobilisation after K562 target stimulation —
           absent/severely reduced NK CD107a = PATHOGNOMONIC of FHL granule secretion defect;
           NOT gene-specific (all FHL2-5 and Griscelli/Chediak-Higashi reduce CD107a);
         Treatment: HLH-2004 protocol (dexamethasone + etoposide + ciclosporin) → then HSCT —
           HSCT IS THE ONLY CURE; without HSCT virtually all patients relapse; early HSCT (CR1)
           gives best outcomes (>60% survival); DEF1 emapalumab (anti-IFN-γ) as bridge to HSCT;
         Variant: p.Ala91Val (PRF1 c.272C>T) — functional hypomorphic variant; associated with
           late-onset HLH or haematological malignancy-triggered HLH in heterozygous/homozygous
           adults; NOT fully penetrant; should be tested in unexplained late-onset HLH;
         seed SEED_BASE+0).
UNC13D (unc-13 homolog D = MUNC13-4; 1090aa; 17q25.3; AR;
         FHL type 3 (FHL3) — second most common FHL, ~20–25% of all FHL;
         MUNC13-4 (Mammalian Unc-13 homolog D) is a PRIMING FACTOR for secretory/lytic granules —
           it primes lytic granules by facilitating ATP-dependent transition of vesicle-docking
           SNARE complexes from a 'closed' (autoinhibited) to an 'open' (fusogenic) state;
           without MUNC13-4, granules dock at the immunological synapse but CANNOT fuse with
           the plasma membrane → complete failure of degranulation → perforin and granzymes
           trapped inside vesicles → NO CTL/NK cytotoxicity;
         CD107a MOBILISATION ASSAY (LAMP-1): CD107a is a lysosome-associated membrane glycoprotein
           exposed on the NK/CTL surface upon granule fusion; NK CD107a mobilisation after K562
           co-culture: absent in all FHL degranulation defects (UNC13D, STX11, STXBP2) —
           mechanistically PATHOGNOMONIC of the vesicle-priming/docking/fusion step; differentiates
           from PRF1 deficiency (perforin expression absent by flow/western);
         PERFORIN EXPRESSION: perforin is PRESENT (normal amount in granules) in UNC13D/FHL3 —
           a key differentiator from FHL2/PRF1 (perforin absent);
         Treatment: identical HLH-2004 protocol → HSCT; emapalumab bridge; outcome similar to FHL2;
         seed SEED_BASE+1).
STX11  (syntaxin 11; 287aa; 6q24.2; AR;
         FHL type 4 (FHL4) — syntaxin 11 vesicle-docking SNARE protein;
         Syntaxin 11 is a SNARE (soluble NSF attachment protein receptor) protein of the
           target-SNARE (t-SNARE) family located on the plasma membrane and late endosomal
           membranes; it forms the final docking complex with vesicle-SNAREs (v-SNAREs) that
           tethers lytic granules immediately before fusion; requires STXBP2/MUNC18-2 as its
           chaperone/regulator (STX11-STXBP2 pair);
         Without STX11: lytic granules are primed and arrive at the immunological synapse
           but the final docking step before membrane fusion is incomplete →
           degranulation severely impaired → same HLH phenotype as FHL2/3;
         FOUNDER EFFECT: most FHL4 cases described in Kurdish and Turkish families;
           p.Arg4Gly_splice variant c.1082+1G>C (intron 1 splice) is the most common Kurdish/Turkish
           founder allele; when HLH presents in Middle Eastern consanguineous families → test STX11;
         STXBP2 DEPENDENCY: STX11 requires STXBP2 for proper folding and stabilisation; STX11
           levels are consequently reduced in STXBP2-deficient cells (interdependency);
         Treatment: HLH-2004 + HSCT as for FHL2/3;
         seed SEED_BASE+2).
STXBP2 (syntaxin-binding protein 2 = MUNC18-2; 592aa; 19p13.2; AR;
         FHL type 5 (FHL5) — syntaxin-binding protein 2 deficiency;
         MUNC18-2 (Mammalian uncoordinated-18 homolog 2) is the SM (Sec1/Munc18) family
           chaperone protein that binds syntaxin 11 (STX11); MUNC18-2 maintains STX11 in
           a stable closed-form conformation competent for SNARE complex assembly, and also
           facilitates the Rab-effector-mediated tethering step prior to SNARE zippering;
         Without STXBP2: STX11 is conformationally unstable and rapidly degraded → markedly
           reduced STX11 protein levels even when STX11 gene is intact (a distinguishing
           diagnostic clue: STXBP2-deficient NK cells show reduced STX11 protein by western blot);
         CLINICAL FEATURE: the most variable FHL subtype — some STXBP2 patients have intestinal
           manifestations (inflammatory bowel disease-like picture, villous atrophy) in addition to
           classic HLH features; this is the most distinctive clinical clue for STXBP2 vs other FHL;
         NEONATAL HLH: STXBP2 deficiency is one of the most common causes of neonatal HLH
           (presenting at birth or within first weeks) — early presentation is a distinguishing feature;
         Treatment: HLH-2004 + HSCT; IBD-like features may require ongoing GI management post-HSCT;
         seed SEED_BASE+3).
LYST   (lysosomal trafficking regulator; 3801aa; 1q42.3; AR;
         Chediak-Higashi syndrome (CHS) — lysosomal trafficking regulator deficiency;
         LYST is a BEACH (Beige and Chediak-Higashi) and WD-repeat domain protein involved in
           regulating late endosome/lysosome trafficking, specifically the fission and sizing of
           lysosome-related organelles (LROs); without LYST, LROs in ALL cell types fail to undergo
           proper fission → form pathologically ENLARGED organelles; in CTLs/NK cells the lytic
           granules are too large to reach the immunological synapse effectively → cytotoxic defect;
           in melanocytes → giant melanosomes → PARTIAL OCULOCUTANEOUS ALBINISM; in neutrophils →
           giant granules → impaired bacterial killing despite normal phagocytosis; in platelets →
           dense-granule defect → bleeding tendency;
         PATHOGNOMONIC: GIANT GRANULES IN PERIPHERAL BLOOD NEUTROPHILS on light microscopy (May-
           Grünwald-Giemsa stain) — PATHOGNOMONIC for CHS; giant granules also seen in NK/CTL on
           blood film or bone marrow; this is the single most important diagnostic clue;
         PARTIAL ALBINISM: characteristic silver-grey hair with metallic sheen + reduced iris
           pigmentation + nystagmus + photophobia; less complete than RAB27A/Griscelli type 1;
         ACCELERATED PHASE: 80–90% of untreated CHS patients develop HLH 'accelerated phase'
           (uncontrolled macrophage activation) before age 10; HSCT is the ONLY cure and must be
           done IN REMISSION (before or after treating the acute accelerated phase);
         NEUROLOGICAL DEGENERATION: even after HSCT, patients develop progressive neurological
           deterioration over decades (spinocerebellar degeneration) — the haematological cure does
           not prevent the neurological component (giant neuronal granules persist);
         Treatment: HSCT IN REMISSION (essential); neurological monitoring lifelong even post-HSCT;
         seed SEED_BASE+4).
RAB27A (RAS-related in brain 27A; 221aa; 15q21.3; AR;
         Griscelli syndrome type 2 (GS2) — RAB27A effector deficiency causing partial albinism + HLH;
         RAB27A is a small GTPase of the RAB family that acts as a molecular switch regulating
           the tethering of secretory/lytic granules at the plasma membrane; in its GTP-bound form,
           RAB27A recruits two key effectors: MELANOPHILIN (MLPH) in melanocytes (for melanosome
           transfer to keratinocytes) and MUNC13-4 (via JFC1/SLP4) in CTLs/NK cells (for lytic
           granule secretion); without RAB27A the effectors cannot be recruited → organelle tethering
           step fails;
         TWO-COMPARTMENT PHENOTYPE: (1) MELANOCYTE: melanosomes normally produced and sized but
           cannot be transferred to keratinocytes → large melanin clusters accumulate in melanocytes
           → SILVERY-METALLIC HAIR (primary diagnostic feature); skin pigmentation may be nearly
           normal (melanin present but trapped); (2) CTL/NK: lytic granules cannot tether/dock →
           degranulation absent → HLH identical to FHL;
         KEY DISTINCTION FROM CHEDIAK-HIGASHI (LYST): Griscelli type 2 does NOT produce giant
           granules — blood film shows NORMAL granule size; the silver hair appearance may be
           macroscopically similar but the molecular basis is opposite (LYST = abnormal sizing;
           RAB27A = abnormal tethering of normally-sized granules); electron microscopy of hair shaft:
           large pigment clumps in medulla (Griscelli) vs small peripheral clumps (normal);
         GRISCELLI SYNDROME CLASSIFICATION: Type 1 (MYO5A): neurological phenotype only, NO HLH,
           NO immune defect — DIFFERENT gene; Type 2 (RAB27A): immune defect + HLH ± partial
           albinism — the clinically dangerous form; Type 3 (MLPH): partial albinism only,
           NO immune defect; only GS TYPE 2 has HLH risk;
         Treatment: HLH-2004 (if HLH has occurred) + HSCT — HSCT is curative for immune defect;
           after HSCT the silvery hair persists (hair follicle melanocytes not replaced by HSCT);
         seed SEED_BASE+5).
FAS    (Fas cell surface death receptor = CD95 = TNFRSF6; 335aa; 10q23.31; AD (type Ia) or AR (type Ib);
         Autoimmune lymphoproliferative syndrome type Ia/Ib (ALPS-FAS) — most common ALPS;
         FAS (CD95) is the prototypical death receptor of the TNFR superfamily; in the FAS-FASL
           pathway, FASL (expressed on activated T cells) binds FAS → FAS trimerises → recruits
           FADD (via death domain) → procaspase-8 activation → apoptosis of the FAS-expressing cell;
           this pathway is essential for 'lymphocyte homeostasis' — clearance of activated effector
           lymphocytes after antigen has been eliminated (the 'restimulation-induced cell death' RICD);
         Without FAS: lymphocytes accumulate after immune activation → LYMPHOPROLIFERATION
           (massive lymphadenopathy + splenomegaly); accumulated T cells undergo abnormal terminal
           differentiation into double-negative T cells (DNT cells: CD3+TCRαβ+CD4-CD8-B220+) —
           DNT CELLS ARE THE PATHOGNOMONIC BIOMARKER FOR ALPS;
         PATHOGNOMONIC: ELEVATED DNT CELLS (>1.5% of total lymphocytes, or >2.5% of CD3+ T cells)
           in peripheral blood + splenomegaly + lymphadenopathy + autoimmunity = ALPS DIAGNOSTIC;
           DNT cell assay is the single most important first investigation;
         AUTOIMMUNITY: ALPS hallmark — particularly EVANS SYNDROME (autoimmune haemolytic anaemia +
           immune thrombocytopenia simultaneously) ± autoimmune neutropenia; elevated FasL-mediated
           ADCC against blood cells; antinuclear antibodies, anti-dsDNA in some;
         FAS MUTATION TYPES: AD type Ia — heterozygous mutations predominantly in the intracellular
           death domain (exons 7–9) — dominant-negative effect on FAS trimer; incomplete penetrance
           (65–70% of carriers develop ALPS); AR type Ib — rare biallelic mutations → more severe;
         MALIGNANCY RISK: 10–15× increased risk of lymphoma (Hodgkin and non-Hodgkin) — lymphoma
           surveillance mandatory; a patient with ALPS and unexplained lymph node growth needs
           urgent biopsy (cannot assume lymphoma is just ALPS);
         TREATMENT: not all ALPS patients require treatment; symptomatic cytopenias → sirolimus
           (mTOR inhibitor, first-line for immune cytopenias in ALPS — superior to steroids);
           mycophenolate mofetil; splenectomy avoided if possible (increases lymphoma risk);
           HSCT reserved for severe cases with malignancy transformation;
         seed SEED_BASE+6).
FOXP3  (forkhead box P3; 431aa; Xp11.22; XL;
         IPEX syndrome — Immune dysregulation Polyendocrinopathy Enteropathy X-linked;
         FOXP3 is the master transcription factor for REGULATORY T CELLS (Tregs; CD4+CD25+FOXP3+);
           FOXP3 expression is both the developmental requirement and the defining molecular marker
           of the thymic Treg lineage; Treg cells suppress effector T cell responses by secreting
           IL-10, TGF-β, and IL-35, expressing CTLA-4, and consuming IL-2;
         Without FOXP3: no functional Tregs → UNRESTRAINED AUTOIMMUNE RESPONSES AGAINST MULTIPLE
           ORGANS → the IPEX TRIAD:
           (1) IMMUNE POLYENDOCRINOPATHY: Type 1 diabetes mellitus presenting in the NEONATAL
               PERIOD (birth–6 weeks) — PATHOGNOMONIC; the only condition causing T1DM at birth
               (vs autoimmune T1DM which is age 5+); also autoimmune thyroid disease, adrenal failure;
           (2) ENTEROPATHY: severe intractable watery diarrhoea (villous atrophy similar to coeliac
               disease but refractory to gluten-free diet); can cause failure-to-thrive and fatal
               malnutrition in infancy;
           (3) ECZEMA: severe atopic dermatitis from birth; ichthyosiform rash;
         X-LINKED: exclusively males affected; females are obligate carriers (one carrier parent
           may have mild autoimmune thyroid disease);
         SEVERITY SPECTRUM: classic severe neonatal triad → fatal within months without treatment;
           milder variants (hypomorphic FOXP3 mutations with residual FOXP3 function) → later onset,
           single organ manifestation, or predominantly IBD-like enteropathy;
         NEONATAL T1DM: the defining diagnostic clue for IPEX — a male neonate with confirmed T1DM
           should have FOXP3 sequenced immediately; other neonatal T1DM causes (KCNJ11, ABCC8, INS,
           GCK, EIF2AK3) do not have the associated polyendocrinopathy + enteropathy + eczema triad;
         TREATMENT: stabilisation with immune suppression: cyclosporin or tacrolimus (calcineurin
           inhibitors) to suppress T cell activity — reduces but does not eliminate autoimmune
           destruction; sirolimus as alternative; HSCT IS THE ONLY CURE and should be pursued urgently
           after initial stabilisation in severe cases; post-HSCT outcomes improving with HLA-matched
           sibling donors;
         seed SEED_BASE+7).
"""

import random
from typing import Any

SEED_BASE = 3894  # seeds 3894-3901

GENES = ["PRF1", "UNC13D", "STX11", "STXBP2", "LYST", "RAB27A", "FAS", "FOXP3"]

GENE_META = {
    "PRF1":   {"locus": "10q22.1", "aa": 555,  "kda": 61.0,  "inh": "AR",    "syndrome": "FHL2 — Perforin-Deficiency-Most-Common-FHL-30-40pct-Ferritin-GT10000-PATHOGNOMONIC-HSCT-Only-Cure",       "mim_gene": 170280, "mim_disease": 267700},
    "UNC13D": {"locus": "17q25.3", "aa": 1090, "kda": 122.0, "inh": "AR",    "syndrome": "FHL3-MUNC13-4-Vesicle-Priming-CD107a-Mobilisation-Absent-PATHOGNOMONIC-Perforin-PRESENT",              "mim_gene": 608897, "mim_disease": 608898},
    "STX11":  {"locus": "6q24.2",  "aa": 287,  "kda": 33.0,  "inh": "AR",    "syndrome": "FHL4-Syntaxin-11-Kurdish-Turkish-Founder-Vesicle-Docking-SNARE-STXBP2-Dependent",                       "mim_gene": 605014, "mim_disease": 603553},
    "STXBP2": {"locus": "19p13.2", "aa": 592,  "kda": 67.0,  "inh": "AR",    "syndrome": "FHL5-MUNC18-2-STX11-Chaperone-IBD-Like-Enteropathy-Neonatal-HLH-DISTINGUISHING",                        "mim_gene": 601717, "mim_disease": 613101},
    "LYST":   {"locus": "1q42.3",  "aa": 3801, "kda": 430.0, "inh": "AR",    "syndrome": "Chediak-Higashi-GIANT-GRANULES-Neutrophils-PATHOGNOMONIC-Partial-Albinism-Neurodegeneration-Persists-Post-HSCT", "mim_gene": 606897, "mim_disease": 214500},
    "RAB27A": {"locus": "15q21.3", "aa": 221,  "kda": 25.0,  "inh": "AR",    "syndrome": "Griscelli-Type2-SILVERY-HAIR-PATHOGNOMONIC-No-Giant-Granules-vs-LYST-HLH-HSCT-Curative",               "mim_gene": 603868, "mim_disease": 607624},
    "FAS":    {"locus": "10q23.31","aa": 335,  "kda": 36.0,  "inh": "AD/AR", "syndrome": "ALPS-TypeIa-DNT-Cells-Elevated-PATHOGNOMONIC-Evans-Syndrome-Lymphoma-10-15x-Sirolimus",                 "mim_gene": 134637, "mim_disease": 601859},
    "FOXP3":  {"locus": "Xp11.22", "aa": 431,  "kda": 47.0,  "inh": "XL",    "syndrome": "IPEX-Neonatal-T1DM-PATHOGNOMONIC-Triad-Enteropathy-Eczema-Treg-Absent-HSCT-Only-Cure",                  "mim_gene": 300292, "mim_disease": 304790},
}

DEFINITIONS = {
    "PRF1":   "Perforin-1 (555aa / 61kDa / 10q22.1 / AR). Pore-forming cytotoxic granule protein. LOF → FHL2 (most common FHL, 30–40%). Without perforin: CTL/NK cannot kill APCs → uncontrolled macrophage activation → HLH. Ferritin >10,000 µg/L near PATHOGNOMONIC. NK CD107a absent (degranulation assay). Perforin protein absent (flow/western). HLH-2004 protocol → HSCT (only cure). p.Ala91Val hypomorphic variant: late-onset/malignancy-associated HLH.",
    "UNC13D": "MUNC13-4 priming factor (1090aa / 122kDa / 17q25.3 / AR). Vesicle priming SNARE regulator. LOF → FHL3 (~20–25% FHL). Granules dock but CANNOT fuse. Perforin PRESENT (key distinction from FHL2/PRF1). CD107a absent after K562 stimulation PATHOGNOMONIC for degranulation defect class. HLH-2004 + HSCT. Emapalumab (anti-IFN-γ) as bridge to HSCT.",
    "STX11":  "Syntaxin-11 t-SNARE (287aa / 33kDa / 6q24.2 / AR). Terminal granule docking SNARE; STX11-STXBP2 pair. LOF → FHL4. Kurdish/Turkish founder allele most common. Vesicle docking step fails. CD107a absent. STX11 protein reduced in STXBP2-deficient cells (interdependency clue). HLH-2004 + HSCT.",
    "STXBP2": "MUNC18-2 SM-protein chaperone (592aa / 67kDa / 19p13.2 / AR). STX11 chaperone; required for syntaxin stability. LOF → FHL5. Distinctive: IBD-like enteropathy (villous atrophy/colitis) + neonatal HLH presentations. STX11 protein levels reduced even when STX11 gene intact. HLH-2004 + HSCT; GI monitoring.",
    "LYST":   "Lysosomal trafficking regulator (3801aa / 430kDa / 1q42.3 / AR). BEACH-domain LRO fission regulator. LOF → Chediak-Higashi. GIANT GRANULES in neutrophils on blood film PATHOGNOMONIC. Partial albinism (silver-grey hair + hypopigmentation). 80–90% develop HLH accelerated phase. HSCT cures immune/HLH component. Neurological degeneration (spinocerebellar) persists and progresses post-HSCT — the unique long-term burden.",
    "RAB27A": "RAB27A GTPase (221aa / 25kDa / 15q21.3 / AR). Granule tethering effector recruiter. LOF → Griscelli type 2. SILVERY METALLIC HAIR PATHOGNOMONIC (large melanin clumps in medullary hair shaft). No giant granules (vs LYST/CHS). Degranulation absent → HLH. HSCT curative for immune defect; silvery hair persists post-HSCT. GS type 1 (MYO5A) = neuro only; GS type 3 (MLPH) = albinism only — only GS2 has HLH.",
    "FAS":    "FAS death receptor CD95 (335aa / 36kDa / 10q23.31 / AD type Ia, AR type Ib). RICD apoptosis signal. LOF → ALPS. DNT cells (CD3+TCRαβ+CD4–CD8–B220+) >1.5% of lymphocytes PATHOGNOMONIC. Evans syndrome (AIHA + ITP). Lymphadenopathy + splenomegaly. Lymphoma risk 10–15×. Sirolimus first-line for cytopenias. Avoid splenectomy. AD incomplete penetrance (~65%). HSCT for severe/transformed disease.",
    "FOXP3":  "FOXP3 Treg master TF (431aa / 47kDa / Xp11.22 / X-linked). Without FOXP3: no functional Tregs → unrestrained autoimmunity. IPEX triad: neonatal T1DM (birth–6 weeks) PATHOGNOMONIC + intractable enteropathy + eczema. Males only. T1DM before age 6 weeks = test FOXP3 immediately. Calcineurin inhibitors (ciclosporin/tacrolimus) to stabilise. HSCT only cure — pursue urgently in severe cases.",
}


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def _gene_cohort(gene: str, seed: int) -> dict[str, Any]:
    rng = _rng(seed)
    n = 40
    meta = GENE_META[gene]

    if gene == "PRF1":
        mean_onset_months = rng.gauss(9, 4); mean_dx_lag = rng.gauss(1.2, 0.5)
        hlh_n = rng.randint(38, 40); ferritin_gt10k_n = rng.randint(32, 38)
        cd107a_absent_n = rng.randint(36, 40); perforin_absent_n = rng.randint(36, 40)
        silver_hair_n = 0; giant_granules_n = 0; dnt_elevated_n = 0
        evans_n = 0; neonatal_t1dm_n = 0; enteropathy_n = 0
        hsct_performed_n = rng.randint(28, 36); relapse_no_hsct_n = rng.randint(24, 32)
        pathognomonic = "Ferritin >10,000 µg/L + absent NK perforin expression (flow cytometry) + absent CD107a mobilisation — FHL2 PATHOGNOMONIC triad; HSCT only cure"
    elif gene == "UNC13D":
        mean_onset_months = rng.gauss(8, 4); mean_dx_lag = rng.gauss(1.4, 0.6)
        hlh_n = rng.randint(37, 40); ferritin_gt10k_n = rng.randint(30, 36)
        cd107a_absent_n = rng.randint(36, 40); perforin_absent_n = rng.randint(2, 6)
        silver_hair_n = 0; giant_granules_n = 0; dnt_elevated_n = 0
        evans_n = 0; neonatal_t1dm_n = 0; enteropathy_n = rng.randint(4, 10)
        hsct_performed_n = rng.randint(26, 34); relapse_no_hsct_n = rng.randint(22, 30)
        pathognomonic = "CD107a absent + PERFORIN PRESENT (key FHL3 vs FHL2 distinction); vesicle priming defect; HLH-2004 + HSCT"
    elif gene == "STX11":
        mean_onset_months = rng.gauss(10, 5); mean_dx_lag = rng.gauss(1.6, 0.7)
        hlh_n = rng.randint(36, 40); ferritin_gt10k_n = rng.randint(28, 34)
        cd107a_absent_n = rng.randint(34, 40); perforin_absent_n = rng.randint(2, 6)
        silver_hair_n = 0; giant_granules_n = 0; dnt_elevated_n = 0
        evans_n = 0; neonatal_t1dm_n = 0; enteropathy_n = rng.randint(4, 10)
        hsct_performed_n = rng.randint(24, 32); relapse_no_hsct_n = rng.randint(20, 28)
        pathognomonic = "Kurdish/Turkish consanguineous family + HLH + CD107a absent; STX11 SNARE docking defect; STXBP2 interdependency (check both)"
    elif gene == "STXBP2":
        mean_onset_months = rng.gauss(3, 2); mean_dx_lag = rng.gauss(0.8, 0.3)
        hlh_n = rng.randint(36, 40); ferritin_gt10k_n = rng.randint(28, 36)
        cd107a_absent_n = rng.randint(34, 40); perforin_absent_n = rng.randint(2, 6)
        silver_hair_n = 0; giant_granules_n = 0; dnt_elevated_n = 0
        evans_n = 0; neonatal_t1dm_n = 0; enteropathy_n = rng.randint(18, 28)
        hsct_performed_n = rng.randint(26, 34); relapse_no_hsct_n = rng.randint(22, 30)
        pathognomonic = "Neonatal HLH + IBD-like enteropathy (DISTINGUISHING feature vs other FHL); STX11 protein reduced even when STX11 intact; CD107a absent"
    elif gene == "LYST":
        mean_onset_months = rng.gauss(36, 18); mean_dx_lag = rng.gauss(2.5, 1.0)
        hlh_n = rng.randint(32, 38); ferritin_gt10k_n = rng.randint(24, 32)
        cd107a_absent_n = rng.randint(32, 38); perforin_absent_n = rng.randint(6, 14)
        silver_hair_n = rng.randint(34, 40); giant_granules_n = rng.randint(38, 40)
        dnt_elevated_n = 0; evans_n = 0; neonatal_t1dm_n = 0; enteropathy_n = rng.randint(6, 14)
        hsct_performed_n = rng.randint(24, 32); relapse_no_hsct_n = rng.randint(26, 36)
        pathognomonic = "GIANT GRANULES on neutrophil blood film PATHOGNOMONIC (May-Grünwald-Giemsa); silver-grey hair; neurological degeneration persists post-HSCT"
    elif gene == "RAB27A":
        mean_onset_months = rng.gauss(24, 12); mean_dx_lag = rng.gauss(1.8, 0.7)
        hlh_n = rng.randint(30, 38); ferritin_gt10k_n = rng.randint(22, 30)
        cd107a_absent_n = rng.randint(32, 38); perforin_absent_n = rng.randint(4, 10)
        silver_hair_n = rng.randint(36, 40); giant_granules_n = rng.randint(0, 4)
        dnt_elevated_n = 0; evans_n = 0; neonatal_t1dm_n = 0; enteropathy_n = rng.randint(4, 10)
        hsct_performed_n = rng.randint(22, 30); relapse_no_hsct_n = rng.randint(18, 26)
        pathognomonic = "SILVERY METALLIC HAIR PATHOGNOMONIC + NO giant granules (vs LYST/CHS); hair EM: large medullary pigment clumps; GS type 2 only has HLH (not GS1/3)"
    elif gene == "FAS":
        mean_onset_months = rng.gauss(18, 12); mean_dx_lag = rng.gauss(3.5, 1.5)
        hlh_n = rng.randint(4, 12); ferritin_gt10k_n = rng.randint(2, 8)
        cd107a_absent_n = rng.randint(2, 6); perforin_absent_n = rng.randint(2, 6)
        silver_hair_n = 0; giant_granules_n = 0
        dnt_elevated_n = rng.randint(36, 40); evans_n = rng.randint(22, 30)
        neonatal_t1dm_n = 0; enteropathy_n = rng.randint(6, 14)
        hsct_performed_n = rng.randint(6, 14); relapse_no_hsct_n = rng.randint(4, 10)
        pathognomonic = "DNT cells >1.5% + Evans syndrome + lymphadenopathy + splenomegaly PATHOGNOMONIC ALPS; lymphoma 10–15×; sirolimus first-line; avoid splenectomy"
    elif gene == "FOXP3":
        mean_onset_months = rng.gauss(0.5, 0.3); mean_dx_lag = rng.gauss(0.3, 0.1)
        hlh_n = rng.randint(8, 16); ferritin_gt10k_n = rng.randint(4, 12)
        cd107a_absent_n = rng.randint(4, 10); perforin_absent_n = rng.randint(4, 10)
        silver_hair_n = 0; giant_granules_n = 0; dnt_elevated_n = 0
        evans_n = rng.randint(10, 20); neonatal_t1dm_n = rng.randint(34, 40)
        enteropathy_n = rng.randint(32, 40)
        hsct_performed_n = rng.randint(18, 26); relapse_no_hsct_n = rng.randint(22, 32)
        pathognomonic = "NEONATAL T1DM (birth–6 weeks) PATHOGNOMONIC + intractable enteropathy + eczema; males only; FOXP3 sequencing immediately for any male neonate with T1DM"
    else:
        mean_onset_months = rng.gauss(12, 6); mean_dx_lag = rng.gauss(2.0, 0.8)
        hlh_n = rng.randint(20, 30); ferritin_gt10k_n = rng.randint(14, 22)
        cd107a_absent_n = rng.randint(16, 24); perforin_absent_n = rng.randint(10, 18)
        silver_hair_n = 0; giant_granules_n = 0; dnt_elevated_n = 0
        evans_n = 0; neonatal_t1dm_n = 0; enteropathy_n = rng.randint(6, 14)
        hsct_performed_n = rng.randint(16, 24); relapse_no_hsct_n = rng.randint(12, 20)
        pathognomonic = "Immune dysregulation defect — see gene-specific entry"

    return {
        "gene": gene,
        "n": n,
        "locus": meta["locus"],
        "aa": meta["aa"],
        "kda": meta["kda"],
        "inheritance": meta["inh"],
        "syndrome": meta["syndrome"],
        "mean_onset_months": round(abs(mean_onset_months), 1),
        "mean_dx_lag_years": round(abs(mean_dx_lag), 2),
        "hlh_n": hlh_n,
        "hlh_pct": round(hlh_n / n * 100, 1),
        "ferritin_gt10k_n": ferritin_gt10k_n,
        "ferritin_gt10k_pct": round(ferritin_gt10k_n / n * 100, 1),
        "cd107a_absent_n": cd107a_absent_n,
        "cd107a_absent_pct": round(cd107a_absent_n / n * 100, 1),
        "perforin_absent_n": perforin_absent_n,
        "perforin_absent_pct": round(perforin_absent_n / n * 100, 1),
        "silver_hair_n": silver_hair_n,
        "silver_hair_pct": round(silver_hair_n / n * 100, 1),
        "giant_granules_n": giant_granules_n,
        "giant_granules_pct": round(giant_granules_n / n * 100, 1),
        "dnt_elevated_n": dnt_elevated_n,
        "dnt_elevated_pct": round(dnt_elevated_n / n * 100, 1),
        "evans_n": evans_n,
        "evans_pct": round(evans_n / n * 100, 1),
        "neonatal_t1dm_n": neonatal_t1dm_n,
        "neonatal_t1dm_pct": round(neonatal_t1dm_n / n * 100, 1),
        "enteropathy_n": enteropathy_n,
        "enteropathy_pct": round(enteropathy_n / n * 100, 1),
        "hsct_performed_n": hsct_performed_n,
        "hsct_performed_pct": round(hsct_performed_n / n * 100, 1),
        "relapse_no_hsct_n": relapse_no_hsct_n,
        "relapse_no_hsct_pct": round(relapse_no_hsct_n / n * 100, 1),
        "pathognomonic": pathognomonic,
        "mim_gene": meta["mim_gene"],
        "mim_disease": meta["mim_disease"],
        "seed": seed,
    }


def generate_overview() -> dict:
    cohorts = [_gene_cohort(g, SEED_BASE + i) for i, g in enumerate(GENES)]
    total_n = sum(c["n"] for c in cohorts)
    total_hlh = sum(c["hlh_n"] for c in cohorts)
    total_ferritin_gt10k = sum(c["ferritin_gt10k_n"] for c in cohorts)
    total_cd107a_absent = sum(c["cd107a_absent_n"] for c in cohorts)
    total_silver_hair = sum(c["silver_hair_n"] for c in cohorts)
    total_giant_granules = sum(c["giant_granules_n"] for c in cohorts)
    total_dnt = sum(c["dnt_elevated_n"] for c in cohorts)
    total_evans = sum(c["evans_n"] for c in cohorts)
    total_neonatal_t1dm = sum(c["neonatal_t1dm_n"] for c in cohorts)
    total_enteropathy = sum(c["enteropathy_n"] for c in cohorts)
    total_hsct = sum(c["hsct_performed_n"] for c in cohorts)
    return {
        "atlas": "Hereditary-Immune-Dysregulation-HLH-ALPS-Atlas",
        "subtitle": "Diseases of Immune Dysregulation — Familial HLH / Griscelli / Chediak-Higashi / ALPS / IPEX (IUIS PID Category 4)",
        "total_patients": total_n,
        "genes_covered": GENES,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "aggregate_stats": {
            "hlh_pct": round(total_hlh / total_n * 100, 1),
            "ferritin_gt10k_pct": round(total_ferritin_gt10k / total_n * 100, 1),
            "cd107a_absent_pct": round(total_cd107a_absent / total_n * 100, 1),
            "silver_hair_pct": round(total_silver_hair / total_n * 100, 1),
            "giant_granules_pct": round(total_giant_granules / total_n * 100, 1),
            "dnt_elevated_pct": round(total_dnt / total_n * 100, 1),
            "evans_pct": round(total_evans / total_n * 100, 1),
            "neonatal_t1dm_pct": round(total_neonatal_t1dm / total_n * 100, 1),
            "enteropathy_pct": round(total_enteropathy / total_n * 100, 1),
            "hsct_performed_pct": round(total_hsct / total_n * 100, 1),
        },
        "cohort_summaries": [
            {k: v for k, v in c.items() if k not in ("pathognomonic",)}
            for c in cohorts
        ],
        "cascade_testing_concepts": [
            "HLH suspected: ferritin + fibrinogen + triglycerides + NK activity + sCD25 (HLH-2004 criteria) → if ≥5/8 → immunology urgent review",
            "NK CD107a mobilisation assay (K562 target): absent → granule secretion defect → order PRF1 perforin flow/western + FHL panel (PRF1, UNC13D, STX11, STXBP2)",
            "Perforin absent: PRF1/FHL2; CD107a absent + perforin PRESENT: UNC13D/FHL3 or STX11/FHL4 or STXBP2/FHL5",
            "Silver hair + HLH → RAB27A (Griscelli type 2) vs LYST (Chediak-Higashi): blood film for giant granules (LYST = giant granules; RAB27A = normal granule size)",
            "Blood film: giant granules in neutrophils → LYST/Chediak-Higashi PATHOGNOMONIC — hair EM: peripheral small clumps",
            "Griscelli type 2 (RAB27A): hair EM shows LARGE pigment clumps in medulla; Griscelli type 1 (MYO5A): neuro only; Griscelli type 3 (MLPH): albinism only — ONLY GS2 has HLH",
            "ALPS screen: DNT cells >1.5% of lymphocytes + splenomegaly + Evans syndrome → FAS sequencing first (type Ia most common); FasL/CASP10 if FAS normal",
            "IPEX screen: male neonate + T1DM (≤6 weeks age) + enteropathy → FOXP3 sequencing IMMEDIATELY; calcineurin inhibitor bridge; urgent HSCT referral",
            "All FHL types: HLH-2004 protocol (dexamethasone + etoposide + ciclosporin) → HSCT as early as possible (CR1 best survival)",
            "Emapalumab (anti-IFN-γ): FDA-approved bridge to HSCT for primary HLH; blocks IFN-γ-driven macrophage activation while HSCT arranged",
            "STXBP2/FHL5 unique: IBD-like enteropathy + neonatal HLH onset — test STXBP2 in any neonatal HLH with GI manifestations",
            "LYST post-HSCT: neurological degeneration (spinocerebellar ataxia + peripheral neuropathy) CONTINUES despite haematological cure — lifelong neurology follow-up mandatory",
            "ALPS-FAS lymphoma surveillance: annual FDG-PET or CT from diagnosis; any rapidly enlarging lymph node = urgent biopsy (cannot assume ALPS alone)",
            "FOXP3/IPEX: sirolimus preferred over steroids for immune suppression bridge; HSCT with HLA-identical sibling preferred if available; FOXP3 expression by flow on CD4+CD25hi Tregs diagnostic aid",
        ],
        "iuis_category": "Category 4 — Diseases of Immune Dysregulation (IUIS 2022)",
    }


def generate_breakdown() -> dict:
    cohorts = [_gene_cohort(g, SEED_BASE + i) for i, g in enumerate(GENES)]
    return {
        "atlas": "Hereditary-Immune-Dysregulation-HLH-ALPS-Atlas",
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "gene_cohorts": cohorts,
        "diagnostic_groups": {
            "fhl_granule_secretion": {
                "genes": ["PRF1", "UNC13D", "STX11", "STXBP2"],
                "mechanism": "CTL/NK lytic granule secretion pathway defects — pore-forming (PRF1) vs priming (UNC13D) vs docking (STX11) vs stabilisation (STXBP2)",
                "shared_feature": "HLH + absent NK CD107a mobilisation; differentiate by perforin expression (PRF1 absent; others present)",
                "key_ddx": "FHL2 (PRF1 absent perforin) vs FHL3/4/5 (perforin present, CD107a absent); STXBP2 unique: IBD-like enteropathy + neonatal onset",
            },
            "lysosome_related_organelle": {
                "genes": ["LYST", "RAB27A"],
                "mechanism": "Lysosome-related organelle (LRO) trafficking defects — sizing (LYST) vs tethering (RAB27A)",
                "shared_feature": "Partial albinism + HLH + NK cytotoxic defect",
                "key_ddx": "LYST: giant granules on blood film + silver hair + neurodegeneration; RAB27A: silver hair + NO giant granules + GS2 only (not GS1/GS3) has HLH",
            },
            "lymphocyte_homeostasis": {
                "genes": ["FAS"],
                "mechanism": "Restimulation-induced cell death (RICD) defect — FAS apoptosis pathway failure → lymphocyte accumulation",
                "shared_feature": "Lymphoproliferation + autoimmunity + elevated DNT cells",
                "key_ddx": "ALPS vs FHL: DNT cells elevated in ALPS (not FHL); Evans syndrome in ALPS not classic HLH; ferritin usually <10,000 in ALPS",
            },
            "regulatory_t_cell": {
                "genes": ["FOXP3"],
                "mechanism": "Treg development and function defect — FOXP3 absent → no Tregs → unrestrained effector T cells",
                "shared_feature": "Multi-organ autoimmunity from birth",
                "key_ddx": "IPEX vs other neonatal T1DM: only IPEX has TRIAD of T1DM + enteropathy + eczema from neonatal period; X-linked (males only)",
            },
        },
        "pathognomonic_features": {
            "PRF1_FHL2": "Ferritin >10,000 + NK perforin protein absent + CD107a absent",
            "UNC13D_FHL3": "HLH + CD107a absent + PERFORIN PRESENT (vs FHL2)",
            "STX11_FHL4": "HLH + Kurdish/Turkish founder + CD107a absent + STX11–STXBP2 interdependency",
            "STXBP2_FHL5": "Neonatal HLH + IBD-like enteropathy/villous atrophy (unique GI manifestation)",
            "LYST_CHS": "GIANT GRANULES in neutrophils on peripheral blood film (visible on light microscopy MGG stain)",
            "RAB27A_GS2": "SILVERY METALLIC HAIR + no giant granules (blood film normal) + HLH",
            "FAS_ALPS": "DNT cells >1.5% lymphocytes + Evans syndrome + splenomegaly + lymphadenopathy",
            "FOXP3_IPEX": "Neonatal T1DM (birth–6 weeks) + enteropathy + eczema in male infant",
        },
    }


def generate_definitions() -> dict:
    return {
        "atlas": "Hereditary-Immune-Dysregulation-HLH-ALPS-Atlas",
        "definitions": DEFINITIONS,
        "gene_metadata": GENE_META,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "key_distinguishers": {
            "FHL2_vs_FHL3_5": "PRF1/FHL2: perforin protein absent; UNC13D/STX11/STXBP2 FHL3-5: CD107a absent but perforin PRESENT — single most important differentiator on flow",
            "LYST_vs_RAB27A": "Both: silver hair + HLH; LYST: GIANT GRANULES on blood film (PATHOGNOMONIC) + neurodegeneration persists post-HSCT; RAB27A: normal granule size + no neurodegeneration",
            "GS_type1_vs_2_vs_3": "MYO5A/GS1: neuro only, NO HLH; RAB27A/GS2: HLH + partial albinism — the only dangerous type; MLPH/GS3: albinism only — no HLH",
            "STXBP2_unique": "FHL5/STXBP2: IBD-like enteropathy + neonatal onset — the most GI-prominent FHL subtype; also: STX11 protein levels reduced in STXBP2-deficient cells",
            "FHL_vs_ALPS": "FHL: HLH picture, ferritin usually >10,000, CD107a absent; ALPS/FAS: DNT cells pathognomonic, Evans syndrome, lymphoma risk, ferritin usually <10,000",
            "IPEX_vs_other_neonatal_T1DM": "FOXP3/IPEX: T1DM + enteropathy + eczema TRIAD, males only; KCNJ11/ABCC8: T1DM only, no triad, sulphonylurea responsive; EIF2AK3: T1DM + skeletal dysplasia (Wolcott-Rallison)",
            "ALPS_lymphoma_risk": "ALPS: 10–15× lymphoma risk — NEVER assume enlarging nodes are just ALPS lymphoproliferation; mandatory annual surveillance; urgent biopsy if rapid growth",
            "HLH_2004_all_FHL": "All FHL types (PRF1, UNC13D, STX11, STXBP2, LYST, RAB27A): same HLH-2004 protocol (dexamethasone + etoposide + ciclosporin) → HSCT; diagnosis NOT treatment changes between subtypes",
            "Emapalumab_bridge": "Anti-IFN-γ (emapalumab) FDA-approved for primary HLH refractory to HLH-2004 — bridges to HSCT without adding myelosuppression",
            "LYST_neuro_post_HSCT": "Chediak-Higashi HSCT cures HLH/immune component ONLY — spinocerebellar degeneration continues and progresses post-HSCT; families must be explicitly counselled this is a SEPARATE, ONGOING problem",
            "FOXP3_HSCT_urgency": "IPEX/FOXP3 HSCT: should be pursued urgently once stabilised (calcineurin inhibitor bridge); every month of delay = additional organ autoimmune destruction; matched sibling best donor",
        },
        "diagnostic_pathway": [
            "Step 1: Phenotype sort — HYPERFERRITINAEMIA + fever + cytopenia + splenomegaly → HLH workup (FHL/Griscelli/CHS); Lymphadenopathy + Evans + DNT → ALPS; Neonatal male T1DM + enteropathy + eczema → IPEX",
            "Step 2: HLH path — NK CD107a mobilisation assay (K562 target): absent → degranulation defect panel; normal → consider secondary HLH or FHL2 (perforin expression flow/western separately)",
            "Step 3: CD107a absent → perforin expression: absent = PRF1/FHL2; present = order FHL3-5 panel (UNC13D, STX11, STXBP2); also check for neonatal onset (STXBP2) + GI (STXBP2)",
            "Step 4: Silver hair → blood film: GIANT GRANULES = LYST/CHS (+ hair EM peripheral small clumps); no giant granules = RAB27A/GS2 (+ hair EM large medullary clumps); confirm with gene panel",
            "Step 5: ALPS path — flow cytometry DNT cells (B220+TCRαβ+CD4–CD8–): >1.5% lymphocytes or >2.5% CD3+ = ALPS; FAS sequencing; if FAS negative → FASLG → CASP10 → somatic FAS",
            "Step 6: IPEX path — FOXP3 sequencing in any male neonate ≤6 weeks with T1DM; send simultaneously with KCNJ11/ABCC8 panel; FOXP3 expression on CD4+CD25hi Tregs by flow; if FOXP3 normal consider CTLA4/IL2RA/STAT5B IPEX-like",
            "Step 7: HLH treatment — do NOT wait for genetic result: start HLH-2004 immediately for confirmed HLH + high suspicion of primary HLH (family history, infant, recurrent); add emapalumab for refractory disease",
            "Step 8: All FHL/GS/CHS confirmed → HSCT referral URGENT; aim for CR1 transplantation; matched sibling best; if none available → MUD/MMUD with RIC conditioning; FHL without HSCT = near-certain relapse",
        ],
    }
