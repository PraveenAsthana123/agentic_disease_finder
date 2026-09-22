#!/usr/bin/env python3
"""Hereditary-Autoimmune-Polyglandular-Syndrome-Atlas — Complete 8-Gene APS / Polyendocrine-Autoimmunity Atlas
AIRE    (autoimmune regulator; 545 aa; 21q22.3; AR LOF;
         APS-1 / APECED — hypoparathyroidism + Addison + mucocutaneous candidiasis;
         AIRE autoantibody signature PATHOGNOMONIC; seed SEED_BASE+0) ·
FOXP3   (forkhead box P3; 431 aa; Xp11.23; XL LOF;
         IPEX — immunodysregulation + polyendocrinopathy + enteropathy + X-linked;
         neonatal T1DM + intractable diarrhoea; Treg absent; seed SEED_BASE+1) ·
CTLA4   (cytotoxic T-lymphocyte antigen 4; 223 aa; 2q33.2; AD LOF haploinsufficiency;
         CHAI — CTLA-4 haploinsufficiency with autoimmune infiltration;
         thyroiditis + T1DM + cytopenias + lymphoproliferation; abatacept CURATIVE;
         seed SEED_BASE+2) ·
LRBA    (LPS-responsive beige-like anchor; 2863 aa; 4q31.3; AR LOF;
         LRBA deficiency — CVID + CTLA-4 recycling defect + autoimmunity;
         abatacept CURATIVE; seed SEED_BASE+3) ·
IL2RA   (interleukin-2 receptor alpha / CD25; 272 aa; 10p15.1; AR LOF;
         CD25/IL-2Rα deficiency — IPEX-like; neonatal autoimmunity;
         IL-2 drives expansion → Tregs absent → uncontrolled autoimmunity;
         seed SEED_BASE+4) ·
STAT3   (signal transducer and activator of transcription 3; 770 aa; 17q21.2; AD GOF;
         STAT3 GOF syndrome — multiorgan autoimmunity + lymphoproliferation;
         T1DM + thyroiditis + cytopenias + short stature; JAK inhibitors effective;
         seed SEED_BASE+5) ·
STAT1   (signal transducer and activator of transcription 1; 750 aa; 2q32.2; AD GOF;
         STAT1 GOF — chronic mucocutaneous candidiasis + autoimmunity;
         thyroiditis + T1DM + aneurysm; ruxolitinib effective; seed SEED_BASE+6) ·
ITCH    (ITCH E3 ubiquitin protein ligase; 864 aa; 20q11.22; AR LOF;
         ITCH deficiency — multisystem autoimmunity + dysmorphic features;
         CTLA-4 ubiquitination impaired → Treg dysfunction; seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 2966-2973)
"""
import random

SEED_BASE = 2966

ATLAS_GENES = [
    {
        "gene": "AIRE",
        "protein": (
            "AIRE -- 21q22.3 AR LOF -- 545aa -- Autoimmune-Regulator-"
            "58kDa-PHD-SAND-CARD-Nuclear-Thymic-Transcription-Factor-"
            "APS-1-APECED-Hypoparathyroidism-Addison-CMC-Triad-OMIM-607358"
        ),
        "locus": "21q22.3",
        "protein_size": (
            "545 aa / 58 kDa (AIRE — autoimmune regulator; nuclear transcription factor; "
            "FUNCTION: expressed in medullary thymic epithelial cells (mTECs); "
            "  drives ectopic expression of >1,000 peripheral tissue antigens in thymus; "
            "  enables negative selection of autoreactive T-cells (clonal deletion); "
            "  domains: CARD (protein-protein interaction), SAND (DNA binding), "
            "    PHD1 + PHD2 (chromatin interaction, H3K4me0 binding), PEST (transcription); "
            "LOF CONSEQUENCE: "
            "  AIRE absent → peripheral antigens not presented in thymus; "
            "  Autoreactive T-cells escape deletion → attack multiple endocrine organs; "
            "  Produces high-titre NEUTRALISING AUTOANTIBODIES to cytokines "
            "    (anti-IFN-ω, anti-IL-12, anti-IL-17A/F, anti-IL-22) — PATHOGNOMONIC; "
            "  Clinical triad: "
            "    1. Mucocutaneous candidiasis (CMC) — FIRST, by age 5 in >90%; "
            "    2. Hypoparathyroidism (HP) — SECOND, median age 7; "
            "    3. Addison disease (PAI) — THIRD, median age 10-12; "
            "  encoded 21q22.3"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (AR) LOF — AIRE / APS-1 (APECED): "
            "  PHENOTYPE (triad in order of appearance): "
            "    1. CMC — oral, nail, oesophageal candida; by age 5 in >90%; "
            "    2. Hypoparathyroidism — low Ca2+, high PTH absent/low; by age 10 in 80%; "
            "    3. Addison disease — primary adrenal insufficiency; by age 15 in 70%; "
            "  ADDITIONAL COMPONENTS (not in triad, variable): "
            "    Autoimmune hepatitis (17%); alopecia (30%); vitiligo (26%); "
            "    Primary gonadal failure (60% females, 15% males); "
            "    Type 1 diabetes (12%); coeliac-like enteropathy; "
            "    Autoimmune keratopathy (corneal opacities, sight-threatening); "
            "    Thymoma (rare); "
            "  KEY CLINICAL RULE — AUTOANTIBODY SIGNATURE PATHOGNOMONIC: "
            "    Anti-IFN-ω (≥100 U/mL) and anti-IL-12 (≥100 U/mL): "
            "      sensitivity >95% for APS-1; "
            "      Present years BEFORE clinical disease — screening test; "
            "    Anti-IL-17A/F + anti-IL-22 → CMC susceptibility (loss of mucosal IL-17); "
            "  FOUNDER MUTATIONS: "
            "    R257X (Finland, ~1:25,000); R139X (Norway/UK); "
            "    del13bp (Finnish — most common Finnish allele); "
            "    Iranian Jews: Y85C (founder); "
            "  DIAGNOSIS: "
            "    Clinical triad + anti-IFN-ω; or homozygous AIRE mutation; "
            "    AD AIRE mutations: R302H, P326L (monoallelic) → dominant negative APS-1 (rare)"
        ),
        "disease_category": (
            "AIRE-APS-1-APECED — CMC-FIRST-THEN-HP-THEN-ADDISON — ANTI-IFN-OMEGA-PATHOGNOMONIC: "
            "  DIAGNOSIS CLUE: child with CMC + HP or Addison; anti-IFN-ω >100 U/mL; "
            "    Triad order: CMC → HP → Addison (each adds a decade); "
            "  TREATMENT: "
            "    CMC: fluconazole long-term; echinocandin if refractory; "
            "    HP: Ca2+ + calcitriol (oral); rPTH if poorly controlled; "
            "    Addison: hydrocortisone + fludrocortisone; sick-day rules mandatory; "
            "    Hepatitis: azathioprine + prednisolone; "
            "    Annual surveillance: Ca2+, cortisol, gonadal function, LFTs, "
            "      anti-IFN-ω panel to predict next component; "
            "  GENETIC TESTING: "
            "    Any child with CMC + 1 endocrine failure → AIRE sequencing + MLPA; "
            "    Anti-IFN-ω testing before sequencing if available"
        ),
        "disease_pathway": (
            "AIRE LOF → IMPAIRED CENTRAL TOLERANCE → POLY-ENDOCRINE AUTOIMMUNITY: "
            "  Normal AIRE function: "
            "    Mature mTECs upregulate AIRE → transcribe peripheral tissue-restricted antigens (TRAs); "
            "    TRAs presented on MHC-II to developing T-cells; "
            "    High-affinity autoreactive T-cells → clonal deletion (negative selection); "
            "    Regulatory T-cell generation (Treg) also depends on AIRE-expressing mTECs; "
            "  AIRE LOF: "
            "    TRAs not expressed → autoreactive T-cells escape thymus; "
            "    Peripheral tolerance mechanisms insufficient for poly-antigenic escape; "
            "    Autoreactive T-cells → organ-specific attack; "
            "    Also: aberrant B-cell tolerance → high-titre cytokine autoantibodies; "
            "    Anti-IL-17 AAb → CMC (loss of mucosal IL-17 defence vs Candida); "
            "    Anti-IFN-ω → may paradoxically increase susceptibility to viral infections"
        ),
    },
    {
        "gene": "FOXP3",
        "protein": (
            "FOXP3 -- Xp11.23 XL LOF -- 431aa -- Forkhead-Box-P3-Scurfin-"
            "47kDa-Master-Treg-Transcription-Factor-IPEX-Neonatal-T1DM-"
            "Enteropathy-Eczema-Treg-Absent-IL-2-Rapamycin-HSCT-OMIM-300292"
        ),
        "locus": "Xp11.23",
        "protein_size": (
            "431 aa / 47 kDa (FOXP3 — forkhead box P3 / scurfin; transcription factor; "
            "FUNCTION: master regulator of CD4+CD25+ regulatory T-cells (Tregs); "
            "  FOXP3 expression is required and sufficient to confer Treg identity; "
            "  Tregs suppress effector T-cells via IL-10, TGF-β, CTLA-4-mediated mechanisms; "
            "  FOXP3 deficiency = NO functional Tregs → uncontrolled immune activation; "
            "  Forkhead domain (DNA binding) + leucine zipper (dimerisation) + RHD domain; "
            "LOF CONSEQUENCE (IPEX = immunodysregulation, polyendocrinopathy, enteropathy, X-linked): "
            "  Neonatal onset in males; "
            "  TRIAD: "
            "    1. Type 1 diabetes mellitus (neonatal/infantile — watery diarrhoea coincident); "
            "    2. Intractable secretory diarrhoea + enteropathy; "
            "    3. Eczema + atopic dermatitis; "
            "  Additional: haemolytic anaemia, thrombocytopenia, hypothyroidism; "
            "  High IgE; multiple food allergies; "
            "  encoded Xp11.23"
        ),
        "inheritance": (
            "X-LINKED (XL) LOF — FOXP3 / IPEX: "
            "  PHENOTYPE (males, neonatal or first year of life): "
            "    Neonatal/infantile T1DM: "
            "      Hyperglycaemia in first weeks-months; anti-insulin AAb often present; "
            "      Insulin dependent from diagnosis; C-peptide absent; "
            "    Secretory diarrhoea: "
            "      Profuse watery diarrhoea from birth; weight loss, malnutrition; "
            "      Villous atrophy on biopsy; not responsive to dietary restriction; "
            "    Eczema: "
            "      Erythroderma / severe atopic dermatitis; high IgE; "
            "  ADDITIONAL AUTOIMMUNE: "
            "    Autoimmune haemolytic anaemia (Coombs positive); "
            "    Immune thrombocytopenia; "
            "    Autoimmune thyroiditis (hypothyroidism); "
            "    Membranous nephropathy; "
            "  FEMALE CARRIERS: usually unaffected (X-inactivation protects); rare mild disease; "
            "  KEY CLINICAL RULE: "
            "    Any male neonate with T1DM + diarrhoea = IPEX/FOXP3 until proven otherwise; "
            "    FOXP3 flow cytometry: absent CD4+CD25+FOXP3+ Tregs = diagnostic; "
            "  TREATMENT: "
            "    Immunosuppression: tacrolimus (calcineurin inhibitor, primary IS); "
            "      Or rapamycin (mTOR inhibitor — preserves Treg more than tacrolimus); "
            "    Supportive: insulin, total parenteral nutrition, skin care; "
            "    CURATIVE: HSCT (haematopoietic stem cell transplantation) — "
            "      Corrects Treg defect; best outcome <2 years with good IS bridge"
        ),
        "disease_category": (
            "FOXP3-IPEX — NEONATAL-T1DM-DIARRHOEA-ECZEMA — TREGS-ABSENT — HSCT-CURATIVE: "
            "  DIAGNOSIS CLUE: male neonate; T1DM + profuse diarrhoea + eczema; "
            "    Absent CD4+CD25+FOXP3+ Tregs on flow cytometry; "
            "  TREATMENT: "
            "    Rapamycin (preferred over tacrolimus — protects Treg reconstitution); "
            "    Bridge to HSCT; HSCT is curative; "
            "    Insulin: insulin infusion; monitor glucose closely during HSCT; "
            "  GENETIC TESTING: "
            "    Any male infant with neonatal diabetes + diarrhoea → FOXP3 first; "
            "    Then KCNJ11/ABCC8 for neonatal DM without diarrhoea"
        ),
        "disease_pathway": (
            "FOXP3 LOF → ABSENT TREGS → UNRESTRAINED EFFECTOR T-CELL ATTACK: "
            "  Normal Treg function: "
            "    Tregs (CD4+CD25+FOXP3+) suppress effector T-cells via: "
            "      1. IL-10 / TGF-β secretion (anti-inflammatory cytokines); "
            "      2. CTLA-4 on Treg surface → depletes CD80/CD86 on APCs → co-stimulation blocked; "
            "      3. IL-2 consumption (metabolic competition vs effector T-cells); "
            "    Without Tregs → effector T-cells proliferate unchecked; "
            "  FOXP3 LOF: "
            "    No Treg generation → effector CD4+/CD8+ T-cells infiltrate every organ; "
            "    Pancreatic islets: insulitis → T1DM; "
            "    Gut epithelium: enteropathy → secretory diarrhoea; "
            "    Skin: eczematous infiltration; "
            "    Thyroid: thyroiditis; Kidney: membranous nephropathy; "
            "    Elevated IgE: Th2 skewing without Treg counter-regulation"
        ),
    },
    {
        "gene": "CTLA4",
        "protein": (
            "CTLA4 -- 2q33.2 AD LOF Haploinsufficiency -- 223aa -- Cytotoxic-T-Lymphocyte-"
            "Antigen-4-25kDa-Ig-Superfamily-B7-CD80-CD86-Ligand-Treg-Effector-T-Cell-Brake-"
            "CHAI-Thyroiditis-T1DM-Cytopenias-Lymphoproliferation-Abatacept-CURATIVE-OMIM-123890"
        ),
        "locus": "2q33.2",
        "protein_size": (
            "223 aa / 25 kDa (CTLA4 — cytotoxic T-lymphocyte-associated protein 4; "
            "FUNCTION: inhibitory receptor on T-cells and Tregs; "
            "  Binds CD80 (B7-1) and CD86 (B7-2) on APCs with higher affinity than CD28; "
            "  Outcompetes CD28 → blocks co-stimulatory signal → T-cell anergy/tolerance; "
            "  Tregs constitutively express high CTLA-4 — essential for Treg suppressive function; "
            "  CTLA-4 also performs transendocytosis: removes CD80/CD86 from APCs → "
            "    reduces co-stimulation available to nearby effector T-cells; "
            "LOF CONSEQUENCE (CHAI = CTLA-4 haploinsufficiency with autoimmune infiltration): "
            "  Half-normal CTLA-4 → impaired Treg function + effector T-cell over-activation; "
            "  Lymphocytic infiltration of multiple organs; "
            "  LATE ONSET (typically 20s-40s) vs FOXP3 (neonatal); "
            "  encoded 2q33.2"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT (AD) LOF haploinsufficiency — CTLA4 / CHAI: "
            "  PHENOTYPE (adult/young adult onset; variable penetrance ~60%): "
            "    Autoimmune thyroiditis (most common; Hashimoto or Graves); "
            "    Type 1 diabetes; "
            "    Cytopenias: immune thrombocytopenia, autoimmune haemolytic anaemia; "
            "    Lymphoproliferation: lymphadenopathy, splenomegaly; "
            "      Bowel infiltration (lymphocytic colitis/enteropathy); "
            "      Lung infiltration (lymphocytic interstitial pneumonitis); "
            "      Brain infiltration (encephalitis — rare but severe); "
            "    Hypogammaglobulinaemia (despite lymphoproliferation — effector B-cell dysfunction); "
            "  KEY CLINICAL RULE — ABATACEPT CURATIVE: "
            "    CTLA4 LOF (or LRBA deficiency) → insufficient CTLA-4 at effector/Treg surface; "
            "    Abatacept (CTLA4-Ig fusion protein) REPLACES deficient CTLA-4 signal; "
            "    Abatacept is the targeted therapy → reverses lymphoproliferation and autoimmunity; "
            "    DISTINGUISH from CTLA-4 CHECKPOINT INHIBITOR toxicity (ipilimumab, which BLOCKS CTLA-4): "
            "      CHAI = too LITTLE CTLA-4 → treat WITH CTLA-4 (abatacept); "
            "      Ipilimumab toxicity = too LITTLE CTLA-4 (drug-induced) → same treat WITH abatacept; "
            "  INCOMPLETE PENETRANCE: ~60%; family members often undiagnosed; "
            "  GENETIC TESTING: "
            "    Immune cytopenias + autoimmune thyroid + hypogamma + lymphoproliferation → CTLA4"
        ),
        "disease_category": (
            "CTLA4-CHAI — THYROIDITIS-CYTOPENIAS-LYMPHOPROLIFERATION — ABATACEPT-CURATIVE: "
            "  DIAGNOSIS CLUE: adult autoimmune multiorgan + lymphoproliferation + hypogammaglobulinaemia; "
            "    Look for Treg dysfunction: reduced CTLA-4 expression on Tregs by flow; "
            "  TREATMENT: "
            "    Abatacept (CTLA4-Ig) — first-line targeted; sc weekly; "
            "    Sirolimus (rapamycin) — second-line; "
            "    Corticosteroids for acute flares; "
            "    AVOID conventional B-cell depletion (rituximab) — worsens lymphoproliferation; "
            "  GENETIC TESTING: "
            "    CTLA4 + LRBA panel (phenotypically overlapping); "
            "    CTLA-4 protein expression on Tregs by flow before genetic results"
        ),
        "disease_pathway": (
            "CTLA4 LOF → REDUCED TREG SUPPRESSION → LYMPHOPROLIFERATION + ORGAN INFILTRATION: "
            "  Normal CTLA-4: "
            "    Tregs constitutively express CTLA-4 at high levels; "
            "    CTLA-4 competes with CD28 for CD80/CD86 binding on APCs; "
            "    Transendocytosis: Treg CTLA-4 strips CD80/CD86 from APCs → reduces co-stimulation; "
            "    Net effect: effector T-cells receive subthreshold co-stimulatory signal → anergy; "
            "  CTLA4 haploinsufficiency: "
            "    50% CTLA-4 → insufficient transendocytosis → CD80/CD86 abundant on APCs; "
            "    Effector T-cells receive full co-stimulation → hyperactivated; "
            "    Autoreactive clones not restrained → organ infiltration; "
            "    Treg suppressive capacity reduced → lymphoproliferation"
        ),
    },
    {
        "gene": "LRBA",
        "protein": (
            "LRBA -- 4q31.3 AR LOF -- 2863aa -- LPS-Responsive-Beige-Like-Anchor-"
            "319kDa-BEACH-WD40-ARM-Endosomal-CTLA4-Recycling-Vesicle-"
            "CVID-Hypogammaglobulinaemia-Autoimmunity-Abatacept-CURATIVE-OMIM-606453"
        ),
        "locus": "4q31.3",
        "protein_size": (
            "2863 aa / 319 kDa (LRBA — LPS-responsive beige-like anchor protein; "
            "FUNCTION: endosomal trafficking protein; "
            "  BEACH domain (beige and Chediak-Higashi): vesicle biogenesis; "
            "  WD40 repeats: protein-protein interaction scaffold; "
            "  ARM repeats: structural; "
            "  KEY FUNCTION: recycles CTLA-4 from endosomes back to cell surface; "
            "    CTLA-4 is normally sorted to lysosomes → degraded if not rescued by LRBA; "
            "    LRBA associates with CTLA-4-containing endosomes → redirects to recycling pathway; "
            "  Secondary function: ICOS (inducible T-cell co-stimulator) trafficking; "
            "LOF CONSEQUENCE: "
            "  LRBA absent → CTLA-4 sorted to lysosomes → rapidly degraded; "
            "  Functional CTLA-4 deficiency at cell surface (phenocopies CTLA4 haploinsufficiency); "
            "  AR (biallelic) → more severe than CTLA4 AD haploinsufficiency; "
            "  encoded 4q31.3"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (AR) LOF — LRBA / LRBA DEFICIENCY (CVID8): "
            "  PHENOTYPE (childhood onset; typically 5-15 years): "
            "    PRIMARY IMMUNODEFICIENCY: "
            "      Hypogammaglobulinaemia (low IgG, IgA, IgM) — CVID phenotype; "
            "      Recurrent sinopulmonary infections (encapsulated bacteria); "
            "      Reduced switched memory B-cells; "
            "    AUTOIMMUNITY (often dominant feature): "
            "      Immune cytopenias (AIHA, ITP, neutropenia); "
            "      Inflammatory bowel disease (colitis); "
            "      Autoimmune liver disease; "
            "      Lymphocytic interstitial pneumonitis; "
            "      Autoimmune thyroiditis; "
            "    LYMPHOPROLIFERATION: "
            "      Splenomegaly; lymphadenopathy; "
            "      Intestinal lymphoid nodular hyperplasia; "
            "  KEY CLINICAL RULE — ABATACEPT CURATIVE (same mechanism as CTLA4): "
            "    LRBA deficiency → CTLA-4 degraded → functional CTLA-4 deficiency; "
            "    Abatacept replaces the missing CTLA-4 signal → dramatic response; "
            "    Response rate: >80% reduction in lymphoproliferation with abatacept; "
            "  DISTINGUISH FROM CTLA4: "
            "    LRBA: AR, earlier onset, more severe, CVID component prominent, "
            "      no CTLA-4 protein on flow cytometry; "
            "    CTLA4: AD, later onset, milder, normal CTLA-4 protein level but haploinsufficiency"
        ),
        "disease_category": (
            "LRBA-DEFICIENCY — CVID-AUTOIMMUNITY-LYMPHOPROLIFERATION — ABATACEPT-CURATIVE: "
            "  DIAGNOSIS CLUE: child with CVID + autoimmunity + lymphoproliferation; "
            "    Flow: absent CTLA-4 on Tregs (CTLA-4 not on surface due to lysosomal degradation); "
            "    LRBA protein absent on Western blot (diagnosis); "
            "  TREATMENT: "
            "    Abatacept — first-line targeted therapy (same as CTLA4 LOF); "
            "    IVIg — for hypogammaglobulinaemia + infections; "
            "    HSCT — for refractory severe disease; "
            "  GENETIC TESTING: "
            "    CTLA4 + LRBA must be tested TOGETHER — phenotypically indistinguishable; "
            "    If CTLA-4 protein absent on flow: suspect LRBA before genetic result"
        ),
        "disease_pathway": (
            "LRBA LOF → CTLA-4 LYSOSOMAL DEGRADATION → FUNCTIONAL CTLA-4 DEFICIENCY: "
            "  Normal LRBA-CTLA-4 recycling circuit: "
            "    CTLA-4 internalised from T-cell surface after ligand binding → endosome; "
            "    In endosome: LRBA recognises CTLA-4 (via BEACH-WD40 domain interaction); "
            "    LRBA recruits recycling vesicle machinery → CTLA-4 trafficked back to surface; "
            "    Net: CTLA-4 recycles efficiently → sustained surface expression; "
            "  LRBA LOF: "
            "    CTLA-4 enters endosome → LRBA absent → no recycling signal; "
            "    CTLA-4 default pathway: lysosomal degradation (t½ shortened); "
            "    Surface CTLA-4 drops → identical phenotype to CTLA4 LOF; "
            "    Compound effect: both Treg and effector T-cell CTLA-4 depleted → "
            "    Treg suppressive defect + effector over-activation"
        ),
    },
    {
        "gene": "IL2RA",
        "protein": (
            "IL2RA -- 10p15.1 AR LOF -- 272aa -- Interleukin-2-Receptor-Alpha-CD25-"
            "30kDa-High-Affinity-IL2-Binding-Chain-Treg-Survival-IL2-Signal-"
            "IPEX-Like-Neonatal-Autoimmunity-Elevated-IL2-IL2-Antibody-Distinguisher-OMIM-147730"
        ),
        "locus": "10p15.1",
        "protein_size": (
            "272 aa / 30 kDa (IL2RA — interleukin-2 receptor alpha chain; CD25; "
            "FUNCTION: alpha subunit of the trimeric high-affinity IL-2 receptor; "
            "  IL-2Rα (CD25) + IL-2Rβ (CD122) + γc (CD132) = high-affinity IL-2Rαβγ; "
            "  IL-2Rα (CD25) alone has low affinity; combined: Kd ~10 pM (100× ↑ affinity); "
            "  Tregs constitutively express high CD25 → preferentially consume IL-2; "
            "  IL-2 signal → Treg survival (Bcl-2), proliferation, FOXP3 maintenance; "
            "  Effector T-cells induce CD25 transiently after activation; "
            "LOF CONSEQUENCE: "
            "  CD25 absent → IL-2 cannot signal effectively on Tregs; "
            "  Treg numbers reduced + function impaired; "
            "  Paradox: serum IL-2 VERY HIGH (not consumed by Tregs); "
            "  Clinical: IPEX-like neonatal autoimmunity; "
            "  encoded 10p15.1"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (AR) LOF — IL2RA / CD25 DEFICIENCY: "
            "  PHENOTYPE (neonatal/early infantile onset, similar to IPEX): "
            "    Lymphoproliferation: hepatosplenomegaly + lymphadenopathy; "
            "    Autoimmune enteropathy: diarrhoea + villous atrophy; "
            "    Autoimmune skin disease: eczema + erythroderma; "
            "    Inflammatory bowel disease (colitis); "
            "    Autoimmune thyroiditis; "
            "    Type 1 diabetes (rare vs FOXP3/IPEX — less dominant feature); "
            "    Recurrent infections: viral (CMV, EBV, herpesviruses) + fungal (Candida); "
            "    Lymphocytic interstitial pneumonitis; "
            "  KEY DISTINGUISHER FROM IPEX (FOXP3 LOF): "
            "    IL2RA: serum IL-2 VERY HIGH (>10,000 pg/mL) — IL-2 not consumed; "
            "    FOXP3/IPEX: serum IL-2 normal or mildly elevated; "
            "    IL2RA: CD25 absent on T-cells/NK by flow (diagnostic); "
            "    FOXP3: FOXP3 absent by intracellular flow; "
            "  FEMALE EFFECT: IL2RA is autosomal → affects males and females equally; "
            "    (vs FOXP3 = X-linked → males predominant); "
            "  TREATMENT: "
            "    Immunosuppression: rapamycin (mTOR inhibitor) + steroids; "
            "    IL-2 pathway: basiliximab (anti-CD25 mAb) paradoxically reported helpful "
            "      (blocks excess IL-2 signalling on effector cells); "
            "    HSCT: curative option for severe cases"
        ),
        "disease_category": (
            "IL2RA-CD25-DEFICIENCY — IPEX-LIKE-FEMALES-ALSO — ELEVATED-IL2-DISTINGUISHER: "
            "  DIAGNOSIS CLUE: IPEX-like but female affected = autosomal → check IL2RA; "
            "    CD25 absent on lymphocytes by flow cytometry (diagnostic); "
            "    Serum IL-2 very high (not consumed by Tregs); "
            "  TREATMENT: "
            "    Rapamycin + steroids; "
            "    HSCT for severe disease; "
            "    Basiliximab (experimental); "
            "  GENETIC TESTING: "
            "    IPEX-like phenotype in female → IL2RA (autosomal) before FOXP3; "
            "    Neonatal diabetes + autoimmunity → FOXP3 first in males, IL2RA in females"
        ),
        "disease_pathway": (
            "IL2RA LOF → TREG IL-2 STARVATION → IMPAIRED TREG SURVIVAL + EFFECTOR OVERACTIVATION: "
            "  Normal IL-2/CD25 Treg circuit: "
            "    Effector T-cells produce IL-2 after antigen activation; "
            "    Tregs constitutively express CD25 (high-affinity IL-2Rα) → "
            "      consume IL-2 from microenvironment (cytokine sink); "
            "    IL-2 → pSTAT5 → Bcl-2 ↑ (Treg survival) + FOXP3 ↑ (Treg function); "
            "    Net: Tregs out-compete effector T-cells for limiting IL-2; "
            "  IL2RA LOF: "
            "    CD25 absent → high-affinity IL-2R non-functional on Tregs; "
            "    Tregs cannot compete for IL-2 → Treg survival compromised; "
            "    Excess IL-2 available to effector T-cells → effector hyperactivation; "
            "    SERUM IL-2 HIGH (key biomarker — Tregs not consuming IL-2); "
            "    Unchecked effector T-cells → multi-organ autoimmunity"
        ),
    },
    {
        "gene": "STAT3",
        "protein": (
            "STAT3 -- 17q21.2 AD GOF -- 770aa -- Signal-Transducer-Activator-Transcription-3-"
            "92kDa-SH2-Domain-JAK-JAK2-JAK1-TYK2-Phospho-Y705-Dimer-"
            "STAT3-GOF-Multiorgan-Autoimmunity-T1DM-Thyroiditis-Short-Stature-JAK-Inhibitor-OMIM-102582"
        ),
        "locus": "17q21.2",
        "protein_size": (
            "770 aa / 92 kDa (STAT3 — signal transducer and activator of transcription 3; "
            "FUNCTION: latent cytoplasmic transcription factor activated by JAKs; "
            "  Activated by: IL-6, IL-10, IL-17, IL-21, IL-22, IFN-α/γ, EGF, GH, leptin; "
            "  JAK1/JAK2/TYK2 phosphorylate STAT3 at Tyr705 → dimerisation → nuclear translocation; "
            "  Targets: anti-apoptotic genes (Bcl-2, Bcl-xL), proliferation, immune regulation; "
            "  In immune context: STAT3 required for Th17 differentiation + IL-10 signalling; "
            "LOF CONSEQUENCE (AD dominant-negative, STAT3 LOF = Hyper-IgE syndrome type 1, not APS); "
            "GOF CONSEQUENCE (AD gain-of-function, distinct phenotype): "
            "  Enhanced STAT3 signalling → "
            "    Disrupted immune homeostasis; impaired Treg function; "
            "    Excessive Th17/effector T-cell activity; lymphoproliferation; "
            "  Multi-organ autoimmunity with young/early childhood onset; "
            "  encoded 17q21.2"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT (AD) GOF — STAT3 / STAT3 GOF SYNDROME: "
            "  PHENOTYPE (childhood onset, variable): "
            "    Polyautoimmunity: "
            "      Type 1 diabetes (most common endocrine manifestation); "
            "      Autoimmune thyroiditis (Hashimoto); "
            "      Autoimmune cytopenia (AIHA, ITP, neutropenia); "
            "      Inflammatory bowel disease (colitis); "
            "      Alopecia areata; vitiligo; "
            "    LYMPHOPROLIFERATION: "
            "      Generalised lymphadenopathy; hepatosplenomegaly; "
      "      Lymphocytic interstitial pneumonitis (LIP); "
            "    GROWTH: "
            "      SHORT STATURE: STAT3 GOF → excess SOCS3 → GH signalling impaired; "
            "        Short stature often first clinical feature in childhood; "
            "    ENDOCRINE: "
            "      Short stature + T1DM + hypothyroidism in same patient → STAT3 GOF; "
            "  KEY CLINICAL RULE — JAK INHIBITORS EFFECTIVE: "
            "    STAT3 GOF pathway: cytokine → JAK → pSTAT3 (excess) → autoimmunity; "
            "    JAK inhibitors (ruxolitinib, baricitinib, tofacitinib) block upstream JAK; "
            "    → Reduce pSTAT3 → reverse autoimmunity + lymphoproliferation; "
            "    Response documented: cytopenias, thyroiditis, IBD improve; "
            "  DISTINGUISH FROM STAT3 LOF (Hyper-IgE syndrome): "
            "    STAT3 LOF (AD dominant-negative): eczema + high IgE + Staph + dental/bone; "
            "    STAT3 GOF: autoimmunity + lymphoproliferation + short stature (NO high IgE)"
        ),
        "disease_category": (
            "STAT3-GOF — T1DM-THYROIDITIS-SHORT-STATURE-LYMPHOPROLIFERATION — JAK-INHIBITORS: "
            "  DIAGNOSIS CLUE: young child with short stature + T1DM + thyroiditis + cytopenias; "
            "    Elevated pSTAT3 on T-cells; GOF variant in STAT3 SH2 or coiled-coil domain; "
            "  TREATMENT: "
            "    JAK inhibitors (ruxolitinib sc/oral — first-line targeted); "
            "    Mycophenolate for organ-specific autoimmunity; "
            "    GH therapy for short stature (monitor closely, JAK-i may help more); "
            "  GENETIC TESTING: "
            "    STAT3 sequencing (SH2 domain hotspot: Y640F, D661Y, K658N — known GOF alleles); "
            "    Distinguish GOF from LOF alleles (opposite phenotype!)"
        ),
        "disease_pathway": (
            "STAT3 GOF → EXCESS pSTAT3 SIGNALLING → TH17/EFFECTOR SKEWING + TREG DYSFUNCTION: "
            "  Normal STAT3 signalling: "
            "    Cytokine binds receptor → JAK transphosphorylation → STAT3 Tyr705 phosphorylated; "
            "    pSTAT3 homodimerises → nuclear translocation → target gene transcription; "
            "    SOCS3 (negative feedback) is STAT3 target → limits own signal; "
            "  STAT3 GOF: "
            "    Enhanced Tyr705 phosphorylation or prolonged nuclear residency; "
            "    Excessive Th17 differentiation (IL-6/IL-23/STAT3 pathway enhanced); "
            "    IL-10 signalling also enhanced but pro-inflammatory Th17 dominates; "
            "    Treg FOXP3 induction impaired (excessive STAT3 competes with TGF-β/Smad); "
            "    GH-STAT5 pathway competing with STAT3 GOF → reduced GH signalling → short stature; "
            "    Effector B-cells: excessive STAT3 → autoantibody production"
        ),
    },
    {
        "gene": "STAT1",
        "protein": (
            "STAT1 -- 2q32.2 AD GOF -- 750aa -- Signal-Transducer-Activator-Transcription-1-"
            "84kDa-IFN-Alpha-Beta-Gamma-JAK1-TYK2-JAK2-CC-DBD-SH2-Domains-"
            "STAT1-GOF-CMC-Autoimmunity-Thyroiditis-T1DM-Aneurysm-Ruxolitinib-OMIM-600555"
        ),
        "locus": "2q32.2",
        "protein_size": (
            "750 aa / 84 kDa (STAT1 — signal transducer and activator of transcription 1; "
            "FUNCTION: central mediator of IFN signalling; "
            "  IFN-α/β → TYK2/JAK1 → STAT1/STAT2 heterodimer + IRF9 = ISGF3 → ISREs; "
            "  IFN-γ → JAK1/JAK2 → STAT1 homodimer (GAF) → GAS elements; "
            "  Type I IFN: antiviral defence; Type II IFN: macrophage activation; "
            "  STAT1 also activated by IL-27, IL-6 (via co-signalling with STAT3); "
            "  KEY: STAT1/STAT3 balance: STAT3 promotes Th17; STAT1 inhibits Th17; "
            "LOF (AR/AD): Mendelian susceptibility to mycobacterial disease (MSMD) / viral; "
            "GOF (AD): EXCESS STAT1 signalling → "
            "  Inhibited Th17 (despite intact IL-17 production) → CMC; "
            "  Paradox: excess IFN signalling → suppresses IL-17-mediated mucosal immunity; "
            "  encoded 2q32.2"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT (AD) GOF — STAT1 / STAT1 GOF SYNDROME: "
            "  PHENOTYPE: "
            "    PRIMARY: Chronic Mucocutaneous Candidiasis (CMC): "
            "      Oral candidiasis; nail onychomycosis; oesophageal candidiasis; "
            "      Onset in childhood; recurrent despite treatment; "
            "      MECHANISM: excess STAT1 → impaired Th17 responses → reduced IL-17 at mucosa; "
            "      PARADOX: anti-candida Th17 immunity suppressed BY excess IFN/STAT1 signalling; "
            "    ENDOCRINE AUTOIMMUNITY: "
            "      Autoimmune thyroiditis (hypothyroidism): >50% of STAT1 GOF; "
            "      Type 1 diabetes: ~20%; "
            "      Alopecia areata; vitiligo; "
            "    VASCULAR: "
            "      INTRACRANIAL / CEREBRAL ANEURYSM — PATHOGNOMONIC FOR STAT1 GOF: "
            "        occurs in ~15%; bilateral internal carotid or intracranial aneurysms; "
            "        Annual MRA surveillance MANDATORY; "
            "        Rupture risk — neurosurgical/interventional radiology referral; "
            "    ADDITIONAL: "
            "      Recurrent bacterial infections; "
            "      Squamous cell carcinoma of oral cavity/skin (HPV-related, IL-17 mucosal defence lost); "
            "  KEY CLINICAL RULE — RUXOLITINIB EFFECTIVE: "
            "    STAT1 GOF pathway: IFN/cytokine → JAK → excess pSTAT1; "
            "    Ruxolitinib (JAK1/2 inhibitor) → reduces pSTAT1 → restores Th17 balance; "
            "    CMC resolves; autoimmunity improves; aneurysm stabilisation reported; "
            "  DISTINGUISH STAT1 GOF from AIRE (APS-1): "
            "    STAT1 GOF: CMC + aneurysm + thyroiditis; anti-IFN-ω NEGATIVE; "
            "    AIRE (APS-1): CMC + HP + Addison; anti-IFN-ω POSITIVE"
        ),
        "disease_category": (
            "STAT1-GOF — CMC-AUTOIMMUNITY-INTRACRANIAL-ANEURYSM — RUXOLITINIB-EFFECTIVE: "
            "  DIAGNOSIS CLUE: CMC + thyroiditis + aneurysm; anti-IFN-ω negative (vs AIRE); "
            "    STAT1 coiled-coil domain GOF variants (L706S, Q293G, K387E, T385M hotspots); "
            "  TREATMENT: "
            "    Ruxolitinib — first-line targeted (JAK1/2); resolves CMC + autoimmunity; "
            "    Azole antifungals for CMC; echinocandin for refractory; "
            "    Annual MRA brain/neck: aneurysm surveillance (rupture prevention); "
            "  GENETIC TESTING: "
            "    CMC + autoimmune thyroid + family history → STAT1 GOF; "
            "    Test coiled-coil domain hotspots first; functional STAT1 phosphorylation assay"
        ),
        "disease_pathway": (
            "STAT1 GOF → EXCESS IFN SIGNALLING → IMPAIRED TH17 + MUCOSAL CANDIDA SUSCEPTIBILITY: "
            "  Normal STAT1/STAT3 balance for Th17: "
            "    IL-6/IL-21/IL-23 → JAK1/2 → STAT3 → RORγt → Th17 differentiation; "
            "    STAT1 normally downregulates this Th17 pathway (counter-balance); "
            "    At mucosa: Th17 cells produce IL-17A/F → antimicrobial peptides → Candida killing; "
            "  STAT1 GOF: "
            "    Excess STAT1 activity → STAT3-driven Th17 pathway suppressed; "
            "    IL-17A/F production at mucosa reduced → mucosal antimicrobial defence lost; "
            "    Candida colonises mucosae → CMC; HPV not cleared → squamous carcinoma risk; "
            "    ALSO: Excess IFN-γ/STAT1 → chronic inflammation → "
            "    Thyroid + pancreatic + vascular wall: "
            "      IFN-γ upregulates MHC-II on parenchymal cells → increased autoantigen presentation; "
            "      Vascular: excess IFN-γ → smooth muscle apoptosis → arterial wall weakening → aneurysm"
        ),
    },
    {
        "gene": "ITCH",
        "protein": (
            "ITCH -- 20q11.22 AR LOF -- 864aa -- ITCH-E3-Ubiquitin-Protein-Ligase-"
            "97kDa-HECT-Domain-WW-Domains-Ubiquitin-E3-Ndfip1-CTLA4-JunB-PPXY-Substrate-"
            "ITCH-Deficiency-Multisystem-Autoimmunity-Developmental-Delay-OMIM-606409"
        ),
        "locus": "20q11.22",
        "protein_size": (
            "864 aa / 97 kDa (ITCH — itchy E3 ubiquitin protein ligase; HECT domain E3; "
            "FUNCTION: ubiquitin E3 ligase of the NEDD4 subfamily; "
            "  HECT domain (homologous to E6-AP C-terminus): ubiquitin transfer from E2 to substrate; "
            "  4 WW domains: recognise PPXY motifs in substrates; "
            "  C2 domain: membrane/phospholipid binding; "
            "  KEY SUBSTRATES: "
            "    JunB (AP-1 component): ubiquitylates → degradation; "
            "      JunB drives IL-4/IL-5 (Th2 cytokines); ITCH LOF → JunB stable → Th2 skewing; "
            "    CTLA-4: ITCH ubiquitylates K48-linked → controls CTLA-4 surface levels; "
            "      ITCH LOF → CTLA-4 not ubiquitylated → altered CTLA-4 trafficking; "
            "    Notch1: ubiquitylates ICN1 → controls Notch signalling; "
            "    Ndfip1 (adaptor protein): reduces Th17 via ITCH-Ndfip1 axis; "
            "    NFAT: regulates cytokine expression; "
            "LOF CONSEQUENCE: "
            "  Stable JunB → Th2 cytokine excess (IL-4, IL-5) → atopy/eosinophilia; "
            "  Impaired immune homeostasis → multisystem autoimmunity; "
            "  encoded 20q11.22"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (AR) LOF — ITCH / ITCH DEFICIENCY: "
            "  PHENOTYPE (paediatric onset; rare — fewer than 20 patients reported): "
            "    DYSMORPHIC FEATURES: "
            "      Developmental delay / intellectual disability (mild-moderate); "
            "      Facial dysmorphism: ptosis, broad forehead, small mouth; "
            "      Short stature; failure to thrive; "
            "    MULTISYSTEM AUTOIMMUNITY: "
            "      Autoimmune hepatitis; "
            "      Pulmonary disease (lymphocytic infiltration); "
            "      Inflammatory bowel disease; "
            "      Autoimmune cytopenias; "
            "      Eczema / atopic dermatitis (JunB-driven Th2 excess); "
            "    SEROLOGY: "
            "      Elevated IgG; autoantibodies (anti-nuclear, anti-smooth muscle); "
            "      Eosinophilia; high IgE; "
            "    IMMUNE FINDINGS: "
            "      Reduced regulatory T-cell function; "
            "      Th2 cytokine excess (IL-4, IL-5); "
            "      Lymphocytic infiltration on biopsies; "
            "  KEY CLINICAL RULE — RARE, SYNDROMIC APS: "
            "    ITCH deficiency is the ONLY hereditary APS with DYSMORPHIC FEATURES; "
            "    The combination of autoimmunity + developmental delay + dysmorphism → ITCH; "
            "    AIRE (APS-1): no dysmorphism; FOXP3 (IPEX): no dysmorphism; "
            "    ITCH: dysmorphism + ID + autoimmunity = distinctive clinical triad; "
            "  TREATMENT: "
            "    Immunosuppression: corticosteroids + azathioprine; "
            "    Liver: ursodeoxycholic acid + IS for hepatitis; "
            "    HSCT: limited experience; considered for severe cases"
        ),
        "disease_category": (
            "ITCH-DEFICIENCY — DYSMORPHIC-AUTOIMMUNITY-RARE — DISTINCTIVE-SYNDROMIC-APS: "
            "  DIAGNOSIS CLUE: multisystem autoimmunity + developmental delay + dysmorphism = ITCH; "
            "    Only hereditary APS gene with neurodevelopmental phenotype + dysmorphic features; "
            "  TREATMENT: "
            "    Corticosteroids + azathioprine; "
            "    HSCT for severe cases; "
            "    Supportive developmental/educational therapy; "
            "  GENETIC TESTING: "
            "    Autoimmune hepatitis + eczema + ID + dysmorphism → ITCH gene sequencing; "
            "    Panel: include ITCH with AIRE, FOXP3, CTLA4, LRBA, IL2RA, STAT3, STAT1"
        ),
        "disease_pathway": (
            "ITCH LOF → STABLE JUNB → TH2 EXCESS + IMPAIRED IMMUNE HOMEOSTASIS: "
            "  Normal ITCH-mediated immune regulation: "
            "    T-cell activation → JunB (AP-1 transcription factor) upregulated; "
            "    JunB drives IL-4, IL-5, IL-13 (Th2 cytokines); "
            "    ITCH ubiquitylates JunB → K48-linked poly-ubiquitin → proteasomal degradation; "
            "    Net: limits Th2 cytokine production duration; "
            "    Also: Ndfip1-ITCH axis limits Th17 differentiation via RORγt ubiquitylation; "
            "    Net: ITCH restrains both Th2 AND Th17 → preserves Treg/effector balance; "
            "  ITCH LOF: "
            "    JunB not degraded → sustained high Th2 cytokines → atopy + eosinophilia; "
            "    Ndfip1-ITCH Th17 control absent → possible Th17 expansion; "
            "    CTLA-4 trafficking altered → Treg function impaired; "
            "    Net: autoimmunity + Th2 skewing + Treg dysfunction → multi-organ infiltration; "
            "    Neurodevelopmental features: ITCH-regulated ubiquitin pathway affects "
            "    neuronal Notch signalling and synaptic pruning during brain development"
        ),
    },
]


def _make_patients(seed: int, gene: str) -> list:
    rng = random.Random(seed)
    gene_profiles = {
        "AIRE":  dict(treg_pct=(3, 8), anti_ifnw_pct=0.95, candida_pct=0.92,
                     hp_pct=0.82, addison_pct=0.70, t1dm_pct=0.14),
        "FOXP3": dict(treg_pct=(0, 1), anti_ifnw_pct=0.04, candida_pct=0.10,
                     hp_pct=0.05, addison_pct=0.05, t1dm_pct=0.95),
        "CTLA4": dict(treg_pct=(4, 9), anti_ifnw_pct=0.03, candida_pct=0.08,
                     hp_pct=0.12, addison_pct=0.15, t1dm_pct=0.28),
        "LRBA":  dict(treg_pct=(2, 6), anti_ifnw_pct=0.03, candida_pct=0.18,
                     hp_pct=0.10, addison_pct=0.12, t1dm_pct=0.22),
        "IL2RA": dict(treg_pct=(1, 4), anti_ifnw_pct=0.05, candida_pct=0.15,
                     hp_pct=0.04, addison_pct=0.08, t1dm_pct=0.40),
        "STAT3": dict(treg_pct=(4, 9), anti_ifnw_pct=0.02, candida_pct=0.12,
                     hp_pct=0.06, addison_pct=0.08, t1dm_pct=0.55),
        "STAT1": dict(treg_pct=(5, 12), anti_ifnw_pct=0.04, candida_pct=0.95,
                     hp_pct=0.08, addison_pct=0.06, t1dm_pct=0.22),
        "ITCH":  dict(treg_pct=(3, 7), anti_ifnw_pct=0.05, candida_pct=0.20,
                     hp_pct=0.04, addison_pct=0.06, t1dm_pct=0.15),
    }
    p = gene_profiles.get(gene, gene_profiles["CTLA4"])

    treatment_map = {
        "AIRE":  ["Ca2+/calcitriol+hydrocortisone/fludro", "Ca2+/calcitriol+HC/fludro+fluconazole", "rPTH+HC/fludro"],
        "FOXP3": ["rapamycin+insulin+TPN", "tacrolimus+insulin", "HSCT"],
        "CTLA4": ["abatacept", "sirolimus+steroids", "abatacept+IVIg"],
        "LRBA":  ["abatacept+IVIg", "abatacept+steroids", "HSCT+IVIg"],
        "IL2RA": ["rapamycin+steroids", "basiliximab+steroids", "HSCT"],
        "STAT3": ["ruxolitinib", "baricitinib+steroids", "ruxolitinib+mycophenolate"],
        "STAT1": ["ruxolitinib+azole", "itraconazole+ruxolitinib", "voriconazole+ruxolitinib"],
        "ITCH":  ["steroids+azathioprine", "MMF+steroids", "HSCT"],
    }
    treatments = treatment_map.get(gene, ["immunosuppression"])

    patients = []
    for i in range(40):
        treg = round(rng.uniform(*p["treg_pct"]), 1)
        anti_ifnw = rng.random() < p["anti_ifnw_pct"]
        candida = rng.random() < p["candida_pct"]
        hp = rng.random() < p["hp_pct"]
        addison = rng.random() < p["addison_pct"]
        t1dm = rng.random() < p["t1dm_pct"]
        age_dx = rng.randint(0, 45) if gene in ("CTLA4", "STAT3", "STAT1") else rng.randint(0, 15)
        treatment = rng.choice(treatments)
        patients.append({
            "id": f"{gene}-{seed}-{i+1:03d}",
            "gene": gene,
            "age_at_diagnosis": age_dx,
            "treg_pct": treg,
            "anti_ifnw_positive": anti_ifnw,
            "candida_infection": candida,
            "hypoparathyroidism": hp,
            "addison_disease": addison,
            "type1_diabetes": t1dm,
            "treatment": treatment,
        })
    return patients


def generate_overview() -> dict:
    """Overview data for Hereditary-Autoimmune-Polyglandular-Syndrome-Atlas."""
    return {
        "atlas":       "Hereditary-Autoimmune-Polyglandular-Syndrome-Atlas",
        "subtitle":    "Complete 8-Gene APS / Polyendocrine Autoimmunity Atlas",
        "total_genes": len(ATLAS_GENES),
        "seed_range":  f"{SEED_BASE}–{SEED_BASE+7}",
        "total_patients": 320,
        "genes": [g["gene"] for g in ATLAS_GENES],
        "gene_loci": {g["gene"]: g["locus"] for g in ATLAS_GENES},
        "inheritance_modes": {
            "AIRE":  "AR LOF",
            "FOXP3": "XL LOF",
            "CTLA4": "AD LOF haploinsufficiency",
            "LRBA":  "AR LOF",
            "IL2RA": "AR LOF",
            "STAT3": "AD GOF",
            "STAT1": "AD GOF",
            "ITCH":  "AR LOF",
        },
        "key_clinical_rules": [
            "AIRE-ANTI-IFN-OMEGA-PATHOGNOMONIC: anti-IFN-ω >100 U/mL sensitivity >95% for APS-1; screen before genetic testing",
            "APS-1-TRIAD-ORDER: CMC first (age <5) → hypoparathyroidism (age ~7) → Addison (age ~10-12)",
            "IPEX-FOXP3-NEONATAL: male neonate + T1DM + diarrhoea + eczema = IPEX until proven otherwise; absent Tregs",
            "CTLA4-LRBA-ABATACEPT-CURATIVE: both CTLA4 LOF and LRBA LOF respond dramatically to abatacept (CTLA4-Ig)",
            "LRBA-CTLA4-DISTINGUISH: LRBA AR biallelic earlier more severe CVID; CTLA4 AD haploinsufficiency later onset",
            "IL2RA-ELEVATED-IL2-DISTINGUISHER: serum IL-2 very high in CD25 deficiency (not consumed by absent Tregs)",
            "STAT3-GOF-SHORT-STATURE: STAT3 GOF causes short stature + T1DM + thyroiditis (vs STAT3 LOF = Hyper-IgE)",
            "STAT1-GOF-ANEURYSM-MANDATORY: intracranial aneurysm in 15% STAT1 GOF — annual MRA brain/neck surveillance",
            "STAT1-GOF-CMC-PARADOX: excess STAT1/IFN suppresses Th17 → CMC (loss of mucosal IL-17 defence)",
            "ITCH-DYSMORPHIC-UNIQUE: only hereditary APS gene with developmental delay + dysmorphic features",
        ],
    }


def generate_breakdown() -> dict:
    """Per-gene breakdown for all 8 APS genes."""
    genes_data = []
    for idx, g in enumerate(ATLAS_GENES):
        pts = _make_patients(SEED_BASE + idx, g["gene"])
        treatments = {}
        for p in pts:
            treatments[p["treatment"]] = treatments.get(p["treatment"], 0) + 1
        genes_data.append({
            "gene":              g["gene"],
            "locus":             g["locus"],
            "protein":           g["protein"],
            "protein_size":      g["protein_size"],
            "inheritance":       g["inheritance"],
            "disease_category":  g["disease_category"],
            "disease_pathway":   g["disease_pathway"],
            "n_patients":        len(pts),
            "mean_age_dx":       round(sum(p["age_at_diagnosis"] for p in pts) / len(pts), 1),
            "mean_treg_pct":     round(sum(p["treg_pct"] for p in pts) / len(pts), 1),
            "anti_ifnw_pct":     round(100 * sum(1 for p in pts if p["anti_ifnw_positive"]) / len(pts), 1),
            "candida_pct":       round(100 * sum(1 for p in pts if p["candida_infection"]) / len(pts), 1),
            "hp_pct":            round(100 * sum(1 for p in pts if p["hypoparathyroidism"]) / len(pts), 1),
            "addison_pct":       round(100 * sum(1 for p in pts if p["addison_disease"]) / len(pts), 1),
            "t1dm_pct":          round(100 * sum(1 for p in pts if p["type1_diabetes"]) / len(pts), 1),
            "treatment_breakdown": treatments,
            "patients":          pts,
        })
    return {
        "atlas":  "Hereditary-Autoimmune-Polyglandular-Syndrome-Atlas",
        "count":  len(genes_data),
        "genes":  genes_data,
    }


def generate_definitions() -> dict:
    """Key clinical terms for Hereditary-Autoimmune-Polyglandular-Syndrome-Atlas."""
    definitions = [
        {
            "term": "APS-1 / APECED (AIRE LOF) — Anti-IFN-ω Diagnostic Key",
            "genes": ["AIRE"],
            "definition": (
                "APS-1 (APECED): biallelic AIRE LOF → failed central tolerance → poly-endocrine autoimmunity. "
                "DIAGNOSTIC HALLMARK: anti-IFN-ω neutralising autoantibodies (>100 U/mL) in >95% of cases. "
                "These antibodies are produced years before clinical disease → "
                "screening anti-IFN-ω detects APS-1 before clinical triad develops. "
                "TRIAD ORDER (sequence is pathognomonic): "
                "1) Mucocutaneous candidiasis (CMC) — oral+nail by age 5 (IL-17 mucosal defence lost); "
                "2) Hypoparathyroidism — median age 7 (CYP11A1 as parathyroid autoantigen); "
                "3) Addison disease — median age 10-12 (21-hydroxylase as adrenal autoantigen); "
                "ADDITIONAL COMPONENTS (not in triad): "
                "Autoimmune hepatitis (CYP1A2/CYP2A6 autoAbs); ovarian failure (StAR autoAbs); "
                "Alopecia (30%); vitiligo (26%); T1DM (12%); coeliac-like. "
                "FOUNDER ALLELES: R257X (Finnish); R139X (UK/Norway); "
                "TREATMENT: component-specific (Ca2+/calcitriol/rPTH; HC/fludro; fluconazole); "
                "annual surveillance of all components."
            ),
        },
        {
            "term": "IPEX Syndrome (FOXP3 LOF) — Treg-Absent Neonatal Autoimmunity",
            "genes": ["FOXP3"],
            "definition": (
                "IPEX (FOXP3 LOF): absent Tregs → unrestrained effector T-cells → neonatal multi-organ autoimmunity. "
                "X-LINKED: affected males; female carriers usually normal. "
                "CLINICAL TRIAD: "
                "1) Neonatal/infantile T1DM (hyperglycaemia in first weeks-months); "
                "2) Intractable secretory diarrhoea (profuse, from birth, villous atrophy); "
                "3) Eczema/erythroderma (high IgE, food allergies). "
                "ADDITIONAL: AIHA, ITP, autoimmune thyroiditis, membranous nephropathy. "
                "DIAGNOSIS: absent CD4+CD25+FOXP3+ Tregs on flow cytometry. "
                "TREATMENT HIERARCHY: "
                "1) Rapamycin (preferred over tacrolimus — less Treg toxicity); "
                "2) Tacrolimus + insulin + TPN; "
                "3) HSCT (curative — restores Treg function); "
                "Bridge to HSCT <2 years for best outcome. "
                "KEY DISTINCTION: FOXP3 (X-linked, neonatal, males) vs "
                "IL2RA (autosomal, neonatal, male+female, elevated serum IL-2)."
            ),
        },
        {
            "term": "CTLA4/LRBA — Shared Pathway, Different Genetics, Same Therapy (Abatacept)",
            "genes": ["CTLA4", "LRBA"],
            "definition": (
                "CTLA4 LOF and LRBA deficiency share a final common pathway: "
                "insufficient CTLA-4 at Treg/effector T-cell surface. "
                "CTLA4 LOF (AD haploinsufficiency, 2q33.2): "
                "50% CTLA-4 → reduced transendocytosis of CD80/CD86 → effector T-cell hyperactivation. "
                "Onset: typically 20s-40s; penetrance ~60%. "
                "LRBA deficiency (AR, 4q31.3): "
                "CTLA-4 internalised after binding → without LRBA, directed to lysosomes → degraded. "
                "Surface CTLA-4 absent → phenocopies CTLA4 LOF but more severe (biallelic, earlier). "
                "FLOW CYTOMETRY KEY: "
                "CTLA4 LOF: CTLA-4 protein present but haploinsufficient; "
                "LRBA deficiency: CTLA-4 protein nearly absent on Tregs. "
                "SHARED CLINICAL FEATURES: "
                "Thyroiditis, T1DM, cytopenias (AIHA/ITP), lymphoproliferation, "
                "lymphocytic interstitial pneumonitis, bowel infiltration, hypogammaglobulinaemia. "
                "SHARED TREATMENT: ABATACEPT (CTLA4-Ig fusion protein) — replaces deficient CTLA-4; "
                ">80% response rate for lymphoproliferation/cytopenias; "
                "Weekly subcutaneous abatacept; may need life-long. "
                "Both: test CTLA4 + LRBA together — phenotypically indistinguishable."
            ),
        },
        {
            "term": "STAT3 GOF vs STAT3 LOF — Opposite Phenotypes (Anti-GOF vs Hyper-IgE)",
            "genes": ["STAT3"],
            "definition": (
                "CRITICAL DISTINCTION: STAT3 GOF ≠ STAT3 LOF. "
                "STAT3 LOF (AD dominant-negative = Job syndrome / Hyper-IgE syndrome type 1): "
                "Eczema + high IgE + Staph abscesses + pneumatoceles + skeletal/dental abnormalities; "
                "no significant autoimmunity; Th17 impaired (opposite to GOF). "
                "STAT3 GOF (AD gain-of-function): "
                "Autoimmunity + lymphoproliferation + short stature; "
                "T1DM (most common endocrine) + thyroiditis + cytopenias; "
                "Th17 EXCESSIVE (STAT3 drives RORγt); "
                "Short stature: STAT3 GOF competes with STAT5B (GH signalling) → GH resistance; "
                "JAK INHIBITORS EFFECTIVE for GOF (ruxolitinib, baricitinib). "
                "KEY FUNCTIONAL ASSAY: pSTAT3 on T-cells after IL-6/IL-10 stimulation: "
                "GOF: pSTAT3 markedly elevated vs normal; "
                "LOF: pSTAT3 impaired or absent. "
                "NEVER confuse these — treatment is opposite: "
                "GOF → JAK inhibitor to reduce STAT3; "
                "LOF → no targeted therapy; antifungal/antibiotic prophylaxis."
            ),
        },
        {
            "term": "STAT1 GOF — CMC with Autoimmunity and Intracranial Aneurysm",
            "genes": ["STAT1"],
            "definition": (
                "STAT1 GOF: excess IFN/STAT1 signalling → paradoxical CMC + autoimmunity. "
                "CMC PARADOX: "
                "STAT1 excess → inhibits STAT3-driven Th17 → reduced IL-17 at mucosa → Candida not cleared. "
                "(Counterintuitive: IFN normally anti-viral, but excess IFN suppresses anti-fungal Th17.) "
                "AUTOIMMUNITY: "
                "Excess IFN-γ upregulates MHC-II on parenchymal cells → autoantigen presentation; "
                "Thyroiditis (>50%), T1DM (~20%), alopecia, vitiligo. "
                "VASCULAR — PATHOGNOMONIC: "
                "Cerebrovascular/intracranial aneurysms (~15%): "
                "IFN-γ excess → smooth muscle apoptosis in vessel wall → aneurysm formation; "
                "MANDATORY: annual MRA brain + neck from diagnosis; "
                "Rupture = life-threatening → neurosurgical/neurointerventional referral. "
                "TREATMENT: Ruxolitinib (JAK1/2 inhibitor): "
                "Reduces pSTAT1 → restores Th17 → CMC resolves; "
                "Reverses autoimmunity; aneurysm stabilisation reported. "
                "DISTINGUISH FROM AIRE (APS-1): "
                "STAT1 GOF: CMC + thyroiditis + aneurysm; anti-IFN-ω NEGATIVE; "
                "AIRE: CMC + HP + Addison; anti-IFN-ω STRONGLY POSITIVE (>100 U/mL)."
            ),
        },
        {
            "term": "IL2RA (CD25) Deficiency — Elevated Serum IL-2 as Diagnostic Biomarker",
            "genes": ["IL2RA"],
            "definition": (
                "IL2RA (CD25) deficiency: absent high-affinity IL-2 receptor α-chain → "
                "Tregs cannot compete for IL-2 → Treg starvation → IPEX-like autoimmunity. "
                "AUTOSOMAL RECESSIVE: females equally affected (distinction from FOXP3/IPEX). "
                "DIAGNOSTIC BIOMARKER: "
                "Serum IL-2 VERY HIGH (often >10,000 pg/mL): "
                "Normal: Tregs constitutively express CD25 → consume IL-2 from microenvironment; "
                "CD25 absent: IL-2 not consumed → accumulates in serum. "
                "FLOW CYTOMETRY: CD25 absent on T-cells and NK cells (diagnostic). "
                "CLINICAL FEATURES (similar to FOXP3/IPEX but AR): "
                "Neonatal/infantile enteropathy, eczema, lymphoproliferation; "
                "T1DM less prominent than FOXP3; viral infections (CMV, herpesviruses) prominent. "
                "TREATMENT: "
                "Rapamycin (mTOR inhibitor — promotes Treg); "
                "Basiliximab (anti-CD25) paradoxically reported helpful "
                "(blocks excess IL-2 signalling on effector T-cells); "
                "HSCT for severe cases. "
                "KEY RULE: IPEX-like phenotype in female infant → check IL2RA first; "
                "then STAT3 GOF; then exclude other causes."
            ),
        },
        {
            "term": "ITCH Deficiency — Syndromic APS with Developmental Delay",
            "genes": ["ITCH"],
            "definition": (
                "ITCH deficiency (AR ITCH E3 ubiquitin ligase LOF): "
                "only hereditary APS gene causing syndromic disease with dysmorphic features + ID. "
                "CLINICAL DISTINGUISHER: "
                "Multisystem autoimmunity + developmental delay/intellectual disability + dysmorphism "
                "= ITCH until proven otherwise. "
                "PATHOMECHANISM: "
                "ITCH ubiquitylates JunB (AP-1 member) → degradation → limits Th2 cytokines; "
                "ITCH LOF → stable JunB → excessive IL-4/IL-5 → eczema/eosinophilia + Th2 autoimmunity; "
                "Also: impaired CTLA-4 ubiquitylation/trafficking → Treg dysfunction. "
                "AUTOIMMUNE MANIFESTATIONS: "
                "Autoimmune hepatitis, pulmonary disease, IBD, cytopenias, eczema. "
                "NEURODEVELOPMENTAL: "
                "Ubiquitin pathway disruption → Notch/synaptic pruning during brain development; "
                "Mild-moderate intellectual disability; ptosis; short stature. "
                "TREATMENT: corticosteroids + azathioprine/MMF; HSCT for severe cases. "
                "PANEL TESTING: Always include ITCH with "
                "AIRE/FOXP3/CTLA4/LRBA/IL2RA/STAT3/STAT1 when evaluating "
                "syndromic polyendocrine autoimmunity with neurodevelopmental features."
            ),
        },
        {
            "term": "Hereditary APS Atlas — Differential Diagnosis Summary",
            "genes": ["AIRE", "FOXP3", "CTLA4", "LRBA", "IL2RA", "STAT3", "STAT1", "ITCH"],
            "definition": (
                "HEREDITARY APS DIFFERENTIAL — 8-GENE GUIDE: "
                "AIRE (AR): CMC → HP → Addison triad; anti-IFN-ω positive; "
                "FOXP3 (XL): neonatal male T1DM + diarrhoea + eczema; absent Tregs; HSCT curative; "
                "CTLA4 (AD): adult thyroiditis + cytopenias + lymphoproliferation; abatacept; "
                "LRBA (AR): child CVID + autoimmunity + lymphoproliferation; CTLA-4 absent flow; abatacept; "
                "IL2RA (AR): IPEX-like BUT female affected + serum IL-2 very high + CD25 absent; "
                "STAT3 GOF (AD): child short stature + T1DM + thyroiditis; pSTAT3 elevated; JAK-i; "
                "STAT1 GOF (AD): CMC + thyroiditis + intracranial aneurysm (MRA mandatory); ruxolitinib; "
                "ITCH (AR): autoimmunity + developmental delay + dysmorphism; syndromic unique; "
                "TARGETED THERAPY SUMMARY: "
                "Abatacept: CTLA4 + LRBA (CTLA-4 replacement); "
                "Ruxolitinib: STAT3 GOF + STAT1 GOF (JAK inhibition); "
                "Rapamycin: FOXP3 + IL2RA (Treg-sparing immunosuppression); "
                "HSCT: FOXP3 + IL2RA + LRBA severe (curative for Treg reconstitution); "
                "Fluconazole: AIRE + STAT1 GOF (CMC antifungal). "
                "FLOW CYTOMETRY PANEL FOR APS: "
                "Treg count (CD4+CD25+FOXP3+): low in FOXP3/IL2RA; "
                "CTLA-4 expression on Tregs: low in CTLA4/LRBA; "
                "CD25 expression: absent in IL2RA; "
                "pSTAT1/pSTAT3 on stimulation: elevated in GOF variants."
            ),
        },
    ]
    return {
        "atlas":  "Hereditary-Autoimmune-Polyglandular-Syndrome-Atlas",
        "count":  len(definitions),
        "definitions": definitions,
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(generate_overview(), indent=2)[:600])
    print("\n=== BREAKDOWN (count) ===")
    bd = generate_breakdown()
    print(f"Genes: {bd['count']}")
    print("\n=== DEFINITIONS (count) ===")
    df = generate_definitions()
    print(f"Terms: {df['count']}")
