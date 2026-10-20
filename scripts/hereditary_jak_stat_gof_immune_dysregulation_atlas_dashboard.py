#!/usr/bin/env python3
"""Hereditary-JAK-STAT-GOF-Immune-Dysregulation-Atlas -- Complete 8-Gene Reference
JAK-STAT Gain-of-Function and Related Signaling Dysregulation Disorders
(STAT1-GOF-750aa-2q32.2-AD-CMC5-Autoimmunity-Chronic-Mucocutaneous-Candidiasis-PATHOGNOMONIC,
 STAT3-GOF-770aa-17q21.2-AD-Multi-Organ-Autoimmune-LPD-Short-Stature-IDDM-Ruxolitinib,
 STAT5B-GOF-787aa-17q21.2-AD-NK-LGL-Leukemia-Immune-Dysregulation-Ruxolitinib,
 STAT6-GOF-847aa-12q13.3-AD-Severe-Atopy-Eosinophilia-Anaphylaxis-Dupilumab,
 JAK1-GOF-1142aa-1p31.3-AD-Systemic-Autoimmunity-Inflammation-Baricitinib,
 SOCS1-LOF-212aa-16p13.13-AD-Myeloproliferation-Autoimmunity-SOCS1-Haploinsufficiency,
 CARD11-GOF-BENTA-1154aa-7p22.2-AD-B-Cell-Expansion-NF-kB-T-Cell-Anergy,
 IL4RA-GOF-825aa-16p12.1-AD-Atopy-IL4-IL13-STAT6-Axis-Dupilumab-Lebrikizumab):

STAT1  (Signal transducer and activator of transcription 1 / STAT1; 750aa; 2q32.2; AD-GOF;
         STAT1 encodes a transcription factor central to type I IFN (IFN-α/β), type II IFN (IFN-γ),
           and IL-27/IL-35 signaling; canonical JAK-STAT pathway: cytokine → receptor dimerization →
           JAK1/JAK2/TYK2 auto/transphosphorylation → JAK phosphorylates STAT1 at Y701 → pSTAT1
           dimerizes (homodimer = GAF for IFN-γ; STAT1/STAT2/IRF9 = ISGF3 for IFN-α/β) → nuclear
           translocation → GAS or ISRE promoter binding → IFN-stimulated gene (ISG) transcription;
         STAT1 COILED-COIL DOMAIN (CCD) GOF MUTATIONS: gain-of-function variants predominantly
           cluster in the coiled-coil domain (CCD, aa 136-315) and less commonly the DNA-binding
           domain (DBD); normal STAT1 activation requires phosphorylation → dimerization → nuclear
           translocation → nuclear export after dephosphorylation by TCP45 (PTPRT/PTPN2); GOF
           mutations impair dephosphorylation (nuclear export failure) → STAT1 hyperactivation →
           prolonged ISG transcription → impaired Th17 differentiation (STAT1 GOF suppresses IL-17
           production by inhibiting STAT3/RORγt-driven Th17 axis) → CMC;
         CHRONIC MUCOCUTANEOUS CANDIDIASIS TYPE 5 (CMC5): most common cause of CMC5; Candida
           tropicalis, albicans; nail dystrophy + oral thrush + esophageal candidiasis PATHOGNOMONIC
           TRIAD; cutaneous Candida; Dermatophyte infections (Trichophyton); ESOPHAGEAL CANDIDIASIS
           IN A NON-IMMUNOCOMPROMISED CHILD = STAT1-GOF until proven otherwise;
         AUTOIMMUNITY: systemic autoimmunity in ~35% (hypothyroidism most common; type 1 DM;
           vitiligo; immune thrombocytopenia; inflammatory bowel disease; alopecia areata);
           autoimmune hepatitis; hemolytic anemia; autoimmune cytopenias; antibody excess despite
           CMC paradox (increased ISG → less IL-17 → Candida susceptibility + increased IFN-γ →
           autoimmunity — both driven by STAT1 hyperactivation);
         INTRACRANIAL ANEURYSM: cerebrovascular aneurysms reported in ~15% of STAT1-GOF patients
           (mechanism unclear; STAT1-driven vascular inflammation); CNS/brain imaging at diagnosis;
         MALIGNANCY: thyroid carcinoma; esophageal SCC; lymphoma; STAT1-GOF surveillance protocols;
         TREATMENT: AZOLES (fluconazole/itraconazole/voriconazole) for CMC — lifelong suppression
           not always required; RUXOLITINIB (JAK1/2 inhibitor; reduces STAT1 phosphorylation →
           reverses STAT1-GOF molecular phenotype; reduces CMC, autoimmunity, cytopenias; FIRST-LINE
           targeted therapy for moderate-severe STAT1-GOF); HSCT considered for severe/refractory;
           IVIG for hypogammaglobulinemia (50% develop hypogammaglobulinemia); seed SEED_BASE+0).
STAT3  (Signal transducer and activator of transcription 3 / STAT3; 770aa; 17q21.2; AD-GOF;
         STAT3 is the major downstream mediator of IL-6, IL-10, IL-11, IL-21, IL-22, IL-27, G-CSF,
           EGF, PDGF, and multiple oncogenic kinases; canonical: cytokine → JAK1/JAK2/TYK2 →
           pSTAT3 (Y705) → STAT3 homodimer or STAT1/STAT3 heterodimer → nuclear SIE/STAT3 target
           genes (BCL-XL, VEGF, MYC, SOCS3, suppressor of Treg/Th17 balance, anti-apoptotic);
         NORMAL STAT3 FUNCTION: essential for anti-inflammatory IL-10 signaling, hepatocyte acute
           phase response, hematopoietic cell differentiation; acute activation (hours) → SOCS3
           feedback negative regulation → STAT3 dephosphorylated by PTP1B/SHP-2 → nuclear export;
         STAT3-GOF MUTATIONS (IUIS 2022 category): heterozygous AD-GOF in SH2 domain or DNA-binding
           domain or transactivation domain → impaired SOCS3 feedback → sustained pSTAT3 → blocked
           Treg differentiation → reduced IL-10 → paradoxical autoimmunity; also blocked STAT3
           nuclear export → constitutive transcription of anti-apoptotic, pro-proliferative targets;
         CLINICAL PHENOTYPE (multi-organ immune dysregulation):
           (1) EARLY-ONSET DIABETES MELLITUS (T1DM-like; islet autoantibodies; ~40%; type 1 DM
               diagnosis in first 2 years of life = STAT3-GOF UNTIL PROVEN OTHERWISE);
           (2) SHORT STATURE/GROWTH FAILURE (GH resistance via STAT3 suppression of IGF-1; growth
               hormone levels normal but IGF-1 low; GH therapy partially effective; prominent finding);
           (3) AUTOIMMUNE CYTOPENIAS (hemolytic anemia, ITP, AIHA — STAT3 suppresses apoptosis of
               autoreactive T cells; lymphopenia with paradoxical lymphoproliferation);
           (4) LYMPHOPROLIFERATION (splenomegaly, lymphadenopathy, large granular lymphocyte LGL
               expansion; lymphoma risk; EBV-driven lymphoproliferative disease);
           (5) HYPOTHYROIDISM/THYROIDITIS (autoimmune Hashimoto's thyroiditis in ~50%);
           (6) ECZEMA/ATOPY (skin inflammation; food allergies);
           (7) INTERSTITIAL LUNG DISEASE (ILD; lymphocytic interstitial pneumonia; bronchiectasis);
         DIAGNOSIS: STAT3 sequencing; STAT3 phosphorylation assay (prolonged pSTAT3 after stimulation);
           flow cytometry: reduced Tregs (CD4+CD25+FoxP3+); elevated activated STAT3 signaling;
         TREATMENT: RUXOLITINIB (most effective — reduces T1DM progression, autoimmune cytopenias,
           lymphoproliferation; start early; monitor glycemia); TOFACITINIB (alternative JAK3/1
           inhibitor); SIROLIMUS for lymphoproliferation (mTOR pathway convergence); IVIG; HSCT for
           severe lymphoma; insulin for T1DM (standard management); GH for growth; seed SEED_BASE+1).
STAT5B (Signal transducer and activator of transcription 5B / STAT5B; 787aa; 17q21.2; AD-GOF;
         STAT5B encodes STAT5B, a critical mediator of IL-2, IL-7, IL-9, IL-15, growth hormone (GH),
           prolactin, erythropoietin, and thrombopoietin signaling; STAT5A (96% identical, encoded
           nearby at 17q21.2) and STAT5B share overlapping functions but differ in tissue expression;
         NORMAL STAT5B FUNCTION: IL-2/IL-15 → JAK1/JAK3 → pSTAT5B (Y699) → Treg induction (FoxP3
           transcription); CD8+ T cell and NK cell survival; GH → JAK2 → STAT5B → IGF-1; erythroid
           and megakaryocyte maturation;
         STAT5B-GOF MUTATIONS: heterozygous AD-GOF in SH2 domain (most common: p.N642H) →
           constitutive JAK2/pSTAT5B signaling → impaired nuclear export → sustained STAT5B
           transcription → anti-apoptotic gene upregulation (BCL-XL) in NK cells and LGL T cells;
         NK-CELL LARGE GRANULAR LYMPHOCYTE (LGL) LEUKEMIA: PATHOGNOMONIC consequence of STAT5B-GOF;
           constitutive STAT5B → clonal NK-LGL or T-LGL expansion → NK-LGL leukemia; STAT5B p.N642H
           is found in ~40% of T-LGL leukemia somatic samples AND as germline GOF variant in this
           hereditary disorder; peripheral blood NK-cell expansion (CD3-CD56+CD16+) → NK-LGL;
           CD3+CD8+CD57+CD16+ T-LGL expansion also seen;
         IMMUNE DYSREGULATION PHENOTYPE: immune dysregulation with eosinophilia, skin rash,
           hemolytic anemia, thrombocytopenia; hepatosplenomegaly; NK-LGL clonal lymphocytosis;
           autoimmune cytopenias; severe opportunistic infections (herpes viruses, Pneumocystis);
           pulmonary lymphocytic infiltrates; STAT5B-LOF (distinct from GOF) causes GH resistance
           → DISTINGUISH from STAT5B-GOF by phenotype (GH resistance in LOF; immune dysreg+LGL in GOF);
         DIAGNOSIS: STAT5B sequencing (p.N642H hotspot — check both germline and somatic); peripheral
           blood NK-LGL count (elevated); T-LGL clonality (TCR Vβ flow cytometry + TCR-γ PCR); bone
           marrow biopsy (LGL infiltration); flow cytometry (CD3-CD56+LGL population elevated);
         TREATMENT: RUXOLITINIB (JAK1/2 inhibitor; reduces pSTAT5B → apoptosis of LGL clones; most
           effective targeted therapy); METHOTREXATE + CYCLOPHOSPHAMIDE (conventional LGL treatment);
           CYCLOSPORIN (reduces LGL-mediated autoimmune cytopenias); HSCT for severe lymphoma or
           refractory disease; avoid immunosuppression that further reduces NK/T-cell surveillance;
           seed SEED_BASE+2).
STAT6  (Signal transducer and activator of transcription 6 / STAT6; 847aa; 12q13.3; AD-GOF;
         STAT6 is the exclusive signaling mediator of IL-4 and IL-13 — the canonical Th2 cytokines;
           IL-4/IL-13 → IL-4Rα/IL-2Rγc (type I) or IL-4Rα/IL-13Rα1 (type II) → JAK1/JAK3 or
           JAK1/TYK2 → pSTAT6 (Y641) → STAT6 homodimer → nuclear translocation → IL-4/STAT6
           target genes: GATA3 (Th2 master regulator), IgE class switch, CD23, eotaxin (CCL11),
           periostin, mucin genes (MUC5AC), TSLP, IL-33 receptor upregulation;
         STAT6-GOF MUTATIONS: heterozygous AD-GOF in DNA-binding domain or SH2 domain or
           transactivation domain → impaired dephosphorylation by SHP-1 → constitutive STAT6
           activation in the absence of cytokine → constitutive Th2 program in all lymphocytes;
           DISCOVERED IN 2022 (Leung et al., NEJM 2023): ultra-severe atopic disease with early
           infantile onset linked to de novo STAT6 GOF mutations;
         PHENOTYPE — ULTRA-SEVERE EARLY-ONSET ATOPY:
           (1) NEONATAL/INFANTILE ECZEMA (Atopic Dermatitis): universally present; severe, early
               onset (weeks of life); refractory to topical steroids; whole-body distribution;
           (2) FOOD ALLERGIES: multiple severe food allergies (IgE-mediated anaphylaxis); peanut,
               tree nut, milk, egg, wheat; ANAPHYLAXIS risk HIGH; epinephrine auto-injector mandatory;
           (3) EOSINOPHILIA: PATHOGNOMONIC blood eosinophilia (often >1.5 × 10⁹/L); tissue
               eosinophilia (esophagus, GI tract); eosinophilic esophagitis (EoE) common;
           (4) ELEVATED IgE: extremely high total IgE (often >10,000 IU/mL); specific IgE
               multisensitized; STAT6-GOF should be suspected when IgE >2000 IU/mL in young child;
           (5) ASTHMA: severe bronchial asthma; progressive; not well controlled by standard therapy;
           (6) ALLERGIC RHINITIS/SINUSITIS: early onset, severe;
           (7) EOSINOPHILIC GASTROINTESTINAL DISEASE (EGID): EoE, eosinophilic gastritis;
         TREATMENT: DUPILUMAB (anti-IL-4Rα antibody; blocks both IL-4 and IL-13 signaling → blocks
           STAT6 activation upstream; dramatic response in STAT6-GOF; FIRST-LINE targeted therapy;
           FDA-approved for moderate-severe atopic dermatitis ≥6 months); LEBRIKIZUMAB (anti-IL-13);
           TRALOKINUMAB (anti-IL-13); RUXOLITINIB topical (for atopic dermatitis skin); strict
           allergen avoidance; epinephrine auto-injector; omalizumab (anti-IgE) adjunct;
           seed SEED_BASE+3).
JAK1   (Janus kinase 1 / JAK1; 1142aa; 1p31.3; AD-GOF;
         JAK1 encodes the 130 kDa tyrosine kinase that is the predominant JAK family member coupled to
           cytokine receptors using the common gamma chain (γc, IL-2Rγ/IL2RG), gp130 (IL-6/11/27/31),
           interferon receptors (IFNAR1/2, IFNGR), and IL-10 family receptors; JAK1 is the most
           broadly expressed JAK family member; JAK1 partners with: JAK3 (γc-dependent cytokines:
           IL-2/-4/-7/-9/-15/-21), JAK2 (gp130-dependent: IL-6; also IFN-γ), TYK2 (type I IFNs);
         JAK1 ACTIVATION MECHANISM: trans-phosphorylation: receptor dimerization → JAK1-JAK partner
           (JAK2/JAK3/TYK2) trans-phosphorylate each other at activation loop Y → activated JAKs
           phosphorylate receptor cytoplasmic tail tyrosines (docking sites) → STATs recruited →
           STAT phosphorylation → dimerization → nuclear translocation → gene transcription;
         JAK1-GOF MUTATIONS: heterozygous AD-GOF in pseudokinase domain (JH2; the regulatory domain
           that holds JAK1 in inactive conformation) → removes JH2 autoinhibition → constitutive
           JAK1 kinase activity without cytokine → constitutive STAT1, STAT3, STAT5 activation →
           systemic inflammatory and autoimmune disease; predominantly somatic in cancers but
           germline GOF is a primary immunodeficiency;
         CLINICAL PHENOTYPE:
           (1) SYSTEMIC AUTOIMMUNITY: polyarthritis (rheumatoid-like); myositis; inflammatory skin
               disease; IgA nephropathy; membranous nephropathy; autoimmune cytopenias;
           (2) ENTEROPATHY: diarrhea, protein-losing enteropathy; autoimmune gastritis;
           (3) EOSINOPHILIA: elevated eosinophils (JAK1 activates STAT3/STAT5 → IL-5 axis);
           (4) RECURRENT INFECTIONS: susceptibility to Herpes viruses, HSV, VZV (JAK1 is essential
               for type I IFN signaling → impaired anti-viral defense when JAK1 is constitutively
               hyperactivated and then downregulated by feedback); paradoxical combined immunodeficiency;
           (5) HEPATOSPLENOMEGALY: lymphocytic infiltration; portal hypertension in severe cases;
         DIAGNOSIS: JAK1 sequencing (germline GOF); pSTAT1/pSTAT3/pSTAT5 flow cytometry in
           unstimulated cells (elevated — constitutive activation without cytokine stimulation);
           cytokine panel; lymphocyte immunophenotyping;
         TREATMENT: BARICITINIB (JAK1/2 inhibitor; preferred for JAK1-GOF given JAK1 specificity;
           reduces constitutional symptoms, arthritis, cytopenias); RUXOLITINIB (JAK1/2 inhibitor;
           alternative); UPADACITINIB (selective JAK1 inhibitor; newer; may have better specificity);
           IVIG for hypogammaglobulinemia; HSCT considered for refractory severe disease; biological
           therapies (IL-6 blockade: tocilizumab) as adjuncts; seed SEED_BASE+4).
SOCS1  (Suppressor of cytokine signaling 1 / SOCS1; 212aa; 16p13.13; AD-LOF haploinsufficiency;
         SOCS1 encodes the smallest SOCS family member (212aa); SOCS proteins are negative feedback
           regulators of JAK-STAT signaling — they are INDUCED by STAT activation and then INHIBIT
           JAK activity (negative feedback loop); SOCS1 structure: SH2 domain (binds phospho-JAK
           activation loop) + SOCS box (recruits ubiquitin E3 ligase → JAK ubiquitination →
           proteasomal degradation) + N-terminal KIR (kinase inhibitory region: pseudosubstrate
           that directly blocks JAK1/JAK2 catalytic site);
         SOCS1 IS THE CRITICAL NEGATIVE REGULATOR OF IFN-γ SIGNALING: IFN-γ → JAK1/JAK2 → STAT1
           → SOCS1 transcription → SOCS1 protein → binds JAK1/2 → inhibits SOCS1 KIR (direct
           catalytic inhibition) → ubiquitin E3 (elongin BC/CRL2) → JAK1/2 degraded → IFN-γ signal
           terminated; without SOCS1: IFN-γ signals indefinitely → STAT1 hyperactivation →
           autoimmunity and inflammation;
         SOCS1 HAPLOINSUFFICIENCY (AD-LOF): monoallelic loss-of-function variants (frameshift,
           nonsense, splice-site, large deletion; missense in critical SH2 or KIR domain) →
           SOCS1 haploinsufficiency → reduced IFN-γ feedback → STAT1 hyperactivation phenotype
           resembling STAT1-GOF (DISCOVERED 2021: Maccari et al., J Allergy Clin Immunol 2021);
         CLINICAL PHENOTYPE:
           (1) AUTOIMMUNITY: systemic autoimmunity — SLE-like (ANA+, dsDNA+, complement low);
               vitiligo; Hashimoto thyroiditis; inflammatory bowel disease; cytopenias (AIHA, ITP);
           (2) MYELOPROLIFERATION: thrombocytosis; monocytosis; splenomegaly; polycythemia; MPN-like
               phenotype (SOCS1 normally suppresses JAK2-driven myeloproliferation);
           (3) MUCOCUTANEOUS CANDIDIASIS: SIMILAR TO STAT1-GOF (shared mechanism: reduced IL-17
               pathway via STAT1 hyperactivation); CMC phenotype in subset of patients;
           (4) HYPERINFLAMMATION: recurrent fever; elevated CRP/ferritin; cytokine storm;
           (5) INTERSTITIAL LUNG DISEASE: lymphocytic ILD; respiratory failure in severe cases;
         DIAGNOSIS: SOCS1 sequencing (gene + exon CNV — haploinsufficiency from deletion); ISG
           (interferon-stimulated gene) score elevated (transcriptome); pSTAT1 elevated in
           unstimulated or IFN-γ-stimulated cells; ANA/dsDNA serology;
         TREATMENT: RUXOLITINIB (JAK1/2 inhibitor → reduces JAK2/pSTAT1 hyperactivation →
           reverses myeloproliferation, autoimmunity, CMC; emerging as first-line for SOCS1
           haploinsufficiency); HYDROXYCHLOROQUINE (SLE-like features); mycophenolate mofetil;
           IVIG; HSCT for severe myeloproliferative or lymphoproliferative disease; seed SEED_BASE+5).
CARD11 (Caspase recruitment domain family member 11 / CARD11 / CARMA1; 1154aa; 7p22.2; AD-GOF;
         CARD11 encodes CARMA1, the lymphocyte-specific scaffold protein of the CBM (CARD11-BCL10-
           MALT1) signalosome; CBM is the key amplifier linking TCR/BCR signaling to NF-κB and NFAT
           activation → lymphocyte activation, proliferation, survival;
         CARD11 STRUCTURE: CARD domain (binds BCL10) + coiled-coil domain (auto-inhibitory linker
           region) + MAGUK domain (PDZ + SH3 + GUK domains; scaffold for BCL10/MALT1 recruitment);
         CARD11 GOF MECHANISM: heterozygous AD-GOF mutations predominantly in the coiled-coil/linker
           domain → disrupt autoinhibitory conformation → constitutive CBM complex formation in the
           absence of TCR/BCR stimulation → constitutive NF-κB activation (IKK-β → IκBα
           phosphorylation → NF-κB nuclear translocation) → B cell hyperproliferation and T cell
           activation defect;
         BENTA DISEASE (B-cell Expansion with NF-κB and T-cell Anergy): the clinical syndrome of
           CARD11-GOF; DISCOVERED 2013 (Snow et al., JCI 2012); key features:
           (1) POLYCLONAL B CELL EXPANSION: markedly elevated B cells (sometimes >90% of lymphocytes);
               splenomegaly; lymphadenopathy; hepatomegaly; POLYCLONAL (not malignant lymphoma) but
               lymphoma risk increased; diffuse large B-cell lymphoma (DLBCL) reported;
           (2) T CELL ANERGY/DYSFUNCTION: paradoxical — constitutive NF-κB in T cells → T cell
               exhaustion/anergy (chronic NF-κB → CTLA-4 upregulation → T cell unresponsiveness);
               recurrent infections (viral, bacterial); vaccine responses poor; combined immunodeficiency
               despite B cell excess;
           (3) ATOPY/ALLERGY: eczema, food allergies, elevated IgE (NF-κB promotes Th2 and IgE class
               switch); resembles hyper-IgE syndrome;
           (4) MUCOCUTANEOUS INFECTIONS: Staphylococcal skin infections; Candida;
         KEY DISTINCTION FROM CARD11-LOF (CID): CARD11-LOF (CBM signalosome atlas) causes
           combined immunodeficiency with ABSENT NF-κB → agammaglobulinemia/SCID-like; CARD11-GOF
           causes BENTA → B cell expansion + T cell anergy (opposite phenotype); SAME GENE,
           OPPOSITE FUNCTION → critical DDx; MALT1-inhibitor (JNJ-67856633) investigational in BENTA;
         TREATMENT: IBRUTINIB (BTK inhibitor; reduces BCR-NF-κB signaling upstream of CBM →
           partially corrects B cell expansion; most evidence for BENTA treatment); LENALIDOMIDE
           (immunomodulation); IVIG; rituximab for lymphoma; HSCT for severe cases;
           seed SEED_BASE+6).
IL4RA  (Interleukin-4 receptor subunit alpha / IL-4Rα / CD124; 825aa; 16p12.1; AD-GOF;
         IL4RA encodes the alpha subunit (825aa, 140kDa) of the type I and type II IL-4 receptors;
         TYPE I IL-4R (IL-4Rα + γc/IL2RG): expressed on lymphocytes and hematopoietic cells;
           mediates IL-4 effects on B cells (IgE class switch) and T cells (Th2 differentiation);
         TYPE II IL-4R (IL-4Rα + IL-13Rα1): expressed on non-hematopoietic cells (epithelium,
           smooth muscle, fibroblasts, neurons); mediates both IL-4 and IL-13 effects on barrier
           tissues (airway, skin, gut); signals via JAK1/TYK2 → pSTAT6;
         IL4RA GOF VARIANTS: AD-GOF polymorphisms and variants in IL-4Rα → enhance IL-4 signaling
           → STAT6 activation → IgE class switch → atopy; classical IL4RA GOF variants:
           (1) p.R576 (R576) variant: common allele (population frequency ~2-8%); creates an ITM
               (immunoreceptor tyrosine-based motif-like) → recruits additional SH2 signaling →
               enhanced IFN regulatory factor activation; INCREASED ATOPY RISK in homozygotes;
           (2) p.Q576R + p.I75V compound variants: clinical HYPER-IgE phenotype; markedly elevated IgE;
               severe eczema; asthma; food allergy; reduced Treg function (IL-4Rα GOF suppresses
               Treg IL-2-STAT5 signaling);
           (3) p.S503P: reduces phosphatase docking → prolonged pSTAT6 → severe atopy;
         BROADER CONTEXT: IL4RA-GOF represents the RECEPTOR-LEVEL IL-4/IL-13/STAT6 axis activation
           (upstream of STAT6-GOF which is the downstream transcription factor activation); both
           converge on the same Th2/IgE pathway but different molecular levels;
         PHENOTYPE:
           (1) SEVERE ATOPIC DERMATITIS: early onset; IgE-mediated; generalized; pruritic;
           (2) MULTIPLE FOOD ALLERGIES: IgE-mediated; anaphylaxis risk;
           (3) ALLERGIC ASTHMA: type 2 high eosinophilic asthma; fractional exhaled NO (FeNO) elevated;
           (4) CHRONIC RHINOSINUSITIS WITH NASAL POLYPS: IL-13-driven;
           (5) ELEVATED IgE + EOSINOPHILIA: total IgE 1000-5000 IU/mL; blood eosinophilia;
         TREATMENT: DUPILUMAB (anti-IL-4Rα antibody; blocks both IL-4 and IL-13 → DIRECTLY targets
           the GOF receptor; FDA-approved for atopic dermatitis, asthma, CRSwNP; FIRST-LINE for
           IL4RA-GOF moderate-severe atopy); LEBRIKIZUMAB/TRALOKINUMAB (anti-IL-13); OMALIZUMAB
           (anti-IgE; adjunct); strict allergen avoidance; epinephrine auto-injector; amlexanox
           (SYK inhibitor — reduces IL-4Rα downstream signaling); seed SEED_BASE+7).

ATLAS_NAME   = "Hereditary-JAK-STAT-GOF-Immune-Dysregulation-Atlas"
SEED_BASE    = 4022   # seeds 4022-4029 (8 genes × 40 patients each)
N_PER_GENE   = 40
TOTAL_N      = N_PER_GENE * 8
"""

import random
import math

ATLAS_NAME   = "Hereditary-JAK-STAT-GOF-Immune-Dysregulation-Atlas"
SEED_BASE    = 4022   # seeds 4022-4029 (8 genes × 40 patients each)
N_PER_GENE   = 40
TOTAL_N      = N_PER_GENE * 8

GENES = ["STAT1", "STAT3", "STAT5B", "STAT6", "JAK1", "SOCS1", "CARD11", "IL4RA"]

GENE_DATA = {
    "STAT1": {
        "full_name":              "Signal transducer and activator of transcription 1 / STAT1-GOF (STAT1; Chronic Mucocutaneous Candidiasis type 5 / CMC5 + Autoimmunity; AD-GOF)",
        "aa":                     750,
        "locus":                  "2q32.2",
        "kda":                    "87",
        "inheritance":            "AD-GOF (autosomal dominant gain-of-function; heterozygous CCD/DBD domain variants; de novo or familial; incomplete penetrance; most common hereditary CMC gene)",
        "syndrome":               "CMC5 — Chronic Mucocutaneous Candidiasis type 5 (STAT1 GOF); Th17 deficiency secondary to STAT1 hyperactivation; PATHOGNOMONIC: esophageal candidiasis + nail dystrophy + oral thrush in non-immunocompromised child",
        "pathway":                "STAT1-GOF (coiled-coil domain) → impaired nuclear dephosphorylation/export → prolonged pSTAT1 → suppression of STAT3/RORγt → reduced IL-17A/F, IL-22 production by Th17 cells → mucosal Candida susceptibility + autoimmunity (excess IFN-γ signaling)",
        "onset_range":            (6, 60),
        "ruxolitinib_pct":        0.60,        # emerging standard for moderate-severe STAT1-GOF
        "azole_antifungal_pct":   0.95,        # fluconazole/itraconazole/voriconazole for CMC
        "autoimmunity_pct":       0.38,        # hypothyroid, T1DM, vitiligo, AIHA
        "esophageal_candida_pct": 0.55,        # esophageal candidiasis PATHOGNOMONIC non-immunocomp
        "nail_dystrophy_pct":     0.80,        # nail dystrophy — chronic dermatophyte/Candida
        "hypogamma_pct":          0.50,        # hypogammaglobulinemia in ~50%
        "ivig_pct":               0.45,
        "hsct_pct":               0.12,        # HSCT reserved for severe refractory
        "recurrent_infect_pct":   0.88,
        "pathognomonic":          "ESOPHAGEAL CANDIDIASIS IN A NON-IMMUNOCOMPROMISED CHILD/YOUNG ADULT = STAT1-GOF UNTIL PROVEN OTHERWISE (Candida esophagitis in someone without HIV, no steroids, no chemotherapy → investigate STAT1-GOF immediately); NAIL DYSTROPHY + ORAL THRUSH + SKIN CANDIDA TRIAD (chronic mucocutaneous candidiasis); ELEVATED STAT1 PHOSPHORYLATION ASSAY: pSTAT1 prolonged after IFN-γ or IFN-α stimulation (prolonged nuclear residence due to impaired nuclear export — nuclear dephosphorylation failure by TC45/PTPRC); REDUCED TH17 CELLS (flow cytometry: CD4+IL-17A+ T cells reduced) paradoxically despite systemic inflammation; AUTOIMMUNITY (hypothyroidism, T1DM, AIHA, ITP) concurrent with CMC; INTRACRANIAL ANEURYSM on brain MRI/MRA (vascular inflammation in ~15%); HYPOGAMMAGLOBULINEMIA (50%) requiring IVIG; STAT1 CCD/DBD domain sequencing",
        "key_drug":               "AZOLES (FLUCONAZOLE first-line for CMC — 150mg weekly suppression; ITRACONAZOLE/VORICONAZOLE for refractory or Aspergillus coinfection); RUXOLITINIB (JAK1/2 inhibitor 10-20mg BID; REDUCES STAT1 PHOSPHORYLATION → reverses Th17 deficiency → CMC resolution, autoimmunity reduction, hypogamma improvement; FIRST-LINE targeted therapy for moderate-severe disease; multiple case series showing dramatic response); IVIG for hypogammaglobulinemia (monthly IgG replacement); HSCT for refractory severe cases; BRAIN MRI/MRA AT DIAGNOSIS and every 2-5yr for aneurysm surveillance; THYROID FUNCTION TESTS annually; GLUCOSE monitoring; ESOPHAGEAL CANDIDIASIS = medical emergency → IV fluconazole + endoscopy + dysphasia evaluation",
    },
    "STAT3": {
        "full_name":              "Signal transducer and activator of transcription 3 / STAT3-GOF (STAT3; Multi-Organ Autoimmunity + Lymphoproliferation + Early-Onset Diabetes + Short Stature; AD-GOF)",
        "aa":                     770,
        "locus":                  "17q21.2",
        "kda":                    "92",
        "inheritance":            "AD-GOF (autosomal dominant gain-of-function; SH2/DBD/TAD domain variants; de novo majority; variable expressivity; multi-organ immune dysregulation)",
        "syndrome":               "STAT3-GOF Immune Dysregulation — Multi-organ autoimmunity (T1DM, thyroiditis, cytopenias), lymphoproliferation (LGL, lymphoma risk), early-onset diabetes, short stature/GH resistance, interstitial lung disease",
        "pathway":                "STAT3-GOF → impaired SOCS3 feedback → sustained pSTAT3 → blocked FoxP3+ Treg differentiation → reduced IL-10 → multi-organ autoimmunity; constitutive anti-apoptotic STAT3 targets (BCL-XL, MCL-1) → lymphocyte survival → LGL expansion; STAT3 hyperactivation in liver → GH resistance → reduced IGF-1 → short stature",
        "onset_range":            (0, 24),
        "ruxolitinib_pct":        0.62,
        "t1dm_pct":               0.42,        # T1DM in neonates/infants — PATHOGNOMONIC context
        "short_stature_pct":      0.65,        # prominent growth failure
        "autoimmunity_pct":       0.75,        # multi-organ autoimmunity: thyroid, cytopenias, GI
        "lymphoproliferation_pct":0.55,        # LGL, splenomegaly, EBV-LPD risk
        "ild_pct":                0.35,        # ILD/bronchiectasis
        "cytopenias_pct":         0.60,        # AIHA, ITP, neutropenia
        "ivig_pct":               0.38,
        "hsct_pct":               0.18,
        "recurrent_infect_pct":   0.70,
        "pathognomonic":          "EARLY-ONSET TYPE 1 DIABETES IN NEONATES/INFANTS (neonatal or infantile T1DM diagnosis = STAT3-GOF UNTIL PROVEN OTHERWISE — neonatal diabetes + autoimmune thyroiditis + eczema/LPD is STAT3-GOF until proven; T1DM before age 2 is <5% idiopathic — investigate STAT3-GOF and IPEX immediately); SHORT STATURE/GROWTH FAILURE WITH NORMAL GH BUT LOW IGF-1 (GH resistance via STAT3 suppression of hepatic IGF-1; growth hormone provocative test normal → check IGF-1 → low in STAT3-GOF → GH resistance confirmed); REDUCED TREGS (flow: CD4+CD25+FoxP3+ markedly reduced — STAT3 suppresses FoxP3 transcription); MULTI-ORGAN AUTOIMMUNITY (thyroid antibodies + T1DM + ITP/AIHA occurring together); PROLONGED pSTAT3 AFTER IL-6 STIMULATION (phospho-STAT3 flow assay — sustained pSTAT3 > 60min after stimulation vs normal 20-30min); LYMPHOPROLIFERATION WITH NK-LGL OR T-LGL EXPANSION",
        "key_drug":               "RUXOLITINIB (10-25mg BID; most effective therapy — reduces T1DM progression (anti-islet T cells), autoimmune cytopenias, lymphoproliferation, IBD; START EARLY before end-organ damage); INSULIN for T1DM (standard; optimize with CGM); SIROLIMUS (mTOR inhibitor; reduces lymphoproliferation, LGL expansion; used when lymphoma risk high); TOFACITINIB (alternative JAK3/1 inhibitor for JAK3-expressing tissue autoimmunity); IVIG for hypogamma; GH therapy for short stature (partially effective — GH resistance means higher GH doses needed); HSCT for severe lymphoma or refractory disease; ANNUAL SURVEILLANCE: thyroid function, CBC, IgE/IgG, bone density (steroid use), eye exam, metabolic panel",
    },
    "STAT5B": {
        "full_name":              "Signal transducer and activator of transcription 5B / STAT5B-GOF (STAT5B; NK-Cell LGL Leukemia + Immune Dysregulation; AD-GOF)",
        "aa":                     787,
        "locus":                  "17q21.2",
        "kda":                    "92",
        "inheritance":            "AD-GOF (autosomal dominant gain-of-function; SH2 domain hotspot p.N642H; de novo or familial; constitutive STAT5B → NK-LGL leukemia; DISTINGUISH from STAT5B-LOF which causes GH resistance)",
        "syndrome":               "STAT5B-GOF NK-LGL Leukemia — NK-cell large granular lymphocyte (LGL) leukemia; immune dysregulation; eosinophilia; autoimmune cytopenias; hepatosplenomegaly; Ruxolitinib-responsive",
        "pathway":                "STAT5B-GOF (p.N642H SH2 domain) → constitutive JAK2/pSTAT5B without cytokine → BCL-XL/MCL-1 upregulation in NK cells → resistance to apoptosis → clonal NK-LGL expansion → NK-LGL leukemia; STAT5B also activates eosinophil survival (IL-5 axis) and mast cell differentiation → eosinophilia and mast cell hyperplasia",
        "onset_range":            (0, 36),
        "ruxolitinib_pct":        0.65,
        "nk_lgl_pct":             0.82,        # NK-LGL expansion HALLMARK
        "eosinophilia_pct":       0.58,
        "splenomegaly_pct":       0.75,
        "cytopenias_pct":         0.62,        # AIHA, ITP, neutropenia from LGL suppression
        "opportunistic_infect_pct":0.42,       # herpes viruses, PJP when immune dysregulation severe
        "ivig_pct":               0.35,
        "hsct_pct":               0.28,        # HSCT for severe refractory lymphoma
        "recurrent_infect_pct":   0.72,
        "pathognomonic":          "NK-CELL LARGE GRANULAR LYMPHOCYTE (LGL) EXPANSION = HALLMARK of STAT5B-GOF (peripheral blood shows >20-40% large granular lymphocytes on film; CD3-CD56+CD16+ NK-LGL population elevated on flow cytometry; bone marrow biopsy shows LGL infiltration); STAT5B p.N642H HOTSPOT MUTATION (found both as germline GOF in this hereditary disorder AND as somatic mutation in sporadic T-LGL/NK-LGL leukemia — check germline DNA not just tumor DNA); EOSINOPHILIA + HEPATOSPLENOMEGALY + LGL COMBINATION is PATHOGNOMONIC for STAT5B-GOF; DISTINGUISH FROM STAT5B-LOF (which causes GH resistance with SHORT STATURE + NORMAL LGL — phenotypic OPPOSITE); constitutive pSTAT5B in unstimulated lymphocytes (phospho-STAT5B flow assay); CYCLOSPORIN/METHOTREXATE TRIAL (if response confirms LGL-mediated autoimmune cytopenias)",
        "key_drug":               "RUXOLITINIB (JAK1/2 inhibitor; reduces constitutive pSTAT5B → NK-LGL apoptosis → reduction of LGL clone; most effective targeted therapy; dramatic response in STAT5B-GOF; 10-20mg BID); METHOTREXATE + CYCLOPHOSPHAMIDE (conventional LGL leukemia regimen; older literature); CYCLOSPORIN (reduces LGL T-cell-mediated autoimmune cytopenias; SH2 domain mediates IL-2 signaling which is STAT5B-dependent); ALEMTUZUMAB (anti-CD52; NK/T cell depleting antibody for refractory severe disease); HSCT for severe lymphoma or aplasia (STAT5B-GOF-driven aplastic anemia reported); AVOID NK-cell depleting therapies long-term (NK cells provide tumor surveillance — balance LGL reduction against viral susceptibility); G-CSF for neutropenia",
    },
    "STAT6": {
        "full_name":              "Signal transducer and activator of transcription 6 / STAT6-GOF (STAT6; Ultra-Severe Atopy + Eosinophilia + Anaphylaxis; AD-GOF; discovered 2022-2023)",
        "aa":                     847,
        "locus":                  "12q13.3",
        "kda":                    "94",
        "inheritance":            "AD-GOF (autosomal dominant gain-of-function; DBD/SH2/TAD domain variants; predominantly de novo; infantile onset; ultra-high IgE; discovered 2022-2023 NEJM)",
        "syndrome":               "STAT6-GOF Hyper-IgE/Atopy — Ultra-severe early-onset atopic dermatitis + multiple food allergies + eosinophilia + asthma + anaphylaxis; constitutive STAT6 activation without IL-4/IL-13; Dupilumab-responsive",
        "pathway":                "STAT6-GOF (DNA-binding domain) → constitutive STAT6 transcription without IL-4/IL-13 → constitutive GATA3 (Th2 master regulator) + IgE class switch + eotaxin/CCL11 → Th2 hyperactivation → IgE overproduction → eosinophilia → atopic multi-organ disease; constitutive IL-4-independent IgE class switch in B cells",
        "onset_range":            (0, 12),
        "dupilumab_pct":          0.88,        # dupilumab: FIRST-LINE targeted therapy
        "eczema_pct":             0.98,        # universally present — universally severe
        "food_allergy_pct":       0.85,        # multiple severe food allergies with anaphylaxis risk
        "eosinophilia_pct":       0.90,        # PATHOGNOMONIC blood eosinophilia
        "asthma_pct":             0.72,
        "high_ige_pct":           0.95,        # IgE often >2000-10000 IU/mL
        "eoe_pct":                0.55,        # eosinophilic esophagitis
        "anaphylaxis_pct":        0.65,        # anaphylaxis risk HIGH — epinephrine mandatory
        "ivig_pct":               0.15,
        "hsct_pct":               0.05,
        "recurrent_infect_pct":   0.52,
        "pathognomonic":          "NEONATAL/INFANTILE ECZEMA + IgE >2000 IU/mL IN FIRST YEAR OF LIFE = STAT6-GOF (or IL4RA-GOF) UNTIL PROVEN OTHERWISE; ULTRA-ELEVATED TOTAL IgE (often >10,000 IU/mL — one of the highest IgE levels seen in ANY disorder; STAT6-GOF patients can have IgE >50,000 IU/mL); BLOOD EOSINOPHILIA >1.5 × 10⁹/L PATHOGNOMONIC in context of severe atopy (combined eosinophilia + hyperIgE + infantile eczema = JAK-STAT axis atopy panel); MULTIPLE SIMULTANEOUS ANAPHYLACTIC FOOD ALLERGIES (unusual even for severe atopic dermatitis — 5+ foods causing anaphylaxis suggests STAT6-GOF); CONSTITUTIVE STAT6 PHOSPHORYLATION IN UNSTIMULATED CELLS (phospho-STAT6 elevated before IL-4 stimulation — GOF confirmation); DUPILUMAB DRAMATIC RESPONSE (blocks IL-4Rα → prevents both IL-4 and IL-13 from activating STAT6-GOF receptor → upstream block bypasses constitutive STAT6 activation somewhat; dramatic clinical response confirms diagnosis functionally)",
        "key_drug":               "DUPILUMAB (anti-IL-4Rα; FDA-approved from 6 months for atopic dermatitis, asthma, CRSwNP; FIRST-LINE for STAT6-GOF — blocks IL-4 and IL-13 receptor upstream → partially bypasses constitutive STAT6 by reducing IL-4/IL-13-driven amplification; 300mg SC q2 weeks adult; weight-based in children; expect 50-80% EASI/IGA improvement); LEBRIKIZUMAB (anti-IL-13 monoclonal; second-line alternative); TRALOKINUMAB (anti-IL-13); RUXOLITINIB topical 1.5% (topical JAK inhibitor for AD skin; FDA-approved); OMALIZUMAB (anti-IgE; add-on for food allergy/asthma; reduces free IgE but doesn't address STAT6-GOF mechanism); EPIPEN (epinephrine auto-injector): MANDATORY in all STAT6-GOF patients (anaphylaxis risk); DIETITIAN REFERRAL (multiple food allergies; prevent nutritional deficiency); ORAL IMMUNOTHERAPY (OIT) for select food allergens with specialist oversight",
    },
    "JAK1": {
        "full_name":              "Janus kinase 1 / JAK1-GOF (JAK1; Systemic Autoimmunity + Inflammation + Enteropathy; AD-GOF)",
        "aa":                     1142,
        "locus":                  "1p31.3",
        "kda":                    "133",
        "inheritance":            "AD-GOF (autosomal dominant gain-of-function; JH2 pseudokinase domain variants removing autoinhibition; de novo or familial; STAT1/STAT3/STAT5 constitutive activation)",
        "syndrome":               "JAK1-GOF Immune Dysregulation — systemic autoimmunity, enteropathy, eosinophilia, recurrent herpes viral infections; constitutive JAK1 kinase activity → multi-STAT hyperactivation",
        "pathway":                "JAK1-GOF (JH2 pseudokinase domain) → loss of autoinhibitory conformation → constitutive JAK1 kinase → constitutive pSTAT1/pSTAT3/pSTAT5 without cytokine → multi-organ inflammation (STAT1 → IFN pathway; STAT3 → IL-6 pathway; STAT5 → IL-2/hematopoiesis); paradoxical impaired type I IFN anti-viral defense via feedback dysregulation",
        "onset_range":            (6, 48),
        "baricitinib_pct":        0.55,        # baricitinib preferred for JAK1 specificity
        "ruxolitinib_pct":        0.40,
        "autoimmunity_pct":       0.80,        # polyarthritis, myositis, nephropathy, skin
        "enteropathy_pct":        0.60,        # diarrhea, protein-losing, autoimmune gastritis
        "eosinophilia_pct":       0.45,
        "herpes_viral_pct":       0.52,        # HSV, VZV, EBV — paradoxical viral susceptibility
        "hepatosplenomegaly_pct": 0.48,
        "ivig_pct":               0.42,
        "hsct_pct":               0.15,
        "recurrent_infect_pct":   0.75,
        "pathognomonic":          "POLYARTHRITIS + ENTEROPATHY + EOSINOPHILIA COMBINATION in a young patient = JAK1-GOF screening (this triad is unusual in standard rheumatologic conditions); CONSTITUTIVE pSTAT1/pSTAT3/pSTAT5 IN UNSTIMULATED CELLS (all three STAT proteins simultaneously elevated — multi-STAT hyperactivation is PATHOGNOMONIC for JAK1-GOF vs more selective STAT-specific GOF variants); PARADOXICAL HERPES VIRAL SUSCEPTIBILITY (recurrent HSV-1 labialis, VZV reactivation, EBV-LPD — JAK1 is essential for type I IFN signaling; chronic JAK1 hyperactivation → SOCS1 feedback upregulation → eventual IFN signaling suppression → viral susceptibility; give prophylactic acyclovir); ELEVATED ANA/ENA + MYOSITIS (CK elevated, anti-Jo-1 or anti-Mi-2 antibodies in some); JAK1 sequencing (JH2 domain variants: p.A634D, p.Y652C, and others); pSTAT flow assay (unstimulated multi-STAT elevation)",
        "key_drug":               "BARICITINIB (selective JAK1/2 inhibitor; 2mg/4mg daily; FDA-approved for RA, atopic dermatitis, alopecia areata; preferred for JAK1-GOF due to JAK1 specificity; reduces polyarthritis, enteropathy, eosinophilia); RUXOLITINIB (JAK1/2 inhibitor; alternative, especially for myeloproliferative features); UPADACITINIB (highly selective JAK1 inhibitor; 15-45mg daily; most specific for JAK1-GOF; FDA-approved for RA, PsA, AS, AD, UC, CD); ACYCLOVIR/VALACYCLOVIR PROPHYLAXIS (prevent HSV/VZV reactivation; lifelong antiviral prophylaxis recommended for JAK1-GOF with viral susceptibility); IVIG for hypogammaglobulinemia; ORAL BUDESONIDE for enteropathy; HSCT for refractory severe disease",
    },
    "SOCS1": {
        "full_name":              "Suppressor of cytokine signaling 1 / SOCS1 haploinsufficiency (SOCS1; JAK-STAT Upregulation + SLE-Like Autoimmunity + Myeloproliferation + CMC; AD-LOF haploinsufficiency; 2021 discovery)",
        "aa":                     212,
        "locus":                  "16p13.13",
        "kda":                    "24",
        "inheritance":            "AD-LOF haploinsufficiency (autosomal dominant loss-of-function; monoallelic frameshift/nonsense/large deletion → SOCS1 haploinsufficiency → IFN-γ/JAK-STAT hyperactivation; ~50% protein production insufficient for negative feedback; 2021 discovery)",
        "syndrome":               "SOCS1 Haploinsufficiency — SLE-like autoimmunity + myeloproliferation + mucocutaneous candidiasis; phenotypic overlap with STAT1-GOF; SOCS1 is the critical IFN-γ feedback regulator whose absence mimics STAT1 hyperactivation",
        "pathway":                "SOCS1-LOF → insufficient SOCS1 negative feedback → IFN-γ → JAK1/JAK2 → pSTAT1 signaling prolonged (normally SOCS1 KIR directly blocks JAK1/2 active site + SOCS-box ubiquitinates JAK for degradation) → chronic STAT1 hyperactivation → Th17 suppression → CMC; SOCS1 also normally suppresses JAK2 (myeloproliferative signaling) → SOCS1 haploinsufficiency → JAK2 hypersignaling → myeloproliferation (thrombocytosis, monocytosis, splenomegaly, MPN-like features)",
        "onset_range":            (12, 120),
        "ruxolitinib_pct":        0.58,
        "autoimmunity_pct":       0.82,        # SLE-like (ANA+, dsDNA+), vitiligo, thyroiditis
        "myeloproliferation_pct": 0.65,        # thrombocytosis, monocytosis, splenomegaly
        "cmc_pct":                0.42,        # CMC similar to STAT1-GOF (shared mechanism)
        "isg_score_elevated_pct": 0.90,        # interferon-stimulated gene score elevated
        "ild_pct":                0.30,
        "cytopenias_pct":         0.45,
        "ivig_pct":               0.28,
        "hsct_pct":               0.14,
        "recurrent_infect_pct":   0.68,
        "pathognomonic":          "SLE-LIKE AUTOIMMUNITY (ANA+, anti-dsDNA+, complement low, lupus nephritis) WITH CONCURRENT MYELOPROLIFERATION (thrombocytosis, splenomegaly) = SOCS1 HAPLOINSUFFICIENCY UNTIL PROVEN OTHERWISE (standard SLE does NOT cause myeloproliferation — the combination of SLE-like disease + MPN-like features in the same patient uniquely suggests SOCS1 LOF or JAK-STAT axis disorder); ELEVATED INTERFERON SIGNATURE SCORE (ISG score: transcriptomic upregulation of IFN-stimulated genes IFIT1/IFIT2/IFIT3/IFI44/MX1 — same elevation as type I interferonopathies but SOCS1 haploinsufficiency is driven by IFN-γ/JAK2 axis not type I IFN specifically); CMC IN CONTEXT OF SLE-LIKE DISEASE (unusual — Candida + lupus combination should trigger SOCS1 testing); SOCS1 protein undetectable by Western blot (50% reduction below haploinsufficiency threshold); SOCS1 sequencing (frameshift, nonsense, large deletion by MLPA)",
        "key_drug":               "RUXOLITINIB (JAK1/2 inhibitor; reduces JAK2-pSTAT1/pSTAT3 hyperactivation → reverses both myeloproliferation AND autoimmunity; SOCS1 normally suppresses JAK2 and JAK1 → ruxolitinib pharmacologically replaces SOCS1 function; 10-20mg BID; monitor CBC — JAK2 inhibition may cause cytopenias in already-cytopenic patients); HYDROXYCHLOROQUINE (SLE-like features: ANA/dsDNA, nephritis, serositis; 5mg/kg/day max); MYCOPHENOLATE MOFETIL (lupus nephritis; antifibrotic for ILD); AZOLES for CMC (fluconazole suppressive therapy same as STAT1-GOF); ASPIRIN (thrombocytosis-related thrombosis prevention); HYDROXYUREA (MPN-like thrombocytosis if JAKi alone insufficient); SOCS1 gene MLPA (large deletion screening) + sequencing",
    },
    "CARD11": {
        "full_name":              "Caspase recruitment domain family member 11 / CARD11-GOF / BENTA disease (CARD11 CARMA1; B-cell Expansion with NF-κB + T-cell Anergy; AD-GOF; DISTINCT from CARD11-LOF/CID)",
        "aa":                     1154,
        "locus":                  "7p22.2",
        "kda":                    "130",
        "inheritance":            "AD-GOF (autosomal dominant gain-of-function; coiled-coil/linker domain mutations; constitutive NF-κB activation; SAME GENE as CARD11-LOF/CID but OPPOSITE PHENOTYPE — critical distinction)",
        "syndrome":               "BENTA Disease — B-cell Expansion with NF-κB and T-cell Anergy; polyclonal B cell hyperproliferation + splenomegaly + atopy + combined immunodeficiency (paradoxical — constitutive NF-κB causes T cell anergy not activation)",
        "pathway":                "CARD11-GOF (coiled-coil domain) → disrupts autoinhibitory linker → constitutive CBM signalosome assembly → constitutive IKK-β → NF-κB nuclear translocation without TCR/BCR stimulation → B cell: constitutive NF-κB → hyperproliferation (BCL-XL + Pax5 upregulation) + IgE class switch → B cell expansion; T cells: chronic NF-κB → CTLA-4 upregulation + TCR signaling exhaustion → T cell anergy → paradoxical combined immunodeficiency despite B cell excess",
        "onset_range":            (0, 18),
        "ibrutinib_pct":          0.42,        # BTK inhibitor reducing CBM-NF-κB upstream
        "b_cell_expansion_pct":   0.95,        # PATHOGNOMONIC polyclonal B cell expansion
        "splenomegaly_pct":       0.85,
        "atopy_pct":              0.72,        # eczema, elevated IgE, food allergy
        "t_cell_anergy_pct":      0.80,        # poor vaccine responses, recurrent viral infections
        "lymphoma_risk_pct":      0.25,        # DLBCL risk elevated (polyclonal → clonal evolution)
        "ivig_pct":               0.65,        # IVIG for functional antibody deficiency
        "hsct_pct":               0.20,
        "recurrent_infect_pct":   0.78,
        "pathognomonic":          "MASSIVE POLYCLONAL B CELL EXPANSION (flow cytometry: B cells >40-70% of lymphocytes; CD19+CD20+ B cell absolute count 10-50× normal; splenomegaly; hepatomegaly) — PATHOGNOMONIC for BENTA (very few other disorders cause this degree of polyclonal B cell expansion); T CELL ANERGY PARADOX (normal or elevated B cells but POOR VACCINE ANTIBODY RESPONSES and recurrent infections — constitutive NF-κB → chronic T cell activation → CTLA-4-mediated T cell anergy; combined immunodeficiency despite B cell excess is the BENTA paradox); ATOPY + HIGH IgE (NF-κB drives IgE class switch and Th2; eczema, food allergy in BENTA); CARD11 COILED-COIL/LINKER DOMAIN MUTATION (DISTINGUISH from CARD11-LOF mutations in MAGUK/CARD domain → CID/SCID-like — completely different phenotype); NF-κB REPORTER ASSAY (constitutive NF-κB activity in patient T cells without stimulation); IBRUTINIB RESPONSE (BTK inhibitor reduces upstream BCR-NF-κB signaling → partially corrects B cell expansion)",
        "key_drug":               "IBRUTINIB (BTK inhibitor 420mg daily; FDA-approved for CLL/MCL; reduces BCR-driven upstream CBM activation → corrects polyclonal B cell expansion in BENTA; most evidence for BENTA-specific treatment; case series show dramatic B cell reduction); LENALIDOMIDE (immunomodulatory; reduces B cell expansion + enhances T cell function; used for BENTA with lymphoma risk); IVIG (monthly IVIG to replace functionally deficient antibodies despite B cell excess — B cells present but T cell anergy means poor affinity maturation → IVIG essential); RITUXIMAB (anti-CD20; for refractory B cell expansion or DLBCL; caution — profound B cell depletion in already-immunodeficient patient); HSCT for severe DLBCL or refractory combined immunodeficiency; DISTINGUISH from CARD11-LOF: CARD11-LOF (CID) → no B cells, SCID-like → HSCT urgently (see hereditary-ICF-CBM-signalosome-atlas); CARD11-GOF (BENTA) → B cell excess + T cell anergy → IBRUTINIB first",
    },
    "IL4RA": {
        "full_name":              "Interleukin-4 receptor subunit alpha / IL-4Rα-GOF / CD124 (IL4RA; Severe Atopy + Allergic Disease + IL-4/IL-13/STAT6 Axis; AD-GOF)",
        "aa":                     825,
        "locus":                  "16p12.1",
        "kda":                    "140",
        "inheritance":            "AD-GOF (autosomal dominant gain-of-function; receptor-level IL-4/IL-13 STAT6 axis activation; IL4RA GOF variants p.R576, p.Q576R, p.I75V, p.S503P enhance IL-4 signaling; complementary to STAT6-GOF at transcription factor level)",
        "syndrome":               "IL4RA-GOF Atopy — Severe atopic dermatitis + multiple food allergies + asthma + CRSwNP; elevated IgE + eosinophilia; receptor-level IL-4/IL-13/STAT6 axis hyperactivation; Dupilumab-responsive (directly blocks IL-4Rα)",
        "pathway":                "IL4RA-GOF → enhanced IL-4/IL-13 receptor activation → JAK1/TYK2 → pSTAT6 → GATA3/Th2 program + IgE class switch + eotaxin → atopic multi-organ disease; IL4RA-GOF also reduces Treg IL-2 signaling (IL-2Rγc shared with IL-4Rα on type I receptor) → reduced FoxP3+ Tregs → less peripheral tolerance → more autoimmunity risk; UPSTREAM ACTIVATION (receptor level) vs STAT6-GOF (transcription factor level) — both converge on same STAT6 targets",
        "onset_range":            (0, 24),
        "dupilumab_pct":          0.90,        # dupilumab directly blocks IL-4Rα — FIRST-LINE
        "eczema_pct":             0.95,
        "food_allergy_pct":       0.72,
        "asthma_pct":             0.68,
        "crsnp_pct":              0.45,        # chronic rhinosinusitis with nasal polyps
        "eosinophilia_pct":       0.62,
        "high_ige_pct":           0.85,        # IgE 1000-5000 IU/mL (less extreme than STAT6-GOF)
        "anaphylaxis_pct":        0.48,
        "ivig_pct":               0.12,
        "hsct_pct":               0.02,
        "recurrent_infect_pct":   0.45,
        "pathognomonic":          "SEVERE ATOPIC DERMATITIS + MULTIPLE FOOD ALLERGIES + ASTHMA + NASAL POLYPS QUADRAD = IL4RA-GOF (or STAT6-GOF) SCREENING (the combination of all four atopic diseases together suggests type 2/STAT6 axis monogenic disorder; standard AD patients rarely have all four simultaneously); DUPILUMAB EXCEPTIONAL RESPONSE (>90% EASI improvement — dupilumab directly blocks IL-4Rα, the GOF receptor, providing mechanism-specific therapy; dramatic response distinguishes IL4RA-GOF from common polygenic atopy); IL4RA SEQUENCING (GOF variants: p.R576, p.Q576R, p.I75V compound, p.S503P; note: p.R576 is present in general population at low frequency — pathogenicity depends on zygosity and clinical severity); ELEVATED FeNO (fractional exhaled NO >50 ppb; type 2 airway inflammation; STAT6-driven iNOS expression in airway epithelium); TREG REDUCTION (flow cytometry: CD4+CD25+FoxP3+ reduced — IL-4 via shared γc chain competes with IL-2 for STAT5/Treg induction); EOSINOPHIL COUNT >1 × 10⁹/L + IgE >1000 IU/mL in young child",
        "key_drug":               "DUPILUMAB (FIRST-LINE; anti-IL-4Rα monoclonal antibody; FDA-approved ≥6 months for AD, 12+ for asthma, 18+ for CRSwNP; DIRECTLY BLOCKS THE GOF RECEPTOR — mechanism-specific; dosing: 200mg/300mg SC q2 weeks adult; weight-based children; expect dramatic improvement in AD, asthma, FeNO, food allergy threshold; can reduce IgE by 70-90% over 6-12 months); LEBRIKIZUMAB/TRALOKINUMAB (anti-IL-13; second-line for AD if dupilumab inadequate); OMALIZUMAB (anti-IgE; add-on for food allergy/asthma; reduces free IgE; approved age ≥6 years); ABROCITINIB/UPADACITINIB (JAK1 inhibitors; for AD if dupilumab inadequate; address downstream JAK1/STAT6 signaling); ALLERGEN IMMUNOTHERAPY (SLIT/SCIT for select allergens after anaphylaxis risk assessment; consult allergist); EPIPEN (mandatory for food anaphylaxis risk); NASAL STEROIDS + SALINE IRRIGATION for CRSwNP; MEPOLIZUMAB/BENRALIZUMAB (anti-IL-5 for refractory eosinophilia)",
    },
}


def _rng(seed):
    r = random.Random(seed)
    return r


def _aggregate_gene(gene, idx):
    gd = GENE_DATA[gene]
    seed = SEED_BASE + idx
    rng = _rng(seed)

    n = N_PER_GENE
    cohort = []
    for i in range(n):
        s = _rng(seed * 1000 + i)
        age_m = s.randint(*gd["onset_range"])
        sex = s.choice(["M", "F"])   # all AD or AR = equal sex
        cohort.append({
            "age_onset_months": age_m,
            "sex":               sex,
        })

    male_n = sum(1 for p in cohort if p["sex"] == "M")

    def _p(key):
        base = gd.get(key, 0.0)
        noise = rng.gauss(0, 0.03)
        return round(max(0.0, min(1.0, base + noise)), 3)

    return {
        "gene":                    gene,
        "full_name":               gd["full_name"],
        "aa":                      gd["aa"],
        "locus":                   gd["locus"],
        "kda":                     gd["kda"],
        "inheritance":             gd["inheritance"],
        "syndrome":                gd["syndrome"],
        "n":                       n,
        "male_n":                  male_n,
        "female_n":                n - male_n,
        "median_onset_months":     sorted(p["age_onset_months"] for p in cohort)[n // 2],
        "ruxolitinib_pct":         _p("ruxolitinib_pct"),
        "hsct_pct":                _p("hsct_pct"),
        "recurrent_infect_pct":    _p("recurrent_infect_pct"),
        "pathognomonic":           gd["pathognomonic"],
        "key_drug":                gd["key_drug"],
        "seed":                    seed,
    }


def generate_overview():
    cohorts = [_aggregate_gene(g, i) for i, g in enumerate(GENES)]

    total_ruxo    = round(sum(c["ruxolitinib_pct"] for c in cohorts) / len(cohorts), 3)
    total_hsct    = round(sum(c["hsct_pct"]         for c in cohorts) / len(cohorts), 3)
    total_infect  = round(sum(c["recurrent_infect_pct"] for c in cohorts) / len(cohorts), 3)

    return {
        "atlas":    ATLAS_NAME,
        "genes":    GENES,
        "total_n":  TOTAL_N,
        "seeds":    f"{SEED_BASE}-{SEED_BASE + 7}",
        "aggregate_stats": {
            "ruxolitinib_pct":             total_ruxo,
            "hsct_pct":                    total_hsct,
            "recurrent_infections_pct":    total_infect,
        },
        "key_facts": [
            "STAT1-GOF (AD 2q32.2 750aa): Chronic Mucocutaneous Candidiasis type 5 (CMC5) — ESOPHAGEAL CANDIDIASIS IN NON-IMMUNOCOMPROMISED CHILD = STAT1-GOF PATHOGNOMONIC; impaired STAT1 nuclear dephosphorylation → prolonged pSTAT1 → suppressed Th17 → CMC; autoimmunity in 35% (hypothyroid, T1DM, AIHA); intracranial aneurysm risk (~15%); RUXOLITINIB first-line targeted therapy; azoles for CMC",
            "STAT3-GOF (AD 17q21.2 770aa): Multi-organ immune dysregulation — EARLY-ONSET T1DM IN NEONATES/INFANTS = STAT3-GOF UNTIL PROVEN OTHERWISE; growth failure/GH resistance (normal GH, low IGF-1); lymphoproliferation (LGL/EBV-LPD); reduced Tregs; multi-organ autoimmunity; RUXOLITINIB + SIROLIMUS most effective; insulin for T1DM",
            "STAT5B-GOF (AD 17q21.2 787aa): NK-Cell LGL Leukemia — NK-LGL EXPANSION PATHOGNOMONIC (CD3-CD56+LGL elevated); p.N642H SH2 HOTSPOT (found both germline GOF and somatic in sporadic T-LGL); eosinophilia; hepatosplenomegaly; autoimmune cytopenias; DISTINGUISH FROM STAT5B-LOF (which causes GH resistance NOT LGL); RUXOLITINIB most effective; HSCT for severe lymphoma",
            "STAT6-GOF (AD 12q13.3 847aa): Ultra-severe atopy — IgE >10,000 IU/mL PATHOGNOMONIC (highest IgE seen in any disorder); constitutive STAT6 WITHOUT IL-4/IL-13; infantile eczema + anaphylactic food allergies + eosinophilia + asthma; DUPILUMAB FIRST-LINE targeted therapy; 2022-2023 NEJM discovery; EPIPEN MANDATORY",
            "JAK1-GOF (AD 1p31.3 1142aa): Systemic autoimmunity — JH2 pseudokinase domain variants → constitutive multi-STAT hyperactivation (STAT1+STAT3+STAT5 simultaneously); polyarthritis + enteropathy + eosinophilia triad; paradoxical herpes viral susceptibility; BARICITINIB/UPADACITINIB preferred (selective JAK1 inhibitors); prophylactic acyclovir for HSV/VZV",
            "SOCS1-LOF haploinsufficiency (AD 16p13.13 212aa): JAK-STAT upregulation — SLE-like autoimmunity (ANA+/dsDNA+) + MYELOPROLIFERATION COMBINATION PATHOGNOMONIC (standard SLE does not cause thrombocytosis/MPN-features — unique to SOCS1); CMC phenotype similar to STAT1-GOF (shared SOCS1/STAT1 axis); elevated ISG score; 2021 discovery; RUXOLITINIB replaces SOCS1 negative feedback pharmacologically",
            "CARD11-GOF / BENTA (AD 7p22.2 1154aa): B-Cell Expansion with NF-κB + T-cell Anergy — POLYCLONAL B CELL EXPANSION PATHOGNOMONIC (B cells >40-70% of lymphocytes); paradoxical combined immunodeficiency despite B cell excess (T cell anergy); atopy + high IgE; CRITICAL DDx from CARD11-LOF/CID (same gene, opposite phenotype); IBRUTINIB reduces B cell expansion; LYMPHOMA RISK 25%",
            "IL4RA-GOF (AD 16p12.1 825aa): Severe atopy — Receptor-level IL-4/IL-13/STAT6 axis (upstream of STAT6-GOF); severe AD + multiple food allergies + asthma + CRSwNP quadrad; IgE 1000-5000 IU/mL; DUPILUMAB DIRECTLY BLOCKS GOF RECEPTOR (mechanism-specific); FeNO elevated (type 2 airway inflammation); Treg reduction; EPIPEN mandatory",
        ],
        "stat_gof_group":          ["STAT1", "STAT3", "STAT5B", "STAT6"],
        "jak_gof_group":           ["JAK1"],
        "socs_lof_group":          ["SOCS1"],
        "cbm_nfkb_group":          ["CARD11"],
        "il4_il13_axis_group":     ["STAT6", "IL4RA"],
        "ruxolitinib_responsive":  ["STAT1", "STAT3", "STAT5B", "JAK1", "SOCS1"],
        "dupilumab_responsive":    ["STAT6", "IL4RA"],
        "neonatal_onset":          ["STAT3", "STAT6", "CARD11"],
        "cmc_group":               ["STAT1", "SOCS1"],
        "nk_lgl_group":            ["STAT5B"],
        "atopy_group":             ["STAT6", "IL4RA", "CARD11"],
        "cohort_summaries":         cohorts,
    }


def generate_breakdown():
    cohorts = [_aggregate_gene(g, i) for i, g in enumerate(GENES)]

    diagnostic_groups = {
        "stat_gof_cmc_autoimmunity": {
            "genes":          ["STAT1", "SOCS1"],
            "mechanism":      "STAT1-GOF (coiled-coil domain): impaired STAT1 nuclear dephosphorylation → prolonged pSTAT1 → STAT3/RORγt/Th17 suppression → CMC + autoimmunity. SOCS1-LOF (haploinsufficiency): insufficient IFN-γ negative feedback → JAK1/JAK2 chronic activation → pSTAT1 excess → same downstream effect as STAT1-GOF; additionally: SOCS1 haploinsufficiency releases JAK2 (myeloproliferative signaling) → thrombocytosis/MPN features absent in STAT1-GOF",
            "shared_feature": "Both cause CMC phenotype via STAT1 hyperactivation → Th17 suppression; autoimmunity (thyroid, cytopenias, IBD); esophageal candidiasis; nail dystrophy; reduced Th17 cells; elevated ISG score; ruxolitinib-responsive",
            "key_ddx":        "STAT1-GOF vs SOCS1-LOF: MYELOPROLIFERATION (thrombocytosis, MPN-like) present in SOCS1-LOF but NOT in STAT1-GOF — this distinguishes them; SOCS1-LOF: SOCS1 protein reduced by Western blot (50%); SLE-like autoimmunity (ANA/dsDNA+) more prominent in SOCS1-LOF; STAT1-GOF: coiled-coil domain variants on sequencing; prolonged pSTAT1 after IFN stimulation specific to STAT1-GOF; BOTH respond to ruxolitinib; BOTH have CMC; intracranial aneurysm in STAT1-GOF but NOT SOCS1-LOF",
        },
        "stat3_stat5b_lgl_lymphoproliferative": {
            "genes":          ["STAT3", "STAT5B"],
            "mechanism":      "STAT3-GOF: sustained pSTAT3 → impaired Treg induction (FoxP3 transcription suppressed) → multi-organ autoimmunity + LGL expansion + early-onset T1DM + GH resistance. STAT5B-GOF: constitutive pSTAT5B → BCL-XL/MCL-1 in NK cells → NK-LGL leukemia; eosinophil survival (IL-5 axis) → eosinophilia; STAT5B normally drives FoxP3 Treg induction from IL-2 signaling — GOF may paradoxically reduce Treg function if chronically activated",
            "shared_feature": "Both cause lymphoproliferation (T-LGL/NK-LGL); both ruxolitinib-responsive; both involve 17q21.2 locus (STAT3 770aa, STAT5B 787aa adjacent); both cause immune dysregulation with cytopenias; both can progress to overt hematologic malignancy",
            "key_ddx":        "STAT3-GOF vs STAT5B-GOF: EARLY-ONSET T1DM + SHORT STATURE = STAT3-GOF (STAT3 suppresses IGF-1 → GH resistance); NK-LGL DOMINANT = STAT5B-GOF (p.N642H hotspot; NK cells CD3-CD56+LGL elevated); STAT5B-LOF (distinct entity: short stature + GH resistance WITHOUT LGL — the OPPOSITE of STAT5B-GOF; confirm by sequencing for LOF vs GOF variant type); eosinophilia more prominent in STAT5B-GOF; multi-organ autoimmunity with T1DM is STAT3-GOF signature; BOTH need JAK inhibitor; STAT3 typically STAT3/SOCS3 axis; STAT5B uses JAK2/STAT5B axis",
        },
        "il4_il13_stat6_atopy_axis": {
            "genes":          ["STAT6", "IL4RA"],
            "mechanism":      "Both converge on IL-4/IL-13 → STAT6 → GATA3/Th2 hyperactivation → IgE class switch + eosinophilia + atopic multi-organ disease. STAT6-GOF: transcription factor level (constitutive STAT6 even without IL-4/IL-13 ligand); discovered 2022-2023. IL4RA-GOF: receptor level (enhanced IL-4/IL-13 receptor signaling → more pSTAT6 per cytokine molecule; requires IL-4/IL-13 ligand but amplified signal)",
            "shared_feature": "Both cause severe early-onset atopy; elevated IgE + eosinophilia; multiple food allergies with anaphylaxis; asthma; DUPILUMAB first-line for both (blocks IL-4Rα); atopic dermatitis universally present; Epipen mandatory",
            "key_ddx":        "STAT6-GOF vs IL4RA-GOF: IgE LEVEL (STAT6-GOF: IgE often >10,000 IU/mL, can exceed 50,000 — extreme hyperIgE; IL4RA-GOF: IgE 1000-5000 IU/mL — severe but less extreme); LIGAND INDEPENDENCE (STAT6-GOF has constitutive pSTAT6 WITHOUT IL-4/IL-13 stimulation in ex vivo assay; IL4RA-GOF requires cytokine for activation — enhanced not constitutive); ONSET (STAT6-GOF more commonly neonatal/weeks of life; IL4RA-GOF typically infant but slightly later); EOSOPHILIC GI DISEASE more common in STAT6-GOF; both respond to DUPILUMAB which directly blocks IL-4Rα (mechanism-specific for IL4RA-GOF, upstream block for STAT6-GOF)",
        },
        "jak1_gof_multi_stat": {
            "genes":          ["JAK1"],
            "mechanism":      "JAK1-GOF (JH2 pseudokinase domain) → constitutive JAK1 kinase → simultaneous activation of STAT1 (IFN-γ pathway), STAT3 (IL-6 pathway), STAT5 (IL-2/hematopoiesis) → multi-pathway inflammatory disease; JAK1 partners with JAK3 (γc cytokines), JAK2 (gp130), TYK2 (type I IFN) → broadest spectrum STAT activation seen in JAK GOF disorders",
            "shared_feature": "Constitutive multi-STAT hyperactivation (STAT1+STAT3+STAT5 simultaneously elevated in unstimulated cells — distinctive vs other STAT GOF which are more selective); systemic inflammation (arthritis, myositis, enteropathy); baricitinib/upadacitinib-preferred (JAK1-selective inhibitors)",
            "key_ddx":        "JAK1-GOF vs STAT-specific GOF: MULTI-STAT ELEVATION IN UNSTIMULATED CELLS distinguishes JAK1-GOF (all three STATs up simultaneously) from STAT-specific GOF (typically one STAT predominantly elevated); POLYARTHRITIS + ENTEROPATHY combination unusual in STAT1/3/5/6-GOF; JAK1-GOF has broader cytokine receptor dependency (γc + gp130 + IFN pathways all affected); paradoxical herpes susceptibility (JAK1 required for type I IFN → anti-viral defense eventually impaired by feedback dysregulation); selective JAK1 inhibitors (upadacitinib) theoretically more specific than ruxolitinib for JAK1-GOF",
        },
        "card11_gof_benta_nfkb": {
            "genes":          ["CARD11"],
            "mechanism":      "CARD11-GOF (coiled-coil/linker domain) → constitutive CBM complex → constitutive IKK-β → NF-κB translocation without TCR/BCR → B cell hyperproliferation (NF-κB drives B cell survival genes BCL-XL, BIM, Pax5) + IgE class switch → B cell expansion; T cells: chronic NF-κB → CTLA-4 upregulation → T cell anergy → combined immunodeficiency paradox; DIFFERENT PATHWAY from JAK-STAT GOF (NF-κB not JAK-STAT)",
            "shared_feature": "Included in JAK-STAT GOF immune dysregulation atlas as a signaling hyperactivation disorder in lymphocytes; shares clinical features of atopy, eosinophilia, and combined immunodeficiency; IBRUTINIB rather than ruxolitinib is targeted therapy; polyclonal B cell expansion is unique to BENTA in this atlas",
            "key_ddx":        "CARD11-GOF (BENTA) vs CARD11-LOF (CID): SAME GENE, POLAR OPPOSITE PHENOTYPES — CARD11-GOF → B cell excess + T cell anergy; CARD11-LOF → B cell absent + agammaglobulinemia (like SCID); SEQUENCING CONFIRMS GOF (coiled-coil mutations) vs LOF (CARD/MAGUK domain loss); clinically: B cell excess = BENTA; B cell absent = CID; BENTA vs CVID: BENTA has massive B cell expansion; CVID has B cell reduction; BENTA NF-κB constitutive; CVID B cell activation defect; IBRUTINIB response expected in BENTA (BTK upstream of CARD11); no response expected in CVID",
        },
    }

    pathognomonic_features = {
        "esophageal_candida_stat1_gof": "ESOPHAGEAL CANDIDIASIS IN NON-IMMUNOCOMPROMISED PATIENT (no HIV, no steroids, no chemotherapy, no diabetes) = STAT1-GOF OR SOCS1-LOF UNTIL PROVEN OTHERWISE: esophageal Candida in an immune-competent-appearing child/young adult is a PATHOGNOMONIC RED FLAG for STAT1-GOF (CMC5); nail dystrophy and oral thrush complement the CMC triad; prolonged pSTAT1 phosphorylation assay confirms; STAT1 CCD/DBD sequencing; ruxolitinib reverses STAT1-GOF molecular phenotype; intracranial aneurysm brain MRI mandatory at diagnosis",
        "neonatal_t1dm_stat3_gof": "NEONATAL OR EARLY INFANTILE TYPE 1 DIABETES = STAT3-GOF UNTIL PROVEN OTHERWISE: T1DM presenting in the first 2 years of life has >20% monogenic cause; STAT3-GOF is a leading cause with additional features (short stature, thyroiditis, cytopenias, lymphoproliferation); screen STAT3 sequencing AND FOXP3 (IPEX) AND KCNJ11/ABCC8 (KATP-channel neonatal DM) in any neonatal/infantile T1DM; reduced Tregs flow cytometry + prolonged pSTAT3 assay confirms STAT3-GOF; ruxolitinib started early can slow T1DM progression",
        "nk_lgl_stat5b_hotspot": "NK-LGL LEUKEMIA IN A CHILD OR YOUNG ADULT = STAT5B p.N642H GERMLINE GOF SCREENING MANDATORY: the p.N642H SH2 domain hotspot is found both as germline GOF in hereditary NK/T-LGL leukemia AND as somatic mutation in sporadic T-LGL leukemia — always test germline DNA (not just tumor DNA) in any young patient with NK-LGL to rule out hereditary STAT5B-GOF; NK-LGL = CD3-CD56+CD16+LGL on flow + peripheral smear large granular lymphocytes; STAT5B-LOF (different entity) causes GH resistance with short stature — phenotypic OPPOSITE; ruxolitinib reduces pSTAT5B constitutive signaling → NK-LGL clone apoptosis",
        "ultra_high_ige_stat6_gof": "IgE >5000 IU/mL IN AN INFANT WITH ECZEMA = STAT6-GOF (or IL4RA-GOF) PANEL IMMEDIATELY: ultra-high IgE in the first year of life with severe eczema, food allergies, and eosinophilia is a genetic atopy flag; STAT6-GOF (2022-2023 NEJM discovery) causes the highest IgE levels seen in any disorder (>10,000-50,000 IU/mL); IL4RA-GOF causes IgE 1000-5000 range (severe but less extreme); both respond to dupilumab; phospho-STAT6 assay in unstimulated cells confirms constitutive STAT6 activation in STAT6-GOF; EPIPEN mandatory in any patient with anaphylactic food allergies regardless of etiology",
        "sle_myeloprolif_socs1": "SLE-LIKE DISEASE + MYELOPROLIFERATION IN SAME PATIENT = SOCS1 HAPLOINSUFFICIENCY RED FLAG: standard SLE (ANA+, dsDNA+, complement low, nephritis) does NOT cause concurrent thrombocytosis, monocytosis, or splenomegaly (myeloproliferative features); when both occur together → SOCS1 haploinsufficiency screens positive; ISG score (interferon signature) elevated; SOCS1 haploinsufficiency is also a CMC mimic; ruxolitinib replaces SOCS1 negative feedback pharmacologically (SOCS1 normally inhibits JAK1/2 directly → ruxolitinib pharmacologically recapitulates this mechanism); SOCS1 MLPA for large deletions + sequencing for point mutations",
        "polyclonal_b_expansion_benta": "MASSIVE POLYCLONAL B CELL EXPANSION = BENTA / CARD11-GOF PATHOGNOMONIC: B cells >40% of peripheral blood lymphocytes (normal <20%) with splenomegaly and hepatomegaly is PATHOGNOMONIC for BENTA in the right clinical context (childhood onset, atopy, recurrent infections despite normal B cells); POLYCLONAL (not lymphoma) — clone check (IGH/κ/λ rearrangement) negative for monoclonality in BENTA (vs DLBCL); PARADOX: massive B cell expansion with POOR VACCINE RESPONSES and recurrent infections (T cell anergy); CARD11 coiled-coil domain mutation confirms GOF; ibrutinib trial reduces B cell expansion; rituximab transiently reduces B cells but T cell anergy persists",
        "jak_stat_flow_cytometry": "JAK-STAT PHOSPHORYLATION FLOW CYTOMETRY (PhosFlow) FOR GOF SCREENING: phospho-STAT assay in UNSTIMULATED cells (no cytokine added): NORMAL = no baseline pSTAT; JAK1/STAT-GOF = elevated pSTAT1+pSTAT3+pSTAT5 in resting cells; STAT1-GOF = sustained pSTAT1 after brief IFN stimulation (prolonged vs normal rapid decay at 30min); STAT3-GOF = prolonged pSTAT3 after IL-6 stimulation; STAT5B-GOF = constitutive pSTAT5B; STAT6-GOF = constitutive pSTAT6 without IL-4; SOCS1-LOF = elevated pSTAT1 after IFN-γ (same as STAT1-GOF but via SOCS1 mechanism); PhosFlow is the FUNCTIONAL SCREENING TEST for JAK-STAT GOF — order on clinical suspicion before sequencing",
        "ruxolitinib_jak_stat_targeted": "RUXOLITINIB (JAK1/2 INHIBITOR) IS THE BACKBONE TARGETED THERAPY FOR MOST JAK-STAT GOF DISORDERS: STAT1-GOF, STAT3-GOF, STAT5B-GOF, JAK1-GOF, SOCS1-LOF all respond to ruxolitinib (10-20mg BID) by blocking the constitutive JAK→STAT phosphorylation: STAT1-GOF: reduces pSTAT1 → Th17 restoration → CMC resolution; STAT3-GOF: reduces pSTAT3 → Treg restoration → autoimmunity/T1DM progression slowed; STAT5B-GOF: reduces pSTAT5B → NK-LGL apoptosis; JAK1-GOF: blocks constitutive multi-STAT activation; SOCS1-LOF: pharmacologically replaces SOCS1 JAK1/2 inhibition. MONITOR: CBC (cytopenias), LFTs, viral reactivation (HSV/VZV — acyclovir co-prophylaxis); adjust dose for cytopenias; do NOT abruptly stop (cytokine rebound)",
    }

    return {
        "gene_cohorts":            cohorts,
        "diagnostic_groups":       diagnostic_groups,
        "pathognomonic_features":  pathognomonic_features,
        "cascade_testing_order": [
            "Step 1: SUSPECTED JAK-STAT GOF (chronic mucocutaneous candidiasis + autoimmunity; early-onset T1DM + short stature + LPD; NK-LGL leukemia in young patient; ultra-severe infantile eczema + IgE >2000 + anaphylaxis; polyclonal B cell expansion + T cell anergy; SLE + myeloproliferation together) → ORDER PHOSPHO-STAT FLOW CYTOMETRY (PhosFlow) ON UNSTIMULATED CELLS: constitutive pSTAT1 → STAT1-GOF/SOCS1-LOF; constitutive pSTAT3 → STAT3-GOF; constitutive pSTAT5B → STAT5B-GOF; constitutive pSTAT6 → STAT6-GOF; multi-STAT elevation → JAK1-GOF",
            "Step 2: CHRONIC MUCOCUTANEOUS CANDIDIASIS + esophageal candidiasis + nail dystrophy → ORDER STAT1 SEQUENCING (CCD/DBD domain mutations first); if STAT1 normal → ORDER SOCS1 SEQUENCING + MLPA (haploinsufficiency); SOCS1-LOF CLUE: concurrent myeloproliferation (thrombocytosis) or SLE-like features absent in STAT1-GOF; pSTAT1 prolongation assay (add IFN-γ → measure pSTAT1 at 30, 60, 90, 120 min — normal decays by 60min; STAT1-GOF persists to 120min); INTRACRANIAL ANEURYSM SCREEN (MRI/MRA brain) at STAT1-GOF diagnosis",
            "Step 3: EARLY-ONSET T1DM (age <2yr) + SHORT STATURE + LYMPHOPROLIFERATION → ORDER STAT3 SEQUENCING simultaneously with FOXP3 (IPEX), KCNJ11/ABCC8 (neonatal DM), DNMT3B (ICF); pSTAT3 prolongation assay after IL-6 stimulation; TREG FLOW (CD4+CD25+FoxP3+ reduced confirms STAT3-GOF); GH stimulation test + IGF-1 (normal GH + low IGF-1 = GH resistance — STAT3-GOF hepatic pathway); START RUXOLITINIB early to slow T1DM progression if STAT3-GOF confirmed",
            "Step 4: NK-LGL LEUKEMIA OR T-LGL + YOUNG AGE → ORDER STAT5B GERMLINE SEQUENCING (check germline, not just tumor) — STAT5B p.N642H hotspot in SH2 domain; NK-LGL flow cytometry (CD3-CD56+CD16+LGL %); bone marrow biopsy (LGL infiltration); T-LGL clonality (TCR Vβ flow + TCR-γ PCR); DISTINGUISH STAT5B-GOF (NK-LGL + eosinophilia + immune dysreg) from STAT5B-LOF (GH resistance + short stature + no LGL); pSTAT5B constitutive elevation confirms GOF",
            "Step 5: ULTRA-SEVERE INFANTILE ATOPY + IgE >2000 IU/mL → ORDER STAT6 SEQUENCING + IL4RA SEQUENCING SIMULTANEOUSLY; pSTAT6 flow in unstimulated cells (STAT6-GOF: constitutive; IL4RA-GOF: elevated but requires cytokine stimulation for maximal signal); IgE LEVEL GUIDE: >5000 IU/mL → STAT6-GOF more likely; 1000-5000 IU/mL → IL4RA-GOF more likely; DUPILUMAB TRIAL (both respond; dramatic response functionally supports diagnosis); FeNO for type 2 airway inflammation; FOOD ALLERGY PANEL (specific IgE + SPT); EPIPEN PRESCRIPTION IMMEDIATELY for any anaphylaxis risk",
            "Step 6: POLYARTHRITIS + ENTEROPATHY + EOSINOPHILIA + HERPES REACTIVATION → ORDER JAK1 SEQUENCING (JH2 pseudokinase domain variants); pSTAT1+pSTAT3+pSTAT5 SIMULTANEOUSLY ELEVATED in unstimulated cells = JAK1-GOF signature (multi-STAT elevation); start ACYCLOVIR/VALACYCLOVIR prophylaxis immediately for herpes susceptibility; BARICITINIB or UPADACITINIB trial (selective JAK1 inhibitors); CBC (cytopenias in JAK1-GOF); joint X-rays and GI endoscopy (enteropathy assessment)",
            "Step 7: SYSTEMIC LUPUS-LIKE DISEASE (ANA+ dsDNA+) + CONCURRENT MYELOPROLIFERATION (thrombocytosis/monocytosis/splenomegaly) → ORDER SOCS1 SEQUENCING + MLPA (haploinsufficiency — frameshift/nonsense + large deletions); ISG score (blood interferon signature elevated); SOCS1 protein Western blot (50% reduction); distinguish from STAT1-GOF (no myeloproliferation in STAT1-GOF) and type I interferonopathies (TREX1/RNASEH2 panel if ISG extremely elevated); RUXOLITINIB replaces SOCS1 feedback pharmacologically; monitor CBC regularly during ruxolitinib (JAK2 inhibition + existing cytopenias may compound)",
            "Step 8: MASSIVE POLYCLONAL B CELL EXPANSION (B cells >40% lymphocytes) + T CELL ANERGY + ATOPY → ORDER CARD11 SEQUENCING (coiled-coil/linker domain variants = BENTA/GOF; MAGUK/CARD domain variants = CID/LOF — opposite phenotype); NF-κB reporter assay in patient T cells (constitutive NF-κB without stimulation); IBRUTINIB TRIAL (BTK inhibitor; reduces upstream BCR-NF-κB signaling → B cell normalization in BENTA; response confirms CARD11-GOF); LYMPHOMA SURVEILLANCE (annual CT neck/chest/abdomen — 25% DLBCL risk); IVIG for functional antibody deficiency; distinguish from CVID (B cells reduced in CVID; expanded in BENTA)",
            "Step 9: JAK-STAT GOF CONFIRMED (any gene) → INITIATE TARGETED THERAPY: RUXOLITINIB for STAT1/STAT3/STAT5B/JAK1/SOCS1 (10-20mg BID; titrate dose; monitor CBC/LFTs); DUPILUMAB for STAT6/IL4RA (weight-based; q2 weeks); IBRUTINIB for CARD11-GOF/BENTA (420mg daily); PROPHYLAXIS: AZOLES (fluconazole/itraconazole) for CMC (STAT1-GOF, SOCS1-LOF); ACYCLOVIR for herpes (JAK1-GOF; STAT5B-GOF); IVIG if hypogammaglobulinemia; EPIPEN for STAT6/IL4RA-GOF (anaphylaxis); INSULIN for T1DM (STAT3-GOF)",
            "Step 10: JAK INHIBITOR MONITORING AND SURVEILLANCE: CBC monthly for 3 months then q3 months (ruxolitinib cytopenias: anemia, thrombocytopenia, neutropenia); VIRAL REACTIVATION PANEL at each visit (CMV/EBV PCR — JAK inhibitors impair viral surveillance); TB SCREENING before starting ruxolitinib (JAK inhibitors increase TB risk; IGRA/TST mandatory); LIPID PANEL (baricitinib/upadacitinib raise LDL); VTE RISK (JAK inhibitors increase DVT/PE risk — assess cardiovascular risk factors); DO NOT ABRUPTLY STOP RUXOLITINIB (cytokine storm rebound — taper over 2 weeks if discontinuing); ANNUAL SURVEILLANCE: brain MRI (STAT1-GOF aneurysm); lymphoma CT (STAT5B-GOF/CARD11-GOF); thyroid function; glucose/HbA1c (all JAK-STAT GOF); bone density (steroid use); ocular exam (cataract risk with JAK inhibitors)",
        ],
    }


def generate_definitions():
    return {
        "definitions": [
            {"term": "JAK-STAT Signaling Pathway (Overview)", "definition": "The JAK-STAT pathway is a rapid, direct signal transduction mechanism: cytokine binds receptor → receptor dimerization → receptor-associated JAK kinases (JAK1/JAK2/JAK3/TYK2) auto/transphosphorylate → JAKs phosphorylate receptor cytoplasmic tyrosines (docking sites) → STAT proteins (STAT1-6) recruited via SH2 domain → STATs phosphorylated at conserved tyrosine (pSTAT) → STATs dimerize → nuclear translocation → target gene transcription → SOCS proteins induced by STATs → SOCS inhibit JAKs (negative feedback). GOF mutations in STATs impair dephosphorylation/nuclear export (prolonged activation); GOF mutations in JAK1 impair autoinhibitory pseudokinase domain (constitutive kinase activity); SOCS1-LOF removes negative feedback (same result: sustained JAK-STAT signaling)."},
            {"term": "Chronic Mucocutaneous Candidiasis (CMC) — Genetic Causes", "definition": "CMC is persistent or recurrent Candida infection affecting nails, oral mucosa, and skin (not systemic) in non-immunocompromised patients. GENETIC CAUSES: (1) STAT1-GOF (most common CMC gene; coiled-coil domain; AD; CMC type 5 = CMCD/CMC5); (2) SOCS1-LOF haploinsufficiency (similar mechanism; SLE-like concurrent); (3) IL-17RA/IL-17RC/ACT1/TRAF3IP2 (CMC types 1-4; purely CMC phenotype without systemic autoimmunity — unlike STAT1-GOF); (4) CARD9-LOF (deep dermatophytosis; systemic candida in CNS); (5) RORC-LOF (IL-17-producing T cells absent). PATHOMECHANISM: Th17 cells (CD4+IL-17A+) and innate lymphoid cells (ILC3) are the primary defense against Candida at mucosal surfaces — IL-17A/F induces β-defensins, S100 proteins, neutrophil recruitment at mucosa; in STAT1-GOF, chronic pSTAT1 suppresses STAT3/RORγt axis → Th17 deficiency → CMC. DIAGNOSIS: nail dystrophy + oral thrush + skin Candida + pSTAT1 prolongation assay + STAT1 sequencing + Th17 flow cytometry (CD4+IL-17A+ reduced)."},
            {"term": "STAT3-GOF vs STAT3-LOF (HYPER-IgE Syndrome Type 1)", "definition": "SAME GENE, OPPOSITE PHENOTYPE: STAT3-LOF (AD-LOF, Hyper-IgE Syndrome type 1 / HIES1, Job syndrome): somatic eczema + recurrent Staphylococcal skin abscesses + elevated IgE + skeletal anomalies (scoliosis, retained primary teeth, joint hyperextensibility) — STAT3 LOF prevents IL-10 anti-inflammatory signaling → Staphylococcal abscesses (distinctive 'cold abscesses' without heat/erythema — impaired IL-1/IL-6 acute phase response). STAT3-GOF (described here): early-onset T1DM + growth failure + lymphoproliferation + MULTI-ORGAN autoimmunity — constitutive STAT3 impairs Treg induction. CLINICAL DISTINCTION: HIES1/STAT3-LOF = Staphylococcal cold abscesses + skeletal; STAT3-GOF = T1DM + LPD + cytopenia triad; BOTH have elevated IgE but for different reasons. Molecular distinction: STAT3-LOF variants in DBD or SH2 domain cause haploinsufficiency; STAT3-GOF variants in SH2/DBD/TAD cause gain-of-function — SAME domain, different variant classes."},
            {"term": "Phospho-STAT (PhosFlow) Assay for JAK-STAT GOF Diagnosis", "definition": "Phospho-STAT flow cytometry (PhosFlow) is the KEY FUNCTIONAL TEST for diagnosing JAK-STAT GOF disorders. PRINCIPLE: permeabilize cells → stain with phospho-specific antibodies (anti-pSTAT1 Y701, anti-pSTAT3 Y705, anti-pSTAT5 Y694, anti-pSTAT6 Y641) → measure fluorescence intensity by flow cytometry. UNSTIMULATED CELLS: GOF mutations cause constitutive STAT phosphorylation even WITHOUT cytokine stimulation (pSTAT elevated at baseline = GOF flag). STIMULATION ASSAYS: add IFN-γ → measure pSTAT1 at 30/60/120min (normal: rapid peak then decay; STAT1-GOF: prolonged plateau >90min); add IL-6 → pSTAT3 (STAT3-GOF: sustained); add IL-2 → pSTAT5 (STAT5B-GOF: constitutive before stimulation). CLINICAL USE: order before sequencing as a rapid functional screen; abnormal PhosFlow → targeted sequencing of relevant STAT/JAK gene; PhosFlow also used to monitor JAK inhibitor therapeutic effect (reduction of constitutive pSTAT = treatment response)."},
            {"term": "Ruxolitinib for JAK-STAT GOF Disorders", "definition": "Ruxolitinib (Jakafi/Jakavi) is a selective JAK1/JAK2 inhibitor (IC50: JAK1 ~3.3 nM, JAK2 ~2.8 nM; less active on JAK3/TYK2) used as targeted therapy across multiple JAK-STAT GOF primary immunodeficiencies. MECHANISM IN JAK-STAT GOF: blocks constitutive JAK1/JAK2 → reduces excess pSTAT1/pSTAT3/pSTAT5 → restores normal JAK-STAT set point. CLINICAL APPLICATIONS: STAT1-GOF (reduces pSTAT1 → restores Th17 → CMC resolution); STAT3-GOF (reduces pSTAT3 → Treg restoration → autoimmune remission, T1DM progression slowed); STAT5B-GOF (reduces pSTAT5B → NK-LGL apoptosis); JAK1-GOF (blocks constitutive JAK1 → multi-STAT normalization); SOCS1-LOF (pharmacologically replaces SOCS1 JAK1/2 inhibitory function). DOSE: 10-20mg BID (adult; weight-adjusted in pediatrics). MONITORING: CBC (cytopenias — dose-limiting anemia/thrombocytopenia); TB screening (IGRA) before start; viral surveillance (CMV/EBV); do NOT abruptly stop (taper — cytokine storm rebound risk). FDA-APPROVED indications: myelofibrosis, PV, GVHD (licensed uses); use in PID is off-label but evidence accumulating rapidly (2015-2024 publications)."},
            {"term": "BENTA Disease (CARD11-GOF) vs CARD11-LOF CID", "definition": "CARD11 encodes CARMA1, the scaffold of the CBM (CARD11-BCL10-MALT1) signalosome: TCR/BCR → CARD11 → NF-κB activation. TWO OPPOSITE DISORDERS FROM SAME GENE: (1) CARD11-GOF (BENTA): coiled-coil/linker domain mutations → constitutive NF-κB → B cell expansion, T cell anergy, atopy, combined immunodeficiency despite B cell excess; IBRUTINIB-responsive. (2) CARD11-LOF (CID/SCID-like): CARD/MAGUK domain mutations → no NF-κB from TCR/BCR → agammaglobulinemia, absent T cell activation → SCID-like → HSCT urgent. CLINICAL PEARL: same gene, COMPLETELY OPPOSITE B cell counts (GOF: B cell excess 70%; LOF: B cells absent); ALWAYS sequence CARD11 in combined immunodeficiency of unknown cause — and identify whether variant is GOF or LOF before treatment."},
            {"term": "Dupilumab for IL-4/IL-13/STAT6 Axis GOF Disorders", "definition": "Dupilumab (Dupixent) is a human monoclonal antibody against IL-4Rα (the shared receptor subunit for both IL-4 and IL-13 signaling): blocks type I IL-4R (IL-4Rα/γc) and type II IL-4R (IL-4Rα/IL-13Rα1) → prevents both IL-4 and IL-13 from signaling → blocks pSTAT6 induction. MECHANISM IN ATOPY GOF: IL4RA-GOF: dupilumab directly neutralizes the GOF receptor → mechanism-specific therapy; STAT6-GOF: dupilumab blocks upstream IL-4/IL-13 input → partially compensates for constitutive STAT6-GOF (STAT6 still constitutively active but upstream amplification removed). FDA APPROVALS: atopic dermatitis (≥6 months), asthma (≥6yr), CRSwNP (≥18yr), EoE (≥12yr), prurigo nodularis (≥18yr). PEDIATRIC DOSING: weight-based (15-60kg: 300mg q4wks then q2wks; >60kg: adult dose); DRAMATIC RESPONSE expected in IL4RA-GOF and STAT6-GOF (>75% EASI improvement in most; IgE reduction 70-90%; FeNO normalization; food allergy threshold improvement). ADVERSE EFFECTS: conjunctivitis (5-10%; use preservative-free eye drops); injection site reactions; rarely arthralgia."},
            {"term": "SOCS1 Haploinsufficiency (2021 Discovery) — JAK-STAT Negative Feedback Loss", "definition": "SOCS1 (Suppressor of Cytokine Signaling 1, 212aa, 16p13.13) is the smallest SOCS family member and the critical negative feedback regulator of IFN-γ/JAK1/JAK2 signaling. MECHANISM: IFN-γ → JAK1/JAK2 → pSTAT1 → SOCS1 transcription → SOCS1 protein: (1) KIR domain directly blocks JAK1/JAK2 active site (pseudosubstrate); (2) SOCS-box recruits elongin BC/CRL2 E3 ubiquitin ligase → JAK ubiquitination → proteasomal degradation. HAPLOINSUFFICIENCY: monoallelic SOCS1 LOF → 50% SOCS1 protein → insufficient feedback → chronic low-grade JAK-STAT hyperactivation → SLE-like autoimmunity + myeloproliferation + CMC (published 2021 by Maccari and Vignoli groups). CLINICAL IMPORTANCE: SOCS1 haploinsufficiency is underdiagnosed because ISG score overlap with type I interferonopathies; SOCS1 Western blot can confirm 50% reduction; treatment = ruxolitinib (pharmacologically replaces SOCS1 JAK inhibitory function — elegant mechanism-based therapy for SOCS1 haploinsufficiency)."},
        ],
        "key_distinguishers": {
            "stat1_gof_vs_socs1_lof":    "Both cause CMC via STAT1 hyperactivation + Th17 suppression. DDx: MYELOPROLIFERATION present in SOCS1-LOF (thrombocytosis, splenomegaly) NOT in STAT1-GOF. SLE-like (ANA/dsDNA) in SOCS1-LOF. Intracranial aneurysm risk in STAT1-GOF not SOCS1-LOF. SOCS1 protein reduced by Western blot. Ruxolitinib for both.",
            "stat3_gof_vs_stat3_lof_hies":"STAT3-GOF: early T1DM + short stature + LPD + cytopenias. STAT3-LOF/HIES: cold abscesses (Staphylococcal) + skeletal anomalies (retained teeth, scoliosis) + high IgE. SAME GENE: GOF in TAD/SH2; LOF in DBD/SH2 (different variant classes).",
            "stat6_gof_vs_il4ra_gof":     "STAT6-GOF: IgE >10,000 often; constitutive pSTAT6 WITHOUT cytokine; EoE common; neonatal onset. IL4RA-GOF: IgE 1000-5000; requires cytokine for activation (enhanced not constitutive); CRSwNP prominent; slightly later onset. Both: dupilumab first-line.",
            "card11_gof_benta_vs_lof_cid": "CARD11-GOF/BENTA: B cell EXPANSION (70% lymphocytes); T cell anergy; polyclonal; ibrutinib. CARD11-LOF/CID: B cells ABSENT; agammaglobulinemia; SCID-like; HSCT urgent. Same gene — mutation location and functional class determines phenotype.",
        },
        "treatment_matrix": {
            "ruxolitinib":   "STAT1-GOF, STAT3-GOF, STAT5B-GOF, JAK1-GOF, SOCS1-LOF: 10-20mg BID; monitor CBC/TB/viral; do not stop abruptly; dose-reduce for cytopenias. Most effective for CMC (STAT1), T1DM progression (STAT3), NK-LGL (STAT5B), polyarthritis (JAK1), myeloproliferation (SOCS1).",
            "dupilumab":     "STAT6-GOF, IL4RA-GOF: anti-IL-4Rα; blocks IL-4 and IL-13; weight-based dosing q2 weeks; dramatic response; reduce IgE 70-90%; treat AD + asthma + EoE + CRSwNP concurrently. Epipen still mandatory (anaphylaxis risk remains until desensitization).",
            "ibrutinib":     "CARD11-GOF/BENTA: BTK inhibitor 420mg daily; reduces upstream BCR-NF-κB signaling → B cell normalization; most evidence for BENTA-specific; combine with IVIG for antibody deficiency.",
            "hsct":          "STAT3-GOF (severe lymphoma), STAT5B-GOF (refractory NK-LGL/aplasia), CARD11-GOF (DLBCL), JAK1-GOF (severe refractory): HSCT curative; lower threshold for STAT3-GOF lymphoma; SOCS1-LOF severe myeloproliferative disease; STAT1-GOF: HSCT rarely needed but used for severe refractory CMC + autoimmunity.",
            "prophylaxis":   "CMC (STAT1-GOF/SOCS1-LOF): fluconazole/itraconazole suppression. Herpes (JAK1-GOF/STAT5B-GOF): acyclovir/valacyclovir lifelong. Anaphylaxis (STAT6/IL4RA-GOF): epinephrine auto-injector always. IVIG: STAT1-GOF (50% hypogamma), CARD11-GOF (functional B cell deficiency), STAT3-GOF (if hypogamma).",
        },
    }
