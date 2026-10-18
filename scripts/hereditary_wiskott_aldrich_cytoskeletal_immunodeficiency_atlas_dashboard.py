#!/usr/bin/env python3
"""Hereditary-Wiskott-Aldrich-Cytoskeletal-Immunodeficiency-Atlas -- Complete 8-Gene Reference
Actin/Cytoskeletal Defects Affecting Immune Cells / WAS-WIPF1-ARPC1B-CORONIN1A-RAC2-RHOH-CARMIL2-STK4
(WAS/XLR-Classic-WAS-Triad-Small-Platelets-PATHOGNOMONIC-EBV-Lymphoma-HSCT-XLN-GOF-Neutropenia,
 WIPF1/AR-WAS-Phenocopy-WASP-Normal-Gene-Absent-Protein-WIP-Stabilises-WASP,
 ARPC1B/AR-Arp2-3-Complex-PID-Eosinophilia-IBD-Vasculitis-Delta-Granule-Defect,
 CORONIN1A/AR-T-B-Plus-NK-Plus-Thymic-Shadow-Preserved-Swiss-Type-HSCT,
 RAC2/AD-AR-Rho-GTPase-GOF-D57N-Neonatal-Sepsis-Cord-Delay-SCID-Severity,
 RHOH/AR-Ultra-Rare-Warts-HPV-PATHOGNOMONIC-Kaposi-Sarcoma-EBV-Lymphoprol,
 CARMIL2/AR-CD28-Signalling-Actin-CMC-Treg-Defect-EBV-LPD-HSCT,
 STK4/AR-Hippo-Kinase-Naive-T-Lymphopenia-Cardiac-ASD-VSD-PATHOGNOMONIC):

WAS   (Wiskott-Aldrich Syndrome Protein; 502aa; Xp11.22; X-linked recessive;
       Classic WAS disease — triad: microTHROMBOCYTOPENIA + ECZEMA + PID;
       WASP is the founding member of the WASp family (N-WASP/WAVE); it is exclusively expressed in
         haematopoietic cells; WASP integrates signals from the actin-nucleation machinery (Arp2/3 complex)
         downstream of Rho GTPases (CDC42 → WASP VCA domain → Arp2/3 activation → actin branching);
         WASP is required for: immunological synapse formation, NK cell killing, T/B cell migration,
         neutrophil chemotaxis, platelet spreading and megakaryocyte cytokinesis;
       CLASSIC WAS TRIAD (all three required for classical presentation):
         (1) MICROTHROMBOCYTOPENIA: platelet count 20-80 × 10⁹/L; SMALL PLATELET SIZE (MPV < 7 fL;
             normally 8-12 fL) — THIS IS PATHOGNOMONIC of WAS/WIPF1/WASp-related defects; ITP has
             large reactive platelets (MPV elevated); WASP defect → platelet cytoskeletal defect →
             impaired spreading and small platelet volume; splenomegaly amplifies thrombocytopenia;
         (2) ECZEMA: typically moderate-severe; IgE elevated; food-specific IgE; exacerbated by
             Th2 polarization from absent Treg (WASP required for Treg function/stability) → Treg
             depletion → Th2 skewing → eczema + elevated IgE;
         (3) PID: combined T+B cell dysfunction; recurrent otitis media, pneumonia, sepsis; impaired
             lymphocyte migration → ineffective adaptive immunity; declining T-cell counts with age;
       XLN (X-linked neutropenia): gain-of-function WASP mutations (L270P, I294T, S272P) in the GBD
         domain prevent autoinhibition → constitutively active → hyperactivated Arp2/3 → cytoskeletal
         defect in neutrophils → neutropenia without classical WAS triad; AML risk 30% (highest of
         all WAS phenotypes); patients are otherwise healthy until neutropenia → infection/AML;
       EBV LYMPHOMA: 13-22% lifetime risk without HSCT; WASP-deficient B cells cannot form effective
         immunological synapses for CTL recognition → EBV persists unchecked → B-cell LPD → lymphoma;
         HSCT MANDATORY before EBV infection causes irreversible transformation (ideally by 3-5 years);
       DIAGNOSTIC TESTS: WASP flow cytometry (CD3+ T cells — WASP absent/reduced in LOF); WASP Western blot;
         platelet count + MPV (small platelets, MPV < 7 fL); WAS gene sequencing for genotype/phenotype;
       WISKOTT-ALDRICH SCORE (0-5): 0 = thrombocytopenia only; 1-2 = WAS mild; 3-4 = moderate; 5 = severe;
       BCG ABSOLUTELY CONTRAINDICATED in all WAS genotypes (combined PID — fatal BCGosis);
       seed SEED_BASE+0).
WIPF1  (WASp-Interacting Protein 1; 503aa; 2q31.1; autosomal recessive;
        WAS phenocopy — clinical presentation IDENTICAL to classical WAS but WAS gene is NORMAL;
        WIP (WIPF1 protein) forms a constitutive complex with WASP and is required for WASP protein
          stability: WIP binds the N-terminal WH1 domain of WASP and shields it from proteasomal
          degradation; WIPF1 biallelic LOF → WIP absent → WASP protein rapidly degraded (ubiquitin-
          proteasome) even though the WAS gene is intact → functionally identical to WASP-null WAS;
        CLINICAL PHENOTYPE: IDENTICAL to classical WAS: microTHROMBOCYTOPENIA (small platelets,
          MPV < 7 fL), ECZEMA, PID; EBV lymphoma risk comparable to WAS; HSCT curative;
        KEY DIAGNOSTIC TRAP: WASP protein ABSENT by flow cytometry/Western blot in WIPF1 deficiency —
          looks like WAS; BUT WAS GENE SEQUENCING IS NORMAL → WIPF1 sequencing mandatory;
          rule: if WASP protein absent + WAS gene normal → sequence WIPF1 immediately;
        INHERITANCE: AUTOSOMAL RECESSIVE — unlike X-linked WAS; FEMALES ARE AFFECTED (important
          clinical clue: female with WAS phenotype → not WAS → WIPF1 or other WAS-like genes);
        SPLENOMEGALY + risk of haemolytic disease (immune dysregulation) described in WIPF1;
        HSCT curative and recommended for WAS score ≥3 equivalent, EBV positivity, or malignancy;
        seed SEED_BASE+1).
ARPC1B (Actin-Related Protein 2/3 Complex Subunit 1B; 323aa; 7q22.1; autosomal recessive;
        Arp2/3 complex subunit — UNIQUE combined inflammatory + immunodeficiency phenotype;
        ARPC1B encodes a structural subunit of the Arp2/3 complex, which is the core actin-nucleation
          machinery activated by WASP/N-WASP/WAVE2; loss of ARPC1B → total Arp2/3 complex instability
          → impaired actin branching → combined lymphocyte migration defect + innate immunity dysregulation;
        UNIQUE COMBINED PHENOTYPE (distinguishes from WAS/WIPF1):
          PID: combined T+B cell dysfunction (recurrent bacterial and fungal infections);
          EOSINOPHILIA: blood eosinophils > 1500/µL — PATHOGNOMONIC in cytoskeletal immunodeficiency context;
          VERY EARLY ONSET IBD: Crohn's-like or ulcerative colitis; onset < 5 years in most patients;
          VASCULITIS: cutaneous and systemic (polyarteritis nodosa-like);
          COLITIS: 60-70% of ARPC1B patients — characteristic finding;
        PLATELET DELTA-GRANULE DEFECT: ARPC1B patients have abnormal platelet morphology on electron
          microscopy — delta (dense) granule absence or reduction (storage pool disease) + platelet
          aggregation defect; DISTINCT from WAS where granule number is normal (small platelet size only);
        NO WASP FLOW DEFECT: WASP protein normal by flow cytometry (different protein); ARPC1B-specific
          flow cytometry test: ARPC1B protein absent from neutrophils/lymphocytes — rapid diagnostic;
        JAKi (ruxolitinib) effective for inflammatory manifestations (vasculitis, IBD, eosinophilia) while
          awaiting HSCT; HSCT curative for both immune and inflammatory phenotypes;
        seed SEED_BASE+2).
CORONIN1A (Coronin-1A; 461aa; 16p11.2; autosomal recessive;
        CORO1A — selective T-CELL EGRESS DEFECT from thymus → T-B+NK+ SCID phenotype;
        Coronin-1A is an F-actin-binding protein required for cortical actin dynamics in thymocytes;
          CORO1A LOF → defective actin-dependent vesicular trafficking in thymocytes → impaired thymic
          egress: T cells complete maturation in thymus but CANNOT EXIT → naïve T cells absent from
          periphery; NK cells and B cells are unaffected (their actin requirements differ);
        PHENOTYPE: T-B+NK+ SCID (or severe T-cell lymphopenia) — often classified as Swiss-type
          agammaglobulinaemia/SCID (historically: Swiss-type = T-B+NK+, no identifiable molecular defect);
        PATHOGNOMONIC IMAGING: THYMIC SHADOW PRESENT AND ENLARGED on CXR/CT — thymus is packed with
          mature T cells unable to egress; contrasts with most SCID where thymic shadow is absent;
          ENLARGED THYMIC SHADOW IN A T-CELL-LYMPHOPENIC INFANT = CORONIN1A until proven otherwise;
        AUTOPHAGY DEFECT: Coronin-1A also required for lysosomal/autophagic flux in mature T cells →
          CORO1A deficiency → impaired autophagy → accelerated naïve T-cell apoptosis (contributes to
          T-cell lymphopenia even from egressed T cells);
        SWISS FOUNDER MUTATION: p.R18C (among Swiss and Central European families);
        BCG ABSOLUTELY CONTRAINDICATED: T cells absent → no adaptive immunity → fatal BCGosis;
        HSCT curative (provides normal haematopoietic progenitors → normal Coronin-1A → normal T-cell
          egress; thymic hyperplasia resolves post-HSCT);
        seed SEED_BASE+3).
RAC2   (Ras-related C3 botulinum toxin substrate 2; 192aa; 22q13.1; autosomal dominant (GOF) or AR (LOF);
        Rho GTPase — NEUTROPHIL + LYMPHOCYTE COMBINED IMMUNODEFICIENCY;
        RAC2 is a haematopoietically restricted Rho GTPase (Rho family); it is required for:
          (1) NADPH oxidase complex assembly (NOX component alongside gp91phox/CYBB);
          (2) Neutrophil migration (lamellipodia formation via Arp2/3);
          (3) NK and T-cell killing and migration;
          (4) Apoptotic pathway regulation in granulocytes;
        TWO MUTATION CLASSES:
          (1) GOF (dominant) — D57N (most common); other: E62K; de novo dominant:
              D57N is constitutively GTP-bound (exchange without GEF) → hyperactivated downstream
              signalling → paradoxical impairment: excess Rac2 activation → receptor internalisation →
              downstream neutrophil killing impaired → neutropenia + T-cell lymphopenia;
              NEONATAL ONSET: omphalitis, delayed cord separation (days 14-21), neonatal sepsis;
              LAD-LIKE FEATURE: umbilical cord separation delayed (LAD-I mimicry) — important DDx;
              LYMPHOCYTE DEFECT: T-cell lymphopenia + impaired T-cell proliferation;
              SEVERITY: SCID-equivalent in D57N; HSCT required urgently from birth;
          (2) LOF (AR): isolated neutrophil defect (migration + superoxide generation) → pyogenic
              bacterial infections; milder phenotype; WASP absent excluded; neutrophil granule analysis;
        BCG ABSOLUTELY CONTRAINDICATED in GOF D57N (SCID-equivalent severity; combined T-cell absence);
        LEUKOCYTOSIS: often 15-30 × 10⁹/L neutrophilia (paradoxical in GOF — impaired egress from BM);
        seed SEED_BASE+4).
RHOH   (Rho GTPase RhoH / TTF; 191aa; 4p13; autosomal recessive;
        Ultra-rare — <30 families described worldwide; constitutively GTP-bound pseudo-GTPase;
        RhoH is an ATYPICAL Rho GTPase: constitutively GTP-bound (intrinsically no GTPase activity);
          not regulated by GEFs or GAPs; instead regulated by transcription and protein stability;
          RhoH functions in TCR-proximal signalling: required for ZAP70 recruitment to CD3ζ ITAMs
          and LAT phosphorylation → without RhoH → TCR signal attenuated → T-cell activation impaired;
        CLINICAL PHENOTYPE: T-cell dysfunction predominating, with specific viral susceptibilities:
          HPV (WARTS): severe, extensive, recalcitrant warts from early childhood — PATHOGNOMONIC of
            RHOH deficiency among cytoskeletal PID genes; HPV2 (common wart), HPV4 (plantar wart),
            HPV3 (flat wart); hundreds of warts; lesions spread to periungual, plantar, facial areas;
            HPV vaccination does not prevent these strains; interferon-α response impaired;
          KAPOSI SARCOMA: HHV8 susceptibility; classical Kaposi sarcoma in young patients — unusual
            without HIV;
          EBV LYMPHOPROLIFERATION: EBV B-cell lymphoproliferative disease; risk of EBV lymphoma;
          MOLLUSCUM CONTAGIOSUM: extensive MCV lesions (cutaneous T-cell defect → MCV unchecked);
        T-CELL LYMPHOPENIA: CD4+ and CD8+ T cells reduced; impaired T-cell proliferation to mitogens
          and TCR stimulation;
        B CELLS: normal numbers but hypogammaglobulinaemia (T-cell help absent) → antibody production
          impaired;
        TREATMENT: rituximab for EBV LPD; HPV lesions refractory to most therapies; IFN-α for
          recalcitrant warts (partial); HSCT curative;
        seed SEED_BASE+5).
RLTPR  (CARMIL2 — Capping protein Regulator And Myosin I Linker 2; 1113aa; 16q22.1; autosomal recessive;
        CD28 co-stimulation actin scaffold — CMC + Treg defect + IBD + EBV susceptibility;
        CARMIL2 (also known as RLTPR) interacts with IQGAP1 and the barbed-end actin capping complex
          (CP/CapZ) to regulate actin dynamics specifically at the CD28 signalling microcluster
          (immunological synapse); CD28 engagement → CARMIL2 recruits WAVE2 and activates Arp2/3 →
          actin remodelling at the synapse → full T-cell activation (IL-2 production, survival);
          CARMIL2 deficiency → impaired CD28 co-stimulation → reduced T-cell activation + reduced
          Treg generation → combined pro-inflammatory (Treg deficiency → colitis) and immunodeficiency
          (impaired T-cell activation → CMC, EBV);
        CLINICAL PHENOTYPE:
          MUCOCUTANEOUS CANDIDIASIS (CMC): oral, oesophageal, nail candidiasis; Th17 dysfunction
            (indirectly via impaired T-cell activation); responds to azoles but recurs off-treatment;
          INFLAMMATORY BOWEL DISEASE / COLITIS: very early onset; Crohn's-like pattern; secondary to
            Treg deficiency allowing unbridled Th17/Th1 intestinal inflammation;
          EBV LYMPHOPROLIFERATIVE DISEASE: EBV lymphoma risk (impaired CTL cytotoxicity via CD28
            signalling defect);
          TREG DEFICIENCY: CD25+FoxP3+ Treg reduced → systemic inflammation + colitis + skin disease;
        DIAGNOSTIC TEST: CD28 stimulation assay — isolated CD28 ligation (anti-CD28 superagonist) +
          IL-2 readout: severely reduced compared to controls (CD3-driven activation preserved);
          CARMIL2 protein absent from T cells by Western blot/flow cytometry;
        CMC OVERLAP NOTE: CARMIL2 deficiency is in the PID/cytoskeletal category, NOT the CMC/IL-17
          pathway category (unlike STAT1 GOF or IL17RA); mechanism is CD28-actin, not IL-17 signalling;
        HSCT curative for immune and inflammatory phenotypes;
        seed SEED_BASE+6).
STK4   (Serine/Threonine Kinase 4 / MST1; 487aa; 20q11.2; autosomal recessive;
        Hippo pathway kinase — NAIVE T-CELL LYMPHOPENIA + CARDIAC DEFECTS (PATHOGNOMONIC association);
        STK4 (MST1) is the mammalian orthologue of Drosophila Hippo; it is a serine/threonine kinase in
          the Hippo tumour suppressor pathway (STK4 → LATS1/2 → YAP/TAZ inhibition);
        IMMUNE FUNCTION: STK4 required in mature T cells for:
          (1) NAIVE T-CELL SURVIVAL: STK4 phosphorylates FOXO3a → nuclear localisation → Bcl-2
              upregulation → naive T-cell antiapoptotic signalling; STK4 LOF → naive T-cell apoptosis
              → naïve CD4+ and CD8+ T cells severely reduced from infancy;
          (2) T-cell homing to lymph nodes (L-selectin/CD62L expression dependent on STK4 kinase);
          (3) NK cell number and function reduced;
        CLINICAL PHENOTYPE:
          NAIVE T-CELL LYMPHOPENIA: recurrent bacterial pneumonia, opportunistic infections; CD45RA+
            (naive) T-cell fraction severely reduced; memory T cells relatively preserved initially;
          CARDIAC CONGENITAL DEFECTS: atrial septal defect (ASD), ventricular septal defect (VSD),
            hypoplastic left heart — 40-60% of STK4-deficient patients; PATHOGNOMONIC COMBINATION:
            congenital heart disease + T-cell lymphopenia + recurrent infections = STRONGLY suspect STK4;
          LUNG DISEASE: bronchiectasis, pulmonary fibrosis (reported in adult patients);
          NK CELL REDUCTION: CMV susceptibility; EBV susceptibility;
          ELEVATED IgE + eosinophilia in a subset (Th2 skewing from Treg/T-cell dysregulation);
        CARDIAC SCREENING: ECHOCARDIOGRAM MANDATORY at diagnosis — congenital heart defects need
          cardiac intervention independent of the immunodeficiency (structural defect not corrected by HSCT);
        HSCT curative for immune defect (naive T cells restored); does NOT correct congenital heart disease;
        BCG ABSOLUTELY CONTRAINDICATED (T-cell lymphopenia → combined T+NK cell impairment → fatal BCGosis);
        seed SEED_BASE+7).
"""

import random
from typing import Any

SEED_BASE = 3918

GENES = ["WAS", "WIPF1", "ARPC1B", "CORONIN1A", "RAC2", "RHOH", "RLTPR", "STK4"]

GENE_META: dict[str, dict[str, Any]] = {
    "WAS": {
        "full_name": "Wiskott-Aldrich Syndrome Protein",
        "aa": 502, "locus": "Xp11.22", "kda": 53, "inh": "X-linked recessive",
        "syndrome": "Wiskott-Aldrich Syndrome / XLN (GOF)",
        "mim_gene": "300392", "mim_disease": "301000",
        "iuis_category": "Cytoskeletal immunodeficiency — WASp family / IUIS 2022 Category 2: Combined with syndromic features",
    },
    "WIPF1": {
        "full_name": "WASp-Interacting Protein 1",
        "aa": 503, "locus": "2q31.1", "kda": 54, "inh": "Autosomal recessive",
        "syndrome": "WAS Phenocopy (WASP absent; WAS gene normal)",
        "mim_gene": "602357", "mim_disease": "614493",
        "iuis_category": "Cytoskeletal immunodeficiency — WASp-stabiliser / IUIS 2022 Category 2",
    },
    "ARPC1B": {
        "full_name": "Actin-Related Protein 2/3 Complex Subunit 1B",
        "aa": 323, "locus": "7q22.1", "kda": 47, "inh": "Autosomal recessive",
        "syndrome": "ARPC1B Immunodeficiency (PID + IBD + Vasculitis + Eosinophilia)",
        "mim_gene": "604223", "mim_disease": "617099",
        "iuis_category": "Cytoskeletal immunodeficiency — Arp2/3 complex / IUIS 2022 Category 2",
    },
    "CORONIN1A": {
        "full_name": "Coronin-1A",
        "aa": 461, "locus": "16p11.2", "kda": 51, "inh": "Autosomal recessive",
        "syndrome": "Coronin-1A Immunodeficiency (T-B+NK+ SCID / Swiss-type)",
        "mim_gene": "605000", "mim_disease": "615401",
        "iuis_category": "Cytoskeletal immunodeficiency — Coronin family / IUIS 2022 Category 1: SCID",
    },
    "RAC2": {
        "full_name": "Ras-related C3 botulinum toxin substrate 2",
        "aa": 192, "locus": "22q13.1", "kda": 21, "inh": "Autosomal dominant (GOF) / Autosomal recessive (LOF)",
        "syndrome": "RAC2 Immunodeficiency (GOF-D57N neonatal SCID / LOF neutrophil defect)",
        "mim_gene": "602049", "mim_disease": "608203",
        "iuis_category": "Cytoskeletal immunodeficiency — Rho GTPase / IUIS 2022 Category 2",
    },
    "RHOH": {
        "full_name": "Rho-related GTP-binding protein RhoH",
        "aa": 191, "locus": "4p13", "kda": 21, "inh": "Autosomal recessive",
        "syndrome": "RhoH Deficiency (warts/HPV + Kaposi + EBV lymphoprol.)",
        "mim_gene": "602037", "mim_disease": "606364",
        "iuis_category": "Cytoskeletal immunodeficiency — Rho GTPase / IUIS 2022 Category 6: Intrinsic immunity defects",
    },
    "RLTPR": {
        "full_name": "CARMIL2 (Capping protein Regulator And Myosin I Linker 2)",
        "aa": 1113, "locus": "16q22.1", "kda": 126, "inh": "Autosomal recessive",
        "syndrome": "CARMIL2 Deficiency (CMC + Treg defect + IBD + EBV LPD)",
        "mim_gene": "615199", "mim_disease": "616067",
        "iuis_category": "Cytoskeletal immunodeficiency — CD28 actin scaffold / IUIS 2022 Category 2",
    },
    "STK4": {
        "full_name": "Serine/Threonine Kinase 4 (MST1 / Hippo kinase)",
        "aa": 487, "locus": "20q11.2", "kda": 56, "inh": "Autosomal recessive",
        "syndrome": "STK4/MST1 Deficiency (naive T-cell lymphopenia + congenital heart defects)",
        "mim_gene": "604965", "mim_disease": "614868",
        "iuis_category": "Cytoskeletal/kinase immunodeficiency — Hippo pathway / IUIS 2022 Category 2",
    },
}

DEFINITIONS = [
    {"term": "WASP (Wiskott-Aldrich Syndrome Protein)", "definition": "Haematopoiesis-exclusive actin nucleation regulator: GBD-WH2-ICA-VCA domain architecture; activated by CDC42-GTP → VCA domain activates Arp2/3 complex → de novo actin branch nucleation; required for immunological synapse formation, platelet spreading, NK killing, T-cell migration; absent in WAS LOF and WIPF1 deficiency (despite normal WAS gene)"},
    {"term": "Arp2/3 Complex", "definition": "7-subunit actin-nucleating complex activated by WASP VCA domain; generates Y-branched actin filaments essential for lamellipodia, phagocytic cups, immunological synapses; ARPC1B encodes one structural subunit — loss destabilises the entire complex"},
    {"term": "Microthrombocytopenia", "definition": "PLATELET COUNT LOW + PLATELET SIZE SMALL (MPV < 7 fL) — PATHOGNOMONIC of WAS/WIPF1; contrasts with ITP (large reactive platelets, high MPV ≥ 12 fL) and Bernard-Soulier (giant platelets); platelet size is the key discriminator"},
    {"term": "WIP (WIPF1 protein)", "definition": "WASp-Interacting Protein 1; constitutively complexed with WASP N-terminal WH1 domain; protects WASP from proteasomal degradation; WIPF1 LOF → WASP protein degraded within hours → WAS-identical phenotype with normal WAS gene"},
    {"term": "Thymic Shadow Preserved (CORONIN1A)", "definition": "Normal or enlarged thymic shadow on CXR/CT in a T-cell-lymphopenic infant = CORONIN1A (or MHC-II) — thymocytes mature but cannot egress; contrasts with most SCID where thymic shadow is absent/small; pathognomonic imaging finding"},
    {"term": "RAC2-D57N GOF", "definition": "Most common RAC2 gain-of-function: Asp57Asn renders Rac2 constitutively GTP-bound (cannot hydrolyse GTP without GAP); hyperactivation → paradoxical neutrophil dysfunction + T-cell lymphopenia; neonatal onset; delayed cord separation; leukocytosis (impaired BM egress); SCID severity → HSCT urgently from birth"},
    {"term": "RhoH (RHOH)", "definition": "Atypical Rho GTPase: constitutively GTP-bound (no intrinsic GTPase); not regulated by GEFs/GAPs; haematopoiesis-exclusive; required for TCR-proximal signalling (ZAP70 recruitment to CD3ζ, LAT phosphorylation); RHOH LOF → TCR signal attenuated → T-cell lymphopenia + viral susceptibilities (HPV warts, Kaposi, EBV)"},
    {"term": "HPV Warts (RHOH Pathognomonic)", "definition": "Severe, extensive, recalcitrant verrucae (HPV2/3/4) from early childhood — PATHOGNOMONIC of RHOH deficiency among cytoskeletal PID; hundreds of plantar, periungual, flat warts; refractory to standard treatments (cryotherapy, salicylic acid); impaired IFN-α/T-cell HPV response"},
    {"term": "CARMIL2 / RLTPR", "definition": "Capping protein Regulator And Myosin I Linker 2; interacts with IQGAP1 to regulate actin dynamics at CD28 signalling microcluster; CARMIL2 deficiency → impaired CD28 co-stimulation → reduced T-cell activation + Treg deficiency → combined CMC + IBD + EBV LPD"},
    {"term": "CD28 Stimulation Assay", "definition": "Diagnostic test for CARMIL2 deficiency: isolated CD28 ligation (anti-CD28 superagonist, no CD3 signal) → IL-2 readout; CARMIL2-null = severely reduced IL-2 (CD28-specific actin remodelling absent); CD3-driven activation relatively preserved; distinguishes CARMIL2 from other T-cell activation defects"},
    {"term": "STK4 / MST1 Kinase", "definition": "Mammalian Sterile20-like kinase 1; Hippo pathway component (STK4 → MOB1 → LATS1/2 → YAP/TAZ phosphorylation → nuclear exclusion → growth suppression); in T cells: phosphorylates FOXO3a → Bcl-2 upregulation → naive T-cell survival; STK4 LOF → naive T-cell apoptosis → T-lymphopenia + cardiac defects"},
    {"term": "Naive T-Cell Lymphopenia (STK4)", "definition": "CD45RA+ CD62L+ naive T-cell fraction severely reduced in STK4/MST1 deficiency; memory T cells (CD45RO+) initially preserved; distinguishable by FACS naive/memory T-cell panel; children present with recurrent bacterial infections before memory compartment is depleted"},
    {"term": "Cardiac Defect + T-Cell Lymphopenia (STK4 Triad)", "definition": "PATHOGNOMONIC ASSOCIATION: congenital heart disease (ASD/VSD 40-60%) + naive T-cell lymphopenia + recurrent pulmonary infections = STK4/MST1 deficiency; no other known PID combines congenital structural cardiac disease with naive T lymphopenia; echocardiogram MANDATORY at diagnosis"},
    {"term": "WASP Flow Cytometry", "definition": "Rapid diagnostic test: anti-WASP antibody staining of permeabilised peripheral blood lymphocytes (CD3+ T cells); WASP absent/severely reduced → WAS LOF OR WIPF1 deficiency; distinguishes from XLN (WASP present in XLN GOF); confirms diagnosis before gene sequencing; pitfall: WIPF1 gives absent WASP despite normal WAS gene"},
    {"term": "ARPC1B Eosinophilia", "definition": "Blood eosinophils > 1500/µL — PATHOGNOMONIC marker of ARPC1B deficiency in the cytoskeletal PID context; caused by dysregulated innate immunity (Arp2/3 complex loss → NLRP3 inflammasome overactivation + Th2 skewing + TSLP-driven eosinophilia); not seen in WAS, WIPF1, or CORONIN1A"},
    {"term": "WAS Score (Ochs / Zhu Classification)", "definition": "0 = thrombocytopenia alone; 1 = thrombocytopenia + mild infections; 2 = WAS triad without autoimmunity; 3 = WAS triad + autoimmunity/vasculitis; 4 = WAS triad + malignancy or haemorrhage; 5 = severe complications or death; Score ≥ 3 or EBV positivity → HSCT indicated"},
]


def _gene_cohort(gene: str, seed: int) -> dict:
    rng = random.Random(seed)
    n = 40
    meta = GENE_META[gene]

    if gene == "WAS":
        mean_onset_months = rng.gauss(4, 2)
        mean_dx_lag = rng.gauss(1.5, 0.8)
        thrombocytopenia_n  = rng.randint(38, 40)    # near-universal
        eczema_n            = rng.randint(28, 36)    # 70-90%
        pid_n               = rng.randint(34, 40)    # very common
        ebv_lymphoma_n      = rng.randint(5, 9)      # 13-22%
        autoimmune_n        = rng.randint(10, 18)    # 40-50%
        xlb_gof_n           = rng.randint(3, 7)      # XLN subgroup
        hsct_n              = rng.randint(26, 34)    # majority undergo HSCT
        small_plt_mpv_n     = rng.randint(37, 40)    # PATHOGNOMONIC
        pathognomonic       = "SMALL PLATELETS (MPV < 7 fL) PATHOGNOMONIC — ITP has large platelets (MPV elevated); WAS platelets are structurally small (cytoskeletal defect); thrombocytopenia + small MPV = WAS/WIPF1 until proven otherwise; WASP flow cytometry absent confirms; EBV lymphoma 13-22% lifetime risk → HSCT before EBV"

    elif gene == "WIPF1":
        mean_onset_months = rng.gauss(5, 2)
        mean_dx_lag = rng.gauss(3.0, 1.5)           # diagnostic delay: WAS gene normal → missed
        thrombocytopenia_n  = rng.randint(36, 40)
        eczema_n            = rng.randint(26, 34)
        pid_n               = rng.randint(32, 38)
        ebv_lymphoma_n      = rng.randint(4, 8)
        autoimmune_n        = rng.randint(8, 16)
        xlb_gof_n           = 0                      # no XLN (different gene)
        hsct_n              = rng.randint(22, 30)
        small_plt_mpv_n     = rng.randint(35, 40)
        pathognomonic       = "WAS PHENOCOPY DIAGNOSTIC TRAP: WASP PROTEIN ABSENT by flow cytometry → looks like WAS; WAS GENE NORMAL by sequencing → WIPF1 must be sequenced next; rule: absent WASP protein + normal WAS gene → WIPF1 deficiency; AR inheritance → FEMALES AFFECTED (distinguishes from X-linked WAS)"

    elif gene == "ARPC1B":
        mean_onset_months = rng.gauss(6, 3)
        mean_dx_lag = rng.gauss(3.5, 1.5)
        thrombocytopenia_n  = rng.randint(24, 34)    # present but different mechanism
        eczema_n            = rng.randint(16, 26)
        pid_n               = rng.randint(32, 38)
        ebv_lymphoma_n      = rng.randint(2, 6)
        autoimmune_n        = rng.randint(28, 36)    # HIGH: vasculitis + IBD
        xlb_gof_n           = 0
        hsct_n              = rng.randint(20, 28)
        small_plt_mpv_n     = rng.randint(18, 28)    # delta-granule defect, not small size
        pathognomonic       = "EOSINOPHILIA > 1500/µL PATHOGNOMONIC in cytoskeletal PID context — ARPC1B is the ONLY cytoskeletal PID gene with eosinophilia; also: EARLY ONSET IBD (Crohn's-like, < 5yr) + VASCULITIS (cutaneous/systemic) + PID triple combination is unique to ARPC1B; PLATELET DELTA-GRANULE ABSENT (storage pool disease) distinguishes from WAS (which has small platelets but normal granules)"

    elif gene == "CORONIN1A":
        mean_onset_months = rng.gauss(3, 1.5)
        mean_dx_lag = rng.gauss(2.0, 1.0)
        thrombocytopenia_n  = rng.randint(2, 8)      # NOT a feature
        eczema_n            = rng.randint(4, 10)      # NOT characteristic
        pid_n               = rng.randint(38, 40)    # T-cell absent → severe
        ebv_lymphoma_n      = rng.randint(6, 12)
        autoimmune_n        = rng.randint(4, 8)
        xlb_gof_n           = 0
        hsct_n              = rng.randint(30, 38)    # most require HSCT
        small_plt_mpv_n     = rng.randint(2, 6)
        pathognomonic       = "THYMIC SHADOW PRESENT/ENLARGED on CXR in a T-cell-lymphopenic infant = CORONIN1A PATHOGNOMONIC; most SCID has absent thymic shadow (empty thymus); CORO1A: thymus packed with mature T cells unable to egress; T-B+NK+ immunophenotype = Swiss-type SCID; p.R18C Swiss founder mutation"

    elif gene == "RAC2":
        mean_onset_months = rng.gauss(1, 0.5)        # neonatal (GOF D57N)
        mean_dx_lag = rng.gauss(0.5, 0.3)
        thrombocytopenia_n  = rng.randint(8, 16)
        eczema_n            = rng.randint(4, 10)
        pid_n               = rng.randint(36, 40)
        ebv_lymphoma_n      = rng.randint(4, 8)
        autoimmune_n        = rng.randint(6, 12)
        xlb_gof_n           = rng.randint(18, 26)    # GOF D57N subgroup majority
        hsct_n              = rng.randint(28, 36)
        small_plt_mpv_n     = rng.randint(4, 10)
        pathognomonic       = "RAC2-D57N GOF: DELAYED CORD SEPARATION (> day 14) + NEONATAL SEPSIS + LEUKOCYTOSIS (15-30 × 10⁹/L) = PATHOGNOMONIC LAD-SCID overlap; umbilical cord delay distinguishes from most SCID; cord separation requires neutrophil elastase + actin dynamics (RAC2 dependent); D57N = constitutively GTP-bound → SCID severity → HSCT from birth; LOF (AR): milder isolated neutrophil defect"

    elif gene == "RHOH":
        mean_onset_months = rng.gauss(18, 6)         # later onset — warts typically present 1-3 years
        mean_dx_lag = rng.gauss(5.0, 2.5)            # often misdiagnosed: warts not linked to PID
        thrombocytopenia_n  = rng.randint(2, 8)
        eczema_n            = rng.randint(4, 10)
        pid_n               = rng.randint(32, 38)
        ebv_lymphoma_n      = rng.randint(10, 18)    # HIGH: EBV susceptibility prominent
        autoimmune_n        = rng.randint(4, 10)
        xlb_gof_n           = 0
        hsct_n              = rng.randint(16, 24)
        small_plt_mpv_n     = rng.randint(2, 6)
        pathognomonic       = "HPV WARTS (SEVERE, EXTENSIVE, RECALCITRANT) PATHOGNOMONIC — hundreds of plantar/periungual/flat warts from childhood; if HPV warts are unresponsive to all standard therapies + recurrent infections → RHOH sequencing; also: KAPOSI SARCOMA (HHV8) in young non-HIV patient + EBV LPD are the other distinctive features; ultra-rare (< 30 families worldwide)"

    elif gene == "RLTPR":
        mean_onset_months = rng.gauss(8, 3)
        mean_dx_lag = rng.gauss(3.0, 1.5)
        thrombocytopenia_n  = rng.randint(4, 10)
        eczema_n            = rng.randint(12, 20)
        pid_n               = rng.randint(30, 36)
        ebv_lymphoma_n      = rng.randint(8, 14)
        autoimmune_n        = rng.randint(20, 28)    # IBD + colitis from Treg deficiency
        xlb_gof_n           = 0
        hsct_n              = rng.randint(20, 28)
        small_plt_mpv_n     = rng.randint(2, 8)
        pathognomonic       = "CD28 STIMULATION ASSAY: isolated CD28 ligation → IL-2 severely reduced = CARMIL2 deficiency PATHOGNOMONIC; CD3-driven activation relatively preserved; CMC + Crohn's-like colitis + EBV lymphoma triple = CARMIL2; CMC here is CD28-actin mechanism (NOT IL-17 pathway like STAT1 GOF or IL17RA); TREG DEFICIENCY (CD25+FoxP3+ reduced) → uncontrolled colitis; distinguish from LRBA (CTLA4 recycling; Abatacept effective)"

    else:  # STK4
        mean_onset_months = rng.gauss(12, 4)         # 1st year of life (cardiac + infections)
        mean_dx_lag = rng.gauss(2.5, 1.0)
        thrombocytopenia_n  = rng.randint(2, 8)
        eczema_n            = rng.randint(6, 12)
        pid_n               = rng.randint(36, 40)
        ebv_lymphoma_n      = rng.randint(4, 8)
        autoimmune_n        = rng.randint(8, 14)
        xlb_gof_n           = 0
        hsct_n              = rng.randint(22, 30)
        small_plt_mpv_n     = rng.randint(2, 6)
        pathognomonic       = "CONGENITAL HEART DISEASE (ASD/VSD) + NAIVE T-CELL LYMPHOPENIA + RECURRENT PULMONARY INFECTIONS = STK4 PATHOGNOMONIC TRIAD; no other known PID combines structural cardiac defects with naive T-cell lymphopenia; ECHOCARDIOGRAM MANDATORY at diagnosis; CARDIAC DEFECT IS NOT CORRECTED BY HSCT (structural); HSCT corrects immune defect only; naive/memory T-cell FACS: CD45RA+ fraction severely reduced"

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
        "thrombocytopenia_n": thrombocytopenia_n,
        "thrombocytopenia_pct": round(thrombocytopenia_n / n * 100, 1),
        "eczema_n": eczema_n,
        "eczema_pct": round(eczema_n / n * 100, 1),
        "pid_n": pid_n,
        "pid_pct": round(pid_n / n * 100, 1),
        "ebv_lymphoma_n": ebv_lymphoma_n,
        "ebv_lymphoma_pct": round(ebv_lymphoma_n / n * 100, 1),
        "autoimmune_n": autoimmune_n,
        "autoimmune_pct": round(autoimmune_n / n * 100, 1),
        "xlb_gof_n": xlb_gof_n,
        "xlb_gof_pct": round(xlb_gof_n / n * 100, 1),
        "hsct_n": hsct_n,
        "hsct_pct": round(hsct_n / n * 100, 1),
        "small_plt_mpv_n": small_plt_mpv_n,
        "small_plt_mpv_pct": round(small_plt_mpv_n / n * 100, 1),
        "pathognomonic": pathognomonic,
        "mim_gene": meta["mim_gene"],
        "mim_disease": meta["mim_disease"],
        "seed": seed,
    }


def generate_overview() -> dict:
    cohorts = [_gene_cohort(g, SEED_BASE + i) for i, g in enumerate(GENES)]
    total_n             = sum(c["n"] for c in cohorts)
    total_thrombo       = sum(c["thrombocytopenia_n"] for c in cohorts)
    total_eczema        = sum(c["eczema_n"] for c in cohorts)
    total_pid           = sum(c["pid_n"] for c in cohorts)
    total_ebv           = sum(c["ebv_lymphoma_n"] for c in cohorts)
    total_autoimmune    = sum(c["autoimmune_n"] for c in cohorts)
    total_hsct          = sum(c["hsct_n"] for c in cohorts)
    total_small_plt     = sum(c["small_plt_mpv_n"] for c in cohorts)
    return {
        "atlas": "Hereditary-Wiskott-Aldrich-Cytoskeletal-Immunodeficiency-Atlas",
        "subtitle": "Actin/Cytoskeletal Defects Affecting Immune Cells / WASp Family–Arp2/3–Rho GTPases–Hippo Kinase (IUIS PID Categories 1, 2, 6)",
        "total_patients": total_n,
        "genes_covered": GENES,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "aggregate_stats": {
            "thrombocytopenia_pct":  round(total_thrombo / total_n * 100, 1),
            "eczema_pct":            round(total_eczema / total_n * 100, 1),
            "pid_pct":               round(total_pid / total_n * 100, 1),
            "ebv_lymphoma_pct":      round(total_ebv / total_n * 100, 1),
            "autoimmune_pct":        round(total_autoimmune / total_n * 100, 1),
            "hsct_pct":              round(total_hsct / total_n * 100, 1),
            "small_plt_mpv_pct":     round(total_small_plt / total_n * 100, 1),
        },
        "cohort_summaries": [
            {k: v for k, v in c.items() if k != "pathognomonic"}
            for c in cohorts
        ],
        "cascade_testing_concepts": [
            "INFANT with MICROTHROMBOCYTOPENIA + SMALL PLATELETS (MPV < 7 fL) → WAS/WIPF1 workup FIRST: check MPV (not just platelet count); WASP flow cytometry; if WASP absent but WAS gene normal → WIPF1 sequencing; if male → XLN excluded only if WAS gene sequenced",
            "WASP PROTEIN ABSENT + WAS GENE NORMAL → WIPF1 DEFICIENCY: critical diagnostic rule; sequence WIPF1 next; females with WAS-like phenotype → cannot be WAS (X-linked) → WIPF1 or ARPC1B immediately",
            "EARLY ONSET IBD (< 5yr) + EOSINOPHILIA (> 1500/µL) + VASCULITIS + RECURRENT INFECTIONS → ARPC1B: this triple combination is pathognomonic; ARPC1B flow cytometry (absent protein) + gene sequencing; JAKi for inflammatory control while planning HSCT",
            "T-CELL LYMPHOPENIA + ENLARGED/PRESENT THYMIC SHADOW on CXR → CORONIN1A: most SCID has absent thymic shadow; if thymus visible/prominent + T cells absent → CORONIN1A first; T-B+NK+ immunophenotype; p.R18C Swiss founder",
            "NEONATAL SEPSIS + DELAYED CORD SEPARATION (> day 14) + LEUKOCYTOSIS → RAC2-D57N GOF: mimics LAD-I (leukocyte adhesion deficiency) but also T-cell lymphopenic; CD18/CD11 expression normal (LAD-I excluded); RAC2 gene sequencing urgently; HSCT from birth",
            "SEVERE RECALCITRANT HPV WARTS (hundreds, plantar/periungual) in a child with recurrent infections → RHOH DEFICIENCY: if standard wart treatment fails + PID features → RHOH sequencing; also check for Kaposi sarcoma (HHV8) and EBV LPD; ultra-rare but treatable with HSCT",
            "CMC + VERY EARLY ONSET COLITIS + EBV LYMPHOPROLIFERATION + TREG REDUCED → CARMIL2 (RLTPR): distinguish from LRBA (CTLA4 recycling → Abatacept effective; CARMIL2 is NOT abatacept-responsive); CD28 stimulation assay (isolated CD28 ligation → reduced IL-2) confirms CARMIL2",
            "CONGENITAL HEART DISEASE (ASD/VSD) + NAIVE T-CELL LYMPHOPENIA + RECURRENT PULMONARY INFECTIONS → STK4/MST1: echocardiogram MANDATORY; cardiac defect is structural (HSCT corrects immune defect only); naive/memory T-cell FACS panel to quantify CD45RA+ depletion",
            "HSCT TIMING: WAS score ≥ 3 OR EBV-positive → HSCT before malignancy; CORONIN1A → urgent; RAC2-D57N → emergency neonatal HSCT; ARPC1B → early (inflammatory damage); STK4 → before immune damage; RHOH/CARMIL2 → variable, HSCT curative",
            "BCG ABSOLUTELY CONTRAINDICATED: ALL 8 genes in this atlas — all have combined T-cell deficiency component; BCGosis is fatal without functional T cells; check BCG vaccination status and onset of symptoms in all newly diagnosed patients",
        ],
        "iuis_category": "IUIS 2022: WAS/WIPF1/ARPC1B/CARMIL2/STK4 = Category 2 (Combined with syndromic features); CORONIN1A = Category 1 (SCID); RAC2/RHOH = Categories 2/6 (cytoskeletal Rho GTPase)",
    }


def generate_breakdown() -> dict:
    cohorts = [_gene_cohort(g, SEED_BASE + i) for i, g in enumerate(GENES)]
    return {
        "atlas": "Hereditary-Wiskott-Aldrich-Cytoskeletal-Immunodeficiency-Atlas",
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "gene_cohorts": cohorts,
        "diagnostic_groups": {
            "wasp_family": {
                "genes": ["WAS", "WIPF1"],
                "mechanism": "WASp complex integrity: WAS LOF = WASP absent; WIPF1 LOF = WASP degraded despite normal WAS gene",
                "shared_feature": "Microthrombocytopenia (small MPV) + eczema + PID triad; WASP absent on flow cytometry in BOTH",
                "key_ddx": "WAS: X-linked (males only, females carriers); WIPF1: AR (females affected); diagnostic rule: absent WASP + normal WAS gene → WIPF1",
            },
            "arp2_3_complex": {
                "genes": ["ARPC1B"],
                "mechanism": "Arp2/3 complex subunit deficiency → impaired actin branching in all immune cells + dysregulated innate inflammasome",
                "shared_feature": "Unique triple: PID + IBD + eosinophilia + vasculitis; platelet delta-granule defect (storage pool disease)",
                "key_ddx": "Eosinophilia > 1500/µL + early IBD + vasculitis = ARPC1B; WASP normal on flow; ARPC1B protein absent",
            },
            "thymic_egress_defect": {
                "genes": ["CORONIN1A"],
                "mechanism": "F-actin dynamics in thymocytes required for egress; CORO1A LOF = T cells trapped in thymus",
                "shared_feature": "T-B+NK+ SCID; thymic shadow present/enlarged on CXR (pathognomonic in SCID)",
                "key_ddx": "Other T-B+NK+ SCID (ZAP70, MHC-II): NO enlarged thymus on CXR; CORONIN1A = enlarged/present thymus",
            },
            "rho_gtpase_defects": {
                "genes": ["RAC2", "RHOH"],
                "mechanism": "Haematopoiesis-restricted Rho GTPases controlling actin dynamics, NADPH oxidase, TCR signalling",
                "shared_feature": "Both affect T-cell function; both rare; both cause distinct pathognomonic features",
                "key_ddx": "RAC2-GOF: neonatal onset, cord delay, leukocytosis, SCID-severity; RHOH: childhood HPV warts, Kaposi, EBV LPD (no neonatal crisis)",
            },
            "cd28_hippo_signalling": {
                "genes": ["RLTPR", "STK4"],
                "mechanism": "CARMIL2: CD28 co-stimulation actin scaffold → T-cell activation + Treg; STK4: Hippo kinase → naive T-cell survival + cardiac development",
                "shared_feature": "Both cause T-cell dysfunction with prominent non-immune features (IBD in CARMIL2; cardiac in STK4)",
                "key_ddx": "CARMIL2: CMC + colitis + EBV; STK4: cardiac defects (ASD/VSD) + naive T lymphopenia (no CMC, no colitis)",
            },
        },
        "pathognomonic_features": {
            "WAS_small_MPV": "SMALL PLATELET SIZE (MPV < 7 fL) — PATHOGNOMONIC of WAS/WIPF1; ITP = large reactive platelets (MPV elevated); Bernard-Soulier = giant platelets; only WAS spectrum = small MPV",
            "WIPF1_absent_WASP_normal_gene": "WASP PROTEIN ABSENT + WAS GENE NORMAL — pathognomonic of WIPF1 deficiency; appears identical to WAS on WASP flow cytometry but molecular diagnosis reveals normal WAS; mandates WIPF1 sequencing",
            "ARPC1B_eosinophilia": "EOSINOPHILIA > 1500/µL + EARLY IBD + VASCULITIS in a PID patient = ARPC1B pathognomonic triple; no other cytoskeletal PID gene produces eosinophilia",
            "CORONIN1A_thymic_shadow": "ENLARGED/PRESENT THYMIC SHADOW on CXR in a T-cell-lymphopenic infant = CORONIN1A; the thymus is packed with mature T cells unable to egress; most SCID has absent/tiny thymic shadow",
            "RAC2_cord_delay": "DELAYED CORD SEPARATION (> 14 days) + NEONATAL SEPSIS + LEUKOCYTOSIS in the first weeks of life = RAC2-D57N GOF; LAD-I mimic but T-cell lymphopenic and CD18 normal",
            "RHOH_HPV_warts": "SEVERE RECALCITRANT HPV WARTS (hundreds, early childhood onset) + PID features = RHOH deficiency; warts unresponsive to all standard treatment; plantar, periungual, flat warts most prominent",
            "CARMIL2_CD28_assay": "CD28 STIMULATION ASSAY showing severely reduced IL-2 with intact CD3 response = CARMIL2 pathognomonic; distinguishes from other T-cell activation defects where both CD3 and CD28 pathways fail",
            "STK4_cardiac_T_lymphopenia": "CONGENITAL HEART DISEASE (ASD/VSD) + NAIVE CD45RA+ T-CELL LYMPHOPENIA + RECURRENT PULMONARY INFECTIONS = STK4 pathognomonic triad; no other PID combines structural heart disease with naive T lymphopenia",
        },
    }


def generate_definitions() -> dict:
    return {
        "atlas": "Hereditary-Wiskott-Aldrich-Cytoskeletal-Immunodeficiency-Atlas",
        "definitions": DEFINITIONS,
        "gene_metadata": GENE_META,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "key_distinguishers": {
            "WAS_vs_WIPF1": "WAS: X-linked (males only); WASP absent + WAS gene mutated; WIPF1: AR (females affected); WASP absent + WAS gene NORMAL → WIPF1; both have WAS clinical triad (small platelets, eczema, PID); both treated with HSCT",
            "WAS_thrombocytopenia_vs_ITP": "WAS/WIPF1: small platelets (MPV < 7 fL) + congenital; ITP: large reactive platelets (MPV ≥ 12 fL) + acquired; platelet transfusion in WAS: rapidly consumed (same WASp defect in donor platelets? No — donor platelets survive longer); splenectomy dangerous in WAS (sepsis risk from combined PID)",
            "ARPC1B_vs_WAS": "WAS: small MPV + no eosinophilia + no IBD; ARPC1B: normal/near-normal MPV + EOSINOPHILIA > 1500 + EARLY IBD + VASCULITIS; WASP flow: normal in ARPC1B (different protein); ARPC1B flow/Western confirms ARPC1B protein absence",
            "CORONIN1A_vs_other_T_B_Plus_NK_Plus_SCID": "T-B+NK+ SCID DDx: ZAP70 (impaired TCR signalling, normal thymus size), MHC-II (CD4 lymphopenia, normal thymus), CORONIN1A (ENLARGED THYMIC SHADOW — unique); thymus imaging MANDATORY in T-B+NK+ SCID",
            "RAC2_GOF_vs_LAD": "RAD2-D57N GOF: cord delay + SCID-severity T-cell lymphopenia + D57N sequencing confirms; LAD-I: cord delay + leukocytosis but CD18/CD11b absent (CD18 deficiency); RAC2: CD18 NORMAL; LAD-I: T cells NORMAL; neutrophil CD18 flow cytometry distinguishes immediately",
            "CARMIL2_vs_LRBA": "CARMIL2: CMC + colitis + EBV LPD; CD28 assay confirms; CTLA4 surface normal (WAS-like mechanism, not CTLA4 recycling); Abatacept NOT effective; LRBA: CTLA4 recycling defect → CTLA4 surface reduced → Abatacept HIGHLY EFFECTIVE; CTLA4 surface flow cytometry distinguishes LRBA from CARMIL2",
            "STK4_vs_DiGeorge": "STK4: congenital heart + T-lymphopenia; chromosome 22q11.2 deletion NORMAL; STK4 gene mutation; cardiac defect is random (ASD/VSD), not conotruncal; calcium NORMAL (no hypocalcaemia); DiGeorge: 22q11.2 deletion; conotruncal heart; hypocalcaemia; thymic aplasia (T-B+NK+ SCID); TBX1 mutation",
            "BCG_risk_ranking_cytoskeletal": "ALL 8 genes CI for BCG; highest severity if BCGosis occurs: CORONIN1A (T absent, fatal BCGosis) = RAC2-D57N (SCID-equivalent) > WAS/WIPF1 (combined PID) > CARMIL2/STK4 (T-cell dysfunction) > ARPC1B (combined PID) > RHOH (T-cell dysfunction); check BCG vaccination date and onset of symptoms in every suspected cytoskeletal PID",
        },
        "diagnostic_pathway": [
            "Step 1: Male infant with thrombocytopenia + eczema + recurrent infections → WAS workup; check MPV: < 7 fL → WAS/WIPF1; WASP flow cytometry immediately",
            "Step 2: WASP absent on flow → WAS gene sequencing; if WAS gene NORMAL → WIPF1 sequencing; female with same phenotype → cannot be WAS → WIPF1 first",
            "Step 3: Thrombocytopenia + eosinophilia (> 1500) + early IBD + vasculitis → ARPC1B; check WASP (normal), then ARPC1B protein by flow/Western; inflammatory control with ruxolitinib while awaiting HSCT evaluation",
            "Step 4: T-cell lymphopenia in infant + CXR showing thymic shadow → CORONIN1A; most SCID has absent thymic shadow — this finding redirects workup; T-B+NK+ immunophenotype; p.R18C sequencing (Swiss-type) and full CORO1A gene panel",
            "Step 5: Neonatal omphalitis + delayed cord separation (> 14 days) + leukocytosis + T-cell lymphopenia → CD18 flow cytometry first (LAD-I excluded if normal); then RAC2 sequencing (D57N most common); HSCT consultation urgently from birth for GOF",
            "Step 6: Child with severe recalcitrant HPV warts (hundreds) not responding to any therapy + recurrent infections → RHOH sequencing; check for Kaposi sarcoma and EBV LPD; rituximab for active EBV LPD; HSCT curative",
            "Step 7: CMC + early colitis (< 3yr) + EBV LPD + reduced Treg → CD28 stimulation assay (anti-CD28 ligation → IL-2); reduced response → CARMIL2/RLTPR sequencing; CTLA4 surface flow (to exclude LRBA); abatacept NOT indicated (CARMIL2 mechanism); HSCT curative",
            "Step 8: Any cardiac defect (ASD/VSD) + naive T-cell lymphopenia + recurrent pulmonary infections → echocardiogram MANDATORY + STK4/MST1 sequencing; naive/memory T-cell FACS (CD45RA+/CD45RO+ ratio); HSCT corrects immune defect; cardiac defect managed separately by cardiology",
        ],
    }
