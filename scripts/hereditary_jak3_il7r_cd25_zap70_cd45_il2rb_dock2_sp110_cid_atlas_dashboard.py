#!/usr/bin/env python3
"""Hereditary-JAK3-IL7R-CD25-ZAP70-CD45-IL2RB-DOCK2-SP110-CID-Variant-Atlas -- Complete 8-Gene Reference
Cytokine-Receptor-Signaling and Proximal-TCR/BCR Defects Causing Distinct CID/SCID-Variant Phenotypes (IUIS 2022)
(JAK3-1124aa-19p13.11-AR-T-B+NK-SCID-γc-Downstream-Both-Sexes-Phenotypically-Identical-to-IL2RG-XSCID,
 IL7RA-459aa-5p13.2-AR-T-B+NK+SCID-T-Cell-Absent-B-NK-Normal-Distinct-from-γc-and-JAK3,
 IL2RA-CD25-272aa-10p15.1-AR-TREG-Absent-Paradox-Lymphoproliferation-Autoimmunity-Infections,
 ZAP70-619aa-2q11.2-AR-SELECTIVE-CD8-Absence-PATHOGNOMONIC-CD4-Normal-Autoimmune,
 PTPRC-CD45-1304aa-1q31.3-AR-T-B-NK-SCID-CD45-Absent-ALL-Lymphocytes-Phosphatase,
 IL2RB-CD122-525aa-22q12.3-AR-NK-Absent-T-Lymphoproliferation-Autoimmunity,
 DOCK2-1862aa-5q35.1-AR-NK-Cell-Deficiency-T-Dysfunction-Viral-Susceptibility-NEJM2015,
 SP110-697aa-2q37.1-AR-VODI-Hepatic-Veno-Occlusive-Disease-PATHOGNOMONIC-Hypogammaglobulinemia-Druze):

JAK3   (Janus kinase 3 / JAK3; 1124aa; 19p13.11; AR (biallelic LOF);
          JAK3 encodes the Janus kinase 3, which is EXCLUSIVELY expressed in hematopoietic cells and
            constitutively associated with the common gamma chain (γc, IL2RG) of the cytokine receptor
            superfamily (IL-2R, IL-4R, IL-7R, IL-9R, IL-15R, IL-21R); upon cytokine binding, JAK3
            (on γc chain) transphosphorylates JAK1 (on the α/β subunit) → STAT5 activation →
            lymphocyte survival/proliferation signals; JAK3 LOF = phenotypically identical to IL2RG
            LOF (X-SCID) because they are in the same signaling axis;
          JAK3-DEFICIENT SCID: AR; both males AND females equally affected (unlike X-SCID which is
            male-predominant); T-B+NK- SCID phenotype: T cells absent (JAK3 downstream of IL-7Rγc →
            T cell development fails at pro-T to pre-T stage); NK cells absent (IL-15Rγc-JAK3 →
            NK maturation fails); B cells present but dysfunctional (IL-4/IL-21 signal absent);
            estimated incidence 1:100,000-1:400,000; accounts for ~7-15% of SCID;
          PATHOGNOMONIC FEATURE: T-B+NK- SCID IN A GIRL (female SCID = autosomal genes; IL2RG/X-SCID
            is in boys predominantly; JAK3 deficiency is the MOST COMMON autosomal cause of T-B+NK-
            SCID identical to X-SCID); both sexes: T lymphopenia (CD3 < 200/μL; CD4 and CD8 both
            absent); absent NK cells (CD56-CD16- < 20/μL); B cells present (IgG low due to absent
            T cell help; B cells may be normal/elevated numerically);
          IMMUNE PROFILE: T-B+NK-; quantitative lymphopenia on TREC NBS screening (TREC absent);
            lymphocyte proliferation absent; in vitro: T cells absent; flowcytometry confirms;
          DIAGNOSIS: CBC lymphopenia → flow cytometry T-B+NK- → JAK3 sequencing; screen for IL2RG
            mutation first (X-SCID exclusion in boys); JAK3 activity assays (pSTAT5 in lymphocytes
            after cytokine stimulation — absent);
          TREATMENT: HSCT curative — allogenic bone marrow transplantation (same protocol as X-SCID;
            myeloablative or RIC); gene therapy for JAK3-SCID under investigation (ex vivo lentiviral
            + ruxolitinib bridge therapy studied); BCG absolutely contraindicated (fatal BCGosis);
            live vaccines contraindicated; IVIG bridge until HSCT; PCP + antifungal prophylaxis;
            seed SEED_BASE+0).

IL7RA  (IL-7 receptor subunit alpha / IL7RA; 459aa; 5p13.2; AR (biallelic LOF);
          IL7RA encodes the IL-7 receptor α chain (CD127), which pairs with γc (IL2RG) to form the
            IL-7 receptor; IL-7 signaling is ESSENTIAL for T cell development in the thymus (CD44-
            CD25+ stage) and for peripheral naive T cell survival; IL-7Rα LOF = loss of IL-7 signal
            only (unlike JAK3 LOF which loses IL-2/IL-4/IL-7/IL-9/IL-15/IL-21 signals);
          IL7RA-DEFICIENT SCID: AR; both males and females; T-B+NK+ SCID phenotype: T cells absent
            (IL-7 signal lost → DN3 block in thymus); B cells NORMAL count and function (IL-7 not
            required for B cell development in humans); NK cells NORMAL (IL-15 signaling through
            IL-15Rβγc = JAK1-JAK3 axis, NOT dependent on IL7RA); estimated incidence 1:200,000-
            1:1,000,000;
          KEY DISTINCTION FROM JAK3/IL2RG SCID: NK CELLS PRESENT (T-B+NK+ vs T-B+NK- in JAK3/IL2RG).
            This is the diagnostic key: flow shows T cells absent, B cells normal, NK cells NORMAL
            (unlike JAK3/IL2RG where NK are also absent);
          PATHOGNOMONIC: T-B+NK+ SCID = IL7RA (or RAG1/RAG2 or ADA or other T-B SCID types);
            distinguished from RAG: RAG has T-B- (B cells absent); distinguished from ADA: ADA has
            T-B-NK- with metabolic findings (adenosine deaminase assay on RBC);
          IMMUNE PROFILE: very low/absent T cells (CD3 < 300/μL); normal/elevated B cells; NK cells
            normal; TREC-NBS positive (TREC absent → flag); JAK3 sequencing negative → IL7RA;
          DIAGNOSIS: flow cytometry T-B+NK+ → IL7RA sequencing (also test RAG1/RAG2 for T-B+ type);
            pSTAT5 in response to IL-7 absent (but present in response to IL-2, IL-15 — JAK3-JAK1
            axis intact); protein expression: IL7Rα absent on CD3 T cells (by flow, when any T cells
            present) or on PBMC;
          TREATMENT: HSCT curative; same urgency as other SCID; BCG absolutely contraindicated;
            IVIG bridge; RIC conditioning effective; gene therapy (IL7RA lentiviral) in pre-clinical
            stage; seed SEED_BASE+1).

IL2RA  (IL-2 receptor subunit alpha / IL2RA (CD25); 272aa; 10p15.1; AR (biallelic LOF);
          IL2RA encodes CD25, the high-affinity α subunit of the IL-2 receptor (IL-2Rα); CD25 is
            constitutively expressed on FOXP3+ regulatory T cells (Tregs) and transiently induced
            on activated T cells; IL-2/IL-2Rα high-affinity trimeric signaling is the PRIMARY
            SURVIVAL SIGNAL FOR TREGS (Tregs depend on IL-2 from conventional T cells);
          IL2RA/CD25 DEFICIENCY: AR; extremely rare (< 30 families described); PARADOXICAL IMMUNE
            DYSREGULATION rather than combined immunodeficiency;
          PARADOXICAL PHENOTYPE: (1) LYMPHOPROLIFERATION — ENLARGED lymph nodes, splenomegaly,
            liver infiltration with lymphocytes (lymphocyte EXPANSION rather than lymphopenia —
            paradox because conventional T cells proliferate unchecked without IL-2Rα-driven Treg
            suppression); (2) TREG ABSENCE OR FUNCTIONAL DEFICIENCY — CD25+ FOXP3+ Treg numbers
            may be low/absent or dysfunctional (cannot respond to IL-2 → cannot suppress auto-
            reactive T cells); (3) INFLAMMATORY BOWEL DISEASE — chronic enteropathy resembling
            IPEX syndrome (FOXP3 deficiency) but milder; (4) RECURRENT INFECTIONS — especially
            CMV, EBV, Pneumocystis, candida → combined immunodeficiency component;
          KEY DISTINCTION FROM IPEX (FOXP3): IPEX males only (FOXP3 on X); IL2RA/CD25 affects
            both sexes; IPEX earlier onset (neonatal); CD25 deficiency: CD25+ T cells absent on
            flow → TEST: IL-2Rα surface on activated T cells (induced CD25 absent on T cell
            surface after stimulation); in IPEX: CD25 surface normal but FOXP3 absent;
          CLINICAL FEATURES: autoimmune thyroiditis, autoimmune hemolytic anemia, CMV organ disease,
            lymphadenopathy, chronic diarrhea, failure to thrive, wasting; onset 2-12 months;
          DIAGNOSIS: lymphadenopathy + recurrent CMV + IBD-like diarrhea → flow: CD25 absent on
            activated T cells (or all T cells) → IL2RA sequencing; Treg CD25+FOXP3+ markedly
            reduced; autoantibody screen positive (AIHA, antinuclear);
          TREATMENT: rapamycin (mTOR inhibitor) for lymphoproliferation; IVIG for infections;
            HSCT curative but technically challenging given lymphoproliferation; CMV ganciclovir;
            rituximab for autoimmune hemolytic anemia; seed SEED_BASE+2).

ZAP70  (zeta-chain associated protein kinase 70 / ZAP70; 619aa; 2q11.2; AR (biallelic LOF);
          ZAP70 encodes the Zeta-chain Associated Protein kinase 70, a key proximal TCR signaling
            kinase; upon TCR engagement: Lck (activated by CD4/CD8 co-receptor) phosphorylates ITAMs
            on CD3ζ chain → ZAP70 recruited and activated → ZAP70 phosphorylates LAT and SLP-76 →
            downstream T cell activation; CD8+ T cells use CD8-Lck for a STRONGER ZAP70 signal than
            CD4+ T cells;
          ZAP70 DEFICIENCY: AR; very rare (< 100 families described); presents as SELECTIVE CD8 T
            CELL LYMPHOPENIA — the MOST PATHOGNOMONIC feature in all of PID;
          PATHOGNOMONIC FEATURE: CD8+ T cells ABSENT (< 50/μL) + CD4+ T cells NORMAL or ELEVATED.
            ALL other SCID causes both CD4 and CD8 to be low together; ZAP70 is the ONLY condition
            where CD8 is absent while CD4 is normal/high; mechanism: CD8+ T cells require a higher
            ZAP70 threshold signal (via CD8-Lck-ZAP70 positive feedback) for thymic positive
            selection → ZAP70 partial LOF results in CD8 thymic deletion; CD4+ T cells (lower ZAP70
            threshold via CD4-Lck) survive thymic selection and can populate periphery; however,
            CD4+ T cells are functionally abnormal (auto-reactive due to altered thymic negative
            selection + absent CD8 regulatory surveillance);
          AUTOIMMUNITY: despite normal CD4 count, CD25+FOXP3+ Tregs function poorly in ZAP70
            deficiency → AUTOIMMUNE CYTOPENIAS, inflammatory bowel disease, arthritis in many
            patients; the paradox of CD4+ lymphocytosis + autoimmunity in an immunodeficient
            patient is pathognomonic;
          CLINICAL FEATURES: recurrent sinopulmonary infections (bacterial), viral infections (CMV,
            EBV), Pneumocystis pneumonia; autoimmune hemolytic anemia; IBD-like enteropathy; onset
            variable (infancy to early childhood); some patients initially diagnosed as ITP or AIHA;
          DIAGNOSIS: CBC — lymphopenia often ABSENT (CD4 normal!) → misdiagnosed; flow cytometry:
            CD8 ABSENT + CD4 normal/elevated; confirm ZAP70 protein by Western blot or
            intracellular flow; ZAP70 sequencing; functional: T cell proliferation to PHA (low);
            anti-CD3/CD28 phospho-ZAP70 absent;
          TREATMENT: HSCT curative — restores CD8+ T cells and corrects autoimmunity; HSCT before
            irreversible autoimmune organ damage; IVIG for infections; immunosuppression for
            autoimmunity (steroids, azathioprine) as bridge; seed SEED_BASE+3).

PTPRC  (protein tyrosine phosphatase receptor type C / PTPRC (CD45); 1304aa; 1q31.3; AR (biallelic LOF);
          PTPRC encodes CD45 (also known as leukocyte common antigen, LCA), the largest known
            cell-surface glycoprotein and the most abundant phosphatase on hematopoietic cells;
            CD45 is expressed on ALL leukocytes (T, B, NK, myeloid, monocytes) — hence 'common'
            antigen; CD45 phosphatase domain dephosphorylates INHIBITORY TYROSINE Tyr505 on LCK
            → LCK activation → TCR/BCR proximal signaling; CD45 is constitutively required for
            all lymphocyte activation (TCR and BCR signaling both require LCK activation via CD45);
          CD45 DEFICIENCY: AR; extremely rare (< 15 families worldwide); the most severe combined
            T-B-NK- SCID with LYMPHOCYTE PHENOTYPE that mirrors reticular dysgenesis BUT WITHOUT
            MYELOID APLASIA;
          PATHOGNOMONIC FEATURE: CD45 (CD45RO/CD45RA) ABSENT on ALL LYMPHOCYTES by flow cytometry.
            This is a simple DIAGNOSTIC TEST: anti-CD45-FITC antibody on peripheral blood → ALL
            lymphocytes are CD45-NEGATIVE (no surface CD45); compare to normals where > 95% of
            PBMC are CD45+. CD45 absent → LCK permanently inhibited (Tyr505 permanently phosphoryl-
            ated) → TCR signal transduction fails → T cells cannot be positively selected →
            T-B-NK- SCID. However, unlike reticular dysgenesis (AK2), MYELOID cells have some
            residual function (neutrophil counts often present);
          CLINICAL FEATURES: T-B-NK- SCID from birth; failure to thrive; recurrent infections;
            TREC-NBS screening positive; B cells absent (CD45 required for B cell BCR signaling →
            B cell development fails at pro-B stage in mice, but in humans B cells may be partially
            present with dysfunctional signal); normal neutrophil count (distinguishes from AK2/RD);
          DIAGNOSIS: WBC lymphopenia → flow: T-B-NK- + CD45 ABSENT on lymphocytes → PTPRC sequencing;
            anti-CD45 flow antibody is part of standard PID flow panel; protein absent by Western blot;
          TREATMENT: HSCT curative; same urgency as other T-B-NK- SCID; BCG absolutely CI;
            IVIG bridge; myeloablative or RIC; seed SEED_BASE+4).

IL2RB  (IL-2 receptor subunit beta / IL2RB (CD122); 525aa; 22q12.3; AR (biallelic LOF);
          IL2RB encodes CD122 (IL-2 receptor β subunit), which is shared between the IL-2 receptor
            (IL-2Rαβγ complex) and the IL-15 receptor (IL-15Rβγ complex); IL-15 signaling through
            IL-2Rβγ (JAK1/JAK3-STAT5) is the PRIMARY SURVIVAL SIGNAL for NK cells and CD8+ memory
            T cells; IL-2 signaling through IL-2Rβγ maintains peripheral T cell homeostasis;
          IL2RB/CD122 DEFICIENCY: AR; described in 2019 (very newly characterized); extremely rare
            (< 15 families described); IMMUNE DYSREGULATION PATTERN (not classic SCID);
          UNIQUE PHENOTYPE: NK CELLS ABSENT + T CELL LYMPHOPROLIFERATION. The combination is
            distinctive: (1) NK cells absent or severely reduced (IL-15 → NK survival completely
            dependent on CD122; without CD122, NK cells die — NK absent); (2) T cells PROLIFERATE
            (paradox: IL-2 cannot signal through IL-2Rβ → auto-reactive T cells are not controlled
            by Tregs; Tregs fail because Treg survival depends partly on IL-2Rβγ signaling too →
            conventional T cells expand unchecked); (3) AUTOIMMUNITY: cytokine storm, IFN-γ
            overproduction, hemophagocytosis; (4) EBV and CMV susceptibility (NK absent);
          CLINICAL FEATURES: neonatal/early-infantile onset; lymphadenopathy; splenomegaly;
            hemophagocytic lymphohistiocytosis (HLH)-like features; recurrent viral infections;
            inflammatory bowel disease; failure to thrive; elevated LDH, ferritin;
          DIAGNOSIS: HLH-like clinical picture + NK CELLS ABSENT + T CELL EXPANSION → CD122 (IL2RB)
            sequencing; flow: NK CD56+CD16+ absent; CD122 absent on T cells/NK cells by flow; pSTAT5
            after IL-2 stimulation absent (IL-2Rβ absent); HLH diagnostic criteria met in some;
          TREATMENT: HSCT curative; ruxolitinib/baricitinib for JAK1/JAK3 blockade as HLH bridge;
            anti-IL-6 (tocilizumab) for cytokine storm; HSCT urgency similar to HLH/FHL5 (STXBP2);
            seed SEED_BASE+5).

DOCK2  (dedicator of cytokinesis 2 / DOCK2; 1862aa; 5q35.1; AR (biallelic LOF);
          DOCK2 encodes an atypical GEF (guanine nucleotide exchange factor) for RAC1 and RAC2 GTPases,
            specifically activating the actin-dependent migration and degranulation of lymphocytes and
            NK cells; DOCK2 is selectively expressed in hematopoietic cells (T cells, B cells, NK cells,
            neutrophils); DOCK2 activates RAC → actin cytoskeleton reorganization → (1) T/B/NK cell
            migration toward chemokine gradients (homing); (2) NK cell degranulation against viral
            targets; (3) T cell receptor clustering at immune synapse; (4) neutrophil superoxide
            production and phagocytosis;
          DOCK2 DEFICIENCY: AR; first described in 2015 (NEJM 2015;373:2413-2422 by Dobbs et al.);
            very rare (< 30 families described); COMBINED T/NK CELL DYSFUNCTION WITH PREDOMINANT
            VIRAL SUSCEPTIBILITY;
          PATHOGNOMONIC FEATURES: (1) SEVERE VIRAL SUSCEPTIBILITY — CMV, HSV-1, RSV, parainfluenza,
            EBV all cause life-threatening disease (NK cells cannot degranulate against virally
            infected cells without DOCK2-RAC actin); (2) NK CELL DEGRANULATION ABSENT — CD107a
            mobilization assay (diagnostic NK degranulation test) is ABSENT in DOCK2 deficiency
            (similar to LYST/CHS, but without giant granules); NK cell NUMBERS may be NORMAL but
            FUNCTION absent; (3) T CELL DYSFUNCTION — T cells present numerically but show
            impaired migration, immune synapse formation, and proliferative responses;
          CLINICAL FEATURES: recurrent severe CMV (organ-invasive: pneumonitis, retinitis, colitis);
            severe HSV (skin dissemination, encephalitis); EBV-driven lymphoproliferation; recurrent
            RSV bronchiolitis; onset from infancy; B cell lymphopenia in some;
          DIAGNOSIS: viral susceptibility profile (CMV + HSV + RSV + EBV) + NK cell numbers NORMAL
            but function ABSENT (CD107a assay) → DOCK2 sequencing; protein expression: DOCK2 absent
            by Western blot in lymphocytes; functional assay: RAC1 GTP-loading absent after
            CXCL12 stimulation;
          TREATMENT: HSCT curative — reconstitutes NK function and T cell migration; anti-CMV
            prophylaxis (valganciclovir) mandatory; IVIG for humoral component; acyclovir for HSV;
            BCG status: controversial — BCG not absolutely CI in DOCK2 (unlike classical SCID) but
            avoid until HSCT; seed SEED_BASE+6).

SP110  (SP110 nuclear body protein / SP110; 697aa; 2q37.1; AR (biallelic LOF);
          SP110 encodes a nuclear body scaffold protein related to SP100 and PML; SP110 is expressed
            in lymphocytes and liver; functions include: (1) transcriptional co-regulation of immune
            genes (particularly IFN-γ-stimulated genes); (2) nuclear body organization (SP110 is a
            component of PML-like nuclear bodies); (3) hepatic stellate cell regulation; SP110
            mutations disrupt both hepatic and lymphoid gene regulation simultaneously;
          SP110 DEFICIENCY / VODI SYNDROME (Veno-Occlusive Disease with Immunodeficiency): AR;
            described initially in Druze families from Lebanon (founder mutation c.1048_1049ins11,
            creating a premature stop codon in exon 15); later described in other populations;
          PATHOGNOMONIC FEATURE #1: HEPATIC VENO-OCCLUSIVE DISEASE (VOD / Sinusoidal Obstruction
            Syndrome) — endothelial damage of hepatic venules → occlusion → sinusoidal hypertension
            → portal hypertension + ascites + hepatomegaly + liver failure; VOD in an infant or
            young child WITHOUT STEM CELL TRANSPLANTATION (VOD in HSCT is common, but spontaneous
            congenital VOD is essentially PATHOGNOMONIC for SP110/VODI). Liver biopsy shows
            centrilobular fibrosis, sinusoidal congestion, obliteration of central veins;
          PATHOGNOMONIC FEATURE #2: COMBINED IMMUNODEFICIENCY — hypogammaglobulinemia (particularly
            low IgG + low IgA) + T cell dysfunction + NK cell reduced; T cells reduced (CD4 lympho-
            penia); recurrent infections (bacterial sinopulmonary, PCP, candida); the combination
            of spontaneous hepatic VOD + combined immunodeficiency in infancy is DIAGNOSTIC for
            VODI/SP110;
          CLINICAL FEATURES: liver failure (ascites, jaundice, coagulopathy); portal hypertension;
            hepatic VOD onset 1-6 months; pulmonary infections; oral candidiasis; failure to thrive;
          DIAGNOSIS: infant with hepatic VOD + immunodeficiency → SP110 sequencing; liver biopsy:
            centrilobular fibrosis + venular obliteration; immunology: low IgG/IgA, CD4 lymphopenia,
            NK reduced; abdominal ultrasound: hepatomegaly + ascites + reversed hepatic venous flow;
          TREATMENT: defibrotide (endothelial protective agent used for HSCT-induced VOD — used
            off-label for VODI) + HSCT curative; liver transplantation for liver failure before HSCT;
            IVIG replacement; TMP-SMX prophylaxis; antifungal; ursodeoxycholic acid for liver;
            seed SEED_BASE+7)."""

import random
import math

# ── seed-deterministic synthetic cohort generator ──────────────────────────

def _cohort(gene: str, seed: int, n: int = 40) -> dict:
    rng = random.Random(seed)

    params = {
        'JAK3':  {'t_lymphopenia_pct': 100, 'nk_absent_pct': 100, 'b_present_pct': 100, 'autoimmune_pct': 10,  'viral_sus_pct': 82,  'age_dx_mo': 3,   'hsct_pct': 90,  'surv_5yr_pct': 82, 'vod_pct': 0},
        'IL7RA': {'t_lymphopenia_pct': 100, 'nk_absent_pct': 0,   'b_present_pct': 100, 'autoimmune_pct': 8,   'viral_sus_pct': 75,  'age_dx_mo': 4,   'hsct_pct': 88,  'surv_5yr_pct': 85, 'vod_pct': 0},
        'IL2RA': {'t_lymphopenia_pct': 15,  'nk_absent_pct': 20,  'b_present_pct': 100, 'autoimmune_pct': 80,  'viral_sus_pct': 70,  'age_dx_mo': 6,   'hsct_pct': 58,  'surv_5yr_pct': 78, 'vod_pct': 0},
        'ZAP70': {'t_lymphopenia_pct': 40,  'nk_absent_pct': 0,   'b_present_pct': 100, 'autoimmune_pct': 68,  'viral_sus_pct': 65,  'age_dx_mo': 8,   'hsct_pct': 72,  'surv_5yr_pct': 80, 'vod_pct': 0},
        'PTPRC': {'t_lymphopenia_pct': 100, 'nk_absent_pct': 100, 'b_present_pct': 10,  'autoimmune_pct': 5,   'viral_sus_pct': 88,  'age_dx_mo': 2,   'hsct_pct': 95,  'surv_5yr_pct': 78, 'vod_pct': 0},
        'IL2RB': {'t_lymphopenia_pct': 20,  'nk_absent_pct': 100, 'b_present_pct': 100, 'autoimmune_pct': 72,  'viral_sus_pct': 80,  'age_dx_mo': 1.5, 'hsct_pct': 85,  'surv_5yr_pct': 72, 'vod_pct': 0},
        'DOCK2': {'t_lymphopenia_pct': 50,  'nk_absent_pct': 80,  'b_present_pct': 80,  'autoimmune_pct': 20,  'viral_sus_pct': 98,  'age_dx_mo': 5,   'hsct_pct': 78,  'surv_5yr_pct': 75, 'vod_pct': 0},
        'SP110': {'t_lymphopenia_pct': 75,  'nk_absent_pct': 55,  'b_present_pct': 70,  'autoimmune_pct': 10,  'viral_sus_pct': 72,  'age_dx_mo': 3,   'hsct_pct': 68,  'surv_5yr_pct': 62, 'vod_pct': 100},
    }[gene]

    def jitter(v, lo=-2, hi=2): return round(min(100, max(0, v + rng.uniform(lo, hi))), 1)

    return {
        "gene":               gene,
        "n":                  n,
        "seed":               seed,
        "t_lymphopenia_pct":  jitter(params['t_lymphopenia_pct'], -2, 2),
        "nk_absent_pct":      jitter(params['nk_absent_pct'], -2, 2),
        "b_present_pct":      jitter(params['b_present_pct'], -2, 2),
        "autoimmune_pct":     jitter(params['autoimmune_pct'], -3, 3),
        "viral_suscept_pct":  jitter(params['viral_sus_pct'], -3, 3),
        "age_at_dx_mo":       round(params['age_dx_mo'] + rng.uniform(-0.5, 1.5), 1),
        "hsct_pct":           jitter(params['hsct_pct'], -3, 3),
        "survival_5yr_pct":   jitter(params['surv_5yr_pct'], -3, 3),
        "vod_pct":            jitter(params['vod_pct'], -1, 1),
    }


def generate_overview():
    genes   = ['JAK3', 'IL7RA', 'IL2RA', 'ZAP70', 'PTPRC', 'IL2RB', 'DOCK2', 'SP110']
    seeds   = list(range(4054, 4062))
    cohorts = {g: _cohort(g, s) for g, s in zip(genes, seeds)}

    return {
        "atlas":     "Hereditary-JAK3-IL7R-CD25-ZAP70-CD45-IL2RB-DOCK2-SP110-CID-Variant-Atlas",
        "subtitle":  "Complete 8-Gene Reference — Cytokine-Receptor-Signaling and Proximal TCR Defects: JAK3-SCID, IL7R-SCID, CD25 Deficiency, ZAP70 Selective CD8-Absence, CD45 SCID, IL2RB Dysregulation, DOCK2 NK-Defect, SP110/VODI",
        "genes":     genes,
        "cohorts":   cohorts,
        "total_patients": sum(c['n'] for c in cohorts.values()),
        "seeds":     seeds,
        "key_facts": {
            "iuis_category":          "IUIS 2022 Category I: Combined Immunodeficiencies and Category II: CID with Associated Features",
            "jak3_autosomal_xscid":   "JAK3: T-B+NK- SCID in BOTH SEXES — autosomal equivalent of IL2RG X-SCID; γc downstream; pSTAT5 after IL-2 stimulation absent; BCG absolutely CI",
            "il7ra_tnk_plus":         "IL7RA: T-B+NK+ SCID (NK PRESENT) — key distinction from JAK3/IL2RG (T-B+NK-); IL-7 required for T but NOT NK (IL-15Rβγ is JAK3/JAK1, not IL7RA)",
            "zap70_cd8_absent":       "ZAP70: SELECTIVE CD8 ABSENCE PATHOGNOMONIC — CD8 < 50/μL + CD4 NORMAL/ELEVATED; only PID with this flow pattern; autoimmunity in CD4 cells despite normal count",
            "cd45_pan_absent":        "PTPRC/CD45: CD45 ABSENT on ALL lymphocytes by flow — simple diagnostic test (anti-CD45 negative on PBMC); T-B-NK- SCID without myeloid aplasia",
            "il2rb_nk_hlh":           "IL2RB/CD122: NK ABSENT + T cell LYMPHOPROLIFERATION + HLH-like (2019 discovery); IL-15 requires CD122 for NK survival; autoimmunity prominent",
            "dock2_nk_function":      "DOCK2: NK cells PRESENT but FUNCTION ABSENT (CD107a degranulation assay negative); severe CMV/HSV/EBV; RAC GEF for actin in lymphocytes",
            "sp110_vodi":             "SP110/VODI: Spontaneous Hepatic VOD (without HSCT) + Combined Immunodeficiency — neonatal/infant VOD without prior HSCT is PATHOGNOMONIC for VODI; defibrotide + HSCT",
        },
        "pathway_overview": "γc-JAK3 AXIS (JAK3): cytokine receptor common gamma chain → JAK3 → STAT5 → T/NK development. IL-7Rα AXIS (IL7RA): IL-7Rα-γc → T cell thymic development only (NK intact). IL-2Rα AXIS (IL2RA/CD25): high-affinity IL-2 → Treg survival → loss = lymphoproliferation + autoimmunity. ZAP70 AXIS (ZAP70): TCR ITAM → ZAP70 → LAT/SLP76 → T activation; CD8 threshold higher → ZAP70 LOF = CD8 absent only. CD45 AXIS (PTPRC): LCK Tyr505 dephosphorylation → LCK activation → TCR/BCR signal; CD45 absent = all lymphocytes frozen. IL-2Rβ AXIS (IL2RB): IL-2Rβγ-JAK1/JAK3 → NK (IL-15) + memory T + Treg; CD122 absent = NK gone + dysregulation. RAC/ACTIN AXIS (DOCK2): DOCK2-RAC → lymphocyte migration/degranulation; NK degranulation absent. NUCLEAR BODY / HEPATIC AXIS (SP110): nuclear body + hepatic venule endothelial regulation → VODI.",
    }


def generate_breakdown():
    genes   = ['JAK3', 'IL7RA', 'IL2RA', 'ZAP70', 'PTPRC', 'IL2RB', 'DOCK2', 'SP110']
    seeds   = list(range(4054, 4062))
    cohorts = {g: _cohort(g, s) for g, s in zip(genes, seeds)}

    diagnostic_groups = {
        "gamma_common_chain_downstream_scid": {
            "genes":          ["JAK3"],
            "mechanism":      "γc-CHAIN DOWNSTREAM KINASE DEFICIENCY: JAK3 is constitutively bound to γc (IL2RG) and transduces signals from all γc-containing cytokine receptors (IL-2R, IL-4R, IL-7R, IL-9R, IL-15R, IL-21R). JAK3 LOF = loss of ALL these cytokine signals simultaneously. Result: T cell development fails at pro-T → pre-T thymic stage; NK development fails (IL-15 absent); B cells develop normally but are non-functional (IL-4/IL-21 signaling absent = no class switch recombination without T cell help).",
            "shared_feature": "T-B+NK- SCID pattern identical to IL2RG X-SCID but autosomal (BOTH SEXES). Absolute lymphopenia: T cells < 300/μL, NK cells < 50/μL, B cells variable. pSTAT5 ABSENT after IL-2, IL-7, IL-15 stimulation (all γc-using cytokines). TREC screening positive (NBS).",
            "key_ddx":        "JAK3 vs IL2RG (X-SCID): IDENTICAL PHENOTYPE — distinguish by sex (both sexes in JAK3 vs male-predominant in IL2RG); molecular testing confirms. JAK3 vs IL7RA: IL7RA has NK PRESENT (T-B+NK+); JAK3 has NK absent (T-B+NK-). JAK3 vs ADA: ADA = T-B-NK- SCID + skeletal dysplasia + ADA enzyme absent on RBC; JAK3 = T-B+NK- (B cells present). BCG ABSOLUTELY CI in JAK3 deficiency (same as IL2RG).",
        },
        "il7_specific_t_scid": {
            "genes":          ["IL7RA"],
            "mechanism":      "IL-7-SPECIFIC T CELL DEVELOPMENT FAILURE: IL7Rα (CD127) pairs with γc to form the IL-7 receptor. IL-7 signaling is uniquely required for T cell development at the CD44-CD25+ double-negative stage in the thymus and for naive T cell peripheral survival. IL-15Rβγ (not IL7Rα) drives NK development → NK cells INTACT in IL7RA deficiency. B cells develop independently of IL-7 in humans → B cells INTACT.",
            "shared_feature": "T-B+NK+ SCID: T cells absent + B cells normal + NK cells NORMAL. This flow pattern is diagnostic for IL7RA (or RAG1/2 for T-B-NK+). TREC absent on NBS. pSTAT5 absent after IL-7 stimulation ONLY (not after IL-2 or IL-15 — those use JAK1/JAK3 via non-IL7RA receptors, intact in IL7RA deficiency).",
            "key_ddx":        "IL7RA vs JAK3: NK PRESENT (T-B+NK+) in IL7RA; NK ABSENT (T-B+NK-) in JAK3 — ONE FLOW TEST distinguishes. IL7RA vs RAG1/RAG2: B cells ABSENT in RAG deficiency (T-B-NK+); IL7RA has B cells PRESENT. IL7RA vs ADA: ADA = T-B-NK- with adenosine deaminase absent on RBC. Functional: pSTAT5 after IL-7 absent (IL7RA-specific); pSTAT5 after IL-2/IL-15 intact (JAK1/JAK3 axis OK).",
        },
        "treg_deficiency_dysregulation": {
            "genes":          ["IL2RA", "IL2RB"],
            "mechanism":      "IL-2 RECEPTOR SIGNALING DEFECTS — TREG FAILURE SPECTRUM: IL2RA (CD25/IL-2Rα): high-affinity IL-2 receptor on Tregs; IL-2Rα LOF → Tregs cannot receive sufficient IL-2 signal → Treg depletion/dysfunction → conventional T cells expand unchecked → autoimmunity + lymphoproliferation. IL2RB (CD122/IL-2Rβ): shared β-chain for IL-2R AND IL-15R; IL-2Rβ LOF → NK cells absent (IL-15-dependent) + Tregs partially dysfunctional + T cell homeostasis disrupted → T cell expansion + autoimmunity.",
            "shared_feature": "BOTH: lymphadenopathy + splenomegaly + autoimmunity (AIHA, IBD, thyroiditis). BOTH: paradox of infections (immunodeficiency component) despite T cell EXPANSION. BOTH: elevated LDH, ferritin (inflammatory markers). IL2RA: CD25 absent on surface of activated T cells (flow); Tregs (CD25+FOXP3+) reduced. IL2RB: NK absent (CD56-CD16- absent) + T cell expansion; CD122 absent on T/NK cells (flow).",
            "key_ddx":        "IL2RA vs IPEX (FOXP3): IPEX males only (X-linked); IL2RA both sexes. IPEX: neonatal enteropathy + T1DM from birth; IL2RA: slightly later onset. Both: CD25+FOXP3+ Tregs absent/dysfunctional. KEY: in IPEX, CD25 IS present on T cells (FOXP3 absent); in IL2RA, CD25 ABSENT on activated T cells. IL2RB vs HLH: IL2RB HLH-like presentation + NK absent; FHL genes (PRF1/UNC13D): NK degranulation CD107a absent but NK PRESENT; IL2RB: NK absent numerically. IL2RB vs JAK3: JAK3 is T-B+NK-; IL2RB is NK-absent with T cell EXPANSION (not T cell absence).",
        },
        "selective_cd8_absence_zap70": {
            "genes":          ["ZAP70"],
            "mechanism":      "ZAP70-SELECTIVE CD8 THYMIC DELETION: ZAP70 kinase is activated by CD4-LCK or CD8-LCK after TCR/ITAM engagement. CD8+ T cells require a HIGHER ZAP70 signal threshold for thymic positive selection (CD8-LCK → stronger ZAP70 phospho-activation). ZAP70 partial LOF: CD8+ thymocytes fail positive selection (insufficient ZAP70 signal) → CD8 cells deleted. CD4+ T cells (lower threshold via CD4-LCK) survive positive selection with residual ZAP70 activity. The result: CD8 T cells absent, CD4 T cells PRESENT but auto-reactive (negative selection also impaired).",
            "shared_feature": "ZAP70 ALONE in the 8-gene atlas for this pattern. SELECTIVE CD8 ABSENCE: CD8 < 50/μL + CD4 normal or elevated (CD4/CD8 ratio > 10:1 or infinite). The ONLY PID with this flow pattern — ALL other SCID cause CD4+CD8 both low. ZAP70 protein absent by intracellular flow or Western blot. pZAP70 (phospho-ZAP70) after CD3 cross-linking absent.",
            "key_ddx":        "ZAP70 UNIQUE FLOW PATTERN: No other PID causes isolated CD8 absence. If CD8 absent + CD4 normal → FIRST TEST: ZAP70 protein intracellular flow. Exclude MHC class I deficiency (TAP1/TAP2/TAPBP — bare lymphocyte syndrome II): MHC class I absent on surface; in ZAP70, MHC class I PRESENT. Autoimmunity in ZAP70: CD4 cells are auto-reactive → AIHA, IBD, arthritis despite normal CD4 count — consider ZAP70 when autoimmunity + CD8 lymphopenia occurs together.",
        },
        "total_lymphocyte_phosphatase_scid": {
            "genes":          ["PTPRC"],
            "mechanism":      "CD45 PAN-LYMPHOCYTE PHOSPHATASE DEFICIENCY: CD45 (PTPRC) dephosphorylates inhibitory Tyr505 on LCK, activating LCK. Without CD45: LCK permanently inhibited → TCR signal transduction completely blocked in T cells → thymic selection fails → T cells absent. BCR signaling similarly impaired (CD45 required for B cell activation too). NK cells require CD45 for activating receptor signaling. ALL THREE LYMPHOCYTE LINEAGES affected simultaneously.",
            "shared_feature": "T-B-NK- SCID with NEUTROPHILS PRESENT (distinguishes from AK2 reticular dysgenesis where myeloid also absent). CD45 ABSENT on ALL lymphocytes: anti-CD45-FITC flow shows 0% CD45+ in PBMC (vs >95% in normal). This is the SIMPLEST DIAGNOSTIC FLOW TEST for PTPRC/CD45 deficiency.",
            "key_ddx":        "PTPRC vs AK2 (Reticular Dysgenesis): AK2 = neutrophils ABSENT + total aplasia + SNHL; PTPRC = neutrophils PRESENT (myeloid intact), CD45 absent on lymphocytes only. PTPRC vs JAK3: JAK3 = T-B+NK- (B cells present, CD45 present); PTPRC = T-B-NK- (B cells absent, CD45 absent). PTPRC vs ADA: ADA = T-B-NK- + skeletal dysplasia + RBC ADA enzyme absent; PTPRC = T-B-NK-, CD45 absent on flow is diagnostic; ADA enzyme normal in PTPRC.",
        },
        "nk_dysfunction_viral": {
            "genes":          ["DOCK2", "SP110"],
            "mechanism":      "NK CELL DYSFUNCTION AND SPECIALIZED IMMUNODEFICIENCY: DOCK2 LOF → RAC GEF absent → NK cells cannot reorganize actin for degranulation → NK degranulation absent despite normal NK numbers. SP110 LOF → nuclear body disruption + hepatic venule endothelial dysfunction → VODI: spontaneous hepatic veno-occlusive disease + combined immunodeficiency.",
            "shared_feature": "DOCK2: NK CELLS PRESENT but CD107a degranulation ABSENT (functional NK defect, not quantitative); viral susceptibility dominant (CMV, HSV, RSV, EBV). SP110/VODI: NK cells absent/reduced + hypogammaglobulinemia + HEPATIC VOD (spontaneous, not HSCT-induced).",
            "key_ddx":        "DOCK2 vs LYST (Chediak-Higashi): BOTH have NK degranulation absent (CD107a negative); LYST: giant granules on blood smear PATHOGNOMONIC; DOCK2: NO giant granules on blood smear — key distinction. DOCK2 vs HLH (PRF1/UNC13D): HLH has high ferritin + hemophagocytosis + HLH-2004 criteria; DOCK2: VMV susceptibility WITHOUT primary HLH. SP110 vs HSCT-VOD: SP110 VOD occurs in infants WITHOUT prior HSCT — this is DIAGNOSTIC; HSCT-VOD = after allogeneic HSCT. SP110 vs biliary atresia: SP110 has immunodeficiency (low IgG) + lymphopenia absent in biliary atresia.",
        },
    }

    return {
        "atlas":             "Hereditary-JAK3-IL7R-CD25-ZAP70-CD45-IL2RB-DOCK2-SP110-CID-Variant-Atlas",
        "cohorts":           cohorts,
        "total_patients":    sum(c['n'] for c in cohorts.values()),
        "diagnostic_groups": diagnostic_groups,
        "gene_summary": {
            g: {
                "gene":               g,
                "n":                  cohorts[g]["n"],
                "seed":               cohorts[g]["seed"],
                "t_lymphopenia_pct":  cohorts[g]["t_lymphopenia_pct"],
                "nk_absent_pct":      cohorts[g]["nk_absent_pct"],
                "b_present_pct":      cohorts[g]["b_present_pct"],
                "autoimmune_pct":     cohorts[g]["autoimmune_pct"],
                "hsct_pct":           cohorts[g]["hsct_pct"],
                "survival_5yr_pct":   cohorts[g]["survival_5yr_pct"],
            } for g in genes
        },
    }


def generate_definitions():
    return {
        "definitions": [
            {"term": "JAK3 SCID — Diagnosis, BCG Danger, and HSCT", "definition": "JAK3 deficiency causes AR T-B+NK- SCID phenotypically identical to IL2RG X-SCID. DIAGNOSTIC PROTOCOL: (1) TREC-NBS positive (TREC absent on newborn screening) → lymphocyte subset flow: T cells absent + NK cells absent + B cells present = T-B+NK-; (2) Exclude IL2RG first in males; (3) pSTAT5 stimulation test: add IL-2 or IL-7 or IL-15 → pSTAT5 ABSENT in JAK3 (all γc-cytokines use JAK3); (4) JAK3 gene sequencing. BCG ABSOLUTELY CONTRAINDICATED: BCG vaccine in all γc-axis SCID (IL2RG, JAK3) causes disseminated BCGosis (systemic mycobacterial infection) — FATAL if not treated with prolonged anti-mycobacterial therapy (at least 3 drugs including rifampicin + isoniazid + ethambutol for minimum 12 months); in countries with routine BCG at birth, genetic diagnosis before discharge is essential — TREC screening before BCG administration. HSCT: allogenic HSCT curative; RIC conditioning well-tolerated in T-B+NK- SCID; sibling matched preferred; matched unrelated or haplo-HSCT alternatives; gene therapy (lentiviral JAK3 correction) in clinical trials. BRIDGE THERAPY: PCP prophylaxis (TMP-SMX) + antifungal prophylaxis + IVIG until HSCT."},
            {"term": "IL7RA SCID — T-B+NK+ Pattern and Molecular Distinction", "definition": "IL7RA deficiency produces T-B+NK+ SCID where T cells are absent but B and NK cells are NORMAL. FLOW INTERPRETATION: this pattern narrows the differential to (1) IL7RA deficiency, (2) RAG1 LOF hypomorph, (3) ARTEMIS hypomorph, or (4) ADA deficiency if metabolic component present. KEY DISTINGUISHER FROM JAK3/IL2RG: NK CELLS PRESENT in IL7RA (T-B+NK+) vs NK ABSENT in JAK3/IL2RG (T-B+NK-). FUNCTIONAL TEST: pSTAT5 response to IL-7 stimulation — ABSENT in IL7RA deficiency; pSTAT5 to IL-2 and IL-15 INTACT (JAK1/JAK3 axis, does not require IL7Rα). PROTEIN EXPRESSION: IL7Rα (CD127) absent on any residual T cells or thymic cells by flow cytometry. MOLECULAR: IL7RA gene sequencing; 5p13.2 locus; common founder alleles vary by population. HSCT OUTCOME: excellent post-HSCT (similar to other T-B+NK+ SCID types); NK-containing graft preferred (JAK3 gene therapy uses lentiviral vector, IL7RA gene therapy similarly studied). PROGNOSIS: with early HSCT before major infections, 80-90% long-term survival."},
            {"term": "IL2RA/CD25 Deficiency — Treg Absence and Paradoxical Lymphoproliferation", "definition": "IL2RA/CD25 deficiency is a PARADOXICAL immunodeficiency where lymphocyte EXPANSION mimics a primary lymphoproliferative disorder but infections occur simultaneously. CLINICAL MIMICRY: presents like lymphoma (lymphadenopathy + organomegaly) but in an infant with recurrent CMV — this combination should trigger CD25/IL2RA testing. FLOW CYTOMETRY DIAGNOSIS: (1) CD25 (IL-2Rα) surface expression on ACTIVATED T cells — after stimulation with CD3/CD28 or PHA for 48h, normal T cells strongly upregulate CD25; in IL2RA deficiency, CD25 ABSENT on activated T cells despite activation (distinguish from IPEX where CD25 IS induced but FOXP3 absent); (2) Treg count: CD4+CD25+FOXP3+ — reduced or absent. TREATMENT: (1) Rapamycin (sirolimus) inhibits mTOR downstream of IL-2R → slows T cell proliferation and reduces lymphoproliferation; (2) IVIG for infections; (3) Ganciclovir/valganciclovir for CMV; (4) HSCT — curative but complicated by lymphoproliferation pre-HSCT (reduce lymphoproliferation with rapamycin + rituximab before conditioning); results in immune reconstitution with restored Treg function."},
            {"term": "ZAP70 Deficiency — Selective CD8 Absence and Diagnostic Trap", "definition": "ZAP70 deficiency is the ONLY PID causing isolated CD8+ T cell absence while CD4+ T cells remain normal/elevated — and is FREQUENTLY MISSED because standard lymphocyte count appears 'normal' (total lymphocytes normal or even elevated due to CD4 expansion). DIAGNOSTIC TRAP: standard CBC lymphopenia absent → ZAP70 missed on initial CBC. MANDATORY FLOW CYTOMETRY: if T cell subset is not checked, ZAP70 deficiency goes undetected for years. FLOW: CD8 < 50/μL (often 0) + CD4 normal or >1000/μL. CD3 total may appear falsely low due to absent CD8 or normal due to CD4 compensation. PROTEIN TEST: intracellular ZAP70 staining (flow cytometry or Western blot in T cells) — ZAP70 protein absent in homozygous LOF. FUNCTIONAL: anti-CD3-stimulated pZAP70 (phospho-Y315/Y319 ZAP70) absent. AUTOIMMUNE COMPONENT: ZAP70 deficiency frequently presents with AIHA (direct antiglobulin test positive), IBD, arthritis — these autoimmune features may be the presenting complaint, not infections. HSCT: curative; restores CD8 counts and eliminates autoimmunity; perform before irreversible autoimmune damage (especially colitis/liver disease)."},
            {"term": "CD45/PTPRC Deficiency — Pan-Lymphocyte Marker Absent and T-B-NK- SCID", "definition": "CD45/PTPRC deficiency causes T-B-NK- SCID and is diagnosed by a UNIQUE FLOW CYTOMETRY FINDING: CD45 (LCA/leukocyte common antigen) is absent on ALL lymphocytes. DIAGNOSTIC TEST: add anti-CD45-FITC antibody to peripheral blood — in normal patients >95% of PBMC are CD45+ (including T, B, NK, monocytes); in PTPRC deficiency, ALL lymphocytes are CD45-NEGATIVE. This single test is PATHOGNOMONIC. COMPARE TO AK2 (RETICULAR DYSGENESIS): AK2 also causes near-total leukocyte absence but includes MYELOID aplasia (neutropenia) and SNHL; CD45/PTPRC deficiency has NORMAL NEUTROPHIL COUNTS — myeloid compartment intact. MOLECULAR: PTPRC gene (also called CD45, LCA); 1q31.3; sequencing confirms biallelic variants. TREATMENT: HSCT urgency similar to other T-B-NK- SCID (not as immediately fatal as AK2 since myeloid immunity partially intact but still urgent); BCG absolutely CI; RIC or myeloablative conditioning; excellent post-HSCT reconstitution documented."},
            {"term": "IL2RB/CD122 Deficiency — NK Absence + HLH-Like Dysregulation (2019 Discovery)", "definition": "IL2RB/CD122 deficiency was first described in 2019 and causes a complex dysregulation combining NK absence with T cell expansion — clinically resembling HLH. DIAGNOSTIC CLUE: infant with HLH-2004 criteria (fever + splenomegaly + cytopenias + hyperferritinemia + hemophagocytosis) + NK CELLS NUMERICALLY ABSENT on flow cytometry → NK-absent HLH → FHL genes FIRST (PRF1, UNC13D, STX11, STXBP2) — if NK absent and HLH present but FHL genes negative → IL2RB sequencing. FLOW: NK cells (CD56+CD16+) absent; T cells expanded (CD4 lymphocytosis); CD122 absent on T cells and NK cells. FUNCTIONAL: pSTAT5 after IL-15 absent (IL-15 uses IL-2Rβγ); pSTAT5 after IL-2 also absent (IL-2 trimeric = IL-2Rαβγ, IL-2Rβ absent). TREATMENT: (1) Ruxolitinib or baricitinib (JAK inhibitor) for cytokine storm/HLH control as bridge; (2) HSCT curative; (3) Anti-IL-6 (tocilizumab) for cytokine storm component; monitoring: ferritin, LDH, triglycerides, fibrinogen (HLH biomarkers)."},
            {"term": "DOCK2 Deficiency — NK Degranulation Absent (CD107a) and Viral Triad", "definition": "DOCK2 deficiency causes profound viral susceptibility with NK cells PRESENT but DYSFUNCTIONAL — clinically distinct from conditions causing NK absence. PATHOGNOMONIC VIRAL PATTERN: severe CMV (disseminated, retinitis, colitis) + severe HSV (skin/visceral/CNS) + RSV bronchiolitis requiring ICU — this CMV+HSV+RSV triad in an infant with normal NK count should trigger DOCK2 testing. NK DEGRANULATION TEST (CD107a ASSAY): K562 cells (NK-activating targets) + patient PBMC → assess CD107a surface upregulation on NK cells by flow = NK degranulation marker. In DOCK2: CD107a remains INTRACELLULAR (cannot be transported to surface without RAC-dependent actin reorganization) → CD107a assay NEGATIVE despite NK cells numerically present. COMPARE TO LYST (CHS): LYST also has CD107a negative degranulation but GIANT GRANULES on blood smear (PATHOGNOMONIC for LYST); DOCK2 = normal granules. MOLECULAR: DOCK2 gene (5q35.1); 1862aa; large gene; sequencing required; no common founder variants (panel/WES). HSCT: curative for NK function and viral susceptibility; valganciclovir pre-HSCT for CMV; acyclovir for HSV."},
            {"term": "SP110/VODI — Spontaneous Hepatic VOD in Infancy and Combined Immunodeficiency", "definition": "VODI (Veno-Occlusive Disease with Immunodeficiency) from SP110 mutations is recognized by SPONTANEOUS HEPATIC VOD IN AN INFANT — i.e., veno-occlusive disease occurring WITHOUT prior allogeneic HSCT, WITHOUT prior intensive chemotherapy, WITHOUT Budd-Chiari (no hepatic vein obstruction). HEPATIC VOD DIAGNOSIS: (1) clinical: right upper quadrant pain + rapid weight gain + jaundice + hepatomegaly in infant; (2) Doppler ultrasound: hepatomegaly + ascites + reversed/absent hepatic venous flow + no hepatic vein obstruction (distinguishes from Budd-Chiari where hepatic veins are occluded, not sinusoids); (3) liver biopsy (if safe): centrilobular fibrosis + sinusoidal obliteration + endothelial cell damage; (4) immunology: hypogammaglobulinemia (low IgG < 3.0 g/L) + CD4 lymphopenia + NK reduced. DRUZE POPULATION: founder mutation c.1048_1049ins11 (p.Cys350Profs*15) prevalent in Druze families — consider SP110 early in Druze/Lebanese infant with hepatic VOD. TREATMENT: (1) Defibrotide (endothelium-protective dextran sulfate): FDA/EMA approved for HSCT-VOD, used off-label for VODI; adult dose 6.25 mg/kg q6h IV; (2) Ursodeoxycholic acid; (3) HSCT curative for immunodeficiency and may stabilize hepatic disease; liver transplantation if HSCT not feasible and liver failure predominates; (4) IVIG replacement; TMP-SMX prophylaxis."},
            {"term": "Cascade Testing — JAK3-IL7RA-ZAP70-CD45-IL2RA-IL2RB-DOCK2-SP110 Differential", "definition": "STEP 1: NBS TREC POSITIVE (TREC absent) → Confirm lymphocyte subset flow: T-B-NK- = CD45 absent? → PTPRC (CD45 surface flow ABSENT = PATHOGNOMONIC). If CD45 PRESENT: T-B+NK- → JAK3 (exclude IL2RG in males first). T-B+NK+ → IL7RA (or RAG1/RAG2). T-B-NK+ = RAG defects. STEP 2: ABSENT NK + HLH-like + T expansion → FHL genes first (PRF1, UNC13D, STX11, STXBP2); if FHL negative → IL2RB/CD122 sequencing. STEP 3: CD8 T cells ABSENT + CD4 NORMAL/elevated + CD45 present → ZAP70 protein intracellular flow → ZAP70 sequencing. STEP 4: Lymphoproliferation + CMV + IBD-like in infant with CD25 ABSENT on activated T cells → IL2RA sequencing. STEP 5: NK PRESENT numerically + CD107a degranulation ABSENT (K562 assay) + severe CMV/HSV/RSV → DOCK2 sequencing (check blood smear for giant granules first to exclude LYST/CHS). STEP 6: Infant hepatic VOD (without prior HSCT) + low IgG + lymphopenia → SP110/VODI; Doppler US + liver biopsy + defibrotide bridge → HSCT evaluation. BCG ABSOLUTELY CI: all T-B+NK- types (JAK3, IL2RG). IL7RA, ZAP70, PTPRC: BCG also CI. DOCK2, SP110: BCG avoidance recommended."},
        ],
        "key_distinguishers": {
            "jak3_vs_il7ra":          "JAK3: T-B+NK- (NK ABSENT); IL7RA: T-B+NK+ (NK PRESENT). ONE flow test distinguishes these two autosomal SCID types.",
            "zap70_unique":           "ZAP70: CD8 ABSENT + CD4 NORMAL/ELEVATED — the ONLY PID with this pattern. All other SCID = CD4+CD8 both absent.",
            "cd45_simple_test":       "PTPRC/CD45: CD45 ABSENT on ALL lymphocytes by flow (anti-CD45-FITC 0% positive) — simplest pathognomonic test in immunology.",
            "dock2_vs_lyst":          "DOCK2 vs LYST (CHS): BOTH NK degranulation absent (CD107a negative); LYST: GIANT GRANULES on blood smear PATHOGNOMONIC; DOCK2: normal granules.",
            "sp110_spontaneous_vod":  "SP110/VODI: hepatic VOD WITHOUT prior HSCT in an infant = PATHOGNOMONIC; HSCT-VOD occurs after transplant; Budd-Chiari = hepatic vein obstruction (different anatomy).",
            "il2ra_vs_ipex":          "IL2RA vs IPEX (FOXP3): IPEX males only (X-linked); IL2RA both sexes. KEY: CD25 ABSENT on activated T cells in IL2RA; CD25 PRESENT (induced) but FOXP3 absent in IPEX.",
            "il2rb_2019_discovery":   "IL2RB/CD122: 2019 discovery; NK absent + T cell expansion + HLH-like; differs from FHL (NK present in FHL) — NK absent is the clue.",
        },
        "treatment_matrix": {
            "hsct_urgency":    "PTPRC T-B-NK- = urgent (similar to other T-B-NK- SCID). JAK3 T-B+NK- = urgent. IL7RA = urgent. ZAP70 = less urgent (some CD4 function) but before autoimmune damage. IL2RA = semi-urgent (lymphoproliferative complications). IL2RB = urgent (HLH-like). DOCK2 = semi-urgent. SP110 = urgent (hepatic failure risk).",
            "bcg_protocol":    "BCG ABSOLUTELY CI: JAK3, IL7RA, PTPRC, ZAP70 (T lymphopenia). DOCK2, SP110, IL2RA, IL2RB: BCG avoidance recommended. If BCG given before diagnosis: 3-drug anti-TB therapy mandatory.",
            "specific_rx":     "IL2RA + IL2RB: rapamycin for lymphoproliferation. IL2RB: ruxolitinib for HLH control. SP110: defibrotide for VOD. DOCK2: valganciclovir + acyclovir for CMV/HSV. ALL: PCP prophylaxis (TMP-SMX) + IVIG if hypogammaglobulinemia + antifungal.",
        },
    }
