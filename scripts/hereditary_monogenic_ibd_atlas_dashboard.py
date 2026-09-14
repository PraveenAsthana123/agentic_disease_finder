#!/usr/bin/env python3
"""Hereditary-Monogenic-IBD-Atlas — Complete 8-Gene Very Early Onset IBD & Monogenic IBD Atlas
(IL10 · IL10RA · IL10RB · XIAP · LRBA · CYBB · CYBA · NCF2).

IL10     (Interleukin-10; 178 aa; 19 kDa; 1q32.1; AR;
          IL-10 deficiency; most common infant-onset (< 3 months) monogenic IBD;
          pan-colitis + perianal disease; curative HSCT MANDATORY — fatal without;
          IL-10 is the master anti-inflammatory cytokine of the gut;
          seed SEED_BASE+0).
IL10RA   (IL-10 receptor alpha subunit; 576 aa; 66 kDa; 11q23.3; AR;
          IL-10 receptor alpha deficiency; identical phenotype to IL10 deficiency;
          infant-onset pan-colitis + perianal abscesses/fistulae;
          multiple ethnic founders — Chinese (p.Cys171Tyr), Turkish, Middle Eastern;
          curative HSCT MANDATORY;
          seed SEED_BASE+1).
IL10RB   (IL-10 receptor beta subunit; 325 aa; 37 kDa; 21q22.11; AR;
          IL-10Rb is shared receptor for IL-10, IFN-λ (IFN-lambda1/2/3), IL-22, IL-26, IL-28/29;
          biallelic loss → COMBINED IL-10 + IFN-λ pathway defect;
          HLH risk HIGHER than IL10RA — IFN-λ antiviral loss + IL-10 gut loss;
          neonatal-onset; curative HSCT MANDATORY;
          seed SEED_BASE+2).
XIAP     (X-linked inhibitor of apoptosis / BIRC4; 497 aa; 57 kDa; Xq25; XLR;
          XLP-2 (X-linked lymphoproliferative disease type 2);
          Crohn-like IBD + HLH (haemophagocytic lymphohistiocytosis);
          MALES ONLY clinically; mothers are obligate carriers (typically healthy);
          HLH episodes recur — HSCT curative for HLH and IBD;
          splenomegaly + cytopenias during HLH;
          seed SEED_BASE+3).
LRBA     (LPS-responsive beige-like anchor protein; 2863 aa; 319 kDa; 4q31.3; AR;
          LRBA deficiency = CVID + autoimmunity + Crohn-like IBD + lymphoproliferation;
          LRBA maintains CTLA4 expression on T-reg surface (endosomal recycling);
          ABATACEPT (CTLA4-Ig) highly effective — mechanism-targeted;
          hypogammaglobulinaemia → recurrent sinopulmonary infections;
          autoimmune cytopenias (AIHA, ITP) pathognomonic;
          wider onset 1-8 years; not infant-onset like IL10 pathway;
          seed SEED_BASE+4).
CYBB     (Cytochrome b-245 heavy chain / gp91phox; 570 aa; 91 kDa; Xp21.1; XLR;
          X-linked chronic granulomatous disease (X-CGD); most common CGD form (65%);
          NADPH oxidase defect → failure to generate reactive oxygen species → cannot kill
          catalase-positive organisms (Aspergillus, Staphylococcus, Burkholderia, Serratia);
          GRANULOMATOUS COLITIS clinically mimics Crohn's;
          NBT (nitroblue tetrazolium) test / DHR flow cytometry DIAGNOSTIC before WES;
          antifungal prophylaxis (itraconazole) MANDATORY lifelong;
          MALES ONLY clinically (X-linked);
          seed SEED_BASE+5).
CYBA     (Cytochrome b-245 light chain / p22phox; 195 aa; 22 kDa; 16q24.2; AR;
          AR-CGD (autosomal recessive); phenotype IDENTICAL to X-CGD;
          p22phox stabilises both gp91phox (CYBB) and other NOX isoforms;
          both sexes affected equally;
          Granulomatous colitis + antifungal prophylaxis mandatory;
          NBT/DHR assay diagnostic (same as CYBB);
          seed SEED_BASE+6).
NCF2     (Neutrophil cytosol factor 2 / p67phox; 526 aa; 67 kDa; 1q25.3; AR;
          AR-CGD variant (p67phox); ~5% of CGD cases;
          PERIANAL DISEASE especially prominent — perianal abscesses + fistulae;
          same granulomatous colitis + systemic granulomata as other CGD;
          NBT/DHR shows abolished oxidative burst (same pattern as X-CGD/CYBA);
          antifungal prophylaxis mandatory; BMT curative option for severe CGD;
          seed SEED_BASE+7).
320-patient aggregate cohort (8 × 40, seeds 2558-2565).
"""

import random

SEED_BASE = 2558

MIBD_GENES = [
    # -- IL10 -- IL-10 deficiency (infant-onset pan-colitis) -----------------------------------
    {
        "gene": "IL10",
        "alt_name": (
            "IL10 (IL10-178aa-1q32.1 / AR -- "
            "INFANT-ONSET-<3-MONTHS-PAN-COLITIS-PERIANAL-DISEASE -- "
            "CURATIVE-HSCT-MANDATORY-FATAL-WITHOUT -- "
            "MASTER-ANTI-INFLAMMATORY-CYTOKINE-GUT -- "
            "IL-10-DEFICIENCY-MOST-COMMON-MONOGENIC-IBD)"
        ),
        "protein": (
            "IL10 -- 1q32.1 AR -- IL10-178aa -- "
            "Interleukin-10-19kDa-Homodimer-Anti-Inflammatory-Cytokine -- "
            "JAK1-STAT3-Signaling-Macrophage-Monocyte-T-Cell-Suppression -- "
            "Master-Gut-Homeostasis-Cytokine-Mucosal-Tolerance -- "
            "OMIM-Gene-124092-Disease-VEO-IBD-612261"
        ),
        "locus": "1q32.1",
        "protein_size": "178 aa / 19 kDa",
        "inheritance": (
            "AR (autosomal recessive — biallelic null/hypomorphic); "
            "ONSET < 3 months of life (neonatal/early infant): "
            "severe watery/bloody diarrhoea, failure to thrive, perianal disease (abscesses, fistulae); "
            "pan-colitis universally; enterocolitis extends to small bowel; "
            "WITHOUT TREATMENT: relentless progression → perforation, sepsis, death; "
            "HSCT is the ONLY curative treatment — conventional IBD therapy (steroids, biologics, immunosuppressants) fails"
        ),
        "disease_category": (
            "IL-10 cytokine deficiency — most common cause of very early onset monogenic IBD; "
            "neonatal/infant-onset pan-colitis + perianal disease; "
            "curative HSCT mandatory (not optional); "
            "gene is IL-10 itself (not receptor) — no upstream rescue pathway possible; "
            "infliximab/vedolizumab bridge therapy used pre-HSCT to reduce gut inflammation"
        ),
        "disease_pathway": (
            "IL-10 is the dominant immunosuppressive cytokine of the intestinal mucosa, secreted by macrophages, "
            "monocytes, dendritic cells, T-regs, and epithelial cells. IL-10 signals via IL-10RA/IL-10RB heterodimer "
            "→ JAK1/TYK2 → STAT3 → anti-inflammatory gene programme (suppresses TNF, IL-12, IL-23, IL-1β). "
            "Complete IL-10 deficiency → uncontrolled mucosal macrophage activation → TNF/IL-12 cytokine storm "
            "→ transmural ulceration + cryptitis + perianal inflammation. "
            "Gut microbiota drive the inflammation (germfree IL-10-KO mice have delayed/attenuated disease). "
            "HSCT replaces the myeloid cell compartment (macrophages) → restores IL-10 secretion → colitis remission. "
            "Bridge therapies: granulocyte-CSF to expand neutrophils, infliximab for luminal control, "
            "parenteral nutrition for nutritional rehabilitation pre-HSCT."
        ),
        "pathognomonic": (
            "PAN-COLITIS ONSET < 3 MONTHS OF LIFE + PERIANAL ABSCESSES/FISTULAE: pathognomonic combination. "
            "Failure of all conventional IBD therapy (steroids, 5-ASA, azathioprine, biologics) to induce remission. "
            "Undetectable serum IL-10 by ELISA (when IL10 gene null — direct test). "
            "Complete histological normalisation after successful HSCT (distinguishes from Crohn's disease). "
            "Colonoscopy: continuous transmural ulceration + deep serpiginous ulcers from anus to caecum."
        ),
        "treatment": (
            "HAEMATOPOIETIC STEM CELL TRANSPLANT (HSCT): curative — matched sibling donor preferred; "
            "myeloablative conditioning required; best outcomes HSCT < 1 year of age; "
            "Bridge to HSCT: infliximab 5 mg/kg q8w (luminal), TPN/elemental formula (nutritional), "
            "granulocyte colony-stimulating factor for secondary infections; "
            "corticosteroids only short-term (bridge bridge); "
            "Post-HSCT: colitis remission in 70-85% by 1 year; "
            "Experimental: rIL-10 infusion (short-term benefit only, not curative); "
            "IL-10 gene therapy trials (AAV-IL10): phase 1 in progress; "
            "HSCT timing: every month of delay → worsening nutritional status and infection risk."
        ),
        "key_features": [
            "Onset < 3 months (neonatal colitis — EARLIEST age onset IBD red flag for monogenic cause)",
            "Perianal disease (abscesses + fistulae) — highly specific for IL10 pathway vs idiopathic IBD",
            "Pan-colitis — continuous from rectum to caecum, no skip lesions",
            "Failure of ALL conventional IBD therapies — infliximab temporarily helps, never remission",
            "Severe malnutrition / failure to thrive — feeding by nasogastric or parenteral route",
            "Recurrent bacterial infections (secondary immunodeficiency from malnutrition + gut barrier loss)",
            "Elevated acute-phase reactants (CRP, ESR, faecal calprotectin >3000 µg/g in acute phase)",
            "Undetectable serum IL-10 by ELISA if gene null variant (direct test)"
        ],
        "key_ddx": [
            "IL10RA deficiency — identical phenotype; molecular test needed to distinguish gene",
            "IL10RB deficiency — same + higher HLH risk due to IFN-λ/IL-22 loss",
            "XIAP deficiency — older onset, male only, HLH dominant feature",
            "Infantile-onset IBD due to epithelial defects (TTC7A, EPCAM, MYO5B) — different mechanism",
            "Necrotising enterocolitis (NEC) — in premature neonates, different context",
            "Hirschsprung's disease + enterocolitis — rectal biopsy for ganglion cells diagnostic"
        ],
        "onset_months_median": 2,
        "perianal_disease_pct": 80,
        "hlh_risk_pct": 5,
        "hsct_rate_pct": 90,
        "on_biologic_pct": 60,
        "cgd_pct": 0,
        "hypogammaglobulinaemia_pct": 0,
        "mortality_pre_hsct_pct": 15,
        "hospitalizations_per_year": 8,
        "nbs_indicated": False,
    },
    # -- IL10RA -- IL-10 receptor alpha deficiency --------------------------------------------
    {
        "gene": "IL10RA",
        "alt_name": (
            "IL10RA (IL10RA-576aa-11q23.3 / AR -- "
            "INFANT-ONSET-PAN-COLITIS-IDENTICAL-IL10-DEFICIENCY -- "
            "MULTIPLE-ETHNIC-FOUNDERS-CHINESE-TURKISH-MIDDLE-EAST -- "
            "p.Cys171Tyr-CHINESE-FOUNDER->60pct-CHINESE-VEOIBD -- "
            "CURATIVE-HSCT-MANDATORY)"
        ),
        "protein": (
            "IL10RA -- 11q23.3 AR -- IL10RA-576aa -- "
            "IL-10-Receptor-Alpha-66kDa-Ligand-Binding-Subunit -- "
            "IL-10-High-Affinity-Binding-Extracellular-D1-D2-Ig-Domains -- "
            "JAK1-Kinase-Association-Intracellular-Box-1-Box-2 -- "
            "OMIM-Gene-146933-Disease-VEO-IBD-613148"
        ),
        "locus": "11q23.3",
        "protein_size": "576 aa / 66 kDa",
        "inheritance": (
            "AR (biallelic null or hypomorphic); "
            "phenotype IDENTICAL to IL10 deficiency (IL-10 cannot signal without functional receptor); "
            "MULTIPLE ETHNIC FOUNDERS — p.Cys171Tyr (c.512G>A): CHINESE FOUNDER present in >60% of Chinese VEO-IBD IL10RA cases; "
            "p.Trp159Arg and other founders in Turkish, Middle Eastern, South Asian consanguineous families; "
            "onset < 3 months (neonatal/infant-onset): identical to IL10 — bloody diarrhoea + perianal disease; "
            "HSCT curative; serum IL-10 levels are ELEVATED (cytokine produced but cannot signal)"
        ),
        "disease_category": (
            "IL-10 receptor alpha deficiency — second most common cause of infant-onset monogenic IBD; "
            "IL-10 produced but cannot signal (receptor defect); "
            "clinically indistinguishable from IL10 deficiency; only molecular testing separates; "
            "curative HSCT mandatory — same as IL10; "
            "serum IL-10 HIGH (cytokine present, cannot signal) vs IL10 deficiency (IL-10 absent) — "
            "this one test distinguishes IL10 gene from IL10RA/RB deficiency clinically"
        ),
        "disease_pathway": (
            "IL-10 receptor is a tetrameric complex (2×IL10RA + 2×IL10RB). IL-10 binds with high affinity to the "
            "IL10RA extracellular domain (Ig-like D1-D2 folds) and recruits IL10RB to complete the signalling unit. "
            "IL10RA loss → IL-10 present in serum but unable to engage receptor → same gut macrophage "
            "dysregulation as IL10 deficiency. In consanguineous families with infant-onset IBD, "
            "IL10RA sequencing should PRECEDE WES (cheaper, targeted, founder alleles known). "
            "p.Cys171Tyr disrupts a disulphide bond in the D1 Ig domain → protein misfolding → "
            "failure of IL-10 binding (critical diagnostic clue: serum IL-10 ELEVATED vs IL10 deficiency where IL-10 ABSENT)."
        ),
        "pathognomonic": (
            "PAN-COLITIS < 3 MONTHS + PERIANAL DISEASE + ELEVATED SERUM IL-10: pathognomonic triad for IL10RA/IL10RB deficiency. "
            "Distinguishes from IL10 deficiency (where serum IL-10 is ABSENT). "
            "p.Cys171Tyr in Chinese ethnicity + infant-onset colitis: >60% probability of IL10RA deficiency. "
            "Functional IL-10 signalling assay (STAT3 phosphorylation after IL-10 stimulation): ABSENT response."
        ),
        "treatment": (
            "HSCT: curative — same as IL10 deficiency (myeloablative conditioning); "
            "pre-HSCT bridge: infliximab + TPN + elemental formula; "
            "corticosteroids (short-term bridge); "
            "tacrolimus has been used as bridge in refractory perianal disease; "
            "targeted sequencing IL10RA/IL10RB before WES in consanguineous families with infant-onset IBD; "
            "early HSCT (< 1 year of age) gives best outcomes — delay worsens nutritional and infectious morbidity."
        ),
        "key_features": [
            "Infant-onset pan-colitis IDENTICAL to IL10 deficiency",
            "p.Cys171Tyr Chinese founder: targeted sequencing first in Chinese infant-onset IBD",
            "Serum IL-10 ELEVATED (cytokine present, cannot signal) — distinguishes from IL10 deficiency",
            "STAT3 phosphorylation after IL-10 stimulation ABSENT: functional receptor assay diagnostic",
            "Perianal disease (abscesses + fistulae) prominent as in IL10 deficiency",
            "Failure of all conventional IBD therapy",
            "Elevated faecal calprotectin + CRP from day 1",
            "Consanguineous family history in Middle East, Turkish, South Asian families"
        ],
        "key_ddx": [
            "IL10 deficiency — serum IL-10 ABSENT (IL10 gene) vs ELEVATED (IL10RA/RB) — key distinction",
            "IL10RB deficiency — clinically identical; molecular testing needed",
            "Classical Crohn's disease in infancy (rare <6 months) — no monogenic cause found on panel",
            "Eosinophilic enterocolitis — eosinophilia on biopsy, responds to exclusion diet"
        ],
        "onset_months_median": 2,
        "perianal_disease_pct": 80,
        "hlh_risk_pct": 5,
        "hsct_rate_pct": 88,
        "on_biologic_pct": 65,
        "cgd_pct": 0,
        "hypogammaglobulinaemia_pct": 0,
        "mortality_pre_hsct_pct": 12,
        "hospitalizations_per_year": 7,
        "nbs_indicated": False,
    },
    # -- IL10RB -- IL-10 receptor beta deficiency (+ IFN-lambda + IL-22 loss) -----------------
    {
        "gene": "IL10RB",
        "alt_name": (
            "IL10RB (IL10RB-325aa-21q22.11 / AR -- "
            "COMBINED-IL10-IFN-LAMBDA-IL22-LOSS-HLH-RISK-HIGHER -- "
            "NEONATAL-ONSET-SHARED-RECEPTOR-IL10-IFNL1-IFNL2-IFNL3-IL22 -- "
            "HLH-RISK-HIGHER-THAN-IL10RA -- "
            "CURATIVE-HSCT-MANDATORY)"
        ),
        "protein": (
            "IL10RB -- 21q22.11 AR -- IL10RB-325aa -- "
            "IL-10-Receptor-Beta-37kDa-Signal-Transducing-Subunit -- "
            "Shared-Receptor-IL10-IFN-Lambda1-2-3-IL22-IL26-IL28A-IL28B-IL29 -- "
            "TYK2-Kinase-Association-Intracellular-Domain -- "
            "OMIM-Gene-123889-Disease-VEO-IBD-613148"
        ),
        "locus": "21q22.11",
        "protein_size": "325 aa / 37 kDa",
        "inheritance": (
            "AR (biallelic null); "
            "IL10RB is the SHARED signal-transducing subunit for IL-10, IFN-λ1/λ2/λ3 (interferons lambda), IL-22, IL-26, IL-28, IL-29; "
            "biallelic loss → COMBINED IL-10 DEFICIENCY + IFN-λ ANTIVIRAL DEFECT + IL-22 BARRIER DEFECT; "
            "HLH RISK HIGHER THAN IL10RA — loss of IFN-λ antiviral innate immunity → viral HLH episodes; "
            "neonatal onset (often earlier than IL10RA); "
            "IL-22 pathway loss → reduced intestinal barrier function (additional to IL-10 defect)"
        ),
        "disease_category": (
            "IL-10 receptor beta deficiency — combined multi-cytokine signalling failure (IL-10 + IFN-λ + IL-22); "
            "neonatal-onset pan-colitis IDENTICAL to IL10/IL10RA + higher HLH risk; "
            "most severe of the three IL-10 pathway genes; "
            "HSCT mandatory and URGENT; "
            "serum IL-10 ELEVATED (same as IL10RA — cytokine present, shared receptor defect)"
        ),
        "disease_pathway": (
            "IL10RB (also named IL10R2) is the obligate shared signalling subunit recruited by multiple cytokine "
            "receptor complexes: IL-10/IL10RA/IL10RB, IFN-λ1-3/IFNLR1/IL10RB, IL-22/IL-22RA1/IL10RB, "
            "IL-26/IL-20RA/IL10RB. IL10RB loss → BROAD cytokine signalling failure. "
            "IFN-λ provides the first line of antiviral defence at mucosal surfaces (gut epithelium, respiratory) — "
            "IL10RB deficiency → viral susceptibility + viral-triggered HLH episodes. "
            "IL-22 maintains mucosal barrier integrity (AMPs, MUC expression) — loss compounds gut leakiness. "
            "This multilayer defect explains the earlier onset and higher severity vs isolated IL10/IL10RA deficiency."
        ),
        "pathognomonic": (
            "NEONATAL PAN-COLITIS + HLH EPISODE + ELEVATED SERUM IL-10: pathognomonic triad for IL10RB deficiency. "
            "IFN-λ signalling assay (STAT1/STAT2 phosphorylation after IFN-λ3 stimulation): ABSENT. "
            "IL-22 signalling assay: ABSENT (shared receptor). "
            "Clinically indistinguishable from IL10RA until HLH episode occurs."
        ),
        "treatment": (
            "URGENT HSCT — most severe IL-10 pathway gene; any HLH episode adds mortality risk; "
            "HLH treatment: dexamethasone + etoposide (HLH-2004 protocol) before HSCT conditioning; "
            "same bridge therapy as IL10/IL10RA (infliximab, TPN, elemental); "
            "viral surveillance mandatory pre-HSCT (EBV, CMV, ADV serology monthly); "
            "antiviral prophylaxis (acyclovir) given IFN-λ antiviral loss; "
            "avoid live-attenuated vaccines pre-HSCT (rotavirus, MMR)."
        ),
        "key_features": [
            "Neonatal-onset (often earliest of IL-10 pathway genes) — first weeks of life",
            "Higher HLH risk than IL10RA — IFN-λ antiviral pathway lost",
            "IFN-λ signalling ABSENT: viral infections (EBV, CMV, ADV) trigger HLH",
            "IL-22 signalling ABSENT: additional mucosal barrier defect vs IL10RA",
            "Serum IL-10 ELEVATED (shared receptor defect — same distinction from IL10 gene)",
            "STAT2 phosphorylation absent after IFN-λ stimulation (functional assay)",
            "Most urgent HSCT timing — HLH episode severely worsens outcome",
            "Avoid live-attenuated vaccines (rotavirus at birth — often given before diagnosis)"
        ],
        "key_ddx": [
            "IL10RA deficiency — no IFN-λ loss; no HLH risk from viral infections; molecular test",
            "IL10 deficiency — serum IL-10 ABSENT (vs elevated in IL10RB); no IFN-λ defect",
            "XLP-1 (SH2D1A) — EBV-triggered HLH + IBD; male only; different pathway",
            "XIAP (XLP-2) — male only; Crohn-like IBD; HLH triggered by non-viral stimuli also"
        ],
        "onset_months_median": 1,
        "perianal_disease_pct": 75,
        "hlh_risk_pct": 35,
        "hsct_rate_pct": 92,
        "on_biologic_pct": 55,
        "cgd_pct": 0,
        "hypogammaglobulinaemia_pct": 0,
        "mortality_pre_hsct_pct": 20,
        "hospitalizations_per_year": 9,
        "nbs_indicated": False,
    },
    # -- XIAP (BIRC4) -- XLP-2 (X-linked Crohn-like + HLH) ------------------------------------
    {
        "gene": "XIAP",
        "alt_name": (
            "XIAP (XIAP-497aa-Xq25 / XLR -- "
            "XLP2-CROHN-LIKE-IBD-PLUS-HLH-MALE-ONLY -- "
            "HSCT-CURATIVE-HLH-AND-IBD -- "
            "SPLENOMEGALY-CYTOPENIAS-HLH-EPISODES -- "
            "BIRC4-IAP-PROTEIN-APOPTOSIS-NOD2-SIGNALLING)"
        ),
        "protein": (
            "XIAP -- Xq25 XLR -- XIAP-497aa -- "
            "X-Linked-Inhibitor-of-Apoptosis-57kDa-BIR1-BIR2-BIR3-RING-Domains -- "
            "Caspase-3-7-9-Inhibitor-IAP-E3-Ubiquitin-Ligase -- "
            "NOD2-Signalling-Augmentation-via-RIP2-Ubiquitination -- "
            "OMIM-Gene-300079-Disease-XLP2-300635"
        ),
        "locus": "Xq25",
        "protein_size": "497 aa / 57 kDa",
        "inheritance": (
            "X-linked recessive; "
            "MALES ONLY clinically — hemizygous males affected; "
            "FEMALE CARRIERS: obligate carriers typically healthy (random X-inactivation); "
            "de novo mutations occur; "
            "BIRC4 (BIR-domain protein) encodes XIAP — inhibits caspases 3, 7, 9; "
            "also augments NOD2 inflammatory signalling via RIP2 ubiquitination; "
            "PARADOX: anti-apoptotic protein loss → hyperinflammation (counterintuitive mechanism)"
        ),
        "disease_category": (
            "XLP-2 (X-linked lymphoproliferative disease type 2) — XIAP deficiency; "
            "Crohn-like IBD (ileitis + colitis + perianal disease in males); "
            "HLH episodes (haemophagocytic lymphohistiocytosis) — recurrent, life-threatening; "
            "splenomegaly + cytopenias during HLH; "
            "HSCT curative for HLH and IBD component; "
            "distinction from XLP-1 (SH2D1A): XIAP has IBD as prominent feature, XLP-1 does not"
        ),
        "disease_pathway": (
            "XIAP is a caspase inhibitor (BIR1/BIR2 → caspase-3/7; BIR3 → caspase-9; RING → E3 ubiquitin ligase). "
            "XIAP loss → excess apoptosis in lymphocytes and macrophages. Paradoxically, "
            "XIAP ALSO promotes NOD2-RIPK2 signalling → loss → REDUCED NOD2-mediated bacterial sensing → "
            "Crohn-like gut inflammation (similar mechanism to NOD2 LOF in IBD susceptibility). "
            "HLH: XIAP regulates NK-cell-mediated cytotoxicity and cytokine storm; "
            "loss → uncontrolled macrophage activation → haemophagocytosis. "
            "X-linkage explains exclusive male clinical penetrance. "
            "HSCT replaces the haematopoietic compartment → normal XIAP expression → resolution."
        ),
        "pathognomonic": (
            "MALE + CROHN-LIKE ILEOCOLITIS + RECURRENT HLH: pathognomonic triad for XLP-2 / XIAP deficiency. "
            "Splenomegaly + hepatomegaly + fever + cytopenias during HLH episode. "
            "Perianal disease (abscesses, fistulae) in boys with recurrent HLH. "
            "XIAP protein level by intracellular flow cytometry: ABSENT in peripheral blood T cells."
        ),
        "treatment": (
            "HSCT: curative for both HLH and IBD — performed after HLH remission; "
            "HLH treatment: HLH-2004 protocol (dexamethasone + etoposide ± cyclosporin); "
            "IBD management pre-HSCT: infliximab/adalimumab (TNF inhibitors), azathioprine; "
            "Sirolimus (mTOR inhibitor): used as bridge immunosuppressant pre-HSCT; "
            "anti-TNF therapy controls Crohn-like IBD but does NOT prevent HLH recurrence; "
            "EBV serology monitoring (EBV can trigger HLH in XLP); "
            "splenectomy CONTRAINDICATED — worsens thrombocytopenic risk; "
            "IVIG for hypogammaglobulinaemia episodes."
        ),
        "key_features": [
            "MALES ONLY — X-linked; carrier females test healthy",
            "Crohn-like ileocolitis + perianal disease — responds partially to anti-TNF",
            "HLH episodes (recurrent) — fever + splenomegaly + cytopenias + ferritin > 500 µg/L",
            "XIAP protein absent by intracellular flow cytometry — diagnostic test",
            "NOD2 signalling defect (XIAP augments RIP2 ubiquitination) — Crohn-like mechanism",
            "EBV seroconversion can trigger HLH (same as XLP-1/SH2D1A)",
            "Splenomegaly prominent during HLH episodes",
            "HSCT: curative for both HLH and IBD — do not delay after HLH remission"
        ],
        "key_ddx": [
            "XLP-1 (SH2D1A) — same X-linked HLH; NO significant IBD; SAP protein absent",
            "IL10 pathway deficiencies — both sexes; no HLH typically; infant onset",
            "Crohn's disease — polygenic; no HLH; no X-linked family history",
            "CVID + IBD — hypogammaglobulinaemia major feature; both sexes"
        ],
        "onset_months_median": 18,
        "perianal_disease_pct": 55,
        "hlh_risk_pct": 70,
        "hsct_rate_pct": 75,
        "on_biologic_pct": 70,
        "cgd_pct": 0,
        "hypogammaglobulinaemia_pct": 15,
        "mortality_pre_hsct_pct": 10,
        "hospitalizations_per_year": 5,
        "nbs_indicated": False,
    },
    # -- LRBA -- LRBA deficiency (CVID + Crohn-like + autoimmunity) ---------------------------
    {
        "gene": "LRBA",
        "alt_name": (
            "LRBA (LRBA-2863aa-4q31.3 / AR -- "
            "CVID-CROHN-LIKE-IBD-AUTOIMMUNE-CYTOPENIAS -- "
            "ABATACEPT-HIGHLY-EFFECTIVE-MECHANISM-TARGETED -- "
            "LRBA-MAINTAINS-CTLA4-T-REG-SURFACE-ENDOSOMAL-RECYCLING -- "
            "LARGEST-MONOGENIC-IBD-GENE-319kDa)"
        ),
        "protein": (
            "LRBA -- 4q31.3 AR -- LRBA-2863aa -- "
            "LPS-Responsive-Beige-Like-Anchor-Protein-319kDa-BEACH-WD-Domain -- "
            "Endosomal-Recycling-Regulator-CTLA4-Recycling-to-T-Reg-Surface -- "
            "TLR4-Signal-Enhancer-Lysosome-Biogenesis-Factor -- "
            "OMIM-Gene-606453-Disease-CVID8-614700"
        ),
        "locus": "4q31.3",
        "protein_size": "2863 aa / 319 kDa",
        "inheritance": (
            "AR (autosomal recessive — biallelic loss of function); "
            "ONSET wider (1-8 years) — not infant-onset like IL10 pathway; "
            "LRBA encodes the largest protein among monogenic IBD genes (319 kDa); "
            "LRBA maintains CTLA4 expression on T-reg surface via endosomal recycling: "
            "LRBA loss → CTLA4 degraded in lysosomes → reduced T-reg suppression → "
            "autoimmune lymphoproliferation + Crohn-like IBD; "
            "hypogammaglobulinaemia (CVID-like) from B-cell dysregulation"
        ),
        "disease_category": (
            "LRBA deficiency — immune dysregulation + CVID + Crohn-like IBD + autoimmune cytopenias + lymphoproliferation; "
            "ABATACEPT (CTLA4-Ig fusion protein) highly effective — directly compensates for T-reg CTLA4 loss; "
            "NOT infant-onset (wider 1-8 years); "
            "hypogammaglobulinaemia → recurrent sinopulmonary infections; "
            "autoimmune haemolytic anaemia (AIHA) + ITP pathognomonic; "
            "Crohn-like IBD responds to abatacept (mechanism-targeted therapy)"
        ),
        "disease_pathway": (
            "LRBA (BEACH/WD-domain protein) acts as an endosomal scaffold that recycles CTLA4 from "
            "intracellular vesicles back to the T-reg cell surface. CTLA4 on T-regs is a critical "
            "immune checkpoint — engages CD80/CD86 on antigen-presenting cells, downregulating T-cell activation. "
            "LRBA loss → CTLA4 diverted to lysosomes for degradation → T-regs cannot suppress effector T cells "
            "→ autoimmune lymphoproliferation + gut inflammation. "
            "Hypogammaglobulinaemia results from B-cell dysregulation (B-cell LRBA also recycles BAFFR and TLR9). "
            "ABATACEPT (CTLA4-Ig) exogenously replaces the lost CTLA4 checkpoint → dramatic immune-regulatory restoration. "
            "HSCT is also curative (second-line after abatacept failure)."
        ),
        "pathognomonic": (
            "CROHN-LIKE IBD + AIHA/ITP + HYPOGAMMAGLOBULINAEMIA + LYMPHADENOPATHY: pathognomonic tetrad for LRBA deficiency. "
            "CTLA4 expression reduced on T-regulatory cells (flow cytometry: CD25+FoxP3+ T-regs with low surface CTLA4). "
            "Dramatic response to ABATACEPT (within weeks): distinguishes LRBA deficiency from idiopathic Crohn's disease. "
            "Splenomegaly + hepatomegaly from lymphoproliferation."
        ),
        "treatment": (
            "ABATACEPT (CTLA4-Ig, 10 mg/kg IV q4 weeks): highly effective — mechanism-targeted; "
            "IVIG (0.4-0.6 g/kg q3-4 weeks): for hypogammaglobulinaemia — reduces infections; "
            "IBD management: abatacept ± infliximab/vedolizumab bridge; "
            "HLH (if occurs): dexamethasone + etoposide; "
            "HSCT: curative (after abatacept failure or severe lymphoproliferation); "
            "Sirolimus (mTOR inhibitor): effective for lymphoproliferation component; "
            "Rituximab: for AIHA/ITP refractory to IVIG; "
            "Prophylactic antibiotics (co-trimoxazole) + antifungal (fluconazole) for immunocompromised state."
        ),
        "key_features": [
            "Wider onset (1-8 years) — NOT infant-onset like IL10 pathway; diagnosed later",
            "AUTOIMMUNE CYTOPENIAS (AIHA + ITP) pathognomonic — check Coombs, platelet antibodies",
            "Hypogammaglobulinaemia — low IgG/IgA/IgM (CVID-like) + recurrent lung infections",
            "Crohn-like IBD — ileocolitis with granulomata (similar to Crohn's histology)",
            "Lymphadenopathy + splenomegaly from lymphoproliferation",
            "CTLA4 LOW on T-regulatory cells — flow cytometry diagnostic clue",
            "DRAMATIC response to abatacept — within 2-4 weeks (pathognomonic treatment response)",
            "LARGEST monogenic IBD gene (2863 aa) — WES essential; targeted panels miss it"
        ],
        "key_ddx": [
            "CTLA4 haploinsufficiency (CTLA4 gene mutation) — same phenotype; AD; abatacept also effective",
            "Common variable immunodeficiency (CVID) — no IBD typically; no LRBA mutation found",
            "Crohn's disease — no hypogammaglobulinaemia; no autoimmune cytopenias; no LRBA mutation",
            "SLE — ANA positive; renal involvement; AIHA + thrombocytopenia overlap but different"
        ],
        "onset_months_median": 36,
        "perianal_disease_pct": 25,
        "hlh_risk_pct": 20,
        "hsct_rate_pct": 35,
        "on_biologic_pct": 75,
        "cgd_pct": 0,
        "hypogammaglobulinaemia_pct": 80,
        "mortality_pre_hsct_pct": 5,
        "hospitalizations_per_year": 4,
        "nbs_indicated": False,
    },
    # -- CYBB -- X-CGD (gp91phox) — X-linked chronic granulomatous disease --------------------
    {
        "gene": "CYBB",
        "alt_name": (
            "CYBB (CYBB-570aa-Xp21.1 / XLR -- "
            "X-CGD-MOST-COMMON-65pct-CGD-GRANULOMATOUS-COLITIS -- "
            "NBT-DHR-FLOW-CYTOMETRY-DIAGNOSTIC-BEFORE-WES -- "
            "ANTIFUNGAL-PROPHYLAXIS-ITRACONAZOLE-MANDATORY-LIFELONG -- "
            "ASPERGILLUS-LUNG-ABSCESS-MOST-DANGEROUS)"
        ),
        "protein": (
            "CYBB -- Xp21.1 XLR -- CYBB-570aa -- "
            "gp91phox-Cytochrome-b-245-Heavy-Chain-91kDa-NOX2-Membrane-Subunit -- "
            "NADPH-Oxidase-Catalytic-Subunit-Flavocytochrome-b558 -- "
            "6-TM-Helix-FAD-Binding-NADPH-Binding-Domain -- "
            "OMIM-Gene-300481-Disease-X-CGD-306400"
        ),
        "locus": "Xp21.1",
        "protein_size": "570 aa / 91 kDa",
        "inheritance": (
            "X-linked recessive; "
            "MALES ONLY clinically; "
            "FEMALE CARRIERS: typically well (mosaic oxidase activity ~50%); "
            "rarely, lyonisation can result in symptomatic female carriers; "
            "MOST COMMON CGD form — 65% of all CGD worldwide; "
            "del/dup Xp21.1 can co-delete CYBB + DMD + NROB1 (McLeod syndrome contiguous)"
        ),
        "disease_category": (
            "X-linked CGD (CYBB) — NADPH oxidase defect → failure to generate reactive oxygen species (superoxide); "
            "cannot kill catalase-positive organisms (Aspergillus, Staphylococcus, Burkholderia, Serratia, Nocardia); "
            "GRANULOMATOUS COLITIS (Crohn-like): obstructive gut granulomata + bloody stool; "
            "antifungal prophylaxis (itraconazole) MANDATORY lifelong; "
            "NBT test + DHR flow cytometry DIAGNOSTIC (< 1% oxidative burst); "
            "steroid treatment of granulomata is effective but increases infection risk"
        ),
        "disease_pathway": (
            "NADPH oxidase complex (NOX2) = membrane-bound Flavocytochrome b558 (gp91phox/CYBB + p22phox/CYBA) "
            "plus cytosolic components (p47phox/NCF1, p67phox/NCF2, p40phox/NCF4, Rac2). "
            "On phagocyte activation, cytosolic subunits translocate to membrane → assembled NOX2 → "
            "electrons from NADPH → O2• (superoxide) → H2O2 → OCl- → kills pathogens. "
            "CYBB loss → NO superoxide → cannot kill catalase-positive organisms (these have their own H2O2 defence). "
            "Persistent intracellular bacteria → granuloma formation (macrophages wall off uncleared pathogens). "
            "GI granulomata are obstructive (pyloric/small bowel obstruction) + cause bloody colitis. "
            "Antifungal prophylaxis mandatory: Aspergillus fumigatus lung abscess is the #1 killer in X-CGD."
        ),
        "pathognomonic": (
            "ABSENT NBT REDUCTION (nitroblue tetrazolium test): < 1% yellow tetrazolium → blue formazan conversion — pathognomonic. "
            "DHR (dihydrorhodamine 123) flow cytometry: absent shift from non-fluorescent DHR123 to fluorescent rhodamine 123. "
            "GRANULOMATOUS COLITIS in a MALE + absent NBT/DHR: pathognomonic for X-CGD. "
            "Aspergillus lung abscess in a male child without HIV: think CGD."
        ),
        "treatment": (
            "ANTIFUNGAL PROPHYLAXIS (itraconazole 200 mg/day): MANDATORY lifelong — #1 priority; "
            "ANTIBIOTIC PROPHYLAXIS (co-trimoxazole daily): reduces bacterial infections; "
            "IFN-gamma SC (50 µg/m2 3×/week): reduces infection frequency by 70% in prophylaxis trials; "
            "GRANULOMATOUS COLITIS: prednisolone 1 mg/kg/day taper — effective but immunosuppressive; "
            "OBSTRUCTION: surgical resection ± steroids; "
            "HSCT: curative — indicated for severe/refractory CGD (rising threshold with good prophylaxis); "
            "Gene therapy (retroviral CYBB): Phase 2 trials positive (Glimm 2021); "
            "Avoid ibuprofen/NSAIDs (impair neutrophil oxidative burst transiently)."
        ),
        "key_features": [
            "NBT test ABSENT (< 1% reduction) — cheap, rapid, diagnostic — do BEFORE WES",
            "DHR flow cytometry ABSENT (< 1% rhodamine+ neutrophils) — gold standard functional test",
            "Aspergillus lung abscess — most dangerous complication; annual chest CT in endemic areas",
            "Granulomatous colitis mimicking Crohn's disease — biopsy shows non-caseating granulomata",
            "Obstructive granulomata (pyloric, small bowel) — may present as bowel obstruction",
            "Males only (X-linked); female carriers have mosaic oxidase activity (~50%)",
            "Antifungal prophylaxis (itraconazole) MANDATORY from diagnosis — never omit",
            "Co-trimoxazole prophylaxis (bacterial) mandatory — reduces infections"
        ],
        "key_ddx": [
            "CYBA (AR-CGD p22phox) — identical phenotype; both sexes; normal carrier females",
            "NCF1 (p47phox AR-CGD) — most common AR-CGD; identical phenotype",
            "NCF2 (p67phox AR-CGD) — rare; perianal disease prominent",
            "Crohn's disease — granulomata less obstructive; NBT/DHR normal; both sexes"
        ],
        "onset_months_median": 12,
        "perianal_disease_pct": 30,
        "hlh_risk_pct": 5,
        "hsct_rate_pct": 30,
        "on_biologic_pct": 20,
        "cgd_pct": 100,
        "hypogammaglobulinaemia_pct": 0,
        "mortality_pre_hsct_pct": 8,
        "hospitalizations_per_year": 3,
        "nbs_indicated": False,
    },
    # -- CYBA -- AR-CGD p22phox (identical phenotype, autosomal) ------------------------------
    {
        "gene": "CYBA",
        "alt_name": (
            "CYBA (CYBA-195aa-16q24.2 / AR -- "
            "AR-CGD-p22phox-IDENTICAL-XCGD-PHENOTYPE-BOTH-SEXES -- "
            "STABILISES-GP91PHOX-AND-NOX1-3-4-5 -- "
            "NBT-DHR-DIAGNOSTIC-SAME-AS-CYBB -- "
            "ANTIFUNGAL-PROPHYLAXIS-MANDATORY)"
        ),
        "protein": (
            "CYBA -- 16q24.2 AR -- CYBA-195aa -- "
            "p22phox-Cytochrome-b-245-Light-Chain-22kDa-NOX2-Membrane-Subunit -- "
            "Heterodimer-Partner-gp91phox-CYBB-Flavocytochrome-b558 -- "
            "Haem-Binding-Stabilises-NOX2-NOX1-NOX3-NOX4 -- "
            "OMIM-Gene-608508-Disease-AR-CGD-233690"
        ),
        "locus": "16q24.2",
        "protein_size": "195 aa / 22 kDa",
        "inheritance": (
            "AR (autosomal recessive — biallelic null); "
            "BOTH SEXES equally affected (unlike CYBB X-linked); "
            "p22phox stabilises gp91phox (CYBB) membrane expression AND NOX1/NOX3/NOX4 activity; "
            "CYBA biallelic loss → absent p22phox → gp91phox degraded → absent NOX2 complex; "
            "phenotype IDENTICAL to X-CGD — Aspergillus, Staphylococcus, Serratia, Burkholderia; "
            "NBT/DHR: absent oxidative burst (same as CYBB)"
        ),
        "disease_category": (
            "AR-CGD (CYBA) — p22phox deficiency; phenotype identical to X-CGD; "
            "both sexes affected equally; "
            "granulomatous colitis (Crohn-like) + lung abscess + lymphadenitis; "
            "antifungal prophylaxis (itraconazole) mandatory; "
            "CYBA also stabilises NOX1 (gut epithelium) and NOX4 (kidney, smooth muscle) — "
            "additional subtle phenotypic differences from pure CYBB deficiency (gut + vascular)"
        ),
        "disease_pathway": (
            "p22phox (CYBA) is the obligate membrane heterodimer partner of gp91phox (CYBB). "
            "Without p22phox, gp91phox protein is unstable and rapidly degraded — "
            "even CYBA mutations destroy both p22phox AND gp91phox (bidirectional stabilisation). "
            "The result is identical to CYBB deficiency: no Flavocytochrome b558 → no NOX2 → no superoxide. "
            "Additionally, CYBA stabilises NOX1 (gut epithelial barrier ROS), NOX3 (cochlear/vestibular), "
            "and NOX4 (kidney mesangial cells) — loss may contribute to mild barrier defects beyond NOX2. "
            "Treatment principles identical to X-CGD: prophylaxis + IFN-gamma + HSCT option."
        ),
        "pathognomonic": (
            "ABSENT NBT + ABSENT DHR + AFFECTED FEMALE: pathognomonic — X-CGD excluded (X-linked), points to AR-CGD. "
            "Granulomatous colitis in a FEMALE with absent DHR → CYBA, NCF1, NCF2, NCF4 differential. "
            "CYBA protein absent by Western blot (diagnostic alongside gp91phox absent together). "
            "Both parents obligate carriers with ~50% oxidative burst (DHR intermediate peak)."
        ),
        "treatment": (
            "Identical to CYBB (X-CGD): "
            "Itraconazole prophylaxis (antifungal — MANDATORY), co-trimoxazole (antibacterial — MANDATORY); "
            "IFN-gamma SC 3×/week; "
            "granulomatous colitis: prednisolone taper; "
            "HSCT: curative option for severe disease; "
            "Gene therapy: retroviral CYBA correction — trials ongoing; "
            "no additional specific therapy above CYBB standard."
        ),
        "key_features": [
            "BOTH SEXES — autosomal recessive (key difference from CYBB X-CGD in male-only)",
            "p22phox stabilises gp91phox AND NOX1/3/4 — mild additional epithelial/vascular effects",
            "NBT + DHR ABSENT — identical to X-CGD; impossible to distinguish clinically",
            "Parents (obligate carriers) show INTERMEDIATE DHR peak (~50% oxidative burst)",
            "Granulomatous colitis same as X-CGD — non-caseating granulomata on biopsy",
            "Aspergillus, Staphylococcus, Serratia, Burkholderia — same pathogen spectrum as CYBB",
            "Antifungal prophylaxis (itraconazole) MANDATORY from diagnosis",
            "Molecular testing (CYBA) essential to confirm — clinically identical to CYBB and NCF1"
        ],
        "key_ddx": [
            "CYBB (X-CGD) — male only; clinically identical; CYBA distinguished by autosomal inheritance",
            "NCF1 (p47phox AR-CGD) — most common AR-CGD; ~50% partial oxidative burst sometimes",
            "NCF2 (p67phox AR-CGD) — perianal disease prominent; rare",
            "Myeloperoxidase deficiency — mild, often asymptomatic; NBT normal (NOX2 intact)"
        ],
        "onset_months_median": 10,
        "perianal_disease_pct": 25,
        "hlh_risk_pct": 3,
        "hsct_rate_pct": 25,
        "on_biologic_pct": 15,
        "cgd_pct": 100,
        "hypogammaglobulinaemia_pct": 0,
        "mortality_pre_hsct_pct": 6,
        "hospitalizations_per_year": 3,
        "nbs_indicated": False,
    },
    # -- NCF2 -- AR-CGD p67phox (perianal disease prominent) ----------------------------------
    {
        "gene": "NCF2",
        "alt_name": (
            "NCF2 (NCF2-526aa-1q25.3 / AR -- "
            "AR-CGD-p67phox-5pct-CGD-PERIANAL-DISEASE-PROMINENT -- "
            "NBT-DHR-ABSENT-SAME-AS-CYBB-CYBA -- "
            "ANTIFUNGAL-MANDATORY-HSCT-CURATIVE -- "
            "RAREST-CLASSIC-CGD-GENE)"
        ),
        "protein": (
            "NCF2 -- 1q25.3 AR -- NCF2-526aa -- "
            "p67phox-Neutrophil-Cytosol-Factor-2-67kDa-Cytosolic-NOX2-Activator -- "
            "4-TPR-PB1-SH3-Domain-Activator-Domain-NOX2-Assembly -- "
            "Cytosolic-Complex-p47phox-p67phox-p40phox-Rac2-Translocates-Membrane -- "
            "OMIM-Gene-608515-Disease-AR-CGD-233710"
        ),
        "locus": "1q25.3",
        "protein_size": "526 aa / 67 kDa",
        "inheritance": (
            "AR (autosomal recessive — biallelic loss); "
            "~5% of all CGD cases (rare); "
            "p67phox (NCF2) is a cytosolic component of the NOX2 complex — provides the activation domain; "
            "NCF2 loss → cytosolic complex cannot activate membrane-bound Flavocytochrome b558 → absent superoxide; "
            "phenotype IDENTICAL to X-CGD/CYBA on NBT/DHR; "
            "PERIANAL DISEASE especially prominent vs other CGD genes — clinical distinction"
        ),
        "disease_category": (
            "AR-CGD (NCF2 / p67phox) — rarest classic CGD gene (~5%); "
            "phenotype: granulomatous colitis + PROMINENT PERIANAL DISEASE + systemic granulomata; "
            "BOTH sexes affected (autosomal); "
            "antifungal prophylaxis mandatory; "
            "HSCT curative; "
            "perianal disease (fistulae + abscesses) can be confused with Crohn's disease — check NBT/DHR"
        ),
        "disease_pathway": (
            "The cytosolic NOX2 activator complex (p47phox/NCF1 + p67phox/NCF2 + p40phox/NCF4 + Rac2) "
            "assembles in the cytoplasm on neutrophil activation signals. "
            "p67phox (NCF2) contains the activation domain (TPR repeats) that directly binds and activates "
            "the Flavocytochrome b558 (gp91phox/CYBB). "
            "NCF2 loss → cytosolic complex cannot dock/activate membrane NOX2 → absent superoxide despite "
            "normal gp91phox and p22phox membrane expression. "
            "NBT/DHR: absent (functionally identical to CYBB/CYBA deficiency on functional assay). "
            "Western blot: gp91phox and p22phox PRESENT (membrane component intact) — "
            "this distinguishes NCF2 from CYBB/CYBA deficiency where gp91phox is absent."
        ),
        "pathognomonic": (
            "ABSENT NBT/DHR + PRESENT gp91phox on Western blot: pathognomonic — distinguishes NCF2/NCF1 from CYBB/CYBA. "
            "PROMINENT PERIANAL DISEASE (abscesses + complex fistulae) in a child with absent DHR: highest NCF2 suspicion. "
            "Both sexes affected (vs CYBB X-linked males only). "
            "Anorectal granulomata on biopsy + absent oxidative burst."
        ),
        "treatment": (
            "Antifungal prophylaxis (itraconazole MANDATORY); "
            "co-trimoxazole prophylaxis (antibacterial mandatory); "
            "IFN-gamma SC 3×/week (same as other CGD); "
            "PERIANAL DISEASE: prednisolone taper + surgical drainage (perianal abscesses); "
            "seton placement for complex fistulae; "
            "HSCT: curative — lower threshold for NCF2 due to severe perianal disease; "
            "anti-TNF (infliximab) for perianal fistulae (short-term bridge — increases infection risk); "
            "Gene therapy: NCF2 retroviral correction — trials in development."
        ),
        "key_features": [
            "PERIANAL DISEASE especially prominent — abscesses + complex anorectal fistulae",
            "BOTH SEXES affected (autosomal recessive)",
            "NBT/DHR ABSENT — same as CYBB/CYBA; functional assay cannot distinguish NCF2 from X-CGD",
            "gp91phox PRESENT on Western blot — membrane component intact; cytosolic activator defect",
            "Anorectal granulomata on proctoscopy biopsy",
            "Systemic granulomata (lymph nodes, liver, lung) same as other CGD",
            "Rarest classic CGD gene (~5% of CGD cases) — WES essential for diagnosis",
            "Antifungal (itraconazole) + antibacterial (co-trimoxazole) prophylaxis MANDATORY"
        ],
        "key_ddx": [
            "CYBB (X-CGD) — male only; gp91phox ABSENT on Western blot (vs PRESENT in NCF2)",
            "NCF1 (p47phox) — most common AR-CGD; gp91phox PRESENT (same as NCF2); most common AR form",
            "CYBA (p22phox AR-CGD) — gp91phox ABSENT (vs PRESENT in NCF2); both sexes",
            "Crohn's disease — normal NBT/DHR; perianal disease similar but CGD excluded"
        ],
        "onset_months_median": 8,
        "perianal_disease_pct": 65,
        "hlh_risk_pct": 3,
        "hsct_rate_pct": 35,
        "on_biologic_pct": 25,
        "cgd_pct": 100,
        "hypogammaglobulinaemia_pct": 0,
        "mortality_pre_hsct_pct": 5,
        "hospitalizations_per_year": 4,
        "nbs_indicated": False,
    },
]


def _make_patients(gene_entry, n=40, seed=None):
    rng = random.Random(seed)
    patients = []
    for i in range(n):
        onset = max(0, int(rng.gauss(gene_entry["onset_months_median"], gene_entry["onset_months_median"] * 0.5 + 2)))
        fatal = rng.random() < gene_entry["mortality_pre_hsct_pct"] / 100
        patients.append({
            "patient_id": f"{gene_entry['gene']}-{seed}-{i:03d}",
            "gene": gene_entry["gene"],
            "onset_months": onset,
            "perianal_disease": rng.random() < gene_entry["perianal_disease_pct"] / 100,
            "hlh_episode": rng.random() < gene_entry["hlh_risk_pct"] / 100,
            "underwent_hsct": rng.random() < gene_entry["hsct_rate_pct"] / 100,
            "on_biologic": rng.random() < gene_entry["on_biologic_pct"] / 100,
            "cgd_confirmed": gene_entry["cgd_pct"] == 100,
            "hypogammaglobulinaemia": rng.random() < gene_entry["hypogammaglobulinaemia_pct"] / 100,
            "pre_hsct_fatal": fatal,
            "hospitalizations_per_year": max(0, round(rng.gauss(gene_entry["hospitalizations_per_year"], 2), 1)),
            "colonoscopy_score": round(rng.uniform(1.5, 3.5) if not fatal else rng.uniform(3.0, 4.0), 1),
            "outcome": "deceased" if fatal else rng.choice(["remission", "partial-remission", "active-disease"]),
        })
    return patients


def generate_overview():
    all_patients = []
    gene_summaries = []
    for idx, entry in enumerate(MIBD_GENES):
        pts = _make_patients(entry, n=40, seed=SEED_BASE + idx)
        all_patients.extend(pts)
        surviving = [p for p in pts if not p["pre_hsct_fatal"]]
        gene_summaries.append({
            "gene": entry["gene"],
            "locus": entry["locus"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"].split(";")[0].strip(),
            "disease_name": entry["disease_category"].split(" —")[0].strip()[:80],
            "pathognomonic_short": entry["pathognomonic"].split(". ")[0][:120],
            "onset_months_median": entry["onset_months_median"],
            "perianal_disease_pct": entry["perianal_disease_pct"],
            "hlh_risk_pct": entry["hlh_risk_pct"],
            "hsct_rate_pct": entry["hsct_rate_pct"],
            "on_biologic_pct": entry["on_biologic_pct"],
            "cgd_pct": entry["cgd_pct"],
            "hypogammaglobulinaemia_pct": entry["hypogammaglobulinaemia_pct"],
            "mortality_pre_hsct_pct": entry["mortality_pre_hsct_pct"],
            "avg_hospitalizations": round(sum(p["hospitalizations_per_year"] for p in pts) / 40, 1),
            "hsct_actual_pct": round(sum(1 for p in pts if p["underwent_hsct"]) / 40 * 100, 1),
        })

    agg = {
        "perianal_disease_pct": round(sum(1 for p in all_patients if p["perianal_disease"]) / 320 * 100, 1),
        "hlh_episode_pct": round(sum(1 for p in all_patients if p["hlh_episode"]) / 320 * 100, 1),
        "hsct_pct": round(sum(1 for p in all_patients if p["underwent_hsct"]) / 320 * 100, 1),
        "on_biologic_pct": round(sum(1 for p in all_patients if p["on_biologic"]) / 320 * 100, 1),
        "cgd_pct": round(sum(1 for p in all_patients if p["cgd_confirmed"]) / 320 * 100, 1),
        "hypogammaglobulinaemia_pct": round(sum(1 for p in all_patients if p["hypogammaglobulinaemia"]) / 320 * 100, 1),
        "mortality_pre_hsct_pct": round(sum(1 for p in all_patients if p["pre_hsct_fatal"]) / 320 * 100, 1),
        "avg_hospitalizations_per_year": round(sum(p["hospitalizations_per_year"] for p in all_patients) / 320, 1),
    }

    return {
        "title": "Hereditary Monogenic IBD Atlas",
        "subtitle": "Complete 8-Gene Very Early Onset & Monogenic IBD Reference",
        "genes": [e["gene"] for e in MIBD_GENES],
        "n_genes": 8,
        "total_patients": 320,
        "seeds": f"{SEED_BASE}–{SEED_BASE + 7}",
        "disease_classes": [
            "IL10 — IL-10 cytokine deficiency; infant-onset pan-colitis; curative HSCT mandatory",
            "IL10RA — IL-10 receptor alpha deficiency; identical to IL10; ethnic founders (p.Cys171Tyr Chinese)",
            "IL10RB — shared IL-10/IFN-λ/IL-22 receptor; combined pathway loss; higher HLH risk",
            "XIAP — XLP-2 (X-linked); Crohn-like IBD + recurrent HLH; males only; HSCT curative",
            "LRBA — CVID + Crohn-like IBD + autoimmune cytopenias; abatacept highly effective",
            "CYBB — X-CGD (gp91phox); granulomatous colitis; antifungal prophylaxis mandatory",
            "CYBA — AR-CGD (p22phox); identical to X-CGD; both sexes; same prophylaxis",
            "NCF2 — AR-CGD (p67phox); perianal disease prominent; rarest classic CGD gene",
        ],
        "gene_summary": gene_summaries,
        "aggregate_metrics": agg,
        "clinical_pearls": [
            "IL10 / IL10RA / IL10RB: HSCT is CURATIVE and MANDATORY — conventional IBD therapy (steroids, biologics, immunosuppressants) NEVER induces remission; delay in HSCT → irreversible nutritional and infectious morbidity.",
            "IL10 vs IL10RA/IL10RB distinction: serum IL-10 is ABSENT in IL10 deficiency (no cytokine produced) but ELEVATED in IL10RA/IL10RB (cytokine present but cannot signal) — this one cheap ELISA distinguishes the group.",
            "IL10RB has HIGHER HLH RISK than IL10RA — IFN-λ antiviral pathway also lost → viral-triggered HLH (EBV, CMV, ADV); antiviral surveillance mandatory.",
            "XIAP (XLP-2): X-LINKED — MALES ONLY; XIAP protein absent by intracellular flow cytometry (rapid diagnostic test); anti-TNF controls IBD but DOES NOT prevent HLH recurrence — HSCT needed for definitive treatment.",
            "LRBA: ABATACEPT (CTLA4-Ig) is mechanism-targeted therapy — LRBA maintains CTLA4 on T-reg surface; dramatic response within 2-4 weeks of abatacept; this treatment response is itself pathognomonic for LRBA/CTLA4 deficiency.",
            "CGD (CYBB/CYBA/NCF2): NBT test (cheap, rapid) and DHR flow cytometry DIAGNOSTIC — perform BEFORE WES in any child with granulomatous colitis or Aspergillus infection; antifungal prophylaxis (itraconazole) MANDATORY lifelong.",
            "CYBB vs CYBA/NCF2 distinction: gp91phox Western blot — ABSENT in CYBB/CYBA (membrane complex destroyed) but PRESENT in NCF1/NCF2 (cytosolic activator defect only); sex also separates (CYBB = male only).",
            "NCF2 (p67phox): PERIANAL DISEASE especially prominent — abscesses and complex anorectal fistulae mimicking Crohn's; NBT/DHR absent; gp91phox present on Western blot (key distinction from CYBB/CYBA).",
            "Age of onset is the STRONGEST clinical discriminator: < 3 months → IL10 pathway; 6-24 months → XIAP (male) or CGD; 1-8 years → LRBA; perianal disease without fever → NCF2 at any age.",
            "ALL infant-onset IBD (< 6 months) should have an urgent monogenic IBD panel (IL10/IL10RA/IL10RB + XIAP + LRBA + CGD genes) — these are all treatable with HSCT or targeted therapy and delay causes preventable mortality.",
        ],
    }


def generate_breakdown():
    breakdowns = []
    for idx, entry in enumerate(MIBD_GENES):
        pts = _make_patients(entry, n=40, seed=SEED_BASE + idx)
        breakdowns.append({
            "gene": entry["gene"],
            "locus": entry["locus"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"],
            "disease_category": entry["disease_category"],
            "disease_pathway": entry["disease_pathway"],
            "pathognomonic": entry["pathognomonic"],
            "treatment": entry["treatment"],
            "key_features": entry["key_features"],
            "key_ddx": entry["key_ddx"],
            "n_patients": 40,
            "perianal_disease_pct": round(sum(1 for p in pts if p["perianal_disease"]) / 40 * 100, 1),
            "hlh_episode_pct": round(sum(1 for p in pts if p["hlh_episode"]) / 40 * 100, 1),
            "hsct_pct": round(sum(1 for p in pts if p["underwent_hsct"]) / 40 * 100, 1),
            "on_biologic_pct": round(sum(1 for p in pts if p["on_biologic"]) / 40 * 100, 1),
            "hypogammaglobulinaemia_pct": round(sum(1 for p in pts if p["hypogammaglobulinaemia"]) / 40 * 100, 1),
            "mortality_pre_hsct_pct": round(sum(1 for p in pts if p["pre_hsct_fatal"]) / 40 * 100, 1),
            "avg_hospitalizations": round(sum(p["hospitalizations_per_year"] for p in pts) / 40, 1),
            "avg_onset_months": round(sum(p["onset_months"] for p in pts) / 40, 1),
            "avg_colonoscopy_score": round(sum(p["colonoscopy_score"] for p in pts) / 40, 2),
            "cgd_confirmed_pct": entry["cgd_pct"],
        })
    return {"gene_breakdowns": breakdowns}


def generate_definitions():
    return {
        "gene_entries": {
            entry["gene"]: {
                "gene": entry["gene"],
                "full_name": entry["alt_name"].split(" (")[0].strip(),
                "locus": entry["locus"],
                "protein_size": entry["protein_size"],
                "inheritance": entry["inheritance"].split(";")[0].strip(),
                "disease_name": entry["disease_category"],
                "disease_pathway": entry["disease_pathway"],
                "pathognomonic": entry["pathognomonic"],
                "treatment": entry["treatment"][:600],
                "key_features": entry["key_features"],
                "key_ddx": entry["key_ddx"],
                "perianal_disease_pct": entry["perianal_disease_pct"],
                "hlh_risk_pct": entry["hlh_risk_pct"],
                "hsct_rate_pct": entry["hsct_rate_pct"],
                "on_biologic_pct": entry["on_biologic_pct"],
                "cgd_pct": entry["cgd_pct"],
                "mortality_pre_hsct_pct": entry["mortality_pre_hsct_pct"],
                "nbs_indicated": entry["nbs_indicated"],
            }
            for entry in MIBD_GENES
        },
        "mibd_glossary": {
            "IL-10 Pathway Deficiencies (IL10 / IL10RA / IL10RB)": (
                "IL-10 (encoded by IL10) is the master mucosal anti-inflammatory cytokine, "
                "signalling via the IL10RA/IL10RB heterodimeric receptor complex → JAK1/TYK2 → STAT3 "
                "→ suppression of TNF, IL-12, IL-23, IL-1β in gut macrophages. "
                "Deficiency of any component (IL10, IL10RA, or IL10RB) → uncontrolled macrophage cytokine storm "
                "→ transmural ulcerative colitis from neonatal age. "
                "DIAGNOSIS: serum IL-10 ABSENT (IL10 gene) vs ELEVATED (IL10RA/IL10RB — cytokine present, cannot signal); "
                "STAT3 phosphorylation after IL-10 stimulation ABSENT (functional assay, gold standard). "
                "TREATMENT: HSCT is the ONLY curative option — replaces myeloid compartment → restores IL-10 production."
            ),
            "XIAP (XLP-2) — X-linked Crohn-like IBD + HLH": (
                "XIAP/BIRC4 encodes X-linked inhibitor of apoptosis (3 BIR domains + RING domain). "
                "XIAP inhibits caspases 3, 7, 9 AND augments NOD2-RIPK2 signalling (via RIP2 ubiquitination). "
                "Loss: excess apoptosis in lymphocytes/macrophages + reduced NOD2 bacterial sensing → Crohn-like gut inflammation. "
                "HLH (haemophagocytic lymphohistiocytosis): uncontrolled macrophage activation → "
                "haemophagocytosis of blood cells, ferritin > 500 µg/L, cytopenias, splenomegaly. "
                "KEY TEST: XIAP protein absent by intracellular flow cytometry (rapid bedside test). "
                "HSCT: curative for both HLH and IBD in XLP-2."
            ),
            "LRBA Deficiency — CTLA4 T-reg Checkpoint Failure": (
                "LRBA (LPS-responsive beige-like anchor; BEACH/WD domain protein) recycles CTLA4 from "
                "endosomes back to the T-reg surface (prevents lysosomal degradation). "
                "CTLA4 on T-regs engages CD80/CD86 on antigen-presenting cells → T-cell activation checkpoint. "
                "LRBA loss → CTLA4 degraded → T-regs cannot suppress effector T cells → "
                "autoimmune lymphoproliferation + Crohn-like IBD + AIHA + ITP + hypogammaglobulinaemia. "
                "ABATACEPT (CTLA4-Ig) directly replaces the lost CTLA4 checkpoint exogenously → "
                "dramatic clinical response (diagnostic + therapeutic). "
                "Identical mechanism to CTLA4 haploinsufficiency (CTLA4 gene AD mutation)."
            ),
            "NADPH Oxidase & Chronic Granulomatous Disease (CGD)": (
                "NOX2/NADPH oxidase = Flavocytochrome b558 [gp91phox (CYBB) + p22phox (CYBA)] in membrane "
                "+ cytosolic activator complex [p47phox (NCF1) + p67phox (NCF2) + p40phox (NCF4) + Rac2]. "
                "Activation: cytosolic complex translocates to membrane → assembled NOX2 → NADPH → O2•¯ (superoxide) "
                "→ H2O2 → HOCl → kills catalase-positive organisms. "
                "CGD = any component missing → no ROS → Aspergillus, Staphylococcus, Burkholderia survive intracellularly. "
                "Gut granulomata form (macrophages wall off uncleared pathogens) → GRANULOMATOUS COLITIS mimicking Crohn's. "
                "NBT TEST (nitroblue tetrazolium) / DHR FLOW CYTOMETRY → absent shift → diagnoses CGD before WES. "
                "gp91phox Western blot: ABSENT in CYBB/CYBA (membrane component destroyed); "
                "PRESENT in NCF1/NCF2 (cytosolic activator defect only) — key molecular distinction."
            ),
            "NBT Test and DHR Flow Cytometry — CGD Diagnosis": (
                "Nitroblue Tetrazolium (NBT) test: PMA-stimulated neutrophils reduce yellow NBT to insoluble "
                "blue formazan in normal cells (ROS-dependent). CGD neutrophils: 0-1% blue cells (vs >95% normal). "
                "Dihydrorhodamine 123 (DHR) flow cytometry: PMA-stimulated neutrophils convert non-fluorescent "
                "DHR123 to fluorescent rhodamine 123 in normal cells. CGD neutrophils: no shift (mean channel 0-5 vs >200 normal). "
                "CARRIER FEMALES (X-CGD): bimodal DHR peak (50% normal neutrophils from normal X-chromosome, "
                "50% CGD from mutant X) — pathognomonic for X-CGD carrier. "
                "NBT/DHR: cheap, rapid, diagnostic within 1 hour — perform BEFORE WES in any child with "
                "granulomatous colitis, Aspergillus infection, or lymphadenitis/liver abscess."
            ),
            "HSCT Timing and Outcomes in Monogenic IBD": (
                "IL10 pathway (IL10/IL10RA/IL10RB): HSCT < 1 year → best outcomes (90% durable remission); "
                "> 2 years → significantly worse (nutritional failure, gut fibrosis, infections). "
                "XIAP (XLP-2): HSCT after HLH remission; myeloablative conditioning preferred for IBD cure; "
                "RIC (reduced-intensity conditioning) curative for HLH but IBD may recur. "
                "LRBA: HSCT second-line after abatacept failure; good outcomes with full conditioning. "
                "CGD (CYBB/CYBA/NCF2): HSCT curative; threshold rising as prophylaxis improves; "
                "myeloablative preferred in children; RIC in older patients; gene therapy emerging (Phase 2). "
                "MATCHED SIBLING DONOR preferred; MATCHED UNRELATED DONOR acceptable (10/10 HLA); "
                "HAPLOIDENTICAL increasingly used in urgent cases when no matched donor available."
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
    print("\n=== DEFINITIONS (IL10) ===")
    defn = generate_definitions()
    print(json.dumps(defn["gene_entries"]["IL10"], indent=2)[:1500])
