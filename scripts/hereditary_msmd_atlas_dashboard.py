#!/usr/bin/env python3
"""Hereditary-MSMD-Atlas -- Complete 8-Gene Reference
Mendelian Susceptibility to Mycobacterial Disease / IFN-γ–IL-12 Axis Defects / Disseminated Mycobacterial Susceptibility
(IFNGR1/AR-Complete+AD-Partial-LOF-Most-Common-MSMD-Serum-IFNg-Elevated-PATHOGNOMONIC,
 IFNGR2/AR-Complete-LOF-IFNg-R2-Heterodimer-Partner-Disseminated-NTM,
 IL12B/AR-LOF-IL12p40-Shared-IL12-IL23-BCG-Salmonella-NTM,
 IL12RB1/AR-LOF-Most-Common-AR-MSMD-Worldwide-IL12-IL23-Impaired-Candida-Risk,
 STAT1-LOF/AD-LOF-HSE+MSMD-AR-LOF-Fatal-Viral-Distinct-from-GOF-CMC,
 IRF8/AD-LOF-mDC-Haploinsufficiency-AR-LOF-Total-DC-Absence-HSCT,
 ISG15/AR-LOF-IFNg-Amplification-Absent-Mycobacterial-CNS-IFN-Excess-Paradox,
 CYBB-Hypomorphic/XL-R-MSMD-Macrophage-Selective-Residual-10-20pct-No-Classic-CGD):

IFNGR1 (interferon gamma receptor 1; 472aa; 6q23.3; AD partial LOF + AR complete LOF;
         Mendelian Susceptibility to Mycobacterial Disease type 1 (MSMD1) — MOST COMMON MSMD GENE;
         IFNGR1 is the ligand-binding subunit of the heterodimeric IFN-γ receptor (IFNGR1/IFNGR2);
           IFNGR1 contains the extracellular IFN-γ binding domain, the transmembrane domain, and
           a short intracellular domain that recruits JAK1 and STAT1 → IFN-γ signalling to STAT1
           phosphorylation (pY701-STAT1) → STAT1 homodimerisation (GAF) → IRF1 transcription →
           macrophage killing programme (iNOS, reactive oxygen species, autophagy, lysosomal fusion);
         TWO DISTINCT MUTATION CLASSES:
           (1) AD PARTIAL LOF (dominant interfering): most common in Europeans; single-allele mutations
               producing a truncated or dysfunctional IFNGR1 that blocks surface trafficking of the
               WT chain (dominant negative); partial residual IFN-γ signalling preserved; recurrent
               NTM/BCG disease but survivable; T3 cell line test shows ~50% STAT1 activation; BCG
               vaccination ABSOLUTELY CONTRAINDICATED — disseminated BCGosis in all AD IFNGR1 patients
               who received BCG; responds partially to IFN-γ therapy;
           (2) AR COMPLETE LOF (recessive null): less common in Europeans, more common in consanguineous
               populations; complete absence of IFN-γ signalling; disseminated NTM/MTB/BCGosis fatal
               without HSCT; no response to exogenous IFN-γ (receptor absent); HSCT curative;
         SERUM IFN-γ ELEVATED: IFN-γ accumulates because the receptor is absent or non-functional →
           SERUM IFN-γ ELEVATION IS PATHOGNOMONIC OF AN IFNGR DEFECT; serum IFN-γ is NOT elevated
           in IL-12/IL-12R defects (IFN-γ production is impaired there, not the receptor);
         ORGANISMS: Mycobacterium avium complex (MAC), M. fortuitum, M. kansasii, M. abscessus,
           M. tuberculosis, M. bovis BCG (most common presenting pathogen in vaccinated patients),
           Salmonella spp. (in AR complete, less prominent than IL-12 defects), Leishmania, CMV
           (AR complete only — broader immunodeficiency);
         ANTI-MYCOBACTERIAL PROPHYLAXIS: clarithromycin/azithromycin + rifampin-based regimen;
           macrophage-targeted therapy required (intracellular pathogen);
         seed SEED_BASE+0).
IFNGR2 (interferon gamma receptor 2; 337aa; 21q22.11; AR complete LOF;
         Mendelian Susceptibility to Mycobacterial Disease type 2 (MSMD2);
         IFNGR2 (IFN-γR2, formerly accessory factor AF-1) is the signal-transducing subunit of the
           IFN-γ receptor heterodimer (IFNGR1/IFNGR2); IFNGR2 carries the intracellular JAK2 and
           STAT1 recruitment motifs; biallelic LOF → complete absence of IFN-γ signalling (identical
           biochemistry to AR-IFNGR1) → disseminated mycobacterial disease;
         CLINICAL PHENOTYPE: indistinguishable from AR-IFNGR1 complete LOF — disseminated NTM/MTB/BCGosis;
           fatal without treatment; SERUM IFN-γ ELEVATED (same diagnostic marker as IFNGR1) — IFN-γ
           accumulates because signal transduction absent even though ligand-binding (IFNGR1) is intact;
         T3 CELL LINE TEST: IFNGR1 surface expression NORMAL (intact ligand-binding chain); IFNGR2
           absent on surface by flow cytometry → distinguishes IFNGR2 from IFNGR1 defects;
         NO DOMINANT INTERFERING FORM: unlike IFNGR1 (which has AD partial LOF), all described
           IFNGR2 mutations are AR complete LOF → always recessive, both alleles must be affected;
         BCGOSIS: BCG vaccination ABSOLUTELY CONTRAINDICATED — immediately fatal disseminated BCGosis
           in IFNGR2-null patients; highest risk is in countries with universal BCG vaccination at birth;
         TREATMENT: HSCT curative (same as AR-IFNGR1 complete); anti-mycobacterial therapy as bridge;
           recombinant IFN-γ NOT effective (signalling pathway absent);
         seed SEED_BASE+1).
IL12B  (interleukin-12 subunit beta; 328aa; 5q33.3; AR;
         Mendelian Susceptibility to Mycobacterial Disease — IL-12p40 deficiency;
         IL12B encodes the p40 subunit (IL-12B/p40) which is SHARED between IL-12 (IL-12p35/p40
           heterodimer) and IL-23 (IL-12B/p19 heterodimer); biallelic LOF → combined IL-12 AND IL-23
           deficiency → impaired Th1 AND Th17 differentiation (IL-23 is required for Th17 maintenance
           and ILC3 survival);
         CLINICAL PHENOTYPE:
           BCG DISEASE: the most frequent presenting problem in vaccinated countries — BCG vaccination
             given at birth → local/regional BCGosis or disseminated BCGosis within months;
           NTM: M. avium complex, M. fortuitum, M. abscessus — recurrent but often survivable with
             anti-mycobacterials (partial residual IFN-γ from NK cells and CD8+ T cells, which can
             produce some IFN-γ independently of IL-12, partially compensates);
           NON-TYPHOIDAL SALMONELLA: recurrent Salmonella bacteremia, a key additional feature that
             DISTINGUISHES IL-12B/IL-12RB1 defects from IFNGR1/IFNGR2 defects; IL-12 is required
             for macrophage activation against intracellular Salmonella;
           CANDIDA: mild CMC in some IL-12B patients (IL-23 loss → reduced Th17) but less prominent
             than dedicated Th17-pathway CMC genes;
         IFN-γ SERUM NORMAL OR LOW: distinguishes from IFNGR1/IFNGR2 defects — in IL-12B deficiency,
           IFN-γ production is impaired (not enough IL-12 drive to NK/T cells) but the receptor is intact;
         TREATMENT: exogenous recombinant IFN-γ (rIFN-γ, actimmune) — DRAMATICALLY EFFECTIVE because
           the IFN-γ receptor is intact; IFN-γ administered 3× weekly subcutaneously; lifelong in
           severe cases; anti-mycobacterials continued concurrently;
         CONSANGUINITY: common founder mutations in Arab, Turkish, North African populations;
         seed SEED_BASE+2).
IL12RB1 (interleukin-12 receptor subunit beta-1; 662aa; 19p13.11; AR;
         Mendelian Susceptibility to Mycobacterial Disease — IL-12Rβ1 deficiency; MOST COMMON AR MSMD WORLDWIDE;
         IL12RB1 encodes the IL-12Rβ1 chain, which pairs with IL-12Rβ2 to form the IL-12 receptor,
           and with IL-23R to form the IL-23 receptor; biallelic LOF → complete absence of BOTH
           IL-12 AND IL-23 receptor signalling (identical to IL12B deficiency in terms of downstream
           consequence but upstream of the IL12B defect in the pathway);
         EPIDEMIOLOGY: most common AR form of MSMD in all populations studied worldwide; more common
           than IL12B deficiency; many consanguineous families with homozygous null mutations;
         COMBINED IL-12 + IL-23 RECEPTOR ABSENCE:
           → No IL-12 signalling → impaired Th1 differentiation → reduced IFN-γ production → poor
             macrophage activation → mycobacterial susceptibility;
           → No IL-23 signalling → impaired Th17 maintenance → mild mucosal Candida susceptibility
             in some patients;
         ORGANISMS: BCG disease (most common); NTM (MAC, M. fortuitum); Salmonella bacteremia;
           mild CMC in a subset (distinguishing from IFNGR defects where CMC is rare);
         IFN-γ SERUM NORMAL OR LOW: same as IL12B (IL-12 production is normal in IL12RB1 — the ligand
           is made but can't signal → distinguishable from IL12B by measuring serum p40 cytokine);
         FLOW CYTOMETRY: IL-12Rβ1 absent on T/NK cell surface → diagnostic rapid test;
         TREATMENT: recombinant IFN-γ — effective (IFN-γ receptor intact); lifelong anti-mycobacterials;
           periodic Salmonella prophylaxis in endemic areas; mild CMC responds to azoles;
         PROGNOSIS: significantly better than AR-IFNGR1/IFNGR2 defects — partial residual IFN-γ from
           receptor-independent NK cell pathways means spontaneous clinical improvement with age in some;
           HSCT reserved for severe refractory cases;
         seed SEED_BASE+3).
STAT1  (signal transducer and activator of transcription 1; 750aa; 2q32.2; AD LOF + AR LOF;
         STAT1 LOSS-OF-FUNCTION (LOF) — MSMD + Herpes Simplex Encephalitis (HSE) susceptibility;
         CRITICAL DISTINCTION: STAT1 GOF causes CMC (see Hereditary-CMC-Atlas); STAT1 LOF causes
           the OPPOSITE — mycobacterial susceptibility + viral susceptibility — SAME GENE, COMPLETELY
           DIFFERENT PHENOTYPES depending on whether the mutation increases or decreases STAT1 function;
         MECHANISM:
           STAT1 is activated downstream of: Type I IFNs (IFNAR → JAK1/TYK2 → pY701-STAT1),
           Type II IFN (IFNGR → JAK1/JAK2 → pY701-STAT1), Type III IFNs (IFN-λ), and also IL-6/IL-27;
           STAT1 LOF → impaired IFN-γ signalling (MSMD component) AND impaired IFN-α/β signalling
           (viral susceptibility component — particularly HSE);
         AD LOF (haploinsufficiency):
           MSMD: recurrent NTM/BCG disease (milder than AR-IFNGR1; partial STAT1 function from normal
             allele) — usually non-life-threatening with treatment;
           HSE: herpes simplex encephalitis in children — AD STAT1 LOF is THE MOST COMMON GENETIC CAUSE
             of sporadic HSE in children worldwide; impaired IFN-α/β responses → HSV-1 reaches CNS;
             DISTINCT from TLR3/UNC93B1-HSE where only TLR3 pathway is impaired (STAT1 LOF impairs
             ALL IFN signalling);
         AR LOF (complete):
           SEVERE: disseminated mycobacterial disease + fatal viral infections (CMV, VZV, HSV, influenza,
             measles) — complete absence of both IFN-γ and IFN-α/β pathways; often fatal in childhood
             without HSCT;
         TREATMENT: AD LOF MSMD — anti-mycobacterials + prophylactic IFN-γ; acyclovir for HSE episodes;
           AR LOF — HSCT;
         BCG CONTRAINDICATED: BCG ABSOLUTELY CONTRAINDICATED in STAT1 LOF — same rationale as other
           IFN-γ pathway defects (no IFN-γ signalling to control BCG replication);
         seed SEED_BASE+4).
IRF8   (interferon regulatory factor 8; 426aa; 16q24.1; AD LOF + AR LOF;
         Interferon regulatory factor 8 deficiency — monocyte dendritic cell immunodeficiency (MDCD) / MSMD;
         IRF8 is the MASTER TRANSCRIPTION FACTOR for monocyte-DC-macrophage lineage development;
           expressed in myeloid progenitors, it drives commitment to monocyte, classical DC (cDC1, cDC2),
           and plasmacytoid DC (pDC) lineages; IRF8 cooperates with PU.1 (SPI1), IRF4, and BATF3;
         TWO DISTINCT MUTATION CLASSES:
           (1) AD LOF — T80A heterozygous (most common; haploinsufficiency):
               Selective cDC1 (BDCA3+, XCR1+) dendritic cell absence in blood — cDC1s are required
               for cross-presentation of intracellular pathogens to CD8+ T cells and for IL-12
               production; NK cell function impaired; MSMD with recurrent NTM/BCG; selective DC1
               deficiency without complete DC absence; spontaneous improvement with age in some;
           (2) AR LOF — K108E biallelic (or other null mutations):
               TOTAL ABSENCE of monocytes + all DCs (cDC1, cDC2, pDC) from birth — SEVERE COMBINED
               IMMUNODEFICIENCY-LIKE phenotype; resembles SCID severity (opportunistic infections, viral,
               mycobacterial, fungal); HSCT REQUIRED URGENTLY;
         DIAGNOSTIC TEST: flow cytometry for blood cDC1 (BDCA3+/CD141+/XCR1+) and monocyte subsets;
           IRF8 surface expression by monocyte stimulation test; gene sequencing confirms;
         IL-12 PRODUCTION IMPAIRED: cDC1s are major IL-12 producers → IRF8 deficiency → low IL-12 →
           impaired Th1 differentiation → mycobacterial susceptibility (same downstream as IL12B defects);
         TREATMENT: AD LOF — anti-mycobacterials + IFN-γ; close monitoring; AR LOF — HSCT urgently;
         seed SEED_BASE+5).
ISG15  (ISG15 ubiquitin-like modifier; 165aa; 1p36.33; AR;
         ISG15 deficiency — mycobacterial susceptibility + CNS IFN-α overactivation paradox;
         ISG15 is a UBIQUITIN-LIKE MODIFIER (Ubl); it has TWO COMPLETELY OPPOSITE BIOLOGICAL ROLES:
           (1) Intracellular: ISGylation (conjugation to target proteins) — broad antiviral ISG function;
               NOT relevant to MSMD phenotype;
           (2) Extracellular/secreted: free ISG15 acts as a CYTOKINE amplifying IFN-γ production from
               NK cells and CD8+ T cells by stabilising USP18 secretion;
               ISG15 LOF → reduced secreted ISG15 → reduced IFN-γ production from NK/CD8+ T cells
               → impaired macrophage activation → MYCOBACTERIAL SUSCEPTIBILITY (BCG disease, NTM);
         THE ISG15 PARADOX — DUAL PHENOTYPE:
           MSMD component: ISG15 LOF → less IFN-γ from NK/T cells → mycobacterial susceptibility;
           CNS OVERACTIVATION component: ISG15 is also required for stabilisation of USP18 (a negative
             regulator of IFN-α/β signalling); ISG15 LOF → USP18 destabilised → TYPE I IFN PATHWAY
             HYPERACTIVATED in CNS → cerebral calcifications (basal ganglia, periventricular),
             progressive intracranial hypertension, pseudo-TORCH syndrome;
         ISG15 LOF = MYCOBACTERIAL SUSCEPTIBILITY + CNS CALCIFICATIONS — unique double phenotype;
           brain MRI: periventricular and basal ganglia calcifications from birth;
         INFLUENZA SUSCEPTIBILITY: ISG15-deficient patients have severe influenza (intracellular
           ISGylation contributes to antiviral defence, though less critical than the IFN-γ/mycobacterial
           pathway defect);
         TREATMENT: anti-mycobacterials; IFN-γ therapy (receptor intact); corticosteroids/acetazolamide
           for CNS intracranial hypertension; no HSCT typically required;
         seed SEED_BASE+6).
CYBB   (cytochrome B-245 heavy chain / gp91phox; 570aa; Xp21.1; XL hypomorphic;
         X-linked MSMD (R-MSMD) — CYBB hypomorphic mutations: macrophage-selective oxidative burst defect;
         CYBB encodes gp91phox (glycoprotein 91kDa), the catalytic subunit of the NADPH oxidase
           (NOX2) complex; classic X-CGD (X-linked chronic granulomatous disease) involves complete
           loss of NOX2 → global neutrophil + macrophage ROS defect → recurrent catalase-positive
           bacteria + Aspergillus (see Hereditary-CGD-Phagocyte-Defect-Atlas);
         CYBB HYPOMORPHIC → R-MSMD (DISTINCT FROM CLASSIC X-CGD):
           Missense mutations that partially reduce, not abolish, NOX2 activity — typically 10-20%
           residual gp91phox function; macrophage-specific NADPH oxidase severely impaired (macrophages
           require high NOX2 activity) but neutrophil NOX2 relatively preserved (residual activity
           sufficient for neutrophil ROS); DHR (dihydrorhodamine) test in neutrophils: NORMAL or
           near-normal in hypomorphic → does NOT detect R-MSMD-CYBB (classic CGD diagnosis missed!);
         PHENOTYPE: BCG disease, NTM, M. tuberculosis in endemic areas; NO Aspergillus
           (macrophage-specific defect; neutrophil antifungal intact); NO recurrent bacterial
           abscesses/granulomas typical of classic X-CGD;
         BCG ABSOLUTELY CONTRAINDICATED: CYBB hypomorphic → BCGosis at vaccination;
         DIAGNOSTIC TRAP: DHR test normal → CYBB hypomorphic patients routinely MISSED on standard
           CGD screening; require gene sequencing of CYBB + macrophage-specific NOX2 activity assay;
         X-LINKED: males only; carrier females protected;
         TREATMENT: anti-mycobacterials; recombinant IFN-γ (partially effective — augments macrophage
           activation via an NOX2-independent pathway); HSCT for severe/refractory cases;
         seed SEED_BASE+7).
"""

import random

GENES     = ["IFNGR1", "IFNGR2", "IL12B", "IL12RB1", "STAT1", "IRF8", "ISG15", "CYBB"]
SEED_BASE = 3910   # 320-patient aggregate: 8 genes × 40 patients, seeds 3910-3917

GENE_META = {
    "IFNGR1":  {"aa": 472,  "kda": "54",   "locus": "6q23.3",   "inh": "AD partial LOF / AR complete LOF", "syndrome": "MSMD1-Most-Common", "mim_gene": 107470, "mim_disease": 209950},
    "IFNGR2":  {"aa": 337,  "kda": "38",   "locus": "21q22.11", "inh": "AR complete LOF",                  "syndrome": "MSMD2",             "mim_gene": 147569, "mim_disease": 614889},
    "IL12B":   {"aa": 328,  "kda": "37",   "locus": "5q33.3",   "inh": "AR",                               "syndrome": "MSMD-IL12B-BCG-Salmonella", "mim_gene": 161561, "mim_disease": 614898},
    "IL12RB1": {"aa": 662,  "kda": "75",   "locus": "19p13.11", "inh": "AR",                               "syndrome": "MSMD-IL12RB1-Most-Common-AR", "mim_gene": 601604, "mim_disease": 614893},
    "STAT1":   {"aa": 750,  "kda": "87",   "locus": "2q32.2",   "inh": "AD LOF / AR LOF",                  "syndrome": "MSMD5-HSE",         "mim_gene": 600555, "mim_disease": 614892},
    "IRF8":    {"aa": 426,  "kda": "49",   "locus": "16q24.1",  "inh": "AD LOF / AR LOF",                  "syndrome": "MDCD-IRF8-MSMD",    "mim_gene": 601565, "mim_disease": 614894},
    "ISG15":   {"aa": 165,  "kda": "18",   "locus": "1p36.33",  "inh": "AR",                               "syndrome": "MSMD-ISG15-CNS-Calcification", "mim_gene": 147571, "mim_disease": 616390},
    "CYBB":    {"aa": 570,  "kda": "91",   "locus": "Xp21.1",   "inh": "XL hypomorphic",                   "syndrome": "R-MSMD-CYBB-Hypomorphic", "mim_gene": 300481, "mim_disease": 300645},
}

DEFINITIONS = {
    "MSMD": "Mendelian Susceptibility to Mycobacterial Disease — rare PID characterised by susceptibility to weakly virulent mycobacteria (NTM: MAC, M. fortuitum, M. kansasii) and BCG vaccine strain; caused by defects in the IFN-γ / IL-12 / STAT1 axis",
    "BCGosis": "Disseminated or severe BCG disease after BCG vaccination — occurs in MSMD patients; BCG (M. bovis BCG) is an attenuated live vaccine that is normally controlled by intact IFN-γ signalling; in IFN-γ pathway defects, BCG proliferates uncontrolled",
    "IFN_gamma_axis": "IFN-γ → binds IFNGR1/IFNGR2 heterodimer → JAK1/JAK2 activation → STAT1 phosphorylation (pY701) → GAF (STAT1 homodimer) formation → nuclear translocation → IRF1 transcription → macrophage killing programme (iNOS, ROS, autophagy)",
    "IL12_IL23_axis": "Macrophages/DCs release IL-12 (p35/p40) → NK/T cell IL-12Rβ1/IL-12Rβ2 receptor → JAK2/TYK2 → STAT4 → IFN-γ production; IL-23 (p19/p40) → IL-23R/IL-12Rβ1 → STAT4/STAT3 → Th17 maintenance",
    "R_MSMD": "Recessive MSMD caused by X-linked hypomorphic CYBB mutations; macrophage NOX2 selectively impaired; DHR test near-normal (neutrophils spared) → misses standard CGD screening; CYBB gene sequencing required",
    "cDC1": "Classical dendritic cell type 1 (BDCA3+/CD141+/XCR1+); the major cross-presenting DC and IL-12 producer; absent in IRF8-AD-LOF haploinsufficiency and IRF8-AR-LOF total DC absence",
    "ISG15_paradox": "ISG15 LOF: (1) MSMD — reduced secreted ISG15 → impaired IFN-γ production from NK/CD8+ T cells → mycobacterial susceptibility; (2) CNS overactivation — USP18 destabilised → TYPE I IFN hypersignalling → basal ganglia/periventricular calcifications",
    "DHR_test": "Dihydrorhodamine 123 flow cytometry — measures neutrophil NADPH oxidase ROS generation; normal in CYBB hypomorphic (R-MSMD) and all IFN-γ/IL-12 axis defects; only abnormal in classic CGD and severe CYBB null mutations",
    "serum_IFNg_elevated": "Serum IFN-γ elevation is PATHOGNOMONIC of an IFN-γ RECEPTOR defect (IFNGR1, IFNGR2) — IFN-γ accumulates because the receptor cannot bind/transduce signal; serum IFN-γ is normal or LOW in IL-12/IL-12R defects (impaired IFN-γ production, not impaired reception)",
    "STAT1_LOF_vs_GOF": "STAT1 LOF: MSMD + HSE (reduced IFN signalling → mycobacterial + viral susceptibility); STAT1 GOF: CMC + autoimmune endocrinopathy (increased IFN signalling → Th17 suppression → fungal susceptibility) — SAME GENE, COMPLETELY OPPOSITE PHENOTYPES",
    "BCG_CI_MSMD": "BCG vaccination ABSOLUTELY CONTRAINDICATED in ALL MSMD genes (IFNGR1, IFNGR2, STAT1 LOF, IRF8, ISG15, CYBB hypomorphic) and IL-12 axis defects (IL12B, IL12RB1) — BCGosis is the most common presenting event in vaccinated MSMD patients; check BCG vaccination status urgently in any infant with suspected MSMD",
    "rIFNg_therapy": "Recombinant IFN-γ (rIFN-γ, actimmune/interferon gamma-1b) — effective in IL-12B, IL-12RB1, STAT1 AD-LOF, ISG15, IRF8 AD-LOF (receptor intact); NOT effective in IFNGR1/IFNGR2 defects (receptor absent) or STAT1 AR-LOF (effector absent)",
}


def _gene_cohort(gene: str, seed: int) -> dict:
    rng = random.Random(seed)
    n = 40
    meta = GENE_META[gene]

    if gene == "IFNGR1":
        # Bimodal: AD partial (survivable) and AR complete (fatal without HSCT)
        mean_onset_months = rng.gauss(3, 2)      # BCGosis at birth / early infancy
        mean_dx_lag = rng.gauss(2.0, 1.5)
        ntm_n           = rng.randint(28, 36)
        bcg_disease_n   = rng.randint(30, 38)
        tb_n            = rng.randint(10, 20)
        salmonella_n    = rng.randint(6, 14)
        viral_suscept_n = rng.randint(6, 14)
        cns_calcif_n    = rng.randint(0, 2)
        hsct_n          = rng.randint(12, 22)
        ifng_elevated_n = rng.randint(34, 38)    # serum IFN-γ elevated in receptor defects
        ifng_response_n = rng.randint(14, 22)    # partial AD response to exogenous IFN-γ
        pathognomonic   = "SERUM IFN-γ ELEVATED (pathognomonic of IFN-γ receptor defect); BCGosis most common presenting event after vaccination; two phenotypes: AD partial (survivable with anti-mycos) vs AR complete (fatal without HSCT)"

    elif gene == "IFNGR2":
        mean_onset_months = rng.gauss(3, 2)
        mean_dx_lag = rng.gauss(2.5, 1.5)
        ntm_n           = rng.randint(28, 36)
        bcg_disease_n   = rng.randint(30, 38)
        tb_n            = rng.randint(10, 20)
        salmonella_n    = rng.randint(6, 14)
        viral_suscept_n = rng.randint(8, 18)
        cns_calcif_n    = rng.randint(0, 2)
        hsct_n          = rng.randint(20, 34)    # AR complete → HSCT required in most
        ifng_elevated_n = rng.randint(34, 38)    # serum IFN-γ elevated
        ifng_response_n = rng.randint(0, 4)      # AR complete → no response to exogenous IFN-γ
        pathognomonic   = "SERUM IFN-γ ELEVATED; IFNGR1 surface expression NORMAL (ligand-binding chain intact) — flow cytometry distinguishes IFNGR2 from IFNGR1 defect; AR complete → HSCT required; NO dominant interfering form unlike IFNGR1"

    elif gene == "IL12B":
        mean_onset_months = rng.gauss(6, 4)
        mean_dx_lag = rng.gauss(4.0, 2.0)
        ntm_n           = rng.randint(22, 32)
        bcg_disease_n   = rng.randint(28, 36)
        tb_n            = rng.randint(12, 22)
        salmonella_n    = rng.randint(24, 34)    # HIGH Salmonella: distinguishing feature
        viral_suscept_n = rng.randint(2, 8)
        cns_calcif_n    = rng.randint(0, 2)
        hsct_n          = rng.randint(4, 12)     # most survive with IFN-γ therapy
        ifng_elevated_n = rng.randint(0, 4)      # serum IFN-γ LOW (production impaired)
        ifng_response_n = rng.randint(32, 38)    # EXCELLENT response to recombinant IFN-γ
        pathognomonic   = "SERUM IFN-γ NORMAL/LOW (production impaired, receptor intact) — distinguishes from IFNGR defects; NON-TYPHOIDAL SALMONELLA BACTEREMIA a key feature; exogenous IFN-γ EFFECTIVE (receptor intact); shared IL-12/IL-23 p40 subunit → mild Th17 impairment"

    elif gene == "IL12RB1":
        mean_onset_months = rng.gauss(6, 4)
        mean_dx_lag = rng.gauss(4.5, 2.0)
        ntm_n           = rng.randint(24, 34)
        bcg_disease_n   = rng.randint(28, 36)
        tb_n            = rng.randint(12, 22)
        salmonella_n    = rng.randint(22, 32)    # HIGH Salmonella: same as IL12B
        viral_suscept_n = rng.randint(2, 8)
        cns_calcif_n    = rng.randint(0, 2)
        hsct_n          = rng.randint(2, 10)     # most survive with IFN-γ; good prognosis
        ifng_elevated_n = rng.randint(0, 4)      # serum IFN-γ LOW
        ifng_response_n = rng.randint(32, 38)    # excellent IFN-γ response
        pathognomonic   = "MOST COMMON AR MSMD WORLDWIDE; IL-12Rβ1 absent on T/NK cells by flow cytometry; combined IL-12 + IL-23 receptor loss; Salmonella bacteremia; mild CMC in some (IL-23 loss → Th17 mild impairment); IFN-γ therapy effective; better prognosis than IFNGR defects"

    elif gene == "STAT1":
        mean_onset_months = rng.gauss(24, 18)    # bimodal: early severe AR, later AD partial
        mean_dx_lag = rng.gauss(3.0, 2.0)
        ntm_n           = rng.randint(22, 32)
        bcg_disease_n   = rng.randint(24, 34)
        tb_n            = rng.randint(8, 18)
        salmonella_n    = rng.randint(4, 12)
        viral_suscept_n = rng.randint(28, 38)    # HIGH viral (HSE most important)
        cns_calcif_n    = rng.randint(0, 4)
        hsct_n          = rng.randint(8, 18)     # AR complete: HSCT
        ifng_elevated_n = rng.randint(0, 6)      # STAT1 LOF → serum IFN-γ variable
        ifng_response_n = rng.randint(8, 18)     # AD LOF partially effective
        pathognomonic   = "STAT1 LOF (DISTINCT from STAT1 GOF causing CMC): AD-LOF → MSMD + HSE (herpes simplex encephalitis, most common genetic cause of sporadic HSE in children); AR-LOF → fatal viral + mycobacterial combined; BCG CONTRAINDICATED; same gene — OPPOSITE phenotypes to STAT1 GOF"

    elif gene == "IRF8":
        mean_onset_months = rng.gauss(12, 10)
        mean_dx_lag = rng.gauss(4.0, 2.5)
        ntm_n           = rng.randint(22, 32)
        bcg_disease_n   = rng.randint(24, 34)
        tb_n            = rng.randint(8, 18)
        salmonella_n    = rng.randint(4, 12)
        viral_suscept_n = rng.randint(6, 16)
        cns_calcif_n    = rng.randint(0, 2)
        hsct_n          = rng.randint(12, 24)    # AR complete: HSCT urgently
        ifng_elevated_n = rng.randint(0, 4)
        ifng_response_n = rng.randint(18, 28)    # AD LOF: IFN-γ partially effective
        pathognomonic   = "IRF8-AD-LOF (T80A): selective cDC1 (BDCA3+/XCR1+) absence in blood; flow cytometry diagnostic; MSMD milder; IRF8-AR-LOF (K108E): TOTAL monocyte + DC absence = SCID-like severity; HSCT urgently required; two completely different severity phenotypes from same gene"

    elif gene == "ISG15":
        mean_onset_months = rng.gauss(6, 4)
        mean_dx_lag = rng.gauss(5.0, 2.5)
        ntm_n           = rng.randint(22, 32)
        bcg_disease_n   = rng.randint(24, 34)
        tb_n            = rng.randint(8, 18)
        salmonella_n    = rng.randint(2, 8)
        viral_suscept_n = rng.randint(16, 26)    # influenza susceptibility
        cns_calcif_n    = rng.randint(28, 36)    # CNS calcifications: PATHOGNOMONIC of ISG15
        hsct_n          = rng.randint(2, 8)      # rare; usually manageable
        ifng_elevated_n = rng.randint(0, 4)
        ifng_response_n = rng.randint(28, 36)    # IFN-γ treatment effective (receptor intact)
        pathognomonic   = "CNS CALCIFICATIONS (basal ganglia + periventricular) PATHOGNOMONIC of ISG15 deficiency — Type I IFN pathway overactivation in CNS (USP18 destabilised); DUAL PHENOTYPE: MSMD + CNS IFN excess; influenza susceptibility; IFN-γ therapy effective (receptor intact)"

    else:  # CYBB hypomorphic
        mean_onset_months = rng.gauss(3, 2)      # BCGosis at vaccination
        mean_dx_lag = rng.gauss(4.0, 2.5)        # often missed: DHR normal
        ntm_n           = rng.randint(24, 34)
        bcg_disease_n   = rng.randint(32, 38)    # BCGosis: most common presenting event
        tb_n            = rng.randint(8, 18)
        salmonella_n    = rng.randint(2, 8)
        viral_suscept_n = rng.randint(2, 8)
        cns_calcif_n    = rng.randint(0, 2)
        hsct_n          = rng.randint(8, 18)
        ifng_elevated_n = rng.randint(0, 4)
        ifng_response_n = rng.randint(18, 28)    # partially effective
        pathognomonic   = "X-LINKED R-MSMD: DHR TEST NORMAL (neutrophils spared — macrophage-specific NOX2 defect); MISSES STANDARD CGD SCREENING — requires CYBB gene sequencing + macrophage-specific NOX2 assay; BCGosis at vaccination; NO Aspergillus (no classic CGD); males only"

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
        "ntm_n": ntm_n,
        "ntm_pct": round(ntm_n / n * 100, 1),
        "bcg_disease_n": bcg_disease_n,
        "bcg_disease_pct": round(bcg_disease_n / n * 100, 1),
        "tb_n": tb_n,
        "tb_pct": round(tb_n / n * 100, 1),
        "salmonella_n": salmonella_n,
        "salmonella_pct": round(salmonella_n / n * 100, 1),
        "viral_suscept_n": viral_suscept_n,
        "viral_suscept_pct": round(viral_suscept_n / n * 100, 1),
        "cns_calcif_n": cns_calcif_n,
        "cns_calcif_pct": round(cns_calcif_n / n * 100, 1),
        "hsct_n": hsct_n,
        "hsct_pct": round(hsct_n / n * 100, 1),
        "ifng_elevated_n": ifng_elevated_n,
        "ifng_elevated_pct": round(ifng_elevated_n / n * 100, 1),
        "ifng_response_n": ifng_response_n,
        "ifng_response_pct": round(ifng_response_n / n * 100, 1),
        "pathognomonic": pathognomonic,
        "mim_gene": meta["mim_gene"],
        "mim_disease": meta["mim_disease"],
        "seed": seed,
    }


def generate_overview() -> dict:
    cohorts = [_gene_cohort(g, SEED_BASE + i) for i, g in enumerate(GENES)]
    total_n          = sum(c["n"] for c in cohorts)
    total_ntm        = sum(c["ntm_n"] for c in cohorts)
    total_bcg        = sum(c["bcg_disease_n"] for c in cohorts)
    total_tb         = sum(c["tb_n"] for c in cohorts)
    total_salmonella = sum(c["salmonella_n"] for c in cohorts)
    total_viral      = sum(c["viral_suscept_n"] for c in cohorts)
    total_cns_calcif = sum(c["cns_calcif_n"] for c in cohorts)
    total_hsct       = sum(c["hsct_n"] for c in cohorts)
    total_ifng_elev  = sum(c["ifng_elevated_n"] for c in cohorts)
    total_ifng_resp  = sum(c["ifng_response_n"] for c in cohorts)
    return {
        "atlas": "Hereditary-MSMD-Atlas",
        "subtitle": "Mendelian Susceptibility to Mycobacterial Disease / IFN-γ–IL-12 Axis Defects (IUIS PID Category 7: Susceptibility to Specific Pathogens)",
        "total_patients": total_n,
        "genes_covered": GENES,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "aggregate_stats": {
            "ntm_pct":          round(total_ntm / total_n * 100, 1),
            "bcg_disease_pct":  round(total_bcg / total_n * 100, 1),
            "tb_pct":           round(total_tb / total_n * 100, 1),
            "salmonella_pct":   round(total_salmonella / total_n * 100, 1),
            "viral_suscept_pct":round(total_viral / total_n * 100, 1),
            "cns_calcif_pct":   round(total_cns_calcif / total_n * 100, 1),
            "hsct_pct":         round(total_hsct / total_n * 100, 1),
            "ifng_elevated_pct":round(total_ifng_elev / total_n * 100, 1),
            "ifng_response_pct":round(total_ifng_resp / total_n * 100, 1),
        },
        "cohort_summaries": [
            {k: v for k, v in c.items() if k != "pathognomonic"}
            for c in cohorts
        ],
        "cascade_testing_concepts": [
            "BCGosis after vaccination (ANY severity) → MSMD workup IMMEDIATELY; check BCG vaccination date and onset of symptoms; serum IFN-γ assay FIRST — if elevated → IFNGR1/IFNGR2 defect; if normal/low → IL-12/IL-12R axis or STAT1/IRF8/ISG15",
            "SERUM IFN-γ ELEVATED → IFNGR1 or IFNGR2 sequencing; distinguish by flow cytometry: IFNGR1 surface absence → IFNGR1 defect; IFNGR1 surface NORMAL but IFNGR2 absent → IFNGR2 defect",
            "SERUM IFN-γ NORMAL/LOW + MSMD → IL-12 axis panel (IL12B, IL12RB1); flow cytometry for IL-12Rβ1 on T/NK cells (IL12RB1 absent on surface → IL12RB1 deficiency confirmed); IL-12Rβ1 normal → IL12B gene sequencing",
            "MSMD + NON-TYPHOIDAL SALMONELLA BACTEREMIA → IL12B or IL12RB1 deficiency strongly suggested; Salmonella bacteremia in a child with NTM/BCG disease = virtually diagnostic of IL-12 axis defect",
            "MSMD + HERPES SIMPLEX ENCEPHALITIS (HSE) → STAT1 LOF (most common genetic cause of sporadic HSE in children); STAT1 LOF impairs BOTH IFN-γ (mycobacterial) AND IFN-α/β (HSE) signalling; check for BCG status urgently",
            "MSMD + CNS CALCIFICATIONS on MRI/CT (basal ganglia/periventricular) → ISG15 deficiency (PATHOGNOMONIC); Type I IFN overactivation in CNS from USP18 destabilisation; IFN-γ therapy effective",
            "MSMD + FLOW CYTOMETRY: cDC1 (BDCA3+/XCR1+) ABSENT from blood → IRF8 deficiency; AD-LOF (T80A): selective cDC1 absence (milder); AR-LOF (K108E): total monocyte + DC absence (SCID-like severity → HSCT urgently)",
            "BCGosis + DHR TEST NORMAL → CYBB HYPOMORPHIC (R-MSMD); DHR normal does NOT exclude MSMD-CYBB (it only measures neutrophil NOX2; macrophage NOX2 defect is missed); CYBB gene sequencing required; X-linked (males only)",
            "IFN-γ THERAPY: EFFECTIVE in IL12B, IL12RB1, IRF8-AD, STAT1-AD, ISG15 (receptor intact); NOT EFFECTIVE in IFNGR1/IFNGR2 complete LOF (receptor absent); NOT effective in CYBB hypomorphic (macrophage NOX2 bypass requires NOX2-independent pathway — partial effect only)",
            "BCG CONTRAINDICATED in ALL MSMD genes: IFNGR1, IFNGR2, STAT1 LOF, IRF8, ISG15, CYBB-hypomorphic, IL12B, IL12RB1 — BCGosis is the most frequent presenting event in vaccinated infants; check vaccination record in every suspected MSMD patient",
            "HSCT: required for IFNGR2 (AR complete), IFNGR1 (AR complete), STAT1 (AR complete), IRF8 (AR complete/K108E); IL12B/IL12RB1 usually managed with IFN-γ therapy; ISG15/CYBB hypomorphic: HSCT reserved for refractory cases",
            "FAMILY CASCADE TESTING: all MSMD genes are rare but phenotype is severe; siblings of BCGosis patients need urgent evaluation before BCG vaccination; carrier detection in XL-CYBB families; prenatal diagnosis available",
        ],
        "iuis_category": "IUIS 2022 Category 7: Susceptibility to Specific Pathogens (Mycobacterial Susceptibility) — IFN-γ/IL-12 axis defects (IFNGR1, IFNGR2, IL12B, IL12RB1, STAT1, ISG15, IRF8) + Phagocyte-MSMD (CYBB hypomorphic)",
    }


def generate_breakdown() -> dict:
    cohorts = [_gene_cohort(g, SEED_BASE + i) for i, g in enumerate(GENES)]
    return {
        "atlas": "Hereditary-MSMD-Atlas",
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "gene_cohorts": cohorts,
        "diagnostic_groups": {
            "ifng_receptor_defects": {
                "genes": ["IFNGR1", "IFNGR2"],
                "mechanism": "IFN-γ receptor absent/non-functional → IFN-γ cannot signal → macrophage cannot activate → mycobacterial killing abolished",
                "shared_feature": "Serum IFN-γ ELEVATED (pathognomonic); BCGosis; NTM; most severe MSMD forms",
                "key_ddx": "IFNGR1 (AD partial LOF — survivable; AR complete — fatal HSCT); IFNGR2 (all AR complete — HSCT; IFNGR1 surface normal on flow cytometry distinguishes from IFNGR1)",
            },
            "il12_il23_axis": {
                "genes": ["IL12B", "IL12RB1"],
                "mechanism": "Impaired IL-12 / IL-23 production or signalling → reduced IFN-γ from NK/T cells → impaired macrophage activation",
                "shared_feature": "Serum IFN-γ LOW; Salmonella bacteremia; NTM/BCG; IFN-γ therapy effective",
                "key_ddx": "IL12B (p40 subunit — measures serum p40); IL12RB1 (most common AR MSMD; IL-12Rβ1 absent on surface by flow cytometry); both have Salmonella as distinguishing feature vs IFNGR defects",
            },
            "stat1_lof_msmd": {
                "genes": ["STAT1"],
                "mechanism": "STAT1 LOF → impaired IFN-γ AND IFN-α/β signalling → mycobacterial + viral susceptibility",
                "shared_feature": "MSMD + HSE (herpes simplex encephalitis); BCG contraindicated; same gene as STAT1 GOF (CMC) — OPPOSITE phenotypes",
                "key_ddx": "AD LOF: MSMD + HSE (most common genetic HSE in children); AR LOF: severe combined IFN deficiency fatal without HSCT; DISTINGUISH from STAT1 GOF: LOF=mycobacterial+viral, GOF=CMC+endocrinopathy+JAKi-responsive",
            },
            "irf8_dc_deficiency": {
                "genes": ["IRF8"],
                "mechanism": "IRF8 master myeloid TF absent/reduced → monocyte/DC lineage development impaired → IL-12 production reduced → IFN-γ impaired → mycobacterial susceptibility",
                "shared_feature": "MSMD with DC lineage defect; flow cytometry shows cDC1 loss",
                "key_ddx": "AD LOF (T80A): selective cDC1 absence (milder MSMD); AR LOF (K108E): total DC+monocyte absence (SCID-like, HSCT urgent); two completely different severity levels from same gene",
            },
            "isg15_isotype_paradox": {
                "genes": ["ISG15"],
                "mechanism": "ISG15 LOF → reduced secreted ISG15 → impaired IFN-γ amplification + USP18 destabilised → Type I IFN overactivation in CNS",
                "shared_feature": "MSMD + CNS calcifications (pathognomonic dual phenotype); influenza susceptibility",
                "key_ddx": "Only MSMD gene with CNS calcifications (pseudo-TORCH on imaging); ISG15 LOF paradox: MSMD from IFN-γ impairment + CNS IFN excess from USP18 loss; IFN-γ therapy effective",
            },
            "cybb_hypomorphic_r_msmd": {
                "genes": ["CYBB"],
                "mechanism": "Hypomorphic CYBB → residual gp91phox (10-20%) → macrophage NOX2 selectively impaired → mycobacterial killing defective → R-MSMD without classic CGD",
                "shared_feature": "BCGosis; NTM; DHR test NORMAL (diagnostic trap); X-linked males only",
                "key_ddx": "DHR NORMAL misses this diagnosis (neutrophils spared); CYBB sequencing + macrophage-specific NOX2 assay required; no Aspergillus (unlike classic X-CGD); no neutrophil catalase+ bacteria susceptibility",
            },
        },
        "pathognomonic_features": {
            "IFNGR1_serum_IFNg": "SERUM IFN-γ ELEVATED — pathognomonic of IFN-γ receptor defect; IFN-γ accumulates because the receptor is non-functional; IFNGR1 AD-partial vs AR-complete: surface IFNGR1 flow cytometry + T3-cell IFN-γ response assay",
            "IFNGR2_flow": "IFNGR1 SURFACE NORMAL + IFNGR2 ABSENT on flow cytometry — distinguishes IFNGR2 from IFNGR1 defect; AR complete only (no AD partial form unlike IFNGR1)",
            "IL12B_salmonella": "NON-TYPHOIDAL SALMONELLA BACTEREMIA — characteristic feature of IL-12 axis defects; Salmonella in an infant with BCG disease/NTM virtually diagnoses IL12B or IL12RB1",
            "IL12RB1_most_common": "MOST COMMON AR MSMD WORLDWIDE — IL-12Rβ1 absent on T/NK cell surface by flow cytometry; combined IL-12 + IL-23 receptor absence; IFN-γ therapy effective",
            "STAT1_LOF_HSE": "STAT1 LOF → MSMD + HSE (herpes simplex encephalitis) — AD LOF causes most common genetic HSE in children (impaired IFN-α/β → HSV-1 reaches CNS); OPPOSITE of STAT1 GOF (CMC); BCG ABSOLUTELY CONTRAINDICATED",
            "IRF8_cDC1": "cDC1 ABSENCE (BDCA3+/XCR1+ cells absent from blood) by flow cytometry — rapid diagnostic test; AD T80A (selective cDC1 loss) vs AR K108E (total DC+monocyte absence = SCID severity)",
            "ISG15_CNS_calcif": "CNS CALCIFICATIONS (basal ganglia + periventricular) on MRI/CT — PATHOGNOMONIC of ISG15 deficiency among MSMD genes; dual phenotype: MSMD + Type I IFN excess; pseudo-TORCH pattern",
            "CYBB_DHR_normal": "DHR TEST NORMAL — PATHOGNOMONIC TRAP for CYBB hypomorphic (R-MSMD); standard CGD screening MISSES this diagnosis; requires CYBB gene sequencing + macrophage-specific oxidative burst assay",
        },
    }


def generate_definitions() -> dict:
    return {
        "atlas": "Hereditary-MSMD-Atlas",
        "definitions": DEFINITIONS,
        "gene_metadata": GENE_META,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "key_distinguishers": {
            "IFNGR_vs_IL12_axis": "IFNGR1/IFNGR2: SERUM IFN-γ ELEVATED (receptor defect → IFN-γ accumulates); IL12B/IL12RB1: SERUM IFN-γ NORMAL/LOW (production impaired → IFN-γ not made); this is the key initial triage test in any MSMD patient",
            "IL12B_vs_IL12RB1": "IL12B: serum IL-12p40 low/absent (production defect); IL12RB1: serum IL-12p40 NORMAL (made but can't signal); IL-12Rβ1 absent on T/NK surface by flow cytometry (IL12RB1) vs normal surface (IL12B); both have Salmonella and respond to rIFN-γ",
            "STAT1_LOF_vs_GOF": "STAT1 LOF: MSMD + HSE (reduced IFN → mycobacterial + viral susceptibility); STAT1 GOF: CMC + autoimmune endocrinopathy + JAKi-responsive (excess IFN → Th17 suppressed); same gene, OPPOSITE phenotypes — the single most important teaching point about STAT1",
            "IRF8_AD_vs_AR": "IRF8 AD-LOF (T80A): selective cDC1 absence; MSMD; milder; managed with IFN-γ/anti-mycos; IRF8 AR-LOF (K108E): TOTAL monocyte + DC absence; SCID-equivalent severity; HSCT URGENTLY required; flow cytometry distinguishes: cDC1 alone absent (AD) vs all DCs + monocytes absent (AR)",
            "ISG15_paradox_mechanism": "ISG15 LOF paradox: intracellular ISGylation lost (but this is not the MSMD mechanism) + secreted ISG15 lost (→ IFN-γ impaired from NK/T cells → MSMD) + USP18 destabilised (→ Type I IFN hyperactivation in CNS → calcifications); three separate mechanisms from one gene LOF",
            "CYBB_CGD_vs_R_MSMD": "Classic X-CGD (CYBB null): DHR ABNORMAL; global neutrophil + macrophage NOX2 absent; Aspergillus + catalase+ bacteria; CYBB hypomorphic (R-MSMD): DHR NORMAL; macrophage-specific NOX2 impaired only; BCGosis + NTM; NO Aspergillus; the DHR assay cannot distinguish these — gene sequencing required",
            "BCG_risk_table": "BCG ABSOLUTELY CI in ALL 8 MSMD genes; highest BCGosis risk: IFNGR1 (AD+AR), IFNGR2 (AR), CYBB-hypomorphic (XL) — these patients have near-zero macrophage IFN-γ response; moderate risk: STAT1 LOF, IRF8-AR (virtually zero IFN response); intermediate: IL12B, IL12RB1 (partial IFN-γ preserved via NK CD8+ cells); lower risk: IRF8-AD, ISG15 (partial residual)",
            "IFNg_therapy_table": "EFFECTIVE (receptor intact): IL12B ✓, IL12RB1 ✓, IRF8-AD ✓, ISG15 ✓, STAT1-AD LOF ✓ (partial), CYBB-hypomorphic ✓ (partial); NOT EFFECTIVE (receptor absent): IFNGR1-AR ✗, IFNGR2-AR ✗; NOT EFFECTIVE (signalling effector absent): STAT1-AR LOF ✗",
        },
        "diagnostic_pathway": [
            "Step 1: Child/young adult with BCGosis, disseminated NTM, or recurrent TB → MSMD workup; confirm no secondary immunodeficiency (HIV, steroids, malnutrition); check BCG vaccination status and date",
            "Step 2: SERUM IFN-γ level: ELEVATED → IFNGR1/IFNGR2 panel; NORMAL/LOW → IL-12 axis panel (IL12B, IL12RB1) + STAT1 LOF + IRF8 + ISG15 + CYBB",
            "Step 3 (IFNGR path): Flow cytometry — IFNGR1 surface expression: ABSENT → IFNGR1 sequencing; NORMAL but IFNGR2 absent → IFNGR2 sequencing; T3 cell line IFN-γ response assay for AD partial vs AR complete",
            "Step 4 (IL-12 axis path): Flow cytometry for IL-12Rβ1 on T/NK cells: ABSENT → IL12RB1 sequencing confirmed; NORMAL → IL12B serum p40 (low) → IL12B sequencing; consider concurrent IFN-γ therapy while awaiting results (safe if receptor intact)",
            "Step 5: MSMD + HSE → STAT1 LOF sequencing urgently; brain MRI with calcification screen → if basal ganglia calcifications → ISG15 sequencing",
            "Step 6: MSMD + cDC1 absent on flow → IRF8 sequencing; if total DC+monocyte absence → IRF8 AR LOF → HSCT consultation immediately",
            "Step 7: MSMD + DHR NORMAL + male → CYBB hypomorphic sequencing; macrophage-specific NOX2 activity assay (monocyte-derived macrophage DHR); X-linked inheritance confirmed",
            "Step 8: BCG status in all household contacts and at-risk siblings: BCG ABSOLUTELY CONTRAINDICATED before MSMD genes excluded; family genetic counselling; country-specific BCG policy considerations",
        ],
    }
