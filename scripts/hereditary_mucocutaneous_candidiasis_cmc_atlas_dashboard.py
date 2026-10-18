#!/usr/bin/env python3
"""Hereditary-Mucocutaneous-Candidiasis-CMC-Atlas -- Complete 8-Gene Reference
Chronic Mucocutaneous Candidiasis / Th17 Pathway Defects / Invasive Fungal Susceptibility
(STAT1-GOF/AD-CMC-Most-Common-GT50pct-IFN-Hypersignaling-Th17-Suppression-JAKi-Ruxolitinib,
 IL17RA/AR-CMC1-Complete-IL17-Signal-Absent-Severe-CMC-Mucosal-Bacterial,
 IL17F/AD-CMC3-Haploinsufficiency-Mild-Candida-Only,
 TRAF3IP2-ACT1/AR-CMC4-IL17-Adapter-Nail-Skin-Oral,
 IL17RC/AR-CMC5-IL17-Receptor-Complex-C-Isolated-CMC,
 RORC/AR-CMC7-Th17-Master-TF-Absent-CMC-Mycobacteria-Staph,
 CARD9/AR-Invasive-CNS-Candida-PATHOGNOMONIC-CLR-CARD9-NF-kB-Axis,
 CLEC7A-Dectin1/AD-Haploinsufficiency-Mucosal-Candida-Y238X-Variant):

STAT1  (signal transducer and activator of transcription 1; 750aa; 2q32.2; AD GOF;
         Chronic mucocutaneous candidiasis (CMC) due to STAT1 gain-of-function — the MOST
           COMMON identifiable genetic cause of CMC (>50% of familial CMC);
         STAT1 is a transcription factor activated downstream of type I IFN receptors (IFNAR),
           type II IFN receptor (IFNGR), and type III IFN receptors (IL28RA/IFNLR); gain-of-function
           (GOF) mutations reduce de-phosphorylation of pY701-STAT1 (enhanced phosphorylation
           persistence) and increase pSTAT1 nuclear dwell time → HYPERACTIVATED IFN-γ/IFN-α
           signalling → IFN-γ suppresses Th17 cell differentiation (STAT1 GOF blocks the RORγt
           transcriptional programme via IFN-γ-induced SOCS1/T-bet competition) → SEVERELY
           REDUCED Th17 CELLS → deficient IL-17A, IL-17F, IL-22 production → impaired mucosal
           epithelial anti-Candida defence (mucosal β-defensins, S100 proteins, calprotectin
           require IL-17/IL-22 induction);
         CLINICAL TRIAD OF STAT1 GOF:
           (1) CHRONIC MUCOCUTANEOUS CANDIDIASIS: oral, oesophageal, genital, nail, skin Candida
               from infancy — the defining feature; often refractory to fluconazole due to
               biofilm and immune escape;
           (2) AUTOIMMUNE ENDOCRINOPATHY: hypothyroidism (most common, ~60%), T1DM, autoimmune
               hepatitis, alopecia, vitiligo — distinct from IPEX because autoimmunity develops
               progressively, not neonatally; check TFTs, anti-TPO, fasting glucose annually;
           (3) VIRAL AND BACTERIAL SUSCEPTIBILITY: HSV, VZV, HPV (extensive warts), EBV, CMV —
               due to impaired T cell cytotoxicity (IFN-γ hypersignaling paradoxically impairs
               cytotoxic T cell killing efficiency) and mucocutaneous barrier breakdown; Staphylococcal
               skin infections from barrier failure;
         ANEURYSMS: cerebral and aortic aneurysms reported in a subset (STAT1 GOF → vascular wall
           inflammation / impaired regulatory responses); brain MRI at diagnosis and every 3–5 years;
         MALIGNANCY: squamous cell carcinoma from chronic CMC-related HPV/mucosal inflammation;
           oesophageal SCC in patients with severe long-standing oesophageal candidiasis;
         JAK INHIBITORS: ruxolitinib (JAK1/2 inhibitor), baricitinib — DRAMATICALLY EFFECTIVE
           for STAT1 GOF CMC; blocks excess pSTAT1 → restores Th17 compartment; response seen
           within weeks; established emerging first-line for severe STAT1 GOF CMC unresponsive
           to azoles; Th17 restoration confirmed by IL-17A production assay post-ruxolitinib;
         HSCT: reserved for severe/malignancy-complicated cases; risk of excessive graft-versus-host
           inflammatory response due to STAT1 GOF background;
         HOT SPOT MUTATIONS: predominantly in the DNA-binding domain (DBD, exons 10–14) and
           coiled-coil domain (CCD) of STAT1; T385M, R274W, R274Q, K388E among most frequent;
         seed SEED_BASE+0).
IL17RA (interleukin-17 receptor A; 866aa; 22q11.1; AR;
         Chronic mucocutaneous candidiasis type 1 (CMC1) — complete IL-17 receptor signalling
           deficiency;
         IL-17RA is the shared signalling subunit of the IL-17 receptor complex; it is REQUIRED
           for signalling of IL-17A (IL-17RA/RC heterodimer), IL-17F (IL-17RA/RC), IL-17AF
           (IL-17RA/RC), and IL-17E/IL-25 (IL-17RA/RB heterodimer); biallelic LOF mutations →
           COMPLETE ABSENCE of IL-17A, IL-17F, IL-17AF, and IL-17E signalling → the most severe
           form of pure IL-17 pathway CMC;
         SEVERE PURE CMC: persistent oral thrush from infancy, nail dystrophy, skin candidiasis,
           oesophageal involvement — all mucosal compartments; no spontaneous resolution;
         ADDITIONAL SUSCEPTIBILITY: beyond Candida, IL-17RA LOF patients have recurrent S. aureus
           skin infections (IL-17 normally induces mucosal CXCL1, CXCL2, CXCL5 → neutrophil
           recruitment to skin/mucosal surfaces → impaired without IL-17R); gram-negative bacteria
           less prominent;
         NORMAL VIRAL IMMUNITY: in contrast to STAT1 GOF, IL-17RA LOF patients have normal viral
           responses (intact IFN and T cell cytotoxicity pathways);
         NO AUTOIMMUNITY: no autoimmune endocrinopathy; IL-17 deficiency itself does not predispose
           to autoimmunity (distinguishing feature from STAT1 GOF);
         TREATMENT: lifelong azole prophylaxis (fluconazole 100–200 mg/day); recombinant IL-17A
           or IL-17RA pathway supplementation not yet clinically available; HSCT consideration
           in severe refractory cases;
         seed SEED_BASE+1).
IL17F  (interleukin-17F; 163aa; 6p12.2; AD;
         Chronic mucocutaneous candidiasis type 3 (CMC3) — IL-17F haploinsufficiency;
         IL-17F is a homodimeric cytokine (IL-17F/F) or heterodimeric with IL-17A (IL-17A/F);
           both forms signal via the IL-17RA/RC receptor complex; heterozygous LOF mutations
           → haploinsufficiency → REDUCED (not absent) IL-17F and IL-17A/F signalling → milder
           mucosal Candida susceptibility than biallelic IL-17RA/RC deficiency;
         MILDER PHENOTYPE: predominantly oral and nail Candida; oesophageal and systemic spread
           uncommon; onset in childhood or adolescence; response to azoles usually maintained;
         CANDIDA-SELECTIVE: unlike IL-17RA deficiency, IL-17F haploinsufficiency does NOT cause
           bacterial susceptibility — IL-17A/IL-17RA-only signalling (driven by the intact IL-17A
           and other cytokines) is sufficient for bacterial defence but insufficient for peak
           anti-Candida mucosal immunity;
         AUTOSOMAL DOMINANT: heterozygous mutations are sufficient — dominant-negative mechanism
           (IL-17F forms homodimers; mutant subunit + wild-type subunit → non-functional
           heterodimer) or haploinsufficiency; autosomal dominant inheritance means vertical
           transmission is common;
         TREATMENT: as-needed azole therapy; prophylaxis for recurrent cases; outcomes generally
           excellent with azole management alone;
         seed SEED_BASE+2).
TRAF3IP2 (TNF-receptor-associated factor 3 interacting protein 2 = ACT1 = CIKS; 574aa; 6q21; AR;
         Chronic mucocutaneous candidiasis type 4 (CMC4) — IL-17 receptor adaptor deficiency;
         TRAF3IP2 (also called ACT1/CIKS) encodes the adaptor protein ACT1, which is the
           OBLIGATE INTRACELLULAR SIGNALLING ADAPTOR immediately downstream of the IL-17RA/RC
           complex; upon IL-17 binding to IL-17RA/RC, ACT1 is recruited via a SEFIR domain
           homotypic interaction, allowing ACT1 to bridge the IL-17 receptor to TRAF6 → NF-κB
           activation → transcription of anti-Candida mediators (β-defensins, CXCL1, CXCL5,
           IL-6, IL-8, S100A8/A9);
         Without ACT1: IL-17 binds receptor normally but cannot transduce signal intracellularly →
           COMPLETE downstream IL-17 signalling failure despite normal extracellular IL-17 levels
           and normal receptor surface expression; IL-17 ligand assays NORMAL but IL-17-induced
           gene activation absent;
         PHENOTYPE SIMILAR TO IL-17RA BUT MILDER: nail candida (onychomycosis), oral thrush, skin
           Candida; mucosal bacterial susceptibility present but mild; oesophageal Candida less common;
         DIAGNOSTIC PITFALL: serum IL-17A and IL-17F are NORMAL or elevated (feedback
           accumulation) because the IL-17 signal is blocked intracellularly, not extracellularly;
           functional IL-17 signalling assay (NF-κB activation after IL-17 stimulation of patient
           fibroblasts or keratinocytes) required to demonstrate the defect;
         TREATMENT: azole prophylaxis; TRAF3IP2 is also linked to psoriasis susceptibility (non-coding
           variants); no targeted therapy beyond antifungals;
         seed SEED_BASE+3).
IL17RC (interleukin-17 receptor C; 734aa; 3p24.3; AR;
         Chronic mucocutaneous candidiasis type 5 (CMC5) — IL-17RC deficiency;
         IL-17RC is the second subunit of the IL-17A/F signalling receptor complex (IL-17RA/RC
           heterodimer); IL-17A and IL-17F both REQUIRE IL-17RC for their full signalling
           efficiency; biallelic LOF mutations → SELECTIVELY impaired IL-17A and IL-17F responses
           (IL-17E/IL-25 signalling is IL-17RA/RB-dependent and thus SPARED in IL-17RC deficiency,
           unlike IL-17RA deficiency);
         SELECTIVE SUSCEPTIBILITY: restricted to Candida at mucosal surfaces — oral, oesophageal,
           nail; does NOT impair bacterial anti-mucosal defence (because IL-17RC is not required for
           IL-17E signalling, and partial IL-17RA homodimer-independent signalling may persist);
         KEY DISTINCTION FROM IL-17RA DEFICIENCY: IL-17E (anti-parasite / eosinophil-activating
           cytokine signalling) is INTACT in IL-17RC deficiency (IL-17E uses IL-17RA/RB, not
           IL-17RC) — this is the immunological distinguishing feature if IL-17E responses are
           tested in research settings;
         RARE CONDITION: fewer than 30 cases described worldwide; autosomal recessive;
           consanguineous families enriched;
         TREATMENT: azole antifungals; prophylaxis for severe cases; generally responds well;
         seed SEED_BASE+4).
RORC   (RAR-related orphan receptor C = RORγt; 519aa; 1q23.3; AR;
         RORC deficiency — Th17 master transcription factor absent;
         RORγt (encoded by RORC) is the MASTER TRANSCRIPTION FACTOR for Th17 cell differentiation:
           it drives expression of IL-17A, IL-17F, IL-22, IL-26, CCR6, and CD161 in CD4+ T cells;
           RORγt is also required for ILC3 (group 3 innate lymphoid cell) development and function
           in the intestinal mucosa; biallelic LOF mutations → ABSENT RORγt → NO Th17 cells AND
           NO ILC3s → dual innate + adaptive mucosal immunity defect;
         TRIPLE SUSCEPTIBILITY: the broadest fungal/bacterial immune defect among CMC genes:
           (1) CANDIDIASIS (CMC): oral, oesophageal, nail, skin Candida from infancy — due to
               Th17 absence; severe, often refractory to azoles alone;
           (2) MYCOBACTERIAL SUSCEPTIBILITY: BCG vaccine is ABSOLUTELY CONTRAINDICATED in RORC
               deficiency — severe disseminated BCGosis after BCG vaccination reported; non-
               tuberculous mycobacterial (NTM) infections common; Mycobacterium tuberculosis
               susceptibility elevated; mechanism: RORγt is also needed for innate ILC3 function
               in granuloma formation;
           (3) STAPHYLOCOCCAL + GRAM-POSITIVE INFECTIONS: recurrent S. aureus skin/mucosal
               infections; similar to IL-17RA but more severe due to combined ILC3 loss;
         ILC3 ABSENCE: intestinal ILC3s are absent in RORC deficiency → intestinal barrier
           defect → IBD-like features possible; failure to produce IL-22 from ILC3s → impaired
           intestinal epithelial regeneration and antibacterial peptide induction;
         BCG CONTRAINDICATION: BCG is a LIVE VACCINE and must NOT be given to any infant suspected
           of RORC deficiency (genetic testing of all family members before vaccination in
           consanguineous families with a known RORC mutation);
         TREATMENT: azole prophylaxis for Candida + anti-mycobacterial prophylaxis (rifabutin ±
           clarithromycin) + HSCT considered for severe/refractory cases (Th17 restoration);
         seed SEED_BASE+5).
CARD9  (caspase recruitment domain-containing protein 9; 536aa; 9q34.3; AR;
         CARD9 deficiency — invasive fungal infections including CNS candidiasis PATHOGNOMONIC;
         CARD9 is the intracellular signal transducer for C-type lectin receptors (CLRs) including
           Dectin-1 (CLEC7A), Dectin-2 (CLEC6A), and Mincle (CLEC4E) — these are the primary
           pattern recognition receptors for fungal cell wall components (β-glucan via Dectin-1,
           mannan via Dectin-2, mycolic acids via Mincle); upon CLR activation → SYK kinase →
           CARD9/BCL10/MALT1 signalosome → NF-κB → pro-inflammatory cytokines (TNF, IL-6,
           IL-12p70, IL-23) and reactive oxygen species → anti-fungal killing;
         WITHOUT CARD9: all CLR signalling to NF-κB is abolished → profound failure of innate
           anti-fungal immune responses (macrophage, dendritic cell, and neutrophil) → INVASIVE
           FUNGAL INFECTIONS not just mucocutaneous;
         PATHOGNOMONIC — CNS CANDIDA INFECTIONS:
           CARD9 deficiency is the only PRIMARY IMMUNODEFICIENCY where Candida species specifically
           invade the central nervous system causing Candida meningitis, brain abscesses, and
           ventriculitis — this is the SINGLE MOST IMPORTANT DISTINGUISHING FEATURE; invasive CNS
           Candida is exceedingly rare even in other immunodeficiencies; any young patient with
           Candida meningitis should have CARD9 urgently sequenced;
         ADDITIONAL INVASIVE FUNGAL INFECTIONS: Trichosporon asahii CNS infections, Phialophora
           verrucosa chromoblastomycosis, Cladophialophora bantiana brain abscess — rare dematiaceous
           fungi causing cerebral infections uniquely frequent in CARD9 deficiency;
         ALSO DERMATOPHYTOSIS: CARD9 deficiency causes severe, extensive, deep dermatophyte
           infections (tinea unguium, tinea corporis, tinea capitis extending subcutaneously) in
           addition to Candida — the combination of severe dermatophytosis + invasive Candida is
           characteristic;
         MUCOCUTANEOUS CANDIDIASIS: also present but less prominent than INVASIVE INFECTIONS —
           the key distinction from other CMC genes where mucosal disease is the predominant feature;
         GM-CSF THERAPY: recombinant GM-CSF has shown benefit in some CARD9 patients by bypassing
           the CARD9 defect and directly activating myeloid cells via GM-CSF receptor → increased
           phagocyte fungicidal activity;
         TREATMENT: aggressive antifungal therapy (liposomal amphotericin B for invasive disease;
           voriconazole for CNS Candida maintenance); GM-CSF adjunct; HSCT for severe refractory
           cases;
         seed SEED_BASE+6).
CLEC7A (C-type lectin domain family 7 member A = Dectin-1; 247aa; 12p13.2; AD;
         CLEC7A/Dectin-1 haploinsufficiency — mucosal candidiasis due to impaired β-glucan sensing;
         Dectin-1 (CLEC7A) is the primary pattern recognition receptor for fungal β-1,3-glucan on
           the surface of innate immune cells (macrophages, dendritic cells, neutrophils); upon
           β-glucan binding → Dectin-1 → SYK kinase → CARD9 (upstream of CARD9 deficiency) →
           NF-κB → anti-fungal cytokines; Dectin-1 also directly activates NLRP3 inflammasome
           → IL-1β → Th17 induction; Dectin-1 is therefore the earliest sensor connecting
           fungal detection to downstream IL-17/Th17 responses;
         HAPLOINSUFFICIENCY: heterozygous LOF mutations are SUFFICIENT to cause susceptibility
           (dominant effect through surface receptor dose reduction on macrophages/DCs/neutrophils);
           the most common CLEC7A variant in CMC patients is Y238X (a premature stop codon in
           exon 6) — this variant is polymorphic in some populations (South Asian frequency ~5%,
           European ~3%) where it provides partial susceptibility; full LOF is rare;
         PHENOTYPE MILDER THAN CARD9: only mucosal candidiasis (oral, vaginal, nail) — NO invasive
           fungal infections (because CARD9 and downstream CLR signalling is still intact through
           other CLRs such as Dectin-2 and Mincle; Dectin-1 is the primary but not sole CLR);
         POPULATION VARIANT Y238X: the Y238X stop codon in exon 6 creates a truncated protein
           that fails to signal; when homozygous → near-complete Dectin-1 loss → more severe
           recurrent vulvovaginal candidiasis or oral candidiasis; when heterozygous → ~50% Dectin-1
           function → mild susceptibility; the y238X homozygous state is the main cause of
           recurrent vulvovaginal candidiasis in otherwise healthy women in some populations;
         TREATMENT: azole antifungals (fluconazole) for acute infections; prophylaxis for recurrent
           cases; no targeted therapy; recombinant Dectin-1 or β-glucan vaccine strategies under
           investigation;
         seed SEED_BASE+7).
"""

import random
from typing import Any

SEED_BASE = 3902  # seeds 3902-3909

GENES = ["STAT1", "IL17RA", "IL17F", "TRAF3IP2", "IL17RC", "RORC", "CARD9", "CLEC7A"]

GENE_META = {
    "STAT1":    {"locus": "2q32.2",  "aa": 750,  "kda": 87.0,  "inh": "AD GOF", "syndrome": "CMC-Most-Common-GT50pct-STAT1-GOF-Th17-Suppressed-Autoimmune-Endocrinopathy-Viral-JAKi-Ruxolitinib",         "mim_gene": 600555, "mim_disease": 614162},
    "IL17RA":   {"locus": "22q11.1", "aa": 866,  "kda": 95.0,  "inh": "AR",     "syndrome": "CMC1-Complete-IL17A-IL17F-Signal-Absent-Severe-CMC-Mucosal-Bacterial-No-Autoimmunity",                        "mim_gene": 605461, "mim_disease": 613953},
    "IL17F":    {"locus": "6p12.2",  "aa": 163,  "kda": 18.0,  "inh": "AD",     "syndrome": "CMC3-IL17F-Haploinsufficiency-Mild-Candida-Only-Oral-Nail-Azole-Responsive",                                  "mim_gene": 606496, "mim_disease": 613956},
    "TRAF3IP2": {"locus": "6q21",    "aa": 574,  "kda": 64.0,  "inh": "AR",     "syndrome": "CMC4-ACT1-IL17-Adaptor-Absent-Nail-Skin-Oral-Candida-IL17-Extracellular-Normal-Functional-Assay-Diagnostic",  "mim_gene": 607463, "mim_disease": 614192},
    "IL17RC":   {"locus": "3p24.3",  "aa": 734,  "kda": 82.0,  "inh": "AR",     "syndrome": "CMC5-IL17RC-Selective-IL17A-IL17F-Impaired-IL17E-SPARED-Pure-Candida-No-Bacterial-No-Autoimmunity",           "mim_gene": 610925, "mim_disease": 613567},
    "RORC":     {"locus": "1q23.3",  "aa": 519,  "kda": 59.0,  "inh": "AR",     "syndrome": "CMC7-RORgammat-Absent-No-Th17-No-ILC3-CMC-Mycobacteria-Staph-BCG-ABSOLUTELY-CI-HSCT",                        "mim_gene": 602943, "mim_disease": 616622},
    "CARD9":    {"locus": "9q34.3",  "aa": 536,  "kda": 61.0,  "inh": "AR",     "syndrome": "CARD9-CLR-Signalosome-Absent-CNS-Candida-PATHOGNOMONIC-Invasive-Fungal-Dermatophytosis-GM-CSF",              "mim_gene": 607212, "mim_disease": 212050},
    "CLEC7A":   {"locus": "12p13.2", "aa": 247,  "kda": 27.0,  "inh": "AD",     "syndrome": "Dectin1-Haploinsufficiency-Y238X-Variant-Mucosal-Candida-Only-No-Invasive-Fungal-Milder-Than-CARD9",         "mim_gene": 606264, "mim_disease": 613108},
}

DEFINITIONS = {
    "STAT1":    "STAT1 GOF (750aa / 87kDa / 2q32.2 / AD GOF). Hyperactivated IFN signalling suppresses Th17 differentiation → CMC. >50% familial CMC. TRIAD: CMC + autoimmune endocrinopathy (hypothyroidism 60%) + viral susceptibility (HSV/HPV). Cerebral/aortic aneurysm risk. Hot spots: T385M, R274W, K388E (DBD/CCD). Ruxolitinib/baricitinib (JAK1/2i) dramatically effective — blocks pSTAT1 persistence → restores Th17. Annual thyroid function, fasting glucose, MRI screening.",
    "IL17RA":   "IL-17RA (866aa / 95kDa / 22q11.1 / AR). Shared IL-17R signalling subunit — LOF abolishes IL-17A, IL-17F, IL-17AF, IL-17E responses. CMC1: severe oral/oesophageal/nail/skin Candida + S. aureus skin infections. Normal viral immunity (contrast STAT1 GOF). No autoimmunity. Lifelong fluconazole prophylaxis. HSCT for refractory severe cases.",
    "IL17F":    "IL-17F (163aa / 18kDa / 6p12.2 / AD). Haploinsufficiency → reduced IL-17F/F and IL-17A/F homodimer/heterodimer levels → mild CMC3. Oral and nail candida only. No bacterial susceptibility (IL-17RA-only signalling sufficient for bacteria). AD inheritance (vertical transmission). Excellent response to azoles. Prophylaxis for frequent recurrences.",
    "TRAF3IP2": "TRAF3IP2/ACT1 (574aa / 64kDa / 6q21 / AR). Obligate IL-17R intracellular adaptor. LOF → IL-17 binds receptor but cannot signal → CMC4. SERUM IL-17A NORMAL or elevated (diagnostic trap). Functional IL-17 signalling assay (NF-κB induction in fibroblasts/keratinocytes) required for diagnosis. Nail + oral + skin Candida; mild mucosal bacterial susceptibility. Azole prophylaxis.",
    "IL17RC":   "IL-17RC (734aa / 82kDa / 3p24.3 / AR). IL-17RA/RC heterodimer component — LOF impairs IL-17A and IL-17F signalling ONLY (IL-17E/IL-25 via IL-17RA/RB spared). CMC5: isolated mucosal candidiasis, no bacterial susceptibility. Key distinction: IL-17E responses intact (vs IL-17RA deficiency where all IL-17 responses absent). Ultra-rare (<30 cases). Azole responsive.",
    "RORC":     "RORC/RORγt (519aa / 59kDa / 1q23.3 / AR). Th17 master TF. LOF → No Th17 cells, No ILC3s → TRIPLE susceptibility: CMC + NTM/mycobacteria + S. aureus/gram-positive. BCG ABSOLUTELY CONTRAINDICATED (severe BCGosis reported). ILC3 absence → intestinal IBD-like features. Broadest susceptibility pattern of all CMC genes. Azole + anti-mycobacterial prophylaxis (rifabutin ± clarithromycin). HSCT for severe cases.",
    "CARD9":    "CARD9 (536aa / 61kDa / 9q34.3 / AR). CLR signalosome (Dectin-1/2, Mincle → SYK → CARD9/BCL10/MALT1 → NF-κB). LOF → CNS Candida meningitis/abscess PATHOGNOMONIC — essentially unique to CARD9 deficiency among PIDs. Also: Trichosporon, Phialophora, Cladophialophora CNS infections; severe dermatophytosis (tinea extending subcutaneously). GM-CSF therapy: bypasses CARD9 → direct myeloid activation. L-AmB for invasive disease; voriconazole for CNS maintenance.",
    "CLEC7A":   "CLEC7A/Dectin-1 (247aa / 27kDa / 12p13.2 / AD). β-1,3-glucan PRR on macrophage/DC/neutrophil. Haploinsufficiency → reduced Dectin-1 density → impaired β-glucan sensing → mucosal candidiasis. Y238X variant: polymorphic in South Asian (5%) and European (3%) populations; homozygous Y238X → recurrent vulvovaginal/oral candidiasis in otherwise healthy women. NO invasive fungal infections (Dectin-2/Mincle/CARD9 intact). Milder than CARD9. Azole-responsive.",
}


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def _gene_cohort(gene: str, seed: int) -> dict[str, Any]:
    rng = _rng(seed)
    n = 40
    meta = GENE_META[gene]

    if gene == "STAT1":
        mean_onset_months = rng.gauss(18, 12)
        mean_dx_lag = rng.gauss(3.5, 1.5)
        cmc_n = rng.randint(38, 40)
        invasive_n = rng.randint(4, 10)
        autoimmune_endo_n = rng.randint(18, 28)
        viral_suscept_n = rng.randint(22, 32)
        mycobacterial_n = rng.randint(2, 6)
        bacterial_n = rng.randint(8, 16)
        cns_candida_n = rng.randint(0, 2)
        dermatophyte_n = rng.randint(4, 10)
        jaki_response_n = rng.randint(28, 36)
        azole_response_n = rng.randint(18, 28)
        pathognomonic = "CMC + autoimmune endocrinopathy (hypothyroidism ~60%) + viral susceptibility (HSV/HPV) TRIAD; JAKi (ruxolitinib) dramatically effective — test STAT1 in all familial CMC first"
    elif gene == "IL17RA":
        mean_onset_months = rng.gauss(6, 4)
        mean_dx_lag = rng.gauss(2.5, 1.0)
        cmc_n = rng.randint(38, 40)
        invasive_n = rng.randint(2, 6)
        autoimmune_endo_n = rng.randint(0, 3)
        viral_suscept_n = rng.randint(2, 6)
        mycobacterial_n = rng.randint(0, 3)
        bacterial_n = rng.randint(16, 26)
        cns_candida_n = rng.randint(0, 2)
        dermatophyte_n = rng.randint(2, 6)
        jaki_response_n = rng.randint(2, 6)
        azole_response_n = rng.randint(14, 22)
        pathognomonic = "Complete IL-17 signal absent (all IL-17A/F/AF/E blocked); severe CMC + S. aureus skin infections; NO autoimmunity (vs STAT1 GOF); azole prophylaxis lifelong"
    elif gene == "IL17F":
        mean_onset_months = rng.gauss(36, 18)
        mean_dx_lag = rng.gauss(5.0, 2.0)
        cmc_n = rng.randint(34, 40)
        invasive_n = rng.randint(0, 2)
        autoimmune_endo_n = rng.randint(0, 3)
        viral_suscept_n = rng.randint(0, 3)
        mycobacterial_n = rng.randint(0, 2)
        bacterial_n = rng.randint(2, 6)
        cns_candida_n = 0
        dermatophyte_n = rng.randint(0, 4)
        jaki_response_n = rng.randint(0, 4)
        azole_response_n = rng.randint(30, 38)
        pathognomonic = "Mild oral/nail CMC only; AD haploinsufficiency; IL-17F partial reduction; no bacterial/viral susceptibility; excellent azole response; vertical transmission"
    elif gene == "TRAF3IP2":
        mean_onset_months = rng.gauss(24, 12)
        mean_dx_lag = rng.gauss(4.0, 1.5)
        cmc_n = rng.randint(36, 40)
        invasive_n = rng.randint(2, 6)
        autoimmune_endo_n = rng.randint(0, 3)
        viral_suscept_n = rng.randint(2, 6)
        mycobacterial_n = rng.randint(0, 3)
        bacterial_n = rng.randint(8, 16)
        cns_candida_n = rng.randint(0, 2)
        dermatophyte_n = rng.randint(2, 6)
        jaki_response_n = rng.randint(2, 6)
        azole_response_n = rng.randint(18, 28)
        pathognomonic = "IL-17 extracellular levels NORMAL (diagnostic trap); functional NF-kB assay in fibroblasts required; nail + oral + skin Candida; CMC4"
    elif gene == "IL17RC":
        mean_onset_months = rng.gauss(24, 12)
        mean_dx_lag = rng.gauss(3.5, 1.5)
        cmc_n = rng.randint(36, 40)
        invasive_n = rng.randint(0, 2)
        autoimmune_endo_n = rng.randint(0, 2)
        viral_suscept_n = rng.randint(0, 3)
        mycobacterial_n = rng.randint(0, 2)
        bacterial_n = rng.randint(2, 6)
        cns_candida_n = 0
        dermatophyte_n = rng.randint(0, 4)
        jaki_response_n = rng.randint(0, 3)
        azole_response_n = rng.randint(28, 36)
        pathognomonic = "Selective IL-17A/F impairment; IL-17E via IL-17RA/RB INTACT (distinguishes from IL-17RA deficiency); pure CMC no bacterial susceptibility; ultra-rare"
    elif gene == "RORC":
        mean_onset_months = rng.gauss(6, 4)
        mean_dx_lag = rng.gauss(2.0, 1.0)
        cmc_n = rng.randint(38, 40)
        invasive_n = rng.randint(4, 10)
        autoimmune_endo_n = rng.randint(2, 6)
        viral_suscept_n = rng.randint(4, 10)
        mycobacterial_n = rng.randint(24, 34)
        bacterial_n = rng.randint(20, 30)
        cns_candida_n = rng.randint(2, 6)
        dermatophyte_n = rng.randint(8, 16)
        jaki_response_n = rng.randint(2, 6)
        azole_response_n = rng.randint(10, 18)
        pathognomonic = "TRIPLE susceptibility: CMC + mycobacteria + Staph; BCG ABSOLUTELY CONTRAINDICATED (live vaccine BCGosis); ILC3 absent → intestinal IBD-like; BROADEST CMC gene phenotype"
    elif gene == "CARD9":
        mean_onset_months = rng.gauss(24, 18)
        mean_dx_lag = rng.gauss(5.0, 2.5)
        cmc_n = rng.randint(28, 36)
        invasive_n = rng.randint(32, 40)
        autoimmune_endo_n = rng.randint(0, 3)
        viral_suscept_n = rng.randint(2, 6)
        mycobacterial_n = rng.randint(2, 6)
        bacterial_n = rng.randint(4, 10)
        cns_candida_n = rng.randint(28, 36)
        dermatophyte_n = rng.randint(26, 34)
        jaki_response_n = rng.randint(2, 6)
        azole_response_n = rng.randint(8, 16)
        pathognomonic = "CNS Candida (meningitis/abscess) PATHOGNOMONIC — essentially unique to CARD9 among PIDs; also Trichosporon/Phialophora/Cladophialophora CNS; severe subcutaneous dermatophytosis; GM-CSF adjunct"
    elif gene == "CLEC7A":
        mean_onset_months = rng.gauss(60, 36)
        mean_dx_lag = rng.gauss(8.0, 3.0)
        cmc_n = rng.randint(34, 40)
        invasive_n = rng.randint(0, 2)
        autoimmune_endo_n = rng.randint(0, 2)
        viral_suscept_n = rng.randint(0, 2)
        mycobacterial_n = rng.randint(0, 2)
        bacterial_n = rng.randint(0, 4)
        cns_candida_n = 0
        dermatophyte_n = rng.randint(2, 6)
        jaki_response_n = rng.randint(0, 2)
        azole_response_n = rng.randint(32, 40)
        pathognomonic = "Y238X stop variant (polymorphic South Asian 5%, European 3%); mucosal Candida ONLY — no invasive fungal (Dectin-2/Mincle/CARD9 intact); recurrent vulvovaginal/oral candidiasis; azole-responsive"
    else:
        mean_onset_months = rng.gauss(24, 12)
        mean_dx_lag = rng.gauss(4.0, 1.5)
        cmc_n = rng.randint(30, 38); invasive_n = rng.randint(4, 10)
        autoimmune_endo_n = rng.randint(4, 10); viral_suscept_n = rng.randint(4, 10)
        mycobacterial_n = rng.randint(2, 6); bacterial_n = rng.randint(6, 14)
        cns_candida_n = rng.randint(0, 4); dermatophyte_n = rng.randint(2, 8)
        jaki_response_n = rng.randint(4, 10); azole_response_n = rng.randint(16, 26)
        pathognomonic = "CMC spectrum defect — see gene-specific entry"

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
        "cmc_n": cmc_n,
        "cmc_pct": round(cmc_n / n * 100, 1),
        "invasive_n": invasive_n,
        "invasive_pct": round(invasive_n / n * 100, 1),
        "autoimmune_endo_n": autoimmune_endo_n,
        "autoimmune_endo_pct": round(autoimmune_endo_n / n * 100, 1),
        "viral_suscept_n": viral_suscept_n,
        "viral_suscept_pct": round(viral_suscept_n / n * 100, 1),
        "mycobacterial_n": mycobacterial_n,
        "mycobacterial_pct": round(mycobacterial_n / n * 100, 1),
        "bacterial_n": bacterial_n,
        "bacterial_pct": round(bacterial_n / n * 100, 1),
        "cns_candida_n": cns_candida_n,
        "cns_candida_pct": round(cns_candida_n / n * 100, 1),
        "dermatophyte_n": dermatophyte_n,
        "dermatophyte_pct": round(dermatophyte_n / n * 100, 1),
        "jaki_response_n": jaki_response_n,
        "jaki_response_pct": round(jaki_response_n / n * 100, 1),
        "azole_response_n": azole_response_n,
        "azole_response_pct": round(azole_response_n / n * 100, 1),
        "pathognomonic": pathognomonic,
        "mim_gene": meta["mim_gene"],
        "mim_disease": meta["mim_disease"],
        "seed": seed,
    }


def generate_overview() -> dict:
    cohorts = [_gene_cohort(g, SEED_BASE + i) for i, g in enumerate(GENES)]
    total_n = sum(c["n"] for c in cohorts)
    total_cmc = sum(c["cmc_n"] for c in cohorts)
    total_invasive = sum(c["invasive_n"] for c in cohorts)
    total_autoimmune = sum(c["autoimmune_endo_n"] for c in cohorts)
    total_viral = sum(c["viral_suscept_n"] for c in cohorts)
    total_mycobacterial = sum(c["mycobacterial_n"] for c in cohorts)
    total_bacterial = sum(c["bacterial_n"] for c in cohorts)
    total_cns_candida = sum(c["cns_candida_n"] for c in cohorts)
    total_dermatophyte = sum(c["dermatophyte_n"] for c in cohorts)
    total_jaki = sum(c["jaki_response_n"] for c in cohorts)
    total_azole = sum(c["azole_response_n"] for c in cohorts)
    return {
        "atlas": "Hereditary-Mucocutaneous-Candidiasis-CMC-Atlas",
        "subtitle": "Chronic Mucocutaneous Candidiasis / Th17 Pathway Defects / Invasive Fungal Susceptibility (IUIS PID Category 6 + STAT1 GOF)",
        "total_patients": total_n,
        "genes_covered": GENES,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "aggregate_stats": {
            "cmc_pct": round(total_cmc / total_n * 100, 1),
            "invasive_pct": round(total_invasive / total_n * 100, 1),
            "autoimmune_endo_pct": round(total_autoimmune / total_n * 100, 1),
            "viral_suscept_pct": round(total_viral / total_n * 100, 1),
            "mycobacterial_pct": round(total_mycobacterial / total_n * 100, 1),
            "bacterial_pct": round(total_bacterial / total_n * 100, 1),
            "cns_candida_pct": round(total_cns_candida / total_n * 100, 1),
            "dermatophyte_pct": round(total_dermatophyte / total_n * 100, 1),
            "jaki_response_pct": round(total_jaki / total_n * 100, 1),
            "azole_response_pct": round(total_azole / total_n * 100, 1),
        },
        "cohort_summaries": [
            {k: v for k, v in c.items() if k not in ("pathognomonic",)}
            for c in cohorts
        ],
        "cascade_testing_concepts": [
            "Recurrent oral/nail/skin Candida → STAT1 GOF sequencing FIRST (>50% familial CMC); check for autoimmune endocrinopathy (TFTs, anti-TPO, glucose) simultaneously",
            "STAT1 GOF confirmed → start ruxolitinib/baricitinib (JAK1/2i) — dramatically effective; monitor CBC, lipids, infection (JAKi immunosuppression risk); brain MRI at baseline",
            "CMC + S. aureus skin infections → IL-17RA or RORC deficiency (both impair bacterial + fungal mucosal defence); distinguish by Th17 count and mycobacterial susceptibility",
            "CMC + mycobacteria + triple susceptibility → RORC deficiency; BCG ABSOLUTELY CONTRAINDICATED — check vaccination history urgently in all suspected RORC deficiency",
            "CNS Candida (meningitis, brain abscess) in any patient → CARD9 deficiency URGENTLY — essentially pathognomonic for CARD9 among primary immunodeficiencies",
            "CARD9 deficiency: L-AmB for invasive/CNS Candida; voriconazole CNS maintenance; consider GM-CSF adjunct; CARD9 panel includes Trichosporon/Phialophora/dematiaceous fungi screen",
            "IL-17 serum level NORMAL → does NOT exclude IL-17 pathway defect — TRAF3IP2/ACT1 deficiency has normal/elevated serum IL-17; requires functional NF-kB signalling assay",
            "Recurrent vulvovaginal candidiasis in otherwise healthy woman → CLEC7A Y238X homozygosity screening; especially in South Asian or East European populations",
            "Mild oral/nail CMC + AD family history + normal workup → IL17F haploinsufficiency; treat with azoles; genetic testing confirms; no aggressive workup needed",
            "IL17RC deficiency (CMC5): test when IL-17RA normal; distinguish by intact IL-17E responses (anti-parasite/eosinophil axis) — only CMC, no bacterial susceptibility",
            "All CMC patients: annual thyroid function + anti-TPO antibodies; fasting glucose; HPV vaccination; screen for SCC in chronic mucosal inflammation sites",
            "HSCT consideration: RORC deficiency (severe mycobacterial), CARD9 (invasive fungal refractory), IL-17RA (severe CMC unresponsive); STAT1 GOF HSCT carries risk of excess GVHD",
        ],
        "iuis_category": "STAT1 GOF: Category 5 (Diseases of Immune Dysregulation); IL17RA/IL17F/TRAF3IP2/IL17RC/RORC: Category 6 (Autoinflammatory disorders / Innate defects); CARD9/CLEC7A: Category 6 (Innate immunity / pattern recognition defects)",
    }


def generate_breakdown() -> dict:
    cohorts = [_gene_cohort(g, SEED_BASE + i) for i, g in enumerate(GENES)]
    return {
        "atlas": "Hereditary-Mucocutaneous-Candidiasis-CMC-Atlas",
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "gene_cohorts": cohorts,
        "diagnostic_groups": {
            "stat1_gof_dysregulation": {
                "genes": ["STAT1"],
                "mechanism": "STAT1 GOF → IFN-γ hypersignalling → Th17 suppression → CMC; unique cause: gain-of-function (not LOF like other CMC genes)",
                "shared_feature": "CMC + autoimmune endocrinopathy + viral susceptibility + JAKi responsive",
                "key_ddx": "STAT1 GOF vs other CMC: autoimmune endocrinopathy (thyroid disease, T1DM) is the key distinguishing feature; only CMC gene where JAKi is primary treatment",
            },
            "il17_receptor_signalling": {
                "genes": ["IL17RA", "IL17F", "TRAF3IP2", "IL17RC"],
                "mechanism": "IL-17 receptor/ligand/adaptor pathway LOF → impaired Th17 effector function → CMC without autoimmunity",
                "shared_feature": "CMC; variable severity based on completeness of IL-17 blockade",
                "key_ddx": "IL17RA (complete, severe, bacterial too); IL17F (mild, Candida only, AD); TRAF3IP2 (normal serum IL-17, functional assay required); IL17RC (selective, IL-17E intact)",
            },
            "th17_development": {
                "genes": ["RORC"],
                "mechanism": "RORγt master TF absent → no Th17 cells AND no ILC3s → triple susceptibility (fungal + mycobacterial + bacterial)",
                "shared_feature": "CMC + mycobacterial + staphylococcal susceptibility; BCG contraindicated",
                "key_ddx": "RORC = BROADEST CMC phenotype; only CMC gene where BCG is CI; mycobacterial susceptibility distinguishes from IL-17RA/RC",
            },
            "clr_innate_fungal": {
                "genes": ["CARD9", "CLEC7A"],
                "mechanism": "C-type lectin receptor (CLR) signalling axis — Dectin-1 (CLEC7A) upstream sensor, CARD9 downstream signalosome",
                "shared_feature": "Impaired innate β-glucan sensing and anti-fungal killing",
                "key_ddx": "CARD9 (complete CLR signalling failure → invasive/CNS fungal PATHOGNOMONIC); CLEC7A (partial Dectin-1 reduction → mucosal candidiasis only, no invasive fungal)",
            },
        },
        "pathognomonic_features": {
            "STAT1_GOF": "CMC + autoimmune endocrinopathy (hypothyroidism ~60%) + viral susceptibility + JAKi responsive — TRIAD; ruxolitinib restores Th17 within weeks",
            "IL17RA_CMC1": "Complete absence of IL-17A/F/AF/E signalling; severe CMC + S. aureus skin; no autoimmunity; lifeong azole prophylaxis",
            "IL17F_CMC3": "Mild AD oral/nail CMC only; vertical transmission; excellent azole response; Candida-selective (no bacterial)",
            "TRAF3IP2_CMC4": "SERUM IL-17 NORMAL (diagnostic trap) — functional NF-kB assay in fibroblasts/keratinocytes required for diagnosis; CMC without autoimmunity",
            "IL17RC_CMC5": "IL-17E/IL-25 responses INTACT (via IL-17RA/RB) — distinguishes from IL-17RA deficiency; pure CMC only; ultra-rare",
            "RORC_CMC7": "Triple susceptibility: CMC + NTM/mycobacteria + S. aureus; BCG ABSOLUTELY CONTRAINDICATED; ILC3 absent — intestinal IBD-like features",
            "CARD9_invasive": "CNS Candida meningitis/abscess PATHOGNOMONIC — essentially unique to CARD9 among PIDs; also subcutaneous/CNS dermatophytosis; GM-CSF adjunct",
            "CLEC7A_Dectin1": "Y238X stop polymorphism; mucosal Candida only (NO invasive); recurrent vulvovaginal/oral candidiasis; azole-responsive; no bacterial/viral susceptibility",
        },
    }


def generate_definitions() -> dict:
    return {
        "atlas": "Hereditary-Mucocutaneous-Candidiasis-CMC-Atlas",
        "definitions": DEFINITIONS,
        "gene_metadata": GENE_META,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "key_distinguishers": {
            "STAT1_GOF_vs_IL17RA": "STAT1 GOF: autoimmune endocrinopathy + viral susceptibility + JAKi responsive — NONE of these in IL17RA; IL17RA: S. aureus bacterial susceptibility — less prominent in STAT1 GOF",
            "TRAF3IP2_diagnostic_trap": "TRAF3IP2/ACT1: serum IL-17A NORMAL or elevated (ligand accumulates because receptor signal blocked) — can falsely exclude IL-17 pathway if only serum IL-17 checked; functional NF-kB assay is essential",
            "IL17RA_vs_IL17RC": "IL17RA LOF: ALL IL-17 responses absent including IL-17E (anti-parasite); IL17RC LOF: only IL-17A and IL-17F impaired, IL-17E via IL-17RA/RB INTACT — distinguishable by IL-17E signalling assay",
            "CARD9_vs_CLEC7A": "CARD9 (complete CLR → NF-kB failure): invasive/CNS Candida + dermatophyte (deep infections) PATHOGNOMONIC; CLEC7A/Dectin-1 haploinsufficiency: mucosal Candida ONLY, NO invasive disease — Dectin-2/Mincle/CARD9 axis still functional",
            "RORC_BCG_CI": "RORC deficiency: BCG live vaccine ABSOLUTELY CONTRAINDICATED — disseminated BCGosis reported; check vaccination record URGENTLY in any infant with RORC mutation or suspected RORC deficiency in family; only CMC gene where BCG is an emergency concern",
            "CMC_severity_spectrum": "Most severe (invasive): CARD9 (CNS Candida) > RORC (triple susceptibility + ILC3 loss) > IL17RA (complete IL-17 signal absent) > TRAF3IP2 > IL17RC > STAT1 GOF (mucosal but JAKi-treatable) > CLEC7A > IL17F (mildest, azole-responsive)",
            "JAKi_for_STAT1_GOF": "Ruxolitinib/baricitinib (JAK1/2i): reduces pSTAT1 persistence → restores Th17 → CMC improvement within 4–8 weeks; only CMC gene where a targeted pathway inhibitor works; NOT indicated for IL-17 pathway or CLR defects",
            "GM_CSF_for_CARD9": "Recombinant GM-CSF (sargramostim/molgramostim): stimulates myeloid phagocytes via GM-CSF receptor independently of CARD9 → increased neutrophil/macrophage fungicidal activity; adjunct to antifungal therapy in CARD9 deficiency; does not restore CARD9 but partially bypasses it",
            "Annual_surveillance_all_CMC": "ALL CMC patients regardless of gene: annual TFTs + anti-TPO (thyroid autoimmunity — especially STAT1 GOF); HPV vaccination (all CMC patients at risk of HPV-related SCC); SCC surveillance of chronically affected mucosal sites; oesophageal CMC → periodic endoscopy",
        },
        "diagnostic_pathway": [
            "Step 1: Chronic/recurrent Candida at mucosal sites (oral, vaginal, nail, oesophageal) from childhood → CMC workup; rule out secondary causes (HIV, immunosuppressants, diabetes, dentures) first",
            "Step 2: STAT1 GOF panel FIRST (most common); simultaneously check for autoimmune endocrinopathy (TFTs, fasting glucose, anti-TPO, cortisol) — if positive → STAT1 GOF highly likely",
            "Step 3: STAT1 GOF confirmed → ruxolitinib/baricitinib; if STAT1 normal → IL-17 pathway panel (IL17RA, IL17F, TRAF3IP2, IL17RC, RORC) by NGS",
            "Step 4: CMC + S. aureus skin + STAT1 normal → IL17RA first (complete IL-17 absence, severe); CMC + bacterial mild → IL17F or TRAF3IP2 (serum IL-17 assay: if normal/elevated → TRAF3IP2 functional assay in fibroblasts)",
            "Step 5: CMC + mycobacteria + STAT1/IL-17R normal → RORC sequencing; check BCG vaccination history URGENTLY — BCG ABSOLUTELY CONTRAINDICATED; anti-mycobacterial prophylaxis immediately",
            "Step 6: INVASIVE FUNGAL / CNS Candida → CARD9 sequencing URGENTLY; start L-AmB empirically; also test for Trichosporon, Phialophora, Cladophialophora; consider GM-CSF adjunct",
            "Step 7: CLEC7A Y238X screening in recurrent vulvovaginal/oral candidiasis in otherwise healthy patients (especially South Asian/East European); homozygous Y238X sufficient for diagnosis",
            "Step 8: All confirmed CMC genes → annual endocrine screen; HPV vaccination if not given; ENT/gastroenterology for oesophageal CMC; dermatology for skin/nail CMC; genetics for family cascade testing",
        ],
    }
