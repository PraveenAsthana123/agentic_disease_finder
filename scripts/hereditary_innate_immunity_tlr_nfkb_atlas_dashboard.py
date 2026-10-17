#!/usr/bin/env python3
"""Hereditary-Innate-Immunity-TLR-NF-kB-Atlas -- Complete 8-Gene Reference
Innate Immune Signalling Defects: TLR-MyD88-NF-κB Pathway and Interferon-Pathway Primary Immunodeficiencies
(IRAK4/AR-Pyogenic-No-Fever-PATHOGNOMONIC-Pneumococcal-Staphylococcal,
 MYD88/AR-Identical-IRAK4-Invasive-Bacterial-No-Systemic-Inflammation,
 NEMO-IKBKG/XL-EDA-ID-Ectodermal-Dysplasia-PATHOGNOMONIC-Anhydrosis-Conical-Teeth,
 NFKBIA-IKBA/AD-GOF-EDA-ID-Mycobacteria-Candida-More-Severe-Than-NEMO,
 IRF3/AD-LOF-Herpes-Simplex-Encephalitis-Susceptibility-Type-I-IFN,
 TLR3/AD-LOF-AR-Herpes-Simplex-Encephalitis-PATHOGNOMONIC-Children,
 UNC93B1/AR-All-Endosomal-TLR-Absent-Herpes-Simplex-Encephalitis-Broadest-Defect,
 IRF7/AR-Life-Threatening-Influenza-PATHOGNOMONIC-Type-I-IFN-Master-Regulator):

IRAK4  (IL-1 receptor-associated kinase 4; 460aa; 12q12; AR;
         IRAK-4 deficiency — hallmark selective susceptibility to Gram-positive bacterial infections;
         IRAK-4 is the serine/threonine kinase that forms the 'Myddosome' complex with MYD88 and
           IRAK-1/IRAK-2 downstream of TLR1/2/4/5/6/7/8/9 and IL-1R/IL-18R/IL-33R signalling;
           IRAK-4 is the proximal kinase that phosphorylates IRAK-1/IRAK-2 → TRAF6 → IKK complex →
           NF-κB activation → TNF, IL-6, IL-1β, IL-8 production;
         Without IRAK-4: TLR-MyD88 and IL-1R pathways are severely impaired → lack of NF-κB activation
           → profoundly reduced IL-6, IL-8, TNF, IL-1β → patients do NOT develop fever, CRP elevation,
           or neutrophil mobilisation in response to bacterial infections;
         PATHOGNOMONIC PRESENTATION: INVASIVE BACTERIAL INFECTIONS WITHOUT SYSTEMIC INFLAMMATION —
           patients present with bacteraemia, meningitis, septic arthritis, pneumonia caused by
           Streptococcus pneumoniae (most common — HIGHLY INVASIVE PNEUMOCOCCAL DISEASE), Staphylococcus
           aureus, Pseudomonas aeruginosa, Listeria, Gram-negative enterics — WITHOUT fever, CRP, ESR
           elevation, or neutrophilia; the absent inflammatory response is the DIAGNOSTIC CLUE;
         CLINICAL IMPROVEMENT WITH AGE: a hallmark of IRAK-4 deficiency — susceptibility to severe
           infections declines markedly after age 14 years (TLR-independent innate immunity, NK cells,
           and adaptive immunity become sufficient); prophylactic penicillin + pneumococcal vaccination
           carried through childhood; adults can often discontinue prophylaxis;
         VIROLOGY: SUSCEPTIBILITY EXCLUSIVELY TO BACTERIA — viral infections (including respiratory
           viruses, HSV, EBV) are handled normally (TRIF/IRF3 pathway intact); this is a key distinction
           from IRF3/TLR3/UNC93B1/IRF7 defects which affect antiviral immunity;
         Treatment: prophylactic penicillin V (amoxicillin) + complete pneumococcal vaccination series;
           IVIG some centres; improve markedly by adolescence;
         seed SEED_BASE+0).
MYD88  (myeloid differentiation primary response 88; 296aa; 3p22.2; AR;
         MyD88 deficiency — clinically and pathophysiologically IDENTICAL to IRAK-4 deficiency;
         MyD88 is the universal Toll/IL-1 receptor (TIR) domain-containing adaptor that connects all
           TLR1/2/4/5/6/7/8/9 and IL-1R superfamily receptors to IRAK-4; without MyD88 →
           downstream Myddosome assembly is completely absent → IRAK-4 cannot be recruited or
           activated → NF-κB and MAPK pathways are not triggered;
         IDENTICAL PHENOTYPE TO IRAK-4: invasive pyogenic bacterial infections without fever/CRP/ESR;
           S. pneumoniae dominant pathogen; S. aureus, Gram-negative bacteria secondary;
         SAME IMPROVEMENT WITH AGE: infections become much less severe after puberty — this is
           consistent across both IRAK-4 and MYD88 deficiencies and represents the most important
           natural history counselling point for families;
         MYD88 SOMATIC GAIN-OF-FUNCTION: unrelated to immunodeficiency — MYD88 p.Leu265Pro GOF is
           the most common mutation in diffuse large B-cell lymphoma (DLBCL) and Waldenström
           macroglobulinaemia; the same gene causes completely opposite clinical effects depending
           on germline LOF (immunodeficiency) vs somatic GOF (lymphoma driver); this is one of the
           clearest examples of LOF vs GOF at the same gene in human disease;
         Treatment: identical to IRAK-4 — penicillin prophylaxis, pneumococcal vaccination, IVIG
           in centres; improves markedly by adulthood;
         seed SEED_BASE+1).
IKBKG  (inhibitor of nuclear factor kappa B kinase regulatory subunit gamma = NEMO; 419aa; Xq28; XL;
         X-linked anhydrotic ectodermal dysplasia with immunodeficiency (XL-EDA-ID);
         NEMO (NF-κB essential modulator) is the regulatory/scaffold subunit of the IKK (IκB kinase)
           complex (IKKα/IKKβ/IKKγ trimer) — IKK phosphorylates IκBα → IκBα polyubiquitinated and
           degraded → NF-κB freed to translocate to nucleus → immune gene expression AND ectodermal
           development gene expression;
         Without NEMO: canonical NF-κB signalling from ALL pathways (TLRs, TNF-R, IL-1R, CD40,
           RANK, LTβR) is abolished → DUAL PHENOTYPE: (1) ectodermal dysplasia + (2) immunodeficiency;
         PATHOGNOMONIC ECTODERMAL DYSPLASIA TRIAD (HED): (1) ANHYDROSIS (absent sweat glands →
           hyperthermia risk in hot environments — fatal hyperthermia if unrecognised in infants);
           (2) HYPOTRICHOSIS (sparse/absent hair, sparse eyebrows); (3) OLIGODONTIA/ANODONTIA with
           CONICAL TEETH — characteristic peg-like teeth are the classic exam finding;
         IMMUNODEFICIENCY: severe — susceptibility to MYCOBACTERIA (NTM and BCG), Candida, pyogenic
           bacteria (S. pneumoniae), Pneumocystis jirovecii; poor antibody responses to polysaccharide
           antigens; NK cell function impaired;
         X-LINKED: males severely affected (hemizygous); females have variable ectodermal findings
           (incontinentia pigmenti — IKBKG heterozygous females present with the IP phenotype);
         HSCT option: HSCT corrects immunological defects but NOT ectodermal features;
         Treatment: IVIG, prophylactic antibiotics/antifungals, BCG CONTRAINDICATED;
         seed SEED_BASE+2).
NFKBIA (NF-κB inhibitor alpha = IκBα; 317aa; 14q13.2; AD gain-of-function;
         Autosomal dominant ectodermal dysplasia with immunodeficiency (AD-EDA-ID);
         IκBα is the inhibitory protein that sequesters NF-κB dimers in the cytoplasm; normally,
           signal-induced phosphorylation of IκBα by the IKK complex → degradation of IκBα → NF-κB
           nuclear translocation; AD GOF variants (typically clustered in the IKK phosphorylation
           sites — S32, S36, or signal-responsive domain) → IκBα resistant to phosphorylation-induced
           degradation → NF-κB PERMANENTLY SEQUESTERED in cytoplasm → constitutive NF-κB deficiency;
         PHENOTYPE: SIMILAR to but DISTINCTLY MORE SEVERE than NEMO/IKBKG-EDA-ID —
           (1) same HED triad (anhydrosis + sparse hair + conical teeth);
           (2) MORE SEVERE IMMUNODEFICIENCY: susceptibility to ALL pathogens of NEMO PLUS additional
               susceptibility to viral infections including herpesviruses; patients have died from
               severe EBV, CMV, varicella-zoster infections — more profound combined immunodeficiency;
           (3) MYCOBACTERIAL SUSCEPTIBILITY: NTM and BCG disease very prominent;
           (4) ANTI-POLYSACCHARIDE RESPONSES: absent;
         AUTOSOMAL DOMINANT: de novo or familial; both sexes affected (contrast to NEMO which is X-linked);
         Treatment: IVIG + prophylactic antibacterials/antifungals/antivirals; HSCT considered in
           severe cases (more evidence than for NEMO);
         seed SEED_BASE+3).
IRF3   (interferon regulatory factor 3; 427aa; 19q13.33; AD loss-of-function;
         IRF3 deficiency — selective susceptibility to HERPES SIMPLEX ENCEPHALITIS (HSE);
         IRF3 is the transcription factor central to the TLR3/TRIF and cGAS/STING antiviral pathways;
           after viral dsRNA sensing by TLR3 → TRIF → TBK1/IKKε → phosphorylation and homodimerisation
           of IRF3 → nuclear translocation → IRF3 directly drives transcription of IFN-β and IFN-α4
           (the immediate-early type I IFN response to viral infection in CNS cells);
         Without functional IRF3 in CNS resident cells (neurons, astrocytes, oligodendrocytes):
           HSV-1 spreads uncontrolled in the brain → necrotising herpes simplex encephalitis with
           high mortality and neurological morbidity;
         PATHOGNOMONIC: HSE IN AN OTHERWISE HEALTHY INDIVIDUAL — if a previously healthy child or
           young adult develops HSE with no other explanation, test TLR3/UNC93B1/TRIF/TBK1/TRAF3/IRF3;
           HSE from this genetic background is one of the archetypes of 'inborn error of immunity to a
           single infectious agent in an otherwise healthy host';
         HAPLOINSUFFICIENCY: AD-LOF variants (haploinsufficiency) are sufficient — one abnormal allele
           is enough to impair IRF3 dimerisation and IFN-β production in TLR3-expressing cells;
         SYSTEMIC IMMUNITY INTACT: outside the CNS the immune system functions normally — no other
           infections; this selectivity reflects the unique dependence of CNS cells on TLR3/IRF3 for
           HSV containment;
         Treatment: Aciclovir IV immediately for HSE — start before genetic result; lifelong aciclovir
           prophylaxis considered in confirmed genetic cases;
         seed SEED_BASE+4).
TLR3   (toll-like receptor 3; 904aa; 4q35.1; AD LOF or AR;
         TLR3 deficiency — hallmark susceptibility to HERPES SIMPLEX ENCEPHALITIS (HSE);
         TLR3 is the endosomal pattern-recognition receptor that detects double-stranded RNA (dsRNA) —
           a replication intermediate of HSV-1 — in endolysosomes of CNS-resident cells; TLR3 signals
           exclusively through the TRIF adaptor (not MyD88) → TBK1 → IRF3 → IFN-β; this TRIF/IRF3
           pathway is the CNS-specific antiviral circuit that HSE susceptibility genes disrupt;
         TLR3 is HIGHLY EXPRESSED IN CNS: neurons, astrocytes, and microglia rely on TLR3 for
           HSV recognition — peripheral immune cells (monocytes, DCs) have redundant PRR pathways
           so systemic HSV control is maintained even in TLR3 deficiency;
         AUTOSOMAL DOMINANT (most common) or AR: many TLR3 HSE mutations are haploinsufficient
           (dominant-negative effect on TLR3 dimerisation); AR alleles described with earlier/more
           severe encephalitis onset;
         SAME PATHOGNOMONIC PRESENTATION as IRF3: first or second episode of HSE in an otherwise
           healthy individual, especially children (70% of childhood HSE cases with identified genetic
           cause have TLR3/UNC93B1/TRIF/IRF3 variants); CHILDREN 3-15 years most common presentation;
         DIAGNOSTIC: TLR3-pathway functional assay in fibroblasts (poly-I:C stimulation → IFN-β
           production) — reduced/absent confirms defect; confirmed by gene sequencing;
         Treatment: IV aciclovir for acute HSE; prophylaxis considered;
         seed SEED_BASE+5).
UNC93B1 (unc-93 homolog B1; 975aa; 11q13.3; AR;
         UNC93B1 deficiency — broadest endosomal TLR-trafficking defect → HSE and impaired antiviral
           immunity across multiple endosomal TLRs;
         UNC93B1 is a 12-transmembrane protein of the ER/endolysosomal compartment that is required
           for the folding and trafficking of endosomal TLRs (TLR3, TLR7, TLR8, TLR9) from the ER
           to endolysosomes where they survey incoming endocytosed material for nucleic acids;
         Without UNC93B1: TLR3, TLR7, TLR8, AND TLR9 are all retained in the ER and degraded →
           COMPLETE ABSENCE of endosomal TLR-mediated signalling across all 4 TLR types; this is the
           BROADEST ENDOSOMAL TLR DEFECT in the panel and affects BOTH dsRNA (TLR3) AND ssRNA (TLR7/8)
           AND CpG-DNA (TLR9) sensing;
         PHENOTYPE: primary clinical manifestation is herpes simplex encephalitis (same as TLR3/IRF3
           deficiency) because TLR3 is the critical CNS antiviral receptor; additionally, impaired
           TLR7/9 → reduced plasmacytoid DC (pDC) type I IFN responses to systemic viral infections;
         AUTOSOMAL RECESSIVE: the first UNC93B1-HSE patients described were homozygous — these were
           the FIRST reported genetic cause of isolated HSE susceptibility (Casanova lab, 2006 Cell);
           historically important as proof-of-concept that a single TLR can be essential for CNS defence
           against a specific pathogen in an otherwise healthy host;
         Treatment: IV aciclovir for acute HSE; prophylaxis considered; otherwise systemic immunity
           is sufficient for most other infections;
         seed SEED_BASE+6).
IRF7   (interferon regulatory factor 7; 457aa; 11p15.5; AR;
         IRF7 deficiency — selective susceptibility to LIFE-THREATENING INFLUENZA;
         IRF7 is the 'master regulator of type I IFN amplification' — after initial IRF3-driven IFN-β
           production, IFN-β signals in an autocrine/paracrine loop through IFNAR1/IFNAR2 → JAK1/TYK2
           → STAT1/STAT2 → IRF9 → ISGF3 → induction of IRF7 → IRF7 then drives massive amplification
           of IFN-α subtypes (IFN-α1-α14) that provide systemic antiviral protection;
         Without IRF7: the IFN-α amplification loop fails → initial IFN-β (IRF3-driven) insufficient
           for control → influenza A virus spreads systemically;
         PATHOGNOMONIC: LIFE-THREATENING INFLUENZA IN A HEALTHY PREVIOUSLY NORMAL INDIVIDUAL —
           a previously healthy child hospitalised with fulminant influenza A pneumonitis + respiratory
           failure + multiorgan failure should raise suspicion for IRF7 deficiency; the FIRST described
           IRF7-deficient patient (Casanova lab, 2015 Science) was a healthy 2.5-year-old girl who
           died from H1N1 influenza despite normal immune development in all other respects;
         SELECTIVE TO INFLUENZA: unlike UNC93B1 (broad endosomal TLR) or IRF3/TLR3 (HSE selective),
           IRF7 deficiency affects the IFN amplification loop which is SPECIFICALLY critical for
           influenza A containment in lungs; most other viral infections are handled by the initial
           IRF3-driven IFN-β or NK-cell/CTL responses;
         CRITICAL CLINICAL IMPLICATION: annual influenza vaccination MANDATORY; IV oseltamivir
           immediately if influenza suspected; consider prophylactic oseltamivir during flu season;
           IRF7 deficiency is one of the reasons why paediatric critical influenza should trigger
           genetic workup;
         Treatment: oseltamivir IV for acute influenza; annual vaccination; prophylactic oseltamivir
           in flu season; type I IFN therapy (IFN-α) as adjunct in critical illness;
         seed SEED_BASE+7).
"""

import random
from typing import Any

SEED_BASE = 3886  # seeds 3886-3893

GENES = ["IRAK4", "MYD88", "IKBKG", "NFKBIA", "IRF3", "TLR3", "UNC93B1", "IRF7"]

GENE_META = {
    "IRAK4":   {"locus": "12q12",   "aa": 460,  "kda": 52.0,  "inh": "AR",      "syndrome": "IRAK4-deficiency — Pyogenic-Bacteria-No-Fever-No-CRP-PATHOGNOMONIC",          "mim_gene": 606883, "mim_disease": 607676},
    "MYD88":   {"locus": "3p22.2",  "aa": 296,  "kda": 33.0,  "inh": "AR",      "syndrome": "MyD88-deficiency — Identical-IRAK4-Invasive-Bacteria-No-Systemic-Inflammation", "mim_gene": 602170, "mim_disease": 612260},
    "IKBKG":   {"locus": "Xq28",    "aa": 419,  "kda": 48.0,  "inh": "XL",      "syndrome": "NEMO — XL-EDA-ID-Anhydrosis-Conical-Teeth-Mycobacteria-Candida-PATHOGNOMONIC", "mim_gene": 300248, "mim_disease": 300291},
    "NFKBIA":  {"locus": "14q13.2", "aa": 317,  "kda": 35.6,  "inh": "AD-GOF",  "syndrome": "IκBα AD-GOF — EDA-ID-More-Severe-NEMO-EBV-CMV-Mycobacteria-Herpesvirus",       "mim_gene": 164008, "mim_disease": 612132},
    "IRF3":    {"locus": "19q13.33","aa": 427,  "kda": 47.0,  "inh": "AD-LOF",  "syndrome": "IRF3-haploinsufficiency — HSE-Susceptibility-Type-I-IFN-Defect-CNS",            "mim_gene": 603734, "mim_disease": 614850},
    "TLR3":    {"locus": "4q35.1",  "aa": 904,  "kda": 103.8, "inh": "AD/AR",   "syndrome": "TLR3-deficiency — Herpes-Simplex-Encephalitis-PATHOGNOMONIC-Children-3-15y",   "mim_gene": 603029, "mim_disease": 614850},
    "UNC93B1": {"locus": "11q13.3", "aa": 975,  "kda": 110.0, "inh": "AR",      "syndrome": "UNC93B1-deficiency — All-Endosomal-TLR3-7-8-9-Absent-HSE-First-Described-2006","mim_gene": 608204, "mim_disease": 610551},
    "IRF7":    {"locus": "11p15.5", "aa": 457,  "kda": 51.5,  "inh": "AR",      "syndrome": "IRF7-deficiency — Life-Threatening-Influenza-PATHOGNOMONIC-IFN-Amplification",  "mim_gene": 605047, "mim_disease": 615953},
}

DEFINITIONS = {
    "IRAK4":   "IL-1R-associated kinase-4 (460aa / 52kDa / 12q12 / AR). Myddosome proximal kinase; downstream of all TLR-MyD88 and IL-1R/IL-18R/IL-33R. LOF → invasive Gram-positive infections (S. pneumoniae, S. aureus) WITHOUT fever, CRP, or ESR — absent systemic inflammation is PATHOGNOMONIC. Improves markedly after age 14 (TLR-independent immunity). Penicillin prophylaxis + pneumococcal vaccination throughout childhood.",
    "MYD88":   "MyD88 TIR-domain adaptor (296aa / 33kDa / 3p22.2 / AR). Universal TLR-IL-1R adaptor; IRAK-4 recruiter. Germline LOF → identical phenotype to IRAK4 deficiency (invasive pyogenic bacteria without fever). Note: somatic GOF p.Leu265Pro is a lymphoma driver (DLBCL/WM) — same gene, opposite biology. Improves by adulthood. Penicillin prophylaxis + vaccination.",
    "IKBKG":   "NEMO IKK regulatory subunit (419aa / 48kDa / Xq28 / X-linked). Canonical NF-κB essential modulator; absent → EDA-ID triad: anhydrosis (hyperthermia risk) + sparse hair + conical teeth PATHOGNOMONIC + immunodeficiency (Mycobacteria, Candida, pneumococcus). BCG CONTRAINDICATED. HSCT corrects immune component only. Females: incontinentia pigmenti phenotype.",
    "NFKBIA":  "IκBα NF-κB inhibitor (317aa / 36kDa / 14q13.2 / AD-GOF). GOF variants resist IKK phosphorylation → NF-κB permanently inhibited. EDA-ID triad (same as NEMO) + MORE SEVERE: EBV, CMV, VZV fatal infections; mycobacteria; Candida. AD: both sexes. HSCT more frequently used than for NEMO.",
    "IRF3":    "Interferon regulatory factor 3 (427aa / 47kDa / 19q13.33 / AD-LOF). TLR3/TRIF/TBK1 effector; drives immediate IFN-β in CNS cells. Haploinsufficiency → HSE in otherwise healthy host. Systemic immunity intact. Aciclovir IV for acute HSE; long-term prophylaxis in confirmed cases.",
    "TLR3":    "Toll-like receptor 3 (904aa / 104kDa / 4q35.1 / AD or AR). Endosomal dsRNA sensor; signals via TRIF (not MyD88) → IRF3 → IFN-β. Critical for CNS HSV-1 detection. AD-LOF (dominant-negative on TLR3 dimerisation) or AR. PATHOGNOMONIC: first/recurrent HSE in children (3–15y). Most common genetic HSE cause. Poly-I:C stimulation assay in fibroblasts diagnostic.",
    "UNC93B1": "Endosomal TLR chaperone (975aa / 110kDa / 11q13.3 / AR). Traffics TLR3, TLR7, TLR8, TLR9 from ER to endolysosomes. LOF → ALL four endosomal TLRs absent. Broadest endosomal defect. Primary phenotype: HSE (TLR3 loss). First genetic HSE cause described (Casanova 2006 Cell). Systemic immunity otherwise preserved.",
    "IRF7":    "Interferon regulatory factor 7 (457aa / 52kDa / 11p15.5 / AR). Master IFN-α amplification regulator. AR-LOF → IFN-α amplification loop failure → influenza A uncontrolled. PATHOGNOMONIC: life-threatening influenza in previously healthy child. Annual flu vaccine MANDATORY. Oseltamivir IV immediately for influenza; prophylaxis in flu season. First case described 2015 Science.",
}


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def _gene_cohort(gene: str, seed: int) -> dict[str, Any]:
    rng = _rng(seed)
    n = 40
    meta = GENE_META[gene]

    if gene == "IRAK4":
        mean_onset = rng.gauss(18, 6); mean_dx_lag = rng.gauss(2.5, 1.0)
        invasive_bacterial_n = rng.randint(38, 40); no_fever_n = rng.randint(34, 38)
        pneumococcal_n = rng.randint(28, 36); staph_n = rng.randint(18, 26)
        eda_triad_n = 0; mycobacterial_n = rng.randint(2, 6)
        hse_n = 0; influenza_severe_n = 0; age_improvement_n = rng.randint(30, 38)
        pathognomonic = "Invasive pyogenic infections WITHOUT fever/CRP/ESR (absent systemic inflammation); improves markedly after age 14"
    elif gene == "MYD88":
        mean_onset = rng.gauss(20, 7); mean_dx_lag = rng.gauss(2.8, 1.1)
        invasive_bacterial_n = rng.randint(36, 40); no_fever_n = rng.randint(32, 38)
        pneumococcal_n = rng.randint(26, 34); staph_n = rng.randint(16, 24)
        eda_triad_n = 0; mycobacterial_n = rng.randint(2, 6)
        hse_n = 0; influenza_severe_n = 0; age_improvement_n = rng.randint(30, 38)
        pathognomonic = "Identical to IRAK4: invasive bacteria without systemic inflammation; somatic GOF same gene = lymphoma driver (opposite biology)"
    elif gene == "IKBKG":
        mean_onset = rng.gauss(6, 3); mean_dx_lag = rng.gauss(1.5, 0.6)
        invasive_bacterial_n = rng.randint(28, 36); no_fever_n = rng.randint(16, 24)
        pneumococcal_n = rng.randint(20, 28); staph_n = rng.randint(10, 18)
        eda_triad_n = rng.randint(36, 40); mycobacterial_n = rng.randint(20, 30)
        hse_n = rng.randint(4, 10); influenza_severe_n = 0; age_improvement_n = rng.randint(8, 16)
        pathognomonic = "HED triad (anhydrosis + sparse hair + conical teeth) PATHOGNOMONIC; BCG ABSOLUTELY CI; X-linked males only"
    elif gene == "NFKBIA":
        mean_onset = rng.gauss(6, 3); mean_dx_lag = rng.gauss(1.2, 0.5)
        invasive_bacterial_n = rng.randint(28, 36); no_fever_n = rng.randint(14, 22)
        pneumococcal_n = rng.randint(18, 26); staph_n = rng.randint(12, 20)
        eda_triad_n = rng.randint(36, 40); mycobacterial_n = rng.randint(24, 34)
        hse_n = rng.randint(8, 16); influenza_severe_n = rng.randint(4, 10); age_improvement_n = rng.randint(6, 12)
        pathognomonic = "EDA-ID triad + MORE SEVERE than NEMO: fatal herpesvirus infections (EBV/CMV/VZV); AD both sexes; HSCT more commonly needed"
    elif gene == "IRF3":
        mean_onset = rng.gauss(72, 36); mean_dx_lag = rng.gauss(0.5, 0.2)
        invasive_bacterial_n = rng.randint(2, 6); no_fever_n = 0
        pneumococcal_n = 0; staph_n = 0
        eda_triad_n = 0; mycobacterial_n = 0
        hse_n = rng.randint(36, 40); influenza_severe_n = 0; age_improvement_n = 0
        pathognomonic = "HERPES SIMPLEX ENCEPHALITIS in otherwise healthy host; systemic immunity intact; CNS-specific IFN-β defect; aciclovir lifelong prophylaxis"
    elif gene == "TLR3":
        mean_onset = rng.gauss(60, 30); mean_dx_lag = rng.gauss(0.4, 0.2)
        invasive_bacterial_n = rng.randint(2, 6); no_fever_n = 0
        pneumococcal_n = 0; staph_n = 0
        eda_triad_n = 0; mycobacterial_n = 0
        hse_n = rng.randint(36, 40); influenza_severe_n = 0; age_improvement_n = 0
        pathognomonic = "HSE PATHOGNOMONIC in children 3-15y; MOST COMMON genetic HSE cause; poly-I:C fibroblast assay diagnostic; AD or AR"
    elif gene == "UNC93B1":
        mean_onset = rng.gauss(66, 30); mean_dx_lag = rng.gauss(0.5, 0.2)
        invasive_bacterial_n = rng.randint(4, 10); no_fever_n = 0
        pneumococcal_n = 0; staph_n = 0
        eda_triad_n = 0; mycobacterial_n = 0
        hse_n = rng.randint(34, 40); influenza_severe_n = rng.randint(4, 10); age_improvement_n = 0
        pathognomonic = "Broadest endosomal TLR defect (TLR3+7+8+9 all absent); primary phenotype HSE; first described genetic HSE cause (Casanova 2006)"
    else:  # IRF7
        mean_onset = rng.gauss(24, 12); mean_dx_lag = rng.gauss(0.3, 0.1)
        invasive_bacterial_n = rng.randint(2, 6); no_fever_n = 0
        pneumococcal_n = 0; staph_n = 0
        eda_triad_n = 0; mycobacterial_n = 0
        hse_n = 0; influenza_severe_n = rng.randint(36, 40); age_improvement_n = 0
        pathognomonic = "LIFE-THREATENING INFLUENZA PATHOGNOMONIC (previously healthy child); annual flu vaccine MANDATORY; oseltamivir prophylaxis in flu season"

    return {
        "gene": gene,
        "syndrome": meta["syndrome"],
        "locus": meta["locus"],
        "aa": meta["aa"],
        "kda": meta["kda"],
        "inh": meta["inh"],
        "n": n,
        "mean_onset_months": max(0.5, mean_onset),
        "mean_dx_lag_yrs": max(0.1, mean_dx_lag),
        "invasive_bacterial_n": invasive_bacterial_n,
        "no_fever_response_n": no_fever_n,
        "pneumococcal_n": pneumococcal_n,
        "staph_n": staph_n,
        "eda_triad_n": eda_triad_n,
        "mycobacterial_n": mycobacterial_n,
        "hse_n": hse_n,
        "influenza_severe_n": influenza_severe_n,
        "age_improvement_n": age_improvement_n,
        "pathognomonic": pathognomonic,
        "seed": seed,
    }


def generate_overview() -> dict:
    cohorts = [_gene_cohort(g, SEED_BASE + i) for i, g in enumerate(GENES)]
    total_n = sum(c["n"] for c in cohorts)
    return {
        "atlas": "Hereditary-Innate-Immunity-TLR-NF-kB-Atlas",
        "title": "Complete 8-Gene Innate Immunity TLR-NF-κB & IFN Pathway Immunodeficiency Reference",
        "subtitle": "TLR-MyD88-NF-κB axis (IRAK4/MYD88/NEMO/IκBα) and Type-I IFN axis (IRF3/TLR3/UNC93B1/IRF7)",
        "genes": GENES,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "total_patients": total_n,
        "cohort_per_gene": 40,
        "disease_groups": {
            "NF-kB_axis": ["IRAK4", "MYD88", "IKBKG", "NFKBIA"],
            "Type_I_IFN_axis": ["IRF3", "TLR3", "UNC93B1", "IRF7"],
        },
        "nfkb_axis_phenotype": "Invasive pyogenic bacterial infections (IRAK4/MYD88: no inflammation; NEMO/IκBα: +ectodermal dysplasia)",
        "ifn_axis_phenotype": "Viral encephalitis (IRF3/TLR3/UNC93B1: herpes simplex encephalitis) or severe influenza (IRF7)",
        "age_improvement_hallmark": "IRAK4 and MYD88: marked improvement by age 14 — TLR-independent adaptive immunity compensates",
        "key_clinical_clues": [
            "Pyogenic bacteraemia WITHOUT fever/CRP/ESR → IRAK4 or MYD88",
            "Recurrent invasive pneumococcal disease → IRAK4/MYD88 first then NEMO/IκBα",
            "Anhydrosis + conical teeth + ectodermal dysplasia → NEMO (XL) or IκBα (AD)",
            "BCG/NTM disease + ectodermal features → NEMO or IκBα",
            "Herpes simplex encephalitis (otherwise healthy child) → TLR3 > UNC93B1 > IRF3",
            "Life-threatening influenza A in previously healthy child → IRF7",
            "Fatal EBV/CMV/VZV + ectodermal dysplasia → IκBα/NFKBIA (more severe than NEMO)",
        ],
        "diagnostic_workup": [
            "IRAK4/MYD88: whole blood TLR stimulation assay (TLR4 LPS, IL-1β) → IL-6/TNF production; gene sequencing",
            "NEMO/IκBα: skin biopsy (sweat gland count/structure); NF-κB activation assay in fibroblasts; gene sequencing",
            "TLR3/IRF3/UNC93B1: fibroblast poly-I:C stimulation → IFN-β ELISA; flow cytometry TLR3 expression",
            "IRF7: IFN-α multiplex in plasma during acute viral illness; IFN-α induction assay post-stimulation",
            "UNC93B1: TLR7/9 function (pDC IFN-α response to R848/CpG) in addition to TLR3",
            "Gene panel: IRAK4, MYD88, IKBKG (NEMO), NFKBIA, IRF3, TLR3, UNC93B1, IRF7",
        ],
        "cohort_summary": cohorts,
    }


def generate_breakdown() -> dict:
    cohorts = [_gene_cohort(g, SEED_BASE + i) for i, g in enumerate(GENES)]
    return {
        "atlas": "Hereditary-Innate-Immunity-TLR-NF-kB-Atlas",
        "breakdown_type": "per-gene cohort n=40 each",
        "genes": GENES,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "cohorts": cohorts,
        "phenotype_comparison": [
            {"gene": "IRAK4",   "axis": "NF-κB",   "pyogenic_bacteria": "+++", "no_inflammation": "+++", "eda_triad": "−",   "mycobacteria": "±",  "hse": "−",   "flu": "−",   "age_improve": "+++", "unique": "Invasive S. pneumoniae/S. aureus WITHOUT fever; improves by age 14; penicillin prophylaxis"},
            {"gene": "MYD88",   "axis": "NF-κB",   "pyogenic_bacteria": "+++", "no_inflammation": "+++", "eda_triad": "−",   "mycobacteria": "±",  "hse": "−",   "flu": "−",   "age_improve": "+++", "unique": "Identical to IRAK4; somatic GOF same gene = lymphoma driver (clinically opposite)"},
            {"gene": "IKBKG",   "axis": "NF-κB",   "pyogenic_bacteria": "++",  "no_inflammation": "+",   "eda_triad": "+++", "mycobacteria": "+++","hse": "±",   "flu": "−",   "age_improve": "+",   "unique": "NEMO X-linked; HED triad PATHOGNOMONIC; BCG ABSOLUTELY CI; anhydrosis → hyperthermia fatal"},
            {"gene": "NFKBIA",  "axis": "NF-κB",   "pyogenic_bacteria": "++",  "no_inflammation": "+",   "eda_triad": "+++", "mycobacteria": "+++","hse": "++",  "flu": "±",   "age_improve": "+",   "unique": "AD-GOF IκBα; MORE SEVERE than NEMO; EBV/CMV/VZV fatal; HSCT more commonly needed"},
            {"gene": "IRF3",    "axis": "Type-I-IFN","pyogenic_bacteria": "−",  "no_inflammation": "−",  "eda_triad": "−",   "mycobacteria": "−",  "hse": "+++", "flu": "−",   "age_improve": "−",   "unique": "HSE in otherwise healthy host; CNS-selective IFN-β defect; systemic immunity intact"},
            {"gene": "TLR3",    "axis": "Type-I-IFN","pyogenic_bacteria": "−",  "no_inflammation": "−",  "eda_triad": "−",   "mycobacteria": "−",  "hse": "+++", "flu": "−",   "age_improve": "−",   "unique": "MOST COMMON genetic HSE cause; children 3-15y; AD (dominant-neg) or AR; poly-I:C assay diagnostic"},
            {"gene": "UNC93B1", "axis": "Type-I-IFN","pyogenic_bacteria": "±",  "no_inflammation": "−",  "eda_triad": "−",   "mycobacteria": "−",  "hse": "+++", "flu": "±",   "age_improve": "−",   "unique": "Broadest defect (TLR3+7+8+9); first described HSE gene 2006; AR; pDC IFN-α also impaired"},
            {"gene": "IRF7",    "axis": "Type-I-IFN","pyogenic_bacteria": "−",  "no_inflammation": "−",  "eda_triad": "−",   "mycobacteria": "−",  "hse": "−",   "flu": "+++", "age_improve": "−",   "unique": "LIFE-THREATENING INFLUENZA PATHOGNOMONIC; IFN-α amplification absent; annual vaccine MANDATORY"},
        ],
        "gene_metadata": GENE_META,
    }


def generate_definitions() -> dict:
    return {
        "atlas": "Hereditary-Innate-Immunity-TLR-NF-kB-Atlas",
        "definitions": DEFINITIONS,
        "gene_metadata": GENE_META,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "key_distinguishers": {
            "IRAK4_vs_MYD88": "Clinically indistinguishable; both AR pyogenic bacteria without inflammation; MYD88 somatic GOF (p.Leu265Pro) drives DLBCL/WM — unrelated to germline LOF immunodeficiency",
            "NEMO_vs_IkBa": "Both cause EDA-ID; NEMO X-linked males, IκBα AD both sexes; IκBα MORE SEVERE (fatal herpesvirus infections, more HSCT needed)",
            "TLR3_vs_UNC93B1": "Both HSE; TLR3 most common genetic HSE cause (children); UNC93B1 broader (TLR3+7+8+9 all absent) + first described 2006",
            "IRF3_vs_TLR3": "Both AD-LOF causing HSE; TLR3 upstream sensor; IRF3 downstream transcription factor; mechanistically sequential",
            "IRF7_unique": "Only gene in panel selective for influenza A; all others affect HSE or bacteria; IRF7 = IFN-α amplification master regulator",
            "NF-kB_axis_bacteria": "IRAK4/MYD88: S. pneumoniae dominant (no fever); NEMO/IκBα: mycobacteria + Candida + bacteria (with ectodermal features)",
            "IFN_axis_viruses": "TLR3/UNC93B1/IRF3 → HSE (HSV-1 CNS specific); IRF7 → influenza (systemic IFN amplification)",
            "Age_improvement": "IRAK4 and MYD88 ONLY — infections improve markedly by age 14; NEMO/IκBα/IRF-axis defects do NOT improve with age",
            "BCG_CI": "IKBKG/NEMO and NFKBIA/IκBα: BCG ABSOLUTELY CONTRAINDICATED (fatal BCG disease in both)",
            "Ectodermal_clue": "Any patient with conical teeth + sparse hair + absent sweat → must test NEMO and IκBα immediately",
        },
        "diagnostic_pathway": [
            "Step 1: Clinical phenotype sorting — pyogenic bacteria without fever → IRAK4/MYD88; ectodermal dysplasia → NEMO/IκBα; HSE healthy child → TLR3/UNC93B1/IRF3; severe influenza → IRF7",
            "Step 2: NF-κB axis workup: whole blood TLR4/IL-1β stimulation (IL-6/TNF absent = IRAK4/MYD88); skin exam (anhydrosis/conical teeth = NEMO/IκBα)",
            "Step 3: IFN-axis workup: fibroblast poly-I:C stimulation (IFN-β absent = TLR3/UNC93B1/IRF3); pDC CpG/R848 stimulation (IFN-α absent = UNC93B1 also affects TLR7/9)",
            "Step 4: Gene sequencing panel: IRAK4, MYD88, IKBKG, NFKBIA, IRF3, TLR3, UNC93B1, IRF7",
            "Step 5: IRAK4/MYD88 confirmed → penicillin prophylaxis + pneumococcal vaccines; counsel re age-improvement by 14",
            "Step 6: NEMO/IκBα confirmed → BCG CONTRAINDICATED flag immediately; HSCT evaluation; IVIG",
            "Step 7: HSE-gene confirmed → lifelong aciclovir prophylaxis; neurology monitoring for sequelae",
            "Step 8: IRF7 confirmed → annual flu vaccine MANDATORY; oseltamivir on hand; prophylactic oseltamivir flu season",
        ],
    }
