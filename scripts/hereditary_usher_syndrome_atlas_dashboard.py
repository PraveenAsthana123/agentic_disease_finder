#!/usr/bin/env python3
"""Hereditary-Usher-Syndrome-Atlas — Complete 8-Gene Atlas
(MYO7A · USH2A · CDH23 · PCDH15 · ADGRV1 · CLRN1 · WHRN · SANS).

MYO7A    (Myosin VIIa; 2314 aa; ~254 kDa; 11q13.5; AR;
           Usher syndrome type 1B (USH1B) — MOST COMMON USH1 ~40-55%;
           Congenital PROFOUND SNHL (flat audiogram, deaf at birth);
           RP onset: ERG extinguished by age 2-3, symptoms late childhood;
           VESTIBULAR DYSFUNCTION: absent caloric responses, delayed walking ~18 months;
           Cochlear implant: EXCELLENT OUTCOMES;
           p.Arg245Leu Acadian founder; seed SEED_BASE+0).
USH2A    (Usherin; 5202 aa; ~570 kDa; 1q41; AR;
           Usher syndrome type 2A (USH2A) — MOST COMMON OVERALL ~40% of all Usher;
           Moderate-severe HIGH-FREQUENCY SNHL sloping audiogram;
           NORMAL VESTIBULAR FUNCTION (key DDx from USH1);
           RP onset ERG changes teens, symptoms 20s-30s;
           c.2299delG European founder ~30% of USH2A alleles;
           seed SEED_BASE+1).
CDH23    (Cadherin-23; 3354 aa; ~360 kDa; 10q22.1; AR;
           Usher syndrome type 1D (USH1D) ~20% of USH1 + DFNB12 (non-syndromic, milder alleles);
           TIP LINK UPPER END — CDH23 anchors upper, PCDH15 lower;
           Congenital profound SNHL + early RP + vestibular dysfunction;
           Truncating → USH1; hypomorphic missense → DFNB12 (no RP);
           seed SEED_BASE+2).
PCDH15   (Protocadherin-15; 1955 aa; ~215 kDa; 10q21.1; AR;
           Usher syndrome type 1F (USH1F) ~15-20% of USH1 + DFNB23 (non-syndromic);
           TIP LINK LOWER END — PCDH15 anchors lower end;
           Gypsy/Roma founder p.Arg929Stop CD2-isoform specific;
           Cochlear implants HIGHLY EFFECTIVE for hearing;
           seed SEED_BASE+3).
ADGRV1   (Adhesion GPCR V1 / VLGR1 / GPR98; 6307 aa; ~692 kDa; 5q14.3; AR;
           Usher syndrome type 2C (USH2C) ~15-20% of USH2 (2nd most common USH2);
           LARGEST HUMAN PROTEIN: 6307 aa / 692 kDa — PATHOGNOMONIC teaching pearl;
           Moderate-severe HF SNHL; NORMAL VESTIBULAR; RP onset teens-20s;
           seed SEED_BASE+4).
CLRN1    (Clarin-1; 232 aa; ~26 kDa; 3q25.1; AR;
           Usher syndrome type 3A (USH3A) — PROGRESSIVE SNHL (unlike USH1/2 congenital);
           PROGRESSIVE RP; variable vestibular (unlike USH1 always absent);
           p.Asn48Lys Finnish founder (1/5000 Finnish births); p.Tyr176Ser Ashkenazi founder;
           Hearing aids temporally effective before cochlear implant needed;
           seed SEED_BASE+5).
WHRN     (Whirlin/DFNB31; 907 aa; ~100 kDa; 9q32; AR;
           Usher syndrome type 2D (USH2D) — rarest USH2 subtype;
           Also DFNB31 non-syndromic HL with severe truncating alleles in some;
           HF SNHL + RP, NORMAL VESTIBULAR; PDZ scaffold protein at stereocilia tip;
           seed SEED_BASE+6).
SANS     (ANKS4B / USH1G; 461 aa; ~52 kDa; 17q25.1; AR;
           Usher syndrome type 1G (USH1G) — rarest USH1;
           Congenital profound SNHL + early RP + vestibular dysfunction;
           SANS scaffold at stereocilia tip links CDH23/PCDH15 complex assembly;
           p.Arg245_Ile248del German founder;
           seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 2430-2437).
"""

import random

SEED_BASE = 2430

USH_GENES = [
    # -- MYO7A -- Usher 1B (Most Common USH1 ~40-55%) -----------------------------------------------
    {
        "gene": "MYO7A",
        "alt_name": (
            "MYO7A (MYO7A-2314aa-11q13.5 / AR -- "
            "USH1B-MOST-COMMON-USH1-~40-55pct-ALL-USH1-CASES -- "
            "CONGENITAL-PROFOUND-SNHL-FLAT-AUDIOGRAM-DEAF-AT-BIRTH -- "
            "RP-ONSET-ERG-EXTINGUISHED-AGE-2-3-SYMPTOMS-LATE-CHILDHOOD -- "
            "VESTIBULAR-DYSFUNCTION-ABSENT-CALORIC-RESPONSES-DELAYED-WALKING-18-MONTHS -- "
            "COCHLEAR-IMPLANT-EXCELLENT-OUTCOMES-FIRST-CHOICE -- "
            "pArg245Leu-ACADIAN-FOUNDER-MUTATION)"
        ),
        "protein": (
            "MYO7A -- 11q13.5 AR -- MYO7A-2314aa -- "
            "Myosin-VIIa-Motor-Protein-254kDa-Unconventional-Myosin-Class-VII -- "
            "N-Terminal-Motor-Domain-IQ-Motifs-Coiled-Coil-MyTH4-FERM-SH3 -- "
            "Expressed-Cochlear-Hair-Cells-AND-RPE-Photoreceptors-Dual-Sensory -- "
            "OMIM-Gene-276903-Disease-USH1B-276900"
        ),
        "locus": "11q13.5",
        "protein_size": "2314 aa / 254 kDa",
        "inheritance": (
            "AR (autosomal recessive); MYO7A encodes myosin VIIa, an unconventional class VII myosin motor protein. "
            "DOMAIN STRUCTURE: N-terminal motor domain (ATPase, actin-binding) → 5 IQ motifs (calmodulin-binding) → "
            "coiled-coil → 2× MyTH4-FERM-SH3 tandem repeats in tail; "
            "COCHLEAR EXPRESSION: MYO7A is essential for stereocilia cohesion and tip-link tension; "
            "expressed in outer and inner hair cell stereocilia; MYO7A LOF → disordered stereocilia bundle → "
            "profound mechanosensory failure → congenital profound SNHL; "
            "RETINAL EXPRESSION: MYO7A expressed in RPE (melanosome transport, opsin transport) and photoreceptors; "
            "MYO7A LOF in RPE → delayed rhodopsin clearance → progressive photoreceptor degeneration → RP; "
            "MOST COMMON USH1 GENE: ~40-55% of all Usher syndrome type 1 cases; "
            "FOUNDER: p.Arg245Leu (c.734G>T) — Acadian (French-Canadian/Louisiana Cajun ancestry); "
            "p.Gln36Stop, p.Arg302Stop common truncating in pan-ethnic populations"
        ),
        "disease_category": (
            "Usher syndrome type 1B (USH1B, AR) — most common USH1 (~40-55%); "
            "triad: congenital profound SNHL + early-onset RP + vestibular areflexia; "
            "cochlear implantation highly effective"
        ),
        "disease_pathway": (
            "MYO7A IN STEREOCILIA: myosin VIIa provides the motor force maintaining tip-link tension in stereocilia. "
            "TIP-LINK COMPLEX: CDH23 (upper end) - PCDH15 (lower end) form the tip link; "
            "MYO7A interacts with the tip-link upper end complex (via SANS/ANKS4B scaffold). "
            "MYO7A LOF → stereocilia disorganisation, loss of tip-link tension → "
            "mechanotransduction channels (TMC1/TMC2) cannot open → no K+ influx → no hair cell signal → "
            "congenital profound SNHL (flat audiogram). "
            "MYO7A IN RPE: transports melanosomes apically; moves rhodopsin from inner to outer segment disk; "
            "MYO7A LOF in RPE → rhodopsin accumulates at base of outer segment → phagocytosis impaired → "
            "oxidative stress → photoreceptor degeneration → RP. "
            "DUAL TISSUE EXPRESSION explains dual sensory loss; "
            "GENE THERAPY: subretinal AAV-MYO7A trials (Sanofi/Horama); "
            "cochlear implant rehabilitates hearing (SNHL is conductive/sensorineural, CI bypasses hair cells). "
            "VESTIBULAR: MYO7A also expressed in vestibular hair cells → absent caloric responses → "
            "delayed ambulation (~18 months); balance compensated by vision."
        ),
        "pathognomonic": (
            "USH1 CLINICAL TRIAD (MYO7A/USH1B — MOST COMMON USH1): "
            "1. CONGENITAL PROFOUND SNHL: flat audiogram (all frequencies ≥90 dBHL); "
            "newborn hearing screen FAILS; deaf at birth; "
            "2. EARLY-ONSET RP: ERG abnormal by age 2-3 years (extinguished rod and cone responses); "
            "night blindness onset late childhood/early teens (nyctalopia); "
            "tunnel vision by 20s-30s; legally blind typically by 40-50; "
            "3. VESTIBULAR AREFLEXIA: absent caloric responses (bilaterally); "
            "DELAYED MOTOR MILESTONES: walking at ~18 months (vs 12 months normal); "
            "tendency to fall in dark (visual compensation for vestibular loss); "
            "KEY DDx FROM USH2: USH1 has PROFOUND (not moderate) SNHL + VESTIBULAR AREFLEXIA; "
            "ERG extinguished in INFANCY in USH1, not teens. "
            "COCHLEAR IMPLANT: ALL USH1 patients are CI candidates; outcomes excellent; "
            "implant before age 2 for best language outcomes."
        ),
        "treatment": (
            "COCHLEAR IMPLANTATION: FIRST-LINE for hearing rehabilitation; bilateral recommended; "
            "implant early (before age 2) for optimal speech-language development; "
            "HEARING AIDS: limited benefit given profound loss; bridge to CI; "
            "RETINAL: no approved treatment 2026; vitamin A palmitate 15,000 IU/day (Level B, field-limited evidence); "
            "AVOID vitamin E (may accelerate RP); "
            "GENE THERAPY: subretinal AAV2-MYO7A trials (HMR59, Sanofi/Horama) in progress; "
            "ORIENTATION AND MOBILITY TRAINING: essential given combined deaf-blindness; "
            "tactile sign language / deafblind manual alphabet as vision fails; "
            "VESTIBULAR REHABILITATION: balance training early; avoid high-risk environments in dark; "
            "ANNUAL REVIEW: audiologist + ophthalmologist + low-vision specialist; "
            "GENETIC COUNSELLING: AR 25% risk per sibling; carrier testing for founder mutations."
        ),
        "key_features": [
            "Congenital profound SNHL (flat audiogram); fails newborn hearing screen",
            "RP onset: ERG extinguished by age 2-3; nyctalopia late childhood; tunnel vision 20s-30s",
            "Vestibular areflexia: absent caloric responses; delayed walking ~18 months",
            "Cochlear implant: EXCELLENT outcomes — implant before age 2",
            "p.Arg245Leu Acadian/French-Canadian founder mutation",
            "Dual expression: cochlear hair cells + RPE/photoreceptors",
            "MYO7A: largest USH1 gene (2314 aa); most common USH1 (~40-55%)",
            "Gene therapy trials ongoing (subretinal AAV-MYO7A)",
        ],
        "key_ddx": (
            "USH2A (USH2): HF SNHL (NOT profound) + NORMAL vestibular — key distinction; RP onset later (teens); "
            "CDH23 (USH1D): identical phenotype to USH1B; genetic test distinguishes; CDH23 DFNB12 milder alleles; "
            "PCDH15 (USH1F): identical phenotype to USH1B; Gypsy/Roma founder p.Arg929Stop; "
            "CLRN1 (USH3A): PROGRESSIVE (not congenital) SNHL — key DDx; vestibular variable; "
            "Nonsyndromic RP + SNHL (mitochondrial A1555G): maternal inheritance; check mitochondrial panel; "
            "SANS/USH1G: rarest USH1; identical triad; genetic panel required."
        ),
        "systemic_involvement": (
            "NO systemic (non-sensory) involvement in classic Usher syndrome. "
            "Dual sensory: SNHL (cochlear) + RP (retinal). Vestibular areflexia. "
            "Some MYO7A variants reported with additional features (rare). "
            "Psychosocial: deafblindness risk — refer to deaf-blind support services."
        ),
        "onset_age": "Congenital SNHL (deaf at birth); RP ERG changes age 2-3; night blindness late childhood",
        "surgical_urgency": "Cochlear implantation urgency: implant before age 2 for speech-language development",
        "gene_family": "Unconventional myosin motor protein family (Class VII); stereocilia and RPE motor",
        "morphology": (
            "AUDIOGRAM: flat profound loss (≥90 dBHL all frequencies); "
            "ERG: extinguished rod + cone responses from infancy; "
            "FUNDUS: RP changes — bone-spicule pigmentation, attenuated vessels, waxy disc pallor; "
            "OCT: outer nuclear layer thinning, photoreceptor loss"
        ),
        "n_patients": 40,
    },

    # -- USH2A -- Usher 2A (Most Common Overall ~40-50% of all Usher) --------------------------------
    {
        "gene": "USH2A",
        "alt_name": (
            "USH2A (USH2A-5202aa-1q41 / AR -- "
            "USH2A-MOST-COMMON-OVERALL-~40pct-ALL-USHER-SYNDROME -- "
            "MODERATE-SEVERE-HIGH-FREQUENCY-SNHL-SLOPING-AUDIOGRAM -- "
            "NORMAL-VESTIBULAR-FUNCTION-KEY-DDx-FROM-USH1 -- "
            "RP-ONSET-TEENS-SYMPTOMS-20s-30s-LATER-THAN-USH1 -- "
            "c.2299delG-EUROPEAN-FOUNDER-~30pct-USH2A-ALLELES)"
        ),
        "protein": (
            "USH2A -- 1q41 AR -- USH2A-5202aa -- "
            "Usherin-Long-Form-570kDa-LamG-EGF-FN3-PDZ-TM-Domains -- "
            "Basal-Body-Periciliary-Membrane-Photoreceptor-Calyceal-Processes -- "
            "Cochlear-Hair-Cell-Ankle-Link-Complex -- "
            "OMIM-Gene-608400-Disease-USH2A-276901"
        ),
        "locus": "1q41",
        "protein_size": "5202 aa / 570 kDa (long isoform, usherin-b)",
        "inheritance": (
            "AR (autosomal recessive); USH2A encodes usherin, a large extracellular matrix/scaffold protein. "
            "TWO ISOFORMS: short form (usherin-a, ~80 kDa) and long form (usherin-b, ~570 kDa); "
            "LONG FORM IS DISEASE-RELEVANT: contains LamG, EGF, FN3, PDZ-binding repeats + single TM domain; "
            "COCHLEAR FUNCTION: usherin localises to hair cell ankle-link complex at base of stereocilia; "
            "ankle links: transient lateral links maintaining stereocilia cohesion in development; "
            "USH2A LOF → ankle-link disruption → moderate-severe SNHL (NOT profound — key difference from USH1); "
            "RETINAL FUNCTION: usherin localises to photoreceptor calyceal processes (periciliary membrane); "
            "USH2A LOF → calyceal process disruption → OS disk shedding defective → progressive RP; "
            "MOST COMMON USH2 GENE: ~50-70% of USH2; ~40% of ALL Usher syndrome (all types); "
            "FOUNDER: c.2299delG (p.Glu767SerfsX21) — European ancestry ~30% of USH2A alleles; "
            "c.11864G>A (p.Cys3952Tyr), c.8559-2A>G also frequent"
        ),
        "disease_category": (
            "Usher syndrome type 2A (USH2A, AR) — most common Usher syndrome overall (~40%); "
            "moderate-severe HF SNHL + RP onset teens + NORMAL vestibular; "
            "c.2299delG European founder ~30% of alleles"
        ),
        "disease_pathway": (
            "USH2A COCHLEAR FUNCTION: usherin forms the ankle-link complex at the base of stereocilia. "
            "ANKLE LINKS: transient structural links (developmental + maintained at base); "
            "loss of ankle links → stereocilia cohesion reduced → "
            "high-frequency hair cells most vulnerable → sloping HF SNHL. "
            "SEVERITY COMPARISON: ankle-link loss (USH2) causes MODERATE-SEVERE loss vs "
            "tip-link disruption (USH1: CDH23/PCDH15) causing PROFOUND loss; "
            "explains why USH2 patients retain some residual hearing (hearing aids partially effective). "
            "USH2A RETINAL FUNCTION: usherin at photoreceptor calyceal processes "
            "(finger-like projections surrounding base of outer segment); "
            "calyceal process disruption → OS disk shedding and renewal impaired → "
            "oxidative photoreceptor stress → progressive rod > cone degeneration → RP. "
            "NORMAL VESTIBULAR: ankle links NOT essential for vestibular hair cells → "
            "normal caloric responses → patients can ride bicycles, normal gait "
            "(critical clinical distinction from USH1 vestibular areflexia). "
            "c.2299delG IMPACT: frameshift at exon 13 removes LamG domain → complete LOF → "
            "severe USH2 phenotype."
        ),
        "pathognomonic": (
            "USH2 CLINICAL PROFILE (USH2A — MOST COMMON): "
            "1. MODERATE-SEVERE HIGH-FREQUENCY SNHL: sloping audiogram (high frequencies worse); "
            "speech frequencies often partially preserved (hearing aids effective early); "
            "NOT profound (can distinguish speech with hearing aids); "
            "2. RP ONSET TEENS: ERG changes in early teens; nyctalopia onset teens; "
            "field constriction 20s-30s; central vision typically preserved until 40s-50s; "
            "LATER RP ONSET THAN USH1 (critical DDx); "
            "3. NORMAL VESTIBULAR FUNCTION: PATHOGNOMONIC DDx FROM USH1 — "
            "patients can ride bicycle, no delayed walking, normal caloric responses; "
            "KEY QUESTION: 'Can you ride a bicycle?' → YES in USH2, NO in USH1; "
            "c.2299delG FOUNDER: test this variant FIRST in European descent patients with USH2 phenotype; "
            "~30% of USH2A alleles are c.2299delG → simple, rapid targeted test."
        ),
        "treatment": (
            "HEARING AIDS: effective early (moderate-severe HF loss retains speech frequencies); "
            "COCHLEAR IMPLANTATION: when hearing aids inadequate; CI candidates — good outcomes; "
            "RETINAL: vitamin A palmitate 15,000 IU/day (discuss benefits/risks; avoid if liver disease); "
            "docosahexaenoic acid (DHA) supplementation: adjunctive (limited evidence); "
            "AVOID vitamin E high-dose; AVOID excessive bright light (sunglasses); "
            "GENE THERAPY: ProQR RNA therapy (QR-421a, sepofarsen for c.2299delG carriers) — Phase 2 trials; "
            "antisense oligonucleotide (ASO) targeting c.2299delG-generated premature termination codon; "
            "LOW VISION AIDS: as vision declines; orientation and mobility training; "
            "ANNUAL REVIEW: audiologist + ophthalmologist (ERG, visual fields, OCT); "
            "GENETIC COUNSELLING: AR 25%; c.2299delG carrier frequency ~1/72 in European population."
        ),
        "key_features": [
            "Moderate-severe HF SNHL sloping audiogram (NOT profound); hearing aids effective early",
            "NORMAL VESTIBULAR FUNCTION — key DDx from USH1; can ride bicycle",
            "RP onset teens (later than USH1 which is infancy-early childhood)",
            "c.2299delG European founder — test FIRST in European-descent USH2 patients",
            "Most common Usher syndrome overall (~40% of all Usher cases)",
            "QR-421a (sepofarsen) ASO therapy in Phase 2 trials for c.2299delG carriers",
            "USH2A: 5202 aa usherin — large scaffold at ankle links + calyceal processes",
            "Cochlear implant: effective when hearing aids no longer sufficient",
        ],
        "key_ddx": (
            "USH1 (MYO7A, CDH23, PCDH15): PROFOUND SNHL (not moderate) + VESTIBULAR AREFLEXIA; "
            "ADGRV1 (USH2C): identical USH2 phenotype; 2nd most common USH2; genetic test; LARGEST protein 6307aa; "
            "WHRN (USH2D): rarest USH2; identical phenotype; PDZ scaffold at stereocilia; "
            "CLRN1 (USH3A): PROGRESSIVE (not congenital) SNHL; vestibular variable; later onset; "
            "Nonsyndromic RP: no SNHL; ERG identical; USH2A panel needed; "
            "DFNB31 (WHRN): non-syndromic HL without RP; milder WHRN alleles."
        ),
        "systemic_involvement": (
            "NO systemic involvement. Dual sensory: SNHL + RP. Vestibular NORMAL. "
            "Psychosocial: progressive combined sensory loss — deafblind services as RP progresses."
        ),
        "onset_age": "SNHL congenital (present at birth, moderate-severe); RP ERG changes early teens, symptoms late teens-20s",
        "surgical_urgency": "Cochlear implant when hearing aids fail; no acute urgency",
        "gene_family": "Usherin — LamG-EGF-FN3 extracellular scaffold; ankle-link and calyceal process component",
        "morphology": (
            "AUDIOGRAM: sloping moderate-severe HF loss (speech frequencies partially spared); "
            "ERG: progressive rod > cone amplitude reduction, extinguished eventually; "
            "FUNDUS: classic RP — bone spicules, attenuated arteries, waxy pallor; "
            "OCT: outer nuclear layer thinning, photoreceptor loss peripheral to central"
        ),
        "n_patients": 40,
    },

    # -- CDH23 -- Usher 1D + DFNB12 -------------------------------------------------------------------
    {
        "gene": "CDH23",
        "alt_name": (
            "CDH23 (CDH23-3354aa-10q22.1 / AR -- "
            "USH1D-~20pct-ALL-USH1 -- "
            "TIP-LINK-UPPER-END-CDH23-ANCHORS-UPPER-PCDH15-ANCHORS-LOWER -- "
            "DFNB12-NON-SYNDROMIC-HL-MILDER-ALLELES-NO-RP -- "
            "CONGENITAL-PROFOUND-SNHL-EARLY-RP-VESTIBULAR-AREFLEXIA -- "
            "TRUNCATING-USH1-HYPOMORPHIC-DFNB12)"
        ),
        "protein": (
            "CDH23 -- 10q22.1 AR -- CDH23-3354aa -- "
            "Cadherin-23-360kDa-27-Extracellular-Cadherin-Repeats-EC1-EC27-Single-TM -- "
            "Stereocilia-Tip-Link-UPPER-End-Calcium-Dependent-Ectodomain -- "
            "CDH23-pHY-Tip-Link-Upper-PCDH15-Lower -- "
            "OMIM-Gene-605516-Disease-USH1D-601067-DFNB12-601386"
        ),
        "locus": "10q22.1",
        "protein_size": "3354 aa / 360 kDa",
        "inheritance": (
            "AR (autosomal recessive); CDH23 encodes cadherin-23, a large single-pass transmembrane protein "
            "with 27 extracellular cadherin (EC) repeats. "
            "TIP-LINK STRUCTURE: the stereocilia tip link connects adjacent rows of stereocilia. "
            "CDH23 ANCHORS THE UPPER END of the tip link: CDH23 homodimers (antiparallel) connect to "
            "PCDH15 homodimers at the lower end — the CDH23-PCDH15 heterodimer forms the tip link filament. "
            "UPPER INSERTION: CDH23 tip links insert into the CUTICULAR PLATE (upper stereocilium); "
            "MECHANOTRANSDUCTION: tip link tension opens TMC1/TMC2 channels at LOWER end (PCDH15 side); "
            "CDH23 LOF → tip link absent → no mechanotransduction → profound SNHL. "
            "GENOTYPE-PHENOTYPE: TRUNCATING variants → USH1D (profound SNHL + RP + vestibular); "
            "HYPOMORPHIC MISSENSE (e.g., p.Val1388Gly) → DFNB12 (non-syndromic HL, no RP); "
            "~20% of USH1 cases; CDH23 is second-largest gene in cochlea after PCDH15."
        ),
        "disease_category": (
            "Usher syndrome type 1D (USH1D, AR) — ~20% of USH1; congenital profound SNHL + early RP + vestibular areflexia; "
            "DFNB12: non-syndromic HL with milder CDH23 alleles (no RP)"
        ),
        "disease_pathway": (
            "CDH23 TIP-LINK MECHANISM: cadherin-23 EC1-EC2 repeats at N-terminus form the calcium-dependent "
            "tip-link bond with PCDH15 EC1-EC2 at lower end. "
            "CALCIUM BINDING: Ca2+ ions between EC repeats maintain tip-link rigidity; "
            "aminoglycoside toxicity: displaces Ca2+ → tip-link fracture → ototoxicity. "
            "CDH23 LOF → no tip link → mechanotransduction channels cannot be gated → "
            "complete K+ influx failure in all hair cells → profound SNHL. "
            "CDH23 IN RETINA: CDH23 localises to photoreceptor ribbon synapses and the "
            "calyceal process region; CDH23 LOF → synaptic ribbon instability → "
            "progressive photoreceptor synaptic dysfunction → RP. "
            "DFNB12 MECHANISM: partial-function missense alleles retain some tip-link function → "
            "residual hearing (moderate loss); sufficient retinal CDH23 for RP prevention. "
            "CI RESPONSE: excellent cochlear implant outcomes (same as all USH1)."
        ),
        "pathognomonic": (
            "USH1D (CDH23) — CLASSIC USH1 TRIAD: "
            "1. CONGENITAL PROFOUND SNHL (≥90 dBHL all frequencies); deaf at birth; "
            "2. EARLY-ONSET RP: ERG extinguished infancy; nyctalopia early childhood; "
            "3. VESTIBULAR AREFLEXIA: absent caloric; delayed walking; "
            "TIP-LINK UPPER END — CDH23 ANCHORS UPPER END: "
            "KEY TEACHING PEARL: 'CDH23 = Upper end, PCDH15 = Lower end' of tip link; "
            "Ca2+ DEPENDENCE: aminoglycosides displace Ca2+ → tip-link fracture → ototoxicity especially in CDH23; "
            "GENOTYPE SWITCH: truncating → USH1D; missense (hypomorphic) → DFNB12 (no RP); "
            "Clinically distinguishable: DFNB12 patients have hearing loss without visual symptoms."
        ),
        "treatment": (
            "COCHLEAR IMPLANTATION: excellent outcomes; implant early (before age 2); "
            "AMINOGLYCOSIDE AVOIDANCE: if possible — CDH23 tip links Ca2+-dependent; heightened ototoxicity risk; "
            "RETINAL: same as USH1B (vitamin A palmitate; gene therapy trials planned); "
            "ORIENTATION AND MOBILITY: as vision deteriorates; "
            "DFNB12 patients: hearing aids + standard audiological follow-up (no RP monitoring needed); "
            "GENETIC COUNSELLING: genotype clarification important — truncating predicts USH1, "
            "hypomorphic predicts DFNB12 (no retinal referral needed)."
        ),
        "key_features": [
            "CDH23 — TIP LINK UPPER END (CDH23 upper; PCDH15 lower) — key structural teaching point",
            "Genotype-phenotype: truncating → USH1D; hypomorphic missense → DFNB12 (no RP)",
            "~20% of all USH1 cases",
            "Aminoglycoside ototoxicity: heightened Ca2+-dependent tip-link vulnerability",
            "Cochlear implant: excellent outcomes (all USH1 types)",
            "DFNB12: non-syndromic HL diagnosis — no ophthalmology referral for pure DFNB12",
            "CDH23: 27 EC repeats, 3354 aa — largest surface area of any tip-link component",
        ],
        "key_ddx": (
            "PCDH15 (USH1F): TIP LINK LOWER END; identical USH1 phenotype; Gypsy/Roma founder; "
            "MYO7A (USH1B): most common USH1; Acadian founder; no tip-link structural role; "
            "GJB2 (Connexin-26): most common non-syndromic HL worldwide; no RP; gap junction; "
            "STRC (stereocilin): non-syndromic mild-moderate HF loss; DFNB16; no RP; "
            "Mitochondrial A3243G: MELAS syndrome; check maternal inheritance."
        ),
        "systemic_involvement": (
            "NO systemic involvement in classic USH1D. "
            "DFNB12 variant: isolated HL, no retinal or vestibular involvement."
        ),
        "onset_age": "Congenital profound SNHL; RP ERG changes infancy-early childhood; vestibular from birth",
        "surgical_urgency": "Cochlear implant urgency: before age 2 for language development",
        "gene_family": "Cadherin superfamily — calcium-dependent cell adhesion; tip-link upper component",
        "morphology": (
            "AUDIOGRAM: flat profound loss all frequencies; "
            "ERG: extinguished rod + cone from early childhood; "
            "FUNDUS: classic RP bone spicules; "
            "TIP-LINK ELECTRON MICROSCOPY: absent tip links on SEM"
        ),
        "n_patients": 40,
    },

    # -- PCDH15 -- Usher 1F + DFNB23 -----------------------------------------------------------------
    {
        "gene": "PCDH15",
        "alt_name": (
            "PCDH15 (PCDH15-1955aa-10q21.1 / AR -- "
            "USH1F-~15-20pct-ALL-USH1 -- "
            "TIP-LINK-LOWER-END-PCDH15-ANCHORS-LOWER-CDH23-ANCHORS-UPPER -- "
            "DFNB23-NON-SYNDROMIC-HL-MILDER-ALLELES -- "
            "GYPSY-ROMA-FOUNDER-pArg929Stop-CD2-ISOFORM -- "
            "MECHANOTRANSDUCTION-CHANNEL-GATING-PCDH15-LOWER-END)"
        ),
        "protein": (
            "PCDH15 -- 10q21.1 AR -- PCDH15-1955aa -- "
            "Protocadherin-15-215kDa-11-Extracellular-Cadherin-Repeats-MAD12-Ectodomain -- "
            "Tip-Link-LOWER-End-Connects-TMC1-TMC2-Mechanotransduction-Channels -- "
            "3-Isoforms-CD1-CD2-CD3-Different-C-Terminal-Cytoplasmic-Domains -- "
            "OMIM-Gene-605514-Disease-USH1F-602083-DFNB23-609533"
        ),
        "locus": "10q21.1",
        "protein_size": "1955 aa / 215 kDa",
        "inheritance": (
            "AR (autosomal recessive); PCDH15 encodes protocadherin-15, a non-classical cadherin. "
            "TIP-LINK LOWER END: PCDH15 homodimers form the LOWER end of the stereocilia tip link; "
            "PCDH15 connects to CDH23 at the upper end via EC1-EC2 interactions; "
            "PCDH15 LOWER TIP: PCDH15 cytoplasmic domain directly interacts with TMC1/TMC2 channels; "
            "PCDH15 LOF → tip-link lower end absent → TMC channels cannot be gated → "
            "no K+ mechanotransduction current → profound SNHL. "
            "3 ISOFORMS (CD1, CD2, CD3): different cytoplasmic domains; CD2 most expressed in cochlea; "
            "GYPSY/ROMA FOUNDER: p.Arg929Stop (c.2785C>T) specific to CD2 isoform — "
            "prevalent in Roma/Sinti (Gypsy) populations with very high USH1F frequency; "
            "GENOTYPE-PHENOTYPE: severe truncating → USH1F; missense hypomorphic → DFNB23; "
            "~15-20% of USH1 cases."
        ),
        "disease_category": (
            "Usher syndrome type 1F (USH1F, AR) — ~15-20% of USH1; congenital profound SNHL + early RP + vestibular areflexia; "
            "DFNB23: non-syndromic HL (milder PCDH15 alleles, no RP)"
        ),
        "disease_pathway": (
            "PCDH15 LOWER END MECHANISM: PCDH15 forms a parallel homodimer anchored at the lower stereocilium; "
            "EC1-EC2 HANDSHAKE with CDH23 (upper): the EC1-EC2 domains of PCDH15 and CDH23 form "
            "the Ca2+-dependent interdigitating heterodimer ('tip-link handshake'); "
            "TMC1/TMC2 CONNECTION: PCDH15 CD1 cytoplasmic domain interacts with TMC1/TMC2; "
            "stereocilia deflection → tip-link tension → PCDH15 pulls TMC channel open → "
            "K+ influx → hair cell depolarisation → sound transduction. "
            "PCDH15 LOF → tip-link lower end absent → TMC channel not gated → "
            "complete K+ influx failure → profound SNHL. "
            "RETINAL: PCDH15 localises to photoreceptor calyceal processes and ribbon synapses; "
            "LOF → calyceal process disruption → RP (same mechanism as MYO7A/CDH23). "
            "ROMA POPULATION: p.Arg929Stop CD2-isoform specific → complete LOF → severe USH1F; "
            "genetic counselling mandatory in Roma communities (founder frequency very high)."
        ),
        "pathognomonic": (
            "USH1F (PCDH15) — USH1 TRIAD IDENTICAL TO USH1B/1D: "
            "1. Congenital profound SNHL (≥90 dBHL); "
            "2. Early-onset RP (ERG extinguished infancy); "
            "3. Vestibular areflexia (delayed walking ~18 months); "
            "TIP LINK LOWER END PEARL: 'PCDH15 = Lower end (contacts TMC channels)'; "
            "'CDH23 = Upper end (contacts cuticular plate)'; "
            "GYPSY/ROMA POPULATION: p.Arg929Stop most common — screen this variant FIRST in Roma families; "
            "DFNB23: PCDH15 missense → HF moderate SNHL without RP (no retinal referral); "
            "CI OUTCOMES: excellent — same as all USH1; "
            "PCDH15/CDH23 CD1-CD2 ISOFORM SPECIFICITY: disease variant may be isoform-specific "
            "(p.Arg929Stop is CD2-specific — only affects CD2-expressing tissue)."
        ),
        "treatment": (
            "COCHLEAR IMPLANTATION: excellent outcomes; bilateral preferred; early implantation; "
            "ROMA COMMUNITY SCREENING: p.Arg929Stop cascade screening in Roma families; "
            "RETINAL: same management as USH1B (vitamin A palmitate; monitor with ERG/OCT); "
            "DFNB23: hearing aids; no ophthalmology referral (no RP); "
            "GENE THERAPY: PCDH15 trials in development; AAV delivery challenging given gene size; "
            "ORIENTATION AND MOBILITY; deafblind services as vision fails; "
            "GENETIC COUNSELLING: founder mutation testing priority in Roma populations."
        ),
        "key_features": [
            "PCDH15 — TIP LINK LOWER END — connects to TMC1/TMC2 mechanotransduction channels",
            "p.Arg929Stop Gypsy/Roma founder mutation (CD2 isoform-specific)",
            "~15-20% of USH1; USH1F phenotype identical to USH1B/1D",
            "DFNB23: non-syndromic HL (milder alleles, no RP — no retinal referral needed)",
            "Cochlear implant excellent outcomes (all USH1 types)",
            "3 isoforms (CD1/CD2/CD3); CD2 most cochlear-expressed",
            "PCDH15: 11 EC repeats, 1955 aa; tip-link lower end with TMC gating",
        ],
        "key_ddx": (
            "CDH23 (USH1D): TIP LINK UPPER END; identical USH1 phenotype; DFNB12 milder alleles; "
            "MYO7A (USH1B): most common USH1; different mechanism (motor protein not tip link); "
            "USH2A: moderate HF SNHL (not profound); normal vestibular; RP later; "
            "SANS (USH1G): tip-link scaffolding protein; identical USH1 phenotype; rarest USH1; "
            "GJB2: most common SNHL; no RP; connexin gap junction."
        ),
        "systemic_involvement": (
            "NO systemic involvement. Dual sensory: SNHL + RP. Vestibular areflexia. "
            "DFNB23 variant: isolated HL only."
        ),
        "onset_age": "Congenital profound SNHL; RP ERG changes infancy; vestibular areflexia from birth",
        "surgical_urgency": "Cochlear implant before age 2 for optimal speech-language outcomes",
        "gene_family": "Non-classical cadherin (protocadherin) — stereocilia tip-link lower end component",
        "morphology": (
            "AUDIOGRAM: flat profound loss; "
            "ERG: extinguished early childhood; "
            "FUNDUS: RP (bone spicules, vascular attenuation); "
            "ELECTRON MICROSCOPY: absent tip links at lower stereocilium end"
        ),
        "n_patients": 40,
    },

    # -- ADGRV1 -- Usher 2C (2nd most common USH2; LARGEST HUMAN PROTEIN) ----------------------------
    {
        "gene": "ADGRV1",
        "alt_name": (
            "ADGRV1 (ADGRV1-6307aa-5q14.3 / AR -- "
            "USH2C-~15-20pct-USH2-2ND-MOST-COMMON-USH2 -- "
            "LARGEST-HUMAN-PROTEIN-6307aa-692kDa-PATHOGNOMONIC-TEACHING-PEARL -- "
            "MODERATE-SEVERE-HF-SNHL-NORMAL-VESTIBULAR-RP-ONSET-TEENS -- "
            "VLGR1-GPR98-ALIAS-ANKLES-LINKS-COCHLEAR-CALYCEAL-RETINAL)"
        ),
        "protein": (
            "ADGRV1 -- 5q14.3 AR -- ADGRV1-6307aa -- "
            "Adhesion-GPCR-V1-VLGR1-GPR98-692kDa-LARGEST-HUMAN-PROTEIN -- "
            "6085-AA-Extracellular-Calx-Beta-EAR-Repeats-GPCR-Autoproteolysis-Site-GAIN-7TM -- "
            "Ankle-Link-Complex-Cochlear-Hair-Cells-Calyceal-Processes-Photoreceptors -- "
            "OMIM-Gene-602851-Disease-USH2C-605472"
        ),
        "locus": "5q14.3",
        "protein_size": "6307 aa / 692 kDa",
        "inheritance": (
            "AR (autosomal recessive); ADGRV1 (also VLGR1, GPR98) encodes Adhesion G protein-coupled receptor V1, "
            "the LARGEST HUMAN PROTEIN AT 6307 AMINO ACIDS / ~692 kDa. "
            "STRUCTURE: enormous extracellular domain (~6085 aa) with tandem CalX-beta/EAR repeats → "
            "GAIN (GPCR autoproteolysis-inducing) domain → 7 transmembrane helices (GPCR-type); "
            "COCHLEAR FUNCTION: ADGRV1 localises to the ankle-link complex at the very base of cochlear stereocilia "
            "(same location as USH2A usherin); part of USH2 interactome complex; "
            "LOF → ankle-link disruption → moderate-severe HF SNHL (not profound); "
            "RETINAL FUNCTION: ADGRV1 localises to photoreceptor calyceal processes; "
            "LOF → calyceal process disruption → progressive RP; "
            "NORMAL VESTIBULAR: ankle links at vestibular hair cells not essential → normal caloric; "
            "~15-20% of USH2 (second most common USH2 gene after USH2A)."
        ),
        "disease_category": (
            "Usher syndrome type 2C (USH2C, AR) — ~15-20% of USH2 (2nd most common USH2); "
            "identical phenotype to USH2A: moderate-severe HF SNHL + RP teens + normal vestibular; "
            "LARGEST HUMAN PROTEIN 6307 aa"
        ),
        "disease_pathway": (
            "ADGRV1 ANKLE-LINK COMPLEX: ADGRV1 extracellular EAR repeats form "
            "the structural backbone of ankle links at the very base of stereocilia. "
            "USH2 COMPLEX: USH2A (usherin) + ADGRV1 + WHRN (whirlin/DFNB31) + "
            "CIB2 + LNX1/2 form the periciliary membrane complex (PMC) / ankle-link complex; "
            "ADGRV1 AS SCAFFOLD: massive extracellular domain serves as structural scaffold; "
            "calmodulin-like binding in EAR repeats; "
            "ADGRV1 LOF → PMC instability → ankle links fragmented → "
            "high-frequency stereocilia most vulnerable → HF SNHL. "
            "RETINAL: ADGRV1 at photoreceptor calyceal processes → calyceal process disruption → RP. "
            "7TM DOMAIN: functional GPCR domain — ADGRV1 couples to Gα proteins; "
            "ligand unknown; autoproteolytic cleavage (GAIN domain) separates N/C fragments. "
            "LARGEST PROTEIN SIGNIFICANCE: large gene → large variety of truncating mutations; "
            "gene therapy delivery challenging (gene too large for standard AAV)."
        ),
        "pathognomonic": (
            "USH2C (ADGRV1) — PHENOTYPICALLY IDENTICAL TO USH2A: "
            "1. Moderate-severe HF SNHL (sloping audiogram); hearing aids effective; "
            "2. RP onset early teens; ERG changes before symptoms; "
            "3. NORMAL VESTIBULAR — patients can ride bicycle; "
            "LARGEST HUMAN PROTEIN — TEACHING PEARL: "
            "'ADGRV1 = 6307 aa = LARGEST HUMAN PROTEIN' — pathognomonic teaching point; "
            "larger than titin (3965 aa in full isoform for cochlear purposes); "
            "CLINICAL DISTINCTION FROM USH2A: only by genetic testing — phenotypes identical; "
            "ADGRV1 vs USH2A: ADGRV1 has no single common founder mutation (unlike USH2A c.2299delG); "
            "sequencing required; "
            "USH2 COMPLEX INTERACTION: ADGRV1 + USH2A + WHRN all co-localise at ankle links; "
            "digenic Usher (heterozygous ADGRV1 + heterozygous USH2A) has been reported."
        ),
        "treatment": (
            "Same USH2 management as USH2A: "
            "HEARING AIDS: effective early (moderate-severe loss, speech spared); "
            "COCHLEAR IMPLANTATION: when hearing aids insufficient; "
            "RETINAL: vitamin A palmitate 15,000 IU/day (discuss risks/benefits); "
            "LOW VISION AIDS; orientation and mobility as vision declines; "
            "GENE THERAPY CHALLENGE: 6307 aa gene exceeds standard AAV capacity (~4.7 kb insert); "
            "strategies: dual-AAV split-intein approaches under development; "
            "ANNUAL REVIEW: audiologist + ophthalmologist (ERG, OCT, visual fields); "
            "GENETIC COUNSELLING: AR; no single prevalent founder in most populations."
        ),
        "key_features": [
            "LARGEST HUMAN PROTEIN: 6307 aa / 692 kDa — ADGRV1/VLGR1/GPR98 — pathognomonic teaching pearl",
            "USH2C: phenotypically identical to USH2A (moderate-severe HF SNHL + RP + normal vestibular)",
            "~15-20% of USH2 cases (2nd most common USH2 after USH2A)",
            "Part of USH2 ankle-link complex: ADGRV1 + USH2A (usherin) + WHRN (whirlin)",
            "No common single founder mutation — full sequencing required",
            "Gene therapy challenging: 6307 aa exceeds standard AAV capacity",
            "Caloric response NORMAL — can ride bicycle (DDx from USH1)",
        ],
        "key_ddx": (
            "USH2A: identical phenotype; c.2299delG European founder (not in ADGRV1); "
            "WHRN (USH2D): identical USH2 phenotype; PDZ scaffold; rarest USH2; "
            "USH1 (MYO7A/CDH23/PCDH15): PROFOUND (not moderate) SNHL + VESTIBULAR AREFLEXIA; "
            "Nonsyndromic RP: no SNHL; panel required; "
            "Digenic Usher: ADGRV1 het + USH2A het reported."
        ),
        "systemic_involvement": (
            "NO systemic involvement. Dual sensory: SNHL + RP. Vestibular NORMAL. "
            "ADGRV1 7TM domain: GPCR — systemic expression but no established systemic disease."
        ),
        "onset_age": "SNHL congenital moderate-severe; RP ERG changes early teens, symptoms late teens-20s",
        "surgical_urgency": "No acute urgency; cochlear implant when hearing aids fail",
        "gene_family": "Adhesion GPCR class — largest human protein; ankle-link complex component",
        "morphology": (
            "AUDIOGRAM: sloping HF moderate-severe loss; "
            "ERG: progressive rod > cone loss, extinguished eventually; "
            "FUNDUS: RP (bone spicules, arteriolar attenuation)"
        ),
        "n_patients": 40,
    },

    # -- CLRN1 -- Usher 3A (PROGRESSIVE SNHL — unique among Usher types) ----------------------------
    {
        "gene": "CLRN1",
        "alt_name": (
            "CLRN1 (CLRN1-232aa-3q25.1 / AR -- "
            "USH3A-PROGRESSIVE-SNHL-PATHOGNOMONIC-DDx-FROM-USH1-USH2-CONGENITAL -- "
            "pAsn48Lys-FINNISH-FOUNDER-1-IN-5000-FINNISH-BIRTHS -- "
            "pTyr176Ser-ASHKENAZI-JEWISH-FOUNDER -- "
            "PROGRESSIVE-RP-VARIABLE-VESTIBULAR-HEARING-AIDS-TEMPORALLY-EFFECTIVE)"
        ),
        "protein": (
            "CLRN1 -- 3q25.1 AR -- CLRN1-232aa -- "
            "Clarin-1-26kDa-4-TM-Helix-Protein-Tetraspanin-Like-Hair-Cell-Photoreceptor -- "
            "Synapse-Organisation-Hair-Cell-Bundle-Maintenance -- "
            "OMIM-Gene-606397-Disease-USH3A-276902"
        ),
        "locus": "3q25.1",
        "protein_size": "232 aa / 26 kDa",
        "inheritance": (
            "AR (autosomal recessive); CLRN1 encodes clarin-1, a 4-transmembrane domain protein "
            "(tetraspanin-like topology). "
            "EXPRESSION: cochlear inner and outer hair cells; retinal photoreceptors; "
            "FUNCTION: hair cell stereocilia bundle maintenance; synapse organisation; "
            "CLRN1 LOF → stereocilia bundle degeneration over time (NOT absent from birth) → "
            "PROGRESSIVE SNHL — hearing present at birth, deteriorates over years; "
            "RETINAL: CLRN1 in photoreceptors → progressive retinal degeneration; "
            "VESTIBULAR: variable (UNLIKE USH1 always absent, UNLIKE USH2 always normal); "
            "UNIQUE CLINICAL PROFILE: USH3A is the ONLY Usher type with PROGRESSIVE SNHL; "
            "USH1 + USH2 = congenital/stable SNHL; USH3 = PROGRESSIVE; "
            "FOUNDER MUTATIONS: p.Asn48Lys (c.144T>A) — Finnish founder, 1/5000 Finnish births; "
            "p.Tyr176Ser — Ashkenazi Jewish founder."
        ),
        "disease_category": (
            "Usher syndrome type 3A (USH3A, AR) — PROGRESSIVE bilateral SNHL (not congenital); "
            "progressive RP; vestibular dysfunction variable; Finnish and Ashkenazi founders"
        ),
        "disease_pathway": (
            "CLRN1 MECHANISM: clarin-1 is a 4-TM domain protein localised to the hair cell "
            "cuticular plate and stereocilia bundle. "
            "PROGRESSIVE BUNDLE DEGRADATION: unlike USH1 (tip links absent from birth) and "
            "USH2 (ankle links disrupted from birth), "
            "CLRN1 maintains bundle INTEGRITY over time; LOF → gradual stereocilia bundle collapse → "
            "progressive K+ mechanotransduction failure → PROGRESSIVE SNHL. "
            "RATE OF PROGRESSION: variable; typically decades; "
            "hearing aids effective until moderate-severe stage, then cochlear implant needed. "
            "RETINAL: CLRN1 in photoreceptor inner segments and synaptic terminals; "
            "LOF → progressive outer segment renewal failure → progressive RP (later onset than USH1). "
            "VESTIBULAR VARIABILITY: CLRN1 vestibular expression less consistent → "
            "some USH3A patients have normal caloric, others abnormal. "
            "FOUNDER EFFECT: Finnish p.Asn48Lys — replaces polar asparagine with charged lysine "
            "in first extracellular loop; destabilises TM structure → complete LOF."
        ),
        "pathognomonic": (
            "USH3A (CLRN1) — PROGRESSIVE SNHL IS THE PATHOGNOMONIC DISTINCTION: "
            "1. PROGRESSIVE bilateral SNHL: hearing PRESENT and USABLE at birth; "
            "deteriorates over years/decades; "
            "hearing aids effective initially (unlike USH1 where CI needed from birth); "
            "FIRST presentation may be in teens or young adulthood with deteriorating hearing; "
            "2. PROGRESSIVE RP: similar to USH1/2 but onset variable; typically teens-20s; "
            "3. VESTIBULAR: VARIABLE — some patients normal, some abnormal (unlike definite USH1/USH2); "
            "KEY DIAGNOSTIC TRAP: PROGRESSIVE SNHL in a young person + RP → ALWAYS test CLRN1; "
            "USH3A may present with apparently idiopathic progressive SNHL before RP noticed; "
            "FINNISH FOUNDER: p.Asn48Lys — test FIRST in Finnish patients with progressive SNHL + RP; "
            "ASHKENAZI FOUNDER: p.Tyr176Ser — screen Ashkenazi Jewish patients; "
            "USH3 clinical overlap with RP + progressive SNHL (mitochondrial) — check mtDNA panel."
        ),
        "treatment": (
            "HEARING AIDS: EFFECTIVE initially (progressive loss — aids can be fitted earlier and later); "
            "COCHLEAR IMPLANTATION: when hearing aids no longer sufficient; CI outcomes GOOD in USH3; "
            "CI TIMING: consider earlier than in non-Usher progressive SNHL "
            "(concurrent RP will eventually limit ability to lip-read); "
            "RETINAL: vitamin A palmitate 15,000 IU/day; standard RP monitoring (ERG, OCT, visual fields); "
            "GENETIC COUNSELLING: AR; founder testing in Finnish + Ashkenazi populations; "
            "COMBINED DEAFBLIND SERVICES: plan ahead — progressive loss both senses; "
            "ANNUAL REVIEW: audiologist + ophthalmologist; "
            "ORIENTATION AND MOBILITY: begin training while vision adequate."
        ),
        "key_features": [
            "PROGRESSIVE bilateral SNHL — pathognomonic DDx from USH1 (congenital profound) and USH2 (congenital moderate)",
            "p.Asn48Lys Finnish founder (1/5000 Finnish births); p.Tyr176Ser Ashkenazi Jewish founder",
            "Vestibular involvement: VARIABLE (not fixed absent as in USH1, not always normal as in USH2)",
            "Hearing aids temporally effective — plan CI when aids insufficient (concurrent RP limits lip-reading)",
            "Progressive RP — onset variable (teens-20s); similar trajectory to USH2",
            "CLRN1: 4-TM domain clarin; stereocilia bundle maintenance protein",
            "USH3 may first present as progressive SNHL without obvious RP — test CLRN1 in progressive SNHL+RP",
        ],
        "key_ddx": (
            "USH1 (MYO7A/CDH23/PCDH15): CONGENITAL PROFOUND SNHL (not progressive); vestibular AREFLEXIA; "
            "USH2A/ADGRV1: CONGENITAL moderate-severe HF SNHL (not progressive); NORMAL vestibular; "
            "Mitochondrial progressive SNHL + RP (A1555G, MELAS): maternal inheritance; check mtDNA; "
            "DFNB: non-syndromic progressive HL; no RP — requires USH gene panel; "
            "ALPORT syndrome: progressive SNHL + kidney disease; not RP."
        ),
        "systemic_involvement": (
            "NO systemic involvement beyond cochlear + retinal + variable vestibular. "
            "Progressive combined sensory loss trajectory requires long-term deafblind planning."
        ),
        "onset_age": "SNHL onset variable (childhood-early adulthood); RP onset teens-20s; vestibular variable",
        "surgical_urgency": "Plan CI early given concurrent progressive RP that will limit lip-reading compensation",
        "gene_family": "Tetraspanin-like 4-TM protein family; hair cell stereocilia bundle maintenance",
        "morphology": (
            "AUDIOGRAM: progressive bilateral SNHL, typically HF predominant; "
            "ERG: progressive rod > cone reduction; "
            "FUNDUS: RP changes progressive"
        ),
        "n_patients": 40,
    },

    # -- WHRN -- Usher 2D / DFNB31 (rarest USH2) ----------------------------------------------------
    {
        "gene": "WHRN",
        "alt_name": (
            "WHRN (WHRN-907aa-9q32 / AR -- "
            "USH2D-RAREST-USH2-SUBTYPE -- "
            "ALSO-DFNB31-NON-SYNDROMIC-HL-SEVERE-TRUNCATING -- "
            "PDZ-SCAFFOLD-PROTEIN-3-PDZ-DOMAINS-STEREOCILIA-TIP -- "
            "USH2-COMPLEX-WHRN-USH2A-ADGRV1-ANKLE-LINK -- "
            "HF-SNHL-NORMAL-VESTIBULAR-RP-ONSET-TEENS)"
        ),
        "protein": (
            "WHRN -- 9q32 AR -- WHRN-907aa -- "
            "Whirlin-DFNB31-100kDa-PDZ-Scaffold-3-PDZ-Domains-Proline-Rich-Region -- "
            "Stereocilia-Tip-Link-Complex-Ankle-Link-Complex -- "
            "OMIM-Gene-607928-Disease-USH2D-611383-DFNB31-607084"
        ),
        "locus": "9q32",
        "protein_size": "907 aa / 100 kDa",
        "inheritance": (
            "AR (autosomal recessive); WHRN (whirlin, DFNB31) is a PDZ domain scaffold protein. "
            "DOMAIN STRUCTURE: 3 PDZ domains (PDZ1, PDZ2 N-terminal; PDZ3 C-terminal) + proline-rich regions; "
            "TWO ISOFORMS: long form (PDZ1+PDZ2+PDZ3) and short form (PDZ3 only); "
            "USH2 COMPLEX: WHRN interacts with USH2A (usherin) C-terminal PDZ-binding motif and "
            "ADGRV1 C-terminal PDZ-binding motif; localises to ankle-link complex; "
            "STEREOCILIA TIP: WHRN also localises to stereocilia TIPS (interacts with MYO15A elongation complex); "
            "COCHLEAR: WHRN LOF → ankle-link instability → moderate-severe HF SNHL; "
            "RETINAL: WHRN in photoreceptor calyceal processes → progressive RP; "
            "DFNB31: more severe truncating without PDZ3 → non-syndromic HL (no RP in some alleles); "
            "RAREST USH2 GENE: <5% of USH2 cases."
        ),
        "disease_category": (
            "Usher syndrome type 2D (USH2D, AR) — rarest USH2 (<5%); HF SNHL + RP + normal vestibular; "
            "DFNB31: non-syndromic HL (truncating alleles without RP in some)"
        ),
        "disease_pathway": (
            "WHRN SCAFFOLD FUNCTION: whirlin's PDZ domains organise the USH2 protein complex. "
            "ANKLE-LINK SCAFFOLD: PDZ1+2 (N-terminal) bind USH2A and ADGRV1 C-termini → "
            "maintains stoichiometry of the ankle-link complex; "
            "STEREOCILIA TIP SCAFFOLD: WHRN interacts with MYO15A (short isoform) and EPS8 at stereocilia tips; "
            "promotes tip elongation and maintenance; "
            "WHRN LOF → ankle links unstable + stereocilia tips shortened → "
            "combined mechanotransduction failure → HF SNHL. "
            "RETINAL: WHRN PDZ scaffold at photoreceptor calyceal processes → "
            "disrupts USH2A/ADGRV1 complex at calyceal membrane → progressive RP. "
            "DFNB31 ALLELE SPECIFICITY: truncating variants removing ALL PDZ domains → USH2D; "
            "partial deletions retaining some scaffold function → DFNB31 (variable RP expression). "
            "GENE THERAPY: WHRN gene size manageable for AAV delivery; trials not yet initiated."
        ),
        "pathognomonic": (
            "USH2D (WHRN) — IDENTICAL PHENOTYPE TO USH2A AND USH2C: "
            "1. Moderate-severe HF SNHL (sloping audiogram); hearing aids effective initially; "
            "2. RP onset early teens; nyctalopia teens-20s; "
            "3. NORMAL VESTIBULAR — can ride bicycle; "
            "RAREST USH2: <5% of USH2 — often only identified on comprehensive panel; "
            "PDZ SCAFFOLD ROLE: whirlin physically links USH2A and ADGRV1 in the ankle-link complex; "
            "disruption of whirlin disrupts entire USH2 complex; "
            "DFNB31 ALLELE: some truncating alleles cause HL without RP — check genotype-phenotype; "
            "THREE PDZ DOMAINS: 'PDZ1+2 = ankle link scaffold; PDZ3 = stereocilia tip elongation'; "
            "genetic panel required for USH2D diagnosis (no clinical distinction from USH2A/USH2C)."
        ),
        "treatment": (
            "HEARING AIDS: effective early (moderate-severe HF loss); "
            "COCHLEAR IMPLANTATION: when aids insufficient; "
            "RETINAL: standard RP management (vitamin A palmitate; monitoring); "
            "DFNB31: hearing aids only; ophthalmology referral may not be needed initially "
            "(confirm genotype-phenotype before assuming RP); "
            "ANNUAL REVIEW: audiologist + ophthalmologist; "
            "GENETIC COUNSELLING: AR; rare — panel testing required for diagnosis."
        ),
        "key_features": [
            "Rarest USH2 subtype (<5% of USH2)",
            "PDZ scaffold — 3 PDZ domains organise USH2A + ADGRV1 ankle-link complex",
            "Phenotypically identical to USH2A/2C — genetic panel required for diagnosis",
            "DFNB31: non-syndromic HL (some alleles lack RP — genotype-phenotype check)",
            "Dual localisation: ankle-link complex AND stereocilia tip scaffold (with MYO15A)",
            "WHRN 907 aa / 100 kDa — manageable size for potential AAV gene therapy",
        ],
        "key_ddx": (
            "USH2A: most common USH2; c.2299delG founder; identical phenotype; "
            "ADGRV1 (USH2C): largest human protein 6307aa; identical USH2 phenotype; "
            "DFNB31: WHRN non-syndromic (no RP) — allele-specific, genotype-phenotype; "
            "CLRN1 (USH3A): PROGRESSIVE SNHL (not congenital); USH3 distinct."
        ),
        "systemic_involvement": "NO systemic involvement. Dual sensory: SNHL + RP. Vestibular NORMAL.",
        "onset_age": "SNHL congenital moderate-severe; RP ERG changes teens, symptoms teens-20s",
        "surgical_urgency": "No acute urgency; cochlear implant when hearing aids fail",
        "gene_family": "PDZ scaffold protein family; stereocilia ankle-link and tip-elongation complex",
        "morphology": (
            "AUDIOGRAM: sloping HF moderate-severe loss; "
            "ERG: progressive rod > cone reduction; "
            "FUNDUS: RP changes"
        ),
        "n_patients": 40,
    },

    # -- SANS -- Usher 1G (Rarest USH1) -------------------------------------------------------------
    {
        "gene": "SANS",
        "alt_name": (
            "SANS (ANKS4B-461aa-17q25.1 / AR -- "
            "USH1G-RAREST-USH1 -- "
            "SCAFFOLD-PROTEIN-ANKYRIN-REPEATS-STEREOCILIA-TIP-CDH23-PCDH15-ASSEMBLY -- "
            "CONGENITAL-PROFOUND-SNHL-EARLY-RP-VESTIBULAR-AREFLEXIA -- "
            "pArg245_Ile248del-GERMAN-FOUNDER -- "
            "SANS-LINKS-HARMONIN-CDH23-PCDH15-TIP-LINK-INSERTION-COMPLEX)"
        ),
        "protein": (
            "SANS -- 17q25.1 AR -- SANS-461aa-ANKS4B -- "
            "SANS-Scaffold-Protein-52kDa-4-Ankyrin-Repeats-SAM-Domain-PDZ-Binding -- "
            "Stereocilia-Tip-USH1-Protein-Network-CDH23-Harmonin-MYO7A-Assembly-Point -- "
            "OMIM-Gene-607696-Disease-USH1G-606943"
        ),
        "locus": "17q25.1",
        "protein_size": "461 aa / 52 kDa",
        "inheritance": (
            "AR (autosomal recessive); SANS (scaffold protein containing ankyrin repeats and SAM domain, "
            "encoded by ANKS4B) is the smallest and rarest Usher type 1 gene. "
            "DOMAIN STRUCTURE: 4 ankyrin repeats (N-terminal) → central linker → SAM domain → "
            "PDZ-binding C-terminal motif (PBM); "
            "USH1 PROTEIN NETWORK AT STEREOCILIA TIP: "
            "SANS serves as a CENTRAL SCAFFOLD assembling the USH1 tip-link insertion complex: "
            "SANS ankyrin repeats → binds HARMONIN (USH1C/DFNB18) → "
            "HARMONIN recruits CDH23 (upper tip link) → CDH23 connects to PCDH15 (lower tip link); "
            "MYO7A INTERACTION: SANS PDZ-BM binds MYO7A MyTH4-FERM domain; "
            "MYO7A transports SANS to stereocilia tips → SANS assembles the CDH23-HARMONIN-SANS complex; "
            "SANS LOF → complex fails to assemble at tips → CDH23 tip links cannot insert → "
            "mechanotransduction absent → congenital profound SNHL; "
            "~1-5% of USH1 (rarest USH1 gene); "
            "GERMAN FOUNDER: p.Arg245_Ile248del (in-frame deletion in SAM domain)."
        ),
        "disease_category": (
            "Usher syndrome type 1G (USH1G, AR) — rarest USH1 (<5% of USH1); "
            "congenital profound SNHL + early RP + vestibular areflexia; "
            "SANS scaffold assembles USH1 tip-link complex at stereocilia tips"
        ),
        "disease_pathway": (
            "SANS ASSEMBLY FUNCTION: SANS is the 'rivet' of the USH1 tip-link insertion complex. "
            "SANS TRANSPORT: MYO7A motor carries SANS along actin to stereocilia tips; "
            "SANS SCAFFOLD: at the TIP, SANS ankyrin repeats bind HARMONIN-b (PDZ domain-containing scaffold); "
            "HARMONIN-b binds CDH23 EC1-EC2 N-terminus; "
            "CDH23-PCDH15 TIP LINK: CDH23 (upper) and PCDH15 (lower) form the link; "
            "TMC1/TMC2 GATING: PCDH15 CD1 gates TMC channels; "
            "COMPLETE PATHWAY: MYO7A → SANS → HARMONIN → CDH23-PCDH15 tip link → TMC channels. "
            "SANS LOF → CDH23 cannot insert at tips → tip links absent → "
            "K+ mechanotransduction channels never open → profound SNHL. "
            "RETINAL: SANS expressed in photoreceptors; LOF → progressive RP (same mechanism as other USH1 genes). "
            "VESTIBULAR: SANS absent from vestibular tips → caloric areflexia → delayed walking. "
            "SAM DOMAIN: protein-protein interaction domain; p.Arg245_Ile248del disrupts SAM fold → "
            "complete HARMONIN binding failure → null phenotype."
        ),
        "pathognomonic": (
            "USH1G (SANS) — CLASSIC USH1 TRIAD (RAREST USH1): "
            "1. Congenital profound SNHL (≥90 dBHL all frequencies); fails newborn hearing screen; "
            "2. Early-onset RP (ERG extinguished infancy-early childhood); nyctalopia early childhood; "
            "3. Vestibular areflexia (absent caloric; delayed walking ~18 months); "
            "CLINICAL PHENOTYPE: IDENTICAL TO USH1B (MYO7A), USH1D (CDH23), USH1F (PCDH15); "
            "diagnosis requires genetic panel; "
            "SANS ASSEMBLY ROLE PEARL: 'SANS = TIP LINK ASSEMBLY SCAFFOLD'; "
            "without SANS, the CDH23-HARMONIN complex CANNOT be transported to and assembled at stereocilia tips; "
            "even if CDH23 and PCDH15 are intact, SANS LOF prevents tip-link formation; "
            "GERMAN FOUNDER: p.Arg245_Ile248del — screen in German/Central European USH1 families; "
            "CI OUTCOMES: excellent (same as all USH1 genes)."
        ),
        "treatment": (
            "COCHLEAR IMPLANTATION: excellent outcomes; bilateral recommended; before age 2; "
            "RETINAL: standard USH1 RP management (vitamin A palmitate; monitoring); "
            "GENE THERAPY: ANKS4B gene delivery feasible (461 aa — fits in standard AAV); "
            "no active clinical trials 2026 (rare gene); "
            "ORIENTATION AND MOBILITY; deafblind services; "
            "VESTIBULAR REHABILITATION: balance training; "
            "GENETIC COUNSELLING: AR 25%; German founder testing in relevant populations; "
            "PANEL TESTING: comprehensive USH1 panel required — SANS variants rare and missed on targeted testing."
        ),
        "key_features": [
            "Rarest USH1 gene (<5% of USH1); USH1 triad identical to USH1B/1D/1F",
            "SANS = central scaffold assembling USH1 tip-link complex (MYO7A → SANS → HARMONIN → CDH23 → PCDH15 → TMC)",
            "p.Arg245_Ile248del German founder (SAM domain in-frame deletion)",
            "SANS 461 aa — fits in single AAV (gene therapy feasible in principle)",
            "Cochlear implant excellent outcomes (all USH1 types equally)",
            "Vestibular areflexia: delayed walking ~18 months",
            "SANS links MYO7A motor domain to CDH23 tip-link insertion at stereocilia tip",
        ],
        "key_ddx": (
            "MYO7A (USH1B): most common USH1; ACADIAN founder; Myosin motor (not scaffold); "
            "CDH23 (USH1D): TIP LINK UPPER END; identical USH1; DFNB12 milder alleles; "
            "PCDH15 (USH1F): TIP LINK LOWER END; identical USH1; Roma founder; "
            "Harmonin (USH1C/DFNB18): HARMONIN scaffold — upstream of SANS in same pathway; rare; "
            "USH2 (USH2A, ADGRV1): MODERATE (not profound) SNHL; NORMAL vestibular."
        ),
        "systemic_involvement": "NO systemic involvement. Classic USH1 dual sensory loss + vestibular areflexia.",
        "onset_age": "Congenital profound SNHL; RP ERG changes infancy-early childhood; vestibular from birth",
        "surgical_urgency": "Cochlear implant before age 2 for language development",
        "gene_family": "Ankyrin repeat + SAM domain scaffold protein; USH1 stereocilia tip-link complex organiser",
        "morphology": (
            "AUDIOGRAM: flat profound loss all frequencies; "
            "ERG: extinguished rod + cone early childhood; "
            "FUNDUS: RP changes (bone spicules, vascular attenuation)"
        ),
        "n_patients": 40,
    },
]


def _patients(gene_data, seed):
    rng = random.Random(seed)
    gene = gene_data["gene"]
    is_ush1 = gene in ("MYO7A", "CDH23", "PCDH15", "SANS")
    is_ush3 = gene == "CLRN1"
    is_ush2 = gene in ("USH2A", "ADGRV1", "WHRN")

    pts = []
    for _ in range(gene_data["n_patients"]):
        if is_ush1:
            age = rng.randint(5, 45)
            snhl_db = rng.randint(90, 120)
            snhl_type = "Profound flat"
            vestibular = "Absent"
            walking_delay = rng.choice([True, True, True, False])
            rp_onset = rng.randint(2, 12)
        elif is_ush3:
            age = rng.randint(15, 60)
            snhl_db = rng.randint(50, 100)
            snhl_type = rng.choice(["Progressive moderate", "Progressive moderate-severe", "Progressive severe"])
            vestibular = rng.choice(["Absent", "Abnormal", "Normal", "Normal"])
            walking_delay = False
            rp_onset = rng.randint(15, 35)
        else:  # USH2
            age = rng.randint(10, 55)
            snhl_db = rng.randint(55, 85)
            snhl_type = "Moderate-severe HF sloping"
            vestibular = "Normal"
            walking_delay = False
            rp_onset = rng.randint(12, 25)

        night_blind_age = rp_onset + rng.randint(2, 8)
        tunnel_vision_age = rp_onset + rng.randint(10, 25)
        ci_fitted = rng.random() < (0.85 if is_ush1 else 0.45)
        ha_fitted = not ci_fitted and rng.random() < 0.85
        va_residual = round(rng.uniform(0.05, 0.4) if age > (rp_onset + 20) else rng.uniform(0.4, 1.0), 2)

        pts.append({
            "age": age,
            "snhl_db": snhl_db,
            "snhl_type": snhl_type,
            "vestibular": vestibular,
            "walking_delay": walking_delay,
            "rp_onset_years": rp_onset,
            "night_blind_age": night_blind_age,
            "tunnel_vision_age": tunnel_vision_age,
            "ci_fitted": ci_fitted,
            "ha_fitted": ha_fitted,
            "va_residual": va_residual,
            "cochlear_implant_outcome": (
                rng.choice(["Excellent", "Very Good", "Good"]) if ci_fitted else None
            ),
        })
    return pts


def _compute_metrics(pts, gene):
    is_ush1 = gene in ("MYO7A", "CDH23", "PCDH15", "SANS")
    is_ush3 = gene == "CLRN1"
    n = len(pts)
    ci_pct = round(100 * sum(1 for p in pts if p["ci_fitted"]) / n, 1)
    ha_pct = round(100 * sum(1 for p in pts if p["ha_fitted"]) / n, 1)
    vestib_absent_pct = round(100 * sum(1 for p in pts if p["vestibular"] == "Absent") / n, 1)
    vestib_normal_pct = round(100 * sum(1 for p in pts if p["vestibular"] == "Normal") / n, 1)
    walking_delay_pct = round(100 * sum(1 for p in pts if p["walking_delay"]) / n, 1)
    profound_snhl_pct = round(100 * sum(1 for p in pts if p["snhl_db"] >= 90) / n, 1)
    low_va_pct = round(100 * sum(1 for p in pts if p["va_residual"] < 0.3) / n, 1)
    progressive_snhl_pct = round(100 * sum(1 for p in pts if "Progressive" in p["snhl_type"]) / n, 1)

    avg_rp_onset = round(sum(p["rp_onset_years"] for p in pts) / n, 1)
    avg_snhl = round(sum(p["snhl_db"] for p in pts) / n, 1)

    return {
        "ci_pct": ci_pct,
        "ha_pct": ha_pct,
        "vestibular_absent_pct": vestib_absent_pct,
        "vestibular_normal_pct": vestib_normal_pct,
        "walking_delay_pct": walking_delay_pct,
        "profound_snhl_pct": profound_snhl_pct,
        "low_va_pct": low_va_pct,
        "progressive_snhl_pct": progressive_snhl_pct,
        "avg_rp_onset_years": avg_rp_onset,
        "avg_snhl_db_hl": avg_snhl,
    }


def _build_gene_record(gd, seed):
    pts = _patients(gd, seed)
    m = _compute_metrics(pts, gd["gene"])
    return {
        "gene": gd["gene"],
        "alt_name": gd["alt_name"],
        "locus": gd["locus"],
        "protein_size": gd["protein_size"],
        "inheritance": gd["inheritance"],
        "disease_category": gd["disease_category"],
        "disease_pathway": gd["disease_pathway"],
        "pathognomonic": gd["pathognomonic"],
        "treatment": gd["treatment"],
        "key_features": gd["key_features"],
        "key_ddx": gd["key_ddx"],
        "systemic_involvement": gd.get("systemic_involvement", False),
        "onset_age": gd["onset_age"],
        "surgical_urgency": gd["surgical_urgency"],
        "gene_family": gd["gene_family"],
        "morphology": gd["morphology"],
        "n_patients": gd["n_patients"],
        **m,
        "sample_patients": pts[:5],
    }


_records = None


def _get_records():
    global _records
    if _records is None:
        _records = [
            _build_gene_record(gd, SEED_BASE + i)
            for i, gd in enumerate(USH_GENES)
        ]
    return _records


def get_overview():
    records = _get_records()
    total = sum(r["n_patients"] for r in records)
    all_pts = []
    for i, gd in enumerate(USH_GENES):
        all_pts.extend(_patients(gd, SEED_BASE + i))

    n = len(all_pts)
    ci_pct = round(100 * sum(1 for p in all_pts if p["ci_fitted"]) / n, 1)
    ha_pct = round(100 * sum(1 for p in all_pts if p["ha_fitted"]) / n, 1)
    vestib_absent_pct = round(100 * sum(1 for p in all_pts if p["vestibular"] == "Absent") / n, 1)
    vestib_normal_pct = round(100 * sum(1 for p in all_pts if p["vestibular"] == "Normal") / n, 1)
    profound_snhl_pct = round(100 * sum(1 for p in all_pts if p["snhl_db"] >= 90) / n, 1)
    walking_delay_pct = round(100 * sum(1 for p in all_pts if p["walking_delay"]) / n, 1)
    low_va_pct = round(100 * sum(1 for p in all_pts if p["va_residual"] < 0.3) / n, 1)
    progressive_snhl_pct = round(100 * sum(1 for p in all_pts if "Progressive" in p["snhl_type"]) / n, 1)
    avg_rp_onset = round(sum(p["rp_onset_years"] for p in all_pts) / n, 1)

    gene_summary = {}
    for r in records:
        gene_summary[r["gene"]] = {
            "gene": r["gene"],
            "disease_category": r["disease_category"],
            "locus": r["locus"],
            "protein_size": r["protein_size"],
            "inheritance": r["inheritance"][:80] + "…",
            "n_patients": r["n_patients"],
            "ci_pct": r["ci_pct"],
            "profound_snhl_pct": r["profound_snhl_pct"],
            "vestibular_absent_pct": r["vestibular_absent_pct"],
            "avg_rp_onset_years": r["avg_rp_onset_years"],
        }

    return {
        "atlas": "Hereditary-Usher-Syndrome-Atlas",
        "subtitle": "Complete 8-Gene Hereditary Usher Syndrome Reference -- MYO7A/USH2A/CDH23/PCDH15/ADGRV1/CLRN1/WHRN/SANS",
        "genes_covered": [g["gene"] for g in USH_GENES],
        "total_patients": total,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "aggregate_metrics": {
            "ci_pct": ci_pct,
            "ha_pct": ha_pct,
            "vestibular_absent_pct": vestib_absent_pct,
            "vestibular_normal_pct": vestib_normal_pct,
            "profound_snhl_pct": profound_snhl_pct,
            "walking_delay_pct": walking_delay_pct,
            "low_va_pct": low_va_pct,
            "progressive_snhl_pct": progressive_snhl_pct,
            "avg_rp_onset_years": avg_rp_onset,
        },
        "gene_summary": gene_summary,
        "key_clinical_alerts": [
            "MYO7A-USH1B-MOST-COMMON-USH1-COCHLEAR-IMPLANT-BEFORE-AGE-2",
            "USH2A-c.2299delG-TEST-FIRST-IN-EUROPEAN-USH2-PATIENTS",
            "CLRN1-PROGRESSIVE-SNHL-PATHOGNOMONIC-DDx-FROM-USH1-USH2-CONGENITAL",
            "ADGRV1-LARGEST-HUMAN-PROTEIN-6307aa-692kDa",
            "CDH23-UPPER-PCDH15-LOWER-TIP-LINK-ANATOMY-KEY-PEARL",
            "VESTIBULAR-AREFLEXIA-USH1-NORMAL-VESTIBULAR-USH2-VARIABLE-USH3",
            "ALL-USHER-CI-CANDIDATES-CI-EXCELLENT-ALL-TYPES",
            "SANS-CENTRAL-SCAFFOLD-MYO7A-SANS-HARMONIN-CDH23-PCDH15-TMC",
            "USH3A-HEARING-AIDS-EFFECTIVE-INITIALLY-PLAN-CI-CONCURRENT-RP",
            "ADGRV1-GENE-THERAPY-CHALLENGE-6307aa-EXCEEDS-STANDARD-AAV",
        ],
        "atlas_summary": (
            "Hereditary Usher Syndrome is the most common cause of combined deaf-blindness worldwide. "
            "Three clinical types: USH1 (congenital profound SNHL + early RP + vestibular areflexia), "
            "USH2 (congenital moderate-severe HF SNHL + RP + NORMAL vestibular), "
            "USH3 (PROGRESSIVE SNHL + progressive RP + variable vestibular). "
            "8-gene atlas covers MYO7A (USH1B, most common USH1), USH2A (most common overall, ~40% all Usher), "
            "CDH23 (USH1D, tip-link upper end), PCDH15 (USH1F, tip-link lower end), "
            "ADGRV1 (USH2C, largest human protein 6307aa), CLRN1 (USH3A, progressive SNHL unique), "
            "WHRN (USH2D, rarest USH2), SANS (USH1G, rarest USH1, tip-link assembly scaffold). "
            "All Usher patients are cochlear implant candidates with excellent outcomes. "
            "USH2A c.2299delG: test first in European-descent USH2 patients (~30% of alleles). "
            "Stereocilia tip-link anatomy: CDH23 (upper end) — PCDH15 (lower end) — TMC channels. "
            "320 patients · 8×40 · seeds 2430–2437."
        ),
    }


def get_breakdown():
    records = _get_records()
    return {
        "atlas": "Hereditary-Usher-Syndrome-Atlas",
        "gene_breakdowns": records,
    }


def get_definitions():
    records = _get_records()
    gene_entries = {}
    for r in records:
        gene_entries[r["gene"]] = (
            f"{r['gene']} — {r['protein_size']} — {r['locus']} — {r['disease_category']} — "
            f"Pathognomonic: {r['pathognomonic'][:200]}… — "
            f"Treatment: {r['treatment'][:200]}…"
        )

    return {
        "atlas": "Hereditary-Usher-Syndrome-Atlas",
        "gene_entries": gene_entries,
        "ush_glossary": {
            "USH1_triad": (
                "USH Type 1 clinical triad: (1) CONGENITAL PROFOUND SNHL (≥90 dBHL flat audiogram, "
                "deaf at birth); (2) EARLY-ONSET RP (ERG extinguished infancy, nyctalopia late childhood, "
                "tunnel vision 20s-30s, legally blind 40s-50s); (3) VESTIBULAR AREFLEXIA (absent caloric, "
                "delayed walking ~18 months, falls in dark). All USH1 patients are CI candidates with "
                "excellent outcomes. Genes: MYO7A (1B), CDH23 (1D), PCDH15 (1F), SANS (1G)."
            ),
            "USH2_triad": (
                "USH Type 2 clinical profile: (1) CONGENITAL MODERATE-SEVERE HF SNHL (sloping audiogram, "
                "speech frequencies partially preserved, hearing aids effective initially); "
                "(2) RP ONSET TEENS (ERG changes early teens, nyctalopia teens, tunnel vision 20s-30s); "
                "(3) NORMAL VESTIBULAR FUNCTION (can ride bicycle, normal caloric — KEY DDx from USH1). "
                "Genes: USH2A (2A, most common), ADGRV1 (2C), WHRN (2D, rarest)."
            ),
            "USH3_triad": (
                "USH Type 3 clinical profile: (1) PROGRESSIVE bilateral SNHL (hearing present at birth, "
                "deteriorates over years/decades — PATHOGNOMONIC DDx from USH1/2); "
                "(2) PROGRESSIVE RP (variable onset); (3) VARIABLE vestibular dysfunction "
                "(neither always absent as USH1, nor always normal as USH2). "
                "Gene: CLRN1 (3A). Finnish founder p.Asn48Lys; Ashkenazi founder p.Tyr176Ser."
            ),
            "tip_link_anatomy": (
                "STEREOCILIA TIP-LINK ANATOMY: "
                "CDH23 (cadherin-23) ANCHORS UPPER END of tip link (inserts into CUTICULAR PLATE at upper stereocilium). "
                "PCDH15 (protocadherin-15) ANCHORS LOWER END (contacts TMC1/TMC2 mechanotransduction channels). "
                "SANS scaffold (ANKS4B): transported by MYO7A to stereocilia tips; "
                "assembles HARMONIN-CDH23 complex at upper tip-link insertion. "
                "MECHANISM: stereocilia deflection → tip-link tension → PCDH15 gates TMC channels → "
                "K+ influx → hair cell depolarisation → sound signal."
            ),
            "cochlear_implant_usher": (
                "COCHLEAR IMPLANT IN USHER SYNDROME: All Usher types are excellent CI candidates. "
                "USH1: implant BEFORE AGE 2 for optimal speech-language development (CI bypasses non-functional hair cells). "
                "USH2: implant when hearing aids no longer sufficient (moderate-severe loss means aids work longer). "
                "USH3: consider early CI given progressive RP will eventually limit lip-reading compensation. "
                "Bilateral CI preferred. Post-CI: auditory verbal therapy mandatory."
            ),
            "USH2A_c2299delG_founder": (
                "USH2A c.2299delG (p.Glu767SerfsX21) EUROPEAN FOUNDER MUTATION: "
                "~30% of all USH2A alleles in European-descent populations. "
                "Simple, rapid targeted Sanger test. Screen FIRST in European-descent patients with USH2 phenotype. "
                "Carrier frequency ~1/72 in European population. "
                "Results in frameshift at exon 13 → loss of LamG domain → complete usherin LOF → severe USH2A."
            ),
            "adgrv1_largest_protein": (
                "ADGRV1 LARGEST HUMAN PROTEIN: ADGRV1/VLGR1/GPR98 = 6307 amino acids / ~692 kDa. "
                "The largest known human protein by amino acid count. "
                "Massive extracellular domain with tandem CalX-beta/EAR repeats (~6085 aa extracellular). "
                "Clinical significance: gene therapy delivery is challenging — gene exceeds standard AAV packaging capacity (~4.7 kb insert). "
                "Dual-vector split-intein AAV approaches under development."
            ),
            "clrn1_progressive_vs_congenital": (
                "CLRN1/USH3A PROGRESSIVE SNHL — KEY CLINICAL DISTINCTION: "
                "USH1 and USH2 both present with CONGENITAL and STABLE SNHL (present at birth, non-progressive). "
                "USH3A presents with PROGRESSIVE bilateral SNHL (hearing present and usable at birth; "
                "deteriorates over years to decades). "
                "Key diagnostic clue: 'My hearing has been getting worse' in a patient who also has RP → test CLRN1. "
                "Hearing aids can be fitted and upgraded as loss progresses. "
                "Plan CI timing with awareness that concurrent RP will reduce lip-reading ability."
            ),
            "usher_vitamin_A": (
                "VITAMIN A PALMITATE IN USHER SYNDROME RP: "
                "Vitamin A palmitate 15,000 IU/day (adult; reduce for children by weight): "
                "Level B recommendation for retinitis pigmentosa generally (Berson et al., 1993 — original RP study). "
                "Evidence in Usher: extrapolated; not directly demonstrated in controlled Usher trials. "
                "Discuss risks: hepatotoxicity (monitor LFTs); teratogenicity (avoid in pregnancy); "
                "AVOID VITAMIN E high-dose (may accelerate RP). "
                "DHA (docosahexaenoic acid) supplementation: adjunctive; limited evidence."
            ),
            "differentiating_USH1_USH2": (
                "CLINICAL DDx USH1 vs USH2: "
                "'Can you ride a bicycle?' → YES (USH2: normal vestibular) | NO (USH1: vestibular areflexia). "
                "Audiogram: profound flat (USH1) vs moderate-severe HF sloping (USH2). "
                "Walking milestone: delayed ~18 months (USH1: vestibular dysfunction) vs normal (USH2). "
                "ERG timing: extinguished in infancy (USH1) vs changes in early teens (USH2). "
                "Hearing aid efficacy: minimal (USH1 profound) vs good initially (USH2 moderate-severe)."
            ),
        },
    }


if __name__ == "__main__":
    import json
    print("OVERVIEW:")
    print(json.dumps(get_overview(), indent=2)[:1000])
    print("\nBREAKDOWN genes:", [r["gene"] for r in get_breakdown()["gene_breakdowns"]])
    print("DEFINITIONS keys:", list(get_definitions()["ush_glossary"].keys()))
