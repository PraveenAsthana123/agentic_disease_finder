"""
TCF4 Pitt-Hopkins Syndrome — bHLH E-protein / 18q21.2 / OMIM 602272 (gene) / 610954 (disease)
================================================================================================
40-patient cohort · TCF4 (18q21.2) · De novo Autosomal Dominant · Both sexes equally affected
Episodic hyperventilation/apnoea PATHOGNOMONIC · Beaked nose + Cupid bow lip dysmorphic
Profound-to-severe ID + absent speech + epilepsy · Corpus callosum absent ~20%

TCF4 BIOLOGY:
TCF4 (Transcription Factor 4, Transforming growth factor beta-inducible Early Gene 2 / ITF2 / E2-2)
encodes an E-protein class basic helix-loop-helix (bHLH) transcription factor of 667 amino acids.
TCF4 belongs to the class I bHLH (E-protein) family along with TCF3 (E2A), TCF12 (HEB), and
ASCL1 (achaete-scute homolog 1). E-proteins form obligate dimers — homodimers or heterodimers
with class II bHLH factors (proneural factors: NEUROD1, NEUROG2, ASCL1, ATOH1) — via their
HLH (helix-loop-helix) dimerisation domain.

TCF4 PROTEIN STRUCTURE (667 aa, 18q21.2):
  N-TERMINAL REGULATORY DOMAIN (aa 1-250): Low complexity activation domain. Contains multiple
    alternatively spliced exons (TCF4 has 20 isoforms differing in N-terminal inclusion).
    Transactivation domain A (AD-A, aa 30-90): interacts with p300/CBP co-activators.
    Transactivation domain B (AD-B, aa 150-210): interacts with SRF (serum response factor).
    Nuclear localisation signal (NLS, aa 220-240): contains NLS1 directing TCF4 to nucleus.
  CENTRAL LINKER REGION (aa 250-548): Connects N-terminal activation domain to the bHLH core.
    Contains regulatory phosphorylation sites (Thr-291 — CK1 site; Ser-374 — PKA site).
    LANA-like domain (aa 320-400): negatively regulates homodimerisation (autoinhibitory).
  bHLH CORE DOMAIN (aa 550-610): The critical functional core. Contains:
    Basic region (aa 550-573): arginine/lysine-rich region that makes direct base-specific
      contacts with E-box consensus sequence (CANNTG — canonical E-box: CAGNTG or CATGTG).
      R576 (Arg576) is the most critical residue — makes hydrogen bonds to E-box guanine.
      Dominant negative missense mutations: R576W (most common), A597T, L600P.
      These DN mutations eliminate E-box DNA binding while retaining HLH dimerisation
      capacity — dominant negative mechanism: TCF4-DN forms non-functional heterodimers
      with ASCL1, NEUROD1, NEUROG2, sequestering them from productive transcriptional
      complexes → gain-of-function at the dimer level / loss-of-function at E-box target level.
    Helix 1 (H1, aa 574-590): amphipathic alpha-helix; forms one arm of the X-shaped dimer
      interface with the partner bHLH helix. Leucines at H1 stabilise hydrophobic core.
    Loop (aa 590-594): flexible glycine-rich loop; connects H1 to H2; length determines
      dimerisation partner selectivity (E-proteins prefer short loops).
    Helix 2 (H2, aa 594-610): second amphipathic alpha-helix; completes HLH dimerisation face.
      L600P disrupts H2 secondary structure → eliminates dimerisation with NEUROD1.
  C-TERMINAL DOMAIN (aa 610-667): Post-bHLH domain. Contains:
    PAS-like domain (aa 620-645): found in some TCF4 isoforms; interacts with aryl hydrocarbon
      receptor nuclear translocator (ARNT) family members.
    C-terminal NLS2 (aa 650-667): second nuclear localisation signal.

TCF4 TARGETS AND NEURAL FUNCTION:
TCF4 is expressed at highest levels in developing brain (peak: embryonic week 8-22 in human;
neonatal in mouse) and in mature cerebellum, brainstem, and limbic structures.

Key transcriptional targets of TCF4 (E-box-dependent activation):
  NRXN1 (Neurexin-1): trans-synaptic adhesion molecule; TCF4 haploinsufficiency → reduced
    NRXN1 expression → impaired presynaptic-postsynaptic connectivity → autism + epilepsy.
  CNTN2 (Contactin-2/TAG-1): axon fasciculation molecule in corticospinal and thalamocortical
    tracts; reduced in Pitt-Hopkins → axonal pathfinding defects.
  SCN1A enhancer: TCF4 occupies SCN1A enhancer elements → haploinsufficiency contributes to
    Nav1.1 expression reduction → interneuron hypofunction → cortical hyperexcitability.
    (Mechanistic overlap with Dravet syndrome phenotype via SCN1A enhancer insufficiency.)
  GABRB3: TCF4 regulates GABRB3 (GABA-A beta3) expression in interneurons → haploinsufficiency
    → reduced GABAergic inhibitory tone → seizure susceptibility.
  DLG4 (PSD-95): postsynaptic density scaffold; TCF4 indirectly regulates via SHANK3 and
    SYNGAP1 enhancer binding → postsynaptic maturation failure in Pitt-Hopkins.
  MASH1/ASCL1 targets: TCF4 heterodimerises with ASCL1 to activate genes of GABAergic
    interneuron lineage. TCF4 haploinsufficiency → ASCL1 functional insufficiency → reduced
    parvalbumin (PV+) interneuron migration from medial ganglionic eminence (MGE) to cortex.
    This is the key interneuronopathy mechanism of Pitt-Hopkins epilepsy.

LOF MECHANISM (haploinsufficiency — 55% of Pitt-Hopkins):
De novo heterozygous TCF4 pathogenic variant → 50% dosage reduction → impaired E-box-dependent
transcription → reduced NRXN1, CNTN2, SCN1A enhancer activity, GABRB3 → cortical hyperexcitability
+ interneuron migration failure (PV+ interneurons critically TCF4-dependent) + white matter
myelination defect (oligodendrocyte differentiation requires TCF4 E-protein activity) →
Pitt-Hopkins syndrome: ID + epilepsy + breathing dysfunction.

DOMINANT NEGATIVE MECHANISM (bHLH missense — 15% of Pitt-Hopkins):
Missense in bHLH basic or H1/H2 region (R576W, A597T, L600P) → protein retains HLH
dimerisation but loses E-box binding → forms non-functional dimers with ASCL1/NEUROD1/NEUROG2
→ dominantly sequesters partner bHLH factors → functional haploinsufficiency PLUS ectopic
partner inhibition → generally MORE severe than simple haploinsufficiency.

BREATHING EPISODES — BRAINSTEM MECHANISM:
TCF4 is expressed in the pre-Bötzinger complex (preBötC) — the primary respiratory rhythm
generator in the medulla — and in the Kölliker-Fuse nucleus (pontine respiratory group).
TCF4 haploinsufficiency → impaired preBötC neuron maturation + reduced PHOX2B-dependent
visceral motor neuron development → dysrhythmic respiratory pattern generation.
The episodic hyperventilation (central neurogenic, not obstructive) alternating with apnoea
reflects an immature/dysregulated respiratory oscillator — unique to TCF4/Pitt-Hopkins among
all Angelman-like syndromes. EEG is NORMAL or mildly slow during breathing episodes (not ictal).

OLIGODENDROCYTE / WHITE MATTER:
TCF4 is an essential driver of oligodendrocyte precursor cell (OPC) differentiation and
myelination. TCF4 LOF → reduced OPC maturation → hypomyelination/dysmyelination →
contributes to thin corpus callosum (absent in ~20%) and delayed myelination on MRI.

PHENOTYPIC SPECTRUM:
  CLASSIC DE NOVO PITT-HOPKINS (TCF4 haploinsufficiency — missense outside bHLH or truncating
    upstream of bHLH + >50 residual activation capacity):
    - Both sexes equally (autosomal dominant de novo); sporadic in >99%.
    - Breathing episodes (episodic hyperventilation + apnoea): onset 2–5 years; PATHOGNOMONIC.
    - Profound ID (IQ typically <40); absent speech in ~80% (some acquire <5 words).
    - Facial dysmorphism: beaked/peaked nasal bridge + thin tip; everted lower lip; Cupid-bow
      upper lip; deep-set eyes; wide mouth + widely-spaced teeth; short philtrum; full cheeks.
    - Epilepsy: focal/multifocal 60%, tonic-clonic 40%, myoclonic 20%, absence 15%.
    - Corpus callosum absent or thin (~20%).
    - Stereotypic hand movements (mouthing, wringing) — Rett-like but progressive speech loss
      NOT seen (unlike Rett: no regression, no normal period followed by loss).
    - Happy/sociable affect (Angelman-like); hand-flapping when excited.
    - ~55% of Pitt-Hopkins cohort.
  TRUNCATING SEVERE EARLY COGNITIVE (frameshift/nonsense exons 1-17, upstream of bHLH):
    - Complete absence of TCF4 protein from mutant allele → maximal haploinsufficiency.
    - Severe DEE; earliest onset epilepsy (<12 months) including hypsarrhythmia in some.
    - More profound cognitive impact.
    - Breathing episodes may appear later (2–4 years).
    - Higher seizure burden; more refractory.
    - ~20% of Pitt-Hopkins cohort.
  bHLH MISSENSE DOMINANT NEGATIVE (R576W, A597T, L600P in bHLH aa 550-610):
    - Dominant negative mechanism → more severe than haploinsufficiency.
    - Breathing episodes prominent.
    - Seizure clustering during febrile illness.
    - Some cortical malformation (simplified gyri).
    - ~15% of Pitt-Hopkins cohort.
  18q21.2 DELETION CONTIGUOUS GENE (chromosomal deletion encompassing TCF4 + flanking genes):
    - FISH/aCGH/SNP array diagnostic; larger deletion → additional features.
    - Growth retardation; cardiac defects (if MBD2 co-deleted).
    - CHARGE-like features if deletion extends to CHD7 territory.
    - ~10% of Pitt-Hopkins cohort.

DISTINGUISHING TCF4 PITT-HOPKINS FROM KEY DDx:
  Angelman (UBE3A/15q11): happy affect + absent speech + epilepsy — BUT: no breathing episodes
    (PATHOGNOMONIC for Pitt-Hopkins), abnormal UBE3A methylation (normal in TCF4/Pitt-Hopkins),
    normal nasal bridge (not beaked), only abnormal UBE3A methylation/mutation diagnostic.
  Rett syndrome (MECP2, Xq28): breathing irregularity (overlap) + hand stereotypies — BUT:
    females predominantly, NORMAL early development then regression (not seen in TCF4),
    MECP2 mutation, hand stereotypies replace purposeful hand use (regression pattern unique to Rett).
  SLC9A6/MRXSCH Christianson: Angelman-like in males — BUT: X-linked (males only), progressive
    CEREBELLAR ATROPHY on MRI (not beaked nose), no breathing episodes, SLC9A6 mutation.
  FOXG1 (14q12): dyskinesias + stereotypies + frontal gyral simplification — BUT: no breathing
    episodes, frontal MRI predominance (not cerebellar), FOXG1 mutation, distinct face.
  Mowat-Wilson (ZEB2, 2q22.3): ID + epilepsy — BUT: Hirschsprung/severe constipation,
    pointed chin + upturned nasal tip (NOT beaked nose), ZEB2 mutation.
  CHARGE syndrome (CHD7, 8q12): coloboma + heart + choanal atresia + retarded growth +
    genital abnormalities + ear abnormalities — BUT: CHD7 mutation, distinct multi-organ.

REFERENCES:
  Pitt D, Hopkins IJ (1978) A syndrome of mental retardation, wide mouth, fleshy lips and
    open-mouthed facial expression. Aust Paediatr J 14:182-184. PMID 310547.
  Amiel J et al. (2007) Mutations in TCF4, encoding a class I basic helix-loop-helix
    transcription factor, are responsible for Pitt-Hopkins syndrome. Am J Hum Genet 80:988-993.
    PMID 17436255.
  Zweier C et al. (2007) Haploinsufficiency of TCF4 causes syndromal mental retardation with
    intermittent hyperventilation (Pitt-Hopkins syndrome). Am J Hum Genet 80:994-1001.
    PMID 17436244.
  de Pontual L et al. (2009) Mutational, functional, and expression studies of the TCF4 gene
    in Pitt-Hopkins syndrome. Hum Mutat 30:669-676. PMID 19235228.
  Flora A et al. (2007) Deletion of Atoh1 disrupts Sonic Hedgehog signaling in the
    developing cerebellum and prevents medulloblastoma. Science 316:1424-1427.
  OMIM 602272 (TCF4 gene) / OMIM 610954 (Pitt-Hopkins syndrome).
"""

import random

random.seed(513)

# ── ETIOLOGY CATALOG ─────────────────────────────────────────────────────────
ETIOLOGY_CATALOG = [
    {
        "category": "TCF4-Classic-DeNovo-PittHopkins",
        "n_target": 22,
        "description": (
            "De novo heterozygous TCF4 pathogenic variants causing haploinsufficiency via missense "
            "outside the bHLH core or truncating variants that escape NMD and retain partial "
            "function. Classic Pitt-Hopkins phenotype: EPISODIC HYPERVENTILATION + APNOEA "
            "(PATHOGNOMONIC — onset 2-5 years, central not obstructive, EEG normal during "
            "episode); profound-to-severe ID (IQ <40); absent speech or very limited (<5 words); "
            "beaked/peaked nasal bridge with thin nasal tip; everted lower lip; Cupid-bow upper "
            "lip; deep-set eyes; wide mouth with widely-spaced teeth; short philtrum. Epilepsy "
            "(focal/multifocal 60%, TC 40%, myoclonic 20%, absence 15%); onset 1-7 years. "
            "Corpus callosum absent/thin in ~20%. Stereotypic hand movements. Happy sociable "
            "affect. Both sexes equally. De novo in >99%. ~55% of Pitt-Hopkins cohort."
        ),
        "typical_variant": (
            "c.1459C>T (p.Arg487*) nonsense upstream of bHLH; "
            "c.1321_1322delAG (p.Arg441fs) frameshift; "
            "splice site variants in intron 14-17 (exon skipping outside bHLH); "
            "c.1687G>A (p.Glu563Lys) — basic region proximal, partial function retained"
        ),
        "inheritance": "De novo autosomal dominant haploinsufficiency (AD, both sexes equally)",
        "functional_deficit": (
            "50% TCF4 dosage reduction → reduced E-box-dependent transcription of NRXN1, "
            "CNTN2, SCN1A enhancer, GABRB3 → cortical hyperexcitability + PV+ interneuron "
            "migration failure (ASCL1 partner insufficiency) + oligodendrocyte differentiation "
            "defect → Pitt-Hopkins phenotype"
        ),
    },
    {
        "category": "TCF4-Truncating-Severe-EarlyCognitive",
        "n_target": 8,
        "description": (
            "Frameshift or nonsense variants in exons 1-17 (upstream of bHLH core domain) → "
            "complete absence of TCF4 protein from mutant allele (NMD-mediated mRNA decay) → "
            "maximal haploinsufficiency. Most severe cognitive trajectory: complete absence of "
            "intentional communication. Early-onset epilepsy (<12 months) including hypsarrhythmia "
            "in some; higher seizure burden; more refractory (3+ AED failures typical). "
            "Breathing episodes may appear later (2-4 years). Greater risk of corpus callosum "
            "agenesis and simplified gyri. ~20% of Pitt-Hopkins cohort."
        ),
        "typical_variant": (
            "c.243_244delCT (p.Leu82fs) — exon 4 frameshift; "
            "c.598C>T (p.Arg200*) — exon 8 nonsense; "
            "c.1103+1G>A — intron 11 splice donor (exon skipping → frameshift); "
            "large exonic deletions encompassing exons 2-8 or 5-12"
        ),
        "inheritance": "De novo autosomal dominant, complete null allele (haploinsufficiency maximum)",
        "functional_deficit": (
            "Zero TCF4 from mutant allele → maximum 50% functional dosage → complete impairment "
            "of all TCF4-dependent E-box programs: NRXN1, CNTN2, GABRB3, SCN1A enhancer, "
            "PV+ interneuron development via ASCL1, oligodendrocyte maturation via OPC "
            "differentiation program → severe DEE + complete cognitive impairment"
        ),
    },
    {
        "category": "TCF4-bHLH-Missense-DominantNegative",
        "n_target": 6,
        "description": (
            "Missense variants in the bHLH core domain (aa 550-610): R576W (Arg576Trp — "
            "disrupts E-box guanine hydrogen bond), A597T (Ala597Thr — H2 helix partial "
            "unfolding), L600P (Leu600Pro — H2 helix proline-kink, eliminates NEUROD1 "
            "dimerisation). Dominant negative mechanism: TCF4-DN retains HLH dimerisation "
            "capacity but loses E-box DNA binding → forms non-functional heterodimers with "
            "ASCL1, NEUROD1, NEUROG2 → sequesters proneural factors from productive complexes "
            "→ more severe than simple haploinsufficiency (DN > LOF phenotype). Breathing "
            "episodes prominent. Seizure clustering during febrile illness. ~15% of cohort."
        ),
        "typical_variant": (
            "c.1726C>T (p.Arg576Trp) R576W — most common bHLH DN missense; "
            "c.1789G>A (p.Ala597Thr) A597T; "
            "c.1799T>C (p.Leu600Pro) L600P — H2 helix disruption"
        ),
        "inheritance": "De novo autosomal dominant, dominant negative bHLH missense",
        "functional_deficit": (
            "TCF4-DN protein forms non-functional dimers with ASCL1/NEUROD1/NEUROG2 → "
            "ectopic sequestration of proneural bHLH partners in addition to loss of TCF4 "
            "own E-box function → more severe NRXN1/CNTN2/GABRB3 deficiency + greater "
            "interneuron migration failure than haploinsufficiency alone"
        ),
    },
    {
        "category": "TCF4-18q21.2-Deletion-ContiguousGene",
        "n_target": 4,
        "description": (
            "Chromosomal deletion encompassing TCF4 at 18q21.2 + flanking genes (MBD2, "
            "SMAD7, NEDD4L depending on deletion size). Detected by FISH, aCGH, or SNP array — "
            "FISH/sequence-negative but methylation-normal → array mandatory. Additional features "
            "beyond classic Pitt-Hopkins: growth retardation, cardiac defects (if MBD2 co-deleted "
            "affecting DNA methylation), renal anomalies, hearing loss. CHARGE-like if deletion "
            "extends far enough to involve nearby regulatory elements. Largest phenotypic impact "
            "in this group; typical deletion size 500 kb to 5 Mb. ~10% of Pitt-Hopkins cohort."
        ),
        "typical_variant": (
            "18q21.2 deletion encompassing TCF4 (500 kb – 5 Mb); arr[hg38] 18q21.2(52,900,000-"
            "55,200,000)x1 de novo; detected by aCGH or SNP array; sequence-negative cases "
            "mandating array analysis"
        ),
        "inheritance": "De novo chromosomal deletion, autosomal, contiguous gene syndrome",
        "functional_deficit": (
            "TCF4 complete deletion from one allele → maximal haploinsufficiency + contiguous "
            "gene effects (MBD2 deletion → DNA methylation defect; NEDD4L deletion → renal "
            "tubular function; SMAD7 deletion → TGF-beta signalling dysregulation)"
        ),
    },
]

# ── PATIENT COHORT (40 patients, seed 513) ──────────────────────────────────
def _build_cohort():
    rng = random.Random(513)
    pts = []
    pid = 1
    for ec in ETIOLOGY_CATALOG:
        n = ec["n_target"]
        for _ in range(n):
            cat = ec["category"]
            is_classic   = cat == "TCF4-Classic-DeNovo-PittHopkins"
            is_severe    = cat == "TCF4-Truncating-Severe-EarlyCognitive"
            is_dn        = cat == "TCF4-bHLH-Missense-DominantNegative"
            is_deletion  = cat == "TCF4-18q21.2-Deletion-ContiguousGene"

            sex = rng.choice(["M", "F"])  # AD — both sexes equally

            # Seizure onset age in years
            seizure_onset_yr = (
                rng.uniform(1.0, 7.0)   if is_classic  else
                rng.uniform(0.3, 2.0)   if is_severe   else
                rng.uniform(1.0, 5.0)   if is_dn       else
                rng.uniform(0.5, 4.0)         # deletion
            )

            # Seizure types
            focal          = rng.random() < (0.60 if is_classic else 0.55 if is_severe else 0.65 if is_dn else 0.58)
            tonic_clonic   = rng.random() < (0.40 if is_classic else 0.45 if is_severe else 0.42 if is_dn else 0.50)
            myoclonic      = rng.random() < (0.20 if is_classic else 0.30 if is_severe else 0.28 if is_dn else 0.25)
            absence        = rng.random() < (0.15 if is_classic else 0.20 if is_severe else 0.18 if is_dn else 0.12)
            infantile_spasms = rng.random() < (0.05 if is_classic else 0.25 if is_severe else 0.08 if is_dn else 0.15)
            hypsarrhythmia   = infantile_spasms and rng.random() < 0.85

            # Breathing episodes
            breathing_episodes = rng.random() < (0.85 if is_classic else 0.55 if is_severe else 0.92 if is_dn else 0.70)

            # Cognitive / speech
            absent_speech  = rng.random() < (0.80 if is_classic else 0.95 if is_severe else 0.88 if is_dn else 0.82)
            speech_words   = 0 if absent_speech else rng.randint(1, 5)
            profound_id    = rng.random() < (0.75 if is_classic else 0.92 if is_severe else 0.85 if is_dn else 0.80)
            any_id         = profound_id or rng.random() < 0.98

            # Morphological
            corpus_callosum_absent = rng.random() < (0.20 if is_classic else 0.28 if is_severe else 0.22 if is_dn else 0.35)
            beaked_nose            = rng.random() < (0.88 if is_classic else 0.82 if is_severe else 0.90 if is_dn else 0.75)
            cupid_bow_lip          = rng.random() < (0.80 if is_classic else 0.72 if is_severe else 0.82 if is_dn else 0.68)
            happy_affect           = rng.random() < (0.82 if is_classic else 0.65 if is_severe else 0.78 if is_dn else 0.70)
            hand_stereotypies      = rng.random() < (0.60 if is_classic else 0.55 if is_severe else 0.65 if is_dn else 0.50)
            hand_flapping          = rng.random() < (0.55 if is_classic else 0.45 if is_severe else 0.60 if is_dn else 0.48)

            # Drug resistance
            drug_resistant  = rng.random() < (0.55 if is_classic else 0.78 if is_severe else 0.70 if is_dn else 0.65)
            n_aeds_failed   = (
                rng.randint(1, 4) if is_classic  else
                rng.randint(2, 6) if is_severe   else
                rng.randint(2, 5) if is_dn       else
                rng.randint(1, 5)
            )

            # MRI
            mri_simplified_gyri    = rng.random() < (0.12 if is_classic else 0.25 if is_severe else 0.20 if is_dn else 0.18)
            mri_white_matter_signal = rng.random() < (0.25 if is_classic else 0.40 if is_severe else 0.30 if is_dn else 0.35)

            # Treatments
            polg_tested     = rng.random() < 0.78
            lev_given       = rng.random() < (0.72 if is_classic else 0.78 if is_severe else 0.75 if is_dn else 0.68)
            vpa_given       = rng.random() < (0.50 if is_classic else 0.55 if is_severe else 0.48 if is_dn else 0.52)
            clb_given       = rng.random() < (0.42 if is_classic else 0.48 if is_severe else 0.45 if is_dn else 0.40)
            kd_tried        = rng.random() < (0.20 if is_classic else 0.35 if is_severe else 0.28 if is_dn else 0.30)
            acth_given      = infantile_spasms and rng.random() < 0.85
            vgb_given       = acth_given and rng.random() < 0.80
            acetazolamide   = breathing_episodes and rng.random() < 0.15  # Level C, used sparingly

            seizure_free    = rng.random() < (0.18 if is_classic else 0.08 if is_severe else 0.12 if is_dn else 0.15)

            # Cardiac (contiguous gene deletion)
            cardiac_defect  = rng.random() < (0.05 if not is_deletion else 0.38)
            growth_retard   = rng.random() < (0.08 if not is_deletion else 0.55)

            pts.append({
                "patient_id":               f"TCF4-{pid:03d}",
                "sex":                      sex,
                "category":                 cat,
                "seizure_onset_yr":         round(seizure_onset_yr, 1),
                "focal":                    focal,
                "tonic_clonic":             tonic_clonic,
                "myoclonic":                myoclonic,
                "absence":                  absence,
                "infantile_spasms":         infantile_spasms,
                "hypsarrhythmia":           hypsarrhythmia,
                "breathing_episodes":       breathing_episodes,
                "absent_speech":            absent_speech,
                "speech_words":             speech_words,
                "profound_id":              profound_id,
                "any_id":                   any_id,
                "corpus_callosum_absent":   corpus_callosum_absent,
                "beaked_nose":              beaked_nose,
                "cupid_bow_lip":            cupid_bow_lip,
                "happy_affect":             happy_affect,
                "hand_stereotypies":        hand_stereotypies,
                "hand_flapping":            hand_flapping,
                "drug_resistant":           drug_resistant,
                "n_aeds_failed":            n_aeds_failed,
                "mri_simplified_gyri":      mri_simplified_gyri,
                "mri_white_matter_signal":  mri_white_matter_signal,
                "polg_tested":              polg_tested,
                "lev_given":                lev_given,
                "vpa_given":                vpa_given,
                "clb_given":                clb_given,
                "kd_tried":                 kd_tried,
                "acth_given":               acth_given,
                "vgb_given":                vgb_given,
                "acetazolamide":            acetazolamide,
                "seizure_free":             seizure_free,
                "cardiac_defect":           cardiac_defect,
                "growth_retard":            growth_retard,
            })
            pid += 1
    return pts


PATIENTS = _build_cohort()

# ── TREATMENTS ────────────────────────────────────────────────────────────────
TREATMENTS = [
    {
        "drug": "Levetiracetam (LEV, Level B) — first-line broad-spectrum AED",
        "level": (
            "Level B (moderate-quality evidence). SV2A synaptic vesicle glycoprotein ligand. "
            "Excellent safety profile: no hepatic, mitochondrial, or cardiac concerns. "
            "Dosed 20-60 mg/kg/day in 2 divided doses. Effective for focal, tonic-clonic, "
            "and myoclonic seizure types in Pitt-Hopkins. May cause behavioural side effects "
            "(irritability, aggression) — monitor carefully; switch to clobazam if intolerable. "
            "IV formulation available for acute status management. Preferred first AED in PHS."
        ),
    },
    {
        "drug": "Valproate (VPA, Level B) — after POLG screen mandatory",
        "level": (
            "Level B (moderate evidence). Broad-spectrum: inhibits multiple seizure mechanisms "
            "(Na+ channel, T-type Ca2+, GABA transaminase). Effective for focal, TC, myoclonic, "
            "absence seizures. POLG SEQUENCING MANDATORY before VPA — fatal Alpers-Huttenlocher "
            "hepatic failure in POLG carriers (applies universally to all DEE syndromes). "
            "Target VPA level 50-100 mg/L. Monitor LFTs. Teratogenic — document contraception "
            "in adolescent females. Weight gain, hair loss, tremor as side effects. Useful in "
            "Pitt-Hopkins for multiple seizure type control after POLG clearance."
        ),
    },
    {
        "drug": "Clobazam (CLB, Level B) — adjunct 1,5-benzodiazepine",
        "level": (
            "Level B adjunct. 1,5-benzodiazepine with lower sedation than 1,4-BZDs "
            "(clonazepam, diazepam). Preferential GABA-A alpha-2/alpha-3 subunit activity. "
            "Effective for focal and myoclonic seizures in Pitt-Hopkins. Start 0.1-0.3 mg/kg/day, "
            "titrate slowly. Buccal midazolam for acute cluster rescue. Tolerance may develop — "
            "drug holiday every 3-6 months if needed. Low respiratory depression risk at "
            "therapeutic doses (important given breathing episodes in PHS — monitor)."
        ),
    },
    {
        "drug": "ACTH (Level A, UKISS) — for infantile spasms component",
        "level": (
            "Level A per UKISS protocol for infantile spasms component. ACTH 150 IU/m2/day IM, "
            "2-week course then taper. Not mechanism-targeted at TCF4 haploinsufficiency but "
            "IS suppression is the primary goal in the spasms window. VGB as adjunct per UKISS "
            "(ACTH+VGB combination outperforms monotherapy for IS resolution). VGB REMS "
            "mandatory. IS component occurs predominantly in the TCF4-Truncating-Severe category "
            "(<12 months onset). ACTH taper over 2 weeks after full-dose course."
        ),
    },
    {
        "drug": "Vigabatrin (VGB, Level A, UKISS) — IS adjunct, REMS mandatory",
        "level": (
            "Level A for infantile spasms (UKISS). VGB 50 mg/kg/day, 2 divided doses, with ACTH. "
            "VGB REMS: ERG baseline; visual field (Goldman) q3 months; max 6 months IS indication. "
            "Not indicated outside IS unless refractory (Level C). Mechanism: GABA-T inhibitor → "
            "raised synaptic GABA. Used only when IS is the presenting seizure type in Pitt-Hopkins "
            "(TCF4-Truncating-Severe category predominantly)."
        ),
    },
    {
        "drug": "Lamotrigine (LTG, Level C) — caution in myoclonic component",
        "level": (
            "Level C (observational evidence). Sodium channel blocker + glutamate release inhibitor. "
            "Generally useful for focal and absence seizures in Pitt-Hopkins. USE WITH CAUTION "
            "in patients with myoclonic component — LTG can worsen myoclonic seizures (pro-myoclonic "
            "in some DEE patients). Start low, titrate very slowly (rash risk; target 1-5 mg/kg/day "
            "over 6-8 weeks). Do not use as first-line if myoclonic seizures are prominent. "
            "Avoid rapid up-titration. Combination with VPA requires 50% LTG dose reduction."
        ),
    },
    {
        "drug": "Ketogenic Diet (KD, Level C) — drug-resistant cases (2+ AED failures)",
        "level": (
            "Level C (limited case series in Pitt-Hopkins; broader evidence in drug-resistant "
            "DEE syndromes). 4:1 or modified Atkins (2:1) ratio. Consider after 2+ AED failures. "
            "RD dietitian + metabolic team mandatory. Screen for fatty acid oxidation disorders "
            "before KD initiation. Monitor beta-OHB (target 2-5 mmol/L), growth, lipid profile, "
            "renal stones, cardiac function. Some reports of breathing episode improvement on KD "
            "(likely via metabolic effects on brainstem respiratory circuits — not proven mechanism)."
        ),
    },
    {
        "drug": "Acetazolamide (carbonic anhydrase inhibitor, Level C) — breathing episodes",
        "level": (
            "Level C (case reports/small series only). Carbonic anhydrase inhibitor — reduces "
            "CO2-drive cycling (attenuates the hypocapnia→apnoea loop in episodic hyperventilation). "
            "Mechanism: by blunting the pH/CO2 overcorrection, acetazolamide may reduce the "
            "severity/frequency of hyperventilation-apnoea cycles. NOT an AED — does not treat "
            "seizures directly. Used specifically for BREATHING EPISODES (episodic hyperventilation "
            "with apnoea) when episodes are severely distressing or causing hypoxic risk. Dosed "
            "5-10 mg/kg/day in 2-3 divided doses. Monitor renal stones (sulfonamide risk)."
        ),
    },
    {
        "drug": "POLG Screen — MANDATORY before any valproate",
        "level": (
            "Universal DEE protocol. POLG (mitochondrial DNA polymerase gamma) biallelic mutations "
            "→ Alpers-Huttenlocher syndrome: fatal hepatic failure triggered by VPA. POLG "
            "sequencing MANDATORY before VPA in all DEE patients including Pitt-Hopkins. "
            "TCF4 haploinsufficiency has no intrinsic mitochondrial involvement, but the "
            "POLG-before-VPA protocol is a non-negotiable DEE safety gate. LFTs baseline "
            "also required before VPA. If POLG positive or equivocal → avoid VPA."
        ),
    },
    {
        "drug": "CBZ/OXC — HIGH CAUTION (avoid in myoclonic/focal-onset bilateral TC)",
        "level": (
            "HIGH CAUTION. Carbamazepine (CBZ) and oxcarbazepine (OXC) are sodium channel blockers "
            "that may worsen focal seizures in some DEE contexts and precipitate absence-status "
            "or myoclonic exacerbation in Pitt-Hopkins. AVOID in patients with myoclonic component. "
            "AVOID if absence seizures are prominent. Use with extreme caution only for isolated "
            "focal seizures without myoclonic/absence component after other first-line options "
            "have failed. Not recommended as empiric first-line in Pitt-Hopkins. Monitor EEG "
            "for worsening after initiation."
        ),
    },
]

# ── CONTRAINDICATIONS ─────────────────────────────────────────────────────────
CONTRAINDICATIONS = [
    {
        "drug": "Carbamazepine (CBZ) / Oxcarbazepine (OXC) — HIGH CAUTION",
        "reason": (
            "CBZ and OXC are sodium channel stabilisers that may precipitate or worsen myoclonic "
            "seizures, absence seizures, and absence-status epilepticus in Pitt-Hopkins. HIGH "
            "CAUTION — especially in patients with myoclonic component or absence seizures. "
            "Avoid as first-line. If used at all, restrict to isolated focal-onset seizures "
            "with no myoclonic/absence component, after other options fail. Monitor EEG for "
            "worsening (absence-status is a known risk with CBZ in non-focal epilepsy syndromes)."
        ),
    },
    {
        "drug": "Valproate (VPA) without POLG screening",
        "reason": (
            "POLG sequencing is MANDATORY before VPA in Pitt-Hopkins. Fatal Alpers-Huttenlocher "
            "hepatic failure in POLG carriers. VPA is otherwise useful in Pitt-Hopkins (broad "
            "spectrum, effective for multiple seizure types) BUT must be preceded by POLG clearance. "
            "Also: document reproductive counselling in adolescent females (VPA teratogenicity — "
            "neural tube defects, neurodevelopmental risks in offspring). LFTs baseline mandatory."
        ),
    },
    {
        "drug": "Phenytoin (PHT) / Fosphenytoin IV — avoid unless no alternative",
        "reason": (
            "IV phenytoin/fosphenytoin: cardiac arrhythmia risk (QTc prolongation, hypotension "
            "with rapid infusion). In Pitt-Hopkins status epilepticus, IV levetiracetam is the "
            "preferred IV AED. PHT/fosphenytoin reserved as last resort only; use IV LEV or "
            "IV phenobarbital first. PHT also worsens absence seizures — especially relevant "
            "given 15% absence prevalence in Pitt-Hopkins cohort."
        ),
    },
    {
        "drug": "Vigabatrin (VGB) without ophthalmology REMS monitoring",
        "reason": (
            "VGB causes irreversible peripheral visual field constriction (bi-nasal/concentric "
            "loss) via retinal GABA accumulation. VGB REMS: ERG at baseline + visual field q3M "
            "on VGB + maximum 6 months for IS indication. Never use VGB for non-IS indications "
            "in Pitt-Hopkins without documented REMS enrolment and ophthalmology review. "
            "Goldman perimetry preferred over Humphrey in cognitively impaired patients."
        ),
    },
    {
        "drug": "Lamotrigine (LTG) — avoid if myoclonic component present",
        "reason": (
            "LTG can paradoxically worsen myoclonic seizures in some DEE patients. Use with "
            "caution and avoid as monotherapy in Pitt-Hopkins patients with prominent myoclonic "
            "component. Generally safe for focal and absence seizures in PHS. Slow titration "
            "mandatory to reduce Stevens-Johnson risk (especially if also on VPA — halve LTG "
            "dose and extend titration schedule). Monitor for rash."
        ),
    },
]

# ── MONITORING ────────────────────────────────────────────────────────────────
MONITORING = [
    {
        "timepoint": "Initial Diagnosis Workup",
        "action": (
            "TCF4 gene panel (exon sequencing) + MLPA/aCGH/SNP array (18q21.2 deletion — "
            "sequence-negative cases → array mandatory). Parental testing (de novo confirmation; "
            "gonadal mosaic risk <1%). MRI brain (3T): corpus callosum, cortical simplification, "
            "white matter myelination. EEG: characterise seizure type (focal/myoclonic/absence/IS). "
            "POLG sequencing before VPA consideration. LFTs baseline. Respiratory/sleep "
            "polysomnography if breathing episodes severe (central apnoea confirmation). "
            "Ophthalmology (optic disc, refractive error — common). Echocardiogram if 18q deletion. "
            "Formal developmental and cognitive assessment (Griffiths/Bayley/DISCO)."
        ),
    },
    {
        "timepoint": "3 Months",
        "action": (
            "EEG: IS/hypsarrhythmia resolution if ACTH given (UKISS protocol). Seizure diary review. "
            "VGB ophthalmology ERG baseline (REMS requirement if VGB started). ACTH taper monitoring "
            "(2-week full dose + 2-week taper). Respiratory assessment: breathing episode frequency "
            "log, pulse oximetry if severe. VPA LFTs if VPA started. Behavioural assessment "
            "(LEV irritability — consider switch to CLB if significant)."
        ),
    },
    {
        "timepoint": "6 Months",
        "action": (
            "EEG: background + epileptiform activity evolution. MRI follow-up: myelination progress, "
            "corpus callosum. VGB visual field q3M (Goldman ERG; discontinue VGB after 6 months "
            "per IS-indication REMS). Breathing episode log: frequency, duration, cyanosis, "
            "triggers (excitement/arousal not fever). Acetazolamide trial if episodes severe. "
            "Developmental assessment (motor, language, adaptive). Neurogenetics: parental results, "
            "reproductive counselling. Ophthalmology at 6 months."
        ),
    },
    {
        "timepoint": "12 Months (Annual)",
        "action": (
            "MRI brain annual: myelination progress, corpus callosum, cortical simplification "
            "evolution. Comprehensive neuropsychological assessment (Griffiths/Bayley). Epilepsy "
            "review: seizure frequency, AED levels. LFTs if on VPA. Breathing episode severity "
            "assessment (polysomnography if deteriorating). Ophthalmology annual. Cardiology if "
            "18q deletion. EHCP educational support review. AED rationalisation (reduce polypharmacy "
            "where possible). Genetics review: reproductive counselling for family."
        ),
    },
    {
        "timepoint": "Ongoing Surveillance (Every 2-3 Years)",
        "action": (
            "Transition planning (paediatric → adult services age 16-18). Scoliosis assessment "
            "(postural instability from hypotonia). Swallowing assessment (dysphagia risk). "
            "Behavioural management review (anxiety, agitation in adolescence). Carer and family "
            "support. Palliative care planning for advanced disability. Respiratory surveillance "
            "in adults (breathing episodes may evolve; central sleep apnoea in some adults). "
            "Genetic counselling update: gonadal mosaic recurrence risk discussion."
        ),
    },
]

# ── THRESHOLDS ────────────────────────────────────────────────────────────────
THRESHOLDS = [
    {
        "metric": "Breathing Episodes — Central Apnoea Duration",
        "normal": "No hyperventilation/apnoea episodes (episodes onset typically age 2-5 years)",
        "alert_value": "Apnoea >15 seconds OR cyanosis during episode → pulse oximetry monitoring",
        "action": "Pulse oximetry during sleep; consider acetazolamide (Level C) for frequent severe episodes",
        "critical_value": "Apnoea >30 seconds + SaO2 <85% → emergency management; polysomnography; anaesthesia consult for procedures",
    },
    {
        "metric": "VPA Plasma Level (if on VPA, post-POLG clearance)",
        "normal": "50-100 mg/L (therapeutic for epilepsy; below 50 mg/L = subtherapeutic)",
        "alert_value": ">100 mg/L supratherapeutic — increased hepatotoxicity + tremor risk",
        "action": "Adjust VPA dose; check LFTs and ammonia (VPA hyperammonaemia); consider dose reduction",
        "critical_value": ">120 mg/L or LFTs >3× ULN → VPA dose reduction or withdrawal; POLG carrier re-screen",
    },
    {
        "metric": "EEG Ictal vs Non-Ictal Breathing Episodes",
        "normal": "EEG normal or mildly slow during breathing episode = NOT ictal",
        "alert_value": "Ictal pattern during episode → reclassify as seizure; adjust AED treatment",
        "action": "Simultaneous video-EEG during episode to confirm ictal vs non-ictal status",
        "critical_value": "Generalised rhythmic discharge during episode = ictal → treat as seizure type, not breathing episode",
    },
    {
        "metric": "Corpus Callosum — MRI Detection",
        "normal": "Corpus callosum present and normal (80% of Pitt-Hopkins patients)",
        "alert_value": "Thin/hypoplastic corpus callosum → functional implications; ~15% of cohort",
        "action": "Formal neuropsychological assessment; interhemispheric communication testing",
        "critical_value": "Complete agenesis (~5%) → absent callosal fibres; assess visual evoked potentials; consider interhemispheric disconnection effects on seizure spread",
    },
    {
        "metric": "Developmental Quotient (DQ) Trajectory",
        "normal": "DQ consistent with profound-to-severe ID; stable (non-progressive — Pitt-Hopkins is non-degenerative)",
        "alert_value": "Accelerating developmental regression → rule out treatable superimposed condition",
        "action": "Rule out Rett-like regression (MECP2 second hit), status epilepticus, metabolic crisis, psychosocial deprivation",
        "critical_value": "Severe acute regression (>3 SD below baseline) → emergency evaluation: metabolic screen, ammonia, lactate, EEG, LP",
    },
]

# ── DEFINITIONS ───────────────────────────────────────────────────────────────
DEFINITIONS = [
    {
        "term": "TCF4 / Pitt-Hopkins Syndrome / bHLH E-Protein / OMIM 610954",
        "definition": (
            "TCF4 (Transcription Factor 4, 18q21.2) encodes a 667-amino-acid E-protein class "
            "bHLH transcription factor. TCF4 forms homodimers and heterodimers with proneural "
            "class II bHLH factors (NEUROD1, NEUROG2, ASCL1) via its HLH dimerisation domain, "
            "binding E-box sequences (CANNTG) in neural gene promoters/enhancers. TCF4 "
            "haploinsufficiency → Pitt-Hopkins syndrome (OMIM 610954): profound ID + absent "
            "speech + epilepsy + PATHOGNOMONIC breathing episodes + distinctive dysmorphic "
            "features. De novo AD, sporadic. Both sexes equally affected."
        ),
    },
    {
        "term": "Breathing Episodes (Episodic Hyperventilation/Apnoea) — PATHOGNOMONIC",
        "definition": (
            "Episodic hyperventilation bursts (RR >40/min, central neurogenic origin) followed "
            "by apnoea (central, not obstructive) are PATHOGNOMONIC for Pitt-Hopkins syndrome. "
            "Onset: typically 2-5 years; may begin in infancy. Triggers: excitement, emotional "
            "arousal, voluntary breath-holding — NOT fever-triggered (unlike PCDH19 febrile seizures). "
            "Duration: 10-60 seconds; cyanosis if apnoea prolonged. EEG during episode: NORMAL "
            "or mildly slow — NOT ictal (video-EEG confirmation diagnostic). Management: "
            "reassurance for mild; acetazolamide (Level C) for severe/frequent. Mechanism: "
            "TCF4 haploinsufficiency → pre-Bötzinger complex (preBötC) maturation failure → "
            "dysrhythmic brainstem respiratory oscillator. Absent in ALL other Angelman-like "
            "syndromes (Angelman, MRXSCH, FOXG1, Mowat-Wilson) — discriminating feature."
        ),
    },
    {
        "term": "bHLH Domain — Basic Helix-Loop-Helix (aa 550-610 of TCF4)",
        "definition": (
            "The bHLH domain (aa 550-610) is the functional core of TCF4. It comprises: "
            "(1) Basic region (aa 550-573): arg/lys-rich, makes base-specific contacts with E-box "
            "CANNTG sequence; R576 is critical for guanine hydrogen bonding. "
            "(2) Helix 1 (H1, aa 574-590): amphipathic alpha-helix, forms one arm of X-shaped "
            "dimer interface. (3) Loop (aa 590-594): flexible glycine-rich; connects H1-H2. "
            "(4) Helix 2 (H2, aa 594-610): second dimer helix; L600P disrupts this helix. "
            "Missense in basic region (R576W) → E-box binding lost but HLH dimerisation retained "
            "→ dominant negative: non-functional dimers with ASCL1/NEUROD1 → more severe than "
            "haploinsufficiency alone."
        ),
    },
    {
        "term": "Dominant Negative Mechanism — bHLH Missense (R576W, A597T, L600P)",
        "definition": (
            "Missense variants in the TCF4 bHLH core (R576W, A597T, L600P) disrupt E-box "
            "DNA binding while retaining HLH dimerisation capacity. The resulting TCF4-DN "
            "protein forms non-functional heterodimers with proneural class II bHLH factors "
            "(ASCL1, NEUROD1, NEUROG2), sequestering them from productive transcriptional "
            "complexes. This dominant negative effect adds ectopic partner inhibition to the "
            "loss of TCF4's own transcriptional function → TCF4 bHLH missense > haploinsufficiency "
            "in phenotypic severity. R576W (c.1726C>T) is the most commonly reported PHS-DN variant."
        ),
    },
    {
        "term": "TCF4 Target Genes — E-box-Dependent Neural Transcription",
        "definition": (
            "TCF4 activates critical neural genes via E-box binding. Key targets: "
            "NRXN1 (Neurexin-1) — trans-synaptic adhesion; TCF4 haploinsufficiency reduces "
            "NRXN1 → impaired synaptogenesis. CNTN2 (Contactin-2) — axon fasciculation in "
            "corticospinal/thalamocortical tracts. SCN1A enhancer — TCF4 occupies SCN1A "
            "enhancers; haploinsufficiency reduces Nav1.1 in interneurons → hyperexcitability. "
            "GABRB3 — GABA-A beta3 subunit in interneurons → reduced GABAergic inhibition. "
            "ASCL1-partnered targets: drives PV+ parvalbumin interneuron lineage from MGE "
            "→ TCF4 haploinsufficiency = interneuronopathy."
        ),
    },
    {
        "term": "PV+ Interneuron Migration Failure — TCF4 Interneuronopathy",
        "definition": (
            "TCF4 heterodimerises with ASCL1 (Mash1) to activate gene programs of GABAergic "
            "interneuron specification and migration from the medial ganglionic eminence (MGE) "
            "to the cortex. Parvalbumin-expressing (PV+) fast-spiking interneurons are "
            "critically TCF4-dependent. TCF4 haploinsufficiency → impaired ASCL1 partner "
            "activity → reduced PV+ interneuron migration and maturation → cortical "
            "E/I (excitation/inhibition) imbalance → epilepsy + cognitive impairment. "
            "This interneuronopathy is the primary mechanism of seizures in Pitt-Hopkins."
        ),
    },
    {
        "term": "Oligodendrocyte Differentiation — White Matter and Corpus Callosum",
        "definition": (
            "TCF4 (E-protein) drives oligodendrocyte precursor cell (OPC) differentiation "
            "and myelination in the central white matter and corpus callosum. TCF4 "
            "haploinsufficiency → impaired OPC maturation → hypomyelination/dysmyelination → "
            "delayed myelination on MRI (white matter signal changes) + thin/absent corpus "
            "callosum (~20% of Pitt-Hopkins). Serial MRI: myelination progress improves "
            "in first years but remains behind normal trajectory. Corpus callosum "
            "agenesis/hypoplasia is a structural biomarker of severe TCF4 LOF."
        ),
    },
    {
        "term": "Pitt-Hopkins Dysmorphic Features — Diagnostic Gestalt",
        "definition": (
            "Distinctive dysmorphic features of Pitt-Hopkins syndrome: (1) Beaked/peaked "
            "nasal bridge with thin nasal tip (most specific facial feature — differentiates "
            "from Angelman, Rett, FOXG1 which lack beaked nose). (2) Everted lower lip. "
            "(3) Cupid-bow upper lip. (4) Deep-set eyes. (5) Wide mouth with widely-spaced "
            "teeth. (6) Short philtrum. (7) Full cheeks. (8) Prominent supraorbital ridges. "
            "Facial gestalt + breathing episodes is sufficient clinical suspicion to order "
            "TCF4 testing. Dysmorphic features become more recognisable with age (adult "
            "face more distinctive than infantile face)."
        ),
    },
    {
        "term": "Angelman DDx — Breathing Episodes Absent in Angelman",
        "definition": (
            "Pitt-Hopkins vs Angelman syndrome: BOTH have profound ID + absent speech + "
            "epilepsy + happy sociable affect. KEY DDx: Breathing episodes (hyperventilation "
            "+ apnoea) are PATHOGNOMONIC for Pitt-Hopkins — completely absent in Angelman. "
            "Additional DDx: (1) UBE3A methylation ABNORMAL in Angelman (chromosome 15q11 "
            "deletion/UPD/IC defect) — NORMAL in Pitt-Hopkins. (2) Beaked nose in Pitt-Hopkins "
            "(not in Angelman — Angelman has normal or broad nasal bridge). (3) Both sexes "
            "equally in Pitt-Hopkins (de novo AD); Angelman affects both sexes via different "
            "mechanism. TCF4 panel + UBE3A methylation must be ordered together."
        ),
    },
    {
        "term": "Rett Syndrome DDx — Breathing Irregularity Overlap",
        "definition": (
            "Rett syndrome (MECP2, Xq28) shares breathing irregularity + hand stereotypies "
            "with Pitt-Hopkins. KEY DDx: (1) Rett has NORMAL early development (6-18 months) "
            "followed by REGRESSION (purposeful hand use lost, acquired language lost) — "
            "Pitt-Hopkins has NO regression, no normal developmental period. (2) Rett "
            "predominantly affects females (MECP2 X-linked dominant). Pitt-Hopkins equally "
            "affects both sexes. (3) MECP2 mutation in Rett; TCF4 mutation in Pitt-Hopkins. "
            "(4) Breathing in Rett: irregular respiratory rate cycles (breath-holding then "
            "forced exhalation); different pattern from TCF4 episodic hyperventilation."
        ),
    },
    {
        "term": "18q21.2 Chromosomal Deletion — Array CGH Mandatory",
        "definition": (
            "TCF4 lies at 18q21.2. Deletions encompassing TCF4 cause Pitt-Hopkins with "
            "additional contiguous gene effects. Detection: FISH/aCGH/SNP array. Sequence "
            "analysis (Sanger/NGS) detects point mutations and small indels but MISSES "
            "deletions — aCGH/SNP array is MANDATORY when sequence is negative and clinical "
            "suspicion is high. Typical deletion: 500 kb to 5 Mb. Co-deleted genes: MBD2 "
            "(DNA methylation binding), SMAD7 (TGF-beta antagonist), NEDD4L (renal tubular). "
            "Deletion size correlates with additional features beyond classic Pitt-Hopkins."
        ),
    },
    {
        "term": "Acetazolamide — Breathing Episode Treatment (Level C)",
        "definition": (
            "Acetazolamide (carbonic anhydrase inhibitor) reduces episodic hyperventilation "
            "severity in Pitt-Hopkins (Level C evidence: case reports/small series). Mechanism: "
            "by inhibiting carbonic anhydrase → blunts rapid CO2/pH cycling → reduces "
            "hypocapnia-driven apnoea trigger → shorter, less severe breathing episodes. "
            "Dose: 5-10 mg/kg/day in 2-3 divided doses. Not an AED — does not treat seizures. "
            "Used specifically for breathing episodes causing cyanosis or severe distress. "
            "Side effects: nephrolithiasis (sulfonamide), paraesthesias, hypokalemia — monitor "
            "electrolytes and renal ultrasound if used long-term."
        ),
    },
    {
        "term": "POLG Mandatory Before Valproate — Universal DEE Protocol",
        "definition": (
            "POLG (mitochondrial DNA polymerase gamma) biallelic mutations → Alpers-Huttenlocher "
            "syndrome: fatal hepatic failure triggered by valproate (VPA). POLG sequencing is "
            "universally mandatory before VPA in ALL DEE patients including Pitt-Hopkins. "
            "TCF4 haploinsufficiency has no intrinsic mitochondrial mechanism, but POLG-before-VPA "
            "is a non-negotiable universal safety gate. If POLG pathogenic biallelic variants "
            "identified → avoid VPA permanently. LFTs baseline before VPA in all cases."
        ),
    },
    {
        "term": "De Novo Inheritance — Gonadal Mosaic Risk",
        "definition": (
            "Pitt-Hopkins is caused by de novo TCF4 variants in >99% of cases — sporadic, "
            "not familial. De novo confirmed by parental testing (TCF4 sequencing of both "
            "parents). Gonadal mosaicism risk: <1% in Pitt-Hopkins (lower than some other "
            "de novo DEE genes). Empirical sibling recurrence risk: ~1% (counselled as low "
            "but non-zero due to gonadal mosaic possibility). Prenatal/preimplantation "
            "genetic diagnosis (PGT) available for confirmed TCF4 variant families. Very "
            "rare familial cases reported (inherited AD, very low penetrance)."
        ),
    },
    {
        "term": "Parvalbumin Interneurons (PV+) — E/I Balance and Epilepsy",
        "definition": (
            "Parvalbumin-expressing (PV+) fast-spiking GABAergic interneurons provide "
            "perisomatic inhibition to pyramidal neurons and are the primary determinant "
            "of cortical oscillatory synchrony and E/I balance. PV+ interneurons arise "
            "from the medial ganglionic eminence (MGE) and their migration/maturation "
            "requires TCF4-ASCL1 heterodimer activity. TCF4 haploinsufficiency → reduced "
            "PV+ interneuron complement → E/I imbalance → cortical hyperexcitability → "
            "focal/multifocal seizures. This interneuronopathy mechanism is shared with "
            "Dravet syndrome (SCN1A, where Nav1.1 loss in PV+ interneurons reduces firing)."
        ),
    },
    {
        "term": "pre-Bötzinger Complex — Brainstem Respiratory Rhythm Generator",
        "definition": (
            "The pre-Bötzinger complex (preBötC) in the ventrolateral medulla is the primary "
            "central pattern generator for respiratory rhythm. TCF4 is expressed in preBötC "
            "neurons and in the Kölliker-Fuse nucleus (pontine respiratory group). TCF4 "
            "haploinsufficiency → impaired preBötC neuron maturation and connectivity → "
            "dysrhythmic respiratory pattern generation → episodic hyperventilation alternating "
            "with apnoea (the Pitt-Hopkins breathing episode). EEG is normal during episodes "
            "(not ictal) — confirms brainstem origin, not cortical seizure. This mechanism "
            "is absent in all other Angelman-like syndromes."
        ),
    },
]

# ── HELPERS ───────────────────────────────────────────────────────────────────
def _pct(pts, key):
    n = len(pts)
    if n == 0:
        return 0
    return round(100 * sum(1 for p in pts if p.get(key)) / n)


def _mean(pts, key):
    vals = [p[key] for p in pts if isinstance(p.get(key), (int, float))]
    if not vals:
        return 0
    return round(sum(vals) / len(vals), 1)


# ── API FUNCTIONS ─────────────────────────────────────────────────────────────
def get_overview():
    pts = PATIENTS
    n = len(pts)
    etiol_dist = []
    for ec in ETIOLOGY_CATALOG:
        cat_pts = [p for p in pts if p["category"] == ec["category"]]
        etiol_dist.append({
            "etiology": ec["category"].replace("TCF4-", "").replace("-", " "),
            "n": len(cat_pts),
            "pct": round(100 * len(cat_pts) / n),
        })
    treat_summary = [
        {"drug": t["drug"].split(" —")[0].split(" (")[0], "level": t["level"][:100]}
        for t in TREATMENTS
    ]
    monitoring_summary = [
        {
            "timepoint": m["timepoint"],
            "action": m["action"][:85] + "..." if len(m["action"]) > 85 else m["action"],
        }
        for m in MONITORING[:5]
    ]
    return {
        "gene": "TCF4",
        "chromosome": "18q21.2",
        "omim_gene": "602272",
        "omim_disease": "610954",
        "protein": "TCF4 — Transcription Factor 4 / E-protein class bHLH / 667 aa",
        "aa_length": 667,
        "domains": (
            "N-terminal activation domain (aa 1-250, AD-A + AD-B + NLS1) + "
            "central linker (aa 250-548, LANA-like autoinhibitory) + "
            "bHLH core (aa 550-610: basic aa 550-573, H1 aa 574-590, loop aa 590-594, H2 aa 594-610) + "
            "C-terminal (aa 610-667: PAS-like + NLS2)"
        ),
        "inheritance": "De novo autosomal dominant (AD); haploinsufficiency or dominant negative; >99% sporadic",
        "disease_spectrum": "Pitt-Hopkins Syndrome — profound ID + absent speech + PATHOGNOMONIC breathing episodes + epilepsy + dysmorphic features",
        "unique_feature": (
            "Episodic hyperventilation alternating with apnoea — PATHOGNOMONIC for Pitt-Hopkins; "
            "absent in ALL other Angelman-like syndromes (Angelman, MRXSCH, FOXG1, Mowat-Wilson). "
            "Beaked nasal bridge + Cupid-bow lip dysmorphic gestalt. "
            "CBZ/OXC HIGH CAUTION (myoclonic/absence). LEV/VPA/CLB Level B. POLG mandatory before VPA. "
            "Acetazolamide (Level C) for breathing episodes."
        ),
        "cohort_seed": 513,
        "kpis": {
            "n_patients": n,
            "breathing_episodes_pct": _pct(pts, "breathing_episodes"),
            "focal_pct": _pct(pts, "focal"),
            "tonic_clonic_pct": _pct(pts, "tonic_clonic"),
            "myoclonic_pct": _pct(pts, "myoclonic"),
            "absence_pct": _pct(pts, "absence"),
            "infantile_spasms_pct": _pct(pts, "infantile_spasms"),
            "drug_resistant_pct": _pct(pts, "drug_resistant"),
            "absent_speech_pct": _pct(pts, "absent_speech"),
            "profound_id_pct": _pct(pts, "profound_id"),
            "corpus_callosum_absent_pct": _pct(pts, "corpus_callosum_absent"),
            "beaked_nose_pct": _pct(pts, "beaked_nose"),
            "happy_affect_pct": _pct(pts, "happy_affect"),
            "hand_stereotypies_pct": _pct(pts, "hand_stereotypies"),
            "kd_tried_pct": _pct(pts, "kd_tried"),
            "polg_tested_pct": _pct(pts, "polg_tested"),
            "acetazolamide_pct": _pct(pts, "acetazolamide"),
            "mean_aeds_failed": _mean(pts, "n_aeds_failed"),
            "seizure_free_pct": _pct(pts, "seizure_free"),
        },
        "etiology_distribution": etiol_dist,
        "treatments_summary": treat_summary,
        "monitoring_summary": monitoring_summary,
        "lifecycle": [
            {
                "stage": "Neonatal / Early Infantile (0-6 months)",
                "events": "Hypotonia; feeding difficulties; no seizures yet in most; dysmorphic features present",
                "key_action": "Clinical gestalt recognition; dysmorphology genetics referral; TCF4 panel; NICU support if hypotonic",
            },
            {
                "stage": "Late Infantile (6-18 months)",
                "events": "Seizure onset (focal/TC) or IS; developmental delay apparent; no regression",
                "key_action": "EEG; ACTH+VGB for IS (UKISS); TCF4 panel + 18q aCGH; POLG screen before VPA",
            },
            {
                "stage": "Toddler / Preschool (18 months - 5 years)",
                "events": "Breathing episodes onset (2-5y); absent/very limited speech; happy affect; hand stereotypies",
                "key_action": "Video-EEG during breathing episode (confirm non-ictal); acetazolamide if severe; Rett/Angelman exclusion",
            },
            {
                "stage": "School Age (5-12 years)",
                "events": "Stable cognitive profile (non-progressive); ongoing epilepsy; breathing episodes ongoing; dysmorphic features prominent",
                "key_action": "EHCP educational support; KD for refractory epilepsy; AED review; ophthalmology; developmental assessment",
            },
            {
                "stage": "Adolescence (12-18 years)",
                "events": "Stable-to-moderate seizure control; breathing episodes may change in pattern; GORD/constipation common",
                "key_action": "Transition to adult services; VPA contraception counselling (teratogenicity); palliative care discussion",
            },
            {
                "stage": "Adult",
                "events": "Severe disability; ongoing epilepsy; dependent care; breathing episodes persist",
                "key_action": "Adult neurology transition; long-term AED maintenance; respiratory surveillance; community care package",
            },
        ],
        "thresholds": THRESHOLDS,
        "contraindications_summary": [c["drug"] for c in CONTRAINDICATIONS],
        "clinical_highlights": [
            "Episodic hyperventilation + apnoea (PATHOGNOMONIC): EEG normal during episode — NOT ictal",
            "Beaked/peaked nasal bridge + Cupid-bow upper lip + everted lower lip — diagnostic gestalt",
            "Corpus callosum absent/thin in ~20% — MRI mandatory at diagnosis",
            "CBZ/OXC HIGH CAUTION — may worsen myoclonic/absence seizures in Pitt-Hopkins",
            "POLG sequencing MANDATORY before any VPA — universal DEE safety protocol",
            "Acetazolamide (Level C) — specific for breathing episodes, not AED action",
            "TCF4 aCGH/SNP array MANDATORY if sequence negative — 18q21.2 deletions not detectable by sequencing",
        ],
        "ddx_table": [
            {"syndrome": "Angelman (UBE3A/15q11)", "key_ddx": "No breathing episodes; UBE3A methylation ABNORMAL; normal nose (not beaked)"},
            {"syndrome": "Rett (MECP2, Xq28)", "key_ddx": "Females predominantly; REGRESSION after normal development; MECP2 mutation"},
            {"syndrome": "MRXSCH SLC9A6", "key_ddx": "Males only (X-linked); progressive cerebellar atrophy MRI; no breathing episodes; SLC9A6"},
            {"syndrome": "FOXG1 (14q12)", "key_ddx": "Frontal gyral simplification MRI; dyskinesias; no breathing episodes; FOXG1"},
            {"syndrome": "Mowat-Wilson (ZEB2)", "key_ddx": "Hirschsprung/constipation; pointed chin + upturned tip (not beaked); ZEB2"},
            {"syndrome": "CHARGE (CHD7)", "key_ddx": "Coloboma + heart + choanal atresia + ear; CHD7 multi-organ; CHD7 mutation"},
        ],
        "mandatory_workup": [
            "TCF4 gene sequencing (point mutations + small indels)",
            "18q21.2 aCGH/SNP array if sequence-negative (deletions not detectable by sequencing)",
            "Parental TCF4 testing (de novo confirmation; gonadal mosaic risk <1%)",
            "POLG sequencing MANDATORY before VPA",
            "MRI brain 3T (corpus callosum, myelination, cortical simplification)",
            "EEG (seizure type characterisation: focal/myoclonic/absence/IS)",
            "Video-EEG during breathing episode (confirm non-ictal — normal EEG = breathing episode, not seizure)",
            "Respiratory/sleep polysomnography if breathing episodes severe",
            "Ophthalmology (refractive error, optic disc)",
            "Echocardiogram if 18q21.2 deletion (cardiac defect risk)",
            "Comprehensive developmental/cognitive assessment (Griffiths/Bayley/DISCO)",
        ],
        "tier_summary": {
            "level_a": "ACTH + VGB (UKISS) for infantile spasms component",
            "level_b": "LEV, VPA (post-POLG), CLB",
            "level_c": "KD (drug-resistant), Acetazolamide (breathing episodes)",
            "high_caution": "CBZ/OXC (avoid in myoclonic/absence); LTG (caution in myoclonic); PHT IV (last resort)",
            "absolute_avoid": "VPA without POLG screen; VGB without REMS monitoring",
        },
    }


def get_breakdown():
    pts = PATIENTS
    by_cat = {}
    for p in pts:
        c = p["category"].replace("TCF4-", "").replace("-", " ")
        if c not in by_cat:
            by_cat[c] = []
        by_cat[c].append(p)

    breakdown = []
    for cat, cat_pts in by_cat.items():
        breakdown.append({
            "category": cat,
            "n": len(cat_pts),
            "breathing_episodes_pct": _pct(cat_pts, "breathing_episodes"),
            "focal_pct": _pct(cat_pts, "focal"),
            "tonic_clonic_pct": _pct(cat_pts, "tonic_clonic"),
            "myoclonic_pct": _pct(cat_pts, "myoclonic"),
            "absence_pct": _pct(cat_pts, "absence"),
            "infantile_spasms_pct": _pct(cat_pts, "infantile_spasms"),
            "drug_resistant_pct": _pct(cat_pts, "drug_resistant"),
            "absent_speech_pct": _pct(cat_pts, "absent_speech"),
            "profound_id_pct": _pct(cat_pts, "profound_id"),
            "corpus_callosum_absent_pct": _pct(cat_pts, "corpus_callosum_absent"),
            "beaked_nose_pct": _pct(cat_pts, "beaked_nose"),
            "happy_affect_pct": _pct(cat_pts, "happy_affect"),
            "hand_stereotypies_pct": _pct(cat_pts, "hand_stereotypies"),
            "kd_tried_pct": _pct(cat_pts, "kd_tried"),
            "acetazolamide_pct": _pct(cat_pts, "acetazolamide"),
            "mean_aeds_failed": _mean(cat_pts, "n_aeds_failed"),
            "seizure_free_pct": _pct(cat_pts, "seizure_free"),
            "cardiac_defect_pct": _pct(cat_pts, "cardiac_defect"),
            "growth_retard_pct": _pct(cat_pts, "growth_retard"),
        })

    etiol_details = [
        {
            "category": ec["category"].replace("TCF4-", "").replace("-", " "),
            "typical_variant": ec["typical_variant"],
            "inheritance": ec["inheritance"],
            "functional_deficit": ec["functional_deficit"],
            "description": ec["description"],
        }
        for ec in ETIOLOGY_CATALOG
    ]

    summary = {
        "breathing_episodes_pct": _pct(pts, "breathing_episodes"),
        "focal_pct": _pct(pts, "focal"),
        "tonic_clonic_pct": _pct(pts, "tonic_clonic"),
        "myoclonic_pct": _pct(pts, "myoclonic"),
        "absence_pct": _pct(pts, "absence"),
        "infantile_spasms_pct": _pct(pts, "infantile_spasms"),
        "drug_resistant_pct": _pct(pts, "drug_resistant"),
        "absent_speech_pct": _pct(pts, "absent_speech"),
        "profound_id_pct": _pct(pts, "profound_id"),
        "corpus_callosum_absent_pct": _pct(pts, "corpus_callosum_absent"),
        "beaked_nose_pct": _pct(pts, "beaked_nose"),
        "happy_affect_pct": _pct(pts, "happy_affect"),
        "hand_stereotypies_pct": _pct(pts, "hand_stereotypies"),
        "kd_tried_pct": _pct(pts, "kd_tried"),
        "polg_tested_pct": _pct(pts, "polg_tested"),
        "acetazolamide_pct": _pct(pts, "acetazolamide"),
        "mean_aeds_failed": _mean(pts, "n_aeds_failed"),
        "seizure_free_pct": _pct(pts, "seizure_free"),
    }

    return {
        "gene": "TCF4",
        "chromosome": "18q21.2",
        "cohort_size": len(pts),
        "cohort_seed": 513,
        "summary": summary,
        "by_category": breakdown,
        "etiology_details": etiol_details,
        "treatments": TREATMENTS,
        "contraindications": CONTRAINDICATIONS,
        "monitoring": MONITORING,
        "thresholds": THRESHOLDS,
    }


def get_definitions():
    return {
        "gene": "TCF4",
        "chromosome": "18q21.2",
        "protein": "TCF4 — Transcription Factor 4 / E-protein bHLH / 667 aa / De novo AD / OMIM 602272",
        "omim_gene": "602272",
        "omim_disease": "610954",
        "disease_name": "Pitt-Hopkins Syndrome (PHS) — bHLH E-protein haploinsufficiency / dominant negative",
        "inheritance": "De novo autosomal dominant (>99% sporadic); haploinsufficiency or dominant negative",
        "definitions": DEFINITIONS,
        "key_ddx": [
            "Angelman Syndrome (UBE3A/15q11): happy affect + absent speech + epilepsy — "
            "BUT: NO breathing episodes (PATHOGNOMONIC for Pitt-Hopkins); UBE3A methylation ABNORMAL "
            "(normal in TCF4); normal nasal bridge (beaked nose is TCF4-specific); "
            "UBE3A methylation + TCF4 panel must both be ordered when Angelman-like",
            "Rett Syndrome (MECP2, Xq28): breathing irregularity + hand stereotypies — "
            "BUT: REGRESSION after normal early development (no normal period in Pitt-Hopkins); "
            "predominantly females (MECP2 X-linked dominant); MECP2 mutation; "
            "hand stereotypies in Rett replace purposeful hand use (loss of function, not additive)",
            "SLC9A6 MRXSCH Christianson Syndrome (Xq26.3): Angelman-like in males — "
            "BUT: X-linked (males only); progressive cerebellar atrophy on MRI (absent in Pitt-Hopkins); "
            "no breathing episodes; no beaked nose; SLC9A6 mutation; NHE6 endosomal mechanism",
            "FOXG1 Syndrome (14q12): dyskinesias + stereotypies + frontal brain MRI — "
            "BUT: no breathing episodes; frontal gyral simplification (not corpus callosum); "
            "more severe hypotonia; FOXG1 mutation; de novo AD",
            "Mowat-Wilson Syndrome (ZEB2, 2q22.3): ID + epilepsy + distinct face — "
            "BUT: Hirschsprung disease/severe constipation (PATHOGNOMONIC for Mowat-Wilson); "
            "pointed chin + upturned nasal tip (NOT beaked nose); ZEB2 mutation; both sexes",
            "CHARGE Syndrome (CHD7, 8q12): coloboma + heart + choanal atresia + retarded growth "
            "+ genital abnormalities + ear abnormalities — BUT: CHD7 mutation; multi-organ involvement "
            "distinguishes from isolated Pitt-Hopkins; CHARGE criteria diagnostic",
        ],
        "concepts": [d["term"] for d in DEFINITIONS],
        "clinical_thresholds": THRESHOLDS,
        "key_facts": [
            "BREATHING EPISODES are PATHOGNOMONIC for Pitt-Hopkins — absent in ALL other "
            "Angelman-like syndromes. EEG is NORMAL during episodes (non-ictal; central brainstem "
            "origin from pre-Bötzinger complex TCF4 haploinsufficiency). Acetazolamide (Level C) "
            "is the specific treatment for severe breathing episodes. Always video-EEG confirm before "
            "treating as seizures.",
            "TCF4 bHLH domain missense (R576W, A597T, L600P): DOMINANT NEGATIVE mechanism — "
            "forms non-functional dimers with ASCL1/NEUROD1, more severe than haploinsufficiency alone. "
            "18q21.2 deletion: contiguous gene syndrome — aCGH/SNP array MANDATORY when sequence "
            "negative. TCF4 sequencing alone misses deletions.",
            "CBZ/OXC are HIGH CAUTION in Pitt-Hopkins — may worsen myoclonic seizures and "
            "precipitate absence-status. LTG caution if myoclonic component. LEV is preferred "
            "first-line AED (Level B, no mitochondrial/cardiac concerns). VPA Level B after "
            "POLG clearance (POLG screen is universal DEE mandatory).",
            "Corpus callosum absent/thin in ~20% — MRI mandatory at diagnosis. "
            "PV+ interneuron migration failure (TCF4-ASCL1 pathway) is the primary epilepsy "
            "mechanism. TCF4 targets: NRXN1, CNTN2, SCN1A enhancer, GABRB3. "
            "Oligodendrocyte differentiation failure → hypomyelination → delayed myelination on MRI.",
            "Pitt-Hopkins is de novo in >99% (sporadic). Gonadal mosaic risk <1%. Parental "
            "testing mandatory to confirm de novo. Prenatal/PGT available for confirmed TCF4 "
            "variant families. Non-progressive (non-degenerative) — cognitive profile stable "
            "once established. Rett DDx: no regression in Pitt-Hopkins (key feature — Rett has "
            "regression after 6-18 months normal development).",
        ],
        "mandatory_steps": [
            "TCF4 gene sequencing (exon-level: Sanger/NGS panel)",
            "18q21.2 aCGH/SNP array if sequence-negative (deletions NOT detected by sequencing)",
            "Parental TCF4 testing (de novo confirmation; document gonadal mosaic risk <1%)",
            "POLG sequencing MANDATORY before any VPA consideration",
            "MRI brain 3T (corpus callosum + myelination + cortical simplification)",
            "EEG characterisation (focal/myoclonic/absence/IS — guides AED selection)",
            "Video-EEG during breathing episode (confirm non-ictal EEG — mandatory for first episode)",
            "Polysomnography if breathing episodes severe (central apnoea severity quantification)",
            "Ophthalmology (refractive error + optic disc — common in Pitt-Hopkins)",
            "Echocardiogram if 18q21.2 deletion detected (cardiac defect risk)",
            "Comprehensive developmental assessment (Griffiths/Bayley/DISCO-2)",
            "Reproductive counselling (de novo confirmed; VPA teratogenicity in adolescent females)",
        ],
        "standards": [
            "OMIM 602272 (TCF4 gene) / OMIM 610954 (Pitt-Hopkins disease)",
            "Amiel J et al. (2007) Am J Hum Genet 80:988-993 — TCF4 mutations cause Pitt-Hopkins",
            "Zweier C et al. (2007) Am J Hum Genet 80:994-1001 — TCF4 haploinsufficiency = Pitt-Hopkins",
            "de Pontual L et al. (2009) Hum Mutat 30:669-676 — TCF4 mutational + functional spectrum",
            "UKISS protocol (ACTH + VGB for infantile spasms): Lux AL et al. (2004) Lancet 364:1485-1492",
            "VGB REMS (SHARE programme): Goldman visual field monitoring q3 months; ERG baseline",
            "POLG Working Group guidelines: POLG-before-VPA universal DEE protocol",
            "ILAE Gene Classification: TCF4 — autosomal dominant DEE, definitive",
            "ClinGen TCF4 variant curation: de novo LOF/missense = pathogenic in Pitt-Hopkins",
            "Pitt D, Hopkins IJ (1978) Aust Paediatr J 14:182-184 — original syndrome description",
        ],
    }
