#!/usr/bin/env python3
"""PME-Atlas — Complete 8-Gene Progressive Myoclonic Epilepsy Atlas
CSTB     (EPM1/Unverricht-Lundborg; AR; dodecamer repeat expansion; most common PME N.Europe; clonazepam/LEV) ·
EPM2A    (Lafora disease type 1; AR; laforin glycogen phosphatase; Lafora bodies PATHOGNOMONIC skin biopsy) ·
NHLRC1   (Lafora disease type 2; AR; malin E3 ubiquitin ligase; clinically identical to EPM2A) ·
MT-TK    (MERRF; mitochondrial; m.8344A>G 80%+; ragged red fibers muscle biopsy; maternal inheritance) ·
SCARB2   (AMRF; AR; LIMP-2/scavenger receptor B2; PME + nephrotic syndrome UNIQUE; no tonic-clonic) ·
GOSR2    (North Sea PME / EPM6; AR; p.Gly144Trp founder; elevated CK; scoliosis; early ambulation loss) ·
KCNC1    (EPM7; AD de novo; Kv3.1 K+ channel; p.Arg320His; cerebellar atrophy MRI; no dementia) ·
PRICKLE1 (EPM1B; AR; Prickle PCP protein; Wnt planar cell polarity; ULD-like phenotype; Dravet DDx)
320-patient aggregate cohort (8 × 40, seeds 1006–1013)

Progressive Myoclonic Epilepsy — Key Neurological Principles:
  - PME TRIAD: (1) PROGRESSIVE MYOCLONUS (action-sensitive, stimulus-sensitive);
    (2) GENERALISED EPILEPTIC SEIZURES (GTCS); (3) PROGRESSIVE NEUROLOGICAL DECLINE
    (cognitive, ataxia — varies by gene; CSTB/KCNC1 milder, Lafora/MERRF severe).
    The rate and degree of neurological decline distinguishes subtypes.
  - DRUG-TO-AVOID (ALL PME): carbamazepine, oxcarbazepine, phenytoin, lamotrigine, vigabatrin.
    Sodium channel blockers/GABA-transaminase inhibitors → worsen myoclonus; may precipitate
    myoclonic status epilepticus. This is the single most important management rule.
    GABAPENTIN/PREGABALIN: also generally avoided (may worsen myoclonus).
  - BACKBONE TREATMENT (ALL PME): valproate (VPA; broad-spectrum; first-line) +
    clonazepam (benzodiazepine; GABAergic; highly effective for myoclonus) +
    levetiracetam (LEV; SV2A; highly effective; particularly CSTB/KCNC1; S-enantiomer brivaracetam).
    Zonisamide: add-on; particularly useful in Lafora (reduces frequency/severity of seizures).
    Perampanel (AMPA antagonist): evidence in Lafora disease; add-on.
    Piracetam: high-dose; anti-myoclonic; historical but still used in CSTB/ULD.
  - LAFORA DISEASE (EPM2A/NHLRC1): SKIN BIOPSY pathognomonic — PAS-positive polyglucosan
    Lafora bodies in apocrine sweat gland ducts. No other PME has this finding.
    OCCIPITAL SEIZURES with formed visual hallucinations (head/eye deviation; visual phenomena)
    are characteristic. Perioral myoclonus (eating epilepsy; myoclonus triggered by eating).
    Rapid progression to vegetative state typically within 10 years of onset.
  - MERRF (MT-TK): RAGGED RED FIBERS (Gomori modified trichrome stain of skeletal muscle).
    MATERNAL INHERITANCE (no male-to-male transmission). Elevated serum lactate.
    Multiple mtDNA point mutations; heteroplasmy level correlates with severity.
    m.8344A>G accounts for >80%; m.8356T>C, m.8363G>A cover most remainder.
    Short stature, hearing loss, cardiomyopathy, lipomas (multiple symmetrical lipomatosis).
  - AMRF (SCARB2): PME + NEPHROTIC SYNDROME — unique combination in PME.
    SCARB2 encodes LIMP-2, a lysosomal membrane protein essential for cathepsin B/D delivery.
    PME in AMRF is predominantly action myoclonus, typically WITHOUT tonic-clonic seizures.
    Renal disease can dominate early course; young adult onset (20-30 years).
    Foot tremor (action tremor of feet) is an early clinical clue.
  - GOSR2 (North Sea PME): p.Gly144Trp founder mutation (Northern European — Dutch, Danish, British).
    ELEVATED SERUM CK (creatine kinase) — not found in other PMEs (suggests myopathic component).
    SCOLIOSIS (early; structural). Loss of independent ambulation within 5-10 years.
    Preserved cognition initially; later mild decline.
  - KCNC1 (EPM7): AUTOSOMAL DOMINANT; p.Arg320His recurrent de novo gain-of-toxic-function.
    Kv3.1 channel (high-frequency firing support in cerebellar/cortical neurons → loss → ataxia).
    CEREBELLAR ATROPHY on MRI (progressive). PURE PME — no/minimal cognitive decline.
    No dementia. Levetiracetam particularly effective. Treatment-responsive compared to Lafora.
  - GENETIC TESTING ALGORITHM for PME:
    Step 1: CSTB repeat expansion analysis (Southern blot/repeat-primed PCR) — if N.European.
    Step 2: EPM2A + NHLRC1 sequencing (if skin biopsy Lafora bodies, or rapid progressive course).
    Step 3: Mitochondrial genome sequencing / MT-TK hotspot (m.8344A>G) — if maternal inheritance,
      elevated lactate, ragged red fibers, multi-system (hearing, cardiac, short stature).
    Step 4: SCARB2 sequencing — if renal involvement / action myoclonus predominant / no GTCS.
    Step 5: GOSR2 — if elevated CK + scoliosis + N.European.
    Step 6: KCNC1 — if AD inheritance or de novo + cerebellar atrophy + pure myoclonus.
    Step 7: PRICKLE1 — if AR PME, ULD-like, panels negative.
    Step 8: PME gene panel / WES/WGS — if above negative.

COHORT: 8 × 40 = 320 patient slots (seeds 1006–1013; gene-specific seeds)
"""

import random

SEED_BASE = 1006

PME_GENES = [
    # ── CSTB — Unverricht-Lundborg Disease ──────────────────────────────
    {
        "gene": "CSTB", "protein": "Cystatin B",
        "alias": "Unverricht-Lundborg Disease (ULD; EPM1; OMIM #254800); AR; dodecamer repeat expansion 5'UTR; most common PME in Northern Europe; stimulus-sensitive myoclonus; preserved cognition long-term",
        "aa": "98 aa", "kDa": "11 kDa",
        "gene_class": (
            "Cysteine protease inhibitor (type I cystatin/stefin family); "
            "inhibits lysosomal cysteine proteases (cathepsins B, H, L, S). "
            "Widely expressed in neurons; functions as endogenous neuroprotective agent "
            "against lysosomal protease-mediated cell death. "
            "GENETIC MECHANISM: ~90% of cases caused by expansion of a 12-nucleotide "
            "(dodecamer) repeat motif in the 5'UTR of CSTB (normal: ≤3 repeats; "
            "pathogenic: ≥30 repeats). Expanded repeats → epigenetic silencing (H3K9 methylation) "
            "→ reduced CSTB expression → loss of cathepsin inhibition → neuronal apoptosis. "
            "~10%: point mutations or small insertions/deletions in CSTB. "
            "21q22.3; OMIM gene 601145."
        ),
        "pme_group": "Autosomal Recessive PME — Repeat Expansion",
        "subtype": "Unverricht-Lundborg Disease — Cystatin B Deficiency",
        "locus": "21q22.3", "omim_gene": 601145, "omim_disease": 254800,
        "inheritance": "Autosomal Recessive (AR). Biallelic CSTB loss (repeat expansion + mutation). Carrier frequency ~1:100 in Finland.",
        "seed_offset": 0,
        "onset_range_y": (6.0, 16.0),
        "gender": "both",
        "severity_weights": [0.25, 0.55, 0.20],
        "mutation_type": "Dodecamer repeat expansion 5'UTR",
        "pathognomonic": False,
        "phenotype": (
            "ONSET: 6-16 years (mean ~10 years). "
            "STIMULUS-SENSITIVE MYOCLONUS: action-induced and stimulus-sensitive jerks; "
            "predominantly affects limbs and face; worsens with intention, stress, light. "
            "GENERALIZED TONIC-CLONIC SEIZURES: occur in most; often morning predominance. "
            "NEUROLOGICAL PROGRESSION: MILD compared to Lafora disease — most patients "
            "retain ambulation, cognitive function relatively preserved long-term. "
            "Ataxia: present and progressive but usually moderate. "
            "No dementia in most cases (distinguishes from Lafora disease). "
            "COURSE: plateau or slow progression; Finnish patients relatively mild. "
            "Photosensitivity: prominent feature; avoid stroboscopic stimuli. "
            "EEG: generalised 3-5 Hz spike-wave and polyspike-wave complexes; photosensitivity."
        ),
        "disease": (
            "Most common PME in Northern Europe (Finland, Mediterranean). "
            "Prevalence ~1:20,000 in Finland. "
            "TREATMENT: Clonazepam (first-line for myoclonus) + Valproate (AED backbone) + "
            "Levetiracetam (highly effective; reduces myoclonus and GTCS). "
            "Piracetam (2,400-24,000 mg/day; older agent; still used in Europe for myoclonus). "
            "AVOID: carbamazepine, phenytoin, oxcarbazepine, lamotrigine (worsen myoclonus). "
            "PROGNOSIS: better than Lafora disease; patients often live normal lifespan "
            "with reasonable quality of life."
        ),
        "treatment_pearl": "LEV + VPA + clonazepam; avoid CBZ/OXC/PHT; piracetam adjunct",
    },
    # ── EPM2A — Lafora Disease Type 1 ──────────────────────────────────
    {
        "gene": "EPM2A", "protein": "Laforin",
        "alias": "Lafora Disease type 1 (EPM2; OMIM #254780); AR; glycogen phosphatase; polyglucosan body (Lafora body) accumulation; SKIN BIOPSY PATHOGNOMONIC; occipital seizures; rapid neurological deterioration",
        "aa": "331 aa", "kDa": "37 kDa",
        "gene_class": (
            "Dual-specificity phosphatase; dephosphorylates glycogen (a rare phosphatase activity). "
            "Laforin removes phosphate groups from glycogen — critical for maintaining normal "
            "glycogen structure (branching, solubility). "
            "Laforin deficiency → hyperphosphorylated, poorly branched, insoluble polyglucosan "
            "accumulates → LAFORA BODIES (intraneuronal, intramuscular, intrahepatic). "
            "Lafora bodies: PAS-positive, round, basophilic polyglucosan inclusions. "
            "Laforin also interacts with malin (NHLRC1) to form E3 ubiquitin ligase complex "
            "targeting glycogen synthesis machinery (PTG, GS) for degradation. "
            "6q24.3; OMIM gene 607566."
        ),
        "pme_group": "Autosomal Recessive PME — Glycogen Phosphatase Deficiency",
        "subtype": "Lafora Disease — Laforin Glycogen Phosphatase Deficiency",
        "locus": "6q24.3", "omim_gene": 607566, "omim_disease": 254780,
        "inheritance": "Autosomal Recessive (AR). Compound heterozygous or homozygous EPM2A mutations. No founder mutation; allelic heterogeneity (missense/nonsense/frameshift).",
        "seed_offset": 1,
        "onset_range_y": (10.0, 18.0),
        "gender": "both",
        "severity_weights": [0.05, 0.25, 0.70],
        "mutation_type": "Missense/nonsense/frameshift; no single founder",
        "pathognomonic": True,
        "phenotype": (
            "ONSET: 10-18 years (mean ~14 years); previously well. "
            "OCCIPITAL SEIZURES (visual seizures): formed visual hallucinations, head/eye deviation "
            "→ highly characteristic; early in disease. "
            "PERIORAL MYOCLONUS / EATING EPILEPSY: myoclonus triggered by eating; "
            "pathognomonic when present. "
            "PROGRESSIVE MYOCLONUS: severe; action-induced; progressively disabling. "
            "TONIC-CLONIC SEIZURES: frequent. "
            "RAPID COGNITIVE DECLINE: dementia develops within 2-5 years of onset. "
            "ATAXIA: severe; rapidly progressive. "
            "PROGRESSION: vegetative state typically within 10 years of diagnosis. "
            "EEG: occipital spike-wave; background slowing; polyspike-wave complexes. "
            "MRI: progressive cerebellar atrophy."
        ),
        "disease": (
            "SKIN BIOPSY (apocrine sweat gland ducts): PAS-positive Lafora bodies "
            "→ PATHOGNOMONIC; diagnostic in >90% of cases. "
            "Also found in muscle, liver, peripheral nerve, skin eccrine glands. "
            "TREATMENT: no disease-modifying therapy available. "
            "Symptomatic: VPA + clonazepam + zonisamide (reduces seizure burden). "
            "Perampanel (AMPA receptor antagonist): Phase 3 trial evidence; reduces myoclonus. "
            "Investigational: anti-sense oligonucleotides (reduce glycogen synthesis); "
            "mTOR inhibitors (rapamycin); gene therapy approaches. "
            "PROGNOSIS: fatal; most patients die within 10 years of diagnosis. "
            "Death from aspiration pneumonia, status epilepticus, respiratory failure."
        ),
        "treatment_pearl": "VPA + ZNS + PER (perampanel); skin biopsy confirms; anti-sense oligo investigational",
    },
    # ── NHLRC1 — Lafora Disease Type 2 ─────────────────────────────────
    {
        "gene": "NHLRC1", "protein": "Malin",
        "alias": "Lafora Disease type 2 (OMIM #254780 allelic); AR; malin E3 ubiquitin ligase; NHL repeat domain; polyglucosan accumulation identical to EPM2A; skin biopsy same as type 1",
        "aa": "395 aa", "kDa": "42 kDa",
        "gene_class": (
            "E3 ubiquitin ligase with NHL repeat domain and RING finger domain. "
            "Malin forms a functional complex with laforin (EPM2A) and acts as the "
            "ubiquitin ligase component targeting glycogen regulatory proteins for proteasomal "
            "degradation: protein targeting to glycogen (PTG), glycogen synthase (GS), "
            "laforin itself. "
            "Malin deficiency → impaired ubiquitination of glycogen synthesis regulators "
            "→ uncontrolled glycogen synthesis → hyperphosphorylated polyglucosan accumulation "
            "→ Lafora bodies (identical to EPM2A). "
            "Malin also participates in stress granule clearance (autophagy pathway). "
            "6p22.3; OMIM gene 608072."
        ),
        "pme_group": "Autosomal Recessive PME — E3 Ubiquitin Ligase Deficiency",
        "subtype": "Lafora Disease — Malin E3 Ligase Deficiency",
        "locus": "6p22.3", "omim_gene": 608072, "omim_disease": 254780,
        "inheritance": "Autosomal Recessive (AR). Biallelic NHLRC1 mutations. Allelic heterogeneity. Founder mutations in certain South Asian (Indian, Pakistani) populations.",
        "seed_offset": 2,
        "onset_range_y": (11.0, 19.0),
        "gender": "both",
        "severity_weights": [0.05, 0.25, 0.70],
        "mutation_type": "Missense/nonsense/frameshift; Indian subcontinent founder variants",
        "pathognomonic": True,
        "phenotype": (
            "CLINICALLY INDISTINGUISHABLE FROM EPM2A LAFORA DISEASE. "
            "ONSET: 11-19 years. "
            "OCCIPITAL SEIZURES with visual phenomena. "
            "PERIORAL MYOCLONUS / EATING EPILEPSY (pathognomonic when present). "
            "PROGRESSIVE MYOCLONUS, GTCS, RAPID DEMENTIA. "
            "SKIN BIOPSY: Lafora bodies — same PAS-positive polyglucosan inclusions as EPM2A. "
            "Cannot clinically distinguish NHLRC1 from EPM2A Lafora disease; "
            "genetics (sequencing) are required. "
            "Some reports suggest slightly milder course in certain NHLRC1 mutations vs EPM2A, "
            "but overlap is substantial. "
            "ATAXIA, cognitive decline, death usually within 10 years of onset."
        ),
        "disease": (
            "NHLRC1 Lafora disease accounts for approximately 40-50% of all Lafora disease; "
            "EPM2A accounts for the other 50-60%. Relative frequencies vary by ethnicity. "
            "NHLRC1 mutations more prevalent in South Asian populations (Indian subcontinent). "
            "SKIN BIOPSY: same diagnostic approach as EPM2A — pathognomonic Lafora bodies. "
            "TREATMENT: identical to EPM2A — VPA + clonazepam + zonisamide + perampanel. "
            "INVESTIGATIONAL: EPM2A and NHLRC1 share the same pathogenic pathway (glycogen "
            "hyperphosphorylation) → same therapeutic targets (GSK3β inhibitors, rapamycin, "
            "anti-sense oligonucleotides targeting PTG). "
            "GENETIC COUNSELLING: 25% recurrence risk (AR)."
        ),
        "treatment_pearl": "Identical to EPM2A; genetic test (sequencing) distinguishes EPM2A from NHLRC1; same skin biopsy",
    },
    # ── MT-TK / MERRF — Mitochondrial PME ──────────────────────────────
    {
        "gene": "MT-TK", "protein": "Mitochondrial tRNA-Lysine",
        "alias": "MERRF syndrome (Myoclonic Epilepsy with Ragged Red Fibers; OMIM #545000); mitochondrial; m.8344A>G 80%+; ragged red fibers (Gomori trichrome); maternal inheritance; elevated lactate; multisystem",
        "aa": "N/A (tRNA 75 nt)", "kDa": "N/A",
        "gene_class": (
            "Mitochondrial transfer RNA for lysine (mt-tRNA-Lys); encoded in the mitochondrial genome "
            "at position 8295-8364 on H-strand. "
            "mt-tRNA-Lys is required for mitochondrial translation of all 13 mtDNA-encoded "
            "OXPHOS subunits (Complex I, III, IV, V). "
            "GENETIC MECHANISM: m.8344A>G (>80% of MERRF) — pathogenic variant impairs "
            "tRNA-Lys aminoacylation and ribosome function → defective mitochondrial protein "
            "synthesis → OXPHOS deficiency → energy failure in high-demand tissues "
            "(brain, muscle, heart). "
            "HETEROPLASMY: variable mtDNA mutation load in different tissues → phenotypic "
            "variability. Higher heteroplasmy = more severe. "
            "Maternal inheritance: mtDNA is maternally transmitted. "
            "MT-TK; mitochondrial genome position 8295-8364."
        ),
        "pme_group": "Mitochondrial PME — OXPHOS Deficiency",
        "subtype": "MERRF Syndrome — MT-TK tRNA-Lys Mutation",
        "locus": "Mitochondrial genome (8295-8364)", "omim_gene": 590060, "omim_disease": 545000,
        "inheritance": "Mitochondrial (maternal) inheritance. No male-to-male transmission. Heteroplasmy level correlates with severity. New mutations can occur de novo.",
        "seed_offset": 3,
        "onset_range_y": (5.0, 65.0),
        "gender": "both",
        "severity_weights": [0.20, 0.45, 0.35],
        "mutation_type": "m.8344A>G (>80%); m.8356T>C; m.8363G>A",
        "pathognomonic": False,
        "phenotype": (
            "ONSET: highly variable (childhood to adulthood; mean ~25 years). "
            "MYOCLONUS: often first symptom; action-sensitive; cortical (EEG correlate). "
            "GENERALIZED SEIZURES: tonic-clonic; myoclonic status. "
            "CEREBELLAR ATAXIA: progressive; gait and limb ataxia. "
            "COGNITIVE DECLINE: present but variable; dementia in severe cases. "
            "HEARING LOSS: sensorineural; common and early. "
            "SHORT STATURE: failure of normal growth. "
            "MUSCLE WEAKNESS: proximal myopathy (ragged red fibers in muscle). "
            "CARDIOMYOPATHY: hypertrophic or dilated (Wolff-Parkinson-White preexcitation in some). "
            "MULTIPLE SYMMETRICAL LIPOMATOSIS (Madelung's disease): lipomas neck/shoulders; "
            "strongly associated with MERRF. "
            "ELEVATED SERUM LACTATE (lactic acidosis during metabolic stress). "
            "EEG: generalised spike-wave; background slowing; focal discharges."
        ),
        "disease": (
            "MUSCLE BIOPSY: Gomori modified trichrome (GMT) stain → RAGGED RED FIBERS "
            "(mitochondrial accumulation at subsarcolemmal region due to compensatory proliferation). "
            "COX-negative (cytochrome c oxidase deficient) fibers on sequential staining. "
            "SDH-positive (succinate dehydrogenase preserved — nuclear-encoded). "
            "ELECTRON MICROSCOPY: mitochondrial inclusions (parking-lot/crystalline). "
            "METABOLIC: elevated serum lactate; elevated CSF lactate; elevated lactate/pyruvate ratio. "
            "TREATMENT: no cure. Supportive: CoQ10 supplementation (antioxidant, OXPHOS cofactor); "
            "riboflavin (Complex I/II cofactor); thiamine; L-carnitine. "
            "AEDs: VPA — use caution (inhibits Complex I; can worsen energy failure — "
            "monitor lactate closely); clonazepam; LEV; zonisamide. "
            "VPA-CAUTION in MERRF (mitochondrial disease): risk of acute liver failure; "
            "carnitine co-supplementation if VPA used. "
            "AVOID: metformin (inhibits Complex I — contraindicated in mitochondrial disease)."
        ),
        "treatment_pearl": "CoQ10 + riboflavin; VPA cautious (inhibits Complex I); NO metformin; maternal inheritance counselling",
    },
    # ── SCARB2 — Action Myoclonus-Renal Failure ─────────────────────────
    {
        "gene": "SCARB2", "protein": "LIMP-2 (Lysosomal Integral Membrane Protein-2)",
        "alias": "AMRF (Action Myoclonus-Renal Failure syndrome; OMIM #254900); AR; scavenger receptor B2; nephrotic syndrome + PME UNIQUE COMBINATION; predominantly action myoclonus; no tonic-clonic; Gaucher DDx",
        "aa": "478 aa", "kDa": "54 kDa",
        "gene_class": (
            "Lysosomal integral membrane protein; member of the CD36 superfamily of scavenger receptors. "
            "LIMP-2 (SCARB2) serves as a receptor for cathepsin A (CTSA) and is essential for "
            "lysosomal targeting and delivery of cathepsins B and D. "
            "LIMP-2 is also the cellular receptor for Enterovirus 71 (EV71) — hand, foot and mouth. "
            "SCARB2 deficiency → lysosomal dysfunction → glucocerebrosidase (GBA) mislocalisation "
            "(LIMP-2 normally escorts GBA from ER/Golgi to lysosome) → glucocerebroside accumulation "
            "in neurons → progressive myoclonic epilepsy. "
            "Renal: podocyte dysfunction → nephrotic syndrome/focal segmental glomerulosclerosis (FSGS). "
            "4q21.1; OMIM gene 602257."
        ),
        "pme_group": "Autosomal Recessive PME — Lysosomal Membrane Defect",
        "subtype": "AMRF Syndrome — LIMP-2/SCARB2 Deficiency",
        "locus": "4q21.1", "omim_gene": 602257, "omim_disease": 254900,
        "inheritance": "Autosomal Recessive (AR). Biallelic SCARB2 mutations. Rare; fewer than 100 families reported worldwide.",
        "seed_offset": 4,
        "onset_range_y": (15.0, 35.0),
        "gender": "both",
        "severity_weights": [0.15, 0.45, 0.40],
        "mutation_type": "Missense/nonsense/frameshift; allelic heterogeneity",
        "pathognomonic": False,
        "phenotype": (
            "ONSET: 15-35 years (young adult onset — later than most PMEs). "
            "ACTION MYOCLONUS: predominantly action-induced myoclonic jerks; "
            "FOOT TREMOR is an early distinctive feature (tremulousness of feet on walking). "
            "NO TONIC-CLONIC SEIZURES (or very rare) — UNIQUE among PME syndromes. "
            "RENAL DISEASE: nephrotic syndrome (proteinuria, hypoalbuminaemia, oedema); "
            "renal biopsy → focal segmental glomerulosclerosis (FSGS) or minimal change; "
            "progressive to end-stage renal disease requiring dialysis or transplantation. "
            "CEREBELLAR ATAXIA: progressive. "
            "COGNITIVE DECLINE: relatively mild initially; dementia in advanced disease. "
            "NO HEPATOSPLENOMEGALY (differentiates from Gaucher disease, which also involves GBA). "
            "EEG: cortical myoclonus; giant SEPs (somatosensory evoked potentials); "
            "normal background in early disease."
        ),
        "disease": (
            "AMRF is UNIQUE in PME: the combination of PME + renal (nephrotic) syndrome "
            "is seen only in SCARB2 deficiency. "
            "DDx: Gaucher disease type III (also involves GBA pathway; has hepatosplenomegaly). "
            "RENAL MANAGEMENT: ACE inhibitor/ARB for proteinuria reduction; dialysis when ESRD; "
            "renal transplantation does NOT affect neurological course. "
            "AEDs: VPA + clonazepam + LEV (same backbone; renal dose adjustment needed for LEV). "
            "Avoid nephrotoxic AEDs. "
            "Piracetam: useful in action myoclonus. "
            "GLUCOSYLCERAMIDE REDUCTION: substrate reduction therapy (miglustat) — limited data; "
            "may slow renal progression; not proven for neurological benefit. "
            "PROGNOSIS: variable; renal disease often determines prognosis; "
            "neurological decline slower than Lafora disease."
        ),
        "treatment_pearl": "PME + nephrotic syndrome → SCARB2 first; ACEi/ARB for proteinuria; LEV dose-adjust for renal; no GTCS typical",
    },
    # ── GOSR2 — North Sea Progressive Myoclonic Epilepsy ───────────────
    {
        "gene": "GOSR2", "protein": "Golgi SNAP Receptor Complex Member 2",
        "alias": "North Sea PME (EPM6; OMIM #614018); AR; p.Gly144Trp founder mutation N.Europe; ELEVATED CK DISTINCTIVE; scoliosis early; progressive; loss of ambulation; preserved cognition relatively",
        "aa": "252 aa", "kDa": "28 kDa",
        "gene_class": (
            "GOSR2 (also known as membrin or GS28) is a Golgi SNARE protein (soluble "
            "NSF attachment protein receptor). "
            "SNARE proteins mediate vesicular fusion between Golgi compartments (ER→cis-Golgi). "
            "GOSR2 specifically mediates homotypic Golgi membrane fusion and Golgi stack assembly. "
            "GOSR2 deficiency → disrupted Golgi trafficking → glycosylation defects → "
            "neuronal and muscle pathology. "
            "GENETIC MECHANISM: p.Gly144Trp (c.430G>T) — recurrent founder mutation "
            "in Northern European populations (Dutch, Danish, British, Norwegian); "
            "homozygous or compound heterozygous. "
            "Glycine-144 is in a conserved SNARE domain; Trp substitution disrupts "
            "Golgi SNARE assembly → vesicular transport defect. "
            "17q21.32; OMIM gene 613442."
        ),
        "pme_group": "Autosomal Recessive PME — Golgi SNARE Defect",
        "subtype": "North Sea PME — GOSR2 Golgi Trafficking Deficiency",
        "locus": "17q21.32", "omim_gene": 613442, "omim_disease": 614018,
        "inheritance": "Autosomal Recessive (AR). p.Gly144Trp founder mutation (Northern European). Compound heterozygotes also reported.",
        "seed_offset": 5,
        "onset_range_y": (2.0, 10.0),
        "gender": "both",
        "severity_weights": [0.10, 0.40, 0.50],
        "mutation_type": "p.Gly144Trp founder (N.Europe); compound heterozygotes",
        "pathognomonic": False,
        "phenotype": (
            "ONSET: 2-10 years (early childhood; typically before school age). "
            "ACTION MYOCLONUS: severe; progressive; generalized. "
            "GTCS: present. "
            "SCOLIOSIS: EARLY and prominent; often requiring surgical intervention. "
            "ELEVATED SERUM CK: creatine kinase raised — DISTINCTIVE feature not seen in other PMEs "
            "(suggests myopathic/membrane transport component). "
            "ATAXIA: progressive cerebellar and/or sensory ataxia. "
            "LOSS OF AMBULATION: typically within 5-10 years of onset; wheelchair by adolescence. "
            "COGNITIVE FUNCTION: relatively preserved initially; mild decline later "
            "(better cognitive profile than Lafora disease). "
            "SHORT STATURE: common. "
            "EEG: generalised spike-wave; photosensitivity."
        ),
        "disease": (
            "North Sea PME: named for geographic clustering around North Sea coastline "
            "(Netherlands, Denmark, UK, Norway — p.Gly144Trp founder effect). "
            "SERUM CK: elevated in GOSR2 — mechanism: Golgi trafficking defect affecting "
            "muscle membrane glycoprotein processing; myopathic component. "
            "MUSCLE BIOPSY: non-specific findings; NOT ragged red fibers (not mitochondrial). "
            "TREATMENT: VPA + clonazepam + LEV (standard PME backbone). "
            "Zonisamide and perampanel as adjuncts. "
            "ORTHOPAEDIC: scoliosis monitoring + surgery (Cobb angle >40°); "
            "bracing for milder curves. "
            "PHYSIOTHERAPY: maintain function; prevent contractures. "
            "PROGNOSIS: progressive; loss of ambulation is near-universal; "
            "lifespan reduced but not as rapid as Lafora disease."
        ),
        "treatment_pearl": "Elevated CK + scoliosis + N.European child PME → GOSR2; standard PME AEDs; early scoliosis orthopedic referral",
    },
    # ── KCNC1 — Progressive Myoclonic Epilepsy 7 ───────────────────────
    {
        "gene": "KCNC1", "protein": "Kv3.1 (Voltage-gated K+ channel subfamily C member 1)",
        "alias": "EPM7 (Progressive Myoclonic Epilepsy 7; OMIM #616187); AD de novo; p.Arg320His recurrent gain-of-toxic-function; Kv3.1 K+ channel; cerebellar atrophy MRI; no dementia; LEV highly effective",
        "aa": "585 aa", "kDa": "67 kDa",
        "gene_class": (
            "KCNC1 encodes Kv3.1 (Shaw-related subfamily), a voltage-gated K+ channel "
            "with high activation threshold (positive to -10 mV). "
            "Kv3.1 channels support rapid repolarisation in fast-spiking neurons "
            "(cerebellar Purkinje cells, interneurons, auditory brainstem neurons). "
            "High-frequency firing capability depends on Kv3.1: fast activation and deactivation "
            "allows neurons to fire at 200-1000 Hz without depolarisation block. "
            "DISEASE MECHANISM: p.Arg320His (c.959G>A) — dominant gain-of-toxic-function "
            "or dominant-negative effect on channel gating → altered K+ conductance → "
            "impaired repolarisation in fast-spiking neurons → cerebellar Purkinje cell "
            "dysfunction → myoclonus and ataxia. "
            "Arg-320 is in the S4 voltage-sensor domain; substitution shifts channel kinetics. "
            "11p15.1; OMIM gene 176258."
        ),
        "pme_group": "Autosomal Dominant PME — Potassium Channelopathy",
        "subtype": "EPM7 — KCNC1 Kv3.1 Channel Dysfunction",
        "locus": "11p15.1", "omim_gene": 176258, "omim_disease": 616187,
        "inheritance": "Autosomal Dominant (AD); typically de novo (p.Arg320His recurrent de novo mutation). No family history in most cases. Familial cases with inherited AD also reported.",
        "seed_offset": 6,
        "onset_range_y": (13.0, 25.0),
        "gender": "both",
        "severity_weights": [0.30, 0.50, 0.20],
        "mutation_type": "p.Arg320His de novo (>90%); rare other KCNC1 variants",
        "pathognomonic": False,
        "phenotype": (
            "ONSET: 13-25 years (adolescent-young adult). "
            "PROGRESSIVE MYOCLONUS: action-sensitive, stimulus-sensitive; progressive. "
            "GTCS: present but often less prominent than myoclonus. "
            "CEREBELLAR ATAXIA: progressive; gait and limb; cerebellar atrophy on MRI "
            "(progressive cerebellar vermis + hemispheres atrophy — DISTINCTIVE MRI finding). "
            "COGNITIVE FUNCTION: PRESERVED or mildly impaired — a key distinguishing feature. "
            "NO DEMENTIA: major differentiator from Lafora disease and MERRF. "
            "COURSE: slower and milder progression than Lafora disease. "
            "PHOTOSENSITIVITY: present. "
            "EEG: generalised spike-wave; occipital predominance; photosensitivity; "
            "giant SEPs; background slowing with progression. "
            "MRI: progressive cerebellar atrophy."
        ),
        "disease": (
            "EPM7 is notable for: (1) AD/de novo inheritance (unlike most PMEs which are AR), "
            "(2) preservation of cognition, (3) MRI cerebellar atrophy. "
            "TREATMENT: Levetiracetam is PARTICULARLY EFFECTIVE in EPM7 "
            "(SV2A mechanism; reduces cortical excitability; anti-myoclonic). "
            "VPA + clonazepam + LEV backbone. Brivaracetam (S-enantiomer; higher SV2A affinity). "
            "Zonisamide (weak carbonic anhydrase + Na+ channel + T-type Ca2+ channel). "
            "AVOID: carbamazepine, phenytoin, oxcarbazepine, lamotrigine. "
            "DE NOVO TESTING: trio WES (proband + parents) confirms de novo p.Arg320His. "
            "GENETIC COUNSELLING: de novo → low recurrence risk for siblings; "
            "50% transmission risk if patient reproduces. "
            "PROGNOSIS: better than Lafora; patients often maintain ambulation and function "
            "for years to decades."
        ),
        "treatment_pearl": "AD/de novo + cerebellar atrophy + preserved cognition → KCNC1; LEV highly effective; avoid CBZ/PHT",
    },
    # ── PRICKLE1 — Progressive Myoclonic Epilepsy 1B ───────────────────
    {
        "gene": "PRICKLE1", "protein": "Prickle Planar Cell Polarity Protein 1",
        "alias": "EPM1B (Progressive Myoclonic Epilepsy 1B; OMIM #612437); AR; Wnt/PCP planar cell polarity signalling; REST/RE1 interactor; Unverricht-Lundborg-like phenotype; panels negative after CSTB",
        "aa": "831 aa", "kDa": "92 kDa",
        "gene_class": (
            "Prickle-1 is an adaptor protein in the Wnt/planar cell polarity (PCP) signalling "
            "pathway, functioning downstream of Frizzled receptors to regulate cell polarity, "
            "directional migration, and axonal growth cone navigation. "
            "PRICKLE1 contains LIM domains (protein-protein interactions) and a PET domain. "
            "PRICKLE1 also interacts with REST (RE1-Silencing Transcription Factor), "
            "a master repressor of neuronal gene expression — PRICKLE1 normally prevents "
            "REST from suppressing neuron-specific genes in neurons. "
            "PRICKLE1 deficiency → impaired PCP signalling → defective neuronal polarity "
            "and migration during development → epileptogenesis. "
            "Synaptic function: PRICKLE1 co-localises with PSD-95 at excitatory synapses; "
            "influences synaptic vesicle release dynamics. "
            "12q12; OMIM gene 608500."
        ),
        "pme_group": "Autosomal Recessive PME — Planar Cell Polarity Defect",
        "subtype": "EPM1B — PRICKLE1 PCP Signalling Deficiency",
        "locus": "12q12", "omim_gene": 608500, "omim_disease": 612437,
        "inheritance": "Autosomal Recessive (AR). Biallelic PRICKLE1 mutations. Rare; limited case series. Allelic heterogeneity.",
        "seed_offset": 7,
        "onset_range_y": (5.0, 15.0),
        "gender": "both",
        "severity_weights": [0.20, 0.55, 0.25],
        "mutation_type": "Missense; biallelic; allelic heterogeneity",
        "pathognomonic": False,
        "phenotype": (
            "ONSET: 5-15 years (childhood; similar to CSTB/ULD). "
            "STIMULUS-SENSITIVE MYOCLONUS: action-induced; progressive. "
            "GENERALIZED TONIC-CLONIC SEIZURES. "
            "ATAXIA: cerebellar; progressive. "
            "COGNITIVE FUNCTION: preserved or mildly impaired (similar to CSTB/ULD). "
            "COURSE: relatively slow progression; better prognosis than Lafora disease. "
            "CLINICAL OVERLAP WITH ULD (CSTB): phenotypically similar — "
            "important to sequence PRICKLE1 when CSTB expansion testing is negative "
            "in a patient with ULD-like PME. "
            "EEG: generalised spike-wave; photosensitivity; similar to CSTB. "
            "MRI: usually normal or mild cerebellar atrophy."
        ),
        "disease": (
            "EPM1B was identified as a separate genetic entity from EPM1 (CSTB). "
            "PCP pathway: Wnt/PCP mutations in other genes cause neural tube defects, "
            "cochlear hair cell orientation abnormalities, limb defects — context-dependent. "
            "In neurons, PCP regulates axon growth, synapse formation. "
            "TREATMENT: same PME backbone as CSTB/ULD — "
            "VPA + clonazepam + LEV + piracetam. "
            "Avoid: CBZ, OXC, PHT, lamotrigine. "
            "GENETIC DIAGNOSIS: typically requires panel sequencing or WES/WGS. "
            "PRICKLE1 often not included in first-tier epilepsy panels — "
            "request PME-specific or comprehensive gene panel. "
            "PROGNOSIS: generally favourable relative to Lafora; "
            "patients maintain ambulation and reasonable quality of life."
        ),
        "treatment_pearl": "ULD-like AR PME with CSTB negative → sequence PRICKLE1; VPA + LEV + clonazepam; avoid CBZ/OXC",
    },
]


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def _make_cohort(gene_data: dict) -> list:
    rng = _rng(SEED_BASE + gene_data["seed_offset"])
    cohort = []
    for i in range(40):
        age_onset = round(
            rng.uniform(*gene_data["onset_range_y"]), 1
        )
        dx_delay = round(rng.uniform(1.5, 6.0), 1)
        severity_weights = gene_data["severity_weights"]
        severity = rng.choices(
            ["mild", "moderate", "severe"],
            weights=severity_weights,
            k=1
        )[0]
        gender = (
            rng.choice(["M", "F"])
            if gene_data["gender"] == "both"
            else ("M" if gene_data["gender"] == "male" else "F")
        )
        myoclonus_severity = round(rng.uniform(3.0, 9.5), 1)  # 0-10 scale
        seizure_freq = rng.choices(
            ["rare (<1/month)", "monthly", "weekly", "daily"],
            weights=[0.15, 0.25, 0.35, 0.25],
            k=1
        )[0]
        valproate = rng.random() < 0.85
        levetiracetam = rng.random() < 0.78
        clonazepam = rng.random() < 0.80
        zonisamide = rng.random() < 0.42
        piracetam = rng.random() < 0.35
        perampanel = rng.random() < (0.55 if gene_data["gene"] in ("EPM2A", "NHLRC1") else 0.18)
        renal_involvement = gene_data["gene"] == "SCARB2" and rng.random() < 0.88
        elevated_ck = gene_data["gene"] == "GOSR2" and rng.random() < 0.90
        ragged_red = gene_data["gene"] == "MT-TK" and rng.random() < 0.85
        lafora_bodies = gene_data["gene"] in ("EPM2A", "NHLRC1") and rng.random() < 0.92
        cerebellar_atrophy_mri = gene_data["gene"] in ("KCNC1", "MT-TK") and rng.random() < 0.72
        scoliosis = gene_data["gene"] == "GOSR2" and rng.random() < 0.78
        hearing_loss = gene_data["gene"] == "MT-TK" and rng.random() < 0.65
        cardiomyopathy = gene_data["gene"] == "MT-TK" and rng.random() < 0.30
        photosensitivity = rng.random() < 0.65
        ambulatory = severity != "severe" or rng.random() < 0.30
        no_gtcs = gene_data["gene"] == "SCARB2" and rng.random() < 0.70
        cohort.append({
            "patient_id": f"PME{(SEED_BASE + gene_data['seed_offset']) * 100 + i:05d}",
            "gene": gene_data["gene"],
            "age_onset_y": age_onset,
            "dx_delay_y": dx_delay,
            "severity": severity,
            "gender": gender,
            "myoclonus_severity_score": myoclonus_severity,
            "seizure_freq": seizure_freq,
            "photosensitivity": photosensitivity,
            "ambulatory": ambulatory,
            "no_tonic_clonic": no_gtcs,
            "renal_involvement": renal_involvement,
            "elevated_ck": elevated_ck,
            "ragged_red_fibers": ragged_red,
            "lafora_bodies_biopsy": lafora_bodies,
            "cerebellar_atrophy_mri": cerebellar_atrophy_mri,
            "scoliosis": scoliosis,
            "hearing_loss": hearing_loss,
            "cardiomyopathy": cardiomyopathy,
            "valproate": valproate,
            "levetiracetam": levetiracetam,
            "clonazepam": clonazepam,
            "zonisamide": zonisamide,
            "piracetam": piracetam,
            "perampanel": perampanel,
        })
    return cohort


def get_overview() -> dict:
    all_patients = []
    for g in PME_GENES:
        all_patients.extend(_make_cohort(g))

    n = len(all_patients)
    avg_onset = round(sum(p["age_onset_y"] for p in all_patients) / n, 1)
    avg_delay = round(sum(p["dx_delay_y"] for p in all_patients) / n, 1)

    lafora_pts = [p for p in all_patients if p["lafora_bodies_biopsy"]]
    ragged_red_pts = [p for p in all_patients if p["ragged_red_fibers"]]
    renal_pts = [p for p in all_patients if p["renal_involvement"]]
    elevated_ck_pts = [p for p in all_patients if p["elevated_ck"]]
    cerabel_pts = [p for p in all_patients if p["cerebellar_atrophy_mri"]]
    scoliosis_pts = [p for p in all_patients if p["scoliosis"]]
    hearing_pts = [p for p in all_patients if p["hearing_loss"]]
    cardiomyopathy_pts = [p for p in all_patients if p["cardiomyopathy"]]
    photosensitive_pts = [p for p in all_patients if p["photosensitivity"]]
    ambulatory_pts = [p for p in all_patients if p["ambulatory"]]
    valproate_pts = [p for p in all_patients if p["valproate"]]
    lev_pts = [p for p in all_patients if p["levetiracetam"]]
    clon_pts = [p for p in all_patients if p["clonazepam"]]
    per_pts = [p for p in all_patients if p["perampanel"]]

    pme_groups = {}
    for g in PME_GENES:
        grp = g["pme_group"]
        pme_groups[grp] = pme_groups.get(grp, 0) + 40

    return {
        "atlas": "PME-Atlas",
        "title": "Complete 8-Gene Progressive Myoclonic Epilepsy Atlas",
        "genes": [g["gene"] for g in PME_GENES],
        "gene_count": len(PME_GENES),
        "total_patients": n,
        "seeds": list(range(SEED_BASE, SEED_BASE + len(PME_GENES))),
        "avg_onset_y": avg_onset,
        "avg_dx_delay_y": avg_delay,
        "lafora_bodies_n": len(lafora_pts),
        "lafora_bodies_pct": round(100 * len(lafora_pts) / n, 1),
        "ragged_red_n": len(ragged_red_pts),
        "ragged_red_pct": round(100 * len(ragged_red_pts) / n, 1),
        "renal_n": len(renal_pts),
        "renal_pct": round(100 * len(renal_pts) / n, 1),
        "elevated_ck_n": len(elevated_ck_pts),
        "elevated_ck_pct": round(100 * len(elevated_ck_pts) / n, 1),
        "cerebellar_atrophy_n": len(cerabel_pts),
        "cerebellar_atrophy_pct": round(100 * len(cerabel_pts) / n, 1),
        "scoliosis_n": len(scoliosis_pts),
        "scoliosis_pct": round(100 * len(scoliosis_pts) / n, 1),
        "hearing_loss_n": len(hearing_pts),
        "hearing_loss_pct": round(100 * len(hearing_pts) / n, 1),
        "cardiomyopathy_n": len(cardiomyopathy_pts),
        "cardiomyopathy_pct": round(100 * len(cardiomyopathy_pts) / n, 1),
        "photosensitivity_n": len(photosensitive_pts),
        "photosensitivity_pct": round(100 * len(photosensitive_pts) / n, 1),
        "ambulatory_n": len(ambulatory_pts),
        "ambulatory_pct": round(100 * len(ambulatory_pts) / n, 1),
        "valproate_n": len(valproate_pts),
        "valproate_pct": round(100 * len(valproate_pts) / n, 1),
        "levetiracetam_n": len(lev_pts),
        "levetiracetam_pct": round(100 * len(lev_pts) / n, 1),
        "clonazepam_n": len(clon_pts),
        "clonazepam_pct": round(100 * len(clon_pts) / n, 1),
        "perampanel_n": len(per_pts),
        "perampanel_pct": round(100 * len(per_pts) / n, 1),
        "pme_groups": pme_groups,
        "drug_avoid_rule": "AVOID in ALL PME: carbamazepine, oxcarbazepine, phenytoin, lamotrigine, vigabatrin — sodium channel blockers worsen myoclonus",
        "clinical_pearl": "Skin biopsy (Lafora bodies) = EPM2A/NHLRC1; Ragged red fibers + maternal = MT-TK/MERRF; PME + renal = SCARB2/AMRF; Elevated CK + scoliosis = GOSR2; Cerebellar atrophy + AD + no dementia = KCNC1/EPM7",
    }


def get_breakdown() -> dict:
    rows = []
    for g in PME_GENES:
        cohort = _make_cohort(g)
        n = len(cohort)
        avg_onset = round(sum(p["age_onset_y"] for p in cohort) / n, 1)
        avg_delay = round(sum(p["dx_delay_y"] for p in cohort) / n, 1)
        avg_myoclonus = round(sum(p["myoclonus_severity_score"] for p in cohort) / n, 1)
        photosensitive = sum(1 for p in cohort if p["photosensitivity"])
        severe = sum(1 for p in cohort if p["severity"] == "severe")
        ambulatory = sum(1 for p in cohort if p["ambulatory"])
        lev = sum(1 for p in cohort if p["levetiracetam"])
        vpa = sum(1 for p in cohort if p["valproate"])
        clon = sum(1 for p in cohort if p["clonazepam"])

        # Gene-specific hallmarks
        hallmark_n = 0
        hallmark_label = ""
        if g["gene"] in ("EPM2A", "NHLRC1"):
            hallmark_n = sum(1 for p in cohort if p["lafora_bodies_biopsy"])
            hallmark_label = "Lafora bodies on skin biopsy"
        elif g["gene"] == "MT-TK":
            hallmark_n = sum(1 for p in cohort if p["ragged_red_fibers"])
            hallmark_label = "Ragged red fibers (muscle biopsy)"
        elif g["gene"] == "SCARB2":
            hallmark_n = sum(1 for p in cohort if p["renal_involvement"])
            hallmark_label = "Renal involvement (nephrotic)"
        elif g["gene"] == "GOSR2":
            hallmark_n = sum(1 for p in cohort if p["elevated_ck"])
            hallmark_label = "Elevated serum CK"
        elif g["gene"] == "KCNC1":
            hallmark_n = sum(1 for p in cohort if p["cerebellar_atrophy_mri"])
            hallmark_label = "Cerebellar atrophy on MRI"
        elif g["gene"] == "GOSR2":
            hallmark_n = sum(1 for p in cohort if p["scoliosis"])
            hallmark_label = "Scoliosis"
        else:
            hallmark_n = photosensitive
            hallmark_label = "Photosensitivity"

        rows.append({
            "gene": g["gene"],
            "protein": g["protein"],
            "locus": g["locus"],
            "inheritance": g["inheritance"].split(".")[0],
            "pme_group": g["pme_group"],
            "subtype": g["subtype"],
            "omim_disease": g["omim_disease"],
            "n_patients": n,
            "avg_onset_y": avg_onset,
            "avg_dx_delay_y": avg_delay,
            "avg_myoclonus_score": avg_myoclonus,
            "severe_pct": round(100 * severe / n, 1),
            "ambulatory_pct": round(100 * ambulatory / n, 1),
            "photosensitive_pct": round(100 * photosensitive / n, 1),
            "vpa_pct": round(100 * vpa / n, 1),
            "lev_pct": round(100 * lev / n, 1),
            "clonazepam_pct": round(100 * clon / n, 1),
            "hallmark_n": hallmark_n,
            "hallmark_pct": round(100 * hallmark_n / n, 1),
            "hallmark_label": hallmark_label,
            "treatment_pearl": g["treatment_pearl"],
            "aa": g["aa"],
            "alias": g["alias"],
            "mutation_type": g["mutation_type"],
        })

    return {
        "atlas": "PME-Atlas",
        "total_genes": len(rows),
        "total_patients": len(PME_GENES) * 40,
        "breakdown": rows,
    }


def get_definitions() -> dict:
    return {
        "atlas": "PME-Atlas",
        "definitions": [
            {
                "term": "PME Triad and Drug-to-Avoid Rule — The Single Most Important Management Principle",
                "definition": (
                    "PROGRESSIVE MYOCLONIC EPILEPSY (PME) TRIAD: "
                    "(1) Progressive myoclonus — action-sensitive jerks, stimulus-sensitive, cortical origin. "
                    "(2) Generalized tonic-clonic seizures (GTCS). "
                    "(3) Progressive neurological deterioration — cognitive decline, ataxia, varying by gene. "
                    "The rate of deterioration is the key distinguishing feature: "
                    "Lafora disease (rapid → vegetative 10 years); CSTB/ULD (slow; survives decades); "
                    "KCNC1/EPM7 (moderate; cognition preserved); MERRF (variable). "
                    "DRUG-TO-AVOID (ABSOLUTE RULE FOR ALL PME): "
                    "CARBAMAZEPINE, OXCARBAZEPINE, PHENYTOIN, LAMOTRIGINE, VIGABATRIN, "
                    "GABAPENTIN, PREGABALIN. "
                    "Mechanism: sodium channel blockers and GABA-transaminase inhibitors "
                    "WORSEN MYOCLONUS; can precipitate myoclonic status epilepticus. "
                    "This is the most common and catastrophic management error in PME. "
                    "PME TREATMENT BACKBONE (all subtypes): "
                    "Valproate (broad-spectrum; anti-myoclonic; first-line). "
                    "Clonazepam (benzodiazepine; GABAergic; potent anti-myoclonic). "
                    "Levetiracetam (SV2A; anti-myoclonic; well-tolerated; particularly EPM1/EPM7). "
                    "Add-on: zonisamide, perampanel (especially Lafora), piracetam (ULD), "
                    "brivaracetam (higher SV2A affinity than LEV)."
                )
            },
            {
                "term": "Lafora Disease (EPM2A/NHLRC1) — Skin Biopsy, Pathogenesis, and Occipital Seizures",
                "definition": (
                    "LAFORA DISEASE: most severe PME; fatal; no disease-modifying treatment. "
                    "PATHOGENESIS: laforin (EPM2A) and malin (NHLRC1) form a complex that "
                    "regulates glycogen phosphorylation and synthesis. "
                    "Deficiency of either → hyperphosphorylated, poorly-branched, insoluble "
                    "polyglucosan → LAFORA BODIES (intraneuronal, intramuscular, hepatic inclusions). "
                    "Lafora bodies accumulate in cortical neurons → neuronal death → rapid progression. "
                    "PATHOGNOMONIC DIAGNOSTIC FINDING: "
                    "SKIN BIOPSY (apocrine sweat gland ducts): "
                    "PAS (periodic acid-Schiff) stain → PAS-POSITIVE LAFORA BODIES. "
                    "Lafora bodies = polyglucosan aggregates; bright magenta/purple on PAS. "
                    "Found in apocrine sweat gland ducts and eccrine glands. "
                    "Axillary or inguinal skin biopsy preferred (highest apocrine gland density). "
                    "No other PME has this finding — pathognomonic. "
                    "Liver biopsy (hepatocytes) and muscle biopsy also show Lafora bodies. "
                    "OCCIPITAL SEIZURES: formed visual hallucinations (coloured, circular, moving images); "
                    "head/eye deviation; occipital EEG discharges; early in disease course. "
                    "PERIORAL MYOCLONUS / EATING EPILEPSY: "
                    "myoclonus of perioral muscles triggered by eating → pathognomonic when present. "
                    "PROGNOSIS: vegetative state typically within 10 years; death in 20s-30s. "
                    "INVESTIGATIONAL: PTG/GYS1 anti-sense oligonucleotides (reduce polyglucosan); "
                    "rapamycin; GSK3β inhibitors."
                )
            },
            {
                "term": "MERRF Syndrome (MT-TK) — Ragged Red Fibers, Maternal Inheritance, Heteroplasmy",
                "definition": (
                    "MERRF SYNDROME: Myoclonic Epilepsy with Ragged Red Fibers. "
                    "GENETICS: m.8344A>G in MT-TK (mitochondrial tRNA-Lysine gene) in >80% of MERRF. "
                    "Other variants: m.8356T>C, m.8363G>A, m.8361G>A. "
                    "MATERNAL INHERITANCE: all mtDNA transmitted via oocyte → no male-to-male transmission. "
                    "A family history showing maternal-lineage affected individuals (aunts, cousins "
                    "through mother) is a key diagnostic clue. "
                    "HETEROPLASMY: variable proportions of mutant vs wild-type mtDNA in different "
                    "tissues → phenotypic variability even within families. "
                    "Higher heteroplasmy = earlier onset, more severe. Threshold effect: typically "
                    ">85% mutant heteroplasmy required for neurological manifestations. "
                    "RAGGED RED FIBERS (RRF): Gomori modified trichrome (GMT) stain of skeletal muscle → "
                    "subsarcolemmal accumulation of abnormal mitochondria → ragged red appearance. "
                    "Mechanism: compensatory mitochondrial proliferation in myofibres. "
                    "COX-NEGATIVE FIBERS: cytochrome c oxidase (Complex IV) deficient fibers → "
                    "sequential SDH/COX staining shows blue (SDH+) but not brown (COX-). "
                    "MULTI-SYSTEM: myoclonus + seizures + ataxia + cognitive decline + "
                    "HEARING LOSS + SHORT STATURE + CARDIOMYOPATHY + lactic acidosis. "
                    "MULTIPLE SYMMETRICAL LIPOMATOSIS (Madelung's disease): "
                    "symmetric lipomas around neck, upper trunk — strongly associated with MERRF. "
                    "TREATMENT: CoQ10 + riboflavin; avoid metformin (Complex I inhibitor). "
                    "VPA: use with caution (can impair mitochondrial function) + carnitine supplement."
                )
            },
            {
                "term": "AMRF Syndrome (SCARB2) — PME + Nephrotic Syndrome, LIMP-2 Biology",
                "definition": (
                    "ACTION MYOCLONUS-RENAL FAILURE (AMRF) SYNDROME: the only PME syndrome "
                    "combining neurological (myoclonus epilepsy) and renal disease (nephrotic syndrome). "
                    "SCARB2 encodes LIMP-2 (Lysosomal Integral Membrane Protein 2). "
                    "LIMP-2 FUNCTIONS: "
                    "(1) Escorts glucocerebrosidase (GBA) from ER/Golgi to lysosomes. "
                    "(2) Transports cathepsin A (CTSA) to lysosomes → CTSA regulates NEU1 (neuraminidase 1). "
                    "(3) Maintains lysosomal membrane integrity. "
                    "SCARB2 deficiency → GBA mislocalisation → glucocerebroside accumulation (lysosomal "
                    "storage disorder component) → neuronal dysfunction. "
                    "Renal: SCARB2 expressed in glomerular podocytes → SCARB2 loss → podocyte "
                    "cytoskeletal defects → focal segmental glomerulosclerosis (FSGS) → nephrotic syndrome. "
                    "UNIQUE CLINICAL FEATURES OF AMRF: "
                    "(1) PME predominantly ACTION MYOCLONUS (foot tremor early). "
                    "(2) TYPICALLY NO TONIC-CLONIC SEIZURES (unlike other PMEs). "
                    "(3) NEPHROTIC SYNDROME (proteinuria > 3.5 g/day, hypoalbuminaemia, oedema). "
                    "(4) Young adult onset (20-35 years). "
                    "RENAL MANAGEMENT: ACE inhibitor/ARB (first-line for proteinuria); "
                    "dialysis when ESRD; renal transplant (does NOT prevent neurological progression). "
                    "NEUROLOGICAL: LEV + clonazepam + VPA (dose-adjust for renal failure). "
                    "DDx: Gaucher type III (also GBA pathway; hepatosplenomegaly present; SCARB2 absent)."
                )
            },
            {
                "term": "GOSR2 (North Sea PME) — Elevated CK, Golgi SNARE Biology, Founder Mutation",
                "definition": (
                    "NORTH SEA PME (EPM6): autosomal recessive PME caused by GOSR2 p.Gly144Trp. "
                    "FOUNDER MUTATION: p.Gly144Trp (c.430G>T) — Northern European origin "
                    "(Dutch, Danish, British, Norwegian populations). "
                    "Named 'North Sea' for geographic clustering around North Sea coastline. "
                    "GOSR2 BIOLOGY: GOSR2 (membrin/GS28) is a Golgi SNARE protein mediating "
                    "vesicular fusion between ER and cis-Golgi compartments. "
                    "SNARE complex assembly requires Gly-144 (conserved Gly-XX-Gly motif); "
                    "Trp substitution disrupts Golgi membrane fusion → glycoprotein processing defect. "
                    "DISTINCTIVE CLINICAL FEATURES: "
                    "(1) ELEVATED SERUM CK (creatine kinase): not found in other PMEs; "
                    "   myopathic component; muscle biopsy non-specific (no ragged red fibers). "
                    "(2) SCOLIOSIS: structural; early; often requiring surgical correction. "
                    "(3) EARLY CHILDHOOD ONSET (2-10 years). "
                    "(4) LOSS OF AMBULATION: typically by early adolescence. "
                    "(5) Cognition: relatively preserved early (better than Lafora). "
                    "TREATMENT: standard PME backbone (VPA + clonazepam + LEV). "
                    "Scoliosis: spinal orthosis (bracing) for mild curves; "
                    "posterior spinal fusion for Cobb angle >40° or rapidly progressing. "
                    "Annual CK monitoring; chest physiotherapy if respiratory involvement. "
                    "DIAGNOSIS: GOSR2 sequencing; p.Gly144Trp hotspot. "
                    "N.European child with PME + elevated CK + scoliosis → GOSR2 first."
                )
            },
            {
                "term": "KCNC1 (EPM7) — Kv3.1 Channelopathy, De Novo AD, Cerebellar Atrophy, LEV Efficacy",
                "definition": (
                    "EPM7 (Progressive Myoclonic Epilepsy 7): caused by KCNC1 p.Arg320His. "
                    "GENETICS: AUTOSOMAL DOMINANT; the mutation is typically DE NOVO "
                    "(p.Arg320His — recurrent; >90% of EPM7 cases). "
                    "No family history in most cases (de novo). Trio WES confirms de novo. "
                    "Transmission risk if patient reproduces: 50% (AD). "
                    "Kv3.1 CHANNEL BIOLOGY: "
                    "KCNC1 (Kv3.1) is a high-threshold K+ channel. "
                    "Kv3.1 channels activate at depolarised potentials (>-10 mV) and deactivate rapidly. "
                    "This property allows neurons (cerebellar Purkinje cells, auditory brainstem neurons, "
                    "fast-spiking cortical interneurons) to repolarise rapidly between spikes → "
                    "sustains high-frequency firing (200-1000 Hz). "
                    "p.Arg320His (S4 voltage-sensor domain): shifts activation or "
                    "dominant-negative effect → impairs high-frequency firing → "
                    "Purkinje cell/cerebellar dysfunction → myoclonus + progressive cerebellar atrophy. "
                    "DISTINCTIVE FEATURES: "
                    "(1) AD/de novo inheritance (vs AR inheritance of most PMEs). "
                    "(2) CEREBELLAR ATROPHY on MRI (progressive; vermis + hemispheres). "
                    "(3) PRESERVED COGNITION (no dementia — unlike Lafora/MERRF). "
                    "(4) LEVETIRACETAM particularly effective (reduces myoclonus and GTCS). "
                    "TREATMENT: LEV (first-line) + VPA + clonazepam. "
                    "Brivaracetam (S-enantiomer; higher SV2A affinity) also effective. "
                    "AVOID: CBZ, OXC, PHT, LTG (worsens myoclonus). "
                    "PROGNOSIS: better than Lafora; cognitive preservation; "
                    "patients often functional for years to decades."
                )
            },
            {
                "term": "PME Genetic Testing Algorithm — Step-by-Step Workup",
                "definition": (
                    "PROGRESSIVE MYOCLONIC EPILEPSY — GENETIC TESTING ALGORITHM: "
                    "STEP 1 — CLINICAL CLASSIFICATION: "
                    "  (a) Ethnicity (Finnish/N.European → CSTB first). "
                    "  (b) Skin biopsy (Lafora bodies → EPM2A/NHLRC1). "
                    "  (c) Maternal family history + elevated lactate → MT-TK/MERRF. "
                    "  (d) PME + renal disease → SCARB2. "
                    "  (e) Elevated CK + scoliosis + N.European → GOSR2. "
                    "  (f) AD/de novo + cerebellar atrophy + no dementia → KCNC1. "
                    "STEP 2 — TARGETED TESTING: "
                    "  CSTB: dodecamer repeat expansion (repeat-primed PCR / Southern blot). "
                    "  EPM2A/NHLRC1: Sanger sequencing (missense/nonsense). "
                    "  MT-TK: mtDNA hotspot testing m.8344A>G (PCR-RFLP or sequencing). "
                    "  SCARB2: sequencing + deletion/duplication. "
                    "  GOSR2: p.Gly144Trp hotspot first. "
                    "  KCNC1: p.Arg320His hotspot → trio WES if negative. "
                    "STEP 3 — PANEL / WES: "
                    "  PME gene panel or trio WES/WGS if targeted tests negative. "
                    "  Include PRICKLE1, KCTD7, LYST, MFSD8 on panel. "
                    "EEG FEATURES (all PME): "
                    "  Generalised 3-5 Hz spike-wave and polyspike-wave complexes. "
                    "  Giant SEPs (somatosensory evoked potentials) = cortical myoclonus. "
                    "  Occipital spike-wave = Lafora disease. "
                    "  Background slowing = severity/progression marker. "
                    "  Photosensitivity = present in most PME syndromes."
                )
            },
            {
                "term": "CSTB (Unverricht-Lundborg Disease) — Dodecamer Repeat, Cystatin B Biology, Prognosis",
                "definition": (
                    "UNVERRICHT-LUNDBORG DISEASE (ULD / EPM1): most common PME in Northern Europe; "
                    "best prognosis among PME syndromes. "
                    "GENETIC MECHANISM: 12-nucleotide (dodecamer) CCCCGCCCCGCG repeat expansion "
                    "in the 5'-untranslated region (5'UTR) of CSTB gene. "
                    "Normal: ≤3 repeats. Pathogenic: ≥30 repeats. "
                    "Repeat expansion → H3K9 methylation (epigenetic silencing) → reduced CSTB mRNA. "
                    "CYSTATIN B: endogenous inhibitor of lysosomal cysteine cathepsins (B, H, L, S). "
                    "Without cystatin B → cathepsins leak from lysosomes → "
                    "aberrant protease activity → mitochondrial damage → neuronal apoptosis. "
                    "REPEAT TESTING: standard sequencing misses expansions; requires "
                    "repeat-primed PCR or Southern blot. This is a MANDATORY technical point: "
                    "negative CSTB sequencing does NOT exclude ULD. "
                    "CLINICAL COURSE: "
                    "  Onset 6-16 years; stimulus-sensitive myoclonus; GTCS; ataxia. "
                    "  PROGRESSIVE BUT SLOW: Finnish patients may plateau after initial progression. "
                    "  NO DEMENTIA in most cases (key distinction from Lafora disease). "
                    "  Life expectancy often normal or near-normal. "
                    "TREATMENT: VPA + clonazepam + LEV + piracetam. "
                    "AVOID: CBZ, OXC, PHT (worsen myoclonus dramatically). "
                    "PIRACETAM: high-dose (up to 24,000 mg/day); reduces action myoclonus; "
                    "well-tolerated; evidence in multiple clinical trials."
                )
            },
        ]
    }


if __name__ == "__main__":
    import json
    ov = get_overview()
    print("=== PME ATLAS OVERVIEW ===")
    print(f"Genes: {ov['genes']}")
    print(f"Total patients: {ov['total_patients']}")
    print(f"Avg onset: {ov['avg_onset_y']} y")
    print(f"Avg dx delay: {ov['avg_dx_delay_y']} y")
    print(f"Lafora bodies: {ov['lafora_bodies_n']} ({ov['lafora_bodies_pct']}%)")
    print(f"Ragged red fibers: {ov['ragged_red_n']} ({ov['ragged_red_pct']}%)")
    print(f"Renal involvement: {ov['renal_n']} ({ov['renal_pct']}%)")
    print(f"Elevated CK: {ov['elevated_ck_n']} ({ov['elevated_ck_pct']}%)")
    print(f"Cerebellar atrophy: {ov['cerebellar_atrophy_n']} ({ov['cerebellar_atrophy_pct']}%)")
    print(f"Scoliosis: {ov['scoliosis_n']} ({ov['scoliosis_pct']}%)")
    print(f"PME groups: {ov['pme_groups']}")
    bd = get_breakdown()
    print(f"\n=== BREAKDOWN ({bd['total_genes']} genes) ===")
    for g in bd["breakdown"]:
        print(f"  {g['gene']}: {g['n_patients']} pts, onset {g['avg_onset_y']}y, delay {g['avg_dx_delay_y']}y, myoclonus {g['avg_myoclonus_score']}/10, severe {g['severe_pct']}%")
    defs = get_definitions()
    print(f"\n=== DEFINITIONS: {len(defs['definitions'])} terms ===")
