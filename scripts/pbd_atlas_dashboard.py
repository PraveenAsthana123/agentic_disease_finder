#!/usr/bin/env python3
"""PBD-Atlas — Complete 8-Gene Peroxisomal Biogenesis Disorders Atlas
PEX1 · PEX6 · PEX26 · PEX10 · PEX2 · PEX3 · PEX5 · PEX16
320-patient aggregate cohort (8 × 40, seeds 888–895)

Peroxisomal Biogenesis Disorders (PBD) facts:
  - Inherited defects in peroxisome biogenesis proteins (peroxins, PEX genes);
    result in failure to import one or both classes of peroxisomal matrix proteins
    (PTS1 and PTS2 cargo) OR failure to form the peroxisomal membrane itself.
  - Collectively called PBD-ZSD (Zellweger Spectrum Disorders) — single clinical continuum:
      ZS  (Zellweger Syndrome): most severe; neonatal hypotonia, seizures, absent peroxisome ghosts
      NALD (Neonatal ALD):    intermediate; survive infancy; progressive leukodystrophy
      IRD  (Infantile Refsum Disease): mildest; survival into adult life; slow regression
  - Incidence: ~1 in 50,000–100,000; PEX1 accounts for ~70% of all ZSD.
  - Biochemical hallmarks (virtually all PBD): elevated VLCFA (C26:0 and C24:0/C22:0 ratio),
    low erythrocyte plasmalogens, elevated phytanic acid + pipecolic acid; peroxisomal catalase
    is CYTOSOLIC not peroxisomal (unlike PEX11B where catalase import is PRESERVED).
  - PEX1 founder allele G843D is the most common PBD variant worldwide (~40% of ZSD alleles).
  - Lorenzo Oil is NOT effective in PBD-ZSD (works in X-ALD [ABCD1] which is a
    peroxisomal TRANSPORTER disorder, NOT a biogenesis disorder) — critical prescribing error.
  - No HSCT/BMT indicated for PBD-ZSD (unlike X-ALD early cerebral form).
  - DHA (docosahexaenoic acid) supplementation: partial benefit for retinal/visual outcomes;
    Level C evidence; does NOT stop neurological regression.
  - ABCD1 (X-ALD) is the most important DDx: single-gene ABCD1 transporter, NOT a PBD;
    VLCFA elevated BUT plasmalogens NORMAL; no dysmorphic features; no ZSD spectrum.

ATLAS SCOPE (8 landmark PBD genes — PEX1/6/26 [AAA-ATPase axis] + RING triad PEX2/10 + PEX3/5/16):
  AAA-ATPase recycling complex (PEX1 · PEX6 · PEX26):
    PEX1   — 1209 aa, 3q27.3  [MOST COMMON ZSD, ~70%; G843D Northern-European founder]
    PEX6   — 980 aa,  6p21.1  [2nd most common, ~15%; A778T frequent allele]
    PEX26  — 305 aa,  22q11.21 [3rd most common, ~6%; anchors PEX1/6 to membrane]
  RING-finger E3 ubiquitin-ligase triad (PEX2 · PEX10 · PEX12 — PEX12 has individual dashboard):
    PEX10  — 326 aa,  1p36.32  [monoubiquitinates PEX5 for recycling; ZSD all tiers]
    PEX2   — 305 aa,  8q21.13  [polyubiquitinates PEX5 for ERAD; severe ZS/NALD]
  Membrane biogenesis + PTS receptor axis (PEX3 · PEX5 · PEX16):
    PEX3   — 373 aa,  6q24.2   [de novo PMP targeting; complete loss → no peroxisomes; severe ZS]
    PEX5   — 639 aa,  12p13.31 [PTS1 cytosolic receptor; ZSD all tiers; key cargo importer]
    PEX16  — 336 aa,  11p11.2  [PMP membrane insertion growth/division; ZS/NALD predominant]

CRITICAL CLINICAL RULES:
  1. PEX1 G843D: MOST COMMON ZSD allele (~40% of all ZSD alleles; ~70% of ZSD from PEX1);
     G843D/G843D homozygous → NALD or IRD (NOT ZS — partial function retained);
     G843D/null compound het → ZS or NALD (severity depends on null allele).
  2. CATALASE LOCALISATION — key differentiator from PEX11B:
     In all PBD-ZSD (PEX1/2/3/5/6/10/16/26): catalase is CYTOSOLIC (import machinery absent);
     In PEX11B (fission defect): catalase import is PRESERVED (import machinery intact);
     Immunofluorescence catalase staining is the first-tier functional test.
  3. LORENZO OIL ABSOLUTE CONTRAINDICATION in PBD-ZSD:
     Lorenzo Oil (erucic acid + oleic acid) reduces VLCFA in X-ALD (ABCD1 transporter defect)
     but has NO benefit and potential harm in PBD-ZSD.
     The VLCFA elevation in ZSD is due to absent beta-oxidation (no functional peroxisomes);
     Lorenzo Oil does not restore peroxisome biogenesis.
  4. HSCT NOT INDICATED for PBD-ZSD (unlike X-ALD cerebral form):
     PBD is a cell-autonomous multi-organ defect; donor cells cannot rescue non-CNS organs;
     no published ZSD series showing HSCT benefit; avoid unless in active trial.
  5. DHA SUPPLEMENTATION — partial retinal/auditory benefit only (Level C):
     DHA (C22:6n-3) is synthesised in peroxisomes; ZSD → DHA deficiency;
     oral DHA (20-100 mg/kg/day) raises plasma DHA and may stabilise retinal function;
     does NOT halt neurological regression; NOT disease-modifying for CNS or liver.
  6. PLASMALOGENS — unique to PBD-ZSD; absent in X-ALD:
     Plasmalogen synthesis is entirely peroxisomal (GNPAT/AGPS steps);
     all true PBD genes → low erythrocyte plasmalogens (ethanolamine plasmalogens);
     X-ALD (ABCD1): VLCFA elevated, plasmalogens NORMAL;
     plasmalogen measurement separates ZSD from X-ALD definitively.
  7. PHYTANIC ACID and PIPECOLIC ACID — secondary markers (not specific to ZSD):
     Phytanic acid elevated in ZSD (alpha-oxidation absent) AND in Refsum disease (PHYH/PEX7);
     pipecolic acid elevated in ZSD AND in hyperpipecolicacidaemia (PEX7);
     VLCFA + plasmalogens are primary; phytanic + pipecolic confirm peroxisomal dysfunction.
  8. NEONATAL ALD (NALD) ≠ X-ALD:
     NALD is the intermediate ZSD phenotype caused by PEX gene mutations;
     X-ALD (X-linked ALD) is caused by ABCD1 (Xq28) — peroxisomal ABCD1 transporter;
     both have VLCFA elevation and demyelination BUT different biochemistry, genetics, treatment.
     Key test: erythrocyte plasmalogens NORMAL in X-ALD, LOW in NALD/ZSD.

COHORT: 8 × 40 = 320 patient slots (seeds 888–895; gene-specific seeds)
"""

import random

SEED_BASE = 888

# ── All 8 PBD Genes ──────────────────────────────────────────────────────────────
PBD_GENES = [
    # ── PEX1 — AAA-ATPase, most common ZSD gene ──────────────────────────────────
    {
        "gene": "PEX1", "alias": "PEX1 — AAA-ATPase (most common ZSD, ~70% of PBD)",
        "aa": "1209 aa", "kDa": "143 kDa",
        "gene_class": "AAA-ATPase",
        "pbd_subgroup": "AAA-ATPase recycling complex (PEX1·PEX6·PEX26)",
        "locus": "3q27.3", "omim_gene": 602136,
        "phenotype": "ZSD all tiers (ZS/NALD/IRD); ~70% of all PBD-ZSD; G843D Northern-European founder → NALD/IRD (partial function); null → ZS",
        "disease": (
            "PEX1 biallelic loss → Peroxisomal Biogenesis Disorder-Zellweger Spectrum Disorder "
            "(PBD-ZSD, OMIM #601539). PEX1 encodes the largest peroxin (1209aa, 143kDa), an "
            "AAA+ ATPase (two AAA domains: D1 and D2) that forms a heterohexameric ring with "
            "PEX6 (also AAA-ATPase). The PEX1/PEX6 complex, anchored to the peroxisomal membrane "
            "by PEX26, provides the mechanical energy (ATP hydrolysis) to extract mono-ubiquitinated "
            "PEX5 from the peroxisomal translocon after cargo delivery — recycling PEX5 back to "
            "the cytosol for the next round of PTS1-cargo import. PEX1 loss → PEX5 trapped on "
            "peroxisomal membrane → degraded (cannot be recycled) → PTS1 import fails → "
            "peroxisomal matrix proteins remain cytosolic → VLCFA accumulate, plasmalogens absent, "
            "phytanic/pipecolic acid elevated. Clinical: ZSD full spectrum — ZS (neonatal: profound "
            "hypotonia, seizures, absent Moro, facial dysmorphism [wide fontanelles, high forehead, "
            "epicanthal folds, hypoplastic midface], die within weeks/months), NALD (survive infancy: "
            "progressive leukodystrophy, retinal dystrophy, SNHL), IRD (mildest: adult survival, "
            "peripheral neuropathy, retinitis pigmentosa, anosmia). Most common ZSD gene (~70%). "
            "PEX1 p.Gly843Asp (G843D, c.2528G>A) is the most frequent variant worldwide (~40% of "
            "all ZSD alleles; ~55-60% in Northern-European ZSD cohorts); G843D is a hypomorphic "
            "missense — residual protein function → milder phenotype (NALD/IRD). "
            "G843D/G843D homozygous → NALD or IRD phenotype (NOT ZS). Null/G843D → ZS or NALD. "
            "Null/null → uniformly severe ZS. Incidence: ~1 in 70,000 for all ZSD; PEX1 alone ~1 in 100,000."
        ),
        "inheritance": "Autosomal recessive. PEX1 3q27.3. Most common ZSD gene (~70% of PBD-ZSD). Pan-ethnic; G843D enriched in Northern European.",
        "hallmark": (
            "PEX1/ZSD HALLMARKS: "
            "(1) G843D GENOTYPE-PHENOTYPE: G843D/G843D → NALD or IRD (NEVER ZS — partial AAA function retained); "
            "G843D/null → ZS or NALD (severity set by null allele); null/null → severe ZS; "
            "G843D genotype predicts milder outcome — important for prognosis counselling; "
            "(2) PEX5 TRAPPED ON MEMBRANE: PEX1 loss → ubiquitinated PEX5 cannot be extracted → "
            "PEX5 degraded → PTS1 import cycle stalls; biochemical consequence: no matrix protein import; "
            "(3) CATALASE CYTOSOLIC: immunofluorescence shows punctate peroxisomal catalase in normal cells; "
            "in ZSD it is diffuse cytosolic — most important single functional test in fibroblasts; "
            "(4) VLCFA ELEVATED (C26:0 plasma): universally elevated in all PBD genes; "
            "primary NBS-adjacent screening marker; also elevated in X-ALD (ABCD1) — "
            "must check plasmalogens to distinguish; "
            "(5) ERYTHROCYTE PLASMALOGENS LOW: plasmalogen synthesis is 100% peroxisomal; "
            "all true biogenesis defects → absent/low; NORMAL in X-ALD (ABCD1 transporter only); "
            "this test SEPARATES ZSD from X-ALD definitively; "
            "(6) RETINAL DYSTROPHY: universal in surviving ZSD (NALD/IRD); "
            "cone-rod dystrophy pattern; early macular involvement; ERG flat/severely reduced; "
            "DHA supplementation may stabilise (Level C); "
            "(7) SNHL (SENSORINEURAL): 70-80% in NALD/IRD; cochlear implants evaluated case-by-case; "
            "(8) FACIAL DYSMORPHISM (ZS only): wide fontanelles, flat occiput, high forehead, "
            "epicanthal folds, depressed nasal bridge, large earlobes — Zellweger gestalt; "
            "not present in milder NALD/IRD; "
            "(9) NO HSCT: unlike X-ALD cerebral form where HSCT halts progression in early disease; "
            "PBD is multi-organ cell-autonomous defect; HSCT not beneficial; "
            "(10) LORENZO OIL ABSOLUTE CI: erucic acid + oleic acid → reduces VLCFA in X-ALD "
            "(ABCD1 transporter); has NO benefit in ZSD; enzyme deficiency cannot be bypassed by oil"
        ),
        "key_ddx": (
            "PEX1 ZSD DDx: (1) X-ALD (ABCD1): VLCFA elevated BUT plasmalogens NORMAL — key separator; "
            "X-linked (males primarily affected); adrenal insufficiency prominent; HSCT effective in early CNS form; "
            "(2) PEX6-ZSD: clinically indistinguishable; distinguish by sequencing; A778T frequent allele; "
            "(3) PEX26-ZSD: milder phenotype predominant (NALD/IRD); smaller gene; "
            "(4) Refsum disease (PHYH/PEX7): phytanic acid elevated BUT VLCFA NORMAL; "
            "peripheral neuropathy + RP + cerebellar ataxia; phytanic acid diet highly effective; "
            "(5) Mitochondrial disease: lactic acidosis; VLCFA normal; "
            "(6) PEX11B: VLCFA elevated BUT catalase peroxisomal (import intact — fission defect only); "
            "photosensitivity 71% HALLMARK; no pachygyria; normal plasmalogens or borderline"
        ),
        "diet_treatment": "Low phytanic acid diet (avoid dairy fat, ruminant fat, fish oil). DHA (docosahexaenoic acid) 20-100mg/kg/day oral — partial retinal benefit (Level C); does NOT halt CNS regression. Fat-soluble vitamins A, D, E, K (malabsorption from liver/bile dysfunction). Lorenzo Oil CONTRAINDICATED (no benefit in ZSD; benefit only in X-ALD). Frequent feeds (fasting avoided — impaired gluconeogenesis in severe forms).",
        "gene_therapy_status": "No approved gene therapy. PEX1 mRNA therapy (RESTORE-PBD approach) in early preclinical development. PEX1 complementation in patient fibroblasts restores peroxisome function in vitro. No clinical trials as of 2025. AAV delivery challenged by large PEX1 coding sequence (>3.6kb).",
        "critical_ci": (
            "CRITICAL CI: (1) Lorenzo Oil — CONTRAINDICATED in ZSD (no enzyme to restore; "
            "works only in X-ALD ABCD1 transporter defect); prescribing error risk is HIGH; "
            "(2) HSCT — NOT indicated in PBD-ZSD (unlike X-ALD cerebral form); "
            "(3) Assuming G843D/G843D = ZS — INCORRECT: G843D homozygous → NALD/IRD (hypomorphic); "
            "(4) Missing X-ALD by not checking plasmalogens — VLCFA elevated in both; "
            "(5) High phytanic acid dietary sources in diet — dairy fat, beef fat, fish oil trigger acute crisis"
        ),
        "nbs_marker": "Not in standard NBS; C26:0 VLCFA detectable by plasma MS/MS (add-on in some expanded NBS programmes). Clinical diagnosis by plasma VLCFA + erythrocyte plasmalogens + plasma phytanic/pipecolic acid + fibroblast catalase IF + PEX1 sequencing (PEX gene panel or WES).",
        "key_biomarker": "Plasma VLCFA (C26:0 elevated; C24:0/C22:0 ratio elevated). Erythrocyte plasmalogens (low). Plasma phytanic acid (elevated). Plasma pipecolic acid (elevated). Fibroblast catalase immunofluorescence (diffuse cytosolic — absent peroxisomal puncta).",
        "severity_spectrum": "Severe ZS (neonatal; die <12 months; no peroxisomes; deep brain malformations) → NALD (survive infancy; progressive leukodystrophy) → IRD (mild; adult survival; peripheral neuropathy + RP + anosmia). G843D genotype → NALD/IRD.",
        "founder_variant": "p.Gly843Asp (G843D, c.2528G>A) — Northern-European founder, ~40% of all ZSD alleles; hypomorphic → NALD/IRD (not ZS)",
        "key_variants": [
            "p.Gly843Asp (G843D) — most common; Northern-European; hypomorphic → NALD/IRD",
            "c.2097_2098insT — null frameshift; severe ZS when compound-het with null",
            "p.Ile700fs — null; ZS phenotype",
            "p.Arg798Gln — D2-AAA domain interface; intermediate NALD",
            "p.Thr758Ile — AAA-ATPase Walker A; moderate NALD",
        ],
        "seed": SEED_BASE + 0,
    },
    # ── PEX6 — AAA-ATPase, 2nd most common ZSD gene ──────────────────────────────
    {
        "gene": "PEX6", "alias": "PEX6 — AAA-ATPase (2nd most common ZSD, ~15%)",
        "aa": "980 aa", "kDa": "116 kDa",
        "gene_class": "AAA-ATPase",
        "pbd_subgroup": "AAA-ATPase recycling complex (PEX1·PEX6·PEX26)",
        "locus": "6p21.1", "omim_gene": 601498,
        "phenotype": "ZSD all tiers; ~15% of PBD-ZSD; A778T allele frequent; clinically indistinguishable from PEX1-ZSD by phenotype alone",
        "disease": (
            "PEX6 biallelic loss → PBD-ZSD (OMIM #614859). PEX6 encodes a 980aa (116kDa) AAA+ ATPase "
            "that forms a tight heterohexameric ring with PEX1 (3 PEX1 + 3 PEX6 subunits, alternating). "
            "The PEX1/PEX6 complex — tethered to the peroxisomal membrane via PEX26 (the PEX6 C-terminus "
            "binding partner) — uses concerted ATP hydrolysis to extract mono-ubiquitinated PEX5 from "
            "the docking/translocation complex (PEX14 pore), recycling PEX5 back to cytosol. "
            "PEX6 loss disrupts the AAA ring → PEX5 trapped → PTS1 import stalls → no functional "
            "matrix proteins in peroxisome → same biochemical consequences as PEX1 loss. "
            "Second most common PBD gene (~10-15% of ZSD cases, ~5-10% of alleles). "
            "Clinical: ZSD full spectrum — ZS, NALD, IRD — phenotypically indistinguishable from "
            "PEX1-ZSD. Severity is genotype-driven (null/null → ZS; missense/missense → NALD/IRD). "
            "p.Ala778Thr (A778T) is a common partial-function allele; A778T/A778T → milder phenotype. "
            "p.Arg860Trp is the French-Canadian founder allele. "
            "ZS patients: VLCFA markedly elevated, plasmalogens absent, brain malformations "
            "(pachygyria/polymicrogyria, neuronal migration defect), die <12 months. "
            "NALD/IRD patients: retinal dystrophy, SNHL, hepatopathy, peripheral neuropathy, "
            "survive years to decades."
        ),
        "inheritance": "Autosomal recessive. PEX6 6p21.1. 2nd most common ZSD gene (~10-15%). Pan-ethnic; p.Arg860Trp French-Canadian founder.",
        "hallmark": (
            "PEX6/ZSD HALLMARKS: "
            "(1) CLINICALLY INDISTINGUISHABLE from PEX1-ZSD — same ZS/NALD/IRD spectrum; "
            "distinguish by gene panel sequencing (not by phenotype alone); "
            "(2) A778T HYPOMORPHIC: p.Ala778Thr reduces but does not abolish PEX6 AAA function; "
            "A778T/A778T or A778T/mild allele → NALD/IRD; A778T/null compound het → ZS or NALD; "
            "(3) PEX6 REQUIRES PEX1 FOR STABILITY: PEX6 protein is destabilised without PEX1 binding; "
            "mutations in PEX1 that destabilise the heterohexamer can show PEX6 protein reduction too; "
            "(4) PEX6 C-TERMINUS BINDS PEX26: PEX26 membrane anchor tethers the whole PEX1/PEX6 "
            "ring to the peroxisomal membrane — PEX6 mutations that disrupt PEX26 binding mimic PEX26 loss; "
            "(5) ALL PBD BIOCHEMICAL MARKERS PRESENT: VLCFA elevated, plasmalogens low, catalase cytosolic; "
            "(6) FRENCH-CANADIAN FOUNDER p.Arg860Trp: enriched in Quebec Saguenay-Lac-Saint-Jean region; "
            "population screening in this cohort suggested for first-degree relatives; "
            "(7) Phenotype range mirrors PEX1: residual AAA activity = milder disease"
        ),
        "key_ddx": (
            "PEX6 DDx: (1) PEX1-ZSD: clinically identical; distinguish by gene panel; "
            "PEX1 is 7× more common — sequence PEX1 first; "
            "(2) PEX26-ZSD: also anchors PEX1/PEX6 ring; milder predominant; smaller gene; "
            "(3) X-ALD: VLCFA elevated, plasmalogens NORMAL — key; "
            "(4) Refsum disease (PHYH/PEX7): phytanic acid elevated, VLCFA normal; "
            "adult onset peripheral neuropathy + RP + ataxia; treatable by phytanic acid diet"
        ),
        "diet_treatment": "Same as PEX1-ZSD: low phytanic acid diet; DHA oral supplementation (Level C, retinal benefit only); fat-soluble vitamins (A,D,E,K); Lorenzo Oil CONTRAINDICATED; fasting avoidance.",
        "gene_therapy_status": "No approved gene therapy. PEX6 smaller than PEX1 (2940bp coding) — more amenable to AAV delivery. Complementation studies in patient cells confirm pathogenicity. No clinical trials as of 2025.",
        "critical_ci": (
            "CRITICAL CI: (1) Lorenzo Oil — contraindicated (same as PEX1); "
            "(2) HSCT — not indicated in ZSD (unlike X-ALD); "
            "(3) Mistaking NALD for X-ALD — check plasmalogens; "
            "(4) A778T/A778T genotype counselled as ZS prognosis — INCORRECT: A778T homozygous → NALD/IRD; "
            "(5) High phytanic acid foods — ruminant fat, fish oil"
        ),
        "nbs_marker": "Not in standard NBS. Same VLCFA-based expanded NBS approach as PEX1. Gene panel (PEX1-PEX6-PEX26-etc.) or WES for diagnosis.",
        "key_biomarker": "Plasma VLCFA elevated (same pattern as PEX1). Erythrocyte plasmalogens low. Fibroblast catalase cytosolic. PEX6 sequencing + protein western (PEX6 reduced/absent; PEX1 also secondarily reduced in severe alleles).",
        "severity_spectrum": "Severe ZS (null/null → die <12 months) → NALD (null/A778T or mild) → IRD (A778T/A778T or two mild alleles). French-Canadian A860W founder → moderate NALD.",
        "founder_variant": "p.Arg860Trp — French-Canadian (Saguenay-Lac-Saint-Jean); p.Ala778Thr — pan-ethnic hypomorphic allele",
        "key_variants": [
            "p.Ala778Thr (A778T) — common pan-ethnic hypomorphic; NALD/IRD phenotype",
            "p.Arg860Trp — French-Canadian founder; moderate NALD",
            "p.Glu566del — in-frame deletion; AAA domain; severe",
            "c.1802_1803insC — null frameshift; ZS when compound-het with null",
            "p.Val625Gly — AAA-D2 domain; intermediate NALD",
        ],
        "seed": SEED_BASE + 1,
    },
    # ── PEX26 — Membrane anchor for PEX1/PEX6 AAA complex ───────────────────────
    {
        "gene": "PEX26", "alias": "PEX26 — PEX1/PEX6 membrane anchor (3rd most common ZSD, ~6%)",
        "aa": "305 aa", "kDa": "34 kDa",
        "gene_class": "membrane anchor (peroxin)",
        "pbd_subgroup": "AAA-ATPase recycling complex (PEX1·PEX6·PEX26)",
        "locus": "22q11.21", "omim_gene": 608666,
        "phenotype": "ZSD predominantly NALD/IRD (milder); ~6% of PBD-ZSD; 22q11.21 locus distinct from DiGeorge region",
        "disease": (
            "PEX26 biallelic loss → PBD-ZSD (OMIM #614862). PEX26 encodes a 305aa (34kDa) single-pass "
            "tail-anchored peroxisomal membrane protein (PMP). PEX26 is the membrane anchor for the "
            "AAA-ATPase recycling complex: its N-terminal cytoplasmic coiled-coil domain binds the "
            "C-terminus of PEX6, tethering the PEX1/PEX6 hexameric ring to the peroxisomal outer "
            "membrane surface. Without PEX26, the PEX1/PEX6 ring cannot dock at the peroxisomal "
            "membrane → PEX5 extraction fails → PTS1 import cycle stalls → identical downstream "
            "consequences to PEX1 or PEX6 mutations. Third most common PBD gene (~5-6% of ZSD). "
            "PEX26 mutations tend to produce milder phenotypes (NALD and IRD predominant) because: "
            "(1) smaller protein — some hypomorphic missense alleles retain partial coiled-coil/binding; "
            "(2) residual membrane-bound AAA activity possible. ZS is seen with true null PEX26 alleles. "
            "Note: PEX26 locus is 22q11.21, near the TBX1/DiGeorge region but DISTINCT — "
            "PEX26 mutations do NOT cause DiGeorge/22q11.2 deletion syndrome."
        ),
        "inheritance": "Autosomal recessive. PEX26 22q11.21. 3rd most common ZSD gene (~5-6%). Pan-ethnic; no major founder allele.",
        "hallmark": (
            "PEX26 HALLMARKS: "
            "(1) MILDER PHENOTYPE BIAS: NALD and IRD predominate (vs PEX1/PEX6 where ZS is prominent); "
            "PEX26 null alleles → ZS; hypomorphic alleles → NALD/IRD; "
            "(2) 22q11.21 LOCUS — NOT DiGeorge: PEX26 locus overlaps 22q11.2 region but is SEPARATE "
            "from the TBX1/CRKL 3Mb DiGeorge deletion; no cardiac or thymic features from PEX26 loss alone; "
            "(3) MEMBRANE-ANCHORED SCAFFOLD: PEX26 is the bridge between the peroxisome membrane and "
            "the soluble PEX1/PEX6 ring; loss → ring released into cytosol → cannot engage PEX5 on membrane; "
            "(4) ALL ZSD BIOMARKERS PRESENT (VLCFA, plasmalogens, catalase cytosolic): identical to PEX1/PEX6; "
            "(5) NO FOUNDER ALLELE: pan-ethnic, no population enrichment unlike PEX1 (G843D) or PEX6 (R860W); "
            "(6) SMALL GENE = FULL SEQUENCING FEASIBLE: 305aa — full sequencing in targeted panel; "
            "exon 3 frameshift most common severe allele"
        ),
        "key_ddx": (
            "PEX26 DDx: (1) PEX1/PEX6-ZSD — same biochemistry; gene panel distinguishes; "
            "(2) 22q11.2 DiGeorge deletion — cardiac + thymic + parathyroid features; "
            "normal VLCFA and plasmalogens; FISH/MLPA confirms; COMPLETELY DIFFERENT from PEX26 loss; "
            "(3) X-ALD — plasmalogens normal; X-linked; "
            "(4) Zellweger-like (PEX5, PEX3) — same spectrum; sequencing distinguishes"
        ),
        "diet_treatment": "Same as PEX1/PEX6-ZSD. Low phytanic acid; DHA (Level C); fat-soluble vitamins; Lorenzo Oil CONTRAINDICATED; fasting avoidance.",
        "gene_therapy_status": "No approved gene therapy. PEX26 smallest of the three AAA-axis peroxins (305aa, ~916bp coding) — AAV delivery most feasible in this group. No clinical trial as of 2025.",
        "critical_ci": (
            "CRITICAL CI: (1) Confusing 22q11.21 locus with DiGeorge syndrome — different gene, different disease; "
            "(2) Lorenzo Oil — contraindicated; "
            "(3) HSCT — not indicated; "
            "(4) High phytanic acid dietary exposure"
        ),
        "nbs_marker": "Not in standard NBS. VLCFA plasma + plasmalogens + gene panel (PEX26 sequencing) or WES.",
        "key_biomarker": "Plasma VLCFA elevated. Erythrocyte plasmalogens low. Fibroblast catalase cytosolic (diffuse). PEX26 protein absent/reduced on western blot. PEX26 sequencing.",
        "severity_spectrum": "Predominantly NALD/IRD (mild-moderate). Null/null → ZS (less common than in PEX1/PEX6). Adult survivors with IRD phenotype reported.",
        "founder_variant": "No major founder allele; pan-ethnic. p.Arg98Trp and exon 3 frameshifts most reported.",
        "key_variants": [
            "p.Arg98Trp — coiled-coil domain; NALD/IRD phenotype (partial binding retained)",
            "c.294+1G>A — splice donor exon 1; null; ZS",
            "p.Ala84Val — N-terminal helix; NALD",
            "p.Leu220Pro — helix-breaker; NALD-IRD",
            "c.793delA — frameshift exon 3; null; ZS when comp-het",
        ],
        "seed": SEED_BASE + 2,
    },
    # ── PEX10 — RING-finger E3 ubiquitin ligase (monoubiquitination of PEX5) ────
    {
        "gene": "PEX10", "alias": "PEX10 — RING E3 ligase, monoubiquitinates PEX5 for recycling",
        "aa": "326 aa", "kDa": "38.6 kDa",
        "gene_class": "RING-finger E3 ubiquitin ligase",
        "pbd_subgroup": "RING-finger E3 triad (PEX2·PEX10·PEX12)",
        "locus": "1p36.32", "omim_gene": 602859,
        "phenotype": "ZSD all tiers; RING E3 monoubiquitinates PEX5 for recycling; collaborates with PEX12 (polyubiquitination for ERAD)",
        "disease": (
            "PEX10 biallelic loss → PBD-ZSD (OMIM #614870). PEX10 encodes a 326aa (38.6kDa) integral "
            "peroxisomal membrane protein with a cytosol-facing RING (Really Interesting New Gene) "
            "zinc-finger domain. PEX10 is one of three PMP RING-finger E3 ubiquitin ligases (PEX2, PEX10, "
            "PEX12) that form the 'RING triad' on the peroxisomal membrane. PEX10 function: "
            "monoubiquitinates PEX5 on a conserved cysteine (Cys11) in the N-terminus — this "
            "monoubiquitination is the 'recycling tag' recognised by the PEX1/PEX6 AAA-ATPase complex "
            "for PEX5 extraction from the membrane back to the cytosol. Without PEX10, PEX5 is not "
            "monoubiquitinated → PEX1/PEX6 cannot extract PEX5 → PEX5 accumulates on membrane → "
            "secondarily polyubiquitinated by PEX12/PEX2 → targeted for ERAD → PTS1 import cycle depleted. "
            "Result: same as PEX1/PEX6/PEX26 loss — cytosolic catalase, VLCFA elevation, plasmalogen loss. "
            "PEX10 has an individual dashboard; this atlas includes it for completeness of the RING triad. "
            "Clinical: ZSD full spectrum. ~4-6% of ZSD cases."
        ),
        "inheritance": "Autosomal recessive. PEX10 1p36.32. RING-triad ZSD. Pan-ethnic.",
        "hallmark": (
            "PEX10 HALLMARKS: "
            "(1) RING TRIAD: PEX10 monoubiquitinates PEX5 (recycling); "
            "PEX12 polyubiquitinates (ERAD of trapped PEX5); "
            "PEX2 assists PEX12 polyubiquitination; triad works cooperatively — loss of any → PTS1 import fails; "
            "(2) PEX5-Cys11 IS THE MONOUBIQUITINATION SITE: mutations affecting PEX10 RING → "
            "PEX5-Cys11 unmodified → PEX1/PEX6 cannot grab → same import block; "
            "(3) RISP (Rieske FeS protein of CIII) is RELATIVELY PRESERVED — not relevant here but "
            "distinguishes from CIII deficiency; "
            "(4) CEREBELLAR ATROPHY: prominent in NALD/IRD survivors with PEX10; "
            "cerebellar neuronal DHA deficiency contributes; "
            "(5) PEX10 HAS INDIVIDUAL DASHBOARD: for per-patient depth; this atlas gives aggregate context; "
            "(6) VLCFA + PLASMALOGEN + CATALASE IF: universal ZSD markers"
        ),
        "key_ddx": (
            "PEX10 DDx: (1) PEX12-ZSD (individual dashboard): RING triad partner; "
            "polyubiquitinates PEX5 (vs PEX10 monoubiquitination); same ZSD phenotype; gene sequencing; "
            "(2) PEX2-ZSD: RING triad member; assists polyubiquitination; same phenotype; "
            "(3) PEX1/6/26-ZSD: AAA-axis; same downstream consequence; gene panel; "
            "(4) X-ALD: plasmalogens normal; X-linked"
        ),
        "diet_treatment": "Same as PEX1-ZSD: low phytanic acid; DHA (Level C); fat-soluble vitamins; Lorenzo Oil CONTRAINDICATED.",
        "gene_therapy_status": "No approved gene therapy. PEX10 coding ~978bp — amenable to AAV delivery in principle. No clinical trial.",
        "critical_ci": (
            "CRITICAL CI: (1) Lorenzo Oil — contraindicated; (2) HSCT — not indicated; "
            "(3) Confusing RING-triad loss with ABCD1/X-ALD — always check plasmalogens"
        ),
        "nbs_marker": "VLCFA plasma + plasmalogens + gene panel (PEX10 included in standard PBD panel) or WES.",
        "key_biomarker": "Plasma VLCFA elevated. Erythrocyte plasmalogens low. Fibroblast catalase cytosolic. PEX10 protein absent/reduced. RING domain sequencing.",
        "severity_spectrum": "ZS (null/null) → NALD → IRD. RING-domain missense alleles → partial function → milder spectrum.",
        "founder_variant": "No major founder allele. p.Gly266Asp (RING domain) and frameshift alleles most reported.",
        "key_variants": [
            "p.Gly266Asp — RING zinc-coordinating Gly; severe E3 loss; NALD/ZS",
            "p.Tyr262Stop — nonsense; null; ZS",
            "c.815+1G>A — splice donor; null; ZS",
            "p.Arg313Cys — C-terminal RING; partial; NALD-IRD",
            "p.Leu148Pro — TM2 helix-breaker; mislocalization; severe",
        ],
        "seed": SEED_BASE + 3,
    },
    # ── PEX2 — RING-finger E3 ligase (PEX5 polyubiquitination for ERAD) ──────────
    {
        "gene": "PEX2", "alias": "PEX2 — RING E3 ligase, polyubiquitinates PEX5 for ERAD",
        "aa": "305 aa", "kDa": "35.6 kDa",
        "gene_class": "RING-finger E3 ubiquitin ligase",
        "pbd_subgroup": "RING-finger E3 triad (PEX2·PEX10·PEX12)",
        "locus": "8q21.13", "omim_gene": 170993,
        "phenotype": "ZSD primarily ZS and NALD (severe end of spectrum); assists PEX12 in polyubiquitinating PEX5 for ERAD quality control",
        "disease": (
            "PEX2 biallelic loss → PBD-ZSD (OMIM #614859 shared locus with PEX6, distinct gene). "
            "PEX2 (305aa, 35.6kDa) is one of the three RING-triad peroxins and is the FIRST "
            "RING-finger PMP identified in mammals. PEX2 function: assists PEX12 in "
            "polyubiquitinating trapped PEX5 on the peroxisomal membrane — this polyubiquitin chain "
            "(Lys48-linked) targets PEX5 for proteasomal ERAD (retrotranslocation-associated degradation). "
            "The polyubiquitination pathway is the quality-control arm: if PEX5 cannot be recycled "
            "(e.g., PEX1/6/10 mutation), PEX12 + PEX2 polyubiquitinate it to prevent indefinite "
            "accumulation. PEX2 loss itself directly disrupts the ERAD arm → trapped PEX5 builds up "
            "chronically → eventually depletes the cytosolic PEX5 pool → PTS1 import fails. "
            "Clinical: ZSD — primarily the SEVERE end (ZS and NALD predominant); "
            "complete PEX2 loss → severe ZS. Represents ~2-3% of ZSD. "
            "PEX2 mutations also found in adult-onset cerebellar ataxia (very mild hypomorphic alleles) — "
            "an important mild end of the spectrum recently characterised."
        ),
        "inheritance": "Autosomal recessive. PEX2 8q21.13. Rare ZSD (2-3%). Pan-ethnic. Note: adult-onset cerebellar ataxia from hypomorphic PEX2 alleles reported.",
        "hallmark": (
            "PEX2 HALLMARKS: "
            "(1) ADULT-ONSET CEREBELLAR ATAXIA from hypomorphic alleles — important mild end; "
            "VLCFA mildly elevated; plasmalogens borderline low; catalase partially cytosolic; "
            "diagnose by fibroblast functional studies + PEX2 sequencing; "
            "(2) SEVERE END (ZS) from null alleles — same neonatal ZS as PEX1 null; "
            "(3) PEX2 ASSISTS PEX12 IN POLYUBIQUITINATION: PEX12 is primary E3 for Lys48-Ub chain; "
            "PEX2 works cooperatively (PEX2's RING domain has weaker intrinsic activity; "
            "cooperation with PEX12 amplifies polyubiquitination signal); "
            "(4) FIRST IDENTIFIED MAMMALIAN RING-TRIAD PEROXIN: PEX2 = PXMP3 historically; "
            "(5) ALL ZSD BIOMARKERS: VLCFA elevated, plasmalogens low, catalase cytosolic; "
            "borderline in very mild alleles — may require peroxisome abundance quantification"
        ),
        "key_ddx": (
            "PEX2 DDx: (1) PEX12-ZSD — polyubiquitination partner; same clinical spectrum; "
            "PEX12 has individual dashboard; gene sequencing; "
            "(2) PEX10-ZSD — monoubiquitination arm; same endpoint; "
            "(3) Adult cerebellar ataxia (non-PBD) — FRDA, SCA, spastic ataxia — "
            "check VLCFA + plasmalogens + PEX2 sequencing in unexplained adult ataxia; "
            "(4) X-ALD adrenomyeloneuropathy (AMN) — adult presentation; plasmalogens NORMAL; X-linked"
        ),
        "diet_treatment": "Same ZSD protocol: low phytanic acid; DHA (Level C); fat-soluble vitamins; Lorenzo Oil CONTRAINDICATED. For mild adult-onset ataxia: DHA trial reasonable (Level C); phytanic acid restriction.",
        "gene_therapy_status": "No approved gene therapy. PEX2 small gene (305aa, ~915bp) — good AAV delivery candidate. No clinical trial.",
        "critical_ci": (
            "CRITICAL CI: (1) Lorenzo Oil — contraindicated in all PBD including PEX2; "
            "(2) Misdiagnosis of adult PEX2 ataxia as Friedreich ataxia or SCA — "
            "always check VLCFA + plasmalogens in adult unexplained cerebellar ataxia; "
            "(3) HSCT — not indicated"
        ),
        "nbs_marker": "VLCFA plasma + plasmalogens (may be borderline in mild alleles). PEX2 sequencing. Fibroblast catalase IF for borderline biochemistry.",
        "key_biomarker": "Plasma VLCFA (elevated; borderline in hypomorphic alleles). Erythrocyte plasmalogens (low; borderline in mild alleles). Fibroblast catalase cytosolic (partial in mild). PEX2 sequencing + protein western.",
        "severity_spectrum": "Severe ZS (null/null; die <12 months) → NALD → IRD → Adult-onset cerebellar ataxia (hypomorphic missense — important mild end). Broad spectrum.",
        "founder_variant": "No major founder allele. p.Cys159Tyr (RING Zn-coordinating cysteine; severe) most reported severe allele.",
        "key_variants": [
            "p.Cys159Tyr — RING zinc-coordinating; abolishes E3 activity; severe ZS/NALD",
            "p.Arg119Gly — RING core; severe",
            "p.Thr300Met — C-terminal; mild; adult ataxia reported",
            "c.603_604del — frameshift; null; ZS",
            "p.Gly89Arg — N-terminal region; NALD",
        ],
        "seed": SEED_BASE + 4,
    },
    # ── PEX3 — De novo peroxisomal membrane assembly ────────────────────────────
    {
        "gene": "PEX3", "alias": "PEX3 — de novo PMP targeting; complete loss → NO peroxisomes",
        "aa": "373 aa", "kDa": "42.7 kDa",
        "gene_class": "peroxisomal membrane protein (PMP) biogenesis",
        "pbd_subgroup": "Membrane biogenesis + PTS receptor axis (PEX3·PEX5·PEX16)",
        "locus": "6q24.2", "omim_gene": 603164,
        "phenotype": "ZS predominantly; de novo peroxisome membrane formation — loss → no peroxisomal ghost membranes, most severe biogenesis defect",
        "disease": (
            "PEX3 biallelic loss → PBD-ZSD (OMIM #614859). PEX3 (373aa, 42.7kDa) is the founding "
            "peroxin for peroxisomal membrane biogenesis. PEX3 is synthesised on cytoplasmic ribosomes "
            "and inserted into the ER membrane (or pre-existing peroxisomal membrane) via PEX19 "
            "(the cytosolic PMP chaperone/receptor). Once in the peroxisomal membrane, PEX3 acts as "
            "the 'dock' for PEX19 to deliver other PMPs — PEX3 is the receptor on the membrane side "
            "that completes PMP insertion. PEX3 is absolutely required for de novo peroxisome "
            "biogenesis from the ER. Without PEX3: PMPs cannot insert into any membrane → "
            "peroxisomal ghost membranes (empty vesicles lacking matrix proteins) are absent — "
            "uniquely, PEX3 loss results in COMPLETE ABSENCE of peroxisomal membranes/ghosts, "
            "distinguishing it from other PBD genes where peroxisomal ghost membranes persist. "
            "Extremely rare. Clinical: uniformly severe ZS (neonatal). "
            "Virtually all reported patients have ZS phenotype with no functional peroxisomes "
            "and no peroxisomal ghost membranes by EM. Severe brain malformations (pachygyria, "
            "polymicrogyria), profound hypotonia, hepatopathy, early death (<12 months)."
        ),
        "inheritance": "Autosomal recessive. PEX3 6q24.2. Very rare ZSD. Predominantly ZS. Pan-ethnic.",
        "hallmark": (
            "PEX3 HALLMARKS: "
            "(1) NO GHOST MEMBRANES ON ELECTRON MICROSCOPY — most severe biogenesis defect; "
            "other PBD genes preserve peroxisomal ghost membranes (empty vesicles); "
            "PEX3 loss → complete absence of peroxisomal membrane structures; "
            "PEX16 loss is similar; "
            "(2) PEX3 IS THE MEMBRANE RECEPTOR FOR PEX19-CARGO DELIVERY: "
            "PEX19 (cytosolic PMP chaperone) captures PMPs → docks PEX19-PMP complex onto PEX3 → "
            "PMP inserted into membrane; no PEX3 → no PMP insertion → no peroxisome; "
            "(3) PEX3 + PEX16 + PEX19 FORM THE 'MEMBRANE BIOGENESIS TRIAD': "
            "the three genes required specifically for peroxisomal membrane formation; "
            "matrix import genes (PEX5, PEX1/6/26, RING triad) are secondarily absent too; "
            "(4) UNIFORMLY SEVERE ZS: no partial-function phenotype reported (extremely limited cohort); "
            "(5) ER-DERIVED DE NOVO FORMATION: in cells without any peroxisomes, "
            "PEX3 + PEX16 re-establish peroxisomes from ER buds — therapeutic complementation proof-of-concept"
        ),
        "key_ddx": (
            "PEX3 DDx: (1) PEX16-ZSD: also lacks ghost membranes; clinically identical severe ZS; "
            "PEX16 is slightly larger (336aa); gene sequencing distinguishes; "
            "(2) PEX19-ZSD (very rare): PEX3/PEX16 receptor-chaperone partner; same ghost-membrane-absent phenotype; "
            "(3) Other ZSD genes (PEX1/6/26/2/10/5): peroxisomal ghost membranes PRESERVED; "
            "EM ghost membranes presence/absence is key differentiator; "
            "(4) X-ALD neonatal: rare severe neonatal X-ALD; plasmalogens NORMAL; X-linked"
        ),
        "diet_treatment": "Supportive ZS care: seizure management (LEV, phenobarbital), fat-soluble vitamins, gastrostomy feeds. DHA has no meaningful benefit in ZS (no peroxisomes to utilise). Lorenzo Oil CONTRAINDICATED. Palliative approach for most ZS.",
        "gene_therapy_status": "No approved gene therapy. PEX3 complementation restores peroxisome formation in patient cells (proof-of-concept de novo peroxisome genesis). No clinical trial; AAV for neonatal delivery is theoretical.",
        "critical_ci": (
            "CRITICAL CI: (1) Lorenzo Oil — contraindicated; (2) HSCT — not indicated; "
            "(3) Aggressive intervention in ZS without goals-of-care discussion — "
            "ZS patients die <12 months; palliative and family support should be initiated early; "
            "(4) Missing ghost-membrane absence on EM — consider PEX3/PEX16 if no ghost membranes"
        ),
        "nbs_marker": "VLCFA plasma + plasmalogens. Gene panel (PEX3 included). Very rare — often diagnosed at autopsy in ZS era pre-genetics.",
        "key_biomarker": "Plasma VLCFA markedly elevated. Plasmalogens absent. Fibroblast catalase cytosolic + ABSENT peroxisomal structures on immunofluorescence (PMP70, PEX14 staining diffuse/absent, not punctate). EM: NO ghost membranes.",
        "severity_spectrum": "Uniformly severe ZS (neonatal death <12 months in all reported cases). No mild phenotype described (too few patients globally for IRD/NALD to be confirmed).",
        "founder_variant": "No founder allele — extremely rare globally (<20 cases reported). All null/severe alleles in literature.",
        "key_variants": [
            "p.Trp154Stop — nonsense; null; ZS",
            "c.419+2T>C — splice site; null; ZS",
            "p.Arg55Stop — early truncation; null; ZS",
            "p.Leu348Pro — C-terminal PEX19-binding domain; severe",
            "p.Glu218Lys — PEX19 docking surface; severe",
        ],
        "seed": SEED_BASE + 5,
    },
    # ── PEX5 — PTS1 cytosolic receptor ───────────────────────────────────────────
    {
        "gene": "PEX5", "alias": "PEX5 — PTS1 receptor; cargo delivery cycle; ZSD all tiers",
        "aa": "639 aa", "kDa": "69.6 kDa",
        "gene_class": "PTS1 import receptor (cytosolic)",
        "pbd_subgroup": "Membrane biogenesis + PTS receptor axis (PEX3·PEX5·PEX16)",
        "locus": "12p13.31", "omim_gene": 600414,
        "phenotype": "ZSD all tiers; PTS1 cytosolic receptor that delivers matrix proteins to peroxisomes; null → severe ZS; hypomorphic → NALD/IRD",
        "disease": (
            "PEX5 biallelic loss → PBD-ZSD (OMIM #214100). PEX5 (639aa, 69.6kDa) is the cytosolic "
            "receptor for peroxisomal targeting signal 1 (PTS1). PTS1 is the C-terminal tripeptide "
            "(-SKL and variants: -SRL, -AKL, -ANL etc.) found on ~90% of peroxisomal matrix proteins "
            "(catalase, thiolase, acyl-CoA oxidase, most beta-oxidation enzymes). The PEX5 cycle: "
            "(1) cytosolic PEX5 recognises PTS1 via its 7 TPR (tetratricopeptide repeat) domains; "
            "(2) PEX5-cargo complex docks at peroxisomal membrane via PEX14 (docking site); "
            "(3) PEX5 inserts into the PEX14/PEX17 translocon pore (PEX5 acts as part of the pore); "
            "(4) cargo released into peroxisomal matrix (mechanism: thermal/pore-assisted translocation); "
            "(5) PEX5 monoubiquitinated by PEX10 → extracted by PEX1/PEX6 → recycled to cytosol; "
            "(6) if stuck: PEX12/PEX2 polyubiquitinate PEX5 → ERAD. "
            "PEX5 loss → all PTS1 cargo remains cytosolic → VLCFA accumulate (ACOX1, MFP2, thiolase all fail), "
            "plasmalogens absent (GNPAT, AGPS are PTS1 proteins), catalase cytosolic. "
            "PEX5 also has a longer isoform (PEX5L) that mediates PTS2 import (via PEX7). "
            "Complete PEX5 loss → both PTS1 AND PTS2 import fail. "
            "Clinical: ZSD full spectrum. ~5% of ZSD. "
        ),
        "inheritance": "Autosomal recessive. PEX5 12p13.31. ~5% of ZSD. Pan-ethnic. PEX5 null → ZS (both PTS1 + PTS2 import fail).",
        "hallmark": (
            "PEX5 HALLMARKS: "
            "(1) PTS1 RECEPTOR — THE CARGO DELIVERY LYNCHPIN: "
            "all PTS1 matrix proteins (catalase, thiolase, ACOX1, MFP2, GNPAT, AGPS, SDHase etc.) "
            "depend on PEX5 for import; loss → global matrix protein import failure; "
            "(2) PEX5 MEDIATES BOTH PTS1 AND PTS2 via PEX5L ISOFORM: "
            "PEX5L (long isoform, contains 37aa insert) binds PEX7 (PTS2 receptor); "
            "PEX5L-PEX7-PTS2-cargo complex → PEX5L docks at PEX14 → PTS2 cargo released; "
            "PEX5 null → both PTS1 and PTS2 import fail (ZS); "
            "PEX7 null → PTS2 import only fails → RCDP (different disease: Rhizomelic Chondrodysplasia Punctata); "
            "(3) PEX5-PEX14 DOCKING — CRITICAL INTERFACE: "
            "PEX14 is the initial docking receptor; anti-PEX14 antibodies are used "
            "clinically to measure peroxisomal ghost abundance; "
            "(4) 7 TPR REPEATS: each PTS1 variant (-SKL/-AKL/-ANL etc.) is recognised by "
            "different TPR combinations; TPR domain mutations may show selective PTS1 cargo loss; "
            "(5) PEX5 PROTEIN ASSAY IN PATIENT FIBROBLASTS: "
            "can check PEX5 abundance + catalase import directly; "
            "(6) VLCFA + PLASMALOGENS + CATALASE IF: universal ZSD markers, all abnormal"
        ),
        "key_ddx": (
            "PEX5 DDx: (1) PEX7-RCDP (Rhizomelic Chondrodysplasia Punctata type 1): "
            "PTS2 import fails but PTS1 import NORMAL (PEX5S isoform intact); "
            "rhizomelia (proximal limb shortening), stippled epiphyses, plasmalogen deficiency; "
            "VLCFA NORMAL in RCDP type 1 (ACOX1/thiolase are PTS1 = import intact); "
            "KEY SEPARATOR: rhizomelia + chondrodysplasia punctata = PEX7 (RCDP1), NOT PEX5 ZSD; "
            "(2) PEX1/6/26-ZSD: same downstream import block; gene panel distinguishes; "
            "(3) X-ALD: plasmalogens NORMAL; X-linked"
        ),
        "diet_treatment": "Same ZSD protocol: low phytanic acid; DHA supplementation (Level C); fat-soluble vitamins; Lorenzo Oil CONTRAINDICATED; fasting avoidance.",
        "gene_therapy_status": "No approved gene therapy. PEX5 coding ~1917bp (PEX5S) — AAV delivery feasible. PEX5L (PEX5 long with 111bp insert) also needed for full PTS2 import rescue. Dual-vector or splice-switching approaches considered. No clinical trial.",
        "critical_ci": (
            "CRITICAL CI: (1) Lorenzo Oil — contraindicated; (2) HSCT — not indicated; "
            "(3) Confusing PEX5 ZSD with PEX7 RCDP — different clinical and biochemical profiles; "
            "rhizomelia + stippled epiphyses = RCDP, not ZSD; VLCFA NORMAL in RCDP type 1; "
            "(4) High phytanic acid foods"
        ),
        "nbs_marker": "VLCFA + plasmalogens. PEX5 sequencing (gene panel or WES). Fibroblast catalase IF confirms.",
        "key_biomarker": "Plasma VLCFA elevated. Erythrocyte plasmalogens low (both ethanolamine and choline). Phytanic + pipecolic elevated. Fibroblast catalase cytosolic. PEX5 protein absent/reduced.",
        "severity_spectrum": "Severe ZS (null/null — both PTS1 and PTS2 fail) → NALD → IRD (partial-function alleles). Adult IRD survivors reported with hypomorphic PEX5 alleles.",
        "founder_variant": "No major founder allele. p.Gly328Glu (TPR domain; moderate) and frameshift nulls most reported.",
        "key_variants": [
            "p.Gly328Glu — TPR2 domain; PEX5-PTS1 binding reduced; moderate NALD/IRD",
            "c.1072_1073insA — frameshift; null; severe ZS",
            "p.Trp503Stop — nonsense; null; ZS",
            "p.Leu293Pro — TPR1 helix-breaker; severe NALD",
            "p.Arg390Gln — PEX14 docking interface; intermediate",
        ],
        "seed": SEED_BASE + 6,
    },
    # ── PEX16 — Peroxisomal membrane insertion and growth ────────────────────────
    {
        "gene": "PEX16", "alias": "PEX16 — PMP membrane insertion; growth/division; ZS/NALD",
        "aa": "336 aa", "kDa": "38.7 kDa",
        "gene_class": "peroxisomal membrane protein (PMP) biogenesis",
        "pbd_subgroup": "Membrane biogenesis + PTS receptor axis (PEX3·PEX5·PEX16)",
        "locus": "11p11.2", "omim_gene": 603360,
        "phenotype": "ZS and NALD predominantly; PMP membrane insertion; complete loss → no ghost membranes (like PEX3); note: 11p11.2 is near PEX26 locus region — distinct chromosomes",
        "disease": (
            "PEX16 biallelic loss → PBD-ZSD (OMIM #614863). PEX16 (336aa, 38.7kDa) is an integral "
            "peroxisomal membrane protein (PMP) with two transmembrane helices. PEX16 works with PEX3 "
            "and PEX19 as part of the 'membrane biogenesis triad.' PEX16 inserts into ER-derived "
            "pre-peroxisomal vesicles first (via the ER-GET pathway), then recruits PEX3 to the "
            "vesicle surface, which in turn allows PEX19-cargo (other PMPs) to dock and insert — "
            "establishing the complete peroxisomal membrane. Without PEX16: PEX3 cannot be efficiently "
            "delivered to ER-derived vesicles → peroxisomal membrane cannot form → no ghost membranes "
            "(similar to PEX3 loss). PEX16 also participates in peroxisomal membrane growth "
            "(enlargement before division) independent of matrix import. "
            "Loss → no functional peroxisomal membranes → secondary loss of all matrix import pathways "
            "(PTS1, PTS2) → same VLCFA/plasmalogen/phytanic acid profile as all ZSD. "
            "Clinical: ZS (null alleles; neonatal; no ghost membranes) and NALD (partial-function). "
            "IRD reported rarely. ~2-3% of ZSD. "
            "The 11p11.2 locus is distinct from PEX26 at 22q11.21; both are pericentromeric "
            "but on different chromosomes."
        ),
        "inheritance": "Autosomal recessive. PEX16 11p11.2. Rare (~2-3% of ZSD). ZS/NALD spectrum. Pan-ethnic.",
        "hallmark": (
            "PEX16 HALLMARKS: "
            "(1) NO GHOST MEMBRANES ON EM (similar to PEX3): most severe membrane biogenesis defect; "
            "PEX16 and PEX3 loss both result in absent peroxisomal membranes — distinguishes from "
            "matrix import ZSD genes (PEX1/2/5/6/10/26) where ghost membranes are preserved; "
            "(2) ER-ROUTE FIRST: PEX16 is inserted at the ER via the GET/TRC40 pathway before "
            "being trafficked to pre-peroxisomal vesicles — unique ER-first insertion mechanism; "
            "(3) PEX16 RECRUITS PEX3 TO ER-DERIVED VESICLES: if PEX16 absent, PEX3 stays on ER "
            "but cannot transfer to peroxisomal precursor → PEX19-cargo delivery fails; "
            "(4) PEROXISOMAL MEMBRANE GROWTH FUNCTION: independent of matrix import; "
            "contributes to peroxisome enlargement before fission; "
            "(5) ALL ZSD BIOMARKERS PRESENT: VLCFA markedly elevated, plasmalogens absent, catalase cytosolic; "
            "(6) 11p11.2 LOCUS: includes STS (steroid sulphatase) and PTPN11 regions nearby — "
            "PEX16 mutations are NOT associated with LEOPARD/NS phenotype (PTPN11); "
            "copy-number variants at 11p11 can co-delete adjacent genes"
        ),
        "key_ddx": (
            "PEX16 DDx: (1) PEX3-ZSD: most similar (no ghost membranes); gene sequencing; "
            "(2) PEX19-ZSD (very rare): third member of membrane biogenesis triad; same absent-ghost phenotype; "
            "(3) Other ZSD (PEX1/6/26/2/5/10): ghost membranes present; matrix import blocked; "
            "EM ghost-membrane presence/absence is the key first separator; "
            "(4) X-ALD severe neonatal: extremely rare; plasmalogens NORMAL; X-linked"
        ),
        "diet_treatment": "Supportive ZS care: seizure management; fat-soluble vitamins; DHA (minimal benefit in ZS; may help NALD). Lorenzo Oil CONTRAINDICATED. Palliative approach for severe ZS.",
        "gene_therapy_status": "No approved gene therapy. PEX16 coding ~1008bp — AAV delivery feasible. PEX16 complementation restores peroxisome formation from ER in patient cells in vitro. No clinical trial.",
        "critical_ci": (
            "CRITICAL CI: (1) Lorenzo Oil — contraindicated; (2) HSCT — not indicated; "
            "(3) Missing ghost-membrane absence — suggests PEX3 or PEX16; pursue EM in typical ZS; "
            "(4) Aggressive intervention without palliative planning for ZS"
        ),
        "nbs_marker": "VLCFA plasma + plasmalogens. Gene panel (PEX16 included in standard PBD panel). EM of fibroblasts for ghost-membrane absence if ZS but panel inconclusive.",
        "key_biomarker": "Plasma VLCFA markedly elevated. Plasmalogens absent. Fibroblast catalase cytosolic. EM: NO peroxisomal ghost membranes. PEX16 absent on western. PEX3 also reduced secondarily.",
        "severity_spectrum": "Severe ZS (null/null; die <12 months; no ghost membranes) → NALD (partial function alleles). IRD extremely rare with hypomorphic alleles.",
        "founder_variant": "No major founder allele. Frameshift nulls and splice-site alleles most reported in ZS. p.Arg176Gln (partial function) NALD reported.",
        "key_variants": [
            "p.Arg176Gln — partial function; NALD phenotype",
            "c.485_486del — frameshift; null; ZS",
            "p.Gln179Stop — nonsense; null; ZS",
            "c.340+1G>A — splice donor; null; ZS",
            "p.Leu218Pro — TM2 helix-breaker; mislocalization; ZS",
        ],
        "seed": SEED_BASE + 7,
    },
]


# ── Patient cohort generation ─────────────────────────────────────────────────────
def _make_patients(gene_dict):
    """Generate 40 synthetic patient records for a given PBD gene."""
    rng = random.Random(gene_dict["seed"])
    gene = gene_dict["gene"]

    # Phenotypic class probabilities per gene
    PHENO_PROBS = {
        "PEX1":  [0.30, 0.38, 0.32],  # ZS/NALD/IRD — G843D enriches NALD+IRD
        "PEX6":  [0.32, 0.40, 0.28],  # ZS/NALD/IRD — A778T enriches NALD
        "PEX26": [0.25, 0.42, 0.33],  # milder bias
        "PEX10": [0.38, 0.40, 0.22],  # moderate severity
        "PEX2":  [0.50, 0.35, 0.15],  # severe bias (ZS dominant) + adult ataxia in mild
        "PEX3":  [0.85, 0.12, 0.03],  # almost all ZS
        "PEX5":  [0.40, 0.38, 0.22],  # full spectrum
        "PEX16": [0.55, 0.38, 0.07],  # ZS/NALD dominant
    }
    probs = PHENO_PROBS.get(gene, [0.40, 0.38, 0.22])
    classes = ["ZS", "NALD", "IRD"]

    patients = []
    for i in range(40):
        r = rng.random()
        if r < probs[0]:
            pheno = "ZS"
        elif r < probs[0] + probs[1]:
            pheno = "NALD"
        else:
            pheno = "IRD"

        # Severity-dependent clinical features
        is_zs   = (pheno == "ZS")
        is_nald = (pheno == "NALD")
        is_ird  = (pheno == "IRD")

        # Age at diagnosis (years)
        if is_zs:
            age_dx = round(rng.uniform(0.0, 0.3), 2)   # neonatal (<4 months)
        elif is_nald:
            age_dx = round(rng.uniform(0.1, 2.0), 1)   # infancy to ~2y
        else:
            age_dx = round(rng.uniform(0.5, 8.0), 1)   # childhood

        # VLCFA elevation (virtually universal in all PBD)
        vlcfa_elev = rng.random() < 0.97

        # Plasmalogen low (universal in all PBD)
        plasmalogen_low = rng.random() < 0.96

        # Retinal dystrophy (cone-rod; ERG flat)
        if is_zs:
            retinal = rng.random() < 0.55      # some ZS die before retina assessed
        elif is_nald:
            retinal = rng.random() < 0.90
        else:
            retinal = rng.random() < 0.95      # almost universal in IRD

        # SNHL (sensorineural hearing loss)
        if is_zs:
            snhl = rng.random() < 0.40
        elif is_nald:
            snhl = rng.random() < 0.78
        else:
            snhl = rng.random() < 0.85

        # Liver disease (hepatomegaly/transaminitis)
        liver = rng.random() < (0.88 if is_zs else 0.62 if is_nald else 0.35)

        # Seizures
        seizures = rng.random() < (0.92 if is_zs else 0.70 if is_nald else 0.45)

        # DHA therapy (supplemented)
        dha_tx = not is_zs and rng.random() < 0.60   # ZS die before therapy meaningful

        # Deceased
        if is_zs:
            deceased = rng.random() < 0.88    # most ZS die <12 months
        elif is_nald:
            deceased = rng.random() < 0.30    # some NALD die in childhood
        else:
            deceased = rng.random() < 0.05    # IRD survive decades

        # Neurological involvement (universal in ZSD)
        neuro = True  # all PBD-ZSD have neurological features

        # Brain MRI abnormal
        if is_zs:
            brain_mri_abnl = rng.random() < 0.98   # pachygyria/cortical dysplasia
        elif is_nald:
            brain_mri_abnl = rng.random() < 0.90   # leukodystrophy/atrophy
        else:
            brain_mri_abnl = rng.random() < 0.65   # cerebellar atrophy

        patients.append({
            "patient_id": f"{gene}-{i+1:03d}",
            "gene": gene,
            "phenotypic_class": pheno,
            "age_dx_y": age_dx,
            "vlcfa_elevated": vlcfa_elev,
            "plasmalogen_low": plasmalogen_low,
            "has_retinal_dystrophy": retinal,
            "has_snhl": snhl,
            "has_liver_disease": liver,
            "has_seizures": seizures,
            "dha_therapy": dha_tx,
            "brain_mri_abnormal": brain_mri_abnl,
            "neurological": neuro,
            "deceased": deceased,
        })
    return patients


# ── Populate patient cohorts ──────────────────────────────────────────────────────
for _g in PBD_GENES:
    _g["patients"] = _make_patients(_g)
    _g["n_patients"] = len(_g["patients"])

ALL_PATIENTS = [p for g in PBD_GENES for p in g["patients"]]


# ─── API: get_overview ───────────────────────────────────────────────────────────
def get_overview():
    total = len(ALL_PATIENTS)
    n_retinal  = sum(1 for p in ALL_PATIENTS if p["has_retinal_dystrophy"])
    n_snhl     = sum(1 for p in ALL_PATIENTS if p["has_snhl"])
    n_liver    = sum(1 for p in ALL_PATIENTS if p["has_liver_disease"])
    n_seizures = sum(1 for p in ALL_PATIENTS if p["has_seizures"])
    n_deceased = sum(1 for p in ALL_PATIENTS if p["deceased"])
    n_dha      = sum(1 for p in ALL_PATIENTS if p["dha_therapy"])
    n_vlcfa    = sum(1 for p in ALL_PATIENTS if p["vlcfa_elevated"])
    n_plasma   = sum(1 for p in ALL_PATIENTS if p["plasmalogen_low"])

    # Phenotypic class counts
    pheno_counts = {"ZS": 0, "NALD": 0, "IRD": 0}
    for p in ALL_PATIENTS:
        pheno_counts[p["phenotypic_class"]] += 1

    gene_summary = []
    for g in PBD_GENES:
        pts = g["patients"]
        gene_summary.append({
            "gene": g["gene"],
            "alias": g["alias"],
            "locus": g["locus"],
            "gene_class": g["gene_class"],
            "pbd_subgroup": g["pbd_subgroup"],
            "n_patients": g["n_patients"],
            "diet_treatment": g["diet_treatment"],
            "nbs_marker": g["nbs_marker"],
            "key_biomarker": g["key_biomarker"],
            "severity_spectrum": g["severity_spectrum"],
            "founder_variant": g["founder_variant"],
            "pct_retinal":  round(100 * sum(1 for p in pts if p["has_retinal_dystrophy"]) / len(pts), 1),
            "pct_snhl":     round(100 * sum(1 for p in pts if p["has_snhl"]) / len(pts), 1),
            "pct_liver":    round(100 * sum(1 for p in pts if p["has_liver_disease"]) / len(pts), 1),
            "pct_seizures": round(100 * sum(1 for p in pts if p["has_seizures"]) / len(pts), 1),
            "pct_deceased": round(100 * sum(1 for p in pts if p["deceased"]) / len(pts), 1),
            "pct_dha":      round(100 * sum(1 for p in pts if p["dha_therapy"]) / len(pts), 1),
            "pct_zs":       round(100 * sum(1 for p in pts if p["phenotypic_class"] == "ZS") / len(pts), 1),
            "pct_nald":     round(100 * sum(1 for p in pts if p["phenotypic_class"] == "NALD") / len(pts), 1),
            "pct_ird":      round(100 * sum(1 for p in pts if p["phenotypic_class"] == "IRD") / len(pts), 1),
            "mean_age_dx_y": round(sum(p["age_dx_y"] for p in pts) / len(pts), 1),
        })

    return {
        "atlas": "PBD-Atlas — Complete 8-Gene Peroxisomal Biogenesis Disorders Atlas",
        "n_genes": len(PBD_GENES),
        "n_patients": total,
        "seeds": [g["seed"] for g in PBD_GENES],
        "genes_covered": [g["gene"] for g in PBD_GENES],
        "gene_subgroups": {
            "AAA-ATPase recycling complex (PEX1·PEX6·PEX26)": ["PEX1", "PEX6", "PEX26"],
            "RING-finger E3 ubiquitin-ligase triad (PEX2·PEX10·PEX12*)": ["PEX10", "PEX2"],
            "Membrane biogenesis + PTS receptor axis (PEX3·PEX5·PEX16)": ["PEX3", "PEX5", "PEX16"],
        },
        "phenotypic_distribution": {
            "ZS":   {"n": pheno_counts["ZS"],   "pct": round(100 * pheno_counts["ZS"]   / total, 1)},
            "NALD": {"n": pheno_counts["NALD"], "pct": round(100 * pheno_counts["NALD"] / total, 1)},
            "IRD":  {"n": pheno_counts["IRD"],  "pct": round(100 * pheno_counts["IRD"]  / total, 1)},
        },
        "aggregate_clinical": {
            "pct_vlcfa_elevated":    round(100 * n_vlcfa    / total, 1),
            "pct_plasmalogen_low":   round(100 * n_plasma   / total, 1),
            "pct_retinal_dystrophy": round(100 * n_retinal  / total, 1),
            "pct_snhl":              round(100 * n_snhl     / total, 1),
            "pct_liver_disease":     round(100 * n_liver    / total, 1),
            "pct_seizures":          round(100 * n_seizures / total, 1),
            "pct_deceased":          round(100 * n_deceased / total, 1),
            "pct_dha_therapy":       round(100 * n_dha      / total, 1),
        },
        "gene_summary": gene_summary,
        "critical_clinical_rules": [
            "PEX1 G843D GENOTYPE-PHENOTYPE: G843D homozygous → NALD or IRD (NEVER ZS — partial AAA function retained); G843D/null → ZS or NALD; most common ZSD variant (~40% of all ZSD alleles); Northern-European enriched; genotyping G843D status is mandatory for prognosis counselling",
            "LORENZO OIL ABSOLUTE CONTRAINDICATION: Lorenzo Oil (erucic + oleic acid) reduces VLCFA only in X-ALD (ABCD1 transporter); has NO benefit in PBD-ZSD; potential harm; critical prescribing error — always confirm ABCD1 vs PEX gene before prescribing Lorenzo Oil",
            "PLASMALOGENS SEPARATE ZSD FROM X-ALD: VLCFA elevated in BOTH ZSD and X-ALD (ABCD1); erythrocyte plasmalogens LOW in all ZSD (biogenesis defect) but NORMAL in X-ALD (transporter defect only); this test definitively distinguishes the two — do it first in any VLCFA-elevated male",
            "HSCT NOT INDICATED in PBD-ZSD: unlike X-ALD cerebral form where HSCT halts progression if done early; PBD-ZSD is a cell-autonomous multi-organ defect; no clinical benefit from HSCT demonstrated; refer to X-ALD expert for any HSCT consideration",
            "CATALASE IMMUNOFLUORESCENCE — GOLD STANDARD FUNCTIONAL TEST: in ZSD fibroblasts, catalase is diffusely cytosolic (NOT punctate peroxisomal); in PEX11B (fission defect), catalase IS peroxisomal (import intact); this single test separates biogenesis defects from fission defects before sequencing",
            "GHOST MEMBRANES ABSENT IN PEX3/PEX16: all other ZSD genes (PEX1/2/5/6/10/26) leave peroxisomal ghost membranes (empty vesicles) visible by EM or PMP70/PEX14 immunostaining; PEX3 and PEX16 loss → NO ghost membranes → most severe membrane defect; EM distinguishes",
            "DHA SUPPLEMENTATION — PARTIAL RETINAL BENEFIT ONLY (Level C): DHA (docosahexaenoic acid) 20-100mg/kg/day raises plasma DHA and may slow retinal degeneration; does NOT arrest neurological progression; NOT disease-modifying; use in all NALD/IRD survivors but counsel expectations honestly",
            "PEX5 LOSS = PTS1 + PTS2 IMPORT BOTH FAIL: PEX5L isoform carries PEX7 (PTS2 receptor) into the membrane; PEX5 null → both PTS1 and PTS2 cargo fail; contrast PEX7 null (RCDP type 1) where only PTS2 fails → rhizomelia + normal VLCFA — completely different disease",
            "PEX2 ADULT-ONSET CEREBELLAR ATAXIA: hypomorphic PEX2 alleles cause adult-onset progressive cerebellar ataxia with mildly elevated VLCFA and borderline plasmalogens — must check VLCFA + plasmalogens + PEX2 sequencing in unexplained adult cerebellar ataxia; treatable with DHA + phytanic acid restriction",
            "PHYTANIC ACID DIETARY RESTRICTION — MANDATORY in ZSD: phytanic acid (ruminant animal fat, dairy fat, fish oil) requires peroxisomal alpha-oxidation (PHYH enzyme, peroxisomal); ZSD → phytanic accumulates → toxic to myelin; AVOID dairy fat, ruminant fat, cod liver oil; plant oils and marine DHA (from algae-derived supplementation without phytanic contamination) preferred",
        ],
    }


# ─── API: get_breakdown ──────────────────────────────────────────────────────────
def get_breakdown():
    gene_rows = []
    for g in PBD_GENES:
        pts = g["patients"]
        gene_rows.append({
            "gene": g["gene"],
            "alias": g["alias"],
            "aa": g["aa"],
            "kDa": g["kDa"],
            "locus": g["locus"],
            "omim_gene": g["omim_gene"],
            "gene_class": g["gene_class"],
            "pbd_subgroup": g["pbd_subgroup"],
            "n_patients": g["n_patients"],
            "seed": g["seed"],
            "phenotype": g["phenotype"],
            "inheritance": g["inheritance"],
            "hallmark": g["hallmark"],
            "key_ddx": g["key_ddx"],
            "diet_treatment": g["diet_treatment"],
            "gene_therapy_status": g["gene_therapy_status"],
            "critical_ci": g["critical_ci"],
            "nbs_marker": g["nbs_marker"],
            "key_biomarker": g["key_biomarker"],
            "severity_spectrum": g["severity_spectrum"],
            "founder_variant": g["founder_variant"],
            "key_variants": g["key_variants"],
            "pct_retinal":   round(100 * sum(1 for p in pts if p["has_retinal_dystrophy"]) / len(pts), 1),
            "pct_snhl":      round(100 * sum(1 for p in pts if p["has_snhl"]) / len(pts), 1),
            "pct_liver":     round(100 * sum(1 for p in pts if p["has_liver_disease"]) / len(pts), 1),
            "pct_seizures":  round(100 * sum(1 for p in pts if p["has_seizures"]) / len(pts), 1),
            "pct_deceased":  round(100 * sum(1 for p in pts if p["deceased"]) / len(pts), 1),
            "pct_dha":       round(100 * sum(1 for p in pts if p["dha_therapy"]) / len(pts), 1),
            "pct_zs":        round(100 * sum(1 for p in pts if p["phenotypic_class"] == "ZS") / len(pts), 1),
            "pct_nald":      round(100 * sum(1 for p in pts if p["phenotypic_class"] == "NALD") / len(pts), 1),
            "pct_ird":       round(100 * sum(1 for p in pts if p["phenotypic_class"] == "IRD") / len(pts), 1),
            "mean_age_dx_y": round(sum(p["age_dx_y"] for p in pts) / len(pts), 1),
        })
    return {
        "genes": gene_rows,
        "total": len(PBD_GENES),
        "total_patients": len(ALL_PATIENTS),
    }


# ─── API: get_definitions ─────────────────────────────────────────────────────
def get_definitions():
    return {
        "atlas": "PBD-Atlas — Complete 8-Gene Peroxisomal Biogenesis Disorders Atlas",
        "pbd_overview": {
            "full_name": "Peroxisomal Biogenesis Disorders (PBD) — inherited defects in peroxin (PEX) genes required to assemble the peroxisomal organelle; result in global failure of all peroxisomal functions",
            "spectrum_name": "PBD-Zellweger Spectrum Disorder (PBD-ZSD) — continuous clinical spectrum from most severe (ZS) to mildest (IRD)",
            "genes_in_atlas": 8,
            "total_known_pex_genes": "14 PEX genes known to cause PBD-ZSD (PEX1/2/3/5/6/10/11B/12/13/14/16/19/26/27); PEX11B causes a fission defect (separate from biogenesis) and has an individual dashboard",
            "collective_incidence": "~1 in 50,000–100,000 for all PBD-ZSD; PEX1 alone causes ~70% of cases",
            "most_common_gene": "PEX1 (3q27.3) — ~70% of all PBD-ZSD; G843D founder allele ~40% of all ZSD alleles",
            "nbs_note": "Not in standard NBS. C26:0 VLCFA measurable by DBS/plasma LC-MS/MS (some programmes). Plasmalogen measurement separates ZSD from X-ALD definitively.",
        },
        "definitions": [
            {
                "term": "Zellweger Spectrum Disorder (ZSD) — PBD Clinical Continuum",
                "definition": "ZSD is a single clinical continuum (not three separate diseases) caused by biallelic mutations in any of the PEX biogenesis genes. Severity runs from: (1) Zellweger Syndrome (ZS): most severe; neonatal hypotonia, seizures, absent Moro reflex, dysmorphic features (wide fontanelles, high forehead, epicanthal folds, depressed nasal bridge), profound VLCFA elevation, absent plasmalogens, brain malformations (pachygyria, cortical dysplasia), hepatic dysfunction, die <12 months; (2) Neonatal ALD (NALD): intermediate; survive infancy to childhood; progressive leukodystrophy, retinal dystrophy, SNHL, developmental regression; (3) Infantile Refsum Disease (IRD): mildest; adult survival; retinitis pigmentosa, SNHL, peripheral neuropathy, anosmia, hepatomegaly, developmental delay. Severity is primarily genotype-driven (residual peroxin function correlates with phenotype). Note: IRD was originally named for its phytanic acid accumulation resembling classical Refsum disease (PHYH mutation), but they are completely different disorders.",
            },
            {
                "term": "Very Long Chain Fatty Acids (VLCFA) — Primary ZSD Biomarker",
                "definition": "VLCFAs are saturated fatty acids with chain length ≥22 carbons. C26:0 (hexacosanoic acid) and the ratio C24:0/C22:0 are the primary ZSD screening markers. Peroxisomal beta-oxidation is the only pathway for VLCFA degradation (mitochondria cannot metabolise C22+). ZSD → absent/dysfunctional peroxisomes → VLCFAs accumulate in plasma, RBC membranes, and tissues → myelin toxicity and cell membrane dysfunction. Same VLCFA elevation occurs in X-ALD (ABCD1 transporter defect) — so VLCFA alone does not diagnose ZSD. Differential: measure erythrocyte plasmalogens (low in ZSD, normal in X-ALD) to separate. VLCFA measured by plasma liquid chromatography-tandem MS (LC-MS/MS). C26:0 >0.65 μmol/L is abnormal; C24:0/C22:0 >1.39 in males (lower threshold in females).",
            },
            {
                "term": "Plasmalogens — Unique PBD Biomarker Separating ZSD from X-ALD",
                "definition": "Plasmalogens (ether phospholipids) are phospholipids with a vinyl-ether bond at the sn-1 position. Synthesis requires two peroxisomal enzymes: GNPAT (dihydroxyacetone phosphate acyltransferase, PTS1) and AGPS (alkyl-DHAP synthase, PTS2). Both are peroxisomal matrix enzymes — if peroxisome biogenesis is defective, both fail → plasmalogens absent. Measured as ethanolamine plasmalogens in erythrocytes (C16:0-DMA and C18:0-DMA). KEY CLINICAL USE: X-ALD (ABCD1 transporter defect) — VLCFA elevated but plasmalogens NORMAL (GNPAT/AGPS are PTS1/PTS2 proteins, unaffected by ABCD1 loss); ZSD — VLCFA elevated AND plasmalogens LOW. This single test separates ZSD from X-ALD definitively. Also low in RCDP type 1 (PEX7 loss) — but VLCFA NORMAL in RCDP (ACOX1/thiolase are PTS1, intact in PEX7 loss).",
            },
            {
                "term": "Peroxisomal Targeting Signal 1 (PTS1) — PEX5 Cargo",
                "definition": "PTS1 is the C-terminal tripeptide -SKL (and conserved variants: -SRL, -AKL, -ANL, -SQL, -SAL, -ARL) that directs newly synthesised cytosolic proteins to peroxisomes for import. Recognised by PEX5 (7 TPR domains). ~90% of peroxisomal matrix proteins use PTS1, including: catalase (-SKL), ACOX1 (acyl-CoA oxidase, beta-oxidation step 1), MFP2 (multifunctional protein 2, beta-oxidation steps 2+3), SCPx (thiolase-related, beta-oxidation step 4), GNPAT (plasmalogen synthesis), AGPS (plasmalogen synthesis), SDHase, and most other matrix enzymes. Loss of PEX5 → ALL PTS1 cargo remains cytosolic → global peroxisomal failure. PTS1 residues are species-conserved — SKL is the canonical tripeptide; variants are lower-affinity PEX5 ligands but still functional.",
            },
            {
                "term": "Peroxisomal Targeting Signal 2 (PTS2) — PEX7 Cargo",
                "definition": "PTS2 is an N-terminal nonapeptide (consensus R/K-[LIVA]-X5-H/Q-[LA]) found on a small subset of peroxisomal proteins: phytanoyl-CoA hydroxylase (PHYH, phytanic acid alpha-oxidation), 3-ketoacyl-CoA thiolase (ACAA1, beta-oxidation step 4, processed to mature form), AGPS (plasmalogen synthesis, converted from PTS1 to PTS2 form in some organisms), and alkylglycerone phosphate synthase. PTS2 receptor: PEX7 (cytosolic). PEX7 docks to PEX5L (long PEX5 isoform containing 37aa insert) → PEX5L carries PEX7-PTS2-cargo to PEX14 pore → PTS2 cargo released. PEX7 biallelic loss → Rhizomelic Chondrodysplasia Punctata type 1 (RCDP1): rhizomelia, stippled epiphyses, low plasmalogens, elevated phytanic acid — but VLCFA NORMAL (ACOX1/MFP2 are PTS1, intact). Contrast ZSD (PEX5 null): BOTH PTS1 and PTS2 fail → VLCFA elevated AND plasmalogens low.",
            },
            {
                "term": "Lorenzo Oil — CONTRAINDICATED in PBD-ZSD (ABCD1/X-ALD Only)",
                "definition": "Lorenzo Oil is a 4:1 mixture of glyceryl trioleate (C18:1) and glyceryl trierucate (C22:1). Its mechanism in X-ALD (ABCD1 mutation): erucic acid competitively inhibits the ELOVL1 elongase that synthesises VLCFA from precursors → reduces plasma C26:0 in X-ALD patients. In X-ALD, the peroxisomal beta-oxidation enzymes are present (peroxisomes are intact; only the ABCD1 transporter is missing) — so there is an intact peroxisome to provide the substrate entry once C26:0 synthesis is reduced. In PBD-ZSD: peroxisomes are absent or non-functional → even if C26:0 synthesis is reduced by erucic acid, there is no functioning peroxisomal beta-oxidation to clear VLCFA → no meaningful VLCFA reduction; and toxic erucic acid accumulation has been associated with cardiac lipidosis in animal studies. CRITICAL PRESCRIBING RULE: Lorenzo Oil is ONLY for X-ALD (ABCD1 mutation confirmed); absolutely contraindicated in PBD-ZSD.",
            },
            {
                "term": "PEX1/PEX6 AAA-ATPase Ring — Mechanism of PEX5 Extraction",
                "definition": "The PEX1/PEX6 ring is a heterohexameric AAA+ ATPase complex composed of three PEX1 and three PEX6 subunits arranged in alternating positions, anchored to the peroxisomal outer membrane by PEX26. Function: after PEX5 delivers PTS1 cargo into the peroxisomal matrix, PEX5 inserts into the docking/translocation complex and becomes mono-ubiquitinated on its N-terminal Cys11 by PEX10 (RING E3). The monoubiquitin is recognised by the PEX1/PEX6 ring, which uses ATP hydrolysis to 'pull' PEX5 out of the membrane (retrotranslocation) and release it back into the cytosol — ready for another import cycle. If PEX5 cannot be extracted (PEX1/6/26 or PEX10 loss): PEX5 accumulates on membrane → PEX12/PEX2 polyubiquitinate it (Lys48) → PEX5 sent to proteasome (ERAD) → cytosolic PEX5 pool depleted → import cycle stalls.",
            },
            {
                "term": "RING-Triad (PEX2·PEX10·PEX12) — Peroxisomal Ubiquitin E3 Ligases",
                "definition": "Three RING-finger peroxisomal membrane proteins (PMPs) constitute the RING triad: PEX2 (8q21.13), PEX10 (1p36.32), and PEX12 (17q12). All three face the cytosol with their RING zinc-finger domains. Working model: PEX10 monoubiquitinates PEX5-Cys11 (Ub-Lys48 or Ub-Lys63 for recycling — exact Lys is debated; mono-Ub allows PEX1/PEX6 extraction). PEX12 (with PEX2 cooperation) polyubiquitinates PEX5 (Lys48-linked chain) for ERAD when PEX5 is stuck (quality-control arm). Loss of any RING triad member → import cycle blocked (different mechanism but same endpoint). Genotype-severity: null alleles → ZS; missense (partial RING function) → NALD/IRD. PEX12 has an individual dashboard in this portal; PEX10 and PEX2 are included in this PBD-Atlas.",
            },
            {
                "term": "PMP Biogenesis Triad (PEX3·PEX16·PEX19) — De Novo Peroxisome Formation",
                "definition": "Three peroxins are required specifically for de novo peroxisomal membrane formation from the ER: PEX3 (6q24.2), PEX16 (11p11.2), and PEX19 (1q22). Pathway: PEX16 is inserted into ER membrane (via GET/TRC40 pathway) → attracts PEX3 to ER-derived pre-peroxisomal vesicles → PEX3 acts as the membrane receptor for PEX19 → PEX19 (cytosolic PMP chaperone) captures newly synthesised PMPs and presents them to PEX3 on the vesicle → PMPs insert into the budding peroxisomal membrane → mature peroxisomes form. Loss of PEX3 or PEX16: NO peroxisomal ghost membranes (complete membrane absence — most severe defect). Loss of PEX19: same result. Contrast: loss of matrix import genes (PEX1/2/5/6/10/12/26) → peroxisomal ghost membranes ARE present (membrane forms but matrix proteins cannot enter). EM ghost-membrane assessment is the first-tier test to distinguish membrane biogenesis defects from matrix import defects.",
            },
            {
                "term": "DHA (Docosahexaenoic Acid) Supplementation — Level C Evidence in ZSD",
                "definition": "DHA (C22:6n-3, omega-3) is synthesised from EPA (C20:5n-3) via elongation and peroxisomal beta-oxidation of C24:6n-3 (the last step is peroxisomal). ZSD → DHA synthesis blocked → DHA deficient in plasma and tissues, especially CNS and retina. Supplementation: oral DHA 20-100mg/kg/day (as ethyl ester or triglyceride) raises plasma DHA. Clinical benefit: retinal stabilisation in NALD/IRD (ERG improvement or slowed deterioration) in some uncontrolled series; auditory benefit unproven. Neurological progression is NOT halted by DHA. Level C evidence (expert opinion + small case series). Sources: algae-derived DHA (preferred — avoids phytanic acid contamination of fish oil); or purified marine DHA with low phytanic content. Phytanic acid dietary restriction must accompany DHA supplementation.",
            },
            {
                "term": "Phytanic Acid Metabolism — PHYH vs PBD",
                "definition": "Phytanic acid (3,7,11,15-tetramethylhexadecanoic acid) is a branched-chain fatty acid from dietary chlorophyll (ruminant meat, dairy fat, fish oil). Degradation requires peroxisomal alpha-oxidation: phytanoyl-CoA → hydroxyphytanoyl-CoA (PHYH enzyme, PTS2) → pristanal → pristanic acid. Then peroxisomal beta-oxidation (3 cycles). ZSD: ALL steps absent → phytanic acid accumulates. Classical Refsum disease (PHYH gene mutation or PEX7 mutation): only alpha-oxidation fails → pristanic acid route blocked → phytanic accumulates → peripheral neuropathy + RP + ataxia + ichthyosis; VLCFA NORMAL. Management: phytanic acid dietary restriction reduces accumulation significantly; effective (Level B) in Refsum disease and beneficial in ZSD NALD/IRD survivors. High phytanic foods: dairy fat (butter, cream, cheese), ruminant fat (beef tallow, lamb fat), cod liver oil — AVOID these.",
            },
            {
                "term": "Catalase Immunofluorescence — First-Tier Functional PBD Test",
                "definition": "Catalase (OMIM 115500) is a peroxisomal matrix enzyme that degrades hydrogen peroxide (H2O2). Its C-terminal PTS1 is -SKL. In normal fibroblasts: catalase shows punctate fluorescence (colocalized with PMP70/peroxisome staining) — peroxisomal import intact. In PBD-ZSD (all matrix import defects): catalase is diffusely cytosolic (no puncta) — PTS1 import absent. In PEX11B (peroxisome fission defect): catalase IS still punctate and peroxisomal (import machinery intact; only division of existing peroxisomes is impaired). This test separates biogenesis defects from fission defects before sequencing and is essential to correctly classify PEX11B vs ZSD. Test: anti-catalase antibody IF in patient fibroblasts. Result: punctate = import intact (fission defect); diffuse = import blocked (biogenesis defect).",
            },
            {
                "term": "Rhizomelic Chondrodysplasia Punctata (RCDP) — PEX7 and PTS2 Defect",
                "definition": "RCDP is caused by loss of PEX7 (PTS2 receptor, 6q23.3) — a RELATED but DISTINCT condition from PBD-ZSD. PEX7 delivers PTS2 cargo only. PTS2 proteins affected: AGPS (alkylglycerone phosphate synthase, plasmalogen synthesis), PHYH (phytanoyl-CoA hydroxylase, phytanic acid alpha-oxidation), and ACAA1 (3-ketoacyl-CoA thiolase, beta-oxidation final step — but ACAA1 has an alternative cytosolic form, so clinical impact is partial). Key RCDP features: proximal limb shortening (rhizomelia — humerus > radius/ulna and femur > tibia/fibula), stippled epiphyses on X-ray (chondrodysplasia punctata), cataracts, intellectual disability, low erythrocyte plasmalogens, elevated phytanic acid. VLCFA NORMAL (ACOX1, MFP2, catalase are PTS1 proteins — PEX7 does not affect PTS1 import). RCDP type 1 (PEX7) is the most common; RCDP types 2 and 3 are AGPS and GNPAT single-enzyme defects. RCDP diagnosis: rhizomelia + stippled epiphyses + low plasmalogens + normal VLCFA + PEX7 sequencing.",
            },
        ],
    }


if __name__ == "__main__":
    import json
    print("=== PBD Atlas — Functional Test ===")
    ov = get_overview()
    print(f"Genes: {ov['n_genes']}, Patients: {ov['n_patients']}, Seeds: {ov['seeds']}")
    print(f"VLCFA elevated: {ov['aggregate_clinical']['pct_vlcfa_elevated']}%")
    print(f"Plasmalogen low: {ov['aggregate_clinical']['pct_plasmalogen_low']}%")
    print(f"Retinal dystrophy: {ov['aggregate_clinical']['pct_retinal_dystrophy']}%")
    print(f"SNHL: {ov['aggregate_clinical']['pct_snhl']}%")
    print(f"Deceased: {ov['aggregate_clinical']['pct_deceased']}%")
    pheno = ov['phenotypic_distribution']
    print(f"Phenotype: ZS={pheno['ZS']['pct']}% NALD={pheno['NALD']['pct']}% IRD={pheno['IRD']['pct']}%")
    bd = get_breakdown()
    print(f"Breakdown genes: {len(bd['genes'])}")
    df = get_definitions()
    print(f"Definitions: {len(df['definitions'])}")
