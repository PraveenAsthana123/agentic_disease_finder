#!/usr/bin/env python3
"""Hereditary-Retinal-Dystrophy-Atlas — Complete 8-Gene Hereditary Retinal Dystrophy Atlas
RPGR   (Retinitis Pigmentosa GTPase Regulator; 903 aa ORF15; Xp11.23; XLR;
         RP3 — most common X-linked RP (~70% of XLRP); ORF15 purine-rich hotspot;
         standard WES/panels MISS ORF15 frameshifts — request ORF15-specific assay;
         males: severe early-onset RP; females: carrier RP via skewed X-inactivation;
         no approved gene therapy (AGTC/Beacon trials, sepofarsen ASO ongoing)) ·
USH2A  (Usherin; 5202 aa; 1q41; AR;
         Usher syndrome type 2A — most common Usher syndrome (>50% of Usher 2);
         RP + congenital moderate-severe SNHL + NO vestibular dysfunction;
         c.2299delG (p.Glu767Sfs) most common European variant;
         concurrent audiological rehabilitation + low-vision rehabilitation mandatory;
         joint ENT-ophthalmology clinic essential) ·
ABCA4  (ABCA4 Transporter; 2273 aa; 1p22.1; AR;
         Stargardt disease (STGD1) — most common hereditary macular dystrophy (~1/8000);
         foveal atrophy + pisciform flecks + lipofuscin accumulation;
         FAF: hyperfluorescent flecks + hypoAF fovea — DIAGNOSTIC;
         AVOID excessive light exposure — phototoxic bisretinoid accumulation;
         N1868I hypomorphic allele — severity depends on compound heterozygosity) ·
RHO    (Rhodopsin; 348 aa; 3q22.1; AD;
         RP4 — most common ADRP (~25% of all ADRP);
         P23H most prevalent in North America (>30% of US ADRP) — Class II misfolding;
         night blindness → peripheral constriction → central preservation late;
         Vitamin A AVOID reduction (needed for rhodopsin chromophore);
         gene therapy knockdown/replacement approach in Phase 1/2) ·
PRPF31 (Pre-mRNA Processing Factor 31; 499 aa; 19q13.42; AD;
         RP11 — reduced penetrance (60-80%) — critical genetic counselling challenge;
         CNOT3 modifier: high CNOT3 expression → unaffected carriers;
         haploinsufficiency of splicing factor → photoreceptor-specific RNA failure;
         unaffected carriers in families — incomplete penetrance complicates predictive testing) ·
CRB1   (Crumbs Homolog 1; 1406 aa; 1q31.3; AR;
         RP12 + LCA8 (Leber Congenital Amaurosis type 8);
         para-arteriolar preservation of RPE (PPRPE) — PATHOGNOMONIC on fundus exam;
         thick retina on OCT early — distinctive; voretigene neparvovec is for RPE65 NOT CRB1;
         gene therapy preclinical; Coats-like exudates rare complication) ·
CNGB3  (Cyclic Nucleotide-Gated Channel Beta 3; 809 aa; 8q21.3; AR;
         Complete Achromatopsia — most common genetic cause;
         p.T383IlefsX most common pathogenic variant (~90% in NW European);
         ERG: absent photopic + normal scotopic — DIAGNOSTIC;
         FL-41 rose/amber tinted lenses reduce photophobia;
         STABLE — does NOT progress to blindness (rods intact);
         AGTC/Beacon gene therapy Phase 2/3 — among most advanced cone dystrophy trials) ·
BEST1  (Bestrophin 1; 585 aa; 11q12.3; AD;
         Best vitelliform macular dystrophy (VMD2) — yolk-like subfoveal lesion;
         EOG Arden ratio (LP:dark) LOW in affected AND unaffected carriers — PATHOGNOMONIC;
         ERG NORMAL — distinguishes from RP;
         CNV complication → anti-VEGF (bevacizumab/ranibizumab) responsive;
         AR bestrophinopathy: biallelic → more widespread involvement)
320-patient aggregate cohort (8 × 40, seeds 1494–1501)
"""

import random

SEED_BASE = 1494

RETINAL_GENES = [
    # ── RPGR — X-linked Retinitis Pigmentosa ──
    {
        "gene": "RPGR",
        "protein": "Retinitis Pigmentosa GTPase Regulator — ORF15 Hotspot X-Linked RP",
        "alias": (
            "RPGR; OMIM gene 312610; RP3 OMIM 300029; Xp11.23; 903 aa (ORF15 isoform); ~90 kDa; "
            "RPGR encodes a GTPase regulator localised to the photoreceptor connecting cilium, "
            "the critical trafficking hub between the inner segment (biosynthesis) and outer segment "
            "(phototransduction). The constitutively expressed isoform (RPGR-ex1-19) and the "
            "retina-specific isoform containing the purine-rich repetitive exon ORF15 are both "
            "disease-relevant. ORF15 is a hotspot for frameshift variants that are refractory to "
            "PCR amplification due to repetitive sequence — standard WES and gene panels MISS these "
            "variants in up to 50% of affected males. RP3 accounts for ~70% of X-linked RP "
            "and 10-15% of all RP. Males: severe progressive RP with childhood night blindness, "
            "central vision preservation into 3rd-4th decade. Females: obligate carriers — "
            "20-25% develop sectoral or classic RP due to skewed X-inactivation. "
            "No approved gene therapy as of 2024; AGTC/Beacon Phase 2/3, sepofarsen ASO trials ongoing."
        ),
        "aa": "903 aa (ORF15)",
        "kDa": "~90 kDa",
        "locus": "Xp11.23",
        "omim_gene": 312610,
        "omim_disease": 300029,
        "inheritance": "XLR — carrier females 20-25% symptomatic (skewed X-inactivation)",
        "gene_class": (
            "RPGR localises to the connecting cilium of rod and cone photoreceptors, where it acts as "
            "a docking platform regulating vectorial protein transport via interaction with RPGRIP1. "
            "The ORF15 isoform (RPGR-ORF15) includes a glutamic acid-rich domain that interacts with "
            "multiple ciliary transport proteins. Loss of RPGR function impairs the trafficking of "
            "phototransduction proteins (rhodopsin, transducin, PDE6) from inner to outer segment, "
            "leading to outer segment disorganisation and progressive photoreceptor degeneration. "
            "The repetitive nature of ORF15 creates a diagnostic blind spot: standard short-read WES "
            "generates PCR slippage artefacts over the purine repeats, resulting in false-negative "
            "sequencing. Long-read sequencing (PacBio, Oxford Nanopore) or specific ORF15 Sanger "
            "protocols are required. Approximately 50% of XLRP families are RPGR-positive; "
            "failure to identify the ORF15 variant means the diagnostic rate is substantially underestimated."
        ),
        "n_patients": 40,
        "seed": SEED_BASE,
        "etiologies": [
            ("ORF15 frameshift — purine-rich deletion/insertion — missed by standard WES", 0.48),
            ("Exon 1-19 missense — connecting cilium interaction domain", 0.28),
            ("Exon 1-19 truncating — haploinsufficiency", 0.15),
            ("Large deletion RPGR (encompassing ORF15)", 0.05),
            ("RPGR VUS — ORF15 sequencing requested", 0.04),
        ],
        "key_alerts": [
            "RPGR-ORF15: Standard WES/gene panels MISS ORF15 frameshifts in up to 50% — REQUEST specific ORF15 sequencing protocol",
            "RPGR-FEMALES: Obligate carrier females — 20-25% develop symptomatic RP (skewed X-inactivation) — ophthalmology assessment mandatory",
            "RPGR-XLRP: Males: childhood night blindness, progressive — AVOID excessive light (UV protection, dark-tinted lenses outdoors)",
            "RPGR-SUPPLEMENTS: AREDS2 only — high-dose Vitamin A not proven in RPGR (unlike USH2A/RHO); avoid beta-carotene in current/ex-smokers (lung cancer risk)",
            "RPGR-GENE-THERAPY: No approved therapy 2024 — AGTC/Beacon Phase 2/3, sepofarsen ASO Phase 2/3 — enrol eligible patients in trials",
            "RPGR-MONITORING: Annual ERG + visual field (Goldmann 30-2) + OCT — document progression rate for trial eligibility",
        ],
    },
    # ── USH2A — Usher Syndrome Type 2A ──
    {
        "gene": "USH2A",
        "protein": "Usherin — Most Common Usher Syndrome Type 2, RP + Congenital SNHL",
        "alias": (
            "USH2A; OMIM gene 608400; Usher 2A OMIM 276901; DFNB39 OMIM; 1q41; 5202 aa; ~570 kDa; "
            "USH2A encodes usherin, a large extracellular matrix scaffold protein localised to "
            "the periciliary membrane of photoreceptors and the stereocilia ankle-link region of "
            "cochlear hair cells. Usher syndrome type 2A: autosomal recessive, accounts for >50% "
            "of all Usher syndrome. Clinical triad: (1) moderate-severe congenital SNHL (not profound, "
            "distinguishing from Usher 1); (2) RP (onset teens-20s — peripheral vision loss, night "
            "blindness); (3) NO vestibular dysfunction (distinguishes from Usher 1). "
            "c.2299delG (p.Glu767Sfs) is the most prevalent European pathogenic variant (~30% of "
            "all USH2A alleles in Europeans). Biallelic USH2A variants without SNHL cause isolated RP "
            "(DFNB39). Management requires joint ENT-ophthalmology clinic; cochlear implants for severe "
            "SNHL; low-vision rehabilitation; orientation and mobility."
        ),
        "aa": "5202 aa",
        "kDa": "~570 kDa",
        "locus": "1q41",
        "omim_gene": 608400,
        "omim_disease": 276901,
        "inheritance": "AR — biallelic pathogenic variants; carrier parents unaffected",
        "gene_class": (
            "Usherin (USH2A) is a transmembrane and extracellular scaffold protein essential for "
            "maintaining the structural integrity of the stereocilia ankle-link apparatus in hair cells "
            "and the periciliary membrane complex (PMC) in photoreceptors. The protein contains "
            "multiple laminin epidermal growth factor (LE) repeats, fibronectin III domains, and "
            "a PDZ-binding motif. In photoreceptors, usherin anchors the PMC — the junction between "
            "the inner segment plasma membrane and the connecting cilium — essential for trafficking "
            "phototransduction proteins. Loss of usherin destabilises this scaffold, leading to "
            "progressive rod and cone degeneration. In cochlear hair cells, usherin is a structural "
            "component of the ankle-link between stereocilia; its absence causes progressive "
            "mechanosensory hair cell dysfunction. The dual retinal-cochlear expression explains "
            "the combined sensory deficit. Gene therapy is actively pursued; exon-skipping approaches "
            "for common variants are in early development."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 1,
        "etiologies": [
            ("c.2299delG (p.Glu767Sfs) compound heterozygous — common European founder", 0.32),
            ("Novel truncating compound heterozygous — LoF", 0.30),
            ("Missense compound heterozygous — LE/FNIII domain", 0.22),
            ("Large exonic deletion (MLPA confirmed)", 0.10),
            ("USH2A VUS + truncating — one pathogenic allele", 0.06),
        ],
        "key_alerts": [
            "USH2A-AUDIOGRAM: Congenital SNHL moderate-severe — cochlear implant evaluation MANDATORY for severe loss; joint ENT-ophthalmology clinic",
            "USH2A-NO-VESTIBULAR: NO vestibular dysfunction — distinguishes Usher 2 from Usher 1 (MYOVIIA/CDH23); balance testing if atypical",
            "USH2A-ISOLATED-RP: Biallelic USH2A without deafness = isolated RP (DFNB39) — always check SNHL in any USH2A-RP",
            "USH2A-FIRST-GENE: First gene to test in any AR-RP patient with unexplained SNHL — highest yield",
            "USH2A-SUPPLEMENTS: AREDS2 multi-vitamin + Vitamin A 15,000 IU/day (evidence: Berson 1993 — slows ERG decline) — avoid in pregnancy",
            "USH2A-LOW-VISION: Orientation & mobility training + white cane training + assistive technology — start when visual field <20 degrees",
        ],
    },
    # ── ABCA4 — Stargardt Disease ──
    {
        "gene": "ABCA4",
        "protein": "ABCA4 Flippase — Stargardt Disease Most Common Hereditary Macular Dystrophy",
        "alias": (
            "ABCA4; OMIM gene 601691; Stargardt OMIM 248200; 1p22.1; 2273 aa; ~256 kDa; "
            "ABCA4 encodes an ATP-binding cassette transporter in the outer segment disc membranes "
            "of rod and cone photoreceptors, functioning as a retinaldehyde flippase that transfers "
            "N-retinylidene-PE (NRPE) from the luminal to the cytoplasmic leaflet of disc membranes. "
            "Loss of ABCA4 leads to accumulation of all-trans-retinal and its condensation products "
            "(A2PE, A2E, A2-DHP-PE) — collectively called bisretinoids — which are toxic to the RPE "
            "and trigger RPE-photoreceptor atrophy. Stargardt disease (STGD1): biallelic ABCA4 — "
            "most common hereditary macular dystrophy (~1/8000-10,000); onset typically first-second "
            "decade; central vision loss, colour disturbance, photophobia. FAF: hyperfluorescent "
            "pisciform flecks + hypoAF foveal region — PATHOGNOMONIC. AVOID excessive light. "
            "Allelic series: N1868I (c.5603A>T) hypomorphic allele — compound het with severe allele → "
            "intermediate severity. Cone-rod dystrophy (CORD) and panretinal RP with severe biallelic variants."
        ),
        "aa": "2273 aa",
        "kDa": "~256 kDa",
        "locus": "1p22.1",
        "omim_gene": 601691,
        "omim_disease": 248200,
        "inheritance": "AR — biallelic (both alleles must be pathogenic/hypomorphic)",
        "gene_class": (
            "ABCA4 is a photoreceptor-specific ABC transporter located in the rim region of rod and "
            "cone outer segment disc membranes. After photoactivation, all-trans-retinal is released "
            "from rhodopsin into the disc lumen. ABCA4 flips N-retinylidene-phosphatidylethanolamine "
            "(NRPE), the Schiff base of all-trans-retinal and PE, from the disc lumen to the "
            "cytoplasmic face, allowing NADPH-dependent reduction to all-trans-retinol for recycling. "
            "ABCA4 deficiency traps NRPE in the luminal disc space, driving formation of A2E and "
            "related bisretinoids — phototoxic fluorophores that accumulate in the RPE, cause oxidative "
            "stress, complement pathway activation, and progressive geographic atrophy. "
            "Light accelerates A2E formation (Stargardt photosensitivity rationale). "
            "The hypomorphic allele N1868I retains partial transport activity — 'two-hit' dosage "
            "model predicts phenotype severity based on residual ABCA4 function."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 2,
        "etiologies": [
            ("Classic Stargardt — two severe variants (frameshift/nonsense/missense ≥0% activity)", 0.42),
            ("Stargardt intermediate — one severe + N1868I hypomorphic allele", 0.22),
            ("Cone-rod dystrophy (CORD) — two severe ABCA4 variants, dominant cone involvement", 0.18),
            ("ABCA4-RP — panretinal degeneration — two severe variants", 0.12),
            ("ABCA4 VUS compound heterozygous — functional testing ordered", 0.06),
        ],
        "key_alerts": [
            "ABCA4-LIGHT: AVOID excessive light exposure — photosensitivity accelerates bisretinoid accumulation and geographic atrophy; UV400 filter lenses mandatory",
            "ABCA4-FAF: Fundus autofluorescence (FAF) — hyperfluorescent pisciform flecks + hypoAF fovea is DIAGNOSTIC; perform FAF at diagnosis and annually",
            "ABCA4-VITAMIN-A: AVOID high-dose Vitamin A supplementation — increased retinoid load worsens A2E accumulation in ABCA4 (OPPOSITE of USH2A/RHO advice)",
            "ABCA4-N1868I: N1868I (c.5603A>T) is a hypomorphic allele — alone causes no disease; must check for second severe pathogenic allele for STGD diagnosis",
            "ABCA4-SCOPE: ABCA4 variants cause Stargardt + CORD + panretinal RP — phenotype determined by residual ABCA4 transport activity (genotype-severity correlation)",
            "ABCA4-TRIALS: Oral ALK-001 (C20-D3-Vitamin A, deuterated) Phase 3 — reduces A2E formation; gene therapy AGTC Phase 2 — register eligible patients",
        ],
    },
    # ── RHO — Rhodopsin ADRP ──
    {
        "gene": "RHO",
        "protein": "Rhodopsin — Most Common Autosomal Dominant RP, P23H Class II Hotspot",
        "alias": (
            "RHO; OMIM gene 180380; RP4 OMIM 613731; 3q22.1; 348 aa; ~38 kDa; "
            "RHO encodes rhodopsin, the visual pigment of rod photoreceptors — a G protein-coupled "
            "receptor (GPCR) in the outer segment disc membrane. Rhodopsin consists of the apoprotein "
            "opsin covalently linked to 11-cis retinal via a protonated Schiff base at Lys296. "
            "Pathogenic variants cause AD RP4 (~25% of all ADRP; most common single-gene ADRP). "
            "Class I variants (surface-expressed, normal trafficking): mild-moderate RP, slower progression. "
            "Class II variants (misfolded, ER-retained, cytotoxic): severe RP, rapid progression — "
            "dominant negative mechanism by recruitment of HSP70/90 chaperones, UPR activation. "
            "P23H (p.Pro23His): most prevalent variant in North America (>30% of US ADRP); Class II. "
            "Clinical: night blindness from childhood, peripheral constriction proceeding centripetally, "
            "foveal preservation until 4th-5th decade in Class I; earlier in Class II. "
            "Vitamin A supplementation (15,000 IU/day): evidence for slowing ERG decline (Berson data) "
            "— do NOT restrict (11-cis retinal needed for chromophore regeneration)."
        ),
        "aa": "348 aa",
        "kDa": "~38 kDa",
        "locus": "3q22.1",
        "omim_gene": 180380,
        "omim_disease": 613731,
        "inheritance": "AD — haploinsufficiency (Class I) or dominant negative misfolding (Class II)",
        "gene_class": (
            "Rhodopsin is the paradigmatic GPCR and visual pigment of vertebrate rod photoreceptors, "
            "comprising seven transmembrane helices that cradle 11-cis retinal in a hydrophobic pocket. "
            "Photon absorption isomerises 11-cis to all-trans retinal, triggering conformational change "
            "to Meta II rhodopsin — activating transducin (Gt) and initiating the phototransduction cascade. "
            "After activation, rhodopsin is phosphorylated by GRK1 and inactivated by arrestin binding. "
            "AD RP4 mutations cause disease by two distinct mechanisms: Class I variants produce "
            "surface-expressed mutant rhodopsin with abnormal kinetics (constitutive activation or "
            "prolonged G-protein coupling); Class II variants misfold in the ER and activate the "
            "unfolded protein response (UPR), triggering rod apoptosis via dominant-negative sequestration "
            "of wild-type rhodopsin and chaperone depletion. P23H (the most common ADRP variant globally) "
            "is a prototypical Class II mutation — its dominant cytotoxicity explains why even a single "
            "copy causes disease despite 50% normal rhodopsin from the WT allele."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 3,
        "etiologies": [
            ("P23H Class II — dominant negative misfolding — North American founder", 0.32),
            ("Class II missense (other) — ER retention, UPR activation — severe/rapid", 0.28),
            ("Class I missense — surface expressed — mild/moderate — slower progression", 0.22),
            ("RHO truncating — haploinsufficiency — rare AD mechanism", 0.12),
            ("RHO AR — biallelic severe variants — severe early-onset RP", 0.06),
        ],
        "key_alerts": [
            "RHO-P23H: P23H is most common North American ADRP variant — Class II misfolding dominant negative — severe/rapid progression vs. Class I",
            "RHO-VITAMIN-A: Vitamin A 15,000 IU/day — evidence for slowing ERG decline (Berson 1993-2010); do NOT restrict retinoid (chromophore source); avoid in pregnancy",
            "RHO-AVOID-BETA-CAROTENE: AREDS2: use formulation WITHOUT beta-carotene in current/ex-smokers (lung cancer risk) — lutein/zeaxanthin instead",
            "RHO-CLASS: Class I vs Class II determines prognosis — Class II (P23H, others): faster progression; gene therapy must account for dominant-negative mechanism",
            "RHO-GENE-THERAPY: Knockdown-and-replace strategy (knockdown both alleles + deliver codon-modified RHO) in Phase 1/2 — enrol eligible patients",
            "RHO-AR: Biallelic RHO (rare) — more severe, early-onset — loss-of-function both alleles — mechanism distinct from AD",
        ],
    },
    # ── PRPF31 — Reduced Penetrance RP ──
    {
        "gene": "PRPF31",
        "protein": "Pre-mRNA Processing Factor 31 — Reduced Penetrance RP11, CNOT3 Modifier",
        "alias": (
            "PRPF31; OMIM gene 606419; RP11 OMIM 600138; 19q13.42; 499 aa; ~55 kDa; "
            "PRPF31 encodes a U4/U6 snRNP-associated splicing factor essential for spliceosome assembly. "
            "AD RP11 is clinically notable for reduced penetrance (60-80%) — unaffected carriers who "
            "carry pathogenic PRPF31 variants are common within RP11 families. The CNOT3 modifier gene "
            "(19p13.3) regulates the transcription of the wild-type PRPF31 allele: individuals with "
            "high wild-type PRPF31 expression (due to high CNOT3-responsive promoter activity) have "
            "sufficient RP protein despite haploinsufficiency and do NOT develop RP. "
            "This modifier effect has profound genetic counselling implications: a sibling of an "
            "affected PRPF31 patient who carries the same pathogenic variant may be permanently unaffected. "
            "Photoreceptor and RPE cells require high levels of precise pre-mRNA splicing due to "
            "their large transcriptomes (rhodopsin, RPE65, ABCA4 all have many exons); "
            "splicing factor haploinsufficiency selectively affects these cell types. "
            "Severity ranges from mild adult-onset (20s-30s) to moderately severe sectoral RP."
        ),
        "aa": "499 aa",
        "kDa": "~55 kDa",
        "locus": "19q13.42",
        "omim_gene": 606419,
        "omim_disease": 600138,
        "inheritance": "AD — reduced penetrance 60-80% (CNOT3 modifier); haploinsufficiency mechanism",
        "gene_class": (
            "PRPF31 is an essential component of the tri-snRNP (U4/U6.U5) spliceosomal complex, "
            "required for catalytic spliceosome assembly. The protein bridges U4 and U5 snRNPs, "
            "stabilising the pre-catalytic spliceosome conformation. Haploinsufficiency of PRPF31 "
            "reduces the efficiency of splicing for multi-intron transcripts — photoreceptors and "
            "RPE cells are particularly vulnerable because they express very long, intron-rich mRNAs "
            "(rhodopsin, guanylate cyclases, RPE65) at high rates. The CNOT3 modifier acts by "
            "controlling transcription from the wild-type PRPF31 allele: high CNOT3 activity → "
            "upregulation of WT PRPF31 → compensation for the haploinsufficient mutant allele → "
            "no RP phenotype. Low CNOT3 activity → insufficient WT PRPF31 → splicing failure → "
            "photoreceptor degeneration. This is one of the clearest examples in retinal genetics "
            "of a modifier gene controlling penetrance, and it means PRPF31 genetic counselling "
            "must explicitly address the possibility of unaffected variant carriers in families."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 4,
        "etiologies": [
            ("Truncating variant (frameshift/nonsense) — haploinsufficiency — penetrance 60-80%", 0.55),
            ("Splice-site variant — altered U4/U6 snRNP interaction — haploinsufficiency", 0.22),
            ("Missense — nCoA interaction domain — partial LOF", 0.14),
            ("Large deletion/duplication (MLPA confirmed)", 0.06),
            ("PRPF31 VUS — functional splicing assay requested", 0.03),
        ],
        "key_alerts": [
            "PRPF31-PENETRANCE: REDUCED PENETRANCE (60-80%) — unaffected family member may carry same pathogenic variant — CNOT3 modifier controls expression from WT allele",
            "PRPF31-COUNSELLING: Genetic counselling MUST explain reduced penetrance — sibling carrier may never develop RP — cannot predict penetrance from variant alone",
            "PRPF31-CNOT3: CNOT3 modifier testing (research labs) can predict penetrance risk — high CNOT3 haplotype → likely unaffected carrier",
            "PRPF31-SPLICING-FACTORS: Other RP splicing factor genes (PRPF3, PRPF4, PRPF6, PRPF8) — same mechanism, similar reduced penetrance — check family history carefully",
            "PRPF31-SUPPLEMENTS: AREDS2 + Vitamin A 15,000 IU/day (Berson data general RP) — monitor LFTs annually with Vit A supplementation",
            "PRPF31-MONITORING: ERG + visual field annually — progression typically slower than RPGR or RHO Class II; low-vision services when VA <6/18",
        ],
    },
    # ── CRB1 — RP12 / LCA8 ──
    {
        "gene": "CRB1",
        "protein": "Crumbs Homolog 1 — RP12/LCA8, Para-arteriolar RPE Preservation PATHOGNOMONIC",
        "alias": (
            "CRB1; OMIM gene 604210; RP12 OMIM 600105; LCA8 OMIM 604210; 1q31.3; 1406 aa; ~160 kDa; "
            "CRB1 encodes a transmembrane cell polarity protein essential for photoreceptor outer "
            "segment morphogenesis and maintaining the tight junctions of the outer limiting membrane. "
            "Biallelic CRB1 variants cause: (1) LCA8 — severe neonatal/infantile visual impairment "
            "(nystagmus, absent fixation, Franceschetti sign — eye pressing); (2) RP12 — later-onset "
            "progressive RP (childhood to early adulthood). "
            "PATHOGNOMONIC SIGN: Para-arteriolar preservation of the RPE (PPRPE) — zones of preserved "
            "pigment epithelium straddling major retinal arterioles on a background of diffuse atrophy. "
            "OCT distinctive: thickened retina in early stages (distinct from other RP). "
            "CRB1 accounts for ~10% of LCA. Critical: voretigene neparvovec (Luxturna) treats RPE65-LCA2, "
            "NOT CRB1-LCA8 — confusion must be avoided. Coats-like exudative retinopathy: rare CRB1 "
            "complication requiring anti-VEGF or laser."
        ),
        "aa": "1406 aa",
        "kDa": "~160 kDa",
        "locus": "1q31.3",
        "omim_gene": 604210,
        "omim_disease": 600105,
        "inheritance": "AR — biallelic LOF/missense; de novo possible for missense",
        "gene_class": (
            "CRB1 is the human orthologue of Drosophila Crumbs, a central determinant of apical-basal "
            "cell polarity in epithelial cells. In the retina, CRB1 is localised to the subapical region "
            "(SAR) of photoreceptors at the outer limiting membrane (OLM), where it organises the "
            "adherens junction complex via interaction with MPP5 (PALS1) and LIN7. "
            "The CRB complex (CRB1-MPP5-MUPP1) anchors photoreceptor inner segments to Müller glia "
            "through a belt of adherens junctions, maintaining the structural integrity of the "
            "photoreceptor layer. Loss of CRB1 causes progressive disruption of the OLM, allowing "
            "invasion of photoreceptor processes into the subretinal space — leading to progressive "
            "degeneration. The para-arteriolar RPE preservation (PPRPE) is explained by the "
            "fact that arteriolar walls provide structural support, locally preserving photoreceptor-RPE "
            "architecture even as the surrounding retina degenerates — this creates the "
            "pathognomonic white stripes alongside arterioles visible on funduscopy."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 5,
        "etiologies": [
            ("LCA8 — biallelic severe variants — congenital visual impairment", 0.35),
            ("RP12 — biallelic moderate variants — childhood-onset progressive RP", 0.40),
            ("CRB1 with PPRPE sign — biallelic truncating/missense", 0.15),
            ("Coats-like exudative CRB1 retinopathy (rare complication)", 0.06),
            ("CRB1 VUS compound heterozygous — functional testing", 0.04),
        ],
        "key_alerts": [
            "CRB1-PPRPE: Para-arteriolar preservation of the RPE (PPRPE) on fundoscopy — WHITE STRIPES alongside arterioles — PATHOGNOMONIC for CRB1; perform careful fundus exam",
            "CRB1-NOT-LUXTURNA: Voretigene neparvovec (Luxturna) treats RPE65-LCA2 ONLY — NOT CRB1-LCA8 — do NOT confuse; genetic diagnosis mandatory before gene therapy referral",
            "CRB1-LCA8: LCA8 — congenital visual impairment — Franceschetti sign (eye pressing/rubbing); nystagmus — low-vision services from birth; portability devices",
            "CRB1-THICK-OCT: Thickened retina on OCT in early CRB1-RP12 — distinctive from most other RP; useful diagnostic clue before genetic testing",
            "CRB1-COATS: Rare Coats-like exudative retinopathy in CRB1 — monitor for exudates — anti-VEGF (bevacizumab) or laser photocoagulation for exudative lesions",
            "CRB1-GENE-THERAPY: CRB1 gene therapy preclinical (size constraint — AAV dual vector approach); no approved therapy — register in natural history studies",
        ],
    },
    # ── CNGB3 — Achromatopsia ──
    {
        "gene": "CNGB3",
        "protein": "CNG Channel Beta 3 — Achromatopsia, Stable Cone Dystrophy, Gene Therapy Phase 2/3",
        "alias": (
            "CNGB3; OMIM gene 605080; Achromatopsia OMIM 262300; 8q21.3; 809 aa; ~92 kDa; "
            "CNGB3 encodes the beta subunit of the cone cyclic nucleotide-gated (CNG) channel, "
            "the primary transduction channel mediating the cone photoreceptor light response. "
            "The CNG channel (CNGA3/CNGB3 heterotetramer) is activated by cGMP in dark-adapted cones "
            "and closes upon photoactivation (PDE6 degrades cGMP). CNGB3 mutations cause complete "
            "achromatopsia in >50% of cases (the other major gene is CNGA3). "
            "p.T383IlefsX (c.1148delC): most prevalent pathogenic variant (~90% of NW European "
            "CNGB3 achromatopsia patients). Clinical: congenital nystagmus, profound photophobia, "
            "total colour blindness (achromacy), reduced visual acuity (0.1-0.2 / 20/200), "
            "central scotoma on photopic ERG. Critical: condition is STABLE — rods are intact — "
            "patient will NOT progress to total blindness. FL-41 rose/amber tinted lenses dramatically "
            "reduce photophobia. Gene therapy: AGTC/Beacon Therapeutics Phase 2/3 (NCT02599922, "
            "NCT02161380) — among the most advanced cone dystrophy gene therapy trials as of 2024."
        ),
        "aa": "809 aa",
        "kDa": "~92 kDa",
        "locus": "8q21.3",
        "omim_gene": 605080,
        "omim_disease": 262300,
        "inheritance": "AR — biallelic (CNGB3 ~50% of achromatopsia; CNGA3 ~25%)",
        "gene_class": (
            "The cone CNG channel is a heterotetrameric complex of CNGA3 (alpha, 3 copies) and CNGB3 "
            "(beta, 1 copy) subunits forming a cation channel in the outer segment plasma membrane. "
            "In darkness, cGMP (produced by retGC) opens the CNG channel, allowing Na+/Ca2+ influx "
            "that depolarises the cone and releases glutamate at the synaptic terminal. "
            "Photoactivation activates PDE6 (via transducin), which hydrolyses cGMP, closing the "
            "CNG channel, hyperpolarising the cone, and reducing glutamate release — the sign change "
            "(signal inversion) that constitutes the phototransduction signal. "
            "CNGB3 loss prevents correct assembly and trafficking of the CNGA3/CNGB3 heterotetramer "
            "to the outer segment membrane. In its absence, some CNGA3 homotetramers reach the membrane "
            "but with altered gating properties, resulting in sustained depolarisation and eventual "
            "cone degeneration. Because rods use a different CNG channel (CNGA1/CNGB1), rod function "
            "is completely preserved — achromatopsia patients have normal rod-mediated night vision "
            "and full peripheral visual fields — a critically important distinction from RP."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 6,
        "etiologies": [
            ("p.T383IlefsX homozygous — NW European founder — complete achromatopsia", 0.45),
            ("p.T383IlefsX compound heterozygous + second truncating allele", 0.28),
            ("Novel truncating biallelic — complete achromatopsia", 0.16),
            ("Missense biallelic — incomplete achromatopsia (partial cone function)", 0.08),
            ("CNGB3 VUS — functional expression testing", 0.03),
        ],
        "key_alerts": [
            "CNGB3-STABLE: Achromatopsia is STABLE — rods intact — patient will NOT go blind — reassurance critical for psychosocial wellbeing",
            "CNGB3-FL41: FL-41 rose/amber tinted lenses — significantly reduce photophobia and improve functional vision outdoors; prescribe immediately at diagnosis",
            "CNGB3-RODS-NORMAL: Normal scotopic ERG + absent photopic ERG = DIAGNOSTIC — distinguishes from cone-rod dystrophy (progressive, rods eventually involved)",
            "CNGB3-GENE-THERAPY: AGTC/Beacon Phase 2/3 gene therapy trials — register ALL eligible patients; best-practice approach for achromatopsia now",
            "CNGB3-CNGA3: If CNGB3 negative, test CNGA3 (25% of achromatopsia), GNAT2, PDE6C, PDE6H, ATF6 — achromatopsia gene panel recommended",
            "CNGB3-DARK-LENS: Dark-tinted lenses (OD 4 filter) + broad-brimmed hat — significantly improve photophobia outdoors; occupational adaptation counselling",
        ],
    },
    # ── BEST1 — Best Vitelliform Macular Dystrophy ──
    {
        "gene": "BEST1",
        "protein": "Bestrophin 1 — Best VMD2, EOG Arden Ratio PATHOGNOMONIC, CNV Anti-VEGF",
        "alias": (
            "BEST1; OMIM gene 607854; Best VMD2 OMIM 153700; AR bestrophinopathy OMIM 611809; "
            "11q12.3; 585 aa; ~68 kDa; "
            "BEST1 encodes bestrophin-1, a calcium-activated chloride channel (CaCC) expressed in "
            "the basolateral membrane of RPE cells. Best vitelliform macular dystrophy (Best disease, "
            "VMD2): autosomal dominant, most common hereditary vitelliform maculopathy. "
            "Staged progression: I (pre-vitelliform — normal vision, abnormal EOG), "
            "II (vitelliform 'egg yolk'), III (scrambled egg), IV (pseudo-hypopyon), V (atrophic scar). "
            "Pathognomonic test: EOG (Electro-oculogram) Arden ratio (light peak:dark trough). "
            "KEY RULE: EOG is ABNORMAL in ALL gene carriers (affected + unaffected) — affected AND "
            "unaffected carriers have subnormal Arden ratio. ERG is NORMAL — pan-retinal rod/cone "
            "function intact. This distinguishes Best disease from RP (abnormal ERG). "
            "CNV (choroidal neovascularisation) complication: anti-VEGF therapy "
            "(bevacizumab, ranibizumab, aflibercept) responsive."
        ),
        "aa": "585 aa",
        "kDa": "~68 kDa",
        "locus": "11q12.3",
        "omim_gene": 607854,
        "omim_disease": 153700,
        "inheritance": "AD (Best VMD2) — dominant negative; AR bestrophinopathy (biallelic LOF)",
        "gene_class": (
            "Bestrophin-1 (BEST1) is a calcium-activated chloride channel residing in the basolateral "
            "membrane of RPE cells, where it is proposed to regulate chloride conductance and intracellular "
            "calcium levels. The protein assembles as a pentameric channel with a central ion-conducting "
            "pore. Best disease mutations cluster in the transmembrane domains and alter channel gating "
            "or abolish Cl- conductance. The precise mechanism linking BEST1 channel dysfunction to "
            "vitelliform material accumulation remains under investigation, but impaired RPE fluid "
            "regulation and phagocytosis of shed photoreceptor outer segments are implicated. "
            "The electrooculogram (EOG) measures the standing potential of the eye — generated by "
            "the RPE — which rises upon light exposure (light peak) and falls in darkness (dark trough). "
            "The Arden ratio (light peak / dark trough > 1.85 normal; ≤1.5 clearly abnormal) is "
            "abnormal even in genetically positive unaffected carriers because the channel defect "
            "impairs the light-induced chloride conductance change regardless of visual phenotype. "
            "This makes EOG the most sensitive diagnostic and carrier-detection tool for BEST1."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 7,
        "etiologies": [
            ("Best VMD2 — AD missense vitelliform stage (vitelliform/scrambled egg)", 0.38),
            ("Best VMD2 — AD — atrophic scar stage (late/advanced)", 0.22),
            ("Best VMD2 — AD — pre-vitelliform (carrier, abnormal EOG, normal vision)", 0.20),
            ("AR bestrophinopathy — biallelic BEST1 LOF — wider involvement", 0.12),
            ("Best VMD2 with CNV complication — anti-VEGF required", 0.08),
        ],
        "key_alerts": [
            "BEST1-EOG: EOG Arden ratio (LP:dark trough) PATHOGNOMONIC — abnormal in ALL BEST1 carriers (affected + unaffected); always perform EOG, not just ERG",
            "BEST1-ERG-NORMAL: ERG is NORMAL in Best VMD2 — abnormal ERG suggests different diagnosis (RP, cone-rod dystrophy); distinguish from panretinal degeneration",
            "BEST1-CNV: Choroidal neovascularisation (CNV) — complication in Best VMD2 — if metamorphopsia or vision drop: OCT-A + IVFA — anti-VEGF (bevacizumab/aflibercept) responsive",
            "BEST1-CARRIERS: Screen first-degree relatives with EOG + genetic testing — unaffected EOG-positive carrier parents identified; important for reproductive counselling",
            "BEST1-AR: AR bestrophinopathy (biallelic BEST1) — more severe, diffuse RPE involvement, earlier visual loss — compound heterozygous or homozygous LOF",
            "BEST1-STAGES: Document stage at diagnosis — pre-vitelliform → vitelliform → scrambled egg → pseudo-hypopyon → atrophic scar; prognosis variable; central VA preserved until atrophic stage",
        ],
    },
]


def _make_cohort(gene_data: dict) -> list:
    r = random.Random(gene_data["seed"])
    gene = gene_data["gene"]
    etiologies = gene_data["etiologies"]
    pts = []

    for i in range(gene_data["n_patients"]):
        # Draw etiology
        roll = r.random()
        cumul = 0.0
        etiol = etiologies[-1][0]
        for et, prob in etiologies:
            cumul += prob
            if roll < cumul:
                etiol = et
                break

        # Sex distribution
        if gene == "RPGR":
            sex = "M" if r.random() < 0.70 else "F"  # XLR — mostly males
        elif gene in ("USH2A", "ABCA4", "CRB1", "CNGB3"):
            sex = "M" if r.random() < 0.50 else "F"  # AR — equal sex
        elif gene in ("RHO", "PRPF31", "BEST1"):
            sex = "M" if r.random() < 0.50 else "F"  # AD — equal sex
        else:
            sex = "M" if r.random() < 0.50 else "F"

        # Onset age (years → months)
        onset_ranges = {
            "RPGR": (7, 18),
            "USH2A": (10, 25),   # RP onset (SNHL congenital)
            "ABCA4": (8, 22),
            "RHO": (10, 25),
            "PRPF31": (18, 40),
            "CRB1": (0, 10),
            "CNGB3": (0, 1),     # congenital
            "BEST1": (6, 20),
        }
        lo, hi = onset_ranges[gene]
        onset_y = round(lo + r.random() * (hi - lo), 1)
        onset_m = round(onset_y * 12)
        age_current_y = round(onset_y + r.random() * 20 + 5)
        dx_delay_m = round(r.gauss(24, 18))  # months to diagnosis
        if dx_delay_m < 0:
            dx_delay_m = 2

        flags = {}

        if gene == "RPGR":
            flags["orf15_variant"] = r.random() < 0.48
            flags["standard_wes_missed"] = flags["orf15_variant"] and r.random() < 0.58
            flags["carrier_female_symptomatic"] = (sex == "F" and r.random() < 0.22)
            flags["areds2_prescribed"] = r.random() < 0.54
            flags["uv_protection_counselled"] = r.random() < 0.68
            flags["annual_erg"] = r.random() < 0.72
            flags["trial_eligible_referred"] = r.random() < 0.28
            flags["gene_therapy_enrolment"] = r.random() < 0.09
        elif gene == "USH2A":
            flags["snhl_congenital"] = r.random() < 0.92
            flags["ci_received"] = r.random() < 0.38
            flags["joint_ent_ophtho_clinic"] = r.random() < 0.52
            flags["c2299delG_variant"] = r.random() < 0.32
            flags["vit_a_prescribed"] = r.random() < 0.54
            flags["orientation_mobility"] = r.random() < 0.44
            flags["annual_audiogram"] = r.random() < 0.78
            flags["annual_erg"] = r.random() < 0.70
        elif gene == "ABCA4":
            flags["stargardt_phenotype"] = r.random() < 0.64
            flags["cord_phenotype"] = r.random() < 0.18
            flags["panretinal_rp_phenotype"] = r.random() < 0.12
            flags["faf_performed"] = r.random() < 0.72
            flags["faf_classic_pattern"] = flags["faf_performed"] and r.random() < 0.78
            flags["light_restriction_counselled"] = r.random() < 0.62
            flags["vit_a_avoided"] = r.random() < 0.48
            flags["n1868i_allele"] = r.random() < 0.22
            flags["trial_referred"] = r.random() < 0.24
        elif gene == "RHO":
            flags["p23h_variant"] = r.random() < 0.32
            flags["class_ii_misfolding"] = r.random() < 0.60
            flags["class_i_surface_expressed"] = not flags["class_ii_misfolding"]
            flags["vit_a_prescribed"] = r.random() < 0.56
            flags["beta_carotene_avoided_smoker"] = r.random() < 0.72
            flags["annual_erg"] = r.random() < 0.74
            flags["gene_therapy_eligible"] = r.random() < 0.32
        elif gene == "PRPF31":
            flags["reduced_penetrance_counselled"] = r.random() < 0.58
            flags["unaffected_carrier_identified_in_family"] = r.random() < 0.45
            flags["cnot3_testing_offered"] = r.random() < 0.14
            flags["vit_a_prescribed"] = r.random() < 0.50
            flags["annual_erg"] = r.random() < 0.70
            flags["predictive_testing_offered_to_relatives"] = r.random() < 0.62
        elif gene == "CRB1":
            flags["lca8_phenotype"] = r.random() < 0.35
            flags["rp12_phenotype"] = r.random() < 0.40
            flags["pprpe_sign_present"] = r.random() < 0.72
            flags["pprpe_noted_on_fundoscopy"] = flags["pprpe_sign_present"] and r.random() < 0.58
            flags["thick_retina_oct"] = r.random() < 0.62
            flags["luxturna_erroneously_referred"] = r.random() < 0.12
            flags["coats_like_complication"] = r.random() < 0.06
            flags["anti_vegf_given"] = flags["coats_like_complication"] and r.random() < 0.78
            flags["franceschetti_sign"] = flags["lca8_phenotype"] and r.random() < 0.58
        elif gene == "CNGB3":
            flags["complete_achromatopsia"] = r.random() < 0.92
            flags["t383fsX_variant"] = r.random() < 0.45
            flags["fl41_lenses_prescribed"] = r.random() < 0.68
            flags["scotopic_erg_normal"] = r.random() < 0.94
            flags["photopic_erg_absent"] = r.random() < 0.92
            flags["gene_therapy_trial_eligible"] = r.random() < 0.52
            flags["gene_therapy_enrolled"] = r.random() < 0.14
            flags["patient_counselled_stable"] = r.random() < 0.76
        elif gene == "BEST1":
            flags["eog_performed"] = r.random() < 0.72
            flags["eog_arden_ratio_abnormal"] = flags["eog_performed"] and r.random() < 0.88
            flags["erg_normal"] = r.random() < 0.90
            flags["cnv_complication"] = r.random() < 0.12
            flags["anti_vegf_prescribed"] = flags["cnv_complication"] and r.random() < 0.84
            flags["ar_bestrophinopathy"] = r.random() < 0.12
            flags["stage_vitelliform"] = r.random() < 0.38
            flags["stage_scrambled_egg"] = r.random() < 0.22
            flags["stage_atrophic"] = r.random() < 0.20
            flags["carrier_relatives_screened"] = r.random() < 0.54

        pts.append({
            "pid": f"{gene}-{i+1:03d}",
            "gene": gene,
            "etiology": etiol,
            "sex": sex,
            "age_onset_years": onset_y,
            "age_onset_months": onset_m,
            "age_current_years": age_current_y,
            "dx_delay_months": dx_delay_m,
            **flags,
        })
    return pts


def get_overview():
    all_patients = []
    gene_summaries = []

    for gd in RETINAL_GENES:
        pts = _make_cohort(gd)
        all_patients.extend(pts)

        gene_summaries.append({
            "gene": gd["gene"],
            "protein": gd["protein"],
            "aa": gd["aa"],
            "kDa": gd["kDa"],
            "locus": gd["locus"],
            "omim_gene": gd["omim_gene"],
            "omim_disease": gd["omim_disease"],
            "inheritance": gd["inheritance"],
            "n_patients": gd["n_patients"],
            "seed": gd["seed"],
            "etiologies": [e[0] for e in gd["etiologies"]],
            "key_alerts": gd["key_alerts"],
            "alias": gd["alias"],
            "gene_class": gd["gene_class"],
        })

    n = len(all_patients)

    def g_pts(gene):
        return [p for p in all_patients if p["gene"] == gene]

    def pct(lst, key, val=True):
        if not lst:
            return 0.0
        return round(100 * sum(1 for p in lst if p.get(key) == val) / len(lst), 1)

    rpgr = g_pts("RPGR")
    usha = g_pts("USH2A")
    abca = g_pts("ABCA4")
    rho = g_pts("RHO")
    prpf = g_pts("PRPF31")
    crb1 = g_pts("CRB1")
    cngb = g_pts("CNGB3")
    best = g_pts("BEST1")

    # Mean dx delay across all
    mean_delay = round(sum(p["dx_delay_months"] for p in all_patients) / n, 1)

    agg = {
        "total_patients": n,
        "total_genes": 8,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "mean_dx_delay_months": mean_delay,
        # RPGR
        "rpgr_orf15_variant_pct": pct(rpgr, "orf15_variant"),
        "rpgr_standard_wes_missed_pct": pct(rpgr, "standard_wes_missed"),
        "rpgr_carrier_female_symptomatic_pct": pct([p for p in rpgr if p["sex"] == "F"], "carrier_female_symptomatic"),
        "rpgr_uv_protection_counselled_pct": pct(rpgr, "uv_protection_counselled"),
        "rpgr_trial_referred_pct": pct(rpgr, "trial_eligible_referred"),
        # USH2A
        "usha_snhl_congenital_pct": pct(usha, "snhl_congenital"),
        "usha_ci_received_pct": pct(usha, "ci_received"),
        "usha_joint_clinic_pct": pct(usha, "joint_ent_ophtho_clinic"),
        "usha_c2299delG_pct": pct(usha, "c2299delG_variant"),
        "usha_vit_a_pct": pct(usha, "vit_a_prescribed"),
        # ABCA4
        "abca4_stargardt_pct": pct(abca, "stargardt_phenotype"),
        "abca4_faf_performed_pct": pct(abca, "faf_performed"),
        "abca4_faf_classic_pct": pct(abca, "faf_classic_pattern"),
        "abca4_light_restricted_pct": pct(abca, "light_restriction_counselled"),
        "abca4_vit_a_avoided_pct": pct(abca, "vit_a_avoided"),
        "abca4_n1868i_pct": pct(abca, "n1868i_allele"),
        # RHO
        "rho_p23h_pct": pct(rho, "p23h_variant"),
        "rho_class_ii_pct": pct(rho, "class_ii_misfolding"),
        "rho_vit_a_pct": pct(rho, "vit_a_prescribed"),
        # PRPF31
        "prpf31_penetrance_counselled_pct": pct(prpf, "reduced_penetrance_counselled"),
        "prpf31_unaffected_carrier_family_pct": pct(prpf, "unaffected_carrier_identified_in_family"),
        "prpf31_cnot3_offered_pct": pct(prpf, "cnot3_testing_offered"),
        # CRB1
        "crb1_pprpe_present_pct": pct(crb1, "pprpe_sign_present"),
        "crb1_pprpe_noted_pct": pct(crb1, "pprpe_noted_on_fundoscopy"),
        "crb1_thick_oct_pct": pct(crb1, "thick_retina_oct"),
        "crb1_luxturna_erroneously_referred_pct": pct(crb1, "luxturna_erroneously_referred"),
        # CNGB3
        "cngb3_complete_pct": pct(cngb, "complete_achromatopsia"),
        "cngb3_fl41_pct": pct(cngb, "fl41_lenses_prescribed"),
        "cngb3_scotopic_normal_pct": pct(cngb, "scotopic_erg_normal"),
        "cngb3_gene_therapy_eligible_pct": pct(cngb, "gene_therapy_trial_eligible"),
        "cngb3_counselled_stable_pct": pct(cngb, "patient_counselled_stable"),
        # BEST1
        "best1_eog_performed_pct": pct(best, "eog_performed"),
        "best1_eog_abnormal_pct": pct(best, "eog_arden_ratio_abnormal"),
        "best1_erg_normal_pct": pct(best, "erg_normal"),
        "best1_cnv_pct": pct(best, "cnv_complication"),
        "best1_anti_vegf_pct": pct(best, "anti_vegf_prescribed"),
        # Cross-gene
        "any_gene_therapy_referred_pct": round(100 * sum(
            1 for p in all_patients if (
                p.get("trial_eligible_referred") or p.get("gene_therapy_trial_eligible")
                or p.get("trial_referred") or p.get("gene_therapy_eligible")
            )
        ) / n, 1),
        "any_annual_erg_pct": round(100 * sum(
            1 for p in all_patients if p.get("annual_erg")
        ) / n, 1),
        "any_vit_a_prescribed_pct": round(100 * sum(
            1 for p in all_patients if p.get("vit_a_prescribed")
        ) / n, 1),
    }

    all_alerts = []
    for gd in RETINAL_GENES:
        all_alerts.extend(gd["key_alerts"])

    return {
        "title": "Hereditary-Retinal-Dystrophy-Atlas — Complete 8-Gene Inherited Retinal Disease Reference",
        "subtitle": (
            "RPGR · USH2A · ABCA4 · RHO · PRPF31 · CRB1 · CNGB3 · BEST1 — "
            "320 patients (8×40, seeds 1494–1501) — X-linked RP, Usher syndrome, Stargardt, "
            "ADRP, reduced-penetrance RP11, LCA8/RP12, Achromatopsia, Best VMD2"
        ),
        "genes": gene_summaries,
        "aggregate_stats": agg,
        "top_alerts": all_alerts,
    }


def get_breakdown():
    breakdown = []
    for gd in RETINAL_GENES:
        pts = _make_cohort(gd)
        sex_dist = {"M": sum(1 for p in pts if p["sex"] == "M"),
                    "F": sum(1 for p in pts if p["sex"] == "F")}
        mean_onset = round(sum(p["age_onset_years"] for p in pts) / len(pts), 1)
        mean_delay = round(sum(p["dx_delay_months"] for p in pts) / len(pts), 1)
        etiol_counts = {}
        for p in pts:
            etiol_counts[p["etiology"]] = etiol_counts.get(p["etiology"], 0) + 1

        breakdown.append({
            "gene": gd["gene"],
            "protein": gd["protein"],
            "aa": gd["aa"],
            "kDa": gd["kDa"],
            "locus": gd["locus"],
            "omim_gene": gd["omim_gene"],
            "omim_disease": gd["omim_disease"],
            "inheritance": gd["inheritance"],
            "n_patients": gd["n_patients"],
            "seed": gd["seed"],
            "mean_onset_years": mean_onset,
            "mean_dx_delay_months": mean_delay,
            "sex_distribution": sex_dist,
            "etiology_counts": etiol_counts,
            "key_alerts": gd["key_alerts"],
            "alias": gd["alias"],
            "gene_class": gd["gene_class"],
            "patients": pts,
        })
    return {"breakdown": breakdown}


def get_definitions():
    return {
        "definitions": [
            {
                "term": "RPGR ORF15 — Why Standard WES Misses Up to 50% of XLRP",
                "definition": (
                    "The RPGR ORF15 exon is a purine-rich, repetitive sequence (poly-GA / poly-A "
                    "stretches) that is highly susceptible to PCR slippage during amplification. "
                    "Short-read WES platforms generate artefactual insertions and deletions over this "
                    "region, making it impossible to reliably call true pathogenic frameshift variants. "
                    "The result: up to 50% of affected males with RPGR-XLRP have their causal variant "
                    "missed by standard WES or gene panels. "
                    "Solution: specifically request ORF15 sequencing using a validated Sanger or "
                    "long-read protocol (PacBio, Oxford Nanopore) alongside WES. "
                    "Every male with X-linked RP (family history or hemizygous pattern) who is "
                    "negative on standard WES must have ORF15 assessed by alternative method. "
                    "This is a systemic diagnostic gap — failure to apply this protocol means the "
                    "most common X-linked RP gene is systematically underdiagnosed."
                ),
            },
            {
                "term": "USH2A vs Usher 1 — Key Clinical Distinguishing Features",
                "definition": (
                    "Usher syndrome has three types with critical clinical distinctions: "
                    "Usher type 1 (MYO7A, CDH23, PCDH15, SANS): profound congenital deafness, "
                    "vestibular areflexia (no caloric response), RP onset in childhood. "
                    "Usher type 2 (USH2A most common): moderate-severe congenital SNHL — NOT profound; "
                    "NO vestibular dysfunction (caloric response NORMAL); RP onset teens-20s. "
                    "Usher type 3 (CLRN1): progressive SNHL (not congenital), variable vestibular. "
                    "Distinguishing Usher 1 from Usher 2 matters: Usher 1 requires vestibular "
                    "physiotherapy and cochlear implant evaluation earlier; sign language may be "
                    "the preferred communication modality. Caloric testing or VEMP is mandatory "
                    "in any Usher patient to classify type before genetic confirmation."
                ),
            },
            {
                "term": "ABCA4 — Why Vitamin A is Contraindicated (Unlike USH2A and RHO)",
                "definition": (
                    "In Stargardt disease (ABCA4), high-dose Vitamin A supplementation is "
                    "CONTRAINDICATED — the opposite of its recommended use in USH2A-RP and RHO-RP. "
                    "Rationale: increased dietary vitamin A → increased production of all-trans-retinal "
                    "after photoactivation → impaired clearance by defective ABCA4 → increased "
                    "A2E and bisretinoid accumulation → accelerated RPE toxicity and geographic atrophy. "
                    "The therapeutic logic for RHO and USH2A (Berson data): high serum retinol "
                    "provides adequate 11-cis retinal chromophore, supporting residual rhodopsin "
                    "function in partially degenerated rods. This mechanism does NOT apply in ABCA4 "
                    "where the problem is bisretinoid over-accumulation, not chromophore shortage. "
                    "Similarly, excessive light (even visible light) accelerates A2E formation "
                    "in ABCA4 — UV400 filter lenses are mandatory at diagnosis."
                ),
            },
            {
                "term": "RHO P23H — Class I vs Class II Rhodopsin Mutations",
                "definition": (
                    "Rhodopsin mutations causing AD RP are classified by biochemical mechanism: "
                    "Class I variants produce surface-expressed rhodopsin with normal cellular "
                    "trafficking but abnormal photocycle kinetics (delayed chromophore release, "
                    "constitutive activation). Clinical: milder, slower progression. "
                    "Class II variants produce misfolded opsin retained in the endoplasmic reticulum; "
                    "the unfolded protein response (UPR) is activated; chaperones (BiP/HSP70) are "
                    "sequestered; dominant-negative toxicity kills rods rapidly. Clinical: severe, "
                    "fast progression. P23H (most common North American ADRP variant) is Class II. "
                    "Gene therapy for RHO-ADRP must address the dominant-negative problem — "
                    "knockdown of BOTH alleles plus delivery of a codon-modified WT RHO is required. "
                    "Replacing only without knockdown leaves the toxic Class II protein in place."
                ),
            },
            {
                "term": "PRPF31 Reduced Penetrance — CNOT3 Modifier Gene",
                "definition": (
                    "RP11 (PRPF31) is the paradigm example of modifier-controlled penetrance in "
                    "retinal dystrophy. Pathogenic PRPF31 variants cause haploinsufficiency — "
                    "whether this haploinsufficiency produces retinal disease depends on the "
                    "expression level of the wild-type PRPF31 allele. "
                    "The CNOT3 gene (19p13.3, adjacent to PRPF31) encodes the CCR4-NOT transcription "
                    "factor subunit 3, which trans-activates PRPF31 transcription. "
                    "Individuals who inherit a high-expression PRPF31 haplotype (high CNOT3 response) "
                    "produce sufficient WT PRPF31 to compensate for haploinsufficiency → no RP. "
                    "Family members with a low-expression WT haplotype → insufficient PRPF31 → RP. "
                    "Genetic counselling implication: a sibling of an affected PRPF31 patient who "
                    "is confirmed to carry the same pathogenic variant may never develop RP — "
                    "this cannot be predicted from the variant alone without haplotype analysis. "
                    "Penetrance is 60-80% across reported families."
                ),
            },
            {
                "term": "CRB1 — Para-Arteriolar Preservation of the RPE (PPRPE) Sign",
                "definition": (
                    "Para-arteriolar preservation of the RPE (PPRPE) is a pathognomonic fundoscopic "
                    "finding in CRB1-associated retinal dystrophy. On dilated fundoscopy, zones of "
                    "intact RPE (appearing lighter, with preserved pigment pattern) are visible "
                    "as stripes flanking major retinal arterioles, set against a background of "
                    "diffuse pigment epithelium atrophy and bone-spicule pigmentation. "
                    "Mechanistic explanation: retinal arterioles provide mechanical support to the "
                    "photoreceptor-RPE complex through their structural contact with Müller glia; "
                    "in CRB1-deficient retina where the outer limiting membrane integrity is lost, "
                    "the arteriolar zones retain local structural support, preventing degeneration "
                    "of the overlying RPE while the non-supported regions atrophy. "
                    "The sign is sufficient to suggest CRB1 testing even without genetic results. "
                    "Do NOT confuse CRB1-LCA8 with RPE65-LCA2 — voretigene neparvovec treats "
                    "RPE65 disease only; CRB1 patients are ineligible and must not be referred for Luxturna."
                ),
            },
            {
                "term": "CNGB3 Achromatopsia — Stability Guarantee and Counselling Imperative",
                "definition": (
                    "Achromatopsia (CNGB3, CNGA3) is a stationary cone dystrophy — a critical "
                    "counselling point that must be communicated clearly to every patient and family. "
                    "Because rods are genetically and functionally normal, patients retain: "
                    "(1) fully normal night vision; (2) full peripheral visual fields; "
                    "(3) intact motion/contrast detection; (4) scotopic ERG is completely normal. "
                    "The condition does NOT progress to blindness. Patients misled into believing "
                    "they will go blind suffer unnecessary psychological harm. "
                    "The disabling symptoms — profound photophobia and visual acuity ~0.1 (20/200) — "
                    "are addressable: FL-41 rose/amber tinted lenses (OD 4 filter) dramatically "
                    "reduce photophobia; dark-tinted wraparound sunglasses outdoors; "
                    "brimmed hats; occupational adaptation to avoid high-luminance environments. "
                    "AGTC/Beacon Therapeutics Phase 2/3 gene therapy trials for CNGB3/CNGA3 are "
                    "the most advanced cone-dystrophy trials — enrol eligible patients promptly."
                ),
            },
            {
                "term": "BEST1 EOG — Why ERG is Normal and EOG is the Diagnostic Test",
                "definition": (
                    "Best vitelliform macular dystrophy is a unique retinal dystrophy where the "
                    "electroretinogram (ERG) is COMPLETELY NORMAL while the electro-oculogram (EOG) "
                    "is invariably abnormal. This distinction is mechanistically explained: "
                    "ERG measures mass photoreceptor electrical responses — rod and cone cells are "
                    "structurally and functionally intact in Best disease (the primary defect is "
                    "in RPE bestrophin chloride channel, not photoreceptors). "
                    "The EOG measures the standing potential of the eye — generated by the RPE "
                    "basolateral membrane — and its light-induced rise (light peak) depends on "
                    "RPE chloride conductance via bestrophin-1. In BEST1 disease, the mutant "
                    "channel fails to mediate the light-induced Cl- conductance increase → "
                    "subnormal Arden ratio (LP:dark trough typically <1.5 vs normal >1.85). "
                    "Critically, the EOG is abnormal even in UNAFFECTED carriers — making EOG "
                    "the most sensitive carrier-detection tool. All first-degree relatives of "
                    "a Best VMD2 patient should have EOG testing."
                ),
            },
            {
                "term": "Vitamin A Supplementation — Gene-Specific Rules in Retinal Dystrophy",
                "definition": (
                    "Vitamin A supplementation rules vary critically by genotype in inherited retinal disease: "
                    "RECOMMEND (15,000 IU/day): USH2A-RP, RHO-ADRP, and most non-ABCA4 RP — "
                    "Berson 1993 trial showed slowing of ERG decline (~20%/year slower); "
                    "monitor LFTs annually; AVOID in pregnancy (teratogenicity). "
                    "CONTRAINDICATE: ABCA4 (Stargardt) — accelerates bisretinoid/A2E accumulation. "
                    "AVOID beta-carotene in any current or ex-smoker taking AREDS2 — "
                    "CARET trial and ATBC trial showed significantly increased lung cancer risk with "
                    "high-dose beta-carotene in smokers; use lutein/zeaxanthin instead. "
                    "UNCERTAIN (do not recommend without specialist guidance): mitochondrial retinopathies, "
                    "choroideremia, gyrate atrophy (ornithine restriction is key in gyrate atrophy). "
                    "Never issue a blanket 'take Vitamin A' recommendation in retinal clinic "
                    "without verifying the genetic diagnosis."
                ),
            },
            {
                "term": "Gene Therapy Landscape in Inherited Retinal Disease (2024)",
                "definition": (
                    "Approved (FDA/EMA 2024): voretigene neparvovec (Luxturna) — RPE65-LCA2/RP20. "
                    "Advanced trials (Phase 2/3): CNGB3 achromatopsia (AGTC/Beacon NCT02599922); "
                    "RPGR-XLRP (AGTC, Beacon, MeiraGTx — multiple trials); sepofarsen ASO "
                    "(ProQR/Arcus) for CEP290-LCA10. "
                    "Phase 1/2: RHO-ADRP (knockdown-and-replace), CRB1 (dual AAV — size constraint); "
                    "ABCA4 (lentiviral vector approach — size constraint for single AAV). "
                    "Critical principle: gene MUST be confirmed genetically before ANY gene therapy "
                    "referral — voretigene is RPE65 ONLY; CNGB3 patients ineligible; CRB1 patients "
                    "ineligible for Luxturna. Misreferral wastes trial slots and delays correct "
                    "enrolment. Maintain a live list of eligible patients mapped to active trials."
                ),
            },
            {
                "term": "Cascade Genetic Testing in Hereditary Retinal Dystrophy",
                "definition": (
                    "Cascade testing strategy by inheritance pattern: "
                    "X-linked (RPGR): maternal relatives — carrier testing for sisters of affected males; "
                    "female carriers should have ophthalmological assessment (20-25% symptomatic); "
                    "sons of carrier females: 50% risk — ophthalmology from age 5-7 (ERG). "
                    "Autosomal recessive (USH2A, ABCA4, CRB1, CNGB3): carrier testing for parents "
                    "and siblings; recurrence risk 25% per pregnancy; prenatal/preimplantation "
                    "genetic testing available; siblings of affected should have genetic testing. "
                    "Autosomal dominant (RHO, PRPF31, BEST1): first-degree relatives 50% risk; "
                    "PRPF31: explain reduced penetrance before testing (positive result does NOT "
                    "guarantee disease development); BEST1: EOG testing of relatives alongside "
                    "genetic testing — EOG abnormal even pre-symptomatically in carriers."
                ),
            },
        ],
    }


if __name__ == "__main__":
    import json
    ov = get_overview()
    print(json.dumps(ov["aggregate_stats"], indent=2))
    print(f"\nTop alerts ({len(ov['top_alerts'])}):")
    for a in ov["top_alerts"]:
        print(f"  • {a}")
