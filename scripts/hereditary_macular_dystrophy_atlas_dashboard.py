#!/usr/bin/env python3
"""Hereditary-Macular-Dystrophy-Atlas — Complete 8-Gene Atlas
(ABCA4 · BEST1 · PRPH2 · TIMP3 · ELOVL4 · C1QTNF5 · PRDM13 · EFEMP1).

ABCA4   (ATP-binding cassette transporter A4; 2273 aa; ~250 kDa; 1p22.1; AR;
          Stargardt disease type 1 (STGD1) — MOST COMMON HEREDITARY MACULAR DYSTROPHY;
          BULL'S EYE MACULOPATHY + PISCIFORM FLECKS PATHOGNOMONIC;
          fundus flavimaculatus variant; FAF dark choroid sign;
          seed SEED_BASE+0).
BEST1   (Bestrophin-1; 585 aa; ~68 kDa; 11q12.3; AD also biallelic AR;
          Best vitelliform macular dystrophy (BVMD) — EGG-YOLK LESION (Vitelliform stage);
          EOG ARDEN RATIO <1.5 PATHOGNOMONIC (retinal pigment epithelium dysfunction);
          5-stage evolution: previtelliform → vitelliform → pseudohypopyon → vitelliruptive → atrophic;
          seed SEED_BASE+1).
PRPH2   (Peripherin-2 / RDS; 346 aa; ~39 kDa; 6p21.1; AD;
          Pattern dystrophy — BUTTERFLY-SHAPED PIGMENT DYSTROPHY + AVMD + CENTRAL AREOLAR;
          highly variable expressivity within same family;
          p.Arg172Trp most common variant; slow progression; anti-VEGF if CNV;
          seed SEED_BASE+2).
TIMP3   (Tissue inhibitor of metalloproteinase 3; 211 aa; ~24 kDa; 22q12.3; AD;
          Sorsby fundus dystrophy (SFD) — BILATERAL CHOROIDAL NEOVASCULARIZATION HAEMORRHAGE;
          ACUTE VISUAL LOSS IN 4TH DECADE PATHOGNOMONIC; Bruch's membrane deposits;
          anti-VEGF responsive; dominant-negative TIMP3 prevents ECM turnover;
          seed SEED_BASE+3).
ELOVL4  (Elongation of very-long-chain fatty acids protein 4; 314 aa; ~34 kDa; 6q14.1; AD;
          Stargardt-like macular dystrophy type 3 (STGD3); yellow flecks + foveal atrophy;
          HOMOZYGOUS LETHAL EQUIVALENT: severe ichthyosis + seizures + profound ID (ARCI);
          heterozygous AD macular dystrophy; very-long-chain PUFAs essential for photoreceptors;
          seed SEED_BASE+4).
C1QTNF5 (C1q and TNF related 5 / CTRP5; 243 aa; ~27 kDa; 11q23.3; AD;
          Late-onset retinal degeneration (LORD/L-ORD) — CRYSTALLINE IRIS DEPOSITS + DRUSEN PATHOGNOMONIC;
          CRYSTALLINE IRIS DEPOSITS UNIQUE TO LORD — differentiates from AMD;
          drusenoid deposits subretinal; CNV in late stage; onset 6th decade;
          seed SEED_BASE+5).
PRDM13  (PR/SET domain 13; 718 aa; ~81 kDa; 6q16.1; AD;
          North Carolina Macular Dystrophy (NCMD/MCDR1/MCDR2) — NON-PROGRESSIVE CENTRAL MACULAR LESION;
          GRADE 0-3 SCALE (0=drusen only, 3=chorioretinal coloboma-like staphyloma);
          STATIONARY from birth — KEY DDx progressive dystrophies;
          regulatory variant in PRDM13 promoter (5'UTR) — standard coding exome may miss;
          seed SEED_BASE+6).
EFEMP1  (EGF-containing fibulin extracellular matrix protein 1; 493 aa; ~55 kDa; 2p16.1; AD;
          Doyne honeycomb retinal dystrophy (DHRD) / Malattia Leventinese (ML) — RADIAL DRUSEN HONEYCOMB;
          p.Arg345Trp (R345W) SINGLE PATHOGENIC VARIANT IN VAST MAJORITY;
          drusen nasal to disc + macula by age 20-30; progression to CNV; FBN3-like domain;
          seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 2406-2413).
"""

import random

SEED_BASE = 2406

MD_GENES = [
    # -- ABCA4 -- Stargardt Disease 1 ---------------------------------------------------------
    {
        "gene": "ABCA4",
        "alt_name": (
            "ABCA4 (ABCA4-2273aa-1p22.1 / AR -- "
            "STARGARDT-DISEASE-1-STGD1-MOST-COMMON-HEREDITARY-MACULAR-DYSTROPHY -- "
            "BULL'S-EYE-MACULOPATHY-PISCIFORM-FLECKS-PATHOGNOMONIC -- "
            "FAF-DARK-CHOROID-SIGN-ABSENT-AUTOFLUORESCENCE-FLECKS -- "
            "VITAMIN-A-SUPPLEMENTATION-ABSOLUTELY-CI-ACCELERATES-A2E-ACCUMULATION)"
        ),
        "protein": (
            "ABCA4 -- 1p22.1 AR -- ABCA4-2273aa -- "
            "ATP-Binding-Cassette-Transporter-A4-250kDa-Photoreceptor-Outer-Segment -- "
            "Flips-N-Retinylidene-PE-and-All-Trans-Retinal-From-Disc-Lumen-to-Cytoplasm -- "
            "Prevents-A2E-Toxic-Bisretinoid-Accumulation-in-RPE -- "
            "Expressed-Exclusively-Photoreceptor-Outer-Segments-Rod-Cone -- "
            "OMIM-Gene-601691-Disease-STGD1-248200"
        ),
        "locus": "1p22.1",
        "protein_size": "2273 aa / 250 kDa",
        "inheritance": (
            "AR — biallelic pathogenic variants required; "
            "ABCA4 transports N-retinylidene-phosphatidylethanolamine (N-ret-PE) and all-trans-retinal from "
            "the disc lumen to the cytoplasmic leaflet of photoreceptor disc membranes; "
            "loss of function → N-ret-PE + all-trans-retinal accumulate in disc lumen → "
            "react to form A2PE → hydrolysed to A2E (toxic bisretinoid) by RPE lysosomes → "
            "RPE cell death → photoreceptor degeneration → macular atrophy; "
            "ABCA4 is the largest known retinal dystrophy gene (21 exons encoding 2273 aa); "
            "pathogenic variants: missense (most common), null (frameshift/nonsense/splice); "
            "common severe variants: p.Leu541Pro, p.Gly1961Glu, p.Ala1038Val, c.5461-10T>C (deep intronic); "
            "allelic heterogeneity >1000 pathogenic variants; "
            "phenotype-genotype: two severe alleles → early childhood onset; "
            "one severe + one mild allele → onset 2nd-3rd decade; "
            "two mild alleles → adult fundus flavimaculatus; "
            "no confirmed homozygous ABCA4 LOF in humans — partial function required"
        ),
        "disease_category": "Stargardt disease type 1 (STGD1) — most common hereditary macular dystrophy; AR; bull's eye maculopathy + pisciform flecks",
        "disease_pathway": (
            "ABCA4 (ATP-binding cassette transporter subfamily A member 4) is the sole retinal-specific ABC transporter, "
            "localised exclusively to the rim region of photoreceptor disc membranes (rod and cone outer segments). "
            "VISUAL CYCLE LINK: in the retinoid visual cycle, bleached all-trans-retinal is released from opsin into the disc lumen; "
            "ABCA4 flips all-trans-retinal (and its Schiff-base adduct N-retinylidene-PE) from the intradisc leaflet "
            "to the cytoplasmic leaflet, where retinal reductase converts it to all-trans-retinol for recycling. "
            "STARGARDT MECHANISM: ABCA4 LOF → all-trans-retinal + N-ret-PE accumulate in disc lumen → "
            "dimerisation → A2PE (precursor) → RPE phagocytosis of disc shedding → "
            "A2PE hydrolysis → A2E (N-retinylidene-N-retinylethanolamine), a fluorescent bisretinoid toxin → "
            "A2E disrupts lysosomal function in RPE → RPE apoptosis → photoreceptor loss → macular atrophy. "
            "FAF DARK CHOROID: A2E autofluorescence is high in flecks but A2E blocks normal autofluorescence in the macula; "
            "fundus autofluorescence (FAF): hyperfluorescent pisciform flecks + dark (hypofluorescent) central macula; "
            "fluorescein angiography: 'dark choroid' (retinal pigment masking; blocks background choroidal fluorescence) in ~80%. "
            "VITAMIN A CI: exogenous retinol supplementation accelerates visual cycle flux → more A2E → "
            "accelerates disease progression — VITAMIN A ABSOLUTELY CONTRAINDICATED in STGD1."
        ),
        "pathognomonic": (
            "BULL'S EYE MACULOPATHY (ring of RPE atrophy around preserved central fovea, then central involvement) + "
            "PISCIFORM (FISH-SHAPED) YELLOW-WHITE FLECKS AT THE LEVEL OF RPE (extending from macula toward mid-periphery) = "
            "STARGARDT PATHOGNOMONIC COMPLEX. "
            "DARK CHOROID SIGN on fluorescein angiography (~80%) — masking of choroidal background fluorescence by A2E; "
            "FUNDUS AUTOFLUORESCENCE: pisciform flecks are HYPERFLUORESCENT (A2E fluorescence); "
            "central macula/atrophy zones HYPOFLUORESCENT (RPE loss). "
            "OCT: outer nuclear layer loss at fovea; IS/OS (ellipsoid zone) disruption; "
            "subfoveal hyper-reflective material (A2E deposits in RPE); "
            "atrophic patches progressing centrifugally over years-decades. "
            "ELECTRORETINOGRAM: full-field ERG initially NORMAL (distinguishes from rod-cone dystrophy); "
            "macular ERG (multifocal ERG) reduced early; "
            "FFERG reduces late when flecks extend beyond 45° (fundus flavimaculatus). "
            "VITAMIN A SUPPLEMENTATION ABSOLUTELY CONTRAINDICATED — accelerates A2E accumulation. "
            "Genetic confirmation: biallelic ABCA4 pathogenic variants (include deep intronic: c.5461-10T>C, c.769-784C>T)."
        ),
        "treatment": (
            "NO PROVEN DISEASE-MODIFYING THERAPY (2026): "
            "VITAMIN A ABSOLUTE CI: never prescribe vitamin A/retinol supplements — accelerates A2E; "
            "SUNGLASSES (UV/blue-light filtering): reduces light-driven visual cycle flux → less A2E formation; "
            "SMOKING CESSATION: oxidative stress accelerates RPE degeneration; "
            "OPTICAL AIDS: magnifiers, dark adaptation aids, eccentric viewing training; "
            "LOW VISION: CCTV, contrast enhancement; "
            "TRIALS (emerging): "
            "Emixustat (visual cycle modulator, ACT-001): RPE65 inhibitor — slows visual cycle, reduces A2E; "
            "Fenretinide (retinol competitor) — reduces retinol delivery to retina; "
            "Gene therapy (AAV2/5-ABCA4): in early trials; ABCA4 large size challenges single-AAV; "
            "Lentiviral vector (large ABCA4); "
            "Stem cell (RPE transplant) trials; "
            "ANNUAL REVIEW: OCT + FAF + visual field (Humphrey 10-2); "
            "DRIVING: most patients legally blind within 20 years (Snellen <6/60); "
            "GENETICS COUNSELLING: 25% recurrence risk siblings; "
            "FAMILY SCREEN: carrier parents → heterozygous siblings (consider cascade if >1 mutation present)."
        ),
        "key_features": [
            "Bull's eye maculopathy — ring RPE atrophy, hallmark STGD1",
            "Pisciform (fish-shaped) yellow-white flecks at RPE level",
            "Dark choroid sign on FFA (~80%) — A2E masking background choroid",
            "FAF: flecks hyperfluorescent; atrophy/macula hypofluorescent",
            "Full-field ERG initially NORMAL — distinguishes from retinitis pigmentosa",
            "Vitamin A supplements ABSOLUTELY CONTRAINDICATED (accelerates A2E)",
            "Onset typically childhood–early adulthood; legal blindness by 3rd–4th decade",
            "Most common hereditary macular dystrophy worldwide",
        ],
        "key_ddx": (
            "AMD (age-related): onset >60y; drusen + CNV; ABCA4 not causative; "
            "BEST1 (BVMD): egg-yolk lesion; EOG Arden ratio pathognomonic; AD; "
            "PRPH2 (pattern dystrophy): butterfly pigment; AD; slow; ERG normal; "
            "ELOVL4 (STGD3): AD inheritance; pedigree key; similar flecks but AD; "
            "Cone dystrophy: ERG cone amplitudes reduced early (vs STGD full-field ERG normal early); "
            "Chloroquine/HCQ maculopathy: drug history; bull's eye; ABCA4 negative; "
            "MEWDS / acute macular neuroretinopathy: acute; self-limited; young females."
        ),
        "systemic_involvement": False,
        "onset_age": "Childhood–early adulthood (most in 2nd–3rd decade); fundus flavimaculatus variant onset 4th–6th decade",
        "surgical_urgency": "No acute surgical urgency; anti-VEGF if CNV develops",
        "gene_family": "ABC transporter (ATP-binding cassette, subfamily A) — retinal-specific",
        "morphology": "Bull's eye maculopathy + pisciform RPE flecks + dark choroid sign",
    },
    # -- BEST1 -- Best Vitelliform Macular Dystrophy ------------------------------------------
    {
        "gene": "BEST1",
        "alt_name": (
            "BEST1 (BEST1-585aa-11q12.3 / AD also AR -- "
            "BEST-VITELLIFORM-MACULAR-DYSTROPHY-BVMD-EGG-YOLK-LESION-PATHOGNOMONIC -- "
            "EOG-ARDEN-RATIO-<1.5-PATHOGNOMONIC-RPE-CHLORIDE-CHANNEL-DYSFUNCTION -- "
            "5-STAGE-EVOLUTION-PREVITELLIFORM-VITELLIFORM-PSEUDOHYPOPYON-VITELLIRUPTIVE-ATROPHIC -- "
            "BIALLELIC-AR-BEST-DISEASE-AUTOSOMAL-RECESSIVE-BESTROPHINOPATHY-ARB)"
        ),
        "protein": (
            "BEST1 -- 11q12.3 AD-AR -- BEST1-585aa -- "
            "Bestrophin-1-68kDa-Calcium-Activated-Chloride-Channel-RPE-Basolateral-Membrane -- "
            "Chloride-Channel-RPE-Basolateral-Membrane-Regulates-Fluid-Transport -- "
            "Modulates-RPE-Phagocytosis-of-Photoreceptor-Outer-Segments -- "
            "Also-Regulates-cAMP-IP3-Calcium-Signalling-in-RPE -- "
            "OMIM-Gene-607854-Disease-BVMD-153700-ARB-611809"
        ),
        "locus": "11q12.3",
        "protein_size": "585 aa / 68 kDa",
        "inheritance": (
            "AD (autosomal dominant) for classic Best vitelliform macular dystrophy (BVMD); "
            "AR (biallelic) for autosomal recessive bestrophinopathy (ARB) — more severe; "
            "BEST1 is a calcium-activated chloride channel (CaCC) expressed on the basolateral membrane of RPE; "
            "regulates Cl⁻ and fluid transport from subretinal space through RPE to choroid; "
            "AD BVMD: heterozygous missense variants (dominant-negative effect on tetrameric channel); "
            "most common variants: p.Tyr227Asn, p.Arg218Cys, p.Thr6Pro, p.Arg200Gln; "
            "EOG ARDEN RATIO <1.5: light peak/dark trough ratio measured by electrooculography; "
            "pathognomonic even in pre-vitelliform / carrier / asymptomatic stage — "
            "RPE function impaired before visible deposits; "
            "PENETRANCE: near-complete for EOG abnormality; phenotype variable (some have normal vision lifelong); "
            "AR (ARB): more severe bilateral multifocal disease; subretinal fluid; choroidal neovascularisation; "
            "BVMD allelic series: VMD2-related pattern dystrophy, BVMD, adult vitelliform macular dystrophy, "
            "autosomal recessive bestrophinopathy"
        ),
        "disease_category": "Best vitelliform macular dystrophy (BVMD) — AD; egg-yolk macular lesion; EOG Arden ratio <1.5 pathognomonic",
        "disease_pathway": (
            "BEST1 (Bestrophin-1) is the founding member of the bestrophin family of anion channels. "
            "PHYSIOLOGICAL FUNCTION: BEST1 forms a pentameric Ca²⁺-activated Cl⁻ channel (CaCC) on the basolateral RPE membrane; "
            "regulates Cl⁻ efflux from RPE into choroid, driving fluid transport from subretinal space → "
            "maintains photoreceptor outer segment (POS) homeostasis. "
            "BVMD MECHANISM: heterozygous missense variants → dominant-negative suppression of BEST1 channel function → "
            "impaired Cl⁻/fluid transport → subretinal fluid + phagocytosed lipofuscin accumulation in RPE → "
            "VITELLIFORM DEPOSIT (lipofuscin-laden material) accumulates in subretinal space below fovea → "
            "EGG-YOLK APPEARANCE (round, orange-yellow, 0.5–3DD diameter). "
            "5 STAGES (Best's own classification): "
            "1. Pre-vitelliform: subtle RPE mottling; EOG already abnormal; "
            "2. Vitelliform: classic egg-yolk lesion (highly fluorescent on FAF — lipofuscin); "
            "3. Pseudohypopyon: yellow material settles inferiorly (gravity-dependent); "
            "4. Vitelliruptive: scrambled-egg appearance (material absorbing); "
            "5. Atrophic / CNV: RPE atrophy or choroidal neovascularisation. "
            "EOG MECHANISM: BEST1 dysfunction → impaired RPE ionic transport → "
            "reduced light-peak response → EOG Arden ratio < 1.5 (normal ≥ 1.85)."
        ),
        "pathognomonic": (
            "EGG-YOLK (VITELLIFORM) MACULAR LESION — round or oval orange-yellow subretinal deposit 0.5–3DD diameter, "
            "sharply demarcated, located at fovea: PATHOGNOMONIC for Best disease. "
            "EOG ARDEN RATIO <1.5 PATHOGNOMONIC: light peak ÷ dark trough < 1.5 (normal ≥ 1.85); "
            "CRITICAL: EOG is abnormal EVEN IN ASYMPTOMATIC CARRIERS and pre-vitelliform stage — "
            "EOG is the FUNCTIONAL TEST that confirms BEST1 dysfunction regardless of phenotype. "
            "FAF: vitelliform lesion is INTENSELY HYPERFLUORESCENT (lipofuscin); atrophic stage hypofluorescent. "
            "PSEUDOHYPOPYON STAGE: yellow material settles to inferior half of lesion (gravity-dependent) — "
            "changes with patient position (upright → settles inferior; supine → redistributes). "
            "ERG FULL-FIELD: NORMAL in BVMD (distinguishes from rod-cone dystrophies); "
            "macular ERG reduced only at fovea. "
            "5-STAGE EVOLUTION: pre-vitelliform → vitelliform → pseudohypopyon → vitelliruptive → atrophic/CNV. "
            "FAMILY SCREEN: EOG mandatory for first-degree relatives — asymptomatic carriers detectable."
        ),
        "treatment": (
            "NO DISEASE-MODIFYING THERAPY PROVEN: "
            "ANTI-VEGF (intravitreal ranibizumab/aflibercept/bevacizumab): for choroidal neovascularisation (CNV) complication; "
            "MOST BVMD PATIENTS RETAIN GOOD CENTRAL VISION until 5th–6th decade; "
            "MONITORING: annual OCT + FAF + visual field; "
            "EOG SCREENING: all first-degree relatives; "
            "OPTICAL AIDS: magnifiers, low vision aids if atrophic stage; "
            "DRIVING: most patients drive into 6th–7th decade (relatively preserved central vision); "
            "PHOTODYNAMIC THERAPY (PDT): for CNV in classic macular location (adjunct to anti-VEGF); "
            "GENE THERAPY: AAV-BEST1 subretinal injection in early Phase I/II trials; "
            "GENETICS: AD 50% recurrence; ARB 25% recurrence (biallelic); "
            "PROGNOSIS: highly variable — some patients never lose reading vision; "
            "CNV complication: anti-VEGF generally responsive."
        ),
        "key_features": [
            "Egg-yolk (vitelliform) orange-yellow foveal lesion — pathognomonic",
            "EOG Arden ratio <1.5 — pathognomonic, even in asymptomatic carriers/pre-vitelliform",
            "Pseudohypopyon stage — gravity-dependent settling of material (positional)",
            "FAF intensely hyperfluorescent (lipofuscin accumulation)",
            "Full-field ERG NORMAL — distinguishes from rod-cone dystrophy",
            "5-stage disease evolution (pre-vitelliform → atrophic/CNV)",
            "AD (BVMD) / AR (ARB — biallelic, more severe multifocal disease)",
            "Family EOG screening mandatory — EOG detects before symptoms",
        ],
        "key_ddx": (
            "ABCA4 (STGD1): bull's eye + flecks; AR; dark choroid; no egg-yolk; EOG normal; "
            "PRPH2 (adult-onset vitelliform): similar lesion but later onset; EOG may be mildly abnormal; AD; "
            "C1QTNF5 (LORD): iris crystalline deposits + drusen; later onset 6th decade; EOG abnormal; AD; "
            "Solar maculopathy: history of sun/eclipse gazing; acute; photoreceptor injury; no EOG change; "
            "Adult-onset foveomacular vitelliform dystrophy (AFVD): BEST1/PRPH2; later; smaller lesion; "
            "Choroidal neovascularisation: drusen + haemorrhage; OCT angio distinguishes; anti-VEGF responsive; "
            "Macular telangiectasia type 2: bilateral parafoveal; OCT specific features; different genetics."
        ),
        "systemic_involvement": False,
        "onset_age": "Childhood (pre-vitelliform detectable by EOG from birth); classic vitelliform stage in 1st–2nd decade",
        "surgical_urgency": "Anti-VEGF urgency if CNV with sudden VA loss; otherwise no surgical emergency",
        "gene_family": "Bestrophin family (BEST1) — calcium-activated chloride channel, RPE basolateral",
        "morphology": "Egg-yolk vitelliform foveal lesion progressing through 5 stages; pseudohypopyon; atrophy/CNV",
    },
    # -- PRPH2 -- Pattern Dystrophy / AVMD ---------------------------------------------------
    {
        "gene": "PRPH2",
        "alt_name": (
            "PRPH2 (PRPH2-346aa-6p21.1 / AD -- "
            "PATTERN-DYSTROPHY-BUTTERFLY-SHAPED-PIGMENT-DYSTROPHY-ADULT-VITELLIFORM-CENTRAL-AREOLAR -- "
            "HIGHLY-VARIABLE-EXPRESSIVITY-SAME-FAMILY-DIFFERENT-PHENOTYPES -- "
            "pARG172TRP-MOST-COMMON-VARIANT-RP-PHENOTYPE-IN-SOME -- "
            "PRPH2-RDS-PERIPHERIN-OUTER-SEGMENT-DISC-MORPHOGENESIS)"
        ),
        "protein": (
            "PRPH2 -- 6p21.1 AD -- PRPH2-346aa -- "
            "Peripherin-2-RDS-39kDa-Outer-Segment-Disc-Morphogenesis-Structural-Protein -- "
            "Tetraspanin-Like-4TM-Homotetramer-Heterooctamer-With-ROM1 -- "
            "Localised-Exclusively-Photoreceptor-Outer-Segment-Disc-Rim -- "
            "Maintains-Disc-Curvature-Stacking-Morphology -- "
            "OMIM-Gene-179605-Disease-Pattern-Dystrophy-169150-RP7-608133"
        ),
        "locus": "6p21.1",
        "protein_size": "346 aa / 39 kDa",
        "inheritance": (
            "AD (autosomal dominant) with highly variable expressivity; "
            "PRPH2 is a tetraspanin-like structural protein at the rim of photoreceptor outer segment discs; "
            "forms homotetramers (rod discs) or heterooctamers with ROM1 (rod outer segments); "
            "disc rim protein essential for outer segment disc morphogenesis and stacking; "
            "AD mechanism: haploinsufficiency; "
            "allelic series: same variant can cause different phenotypes in different family members; "
            "PHENOTYPIC SPECTRUM: butterfly-shaped pattern dystrophy (most classic); "
            "adult vitelliform macular dystrophy (AVMD); "
            "fundus pulverulentus; "
            "multifocal pattern dystrophy simulating STGD; "
            "central areolar choroidal dystrophy (CACD); "
            "retinitis pigmentosa (RP7) — p.Pro210Leu, p.Arg172Trp; "
            "digenic RP with ROM1: PRPH2 + ROM1 heterozygous together → RP; "
            "pathogenic variants: missense, frameshift, nonsense (>100 described)"
        ),
        "disease_category": "Pattern dystrophy — butterfly-shaped pigment dystrophy / AVMD / CACD; AD; variable expressivity; slow macular degeneration",
        "disease_pathway": (
            "PRPH2 (Peripherin-2, also known as RDS — 'retinal degeneration slow' from rds mouse model) is "
            "an outer segment structural protein critical for disc morphogenesis. "
            "DISC RIM STRUCTURE: PRPH2 localises exclusively to the curved rim of photoreceptor disc membranes; "
            "homotetramers in cones; heterooctamers with ROM1 in rods; "
            "the hairpin loop (D2 loop, aa 150-300) is the functional domain for homotypic/heterotypic interactions. "
            "DISEASE MECHANISM: PRPH2 haploinsufficiency (AD) → reduced disc rim protein → "
            "abnormal disc morphogenesis → lipofuscin/pigment accumulation in RPE → "
            "pattern-like deposits in macular RPE; "
            "BUTTERFLY PATTERN: pigment clumps in a triradiate pattern at fovea (resemble butterfly wings); "
            "AVMD: adult vitelliform lesion (later onset than BVMD; EOG may be mildly abnormal); "
            "CACD: progressive areolar RPE + photoreceptor atrophy from fovea, then peripheral spread; "
            "CNV CAN OCCUR in any pattern — anti-VEGF responsive. "
            "REMARKABLE VARIABILITY: identical variants in same family → one member RP phenotype, "
            "another mild butterfly dystrophy — modifier genes + stochastic effects implicated."
        ),
        "pathognomonic": (
            "BUTTERFLY-SHAPED PIGMENT DYSTROPHY: triradiate pattern of pigment clumping at fovea "
            "(resembles butterfly wings) with surrounding RPE mottling — CLASSIC PRPH2 PATTERN. "
            "MULTIFOCAL VARIANTS: single lesions at fovea or multiple lesions simulating STGD flecks; "
            "AVMD VARIANT: small yellow-white subretinal deposit, later onset than BEST1 BVMD; "
            "CACD VARIANT: progressive central scotoma; areolar atrophy at fovea; "
            "FAF: geographic pattern of RPE loss — hypofluorescent atrophy with hyperfluorescent border; "
            "VARIABLE EXPRESSIVITY KEY FEATURE: different family members with same variant → different phenotypes; "
            "ERG FULL-FIELD: NORMAL or mildly reduced (unless RP phenotype); "
            "EOG: may be mildly reduced but less dramatic than BEST1; "
            "DIGENIC RP: PRPH2 heterozygous + ROM1 heterozygous → RP (more severe than either alone). "
            "MOLECULAR: > 100 variants; p.Arg172Trp common (linked to RP7 in some); p.Arg172Gln → pattern dystrophy."
        ),
        "treatment": (
            "NO DISEASE-MODIFYING THERAPY PROVEN: "
            "ANTI-VEGF (intravitreal injection): for CNV complication — pattern dystrophy CNV responds well; "
            "ANNUAL REVIEW: OCT + FAF + visual field (Humphrey 10-2); "
            "GENERALLY BENIGN PROGNOSIS: most patients retain reading vision into 6th–7th decade; "
            "CACD VARIANT: may progress to legal blindness; "
            "GENETIC COUNSELLING: 50% AD recurrence; family members may have very different phenotype; "
            "MOLECULAR PANEL: include ROM1 (digenic RP risk if both PRPH2 + ROM1 heterozygous); "
            "LOW VISION AIDS: eccentric viewing, magnifiers for CACD variant; "
            "DRIVING LICENCE: depends on extent of visual field loss; "
            "SMOKING: avoid — oxidative stress accelerates macular changes."
        ),
        "key_features": [
            "Butterfly-shaped pigment dystrophy at fovea — triradiate pigment pattern",
            "Highly variable expressivity — same variant, different phenotypes in one family",
            "AVMD variant — adult vitelliform macular dystrophy (later onset than BEST1)",
            "CACD variant — central areolar choroidal dystrophy, progressive",
            "ERG full-field NORMAL (unless RP phenotype p.Arg172Trp)",
            "Digenic RP with ROM1 — both heterozygous → RP (more severe)",
            "AD; slow progression; anti-VEGF if CNV",
            "Same gene as RDS (retinal degeneration slow mouse model)",
        ],
        "key_ddx": (
            "BEST1 (BVMD): egg-yolk lesion; EOG Arden ratio <1.5 pathognomonic; earlier onset; "
            "ABCA4 (STGD1): AR; dark choroid; pisciform flecks; full ERG initially normal but different pattern; "
            "ELOVL4 (STGD3): AD; flecks; pedigree; heterozygous only systemic (SCA34 vs ocular); "
            "EFEMP1 (DHRD/ML): radial honeycomb drusen; R345W single variant; nasal to disc; "
            "AMD: older onset; drusen; CFH/ARMS2 variants; no family history of early onset; "
            "RP (retinitis pigmentosa): peripheral bone spicules; nyctalopia; ERG reduced."
        ),
        "systemic_involvement": False,
        "onset_age": "Pattern dystrophy typically 4th–6th decade; AVMD 5th–7th decade; CACD progressive from 3rd decade",
        "surgical_urgency": "Anti-VEGF urgency if CNV; otherwise elective monitoring",
        "gene_family": "Tetraspanin-related (disc rim structural protein) — PRPH2/RDS family",
        "morphology": "Butterfly-shaped foveal pigment; AVMD subretinal deposit; CACD areolar atrophy",
    },
    # -- TIMP3 -- Sorsby Fundus Dystrophy -----------------------------------------------------
    {
        "gene": "TIMP3",
        "alt_name": (
            "TIMP3 (TIMP3-211aa-22q12.3 / AD -- "
            "SORSBY-FUNDUS-DYSTROPHY-SFD-BILATERAL-CHOROIDAL-NEOVASCULARIZATION-HEMORRHAGE -- "
            "ACUTE-VISUAL-LOSS-4TH-DECADE-PATHOGNOMONIC-BRUCH-MEMBRANE-DEPOSITS -- "
            "DOMINANT-NEGATIVE-TIMP3-PREVENTS-ECM-TURNOVER-MMP-INHIBITION-FAILURE -- "
            "HIGH-DOSE-VITAMIN-A-BENEFICIAL-CONTRAST-TO-STGD1-ABCA4)"
        ),
        "protein": (
            "TIMP3 -- 22q12.3 AD -- TIMP3-211aa -- "
            "Tissue-Inhibitor-of-Metalloproteinase-3-24kDa-ECM-Bound-MMP-Inhibitor -- "
            "Inhibits-MMP-1-2-3-7-9-13-14-15-ADAM-10-ADAM-17-ECM-Turnover-Regulation -- "
            "Binds-ECM-Proteoglycan-Heparan-Sulfate-Localised-Bruch-Membrane -- "
            "Regulates-VEGF-Bioavailability-Via-VEGFR-2-Shedding-Inhibition -- "
            "OMIM-Gene-188826-Disease-SFD-136900"
        ),
        "locus": "22q12.3",
        "protein_size": "211 aa / 24 kDa",
        "inheritance": (
            "AD (autosomal dominant) — dominant-negative mechanism; "
            "TIMP3 is the only TIMP family member that is ECM-bound (not secreted freely); "
            "binds heparan sulfate proteoglycans in Bruch's membrane; "
            "heterozygous missense variants in exon 5 (Cys domain) cause SFD; "
            "ALL SFD variants affect cysteine residues in TIMP3 C-terminal domain OR create new cysteines; "
            "aberrant disulfide bonds → protein misfolding → abnormal TIMP3 accumulates in Bruch's membrane → "
            "DOMINANT NEGATIVE: mutant TIMP3 inhibits MMP function + fails to restrain ECM overgrowth → "
            "Bruch's membrane thickening + calcium deposits → CNV breaches Bruch's; "
            "TIMP3 NORMALLY: inhibits ADAM17 (TACE) → prevents VEGFR-2 shedding → reduces VEGF signalling; "
            "mutant TIMP3 → excess VEGF signalling → CNV; "
            "most common variants: p.Ser156Cys, p.Tyr168Cys, p.Gly166Cys, p.Ser181Cys; "
            "high penetrance; onset typically 4th decade"
        ),
        "disease_category": "Sorsby fundus dystrophy (SFD) — AD; bilateral choroidal neovascularization; acute visual loss 4th decade; Bruch's membrane deposits",
        "disease_pathway": (
            "TIMP3 (Tissue Inhibitor of Metalloproteinase-3) is a matrix-bound inhibitor of MMPs and ADAMs, "
            "uniquely localised to Bruch's membrane of the choroid-RPE interface. "
            "NORMAL FUNCTION: TIMP3 inhibits a broad spectrum of MMPs (MMP-1/2/3/7/9/13/14) and ADAMs (ADAM10/17) → "
            "maintains ECM homeostasis in Bruch's membrane by preventing uncontrolled matrix degradation; "
            "also inhibits ADAM17 (TACE) → prevents VEGFR-2 ectodomain shedding → reduces VEGF signalling; "
            "SFD MECHANISM: Cys-domain variants → extra cysteines or altered disulfide pairing → "
            "mutant TIMP3 forms aberrant disulfide-linked oligomers → "
            "accumulates in and thickens Bruch's membrane (dominant-negative deposits); "
            "thickened Bruch's → impaired RPE-photoreceptor oxygen/nutrient exchange; "
            "VEGF overactivation (TIMP3 loss of ADAM17 inhibition) → CNV growth through Bruch's → "
            "subretinal neovascular membrane → HAEMORRHAGE + ACUTE VISUAL LOSS (often over days); "
            "VITAMIN A MECHANISM IN SFD: exogenous retinol improves night vision in some patients (early studies) — "
            "distinct from STGD1 where vitamin A is absolutely contraindicated."
        ),
        "pathognomonic": (
            "BILATERAL SUBRETINAL/CHOROIDAL HAEMORRHAGE + CHOROIDAL NEOVASCULARISATION "
            "IN A PATIENT IN THEIR 4TH DECADE (30s-40s) = SORSBY FUNDUS DYSTROPHY UNTIL PROVEN OTHERWISE. "
            "BRUCH'S MEMBRANE DEPOSITS: bilateral, yellowish calcified deposits along Bruch's membrane "
            "(similar to AMD drusen but age-inappropriate — in young adults); OCT: Bruch's thickening + sub-RPE deposits; "
            "ACUTE PRESENTATION: sudden blurred vision + metamorphopsia + central scotoma in one eye, "
            "then fellow eye months-to-years later; "
            "BILATERAL SYMMETRY: eventual bilateral CNV (often one eye first); "
            "CNV TYPE: Type 1 (sub-RPE) progressing to Type 2 (subretinal); "
            "DARK ADAPTATION: profoundly impaired early (nyctalopia) — Bruch's membrane barrier affects rod recovery; "
            "VITAMIN A: may improve dark adaptation in early disease (opposite to STGD); "
            "GENETIC: ALL SFD variants involve cysteine residues in TIMP3 C-terminal; "
            "p.Ser156Cys most common worldwide."
        ),
        "treatment": (
            "ANTI-VEGF FIRST-LINE FOR CNV (GOOD RESPONSE): "
            "intravitreal ranibizumab/aflibercept/bevacizumab — SFD CNV is generally responsive; "
            "treat both eyes (bilateral disease — fellow eye surveillance mandatory); "
            "DARK ADAPTATION: high-dose vitamin A (50,000 IU/day) — may improve nyctalopia in early SFD; "
            "NOTE: OPPOSITE of STGD1 — Vitamin A BENEFICIAL in SFD (not harmful); "
            "MONITORING: annual OCT + dark adaptation testing + BCVA; "
            "SFD CNV RESPONDS WELL TO ANTI-VEGF (unlike some AMD subtypes); "
            "PHOTODYNAMIC THERAPY (PDT): used historically for classic CNV; now combined with anti-VEGF; "
            "GENETICS COUNSELLING: AD 50% recurrence; confirm by TIMP3 sequencing (all exon 5 cysteine variants); "
            "GENETIC PANEL: ABCA4 + BEST1 + PRPH2 + TIMP3 + ELOVL4 (macular dystrophy panel); "
            "PROGRESSION: without anti-VEGF → legal blindness by 5th-6th decade; with anti-VEGF → vision preserved."
        ),
        "key_features": [
            "Bilateral choroidal neovascularization in 4th decade — age-inappropriate",
            "Subretinal haemorrhage + acute visual loss = Sorsby hallmark",
            "Bruch's membrane thickening + calcium deposits (OCT sub-RPE deposits)",
            "Dark adaptation profoundly impaired (nyctalopia) — early symptom",
            "Vitamin A may improve dark adaptation (OPPOSITE of STGD1 — NOT contraindicated)",
            "TIMP3 variants ALL involve cysteine residues (aberrant disulfide bonds)",
            "Anti-VEGF response GOOD (CNV anti-VEGF responsive)",
            "AD; all SFD variants are Cys domain missense (exon 5 predominantly)",
        ],
        "key_ddx": (
            "AMD: onset >60y; CFH/ARMS2; Bruch's drusen; Vitamin A not beneficial; anti-VEGF also used; "
            "ABCA4 (STGD1): AR; bull's eye; flecks; dark choroid; NO CNV haemorrhage early; Vitamin A CI; "
            "Angioid streaks (ABCC6/pseudoxanthoma elasticum): cracks in Bruch's; skin laxity; echocardiogram; "
            "C1QTNF5 (LORD): iris crystalline deposits; older onset; Bruch's deposits; "
            "Pathological myopia: axial myopia > -6D; lacquer cracks; fundus tessellation; "
            "Idiopathic CNV in young adults: smaller; single eye; TIMP3 negative."
        ),
        "systemic_involvement": False,
        "onset_age": "4th decade (30s-40s) for nyctalopia + CNV; Bruch's deposits detectable earlier",
        "surgical_urgency": "Anti-VEGF urgency when CNV active (acute visual loss); otherwise monitoring",
        "gene_family": "TIMP family (tissue inhibitor of metalloproteinases) — Bruch's membrane ECM regulator",
        "morphology": "Bruch's membrane deposits + bilateral choroidal neovascularization + subretinal haemorrhage",
    },
    # -- ELOVL4 -- Stargardt-like Macular Dystrophy 3 ----------------------------------------
    {
        "gene": "ELOVL4",
        "alt_name": (
            "ELOVL4 (ELOVL4-314aa-6q14.1 / AD -- "
            "STARGARDT-LIKE-MACULAR-DYSTROPHY-3-STGD3-YELLOW-FLECKS-FOVEAL-ATROPHY -- "
            "HETEROZYGOUS-AD-MACULAR-DYSTROPHY-HOMOZYGOUS-AR-ICHTHYOSIS-SEIZURES-SCA34 -- "
            "VLCPUFA-VERY-LONG-CHAIN-POLYUNSATURATED-FATTY-ACID-DEFICIENCY -- "
            "AUTOSOMAL-DOMINANT-SPINOCEREBELLAR-ATAXIA-34-SCA34-ADULT-ONSET)"
        ),
        "protein": (
            "ELOVL4 -- 6q14.1 AD-AR -- ELOVL4-314aa -- "
            "Very-Long-Chain-Fatty-Acid-Elongase-4-34kDa-ER-Membrane-Enzyme -- "
            "Elongates-C26-to-C36-VLC-PUFA-and-VLC-SFA-in-Endoplasmic-Reticulum -- "
            "C28-C36-VLCPUFA-DHA-Elongation-Products-Essential-Photoreceptor-Outer-Segment-Disc -- "
            "Also-Required-Lamellar-Body-Formation-Skin-Barrier-Ichthyosis-When-Biallelic -- "
            "OMIM-Gene-605512-Disease-STGD3-600110-SCA34-133190"
        ),
        "locus": "6q14.1",
        "protein_size": "314 aa / 34 kDa",
        "inheritance": (
            "AD (heterozygous) for STGD3 macular dystrophy; "
            "AR (biallelic) for severe multisystem disease (ARCI + neonatal ichthyosis + seizures); "
            "AD (heterozygous) for Spinocerebellar Ataxia type 34 (SCA34) — different variants than STGD3; "
            "ELOVL4 encodes an ER-membrane enzyme that elongates very-long-chain polyunsaturated fatty acids (VLC-PUFA): "
            "C26 → C28 → C32 → C36 (n-3 and n-6 series); "
            "C32:6 and C34:6 VLC-PUFA are enriched in PHOTORECEPTOR DISC MEMBRANES (rod outer segments >50%); "
            "STGD3 variants (p.Val2Cys, 5bp deletion exon 6): abolish ER retention signal → "
            "mistargeting of enzyme → dominant-negative + loss of VLC-PUFA in retina → "
            "disc membrane destabilisation → photoreceptor degeneration; "
            "HOMOZYGOUS LOF: severe neonate — ichthyosis (skin barrier requires C>26 fatty acids for lamellar bodies) + "
            "neonatal seizures + profound ID; "
            "SCA34: different missense variants → adult-onset spinocerebellar ataxia"
        ),
        "disease_category": "Stargardt-like macular dystrophy type 3 (STGD3) — AD heterozygous; yellow flecks + foveal atrophy; VLC-PUFA deficiency",
        "disease_pathway": (
            "ELOVL4 (Elongation of Very-Long-Chain Fatty Acids Protein 4) is an ER transmembrane enzyme that "
            "catalyses the elongation of very-long-chain polyunsaturated fatty acids (VLC-PUFA) beyond C26. "
            "RETINAL FUNCTION: VLC-PUFA (C28-C36 n-3 and n-6) are major lipid components of photoreceptor disc membranes; "
            "C32:6 and C34:6 DHA-elongation products make up >50% of phosphatidylcholine in rod outer segment discs; "
            "these ultra-long PUFAs create membrane fluidity + packing geometry essential for opsin conformational changes; "
            "STGD3 MECHANISM: p.Val2Cys (most common STGD3 variant) or 5bp insertion (Stargardt-like pedigree) → "
            "loss of ER retention signal or misfolding → "
            "ELOVL4 mislocalised from ER → dominant-negative aggregation with WT ELOVL4 → "
            "VLC-PUFA deficit in retina → disc membrane structural failure → "
            "photoreceptor degeneration mimicking STGD1 (yellow flecks + bull's eye). "
            "ALLELIC SEVERITY: heterozygous = STGD3 or SCA34; homozygous = severe neonatal disease "
            "(ichthyosis because skin lamellar bodies require C>26 fatty acids for barrier formation; "
            "neonatal seizures from CNS VLC-PUFA deficit)."
        ),
        "pathognomonic": (
            "STARGARDT-LIKE YELLOW FLECKS + BULL'S EYE MACULOPATHY IN AN AUTOSOMAL DOMINANT PEDIGREE — "
            "phenotypically similar to STGD1 (ABCA4) but AUTOSOMAL DOMINANT inheritance key differentiator. "
            "TEMPORAL PERIFOVEAL SPARING: some STGD3 variants spare the temporal perifoveal region; "
            "FAF: pisciform hyperfluorescent flecks similar to ABCA4 STGD; "
            "NO DARK CHOROID: unlike ABCA4 STGD1; FFA shows normal choroidal background; "
            "ERG: NORMAL or mildly reduced; "
            "ADULT ONSET (typically 20s-40s) in STGD3; "
            "SCA34 VARIANT: adult-onset cerebellar ataxia + erythrokeratoderma; "
            "IMPORTANT ALLELIC QUESTION: heterozygous vs homozygous/compound heterozygous — "
            "BIALLELIC (rare): neonatal onset ichthyosis + seizures + ID — "
            "SEVERE NEONATE WITH COLLODION BABY APPEARANCE + SEIZURES: rule out biallelic ELOVL4. "
            "FAMILY HISTORY: AD pedigree with Stargardt-like phenotype → ELOVL4 mandatory in panel."
        ),
        "treatment": (
            "NO PROVEN DISEASE-MODIFYING THERAPY FOR STGD3: "
            "VITAMIN A: NOT contraindicated in STGD3 (unlike ABCA4 STGD1 — different mechanism); "
            "ANTI-VEGF: if CNV develops; "
            "OPTICAL AIDS: low vision rehabilitation; "
            "GENETIC COUNSELLING: AD 50% (STGD3); AR 25% recurrence (biallelic severe); "
            "DIFFERENTIATE: sequence ABCA4 + ELOVL4 + PRPH2 (phenotypic overlap); "
            "SCA34 PATIENTS: neurological referral; physiotherapy; speech therapy for ataxia; "
            "BIALLELIC NEONATES: neonatal ICU; anti-epileptic drugs; skin barrier support (emollients); "
            "PROGNOSIS STGD3: similar to ABCA4 STGD1 — progressive visual loss; "
            "FAMILY TESTING: all first-degree relatives should have retinal assessment + genetics."
        ),
        "key_features": [
            "Stargardt-like phenotype with AUTOSOMAL DOMINANT inheritance (unlike ABCA4 AR)",
            "Yellow flecks + bull's eye maculopathy — mimics STGD1",
            "NO dark choroid on FFA (differs from ABCA4 STGD1)",
            "Heterozygous = STGD3 macular dystrophy OR SCA34 ataxia",
            "Biallelic (homozygous) = neonatal ichthyosis + seizures + severe ID",
            "VLC-PUFA C28-C36 deficiency underlies retinal and skin pathology",
            "Vitamin A NOT contraindicated (opposite mechanism to ABCA4)",
            "Molecular: p.Val2Cys and 5bp insertion are classic STGD3 variants",
        ],
        "key_ddx": (
            "ABCA4 (STGD1): AR (vs AD); dark choroid present; Vitamin A absolutely CI; very large gene; "
            "PRPH2 (pattern dystrophy): AD; butterfly pigment; family variability; "
            "PRDM13 (NCMD): non-progressive; grade 0-3; stationary; AD; regulatory variant; "
            "Fundus flavimaculatus (ABCA4 adult): AR; older; similar flecks; "
            "SCA34 vs STGD3: neurological vs pure macular; ELOVL4 variant matters."
        ),
        "systemic_involvement": False,
        "onset_age": "STGD3: 20s-40s; SCA34 (different variants): 4th-6th decade; biallelic: neonatal",
        "surgical_urgency": "No acute urgency; anti-VEGF if CNV",
        "gene_family": "ELOVL (elongation of very-long-chain fatty acids) — ER membrane elongase",
        "morphology": "Yellow-white flecks + bull's eye maculopathy; NO dark choroid (unlike ABCA4)",
    },
    # -- C1QTNF5 -- Late-Onset Retinal Degeneration ------------------------------------------
    {
        "gene": "C1QTNF5",
        "alt_name": (
            "C1QTNF5 (C1QTNF5-243aa-11q23.3 / AD -- "
            "LATE-ONSET-RETINAL-DEGENERATION-LORD-L-ORD -- "
            "CRYSTALLINE-IRIS-DEPOSITS-PATHOGNOMONIC-UNIQUE-TO-LORD -- "
            "DRUSEN-BRUCH-MEMBRANE-DEPOSITS-SUBRETINAL -- "
            "pSER163ARG-FOUNDER-VARIANT-NORTHERN-EUROPEAN-PROGRESSIVE-OUTER-RETINA)"
        ),
        "protein": (
            "C1QTNF5 -- 11q23.3 AD -- C1QTNF5-243aa -- "
            "C1q-And-TNF-Related-Protein-5-CTRP5-27kDa-Secreted-Homotrimer -- "
            "C1q-Like-Globular-Domain-Complement-Collagen-Adipokine-Related -- "
            "Expressed-RPE-Ciliary-Epithelium-Inner-Wall-Schlemm-Canal -- "
            "Secreted-Into-Subretinal-Space-Bruch-Membrane-Aqueous-Humour -- "
            "OMIM-Gene-608752-Disease-LORD-605712"
        ),
        "locus": "11q23.3",
        "protein_size": "243 aa / 27 kDa",
        "inheritance": (
            "AD (autosomal dominant) — dominant-negative or gain-of-toxic-function mechanism; "
            "C1QTNF5 (CTRP5) is a member of the C1q/TNF superfamily; forms homotrimers via C1q-like globular domain; "
            "expressed in RPE, non-pigmented ciliary epithelium, inner wall of Schlemm's canal, and vitreous; "
            "secreted into subretinal space and Bruch's membrane; "
            "pathogenic variant p.Ser163Arg (S163R) found in most LORD families; "
            "p.S163R: substitution in conserved C1q globular domain → protein misfolding → "
            "abnormal aggregation in Bruch's membrane + subretinal space; "
            "HALLMARK: iris crystalline deposits (translucent glistening crystals in iris stroma) — "
            "UNIQUE TO LORD; not seen in AMD or other macular dystrophies; "
            "onset 6th-7th decade; "
            "geographic atrophy (GA) and CNV in late stage; "
            "IOP: long posterior ciliary arteries compressed by deposits → pupil dilation abnormal in some"
        ),
        "disease_category": "Late-onset retinal degeneration (LORD/L-ORD) — AD; crystalline iris deposits + drusenoid deposits; onset 6th decade",
        "disease_pathway": (
            "C1QTNF5 (C1q and TNF-related protein 5, CTRP5) is a secreted homotrimeric adipokine-related protein "
            "expressed by RPE and ciliary epithelium. "
            "NORMAL FUNCTION: secreted C1QTNF5 circulates in subretinal space and Bruch's membrane; "
            "probable role in complement regulation and RPE metabolic support; "
            "binds LTBP-2 (another macular disease gene); "
            "LORD MECHANISM: p.S163R missense → altered protein folding → "
            "mutant C1QTNF5 homotrimers aggregate → "
            "accumulate as CRYSTALLINE DEPOSITS in iris stroma + subretinal space + Bruch's membrane; "
            "iris deposits form visible glistening crystals under slit lamp (pathognomonic); "
            "subretinal deposits → outer nuclear layer loss → progressive photoreceptor degeneration → "
            "geographic atrophy (GA) and/or CNV by 7th-8th decade. "
            "IRIS DEPOSITS MECHANISM: C1QTNF5 is also expressed by non-pigmented ciliary epithelium → "
            "secreted into posterior chamber → deposits in iris stroma during aqueous flow."
        ),
        "pathognomonic": (
            "CRYSTALLINE IRIS DEPOSITS (slit-lamp visible glistening translucent crystals in iris stroma) "
            "IN A PATIENT WITH LATE-ONSET MACULAR DEGENERATION = LORD PATHOGNOMONIC. "
            "NO OTHER HEREDITARY MACULAR DYSTROPHY CAUSES IRIS CRYSTALLINE DEPOSITS — "
            "this distinguishes LORD from AMD, BEST1, ABCA4, and all other macular dystrophies. "
            "DRUSENOID DEPOSITS: yellowish-white deposits at level of RPE/Bruch's, peripapillary + macular; "
            "OCT: sub-RPE deposits extending from disc; outer retinal tubulation in late stage; "
            "FAF: foci of hyper/hypofluorescence; "
            "ONSET 6th DECADE: drusenoid deposits visible from ~50s; iris deposits variable; "
            "LONG POSTERIOR CILIARY ARTERY CHANGES: some patients have iris transillumination + photophobia; "
            "GEOGRAPHIC ATROPHY + CNV: late disease (7th–8th decade); "
            "GENETIC: p.Ser163Arg (S163R) single-site founder variant in most LORD families worldwide."
        ),
        "treatment": (
            "NO PROVEN DISEASE-MODIFYING THERAPY: "
            "ANTI-VEGF for CNV complication; "
            "ANNUAL MONITORING: OCT + FAF + BCVA; dark adaptation; "
            "IRIS DEPOSITS: no treatment required (cosmetically visible but not visually significant); "
            "LOW VISION REHABILITATION: magnifiers, CCTV; "
            "GENETIC COUNSELLING: AD 50%; first-degree relatives need retinal + slit-lamp evaluation; "
            "SLIT-LAMP EVALUATION: iris crystals help identify at-risk relatives before macular changes appear; "
            "COMPLEMENT PATHWAY: theoretically anti-complement therapy (like AMD GA treatment) may be relevant; "
            "PROGNOSIS: slower progression than geographic AMD; many patients retain driving vision into 7th decade; "
            "MOLECULAR: C1QTNF5 sequencing; p.S163R accounts for most cases."
        ),
        "key_features": [
            "Crystalline iris deposits (glistening iris stroma crystals) — PATHOGNOMONIC, unique to LORD",
            "Late-onset drusenoid macular degeneration — onset 6th decade",
            "Sub-RPE + Bruch's membrane deposits extending peripapillary",
            "AD; p.Ser163Arg (S163R) founder variant in most families",
            "Progresses to geographic atrophy + CNV in late stage",
            "No other macular dystrophy causes iris crystals — key differentiator from AMD",
            "OCT: outer retinal tubulation in late atrophy",
            "C1QTNF5 expressed in RPE + ciliary epithelium (explains iris deposits)",
        ],
        "key_ddx": (
            "AMD: onset similar (6th decade); no iris crystals (AMD never causes iris crystals); CFH/ARMS2 variants; "
            "TIMP3 (SFD): 4th decade CNV; Bruch's deposits; no iris crystals; "
            "BEST1 (BVMD): egg-yolk lesion; EOG abnormal; younger onset; no iris deposits; "
            "Cystinosis: cystine crystals in cornea + iris; systemic disease; metabolic; "
            "Bietti crystalline corneoretinal dystrophy (CYP4V2): corneal crystals + retinal crystals; AR."
        ),
        "systemic_involvement": False,
        "onset_age": "6th decade (drusenoid deposits); iris crystals variable timing; CNV/GA 7th-8th decade",
        "surgical_urgency": "Anti-VEGF urgency for CNV; otherwise monitoring",
        "gene_family": "C1q/TNF superfamily (CTRP5) — secreted homotrimeric RPE/ciliary protein",
        "morphology": "Iris crystalline deposits + peripapillary/macular drusenoid deposits + late GA/CNV",
    },
    # -- PRDM13 -- North Carolina Macular Dystrophy -------------------------------------------
    {
        "gene": "PRDM13",
        "alt_name": (
            "PRDM13 (PRDM13-718aa-6q16.1 / AD -- "
            "NORTH-CAROLINA-MACULAR-DYSTROPHY-NCMD-MCDR1-MCDR2 -- "
            "NON-PROGRESSIVE-STATIONARY-MACULAR-LESION-FROM-BIRTH-KEY-DDx -- "
            "GRADE-0-DRUSEN-ONLY-TO-GRADE-3-CHORIORETINAL-COLOBOMA-STAPHYLOMA -- "
            "REGULATORY-VARIANT-PROMOTER-5UTR-EXOME-MISSES-DEDICATED-SEQUENCING-MANDATORY)"
        ),
        "protein": (
            "PRDM13 -- 6q16.1 AD -- PRDM13-718aa -- "
            "PR-SET-Domain-13-81kDa-Transcription-Factor-Histone-Methyltransferase -- "
            "PR-SET-Domain-Zinc-Fingers-H3K9-Methylation-Transcriptional-Repressor -- "
            "Expressed-Retinal-Amacrine-Cells-Horizontal-Cells-Retinal-Development -- "
            "NCMD-Caused-Regulatory-Variants-PRDM13-Promoter-5UTR-Duplication -- "
            "OMIM-Gene-616741-Disease-NCMD-136550"
        ),
        "locus": "6q16.1",
        "protein_size": "718 aa / 81 kDa",
        "inheritance": (
            "AD (autosomal dominant) — regulatory variants in PRDM13 promoter/5'UTR cause NCMD; "
            "PRDM13 encodes a PR domain zinc finger protein (PRDM family) with histone methyltransferase activity; "
            "expressed in retinal amacrine and horizontal cells during foveal development; "
            "CRITICAL DIAGNOSTIC TRAP: NCMD is caused by REGULATORY VARIANTS (promoter/5'UTR duplications/substitutions) — "
            "STANDARD EXOME SEQUENCING MISSES THESE — coding exome reports NEGATIVE; "
            "NCMD diagnosis requires: targeted PRDM13 5'UTR/promoter sequencing OR chromosomal microarray (for tandem duplication); "
            "MCDR1 locus (6q16): most NCMD families (PRDM13 regulatory); "
            "MCDR2 locus (6q16.1): smaller subset (also PRDM13 regulatory); "
            "MCDR3: 5p15 locus (THRB?) — rare; "
            "PATHOGNOMONIC STATIONARITY: NCMD is NON-PROGRESSIVE (born with the lesion; does not worsen); "
            "GRADE SYSTEM: 0=drusen-like deposits only; 1=confluent drusen; 2=chorioretinal atrophy; "
            "3=coloboma-like staphyloma (mimics chorioretinal coloboma)"
        ),
        "disease_category": "North Carolina Macular Dystrophy (NCMD) — AD; NON-PROGRESSIVE stationary central macular lesion; grade 0-3; regulatory variant",
        "disease_pathway": (
            "PRDM13 (PR/SET Domain 13) is a transcriptional repressor and putative histone methyltransferase "
            "expressed in retinal amacrine and horizontal cells during retinal development. "
            "NCMD MECHANISM: gain-of-function regulatory variants (promoter duplications/substitutions) → "
            "abnormal PRDM13 overexpression in retinal development → "
            "disruption of amacrine/horizontal cell foveal maturation → "
            "abnormal foveal development → STATIONARY MACULAR LESION from birth. "
            "NON-PROGRESSIVE NATURE: the lesion is a developmental malformation (not a degeneration); "
            "once retinal development is complete, lesion does not progress — "
            "this is the KEY differentiator from all other hereditary macular dystrophies. "
            "GRADING (Gass/NCMD Working Group): "
            "Grade 0: small irregular drusen-like deposits at fovea; "
            "Grade 1: confluent large drusen or areas of RPE disturbance; "
            "Grade 2: central chorioretinal atrophy; "
            "Grade 3: large coloboma-like excavated staphyloma (fundus may look like optic disc coloboma). "
            "VISUAL ACUITY: highly variable — grade 3 may have 6/60 VA; grade 0 may have 6/6; "
            "STRABISMUS + NYSTAGMUS: common in severe grade (foveal hypoplasia equivalent)."
        ),
        "pathognomonic": (
            "NON-PROGRESSIVE CENTRAL MACULAR LESION IN A PATIENT WITH POSITIVE FAMILY HISTORY (AD) — "
            "discovered in childhood/infancy, NOT worsening with age = NCMD PATHOGNOMONIC. "
            "LESION STATIONARY: VA stable throughout life (no progressive deterioration unlike STGD/BVMD); "
            "GRADE 3 COLOBOMA-LIKE: large excavated macular staphyloma — may be misdiagnosed as "
            "'macular coloboma' (differentiate: NCMD positive PRDM13 family history; "
            "true coloboma = PAX6/CHD7 embryological defect); "
            "OCT: outer retinal loss at fovea ± subretinal fluid; grade 3 excavation may extend posteriorly; "
            "ERG: NORMAL (full-field ERG) — only foveal function impaired; "
            "EXOME SEQUENCING NEGATIVE: THIS IS A DIAGNOSTIC TRAP — regulatory variant missed by exome; "
            "REQUEST: chromosomal microarray (tandem duplication 6q16) OR dedicated PRDM13 promoter sequencing. "
            "VA RANGE: 6/6 (grade 0) → 6/60 or worse (grade 3)."
        ),
        "treatment": (
            "NO TREATMENT REQUIRED (non-progressive lesion): "
            "REASSURANCE: primary need — patients relieved that disease does not worsen; "
            "REFRACTION: full optical correction (hyperopia common in grade 3); "
            "AMBLYOPIA TREATMENT: patching if severe grade in one eye with strabismus/anisometropia; "
            "LOW VISION AIDS: if grade 2-3 (magnifiers, CCTV); "
            "GENETICS: AD 50% recurrence; regulatory variant may be missed by standard exome — "
            "inform geneticist to request specific PRDM13 promoter/5'UTR region; "
            "DRIVING: depends on VA; grade 0-1 usually drive; grade 3 may not qualify; "
            "SCHOOLING: large print, preferential seating if grade 2-3; "
            "OCCUPATIONAL GUIDANCE: avoid roles requiring fine central vision in grade 2-3; "
            "KEY MESSAGE: condition stable — prognosis for NCMD is SIGNIFICANTLY BETTER than other macular dystrophies."
        ),
        "key_features": [
            "Non-progressive (STATIONARY) macular lesion from birth — KEY differentiator",
            "Grade 0 (drusen) to Grade 3 (coloboma-like staphyloma) classification",
            "Standard coding exome sequencing MISSES the diagnosis (regulatory variant)",
            "Require chromosomal microarray or targeted PRDM13 promoter/5'UTR sequencing",
            "Full-field ERG NORMAL (only foveal function affected)",
            "AD pedigree; MCDR1 locus (6q16)",
            "May be misdiagnosed as 'macular coloboma' (important DDx)",
            "Visual acuity highly variable: 6/6 (grade 0) to 6/60 (grade 3)",
        ],
        "key_ddx": (
            "True macular coloboma (PAX6/CHD7): embryological; usually sporadic; no family history; "
            "ABCA4 (STGD1): progressive; AR; flecks; dark choroid; "
            "BEST1 (BVMD): progressive; EOG abnormal; egg-yolk; AD but progressive; "
            "Congenital foveal hypoplasia (nystagmus): no macular lesion; foveal pit absent on OCT; "
            "Toxoplasma scar: pigmented chorioretinal scar; serology; unilateral usually; "
            "C1QTNF5 (LORD): progressive; iris crystals; older onset; "
            "NCMD CERTAINTY: stationary + AD family history + exome negative → always test PRDM13 locus."
        ),
        "systemic_involvement": False,
        "onset_age": "Congenital (lesion present at birth); detected in childhood; non-progressive thereafter",
        "surgical_urgency": "No surgical urgency (non-progressive); amblyopia treatment in childhood if needed",
        "gene_family": "PRDM/PR-SET domain family — transcriptional repressor; retinal development",
        "morphology": "Stationary foveal lesion: grade 0 drusen → grade 3 coloboma-like macular staphyloma",
    },
    # -- EFEMP1 -- Doyne Honeycomb Retinal Dystrophy / Malattia Leventinese ------------------
    {
        "gene": "EFEMP1",
        "alt_name": (
            "EFEMP1 (EFEMP1-493aa-2p16.1 / AD -- "
            "DOYNE-HONEYCOMB-RETINAL-DYSTROPHY-DHRD-MALATTIA-LEVENTINESE-ML -- "
            "RADIAL-DRUSEN-HONEYCOMB-PATTERN-NASAL-TO-DISC-PATHOGNOMONIC -- "
            "pARG345TRP-R345W-SINGLE-PATHOGENIC-VARIANT-VAST-MAJORITY-OF-CASES -- "
            "FIBULIN-3-DOMAIN-DISRUPTION-BRUCH-MEMBRANE-DRUSEN-ACCUMULATION)"
        ),
        "protein": (
            "EFEMP1 -- 2p16.1 AD -- EFEMP1-493aa -- "
            "EGF-Containing-Fibulin-Extracellular-Matrix-Protein-1-Fibulin-3-55kDa -- "
            "6-EGF-Like-Domains-Fibulin-C-Terminal-Module-Secreted-ECM-Glycoprotein -- "
            "Expressed-RPE-Highly-Secreted-Into-Bruch-Membrane-Lamina-Vitrea -- "
            "Regulates-Complement-Cascade-Via-FHL-1-Truncated-Factor-H -- "
            "OMIM-Gene-601548-Disease-DHRD-126600-ML-153700b"
        ),
        "locus": "2p16.1",
        "protein_size": "493 aa / 55 kDa",
        "inheritance": (
            "AD (autosomal dominant) — SINGLE PATHOGENIC VARIANT in vast majority: p.Arg345Trp (R345W); "
            "EFEMP1 (Fibulin-3) is a secreted extracellular matrix glycoprotein with 6 EGF-like domains + fibulin C-terminus; "
            "expressed abundantly in RPE and Bruch's membrane (lamina vitrea); "
            "p.R345W in EGF-like domain 4: Cys pair disrupted → misfolded protein; "
            "misfolded EFEMP1 accumulates in Bruch's membrane → "
            "drusen formation (radial drusen extending from disc + macula); "
            "EFEMP1 also regulates complement: binds FHL-1 (complement factor H-like 1) → "
            "reduces complement activation in Bruch's; mutant EFEMP1 → complement dysregulation; "
            "HISTORICAL CONTEXT: Doyne (1899, Oxford family) and Malattia Leventinese (Swiss Alpine family, Leventina valley) "
            "were described as separate diseases before both mapped to EFEMP1 p.R345W; "
            "phenotypically identical — now unified as DHRD/ML; "
            "variable expressivity but near-100% penetrance with p.R345W"
        ),
        "disease_category": "Doyne honeycomb retinal dystrophy (DHRD) / Malattia Leventinese (ML) — AD; radial honeycomb drusen; p.R345W; nasal to disc + macula",
        "disease_pathway": (
            "EFEMP1 (EGF-Containing Fibulin Extracellular Matrix Protein 1, Fibulin-3) is a secreted ECM glycoprotein "
            "localised to Bruch's membrane (lamina vitrea) and abundantly expressed by RPE. "
            "NORMAL FUNCTION: structural component of Bruch's membrane; regulates complement by binding FHL-1 "
            "(factor H-like protein 1, truncated complement factor H) → reduces C3b deposition on Bruch's; "
            "also expressed in blood vessel walls, basement membranes. "
            "DHRD/ML MECHANISM: p.R345W missense in EGF-like domain 4 → "
            "disrupts Cys146-Cys156 disulfide bond → "
            "protein misfolding → intracellular accumulation and abnormal secretion → "
            "extracellular aggregation in Bruch's membrane → "
            "DRUSEN FORMATION: soft drusen + hard drusen in a RADIAL HONEYCOMB PATTERN "
            "(honeycomb = contiguous drusen forming a reticular network); "
            "nasal to disc distribution (distinguishes from AMD drusen which are macular-centred); "
            "COMPLEMENT DYSREGULATION: mutant EFEMP1 fails to bind FHL-1 → excess complement activation → "
            "RPE/photoreceptor damage → CNV in ~50% by 6th-7th decade. "
            "DRUSEN TIMELINE: visible from 2nd-3rd decade; CNV from 5th-6th decade."
        ),
        "pathognomonic": (
            "RADIAL HONEYCOMB DRUSEN PATTERN EXTENDING NASALLY FROM OPTIC DISC TOWARD MACULA = "
            "DHRD/ML PATHOGNOMONIC. "
            "DRUSEN DISTRIBUTION: radiating from optic disc nasally + extending to macula; "
            "compare to AMD drusen which are macular-centred; "
            "HONEYCOMB APPEARANCE: contiguous drusen form a reticular/lattice network — visible on funduscopy + OCT; "
            "DRUSEN ONSET: visible from 2nd-3rd decade (vs AMD drusen from 6th-7th decade) — "
            "AGE-INAPPROPRIATE DRUSEN IN A YOUNG ADULT = EFEMP1 UNTIL PROVEN OTHERWISE; "
            "OCT: sub-RPE deposits; Bruch's membrane thickening; drusenoid PED; "
            "FAF: drusen hyperfluorescent (early); atrophy hypofluorescent (late); "
            "CNV: ~50% by 7th decade — haemorrhage, metamorphopsia, acute VA loss; "
            "GENETIC: p.R345W in EFEMP1 exon 10 (>95% of DHRD/ML cases); "
            "FAMILY HISTORY: often multiple generations affected — AD pedigree."
        ),
        "treatment": (
            "ANTI-VEGF FOR CNV (RESPONSIVE): "
            "intravitreal ranibizumab/aflibercept — DHRD/ML CNV generally responds well to anti-VEGF; "
            "MONITORING: annual OCT + FAF + BCVA from 3rd decade; "
            "ANTI-VEGF TIMING: early CNV treatment prevents macular atrophy; "
            "BRUCH'S MEMBRANE SUPPORT: no proven agent; "
            "COMPLEMENT MODULATION: theoretically beneficial (complement dysregulation mechanistic); "
            "GENETIC COUNSELLING: AD 50%; p.R345W single-site — easy to test; "
            "FAMILY SCREENING: all first-degree relatives → fundus + OCT; "
            "LOW VISION AIDS: if atrophy or untreated CNV causes permanent VA loss; "
            "DRIVING: maintained if CNV treated early; "
            "PROGNOSIS: without CNV → reasonable central vision until 6th decade; with CNV → vision loss if untreated."
        ),
        "key_features": [
            "Radial honeycomb drusen extending nasally from disc to macula — pathognomonic",
            "Drusen onset in 2nd-3rd decade (age-inappropriate for AMD)",
            "p.Arg345Trp (R345W) single variant accounts for >95% of DHRD/ML cases",
            "AD; two separate disease families (Doyne Oxford + Malattia Leventinese Swiss) = same variant",
            "CNV in ~50% by 7th decade — anti-VEGF responsive",
            "EFEMP1 (Fibulin-3) regulates complement via FHL-1 binding",
            "FAF: hyperfluorescent drusen (lipofuscin + complement deposits)",
            "Distinguished from AMD by: younger onset, radial/nasal distribution, strong AD family history",
        ],
        "key_ddx": (
            "AMD (age-related): drusen macular-centred; onset >60y; no strong AD family history; CFH/ARMS2; "
            "TIMP3 (SFD): Bruch's deposits + CNV but 4th decade, haemorrhagic, no honeycomb; "
            "C1QTNF5 (LORD): iris crystals; drusen; later onset 6th decade; "
            "Familial drusen (HEMICENTIN-1/FBLN1): autosomal dominant; bilateral drusen; check genetic panel; "
            "Pseudoxanthoma elasticum (ABCC6): angioid streaks + Bruch's calcification; skin peau d'orange; "
            "Geographic atrophy early stage: Bruch's deposits; EFEMP1 negative; complement pathway (CFH/ARMS2/C3)."
        ),
        "systemic_involvement": False,
        "onset_age": "2nd-3rd decade (drusen visible); CNV from 5th-6th decade; legal blindness risk from 7th decade if untreated",
        "surgical_urgency": "Anti-VEGF urgency when CNV develops; otherwise monitoring from 3rd decade",
        "gene_family": "Fibulin family (EGF-containing fibulin ECM proteins) — Bruch's membrane structural protein",
        "morphology": "Radial honeycomb drusen nasal to disc + macula; honeycomb pattern; late CNV/atrophy",
    },
]


def _make_cohort(entry, seed):
    rng = random.Random(seed)
    gene = entry["gene"]
    patients = []
    for i in range(40):
        # Visual acuity poor (< 6/18 Snellen equivalent)
        if gene == "ABCA4":
            va_poor = rng.random() < 0.55  # ~50-60% eventually <6/18
        elif gene == "BEST1":
            va_poor = rng.random() < 0.30  # relatively preserved VA in many
        elif gene == "PRPH2":
            va_poor = rng.random() < 0.25  # generally benign
        elif gene == "TIMP3":
            va_poor = rng.random() < 0.50  # CNV haemorrhage causes acute loss
        elif gene == "ELOVL4":
            va_poor = rng.random() < 0.48  # similar to STGD1
        elif gene == "C1QTNF5":
            va_poor = rng.random() < 0.35  # slower progression
        elif gene == "PRDM13":
            va_poor = rng.random() < 0.40  # grade 3 patients
        else:  # EFEMP1
            va_poor = rng.random() < 0.38  # CNV if untreated

        # Choroidal neovascularisation (CNV) present
        if gene == "ABCA4":
            cnv = rng.random() < 0.10  # relatively uncommon in STGD1 (<10%)
        elif gene == "BEST1":
            cnv = rng.random() < 0.25  # CNV in atrophic stage
        elif gene == "PRPH2":
            cnv = rng.random() < 0.20  # pattern dystrophy CNV
        elif gene == "TIMP3":
            cnv = rng.random() < 0.80  # defining feature of Sorsby
        elif gene == "ELOVL4":
            cnv = rng.random() < 0.12
        elif gene == "C1QTNF5":
            cnv = rng.random() < 0.30  # late stage LORD
        elif gene == "PRDM13":
            cnv = rng.random() < 0.05  # very rare (non-progressive)
        else:  # EFEMP1
            cnv = rng.random() < 0.48  # ~50% by 7th decade

        # Drusen / deposits present
        if gene in ("ABCA4",):
            drusen = rng.random() < 0.15  # flecks not drusen; some drusen
        elif gene in ("BEST1",):
            drusen = rng.random() < 0.20  # sub-RPE material
        elif gene in ("PRPH2",):
            drusen = rng.random() < 0.35
        elif gene in ("TIMP3",):
            drusen = rng.random() < 0.90  # Bruch's deposits always
        elif gene in ("ELOVL4",):
            drusen = rng.random() < 0.20
        elif gene in ("C1QTNF5",):
            drusen = rng.random() < 0.85  # drusenoid deposits defining
        elif gene in ("PRDM13",):
            drusen = rng.random() < 0.60  # grade 0-1 drusen
        else:  # EFEMP1
            drusen = rng.random() < 0.95  # honeycomb drusen always

        # Anti-VEGF treatment received
        if gene == "TIMP3" and cnv:
            anti_vegf = rng.random() < 0.88  # high treatment uptake for SFD
        elif gene in ("BEST1", "PRPH2", "C1QTNF5", "EFEMP1") and cnv:
            anti_vegf = rng.random() < 0.82
        elif gene == "ABCA4" and cnv:
            anti_vegf = rng.random() < 0.78
        elif cnv:
            anti_vegf = rng.random() < 0.80
        else:
            anti_vegf = False

        # Nyctalopia / dark adaptation impaired
        if gene == "TIMP3":
            nyctalopia = rng.random() < 0.75  # Bruch's membrane barrier impairs rod recovery
        elif gene == "C1QTNF5":
            nyctalopia = rng.random() < 0.55  # outer retinal loss
        elif gene == "ABCA4":
            nyctalopia = rng.random() < 0.20  # full-field ERG initially normal
        elif gene == "EFEMP1":
            nyctalopia = rng.random() < 0.30
        else:
            nyctalopia = rng.random() < 0.15

        # EOG abnormal (Arden ratio <1.5) — RPE function
        if gene == "BEST1":
            eog_abnormal = True  # pathognomonic
        elif gene == "C1QTNF5":
            eog_abnormal = rng.random() < 0.35  # RPE affected
        elif gene in ("ABCA4", "PRPH2"):
            eog_abnormal = rng.random() < 0.20
        else:
            eog_abnormal = rng.random() < 0.08

        # ERG full-field abnormal
        if gene == "ABCA4":
            erg_abnormal = rng.random() < 0.22  # only when flecks extend beyond 45°
        elif gene == "ELOVL4":
            erg_abnormal = rng.random() < 0.18
        elif gene == "C1QTNF5":
            erg_abnormal = rng.random() < 0.30
        elif gene == "PRDM13":
            erg_abnormal = rng.random() < 0.10  # mostly foveal
        else:
            erg_abnormal = rng.random() < 0.12

        # Consanguineous family (relevant for AR gene)
        if gene == "ABCA4":
            consanguineous = rng.random() < 0.38  # AR; Middle Eastern/SE Asian families
        else:
            consanguineous = rng.random() < 0.06  # AD genes

        patients.append({
            "id": f"{gene}-{seed}-{i+1:03d}",
            "gene": gene,
            "va_poor": va_poor,
            "cnv": cnv,
            "drusen_deposits": drusen,
            "anti_vegf_treatment": anti_vegf,
            "nyctalopia": nyctalopia,
            "eog_abnormal": eog_abnormal,
            "erg_abnormal": erg_abnormal,
            "consanguineous": consanguineous,
            "inheritance": entry["inheritance"].split(";")[0].strip(),
        })
    return patients


def generate_overview():
    all_patients = []
    for idx, entry in enumerate(MD_GENES):
        all_patients.extend(_make_cohort(entry, SEED_BASE + idx))

    total = len(all_patients)
    va_poor_count = sum(1 for p in all_patients if p["va_poor"])
    cnv_count = sum(1 for p in all_patients if p["cnv"])
    drusen_count = sum(1 for p in all_patients if p["drusen_deposits"])
    anti_vegf_count = sum(1 for p in all_patients if p["anti_vegf_treatment"])
    nyctalopia_count = sum(1 for p in all_patients if p["nyctalopia"])
    eog_count = sum(1 for p in all_patients if p["eog_abnormal"])
    consanguineous_count = sum(1 for p in all_patients if p["consanguineous"])

    gene_summary = {}
    for idx, entry in enumerate(MD_GENES):
        gene = entry["gene"]
        cohort = _make_cohort(entry, SEED_BASE + idx)
        gene_summary[gene] = {
            "gene": gene,
            "alt_name": entry["alt_name"],
            "locus": entry["locus"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"].split(";")[0].strip(),
            "disease_category": entry["disease_category"],
            "pathognomonic": entry["pathognomonic"][:300],
            "morphology": entry["morphology"],
            "systemic_involvement": entry["systemic_involvement"],
            "onset_age": entry["onset_age"],
            "surgical_urgency": entry["surgical_urgency"],
            "gene_family": entry["gene_family"],
            "n_patients": len(cohort),
            "va_poor_pct": round(100 * sum(1 for p in cohort if p["va_poor"]) / len(cohort), 1),
            "cnv_pct": round(100 * sum(1 for p in cohort if p["cnv"]) / len(cohort), 1),
            "drusen_pct": round(100 * sum(1 for p in cohort if p["drusen_deposits"]) / len(cohort), 1),
            "anti_vegf_pct": round(100 * sum(1 for p in cohort if p["anti_vegf_treatment"]) / len(cohort), 1),
            "nyctalopia_pct": round(100 * sum(1 for p in cohort if p["nyctalopia"]) / len(cohort), 1),
            "eog_abnormal_pct": round(100 * sum(1 for p in cohort if p["eog_abnormal"]) / len(cohort), 1),
            "consanguineous_pct": round(100 * sum(1 for p in cohort if p["consanguineous"]) / len(cohort), 1),
        }

    return {
        "atlas": "Hereditary-Macular-Dystrophy-Atlas",
        "subtitle": "Complete 8-Gene Hereditary Macular Dystrophy Reference -- ABCA4/BEST1/PRPH2/TIMP3/ELOVL4/C1QTNF5/PRDM13/EFEMP1",
        "genes_covered": [e["gene"] for e in MD_GENES],
        "total_patients": total,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "aggregate_metrics": {
            "va_worse_than_6_18_pct": round(100 * va_poor_count / total, 1),
            "cnv_present_pct": round(100 * cnv_count / total, 1),
            "drusen_deposits_pct": round(100 * drusen_count / total, 1),
            "anti_vegf_treatment_pct": round(100 * anti_vegf_count / total, 1),
            "nyctalopia_pct": round(100 * nyctalopia_count / total, 1),
            "eog_abnormal_pct": round(100 * eog_count / total, 1),
            "consanguineous_family_pct": round(100 * consanguineous_count / total, 1),
        },
        "gene_summary": gene_summary,
    }


def generate_breakdown():
    breakdown = []
    for idx, entry in enumerate(MD_GENES):
        cohort = _make_cohort(entry, SEED_BASE + idx)
        breakdown.append({
            "gene": entry["gene"],
            "alt_name": entry["alt_name"],
            "locus": entry["locus"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"].split(";")[0].strip(),
            "disease_category": entry["disease_category"],
            "pathognomonic": entry["pathognomonic"],
            "treatment": entry["treatment"],
            "key_features": entry["key_features"],
            "key_ddx": entry["key_ddx"],
            "morphology": entry["morphology"],
            "systemic_involvement": entry["systemic_involvement"],
            "onset_age": entry["onset_age"],
            "surgical_urgency": entry["surgical_urgency"],
            "gene_family": entry["gene_family"],
            "n_patients": len(cohort),
            "va_poor_pct": round(100 * sum(1 for p in cohort if p["va_poor"]) / len(cohort), 1),
            "cnv_pct": round(100 * sum(1 for p in cohort if p["cnv"]) / len(cohort), 1),
            "drusen_pct": round(100 * sum(1 for p in cohort if p["drusen_deposits"]) / len(cohort), 1),
            "anti_vegf_pct": round(100 * sum(1 for p in cohort if p["anti_vegf_treatment"]) / len(cohort), 1),
            "nyctalopia_pct": round(100 * sum(1 for p in cohort if p["nyctalopia"]) / len(cohort), 1),
            "eog_abnormal_pct": round(100 * sum(1 for p in cohort if p["eog_abnormal"]) / len(cohort), 1),
            "consanguineous_pct": round(100 * sum(1 for p in cohort if p["consanguineous"]) / len(cohort), 1),
            "sample_patients": cohort[:3],
        })
    return {"gene_breakdowns": breakdown}


def generate_definitions():
    return {
        "gene_entries": {
            entry["gene"]: {
                "gene": entry["gene"],
                "full_name": entry["protein"].split(" --")[0].strip(),
                "locus": entry["locus"],
                "protein_size": entry["protein_size"],
                "inheritance": entry["inheritance"].split(";")[0].strip(),
                "disease_name": entry["disease_category"],
                "disease_pathway": entry["disease_pathway"],
                "pathognomonic": entry["pathognomonic"],
                "treatment": entry["treatment"][:500],
                "key_features": entry["key_features"],
                "key_ddx": entry["key_ddx"],
                "morphology": entry["morphology"],
                "systemic_involvement": entry["systemic_involvement"],
                "onset_age": entry["onset_age"],
                "surgical_urgency": entry["surgical_urgency"],
                "gene_family": entry["gene_family"],
            }
            for entry in MD_GENES
        },
        "md_glossary": {
            "ABCA4 Stargardt — Vitamin A Absolutely Contraindicated": (
                "STARGARDT DISEASE (ABCA4): the SINGLE most important management rule is "
                "VITAMIN A SUPPLEMENTATION ABSOLUTELY CONTRAINDICATED. "
                "MECHANISM: ABCA4 normally flips all-trans-retinal out of photoreceptor disc lumen; "
                "ABCA4 LOF → all-trans-retinal accumulates → condenses to A2E (toxic bisretinoid) in RPE; "
                "EXOGENOUS RETINOL (Vitamin A) → enters visual cycle → more all-trans-retinal generated → "
                "MORE A2E PRODUCED → accelerated RPE death → faster macular atrophy. "
                "CLINICAL RULE: never prescribe Vitamin A supplements, multivitamins with >100% RDA retinol, "
                "or cod liver oil to ABCA4 STGD patients. "
                "DARK CHOROID SIGN ON FFA: ~80% of STGD1 — caused by A2E masking background choroidal fluorescence; "
                "pathognomonic when present in young patient with bull's eye maculopathy. "
                "CONTRAST: TIMP3 (Sorsby) — Vitamin A MAY IMPROVE dark adaptation (opposite)."
            ),
            "BEST1 EOG — Arden Ratio Diagnostic Protocol": (
                "ELECTROOCULOGRAPHY (EOG) is the definitive functional test for BEST1/BVMD. "
                "TECHNIQUE: patient sits in front of large diffuse light source; "
                "eyes track between two fixation lights 30° apart; "
                "electrodes at medial/lateral canthi record the corneoretinal standing potential; "
                "DARK PHASE (10-15 min): dark trough (minimum potential) recorded; "
                "LIGHT PHASE (10-15 min): light peak (maximum potential) recorded; "
                "ARDEN RATIO = Light Peak ÷ Dark Trough; "
                "NORMAL: ≥ 1.85 (i.e. light peak is ≥185% of dark trough); "
                "BVMD PATHOGNOMONIC: Arden ratio < 1.5 in VIRTUALLY ALL BEST1 PATHOGENIC VARIANT CARRIERS; "
                "CRITICALLY: EOG is ABNORMAL EVEN IN: "
                "• pre-vitelliform stage (before any lesion visible on fundus); "
                "• asymptomatic carriers (heterozygous family members with normal VA); "
                "• 'forme fruste' (minimal clinical phenotype); "
                "CLINICAL UTILITY: EOG confirms BEST1 dysfunction before genetic testing; "
                "identifies unaffected-appearing family members who carry the variant; "
                "CONTRAST: ERG full-field is NORMAL in BVMD — EOG is RPE function, ERG is photoreceptor function."
            ),
            "PRDM13 NCMD — Diagnostic Trap: Exome Misses Regulatory Variant": (
                "NORTH CAROLINA MACULAR DYSTROPHY (NCMD) caused by REGULATORY VARIANTS in PRDM13 PROMOTER/5'UTR — "
                "NOT by coding sequence variants. "
                "DIAGNOSTIC TRAP: standard whole-exome sequencing (WES) reads only CODING EXONS → "
                "NCMD patients will have NEGATIVE exome result even with classic phenotype; "
                "non-clinician geneticists may incorrectly report 'no variant found' → misdiagnosis. "
                "WHAT TO REQUEST INSTEAD: "
                "1. Chromosomal MICROARRAY (SNP array / CMA) — detects tandem duplications at 6q16 (MCDR1/MCDR2 locus); "
                "2. Targeted PRDM13 5'UTR/promoter region Sanger sequencing; "
                "3. Long-read whole-genome sequencing (if available); "
                "VARIANTS DESCRIBED: tandem duplications of ~14kb or ~35kb upstream of PRDM13; "
                "point substitutions in 5'UTR; "
                "CLINICAL CLUE for NCMD: stationary macular lesion present in infancy + AD family history + "
                "exome negative → ALWAYS pursue PRDM13 locus testing. "
                "NON-PROGRESSIVE = hallmark: if lesion is worsening, reconsider NCMD diagnosis."
            ),
            "TIMP3 Sorsby — Vitamin A and Anti-VEGF Protocol": (
                "SORSBY FUNDUS DYSTROPHY (SFD) — two key management points that CONTRAST with STGD1: "
                "1. VITAMIN A: "
                "STGD1 (ABCA4): Vitamin A ABSOLUTELY CONTRAINDICATED (accelerates A2E); "
                "SFD (TIMP3): Vitamin A MAY IMPROVE DARK ADAPTATION (50,000 IU/day in early disease); "
                "mechanism: Bruch's membrane thickening impairs retinol transport to photoreceptors; "
                "exogenous retinol supplementation bypasses the Bruch's barrier → improves rod photoreceptor recycling; "
                "NEVER ASSUME all macular dystrophies follow STGD vitamin A rules. "
                "2. CNV ANTI-VEGF: "
                "SFD CNV responds well to anti-VEGF (ranibizumab/aflibercept); "
                "early treatment prevents permanent macular atrophy; "
                "both eyes require surveillance (bilateral disease, may present sequentially); "
                "DOMINANT-NEGATIVE MECHANISM: TIMP3 Cys-domain variants → aberrant disulfide bonds → "
                "mutant TIMP3 aggregates in Bruch's → cannot inhibit MMPs or ADAM17 → "
                "Bruch's ECM overgrowth + excess VEGF signalling → bilateral CNV in 4th decade."
            ),
            "EFEMP1 DHRD/ML — Single Variant and Honeycomb Drusen": (
                "DOYNE HONEYCOMB RETINAL DYSTROPHY (DHRD) and MALATTIA LEVENTINESE (ML) were described "
                "100 years apart in separate European families (Doyne 1899 Oxford; ML 1905 Swiss Alps Leventina valley) "
                "and unified when BOTH mapped to p.Arg345Trp (R345W) in EFEMP1 in 1999. "
                "SINGLE VARIANT: p.R345W accounts for >95% of all DHRD/ML cases worldwide — "
                "targeted R345W testing is sufficient for suspected DHRD/ML; "
                "few additional variants (p.Arg345Gln, p.Ser163Pro) described in <5%. "
                "HONEYCOMB DRUSEN: contiguous drusen forming a reticular/lattice network — "
                "radiating from optic disc nasally (NASAL-TO-DISC DISTRIBUTION IS KEY DDx from AMD); "
                "AMD drusen are macular-centred; EFEMP1 drusen extend nasally from disc. "
                "AGE-INAPPROPRIATE DRUSEN: drusen visible by 2nd-3rd decade in a young adult → EFEMP1 first diagnosis; "
                "COMPLEMENT PATHWAY LINK: EFEMP1 normally binds FHL-1 (factor H-like 1) to suppress complement; "
                "p.R345W mutant → complement dysregulation → drusen + CNV pathway shared with AMD. "
                "PROGNOSIS: CNV in ~50% by 7th decade; anti-VEGF generally responsive."
            ),
        },
    }
