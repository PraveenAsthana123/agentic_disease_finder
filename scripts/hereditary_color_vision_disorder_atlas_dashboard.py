#!/usr/bin/env python3
"""Hereditary-Color-Vision-And-Photoreceptor-Disorder-Atlas — Complete 8-Gene Atlas
(CNGB3 · CNGA3 · GNAT2 · PDE6C · PDE6H · ATF6 · KCNV2 · OPN1LW).

CNGB3    (CNG beta-3 subunit; 809 aa; ~92 kDa; 8q21.3; AR;
           Achromatopsia ACHM3 — MOST COMMON ~50% of all achromatopsia;
           p.Thr383fsTer17 European/North American founder mutation;
           COMPLETE achromatopsia: total color blindness, pendular nystagmus, DAY-BLINDNESS (photophobia);
           FL-41 rose-tinted lenses STANDARD; gene therapy Phase 2/3 trials most advanced;
           ERG: no cone response, normal/near-normal rod response;
           seed SEED_BASE+0).
CNGA3    (CNG alpha-3 subunit; 694 aa; ~80 kDa; 2q11.2; AR;
           Achromatopsia ACHM2 — ~25% of all achromatopsia;
           p.Arg427Trp European founder (~30%);
           COMPLETE achromatopsia: identical phenotype to CNGB3;
           Gene therapy Phase 2 (AGTC-402, BTT-401);
           seed SEED_BASE+1).
GNAT2    (Cone transducin alpha-2; 354 aa; ~40 kDa; 1p13.3; AR;
           Achromatopsia ACHM4 — ~2-3% of achromatopsia;
           Typically complete achromatopsia; less severe nystagmus in some;
           seed SEED_BASE+2).
PDE6C    (cGMP phosphodiesterase alpha' subunit, cone-specific; 858 aa; ~99 kDa; 10q23.33; AR;
           Achromatopsia ACHM5 — incomplete achromatopsia more common;
           Some residual color vision/photopic ERG (incomplete form);
           photophobia dominant complaint; p.Ala146Val pan-ethnic;
           seed SEED_BASE+3).
PDE6H    (cGMP phosphodiesterase gamma' subunit, cone-specific; 84 aa; ~10 kDa; 12p13.33; AR;
           Achromatopsia ACHM6 — incomplete;
           Milder/incomplete achromatopsia: residual color vision, some photopic ERG;
           seed SEED_BASE+4).
ATF6     (ER stress transcription factor; 670 aa; ~75 kDa; 1q23.2; AR;
           Achromatopsia ACHM7 / Macular hypoplasia;
           DISTINCT: OCT shows FOVEAL HYPOPLASIA as primary finding;
           cone ERG may be less extinguished early; macular atrophy develops;
           p.Arg324Cys, p.Gly48Glu founders;
           seed SEED_BASE+5).
KCNV2    (Kv8.2 voltage-gated K+ channel; 545 aa; ~62 kDa; 9p24.2; AR;
           Cone Dystrophy with Supernormal Rod ERG (CDSRR);
           PATHOGNOMONIC: SUPERNORMAL ROD ERG b-wave on dark-adapted bright flash;
           PROGRESSIVE (unlike static achromatopsia); moderate photophobia, macular dystrophy evolves;
           seed SEED_BASE+6).
OPN1LW   (L-cone opsin; 364 aa; ~41 kDa; Xq28; XLR/AD;
           Blue Cone Monochromatism (BCM);
           Gene array deletion/rearrangement at LCR — standard WES MISSES (need targeted gene array testing);
           Males: ONLY S-cones + rods functional, L+M cones absent;
           Pendular nystagmus in infancy improving with age; myopia common (90%);
           foveal hypoplasia on OCT; congenital stationary — NOT progressive;
           seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 2422-2429).
"""

import random

SEED_BASE = 2422

CV_GENES = [
    # -- CNGB3 -- Achromatopsia ACHM3 (Most Common ~50%) ----------------------------------------
    {
        "gene": "CNGB3",
        "alt_name": (
            "CNGB3 (CNGB3-809aa-8q21.3 / AR -- "
            "ACHROMATOPSIA-ACHM3-MOST-COMMON-~50pct-ALL-ACHROMATOPSIA -- "
            "pTHR383FSTER17-EUROPEAN-NORTH-AMERICAN-FOUNDER-MUTATION -- "
            "COMPLETE-ACHROMATOPSIA-TOTAL-COLOR-BLINDNESS-NO-CONE-ERG-PENDULAR-NYSTAGMUS-DAY-BLINDNESS -- "
            "FL-41-ROSE-TINTED-LENSES-STANDARD-GENE-THERAPY-PHASE-2-3-MOST-ADVANCED)"
        ),
        "protein": (
            "CNGB3 -- 8q21.3 AR -- CNGB3-809aa -- "
            "Cyclic-Nucleotide-Gated-Channel-Beta-3-Subunit-92kDa-Cone-Outer-Segment -- "
            "CNG-Channel-Heterotetrameric-2xCNGA3-2xCNGB3-Modulates-cGMP-Sensitivity-Ca2+-Calmodulin -- "
            "Glutamic-Acid-Rich-Protein-GARP-Domain-Unique-to-B-Subunit -- "
            "OMIM-Gene-605080-Disease-ACHM3-262300"
        ),
        "locus": "8q21.3",
        "protein_size": "809 aa / 92 kDa",
        "inheritance": (
            "AR (autosomal recessive); CNGB3 encodes the beta-3 modulatory subunit of the cone "
            "cyclic nucleotide-gated (CNG) channel in cone photoreceptor outer segments; "
            "CNG CHANNEL COMPOSITION: heterotetramer of 2x CNGA3 (alpha, pore-forming) + 2x CNGB3 (beta, modulatory); "
            "CNGB3 FUNCTION: modulates cGMP sensitivity, calmodulin inhibition, channel kinetics; "
            "GARP domain (glutamic acid-rich protein): unique long N-terminal extension connecting disc rim to CNG channel; "
            "CNGB3 LOF -> non-functional CNG channel -> cones cannot depolarise in response to light -> "
            "total cone signal absence -> complete achromatopsia; "
            "MOST COMMON CAUSE: ~50% of all achromatopsia cases worldwide; "
            "FOUNDER MUTATION: p.Thr383fsTer17 (c.1148delC) -- European and North American ancestry; "
            "Prevalence: 1:30,000-1:50,000 (achromatopsia total); CNGB3 accounts for ~50%"
        ),
        "disease_category": "Achromatopsia ACHM3 (AR) — most common ~50% of all achromatopsia; CNGB3 beta-3 CNG channel subunit; complete achromatopsia: total color blindness, pendular nystagmus, photophobia (day-blindness), VA ~0.1; static/non-progressive",
        "disease_pathway": (
            "CNGB3 encodes the beta-3 modulatory subunit of the cone photoreceptor CNG channel. "
            "PHOTOTRANSDUCTION CASCADE: photon -> rhodopsin/cone opsin activation -> transducin (GNAT2) -> "
            "PDE6 (PDE6C/PDE6H in cones) -> cGMP hydrolysis -> CNG channel closes -> "
            "cone hyperpolarisation -> signal to bipolar cells. "
            "CNG CHANNEL NORMAL FUNCTION: in darkness, cGMP high -> CNG channel open -> "
            "Na+/Ca2+ influx -> cone depolarised (dark current); "
            "light -> cGMP falls -> CNG closes -> Ca2+ drops -> cone hyperpolarises -> signal; "
            "CNGB3 LOF MECHANISM: CNGA3 homotetramers form but with markedly reduced cGMP sensitivity and "
            "altered single-channel conductance; cones cannot sustain appropriate light responses -> "
            "total cone dysfunction -> achromatopsia. "
            "STATIC DISEASE: cones present anatomically (early) but non-functional; "
            "progressive outer nuclear layer thinning develops with age (not a primary degenerative process); "
            "GENE THERAPY RATIONALE: cones present -> delivery of CNGB3 cDNA (AAV2/5 or AAV8) "
            "to restore functional CNG channel -> Phase 2/3 trials (most advanced of all achromatopsia genes)."
        ),
        "pathognomonic": (
            "COMPLETE ACHROMATOPSIA (CNGB3/ACHM3) -- ALL FOUR CRITERIA: "
            "1. TOTAL COLOR BLINDNESS: fails ALL Ishihara plates; FM-100 Hue: complete scatter; "
            "D-15: all errors; Nagel anomaloscope: monochromat -- matches ANY red-green mixture with yellow; "
            "2. PENDULAR NYSTAGMUS: horizontal pendular nystagmus onset neonatal/infancy; null point; "
            "contact lenses may reduce amplitude (null point shift); "
            "3. PHOTOPHOBIA (DAY-BLINDNESS): PATHOGNOMONIC -- bright light severely reduces function; "
            "prefers dim light; FL-41 (rose/pink) tinted lenses standard intervention; "
            "4. REDUCED VISUAL ACUITY: ~0.1 (6/60) to 0.2 (6/30); stable throughout life; "
            "ERG PATTERN: FLAT/ABSENT PHOTOPIC ERG (no cone signal); "
            "SCOTOPIC (ROD) ERG: normal or near-normal; "
            "OCT: outer nuclear layer (ONL) thinning at fovea (progressive with age); "
            "Genetic: biallelic CNGB3 variants; p.Thr383fsTer17 founder in European/North American."
        ),
        "treatment": (
            "FL-41 ROSE-TINTED LENSES: standard for photophobia (reduces short-wavelength glare); "
            "dark sunglasses outdoors; wide-brimmed hat; "
            "TINTED CONTACT LENSES: reduce nystagmus amplitude (null point change) + photophobia; "
            "REFRACTIVE CORRECTION: myopia and astigmatism common; correct early; "
            "LOW VISION AIDS: magnification devices; high-contrast materials; prefer dim lighting environments; "
            "GENE THERAPY: Phase 2/3 trials (RD-CURE Consortium, MeiraGTx/Janssen) — refer newly diagnosed patients; "
            "AAV-CNGB3 subretinal injection; "
            "AVOID: vitamin A supplements (NO benefit in achromatopsia -- unlike rod dystrophies); "
            "dark glasses mandatory outdoors; "
            "EDUCATIONAL: low vision classroom support; enlarged print; preferential seating; dim classroom lighting; "
            "GENETIC COUNSELLING: AR 25% risk per sibling; carrier testing for p.Thr383fsTer17."
        ),
        "key_features": [
            "Complete achromatopsia: total color blindness (fails all Ishihara plates)",
            "Pendular nystagmus onset neonatal/infancy",
            "Photophobia (DAY-BLINDNESS) — prefers dim light; FL-41 lenses standard",
            "VA ~0.1 (6/60); stable/non-progressive",
            "Flat photopic ERG; normal scotopic (rod) ERG",
            "p.Thr383fsTer17 European/North American founder mutation",
            "Gene therapy Phase 2/3 trials — refer newly diagnosed",
            "Vitamin A supplements: NO benefit (avoid)",
        ],
        "key_ddx": (
            "CNGA3 (ACHM2): identical phenotype; ~25% achromatopsia; different gene; "
            "ATF6 (ACHM7): foveal hypoplasia on OCT (distinguishes from CNGB3/CNGA3 where OCT initially near-normal); "
            "OPN1LW (BCM): X-linked males; S-cones preserved; myopia 90%; foveal hypoplasia; "
            "KCNV2 (CDSRR): supernormal rod ERG; PROGRESSIVE macular dystrophy; "
            "Congenital stationary night blindness (CSNB): rod dysfunction; different ERG pattern; "
            "Albinism: foveal hypoplasia; iris transillumination; pigment deficiency."
        ),
        "systemic_involvement": False,
        "onset_age": "Congenital / neonatal (nystagmus and photophobia present from birth); VA stable",
        "surgical_urgency": "No acute surgical urgency; gene therapy trial referral for newly diagnosed",
        "gene_family": "Cyclic nucleotide-gated (CNG) channel family — cone phototransduction",
        "morphology": "Normal fundus appearance early; foveal ONL thinning on OCT (progressive with age); flat photopic ERG",
    },
    # -- CNGA3 -- Achromatopsia ACHM2 (~25%) -------------------------------------------------------
    {
        "gene": "CNGA3",
        "alt_name": (
            "CNGA3 (CNGA3-694aa-2q11.2 / AR -- "
            "ACHROMATOPSIA-ACHM2-~25pct-ALL-ACHROMATOPSIA-SECOND-MOST-COMMON -- "
            "pARG427TRP-EUROPEAN-FOUNDER-~30pct -- "
            "COMPLETE-ACHROMATOPSIA-IDENTICAL-PHENOTYPE-CNGB3-CNG-ALPHA-3-PORE-FORMING-SUBUNIT -- "
            "GENE-THERAPY-PHASE-2-AGTC-402-BTT-401)"
        ),
        "protein": (
            "CNGA3 -- 2q11.2 AR -- CNGA3-694aa -- "
            "Cyclic-Nucleotide-Gated-Channel-Alpha-3-Subunit-80kDa-Pore-Forming-Cone-Outer-Segment -- "
            "CNG-Channel-Principal-Pore-Subunit-cGMP-Binding-Cyclic-Nucleotide-Binding-Domain -- "
            "CNBD-C-Linker-S1-S6-TM-Helices-Selectivity-Filter -- "
            "OMIM-Gene-600053-Disease-ACHM2-216900"
        ),
        "locus": "2q11.2",
        "protein_size": "694 aa / 80 kDa",
        "inheritance": (
            "AR (autosomal recessive); CNGA3 encodes the alpha-3 (pore-forming) subunit of the cone CNG channel; "
            "CNGA3 IS THE PRINCIPAL PORE-FORMING SUBUNIT: contains S1-S6 transmembrane helices, "
            "selectivity filter, and cGMP-binding cyclic nucleotide-binding domain (CNBD); "
            "CNGA3 LOF -> loss of pore-forming unit -> non-functional CNG channel -> "
            "total cone signal absence -> complete achromatopsia; "
            "~25% of all achromatopsia cases (second most common after CNGB3); "
            "FOUNDER MUTATION: p.Arg427Trp (c.1279C>T) -- European ancestry ~30% of CNGA3 cases; "
            "also R563H, Y181C common variants; "
            "Identical clinical phenotype to CNGB3-achromatopsia: "
            "cannot be distinguished clinically; gene panel mandatory"
        ),
        "disease_category": "Achromatopsia ACHM2 (AR) — ~25% of achromatopsia; CNGA3 alpha-3 pore-forming CNG channel subunit; phenotype identical to ACHM3/CNGB3; gene therapy Phase 2 ongoing",
        "disease_pathway": (
            "CNGA3 encodes the principal pore-forming subunit (alpha-3) of the cone CNG channel. "
            "CNG CHANNEL STRUCTURE: CNGA3 forms the functional pore; CNGB3 modulates gating. "
            "CNGA3 S6 HELIX + SELECTIVITY FILTER: determines ion selectivity (Na+, K+, Ca2+ permeation); "
            "CNBD AT C-TERMINUS: cGMP binding opens channel; cooperative binding (Hill coefficient ~2). "
            "CNGA3 LOF: unlike CNGB3 LOF (which still allows CNGA3 homotetramers with low cGMP sensitivity), "
            "CNGA3 LOF -> no pore-forming unit -> complete CNG channel abolition -> "
            "even more complete cone signal failure than CNGB3 LOF; "
            "PHOTOTRANSDUCTION BLOCKED: cones cannot generate any light-driven current -> "
            "total achromatopsia, identical severity to CNGB3. "
            "GENE THERAPY: AAV-CNGA3 (AGTC-402 by Applied Genetic Technologies Corporation; "
            "BTT-401 by Beacon Therapeutics) -- Phase 2 trials; "
            "subretinal injection to surviving cone photoreceptors."
        ),
        "pathognomonic": (
            "COMPLETE ACHROMATOPSIA (CNGA3/ACHM2) -- CLINICALLY IDENTICAL TO CNGB3/ACHM3: "
            "1. TOTAL COLOR BLINDNESS: all Ishihara plates failed; Nagel anomaloscope monochromat; "
            "2. PENDULAR NYSTAGMUS: congenital; null point; amplitude reduced by contact lenses; "
            "3. PHOTOPHOBIA (DAY-BLINDNESS): marked; FL-41 rose tints standard; "
            "4. VISUAL ACUITY ~0.1: stable throughout life; "
            "ERG: FLAT PHOTOPIC (no cone response); NORMAL SCOTOPIC (rods intact); "
            "OCT: foveal ONL thinning with age; initially may appear near-normal; "
            "GENETIC DIFFERENTIATION FROM CNGB3: ONLY by sequencing -- "
            "biallelic CNGA3 variants; p.Arg427Trp common in European ancestry; "
            "CLINICAL IMPLICATION: all achromatopsia patients need FULL GENE PANEL "
            "(CNGB3 + CNGA3 + GNAT2 + PDE6C + PDE6H + ATF6) for gene therapy eligibility stratification."
        ),
        "treatment": (
            "FL-41 ROSE-TINTED LENSES: photophobia management (standard); "
            "TINTED CONTACT LENSES: nystagmus amplitude reduction + photophobia; "
            "REFRACTIVE CORRECTION: myopia/astigmatism correction; "
            "LOW VISION AIDS: magnification; high contrast; preferential dim lighting; "
            "GENE THERAPY: Phase 2 trials ongoing (AGTC-402, BTT-401) -- "
            "refer all newly diagnosed CNGA3 patients to gene therapy trial; "
            "AVOID: vitamin A supplements (no benefit in achromatopsia); "
            "EDUCATIONAL SUPPORT: low vision classroom accommodations; enlarged print; "
            "GENETIC COUNSELLING: AR 25% risk; p.Arg427Trp European carrier testing."
        ),
        "key_features": [
            "Complete achromatopsia: identical phenotype to CNGB3/ACHM3",
            "~25% of all achromatopsia (second most common)",
            "p.Arg427Trp European founder (~30% of CNGA3 cases)",
            "Flat photopic ERG; normal scotopic ERG",
            "Pendular nystagmus; photophobia; VA ~0.1 stable",
            "Gene therapy Phase 2 (AGTC-402, BTT-401) — refer newly diagnosed",
            "Cannot be clinically distinguished from CNGB3 — gene panel mandatory",
            "Vitamin A: NO benefit",
        ],
        "key_ddx": (
            "CNGB3 (ACHM3): clinically identical; ~50% achromatopsia; different gene (8q21.3 vs 2q11.2); "
            "GNAT2 (ACHM4): rare ~2-3%; same complete phenotype; transducin subunit; "
            "ATF6 (ACHM7): foveal hypoplasia on OCT distinguishes; less extinguished cone ERG early; "
            "PDE6C (ACHM5): incomplete form common; residual photopic ERG; "
            "OPN1LW (BCM): X-linked males; myopia; foveal hypoplasia; S-cone monochromat (S-cones preserved)."
        ),
        "systemic_involvement": False,
        "onset_age": "Congenital (nystagmus and photophobia from birth); VA stable throughout life",
        "surgical_urgency": "No acute surgical urgency; gene therapy trial referral for newly diagnosed",
        "gene_family": "Cyclic nucleotide-gated (CNG) channel family — cone phototransduction (pore-forming alpha subunit)",
        "morphology": "Normal fundus early; foveal ONL thinning on OCT with age; flat photopic ERG on full-field ERG",
    },
    # -- GNAT2 -- Achromatopsia ACHM4 (~2-3%) -----------------------------------------------------
    {
        "gene": "GNAT2",
        "alt_name": (
            "GNAT2 (GNAT2-354aa-1p13.3 / AR -- "
            "ACHROMATOPSIA-ACHM4-~2-3pct-RARE -- "
            "CONE-TRANSDUCIN-ALPHA-2-SUBUNIT -- "
            "COMPLETE-ACHROMATOPSIA-LESS-SEVERE-NYSTAGMUS-IN-SOME -- "
            "GNAT2-LINKS-CONE-OPSIN-ACTIVATION-TO-PDE6-CASCADE)"
        ),
        "protein": (
            "GNAT2 -- 1p13.3 AR -- GNAT2-354aa -- "
            "Cone-Transducin-Alpha-2-Subunit-40kDa-Cone-Outer-Segment -- "
            "Heterotrimeric-G-Protein-Alpha-Subunit-GTP-Binding-GTPase -- "
            "Couples-Activated-Cone-Opsin-to-PDE6-Amplification-Step -- "
            "OMIM-Gene-139340-Disease-ACHM4-613856"
        ),
        "locus": "1p13.3",
        "protein_size": "354 aa / 40 kDa",
        "inheritance": (
            "AR (autosomal recessive); GNAT2 encodes cone transducin alpha-2 subunit, "
            "the Galpha subunit of the heterotrimeric G-protein transducin specific to cone photoreceptors; "
            "GNAT2 FUNCTION: cone phototransduction cascade amplification step; "
            "activated cone opsin -> GNAT2 GDP->GTP exchange -> GNAT2-GTP dissociates from betagamma -> "
            "GNAT2-GTP activates cone PDE6 (PDE6C/PDE6D alpha/delta) -> cGMP hydrolysis -> CNG channel closure; "
            "GNAT2 LOF -> signal blocked at transducin step -> PDE6 not activated -> "
            "cGMP remains high -> CNG channel stays open -> no photoreceptor hyperpolarisation -> achromatopsia; "
            "RARE: ~2-3% of all achromatopsia cases; "
            "NYSTAGMUS variably less severe than CNGB3/CNGA3 in some patients (unclear mechanism)"
        ),
        "disease_category": "Achromatopsia ACHM4 (AR) — rare ~2-3% of achromatopsia; GNAT2 cone transducin alpha-2; phototransduction block at transducin G-protein step; typically complete achromatopsia",
        "disease_pathway": (
            "GNAT2 encodes the alpha-2 subunit of cone transducin (Gt2), the heterotrimeric G-protein "
            "specific to cone photoreceptors. "
            "TRANSDUCIN CASCADE: activated cone opsin (R*) acts as guanine nucleotide exchange factor (GEF) "
            "for GNAT2; GDP displaced by GTP -> GNAT2-GTP released from betagamma complex -> "
            "GNAT2-GTP binds inhibitory gamma subunit of PDE6 (PDE6H) -> PDE6 catalytic subunits (PDE6C) freed -> "
            "cGMP hydrolysis -> CNG channel closes -> Ca2+ drops -> cone hyperpolarises. "
            "GTPase ACTIVITY: GNAT2 has intrinsic GTPase -> GTP->GDP -> GNAT2 inactivated -> signal terminates. "
            "GNAT2 LOF PATHOMECHANISM: signal transduction blocked between opsin activation and PDE6; "
            "cGMP not hydrolysed -> CNG channel cannot close -> cones unable to hyperpolarise -> "
            "photopic circuit silent -> total achromatopsia. "
            "LESS SEVERE NYSTAGMUS SUBSET: possibly residual signal via non-canonical pathway or "
            "GNAT2 partial function alleles."
        ),
        "pathognomonic": (
            "COMPLETE ACHROMATOPSIA (GNAT2/ACHM4): "
            "1. TOTAL COLOR BLINDNESS: all Ishihara plates failed; Nagel anomaloscope monochromat; "
            "2. NYSTAGMUS: typically pendular; may be LESS SEVERE than CNGB3/CNGA3 in some patients; "
            "3. PHOTOPHOBIA (DAY-BLINDNESS): present; FL-41 standard; "
            "4. VISUAL ACUITY ~0.1: stable; "
            "ERG: FLAT PHOTOPIC (absent cone ERG); NORMAL SCOTOPIC (rod response normal); "
            "OCT: foveal ONL thinning with age; "
            "RARE GENE: biallelic GNAT2 variants; functional testing important; "
            "LABORATORY: fibroblast/lymphocyte transducin functional assay (research); "
            "GENE PANEL INCLUSION: GNAT2 must be included in all achromatopsia gene panels."
        ),
        "treatment": (
            "FL-41 ROSE-TINTED LENSES: photophobia (standard); "
            "TINTED CONTACT LENSES: nystagmus amplitude reduction; "
            "REFRACTIVE CORRECTION: myopia/astigmatism; "
            "LOW VISION AIDS: magnification; preferential dim lighting; "
            "GENE THERAPY: no current clinical trial (rarer gene); preclinical work ongoing; "
            "AVOID: vitamin A supplements (no benefit); "
            "EDUCATIONAL SUPPORT: low vision accommodations; "
            "GENETIC COUNSELLING: AR 25% risk per sibling."
        ),
        "key_features": [
            "Complete achromatopsia: total color blindness (rare ~2-3%)",
            "GNAT2: cone transducin alpha-2; G-protein phototransduction step",
            "Nystagmus may be less severe than CNGB3/CNGA3 in some patients",
            "Flat photopic ERG; normal scotopic ERG",
            "Photophobia (day-blindness); FL-41 standard",
            "VA ~0.1 stable",
            "No current gene therapy trial (preclinical stage)",
            "Vitamin A: NO benefit",
        ],
        "key_ddx": (
            "CNGB3/CNGA3: clinically similar complete achromatopsia; ~75% of achromatopsia; gene panel mandatory; "
            "PDE6C (ACHM5): incomplete form common; residual color vision/photopic ERG; "
            "ATF6: foveal hypoplasia on OCT; macular atrophy; "
            "CSNB (congenital stationary night blindness): rod dysfunction ERG pattern; no photophobia; "
            "Leber congenital amaurosis (LCA): severe rod + cone; VA <0.1; nystagmus; RPE70/CEP290 etc."
        ),
        "systemic_involvement": False,
        "onset_age": "Congenital (nystagmus and photophobia from birth); stable VA",
        "surgical_urgency": "No acute surgical urgency",
        "gene_family": "Heterotrimeric G-protein alpha subunit family (transducin) — cone phototransduction amplification",
        "morphology": "Normal fundus early; foveal ONL thinning on OCT (age-related); flat photopic ERG",
    },
    # -- PDE6C -- Achromatopsia ACHM5 (Incomplete More Common) ------------------------------------
    {
        "gene": "PDE6C",
        "alt_name": (
            "PDE6C (PDE6C-858aa-10q23.33 / AR -- "
            "ACHROMATOPSIA-ACHM5-INCOMPLETE-ACHROMATOPSIA-MORE-COMMON -- "
            "CGMP-PHOSPHODIESTERASE-ALPHA-PRIME-CONE-SPECIFIC -- "
            "RESIDUAL-COLOR-VISION-PHOTOPIC-ERG-INCOMPLETE-FORM -- "
            "pALA146VAL-PAN-ETHNIC-PHOTOPHOBIA-DOMINANT-COMPLAINT)"
        ),
        "protein": (
            "PDE6C -- 10q23.33 AR -- PDE6C-858aa -- "
            "cGMP-Phosphodiesterase-6C-Alpha-Prime-Subunit-Cone-Specific-99kDa -- "
            "Cone-PDE6-Heterodimer-PDE6C-Alpha-Prime-PDE6D-Delta-Catalytic-Subunits -- "
            "GAF-A-GAF-B-Catalytic-Domain-Regulated-by-PDE6H-Gamma-Subunit -- "
            "OMIM-Gene-600827-Disease-ACHM5-613093"
        ),
        "locus": "10q23.33",
        "protein_size": "858 aa / 99 kDa",
        "inheritance": (
            "AR (autosomal recessive); PDE6C encodes the alpha' (alpha-prime) catalytic subunit of "
            "cone-specific cGMP phosphodiesterase (PDE6); "
            "CONE PDE6 STRUCTURE: heterodimer of PDE6C (alpha') + PDE6D (delta) catalytic subunits; "
            "regulated by PDE6H (gamma' inhibitory subunit); "
            "PDE6C FUNCTION: catalyses cGMP hydrolysis in cone outer segments following transducin (GNAT2) activation; "
            "rate-limiting amplification step; "
            "PDE6C LOF -> cGMP not hydrolysed -> CNG channel remains open -> cones cannot hyperpolarise; "
            "INCOMPLETE ACHROMATOPSIA MORE COMMON in PDE6C than in CNGB3/CNGA3: "
            "some missense variants retain partial PDE6C activity -> residual cone function -> "
            "residual color discrimination and partial photopic ERG; "
            "p.Ala146Val: pan-ethnic variant associated with incomplete form; "
            "PHOTOPHOBIA: often the dominant complaint, preceding visual acuity complaint"
        ),
        "disease_category": "Achromatopsia ACHM5 (AR) — incomplete form more common; PDE6C cone-specific cGMP PDE alpha' catalytic subunit; residual color vision and photopic ERG in incomplete form; photophobia dominant",
        "disease_pathway": (
            "PDE6C encodes the cone-specific alpha' catalytic subunit of phosphodiesterase 6 (PDE6). "
            "CONE PDE6 COMPLEX: PDE6C-PDE6D heterodimer, inhibited by PDE6H (two gamma' subunits per complex); "
            "activated GNAT2 (transducin) displaces PDE6H -> PDE6C catalytic site exposed -> "
            "cGMP hydrolysed to 5'-GMP -> cGMP falls -> CNG channel (CNGA3-CNGB3) closes -> "
            "Ca2+ falls -> cone recoverin releases guanylate cyclase activating protein (GCAP) -> "
            "guanylate cyclase resynthesises cGMP (recovery). "
            "PDE6C LOF: complete LOF -> phenotype similar to CNGB3/CNGA3 (complete achromatopsia); "
            "PARTIAL FUNCTION MISSENSE (p.Ala146Val and others): reduced but not absent PDE6 activity -> "
            "cGMP still rises in light but not as high -> partial CNG channel closure -> "
            "residual cone hyperpolarisation -> INCOMPLETE ACHROMATOPSIA (residual colour perception). "
            "PHOTOPHOBIA DOMINANCE: even with partial function, bright light overwhelms compromised PDE6 -> "
            "cGMP transiently unable to be hydrolysed rapidly -> cone overload -> pain/discomfort."
        ),
        "pathognomonic": (
            "ACHROMATOPSIA ACHM5 (PDE6C) -- INCOMPLETE FORM DISTINCTIVE FEATURES: "
            "RESIDUAL COLOR VISION: passes some Ishihara plates (especially high-contrast); "
            "Nagel anomaloscope: not full monochromat -- matches restricted range; "
            "PARTIAL PHOTOPIC ERG: some residual cone response (amplitude reduced but not flat); "
            "distinguishes incomplete form from complete CNGB3/CNGA3 achromatopsia; "
            "PHOTOPHOBIA: DOMINANT COMPLAINT -- often more prominent than acuity complaint; "
            "intense discomfort in bright light; squinting; avoids outdoors; "
            "VISUAL ACUITY: 0.1-0.3 (better than complete achromatopsia in incomplete form); "
            "NYSTAGMUS: may be milder/absent in incomplete form; "
            "COMPLETE FORM (null PDE6C): identical to CNGB3/CNGA3; gene panel distinguishes; "
            "p.Ala146Val on panel: partial function allele -> expect incomplete phenotype."
        ),
        "treatment": (
            "FL-41 ROSE-TINTED LENSES: photophobia primary management; "
            "TINTED CONTACT LENSES: additional photophobia relief + nystagmus (if present); "
            "REFRACTIVE CORRECTION: myopia/astigmatism common; "
            "LOW VISION AIDS: magnification; high contrast; dim lighting preferred; "
            "GENE THERAPY: preclinical (AAV-PDE6C); no current clinical trial; "
            "PHOTOPHOBIA MANAGEMENT PRIORITY: FL-41 + sunglasses + hats; indoor lighting control; "
            "AVOID: vitamin A supplements (no benefit); "
            "EDUCATIONAL SUPPORT: low vision accommodations; note photophobia may be more disabling than VA; "
            "GENETIC COUNSELLING: AR 25% risk; incomplete vs complete form genotype-phenotype correlation."
        ),
        "key_features": [
            "Incomplete achromatopsia more common (residual color vision, partial photopic ERG)",
            "Photophobia: dominant complaint (may exceed acuity as chief concern)",
            "p.Ala146Val pan-ethnic partial-function variant — incomplete phenotype",
            "Complete form: flat photopic ERG (null PDE6C variants)",
            "FL-41 lenses priority for photophobia management",
            "cGMP PDE6 alpha' cone-specific catalytic subunit",
            "No current gene therapy trial (preclinical stage)",
            "Vitamin A: NO benefit",
        ],
        "key_ddx": (
            "CNGB3/CNGA3 (complete ACHM): flat photopic ERG; total colour blindness; no residual; "
            "PDE6H (ACHM6): also incomplete form; smaller gamma' subunit; "
            "ATF6 (ACHM7): foveal hypoplasia on OCT; ERG variable; macular atrophy over time; "
            "Incomplete CSNB: rod ERG abnormal; different pattern; "
            "KCNV2 (CDSRR): supernormal rod ERG distinguishes; progressive macular dystrophy."
        ),
        "systemic_involvement": False,
        "onset_age": "Congenital to early childhood (nystagmus/photophobia from birth in complete form; photophobia prominent early in incomplete)",
        "surgical_urgency": "No acute surgical urgency",
        "gene_family": "cGMP phosphodiesterase family (PDE6) — cone-specific catalytic alpha' subunit",
        "morphology": "Foveal ONL thinning on OCT; partial photopic ERG (incomplete form) or flat (complete form)",
    },
    # -- PDE6H -- Achromatopsia ACHM6 (Incomplete) ------------------------------------------------
    {
        "gene": "PDE6H",
        "alt_name": (
            "PDE6H (PDE6H-84aa-12p13.33 / AR -- "
            "ACHROMATOPSIA-ACHM6-INCOMPLETE-MILD -- "
            "CGMP-PHOSPHODIESTERASE-GAMMA-PRIME-CONE-SPECIFIC-INHIBITORY-SUBUNIT -- "
            "RESIDUAL-COLOR-VISION-SOME-PHOTOPIC-ERG -- "
            "MILDEST-OF-ACHROMATOPSIA-GENES-SMALLEST-PDE6-SUBUNIT)"
        ),
        "protein": (
            "PDE6H -- 12p13.33 AR -- PDE6H-84aa -- "
            "cGMP-Phosphodiesterase-6H-Gamma-Prime-Subunit-Cone-Specific-Inhibitory-10kDa -- "
            "Smallest-PDE6-Subunit-Two-Per-PDE6C-PDE6D-Complex -- "
            "N-Terminal-Transducin-Binding-Domain-C-Terminal-PDEcat-Inhibitory-Region -- "
            "OMIM-Gene-601190-Disease-ACHM6-610024"
        ),
        "locus": "12p13.33",
        "protein_size": "84 aa / 10 kDa",
        "inheritance": (
            "AR (autosomal recessive); PDE6H encodes the gamma' (gamma-prime) inhibitory subunit "
            "of cone-specific cGMP phosphodiesterase; "
            "SMALLEST PDE6 SUBUNIT: 84 aa, ~10 kDa; two PDE6H subunits per cone PDE6 complex; "
            "PDE6H DUAL ROLE: "
            "(1) INHIBITORY: blocks catalytic site of PDE6C in darkness (prevents cGMP hydrolysis); "
            "(2) TRANSDUCIN SENSOR: N-terminal domain binds activated GNAT2-GTP -> "
            "GNAT2 competes with PDE6C active site -> displaces PDE6H -> PDE6C activated; "
            "PDE6H LOF PARADOX: loss of inhibitory subunit -> PDE6C CONSTITUTIVELY ACTIVE -> "
            "cGMP constitutively low -> CNG channel constitutively closed -> cone cannot respond to light; "
            "MILDER/INCOMPLETE phenotype: PDE6C still functional; constitutive activity may be partial; "
            "some residual cone function possible"
        ),
        "disease_category": "Achromatopsia ACHM6 (AR) — incomplete/mild; PDE6H cone-specific gamma' inhibitory subunit; unique mechanism (constitutive PDE6 activation); residual color vision; mildest achromatopsia gene",
        "disease_pathway": (
            "PDE6H encodes the gamma' inhibitory subunit, unique to cone PDE6 (cf. PDE6G for rods). "
            "INHIBITORY MECHANISM: PDE6H C-terminal polycationic region inserts into PDE6C active site -> "
            "blocks cGMP access -> PDE6C inhibited in darkness -> cGMP maintained high -> CNG open -> dark current. "
            "GNAT2 ACTIVATION: GNAT2-GTP binds PDE6H N-terminal domain -> displaces PDE6H from PDE6C active site -> "
            "PDE6C catalyses cGMP hydrolysis -> CNG closes -> cone hyperpolarises. "
            "PDE6H LOF: unique paradoxical mechanism -- "
            "WITHOUT INHIBITION, PDE6C constitutively active even in darkness -> "
            "cGMP constitutively low -> CNG channel constitutively closed -> "
            "no darkness 'baseline' for light response -> cones cannot signal any change; "
            "MILDER THAN COMPLETE ACHM: PDE6C activity may not be fully constitutive "
            "(some residual cGMP/CNG function) -> incomplete achromatopsia."
        ),
        "pathognomonic": (
            "ACHROMATOPSIA ACHM6 (PDE6H) -- MILDER INCOMPLETE PHENOTYPE: "
            "RESIDUAL COLOR VISION: passes some Ishihara plates; partial color discrimination; "
            "PARTIAL PHOTOPIC ERG: some residual cone response (not completely flat); "
            "NYSTAGMUS: may be mild or absent; "
            "VISUAL ACUITY: 0.1-0.4 (better than complete achromatopsia); "
            "PHOTOPHOBIA: present but may be milder; "
            "UNIQUE MECHANISM: constitutive cone PDE6 activation (not blocked PDE6); "
            "GENE PANEL: PDE6H included in all achromatopsia/photoreceptor panels; "
            "CONTRAST WITH PDE6C (ACHM5): PDE6C = catalytic subunit LOF (blocked PDE6); "
            "PDE6H = inhibitory subunit LOF (constitutive PDE6 activation) -- opposite mechanisms, similar incomplete result."
        ),
        "treatment": (
            "FL-41 ROSE-TINTED LENSES: photophobia management; "
            "TINTED CONTACT LENSES: photophobia + nystagmus if present; "
            "REFRACTIVE CORRECTION: myopia/astigmatism correction; "
            "LOW VISION AIDS: magnification; preferential dim lighting; "
            "GENE THERAPY: preclinical; no current clinical trial; "
            "AVOID: vitamin A supplements (no benefit); "
            "EDUCATIONAL SUPPORT: low vision accommodations; note potentially milder disability; "
            "GENETIC COUNSELLING: AR 25% risk; milder phenotype expectation with PDE6H."
        ),
        "key_features": [
            "Incomplete/mild achromatopsia (mildest achromatopsia gene)",
            "Residual color vision; partial photopic ERG",
            "Unique mechanism: constitutive PDE6 activation (loss of inhibitory subunit)",
            "PDE6H: smallest PDE6 subunit (84 aa / 10 kDa); cone-specific gamma' inhibitory",
            "Nystagmus mild or absent; photophobia variable",
            "VA 0.1-0.4 (better than complete achromatopsia)",
            "No current gene therapy trial",
            "Vitamin A: NO benefit",
        ],
        "key_ddx": (
            "PDE6C (ACHM5): incomplete form also; PDE6C = catalytic LOF (blocked); PDE6H = inhibitory LOF (constitutive); "
            "CNGB3/CNGA3 (complete ACHM): flat photopic ERG; total color blindness; "
            "ATF6 (ACHM7): foveal hypoplasia on OCT; macular atrophy; "
            "KCNV2: supernormal rod ERG; progressive; "
            "Stargardt: progressive macular dystrophy; dark choroid 80%; ABCA4."
        ),
        "systemic_involvement": False,
        "onset_age": "Congenital to early childhood; often milder nystagmus; photophobia early",
        "surgical_urgency": "No acute surgical urgency",
        "gene_family": "cGMP phosphodiesterase family (PDE6) — cone-specific gamma' inhibitory subunit",
        "morphology": "Foveal ONL thinning on OCT (may be mild); partial photopic ERG; residual cone response",
    },
    # -- ATF6 -- Achromatopsia ACHM7 / Macular Hypoplasia -----------------------------------------
    {
        "gene": "ATF6",
        "alt_name": (
            "ATF6 (ATF6-670aa-1q23.2 / AR -- "
            "ACHROMATOPSIA-ACHM7-MACULAR-HYPOPLASIA-ER-STRESS-TRANSCRIPTION-FACTOR -- "
            "FOVEAL-HYPOPLASIA-ON-OCT-PRIMARY-DISTINGUISHING-FEATURE-vs-CNGB3-CNGA3 -- "
            "pARG324CYS-pGLY48GLU-FOUNDERS -- "
            "MACULAR-ATROPHY-DEVELOPS-CONE-ERG-LESS-EXTINGUISHED-EARLY)"
        ),
        "protein": (
            "ATF6 -- 1q23.2 AR -- ATF6-670aa -- "
            "Activating-Transcription-Factor-6-Alpha-75kDa-ER-Stress-Sensor-UPR-Transcription-Factor -- "
            "Type-II-ER-Transmembrane-Protein-N-Terminal-bZIP-TF-Domain-C-Terminal-ER-Lumen-Sensor -- "
            "Golgi-Processing-S1P-S2P-Cleavage-Releases-N-terminal-TF -- "
            "OMIM-Gene-605537-Disease-ACHM7-616517"
        ),
        "locus": "1q23.2",
        "protein_size": "670 aa / 75 kDa",
        "inheritance": (
            "AR (autosomal recessive); ATF6 encodes Activating Transcription Factor 6 alpha (ATF6alpha), "
            "a type II ER transmembrane protein and UPR (unfolded protein response) sensor/transcription factor; "
            "ATF6 IS MECHANISTICALLY DISTINCT FROM CNGB3/CNGA3/GNAT2/PDE6C/PDE6H: "
            "NOT a phototransduction cascade gene; instead an ER proteostasis/UPR gene; "
            "NORMAL ATF6 FUNCTION: ER stress -> ATF6 alpha Golgi translocation -> "
            "S1P/S2P proteolytic cleavage -> N-terminal bZIP domain released -> "
            "nuclear import -> transcribes ER chaperones (BiP/GRP78, GRP94, PDI) to resolve ER stress; "
            "ATF6 LOF: impaired UPR -> ER stress unresolved -> FOVEAL CONE APOPTOSIS; "
            "FOUNDER VARIANTS: p.Arg324Cys; p.Gly48Glu (multiple ethnic backgrounds); "
            "DISTINCT OCT PHENOTYPE: foveal hypoplasia (structural) as primary finding"
        ),
        "disease_category": "Achromatopsia ACHM7 / Macular hypoplasia (AR) — distinct from CNG/PDE6 achromatopsia; ATF6 ER stress UPR transcription factor; foveal hypoplasia on OCT (primary distinguishing feature); macular atrophy develops; cone ERG less extinguished early",
        "disease_pathway": (
            "ATF6 is one of three UPR sensors in the ER membrane (others: IRE1, PERK). "
            "UPR ACTIVATION: misfolded proteins accumulate in ER lumen -> GRP78/BiP dissociates from ATF6 -> "
            "ATF6 translocates to Golgi -> S1P cleaves at luminal site 1 -> "
            "S2P cleaves at TM domain (site 2) -> ATF6N (N-terminal bZIP fragment) released -> "
            "nuclear translocation -> activates ERSE (ER stress response element) promoters -> "
            "transcribes ER chaperones (BiP, GRP94, PDI, CHOP) -> ER stress resolved. "
            "ATF6 IN FOVEAL CONES: foveal cones have exceptionally high protein synthesis rates "
            "(dense opsin packing); highest UPR dependency; "
            "ATF6 LOF -> UPR failure in foveal cones -> unresolved ER stress -> "
            "FOVEAL CONE APOPTOSIS during development -> FOVEAL HYPOPLASIA. "
            "MACULAR ATROPHY: progressive RPE/photoreceptor atrophy at fovea over decades; "
            "CONE ERG: less extinguished early (some parafoveal cones survive) -- "
            "distinguishes from complete CNGB3/CNGA3 achromatopsia."
        ),
        "pathognomonic": (
            "ACHROMATOPSIA ACHM7 (ATF6) -- DISTINGUISHING FEATURES vs CNGB3/CNGA3: "
            "1. FOVEAL HYPOPLASIA ON OCT: PRIMARY DISTINGUISHING FEATURE -- "
            "absent or hypoplastic foveal pit; foveal ONL absent or markedly thinned from early age; "
            "inner retinal layers extend into fovea (lack of foveal avascular zone development); "
            "CNGB3/CNGA3: fovea initially near-normal on OCT (functional loss without structural loss early); "
            "2. CONE ERG: LESS COMPLETELY EXTINGUISHED than CNGB3/CNGA3 in early years -- "
            "some residual photopic signal from surviving parafoveal cones; "
            "3. MACULAR ATROPHY PROGRESSION: RPE and photoreceptor loss at macula develops over time; "
            "4. TOTAL COLOR BLINDNESS: present; pendular nystagmus; photophobia; VA 0.05-0.2; "
            "FOUNDER VARIANTS: p.Arg324Cys, p.Gly48Glu; "
            "ERG + OCT COMBINATION: less extinguished ERG + foveal hypoplasia = ATF6 signature."
        ),
        "treatment": (
            "FL-41 ROSE-TINTED LENSES: photophobia management (standard); "
            "TINTED CONTACT LENSES: nystagmus + photophobia; "
            "REFRACTIVE CORRECTION: myopia/astigmatism; "
            "LOW VISION AIDS: magnification; high contrast; "
            "MACULAR ATROPHY MONITORING: annual OCT to track foveal atrophy progression; "
            "GENE THERAPY: preclinical; UPR rescue by AAV-ATF6 under investigation; "
            "AVOID: vitamin A supplements (no benefit); "
            "ANTI-VEGF: for potential CNV if macular atrophy progresses to choroidal neovascularisation; "
            "GENETIC COUNSELLING: AR 25% risk; p.Arg324Cys and p.Gly48Glu testing in at-risk families."
        ),
        "key_features": [
            "Foveal hypoplasia on OCT — PRIMARY distinguishing feature from CNGB3/CNGA3",
            "Cone ERG less completely extinguished early (parafoveal cones surviving)",
            "Macular atrophy progressive (RPE + photoreceptor loss at fovea over time)",
            "Distinct mechanism: ER stress/UPR (NOT phototransduction cascade gene)",
            "Total color blindness; pendular nystagmus; photophobia",
            "p.Arg324Cys and p.Gly48Glu founder variants",
            "ATF6alpha: Golgi-processed ER stress transcription factor",
            "Vitamin A: NO benefit",
        ],
        "key_ddx": (
            "CNGB3/CNGA3 (ACHM3/2): fovea near-normal on OCT early (no hypoplasia); flat photopic ERG; "
            "OPN1LW (BCM): also foveal hypoplasia; X-linked males; S-cones preserved; myopia 90%; "
            "Albinism (OCA/OA): foveal hypoplasia + iris transillumination + pigment deficiency + nystagmus; "
            "Leber congenital amaurosis (LCA): severe rod+cone; RPE70/CEP290; "
            "Macular dystrophy: progressive bilateral macular loss; later onset; different ERG."
        ),
        "systemic_involvement": False,
        "onset_age": "Congenital (nystagmus, photophobia, foveal hypoplasia from birth); macular atrophy progressive",
        "surgical_urgency": "No acute surgical urgency; annual OCT monitoring for macular atrophy progression",
        "gene_family": "UPR/ER stress transcription factor family (ATF/CREB superfamily) — ER proteostasis (NOT phototransduction)",
        "morphology": "Foveal hypoplasia on OCT (absent foveal pit); macular atrophy progressive; less extinguished photopic ERG early",
    },
    # -- KCNV2 -- Cone Dystrophy with Supernormal Rod ERG (CDSRR) ---------------------------------
    {
        "gene": "KCNV2",
        "alt_name": (
            "KCNV2 (KCNV2-545aa-9p24.2 / AR -- "
            "CONE-DYSTROPHY-SUPERNORMAL-ROD-ERG-CDSRR -- "
            "PATHOGNOMONIC-SUPERNORMAL-ROD-ERG-B-WAVE-DARK-ADAPTED-BRIGHT-FLASH -- "
            "PROGRESSIVE-UNLIKE-STATIC-ACHROMATOPSIA -- "
            "KV8-2-VOLTAGE-GATED-K-CHANNEL-KCNV-SUBFAMILY-SILENT-MODIFIER)"
        ),
        "protein": (
            "KCNV2 -- 9p24.2 AR -- KCNV2-545aa -- "
            "Kv8.2-Voltage-Gated-Potassium-Channel-Subfamily-V-Member-2-62kDa -- "
            "Silent-Modifier-Subunit-Heterotetramerises-With-Kv2.1/Kv2.2-KCNB1-KCNB2 -- "
            "Modifies-Activation-Inactivation-Kinetics-Of-Kv2-Channels-In-Photoreceptors -- "
            "OMIM-Gene-607604-Disease-CDSRR-610356"
        ),
        "locus": "9p24.2",
        "protein_size": "545 aa / 62 kDa",
        "inheritance": (
            "AR (autosomal recessive); KCNV2 encodes Kv8.2, a voltage-gated K+ channel subunit "
            "of the KCNV (silent modifier) subfamily; "
            "Kv8.2 DOES NOT FORM HOMOTETRAMERIC CHANNELS: it is a 'silent modifier' subunit -- "
            "cannot form channels alone; instead heterotetramerises with Kv2.1 (KCNB1) or Kv2.2 (KCNB2); "
            "EXPRESSION: cone and rod photoreceptors inner segments; "
            "KCNV2 MODIFIES Kv2 channel kinetics: shifts activation voltage; modifies inactivation; "
            "KCNV2 LOF -> Kv2.1/Kv2.2 homotetramers with altered kinetics -> "
            "abnormal rod photoreceptor repolarisation -> SUPERNORMAL ROD ERG b-wave (paradoxical); "
            "Progressive cone > rod dystrophy: macular dystrophy evolves in adulthood"
        ),
        "disease_category": "Cone Dystrophy with Supernormal Rod ERG (CDSRR) (AR) — KCNV2 Kv8.2 voltage-gated K+ channel; pathognomonic supernormal rod ERG b-wave; PROGRESSIVE macular dystrophy; distinct from static achromatopsia",
        "disease_pathway": (
            "KCNV2 encodes Kv8.2, a 'silent modifier' subunit of voltage-gated K+ (Kv) channels. "
            "SILENT MODIFIER CONCEPT: Kv8.2 has no independent channel activity; "
            "co-assembles with Kv2.1 (KCNB1) or Kv2.2 (KCNB2) in photoreceptors -> "
            "Kv2.1/Kv8.2 or Kv2.2/Kv8.2 heterotetramers. "
            "NORMAL FUNCTION: Kv2.1-Kv8.2 heterotetramers have RIGHT-SHIFTED activation voltage "
            "compared to Kv2.1 homotetramers -> precise control of photoreceptor repolarisation "
            "following light-induced hyperpolarisation. "
            "KCNV2 LOF MECHANISM: Kv8.2 absent -> Kv2.1/Kv2.2 form homotetramers -> "
            "left-shifted activation; faster/stronger repolarisation in rods -> "
            "ROD a-wave normal, but b-wave (Muller cell / ON-bipolar depolarisation) SUPERNORMAL -- "
            "paradoxically enhanced because of abnormal timing/kinetics of rod photoreceptor repolarisation; "
            "CONE DYSFUNCTION: cones more severely affected than rods -> "
            "photophobia, reduced acuity, absent photopic ERG -> progressive macular dystrophy."
        ),
        "pathognomonic": (
            "CDSRR (KCNV2) -- PATHOGNOMONIC FINDING: "
            "SUPERNORMAL ROD ERG B-WAVE: dark-adapted 10 candela (bright flash) ERG -> "
            "b-wave amplitude EXCEEDS NORMAL UPPER LIMIT (>800 microvolts in typical adults); "
            "this is the SINGLE MOST IMPORTANT DIAGNOSTIC TEST for KCNV2; "
            "dark-adapted dim flash ERG may be normal or mildly abnormal; "
            "BRIGHT FLASH is MANDATORY to elicit supernormal b-wave (dim flash may not show it); "
            "PHOTOPIC ERG: reduced or absent (cone dysfunction more severe than rods); "
            "PROGRESSIVE DISEASE (critical DDx from static achromatopsia): "
            "macular dystrophy evolves in 3rd-4th decade; bull's eye maculopathy may develop; "
            "VA deteriorates progressively (not stable like CNGB3/CNGA3 achromatopsia); "
            "PHOTOPHOBIA: moderate; REDUCED VA: 0.1-0.4 initially, progressive decline; "
            "MACULAR OCT: progressive foveal RPE/photoreceptor atrophy; "
            "FUNDUS: progressive macular dystrophy changes (mottling, granularity, bull's eye)."
        ),
        "treatment": (
            "FL-41 ROSE-TINTED LENSES: photophobia management; "
            "REFRACTIVE CORRECTION: myopia/astigmatism; "
            "LOW VISION AIDS: magnification; progressive low vision rehabilitation (disease is progressive); "
            "PROGRESSION MONITORING: annual OCT + photopic ERG + visual field; "
            "ALL-TRANS-RETINOIC ACID (ATRA): potential therapeutic (preclinical evidence for KCNV2); "
            "GENE THERAPY: preclinical; AAV-KCNV2 under development; "
            "DRIVING ADVICE: progressive loss -> regular DVLA/licence reassessment; "
            "ANTI-VEGF: if choroidal neovascularisation develops at macular atrophy edge; "
            "AVOID: vitamin A supplements (uncertain benefit; ATRA different from dietary vitamin A); "
            "GENETIC COUNSELLING: AR 25% risk; PROGRESSIVE prognosis counselling (not static like achromatopsia)."
        ),
        "key_features": [
            "Supernormal rod ERG b-wave (dark-adapted bright flash 10cd) — PATHOGNOMONIC",
            "PROGRESSIVE disease (macular dystrophy develops) — unlike static achromatopsia",
            "Photopic ERG reduced/absent (cone > rod dysfunction)",
            "KCNV2: Kv8.2 silent modifier K+ channel; co-assembles with Kv2.1/Kv2.2",
            "Bull's eye maculopathy develops in 3rd-4th decade",
            "Annual OCT + ERG monitoring mandatory (progressive)",
            "All-trans-retinoic acid (ATRA): potential therapeutic",
            "Bright flash ERG mandatory (supernormal b-wave requires 10cd stimulus)",
        ],
        "key_ddx": (
            "CNGB3/CNGA3 (complete ACHM): STATIC not progressive; no supernormal rod ERG; flat photopic ERG; "
            "Stargardt (ABCA4): progressive macular dystrophy; dark choroid 80%; rod ERG normal (not supernormal); "
            "PDE6C/PDE6H incomplete ACHM: residual color; no supernormal rod ERG; "
            "CSNB1 (NYX): night blindness + normal photopic ERG; no supernormal rod ERG; "
            "Oguchi disease: Mizuo-Nakamura phenomenon; different ERG pattern."
        ),
        "systemic_involvement": False,
        "onset_age": "Symptoms from early childhood (photophobia, nystagmus); progressive visual loss from 3rd decade",
        "surgical_urgency": "No acute surgical urgency; progressive disease requires regular ophthalmology follow-up",
        "gene_family": "Voltage-gated K+ channel silent modifier subfamily (KCNV) — photoreceptor repolarisation",
        "morphology": "Progressive macular dystrophy; bull's eye maculopathy (advanced); supernormal rod ERG b-wave on bright flash",
    },
    # -- OPN1LW -- Blue Cone Monochromatism (BCM) -------------------------------------------------
    {
        "gene": "OPN1LW",
        "alt_name": (
            "OPN1LW (OPN1LW-364aa-Xq28 / XLR-AD -- "
            "BLUE-CONE-MONOCHROMATISM-BCM-X-LINKED-ONLY-S-CONES-RODS-FUNCTIONAL-L-M-CONES-ABSENT -- "
            "GENE-ARRAY-DELETION-REARRANGEMENT-LCR-STANDARD-WES-MISSES-NEED-TARGETED-GENE-ARRAY-TESTING -- "
            "FOVEAL-HYPOPLASIA-OCT-MYOPIA-90pct -- "
            "CONGENITAL-STATIONARY-NOT-PROGRESSIVE)"
        ),
        "protein": (
            "OPN1LW -- Xq28 XLR -- OPN1LW-364aa -- "
            "L-Cone-Long-Wave-Sensitive-Opsin-41kDa-GPCR-7TM-Cone-Photopigment -- "
            "Tandem-Gene-Array-LCR-OPN1LW-OPN1MW-Locus-Control-Region-Dependent-Expression -- "
            "LCR-Deletion-Rearrangement-Eliminates-L-and-M-Cone-Opsins -- "
            "OMIM-Gene-300822-Disease-BCM-303700"
        ),
        "locus": "Xq28",
        "protein_size": "364 aa / 41 kDa",
        "inheritance": (
            "XLR (X-linked recessive) for Blue Cone Monochromatism (BCM); "
            "OPN1LW AND OPN1MW (L- and M-cone opsins) are adjacent genes in tandem array at Xq28 "
            "regulated by a SHARED LOCUS CONTROL REGION (LCR) upstream; "
            "BCM MECHANISM: deletion or rearrangement affecting LCR or both OPN1LW and OPN1MW -> "
            "loss of L-cone AND M-cone opsin expression -> BOTH L and M cones non-functional -> "
            "ONLY S-cones (blue) and rods remain functional = blue cone monochromat; "
            "CRITICAL DIAGNOSTIC POINT: STANDARD WES/EXOME SEQUENCING MISSES BCM -- "
            "LCR deletions/rearrangements are copy-number variants not captured by standard NGS; "
            "NEED: targeted OPN1LW/OPN1MW gene array analysis (Southern blot, MLPA, long-range PCR); "
            "X-LINKED: males fully affected (hemizygous); carrier females usually unaffected (random X-inactivation); "
            "AD form: rare; single-gene missense affecting OPN1LW with secondary M-cone loss; "
            "CONGENITAL STATIONARY: not progressive (unlike KCNV2)"
        ),
        "disease_category": "Blue Cone Monochromatism BCM (XLR) — OPN1LW/OPN1MW LCR deletion; only S-cones + rods functional; L + M cones absent; standard WES misses diagnosis; foveal hypoplasia on OCT; myopia 90%; congenital stationary (non-progressive); males affected",
        "disease_pathway": (
            "OPN1LW (L-opsin) and OPN1MW (M-opsin) genes lie in tandem at Xq28 under shared "
            "transcriptional control of a LOCUS CONTROL REGION (LCR) ~3.5 kb upstream. "
            "NORMAL EXPRESSION: LCR drives expression of first gene in array (OPN1LW) in L-cones "
            "and subsequent array gene (OPN1MW) in M-cones via stochastic promoter-LCR looping. "
            "BCM MUTATIONS: "
            "(1) LCR DELETION: most common; deletes LCR -> both OPN1LW and OPN1MW transcription lost -> "
            "all L and M cones fail to develop functional opsins -> L/M cone apoptosis; "
            "(2) ARRAY REARRANGEMENT: unequal crossover -> hybrid gene or single-gene array with deleterious "
            "missense (e.g. Cys203Arg in exon 4 or exon 3 missense) -> non-functional opsin -> cone death; "
            "RESULT: only S-cones (blue; OPN1SW at 7q32, different chromosome) and rods functional; "
            "FOVEAL HYPOPLASIA: L/M cones essential for foveal pit formation; their absence -> "
            "disrupted foveal development -> hypoplastic fovea (similar to albinism); "
            "MYOPIA: 90% prevalence -- L/M cone absence -> altered emmetropisation; "
            "STATIONARY: S-cones and rods stable; no progressive degeneration."
        ),
        "pathognomonic": (
            "BLUE CONE MONOCHROMATISM (OPN1LW/BCM) -- KEY DIAGNOSTIC FEATURES: "
            "1. MALES ONLY (X-linked): FEMALES UNAFFECTED (carrier); "
            "2. S-CONE MONOCHROMAT: passes ONLY blue-yellow discrimination; "
            "Nagel anomaloscope: cannot match red-green (absent L/M cones); "
            "passes short-wavelength (S-cone) tests; tritanopia ABSENT (has blue); "
            "3. FOVEAL HYPOPLASIA ON OCT: absent/reduced foveal pit (similar to ATF6 and albinism); "
            "4. MYOPIA ~90%: common; may be high myopia; "
            "5. PENDULAR NYSTAGMUS: typically present in infancy; IMPROVES with age (unlike CNGB3/CNGA3 which persists); "
            "6. PHOTOPHOBIA: present; "
            "7. VA: 0.05-0.2; stable (non-progressive); "
            "ERG: absent photopic L/M response; S-cone ERG (blue flash) relatively preserved; "
            "CRITICAL DIAGNOSTIC TRAP: STANDARD WES MISSES BCM -- "
            "must specifically request OPN1LW/OPN1MW gene array analysis / MLPA / Southern blot; "
            "many BCM patients go undiagnosed until targeted testing performed."
        ),
        "treatment": (
            "FL-41 ROSE-TINTED LENSES: photophobia management; "
            "TINTED CONTACT LENSES: nystagmus reduction (contact lenses reduce nystagmus amplitude via null point shift); "
            "MYOPIA CORRECTION: spectacles/contact lenses mandatory; "
            "LOW VISION AIDS: magnification; preferential seating; dim lighting; "
            "CONGENITAL STATIONARY REASSURANCE: counsel patient/family that vision is STABLE (not progressive); "
            "NYSTAGMUS REASSURANCE: improves with age (key point for parents); "
            "CARRIER TESTING: test daughters (X-linked); sons 50% risk of BCM; "
            "GENETIC TESTING GUIDANCE: ensure OPN1LW gene array testing (NOT just standard WES); "
            "GENE THERAPY: preclinical; gene delivery complex due to LCR architecture; "
            "AVOID: vitamin A supplements (no benefit); "
            "EDUCATIONAL SUPPORT: low vision accommodations; stable prognosis for planning."
        ),
        "key_features": [
            "X-linked recessive — males only fully affected; carrier females unaffected",
            "Only S-cones (blue) + rods functional; L and M cones absent",
            "Standard WES MISSES diagnosis — need targeted OPN1LW/OPN1MW gene array (MLPA/Southern blot)",
            "Foveal hypoplasia on OCT (similar to ATF6 and albinism)",
            "Myopia ~90% (may be high myopia)",
            "Pendular nystagmus in infancy — IMPROVES with age (vs CNGB3/CNGA3 persists)",
            "Congenital STATIONARY — not progressive (stable VA)",
            "Contact lenses reduce nystagmus amplitude (null point shift)",
        ],
        "key_ddx": (
            "ATF6 (ACHM7): also foveal hypoplasia; total colour blindness (no S-cone preservation); AR both sexes; "
            "CNGB3/CNGA3 complete ACHM: flat photopic ERG including S-cone; nystagmus persists; no myopia prominence; "
            "Albinism (OCA/OA): foveal hypoplasia + iris transillumination + pigmentary phenotype; "
            "CSNB: night blindness; normal photopic ERG; Xp11.4 (CACNA1F/NYX); "
            "Leber congenital amaurosis: severe rod+cone; <0.1 VA; RPE70/CEP290 etc."
        ),
        "systemic_involvement": False,
        "onset_age": "Congenital (nystagmus from infancy, improves with age); VA stable; myopia progressive in childhood",
        "surgical_urgency": "No acute surgical urgency; myopia progression needs regular refraction",
        "gene_family": "Opsin GPCR family (OPN1LW L-cone opsin) — tandem array Xq28 with LCR regulation",
        "morphology": "Foveal hypoplasia on OCT; absent L/M cone photopic ERG; S-cone ERG relatively preserved; myopic fundus",
    },
]


def _make_cohort(entry: dict, seed: int) -> list:
    """Generate 40 synthetic patients for one gene cohort."""
    rng = random.Random(seed)
    patients = []
    gene = entry["gene"]
    for i in range(40):
        # Visual acuity (poor = worse than 0.2 / 6/30)
        if gene in ("CNGB3", "CNGA3", "GNAT2"):
            va_poor = rng.random() < 0.90  # complete achromatopsia ~VA 0.1
        elif gene == "ATF6":
            va_poor = rng.random() < 0.85  # VA often 0.05-0.2 (foveal hypoplasia)
        elif gene == "OPN1LW":
            va_poor = rng.random() < 0.80  # VA 0.05-0.2 stable
        elif gene == "KCNV2":
            va_poor = rng.random() < 0.60  # progressive; initially 0.1-0.4
        elif gene == "PDE6C":
            va_poor = rng.random() < 0.55  # incomplete form: better VA
        else:  # PDE6H
            va_poor = rng.random() < 0.40  # mildest; VA 0.1-0.4

        # Total color blindness (fails all Ishihara)
        if gene in ("CNGB3", "CNGA3", "GNAT2", "ATF6", "OPN1LW"):
            total_color_blind = True
        elif gene == "PDE6C":
            total_color_blind = rng.random() < 0.50  # complete vs incomplete
        elif gene == "PDE6H":
            total_color_blind = rng.random() < 0.30  # usually incomplete
        elif gene == "KCNV2":
            total_color_blind = rng.random() < 0.70  # significant color defect
        else:
            total_color_blind = False

        # Pendular nystagmus
        if gene in ("CNGB3", "CNGA3", "GNAT2"):
            nystagmus = True  # always present, persists
        elif gene == "OPN1LW":
            nystagmus = rng.random() < 0.85  # present in infancy, improves
        elif gene == "ATF6":
            nystagmus = rng.random() < 0.80
        elif gene == "PDE6C":
            nystagmus = rng.random() < 0.70  # complete > incomplete
        elif gene == "PDE6H":
            nystagmus = rng.random() < 0.50  # mild/absent in incomplete
        elif gene == "KCNV2":
            nystagmus = rng.random() < 0.40  # less prominent
        else:
            nystagmus = False

        # Photophobia (day-blindness)
        if gene in ("CNGB3", "CNGA3", "GNAT2", "ATF6", "OPN1LW"):
            photophobia = True
        elif gene == "PDE6C":
            photophobia = True  # dominant complaint
        elif gene == "PDE6H":
            photophobia = rng.random() < 0.75  # variable milder
        elif gene == "KCNV2":
            photophobia = rng.random() < 0.70
        else:
            photophobia = False

        # Photopic ERG absent (flat)
        if gene in ("CNGB3", "CNGA3", "GNAT2"):
            flat_photopic_erg = True  # complete achromatopsia
        elif gene == "ATF6":
            flat_photopic_erg = rng.random() < 0.55  # less extinguished early
        elif gene == "OPN1LW":
            flat_photopic_erg = rng.random() < 0.80  # L/M cone absent
        elif gene == "KCNV2":
            flat_photopic_erg = rng.random() < 0.75  # cone dominant
        elif gene == "PDE6C":
            flat_photopic_erg = rng.random() < 0.50  # complete vs incomplete
        elif gene == "PDE6H":
            flat_photopic_erg = rng.random() < 0.30  # milder
        else:
            flat_photopic_erg = False

        # Supernormal rod ERG b-wave (KCNV2 only)
        supernormal_rod_erg = (gene == "KCNV2") and (rng.random() < 0.95)

        # Foveal hypoplasia on OCT
        if gene in ("ATF6", "OPN1LW"):
            foveal_hypoplasia = True
        elif gene in ("CNGB3", "CNGA3", "GNAT2"):
            foveal_hypoplasia = rng.random() < 0.10  # subtle/late
        else:
            foveal_hypoplasia = rng.random() < 0.05

        # Myopia (OPN1LW ~90%)
        if gene == "OPN1LW":
            myopia = rng.random() < 0.90
        elif gene in ("ATF6", "CNGB3", "CNGA3"):
            myopia = rng.random() < 0.30
        elif gene == "KCNV2":
            myopia = rng.random() < 0.25
        else:
            myopia = rng.random() < 0.20

        # Progressive macular dystrophy (KCNV2 primarily)
        if gene == "KCNV2":
            progressive_macular = rng.random() < 0.80
        elif gene == "ATF6":
            progressive_macular = rng.random() < 0.60  # macular atrophy
        else:
            progressive_macular = rng.random() < 0.10

        patients.append({
            "id": f"{gene}-{seed}-{i+1:03d}",
            "gene": gene,
            "va_poor": va_poor,
            "total_color_blind": total_color_blind,
            "nystagmus": nystagmus,
            "photophobia": photophobia,
            "flat_photopic_erg": flat_photopic_erg,
            "supernormal_rod_erg": supernormal_rod_erg,
            "foveal_hypoplasia": foveal_hypoplasia,
            "myopia": myopia,
            "progressive_macular": progressive_macular,
            "inheritance": entry["inheritance"].split(";")[0].strip(),
        })
    return patients


def generate_overview():
    all_patients = []
    for idx, entry in enumerate(CV_GENES):
        all_patients.extend(_make_cohort(entry, SEED_BASE + idx))

    total = len(all_patients)
    va_poor_count           = sum(1 for p in all_patients if p["va_poor"])
    total_color_blind_count = sum(1 for p in all_patients if p["total_color_blind"])
    nystagmus_count         = sum(1 for p in all_patients if p["nystagmus"])
    photophobia_count       = sum(1 for p in all_patients if p["photophobia"])
    flat_photopic_count     = sum(1 for p in all_patients if p["flat_photopic_erg"])
    supernormal_rod_count   = sum(1 for p in all_patients if p["supernormal_rod_erg"])
    foveal_hypo_count       = sum(1 for p in all_patients if p["foveal_hypoplasia"])
    myopia_count            = sum(1 for p in all_patients if p["myopia"])
    progressive_mac_count   = sum(1 for p in all_patients if p["progressive_macular"])

    gene_summary = {}
    for idx, entry in enumerate(CV_GENES):
        gene = entry["gene"]
        cohort = _make_cohort(entry, SEED_BASE + idx)
        gene_summary[gene] = {
            "gene":                  gene,
            "alt_name":              entry["alt_name"],
            "locus":                 entry["locus"],
            "protein_size":          entry["protein_size"],
            "inheritance":           entry["inheritance"].split(";")[0].strip(),
            "disease_category":      entry["disease_category"],
            "pathognomonic":         entry["pathognomonic"][:300],
            "morphology":            entry["morphology"],
            "systemic_involvement":  entry["systemic_involvement"],
            "onset_age":             entry["onset_age"],
            "surgical_urgency":      entry["surgical_urgency"],
            "gene_family":           entry["gene_family"],
            "n_patients":            len(cohort),
            "va_poor_pct":           round(100 * sum(1 for p in cohort if p["va_poor"])               / len(cohort), 1),
            "total_color_blind_pct": round(100 * sum(1 for p in cohort if p["total_color_blind"])     / len(cohort), 1),
            "nystagmus_pct":         round(100 * sum(1 for p in cohort if p["nystagmus"])             / len(cohort), 1),
            "photophobia_pct":       round(100 * sum(1 for p in cohort if p["photophobia"])           / len(cohort), 1),
            "flat_photopic_pct":     round(100 * sum(1 for p in cohort if p["flat_photopic_erg"])     / len(cohort), 1),
            "supernormal_rod_pct":   round(100 * sum(1 for p in cohort if p["supernormal_rod_erg"])   / len(cohort), 1),
            "foveal_hypoplasia_pct": round(100 * sum(1 for p in cohort if p["foveal_hypoplasia"])     / len(cohort), 1),
            "myopia_pct":            round(100 * sum(1 for p in cohort if p["myopia"])                / len(cohort), 1),
            "progressive_mac_pct":   round(100 * sum(1 for p in cohort if p["progressive_macular"])   / len(cohort), 1),
        }

    return {
        "atlas":          "Hereditary-Color-Vision-And-Photoreceptor-Disorder-Atlas",
        "subtitle":       (
            "Complete 8-Gene Hereditary Color Vision & Photoreceptor Disorder Reference -- "
            "CNGB3/CNGA3/GNAT2/PDE6C/PDE6H/ATF6/KCNV2/OPN1LW"
        ),
        "genes_covered":  [e["gene"] for e in CV_GENES],
        "total_patients": total,
        "seeds":          f"{SEED_BASE}-{SEED_BASE + 7}",
        "aggregate_metrics": {
            "va_worse_than_0_2_pct":      round(100 * va_poor_count           / total, 1),
            "total_color_blind_pct":      round(100 * total_color_blind_count / total, 1),
            "nystagmus_pct":              round(100 * nystagmus_count         / total, 1),
            "photophobia_pct":            round(100 * photophobia_count       / total, 1),
            "flat_photopic_erg_pct":      round(100 * flat_photopic_count     / total, 1),
            "supernormal_rod_erg_pct":    round(100 * supernormal_rod_count   / total, 1),
            "foveal_hypoplasia_pct":      round(100 * foveal_hypo_count       / total, 1),
            "myopia_pct":                 round(100 * myopia_count            / total, 1),
            "progressive_macular_pct":    round(100 * progressive_mac_count   / total, 1),
        },
        "key_clinical_alerts": [
            "CNGB3 and CNGA3: active gene therapy Phase 2/3 trials — refer ALL newly diagnosed patients",
            "OPN1LW (BCM): standard WES MISSES diagnosis — request targeted OPN1LW/OPN1MW gene array analysis",
            "KCNV2: supernormal rod ERG on bright flash (10cd) is PATHOGNOMONIC — use bright flash protocol",
            "KCNV2: PROGRESSIVE macular dystrophy (not static like achromatopsia) — annual monitoring mandatory",
            "ATF6: foveal hypoplasia on OCT is DISTINGUISHING feature from CNGB3/CNGA3",
            "ALL achromatopsia: FL-41 rose-tinted lenses STANDARD for photophobia",
            "ALL photoreceptor disorders with photophobia: dark sunglasses + FL-41 + hat",
            "Vitamin A supplements: NO benefit in achromatopsia (avoid — unlike some rod dystrophies)",
            "Contact lenses reduce nystagmus amplitude in CNGB3/CNGA3/OPN1LW (null point change)",
            "OPN1LW nystagmus IMPROVES with age (unlike CNGB3/CNGA3 which persists)",
        ],
        "atlas_summary": (
            "This atlas covers the 8 principal genes causing hereditary color vision and cone photoreceptor "
            "disorders. CNGB3 (ACHM3, ~50%) and CNGA3 (ACHM2, ~25%) are the two most common causes of "
            "complete achromatopsia and are the most advanced for gene therapy. GNAT2 (ACHM4) is rare (~2-3%). "
            "PDE6C (ACHM5) and PDE6H (ACHM6) produce incomplete achromatopsia with residual color vision. "
            "ATF6 (ACHM7) is mechanistically distinct (ER stress UPR) with pathognomonic foveal hypoplasia on OCT. "
            "KCNV2 (CDSRR) is the critical progressive outlier — supernormal rod ERG on bright flash is "
            "pathognomonic and the disease causes progressive macular dystrophy unlike the static achromatopsias. "
            "OPN1LW (BCM) is X-linked, requires non-standard gene array testing (WES misses it), and is "
            "distinguished by S-cone preservation, foveal hypoplasia, and nystagmus that improves with age."
        ),
        "gene_summary": gene_summary,
    }


def generate_breakdown():
    breakdown = []
    for idx, entry in enumerate(CV_GENES):
        cohort = _make_cohort(entry, SEED_BASE + idx)
        breakdown.append({
            "gene":                  entry["gene"],
            "alt_name":              entry["alt_name"],
            "locus":                 entry["locus"],
            "protein_size":          entry["protein_size"],
            "inheritance":           entry["inheritance"].split(";")[0].strip(),
            "disease_category":      entry["disease_category"],
            "pathognomonic":         entry["pathognomonic"],
            "treatment":             entry["treatment"],
            "key_features":          entry["key_features"],
            "key_ddx":               entry["key_ddx"],
            "morphology":            entry["morphology"],
            "systemic_involvement":  entry["systemic_involvement"],
            "onset_age":             entry["onset_age"],
            "surgical_urgency":      entry["surgical_urgency"],
            "gene_family":           entry["gene_family"],
            "n_patients":            len(cohort),
            "va_poor_pct":           round(100 * sum(1 for p in cohort if p["va_poor"])               / len(cohort), 1),
            "total_color_blind_pct": round(100 * sum(1 for p in cohort if p["total_color_blind"])     / len(cohort), 1),
            "nystagmus_pct":         round(100 * sum(1 for p in cohort if p["nystagmus"])             / len(cohort), 1),
            "photophobia_pct":       round(100 * sum(1 for p in cohort if p["photophobia"])           / len(cohort), 1),
            "flat_photopic_pct":     round(100 * sum(1 for p in cohort if p["flat_photopic_erg"])     / len(cohort), 1),
            "supernormal_rod_pct":   round(100 * sum(1 for p in cohort if p["supernormal_rod_erg"])   / len(cohort), 1),
            "foveal_hypoplasia_pct": round(100 * sum(1 for p in cohort if p["foveal_hypoplasia"])     / len(cohort), 1),
            "myopia_pct":            round(100 * sum(1 for p in cohort if p["myopia"])                / len(cohort), 1),
            "progressive_mac_pct":   round(100 * sum(1 for p in cohort if p["progressive_macular"])   / len(cohort), 1),
            "sample_patients":       cohort[:3],
        })
    return {"gene_breakdowns": breakdown}


def generate_definitions():
    return {
        "gene_entries": {
            entry["gene"]: {
                "gene":                 entry["gene"],
                "full_name":            entry["protein"].split(" --")[0].strip(),
                "locus":                entry["locus"],
                "protein_size":         entry["protein_size"],
                "inheritance":          entry["inheritance"].split(";")[0].strip(),
                "disease_name":         entry["disease_category"],
                "disease_pathway":      entry["disease_pathway"],
                "pathognomonic":        entry["pathognomonic"],
                "treatment":            entry["treatment"][:600],
                "key_features":         entry["key_features"],
                "key_ddx":              entry["key_ddx"],
                "morphology":           entry["morphology"],
                "systemic_involvement": entry["systemic_involvement"],
                "onset_age":            entry["onset_age"],
                "surgical_urgency":     entry["surgical_urgency"],
                "gene_family":          entry["gene_family"],
            }
            for entry in CV_GENES
        },
        "cv_glossary": {
            "Achromatopsia vs Blue Cone Monochromatism — Key Differential": (
                "BOTH present with colour blindness, nystagmus, photophobia and reduced VA. "
                "ACHROMATOPSIA (CNGB3/CNGA3/GNAT2/PDE6C/PDE6H/ATF6): ALL cone types non-functional; "
                "Nagel anomaloscope: complete monochromat (no red-green discrimination); "
                "S-cone ERG also absent or reduced in complete form; "
                "BLUE CONE MONOCHROMATISM (OPN1LW/BCM): S-CONES PRESERVED -- "
                "only L and M cones absent; blue-yellow discrimination retained; "
                "Nagel anomaloscope: cannot match red-green (L/M absent); S-cone ERG PRESERVED; "
                "BCM ADDITIONAL CLUES: X-linked males only; myopia ~90%; foveal hypoplasia on OCT; "
                "nystagmus IMPROVES with age (vs achromatopsia persists); "
                "CRITICAL: standard WES misses BCM (LCR deletion) -- targeted gene array mandatory."
            ),
            "KCNV2 Supernormal Rod ERG — Diagnostic Protocol": (
                "KCNV2 CONE DYSTROPHY WITH SUPERNORMAL ROD ERG (CDSRR) -- DIAGNOSTIC CRITICAL POINTS: "
                "THE ERG PROTOCOL MATTERS: must use DARK-ADAPTED BRIGHT FLASH (10 candela per ISCEV standard); "
                "dim flash (0.01 cd) may be normal or only mildly abnormal -- does NOT show supernormal b-wave; "
                "SUPERNORMAL b-wave DEFINITION: b-wave amplitude exceeds normal upper limit (typically >800 uV); "
                "MECHANISM: Kv8.2 (KCNV2) absence -> Kv2.1 homotetramers in rods -> "
                "altered repolarisation kinetics -> paradoxically enhanced ON-bipolar (b-wave) response; "
                "PROGRESSIVE DISTINCTION: KCNV2 causes progressive macular dystrophy (bull's eye) -- "
                "not a static disorder like achromatopsia; VA deteriorates over decades; "
                "REQUEST: ISCEV standard full-field ERG with 10cd dark-adapted bright flash when KCNV2 suspected."
            ),
            "ATF6 Foveal Hypoplasia — OCT Distinguishes from CNGB3/CNGA3": (
                "ATF6 (ACHM7) vs CNGB3/CNGA3 (ACHM3/2) -- OCT IS THE KEY DISCRIMINATOR: "
                "ATF6: FOVEAL HYPOPLASIA ON OCT FROM BIRTH -- "
                "absent or markedly reduced foveal pit; inner retinal layers (GCL, IPL) extend to foveal centre; "
                "foveal avascular zone reduced or absent on OCT-A; "
                "CNGB3/CNGA3: OCT INITIALLY NEAR-NORMAL at fovea -- "
                "foveal pit present; outer nuclear layer (ONL) thinning develops progressively with age "
                "but starts near-normal; structural vs functional loss distinction; "
                "ATF6 MECHANISM: ER stress UPR failure during foveal cone development -> apoptosis; "
                "ADDITIONAL ATF6 CLUE: photopic ERG LESS extinguished early (some parafoveal cones survive); "
                "PRACTICAL IMPACT: ATF6 patients should have annual OCT to monitor macular atrophy progression."
            ),
            "OPN1LW Gene Array Testing — Why Standard WES Fails": (
                "BLUE CONE MONOCHROMATISM DIAGNOSTIC TRAP: standard WES/exome sequencing DOES NOT DETECT BCM. "
                "REASON: OPN1LW and OPN1MW are adjacent tandem genes at Xq28 controlled by a shared LCR "
                "(locus control region) ~3.5 kb upstream; BCM is caused by DELETIONS or REARRANGEMENTS "
                "affecting the LCR or both gene array members -- these are copy-number variants (CNVs) "
                "not captured by standard short-read exome NGS; "
                "WHAT TO REQUEST: (1) Southern blot of OPN1LW/OPN1MW locus; "
                "(2) MLPA (multiplex ligation-dependent probe amplification) for OPN1LW/OPN1MW; "
                "(3) long-range PCR spanning the LCR region; "
                "WHEN TO SUSPECT BCM: male; nystagmus improving with age; myopia ~90%; "
                "foveal hypoplasia on OCT; S-cone ERG preserved; standard gene panel negative; "
                "IMPLICATION: a male with suspected achromatopsia + negative standard panel -> "
                "always order targeted OPN1LW/OPN1MW gene array analysis before concluding 'mutation-negative'."
            ),
            "FL-41 Lenses and Photophobia Management in Cone Disorders": (
                "FL-41 (ROSE-TINTED) LENSES: THE STANDARD PHOTOPHOBIA INTERVENTION across ALL achromatopsia subtypes. "
                "MECHANISM: FL-41 filter selectively attenuates 480-520 nm (blue-green) wavelengths "
                "which maximally activate surviving S-cones and rod photoreceptors under photopic conditions; "
                "EVIDENCE: randomised trials show reduced photophobia VAS scores and improved outdoor function; "
                "PRACTICAL: spectacles and contact lens forms available; rose/pink tint; "
                "INDOOR use: may also benefit (fluorescent lighting rich in 480-520nm); "
                "ADDITIONAL: wide-brimmed hat outdoors + dark wraparound sunglasses; "
                "CONTACT LENSES DUAL BENEFIT in CNGB3/CNGA3/OPN1LW: "
                "(1) photophobia reduction via tinted lens + "
                "(2) nystagmus amplitude reduction (contact lenses touch cornea -> sensory feedback "
                "alters null point -> 20-30% amplitude reduction documented); "
                "VITAMIN A SUPPLEMENTS: NO BENEFIT in ANY achromatopsia subtype (unlike rod dystrophies); "
                "do not recommend vitamin A to achromatopsia patients."
            ),
            "Gene Therapy for Achromatopsia — Current Trial Landscape (2026)": (
                "GENE THERAPY TRIAL LANDSCAPE FOR ACHROMATOPSIA (as of 2026): "
                "CNGB3 (ACHM3 ~50%): MOST ADVANCED -- Phase 2/3 trials: "
                "RD-CURE Consortium (Germany/USA): AAV2/5-CNGB3 subretinal; "
                "MeiraGTx/Janssen (AAV5-CNGB3): Phase 2/3 completed; "
                "CNGA3 (ACHM2 ~25%): Phase 2 ongoing: "
                "AGTC-402 (Applied Genetic Technologies): AAV5-CNGA3 subretinal; "
                "BTT-401 (Beacon Therapeutics): AAV-CNGA3; "
                "VECTOR: AAV subretinal injection under local anaesthesia (macula detachment created); "
                "TARGET: surviving cone photoreceptors (cones present but non-functional); "
                "RATIONALE: achromatopsia is an ideal gene therapy target -- "
                "cones anatomically present early in life (degenerative changes later); "
                "single gene; recessive; no immune privilege issues with subretinal; "
                "YOUNGER PATIENTS PREFERRED: intervene before progressive ONL thinning; "
                "GNAT2/PDE6C/PDE6H/ATF6: no current clinical trials (preclinical or early phase); "
                "KCNV2/OPN1LW: preclinical (LCR architecture makes OPN1LW complex)."
            ),
            "Complete vs Incomplete Achromatopsia — Clinical and ERG Distinction": (
                "COMPLETE ACHROMATOPSIA (CNGB3/CNGA3/GNAT2 most common): "
                "ALL Ishihara plates failed; Nagel anomaloscope: complete monochromat; "
                "FLAT PHOTOPIC ERG (no cone signal); normal rod scotopic ERG; "
                "pendular nystagmus (congenital, persists); photophobia (severe); VA ~0.1; "
                "INCOMPLETE ACHROMATOPSIA (PDE6C complete null, ATF6 subset): "
                "fails most but not all Ishihara; PARTIAL photopic ERG; "
                "nystagmus may be milder; VA 0.1-0.3; "
                "PDE6C (ACHM5) AND PDE6H (ACHM6) FAVOUR INCOMPLETE FORM: "
                "partial function alleles retain some PDE6 activity -> residual color discrimination; "
                "CLINICAL IMPORTANCE: incomplete achromatopsia patients may have residual color function "
                "that affects gene therapy outcome expectations and low vision rehabilitation planning; "
                "ERG IS ESSENTIAL for classification -- clinical exam alone insufficient."
            ),
            "Phototransduction Cascade Map — Achromatopsia Gene Positions": (
                "CONE PHOTOTRANSDUCTION PATHWAY AND ACHROMATOPSIA GENE POSITIONS: "
                "1. PHOTON ABSORPTION: L-cone opsin (OPN1LW) / M-cone opsin (OPN1MW) / S-cone opsin (OPN1SW); "
                "BCM = OPN1LW/OPN1MW absent; "
                "2. G-PROTEIN ACTIVATION: GNAT2 (cone transducin alpha-2); ACHM4; "
                "3. PDE6 ACTIVATION: PDE6C (catalytic alpha' subunit, ACHM5) + PDE6D (delta); "
                "regulated by PDE6H (gamma' inhibitory subunit, ACHM6); "
                "4. cGMP HYDROLYSIS: PDE6C active -> cGMP falls; "
                "5. CNG CHANNEL CLOSURE: CNGA3 (pore-forming, ACHM2) + CNGB3 (modulatory, ACHM3) -> closes; "
                "6. CALCIUM FALLS -> CONE HYPERPOLARISES -> SIGNAL TO ON-BIPOLAR CELLS; "
                "ATF6 (ACHM7): NOT in phototransduction cascade -- ER stress/UPR pathway for cone survival; "
                "KCNV2 (CDSRR): Kv2.1/Kv8.2 K+ channel -> cone repolarisation (post-hyperpolarisation); "
                "ERG STEP CORRELATION: a-wave = cone hyperpolarisation; b-wave = ON-bipolar; "
                "ALL phototransduction genes (GNAT2, PDE6C, PDE6H, CNGA3, CNGB3): flat photopic ERG."
            ),
        },
    }


# API-compatible aliases (backend calls get_* variants)
get_overview    = generate_overview
get_breakdown   = generate_breakdown
get_definitions = generate_definitions
