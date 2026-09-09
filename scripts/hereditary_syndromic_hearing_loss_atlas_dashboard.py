#!/usr/bin/env python3
"""Hereditary-Syndromic-Hearing-Loss-Atlas — Complete 8-Gene Atlas
(PAX3 · MITF · SOX10 · EDNRB · EYA1 · CHD7 · TCOF1 · GATA3).

PAX3     (Paired Box 3; 505 aa; ~56 kDa; 2q36.1; AD;
           Waardenburg Syndrome Type 1 (WS1) and Type 3 (WS3/Klein);
           MOST COMMON Waardenburg syndrome gene (~50% of all WS);
           Dystopia canthorum (W-index ≥1.95) — PATHOGNOMONIC for WS1/WS3;
           White forelock; iris heterochromia/hypopigmentation; SNHL ~57%;
           seed SEED_BASE+0).
MITF     (Microphthalmia-associated Transcription Factor; 526 aa; ~58 kDa; 3p13; AD;
           Waardenburg Syndrome Type 2A (WS2A);
           MOST COMMON WS2 gene (~40% of WS2); NO dystopia canthorum (DDx WS1);
           Tietz syndrome: severe MITF alleles → complete albinism;
           seed SEED_BASE+1).
SOX10    (SRY-Box Transcription Factor 10; 466 aa; ~52 kDa; 22q13.1; AD;
           Waardenburg Type 4C (WS4C) / PCWH syndrome;
           HSCR + profound SNHL + peripheral demyelinating neuropathy;
           PCWH: peripheral demyelinating neuropathy + central hypomyelination;
           seed SEED_BASE+2).
EDNRB    (Endothelin Receptor Type B; 442 aa; ~50 kDa; 13q22.3; AR/AD;
           Waardenburg Type 4A (Waardenburg-Shah syndrome);
           Hirschsprung disease + Waardenburg features; AR biallelic most severe;
           S305N Mennonite founder mutation;
           seed SEED_BASE+3).
EYA1     (EYA Transcriptional Coactivator and Phosphatase 1; 559 aa; ~61 kDa; 8q13.3; AD;
           Branchiootorenal (BOR) and Branchiootic (BO) syndrome;
           Branchial anomalies + Otologic anomalies + Renal dysplasia;
           SNHL + CHL; Stickler-like renal phenotype;
           seed SEED_BASE+4).
CHD7     (Chromodomain Helicase DNA Binding Protein 7; 2997 aa; ~337 kDa; 8q12.2; AD de novo;
           CHARGE syndrome — C=Coloboma H=Heart A=choanal Atresia R=Retardation G=Genital E=Ear;
           MOST COMMON cause of syndromic SNHL in neonates requiring urgent airway management;
           SCC (semicircular canal aplasia) on CT = near-pathognomonic for CHARGE;
           seed SEED_BASE+5).
TCOF1    (Treacle Ribosome Biogenesis Factor 1; 1411 aa; ~152 kDa; 5q33.1; AD;
           Treacher Collins Syndrome (TCS) Type 1 / Mandibulofacial Dysostosis;
           Absent/hypoplastic zygoma + malar bones + zygomatic arch + micrognathia;
           Predominantly CHL (ossicular/external canal) ± SNHL;
           seed SEED_BASE+6).
GATA3    (GATA Binding Protein 3; 444 aa; ~48 kDa; 10p15.3; AD;
           HDR syndrome (Hypoparathyroidism-Deafness-Renal dysplasia) / Barakat syndrome;
           TRIAD: SNHL + Hypoparathyroidism (hypocalcaemic tetany) + Renal dysplasia;
           Calcium CORRECTION BEFORE audiometry; hypocalcaemia worsens hearing;
           seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 2446-2453).
"""

import random

SEED_BASE = 2446

SHL_GENES = [
    # -- PAX3 -- Waardenburg Syndrome Type 1 / Type 3 (Most Common WS Gene) -------------------------
    {
        "gene": "PAX3",
        "alt_name": (
            "PAX3 (PAX3-505aa-2q36.1 / AD -- "
            "WS1-WAARDENBURG-TYPE1-MOST-COMMON-WS-GENE-~50pct-ALL-WS -- "
            "DYSTOPIA-CANTHORUM-W-INDEX-≥1.95-PATHOGNOMONIC-WS1 -- "
            "WHITE-FORELOCK-45pct-IRIS-HETEROCHROMIA-HYPOPIGMENTATION -- "
            "SNHL-57pct-BILATERAL-FLAT-AUDIOGRAM-COCHLEAR-IMPLANT-EFFECTIVE -- "
            "WS3-KLEIN-SYNDROME-PAX3-UPPER-LIMB-CONTRACTURES-SAME-GENE)"
        ),
        "protein": (
            "PAX3 -- 2q36.1 AD -- PAX3-505aa -- "
            "Paired-Box-Transcription-Factor-3-56kDa-Paired-Box-Homeodomain -- "
            "Neural-Crest-Melanocyte-Cochlear-Stria-Vascularis -- "
            "OMIM-Gene-606597-Disease-WS1-193500-WS3-148820"
        ),
        "locus": "2q36.1",
        "protein_size": "505 aa / 56 kDa",
        "inheritance": (
            "AD (autosomal dominant; haploinsufficiency); "
            "PAX3 encodes a paired box + homeodomain transcription factor critical for neural crest cell "
            "development and survival. "
            "FUNCTION: PAX3 drives migration and differentiation of melanocyte precursors from neural crest "
            "to cochlear stria vascularis, skin, hair, and iris; also regulates inner ear neuronal survival. "
            "PAX3 LOF → melanocyte failure → absent or dysfunctional intermediate cells in stria vascularis "
            "→ endocochlear potential loss → SNHL; also absent melanocytes in skin/hair/iris → depigmentation. "
            "PENETRANCE: ~100% for dystopia canthorum; SNHL ~57% (bilateral); white forelock ~45%; "
            "iris pigmentation abnormality ~42%; "
            "GENOTYPE-PHENOTYPE: truncating/frameshift → more severe; missense varies; "
            "WS1 (dystopia canthorum) vs WS3 (Klein-Waardenburg, same PAX3 mutation + upper limb abnormality). "
            "FOUNDER: no single dominant founder; private mutations globally; "
            "W-INDEX: W = 2× medial canthal dist / (palpebral fissure L + palpebral fissure R + nose base) — "
            "W ≥1.95 diagnostic for dystopia canthorum."
        ),
        "disease_category": (
            "Waardenburg Syndrome Type 1 (WS1) — AR (dystopia canthorum + SNHL ± pigmentary); "
            "Waardenburg Syndrome Type 3 (WS3/Klein) — same gene + upper limb contractures/anomalies; "
            "SNHL present in ~57%; white forelock ~45%; iris heterochromia ~40%"
        ),
        "disease_pathway": (
            "PAX3 IN NEURAL CREST DEVELOPMENT: PAX3 activated in dorsal neural tube → drives EMT "
            "(epithelial-mesenchymal transition) of neural crest cells → melanoblasts migrate to cochlea, "
            "skin, hair follicles, iris stroma. "
            "COCHLEAR ROLE: melanocytes in stria vascularis intermediate cell layer maintain K+ concentration "
            "in endolymph via ATP-dependent pumps; PAX3 LOF → absent strial melanocytes → endolymph K+ falls "
            "→ endocochlear potential lost → mechanotransduction fails → SNHL. "
            "IRIS: melanocytes absent or reduced in anterior iris stroma → isochromia/heterochromia/hypopigmentation. "
            "SKIN/HAIR: melanocytes absent in specific follicular units → white forelock (bang/temporal), "
            "white patches, premature greying. "
            "MEDIAL CANTHUS: PAX3 regulates periocular mesenchyme → LOF → lateral displacement of medial canthi "
            "(dystopia canthorum) with short palpebral fissures — W-index ≥1.95 = pathognomonic. "
            "COCHLEAR ANATOMY: cochlear structure normal on CT/MRI — no EVA (DDx SLC26A4), no semicircular "
            "canal aplasia (DDx CHD7/CHARGE)."
        ),
        "pathognomonic": (
            "PAX3 / WS1 PATHOGNOMONIC FEATURES: "
            "1. DYSTOPIA CANTHORUM (W-INDEX ≥1.95): lateral displacement of medial canthi; "
            "Calculate: W = 2a/(b+c+a) where a=inner canthal distance, b/c=palpebral fissure widths; "
            "W ≥1.95 = DIAGNOSTIC for WS1 (distinguishes from WS2/MITF where W normal); "
            "2. WHITE FORELOCK (~45%): triangular patch of white hair at frontal midline; "
            "can be subtle or minimal; check hair roots in dyed individuals; "
            "3. IRIS HETEROCHROMIA/HYPOPIGMENTATION (~40-42%): complete heterochromia (blue+brown); "
            "partial heterochromia; isochromia blue; pigmentary patches; "
            "4. SNHL (~57%): bilateral; flat audiogram; prelingual or early onset; "
            "cochlear implant effective (stria vascularis absent, hair cells relatively preserved); "
            "5. PREMATURE GREYING: melanocyte loss in hair follicles across scalp; "
            "6. NORMAL COCHLEAR ANATOMY: CT/MRI cochlea normal — no EVA, no canal aplasia (DDx CHARGE). "
            "WS3 (KLEIN): PAX3 mutation + congenital flexion contractures of elbows/knees/fingers — same gene."
        ),
        "treatment": (
            "HEARING: SNHL present in ~57%; mild-profound range; "
            "HEARING AIDS: for mild-moderate SNHL; digital programmable; "
            "COCHLEAR IMPLANT: for severe-profound SNHL; outcomes good (stria absent but hair cells/SGN preserved); "
            "GENETIC COUNSELLING: AD 50% risk; variable expressivity; full penetrance for dystopia canthorum; "
            "OPHTHAMOLOGY: iris pigment abnormalities — screen for amblyopia in heterochromia; "
            "SKIN: sunscreen for depigmented areas (melanoma risk in normal skin compensatory response); "
            "AUDIOLOGICAL SURVEILLANCE: annual audiogram; SNHL can be stable or progressive; "
            "FAMILY SCREENING: W-index measurement in all first-degree relatives; "
            "molecular confirmation of PAX3 pathogenic variant."
        ),
        "key_features": [
            "Dystopia canthorum (W-index ≥1.95) — PATHOGNOMONIC for WS1; distinguishes WS1 from WS2",
            "White forelock ~45%; iris heterochromia/hypopigmentation ~40%; premature greying",
            "SNHL ~57% bilateral; flat audiogram; cochlear implant effective",
            "Most common WS gene (~50% all WS); WS3 (Klein) = PAX3 + upper limb anomalies",
            "Normal cochlear CT/MRI (DDx CHD7/CHARGE: semicircular canal aplasia)",
            "Neural crest melanocyte development regulator; stria vascularis melanocyte absent",
            "W-index formula: measure before diagnosis; subtle dystopia missed clinically",
            "AD 50% risk; variable expressivity; ophthalmology screen for amblyopia",
        ],
        "key_ddx": (
            "MITF (WS2A): NO dystopia canthorum (W-index normal) — KEY DDx; iris heterochromia; SNHL; "
            "SOX10 (WS4C): HSCR + neuropathy; "
            "EDNRB (WS4A): HSCR + WS features; "
            "CHD7 (CHARGE): semicircular canal aplasia CT; coloboma; heart defect; choanal atresia; "
            "SLC26A4 (Pendred): EVA on CT; no pigmentary changes; "
            "Vitiligo: acquired depigmentation; no ear anomaly; no dystopia canthorum."
        ),
        "systemic_involvement": (
            "SKIN: white forelock; depigmented patches; premature greying — melanocyte loss. "
            "EYE: iris heterochromia; hypopigmented iris; amblyopia risk in complete heterochromia. "
            "EAR: SNHL bilateral (~57%); rare CHL if ossicular anomaly. "
            "LIMBS (WS3 only): congenital flexion contractures of upper limbs (Klein-Waardenburg). "
            "NO HSCR (DDx EDNRB/SOX10); NO neuropathy (DDx SOX10); NO cardiac (DDx CHD7)."
        ),
        "onset_age": "Congenital SNHL; pigmentary features visible at birth; dystopia canthorum measurable at birth",
        "surgical_urgency": "No urgent surgical need in WS1; CI for severe-profound SNHL as per NSHL protocols",
        "gene_family": "Paired box (PAX) transcription factor family; paired box + homeodomain",
        "morphology": (
            "AUDIOGRAM: flat or gently sloping bilateral SNHL moderate-to-profound; "
            "CT/MRI: NORMAL cochlear anatomy; no EVA; no semicircular canal aplasia; "
            "PIGMENTATION: white forelock (bang area); skin patches; iris heterochromia on slit lamp; "
            "FACIAL: telecanthus (dystopia canthorum) — lateral displacement of medial canthi; broad nasal root"
        ),
        "n_patients": 40,
    },

    # -- MITF -- Waardenburg Syndrome Type 2A (Most Common WS2 Gene) --------------------------------
    {
        "gene": "MITF",
        "alt_name": (
            "MITF (MITF-526aa-3p13 / AD -- "
            "WS2A-WAARDENBURG-TYPE2A-MOST-COMMON-WS2-GENE-~40pct-WS2 -- "
            "NO-DYSTOPIA-CANTHORUM-W-INDEX-NORMAL-KEY-DDx-WS1 -- "
            "IRIS-HETEROCHROMIA-SNHL-WHITE-FORELOCK-VARIABLE -- "
            "TIETZ-SYNDROME-SEVERE-ALLELES-COMPLETE-ALBINISM-NO-SNHL -- "
            "MELANOMA-RISK-INCREASED-2-3x-MITF-E318K-COMMON-LOW-RISK)"
        ),
        "protein": (
            "MITF -- 3p13 AD -- MITF-526aa -- "
            "Microphthalmia-Associated-Transcription-Factor-58kDa-bHLH-LZ-PAS -- "
            "Master-Regulator-Melanocyte-Differentiation-Survival -- "
            "OMIM-Gene-156845-Disease-WS2A-193510-Tietz-103500"
        ),
        "locus": "3p13",
        "protein_size": "526 aa / 58 kDa",
        "inheritance": (
            "AD (autosomal dominant; haploinsufficiency or dominant-negative); "
            "MITF is the master transcriptional regulator of melanocyte differentiation and survival. "
            "FUNCTION: MITF (basic helix-loop-helix leucine-zipper transcription factor) controls "
            "melanocyte specification, melanin synthesis genes (TYR, DCT, TYRP1), and melanocyte survival. "
            "MITF LOF → reduced melanocyte development → WS2A phenotype; "
            "SEVERE DOMINANT-NEGATIVE ALLELES (e.g., K206I) → Tietz syndrome — complete albinism, "
            "profound congenital SNHL, absent eyebrows/eyelashes; "
            "MILD ALLELES (e.g., E318K) → increased melanoma risk (~2-3×) without WS2 features; "
            "PENETRANCE FOR WS2A: ~100% for iris/skin features; SNHL in ~80-90% of WS2A; "
            "W-INDEX: NORMAL (≤1.94) — distinguishes WS2A from WS1 (PAX3) where W ≥1.95. "
            "WS2A is the most common genetically confirmed WS2 subtype (~40% of WS2 cases). "
            "FUNCTION DOWNSTREAM OF PAX3: MITF is transcriptionally activated by PAX3 + SOX10 + WNT."
        ),
        "disease_category": (
            "Waardenburg Syndrome Type 2A (WS2A) — AD; SNHL ~80-90%; iris heterochromia; "
            "NO dystopia canthorum (key DDx from WS1/PAX3); "
            "Tietz syndrome (severe alleles) — complete cutaneous albinism + profound congenital SNHL"
        ),
        "disease_pathway": (
            "MITF MASTER REGULATOR: MITF activated by PAX3 + SOX10 + β-catenin (WNT) → "
            "drives expression of melanocyte specification and survival genes. "
            "MELANIN SYNTHESIS: MITF activates TYR (tyrosinase), DCT (DOPAchrome tautomerase), "
            "TYRP1 — all melanin production enzymes; "
            "MITF LOF → reduced melanin → hypopigmented hair, skin, iris. "
            "COCHLEAR: as in PAX3 — absent strial melanocytes → endolymph K+ collapse → SNHL. "
            "TIETZ MECHANISM: certain dominant-negative MITF alleles block all MITF target gene expression → "
            "complete failure of melanocyte differentiation → total albinism + severe bilateral SNHL; "
            "iris appears uniformly grey-blue (no brown pigment); hair uniformly white from birth. "
            "MELANOMA: MITF amplification/gain-of-function in melanoma (opposite direction); "
            "heterozygous E318K → SUMO motif disruption → increased MITF target gene expression → "
            "modest increased melanoma risk; not a WS2 allele clinically."
        ),
        "pathognomonic": (
            "MITF / WS2A CLINICAL FEATURES: "
            "1. NO DYSTOPIA CANTHORUM: W-index NORMAL (≤1.94) — CRITICAL DDx from PAX3/WS1; "
            "medial canthi in normal position; inner canthal distance normal for age; "
            "2. IRIS HETEROCHROMIA (~45%): complete, partial, or segmental; "
            "isochromia blue (complete bilateral hypopigmentation of iris); "
            "3. SNHL (~80-90%): bilateral; moderate-to-profound; flat audiogram; "
            "cochlear implant recommended for severe-profound; "
            "4. WHITE FORELOCK (~30-35%): less frequent than WS1; variable; "
            "5. TIETZ SYNDROME (severe alleles): "
            "— complete uniform albinism of skin, hair (white), eyebrows, eyelashes; "
            "— bilateral profound congenital SNHL (severe); "
            "— NO iris heterochromia (uniform unpigmented iris); "
            "— Tietz cannot clinically be WS2 with this phenotype — test MITF specifically. "
            "COCHLEAR CT/MRI: NORMAL anatomy — no EVA, no semicircular canal aplasia."
        ),
        "treatment": (
            "HEARING: SNHL in ~80-90% of WS2A; mild-profound range; "
            "HEARING AIDS: for mild-moderate SNHL; "
            "COCHLEAR IMPLANT: for severe-profound SNHL; outcomes good; "
            "TIETZ: bilateral profound congenital SNHL → bilateral CI early (before 12 months); "
            "OPHTHALMOLOGY: screen iris heterochromia for amblyopia; "
            "visual function in Tietz usually intact (no nystagmus, normal visual acuity — DDx OCA albinism); "
            "DERMATOLOGY: sun protection for hypopigmented skin patches; "
            "MELANOMA SURVEILLANCE: MITF E318K carriers — annual dermatology review; "
            "GENETIC COUNSELLING: AD 50% risk; variable expressivity."
        ),
        "key_features": [
            "NO dystopia canthorum (W-index normal ≤1.94) — KEY DDx from WS1/PAX3",
            "Most common WS2 gene (~40% of WS2); SNHL ~80-90% bilateral",
            "Iris heterochromia ~45%; white forelock ~30-35%",
            "Tietz syndrome: severe alleles → complete albinism + profound SNHL",
            "Master melanocyte transcription factor; activates TYR, DCT, TYRP1",
            "MITF E318K: melanoma risk ×2-3; NOT a WS2 allele",
            "Normal cochlear CT/MRI (no EVA, no canal aplasia)",
            "CI effective for severe-profound SNHL; Tietz → bilateral CI before 12 months",
        ],
        "key_ddx": (
            "PAX3 (WS1/WS3): dystopia canthorum W-index ≥1.95 — KEY DDx; MITF has NORMAL W-index; "
            "SOX10 (WS4C): HSCR + peripheral neuropathy; "
            "Tietz vs OCA (OCA1-4): Tietz = normal visual acuity + no nystagmus + no foveal hypoplasia (DDx OCA); "
            "SLC45A2/OCA4: uniform albinism but normal hearing; "
            "Vitiligo: acquired; post-inflammatory; anti-melanocyte antibodies."
        ),
        "systemic_involvement": (
            "SKIN: hypopigmented patches; premature greying; Tietz: complete albinism. "
            "EYE: iris heterochromia; amblyopia risk; Tietz: uniformly unpigmented iris (grey-blue). "
            "EAR: SNHL bilateral (mild-profound). "
            "NO HSCR; NO neuropathy; NO cardiac; NO renal. "
            "MELANOMA: E318K low-penetrance risk allele (not clinical WS)."
        ),
        "onset_age": "Congenital SNHL and pigmentary features; Tietz — bilateral profound at birth",
        "surgical_urgency": "Bilateral CI for Tietz syndrome before 12 months; hearing aids for WS2A moderate SNHL",
        "gene_family": "bHLH-LZ (basic helix-loop-helix leucine-zipper) transcription factor; MiT/TFE subfamily",
        "morphology": (
            "AUDIOGRAM: flat bilateral SNHL moderate-to-profound; "
            "CT/MRI: NORMAL cochlear anatomy; no EVA; no semicircular canal aplasia; "
            "IRIS: heterochromia on slit lamp; Tietz: uniformly pale iris; "
            "SKIN: hypopigmented patches; Tietz: total albinism"
        ),
        "n_patients": 40,
    },

    # -- SOX10 -- Waardenburg Type 4C / PCWH Syndrome -----------------------------------------------
    {
        "gene": "SOX10",
        "alt_name": (
            "SOX10 (SOX10-466aa-22q13.1 / AD -- "
            "WS4C-WAARDENBURG-SHAH-TYPE4C-HSCR-WS-PERIPHERAL-NEUROPATHY -- "
            "PCWH-PERIPHERAL-CENTRAL-DEMYELINATING-NEUROPATHY-WAARDENBURG-HIRSCHSPRUNG -- "
            "SNHL-BILATERAL-PROFOUND-COCHLEAR-IMPLANT-EFFECTIVE -- "
            "PERIPHERAL-DEMYELINATING-NEUROPATHY-CMT4-LIKE-HYPOMYELINATION -- "
            "MYELINATION-REGULATOR-PMP22-MPZ-GJB1-TARGET-GENES)"
        ),
        "protein": (
            "SOX10 -- 22q13.1 AD -- SOX10-466aa -- "
            "SRY-Box-Transcription-Factor-10-52kDa-HMG-Box-Transactivation -- "
            "Neural-Crest-Melanocyte-Peripheral-Myelination-Enteric-NS -- "
            "OMIM-Gene-602229-Disease-WS4C-613266-PCWH-609136"
        ),
        "locus": "22q13.1",
        "protein_size": "466 aa / 52 kDa",
        "inheritance": (
            "AD (autosomal dominant; haploinsufficiency); "
            "SOX10 is a high-mobility-group (HMG) box transcription factor with dual roles in "
            "neural crest melanocyte development AND peripheral myelination. "
            "MELANOCYTE FUNCTION (shared with PAX3/MITF): SOX10 → activates MITF → melanocyte differentiation; "
            "SOX10 LOF → same depigmentation phenotype as PAX3/MITF — iris heterochromia, white forelock. "
            "ENTERIC NS: SOX10 required for enteric neural crest → colonise gut → Hirschsprung disease (HSCR) "
            "when LOF — distinguishes WS4 from WS1/WS2. "
            "PERIPHERAL MYELINATION: SOX10 activates PMP22, MPZ (P0), GJB1 (Cx32), MBP — all myelin genes; "
            "SOX10 LOF → peripheral demyelinating neuropathy → CMT4-like phenotype. "
            "PCWH SYNDROME: most severe SOX10 alleles → peripheral demyelinating neuropathy + central "
            "hypomyelination + WS + HSCR. "
            "GENOTYPE-PHENOTYPE: truncating alleles rescued by NMD → milder (WS4C); "
            "alleles escaping NMD → dominant-negative → PCWH (severe); "
            "SOX10 frameshift at different exon positions determines PCWH vs WS4C severity."
        ),
        "disease_category": (
            "Waardenburg Syndrome Type 4C (WS4C) — AD; SNHL + iris heterochromia + white forelock + HSCR; "
            "PCWH syndrome: peripheral + central demyelinating neuropathy + WS + HSCR (most severe SOX10)"
        ),
        "disease_pathway": (
            "SOX10 DUAL NEURAL CREST ROLES: "
            "MELANOCYTE ARM: SOX10 + PAX3 → cooperatively activate MITF → melanocyte differentiation; "
            "SOX10 LOF → MITF haploinsufficiency → hypopigmentation + cochlear melanocyte failure → SNHL. "
            "ENTERIC ARM: SOX10 maintains survival of enteric neural crest progenitors migrating caudally; "
            "SOX10 LOF → enteric progenitor death → absent myenteric ganglia → Hirschsprung disease (HSCR); "
            "HSCR in WS4: aganglionosis most commonly recto-sigmoid (short segment) or total colonic. "
            "MYELINATION ARM: SOX10 binds promoters of PMP22 (peripheral myelin protein 22), "
            "MPZ/P0 (myelin protein zero), GJB1/Cx32 — all required for Schwann cell myelination; "
            "SOX10 LOF → peripheral hypomyelination → demyelinating neuropathy (slow NCVs <38 m/s); "
            "also MBP, CNP, PLP1 (central myelin) → central hypomyelination in PCWH. "
            "ALLELE ESCAPE FROM NMD: C-terminal truncating alleles that escape NMD → dominant-negative "
            "on SOX10 dimers → more severe PCWH; alleles triggering NMD → haploinsufficiency → WS4C."
        ),
        "pathognomonic": (
            "SOX10 / WS4C / PCWH PATHOGNOMONIC FEATURES: "
            "1. WS FEATURES: iris heterochromia; white forelock; SNHL (bilateral profound); "
            "W-index NORMAL (no dystopia canthorum — DDx PAX3/WS1); "
            "2. HIRSCHSPRUNG DISEASE (HSCR): absent enteric ganglia; constipation + failure to thrive; "
            "delayed meconium passage >48h; suction rectal biopsy: aganglionosis PATHOGNOMONIC; "
            "HSCR distinguishes WS4 from WS1 (PAX3) and WS2 (MITF) — WS1/2 have NO HSCR; "
            "3. PERIPHERAL DEMYELINATING NEUROPATHY (WS4C/PCWH): "
            "nerve conduction: slow NCVs (<38 m/s in upper limb) — demyelinating pattern; "
            "onion bulb formation on nerve biopsy; distal weakness + hypo/areflexia; "
            "4. CENTRAL HYPOMYELINATION (PCWH): "
            "MRI white matter: diffuse hypomyelination (T2 hyperintensity); "
            "nystagmus + cerebellar signs + intellectual disability in severe PCWH; "
            "5. COCHLEAR CT/MRI: NORMAL anatomy — no EVA, no semicircular canal aplasia."
        ),
        "treatment": (
            "HEARING: bilateral profound SNHL → cochlear implant; outcomes good; "
            "implant before neuropathy worsens peripheral auditory system; "
            "HSCR: surgical pull-through (endorectal or transabdominal) — must be done early neonatal; "
            "colostomy bridge if needed pre-definitive surgery; "
            "NEUROPATHY: physiotherapy; orthotics; ankle-foot-orthosis (AFO) for foot drop; "
            "CENTRAL (PCWH): supportive; no disease-modifying therapy; early intervention services; "
            "GENETIC COUNSELLING: AD 50% risk; NMD vs escape alleles predicts severity; "
            "cascade family testing."
        ),
        "key_features": [
            "WS4C: HSCR + WS features (heterochromia + white forelock + SNHL) — NO dystopia canthorum",
            "PCWH: peripheral + central demyelinating neuropathy + WS + HSCR (most severe SOX10 alleles)",
            "Peripheral demyelinating neuropathy: NCVs <38 m/s; onion bulbs on biopsy",
            "HSCR: aganglionosis; suction rectal biopsy PATHOGNOMONIC — distinguishes WS4 from WS1/WS2",
            "Alleles escaping NMD → dominant-negative → PCWH; NMD-rescued → haploinsufficiency → WS4C",
            "Activates PMP22, MPZ, GJB1, MBP, PLP1 — all myelin genes (peripheral + central)",
            "Bilateral profound SNHL; CI effective before neuropathy progression",
            "Cochlear CT/MRI: normal (no EVA, no semicircular canal aplasia)",
        ],
        "key_ddx": (
            "PAX3 (WS1): dystopia canthorum W ≥1.95; NO HSCR; NO neuropathy; "
            "EDNRB (WS4A): AR HSCR + WS; SOX10 is AD; no neuropathy in EDNRB; "
            "CMT1A/CMT4: peripheral neuropathy but NO WS features; NO HSCR; "
            "MITF (WS2A): no HSCR; no neuropathy; "
            "RET: isolated HSCR; no WS features; no neuropathy."
        ),
        "systemic_involvement": (
            "EAR: bilateral profound SNHL. "
            "GI: HSCR — aganglionosis rectosigmoid or total colonic; surgical emergency in neonate. "
            "PNS: peripheral demyelinating neuropathy (WS4C/PCWH). "
            "CNS: central hypomyelination (PCWH only) — MRI white matter changes. "
            "SKIN/EYE: depigmentation, iris heterochromia (as WS1/WS2)."
        ),
        "onset_age": "HSCR: neonatal (delayed meconium >48h); SNHL: congenital; Neuropathy: infancy–childhood",
        "surgical_urgency": "HSCR: urgent neonatal surgical assessment; pull-through surgery; CI for profound SNHL",
        "gene_family": "SOX (SRY-box) transcription factor family; Group E; HMG-box DNA binding",
        "morphology": (
            "AUDIOGRAM: bilateral profound flat SNHL; "
            "CT/MRI: NORMAL cochlear anatomy; white matter hyperintensity in PCWH (T2); "
            "NCS: demyelinating NCVs <38 m/s upper limbs; "
            "RECTAL BIOPSY: absent enteric ganglia — aganglionosis DIAGNOSTIC"
        ),
        "n_patients": 40,
    },

    # -- EDNRB -- Waardenburg Type 4A (Waardenburg-Shah Syndrome) -----------------------------------
    {
        "gene": "EDNRB",
        "alt_name": (
            "EDNRB (EDNRB-442aa-13q22.3 / AR-AD -- "
            "WS4A-WAARDENBURG-SHAH-HIRSCHSPRUNG-WS-FEATURES-AR-MOST-SEVERE -- "
            "S305N-MENNONITE-FOUNDER-MUTATION-EDNRB -- "
            "SNHL-BILATERAL-PROFOUND-CONGENITAL-COCHLEAR-IMPLANT -- "
            "TOTAL-COLONIC-AGANGLIONOSIS-RISK-HIGHER-THAN-RET -- "
            "HETEROZYGOUS-MILD-HSCR-ONLY-BIALLELIC-FULL-WS4A)"
        ),
        "protein": (
            "EDNRB -- 13q22.3 AR/AD -- EDNRB-442aa -- "
            "Endothelin-Receptor-Type-B-50kDa-7-TM-GPCR-ET-3-Ligand -- "
            "Enteric-Neural-Crest-Melanoblast-Survival-Migration -- "
            "OMIM-Gene-131244-Disease-WS4A-277580-HSCR2-600155"
        ),
        "locus": "13q22.3",
        "protein_size": "442 aa / 50 kDa",
        "inheritance": (
            "AR (biallelic for full WS4A — Waardenburg-Shah syndrome); "
            "AD/heterozygous (isolated HSCR only — incomplete penetrance ~20-30%); "
            "EDNRB is a G-protein coupled receptor for endothelin-3 (EDN3); "
            "FUNCTION: EDN3-EDNRB signalling in neural crest cells promotes melanoblast and enteric "
            "neural crest survival; also regulates timing of melanoblast differentiation. "
            "BIALLELIC (AR): full Waardenburg-Shah (WS4A) — HSCR + WS pigmentary/hearing features + SNHL; "
            "HETEROZYGOUS: isolated HSCR (~20-30% penetrance) — WS features often absent; "
            "S305N: Mennonite founder — p.Ser305Asn; heterozygous → HSCR; biallelic → WS4A; "
            "GENOTYPE-PHENOTYPE: biallelic null → most severe HSCR (total colonic aganglionosis risk); "
            "EDNRB and EDN3 mutations cause same spectrum — EDNRB more common."
        ),
        "disease_category": (
            "Waardenburg Syndrome Type 4A (WS4A / Waardenburg-Shah syndrome) — AR; "
            "Hirschsprung disease + WS features (iris heterochromia + SNHL + white forelock); "
            "Heterozygous EDNRB → isolated HSCR only (no WS features)"
        ),
        "disease_pathway": (
            "EDN3-EDNRB SIGNALLING IN NEURAL CREST: "
            "EDN3 (endothelin-3) secreted by gut mesenchyme + skin → binds EDNRB on neural crest cells → "
            "Gαq/11 → IP3/DAG → intracellular Ca2+ → prevents premature differentiation → "
            "allows sufficient proliferation and rostro-caudal migration. "
            "ENTERIC: EDNRB on enteric neural crest progenitors — EDN3 from gut mesenchyme delays "
            "differentiation → allows full colonisation of entire gut length; "
            "EDNRB LOF → premature differentiation → insufficient enteric progenitors → aganglionosis; "
            "biallelic → total colonic or long-segment HSCR (most severe aganglionosis extent). "
            "MELANOBLAST: EDNRB on melanoblasts — EDN3 from skin dermis delays terminal differentiation → "
            "allows sufficient melanoblast expansion before differentiating into melanocytes; "
            "EDNRB LOF → premature melanocyte commitment → insufficient melanocytes in stria vascularis/skin → "
            "SNHL + depigmentation (heterochromia, white forelock). "
            "WS4A vs WS4C: EDNRB (AR) vs SOX10 (AD) — both cause HSCR + WS but different inheritance."
        ),
        "pathognomonic": (
            "EDNRB / WS4A PATHOGNOMONIC FEATURES: "
            "1. BIALLELIC EDNRB → FULL WS4A: "
            "Hirschsprung disease + SNHL (bilateral profound) + iris heterochromia + white forelock; "
            "2. HSCR: delayed meconium >48h; constipation; abdominal distension; "
            "suction rectal biopsy: aganglionosis — CONFIRMS DIAGNOSIS; "
            "total colonic aganglionosis risk HIGHER than RET-associated HSCR; "
            "3. HETEROZYGOUS EDNRB: "
            "isolated HSCR only (incomplete penetrance ~20-30%); usually short-segment; "
            "NO WS features in heterozygotes — KEY DDx; "
            "4. S305N MENNONITE FOUNDER: "
            "biallelic S305N → full WS4A (HSCR + WS); screen Mennonite populations; "
            "5. WS FEATURES in biallelic: iris heterochromia; white forelock; SNHL bilateral profound; "
            "W-index NORMAL (no dystopia canthorum); "
            "6. COCHLEAR CT: NORMAL — no EVA, no semicircular canal aplasia."
        ),
        "treatment": (
            "HSCR: urgent neonatal surgical assessment; pull-through colectomy; "
            "total colonic aganglionosis needs total colectomy + ileoanal pouch; "
            "SNHL: bilateral CI for profound SNHL; outcomes good; implant early; "
            "GENETIC COUNSELLING: AR 25% risk for biallelic (WS4A); "
            "heterozygous EDNRB carriers → ~20-30% risk isolated HSCR; "
            "cascade family testing: rectal biopsy + audiology + molecular in relatives; "
            "PRE-CONCEPTION: carrier testing offered to Mennonite families (S305N)."
        ),
        "key_features": [
            "WS4A (Waardenburg-Shah): AR biallelic EDNRB → HSCR + WS features (SNHL + heterochromia)",
            "Heterozygous EDNRB → isolated HSCR only (~20-30% penetrance) — NO WS features",
            "S305N Mennonite founder: biallelic → WS4A; heterozygous → HSCR",
            "Total colonic aganglionosis risk higher than RET-associated HSCR",
            "Suction rectal biopsy: aganglionosis PATHOGNOMONIC for HSCR",
            "EDN3-EDNRB delays neural crest differentiation → allows full gut colonisation",
            "W-index NORMAL (no dystopia canthorum) — DDx WS1/PAX3",
            "Bilateral profound SNHL; CI effective; early implantation recommended",
        ],
        "key_ddx": (
            "SOX10 (WS4C): AD; peripheral demyelinating neuropathy in PCWH; SOX10 AD vs EDNRB AR; "
            "PAX3 (WS1): dystopia canthorum; NO HSCR; "
            "RET (HSCR1): most common HSCR gene; AD; NO WS features; short-segment more common; "
            "EDN3 (WS4B): same spectrum as EDNRB — AR; endothelin-3 ligand for same receptor; "
            "MITF (WS2A): no HSCR; no neuropathy."
        ),
        "systemic_involvement": (
            "GI: HSCR — aganglionosis; neonatal constipation; abdominal distension; risk total colonic. "
            "EAR: bilateral profound SNHL (biallelic). "
            "EYE/SKIN: iris heterochromia; white forelock (biallelic WS4A). "
            "NO neuropathy (DDx SOX10/PCWH); NO dystopia canthorum."
        ),
        "onset_age": "HSCR: neonatal; SNHL: congenital (biallelic); heterozygous: HSCR variable onset",
        "surgical_urgency": "Neonatal HSCR surgical emergency; bilateral CI for profound SNHL",
        "gene_family": "Endothelin receptor (EDRB) family; 7-TM GPCR; Gαq-coupled",
        "morphology": (
            "AUDIOGRAM: bilateral profound SNHL (biallelic); "
            "CT/MRI: NORMAL cochlear; HSCR contrast enema: narrow segment + transition zone; "
            "RECTAL BIOPSY: absent ganglia; absent acetylcholinesterase staining"
        ),
        "n_patients": 40,
    },

    # -- EYA1 -- Branchiootorenal (BOR) Syndrome -----------------------------------------------------
    {
        "gene": "EYA1",
        "alt_name": (
            "EYA1 (EYA1-559aa-8q13.3 / AD -- "
            "BOR-BRANCHIOOTORENAL-SYNDROME-BO-BRANCHIOOTIC -- "
            "BRANCHIAL-ANOMALIES-FISTULAE-CYSTS-TAGS-PATHOGNOMONIC-TRIAD -- "
            "OTOLOGIC-ANOMALIES-MIDDLE-EAR-OSSICULAR-CHL-INNER-EAR-SNHL -- "
            "RENAL-DYSPLASIA-HYPOPLASIA-AGENESIS-RENAL-FUNCTION-BASELINE-MANDATORY -- "
            "SIX1-SIX5-COFACTORS-EYA1-COMPLEX)"
        ),
        "protein": (
            "EYA1 -- 8q13.3 AD -- EYA1-559aa -- "
            "EYA-Transcriptional-Coactivator-Phosphatase-1-61kDa-EYA-Domain -- "
            "Eyes-Absent-Homologue-1-Six-Complex-Ear-Kidney-Branchial-Development -- "
            "OMIM-Gene-601653-Disease-BOR-113650-BO-120502"
        ),
        "locus": "8q13.3",
        "protein_size": "559 aa / 61 kDa",
        "inheritance": (
            "AD (autosomal dominant; haploinsufficiency; variable expressivity); "
            "EYA1 encodes a transcriptional coactivator with protein tyrosine phosphatase activity. "
            "FUNCTION: EYA1 functions with SIX1 and SIX5 in the SIX/EYA transcription complex; "
            "COMPLEX: SIX1-EYA1 heterodimer activates target genes in branchial arch, otic placode, "
            "and metanephric kidney development. "
            "LOF CONSEQUENCES: branchial arch → branchial clefts/fistulae/sinuses/tags remain; "
            "otic placode → outer/middle/inner ear developmental defects; "
            "metanephric kidney → renal dysplasia, hypoplasia, or agenesis. "
            "PENETRANCE: ~100% for branchial anomalies; SNHL/CHL ~80-90%; renal anomalies ~67%; "
            "EXPRESSIVITY: wide intrafamilial variation — same mutation can produce different subsets; "
            "GENOTYPE-PHENOTYPE: EYA domain mutations tend to be more severe; "
            "SIX1 mutations (same pathway) → BO only (no renal — renal requires EYA1). "
            "BRANCHIOOTIC (BO): same ear + branchial without renal anomalies (allelic)."
        ),
        "disease_category": (
            "Branchiootorenal (BOR) syndrome — AD; Branchial anomalies + Otologic anomalies + Renal dysplasia; "
            "Branchiootic (BO) syndrome — ear + branchial only (no renal); "
            "SNHL + CHL mixed; variable expressivity"
        ),
        "disease_pathway": (
            "EYA1 IN ORGANOGENESIS: "
            "BRANCHIAL ARCHES: EYA1 expressed in 1st and 2nd branchial arch-derived mesenchyme; "
            "EYA1 LOF → incomplete involution of branchial apparatus → "
            "persistent cleft sinuses, fistulae (tracts opening anterior sternocleidomastoid), "
            "preauricular tags/pits (1st arch remnants). "
            "OTIC PLACODE: EYA1 required for otic placode induction and patterning; "
            "EYA1 LOF → dysplastic cochlea (Mondini-like), dysplastic semicircular canals, "
            "malformed ossicles (incudomallear, stapes), absent or stenotic EAC. "
            "HEARING TYPES: SNHL (cochlear dysplasia) ± CHL (ossicular/EAC anomaly) ± mixed; "
            "any combination possible; CHL alone or SNHL alone can occur. "
            "KIDNEY: EYA1 + SIX2 in metanephric mesenchyme — LOF → renal dysplasia/hypoplasia/agenesis; "
            "unilateral or bilateral; may be subclinical or cause chronic kidney disease; "
            "RENAL FUNCTION MANDATORY AT DIAGNOSIS: GFR measurement; if bilateral renal anomalies → "
            "nephrology referral; risk of end-stage renal disease in minority."
        ),
        "pathognomonic": (
            "EYA1 / BOR SYNDROME PATHOGNOMONIC FEATURES: "
            "1. BRANCHIAL ANOMALIES (nearly 100% penetrant): "
            "— Branchial fistulae/sinuses: tracts opening on anterior border SCM, "
            "bilaterally (bilateral in ~60%); may drain saliva or mucus; "
            "— Branchial cysts: deep-seated lateral neck masses; "
            "— Preauricular tags/pits: small skin tags or pits anterior to tragus; "
            "2. OTOLOGIC ANOMALIES: "
            "— SNHL: moderate-severe; flat or sloping; cochlear dysplasia (Mondini malformation); "
            "— CHL: ossicular anomalies; EAC stenosis/atresia; "
            "— Mixed HL most common; unilateral or bilateral; variable severity; "
            "CT temporal bone: Mondini cochlea (1.5 turns), dysplastic semicircular canals, "
            "malleus/incus anomalies — KEY IMAGING; "
            "3. RENAL ANOMALIES (~67%): "
            "renal dysplasia/hypoplasia; unilateral/bilateral renal agenesis; duplex kidney; "
            "horseshoe kidney; chronic kidney disease risk. "
            "4. DIAGNOSIS: ≥2 of 3 triad (branchial + otic + renal) + family history = CLINICAL DIAGNOSIS; "
            "EYA1 molecular confirmation."
        ),
        "treatment": (
            "HEARING AIDS: CHL or SNHL management; BAHA (bone-anchored) for CHL with atresia; "
            "COCHLEAR IMPLANT: for severe-profound SNHL; Mondini cochlear anatomy — CI still feasible; "
            "discuss with CI team; partial turn cochlea requires surgical planning; "
            "SURGICAL — EAC/OSSICLE: middle ear exploration + ossiculoplasty; "
            "atresia repair — careful pre-op CT; "
            "BRANCHIAL FISTULAE: elective surgical excision (recurrent infections risk); "
            "complete fistula tract resection required to prevent recurrence; "
            "RENAL: annual GFR/creatinine; renal USS; nephrology co-management; "
            "avoid nephrotoxic drugs (aminoglycosides — compounding renal risk); "
            "GENETIC COUNSELLING: AD 50%; wide variable expressivity even within family."
        ),
        "key_features": [
            "BOR triad: Branchial anomalies (fistulae/cysts/tags) + Otologic anomalies + Renal dysplasia",
            "Branchial fistulae/sinuses anterior to SCM: nearly 100% penetrant; bilateral 60%",
            "Mixed hearing loss (SNHL + CHL): Mondini cochlea + ossicular anomalies on CT",
            "Renal dysplasia/agenesis ~67%: GFR mandatory at diagnosis; nephrology co-management",
            "AVOID aminoglycosides: ototoxic + nephrotoxic double jeopardy in BOR",
            "EYA1-SIX1-SIX5 complex: otic placode + branchial arch + metanephric kidney",
            "SIX1 mutations → BO only (no renal); EYA1 → full BOR triad",
            "CI feasible in Mondini cochlea — pre-operative CT temporal bone planning essential",
        ],
        "key_ddx": (
            "SIX1/SIX5 (BO syndrome): branchial + otic WITHOUT renal — same EYA complex pathway; "
            "CHARGE (CHD7): coloboma + heart + choanal atresia + semicircular canal aplasia; "
            "Treacher Collins (TCOF1): mandibulofacial dysostosis; absent malar/zygoma; CHL; "
            "Isolated HSCR (RET/EDNRB): no branchial/otic/renal triad; "
            "Second branchial cleft remnants (sporadic): no hearing or renal anomalies; negative EYA1."
        ),
        "systemic_involvement": (
            "BRANCHIAL: fistulae, cysts, preauricular tags/pits. "
            "EAR: SNHL + CHL; Mondini cochlea; ossicular/EAC anomalies. "
            "KIDNEY: dysplasia, hypoplasia, agenesis — renal function monitoring mandatory. "
            "NO CARDIAC; NO COLOBOMA; NO CHOANAL ATRESIA (DDx CHARGE)."
        ),
        "onset_age": "Branchial anomalies visible at birth; SNHL identified neonatal screen; renal anomalies antenatal US",
        "surgical_urgency": "Branchial fistulae: elective; SNHL: early amplification/CI; renal function: monitor from birth",
        "gene_family": "EYA (eyes absent) protein family; EYA-domain phosphatase; SIX complex coactivator",
        "morphology": (
            "AUDIOGRAM: mixed HL (SNHL + CHL) or pure SNHL or pure CHL; asymmetric common; "
            "CT temporal bone: Mondini cochlea (1.5 turns); dysplastic SCC; ossicular anomalies; "
            "RENAL USS: dysplasia, small kidney, duplex, horseshoe; "
            "BRANCHIAL: fistulae tracts anterior SCM; preauricular tags anterior to tragus"
        ),
        "n_patients": 40,
    },

    # -- CHD7 -- CHARGE Syndrome (Most Common Syndromic SNHL in Neonates) ---------------------------
    {
        "gene": "CHD7",
        "alt_name": (
            "CHD7 (CHD7-2997aa-8q12.2 / AD-de-novo -- "
            "CHARGE-SYNDROME-C-Coloboma-H-Heart-A-choanal-Atresia-R-Retardation-G-Genital-E-Ear -- "
            "MOST-COMMON-SYNDROMIC-SNHL-NEONATES-CHARGE-AIRWAY-EMERGENCY-FIRST -- "
            "SEMICIRCULAR-CANAL-APLASIA-CT-NEAR-PATHOGNOMONIC-CHARGE -- "
            "SCC-APLASIA-BALANCE-ABSENT-HYPOTONIA-MOTOR-DELAY -- "
            "COLOBOMA-OPTIC-NERVE-CHORIORETINAL-VISUAL-FIELD-LOSS)"
        ),
        "protein": (
            "CHD7 -- 8q12.2 AD (de novo 60-70%) -- CHD7-2997aa -- "
            "Chromodomain-Helicase-DNA-Binding-7-337kDa-CHD-ATPase-Remodeller -- "
            "Neural-Crest-Multi-Organ-Chromatin-Remodelling -- "
            "OMIM-Gene-608892-Disease-CHARGE-214800"
        ),
        "locus": "8q12.2",
        "protein_size": "2997 aa / 337 kDa",
        "inheritance": (
            "AD (autosomal dominant; ~60-70% de novo mutations; 30-40% inherited); "
            "CHD7 encodes a chromodomain helicase DNA-binding protein — ATP-dependent chromatin remodeller; "
            "FUNCTION: CHD7 reads H3K4me1 marks (enhancers) via chromodomain → opens chromatin → "
            "activates enhancer-driven genes in neural crest cells and multiple organ primordia; "
            "CHD7 LOF → global transcriptional dysregulation in neural crest → CHARGE multi-organ syndrome. "
            "DE NOVO RATE: ~60-70% — most affected individuals have unaffected parents; "
            "PENETRANCE: near-complete; recurrence risk for unaffected parents = ~1-2% (gonadal mosaicism); "
            "EXPRESSIVITY: wide — some features severe (airway), others mild (genital); "
            "GENOTYPE-PHENOTYPE: weak — same mutation in different individuals → different features; "
            "truncating variants most common; missense in chromodomain/helicase → severe."
        ),
        "disease_category": (
            "CHARGE syndrome — AD (mostly de novo); C=Coloboma, H=Heart defect, A=choanal Atresia, "
            "R=Retardation (growth + developmental), G=Genital anomaly, E=Ear (SNHL + dysplasia); "
            "MOST COMMON cause of syndromic SNHL identified in neonates; "
            "Semicircular canal aplasia on CT near-pathognomonic"
        ),
        "disease_pathway": (
            "CHD7 IN NEURAL CREST AND ORGANOGENESIS: "
            "CHD7 chromodomain binds H3K4me1 (active enhancer marks) → remodels nucleosomes → "
            "opens chromatin at organ-specific enhancers → activates transcription; "
            "CHD7 LOF → enhancer failure across multiple tissues simultaneously → "
            "COLOBOMA: CHD7 in retinal ganglion cell + optic fissure closure — LOF → failed fissure closure → "
            "iris/retina/optic nerve coloboma → inferior visual field defect; "
            "HEART: CHD7 in cardiac neural crest → conotruncal defects (ToF, interrupted aortic arch, VSD, ASD); "
            "CHOANAL ATRESIA: CHD7 in nasal epithelial-mesenchymal fusion → LOF → bony/membranous atresia; "
            "bilateral choanal atresia = neonatal airway EMERGENCY (obligate nasal breathers); "
            "EAR: CHD7 in otic placode + semicircular canal epithelium → LOF → "
            "SEMICIRCULAR CANAL APLASIA (all 3 canals absent or hypoplastic) — CT hallmark of CHARGE; "
            "cochlear hypoplasia → severe-profound SNHL; "
            "GENITAL: CHD7 in hypothalamic-pituitary axis → hypogonadotropic hypogonadism; "
            "BALANCE: absent semicircular canals → absent vestibulo-ocular reflex → "
            "profound vestibular areflexia → motor delay (late walking 3-5y); hypotonia."
        ),
        "pathognomonic": (
            "CHD7 / CHARGE SYNDROME PATHOGNOMONIC FEATURES: "
            "1. COLOBOMA: iris coloboma (keyhole pupil); chorioretinal coloboma; optic nerve coloboma; "
            "inferior visual field defect; unilateral or bilateral; "
            "2. CHOANAL ATRESIA: "
            "bilateral = NEONATAL AIRWAY EMERGENCY — obligate nasal breathers; "
            "unilateral = unilateral nasal obstruction ± persistent unilateral nasal discharge; "
            "3. SEMICIRCULAR CANAL APLASIA/HYPOPLASIA (CT): "
            "all 3 SCC absent or hypoplastic — NEAR-PATHOGNOMONIC for CHARGE; "
            "balance absent → profound motor delay; late walking; falls; "
            "4. SNHL: severe-profound bilateral; cochlear hypoplasia ± Mondini; "
            "cochlear implant effective but vestibular areflexia complicates balance rehabilitation; "
            "5. CARDIAC DEFECTS: conotruncal — ToF, interrupted aortic arch, VSD; "
            "echo MANDATORY at diagnosis; "
            "6. GENITAL ANOMALIES: micropenis; cryptorchidism (males); "
            "delayed/absent puberty — hypogonadotropic hypogonadism; "
            "7. BALANCE: absent SCC → absent VOR → profound vestibular areflexia → "
            "motor delay; use Romberg/foam; delay CI if vestibular rehab not yet started. "
            "DIAGNOSTIC CRITERIA: 3 major (coloboma/choanal atresia/SCC anomaly/SNHL) = clinical CHARGE."
        ),
        "treatment": (
            "AIRWAY EMERGENCY (bilateral choanal atresia): "
            "IMMEDIATE: oral airway / McGovern nipple / nasal stents; "
            "SURGICAL: choanoplasty (transnasal endoscopic preferred); "
            "CARDIAC: paediatric cardiology; surgical repair per defect type; "
            "HEARING: bilateral profound SNHL → bilateral CI; "
            "CI unique considerations in CHARGE: "
            "— absent SCCs → no canal-related anatomy to avoid; "
            "— vestibular areflexia persists post-CI — balance rehab mandatory; "
            "— bilateral CI preferred for binaural input despite vestibular limitation; "
            "VISION: ophthalmology; low vision aids for coloboma; "
            "DEVELOPMENT: early intervention; multidisciplinary team; "
            "GROWTH/PUBERTY: GH supplementation if GH deficient; hormone replacement at puberty; "
            "GENETIC COUNSELLING: ~60-70% de novo; recurrence <2%; parental testing mandatory."
        ),
        "key_features": [
            "CHARGE: Coloboma + Heart + choanal Atresia + Retardation + Genital + Ear (SNHL)",
            "Semicircular canal aplasia/hypoplasia CT — near-pathognomonic for CHARGE",
            "Bilateral choanal atresia: NEONATAL AIRWAY EMERGENCY (obligate nasal breathers)",
            "Absent VOR + profound vestibular areflexia → severe motor delay (walks at 3-5 years)",
            "Most common syndromic SNHL cause in neonates; de novo ~60-70%",
            "Bilateral CI effective; vestibular areflexia persists — balance rehab mandatory post-CI",
            "Chromatin remodeller: H3K4me1 enhancer binding; LOF disrupts multiple organ enhancers",
            "CHD7 2997 aa — largest syndromic hearing loss gene; wide expressivity (same mutation, different organs)",
        ],
        "key_ddx": (
            "EYA1 (BOR): branchial fistulae + renal; NO coloboma; NO choanal atresia; "
            "TCOF1 (Treacher Collins): mandibulofacial dysostosis; absent malar/zygoma; CHL dominant; "
            "KCNQ1/KCNE1 (JLNS): SNHL + QTc prolongation; no other CHARGE features; "
            "SOX10 (WS4C): HSCR + neuropathy; NO coloboma; NO choanal atresia; "
            "Isolated SNHL: normal CT SCC (CHARGE has absent/hypoplastic SCC as hallmark)."
        ),
        "systemic_involvement": (
            "EYE: coloboma (iris/chorioretinal/optic nerve). "
            "HEART: conotruncal defects — ToF, interrupted aortic arch, VSD. "
            "AIRWAY: choanal atresia — bilateral = neonatal emergency. "
            "EAR: SNHL + SCC aplasia + vestibular areflexia. "
            "GENITAL: hypogonadotropic hypogonadism. "
            "CNS: developmental delay; hypotonia; motor delay. "
            "BALANCE: absent VOR; Romberg positive."
        ),
        "onset_age": "Neonatal; choanal atresia = immediate airway emergency; SNHL identified on newborn screen",
        "surgical_urgency": "IMMEDIATE: bilateral choanal atresia = neonatal airway emergency; cardiac surgical assessment; CI bilateral for SNHL",
        "gene_family": "CHD (chromodomain helicase DNA-binding) protein family; CHD7-9 subfamily; SWI/SNF-like ATPase",
        "morphology": (
            "AUDIOGRAM: bilateral severe-profound SNHL; flat; "
            "CT temporal bone: absent/hypoplastic SCC — PATHOGNOMONIC; cochlear hypoplasia; "
            "CT nose: bony/membranous choanal atresia; "
            "ECHO: conotruncal cardiac defect; "
            "MRI brain: white matter delay; olfactory bulb hypoplasia"
        ),
        "n_patients": 40,
    },

    # -- TCOF1 -- Treacher Collins Syndrome (Mandibulofacial Dysostosis) ----------------------------
    {
        "gene": "TCOF1",
        "alt_name": (
            "TCOF1 (TCOF1-1411aa-5q33.1 / AD -- "
            "TREACHER-COLLINS-SYNDROME-TCS1-MANDIBULOFACIAL-DYSOSTOSIS-FRANCESCHETTI-KLEIN -- "
            "ABSENT-HYPOPLASTIC-ZYGOMA-MALAR-BONES-ZYGOMATIC-ARCH-PATHOGNOMONIC -- "
            "CHL-PREDOMINANTLY-OSSICULAR-EAC-STENOSIS-ATRESIA-BAHA-FIRST-LINE -- "
            "MICROGNATHIA-AIRWAY-MANAGEMENT-NEONATAL -- "
            "TREACLE-RIBOSOME-BIOGENESIS-NUCLEOLUS)"
        ),
        "protein": (
            "TCOF1 -- 5q33.1 AD -- TCOF1-1411aa -- "
            "Treacle-Ribosome-Biogenesis-Factor-1-152kDa-LIS1-Homology-Serine-Rich -- "
            "Nucleolar-RNA-Pol-I-Neural-Crest-Ribosome-Biogenesis -- "
            "OMIM-Gene-606847-Disease-TCS1-154500"
        ),
        "locus": "5q33.1",
        "protein_size": "1411 aa / 152 kDa",
        "inheritance": (
            "AD (autosomal dominant; haploinsufficiency; 40-50% de novo); "
            "TCOF1 encodes Treacle, a nucleolar phosphoprotein involved in rRNA processing and ribosome biogenesis. "
            "FUNCTION: Treacle interacts with UBF (upstream binding factor) and RNA Pol I → "
            "activates rRNA gene transcription → ribosome assembly → neural crest cell proliferation; "
            "TCOF1 LOF → reduced ribosome biogenesis → impaired neural crest cell proliferation → "
            "insufficient neural crest-derived mesenchyme in 1st and 2nd branchial arches → "
            "absent/hypoplastic craniofacial bones derived from neural crest (zygoma, malar, zygomatic arch, "
            "mandibular condyle, middle ear ossicles). "
            "P53 APOPTOSIS: TCOF1 LOF → nucleolar stress → p53 stabilisation → "
            "neural crest apoptosis → worsens neural crest deficit; "
            "PENETRANCE: ~100%; EXPRESSIVITY: wide intrafamilial variation; "
            "40-50% de novo; inherited cases show variable severity. "
            "POLR1C/POLR1D: RNA Pol I subunits causing TCS2/TCS3 (AR inheritance — different from TCS1)."
        ),
        "disease_category": (
            "Treacher Collins Syndrome Type 1 (TCS1 / Mandibulofacial Dysostosis Franceschetti-Klein); "
            "AD; absent/hypoplastic malar + zygoma + zygomatic arch; "
            "Predominantly CHL (ossicular/EAC) ± mixed SNHL; micrognathia"
        ),
        "disease_pathway": (
            "TREACLE IN NEURAL CREST RIBOSOME BIOGENESIS: "
            "TCOF1 → Treacle in nucleolus → interacts with UBF → stimulates rDNA transcription → "
            "rRNA processed → 40S + 60S ribosomes assembled → protein synthesis capacity of neural crest cells. "
            "NEURAL CREST PROLIFERATION: 1st and 2nd branchial arch neural crest cells undergo rapid "
            "proliferation requiring high ribosomal capacity; "
            "TCOF1 haploinsufficiency → ribosome insufficiency → reduced proliferation → "
            "p53 accumulates (nucleolar stress) → apoptosis → neural crest cell loss. "
            "CRANIOFACIAL CONSEQUENCE: "
            "1st branchial arch derivatives reduced: zygoma, malar bones, incus, malleus, mandible; "
            "2nd branchial arch: stapes, external ear (EAC), pinna; "
            "RESULT: absent/hypoplastic zygoma (key feature); malar hypoplasia; "
            "EAC stenosis/atresia; malleus-incus anomaly → CHL (predominantly); "
            "cochlea usually NORMAL → SNHL rare unless cochlear dysplasia; "
            "micrognathia → airway compromise → Pierre Robin-like sequence. "
            "COLOBOMA EYELIDS: lower eyelid coloboma in some (absent cilia, lateral coloboma) — "
            "DIFFERENT from CHD7 retinal coloboma."
        ),
        "pathognomonic": (
            "TCOF1 / TCS PATHOGNOMONIC FEATURES: "
            "1. ABSENT/HYPOPLASTIC ZYGOMA AND MALAR BONES: "
            "malar eminences absent; flat midface; antimongoloid (downslanting) palpebral fissures; "
            "3D CT face: absent zygomatic arch, malar, and zygomatic process — RADIOLOGICAL PATHOGNOMONIC; "
            "2. LOWER EYELID COLOBOMA (50-60%): lateral 1/3 lower eyelid notch; "
            "absent lower eyelid lashes (distinct from CHD7 retinal coloboma); "
            "3. MICROGNATHIA + RETROGNATHIA: small mandible, retrusive chin; "
            "NEONATAL AIRWAY CONCERN: Pierre Robin sequence-like; "
            "4. EAR ANOMALIES: "
            "— EAC stenosis or bilateral atresia → CHL; "
            "— Malformed/absent pinna (microtia Grade 1-3); "
            "— Ossicular chain malformation (incudomallear fusion, absent stapes superstructure); "
            "— Cochlea NORMAL → pure CHL; rarely mixed; "
            "5. HEARING: predominantly CHL 40-100 dB; "
            "BAHA (bone-anchored hearing aid) FIRST-LINE for bilateral CHL with atresia; "
            "unilateral fitting at 6 months on headband; implant BAHA at 4-5 years; "
            "6. FACIAL HAIR: absence of facial hair/eyelashes in coloboma area."
        ),
        "treatment": (
            "HEARING — BAHA FIRST-LINE: bilateral CHL with atresia → bone-anchored hearing aid; "
            "BAHA headband from 6 months; BAHA implant (Baha5/Osia) at 4-5 years; "
            "conventional hearing aids if EAC present (mild stenosis); "
            "ATRESIA SURGERY: EAC reconstruction possible in selected patients (CT assessment); "
            "high complication rate — BAHA generally preferred over atresia surgery; "
            "AIRWAY: neonatal airway assessment; mandibular distraction osteogenesis (MDO) for severe "
            "micrognathia to avoid tracheostomy; "
            "ORBITAL/EYELID: lower eyelid coloboma — reconstruct to protect cornea; "
            "CRANIOFACIAL SURGERY: staged: MDO in infancy; zygoma/orbital floor reconstruction 5-7y; "
            "orthognathic surgery in adolescence; "
            "GENETIC COUNSELLING: AD 50% risk; 40-50% de novo; molecular confirmation TCS1."
        ),
        "key_features": [
            "Absent/hypoplastic zygoma + malar bones — PATHOGNOMONIC (3D CT face)",
            "Predominantly CHL: EAC atresia + ossicular anomaly; cochlea usually NORMAL",
            "BAHA (bone-anchored) FIRST-LINE for bilateral CHL with atresia from 6 months",
            "Lower eyelid coloboma (lateral 1/3): absent eyelashes — distinct from CHD7 retinal coloboma",
            "Micrognathia: neonatal airway concern; mandibular distraction osteogenesis",
            "Treacle (TCOF1): nucleolar rRNA transcription; neural crest ribosome biogenesis",
            "TCOF1 haploinsufficiency → nucleolar stress → p53 → neural crest apoptosis",
            "AD; 40-50% de novo; TCS2/TCS3 (POLR1C/POLR1D) AR — different inheritance",
        ],
        "key_ddx": (
            "CHD7 (CHARGE): semicircular canal aplasia; coloboma is RETINAL (not eyelid); choanal atresia; "
            "EYA1 (BOR): branchial fistulae; renal; Mondini; no malar/zygoma absence; "
            "Nager syndrome (SF3B4): similar facial but preaxial limb anomalies (absent/hypoplastic thumb); "
            "Miller syndrome (DHODH): postaxial limb anomalies; "
            "Goldenhar/OAV: unilateral; dermoid; spine; more asymmetric; "
            "isolated microtia: no malar/zygoma/eyelid features."
        ),
        "systemic_involvement": (
            "FACE: absent malar/zygoma; downslanting palpebral fissures; antimongoloid slant. "
            "EAR: CHL (EAC atresia + ossicular anomaly); microtia. "
            "EYE: lower eyelid coloboma; corneal exposure risk. "
            "AIRWAY: micrognathia + retrognathia. "
            "NO cardiac; NO renal; NO coloboma retinal (DDx CHD7)."
        ),
        "onset_age": "Congenital; CHL identified on BERA/ABR neonatal; facial anomalies visible at birth",
        "surgical_urgency": "BAHA headband at 6 months for CHL; neonatal airway if severe micrognathia (MDO vs tracheostomy)",
        "gene_family": "Treacle/TCOF family; LIS1-homology motif; serine-rich domain; nucleolar phosphoprotein",
        "morphology": (
            "AUDIOGRAM: bilateral CHL 40-100 dB; flat (air-bone gap); bone conduction NORMAL; "
            "CT temporal bone: EAC atresia; ossicular fusion/malformation; NORMAL cochlea; "
            "CT face 3D: absent zygomatic arch and malar eminence DIAGNOSTIC; "
            "PINNA: microtia grade 1-3"
        ),
        "n_patients": 40,
    },

    # -- GATA3 -- HDR Syndrome / Barakat Syndrome (Hypoparathyroidism-Deafness-Renal) ---------------
    {
        "gene": "GATA3",
        "alt_name": (
            "GATA3 (GATA3-444aa-10p15.3 / AD -- "
            "HDR-SYNDROME-HYPOPARATHYROIDISM-DEAFNESS-RENAL-DYSPLASIA-BARAKAT -- "
            "BILATERAL-SNHL-BILATERAL-HYPOCALCAEMIA-RENAL-ANOMALIES-TRIAD -- "
            "CALCIUM-CORRECTION-BEFORE-AUDIOMETRY-HYPOCALCAEMIA-WORSENS-HEARING -- "
            "TETANY-SEIZURES-HYPOCALCAEMIA-EMERGENCY-IV-CALCIUM -- "
            "PARATHYROID-APLASIA-HYPOPLASIA-LOW-PTH-LOW-CALCIUM-HIGH-PHOSPHATE)"
        ),
        "protein": (
            "GATA3 -- 10p15.3 AD -- GATA3-444aa -- "
            "GATA-Binding-Transcription-Factor-3-48kDa-Dual-Zinc-Finger-GATA-Motif -- "
            "Parathyroid-Inner-Ear-Kidney-Thymus-T-Cell-Development -- "
            "OMIM-Gene-131320-Disease-HDR-146255"
        ),
        "locus": "10p15.3",
        "protein_size": "444 aa / 48 kDa",
        "inheritance": (
            "AD (autosomal dominant; haploinsufficiency); "
            "GATA3 is a dual zinc-finger transcription factor (GATA-binding) critical for multiple organ development. "
            "FUNCTION: GATA3 activates target genes in parathyroid gland development, inner ear development, "
            "kidney development, and T-lymphocyte differentiation (CD4+ helper T cell lineage). "
            "LOF → HDR triad: hypoparathyroidism + deafness + renal dysplasia. "
            "PARATHYROID: GATA3 required for parathyroid gland specification from 3rd/4th pharyngeal pouch → "
            "LOF → parathyroid aplasia/hypoplasia → absent PTH → hypocalcaemia + hyperphosphataemia; "
            "INNER EAR: GATA3 in otic vesicle development → LOF → cochlear/vestibular anomalies → SNHL; "
            "KIDNEY: GATA3 in ureteric bud + metanephric mesenchyme → LOF → renal dysplasia/hypoplasia/agenesis; "
            "PENETRANCE: ~100% for SNHL; hypoparathyroidism ~90%; renal anomalies ~80%; "
            "EXPRESSIVITY: variable; hypoparathyroidism may be subclinical (low-normal PTH); "
            "GENOTYPE: haploinsufficiency (most deletions + frameshift); missense in zinc fingers can be "
            "dominant-negative."
        ),
        "disease_category": (
            "HDR syndrome (Hypoparathyroidism-Deafness-Renal dysplasia) / Barakat syndrome — AD; "
            "Triad: bilateral SNHL + hypoparathyroidism (hypocalcaemia) + renal dysplasia; "
            "SNHL bilateral; calcium correction mandatory before audiometry"
        ),
        "disease_pathway": (
            "GATA3 IN THREE ORGAN SYSTEMS: "
            "PARATHYROID: GATA3 activates genes in 3rd/4th pharyngeal pouch epithelium → "
            "parathyroid chief cell specification; "
            "GATA3 LOF → parathyroid gland aplasia or hypoplasia → absent or severely reduced PTH → "
            "HYPOPARATHYROIDISM: LOW PTH + LOW calcium + HIGH phosphate; "
            "symptoms: paraesthesiae, muscle cramps, carpopedal spasm (Trousseau's sign), "
            "laryngospasm, tetanic seizures; "
            "Chvostek's sign (facial nerve percussion spasm); "
            "INNER EAR: GATA3 in otic cup + otic vesicle patterning → cochlear + vestibular development; "
            "GATA3 LOF → cochlear dysplasia → bilateral SNHL (variable severity, usually moderate-severe); "
            "HYPOCALCAEMIA EFFECT ON HEARING: low ionised calcium impairs hair cell and auditory neuron function → "
            "CORRECT CALCIUM BEFORE AUDIOMETRY to avoid confounding the audiogram; "
            "KIDNEY: GATA3 in ureteric bud branching + collecting duct → renal anomalies: "
            "renal dysplasia, vesicoureteric reflux, renal agenesis, hydronephrosis; "
            "RENAL INSUFFICIENCY: bilateral renal anomalies → chronic kidney disease risk."
        ),
        "pathognomonic": (
            "GATA3 / HDR SYNDROME PATHOGNOMONIC FEATURES: "
            "1. HDR TRIAD: "
            "hypoparathyroidism + bilateral SNHL + renal dysplasia = CLINICAL DIAGNOSIS; "
            "2. HYPOPARATHYROIDISM: "
            "LOW serum calcium (<2.0 mmol/L); LOW/absent PTH; HIGH phosphate; "
            "SYMPTOMS: Chvostek's sign; Trousseau's sign; carpopedal spasm; "
            "tetanic seizures; laryngospasm — EMERGENCY; "
            "ECG: prolonged QTc (hypocalcaemia) — monitor; "
            "3. BILATERAL SNHL: "
            "moderate-severe bilateral; audiogram: flat or sloping; "
            "CALCIUM CORRECTION MANDATORY BEFORE AUDIOMETRY — hypocalcaemia worsens hearing; "
            "measure audiogram only after calcium normalised; "
            "4. RENAL ANOMALIES: "
            "renal dysplasia, hypoplasia, unilateral/bilateral agenesis, VUR, hydronephrosis; "
            "RENAL USS MANDATORY at diagnosis; GFR measurement; "
            "5. ABSENT/LOW PTH: distinguishes HDR from other SNHL syndromes; "
            "test PTH + calcium + phosphate in ALL children with SNHL — "
            "HDR may present as isolated SNHL before other triad features appear; "
            "6. ECG: QTc prolongation (hypocalcaemia effect) — cardiac monitoring."
        ),
        "treatment": (
            "HYPOPARATHYROIDISM — CALCIUM REPLACEMENT: "
            "oral calcium carbonate + calcitriol (1,25-OH2D3) — LIFELONG; "
            "target calcium 2.0-2.25 mmol/L (lower than normal to avoid hypercalciuria and renal stones); "
            "DO NOT over-correct — hypercalciuria damages kidneys; "
            "ACUTE HYPOCALCAEMIA EMERGENCY: IV calcium gluconate 10% (slow IV); "
            "TETANY/SEIZURE: IV calcium first; anti-epileptic drugs secondary; "
            "RECOMBINANT PTH (Natpara/PTH1-34): specialist indication for severe cases; "
            "HEARING: "
            "hearing aids for moderate SNHL; CI for severe-profound; "
            "CORRECT CALCIUM BEFORE AUDIOMETRIC ASSESSMENT; "
            "RENAL: annual GFR + creatinine; renal USS; nephrology co-management; "
            "avoid nephrotoxic drugs; calcitriol dose adjusted for renal function; "
            "GENETIC COUNSELLING: AD 50%; GATA3 deletion/sequence analysis."
        ),
        "key_features": [
            "HDR triad: Hypoparathyroidism + Deafness (SNHL bilateral) + Renal dysplasia",
            "Calcium correction MANDATORY BEFORE audiometry — hypocalcaemia worsens hearing",
            "Chvostek's + Trousseau's sign; tetanic seizures; laryngospasm — emergency",
            "Lab: LOW PTH + LOW calcium + HIGH phosphate — screen ALL SNHL children",
            "ECG: prolonged QTc from hypocalcaemia — cardiac monitoring required",
            "Lifelong oral calcium + calcitriol; target Ca 2.0-2.25 mmol/L (not over-corrected)",
            "Renal USS mandatory; bilateral anomalies → CKD risk; avoid nephrotoxics",
            "GATA3: parathyroid + inner ear + kidney + T-cell development master regulator",
        ],
        "key_ddx": (
            "DiGeorge/22q11 (TBX1): parathyroidism + cardiac + palate + SNHL; del22q11 array mandatory; "
            "KBG syndrome (ANKRD11): parathyroidism + macrodontia + short stature; "
            "Autoimmune hypoparathyroidism (AIRE/APS1): acquired; anti-CaSR antibodies; "
            "Pseudo-hypoparathyroidism (GNAS): HIGH PTH + LOW calcium; GATA3 has LOW/absent PTH; "
            "Usher syndrome: SNHL + RP; no hypoparathyroidism; "
            "Alport syndrome (COL4A3/4/5): SNHL + renal disease; normal calcium/PTH."
        ),
        "systemic_involvement": (
            "PARATHYROID: hypoparathyroidism → hypocalcaemia → tetany, seizures, laryngospasm, QTc prolongation. "
            "EAR: bilateral SNHL moderate-severe. "
            "KIDNEY: dysplasia/hypoplasia/agenesis/VUR → CKD risk. "
            "T-CELLS: GATA3 role in CD4+ T cells — mild immune phenotype possible. "
            "NO coloboma; NO branchial fistulae; NO HSCR."
        ),
        "onset_age": "SNHL: congenital or early childhood; Hypoparathyroidism: infancy-childhood; Renal: antenatal USS",
        "surgical_urgency": "Acute hypocalcaemia emergency: IV calcium gluconate immediately; CI for severe-profound SNHL after calcium correction",
        "gene_family": "GATA zinc-finger transcription factor family; C-terminal zinc finger (ZnF2) binds GATA DNA motif",
        "morphology": (
            "AUDIOGRAM: bilateral SNHL moderate-severe (assess only after calcium corrected); "
            "LABS: low Ca, low PTH, high PO4; ECG: prolonged QTc; "
            "RENAL USS: dysplasia, small kidneys, VUR; "
            "CT temporal bone: cochlear/vestibular anomalies variable"
        ),
        "n_patients": 40,
    },
]


def _make_patients(gene_entry: dict) -> list[dict]:
    rng = random.Random(SEED_BASE + SHL_GENES.index(gene_entry))
    gene = gene_entry["gene"]
    patients = []
    for i in range(gene_entry["n_patients"]):
        age = rng.randint(1, 60)
        sex = rng.choice(["M", "F"])

        # Hearing severity by gene/syndrome
        if gene == "PAX3":
            # SNHL ~57%; rest normal hearing
            if rng.random() < 0.57:
                severity = rng.choice(["Moderate", "Severe", "Profound", "Moderate-severe"])
            else:
                severity = "Normal hearing"
        elif gene == "MITF":
            # SNHL ~80-90%
            if rng.random() < 0.85:
                severity = rng.choice(["Moderate", "Severe", "Profound", "Moderate-severe"])
            else:
                severity = "Normal hearing"
            # Tietz alleles (10% severe)
            if i < 4:
                severity = "Profound bilateral (Tietz)"
        elif gene == "SOX10":
            severity = rng.choice(["Severe", "Profound", "Severe-profound"])
        elif gene == "EDNRB":
            # biallelic = profound; het = variable
            if i < 30:
                severity = rng.choice(["Severe", "Profound", "Moderate-severe"])
            else:
                severity = rng.choice(["Mild", "Moderate", "Normal hearing"])
        elif gene == "EYA1":
            severity = rng.choice(["Mild-moderate CHL", "Moderate SNHL", "Moderate-severe mixed",
                                   "Severe SNHL", "Moderate CHL", "Mild SNHL"])
        elif gene == "CHD7":
            severity = rng.choice(["Severe", "Profound", "Severe-profound", "Profound bilateral"])
        elif gene == "TCOF1":
            severity = rng.choice(["Moderate CHL", "Severe CHL", "Profound CHL",
                                   "Moderate CHL bilateral", "Severe-profound CHL"])
        elif gene == "GATA3":
            severity = rng.choice(["Moderate", "Moderate-severe", "Severe", "Severe bilateral"])

        # Management
        if gene in ("SOX10", "CHD7"):
            mgmt = rng.choice(["Bilateral cochlear implant", "Cochlear implant unilateral",
                                "Under CI evaluation", "Bilateral CI + vestibular rehab"])
        elif gene == "TCOF1":
            mgmt = rng.choice(["BAHA headband", "BAHA implant", "Conventional hearing aids",
                                "Surgical atresia repair", "BAHA bilateral"])
        elif gene == "GATA3":
            mgmt = rng.choice(["Hearing aids", "Cochlear implant", "BAHA",
                                "Under evaluation", "Hearing aids bilateral"])
            if "Profound" in severity:
                mgmt = rng.choice(["Bilateral cochlear implant", "Cochlear implant unilateral"])
        elif gene == "PAX3":
            if "Normal" in severity:
                mgmt = "No hearing intervention"
            else:
                mgmt = rng.choice(["Hearing aids", "Cochlear implant", "Under evaluation"])
        elif gene == "MITF":
            if "Normal" in severity:
                mgmt = "No hearing intervention"
            elif "Tietz" in severity or "Profound" in severity:
                mgmt = rng.choice(["Bilateral cochlear implant", "Cochlear implant bilateral early"])
            else:
                mgmt = rng.choice(["Hearing aids", "Cochlear implant", "Under evaluation"])
        elif gene == "EDNRB":
            if "Normal" in severity:
                mgmt = "No hearing intervention"
            else:
                mgmt = rng.choice(["Cochlear implant bilateral", "Hearing aids", "Under CI evaluation"])
        elif gene == "EYA1":
            mgmt = rng.choice(["BAHA", "Conventional hearing aids", "Middle ear surgery",
                                "CI referral", "BAHA headband"])

        # Extra events
        evs = []
        if gene == "PAX3":
            if rng.random() < 0.55:
                evs.append(f"W-index: {rng.uniform(1.95, 2.40):.2f} (≥1.95 diagnostic)")
            if rng.random() < 0.45:
                evs.append("White forelock confirmed")
            if rng.random() < 0.40:
                evs.append("Iris heterochromia on slit lamp")
        elif gene == "MITF":
            if i < 4:
                evs.append("Tietz syndrome: complete albinism + bilateral profound SNHL")
            else:
                if rng.random() < 0.40:
                    evs.append("Iris heterochromia")
            evs.append("W-index normal (≤1.94) — DDx WS1/PAX3 negative")
        elif gene == "SOX10":
            if rng.random() < 0.70:
                evs.append("HSCR confirmed — rectal biopsy aganglionosis")
            if rng.random() < 0.55:
                evs.append("Peripheral neuropathy: NCVs <38 m/s (demyelinating)")
            if rng.random() < 0.20:
                evs.append("PCWH: central hypomyelination on MRI")
        elif gene == "EDNRB":
            if i < 30:
                evs.append("Biallelic EDNRB: WS4A — HSCR + WS features")
                if rng.random() < 0.80:
                    evs.append("HSCR pull-through surgery completed")
            else:
                evs.append("Heterozygous EDNRB: isolated HSCR — no WS features")
            if rng.random() < 0.15:
                evs.append("S305N Mennonite founder allele confirmed")
        elif gene == "EYA1":
            evs.append(rng.choice([
                "Bilateral branchial fistulae anterior SCM",
                "Preauricular tags bilateral",
                "Branchial cyst right neck",
                "Unilateral branchial fistula + preauricular pit"
            ]))
            if rng.random() < 0.67:
                evs.append(rng.choice([
                    "Renal dysplasia unilateral",
                    "Horseshoe kidney",
                    "Duplex right kidney",
                    "Renal hypoplasia bilateral"
                ]))
            evs.append("CT temporal bone: Mondini cochlea or ossicular anomaly")
        elif gene == "CHD7":
            if rng.random() < 0.80:
                evs.append("CT temporal bone: semicircular canal aplasia confirmed")
            if rng.random() < 0.60:
                evs.append(rng.choice([
                    "Iris coloboma bilateral",
                    "Chorioretinal coloboma unilateral",
                    "Optic nerve coloboma"
                ]))
            if rng.random() < 0.50:
                evs.append(rng.choice([
                    "Choanal atresia repaired neonatal",
                    "Bilateral choanal atresia — neonatal airway emergency managed"
                ]))
            if rng.random() < 0.40:
                evs.append("Cardiac defect: ToF/VSD/interrupted aortic arch — repaired")
            evs.append("Absent VOR — vestibular rehabilitation ongoing")
        elif gene == "TCOF1":
            evs.append("CT face 3D: absent zygomatic arch and malar eminence")
            if rng.random() < 0.55:
                evs.append("Lower eyelid coloboma (lateral 1/3)")
            evs.append(rng.choice([
                "Bilateral EAC atresia — BAHA fitted",
                "EAC stenosis bilateral — hearing aids + BAHA",
                "Ossicular malformation — ossiculoplasty attempted"
            ]))
            if rng.random() < 0.35:
                evs.append("Mandibular distraction osteogenesis completed")
        elif gene == "GATA3":
            evs.append(f"Serum calcium: {rng.uniform(1.55, 2.05):.2f} mmol/L (LOW)")
            evs.append("PTH: low/absent — hypoparathyroidism confirmed")
            evs.append("Oral calcium + calcitriol initiated")
            if rng.random() < 0.80:
                evs.append(rng.choice([
                    "Renal dysplasia on USS",
                    "VUR grade II-III bilateral",
                    "Unilateral renal hypoplasia"
                ]))
            if rng.random() < 0.30:
                evs.append("Prior tetanic seizure — IV calcium gluconate acute Rx")

        patients.append({
            "id": f"{gene}-{i+1:03d}",
            "gene": gene,
            "age": age,
            "sex": sex,
            "onset": gene_entry["onset_age"][:60],
            "hearing_severity": severity,
            "management": mgmt,
            "key_features": "; ".join(gene_entry["key_features"][:3]),
            "extra_events": "; ".join(evs) if evs else "—",
        })
    return patients


def _all_patients() -> list[dict]:
    out = []
    for g in SHL_GENES:
        out.extend(_make_patients(g))
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_overview() -> dict:
    """Aggregate summary for the /overview endpoint."""
    patients = _all_patients()
    gene_counts: dict[str, int] = {}
    severity_counts: dict[str, int] = {}
    mgmt_counts: dict[str, int] = {}

    for p in patients:
        gene_counts[p["gene"]] = gene_counts.get(p["gene"], 0) + 1
        severity_counts[p["hearing_severity"]] = severity_counts.get(p["hearing_severity"], 0) + 1
        mgmt_counts[p["management"]] = mgmt_counts.get(p["management"], 0) + 1

    gene_highlights = {
        g["gene"]: {
            "locus": g["locus"],
            "protein_size": g["protein_size"],
            "inheritance": g["inheritance"][:120] + "...",
            "disease_category": g["disease_category"][:120] + "...",
            "pathognomonic_short": g["key_features"][0],
        }
        for g in SHL_GENES
    }

    return {
        "atlas": "Hereditary-Syndromic-Hearing-Loss-Atlas",
        "subtitle": (
            "Complete 8-Gene Syndromic Hearing Loss Reference "
            "(PAX3 · MITF · SOX10 · EDNRB · EYA1 · CHD7 · TCOF1 · GATA3)"
        ),
        "total_patients": len(patients),
        "genes_covered": len(SHL_GENES),
        "seed_range": f"{SEED_BASE}–{SEED_BASE + len(SHL_GENES) - 1}",
        "patients_per_gene": gene_counts,
        "hearing_severity_distribution": severity_counts,
        "management_distribution": mgmt_counts,
        "gene_highlights": gene_highlights,
        "clinical_pearls": [
            "PAX3 (WS1): dystopia canthorum W-index ≥1.95 = PATHOGNOMONIC; SNHL ~57%; most common WS gene",
            "MITF (WS2A): NO dystopia canthorum (DDx WS1); most common WS2 gene; Tietz = severe alleles + complete albinism",
            "SOX10 (WS4C/PCWH): HSCR + demyelinating neuropathy + WS; PCWH = alleles escaping NMD (most severe)",
            "EDNRB (WS4A): AR biallelic = HSCR + full WS4A; heterozygous = isolated HSCR only (no WS features)",
            "EYA1 (BOR): branchial fistulae + Mondini cochlea + renal dysplasia; AVOID aminoglycosides (double jeopardy)",
            "CHD7 (CHARGE): semicircular canal aplasia on CT = near-pathognomonic; bilateral choanal atresia = neonatal airway EMERGENCY",
            "TCOF1 (Treacher Collins): absent malar/zygoma; predominantly CHL; BAHA FIRST-LINE for bilateral atresia",
            "GATA3 (HDR): hypoparathyroidism + SNHL + renal; CORRECT CALCIUM BEFORE audiometry; IV calcium for tetany EMERGENCY",
        ],
    }


def get_breakdown() -> dict:
    """Per-gene breakdown for the /breakdown endpoint."""
    breakdown = {}
    for g in SHL_GENES:
        patients = _make_patients(g)
        mgmt_breakdown: dict[str, int] = {}
        severity_breakdown: dict[str, int] = {}
        for p in patients:
            mgmt_breakdown[p["management"]] = mgmt_breakdown.get(p["management"], 0) + 1
            severity_breakdown[p["hearing_severity"]] = severity_breakdown.get(p["hearing_severity"], 0) + 1

        breakdown[g["gene"]] = {
            "gene": g["gene"],
            "alt_name": g["alt_name"],
            "protein": g["protein"],
            "locus": g["locus"],
            "protein_size": g["protein_size"],
            "inheritance": g["inheritance"],
            "disease_category": g["disease_category"],
            "disease_pathway": g["disease_pathway"],
            "pathognomonic": g["pathognomonic"],
            "treatment": g["treatment"],
            "key_features": g["key_features"],
            "key_ddx": g["key_ddx"],
            "systemic_involvement": g["systemic_involvement"],
            "onset_age": g["onset_age"],
            "surgical_urgency": g["surgical_urgency"],
            "gene_family": g["gene_family"],
            "morphology": g["morphology"],
            "n_patients": g["n_patients"],
            "hearing_severity_distribution": severity_breakdown,
            "management_distribution": mgmt_breakdown,
            "sample_patients": patients[:5],
        }
    return {"breakdown_by_gene": breakdown, "total_genes": len(SHL_GENES)}


def get_definitions() -> dict:
    """Clinical definitions and glossary for the /definitions endpoint."""
    return {
        "atlas": "Hereditary-Syndromic-Hearing-Loss-Atlas",
        "definitions": {
            "Waardenburg_Syndrome": (
                "Autosomal dominant syndromic SNHL with sensorineural hearing loss + iris pigmentary abnormalities + "
                "white forelock + skin depigmentation ± dystopia canthorum (WS1/3) ± Hirschsprung disease (WS4). "
                "GENES: PAX3 (WS1/3) · MITF (WS2A) · SOX10 (WS4C) · EDNRB (WS4A) · EDN3 (WS4B) · SNAI2 (WS2D)"
            ),
            "Dystopia_Canthorum": (
                "Lateral displacement of the medial canthi; measured by W-index: "
                "W = 2a/(b+c+a) where a=inner canthal distance, b/c=palpebral fissure widths; "
                "W ≥1.95 = diagnostic for WS1 (PAX3/WS3); W <1.95 in WS2 (MITF) — KEY DDx"
            ),
            "Tietz_Syndrome": (
                "Severe MITF alleles → complete uniform cutaneous albinism (skin + hair + eyelashes) + "
                "bilateral profound congenital SNHL; NO iris heterochromia (uniformly unpigmented iris); "
                "visual acuity NORMAL (DDx OCA albinism — Tietz: no nystagmus, no foveal hypoplasia)"
            ),
            "PCWH_Syndrome": (
                "Peripheral + Central hypomyelination + Waardenburg syndrome + Hirschsprung disease; "
                "caused by SOX10 alleles that escape nonsense-mediated decay (NMD) → dominant-negative; "
                "FEATURES: SNHL + depigmentation + HSCR + peripheral demyelinating neuropathy + central hypomyelination"
            ),
            "Hirschsprung_Disease": (
                "Congenital absence of enteric ganglia from the rectum extending proximally; "
                "RESULT: functional obstruction — delayed meconium (>48h), constipation, abdominal distension; "
                "DIAGNOSIS: suction rectal biopsy (absent ganglia + acetylcholinesterase staining); "
                "GENES in syndromic SNHL: SOX10 (WS4C), EDNRB (WS4A)"
            ),
            "BOR_Syndrome": (
                "Branchiootorenal syndrome (EYA1 AD): "
                "Branchial anomalies (fistulae/cysts/tags) + Otologic anomalies (Mondini cochlea, ossicular, EAC) + "
                "Renal dysplasia; SNHL + CHL + mixed; "
                "AVOID aminoglycosides (ototoxic + nephrotoxic double jeopardy)"
            ),
            "CHARGE_Syndrome": (
                "CHD7 de novo AD: Coloboma + Heart + choanal Atresia + Retardation + Genital + Ear; "
                "Semicircular canal aplasia/hypoplasia = near-pathognomonic on CT; "
                "bilateral choanal atresia = neonatal airway EMERGENCY; "
                "absent VOR → profound motor delay (walks 3-5 years); CI effective but vestibular rehab mandatory"
            ),
            "Treacher_Collins_Syndrome": (
                "TCOF1 AD (± POLR1C/POLR1D AR): mandibulofacial dysostosis Franceschetti-Klein; "
                "absent/hypoplastic zygoma + malar + zygomatic arch; predominantly CHL (EAC atresia + ossicular); "
                "BAHA FIRST-LINE for bilateral CHL; cochlea usually NORMAL; lower eyelid coloboma (not retinal)"
            ),
            "HDR_Syndrome": (
                "GATA3 AD: Hypoparathyroidism + Deafness (SNHL bilateral) + Renal dysplasia (Barakat syndrome); "
                "LOW PTH + LOW calcium + HIGH phosphate; screen ALL SNHL patients; "
                "CORRECT CALCIUM BEFORE audiometry (hypocalcaemia worsens hearing); "
                "IV calcium gluconate for tetany EMERGENCY"
            ),
            "W_Index": (
                "Measurement for dystopia canthorum: W = 2a / (a + b + c) where "
                "a = inner canthal distance, b = palpebral fissure right, c = palpebral fissure left; "
                "W ≥1.95 = dystopia canthorum = WS1 (PAX3); W ≤1.94 = normal = WS2 (MITF, SOX10, EDNRB)"
            ),
            "Semicircular_Canal_Aplasia": (
                "Absence or severe hypoplasia of semicircular canals on CT temporal bone; "
                "NEAR-PATHOGNOMONIC for CHARGE syndrome (CHD7); "
                "CONSEQUENCE: absent vestibulo-ocular reflex (VOR) + profound vestibular areflexia → "
                "motor delay; falls; late walking (3-5 years); CI patients need vestibular rehab"
            ),
            "BAHA_First_Line_CHL": (
                "Bone-anchored hearing aid (BAHA/Osia): first-line for bilateral CHL with EAC atresia; "
                "headband fitting from 6 months; implant (Baha Connect or Osia) at 4-5 years; "
                "bypasses EAC + middle ear; transmits vibration to cochlea directly; "
                "preferred over atresia surgery in Treacher Collins syndrome"
            ),
            "Mondini_Malformation": (
                "Incomplete partition of the cochlea: 1.5 turns instead of normal 2.5; "
                "associated with EYA1/BOR syndrome, CHD7/CHARGE; "
                "CI still feasible in Mondini — pre-operative CT essential for surgical planning; "
                "may have higher perilymph gusher risk intraoperatively"
            ),
            "NMD_Escape_PCWH": (
                "Nonsense-mediated decay (NMD) escape: certain SOX10 truncating alleles in exon 4-5 → "
                "mRNA not degraded → truncated dominant-negative SOX10 protein → "
                "blocks normal SOX10 function → PCWH syndrome (severe); "
                "alleles triggering NMD → haploinsufficiency → WS4C (less severe); "
                "predict severity by allele position + NMD prediction"
            ),
            "Calcium_Audiometry_Protocol": (
                "GATA3/HDR protocol: ALWAYS normalise serum calcium before audiological assessment; "
                "hypocalcaemia impairs hair cell function + auditory neuron firing → artefactually worse audiogram; "
                "after calcium normalised: repeat audiogram gives true SNHL severity; "
                "avoid over-correcting calcium (hypercalciuria → renal stones in HDR kidney)"
            ),
        },
        "emergency_protocols": [
            "BILATERAL CHOANAL ATRESIA (CHD7): oral airway / McGovern nipple IMMEDIATELY; neonatal choanoplasty",
            "HYPOCALCAEMIC TETANY (GATA3/HDR): IV calcium gluconate 10% SLOW IV; ECG monitoring (QTc)",
            "HSCR NEONATAL OBSTRUCTION (SOX10/EDNRB): colostomy bridge; surgical pull-through planning",
            "CHARGE AIRWAY: multidisciplinary — ENT + paediatric surgery + cardiology simultaneously",
        ],
        "genes": [g["gene"] for g in SHL_GENES],
        "seed_range": f"{SEED_BASE}–{SEED_BASE + len(SHL_GENES) - 1}",
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(get_overview(), indent=2)[:2000])
    print("\n=== BREAKDOWN (PAX3) ===")
    bk = get_breakdown()
    print(json.dumps(bk["breakdown_by_gene"]["PAX3"], indent=2)[:2000])
    print("\n=== DEFINITIONS (first 1000 chars) ===")
    df = get_definitions()
    print(json.dumps(df, indent=2)[:1000])
