#!/usr/bin/env python3
"""GPI-Anchor-Atlas — Complete 8-Gene GPI-Anchor Biosynthesis Disorders Atlas
PIGA   (MCAHS1 — Multiple Congenital Anomalies-Hypotonia-Seizures 1; X-linked; most common GPI disorder) ·
PIGV   (HPMRS1 — Hyperphosphatasia with Mental Retardation 1; AR; high ALP biomarker) ·
PIGL   (CHIME syndrome — Coloboma, Heart, Ichthyosiform, Mental retardation, Ear; AR) ·
PGAP2  (HPMRS3 — Hyperphosphatasia with Mental Retardation 3; AR; ALP elevated) ·
PGAP3  (HPMRS4 — Hyperphosphatasia with Mental Retardation 4; AR; ALP very elevated) ·
PIGN   (MCAHS3-like — Multiple Congenital Anomalies-Hypotonia-Seizures 3; AR) ·
PIGT   (MCAHS3 — Multiple Congenital Anomalies-Hypotonia-Seizures 3; AR) ·
PIGG   (Intellectual Disability + Epilepsy; AR; atypical GPI phenotype)
320-patient aggregate cohort (8 × 40, seeds 974–981)

GPI-Anchor Biosynthesis Disorders — Key Principles:
  - GPI (glycosylphosphatidylinositol) anchors attach >150 surface proteins to the outer
    leaflet of the plasma membrane via a glycolipid moiety synthesised in the ER.
    GPI-anchored proteins include: tissue-nonspecific alkaline phosphatase (TNAP/ALPL),
    urokinase receptor (uPAR), complement regulators (CD55/DAF, CD59/protectin),
    prion protein (PrP), folate receptor (FOLR1), and Thy-1. Defects in GPI anchor
    synthesis → multiple GPI-anchored proteins absent or reduced from cell surfaces.
  - HIGH ALKALINE PHOSPHATASE (ALP/TNAP) IS A BIOMARKER:
    TNAP is GPI-anchored; GPI anchor defects → TNAP lost from cell surface → secreted
    as free enzyme → serum ALP markedly elevated (often 3-10× ULN). This elevated ALP
    (paradoxically, same direction as hepatic/bone disease) is a diagnostic clue.
    HIGH ALP in a child with epilepsy + ID = test GPI anchor pathway genes.
    HPMRS group (PIGV, PGAP2, PGAP3, PIGN, PIGT) all show markedly elevated ALP.
    PIGA and PIGG may have normal or only mildly elevated ALP.
  - SEIZURES ARE CARDINAL:
    Epilepsy is present in >90% of all GPI anchor disorder patients. Types range from
    infantile spasms (West syndrome, hypsarrhythmia) to focal, multifocal, tonic, and
    myoclonic. Most are drug-resistant. Ketogenic diet has been tried with variable success.
    Vigabatrin is a reasonable first-line for infantile spasms in this group.
  - FLOW CYTOMETRIC DIAGNOSIS (FLAER TEST):
    Granulocytes (not lymphocytes) are used. FLAER (Fluorescent Aerolysin) binds directly
    to GPI anchors; reduced FLAER staining = GPI anchor deficit. Simultaneously measure
    CD16 (FcγRIII, GPI-anchored) and CD24 (GPI-anchored) on granulocytes.
    CD59 and CD55 on red blood cells (complement regulators — reduced → PNH-like features).
    CRITICAL: lymphocytes CANNOT be used (they partially shed GPI-anchored proteins normally).
  - X-LINKED PIGA: UNIQUE GENETICS:
    PIGA is X-linked — males severely affected (hemizygous loss-of-function = lethal).
    PIGA mutations in affected males are hypomorphic (partial loss of function) or mosaic.
    Female carriers usually asymptomatic; mosaic females can have mild features.
    Somatic PIGA mutations cause PNH (paroxysmal nocturnal haemoglobinuria) in adults —
    GERMLINE PIGA mutations cause congenital MCAHS1 (completely different disease).
  - TREATMENT REALITY:
    No specific approved therapy for GPI anchor biosynthesis disorders as of 2025.
    SYMPTOMATIC: antiseizure medications (vigabatrin, valproate, ACTH for infantile spasms);
    ketogenic diet (evidence limited); supportive care. EMERGING: statins (modulate ER stress),
    butyrate (increase GPI anchor expression in PIGA mosaics — investigational).
    Bone disease from elevated ALP does NOT require treatment (enzyme is non-functional
    ALP released from cell surface — bone mineralisation typically normal).

COHORT: 8 × 40 = 320 patient slots (seeds 974–981; gene-specific seeds)
"""

import random

SEED_BASE = 974

GPI_GENES = [
    # ── PIGA — Multiple Congenital Anomalies-Hypotonia-Seizures Syndrome 1 (X-linked) ─
    {
        "gene": "PIGA", "protein": "Phosphatidylinositol Glycan Class A",
        "alias": "PIGA — MCAHS1 (OMIM #300868); X-linked GPI anchor synthesis; most common congenital GPI disorder",
        "aa": "484 aa", "kDa": "56 kDa",
        "gene_class": (
            "Phosphatidylinositol glycan anchor biosynthesis class A (PIGA); catalyses the FIRST "
            "committed step of GPI anchor synthesis in the ER: transfer of N-acetylglucosamine (GlcNAc) "
            "from UDP-GlcNAc to phosphatidylinositol (PI) → GlcNAc-PI; PIGA forms a complex with "
            "PIGC, PIGH, PIGY, PIGQ, PIGP in the GlcNAc-PI transferase complex; X-linked (Xp22.2); "
            "complete loss-of-function is lethal; pathogenic alleles are hypomorphic or mosaic; "
            "somatic PIGA mutations cause PNH (paroxysmal nocturnal haemoglobinuria) in adults — "
            "germline hypomorphic mutations cause congenital GPI deficiency syndrome"
        ),
        "gpi_subgroup": "GPI Synthesis — Step 1 (GlcNAc transfer to PI) — X-linked",
        "pathway_step": "Step 1: GlcNAc-PI transfer (first committed step)",
        "locus": "Xp22.2", "omim_gene": 311770, "omim_disease": 300868,
        "inheritance": "X-linked (XL). Xp22.2. Males severely affected (hemizygous hypomorphic). Female carriers usually asymptomatic. Mosaic females possible.",
        "seed_offset": 0,
        "onset_range_y": (0.0, 1.0),
        "phenotype": (
            "MCAHS1 (Multiple Congenital Anomalies-Hypotonia-Seizures Syndrome 1): "
            "neonatal/infantile onset; profound global developmental delay; severe hypotonia; "
            "epileptic encephalopathy (infantile spasms/West syndrome, myoclonic, tonic); "
            "facial dysmorphism (hypertelorism, large fontanelle, broad/depressed nasal bridge, "
            "low-set ears, cleft palate); brain malformations (simplified gyral pattern, "
            "corpus callosum hypoplasia, cortical dysplasia, cerebellar hypoplasia); "
            "congenital heart defects (VSD, ASD); genital abnormalities (hypospadias in males); "
            "anorectal anomalies; variable hearing loss; retinal dystrophy. "
            "ALP: mildly elevated or normal (less striking than HPMRS group). "
            "Most severe form: neonatal death or survival with minimal function. "
            "Less severe (mosaic/hypomorphic): intellectual disability + epilepsy without visceral anomalies."
        ),
        "disease": (
            "PIGA (484 aa, 56 kDa) encodes the catalytic subunit of GlcNAc-PI transferase, executing "
            "the first committed step of GPI anchor biosynthesis in the ER: UDP-GlcNAc + PI → GlcNAc-PI. "
            "PIGA is in complex with PIGC (3-TM structural), PIGH (luminal), PIGY (stability), PIGQ, PIGP. "
            "X-LINKED UNIQUE BIOLOGY: complete loss-of-function is embryonic lethal; pathogenic germline "
            "alleles must retain partial function (hypomorphic). Males carry single allele → severe disease. "
            "Somatic PIGA mutations (acquired in HSC) → PNH: clonal haematopoiesis, haemolytic anaemia, "
            "thrombosis — COMPLETELY DIFFERENT from germline PIGA disorder.\n\n"
            "GPI ANCHOR PATHWAY CONTEXT: GPI anchor synthesis has >20 steps (PIGA through GPAA1, PIGU); "
            "PIGA catalyses step 1. All downstream steps require step 1 product. "
            "GPI-anchored proteins critical for neurodevelopment: folate receptor 1 (FOLR1, essential for "
            "brain folate import), prion protein (PrP, brain function unknown), Thy-1 (axonal signalling), "
            "CD73 (adenosine signalling), contactin-1 (myelination).\n\n"
            "FOLINIC ACID TRIAL: FOLR1 is GPI-anchored; reduced surface FOLR1 → brain folate deficiency "
            "even with normal systemic folate → CSF 5-MTHF low in some PIGA cases → empiric folinic acid "
            "(leucovorin) 2-5mg/kg/day is reasonable (no controlled trial data; low risk).\n\n"
            "DIAGNOSIS: flow cytometry (FLAER + CD16 on granulocytes); panel sequencing; WES/WGS "
            "for X-linked de novo mutations.\n\n"
            "MANAGEMENT: antiseizure medications (ACTH/vigabatrin for infantile spasms); "
            "ketogenic diet (anecdotal benefit); folinic acid empiric trial; supportive care. "
            "Prognosis: generally severe for classic MCAHS1; some attenuated cases survive to adulthood."
        ),
        "hallmark": (
            "PIGA/MCAHS1 HALLMARKS: "
            "(1) X-LINKED — all pathogenic germline PIGA alleles are HYPOMORPHIC (partial function); "
            "   complete loss-of-function is lethal (no patients survive complete PIGA null). "
            "(2) GERMLINE PIGA (MCAHS1) ≠ SOMATIC PIGA (PNH): completely different diseases, different "
            "   ages, mechanisms, and treatments — do not confuse. "
            "(3) BRAIN MALFORMATIONS: simplified gyral pattern, CC hypoplasia — structural abnormality on MRI. "
            "(4) INFANTILE SPASMS commonest seizure type — vigabatrin/ACTH first-line. "
            "(5) ALP: mildly elevated or normal — less striking than HPMRS group (PIGV, PGAP2, PGAP3). "
            "(6) FOLINIC ACID TRIAL: reasonable empiric therapy (FOLR1 GPI-anchored → brain folate deficit). "
            "(7) FLOW CYTOMETRY: FLAER + CD16 on granulocytes (NOT lymphocytes — key technical point). "
            "(8) MOSAIC MALES: some males with somatic PIGA mosaicism have attenuated phenotype."
        ),
        "nbs_marker": "Not on standard NBS; flow cytometric GPI testing (FLAER/CD16/CD24 on granulocytes) for targeted screening",
        "key_biomarker": "Flow cytometry: reduced FLAER, CD16, CD24 on granulocytes; ALP mildly elevated or normal; CSF 5-MTHF low in some",
        "severity_spectrum": "Neonatal lethal (most severe) → classic MCAHS1 (severe) → attenuated ID+epilepsy (mosaic/hypomorphic)",
        "treatments": ["Antiseizure medications (vigabatrin, ACTH for infantile spasms)", "Folinic acid empiric trial 2-5mg/kg/day", "Ketogenic diet (anecdotal)", "Supportive multidisciplinary care"],
        "emergency": "ACTH or vigabatrin for infantile spasms; folinic acid trial for brain folate deficiency",
        "ci_drugs": ["No specific contraindicated drugs in GPI disorders; standard AED precautions apply"],
    },
    # ── PIGV — Hyperphosphatasia with Mental Retardation Syndrome 1 (HPMRS1) ──────────
    {
        "gene": "PIGV", "protein": "Phosphatidylinositol Glycan Class V",
        "alias": "PIGV — HPMRS1 (Mabry syndrome, OMIM #239300); AR; markedly elevated ALP biomarker; facial gestalt",
        "aa": "422 aa", "kDa": "48 kDa",
        "gene_class": (
            "Phosphatidylinositol glycan anchor biosynthesis class V (PIGV); mannosyltransferase enzyme "
            "in the ER; catalyses addition of the SECOND mannose to the GPI anchor intermediate: "
            "Man-GPI → Man2-GPI; PIGV is the step-6 enzyme in GPI synthesis; AR inheritance; "
            "10 transmembrane domains; DXD motif for Mn2+-dependent catalysis; "
            "deficiency → truncated GPI anchor cannot be transferred to proteins → GPI-anchored "
            "proteins (including TNAP/ALP) shed into serum → markedly elevated serum ALP"
        ),
        "gpi_subgroup": "GPI Synthesis — Step 6 (Second mannose addition) — AR/HPMRS",
        "pathway_step": "Step 6: second alpha-1,6-mannose addition (Man2-GPI)",
        "locus": "1p36.11", "omim_gene": 610274, "omim_disease": 239300,
        "inheritance": "AR. 1p36.11. Both sexes. Incidence rare; founder mutations in Middle Eastern populations.",
        "seed_offset": 1,
        "onset_range_y": (0.0, 3.0),
        "phenotype": (
            "HPMRS1 (Mabry syndrome): intellectual disability (mild to severe) + seizures + "
            "markedly elevated serum ALP (3-10× ULN — the diagnostic clue) + facial gestalt. "
            "Facial features: hypertelorism, brachycephaly, tented upper lip, large central incisors, "
            "wide nasal bridge, short palpebral fissures, brachydactyly. "
            "Seizures: generalised tonic-clonic, myoclonic, infantile spasms; drug-resistant in many. "
            "Hypotonia: neonatal hypotonia common; later spasticity may develop. "
            "No liver disease (ALP not hepatic — TNAP is bone/intestinal/placental ALP, all GPI-anchored). "
            "Brain MRI: often normal or nonspecific white-matter changes; no structural malformations. "
            "Life expectancy: variable; many survive to adulthood with significant disability."
        ),
        "disease": (
            "PIGV (422 aa, 48 kDa) catalyses the addition of the second mannose to the growing "
            "GPI anchor chain in the ER: Man1-GPI → Man2-GPI (alpha-1,6-linkage). "
            "PIGV deficiency → GPI anchor synthesis stalls at the Man1-GPI stage → "
            "truncated anchor cannot be transferred to GPI-dependent proteins → "
            "GPI-anchored proteins (TNAP, CD55, CD59, CD16, FOLR1, etc.) cannot be anchored → "
            "proteins secreted as soluble forms → TNAP accumulates in serum → marked ALP elevation.\n\n"
            "PATHOGNOMONIC BIOMARKER: serum ALP >3-10× upper limit of normal in a child with "
            "ID + seizures WITHOUT hepatic or bone disease. This pattern (hyperphosphatasia + "
            "MR + seizures) was described as 'Mabry syndrome' in 1970 before the genetic basis was known. "
            "2010: Mabry syndrome mapped to PIGV. HPMRS umbrella now includes PIGV (HPMRS1), "
            "PIGN (HPMRS2), PGAP2 (HPMRS3), PGAP3 (HPMRS4), PIGG (HPMRS6).\n\n"
            "FOUNDER MUTATIONS: p.Ala395Val — the most common pathogenic variant; found in Middle "
            "Eastern families (Saudi Arabia, Jordan, Kuwait). p.His332Leu — second common. "
            "Both result in complete loss of enzyme function.\n\n"
            "BIOCHEMISTRY: urine/serum: GPI-depleted microparticles detectable; TNAP activity in serum "
            "paradoxically high (enzyme released from membrane). Leukocyte flow cytometry: reduced FLAER, "
            "CD16, CD24. Confirmatory: PIGV gene sequencing.\n\n"
            "MANAGEMENT: antiseizure medications; regular developmental monitoring; "
            "physiotherapy and occupational therapy. No specific treatment. "
            "Bone: despite high ALP, bone mineralisation is typically normal — do not treat "
            "with ALP-lowering agents (enzyme is inactive on the cell surface; elevated serum ALP "
            "is not causing bone pathology in GPI disorders)."
        ),
        "hallmark": (
            "PIGV/HPMRS1 HALLMARKS: "
            "(1) MARKEDLY ELEVATED SERUM ALP (3-10× ULN) IN CHILD WITH ID + SEIZURES — "
            "   this triad = HPMRS until proven otherwise; ALP is a GPI-anchored enzyme (TNAP). "
            "(2) NO LIVER DISEASE: elevated ALP is NOT hepatic — isoenzyme is bone/placental TNAP. "
            "(3) FACIAL GESTALT: hypertelorism, tented upper lip, large central incisors — 'Mabry face'. "
            "(4) MABRY SYNDROME EPONYM: HPMRS1 = original Mabry syndrome (described 1970, PIGV identified 2010). "
            "(5) MIDDLE EAST FOUNDER: p.Ala395Val — common in Saudi/Jordanian/Kuwaiti families. "
            "(6) ALP MONITORING is a surrogate for GPI anchor pathway activity — tracks disease. "
            "(7) BRAIN MRI OFTEN NORMAL (unlike PIGA which has structural malformations). "
            "(8) FLOW CYTOMETRY: reduced FLAER + CD16 + CD24 on granulocytes confirms GPI deficit."
        ),
        "nbs_marker": "Not on NBS; serum ALP is the diagnostic clue (hyperphosphatasia) in HPMRS group",
        "key_biomarker": "Serum ALP markedly elevated (3-10× ULN); flow cytometry FLAER/CD16/CD24 reduced; PIGV sequencing",
        "severity_spectrum": "Mild ID (some walk, talk) to severe (non-ambulant, non-verbal); seizures drug-resistant in ~50%",
        "treatments": ["Antiseizure medications", "Developmental support (PT/OT/SLT)", "Folinic acid empiric trial", "Regular ALP monitoring"],
        "emergency": "No GPI-specific emergency; antiseizure escalation for refractory seizures",
        "ci_drugs": ["Vigabatrin (visual field loss risk — use with caution; monitor ERG)"],
    },
    # ── PIGL — CHIME Syndrome ─────────────────────────────────────────────────────────
    {
        "gene": "PIGL", "protein": "Phosphatidylinositol Glycan Class L",
        "alias": "PIGL — CHIME syndrome (OMIM #280000); AR; coloboma-heart-ichthyosis-MR-ear; unique multi-system GPI disorder",
        "aa": "252 aa", "kDa": "29 kDa",
        "gene_class": (
            "Phosphatidylinositol glycan anchor biosynthesis class L (PIGL); "
            "GlcNAc-PI deacetylase; catalyses step 2 of GPI synthesis: "
            "GlcNAc-PI → GlcN-PI (deacetylation of the N-acetylglucosamine); "
            "252 amino acids; ER-resident enzyme; AR inheritance; "
            "PIGL deficiency → GPI synthesis blocked at step 2 → downstream steps fail → "
            "reduced surface GPI-anchored proteins; unique multi-system phenotype "
            "(CHIME syndrome) not shared by other GPI genes"
        ),
        "gpi_subgroup": "GPI Synthesis — Step 2 (GlcNAc-PI deacetylation) — AR/CHIME",
        "pathway_step": "Step 2: GlcNAc-PI → GlcN-PI (deacetylation)",
        "locus": "17p11.2", "omim_gene": 605947, "omim_disease": 280000,
        "inheritance": "AR. 17p11.2. Both sexes. Extremely rare; <50 cases worldwide.",
        "seed_offset": 2,
        "onset_range_y": (0.0, 5.0),
        "phenotype": (
            "CHIME syndrome (Coloboma, Heart defects, Ichthyosiform dermatosis, Mental retardation, Ear anomalies): "
            "Coloboma: iris/chorioretinal; vision impairment in some. "
            "Heart: structural CHD (ASD, VSD, tetralogy of Fallot). "
            "Ichthyosiform dermatosis: distinctive recurrent/migratory ichthyotic plaques that "
            "resolve and recur — a unique cutaneous feature not seen in other GPI disorders. "
            "Mental retardation: moderate to severe ID; speech delay prominent. "
            "Ear: aural atresia, microtia, mixed conductive and sensorineural hearing loss. "
            "Additional: hypotonic facies, short stature, seizures (not universal but present ~40%), "
            "brachydactyly, low-set ears. "
            "ALP: mildly elevated in some cases; not a consistent finding (unlike HPMRS group). "
            "Distinctive: the migratory/recurrent ichthyotic skin lesions are pathognomonic for CHIME."
        ),
        "disease": (
            "PIGL (252 aa, 29 kDa) catalyses GPI anchor synthesis step 2: deacetylation of "
            "GlcNAc-PI to GlcN-PI, mediated by a zinc metalloenzyme mechanism. Loss of PIGL → "
            "GlcNAc-PI accumulates → GPI synthesis stalls → reduced surface GPI-anchored proteins.\n\n"
            "CHIME SYNDROME: Originally described by Zunich and Kaye in 1983. PIGL identified "
            "as the causative gene in 2012 (Ng et al.). The CHIME acronym captures the key features: "
            "Coloboma (eye structural defect), Heart defects, Ichthyosiform dermatosis "
            "(migratory — pathognomonic), Mental retardation, and Ear anomalies.\n\n"
            "THE MIGRATORY ICHTHYOSIS DISTINGUISHES CHIME: ichthyotic skin plaques that appear, "
            "migrate, and resolve over weeks; triggered by viral illnesses; no other GPI disorder "
            "has this pattern of recurrent migratory skin lesions. Skin biopsy: hyperkeratosis with "
            "parakeratosis, acanthosis.\n\n"
            "GENETICS: bi-allelic PIGL loss-of-function mutations; founders: p.Arg 90*(nonsense) "
            "described in original families; p.Ala58Thr and splice-site variants in European cohorts. "
            "WES/WGS diagnostic approach given rarity.\n\n"
            "MANAGEMENT: CHD surgery if indicated; hearing aids (cochlear implants for severe SNHL); "
            "ophthalmology monitoring; skin emollients during flares; antiseizure medications if seizures; "
            "developmental support. Skin: topical retinoids have been tried during flares with variable results."
        ),
        "hallmark": (
            "PIGL/CHIME HALLMARKS: "
            "(1) CHIME ACRONYM: Coloboma + Heart + Ichthyosiform + Mental retardation + Ear — MEMORISE. "
            "(2) MIGRATORY ICHTHYOSIS IS PATHOGNOMONIC — recurrent migratory ichthyotic plaques "
            "   that appear, migrate, and resolve (triggered by viral illness); unique to CHIME. "
            "(3) ALP: only mildly elevated (or normal) — CHIME is NOT in the HPMRS high-ALP group. "
            "(4) SEIZURES NOT UNIVERSAL (~40% of CHIME patients) — less prominent than PIGA/HPMRS. "
            "(5) EXTREMELY RARE: <50 cases worldwide; consider PIGL sequencing in CHIME phenotype. "
            "(6) STEP 2 BLOCK: GlcNAc-PI deacetylase — early GPI synthesis defect. "
            "(7) AURAL ATRESIA: structural ear canal absence → conductive hearing loss from birth. "
            "(8) SKIN IS THE DIAGNOSTIC CLUE: in any child with ID + cardiac + migratory skin = CHIME."
        ),
        "nbs_marker": "Not on NBS; clinical recognition by CHIME acronym + skin + cardiac workup; PIGL sequencing",
        "key_biomarker": "Migratory ichthyotic skin lesions (clinical); flow cytometry GPI reduction; PIGL sequencing; ALP mildly elevated",
        "severity_spectrum": "Moderate to severe ID; variable cardiac severity; skin flares lifelong; hearing loss progressive",
        "treatments": ["CHD surgical correction if needed", "Hearing aids / cochlear implants", "Emollients for skin flares", "Ophthalmology monitoring", "Antiseizure medications if seizures"],
        "emergency": "Cardiac emergency for structural CHD; skin flares — supportive; no GPI-specific emergency",
        "ci_drugs": ["No specific contraindicated drugs"],
    },
    # ── PGAP2 — Hyperphosphatasia with Mental Retardation Syndrome 3 (HPMRS3) ─────────
    {
        "gene": "PGAP2", "protein": "Post-GPI Attachment to Proteins Factor 2",
        "alias": "PGAP2 — HPMRS3 (OMIM #614207); AR; post-GPI processing step; high ALP + ID + seizures",
        "aa": "343 aa", "kDa": "39 kDa",
        "gene_class": (
            "Post-GPI attachment to proteins factor 2 (PGAP2); ER-resident enzyme; "
            "involved in lipid remodelling of the GPI anchor AFTER attachment to proteins: "
            "removes the sn-2 fatty acid from the inositol ring (deacylation step) — "
            "required for correct GPI anchor fatty acid composition (palmitate remodelling); "
            "AR inheritance; deficiency → GPI-anchored proteins have abnormal fatty acid "
            "composition → unstable surface attachment → proteins shed into serum → "
            "TNAP elevated in serum → hyperphosphatasia; 343 amino acids; 3q25.1"
        ),
        "gpi_subgroup": "Post-GPI Processing — Inositol Deacylation (after protein attachment) — AR/HPMRS",
        "pathway_step": "Post-GPI attachment: inositol deacylation (lipid remodelling after protein attachment)",
        "locus": "11p15.4", "omim_gene": 615187, "omim_disease": 614207,
        "inheritance": "AR. 11p15.4. Both sexes. Rare; founder mutations in Japan, Middle East, Europe.",
        "seed_offset": 3,
        "onset_range_y": (0.0, 3.0),
        "phenotype": (
            "HPMRS3: intellectual disability (mild to moderate more common than HPMRS1/HPMRS4) + "
            "markedly elevated serum ALP + seizures + facial dysmorphism. "
            "Distinctive: PGAP2-associated HPMRS has a MILDER cognitive phenotype on average "
            "compared to PIGV (HPMRS1) — some PGAP2 patients have only borderline ID. "
            "Facial: similar to HPMRS1 but often less striking; hypertelorism, broad nasal bridge. "
            "Seizures: generalised tonic-clonic, febrile-triggered in some; less refractory than PIGA. "
            "Behavior: autistic features, hyperactivity, impulsivity in some. "
            "ALP: consistently and markedly elevated (often most elevated of all HPMRS subtypes). "
            "Additional: brachycephaly, pes planus, relatively mild motor delay. "
            "Brain MRI: usually normal or nonspecific; no consistent structural abnormality."
        ),
        "disease": (
            "PGAP2 (343 aa, 39 kDa) performs a POST-attachment lipid remodelling step: "
            "after the GPI anchor is attached to a protein in the ER, the sn-2 fatty acid "
            "on the inositol phosphate must be removed (deacylated) to allow correct palmitate "
            "remodelling. PGAP2 deficiency → the sn-2 acyl chain is retained → abnormal GPI anchor "
            "lipid composition → unstable membrane anchor → GPI-anchored proteins shed prematurely → "
            "TNAP (ALPL) secreted → serum ALP markedly elevated.\n\n"
            "PGAP2 IS NOT A SYNTHESIS STEP: PGAP2 works AFTER the anchor is fully assembled and "
            "AFTER the anchor has been attached to a protein — this is lipid remodelling in the ER. "
            "This distinguishes PGAP2 from PIGA (step 1), PIGL (step 2), PIGV (step 6).\n\n"
            "FOUNDER MUTATIONS: p.Arg (R) 200* — Turkish/Middle East; p.Tyr(Y)111* — Japanese; "
            "p.Trp(W)186* in European families. Mutations cluster around the enzymatic domain.\n\n"
            "HPMRS3 vs HPMRS1: ALP often HIGHER in HPMRS3; cognitive phenotype MILDER in HPMRS3; "
            "facial features less pronounced in HPMRS3 — can overlap; molecular diagnosis essential.\n\n"
            "MANAGEMENT: antiseizure medications; developmental support; ALP monitoring. "
            "GPI-anchor flow cytometry (FLAER/CD16/CD24) reduced — confirms pathway defect."
        ),
        "hallmark": (
            "PGAP2/HPMRS3 HALLMARKS: "
            "(1) HIGH ALP — often HIGHEST of all HPMRS subtypes; consistently 3-15× ULN. "
            "(2) MILDER COGNITIVE PHENOTYPE than HPMRS1 (PIGV) — some PGAP2 have borderline ID. "
            "(3) POST-ATTACHMENT STEP: PGAP2 acts AFTER GPI is attached to protein — lipid remodelling; "
            "   NOT a GPI synthesis step (unlike PIGA/PIGL/PIGV). "
            "(4) ALP IS BOTH DIAGNOSTIC AND MONITORING TOOL — follow ALP serially; correlates with GPI burden. "
            "(5) AUTISTIC FEATURES + HYPERACTIVITY are more prominent in PGAP2 than other HPMRS subtypes. "
            "(6) FOUNDER MUTATIONS: p.Arg200* (Turkish), p.Tyr111* (Japanese) — ethnicity guides testing order. "
            "(7) BRAIN MRI NORMAL: no structural malformations (unlike PIGA). "
            "(8) FLAER FLOW CYTOMETRY confirms GPI deficit; PGAP2 sequencing is confirmatory."
        ),
        "nbs_marker": "Not on NBS; serum ALP the diagnostic clue in HPMRS group; PGAP2 sequencing confirmatory",
        "key_biomarker": "Serum ALP markedly elevated (often highest in HPMRS group); flow FLAER/CD16 reduced; PGAP2 sequencing",
        "severity_spectrum": "Borderline to moderate ID (milder than HPMRS1); seizures moderately drug-resistant; ALP persistently elevated",
        "treatments": ["Antiseizure medications", "Folinic acid empiric trial", "Developmental support", "ALP monitoring"],
        "emergency": "No GPI-specific emergency; antiseizure escalation for status epilepticus",
        "ci_drugs": ["No specific contraindicated drugs"],
    },
    # ── PGAP3 — Hyperphosphatasia with Mental Retardation Syndrome 4 (HPMRS4) ─────────
    {
        "gene": "PGAP3", "protein": "Post-GPI Attachment to Proteins Factor 3",
        "alias": "PGAP3 — HPMRS4 (OMIM #615716); AR; GPI fatty acid remodelling; high ALP + severe ID + seizures",
        "aa": "375 aa", "kDa": "43 kDa",
        "gene_class": (
            "Post-GPI attachment to proteins factor 3 (PGAP3); ER-resident phospholipase A2-like enzyme; "
            "catalyses fatty acid remodelling of GPI anchors AFTER protein attachment in the ER: "
            "removes the sn-2 unsaturated fatty acid (oleate) from the glycerophosphoinositol → "
            "replaced by stearic acid (C18:0) by MPPE1 → results in GPI anchor with two saturated "
            "fatty acids (sn-1 and sn-2 stearate) → required for correct lipid raft partitioning; "
            "PGAP3 deficiency → retained unsaturated sn-2 fatty acid → unstable GPI anchor → shedding"
        ),
        "gpi_subgroup": "Post-GPI Processing — Fatty Acid Remodelling (oleate removal) — AR/HPMRS",
        "pathway_step": "Post-GPI attachment: oleate removal by phospholipase A2-like activity (Golgi remodelling)",
        "locus": "17q12", "omim_gene": 611801, "omim_disease": 615716,
        "inheritance": "AR. 17q12. Both sexes. Rare; consanguineous families; North African/Middle East/European.",
        "seed_offset": 4,
        "onset_range_y": (0.0, 2.0),
        "phenotype": (
            "HPMRS4: intellectual disability (moderate to severe) + markedly elevated ALP + "
            "refractory epilepsy + distinctive facial dysmorphism. "
            "MOST SEVERE HPMRS SUBTYPE: PGAP3 has the most refractory epilepsy and severest ID "
            "of the HPMRS group. Infantile spasms common; EEG: hypsarrhythmia, multifocal spikes. "
            "Facial: deeply set eyes, low-set and posteriorly rotated ears, thin upper lip, wide nasal bridge. "
            "Hypotonia: profound neonatal hypotonia → later spasticity. "
            "Brain MRI: variable; some have cerebellar hypoplasia or simplified gyral pattern. "
            "ALP: consistently very elevated (3-20× ULN). "
            "Additional: feeding difficulties requiring NG/gastrostomy; respiratory complications; "
            "scoliosis; visual impairment. Survival: some to adulthood with severe disability."
        ),
        "disease": (
            "PGAP3 (375 aa, 43 kDa) performs the second lipid-remodelling step on GPI anchors, "
            "acting in the Golgi: removes the sn-2 unsaturated fatty acid (oleic acid, C18:1) from the "
            "diacylglycerophosphoinositol of the GPI anchor. This unsaturated FA is then replaced by "
            "stearic acid (C18:0) by MPPE1, resulting in a disaturated GPI anchor. "
            "Disaturated GPI anchors are required for correct partitioning into lipid rafts "
            "(cholesterol-rich microdomains where GPI-anchored signalling proteins concentrate).\n\n"
            "PGAP3 DEFICIENCY → retained unsaturated sn-2 oleate → GPI anchor excludes from lipid rafts → "
            "GPI-anchored proteins (TNAP, CD55, CD59, FOLR1) cannot concentrate in rafts → unstable surface "
            "attachment → shedding → hyperphosphatasia + loss of signalling function at cell surface.\n\n"
            "PGAP2 vs PGAP3: both are post-attachment lipid remodelling; PGAP2 removes the inositol "
            "acyl group (step in ER before Golgi); PGAP3 removes sn-2 unsaturated FA (step in Golgi "
            "after PGAP2). PGAP3 deficiency → MOST SEVERE HPMRS phenotype of the PGAP group.\n\n"
            "GENETICS: missense and frameshift variants; no single dominant founder mutation but "
            "certain consanguineous backgrounds (Moroccan, Saudi, Greek). "
            "p.Tyr162* and p.Leu256Pro reported recurrently.\n\n"
            "MANAGEMENT: vigabatrin/ACTH for infantile spasms; ketogenic diet; gastrostomy for feeding; "
            "respiratory monitoring; supportive care. Prognosis guarded."
        ),
        "hallmark": (
            "PGAP3/HPMRS4 HALLMARKS: "
            "(1) MOST SEVERE HPMRS SUBTYPE: most refractory epilepsy + severest ID in the HPMRS group. "
            "(2) ALP VERY HIGH: often 5-20× ULN — consistently the highest across HPMRS4 patients. "
            "(3) GOLGI REMODELLING STEP: PGAP3 acts in GOLGI (vs PGAP2 in ER) — downstream of PGAP2. "
            "(4) LIPID RAFT BIOLOGY: disaturated GPI anchor required for raft partitioning — "
            "   PGAP3 deficiency → GPI proteins excluded from rafts → signalling dysfunction. "
            "(5) INFANTILE SPASMS + HYPSARRHYTHMIA — vigabatrin/ACTH first-line; often drug-resistant. "
            "(6) GASTROSTOMY often needed — feeding difficulties prominent. "
            "(7) PGAP2 vs PGAP3: PGAP2 removes inositol acyl (ER); PGAP3 removes sn-2 oleate (Golgi). "
            "(8) CONSANGUINITY: most PGAP3 cases from consanguineous families (Moroccan, Saudi, Greek)."
        ),
        "nbs_marker": "Not on NBS; serum ALP the clue; PGAP3 sequencing in refractory infantile spasms + high ALP",
        "key_biomarker": "Serum ALP very high (5-20× ULN); flow FLAER/CD16 reduced; brain MRI (cerebellar hypoplasia); PGAP3 sequencing",
        "severity_spectrum": "Severe to profound ID; highly drug-resistant epilepsy; gastrostomy-dependent; high ALP lifelong",
        "treatments": ["Vigabatrin/ACTH for infantile spasms", "Antiseizure medications", "Ketogenic diet (trial)", "Gastrostomy feeding", "Respiratory support if needed"],
        "emergency": "Status epilepticus management; respiratory failure in advanced cases",
        "ci_drugs": ["Vigabatrin long-term — visual field monitoring required (ERG)"],
    },
    # ── PIGN — Multiple Congenital Anomalies-Hypotonia-Seizures Syndrome (HPMRS2/MCAHS) ─
    {
        "gene": "PIGN", "protein": "Phosphatidylinositol Glycan Class N",
        "alias": "PIGN — HPMRS2/MCAHS (OMIM #614080); AR; step 8 GPI synthesis; hyperphosphatasia + seizures",
        "aa": "1069 aa", "kDa": "120 kDa",
        "gene_class": (
            "Phosphatidylinositol glycan anchor biosynthesis class N (PIGN); "
            "18q21.33 locus; AR inheritance; large enzyme (1069 aa, 120 kDa); "
            "catalyses step 8 of GPI anchor synthesis: addition of phosphoethanolamine (EtNP) "
            "to the first mannose of the GPI core structure (Man1); "
            "PIGN deficiency → GPI anchor synthesis truncated at step 7 → GPI-anchored proteins "
            "are absent or markedly reduced from cell surfaces → TNAP shed → hyperphosphatasia; "
            "mutations may show variable severity from HPMRS-like (ALP + ID + seizures) to "
            "MCAHS-like (congenital anomalies + hypotonia + seizures)"
        ),
        "gpi_subgroup": "GPI Synthesis — Step 8 (EtNP addition to Man1) — AR/HPMRS-MCAHS overlap",
        "pathway_step": "Step 8: phosphoethanolamine addition to first mannose (Man1-GPI → EtNP-Man1-GPI)",
        "locus": "18q21.33", "omim_gene": 606547, "omim_disease": 614080,
        "inheritance": "AR. 18q21.33. Both sexes. Rare; both HPMRS-spectrum and MCAHS-spectrum reported.",
        "seed_offset": 5,
        "onset_range_y": (0.0, 2.0),
        "phenotype": (
            "PIGN-related disorder spans HPMRS and MCAHS spectrum: "
            "HPMRS-like: ID + seizures + markedly elevated ALP; similar to PIGV/HPMRS1. "
            "MCAHS-like: hypotonia + congenital anomalies + seizures ± elevated ALP. "
            "Severity varies with residual enzyme activity. "
            "Seizures: generalised, focal, infantile spasms; drug-resistant in many. "
            "Facial dysmorphism: overlapping with HPMRS (hypertelorism, wide nasal bridge). "
            "Congenital anomalies (in MCAHS-spectrum): cardiac (ASD/VSD), renal anomalies, "
            "genital anomalies, corpus callosum hypoplasia. "
            "ALP: elevated in most PIGN patients (HPMRS-spectrum dominant); may be mildly elevated "
            "in MCAHS-spectrum. "
            "Brain MRI: variable — simplified gyri, CC hypoplasia, white-matter changes in some."
        ),
        "disease": (
            "PIGN (1069 aa, 120 kDa) catalyses step 8 of the >20-step GPI anchor synthesis pathway: "
            "addition of phosphoethanolamine (EtNP) to the first mannose (Man1) of the trimannosyl "
            "core of the GPI anchor. This EtNP on Man1 serves as a linker for the subsequent mannose "
            "additions and is required for correct GPI anchor elongation and protein attachment.\n\n"
            "PIGN DEFICIENCY → GPI synthesis stalls at the GlcN-(acyl)PI-Man3 stage → incomplete GPI "
            "anchor cannot be transferred to proteins → GPI-anchored proteins reduced/absent from "
            "cell surface → TNAP shed → hyperphosphatasia.\n\n"
            "PHENOTYPIC OVERLAP: PIGN causes overlapping HPMRS and MCAHS phenotypes. Original families "
            "described with HPMRS phenotype (ALP + ID + seizures + facial features = Mabry syndrome). "
            "Later, families with MCAHS features (congenital anomalies + hypotonia + seizures) identified. "
            "Both phenotypes can occur with PIGN mutations — severity correlates with residual enzyme "
            "activity and mutation type (null > missense).\n\n"
            "GENETICS: compound heterozygous or homozygous loss-of-function; no clear founder mutation "
            "given the gene's size (1069 aa); WES/WGS required. Mutations in the catalytic domain "
            "more severe; hypomorphic missense variants cause attenuated phenotype.\n\n"
            "MANAGEMENT: antiseizure medications; developmental support; cardiac screening for MCAHS-spectrum; "
            "renal ultrasound; ALP monitoring; flow cytometry (FLAER/CD16) for diagnosis."
        ),
        "hallmark": (
            "PIGN HALLMARKS: "
            "(1) OVERLAPPING HPMRS + MCAHS SPECTRUM — same gene, both phenotypes possible based on "
            "   mutation severity. Null mutations → MCAHS; hypomorphic → HPMRS-like. "
            "(2) STEP 8 OF GPI SYNTHESIS: EtNP addition to Man1 — mid-pathway enzyme. "
            "(3) LARGEST GPI GENE IN THIS ATLAS: 1069 aa / 120 kDa — WES/WGS required for full coverage. "
            "(4) ALP ELEVATED in HPMRS-spectrum; mildly elevated or normal in MCAHS-spectrum. "
            "(5) CARDIAC + RENAL ANOMALIES in MCAHS-spectrum — screen all PIGN patients. "
            "(6) BRAIN MRI: structural anomalies (simplified gyri, CC hypoplasia) in MCAHS-spectrum; "
            "   often normal in HPMRS-spectrum. "
            "(7) DRUG-RESISTANT EPILEPSY: most PIGN patients have challenging seizure control. "
            "(8) FLOW CYTOMETRY: FLAER/CD16/CD24 on granulocytes confirms GPI pathway deficit."
        ),
        "nbs_marker": "Not on NBS; serum ALP and clinical phenotype guide testing; WES/WGS for PIGN (large gene)",
        "key_biomarker": "Serum ALP elevated; flow FLAER/CD16 reduced; cardiac/renal screening; PIGN sequencing (WES/WGS)",
        "severity_spectrum": "Null → severe MCAHS (neonatal); missense → HPMRS-like (childhood ID+epilepsy+high ALP)",
        "treatments": ["Antiseizure medications", "Cardiac surgery if needed", "Renal monitoring", "Developmental support", "ALP monitoring"],
        "emergency": "Cardiac emergency for structural CHD; antiseizure escalation",
        "ci_drugs": ["No specific contraindicated drugs"],
    },
    # ── PIGT — Multiple Congenital Anomalies-Hypotonia-Seizures Syndrome 3 (MCAHS3) ────
    {
        "gene": "PIGT", "protein": "Phosphatidylinositol Glycan Class T",
        "alias": "PIGT — MCAHS3 (OMIM #615398); AR; GPI transamidase complex; multi-system epileptic encephalopathy",
        "aa": "579 aa", "kDa": "65 kDa",
        "gene_class": (
            "Phosphatidylinositol glycan anchor biosynthesis class T (PIGT); "
            "20q13.12; AR inheritance; 579 aa; component of the GPI transamidase complex "
            "(PIGT, PIGU, GPAA1, PIGS, PIGK) in the ER; catalyses the ATTACHMENT of the "
            "completed GPI anchor to proteins (transamidase reaction): protein pre-pro-sequence "
            "GPI signal peptide is cleaved and replaced by GPI anchor (transamidation); "
            "PIGT is the regulatory/structural subunit of the transamidase complex; "
            "deficiency → GPI transamidase inactive → proteins cannot receive GPI anchor → "
            "all GPI-anchored proteins fail surface attachment; ALP markedly elevated"
        ),
        "gpi_subgroup": "GPI Transamidase Complex — GPI Attachment to Protein (final ER step) — AR/MCAHS3",
        "pathway_step": "Final ER step: GPI transamidase (PIGT-PIGU-GPAA1-PIGS-PIGK complex) attaches GPI to protein",
        "locus": "20q13.12", "omim_gene": 610272, "omim_disease": 615398,
        "inheritance": "AR. 20q13.12. Both sexes. Rare; de novo and compound heterozygous reported.",
        "seed_offset": 6,
        "onset_range_y": (0.0, 1.5),
        "phenotype": (
            "MCAHS3 (Multiple Congenital Anomalies-Hypotonia-Seizures Syndrome 3): "
            "Hypotonia: severe neonatal/infantile hypotonia; feeding difficulties; delayed milestones. "
            "Seizures: epileptic encephalopathy; infantile spasms, myoclonic, tonic; drug-resistant. "
            "Congenital anomalies: variable — cardiac (VSD/ASD), renal, skeletal (brachydactyly, "
            "clinodactyly), vertebral. Facial: hypertelorism, broad nasal bridge, ear anomalies. "
            "ALP: markedly elevated (TNAP fails to attach to membrane via GPI → shed into serum). "
            "Brain MRI: simplified gyri, corpus callosum hypoplasia, cortical dysplasia in some. "
            "Additional: intellectual disability (severe); visual impairment (optic atrophy or "
            "cortical visual impairment); hearing loss; failure to thrive. "
            "Prognosis: variable — neonatal forms severe; some attenuated cases survive to childhood."
        ),
        "disease": (
            "PIGT (579 aa, 65 kDa) is the regulatory subunit of the GPI transamidase (GPI-T) complex, "
            "which executes the FINAL step of GPI anchor attachment in the ER: "
            "the pre-pro-protein's C-terminal GPI signal sequence is recognised, cleaved, and "
            "the completed GPI anchor is covalently attached to the omega-site amino acid "
            "(transamidation reaction). PIGT provides structural stability and substrate binding "
            "to the catalytic subunit PIGK.\n\n"
            "PIGT DEFICIENCY → GPI transamidase complex unstable → proteins cannot receive GPI anchor → "
            "ALL GPI-anchored proteins fail to reach the cell surface → most severe GPI defect because "
            "it affects the final attachment step (all upstream synthesis may be intact but useless). "
            "TNAP secreted as free enzyme → marked hyperphosphatasia.\n\n"
            "GPI TRANSAMIDASE COMPLEX (GPI-T): 5 subunits — PIGT (regulatory), PIGU (regulatory), "
            "GPAA1 (structural, GPI binding), PIGS (regulatory), PIGK (catalytic cysteine protease). "
            "Mutations in any of the 5 subunits cause GPI anchor attachment deficiency.\n\n"
            "SOMATIC PIGT MUTATIONS: like PIGA, somatic PIGT mutations in HSC can cause PNH-like "
            "syndrome — distinguishable from germline MCAHS3 by age of onset and setting.\n\n"
            "GENETICS: de novo dominant-negative (heterozygous severe allele) or compound heterozygous "
            "AR; some patients with bi-allelic hypomorphic alleles have attenuated phenotype. "
            "WES/WGS essential for diagnosis.\n\n"
            "MANAGEMENT: antiseizure medications; cardiac/renal screening; gastrostomy; "
            "respiratory monitoring; developmental support. Folinic acid empiric trial reasonable."
        ),
        "hallmark": (
            "PIGT/MCAHS3 HALLMARKS: "
            "(1) FINAL ER ATTACHMENT STEP: PIGT is part of the GPI transamidase — attaches the "
            "   completed GPI anchor to its protein; deficiency = most downstream synthesis defect. "
            "(2) ALP MARKEDLY ELEVATED: GPI transamidase deficiency → TNAP cannot be GPI-anchored "
            "   → shed → high serum ALP; ALP useful as biomarker for disease activity. "
            "(3) GPI-T COMPLEX: 5-subunit complex (PIGT/PIGU/GPAA1/PIGS/PIGK); "
            "   mutations in any subunit cause GPI attachment deficiency. "
            "(4) SEVERE MULTI-SYSTEM: neonatal hypotonia + seizures + cardiac/renal anomalies "
            "   + brain malformations — one of the most severe GPI phenotypes. "
            "(5) DE NOVO DOMINANT-NEGATIVE possible: unlike other GPI genes where AR biallelic "
            "   is the rule, some severe PIGT alleles act dominant-negative. "
            "(6) FOLINIC ACID TRIAL: reasonable empiric therapy (FOLR1 GPI-anchored). "
            "(7) BRAIN MRI: structural malformations in some (simplified gyri, CC hypoplasia) — "
            "   similar to PIGA. "
            "(8) DRUG-RESISTANT EPILEPSY: vigabatrin/ACTH for spasms; KD anecdotal benefit."
        ),
        "nbs_marker": "Not on NBS; serum ALP + congenital anomalies + seizures → PIGT sequencing (WES/WGS)",
        "key_biomarker": "Serum ALP elevated; flow FLAER/CD16 reduced; multi-system anomaly screen; PIGT sequencing (WES/WGS)",
        "severity_spectrum": "Neonatal severe (multi-organ) to attenuated (ID + epilepsy alone); de novo → severe",
        "treatments": ["Antiseizure medications (vigabatrin/ACTH for spasms)", "Folinic acid empiric trial", "Cardiac/renal surgical correction", "Gastrostomy", "Developmental support"],
        "emergency": "Status epilepticus; cardiac emergency; respiratory failure in neonates",
        "ci_drugs": ["No specific contraindicated drugs"],
    },
    # ── PIGG — Intellectual Disability with Epilepsy / HPMRS6 ─────────────────────────
    {
        "gene": "PIGG", "protein": "Phosphatidylinositol Glycan Class G",
        "alias": "PIGG — HPMRS6/ID+epilepsy (OMIM #617582); AR; EtNP addition to Man2; atypical GPI phenotype",
        "aa": "866 aa", "kDa": "98 kDa",
        "gene_class": (
            "Phosphatidylinositol glycan anchor biosynthesis class G (PIGG); "
            "4p16.3; AR inheritance; 866 aa, 98 kDa; "
            "GPI anchor biosynthesis enzyme in the ER; catalyses addition of "
            "phosphoethanolamine (EtNP) to the SECOND mannose (Man2) of the trimannosyl "
            "GPI core structure; EtNP on Man2 serves as a sidebranch — its function is "
            "less critical than EtNP on Man3 (the protein-attachment ethanolamine); "
            "PIGG deficiency → partial/abnormal GPI anchor → mildly reduced surface GPI proteins; "
            "phenotype MILDER than other GPI disorders (HPMRS6 — intellectual disability + epilepsy "
            "without severe congenital anomalies or markedly elevated ALP)"
        ),
        "gpi_subgroup": "GPI Synthesis — Step 9 (EtNP addition to Man2, sidebranch) — AR/HPMRS6-atypical",
        "pathway_step": "Step 9: phosphoethanolamine addition to second mannose (Man2) — sidebranch, less critical",
        "locus": "4p16.3", "omim_gene": 616918, "omim_disease": 617582,
        "inheritance": "AR. 4p16.3. Both sexes. Rare; attenuated GPI phenotype.",
        "seed_offset": 7,
        "onset_range_y": (0.0, 5.0),
        "phenotype": (
            "PIGG-related disorder (HPMRS6): MILDEST of the GPI anchor disorder spectrum. "
            "Intellectual disability: mild to moderate; often better cognitive outcomes than PIGV/PGAP3. "
            "Epilepsy: focal and generalised seizures; less refractory than other GPI disorders; "
            "febrile seizures + afebrile seizures; often respond to 1-2 AEDs. "
            "Behavioral: autistic features, hyperactivity, attention deficit. "
            "ALP: mildly elevated or normal (PIGG only partially reduces surface GPI proteins "
            "because EtNP on Man2 is a sidebranch — less critical than Man3 EtNP). "
            "No major congenital anomalies (no cardiac, no renal, no ichthyosis). "
            "Brain MRI: typically normal; sometimes nonspecific white-matter changes. "
            "Facial: subtle dysmorphism; may be overlooked clinically. "
            "Overall: PIGG should be considered in mild-moderate ID + epilepsy + "
            "mildly elevated ALP without a more striking GPI phenotype."
        ),
        "disease": (
            "PIGG (866 aa, 98 kDa) catalyses step 9 of GPI anchor synthesis: addition of "
            "phosphoethanolamine (EtNP) to the second mannose (Man2) of the trimannosyl core "
            "(Man1-Man2-Man3-GlcN-PI). This EtNP on Man2 is a SIDEBRANCH modification — "
            "unlike the critical EtNP on Man3 (the one that actually links to the omega-site of "
            "the protein via transamidation), the Man2-EtNP is not strictly required for "
            "GPI anchor synthesis or protein attachment.\n\n"
            "WHY PIGG PHENOTYPE IS MILDER: since Man2-EtNP is a sidebranch modification, "
            "PIGG deficiency does not completely block GPI anchor synthesis or protein attachment. "
            "Surface GPI-anchored proteins are mildly reduced (not absent). TNAP shed slightly → "
            "ALP mildly elevated (vs 3-10× ULN in HPMRS1-5). This explains the milder clinical "
            "phenotype (no severe congenital anomalies, milder ID, less refractory epilepsy).\n\n"
            "HPMRS6: the PIGG phenotype was categorised as HPMRS6 to indicate the hyperphosphatasia "
            "component, but it is the mildest within the HPMRS spectrum. "
            "Initially described as HPMRS in families with mildly elevated ALP + ID + epilepsy "
            "without other Mabry syndrome features.\n\n"
            "GENETICS: homozygous or compound heterozygous PIGG missense/frameshift; "
            "p.Arg 468* and p.Leu422Pro described in original cohorts; consanguineous families. "
            "WES/WGS captures all variants; targeted panel for GPI pathway.\n\n"
            "MANAGEMENT: antiseizure medications (usually well-controlled with 1-2 drugs); "
            "developmental support; behavioural therapy for ASD features; ALP monitoring. "
            "Folinic acid empiric trial may be beneficial given FOLR1 GPI-anchoring."
        ),
        "hallmark": (
            "PIGG/HPMRS6 HALLMARKS: "
            "(1) MILDEST GPI ANCHOR DISORDER: EtNP on Man2 is a SIDEBRANCH — not critical for "
            "   protein attachment → milder phenotype than all other GPI gene disorders in this atlas. "
            "(2) ALP MILDLY ELEVATED OR NORMAL: unlike HPMRS1-4 which have 3-20× ULN; "
            "   PIGG has only mild ALP elevation — easy to miss clinically. "
            "(3) NO MAJOR CONGENITAL ANOMALIES: no cardiac, no renal, no ichthyosis — distinct from "
            "   PIGA (MCAHS1), PIGL (CHIME), PIGT (MCAHS3). "
            "(4) EPILEPSY OFTEN DRUG-RESPONSIVE: 1-2 AEDs typically effective (vs refractory in PGAP3). "
            "(5) MILD-MODERATE ID + ASD FEATURES: the cognitive phenotype — consider PIGG in "
            "   mild-moderate ID + epilepsy + autistic features + subtle ALP elevation. "
            "(6) BRAIN MRI NORMAL: no structural malformations. "
            "(7) STEP 9 — SIDEBRANCH: Man2-EtNP is a modification that modulates GPI but is not "
            "   essential — explains why PIGG deficiency has partial rather than complete GPI deficit. "
            "(8) PIGG IS THE DDx for other GPI disorders: milder ALP, no anomalies, mild-moderate ID."
        ),
        "nbs_marker": "Not on NBS; mildly elevated ALP + ID + epilepsy without anomalies → consider PIGG; WES/WGS diagnostic",
        "key_biomarker": "Serum ALP mildly elevated or normal; flow FLAER/CD16 mildly reduced; PIGG sequencing (WES/WGS)",
        "severity_spectrum": "Mild to moderate ID (best cognitive outcomes in GPI disorders); seizures drug-responsive; ALP mildly elevated",
        "treatments": ["Antiseizure medications (usually 1-2 AEDs sufficient)", "Behavioural therapy for ASD features", "Folinic acid empiric trial", "Developmental support"],
        "emergency": "Standard seizure emergency protocols; no GPI-specific emergency",
        "ci_drugs": ["No specific contraindicated drugs"],
    },
]


def _make_patients(gene_data):
    """Generate deterministic 40-patient cohort for a GPI anchor disorder gene."""
    gene = gene_data["gene"]
    rng = random.Random(SEED_BASE + gene_data["seed_offset"])
    patients = []
    onset_lo, onset_hi = gene_data["onset_range_y"]

    # PIGA is X-linked — weight males more heavily
    male_weight = 3 if gene == "PIGA" else 1

    for i in range(40):
        if gene == "PIGA":
            sex = rng.choices(["M", "F"], weights=[3, 1], k=1)[0]
        else:
            sex = rng.choice(["M", "F"])
        age_onset = round(rng.uniform(onset_lo, onset_hi), 2)
        dx_delay = round(rng.uniform(0.2, 6.0), 2)
        age_dx = round(age_onset + dx_delay, 2)
        age_now = round(age_dx + rng.uniform(0.5, 18.0), 2)
        severity = rng.choices(
            ["Severe", "Moderate", "Mild"],
            weights={
                "PIGA":  [60, 30, 10],
                "PIGV":  [40, 45, 15],
                "PIGL":  [35, 45, 20],
                "PGAP2": [25, 50, 25],
                "PGAP3": [65, 30, 5],
                "PIGN":  [50, 35, 15],
                "PIGT":  [65, 30, 5],
                "PIGG":  [15, 50, 35],
            }[gene], k=1
        )[0]
        tx_map = {
            "PIGA":  rng.choice(["Vigabatrin + ACTH", "Vigabatrin alone", "Folinic acid + valproate", "KD + levetiracetam", "Supportive (died neonatal)"]),
            "PIGV":  rng.choice(["Valproate", "Levetiracetam", "Phenobarbital", "Folinic acid + valproate", "KD + levetiracetam"]),
            "PIGL":  rng.choice(["Levetiracetam", "Valproate", "Supportive", "Vigabatrin", "Carbamazepine"]),
            "PGAP2": rng.choice(["Valproate", "Levetiracetam + clobazam", "Folinic acid + valproate", "KD", "Vigabatrin"]),
            "PGAP3": rng.choice(["Vigabatrin + ACTH", "KD + vigabatrin", "Valproate polytherapy", "Fenfluramine trial", "Supportive (neonatal)"]),
            "PIGN":  rng.choice(["Vigabatrin", "Valproate", "Levetiracetam", "KD + clobazam", "ACTH + vigabatrin"]),
            "PIGT":  rng.choice(["Vigabatrin + ACTH", "Folinic acid + vigabatrin", "KD", "Valproate polytherapy", "Supportive"]),
            "PIGG":  rng.choice(["Levetiracetam", "Valproate", "Lamotrigine", "Levetiracetam + clobazam", "Oxcarbazepine"]),
        }[gene]
        alp_fold = {
            "PIGA":  round(rng.uniform(1.0, 3.0), 1),   # mildly elevated
            "PIGV":  round(rng.uniform(3.0, 10.0), 1),  # markedly elevated
            "PIGL":  round(rng.uniform(1.0, 2.5), 1),   # mildly elevated
            "PGAP2": round(rng.uniform(4.0, 15.0), 1),  # very elevated
            "PGAP3": round(rng.uniform(5.0, 20.0), 1),  # highest
            "PIGN":  round(rng.uniform(2.0, 8.0), 1),   # elevated
            "PIGT":  round(rng.uniform(3.0, 12.0), 1),  # elevated
            "PIGG":  round(rng.uniform(1.0, 2.0), 1),   # mildly elevated or normal
        }[gene]
        flaer_pct = round(rng.uniform(15, 65), 1)  # % of normal GPI expression
        outcome = rng.choices(
            ["Seizure-free", "Seizure reduction >50%", "Partial response", "Drug-resistant", "Neonatal death"],
            weights={
                "PIGA":  [5, 15, 30, 35, 15],
                "PIGV":  [10, 25, 35, 30, 0],
                "PIGL":  [15, 30, 35, 20, 0],
                "PGAP2": [15, 30, 35, 20, 0],
                "PGAP3": [5, 10, 30, 50, 5],
                "PIGN":  [10, 20, 35, 30, 5],
                "PIGT":  [5, 10, 30, 40, 15],
                "PIGG":  [30, 40, 25, 5, 0],
            }[gene], k=1
        )[0]
        patients.append({
            "patient_id": f"{gene}-{i+1:03d}",
            "gene": gene,
            "age_onset_y": age_onset,
            "age_dx_y": age_dx,
            "age_now_y": age_now,
            "dx_delay_y": round(dx_delay, 2),
            "sex": sex,
            "severity": severity,
            "treatment": tx_map,
            "alp_fold_uln": alp_fold,
            "flaer_pct_normal": flaer_pct,
            "outcome": outcome,
        })
    return patients


def _all_patients():
    all_pts = []
    for g in GPI_GENES:
        all_pts.extend(_make_patients(g))
    return all_pts


# ─── Public API ───────────────────────────────────────────────────────────────

def get_overview():
    pts = _all_patients()
    n = len(pts)
    sev_counts = {"Severe": 0, "Moderate": 0, "Mild": 0}
    for p in pts:
        sev_counts[p["severity"]] += 1
    avg_delay = round(sum(p["dx_delay_y"] for p in pts) / n, 2)
    avg_onset = round(sum(p["age_onset_y"] for p in pts) / n, 2)
    avg_alp = round(sum(p["alp_fold_uln"] for p in pts) / n, 1)
    avg_flaer = round(sum(p["flaer_pct_normal"] for p in pts) / n, 1)
    gene_counts = {}
    for p in pts:
        gene_counts[p["gene"]] = gene_counts.get(p["gene"], 0) + 1
    outcome_counts = {}
    for p in pts:
        outcome_counts[p["outcome"]] = outcome_counts.get(p["outcome"], 0) + 1
    return {
        "atlas": "GPI-Anchor Biosynthesis Disorders Atlas",
        "subtitle": "Complete 8-Gene GPI-Anchor Biosynthesis Disorders Reference",
        "description": (
            "The GPI-Anchor Biosynthesis Disorders Atlas covers 8 genes in the GPI "
            "glycolipid anchor pathway, all causing epileptic encephalopathy with "
            "intellectual disability. GPI anchors attach >150 surface proteins to the "
            "outer plasma membrane leaflet. Defects in GPI anchor synthesis or processing "
            "cause a spectrum from severe neonatal multi-system disease (PIGA, PIGT) to "
            "attenuated ID+epilepsy (PIGG). Markedly elevated serum ALP (released TNAP) "
            "is a key biomarker in the HPMRS group (PIGV, PGAP2, PGAP3, PIGN, PIGT). "
            "Flow cytometric GPI testing (FLAER/CD16/CD24 on granulocytes) is the screening test."
        ),
        "genes": [g["gene"] for g in GPI_GENES],
        "n_genes": len(GPI_GENES),
        "total_patients": n,
        "avg_onset_y": avg_onset,
        "avg_dx_delay_y": avg_delay,
        "avg_alp_fold_uln": avg_alp,
        "avg_flaer_pct_normal": avg_flaer,
        "severity_distribution": sev_counts,
        "gene_counts": gene_counts,
        "outcome_distribution": outcome_counts,
        "gpi_groups": {
            "GPI Synthesis (early steps)": ["PIGA", "PIGL", "PIGV", "PIGN"],
            "Post-GPI Processing (lipid remodelling)": ["PGAP2", "PGAP3"],
            "GPI Transamidase (protein attachment)": ["PIGT"],
            "GPI Synthesis (sidebranch, mild phenotype)": ["PIGG"],
        },
        "phenotype_groups": {
            "MCAHS (Congenital Anomalies + Hypotonia + Seizures)": ["PIGA", "PIGN", "PIGT"],
            "HPMRS (Hyperphosphatasia + Mental Retardation + Seizures)": ["PIGV", "PGAP2", "PGAP3", "PIGN", "PIGT"],
            "CHIME (Coloboma + Heart + Ichthyosis + MR + Ear)": ["PIGL"],
            "Mild ID + Epilepsy (HPMRS6)": ["PIGG"],
        },
        "key_teaching": [
            "HIGH SERUM ALP IN CHILD WITH ID + SEIZURES = GPI ANCHOR DISORDER UNTIL PROVEN OTHERWISE",
            "ALP IS GPI-ANCHORED (TNAP): GPI defects shed TNAP into serum → hyperphosphatasia is a biomarker",
            "ALP IS NOT HEPATIC/BONE DISEASE: do not treat with ALP-lowering agents; bone mineralisation typically normal",
            "FLOW CYTOMETRY (FLAER + CD16 + CD24 on GRANULOCYTES): the diagnostic screening test; lymphocytes NOT usable",
            "PIGA IS X-LINKED: only gene in GPI pathway that is X-linked; all pathogenic germline alleles are hypomorphic",
            "GERMLINE PIGA (MCAHS1) vs SOMATIC PIGA (PNH): completely different diseases; do not confuse",
            "CHIME SYNDROME (PIGL): migratory ichthyotic skin lesions are PATHOGNOMONIC for PIGL",
            "PIGG IS MILDEST: Man2-EtNP is a sidebranch — PIGG deficiency partially reduces GPI; mild ID + epilepsy only",
            "FOLINIC ACID EMPIRIC TRIAL: FOLR1 is GPI-anchored → brain folate deficiency in all GPI disorders; low risk to try",
            "EPILEPSY CARDINAL: >90% of GPI anchor disorder patients have epilepsy; infantile spasms (vigabatrin/ACTH) common",
        ],
        "emergency_summary": {
            "PIGA":  "Infantile spasms → vigabatrin/ACTH immediately; folinic acid empiric trial",
            "PIGV":  "Refractory seizures → antiseizure escalation; folinic acid trial",
            "PIGL":  "CHD surgical emergency if haemodynamically compromised; seizures → standard AEDs",
            "PGAP2": "Refractory seizures → escalate; folinic acid trial; no specific emergency",
            "PGAP3": "Infantile spasms + refractory → vigabatrin/ACTH; consider KD; respiratory monitoring",
            "PIGN":  "CHD emergency if MCAHS-spectrum; antiseizure escalation; folinic acid trial",
            "PIGT":  "Neonatal multi-system emergency; cardiac/respiratory support; vigabatrin/ACTH for spasms",
            "PIGG":  "Standard seizure protocols; usually responds to 1-2 AEDs; folinic acid trial",
        },
        "seeds_used": f"{SEED_BASE}–{SEED_BASE + len(GPI_GENES) - 1}",
    }


def get_breakdown():
    result = []
    for gd in GPI_GENES:
        pts = _make_patients(gd)
        sev = {"Severe": 0, "Moderate": 0, "Mild": 0}
        for p in pts:
            sev[p["severity"]] += 1
        avg_delay = round(sum(p["dx_delay_y"] for p in pts) / len(pts), 1)
        avg_onset = round(sum(p["age_onset_y"] for p in pts) / len(pts), 1)
        avg_alp = round(sum(p["alp_fold_uln"] for p in pts) / len(pts), 1)
        avg_flaer = round(sum(p["flaer_pct_normal"] for p in pts) / len(pts), 1)
        treatments = {}
        for p in pts:
            treatments[p["treatment"]] = treatments.get(p["treatment"], 0) + 1
        top_tx = sorted(treatments.items(), key=lambda x: -x[1])[:3]
        outcomes = {}
        for p in pts:
            outcomes[p["outcome"]] = outcomes.get(p["outcome"], 0) + 1
        result.append({
            "gene": gd["gene"],
            "protein": gd["protein"],
            "alias": gd["alias"],
            "aa": gd["aa"],
            "locus": gd["locus"],
            "omim_gene": gd["omim_gene"],
            "omim_disease": gd["omim_disease"],
            "inheritance": gd["inheritance"],
            "gpi_subgroup": gd["gpi_subgroup"],
            "pathway_step": gd["pathway_step"],
            "n_patients": len(pts),
            "severity_distribution": sev,
            "avg_onset_y": avg_onset,
            "avg_dx_delay_y": avg_delay,
            "avg_alp_fold_uln": avg_alp,
            "avg_flaer_pct_normal": avg_flaer,
            "top_treatments": [{"treatment": t, "n": n_} for t, n_ in top_tx],
            "outcome_distribution": outcomes,
            "hallmark": gd["hallmark"],
            "nbs_marker": gd["nbs_marker"],
            "key_biomarker": gd["key_biomarker"],
            "severity_spectrum": gd["severity_spectrum"],
            "emergency": gd["emergency"],
            "ci_drugs": gd["ci_drugs"],
            "disease_summary": gd["disease"][:600] + "…",
            "phenotype_summary": gd["phenotype"][:400] + "…",
            "gene_class": gd["gene_class"][:400] + "…",
        })
    return {"breakdown": result, "total_genes": len(result)}


def get_definitions():
    return {
        "definitions": [
            {"term": "GPI Anchor (Glycosylphosphatidylinositol)", "definition": "A glycolipid post-translational modification that attaches >150 proteins to the outer leaflet of the plasma membrane. Structure: phosphatidylinositol (PI) lipid tail → glucosamine → trimannosyl core → ethanolamine phosphate linker → protein C-terminus. Synthesised in the ER by >20 sequential enzymes (PIGA through GPAA1/PIGU). Required for membrane localisation of complement regulators (CD55, CD59), alkaline phosphatase (TNAP), folate receptor (FOLR1), prion protein, Thy-1, uPAR, and others."},
            {"term": "Hyperphosphatasia (HPMRS)", "definition": "Markedly elevated serum alkaline phosphatase (ALP/TNAP) in GPI anchor disorders. Mechanism: tissue-nonspecific alkaline phosphatase (TNAP) is a GPI-anchored enzyme; GPI anchor deficiency → TNAP cannot be anchored to cell surface → secreted as free (soluble) enzyme into serum → serum ALP elevated 3-20× ULN. Elevated ALP does NOT indicate hepatic or bone disease in GPI disorders — isoenzyme fractionation or clinical context confirms. HPMRS = Hyperphosphatasia with Mental Retardation Syndrome."},
            {"term": "FLAER (Fluorescent Aerolysin)", "definition": "A flow cytometric reagent for GPI anchor detection. Aerolysin (from Aeromonas hydrophila) specifically binds to GPI anchors; conjugated to a fluorochrome → measures total GPI anchor density on cell surface. Used on GRANULOCYTES (not lymphocytes — lymphocytes partially shed GPI proteins normally, giving false-low results). Reduced FLAER staining = GPI anchor deficit. Combined with anti-CD16 (GPI-anchored FcγRIII) and anti-CD24 for comprehensive GPI-deficit testing."},
            {"term": "GPI Transamidase Complex (GPI-T)", "definition": "Five-subunit ER enzyme complex that performs the final step of GPI anchor attachment: the C-terminal GPI signal peptide of the nascent protein is cleaved and replaced by the completed GPI anchor (transamidation reaction). Subunits: PIGK (catalytic cysteine protease), PIGT (regulatory/structural), PIGU (regulatory), GPAA1 (structural, GPI binding), PIGS (regulatory). Mutations in any subunit cause GPI anchor attachment failure. PIGT deficiency causes MCAHS3."},
            {"term": "MCAHS (Multiple Congenital Anomalies-Hypotonia-Seizures Syndrome)", "definition": "Clinical phenotype of severe GPI anchor disorders characterised by: Multiple congenital anomalies (cardiac, renal, skeletal, urogenital), neonatal/infantile Hypotonia, and epileptic Seizures (often infantile spasms). MCAHS1: PIGA (X-linked). MCAHS2: PGAP3. MCAHS3: PIGT. Overlapping MCAHS/HPMRS phenotype: PIGN. The congenital anomalies reflect the role of GPI-anchored proteins in organogenesis."},
            {"term": "Phosphoethanolamine (EtNP) Bridging", "definition": "The mechanism linking the completed GPI trimannosyl core to the protein omega-site. After synthesis of the trimannosyl core (Man1-Man2-Man3), an ethanolamine phosphate (EtNP) is added to Man3 — this EtNP bridges Man3 to the omega-site amino acid of the protein via transamidation. PIGN adds EtNP to Man1 (step 8); PIGG adds EtNP to Man2 (step 9, sidebranch); PIGN-added EtNP and PIGG-added EtNP are sidebranches — the Man3-EtNP is the critical attachment point."},
            {"term": "Lipid Raft Partitioning of GPI Proteins", "definition": "GPI-anchored proteins preferentially localise to cholesterol-rich membrane microdomains called lipid rafts. Correct GPI anchor lipid composition (with two saturated fatty acids — stearate at sn-1 and sn-2 positions) is required for raft partitioning. PGAP3 removes the sn-2 unsaturated fatty acid (oleate) from the GPI anchor in the Golgi, enabling replacement by stearate. PGAP3 deficiency → retained unsaturated sn-2 oleate → GPI anchor excluded from lipid rafts → signalling dysfunction."},
            {"term": "CHIME Syndrome (PIGL)", "definition": "Rare syndrome caused by biallelic PIGL mutations: Coloboma (chorioretinal/iris), Heart defects (structural CHD), Ichthyosiform dermatosis (migratory/recurrent — PATHOGNOMONIC), Mental retardation (moderate-severe), Ear anomalies (aural atresia, microtia). The migratory ichthyotic skin lesions are pathognomonic — they appear, migrate, and resolve over weeks, triggered by viral illness. PIGL catalyses step 2 of GPI synthesis (GlcNAc-PI deacetylation)."},
            {"term": "FOLR1 (Folate Receptor 1) Brain Folate Deficiency", "definition": "FOLR1 (alpha-folate receptor) is a GPI-anchored protein expressed on the choroid plexus epithelium; it imports 5-methyltetrahydrofolate (5-MTHF) into the CSF. GPI anchor disorders reduce surface FOLR1 → impaired CSF folate import → low CSF 5-MTHF (cerebral folate deficiency) despite normal serum folate. Empiric folinic acid (leucovorin, 5-formyl-THF) supplementation is recommended in all GPI anchor disorders — it bypasses FOLR1 by passive diffusion at pharmacological doses. CSF 5-MTHF should be measured to confirm deficiency."},
            {"term": "Somatic vs Germline PIGA Mutations", "definition": "Two completely different diseases share PIGA as the causative gene: (1) SOMATIC PIGA mutations acquired in haematopoietic stem cells → PNH (Paroxysmal Nocturnal Haemoglobinuria): clonal haematopoiesis, complement-mediated haemolysis, thrombosis — onset in adults; treated with eculizumab. (2) GERMLINE hypomorphic PIGA mutations → MCAHS1 (Congenital GPI Deficiency): neonatal/infantile epileptic encephalopathy + congenital anomalies — treated with AEDs + supportive care. PNH is an acquired clonal disorder; MCAHS1 is a germline congenital IEM."},
            {"term": "GPI Anchor Pathway Step Order", "definition": "GPI anchor synthesis occurs in the ER by >20 sequential enzymes. Key steps relevant to this atlas: Step 1 (PIGA complex): GlcNAc-PI synthesis; Step 2 (PIGL): GlcNAc-PI → GlcN-PI deacetylation; Step 6 (PIGV): second mannose addition (Man2); Step 8 (PIGN): EtNP to Man1; Step 9 (PIGG): EtNP to Man2 (sidebranch). Post-attachment ER remodelling: PGAP2 (inositol deacylation). Post-attachment Golgi remodelling: PGAP3 (sn-2 oleate removal). Final attachment: GPI-T complex (PIGT subunit)."},
            {"term": "Alkaline Phosphatase Isoenzyme Fractionation", "definition": "When serum ALP is elevated, isoenzyme fractionation distinguishes the source: liver-ALP (hepatic disease), bone-ALP (rickets/Paget), intestinal-ALP, placental-ALP, and tissue-nonspecific ALP (TNAP, the GPI-anchored form). In GPI anchor disorders, the elevated isoenzyme is TNAP (tissue-nonspecific); liver and bone isoenzymes are normal. This confirms the elevated ALP is NOT from liver or bone disease. Alternatively, gamma-GT (GGT) is normal in GPI disorders (GGT is not GPI-anchored)."},
            {"term": "Vigabatrin for GPI-Related Infantile Spasms", "definition": "Vigabatrin (GABA transaminase inhibitor) is a first-line agent for infantile spasms in GPI anchor disorders, especially PIGA/MCAHS1 and PGAP3/HPMRS4. Combined with ACTH for refractory cases. IMPORTANT MONITORING: vigabatrin causes irreversible concentric visual field constriction (retinal toxicity) in 30-50% of patients — requires serial electroretinograms (ERG) every 3-6 months during treatment. Lower doses (<100mg/kg/day) may reduce but not eliminate this risk. Benefits for infantile spasms typically outweigh visual risk in the context of severe GPI encephalopathy."},
            {"term": "Ketogenic Diet in GPI Anchor Disorders", "definition": "The ketogenic diet (KD) has been reported anecdotally in several GPI anchor disorder patients with refractory seizures, particularly PGAP3 and PIGA. Proposed mechanism: ketone bodies provide alternative cerebral energy substrate and modulate GABA/glutamate neurotransmission. Evidence base is limited to case series (<20 published cases with GPI disorders). No randomised controlled trial data. KD is a reasonable option in drug-resistant GPI-related epilepsy after failure of 2-3 appropriate AEDs."},
            {"term": "Post-GPI Processing (PGAP2 and PGAP3)", "definition": "After the GPI anchor is fully synthesised and attached to a protein in the ER, two lipid remodelling steps optimise the anchor: (1) PGAP2 (ER): removes the inositol-linked acyl chain (inositol deacylation) — allows the GPI anchor to move from ER to Golgi. (2) PGAP3 (Golgi): removes the sn-2 unsaturated fatty acid (oleate) from the diacylglycerol — oleate is replaced by stearate by MPPE1 → disaturated GPI anchor required for lipid raft partitioning. PGAP2 acts BEFORE PGAP3 in the pathway; both deficiencies cause HPMRS with high ALP."},
        ]
    }


if __name__ == "__main__":
    import json
    ov = get_overview()
    print("=== GPI ANCHOR ATLAS OVERVIEW ===")
    print(f"Genes: {ov['genes']}")
    print(f"Total patients: {ov['total_patients']}")
    print(f"Avg onset: {ov['avg_onset_y']} y")
    print(f"Avg dx delay: {ov['avg_dx_delay_y']} y")
    print(f"Avg ALP fold ULN: {ov['avg_alp_fold_uln']}")
    print(f"Avg FLAER %: {ov['avg_flaer_pct_normal']}")
    print(f"GPI groups: {list(ov['gpi_groups'].keys())}")
    bd = get_breakdown()
    print(f"\n=== BREAKDOWN ({bd['total_genes']} genes) ===")
    for g in bd["breakdown"]:
        print(f"  {g['gene']}: {g['n_patients']} pts, onset {g['avg_onset_y']}y, delay {g['avg_dx_delay_y']}y, ALP {g['avg_alp_fold_uln']}x ULN")
    defs = get_definitions()
    print(f"\n=== DEFINITIONS: {len(defs['definitions'])} terms ===")
