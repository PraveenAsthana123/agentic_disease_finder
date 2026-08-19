#!/usr/bin/env python3
"""GNPTAB / Mucolipidosis II (I-Cell Disease) & ML-IIIA/B (Pseudo-Hurler) Epilepsy Dashboard — seed data module.

Mucolipidosis II (ML-II, I-Cell Disease) and Mucolipidosis IIIA/B (Pseudo-Hurler Polydystrophy):
caused by biallelic LOF in GNPTAB gene (12q23.2), encoding the alpha/beta subunit of
UDP-GlcNAc:lysosomal enzyme GlcNAc-1-phosphotransferase → failure to add mannose-6-phosphate (M6P)
targeting tag → lysosomal enzymes secreted into PLASMA instead of trafficked to lysosomes.

KEY DISTINGUISHING FEATURES:
  1. INVERSE PLASMA/LEUKOCYTE ENZYME PATTERN (PATHOGNOMONIC) — lysosomal enzymes are
     MARKEDLY ELEVATED in PLASMA (10-50x normal) but LOW or absent in LEUKOCYTES; the OPPOSITE
     of most LSDs where leukocyte enzyme activity is low but plasma not elevated; this inverse
     pattern is the biochemical hallmark; dual panel: plasma HYAL1/HEX-B ELEVATED +
     leukocyte HEX-A/B LOW → diagnostic for ML-II/III before gene sequencing.
  2. GINGIVAL HYPERPLASIA (ML-II PATHOGNOMONIC) — thick, fibrotic, hyperplastic gums from
     birth in ML-II; visible at first clinical examination; one of earliest clinical signs;
     distinguishes ML-II from MPS-I Hurler (gingival disease mild) and all other coarse facies
     LSDs; absence of this feature in ML-IIIA/B is part of the phenotypic spectrum difference.
  3. NO CORNEAL CLOUDING (KEY DISTINGUISHER) — ML-II and ML-IIIA/B have NO corneal clouding;
     this sharply distinguishes from MPS-I Hurler (corneal clouding universal), MPS-VI Maroteaux-
     Lamy (universal), MPS-VII Sly (present), and ML-IV MCOLN1 (present); absence of corneal
     clouding in the presence of coarse facies narrows the differential critically.
  4. DUAL PHENOTYPE FROM ONE GENE — GNPTAB generates two distinct clinical phenotypes:
     ML-II (null alleles → severe, onset birth-6mo, death age 5-8yr) and ML-IIIA/B (missense
     alleles → attenuated, onset 2-5yr, survival into adulthood); phenotype-genotype correlation
     is strong (null vs missense), making genotype clinically informative.
  5. CARPAL TUNNEL SYNDROME (ML-IIIA — PRESENTING SYMPTOM) — bilateral carpal tunnel syndrome
     in 85% of ML-IIIA/B patients by age 10; often the FIRST clinical sign leading to diagnosis
     in the attenuated phenotype; unexplained bilateral CTS in a child with joint stiffness →
     metabolic workup; CTS release is Level A therapeutic intervention.
  6. CARDIOMYOPATHY (ML-II — 80%) — cardiomegaly/cardiomyopathy in 80% of ML-II patients;
     major drug safety driver: PHT ABSOLUTE CI (cardiac conduction), VGB ABSOLUTE CI for IS
     (cardiac toxicity + cardiomyopathy), CBZ/OXC CAUTION (PR prolongation); ACTH preferred
     over VGB for infantile spasms due to cardiomyopathy; fosphenytoin ABSOLUTE CI in SE.
  7. I-CELL INCLUSIONS (FIBROBLAST — PHASE CONTRAST) — dense lysosomal inclusions visible
     in cultured skin fibroblasts by phase contrast microscopy → "I-cells" (inclusion cells) →
     disease name "I-Cell Disease"; historical name predates biochemical characterization;
     inclusions represent accumulated undigested glycoproteins, glycolipids, GAGs in lysosomes.
  8. NO ERT APPROVED (2026) — despite being related to the M6P targeting system; GNPTAB
     enzyme replacement is complicated by the phosphotransferase machinery itself being the
     defective enzyme (cannot use standard M6P-targeted ERT); no HSCT evidence in ML-II.
  9. PERIOSTEAL NEW BONE FORMATION (X-RAY — ML-II) — new bone formation around long bones
     on plain X-ray; seen from infancy in ML-II; part of dysostosis multiplex pattern but with
     the periosteal reaction being more prominent than other LSDs.

Enzyme biology:
  GNPTAB encodes the alpha (alpha) and beta (beta) subunits of UDP-GlcNAc:lysosomal enzyme
  GlcNAc-1-phosphotransferase (the GNPTAB complex). This enzyme adds GlcNAc-1-phosphate to
  high-mannose N-linked oligosaccharides on nascent lysosomal hydrolases in the cis-Golgi,
  generating the mannose-6-phosphate (M6P) recognition marker. Without the M6P tag, lysosomal
  enzymes fail to bind M6P receptors in the trans-Golgi → are not sorted to lysosomes →
  are instead SECRETED into the extracellular space (plasma). Two phenotypes:
  ML-II (null alleles): complete loss of phosphotransferase activity → severe phenotype, early death.
  ML-IIIA/B (missense alleles): partial residual phosphotransferase activity → attenuated phenotype.
  GNPTAB gene at 12q23.2; OMIM gene 607840; OMIM ML-II 252500; OMIM ML-IIIA/B 252600.
  The gamma subunit (GNPTG gene, 16p13.3) completes the hexameric alpha2-beta2-gamma2 complex;
  GNPTG mutations cause the milder ML-IIIC (Pseudo-Hurler Polydystrophy, variant C) — distinct gene.

Pharmacology:
  Epilepsy management is symptomatic (no disease-modifying ERT available).
  POLG1 exclusion MANDATORY before VPA (CPIC Grade A — same rationale as other storage disorders).
  ACTH Level A for infantile spasms in ML-II (VGB ABSOLUTE CI due to cardiomyopathy).
  PHT/Fosphenytoin ABSOLUTE CI in ML-II (cardiac — QTc prolongation + cardiomyopathy).
  CBZ/OXC CAUTION in ML-II (PR prolongation risk in cardiomyopathy); lacosamide preferred.
  Levetiracetam (IV) first-line for SE (NOT fosphenytoin in ML-II — ABSOLUTE CI).
  Typical antipsychotics HIGH RISK: basal ganglia lysosomal storage → severe EPS.
  General anesthesia EXTREME HAZARD: gingival hyperplasia (difficult airway) + cardiomyopathy
  + joint contractures; multi-specialist perioperative planning mandatory in ML-II.
"""
import random

GENE = "GNPTAB"
LOCUS = "12q23.2"
OMIM_GENE = "607840"
OMIM_DISEASE = "252500 (ML-II I-Cell Disease); 252600 (ML-IIIA/B Pseudo-Hurler)"
INHERITANCE = (
    "Autosomal Recessive (AR) — biallelic GNPTAB LOF; both males AND females equally affected; "
    "null alleles → ML-II severe; missense alleles → ML-IIIA/B attenuated"
)
COHORT_SIZE = 40
DISEASE_MECHANISM = (
    "GNPTAB (GlcNAc-1-phosphotransferase alpha/beta subunit) deficiency → failure to add "
    "mannose-6-phosphate (M6P) targeting tag to nascent lysosomal hydrolases in the cis-Golgi. "
    "M6P-TARGETING FAILURE: Without M6P tags, lysosomal enzymes cannot bind M6P receptors in the "
    "trans-Golgi network → are not sorted to lysosomes → are secreted into the extracellular space "
    "and PLASMA. Result: lysosomal hydrolases ACCUMULATE in PLASMA (10-50x elevated) while being "
    "LOW or ABSENT in LEUKOCYTES and lysosomes — the INVERSE PATTERN that is biochemically "
    "PATHOGNOMONIC for ML-II/III (opposite of most LSDs where leukocyte enzyme is low). "
    "CONSEQUENCE: without lysosomal hydrolases, undegraded glycoproteins, glycolipids, and GAGs "
    "accumulate in lysosomes → 'I-cell inclusions' (dense lysosomal aggregates visible under "
    "phase contrast microscopy in cultured fibroblasts → historical name 'I-Cell Disease'). "
    "DUAL PHENOTYPE: GNPTAB generates two distinct diseases depending on allele type: "
    "ML-II I-Cell Disease (null alleles → complete loss of phosphotransferase activity → severe, "
    "onset birth-6mo, gingival hyperplasia pathognomonic, periosteal new bone, cardiomyopathy 80%, "
    "epilepsy 40-65%, death age 5-8yr from cardiomyopathy + respiratory failure); "
    "ML-IIIA/B Pseudo-Hurler Polydystrophy (missense alleles → partial residual activity → "
    "attenuated, onset age 2-5yr, carpal tunnel syndrome 85% by age 10 as presenting symptom, "
    "joint stiffness, scoliosis, mild coarse facies, normal to mild ID, survival into adulthood). "
    "EPILEPSY MECHANISM: Progressive lysosomal storage in cortical neurons from undegraded "
    "substrates → neuronal dysfunction → cortical hyperexcitability; ML-II epilepsy 40-65% "
    "(infantile spasms predominant, GTCS, myoclonic), ML-IIIA 15-25% (lower, milder); "
    "DRE 25-35% in ML-II; infantile spasms (IS) managed with ACTH Level A (VGB ABSOLUTE CI "
    "in ML-II due to cardiomyopathy). "
    "CARDIAC DRIVER OF ALL DRUG DECISIONS: Cardiomyopathy/cardiomegaly in 80% of ML-II "
    "patients → PHT/fosphenytoin ABSOLUTE CI (QTc/PR prolongation), VGB ABSOLUTE CI (cardiac "
    "toxicity), CBZ/OXC CAUTION (PR prolongation), anesthesia EXTREME HAZARD; this cardiac "
    "burden makes ML-II the most drug-constrained epilepsy in the mucolipidosis group. "
    "NO CORNEAL CLOUDING: distinguishes ML-II/III from MPS-I Hurler (universal clouding), "
    "MPS-VI (universal), MPS-VII Sly (present), and ML-IV MCOLN1 (present). "
    "NO ERT APPROVED (2026): ERT complicated by the phosphotransferase being the defective "
    "enzyme itself (cannot use standard M6P-targeted ERT to replace GNPTAB); NO HSCT evidence "
    "in ML-II (unlike MPS-I Hurler where HSCT before age 2.5yr is Level A)."
)

ETIOLOGIES = [
    {
        "name": "GNPTAB Null/Null — ML-II Severe Classic",
        "pct": 25,
        "n": 10,
        "ml_type": "ML-II",
        "seizure_risk": (
            "40-65% (IS dominant; GTCS; myoclonic; early onset; DRE 25-35%; "
            "cardiomyopathy drives all AED decisions; PHT/VGB ABSOLUTE CI)"
        ),
        "eeg": (
            "Hypsarrhythmia in IS subset (chaotic high-amplitude multi-focal spikes + slow-waves); "
            "high-amplitude multifocal discharges; generalized slow spike-wave in post-IS stage; "
            "polyspike-wave in myoclonic subset (20-30%); background severely disorganized; "
            "burst-suppression pattern in neonatal seizures; EEG mandatory before all AED starts; "
            "progressive background deterioration paralleling neurological decline; "
            "IS typically onset age 3-9 months; ACTH response monitoring by EEG (hypsarrhythmia "
            "resolution = treatment success marker); "
            "avoid fosphenytoin loading for acute SE — ABSOLUTE CI in ML-II; IV LEV instead"
        ),
        "variant_detail": (
            "Biallelic null alleles (nonsense + frameshift + large deletion): complete absence of "
            "GlcNAc-1-phosphotransferase alpha/beta activity (<1% control); plasma lysosomal enzymes "
            "10-50x elevated (HYAL1, HEX-B, beta-gal, arylsulfatase); leukocyte enzymes absent or "
            "<5% control — INVERSE PATTERN pathognomonic; gingival hyperplasia visible from birth "
            "(thick fibrotic hyperplastic gums — pathognomonic for ML-II); coarse facies severe from "
            "birth; joint contractures; periosteal new bone on X-ray from infancy; "
            "cardiomegaly/cardiomyopathy 80-90% (major mortality contributor + drug CI driver); "
            "I-cell inclusions in fibroblasts (phase contrast microscopy); "
            "infantile spasms 40-55%; GTCS, myoclonic; DRE 25-35%; "
            "death age 5-8yr from cardiomyopathy + pneumonia/respiratory failure; "
            "NO corneal clouding; NO ERT; NO HSCT evidence"
        ),
        "hsct_eligible": False,
        "ert_eligible": False,
    },
    {
        "name": "GNPTAB Null/Missense — ML-II Intermediate",
        "pct": 30,
        "n": 12,
        "ml_type": "ML-II",
        "seizure_risk": (
            "35-55% (IS + GTCS predominant; variable severity; cardiomyopathy present "
            "in 70-80%; PHT/VGB still ABSOLUTE CI; LEV + ACTH backbone)"
        ),
        "eeg": (
            "Hypsarrhythmia in IS subset; generalized slow spike-wave; "
            "multifocal discharges; background slower than null-null but severely abnormal; "
            "IS onset typically 4-10 months; EEG-guided ACTH response tracking; "
            "myoclonic component in 20-25%; polyspike-wave in myoclonic subset; "
            "background disorganization moderate-to-severe; "
            "progressive deterioration with disease course; EEG mandatory before all AEDs"
        ),
        "variant_detail": (
            "Compound-het: one null allele (nonsense/frameshift) + one missense allele with "
            "severely reduced but not absent phosphotransferase activity; "
            "plasma lysosomal enzymes 8-40x elevated; leukocyte enzymes 3-10% control (very low); "
            "INVERSE PATTERN maintained (elevated plasma, low leukocyte); "
            "ML-II phenotype expressed (null allele dominates); variable severity range; "
            "gingival hyperplasia present (less florid than null-null in some); "
            "cardiomyopathy 70-80%; joint contractures; I-cell inclusions in fibroblasts; "
            "IS 35-45%; GTCS; DRE 20-30%; survival somewhat longer than null-null (age 6-12yr); "
            "NO corneal clouding; NO ERT approved"
        ),
        "hsct_eligible": False,
        "ert_eligible": False,
    },
    {
        "name": "GNPTAB Biallelic Missense Severe — ML-IIIA Classic",
        "pct": 20,
        "n": 8,
        "ml_type": "ML-IIIA",
        "seizure_risk": (
            "15-25% (GTCS + focal dominant; no IS; carpal tunnel 85% by age 10; "
            "mild-moderate ID; LEV first-line; POLG1 mandatory before VPA)"
        ),
        "eeg": (
            "Near-normal to mildly abnormal background; focal temporal or frontal discharges in some; "
            "generalized slow spike-wave in GTCS subset; "
            "absence of hypsarrhythmia (ML-IIIA does not present with IS); "
            "EEG mandated before AED prescription; "
            "background preserved better than ML-II null cases; "
            "seizures may be first neurological presentation (CTS is often the first clinical sign); "
            "focal onset secondary generalization pattern common; "
            "myoclonic component rare in ML-IIIA (<10%)"
        ),
        "variant_detail": (
            "Biallelic missense alleles with moderate-significant functional loss: "
            "partial residual phosphotransferase activity (5-25% control); "
            "plasma lysosomal enzymes elevated (3-15x) but less dramatically than ML-II; "
            "leukocyte enzymes low (10-30% control); INVERSE PATTERN maintained; "
            "ML-IIIA phenotype: onset age 2-5yr; joint stiffness ALL joints; "
            "carpal tunnel syndrome 85% by age 10 — often FIRST clinical sign leading to diagnosis; "
            "scoliosis, mild coarse facies; intelligence normal to mildly impaired; "
            "gingival hyperplasia absent or minimal (unlike ML-II); "
            "cardiomyopathy less prominent than ML-II (15-25%); "
            "survival may reach adulthood (20-40yr); "
            "NO corneal clouding; NO ERT; carpal tunnel release Level A"
        ),
        "hsct_eligible": False,
        "ert_eligible": False,
    },
    {
        "name": "GNPTAB Biallelic Missense Attenuated — ML-IIIB",
        "pct": 15,
        "n": 6,
        "ml_type": "ML-IIIB",
        "seizure_risk": (
            "10-20% (rare; late onset GTCS; normal intelligence; "
            "carpal tunnel + severe joint disease dominant presentation; "
            "lacosamide or LEV; PHT CAUTION — not absolute CI in ML-IIIB)"
        ),
        "eeg": (
            "Near-normal or minimal background slowing; focal discharges in seizure subset; "
            "EEG may be essentially normal in non-seizure subset; "
            "first seizure EEG may be near-normal; "
            "absence of hypsarrhythmia, absence of polyspike-wave; "
            "EEG abnormality correlates with joint/skeletal disease severity rather than "
            "neurological progression (given normal intelligence in most); "
            "epilepsy if present: focal > GTCS; no IS; no myoclonic component"
        ),
        "variant_detail": (
            "Biallelic hypomorphic missense alleles: high residual phosphotransferase activity "
            "(25-45% control); plasma lysosomal enzymes mildly elevated (2-5x control); "
            "leukocyte enzymes mildly reduced (30-50% control); INVERSE PATTERN present but subtle; "
            "ML-IIIB phenotype: attenuated, dominant skeletal/joint disease; "
            "carpal tunnel syndrome 75-85%; severe joint stiffness; scoliosis; "
            "NORMAL intelligence (key distinguisher from ML-II, ML-IIIA); "
            "survival into adulthood (30-50yr with supportive care); "
            "coarse facies mild; cardiomyopathy rare (<10%); "
            "epilepsy rare (10-20%); late onset GTCS if present; "
            "carpal tunnel release Level A; orthopedic management dominant"
        ),
        "hsct_eligible": False,
        "ert_eligible": False,
    },
    {
        "name": "GNPTAB Deep Intronic / Splice — Variable ML-II/III",
        "pct": 10,
        "n": 4,
        "ml_type": "Variable",
        "seizure_risk": (
            "25-50% (variable; depends on splice consequence — severe splice → ML-II IS; "
            "partial splice → ML-IIIA phenotype; genotype-phenotype correlation via RNA study)"
        ),
        "eeg": (
            "Variable: severe splice consequence → hypsarrhythmia (ML-II-like); "
            "partial splice → focal/GTCS pattern (ML-IIIA-like); "
            "EEG phenotype predicts underlying splice severity before RNA study returns; "
            "myoclonic component in severe splice subset; "
            "background slowing severity parallels clinical ML-II vs ML-IIIA determination; "
            "RNA analysis (RT-PCR of GNPTAB transcript) required to quantify splice impact"
        ),
        "variant_detail": (
            "Deep intronic or consensus splice-site variants in GNPTAB; phenotype depends on "
            "degree of exon skipping / intron retention / cryptic splice site activation; "
            "cryptic splice activation may produce partial residual GNPTAB transcript → variable "
            "residual phosphotransferase activity; severe splice (functional null) → ML-II phenotype "
            "with IS, cardiomyopathy, early death; partial splice (residual activity) → ML-IIIA "
            "phenotype with carpal tunnel, joint stiffness, better prognosis; "
            "RNA analysis (RT-PCR, long-read sequencing) mandatory to quantify splice consequence; "
            "plasma + leukocyte enzyme panel confirms inverse pattern regardless of splice severity; "
            "management depends on which phenotype is expressed; "
            "genetic counseling complex given variable expressivity"
        ),
        "hsct_eligible": False,
        "ert_eligible": False,
    },
]

SEIZURE_TYPES = [
    {
        "type": "Infantile Spasms (IS)",
        "pct": 55,
        "note": (
            "ML-II dominant seizure type (55% of ML-II epilepsy cases); onset age 3-12 months; "
            "hypsarrhythmia on EEG; ACTH Level A PREFERRED over VGB (VGB ABSOLUTE CI in ML-II "
            "due to cardiomyopathy + cardiac toxicity + visual monitoring impossible in severe ID); "
            "ACTH 150 IU/m2/day × 2 weeks then taper; EEG response monitoring mandatory; "
            "VPA second-line if ACTH fails (POLG1 mandatory first); NOT VGB in ML-II IS"
        ),
    },
    {
        "type": "Generalized Tonic-Clonic Seizures (GTCS)",
        "pct": 45,
        "note": (
            "Across ML-II and ML-IIIA; LEV Level B first-line (safest cardiac profile); "
            "VPA Level B broad-spectrum (POLG1 mandatory); CBZ/OXC CAUTION in ML-II "
            "(PR prolongation risk in cardiomyopathy); PHT ABSOLUTE CI in ML-II; "
            "lacosamide preferred over CBZ for focal-onset-GTCS in ML-II cardiac patients"
        ),
    },
    {
        "type": "Myoclonic Seizures",
        "pct": 30,
        "note": (
            "ML-II subset (30%); VPA + CLZ adjunct; CBZ/OXC CAUTION (may worsen myoclonus + "
            "cardiac concern); PHT ABSOLUTE CI in ML-II; IV LEV for acute myoclonic SE; "
            "EEG polyspike-wave confirmation before any sodium-channel AED"
        ),
    },
    {
        "type": "Focal Onset Seizures",
        "pct": 20,
        "note": (
            "Both ML-II and ML-IIIA; lacosamide PREFERRED (safest sodium-channel blocker for "
            "ML-II cardiomyopathy vs CBZ/OXC); CBZ/OXC acceptable in ML-IIIB (no cardiomyopathy); "
            "secondary generalization common; LEV as adjunct"
        ),
    },
    {
        "type": "Status Epilepticus (SE)",
        "pct": 15,
        "note": (
            "Febrile provoked in ML-II (respiratory infection → fever → SE cascade); "
            "IV LEV MANDATORY for SE management; fosphenytoin ABSOLUTE CI in ML-II "
            "(cardiac: QTc + PR prolongation in cardiomyopathy → lethal arrhythmia); "
            "rescue benzodiazepine (diazepam rectal or midazolam buccal) as first-response; "
            "ICU cardiac monitoring mandatory during SE in ML-II"
        ),
    },
]

TRIGGERS = [
    {
        "trigger": "Febrile Illness / Respiratory Infection",
        "pct": 70,
        "note": (
            "DOMINANT TRIGGER in ML-II: respiratory compromise (weak chest muscles, cardiomyopathy, "
            "recurrent pneumonia) → fever → seizure threshold drop → status epilepticus cascade; "
            "aggressive antipyretics; rescue BDZ protocol; ICU-level monitoring during febrile "
            "illness in ML-II; respiratory infection management = seizure prevention"
        ),
    },
    {
        "trigger": "Sleep Deprivation",
        "pct": 45,
        "note": (
            "Disrupted sleep in ML-II (nocturnal seizures, nocturnal respiratory distress, "
            "caregiver exhaustion); melatonin Level B for sleep-seizure nexus; "
            "sleep study if OSA suspected (cardiomyopathy may have associated OSA)"
        ),
    },
    {
        "trigger": "Missed AED Dose",
        "pct": 40,
        "note": (
            "Caregiver-administered AEDs; joint contractures (ML-II) complicate oral dosing; "
            "liquid formulations preferred; nasogastric if severe swallowing difficulty; "
            "IV LEV bridge formulation if NPO during hospitalizations"
        ),
    },
    {
        "trigger": "Respiratory Infection / Pneumonia",
        "pct": 55,
        "note": (
            "ML-II specific: recurrent pneumonia from aspiration (gingival hyperplasia → "
            "dysphagia → aspiration) + cardiomyopathy → combined cardiorespiratory compromise "
            "→ fever → seizure precipitation; influenza and RSV vaccination mandatory; "
            "chest physiotherapy protocol"
        ),
    },
    {
        "trigger": "Physiological Stress / Surgical Procedures",
        "pct": 35,
        "note": (
            "GNPTAB-SPECIFIC EXTREME HAZARD: General anesthesia in ML-II — difficult airway "
            "(gingival hyperplasia + joint contractures of TMJ + short neck), cardiomyopathy "
            "(anesthetic agents reduce cardiac output), joint contractures (positioning challenges); "
            "any surgical procedure (carpal tunnel release in ML-IIIA, orthopedic in ML-IIIB) "
            "requires multi-specialist perioperative team: metabolic disease, cardiac, anesthesia"
        ),
    },
    {
        "trigger": "Photic Stimulation",
        "pct": 20,
        "note": (
            "Photoparoxysmal response in minority of ML-II cases; photosensitivity screen "
            "on EEG; avoid strobe environments; less prominent trigger compared to respiratory/febrile"
        ),
    },
]

TREATMENTS = [
    {
        "drug": "Levetiracetam (LEV)",
        "level": "Level B",
        "indication": (
            "FIRST-LINE broad-spectrum: GTCS + focal + myoclonic + SE; SV2A modulation; "
            "1000-3000mg/day (weight-based in infants); IV formulation MANDATORY for SE "
            "management (replaces fosphenytoin which is ABSOLUTE CI in ML-II); "
            "safest cardiac profile among AEDs — NO PR/QTc prolongation; "
            "critical advantage in ML-II cardiomyopathy; B6 supplement if behavioral SE develops; "
            "preferred AED in ML-II across all seizure types; IV bridge during NPO/hospitalization"
        ),
        "ci": (
            "Behavioral worsening (irritability) — monitor in ML-II (pre-existing behavioral burden); "
            "B6 pyridoxine supplement if LEV-induced behavioral SE develops; "
            "dose-adjust for renal impairment"
        ),
    },
    {
        "drug": "ACTH (Adrenocorticotropic Hormone)",
        "level": "Level A",
        "indication": (
            "INFANTILE SPASMS in ML-II — PREFERRED over VGB (VGB ABSOLUTE CI in ML-II "
            "due to cardiomyopathy); ACTH 150 IU/m2/day IM × 2 weeks then taper; "
            "EEG monitoring: hypsarrhythmia resolution = treatment success; "
            "highest evidence level (Level A) for IS management; "
            "ACTH vs VGB: ACTH is the ONLY viable IS treatment in ML-II (VGB forbidden — "
            "cardiac toxicity + visual monitoring impossible + cardiomyopathy compound risk)"
        ),
        "ci": (
            "Hypertension monitoring (ACTH → cortisol → BP elevation); "
            "blood glucose monitoring (glucose elevation); "
            "infection risk (immunosuppression); "
            "cardiac monitoring (blood pressure + fluid retention in cardiomyopathy)"
        ),
    },
    {
        "drug": "Valproate / VPA (Depakote / Epilim)",
        "level": "Level B",
        "indication": (
            "Broad-spectrum: GTCS + myoclonic + IS (if ACTH fails); 500-2500mg/day; "
            "MANDATORY POLG1 exclusion BEFORE prescribing (CPIC Grade A); "
            "second-line for IS after ACTH failure; "
            "VPA + LEV combination for myoclonic-GTCS mixed phenotype; "
            "CAUTION in ML-II cardiac compromise — hepatic/metabolic monitoring intensive"
        ),
        "ci": (
            "POLG1/POLG2 carriers: ABSOLUTE CI (Alpers-Huttenlocher; fulminant hepatic failure); "
            "CAUTION in ML-II cardiac compromise (VPA-associated carnitine depletion); "
            "pregnancy (teratogenic); pancreatitis; weight gain; "
            "ML-II hepatomegaly (if present) → closer LFT monitoring"
        ),
    },
    {
        "drug": "Clonazepam (CLZ)",
        "level": "Level B",
        "indication": (
            "Adjunct for myoclonic component; 0.5-2mg TID; GABAergic modulation; "
            "useful combination with LEV or VPA for myoclonus burden in ML-II; "
            "rescue benzodiazepine protocol for SE: diazepam rectal / midazolam buccal "
            "(gingival hyperplasia may complicate buccal administration — nasal route preferred)"
        ),
        "ci": (
            "Sedation (adds to respiratory compromise in ML-II — careful titration); "
            "tolerance; withdrawal rebound; hypersalivation (+ gingival hyperplasia → aspiration); "
            "never abrupt discontinuation"
        ),
    },
    {
        "drug": "Lacosamide (Vimpat)",
        "level": "Level C",
        "indication": (
            "Focal seizures in ML-II/III — PREFERRED over CBZ/OXC in ML-II (safer cardiac "
            "profile; no PR prolongation concern vs CBZ/OXC); 200-400mg/day; IV formulation; "
            "slow-inactivation sodium-channel modulator; does not worsen myoclonus; "
            "useful when CBZ causes hyponatremia or EEG shows borderline myoclonus; "
            "advantageous safety profile in ML-II cardiomyopathy"
        ),
        "ci": (
            "PR interval prolongation (less than CBZ but still ECG monitoring in ML-II); "
            "dizziness; not first-line; drug interaction monitoring"
        ),
    },
    {
        "drug": "Carbamazepine (CBZ) / Oxcarbazepine (OXC)",
        "level": "Level B",
        "indication": (
            "Focal seizures in ML-IIIA/B (NO cardiomyopathy or mild only); "
            "CBZ 400-1600mg/day; OXC 600-2400mg/day; "
            "CAUTION in ML-II with cardiomyopathy (PR prolongation risk); "
            "acceptable in ML-IIIB (normal cardiac function); "
            "HLA-B1502 test before CBZ in SE-Asian patients; "
            "EEG mandatory — AVOID if myoclonic component present"
        ),
        "ci": (
            "CAUTION/RELATIVE CI in ML-II cardiomyopathy (PR prolongation → arrhythmia); "
            "RELATIVE CI if myoclonic component on EEG (worsens myoclonus); "
            "hyponatremia (OXC); HLA-B1502 SJS risk; "
            "lacosamide preferred when cardiac concern present"
        ),
    },
    {
        "drug": "Carpal Tunnel Release (Surgical — ML-IIIA/B)",
        "level": "Level A",
        "indication": (
            "ML-IIIA/B presenting symptom management: bilateral carpal tunnel syndrome in 85% "
            "by age 10; surgical release Level A; often FIRST therapeutic intervention in ML-IIIA; "
            "early release prevents median nerve damage and functional loss; "
            "multi-specialist perioperative: metabolic + cardiac + anesthesia teams mandatory"
        ),
        "ci": (
            "General anesthesia extreme hazard (see contraindications); "
            "multi-specialist perioperative planning mandatory; "
            "monitor for post-surgical seizure precipitation (physiological stress trigger)"
        ),
    },
]

CONTRAINDICATIONS = [
    {
        "drug": "Fosphenytoin / Phenytoin (PHT) — IV/Oral",
        "severity": "ABSOLUTE CI in ML-II (cardiac: QTc/PR prolongation + cardiomyopathy)",
        "reason": (
            "ABSOLUTE CI in ML-II: cardiomyopathy (80% of ML-II) + PHT/fosphenytoin → "
            "QTc prolongation + PR prolongation → lethal ventricular arrhythmia; "
            "IV fosphenytoin (standard SE drug) is FORBIDDEN in ML-II SE → use IV LEV instead; "
            "oral PHT for chronic therapy: ABSOLUTE CI in ML-II; "
            "DISTINCTION from MANBA (where PHT is RELATIVE CI for myoclonus only): "
            "in ML-II the cardiac cardiomyopathy makes this ABSOLUTE, not relative; "
            "ML-IIIB (no cardiomyopathy): PHT CAUTION but not absolute CI"
        ),
    },
    {
        "drug": "Vigabatrin (VGB) — for Infantile Spasms",
        "severity": "ABSOLUTE CI in ML-II IS (cardiac toxicity + visual monitoring impossible)",
        "reason": (
            "VGB ABSOLUTE CI in ML-II IS management: cardiac toxicity (VGB associated with "
            "cardiac conduction changes) compounds cardiomyopathy risk; visual field testing "
            "is IMPOSSIBLE in ML-II severe ID + cardiomegaly + contractures; "
            "ACTH is the ONLY viable IS treatment in ML-II; "
            "VGB RELATIVE CI in ML-IIIA/B (attenuated cardiac risk but visual monitoring "
            "still difficult in mild-moderate ID); "
            "do NOT use VGB for IS in ML-II regardless of EEG response — ACTH mandatory"
        ),
    },
    {
        "drug": "CBZ / OXC — in ML-II Cardiomyopathy",
        "severity": "CAUTION / RELATIVE CI in ML-II (cardiac arrhythmia, PR prolongation)",
        "reason": (
            "CBZ/OXC cause PR prolongation → arrhythmia risk in ML-II cardiomyopathy (80%); "
            "safe in ML-IIIB (no cardiomyopathy); "
            "lacosamide preferred for focal seizures in ML-II (safer cardiac profile); "
            "if CBZ used in ML-II: continuous cardiac monitoring, baseline ECG, "
            "cardiology clearance; same RELATIVE CI principle as cardiac LSDs generally; "
            "myoclonic component (if present): additional RELATIVE CI (worsens myoclonus)"
        ),
    },
    {
        "drug": "POLG1 Carriers — Valproate",
        "severity": "ABSOLUTE CI (CPIC Grade A)",
        "reason": (
            "MANDATORY POLG1/POLG2 sequencing before VPA in ALL GNPTAB patients; "
            "POLG1 carriers → VPA → Alpers-Huttenlocher (acute hepatic failure + neurological "
            "deterioration + death); CPIC Grade A evidence; "
            "if POLG1+ → LEV backbone; CLZ adjunct for myoclonus; lacosamide for focal; "
            "do NOT delay POLG1 testing — order simultaneously with enzyme panel"
        ),
    },
    {
        "drug": "Typical Antipsychotics (Haloperidol, Chlorpromazine, Fluphenazine)",
        "severity": "HIGH RISK (EPS from basal ganglia lysosomal storage)",
        "reason": (
            "Typical APs prescribed for behavioral features in ML-II (aggression, hyperactivity) → "
            "severe EPS due to lysosomal storage in basal ganglia (globus pallidus, substantia nigra "
            "→ D2 hypersensitivity → acute dystonic reactions, Parkinsonism, tardive dyskinesia); "
            "ALSO lowers seizure threshold; use ATYPICAL AP (quetiapine preferred — lowest EPS risk); "
            "same principle as MANBA and AGA behavioral phenotype trap"
        ),
    },
    {
        "drug": "General Anesthesia — Unplanned / Non-Specialist",
        "severity": "EXTREME HAZARD in ML-II (multi-system hazard)",
        "reason": (
            "ML-II EXTREME HAZARD: gingival hyperplasia (thick hyperplastic gums → difficult "
            "laryngoscopy + intubation; ETT sizing challenging); joint contractures (TMJ, cervical "
            "spine → limited neck extension → difficult airway positioning); "
            "cardiomyopathy (volatile anesthetics depress myocardium → cardiac arrest); "
            "perioperative team MANDATORY: metabolic/LSD specialist + pediatric cardiology + "
            "experienced pediatric anesthesia + ICU post-op; "
            "ML-IIIA/B: lower risk (less cardiomyopathy) but still CTS surgery requires specialized team"
        ),
    },
]


def get_overview():
    random.seed(42)
    return {
        "gene": GENE,
        "locus": LOCUS,
        "omim_gene": OMIM_GENE,
        "omim_disease": OMIM_DISEASE,
        "inheritance": INHERITANCE,
        "cohort_size": COHORT_SIZE,
        "disease_mechanism": DISEASE_MECHANISM,
        "color_scheme": {
            "ACCENT": "#1a237e",   # deep indigo — ML-II severe, I-cell disease
            "ACCENT2": "#b71c1c",  # dark red — ABSOLUTE CI, cardiac risk, gingival hyperplasia
            "ACCENT3": "#e65100",  # deep orange — RELATIVE CI, CAUTION, cardiac monitoring
            "ACCENT4": "#4a148c",  # deep purple — no ERT, inverse enzyme pattern, M6P
            "ACCENT5": "#006064",  # teal — ML-IIIA carpal tunnel, joint stiffness, attenuated
            "ACCENT6": "#1565c0",  # blue — LEV first-line, ACTH IS, safe treatments
        },
        "kpis": {
            "epilepsy_prevalence_ml2": "40-65% (ML-II I-Cell Disease; IS dominant; GTCS; myoclonic)",
            "epilepsy_prevalence_ml3": "15-25% (ML-IIIA Pseudo-Hurler; lower; GTCS + focal)",
            "drug_resistance_rate": "25-35% (DRE in ML-II; lower in ML-IIIA)",
            "inverse_enzyme_pattern": "PATHOGNOMONIC: plasma enzymes 10-50x ELEVATED; leukocyte enzymes LOW",
            "gingival_hyperplasia": "Pathognomonic ML-II: thick fibrotic hyperplastic gums FROM BIRTH",
            "no_corneal_clouding": "NO corneal clouding — distinguishes from MPS-I, MPS-VI, MPS-VII, ML-IV",
            "cardiomyopathy_ml2": "Cardiomyopathy/cardiomegaly 80% ML-II — major drug CI driver",
            "carpal_tunnel_ml3": "Carpal tunnel syndrome 85% by age 10 in ML-IIIA — first presenting sign",
            "i_cell_inclusions": "I-cell inclusions in fibroblasts (phase contrast) — historical diagnostic",
            "infantile_spasms": "IS 40-55% of ML-II epilepsy; ACTH Level A (VGB ABSOLUTE CI in ML-II)",
            "pht_status_ml2": "PHT/Fosphenytoin ABSOLUTE CI in ML-II (cardiac: QTc/PR + cardiomyopathy)",
            "vgb_status_ml2": "VGB ABSOLUTE CI in ML-II IS (cardiac toxicity + cardiomyopathy)",
            "ert_approved": "None approved (2026) — GNPTAB enzyme replacement technically constrained",
            "hsct_evidence": "None in ML-II (unlike MPS-I Hurler HSCT Level A before age 2.5yr)",
            "polg1_mandatory": "MANDATORY before VPA (CPIC Grade A)",
            "anesthesia_hazard": "EXTREME HAZARD in ML-II (gingival hyperplasia airway + cardiomyopathy)",
            "dual_disease_gene": "One GNPTAB gene → two diseases: ML-II (null) and ML-IIIA/B (missense)",
            "acth_vs_vgb": "ACTH Level A for IS in ML-II — only viable option (VGB forbidden)",
            "lacosamide_preferred": "Lacosamide preferred over CBZ/OXC for focal in ML-II (cardiac safety)",
            "periosteal_bone": "Periosteal new bone formation on X-ray — ML-II specific from infancy",
        },
        "etiologies": ETIOLOGIES,
    }


def get_breakdown():
    random.seed(42)
    patients = []

    # Seizure type pools by phenotype
    ml2_seizure_types = [
        "Infantile Spasms (IS)",
        "IS + GTCS",
        "GTCS",
        "GTCS + Myoclonic",
        "Myoclonic + GTCS",
        "Status Epilepticus",
        "Focal + GTCS",
    ]
    ml3_seizure_types = [
        "GTCS",
        "Focal Onset",
        "Focal + GTCS",
        "GTCS",
        "Focal Onset",
    ]
    mlvar_seizure_types = [
        "IS + GTCS",
        "GTCS",
        "Focal + GTCS",
        "Myoclonic + GTCS",
    ]

    treatment_map = {
        "ML-II": ["Levetiracetam", "ACTH", "Valproate", "Clonazepam", "Lacosamide"],
        "ML-IIIA": ["Levetiracetam", "Valproate", "Carbamazepine", "Lacosamide"],
        "ML-IIIB": ["Levetiracetam", "Lacosamide", "Oxcarbazepine", "Valproate"],
        "Variable": ["Levetiracetam", "ACTH", "Valproate", "Lacosamide"],
    }

    for eth in ETIOLOGIES:
        n = eth["n"]
        ml_type = eth["ml_type"]

        for i in range(n):
            is_ml2 = ml_type == "ML-II"
            is_ml3a = ml_type == "ML-IIIA"
            is_ml3b = ml_type == "ML-IIIB"
            is_var = ml_type == "Variable"
            is_null_null = "Null/Null" in eth["name"]

            # Age of onset (decimal for months in ML-II IS)
            if is_ml2:
                # Mix of IS (0.3-0.9yr) and later GTCS (1-3yr)
                age_onset = round(random.uniform(0.3, 2.5), 1)
            elif is_ml3b:
                age_onset = round(random.uniform(4, 15), 1)
            elif is_ml3a:
                age_onset = round(random.uniform(2, 8), 1)
            else:
                # Variable splice — could be ML-II-like or ML-IIIA-like
                age_onset = round(random.uniform(0.5, 6), 1)

            # Seizure type selection
            if is_ml2:
                sz_type = random.choice(ml2_seizure_types)
            elif is_var:
                sz_type = random.choice(mlvar_seizure_types)
            else:
                sz_type = random.choice(ml3_seizure_types)

            # Gingival hyperplasia: pathognomonic in ML-II, absent/minimal in ML-IIIA/B
            gingival_hyperplasia = (
                (random.random() < 0.92) if is_ml2
                else (random.random() < 0.20) if is_ml3a
                else (random.random() < 0.05) if is_ml3b
                else (random.random() < 0.55)  # variable
            )

            # Carpal tunnel: dominant in ML-IIIA, present in ML-IIIB, rare in ML-II
            carpal_tunnel = (
                (random.random() < 0.10) if is_ml2
                else (random.random() < 0.88) if is_ml3a
                else (random.random() < 0.82) if is_ml3b
                else (random.random() < 0.35)  # variable
            )

            # Cardiomyopathy: 80% in ML-II, lower in ML-IIIA/B
            cardiomyopathy = (
                (random.random() < 0.82) if is_ml2
                else (random.random() < 0.22) if is_ml3a
                else (random.random() < 0.08) if is_ml3b
                else (random.random() < 0.52)  # variable
            )

            # Treatment selection
            t_pool = treatment_map.get(ml_type, treatment_map["Variable"])
            treatment_1 = t_pool[0]  # LEV always first-line
            treatment_2 = random.choice(t_pool[1:]) if len(t_pool) > 1 else "Clonazepam"

            # CI avoided — most critical for ML-II
            if is_ml2:
                ci_avoided = random.choice([
                    "Fosphenytoin/PHT (ABSOLUTE CI — cardiac QTc/PR + cardiomyopathy)",
                    "VGB (ABSOLUTE CI — cardiac toxicity + cardiomyopathy + IS)",
                    "CBZ/OXC (CAUTION — PR prolongation in cardiomyopathy)",
                    "Typical antipsychotics (HIGH RISK — EPS basal ganglia storage)",
                    "General anesthesia without specialist team (EXTREME HAZARD)",
                ])
            elif is_ml3a or is_ml3b:
                ci_avoided = random.choice([
                    "POLG1/VPA (ABSOLUTE CI — CPIC Grade A)",
                    "CBZ/OXC with cardiomyopathy monitoring",
                    "Typical antipsychotics (HIGH RISK — EPS)",
                    "VGB (RELATIVE CI — visual monitoring difficult)",
                    "General anesthesia without metabolic team",
                ])
            else:
                ci_avoided = random.choice([
                    "Fosphenytoin/PHT (ABSOLUTE CI if ML-II phenotype)",
                    "VGB (ABSOLUTE CI if IS + ML-II cardiomyopathy)",
                    "POLG1/VPA (ABSOLUTE CI — CPIC Grade A)",
                    "Typical antipsychotics (HIGH RISK — EPS)",
                ])

            pt = {
                "id": f"GNPTAB-{len(patients)+1:03d}",
                "etiology": eth["name"],
                "age_onset_yr": age_onset,
                "seizure_type": sz_type,
                "ml_type": ml_type,
                "gingival_hyperplasia": gingival_hyperplasia,
                "carpal_tunnel": carpal_tunnel,
                "cardiomyopathy": cardiomyopathy,
                "treatment_1": treatment_1,
                "treatment_2": treatment_2,
                "ci_avoided": ci_avoided,
                "plasma_enzymes": "ELEVATED (10-50x control — M6P-targeting failure)",
                "leukocyte_enzymes": "LOW — inverse pattern (PATHOGNOMONIC ML-II/III)",
                "trigger": random.choice([t["trigger"] for t in TRIGGERS]),
                "eeg_pattern": eth["eeg"][:100] + "…",
                "i_cell_inclusions": is_ml2 or (is_var and random.random() < 0.55),
                "periosteal_new_bone": is_ml2 and random.random() < 0.85,
                "corneal_clouding": False,   # ML-II/III: NO corneal clouding (key distinguisher)
                "polg1_tested": True,
                "hsct": False,
                "ert": False,
                "seizure_risk": eth["seizure_risk"],
            }
            patients.append(pt)

    return {
        "cohort_size": COHORT_SIZE,
        "patients": patients,
        "seizure_summary": SEIZURE_TYPES,
        "trigger_summary": TRIGGERS,
        "treatment_summary": TREATMENTS,
        "contraindication_summary": CONTRAINDICATIONS,
        "etiology_breakdown": [
            {"name": e["name"], "pct": e["pct"], "n": e["n"], "ml_type": e["ml_type"]}
            for e in ETIOLOGIES
        ],
    }


def get_definitions():
    return {
        "glossary": [
            {
                "term": "GNPTAB (GlcNAc-1-Phosphotransferase Alpha/Beta Subunit)",
                "definition": (
                    "Encodes the alpha and beta catalytic subunits of UDP-GlcNAc:lysosomal enzyme "
                    "GlcNAc-1-phosphotransferase (hexameric complex: alpha2-beta2-gamma2); "
                    "located at 12q23.2; OMIM 607840; adds GlcNAc-1-phosphate to high-mannose "
                    "N-linked oligosaccharides on nascent lysosomal hydrolases in the cis-Golgi, "
                    "generating the mannose-6-phosphate (M6P) recognition marker; deficiency causes "
                    "ML-II (null alleles) or ML-IIIA/B (missense alleles); gamma subunit (GNPTG) "
                    "is a separate gene (16p13.3) — GNPTG mutations cause the milder ML-IIIC."
                ),
            },
            {
                "term": "ML-II I-Cell Disease (Mucolipidosis II)",
                "definition": (
                    "Severe lysosomal storage disorder caused by GNPTAB biallelic null alleles; "
                    "OMIM 252500; onset birth to 6 months; hallmarks: gingival hyperplasia "
                    "(PATHOGNOMONIC — thick fibrotic hyperplastic gums from birth), periosteal new "
                    "bone on X-ray, coarse facies, joint contractures, cardiomyopathy 80%; "
                    "I-cell inclusions in fibroblasts (phase contrast — gives disease its name); "
                    "epilepsy 40-65% (IS dominant; GTCS; myoclonic); death age 5-8yr; "
                    "NO corneal clouding (distinguishes from MPS-I, MPS-VI, MPS-VII, ML-IV); "
                    "NO ERT; NO HSCT evidence."
                ),
            },
            {
                "term": "ML-IIIA Pseudo-Hurler Polydystrophy (Mucolipidosis IIIA)",
                "definition": (
                    "Attenuated form caused by GNPTAB biallelic missense alleles; OMIM 252600; "
                    "onset age 2-5yr; hallmarks: joint stiffness (ALL joints), carpal tunnel "
                    "syndrome (85% by age 10 — often FIRST clinical sign), scoliosis, mild coarse "
                    "facies; intelligence normal to mildly impaired (UNLIKE ML-II severe ID); "
                    "survival into adulthood (20-40yr); epilepsy 15-25% (GTCS + focal); "
                    "gingival hyperplasia absent or minimal; cardiomyopathy 15-25% (much less than "
                    "ML-II 80%); carpal tunnel release Level A therapeutic intervention."
                ),
            },
            {
                "term": "M6P Targeting Failure — Inverse Plasma/Leukocyte Enzyme Pattern",
                "definition": (
                    "PATHOGNOMONIC biochemical hallmark of ML-II/III: GNPTAB deficiency → failure "
                    "to add M6P tags to lysosomal hydrolases → enzymes cannot bind M6P receptors "
                    "in trans-Golgi → secreted into PLASMA (10-50x elevated: HYAL1, HEX-B, "
                    "beta-galactosidase, arylsulfatase) while ABSENT or LOW in LEUKOCYTES "
                    "(<5-20% control); OPPOSITE of most LSDs (where leukocyte enzyme is low but "
                    "plasma is not elevated); diagnostic dual panel: plasma enzyme ELEVATED + "
                    "leukocyte enzyme LOW = INVERSE PATTERN = ML-II/III diagnosis."
                ),
            },
            {
                "term": "Gingival Hyperplasia (ML-II — Pathognomonic)",
                "definition": (
                    "Thick, fibrotic, hyperplastic gums from BIRTH in ML-II — one of the earliest "
                    "and most specific clinical signs; visible on first physical examination; "
                    "caused by glycoprotein/GAG accumulation in gingival fibroblasts; "
                    "PATHOGNOMONIC for ML-II when combined with coarse facies; "
                    "differentiates ML-II from MPS-I Hurler (gingival disease present but less "
                    "florid), MPS-III Sanfilippo (minimal gingival disease), ML-IIIA/B (absent); "
                    "also creates AIRWAY HAZARD: difficult laryngoscopy + intubation under anesthesia; "
                    "absent in ML-IIIA/B (key phenotype discriminator)."
                ),
            },
            {
                "term": "I-Cells (Inclusion Cells) — Phase Contrast Fibroblasts",
                "definition": (
                    "Dense lysosomal inclusions visible in cultured skin fibroblasts under phase "
                    "contrast microscopy; represent accumulated undigested glycoproteins, glycolipids, "
                    "and GAGs in lysosomes of cells that lack lysosomal hydrolases (secreted into "
                    "plasma due to M6P targeting failure); historical discovery preceded biochemical "
                    "characterization → named 'I-Cell Disease'; inclusions are autofluorescent under "
                    "UV; electron microscopy shows pleomorphic dense membrane-bound organelles; "
                    "skin fibroblast culture required (not seen in peripheral blood lymphocytes as "
                    "prominently as in cultured cells)."
                ),
            },
            {
                "term": "Carpal Tunnel Syndrome (ML-IIIA — Presenting Symptom)",
                "definition": (
                    "Bilateral carpal tunnel syndrome in 85% of ML-IIIA patients by age 10; "
                    "often the FIRST clinical sign leading to metabolic workup in ML-IIIA; "
                    "caused by glycoprotein accumulation in synovium and connective tissue of "
                    "the carpal tunnel → median nerve compression; children presenting with bilateral "
                    "unexplained CTS + joint stiffness → METABOLIC WORKUP mandatory; "
                    "carpal tunnel release (surgical) Level A — major therapeutic intervention; "
                    "CTS is MORE PROMINENT in ML-IIIA than in any MPS (where joint disease is "
                    "diffuse but CTS less specifically presenting)."
                ),
            },
            {
                "term": "ACTH Level A for Infantile Spasms (ML-II)",
                "definition": (
                    "ACTH is the Level A treatment of choice for infantile spasms in ML-II; "
                    "preferred OVER VGB (VGB ABSOLUTE CI in ML-II: cardiac toxicity + cardiomyopathy "
                    "+ visual monitoring impossible in severe ID); ACTH 150 IU/m2/day IM × 2 weeks; "
                    "EEG monitoring: hypsarrhythmia resolution = treatment success; "
                    "VPA second-line if ACTH fails (POLG1 mandatory); "
                    "this ML-II IS → ACTH rule is DISEASE-SPECIFIC and must NOT be overridden "
                    "by VGB even though VGB is standard IS therapy in non-cardiac-compromised infants."
                ),
            },
            {
                "term": "PHT/Fosphenytoin ABSOLUTE CI (ML-II Cardiac)",
                "definition": (
                    "Phenytoin and fosphenytoin (IV) are ABSOLUTE CI in ML-II: QTc prolongation + "
                    "PR prolongation in the context of cardiomyopathy (80% of ML-II) → ventricular "
                    "arrhythmia → cardiac arrest; IV fosphenytoin is the standard SE second-line drug "
                    "in general epilepsy → must NOT be used in ML-II SE → IV LEV replaces it; "
                    "DISTINCTION from MANBA (where PHT is RELATIVE CI for myoclonus only): "
                    "in ML-II the cardiomyopathy makes this ABSOLUTE, not relative; "
                    "ML-IIIB (no/minimal cardiomyopathy): PHT is CAUTION, not absolute CI."
                ),
            },
            {
                "term": "VGB ABSOLUTE CI in ML-II IS (Cardiac + Monitoring Failure)",
                "definition": (
                    "Vigabatrin is ABSOLUTE CI for infantile spasms in ML-II for compound reasons: "
                    "1) VGB cardiac toxicity (associated with cardiac conduction changes) compounds "
                    "cardiomyopathy → arrythmia risk; "
                    "2) VGB visual field loss monitoring requires reliable visual field perimetry → "
                    "IMPOSSIBLE in ML-II severe intellectual disability + visual assessment impossible; "
                    "3) Cardiomyopathy (80%) creates additional cardiac CI; "
                    "ACTH is the ONLY viable IS treatment in ML-II; "
                    "VGB RELATIVE CI in ML-IIIA/B (attenuated cardiac risk but still monitoring concern)."
                ),
            },
            {
                "term": "Anesthesia Extreme Hazard (ML-II)",
                "definition": (
                    "General anesthesia in ML-II carries EXTREME HAZARD from three simultaneous risks: "
                    "1) AIRWAY: gingival hyperplasia (thick fibrous gums) + joint contractures (TMJ, "
                    "cervical spine limited extension) → difficult laryngoscopy, difficult intubation, "
                    "risk of airway loss; "
                    "2) CARDIAC: cardiomyopathy (80%) → volatile anesthetic agents depress myocardium "
                    "→ cardiac arrest during induction; "
                    "3) POSITIONING: joint contractures throughout → positioning for surgery is "
                    "technically demanding; MANDATORY perioperative team: metabolic/LSD specialist, "
                    "pediatric cardiology, experienced pediatric/difficult-airway anesthesia, ICU post-op."
                ),
            },
            {
                "term": "POLG1 Mandatory (GNPTAB, CPIC Grade A)",
                "definition": (
                    "POLG1/POLG2 sequencing MANDATORY before valproate in all GNPTAB patients "
                    "(ML-II and ML-IIIA/B); POLG1 carriers → VPA → Alpers-Huttenlocher syndrome "
                    "(acute hepatic failure + progressive neurological deterioration + death); "
                    "CPIC Grade A evidence; order POLG1 simultaneously with diagnostic enzyme panel; "
                    "if POLG1+ → LEV as backbone; CLZ for myoclonus; lacosamide for focal seizures; "
                    "never delay POLG1 test — consider ML-II urgent given rapid disease progression."
                ),
            },
            {
                "term": "No ERT Approved (GNPTAB, 2026)",
                "definition": (
                    "Unlike MPS-I (laronidase), MPS-II (idursulfase), MPS-IVA (elosulfase alfa), "
                    "MPS-VI (galsulfase), MPS-VII (vestronidase alfa), no ERT is approved for ML-II "
                    "or ML-IIIA/B (2026); technical obstacle: the defective enzyme IS the "
                    "phosphotransferase needed to add M6P tags → standard M6P-targeted ERT "
                    "manufacturing cannot use GNPTAB's own delivery mechanism; "
                    "gene therapy and substrate reduction strategies under early research."
                ),
            },
            {
                "term": "Plasma Enzyme Panel Diagnosis (ML-II/III)",
                "definition": (
                    "Diagnostic gold-standard: DUAL panel approach: "
                    "1) PLASMA lysosomal enzyme panel: HYAL1, HEX-A, HEX-B, arylsulfatase A+B, "
                    "beta-galactosidase, beta-glucuronidase → MARKEDLY ELEVATED (10-50x normal); "
                    "2) LEUKOCYTE lysosomal enzyme panel: same enzymes → LOW or absent (<5-20%); "
                    "INVERSE PATTERN (plasma elevated, leukocyte low) = PATHOGNOMONIC for ML-II/III; "
                    "plasma HEX-B elevation is particularly sensitive; "
                    "confirmatory: GNPTAB gene sequencing identifies biallelic pathogenic variants; "
                    "GNPTAB sequencing also determines allele type (null vs missense) → predicts "
                    "ML-II vs ML-IIIA/B phenotype."
                ),
            },
            {
                "term": "No Corneal Clouding (ML-II/III — Key Differential)",
                "definition": (
                    "ML-II I-Cell Disease and ML-IIIA/B do NOT have corneal clouding; "
                    "this ABSENCE is a critical diagnostic discriminator: "
                    "MPS-I Hurler (IDUA): corneal clouding UNIVERSAL → ERT + HSCT; "
                    "MPS-VI Maroteaux-Lamy (ARSB): corneal clouding UNIVERSAL; "
                    "MPS-VII Sly (GUSB): corneal clouding present; "
                    "ML-IV (MCOLN1): corneal clouding PRESENT + psychomotor retardation; "
                    "ML-II: coarse facies + dysostosis + NO corneal clouding = narrow differential "
                    "to ML-II (plasma/leukocyte enzyme panel confirms); "
                    "note: MPS-II Hunter (IDS) also lacks corneal clouding but is X-linked."
                ),
            },
        ],
        "diagnostic_algorithm": [
            "Step 1: Suspect ML-II — neonate/infant with coarse facies + gingival hyperplasia "
            "(thick fibrotic hyperplastic gums FROM BIRTH — pathognomonic) + joint contractures; "
            "or suspect ML-IIIA — child age 2-5yr with bilateral carpal tunnel syndrome + joint "
            "stiffness + mild coarse facies (CTS as presenting symptom = ML-IIIA until proven otherwise)",
            "Step 2: Clinical clue checklist — gingival hyperplasia (ML-II: 90%+, pathognomonic); "
            "periosteal new bone on plain X-ray (ML-II from infancy); carpal tunnel syndrome "
            "(ML-IIIA: 85% by age 10); cardiomegaly/cardiomyopathy (ML-II: 80%); "
            "I-cell inclusions in fibroblasts; NO corneal clouding (both ML-II and ML-IIIA/B)",
            "Step 3: PLASMA lysosomal enzyme panel — HYAL1, HEX-A, HEX-B, arylsulfatase, "
            "beta-galactosidase: should be MARKEDLY ELEVATED (10-50x control); "
            "plasma HEX-B elevation most sensitive; if NORMAL → ML-II/III unlikely",
            "Step 4: LEUKOCYTE lysosomal enzyme panel — same panel on leukocytes: should be LOW "
            "or absent (<5-20% control); INVERSE PATTERN (plasma ELEVATED + leukocyte LOW) = "
            "PATHOGNOMONIC for ML-II/III — diagnose before sequencing is available",
            "Step 5: GNPTAB gene sequencing (12q23.2) — biallelic pathogenic variants; "
            "allele type determination: null/null → ML-II; biallelic missense → ML-IIIA/B; "
            "splice variants → RNA analysis (RT-PCR) to determine splice consequence and severity",
            "Step 6: Skin fibroblast culture — phase contrast microscopy for I-cell inclusions "
            "(confirmatory in ML-II; dense lysosomal inclusions in fibroblasts are the historical "
            "hallmark); electron microscopy for detailed inclusion characterization if needed",
            "Step 7: EEG MANDATORY — before prescribing any AED; classify seizure type "
            "(IS → hypsarrhythmia → ACTH Level A, NOT VGB; GTCS → LEV; Myoclonic → VPA+CLZ); "
            "IS in ML-II: hypsarrhythmia pattern guides ACTH decision and response monitoring",
            "Step 8: POLG1/POLG2 sequencing — MANDATORY before any valproate prescription; "
            "order simultaneously with enzyme panel (do not wait for ML-II diagnosis to order POLG1)",
            "Step 9: Cardiac evaluation — echo + ECG for cardiomyopathy/cardiomegaly in ML-II; "
            "cardiomyopathy (80%) is the PRIMARY drug CI driver: PHT/fosphenytoin ABSOLUTE CI, "
            "VGB ABSOLUTE CI for IS, CBZ/OXC CAUTION; cardiology co-management mandatory in ML-II",
            "Step 10: Skeletal survey — plain X-ray periosteal new bone formation (ML-II from "
            "infancy); dysostosis multiplex pattern; periosteal reaction more prominent than MPS; "
            "assess joint contracture extent; spine X-ray for scoliosis (ML-IIIA/B)",
            "Step 11: Ophthalmology — CONFIRM absence of corneal clouding (ML-II/III: absent); "
            "if corneal clouding PRESENT → reconsider MPS-I, MPS-VI, MPS-VII, ML-IV; "
            "retinal assessment (retinal degeneration absent in ML-II/III unlike MPS-II Hunter)",
            "Step 12: Carpal tunnel assessment (ML-IIIA/B) — nerve conduction study for CTS; "
            "carpal tunnel release Level A if confirmed; orthopedic assessment for scoliosis; "
            "joint physiotherapy program; multi-specialist team: metabolic + neurology + orthopedics "
            "+ cardiology + anesthesia; register in ML-II/III natural history database",
        ],
        "pharmacological_distinctions": [
            "ACTH vs VGB for IS in ML-II: ACTH Level A is THE treatment; VGB is ABSOLUTE CI "
            "(cardiac toxicity + cardiomyopathy + visual monitoring impossible); this ML-II rule "
            "is DISEASE-SPECIFIC — in other IS etiologies (TSC, cryptogenic) VGB is standard "
            "but in ML-II ACTH is the only option; VGB must be actively flagged in prescribing "
            "systems for ML-II patients",
            "Lacosamide vs CBZ/OXC for focal seizures in ML-II: lacosamide PREFERRED (PR prolongation "
            "with CBZ/OXC compounds cardiomyopathy risk); CBZ/OXC acceptable in ML-IIIB (no "
            "cardiomyopathy); cardiology review before ANY sodium-channel AED in ML-II",
            "IV LEV vs Fosphenytoin for SE in ML-II: IV LEV MANDATORY (20-60mg/kg loading); "
            "fosphenytoin ABSOLUTE CI (QTc/PR prolongation → arrhythmia in cardiomyopathy); "
            "this reverses the standard SE protocol (where fosphenytoin/PHT is second-line); "
            "emergency physicians must be briefed on ML-II SE management deviation",
            "VPA role in ML-II: broad-spectrum (GTCS + myoclonic + IS backup); POLG1 mandatory; "
            "CAUTION with hepatomegaly (if present); carnitine supplementation consider; "
            "monitor LFTs intensively; VPA is second-line for IS (after ACTH failure) in ML-II",
            "No HSCT in ML-II vs MPS-I Hurler: MPS-I Hurler HSCT before 2.5yr is Level A "
            "(cognitive benefit established); ML-II NO HSCT evidence; do NOT extrapolate MPS-I "
            "HSCT evidence to ML-II; disease mechanisms differ (ML-II: phosphotransferase "
            "deficiency vs MPS-I: alpha-L-iduronidase deficiency with different bone marrow "
            "microchimerism expectations)",
            "PHT absolute vs relative CI spectrum: ML-II (cardiomyopathy 80%) → PHT ABSOLUTE CI; "
            "ML-IIIB (minimal cardiomyopathy) → PHT CAUTION; MANBA (no cardiomyopathy, myoclonus "
            "risk) → PHT RELATIVE CI only if myoclonus on EEG; this spectrum from absolute to "
            "relative demonstrates how cardiac phenotype drives the CI severity classification",
        ],
        "differential_diagnosis": [
            "MPS-I Hurler (IDUA, 4p16.3): coarse facies + dysostosis + severe ID; "
            "DISTINGUISHES from ML-II: corneal clouding UNIVERSAL in MPS-I Hurler (ML-II: ABSENT); "
            "MPS-I: alpha-L-iduronidase LOW in leukocytes + plasma NOT elevated (not inverse "
            "pattern); urine GAG: HS+DS elevated; HSCT available before age 2.5yr (Level A); "
            "laronidase ERT available; ML-II has NO corneal clouding, NO ERT, NO HSCT",
            "MPS-II Hunter (IDS, Xq28): coarse facies + dysostosis + retinal involvement; "
            "X-linked (males predominantly); NO corneal clouding (like ML-II — key overlap); "
            "DISTINGUISHES: IDS gene X-linked; urine GAG: HS+DS elevated (ML-II: urine GAG "
            "NORMAL — not a GAG disorder); idursulfase ERT available; IDS enzyme low in "
            "leukocytes (not inverse plasma pattern); retinal involvement in MPS-II (differs ML-II)",
            "MPS-VI Maroteaux-Lamy (ARSB, 5q14.1): coarse facies + severe joint disease + "
            "NORMAL intelligence; DISTINGUISHES: corneal clouding UNIVERSAL in MPS-VI (ML-II: ABSENT); "
            "DS+C4S elevated in urine (ML-II: urine GAG NORMAL); ARSB enzyme low in leukocytes; "
            "galsulfase ERT + HSCT available; carpal tunnel prominent in MPS-VI but gingival "
            "hyperplasia less pathognomonic than ML-II",
            "ML-IV (MCOLN1, 19p13.2): lysosomal storage + psychomotor retardation; "
            "DISTINGUISHES: ML-IV has CORNEAL CLOUDING (present from early life — OPPOSITE of ML-II); "
            "Ashkenazi Jewish founder mutation; no periosteal new bone (ML-II has it); "
            "ML-IV: serum gastrin markedly elevated + achlorhydria (specific biomarker); "
            "MCOLN1 (mucolipin-1) gene deficiency — different pathway from M6P (GNPTAB); "
            "plasma lysosomal enzymes NORMAL in ML-IV (not elevated — opposite of ML-II)",
            "MPS-IVA Morquio-A (GALNS, 16q24.3): severe skeletal disease + odontoid hypoplasia; "
            "DISTINGUISHES: NORMAL intelligence in Morquio-A (unlike ML-II severe ID); "
            "NO gingival hyperplasia; keratan sulfate (KS) elevated in urine; "
            "elosulfase alfa ERT available; cervical spine instability dominant (vs ML-II "
            "where cardiomyopathy is the life-threatening feature); "
            "no carpal tunnel syndrome as presenting symptom (unlike ML-IIIA 85%)",
        ],
    }
