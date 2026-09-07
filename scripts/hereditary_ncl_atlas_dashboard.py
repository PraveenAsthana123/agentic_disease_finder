#!/usr/bin/env python3
"""Hereditary-NCL-Atlas — Complete 8-Gene Atlas
CLN1    (PPT1 / palmitoyl-protein thioesterase 1; 306 aa; 1p34.2; AR;
         Infantile NCL (INCL) — Santavuori-Haltia disease;
         onset 6–18 months; most rapid regression of all NCLs;
         granular osmiophilic deposits (GRODs) on EM PATHOGNOMONIC;
         no FDA-approved therapy; rapid death age 5–10 yr;
         seed SEED_BASE+0) .
CLN2    (TPP1 / tripeptidyl peptidase 1; 563 aa; 11p15.4; AR;
         Late-infantile NCL — Jansky-Bielschowsky disease;
         cerliponase alfa (Brineura) ICV infusion FDA2017 — FIRST CNS enzyme replacement therapy worldwide;
         curvilinear bodies on EM PATHOGNOMONIC;
         giant VEPs to 1–2 Hz flash PATHOGNOMONIC;
         onset 2–4 yr; tripling of CNS seizure severity score = treatment initiation threshold;
         seed SEED_BASE+1) .
CLN3    (Battenin / CLN3; 438 aa; 16p12.1; AR;
         Juvenile NCL — Spielmeyer-Vogt-Sjögren / classic Batten disease;
         most common NCL worldwide;
         blindness FIRST — vision loss precedes seizures by 5 yr PATHOGNOMONIC ordering;
         sea-blue histiocytes on bone marrow biopsy PATHOGNOMONIC;
         vacuolated lymphocytes on blood smear;
         1-kb deletion c.461-280_677+382del accounts for ~73% of CLN3 alleles;
         fingerprint profiles on EM;
         seed SEED_BASE+2) .
CLN5    (CLN5; 407 aa; 13q22.3; AR;
         Finnish variant late-infantile NCL;
         TPP1 enzyme activity NORMAL — distinguishes from CLN2;
         molecular: ER/Golgi membrane protein (not lysosomal enzyme);
         p.Y392X Finnish founder mutation;
         onset 4–7 yr; ataxia more prominent than classic late-infantile;
         mixed deposits on EM: fingerprints + rectilinear;
         seed SEED_BASE+3) .
CLN6    (Linclin / CLN6; 311 aa; 15q23; AR;
         Variant late-infantile NCL and adult Kufs disease (AD alleles);
         ER transmembrane protein — no lysosomal enzyme defect;
         Sri Lankan / Costa Rican / Romani founder mutations;
         Kufs disease (adult-onset): CLN6 (AR) or DNAJC5 — CLN4 (AD);
         cortical neuron EM: mixed curvilinear + fingerprint;
         seed SEED_BASE+4) .
CLN7    (MFSD8 / major facilitator superfamily domain-containing 8; 518 aa; 4q28.2; AR;
         Late-infantile variant NCL;
         lysosomal membrane protein (not enzyme);
         Turkish founder c.103C>T (p.Arg35Trp) and Pakistani founder;
         fingerprint profiles predominate on EM — unlike CLN2 curvilinear;
         onset 3–6 yr;
         seed SEED_BASE+5) .
CLN8    (CLN8; 286 aa; 8p23.3; AR;
         Northern epilepsy syndrome (EPMR) — Finnish founder p.Arg24Gly;
         SLOWEST progression of all NCLs — Finnish patients survive to age 35–50 yr;
         ER membrane protein;
         progressive myoclonic epilepsy + intellectual decline, without visual loss (EPMR subtype);
         variant late-infantile form (non-Finnish): rapid, similar to CLN6;
         EM: granular / curvilinear mixed;
         seed SEED_BASE+6) .
CLN10   (CTSD / cathepsin D; 412 aa; 11p15.5; AR;
         Congenital NCL (MOST SEVERE) — born with seizures, microcephaly, respiratory failure → death within days–weeks;
         OR juvenile/adult form (partial loss-of-function) with slower progression;
         CTSD is the ONLY NCL gene encoding an aspartyl protease (all others: PPT, TPP1, or membrane proteins);
         p.Y199C hypomorphic → juvenile form; null mutations → congenital lethal;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1902–1909)
"""

import random

SEED_BASE = 1902

NCL_GENES = [
    # -- CLN1 / PPT1 — Infantile NCL (INCL) -----------------------------------------
    {
        "gene": "CLN1",
        "alt_name": "PPT1",
        "protein": (
            "CLN1/PPT1 -- 1p34.2 AR -- PPT1-306aa -- "
            "Infantile-NCL-INCL-Santavuori-Haltia -- "
            "GRODs-Granular-Osmiophilic-Deposits-EM-PATHOGNOMONIC -- "
            "Onset-6-18m-Rapid-Regression-No-FDA-Therapy -- "
            "PPT1-Enzyme-Blood-Leukocytes-Diagnostic-Assay"
        ),
        "locus": "1p34.2",
        "protein_size": "306 aa",
        "inheritance": "AR",
        "age_of_onset": "6–18 months (infantile)",
        "key_biomarker": (
            "PPT1 enzyme activity markedly reduced in leukocytes/fibroblasts PATHOGNOMONIC; "
            "GRODs on skin or conjunctival biopsy EM; EEG: high-amplitude slow waves early → isoelectric late; "
            "brain MRI: severe atrophy, T2 signal loss in thalami/white matter"
        ),
        "pathognomonic": (
            "GRODs (granular osmiophilic deposits) on EM biopsy — round, dense, 50–80 nm granules; "
            "combined with PPT1 enzyme deficiency = INCL diagnosis confirmed; "
            "absent VEPs and ERG by age 3 yr"
        ),
        "treatment": (
            "No FDA-approved disease-modifying therapy; "
            "cysteamine + N-acetylcysteine (antioxidants) — used in some centres, no proven efficacy; "
            "symptomatic: anti-seizure medications (valproate CAUTION — hepatotoxicity); "
            "palliative care; gene therapy trials (AAV9-PPT1) in progress; "
            "multidisciplinary: neurology, dietetics, palliative care"
        ),
        "critical_flags": [
            "GRODS-EM-PATHOGNOMONIC — skin punch biopsy or conjunctival biopsy; round granular deposits 50-80 nm; do NOT mistake for lipofuscin of aging",
            "PPT1-ENZYME-BLOOD-TEST — leukocyte PPT1 activity <5% of normal is diagnostic; available in specialist labs; fastest diagnostic step",
            "MOST-RAPID-REGRESSION-NCL — INCL has fastest decline; walking/talking lost by age 2-3 yr; death age 5-10 yr without respiratory support",
            "VALPROATE-HEPATOTOXICITY-CAUTION — VPA used but monitor LFTs closely; NCL patients have mitochondrial-like vulnerability to VPA",
            "NO-APPROVED-THERAPY — unlike CLN2 (cerliponase alfa); enrol in gene therapy trial if eligible; AAV9-PPT1 intrathecal",
            "VISION-LOSS-EARLY — retinal degeneration by 6-12 months; ERG flat early; ophthalmology mandatory at diagnosis",
            "EEG-ISOELECTRIC-LATE — high-amplitude slow waves → suppression-burst → isoelectric; timing correlates with cognitive decline",
            "PPT1-PALMITOYL-THIOESTERASE — cleaves fatty acid thioesters from palmitoylated proteins in lysosomes; dysfunction → ceroid/lipofuscin accumulation"
        ],
        "alias": (
            "CLN1 (PPT1 — palmitoyl-protein thioesterase 1); OMIM gene 600722; "
            "Neuronal ceroid lipofuscinosis type 1 (INCL / Santavuori-Haltia disease) OMIM 256730. "
            "1p34.2; 306 aa; ~34 kDa; lysosomal serine thioesterase; autosomal recessive. "
            "FUNCTION: PPT1 encodes palmitoyl-protein thioesterase 1, a lysosomal enzyme that cleaves long-chain fatty acids "
            "(palmitoyl chains) from palmitoylated proteins during lysosomal catabolism. "
            "Loss of PPT1 → accumulation of palmitoylated proteins and lipids → ceroid storage (GRODs). "
            "GRODs are electron-dense granular deposits ~50-80 nm in diameter, visible on EM of biopsy material. "
            "They represent autofluorescent material (ceroid/lipofuscin) in secondary lysosomes. "
            "CLINICAL PRESENTATION: "
            "Infantile NCL (INCL) / Santavuori-Haltia disease: "
            "Normal development until 6-18 months; rapid psychomotor regression; "
            "Myoclonic jerks and tonic-clonic seizures begin age 1-2 yr; "
            "Hypotonia, progressive spasticity; "
            "Retinal degeneration → blindness by 18-24 months; "
            "Loss of walking by 2-3 yr; loss of speech; "
            "Death typically age 5-10 yr (may extend to teenage years with respiratory support). "
            "ATYPICAL FORMS: "
            "Late-infantile/juvenile-onset (PPT1) — residual enzyme activity; slower progression; "
            "Adult-onset PPT1 — extremely rare. "
            "DIAGNOSIS: "
            "PPT1 enzyme assay in leukocytes or dried blood spot — fastest test; "
            "Skin biopsy EM: GRODs in eccrine sweat gland cells, pericytes; "
            "PPT1 gene sequencing: c.364A>T (p.Thr122Met) Scandinavian founder; c.169C>T Finnish; "
            "Brain MRI: progressive cortical + cerebellar atrophy; T2 signal loss thalami. "
            "MANAGEMENT: "
            "No approved disease-modifying therapy. "
            "Anti-epileptic drugs (avoid phenytoin early; VPA with caution). "
            "Physiotherapy, speech therapy, palliative care. "
            "Gene therapy trials: intrathecal AAV9-PPT1 (Phase 1/2 active). "
            "Enzyme replacement trials. "
            "Biobank enrolment strongly recommended."
        ),
    },
    # -- CLN2 / TPP1 — Late-infantile NCL (Brineura FDA2017) -------------------------
    {
        "gene": "CLN2",
        "alt_name": "TPP1",
        "protein": (
            "CLN2/TPP1 -- 11p15.4 AR -- TPP1-563aa -- "
            "Late-Infantile-NCL-Jansky-Bielschowsky -- "
            "Cerliponase-Alfa-Brineura-ICV-FDA2017-ONLY-NCL-Enzyme-Therapy -- "
            "Curvilinear-Bodies-EM-PATHOGNOMONIC -- "
            "Giant-VEPs-1-2Hz-Flash-PATHOGNOMONIC"
        ),
        "locus": "11p15.4",
        "protein_size": "563 aa",
        "inheritance": "AR",
        "age_of_onset": "2–4 years (late-infantile)",
        "key_biomarker": (
            "TPP1 enzyme activity reduced in leukocytes/fibroblasts PATHOGNOMONIC; "
            "curvilinear bodies on EM biopsy; "
            "giant VEPs to 1-2 Hz flash stimulation (abolished at higher frequencies) PATHOGNOMONIC; "
            "CLN2 clinical rating scale (motor/language/seizure 0-6 each; total 18); "
            "MRI: cerebellar + cortical atrophy progressive"
        ),
        "pathognomonic": (
            "CLN2 = ONLY NCL with FDA-approved enzyme replacement therapy (cerliponase alfa / Brineura); "
            "biweekly intracerebroventricular (ICV) infusions via Ommaya reservoir; "
            "giant VEPs (>50 µV) at 1-2 Hz flash PATHOGNOMONIC — abolished at 3+ Hz; "
            "curvilinear bodies on EM (parallel stacked membrane profiles)"
        ),
        "treatment": (
            "CERLIPONASE ALFA (Brineura) — biweekly ICV infusion 300 mg; "
            "FDA approved 2017 (first CNS enzyme replacement worldwide); EMA approved 2017; "
            "slows motor and language decline; initiate before severe motor loss (CLN2 score ≥6/6 motor); "
            "Ommaya reservoir implantation required; "
            "anti-epileptics: lamotrigine, levetiracetam (avoid carbamazepine — worsens); "
            "monitor CLN2 rating scale every 3-6 months for treatment response"
        ),
        "critical_flags": [
            "CERLIPONASE-ALFA-ICV-FDA2017 — ONLY NCL with enzyme replacement; biweekly Ommaya infusion; do NOT miss this — changes prognosis",
            "GIANT-VEPs-1-2Hz-PATHOGNOMONIC — >50 µV at 1-2 Hz flash only; absent at higher rates; distinctive EEG-EP finding; screen all late-infantile seizure patients",
            "CLN2-RATING-SCALE — score motor/language/seizure 0-6 each (18 max); initiate cerliponase when score ≥12 or declining; monitor 3-6 monthly",
            "CURVILINEAR-BODIES-EM — parallel stacked membrane profiles in skin biopsy cells; PATHOGNOMONIC for CLN2 among NCLs",
            "TPP1-ENZYME-DBS-AVAILABLE — dried blood spot enzyme assay; fast, inexpensive first-line test; <5% activity = CLN2 confirmed",
            "CARBAMAZEPINE-WORSENS — avoid in CLN2; can precipitate acute encephalopathy; use lamotrigine or levetiracetam",
            "ONSET-2-4yr-SEIZURES-FIRST — typically myoclonic + atonic seizures as first symptom; vision loss follows (contrast CLN3 where vision loss is first)",
            "OMMAYA-RESERVOIR-MANDATORY — permanent implant for ICV drug delivery; neurosurgery team required before cerliponase start"
        ],
        "alias": (
            "CLN2 (TPP1 — tripeptidyl peptidase 1); OMIM gene 607998; "
            "Neuronal ceroid lipofuscinosis type 2 (late-infantile / Jansky-Bielschowsky) OMIM 204500. "
            "11p15.4; 563 aa; ~66 kDa (pro-enzyme) / ~45 kDa (mature); lysosomal serine protease; autosomal recessive. "
            "FUNCTION: TPP1 encodes tripeptidyl peptidase 1, a lysosomal serine protease that cleaves tripeptides "
            "from the N-terminus of proteins under acidic conditions. "
            "It is required for lysosomal protein catabolism; loss of TPP1 → protein/lipid accumulation → curvilinear body formation. "
            "CLINICAL PRESENTATION: "
            "Late-infantile NCL (onset age 2-4 yr): "
            "Seizures are the first symptom (myoclonic, atonic, tonic-clonic); "
            "Language and motor regression follow; "
            "Retinal dystrophy → visual loss (later than CLN3); "
            "CLN2 clinical rating scale (0-18): motor (0-6) + language (0-6) + seizure (0-6); "
            "Rapid decline without treatment; death typically age 5-10 yr. "
            "With cerliponase alfa treatment: slower decline, motor/language preservation for 2-4 additional years. "
            "DIAGNOSIS: "
            "TPP1 enzyme assay — leukocytes or DBS; <5% normal = diagnostic; "
            "Skin/conjunctival biopsy EM: curvilinear bodies in eccrine secretory cells; "
            "VEPs: giant responses >50 µV at 1-2 Hz flash, disappearing at 3+ Hz — PATHOGNOMONIC; "
            "CLN2 gene sequencing: c.509-1G>C (IVS5-1G>C) and c.622C>T (p.Arg208X) most common European alleles. "
            "TREATMENT: "
            "Cerliponase alfa (Brineura): 300 mg biweekly via Ommaya reservoir; "
            "Only enzyme replacement that crosses blood-brain barrier when given ICV; "
            "Start before severe motor loss; monitor with CLN2 rating scale; "
            "Premedication with antihistamines ± corticosteroids recommended. "
            "Anti-epileptics: lamotrigine, levetiracetam (avoid carbamazepine, phenytoin). "
            "Multidisciplinary: neurology, ophthalmology, physiotherapy, palliative care."
        ),
    },
    # -- CLN3 — Juvenile NCL / Batten disease -----------------------------------------
    {
        "gene": "CLN3",
        "alt_name": "Battenin",
        "protein": (
            "CLN3 -- 16p12.1 AR -- Battenin-438aa -- "
            "Juvenile-NCL-JNCL-Spielmeyer-Vogt-Sjogren-Classic-Batten-Disease -- "
            "Sea-Blue-Histiocytes-Bone-Marrow-PATHOGNOMONIC -- "
            "Vision-Loss-FIRST-Precedes-Seizures-5yr-PATHOGNOMONIC-Ordering -- "
            "1-kb-Deletion-73pct-Alleles-Vacuolated-Lymphocytes"
        ),
        "locus": "16p12.1",
        "protein_size": "438 aa",
        "inheritance": "AR",
        "age_of_onset": "4–7 years (juvenile)",
        "key_biomarker": (
            "Vacuolated lymphocytes on peripheral blood smear PATHOGNOMONIC for CLN3; "
            "sea-blue histiocytes on bone marrow biopsy PATHOGNOMONIC; "
            "1-kb deletion (c.461-280_677+382del) detectable by PCR — 73% of CLN3 alleles; "
            "fingerprint profiles on EM; "
            "CLN3 gene sequencing; "
            "ERG: extinguished early; VEPs: diminished"
        ),
        "pathognomonic": (
            "Vision loss (macular degeneration/retinal dystrophy) PRECEDES seizures by 5 years — PATHOGNOMONIC for CLN3 ordering; "
            "sea-blue histiocytes on BM biopsy PATHOGNOMONIC; "
            "vacuolated lymphocytes on blood film; "
            "1-kb deletion accounts for 73% of alleles worldwide"
        ),
        "treatment": (
            "No FDA-approved disease-modifying therapy; "
            "mycophenolate mofetil (immunomodulation) — trial evidence limited; "
            "anti-epileptics: lamotrigine, levetiracetam, clobazam; "
            "avoid carbamazepine; "
            "vision aids / low vision rehabilitation; "
            "gene therapy trials (AAV-CLN3 intrathecal); "
            "substrate reduction and chaperone research ongoing; "
            "multidisciplinary neurology, ophthalmology, psychiatry (behavioural), palliative care"
        ),
        "critical_flags": [
            "VISION-LOSS-5yr-BEFORE-SEIZURES-PATHOGNOMONIC — macular dystrophy onset 5-7yr; first symptom BEFORE neurological decline; contrast CLN2 (seizures first)",
            "SEA-BLUE-HISTIOCYTES-BM-PATHOGNOMONIC — bone marrow biopsy; sea-blue histiocytes contain ceroid; diagnostic in correct clinical context; do NOT mistake for Niemann-Pick",
            "VACUOLATED-LYMPHOCYTES-BLOOD-SMEAR — peripheral blood; lymphocytes with cytoplasmic vacuoles; rapid inexpensive screen; not specific but highly suggestive of CLN3",
            "1-KB-DELETION-73pct — c.461-280_677+382del; detectable by simple PCR; homozygous deletion confirms CLN3; test FIRST before full gene sequencing",
            "NO-APPROVED-THERAPY — unlike CLN2; mycophenolate used without strong evidence; enrol in gene therapy trials",
            "BEHAVIOURAL-PSYCHIATRIC-EARLY — depression, anxiety, psychosis in early teens before cognitive decline evident; psychiatry co-management essential",
            "FINGERPRINT-PROFILES-EM — curvilinear + fingerprint mixed deposits in skin biopsy; fingerprint predominant in JNCL",
            "MOST-COMMON-NCL-WORLDWIDE — CLN3 JNCL is the most prevalent NCL; 1:25,000 in Finland; 1:100,000 globally; Spielmeyer-Vogt eponym"
        ],
        "alias": (
            "CLN3 (Battenin / CLN3); OMIM gene 607042; "
            "Neuronal ceroid lipofuscinosis type 3 (JNCL / Juvenile Batten disease / Spielmeyer-Vogt-Sjögren) OMIM 204200. "
            "16p12.1; 438 aa; ~48 kDa; lysosomal/late endosomal membrane protein; autosomal recessive. "
            "FUNCTION: CLN3 encodes Battenin, a multi-pass transmembrane protein of lysosomal and late endosomal membranes. "
            "Its exact function is incompletely understood; "
            "putative roles in lysosomal pH regulation, arginine transport, and autophagosome-lysosome fusion. "
            "Loss of CLN3 → fingerprint body formation (concentric lamellar membrane deposits) in neurons and other cells. "
            "CLINICAL PRESENTATION: "
            "Juvenile NCL (JNCL) — onset age 4-7 yr: "
            "FIRST symptom: progressive visual failure (macular dystrophy → retinitis pigmentosa pattern → blindness by 10 yr); "
            "Seizures begin 2-5 years AFTER visual loss onset (myoclonic, complex partial, tonic-clonic); "
            "Psychiatric symptoms: depression, anxiety, hallucinations, behavioural change in early-mid teens; "
            "Motor deterioration: ataxia, dysarthria, rigidity (onset mid-teens); "
            "Cognitive decline: intellectual disability progressive; "
            "Death: 15-35 yr typically. "
            "DIAGNOSIS: "
            "1-kb deletion PCR — screen first; homozygous = CLN3 diagnosed; "
            "Peripheral blood smear: vacuolated lymphocytes; "
            "Bone marrow biopsy: sea-blue histiocytes; "
            "Skin biopsy EM: fingerprint profiles + curvilinear in eccrine cells; "
            "Ophthalmology: ERG extinguished early; "
            "CLN3 gene sequencing for non-deletion alleles. "
            "MANAGEMENT: "
            "No disease-modifying therapy approved. "
            "Anti-epileptics: lamotrigine, clobazam, levetiracetam (carbamazepine exacerbates). "
            "Low vision aids, guide dog, braille education. "
            "Psychiatric medications for behavioural symptoms. "
            "Physical, occupational, speech therapy. "
            "Gene therapy trial: intrathecal AAV9-CLN3 (clinical trials ongoing). "
            "Palliative care involvement from diagnosis."
        ),
    },
    # -- CLN5 — Finnish variant late-infantile NCL ------------------------------------
    {
        "gene": "CLN5",
        "alt_name": "CLN5",
        "protein": (
            "CLN5 -- 13q22.3 AR -- CLN5-407aa -- "
            "Finnish-Variant-Late-Infantile-NCL -- "
            "TPP1-Enzyme-Normal-Distinguishes-From-CLN2 -- "
            "p.Y392X-Finnish-Founder -- "
            "ER-Golgi-Membrane-Protein-Not-Lysosomal-Enzyme"
        ),
        "locus": "13q22.3",
        "protein_size": "407 aa",
        "inheritance": "AR",
        "age_of_onset": "4–7 years (Finnish variant late-infantile)",
        "key_biomarker": (
            "TPP1 enzyme activity NORMAL (distinguishes from CLN2); "
            "CLN5 gene sequencing with p.Y392X Finnish founder; "
            "skin biopsy EM: mixed fingerprint + rectilinear profiles; "
            "brain MRI: cerebellar and cortical atrophy; "
            "CLN5 protein immunohistochemistry on patient fibroblasts"
        ),
        "pathognomonic": (
            "Finnish variant late-infantile NCL with TPP1 enzyme NORMAL — rules out CLN2; "
            "p.Y392X (c.1175T>G) founder mutation in Finnish population; "
            "mixed EM deposits: fingerprint + rectilinear profiles (not pure curvilinear = CLN2)"
        ),
        "treatment": (
            "No FDA-approved therapy; "
            "anti-epileptics: lamotrigine, levetiracetam; "
            "physiotherapy, communication aids; "
            "gene therapy trials (AAV9-CLN5) in development; "
            "enzyme replacement strategy (CLN5 is soluble, secretable — potential for ERT if produced); "
            "palliative care"
        ),
        "critical_flags": [
            "TPP1-ENZYME-NORMAL-KEY-DISTINCTION — TPP1 activity normal in CLN5; do NOT stop at CLN2 enzyme test if negative; proceed to gene panel",
            "FINNISH-FOUNDER-p.Y392X — c.1175T>G; homozygous in Finnish patients; screen targeted mutation first in Finnish ancestry",
            "ATAXIA-PROMINENT-CLN5 — cerebellar ataxia is relatively more prominent than in classic CLN2 late-infantile",
            "MIXED-EM-PROFILES — fingerprint + rectilinear (not pure curvilinear); EM biopsy still useful but pattern different from CLN2",
            "CLN5-SOLUBLE-SECRETED — unlike CLN1/2 which are also soluble, CLN5 is secreted and taken up by neighbouring cells; ERT cross-correction possible",
            "NO-GIANT-VEPs — VEPs are reduced but NOT the characteristic giant 1-2 Hz VEPs of CLN2; helps distinguish",
            "ONSET-4-7yr-SIMILAR-CLN2 — clinical overlap with CLN2 makes enzyme test critical; gene panel if TPP1 normal",
            "PREVALENCE-FINLAND — higher frequency in Finland due to p.Y392X founder; ~1:30,000 in Finland; rare elsewhere"
        ],
        "alias": (
            "CLN5 (CLN5 — ceroid lipofuscinosis neuronal 5); OMIM gene 608102; "
            "Neuronal ceroid lipofuscinosis type 5 (Finnish variant late-infantile / CLN5 disease) OMIM 256731. "
            "13q22.3; 407 aa; ~46 kDa; ER/Golgi soluble lysosomal-targeted protein; autosomal recessive. "
            "FUNCTION: CLN5 encodes a soluble protein targeted to lysosomes via the Golgi. "
            "It is not an enzyme in the conventional sense but rather a lysosomal accessory protein. "
            "CLN5 interacts with CLN1 (PPT1), CLN2 (TPP1), and CLN3 proteins. "
            "Loss of CLN5 → mixed ceroid/lipofuscin accumulation with fingerprint + rectilinear profiles on EM. "
            "CLINICAL PRESENTATION: "
            "Finnish variant late-infantile NCL (onset 4-7 yr): "
            "Progressive visual failure, seizures, motor deterioration, cognitive decline; "
            "Similar to CLN2 clinically but slower progression; "
            "Ataxia more prominent than typical CLN2; "
            "Death in 2nd-3rd decade. "
            "DIAGNOSIS: "
            "CLN2 (TPP1) enzyme assay NORMAL — critical distinction; "
            "CLN5 gene sequencing: p.Y392X (c.1175T>G) Finnish founder; "
            "EM skin biopsy: mixed fingerprint + rectilinear profiles; "
            "NCL gene panel recommended for all late-infantile NCL with normal TPP1. "
            "MANAGEMENT: "
            "No disease-modifying therapy approved. "
            "Symptomatic: anti-epileptics, physiotherapy, speech therapy. "
            "Palliative care. "
            "Gene therapy trials active (CLN5 AAV9). "
            "ERT potential (CLN5 secretable — cross-correction experiments ongoing)."
        ),
    },
    # -- CLN6 — Variant late-infantile / Kufs disease --------------------------------
    {
        "gene": "CLN6",
        "alt_name": "Linclin",
        "protein": (
            "CLN6 -- 15q23 AR -- Linclin-311aa -- "
            "Variant-Late-Infantile-NCL-Sri-Lankan-Costa-Rican-Romani-Founders -- "
            "ER-Membrane-Protein-No-Enzyme-Defect -- "
            "Kufs-Disease-Adult-NCL-CLN6-AR-or-DNAJC5-AD"
        ),
        "locus": "15q23",
        "protein_size": "311 aa",
        "inheritance": "AR (variant late-infantile / Kufs type A); AD (Kufs type B via DNAJC5)",
        "age_of_onset": "18 months–8 years (variant late-infantile); adult (Kufs disease)",
        "key_biomarker": (
            "TPP1 enzyme NORMAL; PPT1 enzyme NORMAL; "
            "CLN6 gene sequencing; "
            "EM biopsy: mixed curvilinear + fingerprint profiles; "
            "brain MRI: cerebellar predominant atrophy early"
        ),
        "pathognomonic": (
            "Variant late-infantile NCL with normal TPP1 AND PPT1 enzyme — gene panel required; "
            "founder mutations: p.Ile154del Sri Lankan; p.Cys267ArgfsTer1 Romani; Costa Rican c.214G>T; "
            "Kufs disease (adult NCL) — progressive myoclonic epilepsy ± dementia + ataxia WITHOUT visual failure"
        ),
        "treatment": (
            "No FDA-approved therapy; "
            "anti-epileptics; "
            "gene therapy trials (AAV-CLN6 intrathecal — Phase 1/2); "
            "palliative care; "
            "genetic counselling; "
            "Kufs disease: benzodiazepines + valproate for myoclonus"
        ),
        "critical_flags": [
            "NORMAL-TPP1-PPT1-ENZYMES — both CLN1 (PPT1) and CLN2 (TPP1) enzyme assays normal; proceed to NCL gene panel",
            "SRI-LANKAN-FOUNDER-p.Ile154del — enriched in Sri Lankan/South Asian populations; targeted first in appropriate ancestry",
            "KUFS-DISEASE-ADULT-NCL — CLN6 (AR) causes Kufs type A: adult progressive myoclonic epilepsy + dementia + ataxia; vision NORMAL (CONTRAST to juvenile CLN3)",
            "KUFS-NO-VISUAL-LOSS — adult NCL (Kufs) has no retinal involvement; distinguishes from all paediatric NCLs where vision loss is common",
            "ER-MEMBRANE-PROTEIN — CLN6 is ER-resident transmembrane protein; not a lysosomal enzyme; enzyme replacement strategy not applicable",
            "MIXED-EM-DEPOSITS — curvilinear + fingerprint; less distinctive than CLN2 or CLN3; EM still informative",
            "GENE-THERAPY-TRIAL-ACTIVE — AAV-CLN6 intrathecal (Phase 1/2); enrol eligible patients; CLN6 restoration via gene therapy most promising current strategy",
            "ROMA-POPULATION-ENRICHED — Romani (Gypsy) population has higher CLN6 frequency due to founder effect; p.Cys267ArgfsTer1 Romani"
        ],
        "alias": (
            "CLN6 (Linclin / CLN6); OMIM gene 606725; "
            "Neuronal ceroid lipofuscinosis type 6 (variant late-infantile NCL / Kufs disease type A) OMIM 601780. "
            "15q23; 311 aa; ~36 kDa; ER transmembrane protein; autosomal recessive (most); AD Kufs (DNAJC5 gene). "
            "FUNCTION: CLN6 encodes a multi-pass transmembrane protein of the ER membrane. "
            "It is not a lysosomal enzyme; its function is poorly defined. "
            "Proposed roles: glycolipid biosynthesis, protein trafficking, ER quality control. "
            "Loss of CLN6 → ceroid accumulation with mixed curvilinear + fingerprint profiles on EM. "
            "CLINICAL PRESENTATION: "
            "Variant late-infantile NCL (typical onset 18m-8 yr): "
            "Seizures, motor deterioration, visual loss, cognitive regression — similar to CLN2; "
            "Sri Lankan/Romani/Costa Rican founders show higher frequency. "
            "Kufs disease (CLN6-related adult NCL — type A): "
            "Adult onset (usually >30 yr); progressive myoclonic epilepsy ± dementia; "
            "Ataxia; NO retinal degeneration (vision normal) — key clinical clue; "
            "Kufs type B: AD CLN6 (some cases) or DNAJC5 (CLN4). "
            "DIAGNOSIS: "
            "Both PPT1 and TPP1 enzymes normal — then NCL gene panel; "
            "CLN6 gene sequencing; "
            "EM: mixed curvilinear + fingerprint; "
            "Kufs disease: brain MRI (cortical + cerebellar atrophy); EEG (giant SSEPs, PLEDs); "
            "Brain biopsy rarely needed if gene panel diagnostic. "
            "MANAGEMENT: "
            "No approved therapy. Symptomatic management. "
            "AAV-CLN6 gene therapy trial: enrol eligible patients. "
            "Kufs: valproate + benzodiazepines for myoclonus; piracetam."
        ),
    },
    # -- CLN7 / MFSD8 — Late-infantile variant -----------------------------------------
    {
        "gene": "CLN7",
        "alt_name": "MFSD8",
        "protein": (
            "CLN7/MFSD8 -- 4q28.2 AR -- MFSD8-518aa -- "
            "Late-Infantile-Variant-NCL-Turkish-Pakistani-Founders -- "
            "Lysosomal-Membrane-Protein-Major-Facilitator-Superfamily -- "
            "Fingerprint-Profiles-Predominate-EM-Contrast-CLN2-Curvilinear -- "
            "c.103C-T-p.Arg35Trp-Turkish-Founder"
        ),
        "locus": "4q28.2",
        "protein_size": "518 aa",
        "inheritance": "AR",
        "age_of_onset": "3–6 years (late-infantile variant)",
        "key_biomarker": (
            "TPP1 enzyme NORMAL; PPT1 enzyme NORMAL; "
            "CLN7/MFSD8 gene sequencing; "
            "EM: fingerprint profiles predominant (contrast CLN2: curvilinear predominant); "
            "brain MRI: progressive cerebral + cerebellar atrophy"
        ),
        "pathognomonic": (
            "Late-infantile NCL with normal TPP1+PPT1 AND fingerprint-predominant EM deposits; "
            "Turkish founder c.103C>T (p.Arg35Trp); "
            "Pakistani founder c.881C>A (p.Ala294Asp); "
            "MFSD8 is a lysosomal membrane transporter (MFS family)"
        ),
        "treatment": (
            "No FDA-approved therapy; "
            "anti-epileptics: lamotrigine, levetiracetam; "
            "physiotherapy, visual rehabilitation; "
            "gene therapy research (MFSD8 AAV); "
            "palliative care"
        ),
        "critical_flags": [
            "MFSD8-LYSOSOMAL-MEMBRANE-TRANSPORTER — not an enzyme; major facilitator superfamily; substrate unknown; ERT not applicable",
            "TURKISH-FOUNDER-c.103C>T — p.Arg35Trp; homozygous in Turkish patients; screen targeted mutation first",
            "FINGERPRINT-EM-PREDOMINANT — fingerprint profiles dominate; contrast CLN2 (curvilinear predominant); EM subtype helps narrow gene",
            "NORMAL-TPP1-PPT1 — both enzyme tests normal; gene panel mandatory; CLN7 is identified by sequencing",
            "ONSET-3-6yr-SEIZURES-VISUAL-LOSS — similar presentation to CLN2/CLN5/CLN6; enzyme + EM distinguish",
            "RARE-OUTSIDE-FOUNDER-POPULATIONS — CLN7 is less common than CLN2/CLN3 globally; higher in Turkish/Pakistani/Turkish-Cypriot",
            "NO-GIANT-VEPs — does not show the characteristic giant 1-2 Hz VEPs of CLN2; EEG/EP helps differentiate",
            "MFSD8-SUBSTRATE-UNKNOWN — transporter function not characterised; may transport modified amino acids or lipids out of lysosome"
        ],
        "alias": (
            "CLN7 (MFSD8 — major facilitator superfamily domain-containing 8); OMIM gene 611124; "
            "Neuronal ceroid lipofuscinosis type 7 (CLN7 disease / MFSD8-related) OMIM 610951. "
            "4q28.2; 518 aa; ~58 kDa; lysosomal membrane protein (MFS transporter family); autosomal recessive. "
            "FUNCTION: MFSD8 encodes a multi-pass transmembrane protein of the major facilitator superfamily, "
            "localised to lysosomal membranes. "
            "It is predicted to function as a lysosomal transporter, though its substrate is not established. "
            "Loss of MFSD8 → fingerprint body accumulation in neurons and other cells. "
            "CLINICAL PRESENTATION: "
            "Late-infantile variant NCL (onset 3-6 yr): "
            "Seizures (often myoclonic and generalised), visual deterioration, "
            "motor and speech regression, cognitive decline; "
            "Similar to CLN2 clinically; Turkish, Pakistani, and Turkish-Cypriot populations enriched. "
            "DIAGNOSIS: "
            "TPP1 and PPT1 enzyme assays both normal; "
            "NCL gene panel (MFSD8/CLN7); "
            "EM: fingerprint profiles predominant (curvilinear less prominent than CLN2); "
            "Turkish founder: c.103C>T (p.Arg35Trp) — targeted PCR. "
            "MANAGEMENT: "
            "No approved therapy. "
            "Symptomatic: anti-epileptics. "
            "Gene therapy research (pre-clinical). "
            "Palliative care."
        ),
    },
    # -- CLN8 — Northern epilepsy / EPMR -----------------------------------------------
    {
        "gene": "CLN8",
        "alt_name": "CLN8",
        "protein": (
            "CLN8 -- 8p23.3 AR -- CLN8-286aa -- "
            "Northern-Epilepsy-EPMR-Finnish-Founder-p.Arg24Gly -- "
            "SLOWEST-Progression-All-NCLs-Survival-35-50yr -- "
            "ER-Membrane-Protein-Progressive-Myoclonic-Epilepsy -- "
            "Variant-Late-Infantile-Non-Finnish-Rapid-Form-Also"
        ),
        "locus": "8p23.3",
        "protein_size": "286 aa",
        "inheritance": "AR",
        "age_of_onset": "5–10 years (Northern epilepsy EPMR); 1.5–8 years (variant late-infantile form)",
        "key_biomarker": (
            "TPP1 and PPT1 enzymes NORMAL; "
            "CLN8 gene sequencing: p.Arg24Gly Finnish founder; "
            "EM: mixed granular/curvilinear deposits; "
            "EEG: progressive myoclonic epilepsy pattern; "
            "brain MRI: initially subtle → progressive cerebral atrophy"
        ),
        "pathognomonic": (
            "CLN8 Northern epilepsy (EPMR) — p.Arg24Gly Finnish founder; "
            "SLOWEST NCL progression — Finnish children survive to age 35-50 yr; "
            "progressive myoclonic epilepsy + cognitive decline WITHOUT early blindness (EPMR subtype); "
            "vision usually preserved until late stages (contrast other NCLs)"
        ),
        "treatment": (
            "No FDA-approved therapy; "
            "anti-epileptics: valproate + clobazam (EPMR); lamotrigine; "
            "piracetam for myoclonus; "
            "physiotherapy, cognitive support; "
            "gene therapy trials (AAV-CLN8); "
            "palliative care — much later than other NCLs due to slow progression"
        ),
        "critical_flags": [
            "SLOWEST-PROGRESSION-NCL — Finnish EPMR patients survive to 35-50yr; very different prognosis from CLN1 (death age 5-10yr); communicate this clearly to families",
            "EPMR-NO-EARLY-BLINDNESS — Northern epilepsy (EPMR): progressive myoclonic epilepsy + cognitive decline; vision preserved until late; contrast all other NCLs",
            "FINNISH-FOUNDER-p.Arg24Gly — c.70C>G; Northern epilepsy confined to northern Finland; regional; if Finnish ancestry + PME, test CLN8 first",
            "VARIANT-LI-FORM-RAPID — non-Finnish CLN8 mutations (e.g., c.749C>T p.Ser250Phe) cause rapid late-infantile NCL similar to CLN6; DIFFERENT prognosis from EPMR",
            "ER-MEMBRANE-PROTEIN — CLN8 ER-localised; interacts with COPI vesicles; ER-to-lysosome trafficking role proposed; not an enzyme",
            "NORMAL-ENZYME-BOTH — both PPT1 and TPP1 normal; NCL gene panel required; CLN8 by sequencing",
            "MYOCLONIC-EPILEPSY-PROMINENT — in EPMR: severe myoclonic jerks; valproate + clobazam effective initially; piracetam adjunct",
            "GENE-THERAPY-BEING-DEVELOPED — AAV9-CLN8 intrathecal; Phase 1/2 trials emerging; enrol eligible patients"
        ],
        "alias": (
            "CLN8 (CLN8 — ceroid lipofuscinosis neuronal 8); OMIM gene 607837; "
            "Northern epilepsy (EPMR) / NCL type 8 OMIM 600143; variant late-infantile CLN8 OMIM 600143. "
            "8p23.3; 286 aa; ~33 kDa; ER membrane protein; autosomal recessive. "
            "FUNCTION: CLN8 encodes a multi-pass transmembrane ER protein. "
            "It cycles between ER and Golgi-related membranes, proposed role in lipid synthesis or protein trafficking. "
            "Interacts with coat protein complex I (COPI) for ER retrieval. "
            "Loss of CLN8 → mixed granular/curvilinear ceroid accumulation. "
            "CLINICAL PRESENTATION: "
            "NORTHERN EPILEPSY (EPMR) — Finnish founder p.Arg24Gly: "
            "Onset 5-10 yr; progressive myoclonic epilepsy, intellectual decline; "
            "Vision preserved until late stages; relatively mild cognitive involvement; "
            "Survival 35-50 yr — the SLOWEST NCL. "
            "VARIANT LATE-INFANTILE CLN8 (non-Finnish): "
            "Onset 1.5-8 yr; rapid; seizures, visual loss, motor regression; "
            "Similar to CLN6 clinically; death in childhood/early teens. "
            "DIAGNOSIS: "
            "Both enzyme assays normal; CLN8 gene sequencing; "
            "Finnish EPMR: targeted p.Arg24Gly PCR; "
            "EM: mixed granular/curvilinear deposits (less distinctive than CLN1 GRODs or CLN2 curvilinear). "
            "MANAGEMENT: "
            "EPMR: valproate + clobazam; piracetam adjunct for myoclonus. "
            "Variant LI form: broader AED choices. "
            "Physiotherapy, cognitive rehabilitation. "
            "Gene therapy trials: AAV-CLN8. "
            "Palliative care (timing differs greatly by subtype)."
        ),
    },
    # -- CLN10 / CTSD — Congenital NCL ------------------------------------------------
    {
        "gene": "CLN10",
        "alt_name": "CTSD",
        "protein": (
            "CLN10/CTSD -- 11p15.5 AR -- CathepsinD-412aa -- "
            "Congenital-NCL-Most-Severe-Born-Seizures-Microcephaly-Death-Days-Weeks -- "
            "Juvenile-Adult-Hypomorphic-Slower -- "
            "CTSD-ONLY-NCL-Aspartyl-Protease -- "
            "Null-Mutations-Congenital-Lethal-p.Y199C-Juvenile"
        ),
        "locus": "11p15.5",
        "protein_size": "412 aa",
        "inheritance": "AR",
        "age_of_onset": "Congenital (null mutations — neonatal lethal); juvenile/adult (hypomorphic p.Y199C)",
        "key_biomarker": (
            "CTSD enzyme activity absent or severely reduced (null mutations = congenital lethal); "
            "CLN10/CTSD gene sequencing; "
            "EM: granular deposits (GRODs-like) in congenital form; "
            "brain MRI: severe microcephaly, cortical dysplasia, lissencephaly-like; "
            "p.Y199C = juvenile/adult form with slower progression"
        ),
        "pathognomonic": (
            "Congenital NCL — born with seizures + microcephaly + respiratory failure → death within days to weeks; "
            "CTSD is the ONLY NCL gene encoding an aspartyl protease (lysosomal cathepsin D); "
            "null mutations → congenital lethal; p.Y199C → juvenile/adult slower form; "
            "EM: granular amorphous deposits resembling GRODs"
        ),
        "treatment": (
            "CONGENITAL FORM: palliative care only; death within days to weeks; "
            "JUVENILE/ADULT FORM: anti-epileptics; physiotherapy; "
            "CTSD-directed therapy research (ERT possible — secreted aspartyl protease); "
            "gene therapy (AAV-CTSD); "
            "palliative care with specialist support; "
            "genetic counselling for future pregnancies (25% recurrence AR)"
        ),
        "critical_flags": [
            "CONGENITAL-LETHAL-NULL-MUTATIONS — born with seizures, lissencephaly, respiratory failure; die within days to weeks; most severe of all NCLs",
            "CTSD-ONLY-ASPARTYL-PROTEASE-IN-NCL — all other NCL genes encode serine proteases (PPT1/TPP1) or membrane proteins; CTSD is unique cathepsin D",
            "p.Y199C-JUVENILE-FORM — hypomorphic allele; residual CTSD activity; onset in childhood; slower progression; DIFFERENT prognosis from null mutations",
            "PRENATAL-DIAGNOSIS-CRITICAL — null mutations: counsel 25% recurrence; chorionic villus sampling CTSD enzyme + gene testing available; congenital form lethal",
            "MICROCEPHALY-BORN — congenital NCL: microcephaly detectable in utero by 20-week ultrasound; may prompt earlier genetic evaluation",
            "ERT-POTENTIAL-CTSD — CTSD is secreted aspartyl protease; mannose-6-phosphate receptor uptake possible; ERT research ongoing; preclinical evidence",
            "RARE-OVERALL — CLN10 congenital NCL is rare; ~30 cases published; mostly severe null homozygotes; p.Y199C juvenile form more survivable",
            "CTSD-CATHEPSIN-D-PROTEASE — normally degrades amyloid precursor protein, intracellular proteins; loss → GM2/GM3 gangliosidosis-like accumulation"
        ],
        "alias": (
            "CLN10 (CTSD — cathepsin D); OMIM gene 116840; "
            "Neuronal ceroid lipofuscinosis type 10 (congenital NCL / CLN10) OMIM 610127. "
            "11p15.5; 412 aa; ~43 kDa (mature heavy chain ~34 kDa + light chain ~14 kDa); "
            "lysosomal aspartyl protease; autosomal recessive. "
            "FUNCTION: CTSD encodes cathepsin D, a lysosomal aspartyl protease. "
            "It cleaves proteins at low pH, including amyloid precursor protein, pro-forms of other cathepsins, "
            "and various intracellular substrates. "
            "Loss of cathepsin D → accumulation of granular deposits and lipofuscin in neurons. "
            "CTSD is the ONLY NCL gene encoding a protease of the aspartyl class (all others: serine proteases or membrane proteins). "
            "CLINICAL PRESENTATION: "
            "CONGENITAL NCL (null mutations): "
            "Born with microcephaly, lissencephaly/pachygyria, tonic seizures, apnoea; "
            "Death within days to weeks of birth (rarely survives to months with intensive support); "
            "Most severe NCL — congenital lethal. "
            "JUVENILE/ADULT FORM (hypomorphic mutations — p.Y199C): "
            "Onset childhood to early adult; progressive visual failure, myoclonic epilepsy, cognitive decline; "
            "Slower course; survival into adulthood possible. "
            "DIAGNOSIS: "
            "CTSD enzyme assay (lysosomal cathepsin D activity) — absent in null forms; "
            "CTSD/CLN10 gene sequencing: null mutations (frameshift/nonsense) = congenital; p.Y199C = juvenile; "
            "EM: granular dense deposits (GRODs-like) — not curvilinear; "
            "Brain MRI (prenatal or neonatal): microcephaly, cortical simplification. "
            "MANAGEMENT: "
            "Congenital: palliative; end-of-life planning at birth. "
            "Juvenile: anti-epileptics (lamotrigine, levetiracetam); physiotherapy; palliative care. "
            "ERT research (CTSD secretable via M6P pathway). "
            "Gene therapy (AAV-CTSD) in development. "
            "Prenatal testing: chorionic villus sampling (enzyme + gene)."
        ),
    },
]


def _make_cohort(gene_entry: dict, seed: int, n: int = 40) -> list:
    rng = random.Random(seed)
    # NCL age at diagnosis varies by type
    mean_age = {
        "CLN1": 1.2,
        "CLN2": 3.2,
        "CLN3": 5.8,
        "CLN5": 5.5,
        "CLN6": 4.5,
        "CLN7": 4.8,
        "CLN8": 7.5,
        "CLN10": 0.0,  # congenital
    }.get(gene_entry["gene"], 4.0)
    sd_age = {
        "CLN1": 0.5,
        "CLN2": 0.8,
        "CLN3": 1.0,
        "CLN5": 1.2,
        "CLN6": 1.5,
        "CLN7": 1.2,
        "CLN8": 2.5,
        "CLN10": 0.05,
    }.get(gene_entry["gene"], 1.0)
    ages = [round(rng.gauss(mean_age, sd_age), 1) for _ in range(n)]
    ages = [max(0.0, min(20.0, a)) for a in ages]
    sexes = [rng.choice(["M", "F"]) for _ in range(n)]
    # CLN10 congenital: all severe
    if gene_entry["gene"] == "CLN10":
        severities = ["severe"] * n
    # CLN1: mostly severe (rapid progression)
    elif gene_entry["gene"] == "CLN1":
        severities = [rng.choices(["moderate", "severe"], weights=[1, 4])[0] for _ in range(n)]
    # CLN8 EPMR: mostly mild
    elif gene_entry["gene"] == "CLN8":
        severities = [rng.choices(["mild", "moderate", "severe"], weights=[3, 2, 1])[0] for _ in range(n)]
    else:
        severities = [rng.choice(["mild", "moderate", "severe"]) for _ in range(n)]
    return [
        {
            "patient_id": f"{gene_entry['gene']}-{i+1:03d}",
            "gene": gene_entry["gene"],
            "age_at_diagnosis_yr": ages[i],
            "sex": sexes[i],
            "severity": severities[i],
            "inheritance": gene_entry["inheritance"],
            "locus": gene_entry["locus"],
        }
        for i in range(n)
    ]


def overview() -> dict:
    all_patients = []
    for idx, g in enumerate(NCL_GENES):
        cohort = _make_cohort(g, SEED_BASE + idx)
        all_patients.extend(cohort)

    total = len(all_patients)
    gene_counts = {}
    for p in all_patients:
        gene_counts[p["gene"]] = gene_counts.get(p["gene"], 0) + 1

    age_vals = [p["age_at_diagnosis_yr"] for p in all_patients]
    avg_age = round(sum(age_vals) / len(age_vals), 1)
    severe_count = sum(1 for p in all_patients if p["severity"] == "severe")

    gene_summary = []
    for g in NCL_GENES:
        gene_summary.append({
            "gene": g["gene"],
            "alt_name": g.get("alt_name", ""),
            "locus": g["locus"],
            "protein_size": g["protein_size"],
            "inheritance": g["inheritance"],
            "age_of_onset": g["age_of_onset"],
            "key_biomarker": g["key_biomarker"],
            "pathognomonic": g["pathognomonic"],
            "treatment": g["treatment"],
            "n_patients": gene_counts.get(g["gene"], 0),
            "critical_flags": g["critical_flags"],
        })

    return {
        "atlas": "Hereditary-NCL-Atlas",
        "subtitle": (
            "Complete 8-Gene Hereditary Neuronal Ceroid Lipofuscinosis (Batten Disease) Atlas — "
            "CLN1/PPT1-306aa-1p34.2-AR-INCL-GRODs-EM-PATHOGNOMONIC-No-Therapy-Rapid-Death | "
            "CLN2/TPP1-563aa-11p15.4-AR-Late-Infantile-Cerliponase-Alfa-Brineura-ICV-FDA2017-ONLY-NCL-ERT | "
            "CLN3-438aa-16p12.1-AR-JNCL-Batten-Vision-Loss-FIRST-5yr-Before-Seizures-Sea-Blue-Histiocytes | "
            "CLN5-407aa-13q22.3-AR-Finnish-Variant-TPP1-Normal-p.Y392X-Founder | "
            "CLN6/Linclin-311aa-15q23-AR-Variant-LI-Kufs-Adult-NCL-No-Vision-Loss-Kufs | "
            "CLN7/MFSD8-518aa-4q28.2-AR-Turkish-Founder-Fingerprint-EM-Predominant | "
            "CLN8-286aa-8p23.3-AR-EPMR-Finnish-p.Arg24Gly-SLOWEST-NCL-Survival-35-50yr | "
            "CLN10/CTSD-412aa-11p15.5-AR-Congenital-NCL-Most-Severe-Die-Days-Weeks-Aspartyl-Protease | "
            "320-Patient-Aggregate-8x40-seeds-1902-1909"
        ),
        "aggregate_stats": {
            "total_patients": total,
            "genes_covered": len(NCL_GENES),
            "avg_age_at_diagnosis_yr": avg_age,
            "severe_cases_pct": round(100 * severe_count / total, 1),
            "seed_range": f"{SEED_BASE}–{SEED_BASE + len(NCL_GENES) - 1}",
        },
        "gene_summary": gene_summary,
        "key_clinical_distinctions": [
            "CLN2-ONLY-NCL-WITH-FDA-THERAPY: cerliponase alfa (Brineura) ICV biweekly — ONLY approved NCL enzyme replacement; all other NCLs: supportive only",
            "CLN3-VISION-FIRST-5yr: CLN3 (Batten) — vision loss 5 years BEFORE seizures; contrast CLN2 (seizures first); ophthalmology referral before neurology in CLN3",
            "CLN10-CONGENITAL-LETHAL: CTSD null mutations — born dying; microcephaly, seizures, respiratory failure; death within days-weeks; MOST severe NCL",
            "CLN8-EPMR-SLOWEST: Finnish p.Arg24Gly — Northern epilepsy; survival 35-50 yr; vastly different prognosis from CLN1/CLN2/CLN10; do NOT conflate",
            "CLN1-GRODS-EM: PPT1 deficiency — granular osmiophilic deposits; CLN2: curvilinear; CLN3: fingerprint; EM biopsy guides gene diagnosis",
            "TPP1-ENZYME-ASSAY-FIRST: if late-infantile NCL suspected — check TPP1 enzyme first; positive = CLN2 → cerliponase; negative → NCL gene panel",
            "CLN6-KUFS-NO-VISION: adult NCL (Kufs disease) — CLN6/CLN8 — NO retinal involvement; progressive myoclonic epilepsy + dementia; ophthalmologically normal",
            "SEA-BLUE-HISTIOCYTES-CLN3: bone marrow biopsy PATHOGNOMONIC for CLN3; do NOT confuse with Niemann-Pick type A/B (sphingomyelinase deficiency)",
            "CLN7-MFSD8-TURKISH: c.103C>T (p.Arg35Trp) Turkish founder; fingerprint EM; onset 3-6yr; similar to CLN2 clinically but enzyme normal",
            "CARBAMAZEPINE-AVOID-ALL-NCL: carbamazepine precipitates acute encephalopathy in NCL; never use; lamotrigine + levetiracetam safe alternatives",
        ],
    }


def breakdown() -> dict:
    result = []
    for idx, g in enumerate(NCL_GENES):
        cohort = _make_cohort(g, SEED_BASE + idx)
        severities = {}
        for p in cohort:
            severities[p["severity"]] = severities.get(p["severity"], 0) + 1
        result.append({
            "gene": g["gene"],
            "alt_name": g.get("alt_name", ""),
            "protein": g["protein"],
            "locus": g["locus"],
            "protein_size": g["protein_size"],
            "inheritance": g["inheritance"],
            "age_of_onset": g["age_of_onset"],
            "key_biomarker": g["key_biomarker"],
            "pathognomonic": g["pathognomonic"],
            "treatment": g["treatment"],
            "critical_flags": g["critical_flags"],
            "severity_distribution": severities,
            "n_patients": len(cohort),
            "patients": cohort[:5],
        })
    return {"genes": result, "total_genes": len(NCL_GENES)}


def definitions() -> dict:
    return {
        "atlas": "Hereditary-NCL-Atlas",
        "genes": [
            {
                "gene": g["gene"],
                "alt_name": g.get("alt_name", ""),
                "definition": g["alias"],
                "locus": g["locus"],
                "protein_size": g["protein_size"],
                "inheritance": g["inheritance"],
                "age_of_onset": g["age_of_onset"],
                "critical_flags": g["critical_flags"],
            }
            for g in NCL_GENES
        ],
        "glossary": {
            "NCL": "Neuronal ceroid lipofuscinosis — group of inherited lysosomal storage disorders characterised by progressive neurodegeneration, seizures, visual failure, and ceroid accumulation; 14 genetic forms (CLN1–CLN14); most AR; collectively 1:25,000–1:100,000",
            "Batten disease": "Broad term for NCL group, especially juvenile NCL (CLN3); eponym from Frederick Batten (1903); now used loosely for all NCLs",
            "Ceroid": "Autofluorescent lipopigment accumulating in NCL lysosomes; composed of subunit c of mitochondrial ATP synthase (major component), dolichol-pyrophosphate-linked sugars, and oxidised proteins; storage signature of NCL",
            "GRODs": "Granular osmiophilic deposits — round electron-dense granules 50-80 nm; PATHOGNOMONIC for CLN1 (PPT1 deficiency) on EM biopsy",
            "Curvilinear bodies": "Parallel curved membrane profiles on EM; PATHOGNOMONIC for CLN2 (TPP1 deficiency); stacked 'C'-shaped membranes",
            "Fingerprint profiles": "Concentric curvilinear lamellae resembling fingerprint whorls on EM; predominant in CLN3 (juvenile NCL); also seen in CLN6, CLN7",
            "Cerliponase alfa (Brineura)": "Recombinant human TPP1 — intracerebroventricular enzyme replacement for CLN2 disease; FDA/EMA approved 2017; biweekly ICV infusion via Ommaya reservoir; first CNS enzyme therapy worldwide",
            "Ommaya reservoir": "Surgically implanted subcutaneous device connected to a catheter in brain ventricle; allows repeated intracerebroventricular injections; required for cerliponase alfa delivery",
            "CLN2 rating scale": "Validated 18-point scale (motor 0-6 + language 0-6 + seizure 0-6); used to monitor CLN2 disease progression and cerliponase response; initiate therapy before severe decline",
            "EPMR": "Progressive epilepsy with intellectual disability, Finnish type — CLN8 Northern epilepsy; very slow progression; Finnish founder p.Arg24Gly; survival 35-50 yr",
            "Kufs disease": "Adult-onset NCL — progressive myoclonic epilepsy ± dementia, NO retinal involvement; type A (AR): CLN6 or CLN8; type B (AD): DNAJC5 (CLN4)",
            "Sea-blue histiocytes": "Bone marrow histiocytes containing ceroid — stain blue-green with Giemsa; PATHOGNOMONIC for CLN3 in correct clinical context; also seen in Niemann-Pick types A/B and sea-blue histiocyte syndrome",
            "Vacuolated lymphocytes": "Peripheral blood lymphocytes with cytoplasmic vacuoles — rapid inexpensive screen for CLN3; not specific alone but highly suggestive with clinical picture",
            "Giant VEPs": "Markedly enlarged visual evoked potentials (>50 µV) at 1-2 Hz flash stimulation; PATHOGNOMONIC for CLN2; amplitude decreases at higher flash rates; EEG-EP finding",
            "PPT1": "Palmitoyl-protein thioesterase 1 — CLN1 gene product; lysosomal serine thioesterase; cleaves palmitoyl residues from cysteine in proteins; deficient in INCL",
            "TPP1": "Tripeptidyl peptidase 1 — CLN2 gene product; lysosomal serine protease; cleaves N-terminal tripeptides; enzyme assay used for CLN2 diagnosis and treatment monitoring",
            "Battenin": "CLN3 gene product — lysosomal/endosomal membrane protein; function partially characterised; involved in pH regulation and arginine transport; 438 aa; most common NCL worldwide",
            "Cathepsin D (CTSD)": "CLN10 gene product — lysosomal aspartyl protease; the only NCL protein that is a protease of aspartyl class; null mutations = congenital lethal NCL; p.Y199C = juvenile form",
        },
    }


if __name__ == "__main__":
    import json
    print("=== HEREDITARY-NCL-ATLAS — OVERVIEW ===")
    print(json.dumps(overview(), indent=2)[:3000])
    print("\n=== BREAKDOWN (CLN2 — cerliponase gene) ===")
    bd = breakdown()
    cln2 = next(g for g in bd["genes"] if g["gene"] == "CLN2")
    print(json.dumps(cln2, indent=2)[:2000])
    print("\n=== DEFINITIONS (glossary sample) ===")
    df = definitions()
    print(json.dumps(list(df["glossary"].items())[:5], indent=2))
