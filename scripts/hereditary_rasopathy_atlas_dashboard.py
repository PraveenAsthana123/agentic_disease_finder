#!/usr/bin/env python3
"""Hereditary-RASopathy-Atlas — Complete 8-Gene RAS-MAPK Pathway Disorder Atlas
PTPN11  (SHP-2 / Noonan syndrome type 1; 593 aa; 12q24.13; AD;
         Most common RASopathy -- ~50% of Noonan syndrome;
         GOF SH2-domain variants dysregulate RAS-MAPK via SHP-2 phosphatase;
         JMML risk 200x in NS-PTPN11 -- screen all children;
         seed SEED_BASE+0) .
SOS1    (Son of Sevenless 1; 1333 aa; 2p22.1; AD;
         ~10-13% of Noonan syndrome (NS type 4);
         Full ectodermal involvement: keratosis pilaris, sparse eyebrows, hypertelorism;
         Normal to near-normal intelligence -- BEST cognitive prognosis in NS;
         seed SEED_BASE+1) .
RAF1    (RAF proto-oncogene serine/threonine-protein kinase; 648 aa; 3p25.2; AD;
         ~3-17% of NS; HIGHEST HCM risk -- >90% if HCM-type variant (p.S257L hotspot);
         Most common HCM-causing RASopathy gene; cardiomyopathy can be severe/fatal;
         seed SEED_BASE+2) .
RIT1    (Ras-like without CAAX 1; 219aa; 1q22; AD;
         ~5% of NS (NS type 8); clinically overlaps RAF1;
         SECOND highest HCM risk gene in RASopathies (~70-75%);
         Lymphatic anomalies: chylothorax, lymphoedema more frequent;
         seed SEED_BASE+3) .
BRAF    (B-Raf proto-oncogene serine/threonine kinase; 766 aa; 7q34; AD;
         ~75% of Cardio-facio-cutaneous (CFC) syndrome;
         Severe intellectual disability DISTINCTIVE vs. NS (NS-like but worse cognition);
         No JMML association; different hotspots than cancer BRAF-V600E;
         seed SEED_BASE+4) .
MAP2K1  (Mitogen-activated protein kinase kinase 1 / MEK1; 393 aa; 15q22.31; AD;
         ~25% of CFC syndrome (type 3); ~5-10% NS-like;
         Ichthyosis PATHOGNOMONIC for MAP2K1/MAP2K2 CFC -- the skin clue;
         Trametinib (MEK inhibitor) emerging therapy;
         seed SEED_BASE+5) .
HRAS    (Harvey rat sarcoma viral proto-oncogene; 189 aa; 11p15.5; AD;
         >80% of Costello syndrome; de novo GOF at codons 12/13/34;
         Malignant solid tumours 15-17% by age 20 -- RHABDOMYOSARCOMA + BLADDER Ca;
         Papillomata around nose/mouth post-infancy PATHOGNOMONIC;
         seed SEED_BASE+6) .
SHOC2   (Leucine-rich repeat protein SHOC-2; 580 aa; 10q25.2; AD;
         Noonan syndrome with Loose Anagen Hair (NSLH / Mazzanti syndrome);
         p.S2G (c.4A>G) founder variant in >80% of NSLH -- hotspot;
         Hair pulled out PAINLESSLY and EASILY -- anagen effluvium PATHOGNOMONIC;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 1990-1997)
"""

import random

SEED_BASE = 1990

RASOPATHY_GENES = [
    # -- PTPN11 -- SHP-2 / Noonan syndrome type 1 -------------------------------------------
    {
        "gene": "PTPN11",
        "alt_name": "PTPN11 (SHP-2 / Noonan Syndrome Type 1 -- Most Common RASopathy ~50% NS)",
        "protein": (
            "PTPN11 -- 12q24.13 AD -- PTPN11-593aa -- "
            "Noonan-Syndrome-Type1-Most-Common-RASopathy-50pct-NS -- "
            "GOF-SH2-Domain-Variants-SHP2-Phosphatase-Dysregulate-RAS-MAPK -- "
            "JMML-Risk-200x-Screen-All-NS-PTPN11-Children -- "
            "Pulmonary-Stenosis-Most-Common-CVD-60-70pct -- "
            "Coagulation-Defect-Factor-XI-Deficiency-Common"
        ),
        "locus": "12q24.13",
        "protein_size": "593 aa",
        "inheritance": "AD (autosomal dominant)",
        "age_of_onset": (
            "Prenatal: increased nuchal translucency (NT) on US -- RASopathy screen; polyhydramnios; "
            "Neonatal: congenital heart disease (pulmonary stenosis ~60-70%); chylothorax; "
            "Infancy: feeding difficulties, hypotonia, developmental delay; "
            "Childhood: short stature (final height ~162cm M / ~152cm F); learning difficulties; "
            "JMML: juvenile myelomonocytic leukaemia -- presents 0-5yrs; PTPN11 somatic + germline; "
            "Adult: facial features less distinctive; reproductive issues (cryptorchidism in males ~77%); "
            "Coagulation: Factor XI, XII, Factor V/vWF defects -- perioperative bleeding risk"
        ),
        "key_biomarker": (
            "ECG/Echo: pulmonary stenosis (domed valve, dysplastic); Doppler gradient; "
            "Height: growth curve -- final height SD; Growth hormone stimulation if GH deficient; "
            "Peripheral blood smear + FBC: monocytosis + left shift = JMML screen; "
            "Haematology: Factor VIII, XI, XII, vWF, PTT -- coagulopathy pre-op MANDATORY; "
            "Ophthalmology: strabismus, nystagmus, refractive error (ROUTINE surveillance); "
            "Molecular: PTPN11 panel (exon 3, 7, 8, 13 hotspots -- E69K, G60A, Y62D, T73I, Q79R, D61G/N/Y); "
            "Bone age: delayed -- assess GH axis if height velocity <-2 SD; "
            "Echocardiogram: HOCM (hypertrophic obstructive) if septal thickness elevated"
        ),
        "pathognomonic": (
            "Dysmorphic triad: short stature + webbed neck + pulmonary stenosis = Noonan syndrome until proven otherwise; "
            "Facial: hypertelorism, low-set posteriorly rotated ears, ptosis, downslanting palpebral fissures; "
            "DISTINGUISH from Turner syndrome: Noonan affects BOTH sexes; normal karyotype; PTPN11 GOF; "
            "DISTINGUISH from CFC/Costello: CFC has worse cognition (BRAF/MAP2K1); Costello has papillomata (HRAS); "
            "JMML clue: young NS-PTPN11 child + monocytosis + splenomegaly + immature cells = JMML workup STAT; "
            "Bleeding clue: post-operative haemorrhage in NS child = coagulation screen (factor deficiency common)"
        ),
        "treatment": (
            "Cardiology: pulmonary valvuloplasty for significant PS; dysplastic valve may need surgical repair; "
            "Growth hormone: GH therapy approved in NS -- FDA 2007 (Norditropin); improves final height 4-7cm; "
            "JMML: HSCT is curative for JMML (only treatment); "
            "MEK inhibitor (trametinib): compassionate use for JMML/tumours; clinical trials NS; "
            "Hearing: hearing aids if sensorineural hearing loss; school support if cognitive issues; "
            "Coagulopathy: FFP/factor replacement peri-operatively; DDAVP for vWD component; "
            "Cryptorchidism: orchidopexy before 12 months (fertility preservation)"
        ),
        "critical_flags": [
            "PTPN11-JMML-RISK-200X-SCREEN-ALL-CHILDREN",
            "PTPN11-COAGULOPATHY-PREOP-SCREEN-MANDATORY",
            "PTPN11-PULMONARY-STENOSIS-60-70pct",
            "PTPN11-GH-APPROVED-FDA-2007-NORDITROPIN",
            "PTPN11-TURNER-DDX-KARYOTYPE-NORMAL",
            "PTPN11-DYSPLASTIC-PULMONARY-VALVE-SURGICAL",
        ],
        "seed": SEED_BASE + 0,
    },
    # -- SOS1 -- Noonan syndrome type 4 -------------------------------------------
    {
        "gene": "SOS1",
        "alt_name": "SOS1 (Son of Sevenless 1 / Noonan Syndrome Type 4 -- Best Cognitive Prognosis NS)",
        "protein": (
            "SOS1 -- 2p22.1 AD -- SOS1-1333aa -- "
            "Noonan-Syndrome-Type4-10-13pct-NS -- "
            "RAS-GEF-GOF-Promotes-Active-GTP-RAS -- "
            "Full-Ectodermal-Involvement-Keratosis-Pilaris-Sparse-Eyebrows -- "
            "Normal-Near-Normal-Intelligence-BEST-Cognitive-Prognosis-NS -- "
            "Lower-JMML-Risk-vs-PTPN11-Reassurance"
        ),
        "locus": "2p22.1",
        "protein_size": "1333 aa",
        "inheritance": "AD (autosomal dominant)",
        "age_of_onset": (
            "Prenatal: nuchal translucency increased (less marked than PTPN11); "
            "Neonatal: cardiac defects (pulmonary stenosis, ASD); "
            "Childhood: ectodermal features emerge -- keratosis pilaris (rough follicular skin), sparse outer eyebrows; "
            "Cognition: NORMAL to near-normal in most SOS1-NS -- best neuropsychological prognosis in NS spectrum; "
            "Short stature: present but GH response generally good; "
            "Sparse/absent outer third of eyebrows: characteristic ectodermal clue for SOS1; "
            "Adult: phenotype stable; reproductive function typically preserved"
        ),
        "key_biomarker": (
            "ECG/Echo: pulmonary stenosis, ASD; HOCM less common than RAF1/RIT1; "
            "Dermatology: keratosis pilaris of arms/face -- follicular hyperkeratosis; "
            "Ophthalmology: anterior segment (iris, lens) -- keratoconus rare; "
            "Neuropsychology: formal IQ testing -- typically >70, often >90; "
            "Molecular: SOS1 gene sequencing (hotspot exons 3, 6, 9, 10 -- E433K, R552G, W729L most common); "
            "Growth: GH stimulation if growth velocity low; GH response generally better than PTPN11-NS"
        ),
        "pathognomonic": (
            "NS + keratosis pilaris + absent outer eyebrows + NORMAL cognition = SOS1 until proven otherwise; "
            "Sparse lateral eyebrows in NS = ECTODERMAL involvement = SOS1 over PTPN11; "
            "DISTINGUISH from PTPN11-NS: SOS1 better cognition, more ectodermal features, lower JMML risk; "
            "DISTINGUISH from CFC (BRAF): CFC has severe intellectual disability; SOS1 has normal intelligence; "
            "DISTINGUISH from LEOPARD/Noonan with Café-au-lait: RAS-MAPK spectrum overlap; molecular essential"
        ),
        "treatment": (
            "Cardiology: pulmonary valvuloplasty if significant PS; HOCM management if present; "
            "Growth hormone: generally good response; approved for NS; "
            "Dermatology: emollients + urea cream for keratosis pilaris (cosmetic); "
            "Education: mainstream schooling usually appropriate (normal cognition); "
            "Genetics: recurrence risk 50% (AD); genetic counselling; "
            "No JMML prophylaxis required (low risk vs. PTPN11)"
        ),
        "critical_flags": [
            "SOS1-NORMAL-COGNITION-BEST-PROGNOSIS-NS",
            "SOS1-KERATOSIS-PILARIS-ECTODERMAL-CLUE",
            "SOS1-SPARSE-OUTER-EYEBROWS-DIAGNOSTIC",
            "SOS1-LOWER-JMML-RISK-VS-PTPN11",
            "SOS1-GH-RESPONSE-GENERALLY-GOOD",
            "SOS1-MAINSTREAM-SCHOOL-APPROPRIATE",
        ],
        "seed": SEED_BASE + 1,
    },
    # -- RAF1 -- Noonan syndrome with HCM -------------------------------------------
    {
        "gene": "RAF1",
        "alt_name": "RAF1 (RAF Proto-Oncogene / Noonan Syndrome with HCM -- Highest Cardiomyopathy Risk Gene)",
        "protein": (
            "RAF1 -- 3p25.2 AD -- RAF1-648aa -- "
            "Noonan-Syndrome-3-17pct -- "
            "HIGHEST-HCM-Risk-RASopathy-90pct-If-HCM-Variant-pS257L-Hotspot -- "
            "HCM-Can-Be-Severe-Fatal-Neonatal-Biventricular -- "
            "RAF1-NOT-BRAF-Different-Hotspots-From-Cancer -- "
            "Lentigines-Phenotype-In-Some-RAF1-LEOPARD-Like"
        ),
        "locus": "3p25.2",
        "protein_size": "648 aa",
        "inheritance": "AD (autosomal dominant)",
        "age_of_onset": (
            "Prenatal: severe HCM can cause non-immune hydrops fetalis (cardiac failure); "
            "Neonatal: biventricular HCM in most RAF1-HCM variants -- severe neonatal form; "
            "Infancy: HCM progression; pulmonary stenosis co-existing; "
            "Childhood: short stature, Noonan facies, learning difficulties; "
            "HCM: ONSET in fetal/neonatal period in RAF1 -- unlike sarcomeric HCM (teenage/adult); "
            "Some RAF1: lentigines (multiple skin spots) -- overlaps LEOPARD syndrome; "
            "Adult: HCM stabilises or progresses; sudden cardiac death risk in severe HCM"
        ),
        "key_biomarker": (
            "Echo: biventricular HCM, LVOTO, HOCM, concentric hypertrophy; assess gradient; "
            "ECG: LVH pattern, ST changes, repolarisation abnormalities; "
            "BNP/NT-proBNP: cardiac stress biomarker; elevated in HCM; "
            "Holter: arrhythmia surveillance (VT, AF) -- annual in established HCM; "
            "Molecular: RAF1 sequencing (p.S257L ~40% of HCM-type RAF1 -- exon 7 hotspot; p.L613V); "
            "Prenatal: fetal echo if known RAF1 parent; increased NT on first-trimester US; "
            "Cardiac MRI: fibrosis quantification (late gadolinium enhancement) in older patients"
        ),
        "pathognomonic": (
            "NS features + BIVENTRICULAR HCM in neonate/infant = RAF1 or RIT1 until proven otherwise; "
            "HCM in Noonan syndrome = check RAF1 + RIT1 FIRST (highest-risk genes for HCM); "
            "DISTINGUISH from sarcomeric HCM (MYH7, MYBPC3): no Noonan dysmorphia; adult onset; no RASopathy pathway; "
            "DISTINGUISH from PTPN11-NS: pulmonary stenosis >> HCM in PTPN11; RAF1 >> HCM; "
            "DISTINGUISH from Pompe disease: HCM + hypotonia + elevated CK = Pompe; not Noonan; "
            "Lentigines + HCM = LEOPARD syndrome (RAF1 or PTPN11 p.T468M) -- overlap phenotype"
        ),
        "treatment": (
            "HCM: beta-blockers (atenolol) first-line for LVOTO and symptom control; "
            "Severe neonatal HCM: may require ECMO, urgent surgical myomectomy; "
            "ICD: for high-risk HCM (family history SCD, massive hypertrophy, NSVT, exercise hypotension); "
            "Mavacamten (MyoKardia): cardiac myosin inhibitor for LVOTO -- emerging in RASopathy HCM; "
            "Pulmonary stenosis: valvuloplasty if significant; "
            "Growth hormone: approved for NS; use with caution if HCM present (GH may worsen HCM); "
            "Heart transplant: last resort for refractory HCM heart failure"
        ),
        "critical_flags": [
            "RAF1-HCM-RISK-90pct-HIGHEST-RASOPATHY-HCM-GENE",
            "RAF1-BIVENTRICULAR-HCM-NEONATAL-FATAL",
            "RAF1-PS257L-HOTSPOT-EXON7",
            "RAF1-GH-CAUTION-MAY-WORSEN-HCM",
            "RAF1-ICD-HIGH-RISK-HCM",
            "RAF1-DIFFERENT-HOTSPOTS-FROM-CANCER-BRAF",
        ],
        "seed": SEED_BASE + 2,
    },
    # -- RIT1 -- Noonan syndrome type 8 -------------------------------------------
    {
        "gene": "RIT1",
        "alt_name": "RIT1 (Ras-like Without CAAX / Noonan Syndrome Type 8 -- Second Highest HCM Risk)",
        "protein": (
            "RIT1 -- 1q22 AD -- RIT1-219aa -- "
            "Noonan-Syndrome-Type8-5pct-NS -- "
            "SECOND-Highest-HCM-Risk-70-75pct -- "
            "Lymphatic-Anomalies-Chylothorax-Lymphoedema-More-Frequent -- "
            "Neonatal-Pulmonary-Oedema-And-HCM-Combination -- "
            "Facial-Severe-Round-Face-Low-Set-Ears"
        ),
        "locus": "1q22",
        "protein_size": "219 aa",
        "inheritance": "AD (autosomal dominant)",
        "age_of_onset": (
            "Prenatal: very high NT (nuchal translucency) -- often >99th centile; hydrops fetalis risk; "
            "Neonatal: HCM + chylothorax combination; pulmonary oedema; need respiratory support; "
            "Infancy: lymphatic issues more prominent vs. other NS genes -- lymphoedema, pleural effusions; "
            "HCM: SECOND highest risk after RAF1 -- 70-75% of RIT1 patients; "
            "Developmental: delay common; learning difficulties; hypotonia; "
            "Short stature: present; GH approved and indicated; "
            "Facial: more severe round/full face than PTPN11-NS; broad forehead"
        ),
        "key_biomarker": (
            "Echo: HCM prevalence 70-75%; biventricular pattern (like RAF1); severity variable; "
            "Chest imaging: chylothorax -- milky pleural fluid on tap; lymphoscintigraphy if lymphoedema; "
            "Lymphocyte count: CD4 low if persistent chylothorax (lymphocyte loss into chyle); "
            "Albumin: low if protein-losing chylothorax; "
            "Molecular: RIT1 sequencing (p.A57G, p.F82L, p.M90I hotspots -- exon 5); "
            "Prenatal: first-trimester NT >3.5mm in any fetus -- RASopathy panel; "
            "Cardiac MRI: fibrosis in HCM if established cardiomyopathy"
        ),
        "pathognomonic": (
            "NS + HCM + CHYLOTHORAX in neonatal period = RIT1 until proven otherwise; "
            "Very high NT (>99th centile) on prenatal US in any sex = RASopathy screen including RIT1; "
            "DISTINGUISH from RAF1: clinically very similar; RIT1 has more lymphatic features; molecular confirms; "
            "DISTINGUISH from Turner syndrome: RIT1 affects both sexes; Turner 45X/mosaic; different cardiac (CoA); "
            "DISTINGUISH from congenital lymphoedema (Milroy): Milroy has FLT4 gene; no Noonan dysmorphia; "
            "RAF1 vs. RIT1: both highest HCM risk; RIT1 adds lymphatic anomaly burden"
        ),
        "treatment": (
            "HCM: beta-blockers, myomectomy if LVOTO; same approach as RAF1; "
            "Chylothorax: low-fat MCT diet; octreotide (somatostatin analogue) to reduce chyle flow; "
            "Pleurodesis or surgical ligation: for refractory chylothorax; "
            "Lymphoedema: compression, physiotherapy, lymphatic mapping; "
            "Pulmonary: respiratory support in neonatal period (CPAP, mechanical ventilation); "
            "Growth hormone: approved for NS; "
            "Nutritional support: TPN if severe chylothorax with protein/lymphocyte loss"
        ),
        "critical_flags": [
            "RIT1-SECOND-HIGHEST-HCM-RISK-70-75pct",
            "RIT1-CHYLOTHORAX-LYMPHATIC-HALLMARK",
            "RIT1-HIGH-NT-PRENATAL-SCREEN",
            "RIT1-NEONATAL-COMBINED-HCM-CHYLOTHORAX",
            "RIT1-MCT-DIET-OCTREOTIDE-CHYLOTHORAX",
            "RIT1-RAF1-DDX-MOLECULAR-REQUIRED",
        ],
        "seed": SEED_BASE + 3,
    },
    # -- BRAF -- Cardio-facio-cutaneous syndrome -------------------------------------------
    {
        "gene": "BRAF",
        "alt_name": "BRAF (B-Raf / Cardio-Facio-Cutaneous Syndrome -- Most Common CFC Gene ~75%)",
        "protein": (
            "BRAF -- 7q34 AD -- BRAF-766aa -- "
            "Cardio-Facio-Cutaneous-Syndrome-Most-Common-CFC-Gene-75pct -- "
            "Severe-Intellectual-Disability-DISTINCTIVE-CFC-Not-NS -- "
            "Sparse-Absent-Eyebrows-Curly-Sparse-Hair-Hyperkeratosis -- "
            "BRAF-CFC-Hotspots-Different-From-Cancer-V600E -- "
            "No-JMML-Association-Unlike-PTPN11"
        ),
        "locus": "7q34",
        "protein_size": "766 aa",
        "inheritance": "AD (autosomal dominant, almost always de novo)",
        "age_of_onset": (
            "Prenatal: increased NT, polyhydramnios, reduced fetal movement; "
            "Neonatal: severe hypotonia, poor feeding, cardiac defect; "
            "Infancy: severe intellectual disability emerging; seizures (30-50%); "
            "Ectodermal features: absent/sparse eyebrows, curly sparse scalp hair, hyperkeratosis, ichthyosis; "
            "COGNITION: severe ID in most CFC patients -- distinguishes from Noonan syndrome; "
            "Short stature: universally severe; "
            "Cardiac: pulmonary stenosis, HCM; similar to NS but more severe phenotype overall"
        ),
        "key_biomarker": (
            "ECG/Echo: PS, HCM, ASD; "
            "Neurology: EEG if seizures; brain MRI (periventricular white matter abnormalities, ventriculomegaly); "
            "Neuropsychology: severe ID typically -- formal assessment; "
            "Dermatology: ichthyosis, hyperkeratosis biopsy if uncertain; "
            "Molecular: BRAF sequencing (exons 11, 15 -- p.Q257R, p.E501K, p.D638E hotspots; NOT V600E which is cancer); "
            "Ophthalmology: nystagmus, strabismus, optic nerve hypoplasia; "
            "GH axis: multiple pituitary hormone deficiency possible"
        ),
        "pathognomonic": (
            "NS features + SEVERE intellectual disability + absent eyebrows + sparse curly hair = CFC syndrome; "
            "CFC triad: cardiac defect + facial (sparse/absent eyebrows) + ectodermal (ichthyosis/hyperkeratosis); "
            "DISTINGUISH from NS: NS has milder/no ID; CFC has severe ID; both RASopathy; different gene; "
            "DISTINGUISH from Costello syndrome (HRAS): Costello has papillomata, HRAS tumour risk; CFC has ichthyosis, no tumour risk; "
            "BRAF V600E = CANCER mutation (melanoma, colorectal) -- NOT the Noonan/CFC variants; "
            "CFC BRAF variants are activating but different sites from oncogenic V600E -- critical to distinguish"
        ),
        "treatment": (
            "Cardiac: pulmonary valvuloplasty for PS; HCM management; "
            "Neurology: anti-epileptic drugs for seizures; MEK inhibitor trials for severe CFC; "
            "MEK inhibitor (trametinib): early clinical trials in CFC -- compassionate use for life-threatening manifestations; "
            "Dermatology: emollients, topical keratolytics for ichthyosis/hyperkeratosis; "
            "Education/therapy: specialist intellectual disability input; physiotherapy, OT, speech therapy; "
            "Growth hormone: consider for short stature; caution if HCM; "
            "Ophthalmology: glasses/patching for amblyopia"
        ),
        "critical_flags": [
            "BRAF-CFC-SEVERE-ID-NOT-NS",
            "BRAF-V600E-CANCER-VARIANT-NOT-RASOPATHY-VARIANT",
            "BRAF-CFC-ABSENT-EYEBROWS-ICHTHYOSIS-ECTODERMAL",
            "BRAF-TRAMETINIB-MEK-INHIBITOR-TRIALS",
            "BRAF-NO-JMML-UNLIKE-PTPN11",
            "BRAF-DE-NOVO-ALMOST-ALWAYS",
        ],
        "seed": SEED_BASE + 4,
    },
    # -- MAP2K1 -- CFC syndrome type 3 -------------------------------------------
    {
        "gene": "MAP2K1",
        "alt_name": "MAP2K1 (MEK1 / CFC Syndrome Type 3 -- Ichthyosis PATHOGNOMONIC)",
        "protein": (
            "MAP2K1 -- 15q22.31 AD -- MAP2K1-393aa -- "
            "Cardio-Facio-Cutaneous-Type3-25pct-CFC-5-10pct-NS-Like -- "
            "Ichthyosis-PATHOGNOMONIC-For-MAP2K1-MAP2K2-CFC -- "
            "MEK1-Kinase-GOF-Phosphorylates-ERK -- "
            "Trametinib-MEK-Inhibitor-Emerging-Therapy -- "
            "MAP2K2-Also-Causes-CFC-Test-Both"
        ),
        "locus": "15q22.31",
        "protein_size": "393 aa",
        "inheritance": "AD (autosomal dominant, de novo)",
        "age_of_onset": (
            "Prenatal: NT increased; polyhydramnios; "
            "Neonatal: cardiac defect, hypotonia, poor feeding; "
            "Infancy: ichthyosis skin changes appear early -- dry scaly skin, hyperkeratosis; "
            "Childhood: intellectual disability (severe, like BRAF-CFC); seizures; "
            "Ectodermal: ichthyosis is the defining skin sign for MAP2K1/MAP2K2 in CFC; "
            "Sparse/absent eyebrows also present; "
            "Cardiac: PS, HCM, ASD -- similar spectrum to BRAF-CFC"
        ),
        "key_biomarker": (
            "ECG/Echo: PS, HCM; "
            "Dermatology: ichthyosis biopsy -- lamellar/epidermolytic pattern; "
            "Neurology: EEG, brain MRI (white matter); "
            "Molecular: MAP2K1 sequencing (exon 2, 3, 6 -- p.P124S, p.Y130C, p.Q56P hotspots); "
            "Also test MAP2K2 (MEK2) -- same disease, different gene (15-20% of CFC); "
            "Ophthalmology: nystagmus, strabismus; "
            "GH: pituitary evaluation for multiple hormone deficiency"
        ),
        "pathognomonic": (
            "CFC features + ICHTHYOSIS = MAP2K1 or MAP2K2 until proven otherwise; "
            "Ichthyosis is the skin fingerprint for MEK1/MEK2 CFC within RASopathy spectrum; "
            "DISTINGUISH from BRAF-CFC: ichthyosis more prominent in MAP2K1/2; BRAF has hyperkeratosis not true ichthyosis; molecular required; "
            "DISTINGUISH from isolated ichthyosis (TGM1, ABCA12): those lack cardiac/facial RASopathy features; "
            "MEK inhibitor (trametinib) rationale: MAP2K1 IS MEK1 -- direct target of trametinib; "
            "MAP2K1 vs. MAP2K2: clinically indistinguishable -- always test both genes on CFC panel"
        ),
        "treatment": (
            "MEK inhibitor (trametinib): DIRECT mechanistic rationale -- MAP2K1 encodes MEK1; trials ongoing; "
            "Dermatology: retinoids for ichthyosis (acitretin); emollients; topical keratolytics; "
            "Cardiology: PS valvuloplasty; HCM beta-blocker; "
            "Neurology: AED for seizures; developmental support; "
            "Education: intensive ID support; physiotherapy, OT, SLT; "
            "Ophthalmology: amblyopia management"
        ),
        "critical_flags": [
            "MAP2K1-ICHTHYOSIS-PATHOGNOMONIC-MEK1-CFC",
            "MAP2K1-TRAMETINIB-DIRECT-MECHANISTIC-TARGET",
            "MAP2K1-TEST-MAP2K2-ALSO-BOTH-GENES",
            "MAP2K1-MEK1-NOT-BRAF-V600E-DIFFERENT-TARGET",
            "MAP2K1-DE-NOVO-ALWAYS",
            "MAP2K1-RETINOIDS-ICHTHYOSIS",
        ],
        "seed": SEED_BASE + 5,
    },
    # -- HRAS -- Costello syndrome -------------------------------------------
    {
        "gene": "HRAS",
        "alt_name": "HRAS (Harvey RAS / Costello Syndrome -- TUMOUR RISK 15-17% by Age 20)",
        "protein": (
            "HRAS -- 11p15.5 AD -- HRAS-189aa -- "
            "Costello-Syndrome-Over-80pct-HRAS-De-Novo-GOF -- "
            "Malignant-Tumour-Risk-15-17pct-By-Age-20-Rhabdomyosarcoma-Bladder-Ca -- "
            "Papillomata-Around-Nose-Mouth-Post-Infancy-PATHOGNOMONIC -- "
            "HRAS-Codons-12-13-34-GOF-Hotspots -- "
            "Severe-ID-Like-CFC-NOT-Like-NS"
        ),
        "locus": "11p15.5",
        "protein_size": "189 aa",
        "inheritance": "AD (autosomal dominant, >95% de novo)",
        "age_of_onset": (
            "Prenatal: severe polyhydramnios, macrosomia, increased NT; "
            "Neonatal: severe hypotonia, macrosomia at birth then failure to thrive paradox; "
            "Infancy: failure to thrive, severe hypotonia, developmental delay; "
            "Infancy-Childhood: papillomata appear 1-5yrs -- around nose/mouth/perianal -- PATHOGNOMONIC; "
            "Childhood: tumour surveillance begins; rhabdomyosarcoma most common malignancy; "
            "Skin: coarse facies, deep palmar/plantar creases (DISTINCTIVE); "
            "Cardiac: HCM; PS; arrhythmias (multifocal atrial tachycardia specific to Costello)"
        ),
        "key_biomarker": (
            "Cardiac: Echo (HCM, PS), Holter (multifocal atrial tachycardia -- Costello-specific arrhythmia); "
            "Tumour surveillance: abdominal/pelvic US 3-6 monthly for rhabdomyosarcoma; urinalysis annually (bladder Ca); "
            "Urinalysis: haematuria, cytology for bladder carcinoma (transitional cell Ca); "
            "Molecular: HRAS sequencing (codons 12: p.G12S most common; codon 13: p.G13C; codon 34: p.Q22K); "
            "Hypoglycaemia: fasting glucose -- hypoglycaemia in neonatal/infant period; "
            "CK: mild elevation possible; "
            "Dermatology: papilloma biopsy if uncertain; wart-like HPV-negative lesions"
        ),
        "pathognomonic": (
            "NS-like features + PAPILLOMATA around nose/mouth + DEEP PALMAR CREASES = Costello syndrome; "
            "Perinasal/perioral wart-like papillomata in child with dysmorphia = HRAS until proven otherwise; "
            "TUMOUR RISK: highest of all RASopathies -- MANDATORY cancer surveillance from diagnosis; "
            "DISTINGUISH from CFC (BRAF): CFC has ichthyosis/hyperkeratosis NOT papillomata; HRAS has papillomata not ichthyosis; "
            "DISTINGUISH from NS: NS has normal/near-normal cancer risk; Costello has 15-17% tumour risk; "
            "Multifocal atrial tachycardia: highly specific for Costello -- not seen in other NS spectrum; "
            "HRAS codons 12/13 also mutated in bladder, thyroid cancer (somatic) -- germline GOF causes Costello"
        ),
        "treatment": (
            "Tumour surveillance: abdominal US + urinalysis every 6 months; "
            "Rhabdomyosarcoma treatment: chemotherapy (vincristine/actinomycin/cyclophosphamide) + surgery + radiation; "
            "Bladder carcinoma: cystoscopy + resection; "
            "Cardiac: beta-blocker or flecainide for MAT (multifocal atrial tachycardia); HCM management; "
            "Papillomata: observation; surgical/laser removal if symptomatic/cosmetic; not HPV-vaccine responsive; "
            "Growth hormone: SHORT STATURE but GH use controversial in Costello (HRAS oncogene -- theoretical tumour promotion risk); "
            "Hypoglycaemia: glucose supplementation in neonatal period"
        ),
        "critical_flags": [
            "HRAS-TUMOUR-RISK-15-17pct-MANDATORY-SURVEILLANCE",
            "HRAS-RHABDOMYOSARCOMA-BLADDER-Ca-MOST-COMMON",
            "HRAS-PAPILLOMATA-PERINASAL-PATHOGNOMONIC",
            "HRAS-GH-CONTROVERSIAL-ONCOGENE-RISK",
            "HRAS-MULTIFOCAL-ATRIAL-TACHYCARDIA-SPECIFIC",
            "HRAS-DEEP-PALMAR-CREASES-DISTINCTIVE",
        ],
        "seed": SEED_BASE + 6,
    },
    # -- SHOC2 -- Noonan syndrome with loose anagen hair -------------------------------------------
    {
        "gene": "SHOC2",
        "alt_name": "SHOC2 (SHOC-2 / Noonan Syndrome with Loose Anagen Hair -- Hair Pulled Painlessly PATHOGNOMONIC)",
        "protein": (
            "SHOC2 -- 10q25.2 AD -- SHOC2-580aa -- "
            "Noonan-Syndrome-With-Loose-Anagen-Hair-NSLH-Mazzanti-Syndrome -- "
            "pS2G-c4A>G-Founder-Variant-Over-80pct-NSLH-Hotspot -- "
            "Hair-Pulled-Out-Painlessly-Easily-Anagen-Effluvium-PATHOGNOMONIC -- "
            "Intellectual-Disability-More-Prominent-vs-Classic-NS -- "
            "Anagen-Hair-Root-On-Pulling-Diagnostic-Microscopy"
        ),
        "locus": "10q25.2",
        "protein_size": "580 aa",
        "inheritance": "AD (autosomal dominant, de novo or inherited)",
        "age_of_onset": (
            "Prenatal: increased NT, normal or mildly increased; "
            "Neonatal: cardiac defect (PS, ASD); hypotonia; "
            "Infancy: HAIR easily pulled painlessly -- caregiver notices first; "
            "Childhood: intellectual disability more prominent than classic NS; learning difficulties; "
            "Hair: sparse, thin, light-coloured, easily extractable -- the defining feature; "
            "Short stature: universal in NSLH; "
            "Ectodermal: keratosis pilaris (like SOS1); but hair finding is the pathognomonic differentiator"
        ),
        "key_biomarker": (
            "Trichogram: pull test -- 10-20 hairs easily removed PAINLESSLY; microscopy shows anagen bulb (not telogen) = anagen effluvium pattern; "
            "ECG/Echo: PS, ASD, HOCM; "
            "Neuropsychology: formal IQ -- usually below average to moderate ID (worse than PTPN11-NS); "
            "Molecular: SHOC2 sequencing (p.S2G in >80% of NSLH -- exon 2 hotspot c.4A>G; also p.M173I); "
            "Light microscopy: anagen hair with misshapen/triangular cross-section; cuticle abnormalities; "
            "Skin: keratosis pilaris (rough follicular skin); eczema association"
        ),
        "pathognomonic": (
            "NS features + HAIR PULLED PAINLESSLY WITHOUT RESISTANCE + anagen bulb on microscopy = NSLH/SHOC2; "
            "Painless easy hair extraction in Noonan-like child = SHOC2 p.S2G until proven otherwise; "
            "Anagen effluvium pattern on trichogram = anagen loose hair syndrome = SHOC2 in RASopathy context; "
            "DISTINGUISH from alopecia areata: AA has telogen hair, regrowth typical, no Noonan features; "
            "DISTINGUISH from NS-PTPN11: PTPN11 does not have hair pulling phenomenon; cognition better in PTPN11; "
            "SHOC2 p.S2G >80% of NSLH -- highly recurrent founder/hotspot mutation; targeted assay possible"
        ),
        "treatment": (
            "Hair: no curative treatment for loose anagen hair -- avoidance of trauma; gentle handling; "
            "Short stature: GH therapy approved for NS spectrum; "
            "Cardiac: PS valvuloplasty; HOCM beta-blocker; "
            "Education: intellectual disability support; moderate cognitive difficulties; "
            "Dermatology: emollients for keratosis pilaris; "
            "Genetics: 50% recurrence risk if inherited; counselling"
        ),
        "critical_flags": [
            "SHOC2-PAINLESS-HAIR-EXTRACTION-PATHOGNOMONIC",
            "SHOC2-pS2G-HOTSPOT-80pct-NSLH",
            "SHOC2-ANAGEN-EFFLUVIUM-TRICHOGRAM-DIAGNOSTIC",
            "SHOC2-ID-MORE-PROMINENT-VS-PTPN11-NS",
            "SHOC2-NSLH-MAZZANTI-SYNDROME",
            "SHOC2-TARGETED-ASSAY-pS2G-FIRST",
        ],
        "seed": SEED_BASE + 7,
    },
]


def _generate_cohort(gene_entry: dict) -> list:
    """Generate 40-patient synthetic cohort for the given RASopathy gene."""
    rng = random.Random(gene_entry["seed"])
    gene = gene_entry["gene"]
    cohort = []

    for i in range(40):
        age = rng.randint(0, 55)
        sex = rng.choice(["M", "F"])

        # Gene-specific clinical features
        if gene == "PTPN11":
            pulmonary_stenosis = rng.random() < 0.65
            hcm = rng.random() < 0.12
            jmml = age < 10 and rng.random() < 0.04
            cognitive_impairment = rng.random() < 0.30
            short_stature = rng.random() < 0.90
            coagulopathy = rng.random() < 0.35
            cryptorchidism = sex == "M" and rng.random() < 0.77
            gh_therapy = rng.random() < 0.50
            keratosis_pilaris = rng.random() < 0.15
            papillomata = False
            loose_anagen_hair = False
            tumour_malignant = jmml
            chylothorax = rng.random() < 0.10
            ichthyosis = False
            severe_id = rng.random() < 0.05

        elif gene == "SOS1":
            pulmonary_stenosis = rng.random() < 0.55
            hcm = rng.random() < 0.08
            jmml = False
            cognitive_impairment = rng.random() < 0.15
            short_stature = rng.random() < 0.88
            coagulopathy = rng.random() < 0.15
            cryptorchidism = sex == "M" and rng.random() < 0.60
            gh_therapy = rng.random() < 0.55
            keratosis_pilaris = rng.random() < 0.70
            papillomata = False
            loose_anagen_hair = False
            tumour_malignant = False
            chylothorax = rng.random() < 0.05
            ichthyosis = False
            severe_id = rng.random() < 0.02

        elif gene == "RAF1":
            pulmonary_stenosis = rng.random() < 0.45
            hcm = rng.random() < 0.90
            jmml = False
            cognitive_impairment = rng.random() < 0.40
            short_stature = rng.random() < 0.92
            coagulopathy = rng.random() < 0.20
            cryptorchidism = sex == "M" and rng.random() < 0.65
            gh_therapy = rng.random() < 0.40
            keratosis_pilaris = rng.random() < 0.20
            papillomata = False
            loose_anagen_hair = False
            tumour_malignant = False
            chylothorax = rng.random() < 0.10
            ichthyosis = False
            severe_id = rng.random() < 0.08
            lentigines = rng.random() < 0.15
            coagulopathy = coagulopathy or False

        elif gene == "RIT1":
            pulmonary_stenosis = rng.random() < 0.50
            hcm = rng.random() < 0.72
            jmml = False
            cognitive_impairment = rng.random() < 0.45
            short_stature = rng.random() < 0.93
            coagulopathy = rng.random() < 0.20
            cryptorchidism = sex == "M" and rng.random() < 0.70
            gh_therapy = rng.random() < 0.45
            keratosis_pilaris = rng.random() < 0.20
            papillomata = False
            loose_anagen_hair = False
            tumour_malignant = False
            chylothorax = rng.random() < 0.40
            ichthyosis = False
            severe_id = rng.random() < 0.10

        elif gene == "BRAF":
            pulmonary_stenosis = rng.random() < 0.50
            hcm = rng.random() < 0.40
            jmml = False
            cognitive_impairment = rng.random() < 0.95
            short_stature = rng.random() < 0.98
            coagulopathy = rng.random() < 0.10
            cryptorchidism = sex == "M" and rng.random() < 0.70
            gh_therapy = rng.random() < 0.30
            keratosis_pilaris = rng.random() < 0.60
            papillomata = False
            loose_anagen_hair = False
            tumour_malignant = False
            chylothorax = rng.random() < 0.05
            ichthyosis = rng.random() < 0.60
            severe_id = rng.random() < 0.85

        elif gene == "MAP2K1":
            pulmonary_stenosis = rng.random() < 0.45
            hcm = rng.random() < 0.38
            jmml = False
            cognitive_impairment = rng.random() < 0.90
            short_stature = rng.random() < 0.98
            coagulopathy = rng.random() < 0.10
            cryptorchidism = sex == "M" and rng.random() < 0.68
            gh_therapy = rng.random() < 0.28
            keratosis_pilaris = rng.random() < 0.50
            papillomata = False
            loose_anagen_hair = False
            tumour_malignant = False
            chylothorax = rng.random() < 0.05
            ichthyosis = rng.random() < 0.80
            severe_id = rng.random() < 0.82

        elif gene == "HRAS":
            pulmonary_stenosis = rng.random() < 0.40
            hcm = rng.random() < 0.55
            jmml = False
            cognitive_impairment = rng.random() < 0.88
            short_stature = rng.random() < 0.98
            coagulopathy = rng.random() < 0.10
            cryptorchidism = sex == "M" and rng.random() < 0.75
            gh_therapy = rng.random() < 0.10  # controversial
            keratosis_pilaris = rng.random() < 0.20
            papillomata = rng.random() < 0.80
            loose_anagen_hair = False
            tumour_malignant = rng.random() < 0.16
            chylothorax = rng.random() < 0.05
            ichthyosis = False
            severe_id = rng.random() < 0.75

        else:  # SHOC2
            pulmonary_stenosis = rng.random() < 0.50
            hcm = rng.random() < 0.25
            jmml = False
            cognitive_impairment = rng.random() < 0.70
            short_stature = rng.random() < 0.95
            coagulopathy = rng.random() < 0.15
            cryptorchidism = sex == "M" and rng.random() < 0.65
            gh_therapy = rng.random() < 0.50
            keratosis_pilaris = rng.random() < 0.50
            papillomata = False
            loose_anagen_hair = rng.random() < 0.92
            tumour_malignant = False
            chylothorax = rng.random() < 0.08
            ichthyosis = False
            severe_id = rng.random() < 0.20

        cohort.append({
            "patient_id": f"{gene}-{gene_entry['seed']}-{i+1:03d}",
            "gene": gene,
            "age": age,
            "sex": sex,
            "pulmonary_stenosis": pulmonary_stenosis,
            "hcm": hcm,
            "jmml": jmml,
            "cognitive_impairment": cognitive_impairment,
            "severe_intellectual_disability": severe_id,
            "short_stature": short_stature,
            "coagulopathy": coagulopathy,
            "cryptorchidism": cryptorchidism,
            "gh_therapy": gh_therapy,
            "keratosis_pilaris": keratosis_pilaris,
            "papillomata": papillomata,
            "loose_anagen_hair": loose_anagen_hair,
            "tumour_malignant": tumour_malignant,
            "chylothorax": chylothorax,
            "ichthyosis": ichthyosis,
            "on_mek_inhibitor": gene in ("MAP2K1", "BRAF") and rng.random() < 0.08,
        })

    return cohort


def overview() -> dict:
    all_cohorts = [_generate_cohort(g) for g in RASOPATHY_GENES]
    total = sum(len(c) for c in all_cohorts)
    all_pts = [p for c in all_cohorts for p in c]

    hcm_n = sum(1 for p in all_pts if p["hcm"])
    ps_n = sum(1 for p in all_pts if p["pulmonary_stenosis"])
    jmml_n = sum(1 for p in all_pts if p["jmml"])
    tumour_n = sum(1 for p in all_pts if p["tumour_malignant"])
    cognitive_n = sum(1 for p in all_pts if p["cognitive_impairment"])
    severe_id_n = sum(1 for p in all_pts if p["severe_intellectual_disability"])
    short_stature_n = sum(1 for p in all_pts if p["short_stature"])
    gh_n = sum(1 for p in all_pts if p["gh_therapy"])
    papillomata_n = sum(1 for p in all_pts if p["papillomata"])
    loose_hair_n = sum(1 for p in all_pts if p["loose_anagen_hair"])
    ichthyosis_n = sum(1 for p in all_pts if p["ichthyosis"])
    chylothorax_n = sum(1 for p in all_pts if p["chylothorax"])
    mek_inhibitor_n = sum(1 for p in all_pts if p["on_mek_inhibitor"])
    gene_counts = {g["gene"]: len(_generate_cohort(g)) for g in RASOPATHY_GENES}

    return {
        "atlas": "Hereditary-RASopathy-Atlas",
        "subtitle": (
            "Complete 8-Gene RAS-MAPK Pathway Disorder Atlas: "
            "Noonan Syndrome, Cardio-Facio-Cutaneous Syndrome, and Costello Syndrome"
        ),
        "genes": [g["gene"] for g in RASOPATHY_GENES],
        "total_patients": total,
        "seeds": f"{SEED_BASE}-{SEED_BASE+7}",
        "hcm_patients": hcm_n,
        "pulmonary_stenosis_patients": ps_n,
        "jmml_patients": jmml_n,
        "tumour_patients": tumour_n,
        "cognitive_impairment_patients": cognitive_n,
        "severe_id_patients": severe_id_n,
        "short_stature_patients": short_stature_n,
        "gh_therapy_patients": gh_n,
        "papillomata_patients": papillomata_n,
        "loose_anagen_hair_patients": loose_hair_n,
        "ichthyosis_patients": ichthyosis_n,
        "chylothorax_patients": chylothorax_n,
        "mek_inhibitor_patients": mek_inhibitor_n,
        "gene_patient_counts": gene_counts,
        "pathway": (
            "All 8 genes encode RAS-MAPK signalling components: "
            "PTPN11 (SHP-2 phosphatase activates RAS via SH2-domain GOF) → "
            "SOS1 (RAS-GEF GOF locks RAS in active GTP state) → "
            "RIT1/HRAS (small GTPases — active GTP-bound signals RAF) → "
            "RAF1/BRAF (serine/threonine kinases — MAP3K level) → "
            "MAP2K1 (MEK1 — MAP2K level, phosphorylates ERK1/2) → "
            "SHOC2 (scaffold protein potentiates RAF1 dephosphorylation at pS259 — keeps RAF1 active). "
            "GOF variants in all 8 genes cause constitutive RAS-MAPK activation: "
            "→ proliferation, growth, cardiomyocyte hypertrophy, lymphangiogenesis."
        ),
        "key_clinical_insight": (
            "PTPN11 (NS type 1): JMML risk 200x — screen all NS-PTPN11 children with FBC + differential. "
            "SOS1 (NS type 4): BEST cognitive prognosis in NS — mainstream schooling usually appropriate; ectodermal clue (sparse eyebrows). "
            "RAF1 (NS+HCM): HIGHEST HCM risk (>90%) — check RAF1 first in any Noonan child with HCM. "
            "RIT1 (NS type 8): SECOND highest HCM + chylothorax combination — MCT diet + octreotide for chyle leak. "
            "BRAF (CFC): SEVERE intellectual disability — distinguishes CFC from NS; V600E is cancer not CFC. "
            "MAP2K1 (CFC type 3): ICHTHYOSIS is pathognomonic for MEK1/MEK2 CFC — trametinib direct target. "
            "HRAS (Costello): TUMOUR risk 15-17% by age 20 — MANDATORY 6-monthly surveillance; papillomata diagnostic. "
            "SHOC2 (NSLH): PAINLESS HAIR EXTRACTION pathognomonic — trichogram confirms anagen effluvium; p.S2G in >80%."
        ),
    }


def breakdown() -> dict:
    result = {}
    for gene_entry in RASOPATHY_GENES:
        cohort = _generate_cohort(gene_entry)
        gene = gene_entry["gene"]

        hcm_pct = round(100 * sum(1 for p in cohort if p["hcm"]) / len(cohort))
        ps_pct = round(100 * sum(1 for p in cohort if p["pulmonary_stenosis"]) / len(cohort))
        jmml_pct = round(100 * sum(1 for p in cohort if p["jmml"]) / len(cohort))
        tumour_pct = round(100 * sum(1 for p in cohort if p["tumour_malignant"]) / len(cohort))
        cognitive_pct = round(100 * sum(1 for p in cohort if p["cognitive_impairment"]) / len(cohort))
        severe_id_pct = round(100 * sum(1 for p in cohort if p["severe_intellectual_disability"]) / len(cohort))
        short_stature_pct = round(100 * sum(1 for p in cohort if p["short_stature"]) / len(cohort))
        gh_pct = round(100 * sum(1 for p in cohort if p["gh_therapy"]) / len(cohort))
        papillomata_pct = round(100 * sum(1 for p in cohort if p["papillomata"]) / len(cohort))
        loose_hair_pct = round(100 * sum(1 for p in cohort if p["loose_anagen_hair"]) / len(cohort))
        ichthyosis_pct = round(100 * sum(1 for p in cohort if p["ichthyosis"]) / len(cohort))
        chylothorax_pct = round(100 * sum(1 for p in cohort if p["chylothorax"]) / len(cohort))
        kp_pct = round(100 * sum(1 for p in cohort if p["keratosis_pilaris"]) / len(cohort))
        mek_pct = round(100 * sum(1 for p in cohort if p["on_mek_inhibitor"]) / len(cohort))

        result[gene] = {
            "gene": gene,
            "alt_name": gene_entry["alt_name"],
            "locus": gene_entry["locus"],
            "protein_size": gene_entry["protein_size"],
            "inheritance": gene_entry["inheritance"],
            "n_patients": len(cohort),
            "hcm_pct": hcm_pct,
            "pulmonary_stenosis_pct": ps_pct,
            "jmml_pct": jmml_pct,
            "tumour_pct": tumour_pct,
            "cognitive_impairment_pct": cognitive_pct,
            "severe_id_pct": severe_id_pct,
            "short_stature_pct": short_stature_pct,
            "gh_therapy_pct": gh_pct,
            "papillomata_pct": papillomata_pct,
            "loose_anagen_hair_pct": loose_hair_pct,
            "ichthyosis_pct": ichthyosis_pct,
            "chylothorax_pct": chylothorax_pct,
            "keratosis_pilaris_pct": kp_pct,
            "mek_inhibitor_pct": mek_pct,
            "age_of_onset": gene_entry["age_of_onset"],
            "key_biomarker": gene_entry["key_biomarker"],
            "pathognomonic": gene_entry["pathognomonic"],
            "treatment": gene_entry["treatment"],
            "critical_flags": gene_entry["critical_flags"],
            "seed": gene_entry["seed"],
            "cohort_preview": cohort[:5],
        }
    return result


def definitions() -> dict:
    return {
        "atlas": "Hereditary-RASopathy-Atlas",
        "pathway": "RAS-MAPK (Mitogen-Activated Protein Kinase) Signalling Pathway",
        "shared_mechanism": (
            "All RASopathies share gain-of-function dysregulation of the RAS-ERK cascade: "
            "Ligand → RTK → (SOS1/SHP-2 → RAS-GTP) → RAF1/BRAF → MEK1/MEK2 → ERK1/ERK2 → "
            "nucleus (proliferation, differentiation, survival). "
            "GOF variants cause constitutive pathway activation without ligand input."
        ),
        "genes": {
            g["gene"]: {
                "full_name": g["alt_name"],
                "locus": g["locus"],
                "protein_size": g["protein_size"],
                "inheritance": g["inheritance"],
                "critical_flags": g["critical_flags"],
                "pathognomonic": g["pathognomonic"],
                "treatment_summary": g["treatment"],
            }
            for g in RASOPATHY_GENES
        },
        "glossary": {
            "RASopathy": "Group of conditions caused by GOF germline variants in RAS-MAPK pathway genes; shared features: short stature, cardiac defects, facial dysmorphia, variable cognition",
            "Noonan syndrome (NS)": "Most common RASopathy; pulmonary stenosis + short stature + characteristic facies; PTPN11 #1 cause (~50%)",
            "CFC syndrome": "Cardio-Facio-Cutaneous; BRAF (~75%) or MAP2K1/2/KRAS; severe ID distinguishes from NS; ectodermal features",
            "Costello syndrome": "HRAS GOF; papillomata (pathognomonic); highest solid tumour risk of all RASopathies (15-17% by age 20)",
            "NSLH": "Noonan syndrome with Loose Anagen Hair (Mazzanti syndrome); SHOC2 p.S2G hotspot; painless hair extraction pathognomonic",
            "JMML": "Juvenile Myelomonocytic Leukaemia — high risk in PTPN11-NS (200x population); HSCT curative",
            "HCM": "Hypertrophic cardiomyopathy — highest risk genes: RAF1 (~90%), RIT1 (~72%); biventricular onset neonatal period",
            "GOF": "Gain-of-function — variant increases RAS-MAPK signalling constitutively without ligand",
            "MEK inhibitor": "Trametinib targets MAP2K1 (MEK1) and MAP2K2 (MEK2) — rational therapy for MAP2K1 CFC and trials in PTPN11-JMML",
            "Anagen effluvium": "Hair loss during growth phase (anagen) — hair pulled painlessly; not the same as telogen effluvium; trichogram diagnostic",
            "Chylothorax": "Pleural effusion of chyle (lymphatic fluid + fat) — RIT1 > other RASopathies; MCT diet + octreotide first-line",
            "Nuchal translucency (NT)": "Ultrasound measure of nuchal fluid 11-14 weeks; elevated in all RASopathies — prenatal screening trigger",
            "Papillomata": "Wart-like perinasal/perioral skin lesions in Costello syndrome (HRAS); HPV-negative; pathognomonic for Costello",
            "Ichthyosis": "Scaly dry skin from defective cornification; MAP2K1 > BRAF CFC; keratolytics + retinoids",
            "Keratosis pilaris": "Follicular hyperkeratosis (rough skin); SOS1 >> other RASopathies; cosmetically managed",
            "Growth hormone therapy": "FDA-approved for NS (Norditropin 2007); use with caution in RAF1-HCM (may worsen HCM) and HRAS-Costello (theoretical oncogene concern)",
            "Multifocal atrial tachycardia": "Arrhythmia highly specific to Costello syndrome (HRAS) — multiple ectopic atrial foci; beta-blocker/flecainide",
            "Pulmonary stenosis (PS)": "Congenital pulmonary valve stenosis — most common cardiac defect in NS (60-70% PTPN11); dysplastic valve may need surgery over valvuloplasty",
            "Cryptorchidism": "Undescended testes — common in all NS-spectrum males (~65-77%); orchidopexy <12 months mandatory",
            "Rhabdomyosarcoma": "Most common malignancy in Costello syndrome — soft tissue sarcoma; 6-monthly surveillance from diagnosis",
            "Deep palmar creases": "Prominent deep creases of palm/sole — distinctive for Costello syndrome; also seen in Down syndrome but context different",
        },
        "surveillance_protocols": {
            "PTPN11": "Annual FBC + diff (JMML screen); echo annually if HCM/PS; coagulation pre-op; growth monitoring",
            "SOS1": "Echo annually; growth/GH axis; education assessment; dermatology for KP",
            "RAF1": "Echo 6-monthly in HCM; Holter annually; GH with caution; ICD risk stratification",
            "RIT1": "Echo 6-monthly in HCM; chest imaging chylothorax; lymphatic mapping",
            "BRAF": "Echo; neurology/EEG; educational assessment (ID support); ophthalmology; dermatology",
            "MAP2K1": "Echo; neurology/EEG; dermatology (ichthyosis); MEK inhibitor trial eligibility",
            "HRAS": "Abdominal/pelvic US 6-monthly; urinalysis annually; echo + Holter; tumour surveillance MANDATORY",
            "SHOC2": "Trichogram if uncertain; echo; neuropsychology; GH; dermatology (KP)",
        },
    }


if __name__ == "__main__":
    import json
    ov = overview()
    print(f"Atlas: {ov['atlas']}")
    print(f"Total patients: {ov['total_patients']}")
    print(f"Seeds: {ov['seeds']}")
    print(f"Genes: {', '.join(ov['genes'])}")
    print(f"HCM patients: {ov['hcm_patients']}")
    print(f"JMML patients: {ov['jmml_patients']}")
    print(f"Tumour patients: {ov['tumour_patients']}")
    print(f"Papillomata patients: {ov['papillomata_patients']}")
    print(f"Loose anagen hair patients: {ov['loose_anagen_hair_patients']}")
    print("Breakdown gene keys:", list(breakdown().keys()))
    print("Definitions gene keys:", list(definitions()["genes"].keys()))
