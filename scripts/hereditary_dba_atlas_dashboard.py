"""Hereditary Diamond-Blackfan Anaemia (DBA) & Ribosomopathy Atlas — 8-Gene Reference
RPS19-RPL5-RPL11-RPS26-RPL35A-RPS17-RPL26-TSR2
Pure Red Cell Aplasia / Ribosomopathy / DBA Spectrum
320 patients (8 x 40), seeds 2806-2813.
Endpoints: /api/hereditary-dba-atlas/overview|breakdown|definitions
"""
import random

ATLAS_GENES = [
    {
        "gene": "RPS19",
        "protein": (
            "RPS19 -- 19q13.2 AD -- 145aa -- 40S-Ribosomal-Protein-S19-16kDa-"
            "Small-Subunit-Scaffold-Nucleolar-Stress-p53-Axis-"
            "OMIM-Gene-603474-Disease-DBA1-105650"
        ),
        "locus": "19q13.2",
        "protein_size": "145 aa / 16 kDa (40S ribosomal small subunit; nucleolar stress sensor; p53-stabilizer under haploinsufficiency; direct interaction with RPS14 and eS26; mutations cluster in C-terminus interface)",
        "inheritance": (
            "AUTOSOMAL DOMINANT (haploinsufficiency) — most common DBA gene (~25% of all DBA); "
            "RPS19 encodes Ribosomal Protein S19, a structural component of the 40S small ribosomal subunit; "
            "MECHANISM: RPS19 haploinsufficiency → impaired 18S rRNA processing (accumulation of 21S pre-rRNA) → "
            "  → ribosomal stress → FREE RPL5 and RPL11 release from nucleolus → "
            "  → RPL5+RPL11 bind MDM2 → MDM2 inhibited → p53 stabilised → "
            "  → p53 target gene transcription → G1 cell cycle arrest → erythroid progenitor apoptosis; "
            "ERYTHROID SPECIFICITY: erythroid progenitors have highest ribosomal demand during rapid proliferation; "
            "  haploinsufficiency threshold crossed selectively in erythroid lineage; "
            "PREVALENCE: DBA overall ~5-7 per million live births; RPS19 = most common single-gene cause; "
            "MUTATION TYPES: missense (~50%), frameshift (~25%), nonsense, splice site, large deletions; "
            "DE NOVO RATE: ~45% of RPS19 cases are de novo; "
            "STEROID RESPONSE: 40-60% respond to prednisolone (2 mg/kg/day → taper to alternate day); "
            "LEUCINE SUPPLEMENTATION: phase 2 trial showed partial response; activates mTORC1 → bypasses RP-stress; "
            "CURATIVE: allogeneic HSCT (HLA-matched sibling preferred; unrelated donor acceptable); "
            "CANCER RISK: MDS/AML (2-5% cumulative by age 50), osteosarcoma, colon cancer — long-term surveillance mandatory; "
            "REGISTRIES: Diamond-Blackfan Anemia Registry (DBAR) since 1992 — oldest rare disease registry; "
            "KEY MUTATIONS: R62W (most common missense); c.2T>C (initiator codon); del exon 2-3 (recurrent)"
        ),
        "disease_category": (
            "DIAMOND-BLACKFAN ANAEMIA TYPE 1 (DBA1) — OMIM 105650; "
            "AGE OF ONSET: first year of life (median 2-3 months); "
            "HAEMATOLOGY — DIAGNOSTIC CRITERIA (Alter 1987 + 2001): "
            "  1. Normochromic macrocytic anaemia presenting <1 year of age; "
            "  2. Reticulocytopenia (reticulocyte count <10,000/μL); "
            "  3. Near-absent erythroid precursors on bone marrow aspirate (erythroblastopenia); "
            "  4. Normal or mildly decreased platelet and neutrophil counts (NOT pancytopenia); "
            "SUPPORTING CRITERIA: "
            "  Elevated erythrocyte adenosine deaminase (eADA) — PATHOGNOMONIC when elevated; "
            "    eADA >1.04 U/g Hb (age-adjusted): 80-85% sensitivity; 95% specificity for DBA vs other PRCA; "
            "  Elevated fetal haemoglobin (HbF) persistence; "
            "  Elevated MCV/MCH relative to age even before profound anaemia; "
            "  i-antigen expression on red cells (fetal pattern persistence); "
            "CONGENITAL ANOMALIES (~50% of DBA patients): "
            "  Upper limb anomalies: triphalangeal thumb (PATHOGNOMONIC for DBA); flat thenar eminence; "
            "  Craniofacial: cleft palate, microcephaly, abnormal ears; "
            "  Cardiac defects (30%): VSD, ASD, coarctation of aorta; "
            "  Renal anomalies (15%); "
            "  Short stature (very common; corticosteroids worsen growth); "
            "  Eye anomalies: strabismus, glaucoma; "
            "PROGNOSIS: "
            "  Steroid-responsive: ongoing low-dose prednisolone; growth monitoring essential; "
            "  Transfusion-dependent: chronic iron overload → chelation (deferasirox/deferoxamine); "
            "  HSCT: curative for non-responders; best results sibling donor <10 years of age; "
            "LONG-TERM CANCER RISK: MDS/AML, osteosarcoma, colorectal cancer — annual surveillance from age 20"
        ),
        "disease_pathway": (
            "RIBOSOMAL STRESS / p53 ACTIVATION PATHWAY: "
            "NORMAL RIBOSOME BIOGENESIS: "
            "  rDNA transcription (RNA Pol I) → pre-47S rRNA → processing → "
            "    → 18S rRNA (40S subunit) + 5.8S/28S rRNA (60S subunit); "
            "  RPS proteins assemble onto 18S rRNA in nucleolus → export as 40S subunit; "
            "HAPLOINSUFFICIENCY MECHANISM: "
            "  RPS19 (or other RP) haploinsufficiency → "
            "  → imbalance: rRNA > RP protein → excess rRNA intermediates accumulate → "
            "  → NUCLEOLAR STRESS SIGNAL: free RPL5 and RPL11 (not incorporated into 60S subunit) → "
            "  → RPL5·RPL11 complex binds MDM2 (p53-specific E3 ubiquitin ligase) → "
            "  → MDM2 inhibited — cannot ubiquitinate p53 → "
            "  → p53 stabilised and activated → "
            "  → p21 (CDKN1A) transcription → G1 cell cycle arrest; "
            "  → PUMA/BAX transcription → apoptosis; "
            "ERYTHROID SELECTIVITY: "
            "  Erythroid colony-forming units (BFU-E, CFU-E) have highest proliferative rate; "
            "  Require maximum ribosome biogenesis; "
            "  Most vulnerable to RP haploinsufficiency → selective erythroid failure; "
            "  Myeloid and lymphoid progenitors less severely affected (different RP demand); "
            "LEUCINE RESCUE MECHANISM: "
            "  Leucine → activates GCN2 → activates mTORC1 → "
            "  → enhances translation of remaining RPS19 allele; "
            "  → partial correction of ribosomal imbalance → partial erythroid rescue; "
            "CORTICOSTEROID MECHANISM: "
            "  Not fully understood; "
            "  SBDS upregulation? Anti-apoptotic effect on erythroid progenitors; "
            "  Dexamethasone-responsive element in RPS19 promoter region"
        ),
        "pathognomonic": (
            "eADA ELEVATION PATHOGNOMONIC: "
            "  Erythrocyte adenosine deaminase (eADA) elevated (>1.04 U/g Hb, age-adjusted) in 80-85% of DBA; "
            "  Mechanism: eADA is an mRNA stability factor; in DBA, p53 activation upregulates eADA mRNA; "
            "  HIGH SPECIFICITY: eADA normal in: aplastic anemia, Fanconi anemia, TEC, Diamond-Blackfan remission; "
            "  eADA NORMAL in DBA remission (can revert to normal on steroids — check before starting therapy!); "
            "  TRIPHALANGEAL THUMB PATHOGNOMONIC FOR DBA: flat thenar / triphalangeal (3-phalanged) = "
            "    preaxial limb anomaly specific to DBA (not Fanconi, not other BMF); "
            "  i-ANTIGEN POSITIVITY: fetal red cell antigen persistence in DBA (not Fanconi); "
            "  ELEVATED MCV/MCH FROM BIRTH: macrocytosis present even in neonatal period"
        ),
        "treatment": (
            "STEROID THERAPY (first-line): "
            "  Prednisolone 2 mg/kg/day → assess at 4 weeks; "
            "  40-60% respond (Hb rises to ≥9 g/dL, transfusion-free); "
            "  Maintenance: taper to lowest alternate-day dose maintaining Hb ≥9 g/dL; "
            "  GROWTH: corticosteroids cause growth failure — monitor height velocity every 6 months; "
            "  Steroid holiday: planned periods off steroids (watch for relapse); "
            "TRANSFUSION PROGRAMME: "
            "  For non-responders or steroid-dependent with unacceptable side effects; "
            "  Chronic transfusions every 3-4 weeks (target Hb >8 g/dL); "
            "  Iron chelation: deferasirox (oral, age ≥2) or deferoxamine SC; "
            "  Ferritin target <1000 ng/mL (cardiac, liver, endocrine protection); "
            "HAEMATOPOIETIC STEM CELL TRANSPLANTATION (HSCT): "
            "  CURATIVE — eliminates transfusion dependence and cancer risk; "
            "  Best results: matched sibling donor < age 10 (OS >90%); "
            "  Unrelated donor: OS 80-85% (DBAR outcome data); "
            "  Conditioning: reduced-intensity preferred; "
            "  TIMING: before iron overload develops; "
            "LEUCINE SUPPLEMENTATION: "
            "  0.5 g/kg/day; partial response in ~50% (Hb rise 1-2 g/dL); "
            "  May allow steroid-sparing in partial responders; "
            "LUSPATERCEPT (investigational): "
            "  TGF-β trap; phase 2 DBA trial; some responses in transfusion-dependent DBA; "
            "ERYTHROPOIETIN: "
            "  Not effective in DBA (erythroid progenitors absent — no target cells)"
        ),
        "seed": 2806,
        "pt_vars": {
            "hb_range": (4.5, 7.8),
            "reticulocyte_pct": (0.01, 0.15),
            "mcv_range": (93, 112),
            "eada_range": (1.05, 3.80),
            "steroid_response_pct": 52,
        }
    },
    {
        "gene": "RPL5",
        "protein": (
            "RPL5 -- 1p22.1 AD -- 297aa -- 60S-Ribosomal-Protein-L5-34kDa-"
            "Large-Subunit-5S-rRNA-Scaffold-MDM2-Binding-Partner-"
            "OMIM-Gene-603634-Disease-DBA6-612561"
        ),
        "locus": "1p22.1",
        "protein_size": "297 aa / 34 kDa (60S ribosomal large subunit; directly contacts 5S rRNA; forms RPL5·RPL11 surveillance complex that binds MDM2; critical for 60S assembly; mutations in 5S rRNA-binding domain most severe)",
        "inheritance": (
            "AUTOSOMAL DOMINANT (haploinsufficiency) — second most common DBA gene (~9% of all DBA); "
            "RPL5 encodes Ribosomal Protein L5, a structural component of the 60S large ribosomal subunit; "
            "CRITICAL FUNCTION: RPL5 directly binds and stabilises 5S rRNA; "
            "  In RPL5 haploinsufficiency: 5S rRNA not stabilised → 5S rRNA degraded → 60S assembly defective; "
            "SURVEILLANCE MECHANISM: "
            "  Free (non-ribosome-incorporated) RPL5 + RPL11 complex → binds MDM2 → "
            "  → MDM2 catalytic activity inhibited → p53 stabilised → erythroid apoptosis; "
            "  RPL5 haploinsufficiency PARADOX: reduced total RPL5 but relatively more FREE RPL5 available → "
            "  → enhanced MDM2 inhibition in erythroid progenitors; "
            "DISTINCTIVE PHENOTYPE vs RPS19: "
            "  HIGHER FREQUENCY OF MIDLINE/CLEFT ANOMALIES: "
            "    Cleft lip ± palate (15-20% of RPL5 patients — pathognomonic for RPL5-DBA); "
            "    Midline abnormalities (bifid uvula, high-arched palate); "
            "    Thumb anomalies: triphalangeal thumb + bilateral broad/flat thumbs; "
            "  HIGHEST CANCER RISK AMONG DBA GENES: "
            "    Elevated osteosarcoma, colon cancer, breast cancer, MDS/AML; "
            "    Mechanism: RPL5+RPL11 are p53 activators; RPL5 haploinsufficiency impairs p53 surveillance; "
            "    Lifetime cancer risk estimated 20-30% (higher than other DBA genes); "
            "  STEROID RESPONSE: similar to RPS19 (40-55%); "
            "DE NOVO RATE: ~55% of RPL5 cases; "
            "MUTATION TYPES: frameshift/nonsense dominant; missense clustered in 5S-binding domain"
        ),
        "disease_category": (
            "DIAMOND-BLACKFAN ANAEMIA TYPE 6 (DBA6) — OMIM 612561; "
            "ERYTHROID PHENOTYPE: identical to DBA1 (pure red cell aplasia); "
            "DISTINCTIVE FEATURES OF RPL5-DBA (vs other DBA genes): "
            "  1. CLEFT PALATE/LIP — PATHOGNOMONIC for RPL5 when found with DBA: "
            "     Isolated cleft palate OR cleft lip ± palate in 15-20%; "
            "     Not seen at this frequency in RPS19, RPL11, or other RP genes; "
            "  2. THUMB ANOMALIES: triphalangeal thumb (preaxial limb anomaly); "
            "     Thenar hypoplasia; first ray anomalies; "
            "  3. CARDIAC DEFECTS: VSD (~25%), ASD; "
            "  4. HIGHEST CANCER RISK: osteosarcoma (particularly post-irradiation), colon cancer; "
            "     Annual MRI surveillance + colonoscopy from age 20 in RPL5 carriers; "
            "HAEMATOLOGICAL CRITERIA: "
            "  Same as DBA1 — macrocytic PRCA, elevated eADA, reticulocytopenia, erythroblastopenia on BM; "
            "PROGNOSIS: "
            "  ~50% steroid-responsive; ~50% transfusion-dependent; "
            "  HSCT recommended earlier in RPL5 given high cancer risk + poor HLA-match risk with advancing age"
        ),
        "disease_pathway": (
            "RPL5·RPL11 MDM2-INHIBITION / 5S RIBONUCLEOPROTEIN PATHWAY: "
            "5S RNP COMPLEX: "
            "  RPL5 + RPL11 + 5S rRNA = 5S ribonucleoprotein (5S RNP); "
            "  In normal cells: 5S RNP incorporated into 60S ribosome → ribosome assembly; "
            "  In RPL5 haploinsufficiency: insufficient RPL5 to sequester all 5S rRNA → "
            "    → excess RPL11 released from 5S RNP → free RPL11 available; "
            "MDM2 BINDING: "
            "  Free RPL5 and free RPL11 each have independent MDM2-binding capability; "
            "  RPL5 binds MDM2 N-terminal domain; RPL11 binds MDM2 central domain; "
            "  Together they cause allosteric inhibition of MDM2 E3 ubiquitin ligase; "
            "  MDM2 cannot ubiquitinate p53 → p53 half-life increases 4-10 fold; "
            "p53 TRANSCRIPTIONAL ACTIVATION: "
            "  p53 activates: p21 (CDKN1A) → G1 arrest; PUMA (BBC3) → mitochondrial apoptosis; "
            "  TIGAR → glycolytic shift; DEC1 → differentiation block; "
            "MIDLINE DEVELOPMENTAL DEFECT MECHANISM: "
            "  RPL5 also regulates rRNA processing in neural crest cells; "
            "  Neural crest → palate/lip/thumb structures; "
            "  Haploinsufficiency during craniofacial development → midline defects (cleft, thumb anomalies); "
            "  Window: 4-8 weeks post-conception (neural crest migration critical period)"
        ),
        "pathognomonic": (
            "CLEFT LIP/PALATE + DBA = RPL5 PATHOGNOMONIC: "
            "  Cleft palate (isolated) or cleft lip ± palate in a DBA patient strongly predicts RPL5 mutation; "
            "  15-20% of RPL5-DBA have midline cleft — much higher than RPS19 (<1%) or RPL11 (<3%); "
            "  CLINICAL RULE: DBA + cleft palate → genotype RPL5 FIRST; "
            "TRIPHALANGEAL THUMB + CLEFT PALATE + DBA: "
            "  Three-pronged diagnostic triad pointing to RPL5; "
            "HIGHEST CANCER RISK PEARLS: "
            "  Osteosarcoma risk: never give therapeutic irradiation to RPL5-DBA (worsens risk); "
            "  Colonoscopy from age 20 (colon cancer risk); "
            "  Annual breast MRI from age 25 in RPL5 female carriers"
        ),
        "treatment": (
            "SAME DBA FRAMEWORK + RPL5-SPECIFIC MANAGEMENT: "
            "Steroids: prednisolone 2 mg/kg/day → taper; ~50% response; "
            "CLEFT PALATE: palatal surgery timing coordinated with haematology to avoid anaemia risk; "
            "  ENT and craniofacial surgery team involved from birth; "
            "CANCER SURVEILLANCE (RPL5-SPECIFIC — more intensive than other DBA genes): "
            "  Annual whole-body MRI from diagnosis (children) or age 10 (adults); "
            "  Colonoscopy: begin age 20, every 5 years; "
            "  Breast: annual MRI from age 25 in females; "
            "  Bone surveys + AFP/LDH: osteosarcoma surveillance; "
            "  AVOID THERAPEUTIC RADIATION (all DBA — but especially RPL5 given TP53 pathway impairment); "
            "HSCT TIMING: "
            "  Consider earlier HSCT in RPL5 to reduce cumulative cancer risk; "
            "  Recommend sibling donor BEFORE age 10 if available; "
            "GENETIC COUNSELLING: "
            "  RPL5 cleft palate can be a forme fruste — cleft palate without overt DBA possible; "
            "  Cleft palate probands should have eADA + FBC to rule out subclinical DBA"
        ),
        "seed": 2807,
        "pt_vars": {
            "hb_range": (4.2, 7.5),
            "reticulocyte_pct": (0.01, 0.12),
            "mcv_range": (94, 114),
            "eada_range": (1.10, 4.20),
            "steroid_response_pct": 48,
        }
    },
    {
        "gene": "RPL11",
        "protein": (
            "RPL11 -- 1p36.1 AD -- 178aa -- 60S-Ribosomal-Protein-L11-20kDa-"
            "Large-Subunit-MDM2-Binder-Thumb-Anomaly-Cancer-Risk-"
            "OMIM-Gene-604175-Disease-DBA7-612562"
        ),
        "locus": "1p36.1",
        "protein_size": "178 aa / 20 kDa (60S ribosomal large subunit; zinc-finger-like domain; key MDM2-binding partner; p53 activator; nucleolar localisation in steady state; released during 60S assembly stress)",
        "inheritance": (
            "AUTOSOMAL DOMINANT (haploinsufficiency) — third most common DBA gene (~5% of all DBA); "
            "RPL11 encodes Ribosomal Protein L11, a 60S structural component essential for 60S assembly; "
            "MECHANISM: RPL11 haploinsufficiency → impaired 60S pre-rRNA processing → nucleolar stress → "
            "  → free RPL11 binds MDM2 → p53 stabilised → erythroid apoptosis; "
            "DISTINCTIVE RPL11 PHENOTYPE: "
            "  THUMB / THENAR ANOMALIES: "
            "    Thumb anomalies in ~30% of RPL11-DBA (flat thenar, triphalangeal thumb); "
            "    PATHOGNOMONIC COMBINATION: DBA + bilateral thenar hypoplasia + no cleft palate → consider RPL11; "
            "  ABSENCE OF CLEFT PALATE (contrast with RPL5): no significant cleft association; "
            "  CANCER RISK: elevated but less than RPL5; MDS/AML, osteosarcoma; "
            "DE NOVO RATE: ~60% of RPL11 cases; "
            "MUTATION TYPES: missense common; nonsense/frameshift; "
            "5q- SYNDROME CONNECTION: 5q- in MDS causes haploinsufficiency of RPS14 (not RPL11) but similar p53 mechanism; "
            "  RPL11 at 1p36 — del1p36 can rarely cause DBA-like phenotype without point mutation"
        ),
        "disease_category": (
            "DIAMOND-BLACKFAN ANAEMIA TYPE 7 (DBA7) — OMIM 612562; "
            "ERYTHROID PHENOTYPE: same as DBA1 — macrocytic PRCA, elevated eADA, reticulocytopenia; "
            "DISTINCTIVE FEATURES vs RPL5: "
            "  Thumb/thenar anomalies WITHOUT cleft palate; "
            "  DIAGNOSTIC CLUE: DBA + thumb anomaly → consider RPL11 (RPL5 has cleft + thumb; RPL11 has thumb only); "
            "  Cardiac defects present in ~20%; "
            "  Cancer risk: intermediate (lower than RPL5, higher than RPS19); "
            "GENOTYPE-PHENOTYPE: "
            "  N-terminal missense mutations → milder phenotype; "
            "  Protein-truncating variants → more severe haematological + structural phenotype"
        ),
        "disease_pathway": (
            "RPL11-MDM2 SURVEILLANCE — 60S ASSEMBLY CHECKPOINT: "
            "NORMAL 60S ASSEMBLY: "
            "  60S pre-rRNA (5.8S + 28S + 5S rRNA) + RPL proteins → 60S subunit; "
            "  RPL11 specifically binds the peptidyl transferase centre and bridges 5S rRNA; "
            "HAPLOINSUFFICIENCY STRESS: "
            "  RPL11 limiting → 60S assembly stalls → free RPL11 in nucleoplasm; "
            "  Free RPL11 binds MDM2 central zinc-finger domain; "
            "  MDM2 self-ubiquitination impaired; MDM2 cannot degrade p53; "
            "SHARED PATH WITH RPL5: "
            "  Both RPL5 and RPL11 independently bind MDM2; "
            "  Together form RPL5·RPL11 complex for cooperative MDM2 inhibition; "
            "  Single gene haploinsufficiency sufficient to activate pathway selectively in erythroid cells"
        ),
        "pathognomonic": (
            "THENAR HYPOPLASIA + DBA without CLEFT PALATE → RPL11: "
            "  DBA + flat/hypoplastic thenar eminence + absent cleft palate = RPL11 pattern; "
            "  RPL5: cleft palate + thumb anomaly; RPL11: thumb anomaly alone (no cleft); "
            "  BILATERAL THENAR HYPOPLASIA strongly predicts RPL11 in DBA; "
            "eADA ELEVATION: same as other DBA genes (80-85% sensitivity); "
            "TRIPHALANGEAL THUMB (not thenar): both RPL5 and RPL11 can show; cleft differentiates"
        ),
        "treatment": (
            "SAME DBA FRAMEWORK as RPS19: "
            "  Steroids, transfusions, chelation, HSCT; "
            "  ~50% steroid response; "
            "HAND/THUMB RECONSTRUCTION: "
            "  Early orthopaedic involvement for thumb/hand anomalies; "
            "  Opponensplasty if thenar function severely compromised; "
            "CANCER SURVEILLANCE: "
            "  Intermediate-intensity protocol (between RPS19 and RPL5); "
            "  Annual CBC + bone survey; colonoscopy from age 20; "
            "  Whole-body MRI every 2 years from age 10"
        ),
        "seed": 2808,
        "pt_vars": {
            "hb_range": (4.3, 7.9),
            "reticulocyte_pct": (0.01, 0.14),
            "mcv_range": (92, 113),
            "eada_range": (1.08, 3.95),
            "steroid_response_pct": 50,
        }
    },
    {
        "gene": "RPS26",
        "protein": (
            "RPS26 -- 12q13.2 AD -- 119aa -- 40S-Ribosomal-Protein-S26-13kDa-"
            "Small-Subunit-E-Site-mRNA-Decoding-Somatic-AML-12q-Deletions-"
            "OMIM-Gene-603473-Disease-DBA10-613309"
        ),
        "locus": "12q13.2",
        "protein_size": "119 aa / 13 kDa (40S ribosomal small subunit; mRNA decoding E-site; directly contacts tRNA in exit site; mutations cluster in rRNA-binding surface; smallest DBA ribosomal protein by size)",
        "inheritance": (
            "AUTOSOMAL DOMINANT (haploinsufficiency) — ~3% of all DBA; "
            "RPS26 encodes Ribosomal Protein S26, one of the smallest 40S subunit proteins; "
            "MECHANISM: same nucleolar stress → free RPL5·RPL11 → MDM2 inhibition → p53 pathway; "
            "DISTINCTIVE FEATURE: "
            "  RPS26 somatic loss (deletions, LOH) found in 10-15% of de novo AML (del12q13); "
            "  Germline RPS26 haploinsufficiency (DBA) → increased leukemic transformation; "
            "  RPS26 is a tumour suppressor through the p53 pathway in haematopoietic cells; "
            "CLINICAL PHENOTYPE: "
            "  Haematological DBA phenotype (macrocytic PRCA); "
            "  Congenital anomalies in ~40% (cardiac, renal, craniofacial); "
            "  SHORT STATURE frequent (corticosteroid side effect + intrinsic GH-axis effect of RP stress); "
            "  Intermediate cancer risk; "
            "STEROID RESPONSE: ~45-50%; "
            "DE NOVO RATE: ~50%; "
            "MUTATION SPECTRUM: missense (C-terminal rRNA-binding surface), nonsense, frameshift"
        ),
        "disease_category": (
            "DIAMOND-BLACKFAN ANAEMIA TYPE 10 (DBA10) — OMIM 613309; "
            "PHENOTYPE: classic DBA haematology (erythroblastopenia, macrocytic anaemia, elevated eADA); "
            "NOTABLE FEATURES: "
            "  No single pathognomonic structural anomaly (unlike RPL5-cleft or RPL11-thumb); "
            "  Short stature prominent feature (intrinsic + corticosteroid-compounded); "
            "  Genitourinary anomalies in ~20%; "
            "SOMATIC LINK: "
            "  12q13 deletions with RPS26 somatic LOH found in ~12% of non-DBA AML — monitoring important; "
            "  Germline DBA10 → higher AML transformation risk than DBA1 (data limited)"
        ),
        "disease_pathway": (
            "40S SMALL SUBUNIT ASSEMBLY — RPS26 HAPLOINSUFFICIENCY: "
            "NORMAL FUNCTION: "
            "  RPS26 in mRNA exit site (E site) of 40S subunit → stabilises tRNA·mRNA during elongation; "
            "  Critical for accurate codon reading → translation fidelity; "
            "HAPLOINSUFFICIENCY: "
            "  Reduced 40S subunit output → ribosomal stress → same p53 pathway as RPS19; "
            "  E-site defect → codon skipping → translation errors → abnormal proteins → UPR contribution; "
            "AML LINK: "
            "  Somatic del12q: RPS26 LOH in AML → p53 surveillance impaired → clonal outgrowth; "
            "  DBA10 germline carriers: one allele already inactivated → lower threshold for AML"
        ),
        "pathognomonic": (
            "NO SINGLE STRUCTURAL PATHOGNOMONIC: "
            "  DBA10/RPS26-DBA lacks a single structural PATHOGNOMONIC anomaly; "
            "  Diagnosis: combination of haematological DBA criteria + eADA elevation + molecular confirmation; "
            "SOMATIC AML LINK: "
            "  Haematologist pearl: 12q13 deletion in AML patient → screen family for DBA10; "
            "  Treat AML in DBA10 with caution: TP53 pathway impaired — radiation risk increased; "
            "eADA: elevated in ~80% — same as other DBA genes"
        ),
        "treatment": (
            "SAME DBA FRAMEWORK: steroids → transfusion → chelation → HSCT; "
            "GROWTH MONITORING: short stature frequent — paediatric endocrinology involvement; "
            "  Growth hormone trials if height SDS < −2 and corticosteroid-sparing achieved; "
            "CANCER SURVEILLANCE: "
            "  Annual CBC, LFTs, LDH; "
            "  Colonoscopy from age 20; "
            "  Haematology vigilance for MDS/AML (del12q cytogenetics helpful); "
            "AML TREATMENT NOTE: avoid radiation therapy; platinum + targeted approaches preferred"
        ),
        "seed": 2809,
        "pt_vars": {
            "hb_range": (4.4, 7.7),
            "reticulocyte_pct": (0.01, 0.13),
            "mcv_range": (91, 111),
            "eada_range": (1.04, 3.60),
            "steroid_response_pct": 47,
        }
    },
    {
        "gene": "RPL35A",
        "protein": (
            "RPL35A -- 3q29 AD -- 110aa -- 60S-Ribosomal-Protein-L35a-12kDa-"
            "Large-Subunit-Peptide-Exit-Tunnel-Genitourinary-Anomalies-"
            "OMIM-Gene-180468-Disease-DBA5-612528"
        ),
        "locus": "3q29",
        "protein_size": "110 aa / 12 kDa (60S ribosomal large subunit; adjacent to peptide exit tunnel; contacts 28S rRNA domain V; one of the smallest 60S proteins; mutations in RNA-binding residues most pathogenic)",
        "inheritance": (
            "AUTOSOMAL DOMINANT (haploinsufficiency) — ~3% of DBA; "
            "RPL35A encodes Ribosomal Protein L35a, a 60S large subunit component near the peptide exit tunnel; "
            "MECHANISM: same nucleolar stress → RPL5·RPL11·MDM2 → p53 → erythroid apoptosis; "
            "DISTINCTIVE FEATURE — GENITOURINARY ANOMALIES: "
            "  Renal anomalies: 20-25% of RPL35A-DBA; horseshoe kidney, duplex collecting system; "
            "  Genital anomalies in males: hypospadias, cryptorchidism; "
            "  GU anomalies more prominent than in RPS19/RPL5/RPL11; "
            "  DIAGNOSTIC CLUE: DBA + renal/GU anomaly → genotype RPL35A early; "
            "STEROID RESPONSE: ~45-50%; "
            "DE NOVO RATE: ~50%; "
            "3q29 locus: rare deletion syndromes involving RPL35A described in literature"
        ),
        "disease_category": (
            "DIAMOND-BLACKFAN ANAEMIA TYPE 5 (DBA5) — OMIM 612528; "
            "ERYTHROID PHENOTYPE: classic DBA macrocytic PRCA; elevated eADA; "
            "DISTINCTIVE RPL35A FEATURES: "
            "  Genitourinary anomalies: renal + male genital (hypospadias) more frequent than in other DBA subtypes; "
            "  Cardiac defects ~20%; "
            "  Cleft palate <5% (contrast RPL5); "
            "  Thumb anomalies <10% (contrast RPL11); "
            "  Short stature common; "
            "CLINICAL RULE: DBA + hypospadias + renal duplex → sequence RPL35A"
        ),
        "disease_pathway": (
            "60S PEPTIDE EXIT TUNNEL INTEGRITY — RPL35A MECHANISM: "
            "PEPTIDE EXIT TUNNEL (PET): "
            "  Newly synthesised polypeptide exits ribosome through PET (60S subunit tunnel); "
            "  RPL35A contributes to PET architecture near tunnel exit; "
            "  RPL35A LOF → PET integrity compromised → translation elongation stall → "
            "    → ribosomal quality control (RQC) pathway activated → "
            "    → RPL5·RPL11 release → p53 pathway (same as all DBA genes); "
            "DEVELOPMENTAL BIOLOGY: "
            "  RPL35A expressed in ureteric bud and metanephric mesenchyme during kidney development; "
            "  Haploinsufficiency during nephrogenesis (weeks 5-12 post-conception) → "
            "    → reduced ribosomal output → impaired renal branching morphogenesis → GU anomalies"
        ),
        "pathognomonic": (
            "GENITOURINARY ANOMALY + DBA → SEQUENCE RPL35A FIRST: "
            "  Renal anomaly (horseshoe, duplex, absent) OR hypospadias in male DBA patient → RPL35A; "
            "  20-25% of RPL35A-DBA have significant GU anomaly; "
            "  eADA elevated as with all DBA genes; "
            "  RPL35A-DBA does NOT commonly have cleft or triphalangeal thumb (DDx from RPL5/RPL11)"
        ),
        "treatment": (
            "SAME DBA FRAMEWORK (steroids → transfusion → HSCT); "
            "GENITOURINARY MANAGEMENT: "
            "  Renal USS at diagnosis (all DBA — but especially RPL35A); "
            "  Duplex collecting system: annual MCUG + renal function monitoring; "
            "  Hypospadias: urology referral in first year of life; "
            "  Horseshoe kidney: low threshold for UTI workup; "
            "NEPHROLOGY CO-MANAGEMENT: "
            "  Avoid nephrotoxic drugs (gentamicin, NSAIDs) — increased renal malformation risk; "
            "  Chronic kidney disease monitoring if significant renal anomaly"
        ),
        "seed": 2810,
        "pt_vars": {
            "hb_range": (4.3, 7.6),
            "reticulocyte_pct": (0.01, 0.13),
            "mcv_range": (90, 112),
            "eada_range": (1.05, 3.55),
            "steroid_response_pct": 46,
        }
    },
    {
        "gene": "RPS17",
        "protein": (
            "RPS17 -- 15q25.2 AD -- 135aa -- 40S-Ribosomal-Protein-S17-15kDa-"
            "Small-Subunit-Head-Domain-First-Non-RPS19-Gene-Discovered-"
            "OMIM-Gene-180472-Disease-DBA4-612527"
        ),
        "locus": "15q25.2",
        "protein_size": "135 aa / 15 kDa (40S ribosomal small subunit head domain; involved in mRNA accommodation; second DBA gene discovered historically; biallelic inherited variant reported in two unrelated DBA families)",
        "inheritance": (
            "AUTOSOMAL DOMINANT (haploinsufficiency) — ~1-2% of DBA; "
            "RPS17 encodes Ribosomal Protein S17; historically significant as the FIRST RP gene after RPS19 to be confirmed as a DBA cause (2006); "
            "MECHANISM: same 40S assembly stress → nucleolar stress → p53 pathway; "
            "CLINICAL PHENOTYPE: "
            "  Classic DBA erythroid phenotype; "
            "  Congenital anomalies in ~35%; "
            "  STEROID RESPONSE: ~50-60% (slightly higher response rate than RPL genes); "
            "  Cancer risk: intermediate (similar to RPS19); "
            "HISTORICAL IMPORTANCE: "
            "  RPS19 discovered 1999; RPS17 2006; opened floodgates for RP gene discovery; "
            "  Now >20 RP genes confirmed in DBA; "
            "MUTATION SPECTRUM: heterozygous loss-of-function; both point mutations and deletions reported"
        ),
        "disease_category": (
            "DIAMOND-BLACKFAN ANAEMIA TYPE 4 (DBA4) — OMIM 612527; "
            "Classic macrocytic PRCA phenotype; elevated eADA; "
            "No single pathognomonic structural anomaly; "
            "Steroid response ~50-60%; "
            "Importance: historical landmark in DBA molecular diagnosis — confirmed ribosome haploinsufficiency as DBA mechanism"
        ),
        "disease_pathway": (
            "40S HEAD DOMAIN ASSEMBLY — RPS17: "
            "RPS17 localises to head domain of 40S subunit; "
            "Required for 18S rRNA maturation in head region; "
            "Haploinsufficiency → same p53 pathway as RPS19; "
            "Steroid response slightly higher may reflect: 40S proteins (RPS genes) respond better to prednisolone "
            "than 60S proteins (RPL genes) — hypothesis based on clinical observation, not proven mechanistically"
        ),
        "pathognomonic": (
            "No single structural PATHOGNOMONIC feature for RPS17-DBA; "
            "eADA elevated (~80%); "
            "Classic DBA haematological criteria confirm diagnosis; "
            "HISTORICAL PEARL: if family study of DBA kindred with no RPS19 mutation → check RPS17 (15q25.2 in silico)"
        ),
        "treatment": (
            "Standard DBA framework: steroids → transfusions → HSCT; "
            "~50-60% steroid response rate; "
            "Growth monitoring essential; "
            "Cancer surveillance: annual CBC + whole-body MRI every 2 years"
        ),
        "seed": 2811,
        "pt_vars": {
            "hb_range": (4.5, 8.0),
            "reticulocyte_pct": (0.01, 0.15),
            "mcv_range": (91, 112),
            "eada_range": (1.04, 3.50),
            "steroid_response_pct": 54,
        }
    },
    {
        "gene": "RPL26",
        "protein": (
            "RPL26 -- 17p13.1 AD -- 145aa -- 60S-Ribosomal-Protein-L26-17kDa-"
            "Large-Subunit-mRNA-5prime-UTR-Binding-p53-mRNA-IRES-Translation-"
            "OMIM-Gene-603704-Disease-DBA11-614900"
        ),
        "locus": "17p13.1",
        "protein_size": "145 aa / 17 kDa (60S ribosomal large subunit; unique direct interaction with mRNA 5'-UTR during translation initiation; binds p53 mRNA 5'-UTR; directly upregulates p53 translation — dual role in ribosome structure AND p53 regulation)",
        "inheritance": (
            "AUTOSOMAL DOMINANT (haploinsufficiency) — ~1% of DBA; "
            "RPL26 has a UNIQUE DUAL ROLE distinguishing it from all other DBA RP genes: "
            "  STRUCTURAL: 60S large subunit component; "
            "  REGULATORY: RPL26 directly binds and stimulates p53 mRNA translation (5'-UTR binding); "
            "UNIQUE MECHANISM: "
            "  Normal: RPL26 promotes p53 mRNA translation → normal low-level p53 activity; "
            "  RPL26 haploinsufficiency: PARADOX — REDUCED RPL26 leads to LESS p53 activation from this direct path; "
            "  BUT ribosomal stress path STILL active (via RPL5·RPL11·MDM2); "
            "  Net: erythroid apoptosis still results from nucleolar stress; "
            "  INTERESTING: RPL26 overexpression → p53 induction — RPL26 is a POSITIVE regulator of p53 translation; "
            "CLINICAL FEATURES: "
            "  Classic DBA erythroid phenotype; "
            "  Less extensive congenital anomaly data (small cohort); "
            "  17p13.1 locus — NOTE: TP53 is at 17p13.1 — del17p deletions can involve BOTH RPL26 AND TP53; "
            "  COMPOUND EFFECT: del17p → TP53 + RPL26 haploinsufficiency simultaneously → very aggressive phenotype"
        ),
        "disease_category": (
            "DIAMOND-BLACKFAN ANAEMIA TYPE 11 (DBA11) — OMIM 614900; "
            "Classic DBA macrocytic PRCA phenotype; elevated eADA; "
            "GENETIC CURIOSITY: RPL26 at 17p13.1 — adjacent to TP53; "
            "  Large del17p → RPL26 + TP53 haploinsufficiency; "
            "  Clinical presentation: DBA phenotype + very early-onset tumour predisposition; "
            "  Check for del17p cytogenetics in RPL26-DBA patients (MLPA); "
            "Steroid response ~45-55%"
        ),
        "disease_pathway": (
            "RPL26 — DUAL ROLE IN p53 REGULATION: "
            "PATH 1 (STRUCTURAL): "
            "  RPL26 haploinsufficiency → 60S assembly defect → nucleolar stress → "
            "    → free RPL5·RPL11 → MDM2 inhibition → p53 stabilised; "
            "PATH 2 (DIRECT p53 mRNA BINDING): "
            "  RPL26 binds p53 mRNA 5'-UTR → promotes IRES-mediated translation of p53; "
            "  RPL26 + nucleolin compete for same p53 5'-UTR element; "
            "  After DNA damage: RPL26 enriched on p53 mRNA → rapid p53 translation surge; "
            "  In DBA: RPL26 deficiency REDUCES this direct path; "
            "    BUT increases nucleolar stress path — net effect: erythroid apoptosis; "
            "del17p SYNERGY: "
            "  TP53 loss (del17p) PLUS RPL26 loss → p53 completely inactivated; "
            "  NO nucleolar stress response; "
            "  Severe immunodeficiency + tumour predisposition (Li-Fraumeni-like)"
        ),
        "pathognomonic": (
            "RPL26 + TP53 DEL17p COMPOUND — CLINICAL ALERT: "
            "  DBA11 patient with severe clinical course: check MLPA for del17p (RPL26 + TP53 co-deletion); "
            "  del17p compound: DBA + Li-Fraumeni equivalent → cancer by adolescence; "
            "DIRECT p53 mRNA REGULATOR: "
            "  RPL26 is UNIQUE among all RP proteins in directly binding p53 mRNA; "
            "  Research value: RPL26 manipulation as potential therapeutic p53 modulator; "
            "eADA elevated in ~80% DBA11"
        ),
        "treatment": (
            "Standard DBA: steroids → transfusions → HSCT; "
            "del17p CO-DELETION MANAGEMENT: "
            "  If del17p detected: Li-Fraumeni-equivalent surveillance; "
            "  WNT1 breast cancer surveillance (annual MRI from age 25); "
            "  Sarcoma vigilance; "
            "  AVOID RADIATION (TP53 haploinsufficiency + RPL26 deficiency = extreme radiosensitivity); "
            "  Early HSCT discussion to reduce cancer risk"
        ),
        "seed": 2812,
        "pt_vars": {
            "hb_range": (4.4, 7.8),
            "reticulocyte_pct": (0.01, 0.13),
            "mcv_range": (92, 113),
            "eada_range": (1.06, 3.70),
            "steroid_response_pct": 50,
        }
    },
    {
        "gene": "TSR2",
        "protein": (
            "TSR2 -- Xp11.22 XLR -- 228aa -- TSR2-Ribosome-Maturation-Factor-26kDa-"
            "X-Linked-DBA-RPS26-Chaperone-Males-Affected-FEMALE-Carriers-"
            "OMIM-Gene-300945-Disease-DBA13-300946"
        ),
        "locus": "Xp11.22",
        "protein_size": "228 aa / 26 kDa (ribosome maturation factor; specific chaperone for RPS26 nuclear import; mediates RPS26 assembly into 40S subunit; contains HEAT-like repeats; Xp11.22 locus — X-linked inheritance)",
        "inheritance": (
            "X-LINKED RECESSIVE — hemizygous males affected; heterozygous females usually unaffected; "
            "TSR2 is the only X-linked DBA gene identified (distinct from DKC1 which causes a different X-linked BMF syndrome); "
            "TSR2 encodes TSR2 Ribosome Maturation Factor — a DEDICATED CHAPERONE for RPS26; "
            "MECHANISM: "
            "  TSR2 escorts newly synthesised RPS26 protein from cytoplasm → nucleus → 40S assembly; "
            "  TSR2 haploinsufficiency → RPS26 cannot reach nucleolus → "
            "    → 40S subunit assembly fails despite RPS26 gene being INTACT; "
            "  Functionally equivalent to RPS26 haploinsufficiency (DBA10) via loss of chaperone, not RP itself; "
            "  IMPLICATION: TSR2-DBA patients have NORMAL RPS26 gene sequence but functionally RPS26-deficient; "
            "  NGS DIAGNOSTIC TRAP: sequencing RPS26 is NORMAL in TSR2-DBA; "
            "X-LINKED PHENOTYPE: "
            "  Males: classic DBA erythroid phenotype; "
            "  Female carriers: usually unaffected (skewed X-inactivation); "
            "    Rare female carriers with haematological features (random or preferential X-inactivation); "
            "PREVALENCE: <1% of DBA; very rare; important to diagnose (X-linked inheritance counselling); "
            "MUTATION TYPES: hemizygous LOF; missense and truncating variants in HEAT-like domain; "
            "STEROID RESPONSE: ~50%"
        ),
        "disease_category": (
            "DIAMOND-BLACKFAN ANAEMIA TYPE 13 (DBA13) — OMIM 300946; "
            "X-LINKED DBA (only X-linked DBA subtype); "
            "ERYTHROID PHENOTYPE: same as autosomal DBA (macrocytic PRCA, elevated eADA, reticulocytopenia); "
            "DISTINCTIVE FEATURES: "
            "  X-linked inheritance pattern — all affected children are male; "
            "  Female carriers clinically unaffected (95%); "
            "  Rare carrier females: may show mild macrocytosis without frank anaemia; "
            "GENETIC COUNSELLING: "
            "  Carrier mother → 50% of sons affected, 50% of daughters are carriers; "
            "  De novo hemizygous TSR2 mutations documented in isolated cases; "
            "DIAGNOSIS TRAP: "
            "  Standard DBA gene panel sequences RP genes — RPS26 NORMAL in TSR2-DBA; "
            "  Must include TSR2 on panel; "
            "  Or whole-exome sequencing to pick up TSR2 mutation"
        ),
        "disease_pathway": (
            "TSR2-RPS26 CHAPERONE AXIS — X-LINKED DBA MECHANISM: "
            "NORMAL TSR2 FUNCTION: "
            "  RPS26 translated in cytoplasm → "
            "  TSR2 (chaperone) binds RPS26 in cytoplasm → "
            "  TSR2·RPS26 complex translocates to nucleus → "
            "  RPS26 deposited onto 40S pre-ribosome in nucleolus → "
            "  TSR2 recycled back to cytoplasm; "
            "TSR2 DEFICIENCY: "
            "  RPS26 protein produced but cannot reach nucleolus without chaperone; "
            "  Cytoplasmic RPS26 aggregates and is degraded; "
            "  40S assembly fails → nucleolar stress → same p53 pathway; "
            "RPS26 CONNECTION: "
            "  TSR2-DBA and RPS26-DBA (DBA10) share identical pathway; "
            "  TSR2 is the upstream chaperone; RPS26 is the downstream ribosomal protein; "
            "  Phenotype very similar between DBA10 and DBA13"
        ),
        "pathognomonic": (
            "X-LINKED DBA INHERITANCE — TSR2: "
            "  All-male affected pedigree with DBA + X-linked inheritance = TSR2 (or DKC1, but DKC1 causes DC triad, NOT pure PRCA); "
            "  TSR2 = X-linked PRCA (pure red cell aplasia); DKC1 = X-linked DC (skin+nails+mouth triad + PRCA); "
            "  DIFFERENTIATION: TSR2-affected males do NOT have nail dystrophy, leucoplakia, or skin pigmentation (those are DKC1/DC); "
            "RPS26 NORMAL IN TSR2-DBA: "
            "  Diagnostic pitfall: standard RP panel shows no RPS26 mutation; "
            "  Always include TSR2 in DBA panel for X-linked family history; "
            "eADA elevated in TSR2-DBA (~80%)"
        ),
        "treatment": (
            "SAME DBA FRAMEWORK: steroids → transfusions → HSCT; "
            "GENETIC COUNSELLING SPECIFIC TO X-LINKED: "
            "  Carrier testing for sisters and maternal female relatives; "
            "  Pre-natal diagnosis available for at-risk pregnancies; "
            "  PGT (pre-implantation genetic testing) available; "
            "FEMALE CARRIER MONITORING: "
            "  Annual FBC; eADA if any haematological concern; "
            "BONE MARROW TRANSPLANTATION: "
            "  Allogeneic HSCT curative; check carrier females in family as potential donors (may be obligate carriers — test first); "
            "  Cord blood banking recommended for carrier mothers (affected sons)"
        ),
        "seed": 2813,
        "pt_vars": {
            "hb_range": (4.3, 7.7),
            "reticulocyte_pct": (0.01, 0.12),
            "mcv_range": (93, 114),
            "eada_range": (1.06, 3.65),
            "steroid_response_pct": 49,
        }
    },
]

DEFINITIONS = {
    "definitions": [
        {
            "term": "Diamond-Blackfan Anaemia (DBA)",
            "definition": (
                "Congenital hypoplastic anaemia — pure red cell aplasia (PRCA) presenting in first year of life; "
                "caused by haploinsufficiency of ribosomal protein genes (40S or 60S subunit proteins) → "
                "nucleolar stress → p53 activation → selective erythroid apoptosis; "
                "OMIM 105650 (DBA1/RPS19-based original description); "
                "Incidence: 5-7 per million live births; equal sex ratio in autosomal forms; "
                "Named after Louis Diamond (Boston) and Kenneth Blackfan (1938 original description)"
            )
        },
        {
            "term": "Pure Red Cell Aplasia (PRCA)",
            "definition": (
                "Selective absence of erythroid precursors (proerythroblasts, erythroblasts) in bone marrow "
                "with normal myeloid and megakaryocytic precursors; "
                "Reticulocytopenia (<10,000/μL); macrocytic anaemia; preserved platelet and neutrophil counts; "
                "DBA is congenital PRCA; TEC (transient erythroblastopenia of childhood) is acquired/temporary PRCA"
            )
        },
        {
            "term": "Erythrocyte Adenosine Deaminase (eADA)",
            "definition": (
                "Enzyme in red cells that deaminates adenosine → inosine; "
                "Elevated in DBA (>1.04 U/g Hb, age-adjusted) in 80-85% of cases — PATHOGNOMONIC when elevated; "
                "Mechanism: p53 target gene upregulation increases eADA mRNA stability in DBA erythrocytes; "
                "Important distinction: eADA NORMAL in aplastic anaemia, Fanconi anaemia, TEC; "
                "Can revert to normal in DBA remission (on steroids) — test before starting steroids if possible"
            )
        },
        {
            "term": "Nucleolar Stress / Ribosomal Stress",
            "definition": (
                "Cellular response to impaired ribosome biogenesis; "
                "Triggered by: RP haploinsufficiency, rRNA transcription inhibition, nucleolar disruption; "
                "Key event: excess free RPL5 and RPL11 (not incorporated into ribosomes) → "
                "  bind MDM2 → inhibit MDM2 E3 ubiquitin ligase activity → p53 stabilisation; "
                "Downstream: p53 → p21 (G1 arrest) + PUMA (apoptosis); "
                "Selective in erythroid progenitors: highest demand for ribosome biogenesis during rapid proliferation"
            )
        },
        {
            "term": "MDM2 — Mouse Double Minute 2 Homolog",
            "definition": (
                "E3 ubiquitin ligase (RING domain); primary negative regulator of p53; "
                "Ubiquitinates p53 → proteasomal degradation (p53 half-life ~20 min in normal cells); "
                "MDM2 inhibited by: RPL5·RPL11 (ribosomal stress), ARF (oncogenic stress), ATM (DNA damage); "
                "All three converge on MDM2 to stabilise p53 in different stress contexts; "
                "MDM2 inhibitors (nutlins) under investigation for DBA treatment (activate p53 further — complex)"
            )
        },
        {
            "term": "Triphalangeal Thumb",
            "definition": (
                "Congenital limb anomaly: thumb with three phalanges (proximal, middle, distal) instead of normal two; "
                "Preaxial limb defect (thumb = preaxial digit, radial side); "
                "PATHOGNOMONIC for DBA (most commonly RPL5, RPL11) when isolated; "
                "NOT seen in Fanconi anaemia (Fanconi has thumb hypoplasia/aplasia, not triphalangeal); "
                "Bilateral triphalangeal thumb + DBA: haematology-orthopaedics co-management from birth"
            )
        },
        {
            "term": "Corticosteroid Response in DBA",
            "definition": (
                "40-80% of DBA patients respond to prednisolone (2 mg/kg/day for 4 weeks → taper); "
                "Definition of response: Hb ≥9 g/dL and transfusion independence; "
                "Mechanism: uncertain — may upregulate residual RP allele, anti-apoptotic in erythroid progenitors; "
                "GROWTH: corticosteroids cause linear growth failure — switch to transfusion programme if growth SDS < -2; "
                "STEROID HOLIDAY: planned off-steroid periods (monitor for relapse); "
                "RPL genes (60S) may have lower response rate than RPS genes (40S) — observational data"
            )
        },
        {
            "term": "Allogeneic HSCT in DBA",
            "definition": (
                "Only CURATIVE treatment for DBA; eliminates transfusion dependence AND cancer predisposition; "
                "INDICATIONS: transfusion-dependence, steroid side effects (growth/Cushing), patient/family preference; "
                "BEST RESULTS: HLA-identical sibling donor, age < 10 years (OS >90%); "
                "Unrelated donor: OS 80-85% in modern series (DBAR data); "
                "Conditioning: reduced-intensity conditioning (RIC) preferred (myeloablative possible); "
                "TIMING: before iron overload develops (ferritin <1000 ng/mL ideal at time of transplant)"
            )
        },
        {
            "term": "Diamond-Blackfan Anemia Registry (DBAR)",
            "definition": (
                "Longest-running rare disease registry; founded 1992 by Jeffrey Lipton (Cohen Children's, NY); "
                "International collaboration: >700 patients enrolled; "
                "Key contributions: established eADA biomarker, cancer risk quantification, steroid response rates, "
                "  HSCT outcome data, genotype-phenotype correlations; "
                "Registry enrolment: all DBA patients should be registered"
            )
        },
        {
            "term": "Transient Erythroblastopenia of Childhood (TEC)",
            "definition": (
                "Acquired, self-limiting PRCA in previously healthy children 1-4 years; "
                "NORMAL eADA (key DDx from DBA); "
                "NORMAL MCV (no macrocytosis — DBA is macrocytic); "
                "Usually follows viral infection (parvovirus B19, others); "
                "Spontaneous recovery in 1-2 months; "
                "NO congenital anomalies; positive family history absent; "
                "CRITICAL DISTINCTION: eADA normal in TEC → DBA excluded"
            )
        },
        {
            "term": "Leucine Supplementation in DBA",
            "definition": (
                "Leucine (branched-chain essential amino acid) activates mTORC1 → enhances translation of "
                "remaining RP gene allele → partial correction of ribosomal haploinsufficiency; "
                "Dose: 0.5 g/kg/day oral; "
                "Response: ~50% show modest Hb improvement (+1-2 g/dL); may allow steroid sparing; "
                "Well tolerated; no serious side effects in DBA trials; "
                "Not curative; adjunct to primary therapy"
            )
        },
        {
            "term": "RPS14 / 5q- Syndrome Connection",
            "definition": (
                "5q- in myelodysplastic syndrome (MDS): interstitial deletion of chromosome 5q including RPS14; "
                "RPS14 (ribosomal protein S14) haploinsufficiency → same p53 pathway as DBA; "
                "Phenotype: macrocytic anaemia + hypolobated megakaryocytes (PATHOGNOMONIC of 5q-); "
                "DBA vs 5q- MDS: DBA = germline/congenital; 5q- MDS = acquired/somatic in adults; "
                "Lenalidomide FDA-approved for del5q MDS (mechanism: RPS14 restoration via degradation of CSNK1A1)"
            )
        },
    ],
    "standards": [
        "Diamond-Blackfan Anemia Registry (DBAR) — International DBA Registry (IDBAR)",
        "British Society for Haematology (BSH) Guidelines — DBA Diagnosis and Management",
        "European Working Group on MDS in Childhood (EWOG-MDS) — DBA Management Protocol",
        "OMIM: DBA1 (105650 / RPS19), DBA4 (612527 / RPS17), DBA5 (612528 / RPL35A), DBA6 (612561 / RPL5), DBA7 (612562 / RPL11), DBA10 (613309 / RPS26), DBA11 (614900 / RPL26), DBA13 (300946 / TSR2)",
        "Alter BP et al. Cancer Risk in Diamond-Blackfan Anemia (DBAR data)",
        "Vlachos A et al. Diagnosing and treating Diamond Blackfan anaemia: results of an international clinical consensus conference. Br J Haematol. 2008",
        "ClinVar / LOVD RPS19/RPL5/RPL11 variant databases",
        "ESID (European Society for Immunodeficiencies) — DBA working group",
    ]
}


def _make_patients(gene_data):
    rng = random.Random(gene_data["seed"])
    pts = []
    hb_lo, hb_hi = gene_data["pt_vars"]["hb_range"]
    reti_lo, reti_hi = gene_data["pt_vars"]["reticulocyte_pct"]
    mcv_lo, mcv_hi = gene_data["pt_vars"]["mcv_range"]
    eada_lo, eada_hi = gene_data["pt_vars"]["eada_range"]
    steroid_resp_pct = gene_data["pt_vars"]["steroid_response_pct"]

    for i in range(40):
        hb = round(rng.uniform(hb_lo, hb_hi), 1)
        reti = round(rng.uniform(reti_lo, reti_hi), 3)
        mcv = round(rng.uniform(mcv_lo, mcv_hi), 1)
        eada = round(rng.uniform(eada_lo, eada_hi), 2)
        age_diag = round(rng.uniform(0.1, 0.9), 2)  # years at diagnosis
        responds_steroid = rng.random() < steroid_resp_pct / 100
        transfusion_dep = not responds_steroid
        hsct = rng.random() < 0.18
        anomaly_present = rng.random() < 0.45

        sex = "M" if gene_data["gene"] == "TSR2" else rng.choice(["M", "F"])

        pts.append({
            "patient_id": f"{gene_data['gene']}-{i+1:03d}",
            "gene": gene_data["gene"],
            "sex": sex,
            "age_at_diagnosis_years": age_diag,
            "hb_at_diagnosis_gdl": hb,
            "reticulocyte_pct": reti,
            "mcv_fl": mcv,
            "eada_ugHb": eada,
            "eada_elevated": eada > 1.04,
            "steroid_response": responds_steroid,
            "transfusion_dependent": transfusion_dep,
            "hsct_performed": hsct,
            "congenital_anomaly_present": anomaly_present,
            "seed": gene_data["seed"],
        })
    return pts


def generate_overview():
    total_patients = 0
    all_genes = []

    for gene_data in ATLAS_GENES:
        patients = _make_patients(gene_data)
        total_patients += len(patients)
        steroid_resp = sum(1 for p in patients if p["steroid_response"])
        transfusion_dep = sum(1 for p in patients if p["transfusion_dependent"])
        hsct = sum(1 for p in patients if p["hsct_performed"])
        eada_high = sum(1 for p in patients if p["eada_elevated"])
        anomaly = sum(1 for p in patients if p["congenital_anomaly_present"])

        all_genes.append({
            "gene": gene_data["gene"],
            "locus": gene_data["locus"],
            "protein": gene_data["protein"],
            "disease_category": gene_data["disease_category"],
            "inheritance": gene_data["inheritance"],
            "n_patients": len(patients),
            "median_hb": round(
                sorted(p["hb_at_diagnosis_gdl"] for p in patients)[len(patients) // 2], 1
            ),
            "pct_eada_elevated": round(eada_high / len(patients) * 100, 1),
            "pct_steroid_response": round(steroid_resp / len(patients) * 100, 1),
            "pct_transfusion_dep": round(transfusion_dep / len(patients) * 100, 1),
            "pct_hsct": round(hsct / len(patients) * 100, 1),
            "pct_anomaly": round(anomaly / len(patients) * 100, 1),
            "seed": gene_data["seed"],
        })

    return {
        "atlas": "Hereditary Diamond-Blackfan Anaemia Atlas",
        "subtitle": "Complete 8-Gene DBA & Ribosomopathy Reference — RPS19·RPL5·RPL11·RPS26·RPL35A·RPS17·RPL26·TSR2",
        "genes": [g["gene"] for g in ATLAS_GENES],
        "total_patients": total_patients,
        "gene_summaries": all_genes,
        "seeds": "2806-2813",
        "pathway_categories": [
            {
                "pathway": "40S Ribosomal Small Subunit RP Genes",
                "genes": ["RPS19", "RPS26", "RPS17"],
                "note": (
                    "RPS genes assemble into 40S subunit (18S rRNA + ~33 RPS proteins); "
                    "40S + 60S → 80S ribosome; haploinsufficiency → 18S pre-rRNA accumulation → "
                    "nucleolar stress → RPL5·RPL11 freed → MDM2 inhibition → p53 → erythroid apoptosis; "
                    "40S RP genes (RPS) may have slightly higher steroid response than 60S RPL genes"
                ),
            },
            {
                "pathway": "60S Ribosomal Large Subunit RP Genes",
                "genes": ["RPL5", "RPL11", "RPL35A", "RPL26"],
                "note": (
                    "RPL genes assemble into 60S subunit (5.8S + 28S + 5S rRNA + ~47 RPL proteins); "
                    "RPL5 and RPL11 are DUAL FUNCTION: ribosomal structural + MDM2 inhibitors; "
                    "Free RPL5·RPL11 complex is the CENTRAL MEDIATOR of nucleolar stress p53 activation; "
                    "RPL5: cleft palate PATHOGNOMONIC; RPL11: thenar hypoplasia; RPL26: adjacent to TP53 on 17p13"
                ),
            },
            {
                "pathway": "Ribosome Assembly Chaperone (X-linked)",
                "genes": ["TSR2"],
                "note": (
                    "TSR2 = RPS26-specific nuclear import chaperone; NOT a ribosomal protein itself; "
                    "TSR2 haploinsufficiency → RPS26 cannot reach nucleolus → functional RPS26 deficiency; "
                    "Only X-linked DBA gene; hemizygous males affected; carrier females usually unaffected; "
                    "DIAGNOSTIC TRAP: RPS26 sequencing normal in TSR2-DBA → must test TSR2 on panel"
                ),
            },
            {
                "pathway": "MDM2·p53 Axis (Universal DBA Pathway)",
                "genes": ["RPS19", "RPL5", "RPL11", "RPS26", "RPL35A", "RPS17", "RPL26", "TSR2"],
                "note": (
                    "ALL 8 DBA genes converge on: RP haploinsufficiency → ribosomal stress → "
                    "free RPL5·RPL11 → MDM2 inhibition → p53 stabilisation → G1 arrest + erythroid apoptosis; "
                    "Selective for erythroid progenitors (highest ribosomal demand per cell cycle); "
                    "MDM2 inhibitors (nutlins/idasanutlin) counterproductive in DBA (increase p53 further)"
                ),
            },
        ],
        "critical_distinctions": [
            "DBA vs TEC: eADA ELEVATED in DBA (80-85%); eADA NORMAL in TEC — single most important DDx test; perform BEFORE steroids",
            "DBA vs Fanconi Anaemia: DBA = pure red cell aplasia (isolated erythropenia); FA = pancytopenia + chromosomal breakage (DEB/MMC test); DBA triphalangeal thumb vs FA thumb hypoplasia/aplasia",
            "RPS19 (most common, 25%) vs RPL5 (9%): RPL5 has CLEFT PALATE + highest cancer risk; RPS19 has no cleft; always check for cleft when DBA diagnosed",
            "RPL5 vs RPL11: RPL5 = cleft palate + thumb; RPL11 = thumb ONLY (no cleft); cleft differentiates",
            "TSR2 vs DKC1 (both X-linked): TSR2 = pure red cell aplasia; DKC1/DC = skin triad (nails+leukoplakia+pigmentation) + BMF — completely different syndromes at same X-linked locus neighbourhood",
            "RPL26 at 17p13.1: adjacent to TP53; del17p → COMPOUND haploinsufficiency (RPL26 + TP53); check MLPA for 17p deletion in RPL26-DBA patients",
            "eADA TIMING: revert to normal in DBA remission on steroids; always measure eADA BEFORE starting steroids for accurate diagnostic value",
            "CANCER RISK: RPL5 > RPL11 > RPS26 > RPS19; RPL5 patients: colonoscopy from age 20, annual whole-body MRI, breast MRI from age 25",
            "HSCT CURATIVE: only treatment that eliminates cancer risk; best outcomes sibling donor + age < 10; do NOT delay for steroid trial if sibling donor available early",
            "STEROID GROWTH FAILURE: if height SDS < −2 while on steroids → switch to transfusion programme; growth failure is an indication to switch, not to increase steroids",
        ],
    }


def generate_breakdown():
    result = []
    for gene_data in ATLAS_GENES:
        patients = _make_patients(gene_data)
        result.append({
            "gene": gene_data["gene"],
            "locus": gene_data["locus"],
            "protein": gene_data["protein"],
            "protein_size": gene_data["protein_size"],
            "inheritance": gene_data["inheritance"],
            "disease_category": gene_data["disease_category"],
            "disease_pathway": gene_data["disease_pathway"],
            "pathognomonic": gene_data["pathognomonic"],
            "treatment": gene_data["treatment"],
            "n_patients": len(patients),
            "patients": patients[:5],
        })
    return {"genes": result, "total": len(ATLAS_GENES), "seeds": "2806-2813"}


def generate_definitions():
    return {
        "atlas": "Hereditary Diamond-Blackfan Anaemia Atlas",
        "definitions": DEFINITIONS["definitions"],
        "standards": DEFINITIONS["standards"],
        "gene_count": len(ATLAS_GENES),
        "seeds": "2806-2813",
    }
