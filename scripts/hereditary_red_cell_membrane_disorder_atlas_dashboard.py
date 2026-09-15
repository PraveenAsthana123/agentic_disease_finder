"""Hereditary Red Cell Membrane Disorder Atlas — 8-Gene Reference
ANK1-SPTA1-SPTB-SLC4A1-EPB42-EPB41-PIEZO1-KCNN4
Hereditary Spherocytosis / Elliptocytosis / Stomatocytosis / Xerocytosis
320 patients (8 x 40), seeds 2846-2853.
Endpoints: /api/hereditary-red-cell-membrane-disorder-atlas/overview|breakdown|definitions
"""
import random

ATLAS_GENES = [
    {
        "gene": "ANK1",
        "protein": (
            "ANK1 -- 8p11.21 AD -- 1881aa -- Ankyrin-1-206kDa-Erythrocyte-Ankyrin-"
            "Anchors-Spectrin-Cytoskeleton-Band3-AE1-"
            "Most-Common-HS-Gene-60-65pct-Hereditary-Spherocytosis-"
            "OMIM-Gene-612641-Disease-HS1-182900"
        ),
        "locus": "8p11.21",
        "protein_size": (
            "1881 aa / 206 kDa (Ankyrin-1; anchors beta-spectrin cytoskeleton to Band 3 (SLC4A1) and Rh complex "
            "in the erythrocyte lipid bilayer; loss-of-function → spectrin-lipid bilayer uncoupling → "
            "membrane vesiculation → microspherocyte formation → splenic entrapment → hemolysis; "
            "most common HS gene accounting for 60-65% of all HS cases; "
            "autosomal dominant (new mutations account for ~25% of cases — de novo); "
            "repeat variants in exons 1-6 (spectrin-binding domain) most pathogenic; "
            "3 ankyrin-binding sites on Band 3 cytoplasmic N-terminal domain)"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT (AD) — most common; de novo mutations ~25%; "
            "EPIDEMIOLOGY: "
            "  HS prevalence 1:2000 Northern Europeans (highest); "
            "  ANK1 = 60-65% of HS; "
            "  SPTA1/SPTB combined ~20%; SLC4A1 ~15-20%; EPB42 ~2-3%; EPB41 <1% (HS phenotype rare); "
            "CLINICAL SPECTRUM: "
            "  MILD: Hb 11-15 g/dL, retic <10%, compensated; may go undiagnosed; "
            "  MODERATE: Hb 8-11 g/dL, retic 10-20%, jaundice, splenomegaly; "
            "  SEVERE: Hb <8 g/dL, retic >20%, transfusion-dependent; rare; "
            "TRIAD (CLASSIC): "
            "  1. Microspherocytes on blood film (>5% spherocytes); "
            "  2. Elevated MCHC (>36 g/dL) — water loss from membrane vesiculation; "
            "  3. Increased osmotic fragility; "
            "COMPLICATIONS: "
            "  Gallstones (pigment = unconjugated bilirubin): ~50% by age 30 if untreated; "
            "  Aplastic crisis: Parvovirus B19 (PVB19) ablates erythroid precursors → acute Hb drop 5-7 g/dL → medical emergency; "
            "  Megaloblastic crisis: folate depletion (high erythroid turnover); "
            "  Haemolytic crisis: intercurrent infection → splenic enlargement → acute worsening"
        ),
        "disease_category": (
            "HEREDITARY SPHEROCYTOSIS TYPE 1 (HS1) — OMIM 182900; "
            "RED CELL MEMBRANE STRUCTURAL DEFECT: "
            "  PRIMARY DEFECT: ANK1 loss → reduced ankyrin → reduced spectrin incorporation → "
            "    spectrin-lipid bilayer uncoupling → membrane lipid vesiculation; "
            "  MORPHOLOGY: microspherocytes — loss of central pallor; decreased deformability; "
            "  KEY LAB: "
            "    EMA (eosin-5′-maleimide) binding test: REDUCED to 70-85% of normal (flow cytometry); "
            "    most sensitive/specific HS screening test (>90% sensitivity); "
            "    Osmotic fragility: INCREASED (shift to left in incubation test); "
            "    Cryohemolysis: POSITIVE (chilled RBCs lyse at 4°C); "
            "    MCHC elevated (>36 g/dL); "
            "    MCV: low-normal (small spherocytes); "
            "ANAEMIA CLASSIFICATION: "
            "  Predominantly extravascular hemolysis (spleen filters rigid spherocytes); "
            "  LDH elevated (moderate); bilirubin elevated (unconjugated); "
            "  DAT (Coombs): NEGATIVE — distinguishes from autoimmune hemolysis"
        ),
        "disease_pathway": (
            "ERYTHROCYTE MEMBRANE CYTOSKELETON — ANKYRIN VERTICAL INTERACTION: "
            "NORMAL ARCHITECTURE: "
            "  Vertical (cytoskeleton-to-bilayer): Ankyrin-1 bridges beta-spectrin to Band 3 (AE1); "
            "  Secondary vertical: Protein 4.2 (EPB42) stabilises Ankyrin-Band 3 interaction; "
            "  Horizontal: spectrin heterodimers (SPTA1+SPTB) form lateral network; "
            "  Actin/Protein 4.1R (EPB41) junction complex links spectrin ends; "
            "  Result: RBC deformable enough to traverse 2-3 µm splenic sinusoids (RBC diameter 8 µm); "
            "ANK1 LOSS — MOLECULAR CONSEQUENCES: "
            "  Reduced ankyrin → spectrin-bilayer uncoupling → surface area loss; "
            "  Membrane blebbing (vesiculation) → microspherocyte (reduced surface-to-volume ratio); "
            "  Rigid spherocyte fails to traverse splenic red pulp → splenic hemolysis; "
            "  Splenomegaly develops over years → progressive hypersplenism; "
            "SPLEEN AS THERAPEUTIC TARGET: "
            "  Splenectomy removes filter → spherocytes survive → Hb normalises; "
            "  Spherocytes persist post-splenectomy (morphology unchanged; ankyrin defect persists); "
            "  OPSI RISK: Streptococcus pneumoniae, H. influenzae, N. meningitidis — vaccinate + penicillin prophylaxis"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC / HIGHLY CHARACTERISTIC: "
            "  1. EMA (eosin-5′-maleimide) flow cytometry: reduced to <80% of control — most specific HS screening test; "
            "  2. MICROSPHEROCYTES on peripheral blood film (loss of central pallor; MCV low-normal; MCHC >36 g/dL); "
            "  3. NEGATIVE DIRECT ANTIGLOBULIN TEST (DAT/Coombs) — excludes AIHA; "
            "  4. POSITIVE FAMILY HISTORY in AD cases (first-degree relative affected); "
            "  5. APLASTIC CRISIS trigger: PVB19 infection + rapid Hb drop 5-7 g/dL in days — pure red cell aplasia; "
            "  6. PIGMENT GALLSTONES: unconjugated bilirubin stones (not cholesterol); "
            "CRITICAL DIAGNOSTIC TRAP: "
            "  Do NOT use osmotic fragility as sole test — falsely normal in 10-20% mild HS; "
            "  EMA binding test superior; "
            "  MCHC >36 in context of hemolysis = spherocytosis until proven otherwise; "
            "NEONATAL HS: jaundice, anaemia, need phototherapy or exchange transfusion; "
            "EMA CUT-OFF: <80% of matched control mean = positive"
        ),
        "treatment": (
            "MANAGEMENT — ANK1-HS1: "
            "SUPPORTIVE: "
            "  Folic acid 1 mg/day MANDATORY (high RBC turnover depletes folate); "
            "  PVB19 alert: seek care if Hb drops acutely; "
            "  Penicillin V prophylaxis post-splenectomy (lifelong or minimum 2 years post-op); "
            "SPLENECTOMY: "
            "  INDICATION: Severe HS (Hb <8 g/dL, transfusion dependence); "
            "    Moderate HS with symptomatic gallstones (cholecystectomy concurrently); "
            "  DEFER <5 years (OPSI risk highest; vaccinate first); "
            "  Laparoscopic preferred; "
            "  Pre-operative: Pneumovax/Pneumococcal conjugate + Hib + Meningococcal ACWY + B; "
            "  Expected result: Hb normalises; reticulocytes fall; spherocytes PERSIST (morphology unchanged); "
            "MONITORING: "
            "  Annual FBC + bilirubin + LDH; "
            "  Ultrasound abdomen for gallstones every 3-5 years (or if symptomatic); "
            "  Post-splenectomy: monitor Howell-Jolly bodies (absence = splenic regrowth/splenosis); "
            "TRANSFUSION: aplastic crisis — often single PRBC transfusion + supportive; "
            "EMERGING: Mitapivat not applicable; luspatercept investigational for severe HS; "
            "CI: AVOID splenectomy in DHS (PIEZO1/KCNN4) — see those entries"
        ),
        "seed": 2846,
    },
    {
        "gene": "SPTA1",
        "protein": (
            "SPTA1 -- 1q23.1 AR-biallelic-severe / AD-mild -- 2429aa -- Spectrin-alpha-chain-erythrocyte-1-"
            "281kDa-Horizontal-Spectrin-Lattice-SPTA1-SPTB-Heterodimer-"
            "Biallelic-HPP-Most-Severe-HE-LELY-Allele-Modifier-"
            "OMIM-Gene-182860-Disease-HPP-266140-HE1-130500"
        ),
        "locus": "1q23.1",
        "protein_size": (
            "2429 aa / 281 kDa (Spectrin alpha-chain-erythrocyte-1; forms the alpha subunit of the spectrin heterodimer; "
            "SPTA1+SPTB heterodimers self-associate head-to-head to form spectrin tetramers; "
            "spectrin tetramers form the two-dimensional horizontal cytoskeletal lattice; "
            "SPTA1 contains 22 triple-helical repeat units + EF-hand Ca2+-binding domains + SH3 domain; "
            "alpha-spectrin is synthesised in 3-4× excess over beta-spectrin in normal erythropoiesis; "
            "LELY allele (Low Expression LYon): common polymorphism causing partial alpha-spectrin deficiency; "
            "LELY + severe SPTA1 mutation in trans → HPP phenotype; "
            "heterozygous SPTA1 mutations (one functional allele) → mild HE only due to alpha excess)"
        ),
        "inheritance": (
            "COMPLEX GENETICS — SPTA1: "
            "  BIALLELIC AR (homozygous or compound het): SEVERE — HPP (Hereditary Pyropoikilocytosis); "
            "  HETEROZYGOUS AD (with LELY in trans): MODERATE-SEVERE — HPP; "
            "  HETEROZYGOUS AD (without LELY): MILD — HE1 or asymptomatic; "
            "  HETEROZYGOUS normal allele + LELY only: ASYMPTOMATIC; "
            "LELY ALLELE: "
            "  Lys2069Ile (alpha-spectrin exon 46) + intron 45 alternative splice site; "
            "  Low expression (normal spectrin excess lost); "
            "  Common in African populations (~25% allele frequency); "
            "  In trans with severe SPTA1 mutation: LELY contributes to alpha-spectrin deficiency → HPP; "
            "HPP EPIDEMIOLOGY: "
            "  Rare; African-American families predominantly affected; "
            "  Presentation: neonatal haemolytic anaemia (severe, transfusion-dependent); "
            "  Thermal sensitivity: HPP erythrocytes lyse at 45-46°C (normal >49°C) — diagnostic; "
            "  Microspherocytes + poikilocytes + fragments; extreme poikilocytosis; "
            "SPTA1 alpha-spectrin TRUNCATION mutations: "
            "  p.Trp182Ter (codon 182), p.Arg45Ter — associated with severe alpha-spectrin deficiency"
        ),
        "disease_category": (
            "HPP (HEREDITARY PYROPOIKILOCYTOSIS) — OMIM 266140; HE TYPE 1 — OMIM 130500; "
            "SEVERITY SPECTRUM: "
            "  HPP: most severe; "
            "    Hb 5-9 g/dL; reticulocytes 15-30%; MCHC elevated; extreme poikilocytosis; "
            "    Requires transfusions; may need splenectomy (partial benefit); "
            "  HE (SPTA1 heterozygous, no LELY): mild; "
            "    Often asymptomatic; elliptocytes >25% on film; Hb normal or mildly low; "
            "DIAGNOSTIC HALLMARK: "
            "  RBC HEAT SENSITIVITY TEST: HPP cells lyse/fragment at 45-46°C; normal RBCs >49°C; "
            "  Ektacytometry: reduced deformability index (DI) at low osmolality; "
            "  EMA binding: reduced (like other membrane disorders); "
            "  Peripheral blood: microspherocytes + elliptocytes + ovalocytes + fragments + budding cells; "
            "NEONATAL PRESENTATION: "
            "  Severe HPP: jaundice at birth, phototherapy or exchange transfusion, transfusion-dependence; "
            "  Mild HE: often incidental finding in later childhood"
        ),
        "disease_pathway": (
            "ERYTHROCYTE MEMBRANE CYTOSKELETON — SPECTRIN HORIZONTAL LATTICE (SPTA1): "
            "SPECTRIN HETERODIMER ASSEMBLY: "
            "  Alpha (SPTA1) + Beta (SPTB) → non-covalent heterodimer (in tail-to-tail orientation); "
            "  Heterodimers self-associate head-to-head → tetramer; "
            "  Tetramers cross-linked by actin (6-7 spectrin ends per actin filament) + Protein 4.1R; "
            "SPTA1 MUTATIONS — MOLECULAR MECHANISMS: "
            "  1. Truncation/nonsense: alpha-spectrin deficiency → insufficient heterodimer → membrane instability; "
            "  2. Missense (repeat unit): defective self-association → heterodimer formation intact "
            "     but tetramer assembly impaired → abnormal horizontal lattice; "
            "  3. LELY in trans: normal alpha excess lost → net spectrin deficiency → vertical AND horizontal defect; "
            "HPP vs HE DISTINCTION: "
            "  HE (AD single hit): alpha excess maintains sufficient normal spectrin → mild elliptocytosis only; "
            "  HPP (biallelic or LELY in trans): alpha deficit → severe destabilisation → extreme poikilocytosis; "
            "THERMAL STABILITY EXPLANATION: "
            "  Normal spectrin melting temperature ~49°C (denatures and breaks cell); "
            "  SPTA1 mutations destabilise spectrin → lower melting point → lysis at 45-46°C"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC / HIGHLY CHARACTERISTIC — SPTA1-HPP: "
            "  1. RBC HEAT SENSITIVITY: fragmentation/lysis at 45-46°C (normal RBCs stable to >49°C) — PATHOGNOMONIC for HPP; "
            "  2. EXTREME POIKILOCYTOSIS: microspherocytes + elliptocytes + fragments + budding forms (not seen in other HAs); "
            "  3. NEONATAL SEVERE HEMOLYTIC ANEMIA: severe HPP presents in neonate; "
            "  4. AFRICAN-AMERICAN FAMILY HISTORY with HE in parents (mild) + HPP in child (biallelic); "
            "  5. EMA BINDING: reduced; "
            "  6. LELY ALLELE: common in African population; compound with SPTA1 mutation → HPP; "
            "DIAGNOSTIC TRAP: "
            "  Heterozygous SPTA1 alone (no LELY) → ONLY mild HE; "
            "  LELY status MUST be checked in suspected HPP (not on standard WES); "
            "  Ektacytometry: OSMOTIC GRADIENT EKTACYTOMETRY distinguishes HPP from HS definitively; "
            "HE (AD, single hit, no LELY): "
            "  >25% elliptocytes on film; "
            "  Usually asymptomatic adults; "
            "  Haemolytic exacerbation during intercurrent illness/fever"
        ),
        "treatment": (
            "MANAGEMENT — SPTA1 (HPP / HE): "
            "HPP (SEVERE): "
            "  Folic acid 5 mg/day (high erythroid demand); "
            "  Transfusion support: PRBCs for severe anaemia (HPP); "
            "  Splenectomy: partial benefit; "
            "    Hb improves (reduces by 1-2 g/dL less than in HS); "
            "    NOT fully curative as horizontal lattice defect persists; "
            "    Defer <5 years; post-op vaccinations + penicillin; "
            "MONITORING: "
            "  FBC monthly until stable post-diagnosis; "
            "  Neonatal: daily bilirubin monitoring; "
            "  Transfusion iron overload: ferritin monitoring if transfusion-dependent; "
            "HE (MILD, HETEROZYGOUS): "
            "  Usually observation only; "
            "  Folic acid supplementation; "
            "  Counsel on aplastic crisis (PVB19) risk; "
            "GENETIC COUNSELLING: "
            "  LELY allele frequency matters: African families at higher risk HPP; "
            "  Test partner for LELY if proband has SPTA1 mutation (risk for offspring); "
            "EMERGING: gene therapy under investigation; ektacytometry-guided splenectomy timing"
        ),
        "seed": 2847,
    },
    {
        "gene": "SPTB",
        "protein": (
            "SPTB -- 14q23.3 AD -- 2137aa -- Spectrin-beta-chain-erythrocyte-1-"
            "246kDa-Horizontal-Spectrin-Lattice-SPTA1-SPTB-Heterodimer-"
            "AD-Hereditary-Elliptocytosis-HS2-"
            "OMIM-Gene-182870-Disease-HE2-130600-HS2-616649"
        ),
        "locus": "14q23.3",
        "protein_size": (
            "2137 aa / 246 kDa (Spectrin beta-chain erythrocyte; beta subunit of spectrin heterodimer; "
            "contains 17 triple-helical repeat units + actin-binding calponin homology domains; "
            "C-terminal: pleckstrin homology (PH) domain (membrane PI(4,5)P2 binding); "
            "N-terminal: actin-binding domain (ABS) + interacts with Protein 4.1R at junctional complex; "
            "ankyrin-binding domain in repeat 14-15 region; "
            "unlike alpha-spectrin (3-4× excess), beta-spectrin produced in stoichiometric amounts; "
            "dominant SPTB mutations: truncations + self-association domain missense)"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT (AD): "
            "  HE2: most SPTB mutations → mild-moderate elliptocytosis; "
            "  HS2: ankyrin-binding or self-association mutations → spherocytosis phenotype; "
            "  Compound heterozygous SPTB + SPTA1 → HPP (very rare); "
            "EPIDEMIOLOGY: "
            "  SPTB-HE2 more common than SPTA1-HE1 in most populations; "
            "  Prevalence elliptocytosis 1:2000-4000 globally; "
            "  West Africa, Cameroon, Italy: higher prevalence; "
            "CLINICAL SPECTRUM: "
            "  Most heterozygous carriers ASYMPTOMATIC; "
            "  HE2 MILD: elliptocytes on film; no anaemia; compensated hemolysis; "
            "  HE2 MODERATE: mild anaemia (Hb 10-12 g/dL), reticulocytosis, splenomegaly; "
            "  SPHEROCYTIC HE (HS2): splenomegaly, moderate anaemia, sphero-elliptocytes; "
            "  ACUTE HAEMOLYSIS: fever, infection, pregnancy trigger acute exacerbation"
        ),
        "disease_category": (
            "HEREDITARY ELLIPTOCYTOSIS TYPE 2 (HE2) / HS TYPE 2 — OMIM 130600 / 616649; "
            "ELLIPTOCYTE MORPHOLOGY: "
            "  >25% elongated elliptocytes on peripheral film (normal <5%); "
            "  Long axis : short axis ratio >2:1 in severe cases; "
            "  No poikilocytes (distinguishes HE from HPP); "
            "LABORATORY: "
            "  EMA binding: mildly-to-moderately reduced; "
            "  Osmotic fragility: NORMAL or mildly increased; "
            "  Ektacytometry: reduced DI (deformability index); elongated osmotic ektacytometry curve; "
            "  LDH + bilirubin: mildly elevated (compensated hemolysis); "
            "HS2 PHENOTYPE (SPTB ankyrin-binding repeat mutations): "
            "  Mixed sphero-elliptocyte morphology; "
            "  More severe anaemia (Hb 8-11 g/dL); "
            "  Elevated MCHC; increased osmotic fragility; "
            "DAT: NEGATIVE in all HE/HS"
        ),
        "disease_pathway": (
            "ERYTHROCYTE MEMBRANE CYTOSKELETON — BETA-SPECTRIN LATERAL LATTICE (SPTB): "
            "SELF-ASSOCIATION DOMAIN: "
            "  N-terminal 156 amino acids of SPTB + C-terminal of SPTA1 → head-to-head self-association site; "
            "  Mutations here (e.g. p.Leu260Pro, p.Arg45Ser) → impaired tetramer formation; "
            "  Result: excess spectrin dimers (not tetramers) → unstable horizontal lattice; "
            "TRUNCATION MUTATIONS: "
            "  Premature stop codon: truncated beta-spectrin; "
            "  Truncations in repeat 15-17 → loss of C-terminal PH domain → membrane attachment defect; "
            "  Haploinsufficiency (50% SPTB) → spectrin deficiency → vertical + horizontal defects; "
            "  Reduced membrane stability → elliptocytes (mild) or spherocytes (severe); "
            "ACTIN-BINDING DOMAIN: "
            "  N-terminal ABD mutations → disruption of spectrin-actin junctional complex; "
            "  More severe phenotype when junctional complex disrupted"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC / HIGHLY CHARACTERISTIC — SPTB-HE2/HS2: "
            "  1. ELLIPTOCYTES >25% of RBCs on peripheral film (in HE2); "
            "  2. NO SIGNIFICANT POIKILOCYTOSIS — distinguishes from HPP (SPTA1); "
            "  3. NEGATIVE DAT — excludes autoimmune; "
            "  4. EKTACYTOMETRY: elongated curve + reduced DI at low osmolality; "
            "  5. AD FAMILY HISTORY: parent-to-child transmission; "
            "  6. MALARIA PARADOX: elliptocytosis common in malaria-endemic Africa — mild carriers protected; "
            "DIAGNOSTIC TRAP: "
            "  Elliptocytes occur in iron deficiency, megaloblastic anaemia, myelodysplasia; "
            "  EMA binding: must compare to matched healthy control; "
            "  SPTB HS2: sphero-elliptocytes — may look like HS; EMA + family history clarify; "
            "  Self-association domain mutations: may only be detected by protein electrophoresis "
            "  (reduced tetramer:dimer ratio on native PAGE)"
        ),
        "treatment": (
            "MANAGEMENT — SPTB-HE2/HS2: "
            "MILD HE2: "
            "  No specific therapy; "
            "  Folic acid 1 mg/day if any hemolysis; "
            "  Monitor for aplastic crisis (PVB19); "
            "  Counsel regarding disease-exacerbating illnesses; "
            "MODERATE-SEVERE HE2 / HS2: "
            "  Splenectomy: partial-to-complete benefit (better for HS2 than HE2); "
            "  Indication: Hb <10 g/dL sustained + symptomatic; "
            "  Laparoscopic; defer <5 years; vaccinate + penicillin; "
            "  Post-op: elliptocytes persist but Hb improves; "
            "MONITORING: "
            "  FBC annually; bilirubin; ultrasound abdomen (gallstones); "
            "  Neonatal SPTB-HS2: close monitoring for jaundice; exchange transfusion may be needed; "
            "GENETIC COUNSELLING: "
            "  AD: 50% risk to offspring; "
            "  Prenatal diagnosis available if variant confirmed"
        ),
        "seed": 2848,
    },
    {
        "gene": "SLC4A1",
        "protein": (
            "SLC4A1 -- 17q21.31 AD-HS3-SAO / AR-dRTA-HA -- 911aa -- Band-3-Anion-Exchanger-1-AE1-"
            "102kDa-Cl-HCO3-Exchanger-CO2-Transport-Vertical-Anchor-Ankyrin-"
            "AD-HS3-Southeast-Asian-Ovalocytosis-SAO-AR-dRTA-Hemolytic-Anemia-"
            "OMIM-Gene-109270-Disease-HS3-270970-SAO-166900-dRTA-611590"
        ),
        "locus": "17q21.31",
        "protein_size": (
            "911 aa / 102 kDa (Band 3 = Anion Exchanger 1 AE1; most abundant RBC membrane protein (~1×10⁶ copies/cell); "
            "two structural domains: "
            "  N-terminal cytoplasmic domain (412 aa): binds ankyrin-1, protein 4.1R, protein 4.2, haemoglobin, "
            "    aldolase, glyceraldehyde-3-phosphate dehydrogenase; "
            "  C-terminal transmembrane domain (~500 aa): 14 transmembrane spans; Cl⁻/HCO₃⁻ exchanger; "
            "  function: CO₂ as HCO₃⁻ in red cells → transported to lungs → CO₂ exhaled; "
            "  GPI-linked glycophorin C anchored through protein 4.1R; "
            "N-terminal domain truncation (p.Glu40Ter): SAO — produces rigid ovalocytes)"
        ),
        "inheritance": (
            "DUAL INHERITANCE PATTERN — SLC4A1: "
            "AD PHENOTYPES: "
            "  HS3: missense/frameshift in ankyrin/cytoskeletal-binding N-terminal domain → spherocytosis; "
            "    Hb 10-13 g/dL; moderate HS; standard HS management; "
            "  SAO (SOUTHEAST ASIAN OVALOCYTOSIS): "
            "    p.Ala400_Ala408del (27-bp deletion in transmembrane domain) + p.Lys56Glu; "
            "    AD; near-ubiquitous among indigenous SE Asians (Philippines, Malaysia, Papua New Guinea, Indonesia); "
            "    RIGID OVALOCYTES: band 3 dimer locked in rigid conformation → cells resist deformation; "
            "    MILD ANAEMIA or compensated; often asymptomatic; "
            "    PROTECTION: SAO cells RESIST Plasmodium falciparum invasion → protective against cerebral malaria; "
            "    LETHAL HOMOZYGOUS: no live-born SAO homozygotes (embryo lethal); "
            "AR PHENOTYPES: "
            "  dRTA + haemolytic anaemia (SLC4A1 dRTA): "
            "    Biallelic loss of kidney AE1 (kAE1) isoform; "
            "    Metabolic acidosis (cannot acidify urine); nephrocalcinosis; growth retardation; "
            "    Haemolytic anaemia (Southeast Asian AR variant: p.Gly701Asp, p.Ser773Pro); "
            "    Nephrocalcinosis → renal failure if untreated; "
            "    Bicarbonate supplementation corrects acidosis + improves growth"
        ),
        "disease_category": (
            "HS3 (AD-AE1 HS) / SAO (AD-RIGID OVALOCYTOSIS) / dRTA-HA (AR-RENAL): "
            "HS3 PHENOTYPE: "
            "  Spherocytes; elevated MCHC; increased osmotic fragility; EMA reduced; "
            "  Clinical severity: mild-moderate HS; "
            "SAO PHENOTYPE: "
            "  RIGID OVALOCYTES on film: oval/elliptical; thickened rim; transverse ridge across cell; "
            "  EMA BINDING: increased (unique to SAO — band 3 EMA-accessible dimer); "
            "  OSMOTIC FRAGILITY: DECREASED (rigid cells resist hypotonic lysis — opposite HS); "
            "  Malaria protection: P. falciparum MEROZOITE INVASION blocked by rigid ovalocyte membrane; "
            "  CRITICAL: SAO cells are rigid but NOT fragile; do NOT confuse with HS; "
            "  SAO-dRTA: compound phenotype — some SAO patients have dRTA (heterozygous truncated kAE1); "
            "dRTA-HA PHENOTYPE: "
            "  Metabolic acidosis (urine pH cannot fall below 5.5); "
            "  Nephrocalcinosis on ultrasound; "
            "  Growth failure; rickets (type 3); "
            "  Haemolytic anaemia (variable severity)"
        ),
        "disease_pathway": (
            "BAND 3 — VERTICAL MEMBRANE ANCHOR + ANION EXCHANGER: "
            "VERTICAL INTERACTION ROLE: "
            "  AE1 N-terminal: bound by ankyrin-1 → connects spectrin cytoskeleton to lipid bilayer; "
            "  AE1 bound by protein 4.2 (EPB42): stabilises the AE1-ankyrin interaction; "
            "  AE1 also binds protein 4.1R via glycophorin C (secondary vertical link); "
            "HS3 MECHANISM: "
            "  N-terminal missense/truncation mutations → impaired ankyrin binding → "
            "  vertical uncoupling → membrane vesiculation → microspherocytes → splenic hemolysis; "
            "SAO MECHANISM (27-bp deletion): "
            "  27-bp deletion: AE1 transmembrane domain adopts locked dimer conformation; "
            "  Locks cell shape as ovalocyte; increases membrane rigidity 3-10 fold; "
            "  Cl⁻/HCO₃⁻ exchange activity ABSENT in SAO (transport-null mutation); "
            "  CO₂ transport: compensated by alternative routes; "
            "  SAO-specific: resistant to echinocytosis by ATP-depletion; "
            "dRTA MECHANISM: "
            "  kAE1 (kidney AE1, N-terminal truncated isoform): expressed in alpha-intercalated collecting duct cells; "
            "  AR mutations misroute kAE1 to apical (wrong) membrane → HCO₃⁻ absorption impaired → "
            "  failure to excrete acid → distal RTA; "
            "  eAE1 (erythrocyte AE1) may be partially affected in some AR mutations"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC / HIGHLY CHARACTERISTIC — SLC4A1: "
            "SAO: "
            "  1. RIGID OVALOCYTES with transverse ridge on peripheral film — PATHOGNOMONIC for SAO; "
            "  2. DECREASED OSMOTIC FRAGILITY (opposite of HS — cells resist lysis); "
            "  3. EMA BINDING INCREASED (unique to SAO among red cell membrane disorders); "
            "  4. Southeast Asian ethnicity + malaria-endemic ancestry; "
            "  5. LETHAL HOMOZYGOUS: screen partner if SAO identified; "
            "HS3: "
            "  Spherocytes; negative DAT; EMA reduced; positive family history; "
            "dRTA-HA: "
            "  1. METABOLIC ACIDOSIS + NORMAL ANION GAP + HIGH URINE pH (>5.5 despite systemic acidosis); "
            "  2. NEPHROCALCINOSIS on renal ultrasound; "
            "  3. HAEMOLYTIC ANEMIA in same patient; "
            "  4. SE Asian ethnicity (common AR founder); "
            "CRITICAL DIAGNOSTIC TRAPS: "
            "  SAO mistaken for elliptocytosis (different mechanism; EMA increased not decreased); "
            "  dRTA-HA: do NOT give ammonium chloride loading test (not needed; urine pH already elevated); "
            "  SAO-dRTA compound: band 3 gene — always check renal function in SE Asian ovalocytosis"
        ),
        "treatment": (
            "MANAGEMENT — SLC4A1 (HS3 / SAO / dRTA-HA): "
            "HS3: "
            "  Same as ANK1-HS1: folic acid; splenectomy for severe/symptomatic; vaccinations; penicillin; "
            "  Gallstone monitoring; PVB19 aplastic crisis awareness; "
            "SAO: "
            "  USUALLY NO TREATMENT REQUIRED; compensated or mild anaemia; "
            "  Folic acid supplementation; "
            "  CRITICAL: AVOID SPLENECTOMY in SAO — rigid ovalocytes do NOT benefit; "
            "    splenectomy may unmask or worsen hemolysis without benefit; "
            "    Thrombosis risk post-splenectomy elevated in SAO (rigid cells); "
            "  Malaria prevention: standard antimalarial prophylaxis; "
            "  Genetic counselling: do NOT have children with another SAO carrier (homozygous lethal); "
            "dRTA-HA: "
            "  BICARBONATE SUPPLEMENTATION: sodium bicarbonate or citrate 1-3 mEq/kg/day; "
            "  Goal: normalise serum bicarbonate → prevents nephrocalcinosis progression; "
            "    improves linear growth; reduces hemolysis severity; "
            "  Monitoring: serum electrolytes; bicarbonate; renal ultrasound; eGFR; growth charts; "
            "  Diuretics: thiazide diuretics reduce urinary calcium → slow nephrocalcinosis; "
            "  Avoid: acidifying foods; NSAIDs (nephrotoxic); nephrotoxic aminoglycosides"
        ),
        "seed": 2849,
    },
    {
        "gene": "EPB42",
        "protein": (
            "EPB42 -- 15q15.2 AR -- 721aa -- Erythrocyte-Membrane-Protein-Band-4.2-"
            "77kDa-Palmitoylated-Transglutaminase-Homologue-Stabilises-AE1-Ankyrin-Complex-"
            "AR-HS5-Japanese-Founder-Southern-Mediterranean-"
            "OMIM-Gene-177070-Disease-HS5-612690"
        ),
        "locus": "15q15.2",
        "protein_size": (
            "721 aa / 77 kDa (Erythrocyte membrane protein band 4.2; also known as Protein 4.2; "
            "palmitoylated peripheral membrane protein; structurally homologous to transglutaminases (inactive); "
            "binds AE1 N-terminal cytoplasmic domain + ankyrin-1; "
            "stabilises the AE1-ankyrin-1-spectrin vertical interaction complex; "
            "copiously expressed (~2×10⁵ molecules per RBC); "
            "essential for optimal anchoring of spectrin cytoskeleton to lipid bilayer; "
            "loss → reduced ankyrin/spectrin content → membrane vesiculation → microspherocytes; "
            "ethnic-specific founder variants: p.Ala142Thr (Japanese HS5 founder); "
            "p.Tyr142His (Southern Mediterranean); p.Glu90Lys (Japanese); c.523G>T (Japanese))"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (AR): "
            "  HOMOZYGOUS or COMPOUND HETEROZYGOUS → HS5; "
            "  HETEROZYGOUS carriers: usually asymptomatic (normal RBC morphology); "
            "ETHNIC DISTRIBUTION: "
            "  JAPANESE: most common non-European HS gene; p.Ala142Thr founder; "
            "    Japanese HS5 may account for 50% of Japanese HS cases; "
            "    Prevalence higher in Japan than European HS5; "
            "  SOUTHERN MEDITERRANEAN (Greece, Italy, Sardinia, Spain): "
            "    p.Tyr142His founder + p.Glu90Lys; "
            "  EUROPEAN (non-Mediterranean): rare; "
            "  AFRICAN-AMERICAN: very rare; "
            "SEVERITY: "
            "  Variable (milder than SPTA1-HPP); "
            "  Hb 9-12 g/dL typically; "
            "  Moderate hemolytic anemia with splenomegaly; "
            "  Gallstones; folic acid depletion; "
            "COMPLETE PROTEIN 4.2 ABSENCE: "
            "  Homozygous null mutations → COMPLETE band 4.2 absence on SDS-PAGE; "
            "  Diagnosis: band 4.2 absent band on erythrocyte membrane protein gel (SDS-PAGE)"
        ),
        "disease_category": (
            "HEREDITARY SPHEROCYTOSIS TYPE 5 (HS5) — OMIM 612690; "
            "MICROSPHEROCYTOSIS: "
            "  Spherocytes + microspherocytes on peripheral film; "
            "  Reduced surface-to-volume ratio; "
            "  MCHC elevated (>36 g/dL); "
            "  Increased osmotic fragility; "
            "LABORATORY: "
            "  EMA BINDING: REDUCED (Band 4.2 loss → Band 3 topology altered → EMA-accessible sites reduced); "
            "  SDS-PAGE membrane protein electrophoresis: ABSENT BAND 4.2 BAND (diagnostic); "
            "  LDH + indirect bilirubin elevated; "
            "  DAT: NEGATIVE; "
            "  Reticulocytes: 5-20%; "
            "  Haptoglobin: reduced; "
            "CLINICAL: "
            "  Mild-moderate hemolytic anaemia; "
            "  Splenomegaly in most patients; "
            "  Gallstones (pigment); "
            "  Aplastic crisis risk (PVB19); "
            "  Well-compensated in mild cases (asymptomatic until illness)"
        ),
        "disease_pathway": (
            "ERYTHROCYTE MEMBRANE — PROTEIN 4.2 VERTICAL STABILISER FUNCTION (EPB42): "
            "MOLECULAR ROLE: "
            "  Band 4.2 binds AE1 N-terminal cytoplasmic domain (aa 1-360) → stabilises AE1 in membrane; "
            "  Band 4.2 also binds Ankyrin-1 → stabilises AE1-ankyrin bridge; "
            "  Band 4.2 loss → AE1-ankyrin bridge weakened → spectrin-bilayer coupling reduced; "
            "  Secondary: ankyrin content reduced (30-50%) in 4.2-deficient RBCs; "
            "  Beta-spectrin content also reduced (20-30%); "
            "  Net: VERTICAL INSTABILITY → membrane vesiculation → microspherocytes; "
            "TRANSGLUTAMINASE HOMOLOGY: "
            "  Structurally homologous to transglutaminases but CATALYTICALLY INACTIVE; "
            "  No crosslinking activity; structural scaffold role only; "
            "  Three-dimensional structure: TGase fold (beta-sandwich + barrel); "
            "ETHNIC FOUNDER VARIANT MECHANISM: "
            "  p.Ala142Thr (Japanese): threonine introduces bulky + hydrophilic residue → protein misfolding + ER retention; "
            "  p.Tyr142His: similar position → same mechanism; "
            "  c.523G>T: introduces premature stop → haploinsufficiency (AR)"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC / HIGHLY CHARACTERISTIC — EPB42-HS5: "
            "  1. SDS-PAGE MEMBRANE PROTEIN GEL: ABSENT BAND 4.2 — PATHOGNOMONIC for homozygous HS5; "
            "  2. EMA BINDING REDUCED: confirms membrane protein defect; "
            "  3. JAPANESE or MEDITERRANEAN ANCESTRY: ethnic founder variants; "
            "  4. AR PATTERN: both parents often carriers (mildly abnormal film or normal); "
            "  5. MICROSPHEROCYTES + elevated MCHC + increased osmotic fragility + negative DAT: "
            "     standard HS lab triad; "
            "DIAGNOSTIC APPROACH: "
            "  EMA flow cytometry: screen (reduced); "
            "  SDS-PAGE: confirms absent band 4.2 (diagnostic in homozygous); "
            "  Molecular: sequence EPB42 gene (especially p.Ala142Thr in Japanese patients); "
            "  Next-gen sequencing panel (ANK1+SPTB+SPTA1+SLC4A1+EPB42) if ethnicity suggests HS5; "
            "CLINICAL TRAP: "
            "  Heterozygous carriers: NO phenotype → family screening by molecular testing only; "
            "  AR pattern: may be missed without family testing"
        ),
        "treatment": (
            "MANAGEMENT — EPB42-HS5: "
            "STANDARD HS APPROACH: "
            "  Folic acid 1 mg/day (mandatory); "
            "  Hydroxyurea: not standard for HS; "
            "  Transfusion for severe anaemia; aplastic crisis management; "
            "SPLENECTOMY: "
            "  Indication: Hb <9 g/dL sustained or symptomatic; "
            "  EFFECTIVE: HS5 responds well to splenectomy; Hb normalises post-op; "
            "  Defer <5 years; laparoscopic preferred; "
            "  Pre-op: pneumococcal + Hib + meningococcal vaccines; "
            "  Post-op: penicillin V lifelong (or until adulthood); "
            "MONITORING: "
            "  Annual FBC + bilirubin + LDH; "
            "  Ultrasound abdomen: gallstones every 3-5 years; "
            "  Concurrent cholecystectomy if gallstones symptomatic at time of splenectomy; "
            "GENETIC COUNSELLING: "
            "  AR: parents usually unaffected carriers; "
            "  Sibling risk 25%; "
            "  Partner testing if founder variant population"
        ),
        "seed": 2850,
    },
    {
        "gene": "EPB41",
        "protein": (
            "EPB41 -- 1p35.3 AD -- 823aa -- Erythrocyte-Membrane-Protein-Band-4.1-"
            "80kDa-FERM-Domain-Spectrin-Actin-Junctional-Complex-Stabiliser-"
            "AD-Hereditary-Elliptocytosis-HE1-Southeast-Asian-Malaria-Protection-"
            "OMIM-Gene-130500-Disease-HE1-130500"
        ),
        "locus": "1p35.3",
        "protein_size": (
            "823 aa / 80 kDa (Erythrocyte membrane protein band 4.1R; large isoform (4.1R); "
            "FERM domain (4.1R/Ezrin/Radixin/Moesin homology domain) at N-terminus — binds glycophorin C/D; "
            "spectrin-actin binding domain (SABD): central 10-kDa domain binds SPTA1+SPTB+F-actin simultaneously; "
            "linker between vertical (GPC-membrane) and horizontal (spectrin-actin) skeleton; "
            "multiple isoforms by alternative splicing (4.1a, 4.1b, 4.1R, 4.1N, 4.1G); "
            "4.1R predominantly erythroid (exon 16 SABD inclusion); "
            "protein 4.1R loss → junctional complex weakened → elliptocytes/poikilocytes; "
            "homozygous loss → HPP-like severe phenotype)"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT (AD) — HAPLOINSUFFICIENCY: "
            "  Most HETEROZYGOUS carriers: MILD HE or ASYMPTOMATIC; "
            "  Exacerbated in: febrile illness, malaria, pregnancy; "
            "ETHNIC DISTRIBUTION: "
            "  HIGH FREQUENCY in SE Asia (Malaysia, Philippines): "
            "    4.1R deficiency allele frequency ~0.1-0.5% in some SE Asian populations; "
            "    Mild HE carriers have some protection against P. falciparum malaria (like SAO); "
            "  Algeria, North Africa: founder variants; "
            "  European: rare; "
            "HOMOZYGOUS RARE: "
            "  Compound het or homozygous → severe HPP-like phenotype; "
            "  Complete 4.1R absence on SDS-PAGE; "
            "  Neonatal haemolytic anaemia; poikilocytosis; "
            "CLINICAL SPECTRUM: "
            "  MILD (het): elliptocytes, no anaemia, no treatment needed; "
            "  MODERATE (het with modifier): anaemia Hb 10-12 g/dL, mild splenomegaly; "
            "  SEVERE (homozygous): HPP-like; severe CNHA; poikilocytes + elliptocytes"
        ),
        "disease_category": (
            "HEREDITARY ELLIPTOCYTOSIS TYPE 1 (HE1) — OMIM 130500; "
            "ELLIPTOCYTE MORPHOLOGY: "
            "  >25% elongated elliptocytes (HE1 het); "
            "  Some ovalocytes; occasional stomatocytes; "
            "  Homozygous: extreme poikilocytosis (HPP-like); "
            "LABORATORY: "
            "  SDS-PAGE: ABSENT or REDUCED BAND 4.1 (diagnostic in homozygous/compound het); "
            "  EMA BINDING: mildly reduced; "
            "  Osmotic fragility: NORMAL (unlike HS); "
            "  Ektacytometry: reduced DI; elongated ektacytometry curve; "
            "  Glycophorin C: REDUCED (GPC anchored through 4.1R); "
            "    Reduced GPC by flow cytometry + Western blot; "
            "  DAT: NEGATIVE; "
            "MALARIA PROTECTION: "
            "  4.1R required for P. falciparum invasion (P. falciparum binds GPC via EBA-140 using 4.1R); "
            "  Reduced GPC in 4.1R-null cells → reduced EBA-140 binding → merozoite invasion blocked"
        ),
        "disease_pathway": (
            "ERYTHROCYTE MEMBRANE — PROTEIN 4.1R JUNCTIONAL COMPLEX (EPB41): "
            "JUNCTIONAL COMPLEX STRUCTURE: "
            "  One short actin filament (14-16 monomers) + "
            "    Protein 4.1R (bridges spectrin ends + actin) + "
            "    Adducin (caps actin + stabilises spectrin-actin) + "
            "    Dematin (p55) + Tropomyosin + Tropomodulin (caps pointed actin end); "
            "  Six spectrin ends (from 3 spectrin tetramers) attach to each actin filament; "
            "  Result: hexagonal lattice of spectrin-actin; "
            "4.1R SABD FUNCTION: "
            "  10-kDa SABD binds SPTA1 (repeat 15) + SPTB (N-terminal) + F-actin simultaneously; "
            "  Stabilises spectrin-actin junction; loss → junction weak → spectrin-actin detachment; "
            "  Lateral (horizontal) lattice destabilised → elliptocytes; "
            "GPC VERTICAL LINK: "
            "  4.1R FERM binds glycophorin C N-terminal cytoplasmic domain; "
            "  GPC → 4.1R → spectrin-actin: secondary vertical linkage (like Band 3 → ankyrin → spectrin); "
            "  4.1R loss → GPC-membrane attachment defect → GPC reduced → malaria protection lost"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC / HIGHLY CHARACTERISTIC — EPB41-HE1: "
            "  1. SDS-PAGE MEMBRANE PROTEIN GEL: ABSENT BAND 4.1 (homozygous/compound het) — diagnostic; "
            "  2. GLYCOPHORIN C REDUCED by flow cytometry + Western blot (secondary to 4.1R loss); "
            "  3. ELLIPTOCYTES >25% on peripheral blood film; "
            "  4. SE ASIAN or NORTH AFRICAN ancestry (founder alleles); "
            "  5. POSITIVE FAMILY HISTORY: AD — parents with mild elliptocytosis; "
            "  6. EXACERBATION WITH FEVER/MALARIA: acute haemolysis during febrile illness; "
            "DIAGNOSTIC TRAP: "
            "  Heterozygous carriers: OFTEN ASYMPTOMATIC — may be missed unless specifically looking; "
            "  Reduced GPC on flow: may be mistaken for PNH (GPI-anchored, different mechanism); "
            "    Confirm: 4.1R Western blot + DAT + Ham's test/FLAER distinguishes; "
            "  Homozygous: HPP-like — severe neonatal HA; distinguish from SPTA1-HPP by SDS-PAGE bands"
        ),
        "treatment": (
            "MANAGEMENT — EPB41-HE1: "
            "MILD HETEROZYGOUS HE1: "
            "  Usually OBSERVATION ONLY; "
            "  Folic acid 1 mg/day if any hemolysis; "
            "  Patient education: report acute pallor/jaundice (aplastic crisis); "
            "  Malaria prevention (standard prophylaxis in endemic areas); "
            "MODERATE-SEVERE: "
            "  Splenectomy: benefit in severe heterozygous HE1 or homozygous; "
            "  Homozygous: splenectomy may be life-saving; "
            "  Defer <5 years; laparoscopic; pre-op vaccines; post-op penicillin; "
            "MONITORING: "
            "  FBC + bilirubin + LDH annually; "
            "  Growth monitoring in children (if anaemic); "
            "  Gallstone surveillance (if haemolytic); "
            "GENETIC COUNSELLING: "
            "  AD mild HE: reassure; 50% risk to offspring; most will have mild/no symptoms; "
            "  Homozygous risk: if both parents have HE1 allele → 25% risk severe phenotype"
        ),
        "seed": 2851,
    },
    {
        "gene": "PIEZO1",
        "protein": (
            "PIEZO1 -- 16q24.3 AD-DHS / AR-LMPHD3 -- 2521aa -- Piezo-Type-Mechanosensitive-Ion-Channel-"
            "Component-1-286kDa-Trimeric-Homotrimer-Mechanically-Activated-Ca2plus-Permeable-Cation-Channel-"
            "GOF-Dehydrated-Hereditary-Stomatocytosis-DHS-Xerocytosis-ABSOLUTE-CI-Splenectomy-Fatal-Thrombosis-"
            "OMIM-Gene-611184-Disease-DHS-194380-LMPHD3-616843"
        ),
        "locus": "16q24.3",
        "protein_size": (
            "2521 aa / 286 kDa monomer (PIEZO1 homotrimer forms propeller-shaped mechanosensitive channel; "
            "3 blades × ~700 aa each + central ion-conducting pore; "
            "mechanically-gated: membrane tension opens channel → Ca2+ influx; "
            "Ca2+ influx activates KCa3.1 (KCNN4 Gardos channel) → K+ efflux + H2O loss → "
            "RBC dehydration → increased MCHC → xerocytes/stomatocytes; "
            "GOF mutations: gain-of-function → channel stays open at lower tension threshold → "
            "excessive Ca2+ entry → persistent K+ loss → dehydrated RBCs; "
            "AR LMPHD3: biallelic loss-of-function → generalised lymphatic dysplasia + HDFN)"
        ),
        "inheritance": (
            "DUAL INHERITANCE MECHANISM: "
            "AD GOF — Dehydrated Hereditary Stomatocytosis (DHS) / Hereditary Xerocytosis: "
            "  GOF missense mutations: channel over-activated → Ca2+ excess → K+ loss → dehydration; "
            "  Most common PIEZO1 DHS mutation: p.Arg2456His; p.Arg1334Trp; "
            "  EPIDEMIOLOGY: "
            "    DHS/xerocytosis: rare (estimated 1:50,000-100,000); "
            "    Common in patients initially diagnosed with congenital haemolytic anaemia of unknown cause; "
            "  CLINICAL: Hb 10-14 g/dL; compensated hemolysis; "
            "    PSEUDOHYPERKALEMIA: falsely elevated serum K+ at room temperature (K+ leaks from dehydrated RBCs); "
            "    MCHC >36 g/dL (PATHOGNOMONIC); "
            "    Stomatocytes on fresh blood film (may normalise in stored samples); "
            "    Perinatal oedema (hydrops fetalis): uncommon but reported; "
            "AR LOF — LMPHD3 (Lymphatic dysplasia-3 + HDFN): "
            "  Biallelic PIEZO1 loss → generalised lymphatic dysplasia; "
            "  Haemolytic disease of the fetus/newborn (HDFN); "
            "  Severe lymphoedema; "
            "  Distinct phenotype from AD DHS"
        ),
        "disease_category": (
            "DHS (DEHYDRATED HEREDITARY STOMATOCYTOSIS) / HEREDITARY XEROCYTOSIS — OMIM 194380; "
            "OVERHYDRATED STOMATOCYTOSIS (OHS) — OPPOSITE PHENOTYPE (SLC4A2/RHAG mutations, not PIEZO1); "
            "KEY DISTINCTION DHS vs NORMAL HS: "
            "  DHS: MCHC ELEVATED (dehydrated cells); "
            "  HS: MCHC elevated (different mechanism — microspherocytes); "
            "  DHS: OSMOTIC FRAGILITY REDUCED or NORMAL (stiff cells resist hypotonic lysis); "
            "  HS: OSMOTIC FRAGILITY INCREASED; "
            "  DHS: EKTACYTOMETRY = characteristic xerocyte profile (high DI at low osmolality); "
            "RBC MORPHOLOGY: "
            "  STOMATOCYTES: mouth-shaped central pallor area; "
            "  XEROCYTES: dehydrated, dense, crenated; "
            "  NOTE: stomatocytes may disappear in stored or EDTA blood — examine FRESH FILM; "
            "LABORATORY: "
            "  MCHC >36 g/dL; MCV normal or slightly high; "
            "  Reticulocytes: 5-20%; "
            "  Indirect bilirubin elevated; "
            "  PSEUDOHYPERKALEMIA: serum K+ >5.5 mEq/L without haemolysis marker; "
            "    Confirm: measure K+ in heparinised plasma (spun immediately at 37°C); "
            "  LDH: mildly elevated"
        ),
        "disease_pathway": (
            "PIEZO1 GOF — MECHANOSENSITIVE Ca2+ CHANNEL OVERACTIVATION → RBC DEHYDRATION: "
            "NORMAL PIEZO1 FUNCTION: "
            "  Membrane tension during RBC deformation → brief PIEZO1 opening → Ca2+ entry → "
            "  KCNN4 activation → K+ efflux → osmotic compensation; "
            "  Self-limiting by rapid inactivation (fast kinetics); "
            "GOF MUTATION MECHANISM: "
            "  Inactivation impaired → channel remains open longer → excess Ca2+ influx; "
            "  Excess Ca2+ → KCNN4 (Gardos channel) constitutively active; "
            "  K+ efflux + Cl⁻ follows (KCl loss); "
            "  Water follows KCl → RBC DEHYDRATION; "
            "  Increased MCHC → dense, rigid cells; "
            "  Spleen: partially traps dense RBCs → moderate hemolysis; "
            "POST-SPLENECTOMY THROMBOSIS MECHANISM: "
            "  Dense, rigid RBCs after splenectomy: "
            "    Phosphatidylserine (PS) exposure on outer leaflet (Ca2+ activates scramblases); "
            "    PS = procoagulant surface → platelet + coagulation factor binding; "
            "    Post-splenectomy: these PS-exposing RBCs accumulate in circulation; "
            "    FATAL PORTAL, HEPATIC, MESENTERIC VEIN THROMBOSIS reported; "
            "    Splenectomy absolutely contraindicated in DHS"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC / HIGHLY CHARACTERISTIC — PIEZO1-DHS: "
            "  1. ELEVATED MCHC (>36 g/dL) in context of haemolytic anaemia; "
            "  2. PSEUDOHYPERKALEMIA: serum K+ elevated without true hyperkalemia "
            "     (confirmed by heparinised plasma K+ measured immediately at 37°C = NORMAL); "
            "  3. STOMATOCYTES on FRESH peripheral blood film (disappear on stored/EDTA blood); "
            "  4. OSMOTIC FRAGILITY: DECREASED or NORMAL (dehydrated cells resist lysis); "
            "  5. EKTACYTOMETRY: xerocyte pattern (increased DI at low osmolality); "
            "  6. NEGATIVE DAT; "
            "  7. AD FAMILY HISTORY of compensated haemolytic anaemia; "
            "ABSOLUTE CONTRAINDICATION: "
            "  SPLENECTOMY — ABSOLUTELY CONTRAINDICATED; "
            "  FATAL thrombotic complications reported: portal vein thrombosis, mesenteric ischaemia; "
            "  Mechanism: procoagulant PS-exposing dehydrated RBCs accumulate post-splenectomy; "
            "  If splenectomy already done: full anticoagulation + thrombosis monitoring; "
            "DIAGNOSTIC TRAP: "
            "  MCHC >36 suggests HS — but HS has INCREASED osmotic fragility; DHS has DECREASED; "
            "  Stomatocytes: missed if blood stored (>2h) or EDTA anticoagulated — examine fresh EDTA-free film; "
            "  Pseudohyperkalemia: may trigger unnecessary cardiac workup; confirm with immediate spun plasma"
        ),
        "treatment": (
            "MANAGEMENT — PIEZO1-DHS: "
            "CRITICAL: SPLENECTOMY ABSOLUTELY CONTRAINDICATED — fatal thrombosis risk; "
            "SUPPORTIVE: "
            "  Folic acid 1 mg/day; "
            "  Avoid triggers of acute haemolysis; "
            "HYDROXYUREA: some evidence for reducing haemolysis (reduces haematocrit effect on deformability); "
            "SENICAPOC (ICA-17043) — GARDOS CHANNEL INHIBITOR: "
            "  Mechanism: senicapoc blocks KCNN4 (Gardos channel) → reduces K+ efflux → "
            "    reduces RBC dehydration → MCHC normalises → less hemolysis; "
            "  STATUS: Phase 2/3 trials positive for SCD (reduces MCHC); investigational for DHS; "
            "  Rationale: PIEZO1 GOF → excess Ca2+ → KCNN4 activation → K+ loss; "
            "    blocking KCNN4 downstream of PIEZO1 → reverses dehydration even without PIEZO1 correction; "
            "ANTICOAGULATION: "
            "  NOT routinely indicated in uncomplicated DHS; "
            "  Consider if prior thrombosis or post-splenectomy; "
            "  Aspirin: consider post-splenectomy (if unavoidably done); "
            "MONITORING: "
            "  MCHC at each visit (therapeutic target); "
            "  Pseudohyperkalemia: always check heparinised plasma if electrolytes abnormal; "
            "  Thrombosis vigilance: D-dimer; Doppler if symptoms"
        ),
        "seed": 2852,
    },
    {
        "gene": "KCNN4",
        "protein": (
            "KCNN4 -- 19q13.31 AD -- 427aa -- Potassium-Intermediate-Conductance-Calcium-Activated-Channel-"
            "Subfamily-N-Member-4-47kDa-Gardos-Channel-IKCa1-KCa3.1-SK4-"
            "AD-Dehydrated-Stomatocytosis-Type-2-DHS2-AVOID-Splenectomy-Same-Thrombosis-Risk-Senicapoc-Direct-Target-"
            "OMIM-Gene-602754-Disease-DHS2-616689"
        ),
        "locus": "19q13.31",
        "protein_size": (
            "427 aa / 47 kDa (KCNN4 = KCa3.1 = Gardos channel = IKCa1 = SK4; "
            "intermediate-conductance calcium-activated potassium channel; "
            "6 transmembrane segments (S1-S6) + pore loop between S5-S6; "
            "forms homotetramers (4 × 427 aa); "
            "calmodulin constitutively bound to C-terminal domain: "
            "  Ca2+ binds calmodulin → conformational change → channel opens → K+ efflux; "
            "half-maximal activation: [Ca2+]i ~300 nM; "
            "channel activated by PIEZO1 → Ca2+ entry; "
            "KCNN4 GOF mutations → channel activates at lower Ca2+ threshold → "
            "spontaneous K+ efflux → RBC dehydration without PIEZO1 triggering; "
            "originally named Gardos channel after György Gárdos who discovered Ca2+-activated K+ transport in RBCs (1958))"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT (AD) — GAIN-OF-FUNCTION: "
            "  Heterozygous GOF missense → DHS2; "
            "  VERY RARE: <50 families described worldwide; "
            "  Key mutations: p.Val282Met; p.Arg352His; p.Ala322Gly; "
            "MECHANISM DISTINCTION FROM PIEZO1-DHS1: "
            "  PIEZO1-DHS1: upstream mechanosensory channel GOF → excess Ca2+ → activates KCNN4 indirectly; "
            "  KCNN4-DHS2: direct GOF of Gardos channel → opens at LOWER [Ca2+]i → "
            "    spontaneous K+ efflux even at resting Ca2+ levels; "
            "CLINICAL PHENOTYPE: "
            "  IDENTICAL to PIEZO1-DHS: "
            "    Compensated haemolytic anaemia (Hb 10-14 g/dL); "
            "    Stomatocytes on fresh film; "
            "    Elevated MCHC; "
            "    Pseudohyperkalemia; "
            "    Reduced osmotic fragility; "
            "    Ektacytometry xerocyte pattern; "
            "  CANNOT distinguish PIEZO1-DHS from KCNN4-DHS2 on clinical grounds alone; "
            "  Genetic panel (PIEZO1 + KCNN4) required"
        ),
        "disease_category": (
            "DEHYDRATED HEREDITARY STOMATOCYTOSIS TYPE 2 (DHS2) — OMIM 616689; "
            "PHENOTYPE IDENTICAL TO PIEZO1-DHS1: "
            "  Elevated MCHC; "
            "  Stomatocytes (fresh film); "
            "  Pseudohyperkalemia; "
            "  Reduced osmotic fragility; "
            "  Compensated haemolytic anaemia; "
            "  Negative DAT; "
            "SENICAPOC IS DIRECT TARGET: "
            "  PIEZO1-DHS: senicapoc blocks KCNN4 downstream; "
            "  KCNN4-DHS2: senicapoc directly blocks the mutant KCNN4 channel; "
            "    More mechanistically direct; potentially greater efficacy in KCNN4-DHS2 vs PIEZO1-DHS1; "
            "PERINATAL OEDEMA: "
            "  Hydrops fetalis reported in KCNN4-DHS2 (as in PIEZO1-DHS1); "
            "  Mechanism: severe intrauterine haemolysis → anaemia → cardiac failure → hydrops; "
            "  Oedema may resolve spontaneously after birth as haemolysis compensates"
        ),
        "disease_pathway": (
            "KCNN4-DHS2 — DIRECT GARDOS CHANNEL GOF → RBC DEHYDRATION: "
            "GARDOS EFFECT (NORMAL): "
            "  Named for Gárdos 1958 discovery: Ca2+-activated K+ transport in RBCs; "
            "  Ca2+ binds calmodulin (constitutively associated with KCNN4 C-terminus); "
            "  Calmodulin-Ca2+ → channel opening → K+ efflux → KCl co-transport → water loss; "
            "  Normally brief and regulated; "
            "KCNN4 GOF MECHANISM: "
            "  GOF mutation: calmodulin activation threshold LOWERED → channel opens spontaneously "
            "  at normal (low) intracellular Ca2+ concentrations; "
            "  Or: channel gating kinetics altered → stays open longer; "
            "  Result: chronic K+ efflux → KCl + H2O loss → DEHYDRATED RBCs; "
            "  MCHC rises (denser = higher haemoglobin concentration per cell); "
            "SENICAPOC PHARMACOLOGY: "
            "  Senicapoc = ICA-17043: selective KCNN4 (KCa3.1) blocker; "
            "  Originally developed for SCD (reduces MCHC in sickle cells); "
            "  Mechanism: blocks K+ efflux → preserves RBC volume → lower MCHC → less dehydration; "
            "  In KCNN4-DHS2: directly inhibits the mutant channel (MOST DIRECT MECHANISM); "
            "  In PIEZO1-DHS1: blocks downstream KCNN4 (Ca2+ source is PIEZO1; block is downstream)"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC / HIGHLY CHARACTERISTIC — KCNN4-DHS2: "
            "  1. SAME FULL CLINICAL + LAB PICTURE AS PIEZO1-DHS1: "
            "     elevated MCHC + stomatocytes (fresh film) + pseudohyperkalemia + reduced osmotic fragility; "
            "  2. CANNOT DISTINGUISH FROM PIEZO1-DHS1 CLINICALLY — requires genetic panel; "
            "  3. VERY RARE: <50 families — important to include KCNN4 in DHS gene panel; "
            "  4. SENICAPOC: most rationally targeted in KCNN4-DHS2 (direct channel block); "
            "  5. AD FAMILY HISTORY: one affected parent; "
            "  6. NEGATIVE DAT; "
            "ABSOLUTE CONTRAINDICATION — SAME AS PIEZO1: "
            "  SPLENECTOMY ABSOLUTELY CONTRAINDICATED; "
            "  Same thrombosis risk mechanism (PS-exposing dehydrated RBCs); "
            "  Same fatal consequences; "
            "DIAGNOSTIC WORKUP FOR DHS (PIEZO1 + KCNN4): "
            "  Step 1: MCHC elevated + stomatocytes + negative DAT + reduced osmotic fragility; "
            "  Step 2: Pseudohyperkalemia confirmation (heparinised plasma immediately); "
            "  Step 3: Ektacytometry (xerocyte pattern); "
            "  Step 4: DHS gene panel (PIEZO1 + KCNN4); "
            "  Step 5: Functional assay (Gardos effect assay: [Ca2+]-activated K+ transport in RBCs)"
        ),
        "treatment": (
            "MANAGEMENT — KCNN4-DHS2: "
            "CRITICAL: SPLENECTOMY ABSOLUTELY CONTRAINDICATED (same as PIEZO1-DHS1); "
            "SUPPORTIVE: "
            "  Folic acid 1 mg/day; "
            "SENICAPOC (ICA-17043) — DIRECT KCNN4 INHIBITOR: "
            "  Phase 2 study (SCD): >1 g/dL Hb rise; MCHC normalised; "
            "  DHS2: directly inhibits the GOF mutant channel; "
            "  Superior mechanistic rationale in DHS2 vs DHS1 (direct vs downstream); "
            "  Investigational (not yet FDA/EMA approved for DHS specifically); "
            "PSEUDOHYPERKALEMIA MANAGEMENT: "
            "  Educate patient + all treating clinicians; "
            "  Label chart: 'KCNN4 DHS2 — pseudohyperkalemia expected — confirm with immediate plasma K+'; "
            "  Avoid unnecessary potassium-restriction or dangerous treatments (insulin/glucose, dialysis); "
            "MONITORING: "
            "  MCHC + haemoglobin; "
            "  Reticulocyte count; "
            "  Thrombosis surveillance; "
            "GENETIC COUNSELLING: "
            "  AD: 50% risk to offspring; "
            "  DHS gene panel for family members with anaemia; "
            "PERINATAL: "
            "  Monitor fetus by serial ultrasound if parent has DHS2 (hydrops risk); "
            "  IVIG + early delivery if hydrops develops; "
            "  Exchange transfusion for severe neonatal haemolysis"
        ),
        "seed": 2853,
    },
]

DEFINITIONS = {
    "definitions": [
        {
            "term": "Hereditary Spherocytosis (HS) — MCHC, Osmotic Fragility, EMA Test",
            "definition": (
                "Hereditary Spherocytosis (HS): autosomal dominant (or occasionally AR) red cell membrane disorder "
                "characterised by microspherocytes on peripheral blood film, elevated MCHC (>36 g/dL), increased "
                "osmotic fragility, and negative DAT (Coombs test). Five genes account for most HS cases: "
                "ANK1 (60-65%), SPTA1+SPTB combined (~20%), SLC4A1 (~15%), EPB42 (~3%), EPB41 (<1% HS). "
                "EMA (eosin-5'-maleimide) binding test by flow cytometry: most sensitive HS screening; "
                "<80% of control = positive. Increased osmotic fragility: RBCs lyse at higher NaCl concentrations. "
                "Peripheral film: spherocytes lack central pallor; MCHC elevated due to membrane surface loss. "
                "All HS genes disrupt VERTICAL membrane interactions (cytoskeleton-to-bilayer). "
                "Splenectomy: curative for severe HS (Hb normalises); spherocytes persist but survive longer. "
                "DEFER splenectomy <5y (OPSI risk)."
            ),
        },
        {
            "term": "Hereditary Elliptocytosis (HE) — HPP, SPTA1, SPTB, EPB41, Heat Sensitivity",
            "definition": (
                "Hereditary Elliptocytosis (HE): >25% elongated elliptocytes on peripheral film. "
                "Genes: SPTA1 (HE1/HPP), SPTB (HE2), EPB41 (HE1), SLC4A1 minor. "
                "All disrupt HORIZONTAL spectrin lattice (spectrin-actin junctional complex). "
                "HE mild (heterozygous SPTA1, SPTB, EPB41): usually asymptomatic or mild compensated hemolysis; "
                "osmotic fragility NORMAL (differs from HS). "
                "HPP (Hereditary Pyropoikilocytosis): most severe HE; biallelic SPTA1 or SPTA1+LELY in trans; "
                "RBC heat sensitivity at 45-46°C (pathognomonic; normal >49°C); extreme poikilocytosis; "
                "neonatal severe hemolytic anaemia. "
                "LELY allele: common SPTA1 low-expression allele (Lys2069Ile); acts as modifier "
                "when in trans with SPTA1 pathogenic variant → HPP from what would otherwise be mild HE. "
                "Splenectomy: partial benefit in HPP (not fully curative like HS)."
            ),
        },
        {
            "term": "Dehydrated Hereditary Stomatocytosis (DHS) — PIEZO1, KCNN4, Xerocytosis, Pseudohyperkalemia",
            "definition": (
                "Dehydrated Hereditary Stomatocytosis (DHS) / Hereditary Xerocytosis: AD GOF mutations in "
                "PIEZO1 (mechanosensitive Ca2+ channel, DHS1) or KCNN4 (Gardos K+ channel, DHS2). "
                "Mechanism: excess Ca2+ entry (PIEZO1 GOF) or direct channel activation (KCNN4 GOF) → "
                "K+ efflux via Gardos channel → KCl + H2O loss → RBC dehydration. "
                "Clinical: compensated haemolytic anaemia; elevated MCHC >36 g/dL; stomatocytes on FRESH film "
                "(disappear in stored/EDTA blood); REDUCED osmotic fragility (dense cells resist lysis — OPPOSITE of HS). "
                "Pseudohyperkalemia: K+ leaks from dehydrated RBCs at room temperature → falsely high serum K+; "
                "confirm by heparinised plasma measured immediately at 37°C. "
                "CRITICAL: SPLENECTOMY ABSOLUTELY CONTRAINDICATED — fatal portal/mesenteric thrombosis; "
                "phosphatidylserine-exposing dense RBCs activate coagulation post-splenectomy. "
                "Senicapoc (ICA-17043): Gardos channel blocker; investigational; more direct in KCNN4-DHS2."
            ),
        },
        {
            "term": "Southeast Asian Ovalocytosis (SAO) — SLC4A1 27-bp deletion, Malaria Protection",
            "definition": (
                "Southeast Asian Ovalocytosis (SAO): AD SLC4A1 27-bp deletion (p.Ala400_Ala408del) + p.Lys56Glu. "
                "Produces RIGID OVALOCYTES with transverse ridge — pathognomonic morphology. "
                "Epidemiology: high prevalence in SE Asian indigenous populations (Philippines, Malaysia, PNG, Indonesia). "
                "Mechanism: 27-bp deletion locks band 3 dimer in rigid conformation; transport null (no Cl⁻/HCO₃⁻ exchange). "
                "Clinical: MILD or COMPENSATED anaemia; distinct from HS. "
                "EMA binding: INCREASED (unique to SAO — opposite of HS). "
                "Osmotic fragility: DECREASED (rigid cells resist hypotonic lysis — opposite of HS). "
                "Malaria protection: rigid SAO ovalocytes block P. falciparum merozoite invasion → protection against cerebral malaria. "
                "LETHAL HOMOZYGOUS: no live-born SAO homozygotes — embryonic lethal. "
                "MANAGEMENT: usually none; AVOID splenectomy in SAO."
            ),
        },
        {
            "term": "EMA (Eosin-5'-Maleimide) Binding Test — Flow Cytometry HS Screening",
            "definition": (
                "EMA (eosin-5'-maleimide) binding test: flow cytometric assay measuring EMA binding to band 3 (AE1) "
                "and Rh complex on erythrocyte surface. "
                "Result expressed as % of matched healthy control MFI (mean fluorescence intensity). "
                "REDUCED (<80%) in ALL 5 HS genes (ANK1, SPTB, SPTA1, SLC4A1-HS3, EPB42): "
                "  Loss of membrane protein interaction reduces EMA-accessible band 3 epitopes; "
                "  Most sensitive + specific HS screening test (>90% sensitivity, >95% specificity); "
                "  Superior to osmotic fragility (10-20% false-negative rate for mild HS). "
                "INCREASED in SAO: band 3 dimer locking increases EMA-accessible epitopes (opposite direction — diagnostic). "
                "NORMAL in: DHS (PIEZO1/KCNN4), OHS, HE (SPTB, SPTA1 mild) — EMA does NOT screen all RBC membrane disorders. "
                "Limitation: requires 12h fresh blood; matched healthy control must be run concurrently."
            ),
        },
        {
            "term": "Ektacytometry — Osmotic Gradient, Laser Diffraction, DI Curve",
            "definition": (
                "Ektacytometry: laser diffraction technique measuring RBC deformability index (DI) across "
                "osmotic gradient (from 60 to 400 mOsm/kg). Gold standard functional test for RBC membrane disorders. "
                "Parameters: "
                "  DI_min: minimum deformability at low osmolality (indicator of surface-to-volume ratio); "
                "  O_min: osmolality at minimum DI (osmotic resistance); "
                "  DI_max: peak deformability at optimal osmolality; "
                "CHARACTERISTIC PATTERNS: "
                "  HS: left-shifted curve (lower O_min); reduced DI_min; "
                "  HE/HPP: reduced DI_max; elongated curve shape; "
                "  DHS/xerocytosis: RIGHT-shifted curve + high DI at low osmolality (opposite HS); "
                "  SAO: reduced DI_max with distinct rigid ovalocyte plateau; "
                "  Overhydrated stomatocytosis (OHS): Left-shifted with high MCV + low MCHC opposite of DHS. "
                "Differentiates all forms definitively when combined with blood film + EMA + MCHC."
            ),
        },
        {
            "term": "Aplastic Crisis — Parvovirus B19, PVB19, Red Cell Aplasia in HS",
            "definition": (
                "Aplastic Crisis: acute, temporary arrest of erythropoiesis due to Parvovirus B19 (PVB19) infection "
                "in patients with chronic haemolytic anaemia (HS, HE, SCD, thalassemia, etc.). "
                "Mechanism: PVB19 infects and lyses erythroid progenitors (CFU-E) via globoside receptor (P antigen); "
                "erythropoiesis ceases for 7-10 days → RBC lifespan already reduced (HS: 20-30 days) → "
                "rapid Hb fall of 5-7 g/dL in days. "
                "Presentation: acute pallor + fatigue + worsening jaundice; "
                "  reticulocytes: DISAPPEAR (paradoxical for hemolytic anaemia); "
                "  Hb may drop to critical levels requiring urgent transfusion. "
                "Diagnosis: PVB19 IgM + PCR in acute phase; IgG = prior infection (immune). "
                "Treatment: PRBC transfusion (often single unit sufficient); "
                "  IVIg if immunocompromised (B-cell immunodeficiency prolongs aplasia). "
                "Family clustering: household contacts with PVB19 all at risk for sequential aplastic crises."
            ),
        },
        {
            "term": "Splenectomy in Red Cell Membrane Disorders — Indications, CIs, OPSI, DHS Fatal Thrombosis",
            "definition": (
                "Splenectomy indications and contraindications in red cell membrane disorders: "
                "INDICATIONS: "
                "  HS (all genes): severe (Hb <8 g/dL) or moderate with symptomatic gallstones; "
                "  HPP (SPTA1 biallelic): partial benefit; "
                "  HE (SPTB, EPB41): severe or symptomatic; "
                "ABSOLUTE CONTRAINDICATIONS: "
                "  PIEZO1-DHS1 and KCNN4-DHS2: SPLENECTOMY ABSOLUTELY CONTRAINDICATED; "
                "    Fatal portal, hepatic, mesenteric vein thrombosis reported; "
                "    PS-exposing dehydrated RBCs → procoagulant → post-splenectomy thrombosis; "
                "  SAO: generally contraindicated (rigid cells; no benefit); "
                "OPSI (Overwhelming Post-Splenectomy Infection): "
                "  Streptococcus pneumoniae (most common) + H. influenzae type b + N. meningitidis; "
                "  Lifetime risk ~3-5% without prophylaxis; can be rapidly fatal; "
                "  Prevention: vaccines (pneumococcal, Hib, meningococcal ACWY+B) + penicillin V lifelong; "
                "  DEFER splenectomy <5 years of age (highest OPSI risk); "
                "TIMING: concurrent cholecystectomy if gallstones symptomatic; laparoscopic preferred."
            ),
        },
        {
            "term": "Senicapoc (ICA-17043) — Gardos Channel Blocker, DHS, SCD, KCNN4",
            "definition": (
                "Senicapoc (ICA-17043): orally active, highly selective blocker of KCNN4 (KCa3.1 / Gardos channel). "
                "Pharmacology: binds inner vestibule of KCNN4 K+ pore; IC50 ~11 nM; "
                "  blocks Ca2+-activated K+ efflux → prevents KCl-driven RBC dehydration; "
                "  normalises MCHC (reduced from elevated) → less dense cells → reduced sickling (SCD) or xerocytosis (DHS). "
                "Clinical trials: "
                "  SCD Phase 3: reduced MCHC by ~0.8 g/dL; Hb rise ~0.9 g/dL; no effect on painful crises endpoint; "
                "    → FDA not approved for SCD; "
                "  DHS Phase 2 (PIEZO1 + KCNN4 combined): MCHC normalised; haemolysis reduced; "
                "    → potentially more efficacious in DHS than SCD; ongoing registrational studies. "
                "Mechanism in DHS: "
                "  PIEZO1-DHS1: senicapoc blocks KCNN4 DOWNSTREAM of the excess Ca2+ entry; "
                "  KCNN4-DHS2: senicapoc blocks the GOF channel DIRECTLY (most rational use). "
                "Adverse effects: well-tolerated; transient elevation in serum creatinine; "
                "No splenectomy needed if senicapoc effective."
            ),
        },
        {
            "term": "Protein 4.2 (EPB42), Protein 4.1R (EPB41) — SDS-PAGE Membrane Protein Gel Diagnosis",
            "definition": (
                "SDS-PAGE membrane protein gel (Fairbanks system) separates erythrocyte membrane proteins "
                "by molecular weight. Key bands: "
                "  Band 1/2: alpha/beta-spectrin (~240/220 kDa); "
                "  Band 2.1: Ankyrin-1 (~206 kDa); "
                "  Band 3: AE1 (~102 kDa — most abundant); "
                "  Band 4.1: Protein 4.1R (~80 kDa); "
                "  Band 4.2: Protein 4.2 (~77 kDa); "
                "  Band 5: Actin (~42 kDa); "
                "  Band 6: G3PD (~37 kDa); "
                "  PAS bands: glycophorins (carbohydrate-staining). "
                "Absent band 4.2 (EPB42 homozygous null): diagnoses EPB42-HS5 definitively. "
                "Absent/reduced band 4.1 (EPB41 homozygous): diagnoses EPB41-HE1 homozygous. "
                "Reduced spectrin (bands 1+2): in ANK1/SPTB-HS (secondary spectrin deficiency). "
                "Self-association defect (SPTB/SPTA1 HE): reduced tetramer:dimer ratio on native PAGE "
                "(requires non-denaturing conditions — separate assay from SDS-PAGE)."
            ),
        },
        {
            "term": "Gardos Effect — Historical Context, Ca2+-Activated K+ Transport, Gárdos 1958",
            "definition": (
                "Gardos Effect: Ca2+-activated K+ transport in erythrocytes, first described by György Gárdos in 1958. "
                "Original observation: ATP-depleted RBCs (which accumulate Ca2+ as Ca2+-ATPase pump fails) "
                "undergo rapid K+ loss → cell shrinkage → density increase. "
                "Molecular basis (identified 1990s): KCNN4 (KCa3.1) channel — Ca2+ binds constitutively-associated "
                "calmodulin → channel opens → K+ efflux → KCl cotransport → H2O loss. "
                "Physiological role: RBC volume regulation during passage through microcirculation; "
                "  brief, self-limiting K+ loss compensates for osmotic stress. "
                "Pathophysiology: "
                "  SCD: sickling-induced Ca2+ entry activates Gardos channel → dense sickle cells; "
                "  PIEZO1-DHS1: PIEZO1 GOF → excess Ca2+ → Gardos constitutively active; "
                "  KCNN4-DHS2: direct GOF → Gardos active at normal [Ca2+]. "
                "Senicapoc: blocks KCNN4 → reverses Gardos effect → reduces pathological dehydration."
            ),
        },
        {
            "term": "LELY Allele — Low Expression LYon, SPTA1 Modifier, HPP Risk",
            "definition": (
                "LELY allele (Low Expression LYon): common SPTA1 polymorphism causing partial alpha-spectrin deficiency. "
                "Molecular basis: two linked changes — "
                "  1. p.Lys2069Ile (exon 46): reduces SPTA1 mRNA stability; "
                "  2. Intron 45 splice site variant: causes alternative splicing with partial exon skipping; "
                "  Combined: LELY allele produces ~50% less alpha-spectrin than normal allele. "
                "IMPORTANCE: normal alpha-spectrin is produced in 3-4× excess over beta-spectrin. "
                "  SPTA1 mutation in trans with LELY: alpha excess eliminated → ALPHA SPECTRIN DEFICIENCY; "
                "  Biallelic SPTA1 (het + LELY in trans) → HPP phenotype (severe HE); "
                "  Biallelic normal SPTA1 → MILD HE only (enough alpha-excess remains). "
                "Frequency: ~25% allele frequency in African populations; 5% in European. "
                "Clinical significance: "
                "  African family with mild HE parent (het SPTA1) + HPP child (biallelic) → LELY likely in trans; "
                "  LELY testing: not on standard WES panels; needs targeted SPTA1 exon 46 + intron 45 assay."
            ),
        },
    ],
    "standards": [
        "ICSH 2015 — Guidelines for laboratory diagnosis of hereditary red cell membrane disorders (EMA, osmotic fragility, ektacytometry)",
        "Bolton-Maggs PHB et al. 2011 — Br J Haematol: Guidelines for diagnosis and management of hereditary spherocytosis",
        "Gallagher PG 2013 — Blood: Hereditary elliptocytosis: spectrin and protein 4.1R",
        "Zarychanski R et al. 2012 — Blood: Mutations in the mechanotransduction protein PIEZO1 are associated with hereditary xerocytosis",
        "Andolfo I et al. 2013 — Blood: Multiple clinical forms of dehydrated hereditary stomatocytosis arise from mutations in PIEZO1",
        "Gárdos G 1958 — Biochim Biophys Acta: The function of calcium in the potassium permeability of human erythrocytes (Gardos effect)",
        "Guizouarn H & Borgese F 2020 — Front Physiol: Dehydrated hereditary stomatocytosis (DHS): PIEZO1 and KCNN4 mutations",
        "Fermo E et al. 2022 — Haematologica: KCNN4 mutations cause dehydrated hereditary stomatocytosis type 2",
        "Iolascon A et al. 2020 — Am J Hematol: PIEZO1 — Hereditary xerocytosis and beyond",
        "Stewart GW et al. 2019 — Br J Haematol: Hereditary stomatocytosis: spectrum of clinical and laboratory features",
        "Piel FB et al. 2013 — Nat Genet: Global distribution of the sickle cell gene and geographical confirmation of the malaria hypothesis (SAO/malaria context)",
        "Inaba M et al. 1992 — Blood: Complete deficiency of protein 4.2 in erythrocytes from a patient with hemolytic anemia (EPB42-HS5 Japan)",
    ],
}


def _make_patients(gene_data):
    rng = random.Random(gene_data["seed"])
    patients = []
    gene = gene_data["gene"]
    for i in range(40):
        pid = f"{gene}-M{gene_data['seed']}-P{i+1:03d}"
        sex = rng.choice(["M", "F"])
        age_dx = rng.randint(0, 45)

        # Gene-specific parameter distributions
        if gene == "ANK1":
            hb = round(rng.uniform(7.5, 13.5), 1)
            retic = round(rng.uniform(5, 22), 1)
            ldh = rng.randint(280, 750)
            bili = rng.randint(25, 120)
            mchc = round(rng.uniform(35.5, 40.0), 1)
            osmotic_frag = "increased"
            ema_pct = round(rng.uniform(62, 79), 1)
            morphology = "microspherocytes"
            splenomegaly = rng.random() < 0.82
            splenectomy = rng.random() < 0.35
            dehydrated = False
            neuro = False
            gallstones = rng.random() < 0.48

        elif gene == "SPTA1":
            hb = round(rng.uniform(5.5, 12.0), 1)
            retic = round(rng.uniform(8, 30), 1)
            ldh = rng.randint(350, 1100)
            bili = rng.randint(35, 180)
            mchc = round(rng.uniform(35.0, 40.0), 1)
            osmotic_frag = rng.choice(["increased", "normal"])
            ema_pct = round(rng.uniform(60, 80), 1)
            morphology = rng.choice(["elliptocytes+microspherocytes", "extreme_poikilocytosis_HPP"])
            splenomegaly = rng.random() < 0.78
            splenectomy = rng.random() < 0.42
            dehydrated = False
            neuro = False
            gallstones = rng.random() < 0.45

        elif gene == "SPTB":
            hb = round(rng.uniform(9.0, 14.0), 1)
            retic = round(rng.uniform(3, 18), 1)
            ldh = rng.randint(200, 650)
            bili = rng.randint(18, 90)
            mchc = round(rng.uniform(33.0, 37.5), 1)
            osmotic_frag = rng.choice(["normal", "normal", "increased"])
            ema_pct = round(rng.uniform(65, 85), 1)
            morphology = "elliptocytes"
            splenomegaly = rng.random() < 0.52
            splenectomy = rng.random() < 0.20
            dehydrated = False
            neuro = False
            gallstones = rng.random() < 0.28

        elif gene == "SLC4A1":
            phenotype = rng.choice(["HS3", "SAO", "dRTA_HA"])
            hb = round(rng.uniform(8.0, 13.5) if phenotype != "dRTA_HA" else rng.uniform(7.0, 11.5), 1)
            retic = round(rng.uniform(4, 20), 1)
            ldh = rng.randint(220, 680)
            bili = rng.randint(20, 100)
            mchc = round(rng.uniform(33.5, 38.5) if phenotype == "HS3" else rng.uniform(30, 35), 1)
            osmotic_frag = "decreased" if phenotype == "SAO" else "increased" if phenotype == "HS3" else "normal"
            ema_pct = round(rng.uniform(105, 125) if phenotype == "SAO" else rng.uniform(60, 80), 1)
            morphology = "rigid_ovalocytes_SAO" if phenotype == "SAO" else "microspherocytes" if phenotype == "HS3" else "ovalocytes"
            splenomegaly = rng.random() < 0.60
            splenectomy = False if phenotype == "SAO" else rng.random() < 0.25
            dehydrated = False
            neuro = False
            gallstones = rng.random() < 0.38

        elif gene == "EPB42":
            hb = round(rng.uniform(8.5, 12.5), 1)
            retic = round(rng.uniform(6, 20), 1)
            ldh = rng.randint(260, 700)
            bili = rng.randint(22, 110)
            mchc = round(rng.uniform(35.0, 39.5), 1)
            osmotic_frag = "increased"
            ema_pct = round(rng.uniform(62, 80), 1)
            morphology = "microspherocytes"
            splenomegaly = rng.random() < 0.75
            splenectomy = rng.random() < 0.38
            dehydrated = False
            neuro = False
            gallstones = rng.random() < 0.44

        elif gene == "EPB41":
            hb = round(rng.uniform(10.0, 14.5), 1)
            retic = round(rng.uniform(2, 14), 1)
            ldh = rng.randint(180, 520)
            bili = rng.randint(12, 75)
            mchc = round(rng.uniform(32.0, 36.0), 1)
            osmotic_frag = "normal"
            ema_pct = round(rng.uniform(70, 88), 1)
            morphology = "elliptocytes"
            splenomegaly = rng.random() < 0.38
            splenectomy = rng.random() < 0.12
            dehydrated = False
            neuro = False
            gallstones = rng.random() < 0.18

        elif gene == "PIEZO1":
            hb = round(rng.uniform(9.5, 14.0), 1)
            retic = round(rng.uniform(5, 20), 1)
            ldh = rng.randint(220, 620)
            bili = rng.randint(20, 95)
            mchc = round(rng.uniform(36.0, 41.5), 1)
            osmotic_frag = "decreased"
            ema_pct = round(rng.uniform(88, 110), 1)
            morphology = "stomatocytes_xerocytes"
            splenomegaly = rng.random() < 0.45
            splenectomy = False  # ABSOLUTELY CONTRAINDICATED
            dehydrated = True
            neuro = False
            gallstones = rng.random() < 0.25

        else:  # KCNN4
            hb = round(rng.uniform(10.0, 14.5), 1)
            retic = round(rng.uniform(4, 18), 1)
            ldh = rng.randint(200, 580)
            bili = rng.randint(18, 88)
            mchc = round(rng.uniform(36.5, 41.0), 1)
            osmotic_frag = "decreased"
            ema_pct = round(rng.uniform(90, 112), 1)
            morphology = "stomatocytes_xerocytes"
            splenomegaly = rng.random() < 0.40
            splenectomy = False  # ABSOLUTELY CONTRAINDICATED
            dehydrated = True
            neuro = False
            gallstones = rng.random() < 0.22

        patients.append({
            "patient_id": pid,
            "sex": sex,
            "age_at_diagnosis_years": age_dx,
            "hemoglobin_g_dl": hb,
            "reticulocyte_pct": retic,
            "ldh_u_l": ldh,
            "bilirubin_umol_l": bili,
            "mchc_g_dl": mchc,
            "osmotic_fragility": osmotic_frag,
            "ema_binding_pct_control": ema_pct,
            "morphology": morphology,
            "splenomegaly": splenomegaly,
            "splenectomy_done": splenectomy,
            "rbc_dehydrated": dehydrated,
            "neuro_involvement": neuro,
            "gallstones": gallstones,
            "transfusion_dependent": hb < 8.5,
        })
    return patients


def generate_overview():
    all_genes = []
    total_patients = 0
    for gene_data in ATLAS_GENES:
        patients = _make_patients(gene_data)
        total_patients += len(patients)
        all_genes.append({
            "gene": gene_data["gene"],
            "locus": gene_data["locus"],
            "n_patients": len(patients),
            "median_hb_g_dl": sorted(p["hemoglobin_g_dl"] for p in patients)[len(patients)//2],
            "mean_retic_pct": round(sum(p["reticulocyte_pct"] for p in patients) / len(patients), 1),
            "mean_ldh_u_l": round(sum(p["ldh_u_l"] for p in patients) / len(patients)),
            "mean_mchc_g_dl": round(sum(p["mchc_g_dl"] for p in patients) / len(patients), 1),
            "pct_splenomegaly": round(100 * sum(p["splenomegaly"] for p in patients) / len(patients), 1),
            "pct_dehydrated": round(100 * sum(p["rbc_dehydrated"] for p in patients) / len(patients), 1),
            "pct_transfusion_dependent": round(100 * sum(p["transfusion_dependent"] for p in patients) / len(patients), 1),
            "pct_gallstones": round(100 * sum(p["gallstones"] for p in patients) / len(patients), 1),
            "mean_ema_pct": round(sum(p["ema_binding_pct_control"] for p in patients) / len(patients), 1),
            "seed": gene_data["seed"],
        })

    return {
        "atlas": "Hereditary Red Cell Membrane Disorder Atlas",
        "subtitle": (
            "Complete 8-Gene Hereditary Red Cell Membrane Disorder Reference — "
            "ANK1·SPTA1·SPTB·SLC4A1·EPB42·EPB41·PIEZO1·KCNN4"
        ),
        "genes": [g["gene"] for g in ATLAS_GENES],
        "total_patients": total_patients,
        "gene_summaries": all_genes,
        "seeds": "2846-2853",
        "disorder_categories": [
            {
                "category": "Hereditary Spherocytosis (HS) — Vertical Interaction Defects",
                "genes": ["ANK1", "SPTA1", "SLC4A1", "EPB42"],
                "note": (
                    "ANK1 (60-65% HS): most common; microspherocytes; EMA reduced; osmotic fragility increased; "
                    "splenectomy curative (defer <5y); aplastic crisis PVB19 risk; folic acid mandatory. "
                    "SPTA1 (biallelic HPP): heat sensitivity 45-46°C pathognomonic; extreme poikilocytosis; "
                    "LELY allele modifier (African ancestry); splenectomy partial benefit. "
                    "SLC4A1-HS3: band 3 N-terminal mutations; standard HS; concurrent SAO variant (SE Asia) — "
                    "rigid ovalocytes; EMA INCREASED (opposite HS); malaria protection; lethal homozygous. "
                    "EPB42 (HS5): AR; Japanese p.Ala142Thr + Mediterranean founders; "
                    "absent band 4.2 on SDS-PAGE pathognomonic; splenectomy effective"
                ),
            },
            {
                "category": "Hereditary Elliptocytosis (HE) — Horizontal Spectrin Lattice Defects",
                "genes": ["SPTA1", "SPTB", "EPB41"],
                "note": (
                    "SPTB (HE2): most common HE gene; AD; mostly mild elliptocytosis; "
                    "elliptocytes >25% on film; osmotic fragility NORMAL; splenectomy for severe/HS2 phenotype. "
                    "EPB41 (HE1): AD; junctional complex defect; SE Asian + North African founders; "
                    "GPC reduced (4.1R-GPC link lost); malaria protection (EBA-140 invasion blocked); "
                    "absent band 4.1 on SDS-PAGE (homozygous). "
                    "HE vs HS key distinction: HE = NORMAL osmotic fragility; HS = INCREASED osmotic fragility; "
                    "ektacytometry distinguishes definitively"
                ),
            },
            {
                "category": "Dehydrated Hereditary Stomatocytosis (DHS) — Cation Channel GOF — SPLENECTOMY CONTRAINDICATED",
                "genes": ["PIEZO1", "KCNN4"],
                "note": (
                    "PIEZO1-DHS1 (GOF AD): mechanosensitive Ca2+ channel excess Ca2+ → Gardos channel (KCNN4) → "
                    "K+ efflux → KCl loss → RBC dehydration → elevated MCHC + xerocytes/stomatocytes. "
                    "KCNN4-DHS2 (GOF AD): direct Gardos channel GOF → spontaneous K+ efflux at normal Ca2+. "
                    "SAME PHENOTYPE: MCHC >36; stomatocytes (FRESH film only); reduced osmotic fragility; "
                    "pseudohyperkalemia (confirm with immediate heparinised plasma K+). "
                    "ABSOLUTE CI: SPLENECTOMY in ALL DHS — fatal portal/mesenteric thrombosis. "
                    "Senicapoc: Gardos channel blocker; most direct in KCNN4-DHS2; investigational"
                ),
            },
        ],
        "critical_distinctions": [
            "DHS vs HS (both: MCHC >36 g/dL): DHS = DECREASED osmotic fragility (rigid/dehydrated RBCs resist lysis); HS = INCREASED fragility; ektacytometry is definitive — right-shifted curve = DHS; left-shifted = HS",
            "SPLENECTOMY ABSOLUTE CI in DHS (PIEZO1 + KCNN4): fatal thrombosis post-splenectomy; phosphatidylserine-exposing dense RBCs → procoagulant → portal/mesenteric/hepatic vein thrombosis; NEVER splenectomise DHS",
            "PSEUDOHYPERKALEMIA in DHS: serum K+ falsely elevated (K+ leaks from dehydrated RBCs at room temperature); always confirm with heparinised plasma spun immediately at 37°C before treating 'hyperkalemia'",
            "STOMATOCYTES DISAPPEAR on stored/EDTA blood: examine FRESH film in suspected DHS; storage artifact mimics recovery; always specify fresh film request to lab",
            "SAO (SLC4A1 27-bp deletion): EMA INCREASED (opposite all other HS genes); osmotic fragility DECREASED (opposite HS); rigid ovalocytes with transverse ridge; malaria protection; LETHAL HOMOZYGOUS — screen partner if SAO found",
            "HPP (SPTA1 biallelic + LELY): heat sensitivity 45-46°C pathognomonic; LELY allele not on standard WES — test separately; African family: HE parent + HPP child = LELY in trans until proven otherwise",
            "EMA binding test: REDUCED in all 5 HS genes; INCREASED in SAO; NORMAL in DHS (PIEZO1/KCNN4), HE (most); EMA does NOT exclude DHS",
            "dRTA-HA (SLC4A1 AR): metabolic acidosis + urine pH >5.5 + nephrocalcinosis + haemolytic anaemia in same patient = SLC4A1 dRTA until proven otherwise; bicarb supplementation corrects acidosis + prevents nephrocalcinosis progression",
            "Band 4.2 (EPB42) absent on SDS-PAGE = EPB42-HS5 definitive diagnosis; Japanese p.Ala142Thr founder; Southern Mediterranean p.Tyr142His; AR — parents asymptomatic carriers",
            "GPC (glycophorin C) reduced in EPB41-HE1: 4.1R required for GPC membrane anchoring; EPB41 loss → GPC reduced; do NOT confuse with PNH (GPI-anchored; FLAER/Ham's test distinguishes)",
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
    return {"genes": result, "total": len(ATLAS_GENES), "seeds": "2846-2853"}


def generate_definitions():
    return {
        "atlas": "Hereditary Red Cell Membrane Disorder Atlas",
        "definitions": DEFINITIONS["definitions"],
        "standards": DEFINITIONS["standards"],
        "gene_count": len(ATLAS_GENES),
        "seeds": "2846-2853",
    }
