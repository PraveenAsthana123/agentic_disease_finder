#!/usr/bin/env python3
"""Hereditary-Haemolytic-Anaemia-Atlas — Complete 8-Gene Hereditary Haemolytic Anaemia Atlas
ANK1    (Ankyrin-1; 1881 aa; ~206 kDa; 8p11.21; AD/AR;
         OMIM gene 612641; HS1 OMIM 182900;
         most common HS gene 30-40%; haploinsufficiency;
         osmotic fragility + EMA binding flow cytometry;
         splenectomy effective — vaccination MANDATORY before;
         seed SEED_BASE+0) ·
SPTB    (Beta-spectrin; 2137 aa; ~246 kDa; 14q23.3; AD;
         OMIM gene 182870; HS2 OMIM 616649;
         2nd most common HS gene; partial deficiency;
         also hereditary elliptocytosis + pyropoikilocytosis;
         seed SEED_BASE+1) ·
SLC4A1  (Band 3 / AE1; 911 aa; ~102 kDa; 17q21.31; AD/AR;
         OMIM gene 109270; HS4 OMIM 612653;
         AD: HS4; AD: SAO (Ala400-Ala408 deletion) — malaria protection;
         AR: severe HA + DISTAL RTA — alkali mandatory;
         seed SEED_BASE+2) ·
SPTA1   (Alpha-spectrin; 2429 aa; ~280 kDa; 1q23.1; AD/AR;
         OMIM gene 182860; HE1 OMIM 182900;
         AD: hereditary elliptocytosis (Arg28His common Africa);
         AR: hereditary pyropoikilocytosis — most severe neonatal HA;
         aLELY allele in trans with pathogenic SPTA1 = HPP;
         seed SEED_BASE+3) ·
PKLR    (Pyruvate kinase L/R; 574 aa; ~63 kDa; 1q22; AR;
         OMIM gene 609712; PK deficiency OMIM 266200;
         most common non-spherocytic hereditary HA;
         Mitapivat (Pyrukynd FDA 2022) — first approved treatment;
         post-splenectomy paradoxical reticulocytosis;
         seed SEED_BASE+4) ·
G6PD    (Glucose-6-phosphate dehydrogenase; 515 aa; ~59 kDa; Xq28; XL;
         OMIM gene 305900; G6PD deficiency OMIM 300908;
         most common human enzymopathy — 400M worldwide;
         rasburicase ABSOLUTELY CONTRAINDICATED;
         fava beans, primaquine, dapsone triggers;
         seed SEED_BASE+5) ·
PIEZO1  (Mechanosensitive channel; 2521 aa; ~286 kDa; 16q24.3; AD GOF;
         OMIM gene 611184; DHS OMIM 194380;
         dehydrated hereditary stomatocytosis / xerocytosis;
         high MCHC; pseudohyperkalaemia on stored blood;
         SPLENECTOMY CONTRAINDICATED — DVT/PE risk;
         seed SEED_BASE+6) ·
KCNN4   (Gardos channel; 427 aa; ~48 kDa; 19q13.31; AD GOF;
         OMIM gene 602754; DHS/Gardos channelopathy OMIM 194380;
         dehydrated HS via K+ loss; senicapoc clinical trials;
         SPLENECTOMY CONTRAINDICATED — DVT/PE risk;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1526–1533)
"""

import random

SEED_BASE = 1526

HAEMOLYTIC_ANAEMIA_GENES = [
    # ── ANK1 — Most Common Hereditary Spherocytosis Gene ──
    {
        "gene": "ANK1",
        "protein": "Ankyrin-1 — Most Common HS Gene 30-40%, Haploinsufficiency, Splenectomy Effective",
        "alias": (
            "ANK1; OMIM gene 612641; HS1 OMIM 182900; 8p11.21; 1881 aa; ~206 kDa; "
            "ANK1 encodes ankyrin-1 (erythrocyte ankyrin, ankyrin-R), the linker protein "
            "that tethers the spectrin-actin cytoskeleton to the lipid bilayer of the red "
            "cell membrane. Ankyrin-1 bridges beta-spectrin (SPTB) to the cytoplasmic domain "
            "of Band 3 (SLC4A1) and Rh-associated glycoprotein complex, creating the vertical "
            "interaction that maintains red cell structural integrity, biconcave shape, and "
            "deformability. ANK1 is the most common hereditary spherocytosis gene, accounting "
            "for 30-40% of all HS cases. Haploinsufficiency (predominantly truncating variants "
            "— frameshift, nonsense, splice-site) reduces ankyrin density in the membrane "
            "skeleton, weakening vertical linkages between lipid bilayer and underlying spectrin "
            "meshwork. Membrane lipid vesiculates and is shed from the red cell surface, "
            "reducing surface-to-volume ratio and producing the spheroidal shape characteristic "
            "of HS. Spherocytes have reduced deformability and are selectively trapped and "
            "destroyed in the narrow sinusoids of the splenic red pulp — resulting in "
            "extravascular haemolysis. Clinical presentation: mild-to-moderate haemolytic "
            "anaemia (Hb 8-12 g/dL), elevated MCHC (>36 g/dL), elevated reticulocytes, "
            "elevated unconjugated bilirubin, hyperbilirubinuria (urine discolouration). "
            "Long-term: pigment (bilirubin) gallstones requiring cholecystectomy in 50-70% "
            "by adulthood. Aplastic crisis precipitated by parvovirus B19 (erythroid "
            "precursor destruction) — ACUTE HAEMATOLOGICAL EMERGENCY requiring transfusion. "
            "Splenectomy reduces haemolysis by 90-95% but does NOT correct the underlying "
            "membrane defect — red cells remain spherocytes. Pre-splenectomy vaccination "
            "MANDATORY (pneumococcal, meningococcal, Hib). Penicillin prophylaxis "
            "post-splenectomy. Splenectomy deferred until age ≥6 years to preserve "
            "immunological development."
        ),
        "aa": "1881 aa",
        "kDa": "~206 kDa",
        "locus": "8p11.21",
        "omim_gene": 612641,
        "omim_disease": 182900,
        "inheritance": "AD — haploinsufficiency; AR biallelic → more severe HS with SPTA1 compound het",
        "gene_class": (
            "Ankyrin-1 is a 1881-amino acid modular adaptor protein organised into three "
            "functional domains: (1) N-terminal membrane-binding domain (MBD, residues 1-807) "
            "— 24 ANK repeat modules arranged in 6 tandem pairs that bind Band 3 (SLC4A1) "
            "cytoplasmic domain, Rh-associated glycoprotein (RhAG), RhCE, and other membrane "
            "proteins; (2) central spectrin-binding domain (SBD, residues 808-1423) — binds "
            "the beta-spectrin (SPTB) C-terminal region via a specific ankyrin-binding site, "
            "tethering the 2D spectrin-actin meshwork to the bilayer; (3) C-terminal "
            "regulatory domain (CTD, residues 1424-1881) — auto-inhibitory domain that "
            "modulates MBD affinity for Band 3; isoform-specific splicing of exon 39 (CTD) "
            "generates the erythrocyte-specific Ank1.8 isoform and the brain/muscle ankyrin-B "
            "isoforms. In erythrocytes, ankyrin acts as the primary vertical connector: "
            "four-point attachment to Band 3 (MBD) and beta-spectrin (SBD) at each ankyrin "
            "molecule creates ~1 ankyrin per 25 spectrin heterodimers in the membrane. "
            "Haploinsufficiency → 50% reduction in ankyrin → reduced Band 3/spectrin vertical "
            "tethers → lipid bilayer instability → vesiculation of lipid bilayer microdomains "
            "→ loss of surface area without loss of volume → spherocytosis. "
            "Spectrin-to-ankyrin ratio: normal 6:1; HS 3:1 or worse. "
            "Missense ANK1 variants in the MBD impair Band 3 binding; in SBD impair "
            "spectrin binding — both cause HS. Ankyrin-1 is also expressed in muscle (Ank1.9) "
            "and cerebellum (Ank1.8) — rare ANK1 neurological phenotypes in severe biallelic."
        ),
        "n_patients": 40,
        "seed": SEED_BASE,
        "etiologies": [
            ("ANK1 truncating AD heterozygous — haploinsufficiency, moderate HS", 0.55),
            ("ANK1 missense AD heterozygous — MBD/SBD domain, mild-moderate HS", 0.25),
            ("ANK1 splice-site AD — exon skipping, reduced ankyrin, moderate HS", 0.10),
            ("ANK1 biallelic AR — severe HS, transfusion-dependent, early splenectomy", 0.10),
        ],
        "key_alerts": [
            "ANK1-PARVOVIRUS-B19-APLASTIC-CRISIS: Parvovirus B19 destroys erythroid precursors → ACUTE aplastic crisis in ANY chronic haemolytic anaemia — sudden Hb drop, absent reticulocytes; EMERGENCY transfusion; inform infection control",
            "ANK1-SPLENECTOMY-VACCINATION-MANDATORY: Splenectomy reduces haemolysis 90-95% but MUST vaccinate against pneumococcus, meningococcus, Hib BEFORE splenectomy; post-splenectomy penicillin prophylaxis; defer surgery until age ≥6yr",
            "ANK1-GALLSTONES-PIGMENT: Pigment (bilirubin) gallstones in 50-70% of HS by adulthood — offer cholecystectomy at time of splenectomy if gallstones present",
            "ANK1-FOLATE-SUPPLEMENTATION-MANDATORY: High erythropoietic turnover depletes folate — folic acid supplementation mandatory (5 mg/day) especially during pregnancy and haemolytic crises",
            "ANK1-EMA-BINDING-DIAGNOSIS: Eosin-5'-maleimide (EMA) flow cytometry is the most sensitive diagnostic test for HS (sensitivity ~93%) — reduces EMA binding in HS; request alongside osmotic fragility; DOES NOT require fresh blood like osmotic fragility",
            "ANK1-SPLENECTOMY-DEFERRED-UNDER-6yr: Splenectomy deferred until age ≥6yr to preserve immune development; wearable + prophylactic penicillin in interim; if severe HA (transfusion-dependent) partial splenectomy may be considered earlier",
        ],
    },
    # ── SPTB — Beta-Spectrin HS2 ──
    {
        "gene": "SPTB",
        "protein": "Beta-Spectrin — HS2, Partial Beta-Spectrin Deficiency, Also HE/Pyropoikilocytosis",
        "alias": (
            "SPTB; OMIM gene 182870; HS2 OMIM 616649 / HE2 OMIM 130600; 14q23.3; 2137 aa; ~246 kDa; "
            "SPTB encodes beta-spectrin (spectrin beta chain, erythrocyte; also designated "
            "spectrin-beta-1 or SpBeta), the heterodimeric partner of alpha-spectrin (SPTA1) "
            "in the horizontal spectrin-actin submembrane cytoskeleton. The 2D hexagonal "
            "spectrin-actin meshwork maintains red cell biconcave shape and resilience to "
            "deformation. SPTB is the second most common hereditary spherocytosis gene "
            "(accounting for 20-30% of genotype-positive HS). SPTB haploinsufficiency "
            "(truncating variants — frameshift, nonsense, splice-site) results in partial "
            "beta-spectrin deficiency, reduced spectrin heterodimerisation, and reduced "
            "spectrin content in the membrane skeleton (spectrin deficiency HS2). "
            "Clinically similar to ANK1-HS: spherocytes, haemolytic anaemia, splenomegaly, "
            "hyperbilirubinemia, pigment gallstones. Severity correlates with degree of "
            "spectrin deficiency: mild (spectrin 75-85% of normal → mild HS), moderate "
            "(60-75% → moderate HS), severe (<60% → severe/transfusion-dependent HS). "
            "IMPORTANT: SPTB missense variants in the repeat 1-2 domain (actin-binding "
            "calponin-homology domain, and adjacent tandem repeats) cause HEREDITARY "
            "ELLIPTOCYTOSIS (HE) — elliptical red cells on blood film, usually mild haemolysis "
            "in heterozygotes. SPTB missense + SPTA1 low-expression allele (aLELY) in trans "
            "→ hereditary PYROPOIKILOCYTOSIS (HPP) — severe neonatal haemolytic anaemia "
            "with microspherocytes and poikilocytes, high RBC thermolability at 45-46°C. "
            "Management: identical to ANK1-HS (splenectomy, vaccination, folate)."
        ),
        "aa": "2137 aa",
        "kDa": "~246 kDa",
        "locus": "14q23.3",
        "omim_gene": 182870,
        "omim_disease": 616649,
        "inheritance": "AD — partial beta-spectrin deficiency; SPTB missense + aLELY → AR HPP severe neonatal HA",
        "gene_class": (
            "Beta-spectrin is a 2137-amino acid anti-parallel heterodimer partner of alpha-spectrin. "
            "SPTB structure: (1) N-terminal actin/protein 4.1-binding domain — two calponin "
            "homology (CH) domains (tandem ABD) bind F-actin; an adjacent segment binds protein "
            "4.1R, forming the spectrin-actin-4.1R ternary complex at the pointed end of "
            "short actin protofilaments; (2) central rod domain — 17 tandem triple-helical "
            "spectrin repeats (R1-R17), forming the flexible 100 nm coiled-coil rod; "
            "(3) C-terminal EF-hand/pleckstrin homology domain — binds calmodulin and "
            "phosphoinositides; the C-terminus participates in ankyrin-1 (ANK1) binding "
            "via a site within repeats R14-R15. Alpha-spectrin (SPTA1) and beta-spectrin "
            "self-associate in antiparallel orientation by headpiece-to-tailpiece interaction: "
            "SPTA1 tail (repeats R20-R21) + SPTB head (N-terminal CH domains) and vice versa. "
            "This heterodimers further self-associate laterally to form tetramers "
            "(two antiparallel heterodimers joined at the D1/D2 tetramerisation site of "
            "SPTA1 and corresponding SPTB region). HE/HPP mutations cluster in SPTB repeat 1 "
            "(and SPTA1 repeats 1-2): they weaken the self-association of spectrin tetramers "
            "→ increased heterodimer:tetramer ratio → reduced membrane skeleton cohesion → "
            "elliptocytosis. At 45-46°C, normal spectrin tetramers dissociate; HE/HPP spectrins "
            "dissociate at lower temperatures — RBC thermolability test (37-46°C) is diagnostic "
            "for HPP. SPTB haploinsufficiency in HS1/HS2: reduced spectrin content → "
            "vertical linkage loss via ankyrin → spherocytosis (different from horizontal "
            "HE spectrin self-association defect)."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 1,
        "etiologies": [
            ("SPTB truncating AD heterozygous — HS2, partial spectrin deficiency, moderate", 0.50),
            ("SPTB missense R1-domain AD — hereditary elliptocytosis (HE), mild haemolysis", 0.25),
            ("SPTB missense + aLELY (SPTA1) in trans — hereditary pyropoikilocytosis (HPP), severe neonatal", 0.15),
            ("SPTB large deletion AD — severe HS, transfusion-dependent, early splenectomy", 0.10),
        ],
        "key_alerts": [
            "SPTB-HPP-NEONATAL-SEVERE-HA: Hereditary pyropoikilocytosis (HPP = SPTB HE + SPTA1 aLELY in trans) — SEVERE neonatal haemolytic anaemia with microspherocytes; distinguish from AIHA and infections; RBC thermolability test at 45°C is diagnostic",
            "SPTB-HE-MILD-HETEROZYGOUS-CAUTION: SPTB HE missense heterozygotes typically have mild/compensated haemolysis — do NOT misclassify as benign; compound het with aLELY or second SPTA1 variant = severe HPP",
            "SPTB-SPECTRIN-DEFICIENCY-SEVERITY: Degree of spectrin deficiency correlates with HS severity — spectrin quantitation by SDS-PAGE can guide prognosis; <60% normal spectrin = severe HS with transfusion dependency",
            "SPTB-PARVOVIRUS-B19-CRISIS: As with all hereditary HA, parvovirus B19 → aplastic crisis — maintain high suspicion; check reticulocyte count during any acute Hb drop; EMERGENCY blood transfusion if Hb critically low",
            "SPTB-BLOOD-FILM-CRITICAL: Blood film is diagnostic — spherocytes in HS, elliptocytes in HE, microspherocytes + poikilocytes in HPP; request expert haematology morphology review; EMA flow cytometry complements morphology",
            "SPTB-SPLENECTOMY-CHOLECYSTECTOMY: Combined splenectomy + cholecystectomy at time of intervention if gallstones present; laparoscopic splenectomy preferred; anticoagulation post-splenectomy for 3 months (portal vein thrombosis risk)",
        ],
    },
    # ── SLC4A1 — Band 3 HS4, SAO, Distal RTA ──
    {
        "gene": "SLC4A1",
        "protein": "Band 3 / AE1 — HS4 + Southeast Asian Ovalocytosis (SAO) + Distal RTA AR/Compound Het",
        "alias": (
            "SLC4A1; OMIM gene 109270; HS4 OMIM 612653 / SAO OMIM 166900 / dRTA OMIM 179800; "
            "17q21.31; 911 aa; ~102 kDa; "
            "SLC4A1 encodes Band 3 (AE1 — anion exchanger 1), the most abundant integral "
            "membrane protein of the erythrocyte. Band 3 has two functional domains: "
            "(1) N-terminal cytoplasmic domain (cdAE1, 1-360 aa) — scaffold that binds "
            "ankyrin-1 (ANK1), protein 4.1R, protein 4.2 (EPB42), haemoglobin, glycolytic "
            "enzymes; (2) C-terminal transmembrane domain (tmAE1, 361-911 aa) — 14 "
            "transmembrane helices forming the Cl-/HCO3- anion exchanger that mediates "
            "CO2 transport from tissues to lungs (one of the highest-flux membrane transport "
            "systems in biology, ~5×10^8 anion exchanges per red cell per second). "
            "Three distinct SLC4A1 disease mechanisms: (1) AD HS4 — heterozygous missense "
            "or truncating variants in cdAE1 disrupt ankyrin binding → membrane skeleton "
            "instability → spherocytosis; (2) AD Southeast Asian Ovalocytosis (SAO) — in-frame "
            "9-amino acid (27 bp) deletion removing residues 400-408 (Ala400-Ala408) at the "
            "interface of cdAE1 and first transmembrane segment → rigid oval red cells, reduced "
            "deformability; homozygous SAO is LETHAL IN UTERO; SAO heterozygotes have almost "
            "NO haemolysis but have REDUCED susceptibility to cerebral malaria by ~90% "
            "(Plasmodium falciparum cannot invade SAO red cells); (3) AR/compound heterozygous "
            "dRTA — both alleles loss-of-function in the kidney AE1 (kAE1 isoform, exons 1-3 "
            "skipped in RBC → different start, same cdAE1/tmAE1) → impaired Cl-/HCO3- "
            "exchange in alpha-intercalated cells of collecting duct → failure to acidify urine "
            "→ distal RTA: metabolic acidosis, hypokalemia, nephrocalcinosis, rickets/growth "
            "failure. AR dRTA + HAEMOLYTIC ANAEMIA (HS phenotype) in compound heterozygotes."
        ),
        "aa": "911 aa",
        "kDa": "~102 kDa",
        "locus": "17q21.31",
        "omim_gene": 109270,
        "omim_disease": 612653,
        "inheritance": "AD: HS4 (missense) / SAO (Ala400-408del); AR/compound het: severe HA + distal RTA + nephrocalcinosis",
        "gene_class": (
            "SLC4A1 is a 911-amino acid bicarbonate transporter expressed in two major isoforms: "
            "the erythrocyte isoform (eAE1, all 911 aa, initiating at exon 1-Met1) and the "
            "kidney isoform (kAE1, residues 66-911, initiating at Met65 in exon 4, skipping "
            "exons 1-3 and their corresponding N-terminal 65 aa). eAE1 forms obligate homodimers "
            "in the membrane (dimer interface via TM5-TM6 contacts) and further assembles into "
            "tetramers via cdAE1-cdAE1 dimerisation. The tmAE1 operates via an alternating "
            "access mechanism (SLMB fold): outward-open (Cl-) ↔ occluded ↔ inward-open (HCO3-) "
            "conformational transitions, mediated by movements of the transport domain relative "
            "to the scaffold domain. In SAO (Ala400-408 deletion): the deletion rigidifies the "
            "junction between cdAE1 and TM1, locking the transporter in a conformation that "
            "prevents membrane bilayer thermal fluctuations — the SAO RBC membrane has 3-4x "
            "higher membrane shear viscosity than normal, explaining rigidity without spherocytosis. "
            "SAO Band 3 has abolished anion transport (non-functional in the truncated region) "
            "but the cytoskeletal scaffold function of cdAE1 is retained. In HS4: missense "
            "variants at the ANK1/protein 4.2 binding sites of cdAE1 (Lys56, Glu40, Asp399 "
            "cluster) reduce ankyrin and protein 4.2 recruitment → reduced vertical linkages → "
            "spherocytosis. Protein 4.2 (encoded by EPB42) is entirely dependent on Band 3 "
            "cdAE1 for its membrane association — EPB42 mutations phenocopy SLC4A1 HS4."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 2,
        "etiologies": [
            ("SLC4A1 HS4 missense AD — cdAE1 ANK1-binding domain, moderate HS", 0.45),
            ("SLC4A1 SAO AD Ala400-408del — Southeast Asian origin, ovalocytosis, malaria protection", 0.25),
            ("SLC4A1 compound het AR — severe HA + distal RTA, nephrocalcinosis, metabolic acidosis", 0.20),
            ("SLC4A1 truncating AD — HS4 severe, reduced ankyrin binding", 0.10),
        ],
        "key_alerts": [
            "SLC4A1-SAO-HOMOZYGOUS-LETHAL: Southeast Asian Ovalocytosis (SAO = Ala400-408del) is LETHAL when homozygous — in utero death; when counselling SAO couples confirm both partners' genotype before conception",
            "SLC4A1-DISTAL-RTA-ALKALI-MANDATORY: AR/compound het SLC4A1 → distal RTA → metabolic acidosis + hypokalemia + nephrocalcinosis — MUST give alkali therapy (sodium bicarbonate or Shohl's solution) to prevent renal damage and rickets",
            "SLC4A1-SAO-MALARIA-PROTECTION-NOT-ANAEMIA: SAO heterozygotes have NEAR-ZERO haemolysis in normal conditions — do NOT diagnose haemolytic anaemia on SAO alone; SAO confers 90% protection against cerebral malaria; ovalocytes on blood film",
            "SLC4A1-dRTA-NEPHROCALCINOSIS-SCREEN: In biallelic SLC4A1 — renal ultrasound MANDATORY to assess nephrocalcinosis; urine pH (>5.5 in acidosis = positive dRTA); urine bicarbonate loss; potassium supplementation mandatory alongside alkali",
            "SLC4A1-PROTEIN-4.2-LINKAGE: SLC4A1 cdAE1 is the binding site for protein 4.2 (EPB42) — EPB42 deficiency (Japanese HS5) is phenotypically identical to SLC4A1 HS4; request EPB42 protein quantitation alongside Band 3 in unexplained HS",
            "SLC4A1-TRANSFUSION-dRTA: dRTA with chronic HA may require transfusion support — each unit provides anion-loaded red cells temporarily; long-term alkali therapy reduces haemolysis by improving metabolic environment",
        ],
    },
    # ── SPTA1 — Alpha-Spectrin HE/HPP ──
    {
        "gene": "SPTA1",
        "protein": "Alpha-Spectrin — HE1 (AD Elliptocytosis), HPP (AR Pyropoikilocytosis — Most Severe Neonatal HA)",
        "alias": (
            "SPTA1; OMIM gene 182860; HE1 OMIM 182900 / HPP OMIM 266140 / HS3 OMIM 270970; "
            "1q23.1; 2429 aa; ~280 kDa; "
            "SPTA1 encodes alpha-spectrin (spectrin alpha chain, erythrocyte; SpAlpha), the "
            "partner chain of the spectrin heterodimer in the red cell membrane skeleton. "
            "SPTA1 is responsible for the most common cause of hereditary elliptocytosis (HE) "
            "in African and Southern European populations. Key SPTA1 biology: alpha-spectrin "
            "is synthesised in 3-4x excess over beta-spectrin in erythropoiesis; the "
            "aLELY (low-expression alpha-spectrin, carried in ~20-30% of African chromosomes) "
            "is a common polymorphism in intron 45 (IVS46-1 G>A) that reduces SPTA1 alpha "
            "chain synthesis by 50% on that allele. Critical disease mechanisms: (1) AD HE1 — "
            "SPTA1 missense variants in repeats R1-R2 (Arg28His — most common, Africa; "
            "Arg45Ser, Leu49Phe etc.) destabilise the spectrin tetramer self-association "
            "site: HE1 heterozygotes have 50-60% spectrin dimers vs 5-10% normal — mild "
            "elliptocytosis, usually compensated (minimal haemolysis); (2) AR HPP = "
            "HEREDITARY PYROPOIKILOCYTOSIS — most severe form: HE1 allele PLUS aLELY on "
            "the trans chromosome (compound heterozygous) → severely reduced alpha-spectrin "
            "→ unstable tetramers + low spectrin content → microspherocytes, elliptocytes, "
            "poikilocytes, severe haemolytic anaemia (Hb 4-8 g/dL), MCV 50-60 fL, markedly "
            "elevated reticulocytes. HPP is one of the most severe hereditary haemolytic "
            "anaemias, presenting in the NEONATAL period. RBC thermolability at 45-46°C "
            "is pathognomonic for HPP (normal RBCs thermolabile >49°C). "
            "Splenectomy markedly effective in HPP — most patients transfusion-dependent "
            "prior to splenectomy."
        ),
        "aa": "2429 aa",
        "kDa": "~280 kDa",
        "locus": "1q23.1",
        "omim_gene": 182860,
        "omim_disease": 182900,
        "inheritance": "AD: HE1 (mild, elliptocytes); AR compound het (HE1 allele + aLELY) → HPP severe neonatal HA",
        "gene_class": (
            "Alpha-spectrin is a 2429-amino acid elongated protein with 20 tandem triple-helical "
            "spectrin repeats (R1-R20) plus N-terminal calponin homology (CH) domains and "
            "C-terminal EF-hand motifs. Key structural features: the N-terminal CH1-CH2 "
            "tandem (residues 1-270) constitute the partial actin-binding domain of alpha- "
            "spectrin (complemented by SPTB CH1-CH2 to form the complete actin-binding domain "
            "in the heterodimer); tetramerisation domain = partial helix C of R1 + N-terminal "
            "helix A of R2 forms a 'triple helix' with R17 of the partner SPTB (C-terminus) — "
            "this R17-R1-R2 inter-molecular triple helix is the spectrin tetramerisation site "
            "that HE/HPP mutations disrupt. The aLELY allele (IVS46 variant + Leu1854Arg "
            "in exon 46): the IVS46 variant reduces mRNA splicing efficiency → only 50% "
            "of normal alpha-spectrin mRNA is produced on the aLELY allele; Leu1854Arg "
            "within the aLELY allele reduces spectrin tetramer formation (aLELY alone is "
            "clinically silent because alpha-spectrin is produced in 3-4x excess over "
            "beta-spectrin). When aLELY is combined with a pathogenic HE1 mutation on the "
            "trans chromosome (compound het): total alpha-spectrin falls below threshold "
            "needed for normal membrane skeleton → HPP. The EMA flow cytometry test "
            "detects Band 3 (ANK1/SPTB HS) but is less sensitive for isolated spectrin "
            "deficiency (HE/HPP); acidified glycerol lysis time (AGLT) and osmotic gradient "
            "ektacytometry are better for HE/HPP."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 3,
        "etiologies": [
            ("SPTA1 Arg28His AD — most common HE1 Africa, elliptocytosis, mild/compensated haemolysis", 0.40),
            ("SPTA1 other R1-R2 missense AD — HE1, elliptocytosis, variable haemolysis", 0.25),
            ("SPTA1 HE1 + aLELY in trans — HPP, severe neonatal HA, microspherocytes, early transfusion", 0.25),
            ("SPTA1 biallelic truncating AR — severe HS3, profound haemolysis, splenectomy mandatory", 0.10),
        ],
        "key_alerts": [
            "SPTA1-HPP-NEONATAL-EMERGENCY: Hereditary pyropoikilocytosis (HPP) presents at birth with SEVERE haemolytic anaemia (Hb 4-8 g/dL), MCV 50-60 fL, microspherocytes + poikilocytes — EMERGENCY transfusion; early haematology consult; do NOT delay splenectomy once age-appropriate",
            "SPTA1-ALÉLY-SILENT-TRAP: The aLELY allele is CLINICALLY SILENT alone but DANGEROUS in trans with HE1 mutation → HPP; always test both parents when HE1 allele identified; aLELY prevalence ~20-30% in African populations",
            "SPTA1-RBC-THERMOLABILITY-DIAGNOSTIC: RBC thermolability at 45-46°C (HPP RBCs fragment; normal RBCs intact until >49°C) — PATHOGNOMONIC for HPP; specific, simple test; request alongside ektacytometry",
            "SPTA1-HE1-HETEROZYGOTE-BENIGN-BUT-WATCH: HE1 heterozygotes usually have compensated haemolysis with minimal anaemia — but haemolysis worsens during infections, pregnancy, stress; annual Hb monitoring mandatory",
            "SPTA1-SPLENECTOMY-TRANSFORMS-HPP: Splenectomy dramatically improves HPP — transforms severe transfusion-dependent HA into mild/compensated HA with elliptocytes; strongest recommendation in haematology for HPP; defer until ≥6yr if possible",
            "SPTA1-EKTACYTOMETRY-DEFINITIVE: Laser diffraction ektacytometry (osmotic gradient) gives a characteristic 'HE/HPP' curve pattern — diagnostic when blood film and EMA are equivocal; reference laboratories perform this test",
        ],
    },
    # ── PKLR — Pyruvate Kinase Deficiency ──
    {
        "gene": "PKLR",
        "protein": "Pyruvate Kinase L/R — Most Common Non-Spherocytic HA, Mitapivat FDA 2022, Paradoxical Reticulocytosis Post-Splenectomy",
        "alias": (
            "PKLR; OMIM gene 609712; PK deficiency OMIM 266200; 1q22; 574 aa; ~63 kDa; "
            "PKLR encodes the tissue-specific pyruvate kinase expressed in erythrocytes "
            "(erythrocyte PKR isoform, exons 1-11 with erythroid promoter) and liver "
            "(PKL isoform, exon 1L replaced by exon 1R). Pyruvate kinase catalyses the "
            "final step of glycolysis: phosphoenolpyruvate (PEP) + ADP → pyruvate + ATP. "
            "In mature red cells (no mitochondria), glycolysis is the SOLE source of ATP. "
            "Biallelic PKLR loss-of-function variants → reduced PKR activity → ATP depletion "
            "in red cells → impaired ion pump (Na/K-ATPase) function → red cell dehydration "
            "and increased rigidity → haemolysis in the spleen. PK deficiency is the most "
            "common hereditary non-spherocytic haemolytic anaemia (HNSHA), accounting for "
            "~90% of glycolytic enzyme defects causing HA. Global prevalence ~1/20,000; "
            "carrier frequency up to 1/60 in some populations (Amish, Pennsylvania Dutch "
            "Arg479His homozygous; Pakistani Arg532Trp). Clinical presentation: chronic "
            "haemolytic anaemia (Hb 5-12 g/dL), splenomegaly, jaundice, pigment gallstones. "
            "Blood film: POLYCHROMASIA (reticulocytosis), crenated/echinocyte-like RBCs, "
            "NO spherocytes (non-spherocytic). Chronic HA → elevated 2,3-DPG (PEP-pathway "
            "shunted through 2,3-DPG mutase) → RIGHT-SHIFTED oxygen dissociation curve → "
            "patients paradoxically WELL-TOLERATED despite low Hb (better O2 unloading). "
            "MITAPIVAT (Pyrukynd; FDA Feb 2022) — first approved disease-modifying treatment: "
            "allosteric activator of PKR (binds the activator site) → increases affinity for "
            "PEP, increases ATP production → improved RBC survival. Clinical trial (ACTIVATE): "
            "Hb increase >1 g/dL in ~40% of non-transfusion-dependent adults. Post-splenectomy: "
            "PARADOXICAL RETICULOCYTE RISE — reticulocytes (which still have some mitochondria) "
            "rely on spleen for sequestration; after splenectomy they re-enter circulation, "
            "increasing reticulocyte count (a SIGN OF SUCCESS, not failure)."
        ),
        "aa": "574 aa",
        "kDa": "~63 kDa",
        "locus": "1q22",
        "omim_gene": 609712,
        "omim_disease": 266200,
        "inheritance": "AR — biallelic pathogenic variants; compound heterozygous most common outside consanguineous populations",
        "gene_class": (
            "Pyruvate kinase L/R is a 574-amino acid allosteric enzyme that forms a homotetramer "
            "with each subunit consisting of four domains: (1) N-terminal barrel domain; "
            "(2) A-barrel (TIM barrel) — catalytic domain, PEP-binding site (Lys269, Arg73, "
            "Arg246 — substrate binding cleft); (3) B-domain (inserted into A-barrel) — "
            "mobile domain that closes over the active site upon ADP binding; (4) C-domain — "
            "allosteric regulatory domain, fructose-1,6-bisphosphate (FBP) activator binding "
            "site. PK is allosterically activated by FBP (positive feedback in glycolysis) "
            "and inhibited by ATP (product inhibition) and alanine. In mature erythrocytes, "
            "the enzyme is already in a lower-activity state (lacking FBP activation, low "
            "metabolic throughput) — pathogenic variants must reduce activity below the "
            "residual threshold (~25% normal) to cause clinical HA. Mitapivat binds the "
            "FBP activator pocket allosterically, stabilising the active R-state conformation "
            "of PKR and increasing substrate affinity (reduces Km for PEP): this bypasses "
            "the loss of FBP activation in red cells. Compound heterozygous combinations: "
            "one missense (reduced activity) + one truncating (no protein) = common compound "
            "het. Homozygous null = lethal in utero. 2,3-DPG accumulation in PK deficiency: "
            "PEP accumulates upstream of the PK block → shunted into 2,3-DPG mutase pathway "
            "→ 2,3-DPG rises to 2-3x normal → binds beta-globin deoxyHb T state → shifts "
            "O2 dissociation curve RIGHT → better tissue oxygenation at low Hb (explains "
            "why some patients tolerate Hb 6-7 g/dL without symptoms)."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 4,
        "etiologies": [
            ("PKLR compound het (missense + truncating) AR — moderate HA, splenomegaly, Hb 7-10 g/dL", 0.50),
            ("PKLR homozygous missense AR — consanguineous, moderate-severe HA, recurrent crises", 0.25),
            ("PKLR Arg479His homozygous AR — Amish/Pennsylvania Dutch founder, severe HA", 0.15),
            ("PKLR severe biallelic — transfusion-dependent, splenectomy required, iron chelation", 0.10),
        ],
        "key_alerts": [
            "PKLR-MITAPIVAT-FDA-2022-FIRST-TREATMENT: Mitapivat (Pyrukynd) FDA Feb 2022 — FIRST approved disease-modifying treatment for PK deficiency in adults; allosteric PKR activator; discuss with ALL non-transfusion-dependent PK deficiency adults; monitor haemolysis labs",
            "PKLR-PARADOXICAL-RETICULOCYTOSIS-POST-SPLENECTOMY: After splenectomy, reticulocyte count RISES paradoxically (sequestered reticulocytes re-enter circulation) — reassure patient this is a sign of spleen removal, NOT worsening haemolysis; overall Hb IMPROVES",
            "PKLR-2,3-DPG-RIGHT-SHIFT-TOLERATED: PK deficiency patients tolerate low Hb better than expected due to 2,3-DPG accumulation shifting O2 curve right — do NOT transfuse based on Hb threshold alone; use symptoms + exercise tolerance as guide",
            "PKLR-IRON-OVERLOAD-CHELATION: Regular transfusions → iron overload; even non-transfused PK deficiency has ineffective erythropoiesis → iron hyperabsorption → liver iron deposits; ferritin monitoring 6-monthly; chelation if ferritin >1000 μg/L",
            "PKLR-APLASTIC-CRISIS-PARVOVIRUS: Parvovirus B19 → aplastic crisis — ACUTE EMERGENCY in PK deficiency (as in any chronic HA); IVIg may be needed in immunocompromised; TRANSFUSION mandatory; report to public health if outbreak suspected",
            "PKLR-SPLENECTOMY-INDICATION: Splenectomy indicated for transfusion-dependent PK deficiency or growth failure in children; defer to ≥6yr; vaccination mandatory; may not completely eliminate transfusion requirement in severe cases",
        ],
    },
    # ── G6PD — Glucose-6-Phosphate Dehydrogenase Deficiency ──
    {
        "gene": "G6PD",
        "protein": "G6PD Deficiency — Most Common Human Enzymopathy 400M Worldwide, Rasburicase ABSOLUTE CI, Oxidant Triggers",
        "alias": (
            "G6PD; OMIM gene 305900; G6PD deficiency OMIM 300908; Xq28; 515 aa; ~59 kDa; "
            "G6PD encodes glucose-6-phosphate dehydrogenase, the rate-limiting enzyme of the "
            "pentose phosphate pathway (hexose monophosphate shunt). G6PD catalyses: "
            "glucose-6-phosphate + NADP+ → 6-phosphogluconolactone + NADPH. NADPH is "
            "the sole reducing agent in mature erythrocytes (no mitochondria), providing "
            "reducing equivalents to glutathione reductase to maintain reduced glutathione "
            "(GSH). GSH neutralises reactive oxygen species (H2O2, superoxide) generated "
            "during oxidative stress. G6PD deficiency → reduced NADPH → reduced GSH → "
            "oxidised haemoglobin (metHb) → Heinz body formation (denatured Hb precipitates) "
            "→ Heinz body haemolytic anaemia. G6PD deficiency is X-LINKED: males are "
            "hemizygous (fully affected or normal); females are heterozygous (variable "
            "due to X-inactivation/lyonisation — some severely affected). Epidemiology: "
            "400 million affected worldwide — most common human enzymopathy; protective "
            "against severe malaria (P. falciparum). WHO classification: Class I (severe, "
            "chronic CNSHA — very rare variants); Class II (severe episodic — G6PDd "
            "Mediterranean Ser218Phe, G6PDd Canton, G6PDd Chatham); Class III "
            "(moderate episodic — G6PDd A- Gly202Asp+Val68Met, most common African variant); "
            "Class IV (no deficiency); Class V (increased activity). KEY TRIGGERS: "
            "RASBURICASE (recombinant uricase for hyperuricaemia/tumour lysis) is "
            "ABSOLUTELY CONTRAINDICATED — causes catastrophic acute haemolytic crisis; "
            "primaquine/tafenoquine (anti-malarials); dapsone (anti-infective/leprosy); "
            "nitrofurantoin; high-dose vitamin C (>2g/day); methylene blue (CI in G6PD "
            "for metHb treatment — the only safe alternative is ascorbic acid); fava beans "
            "(favism — Class II/III only, not A-). NEONATAL JAUNDICE in G6PD deficient "
            "male neonates — phototherapy ± exchange transfusion."
        ),
        "aa": "515 aa",
        "kDa": "~59 kDa",
        "locus": "Xq28",
        "omim_gene": 305900,
        "omim_disease": 300908,
        "inheritance": "XL — males hemizygous (full phenotype); females heterozygous (lyonisation → variable, 5-10% severely affected)",
        "gene_class": (
            "G6PD is a 515-amino acid homodimeric enzyme (functional dimer of 59 kDa monomers) "
            "with a two-domain structure: (1) N-terminal beta+alpha domain (residues 1-200) — "
            "structural domain with the NADP+ structural binding site (distinct from the "
            "coenzyme/substrate NADP+ catalytic site); (2) C-terminal beta/alpha TIM barrel "
            "domain (residues 200-515) — catalytic domain containing the glucose-6-phosphate "
            "binding site (Arg72, His201, Arg257, Lys205) and catalytic NADP+ binding site "
            "(Arg72, Asn126, Tyr139, Lys205). The structural NADP+ binding site (at the "
            "C-terminal tail) stabilises the G6PD dimer — variants destabilising this site "
            "reduce dimer stability and enzyme half-life: G6PDd Mediterranean (Ser218Phe) "
            "causes a conformational change reducing thermal stability → accelerated "
            "inactivation in aging red cells (reticulocytes have normal G6PD; older cells "
            "are severely deficient). The G6PDd A- variant (Gly202Asp + Val68Met): Gly202Asp "
            "is the functionally destabilising mutation (reduces NADP+ binding); Val68Met "
            "is an evolutionarily older linked polymorphism. In oxidative stress: NADPH "
            "demand exceeds residual G6PD capacity → NADP+ accumulates → inhibits "
            "glycolysis (phosphoglucose isomerase) → compounding ATP depletion → haemolysis "
            "in 24-72h after trigger exposure. Heinz bodies (denatured Hb precipitates) "
            "detected by brilliant cresyl blue staining — pathognomonic for Heinz body "
            "haemolytic anaemia including G6PD crisis. G6PD assay: fluorescent spot test "
            "(point-of-care); quantitative spectrophotometric assay (standard); CAUTION "
            "— assay unreliable during acute haemolytic crisis (deficient old cells are "
            "lysed, leaving only younger G6PD-normal reticulocytes) — recheck 3 months "
            "after crisis resolution."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 5,
        "etiologies": [
            ("G6PD A- (Gly202Asp) hemizygous male — Class III African, episodic haemolysis with triggers", 0.40),
            ("G6PD Mediterranean (Ser218Phe) hemizygous male — Class II, severe favism, dapsone CI", 0.30),
            ("G6PD heterozygous female — lyonisation variable, trigger-induced haemolysis", 0.15),
            ("G6PD Class I (chronic CNSHA) — rare severe variant, transfusion-dependent", 0.15),
        ],
        "key_alerts": [
            "G6PD-RASBURICASE-ABSOLUTE-CI: Rasburicase (recombinant uricase for tumour lysis hyperuricaemia) ABSOLUTELY CONTRAINDICATED in G6PD deficiency — causes catastrophic acute haemolytic crisis; check G6PD status BEFORE prescribing rasburicase; use allopurinol or febuxostat instead",
            "G6PD-PRIMAQUINE-TAFENOQUINE-CI: Primaquine and tafenoquine CONTRAINDICATED in G6PD deficiency — essential for Plasmodium vivax radical cure; test G6PD status BEFORE anti-malarial therapy; chloroquine/artemisinin safe alternatives",
            "G6PD-SCREEN-BEFORE-PRESCRIBING: ALWAYS check G6PD status before prescribing: dapsone, primaquine, tafenoquine, nitrofurantoin, high-dose vitamin C, methylene blue, rasburicase; document in drug allergy/alerts field",
            "G6PD-ASSAY-TIMING-UNRELIABLE-ACUTE-CRISIS: G6PD assay gives FALSE NORMAL during acute crisis (G6PD-deficient old cells lysed, reticulocytes remain) — always recheck G6PD activity 3 months post-crisis for accurate quantitation",
            "G6PD-NEONATAL-JAUNDICE-MALES: G6PD deficient male neonates at risk for severe neonatal jaundice (neonatal haemolysis + reduced hepatic conjugation) — phototherapy; exchange transfusion if bilirubin critical; screen G6PD in all jaundiced male neonates in endemic regions",
            "G6PD-METHYLENE-BLUE-CI-FOR-METHAEMOGLOBINAEMIA: Methylene blue (first-line metHb treatment) is CI in G6PD — G6PD deficient patients CANNOT generate NADPH to reduce methylene blue to leucomethylene blue; use high-flow O2 + ascorbic acid instead; pre-check G6PD before any procedure using methylene blue dye",
        ],
    },
    # ── PIEZO1 — Dehydrated Hereditary Stomatocytosis ──
    {
        "gene": "PIEZO1",
        "protein": "PIEZO1 — Dehydrated HS/Xerocytosis, AD GOF, High MCHC, Pseudohyperkalaemia, SPLENECTOMY CONTRAINDICATED DVT/PE",
        "alias": (
            "PIEZO1; OMIM gene 611184; DHS OMIM 194380; 16q24.3; 2521 aa; ~286 kDa; "
            "PIEZO1 encodes a mechanosensitive cation channel that is the largest known "
            "single-polypeptide ion channel (2521 amino acids; ~3 MDa homotrimeric complex). "
            "PIEZO1 is a mechanically activated non-selective cation channel that opens in "
            "response to membrane tension and allows Na+, K+, and Ca2+ influx. In erythrocytes, "
            "PIEZO1 activity is normally brief (rapid inactivation) — gain-of-function (GOF) "
            "variants reduce inactivation rate → prolonged cation influx → increased intracellular "
            "Ca2+ → activation of KCNN4 (Gardos channel) → K+ efflux → Cl- follows (osmotic) "
            "→ cell dehydration. Dehydrated red cells: HIGH MCHC (>36 g/dL in many patients), "
            "reduced MCV, and STOMATOCYTES (target cells with central pallor) on blood film — "
            "or may show subtly irregular cells. MCHC >36 g/dL + mild haemolytic anaemia + "
            "splenomegaly = DHS phenotype. Clinical: mild-to-moderate haemolytic anaemia "
            "(Hb 9-12 g/dL), splenomegaly, jaundice, RARE gallstones (relative to HS). "
            "MOST IMPORTANT DIAGNOSTIC CLUE: PSEUDOHYPERKALAEMIA — blood samples show "
            "spuriously HIGH potassium on routine biochemistry; this is because dehydrated, "
            "rigid DHS red cells are fragile to handling and storage at room temperature → "
            "K+ leaks out of cells into plasma during tube transport and analysis; "
            "immediately centrifuge and separate plasma at 37°C (not room temp) to get "
            "TRUE potassium; confirm with plasma separated at 37°C. SPLENECTOMY IS "
            "CONTRAINDICATED in DHS (PIEZO1 and KCNN4): multiple series show HIGH "
            "INCIDENCE of portal vein thrombosis, pulmonary embolism, and DVT post-splenectomy "
            "in DHS — LIFE-THREATENING thrombotic events. Management: folate, avoid "
            "fava beans (haemolysis triggers), hydroxyurea under investigation, "
            "senicapoc (KCNN4/Gardos blocker) studied as DHS treatment. "
            "Recessive PIEZO1 LOF (biallelic): completely different phenotype — "
            "GENERALISED LYMPHATIC DYSPLASIA with lymphoedema."
        ),
        "aa": "2521 aa",
        "kDa": "~286 kDa",
        "locus": "16q24.3",
        "omim_gene": 611184,
        "omim_disease": 194380,
        "inheritance": "AD — gain-of-function (prolonged channel activation); AR LOF → generalised lymphatic dysplasia (completely different)",
        "gene_class": (
            "PIEZO1 is a 2521-amino acid mechanosensitive ion channel that forms a propeller-shaped "
            "homotrimer (~8 MDa, one of the largest known ion channel complexes). Structure "
            "(cryo-EM at 3.8 Å): each subunit consists of: (1) peripheral blade region — "
            "9 blades each made of 4 Transmembrane Helical Units (THUs), totalling 38 TM "
            "helices per subunit; these blades curve away from the central pore like a "
            "propeller and sense membrane curvature/tension; (2) central pore module — "
            "inner helix (IH) and outer helix (OH) form the ion-conducting pore; selectivity "
            "filter (Glu2456 in humans) determines Ca2+ and Na+ permeability; (3) "
            "beam/latch/anchor structures — long intracellular helices that transmit "
            "mechanical force from peripheral blades to the central pore gate. Activation "
            "mechanism: membrane tension (from deformation) displaces peripheral blades → "
            "lever arm mechanism → opens the pore; rapid inactivation occurs via movement "
            "of the intracellular beam back toward the pore. DHS GOF variants cluster in: "
            "the beam region (E756del most common European DHS variant, Glu756del causes "
            "impaired inactivation → prolonged opening → excess Ca2+ influx → KCNN4 "
            "activation → K+ efflux → dehydration); C-terminal domain variants; "
            "IH domain variants. PIEZO1 GOF E756del: frequency ~8% in sub-Saharan African "
            "population — confers some protection against malaria. Recessive PIEZO1 LOF "
            "→ generalised lymphatic dysplasia (PIEZO1 required for lymphatic valve "
            "development via GATA2 signalling)."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 6,
        "etiologies": [
            ("PIEZO1 E756del AD GOF — European DHS, high MCHC, pseudohyperkalaemia, mild haemolysis", 0.45),
            ("PIEZO1 other GOF missense AD — DHS, variable severity, stomatocytes on film", 0.30),
            ("PIEZO1 GOF high MCHC + splenomegaly — moderate DHS, splenectomy AVOIDED", 0.15),
            ("PIEZO1 biallelic LOF AR — generalised lymphatic dysplasia, NOT haemolytic anaemia", 0.10),
        ],
        "key_alerts": [
            "PIEZO1-SPLENECTOMY-CONTRAINDICATED-DVT-PE: Splenectomy is CONTRAINDICATED in DHS (PIEZO1 and KCNN4) — HIGH risk of portal vein thrombosis, pulmonary embolism, and fatal DVT post-splenectomy; this is opposite to HS where splenectomy is first-line treatment",
            "PIEZO1-PSEUDOHYPERKALAEMIA-ARTIFACT: High serum K+ in DHS patients is PSEUDOHYPERKALAEMIA (K+ leaks from fragile DHS red cells during storage at RT) — confirm with plasma separated immediately at 37°C; do NOT treat with kayexalate or dialysis for the artefact",
            "PIEZO1-HIGH-MCHC-DIAGNOSTIC-CLUE: MCHC >36 g/dL + mild haemolytic anaemia = DHS phenotype; request ektacytometry (osmotic dehydration curve shifts left) and blood film for stomatocytes; MCHC elevation is more pronounced than in HS",
            "PIEZO1-RECESSIVE-LOF-DIFFERENT-PHENOTYPE: Biallelic PIEZO1 LOF = GENERALISED LYMPHATIC DYSPLASIA — COMPLETELY DIFFERENT from the haematological DHS phenotype of heterozygous GOF; do NOT confuse the two",
            "PIEZO1-E756DEL-AFRICAN-FOUNDER: PIEZO1 E756del has ~8% carrier frequency in sub-Saharan Africa — when genetic panel testing Africans with mild haemolytic anaemia, always check PIEZO1; malaria protection may be part of evolutionary selection",
            "PIEZO1-FOLATE-ANTICOAGULATION: Folate supplementation mandatory (as with all chronic HA); thromboprophylaxis with LMWH or DOAC should be considered perioperatively or during other high-thrombosis-risk situations in DHS",
        ],
    },
    # ── KCNN4 — Gardos Channel Channelopathy ──
    {
        "gene": "KCNN4",
        "protein": "KCNN4 Gardos Channel — AD GOF Dehydrated HS, Senicapoc Trial, SPLENECTOMY CONTRAINDICATED DVT/PE Risk",
        "alias": (
            "KCNN4; OMIM gene 602754; DHS/Gardos channelopathy OMIM 194380; 19q13.31; 427 aa; ~48 kDa; "
            "KCNN4 encodes the IKCa1/SK4 intermediate-conductance Ca2+-activated potassium "
            "channel (also known as the Gardos channel in erythrocytes, named for György "
            "Gárdos who described calcium-activated K+ transport in red cells in 1958). "
            "KCNN4 is activated by intracellular Ca2+: Ca2+/calmodulin binding to the "
            "calmodulin-binding domain (CaMBD) on the cytoplasmic C-terminal opens the "
            "channel, allowing K+ efflux down its electrochemical gradient (K+_in >> K+_out). "
            "K+ efflux → Cl- efflux (via CFTR or other Cl- channels) → osmotic water "
            "loss → red cell dehydration. In normal red cells, KCNN4 activity is controlled "
            "by intracellular Ca2+ concentration (normally very low, <0.1 μM in rest). "
            "KCNN4 gain-of-function variants → lower Ca2+ threshold for activation → "
            "constitutive/excessive K+ efflux → dehydrated stomatocytes → DHS phenotype. "
            "The DHS phenotype from KCNN4 is clinically IDENTICAL to PIEZO1 DHS: high MCHC, "
            "mild-moderate haemolytic anaemia, splenomegaly, pseudohyperkalaemia. "
            "CRITICAL MANAGEMENT RULE identical to PIEZO1: SPLENECTOMY IS CONTRAINDICATED "
            "due to high postoperative thrombotic risk. SENICAPOC (ICA-17043): a KCNN4 "
            "channel blocker that was investigated in Phase III trials for sickle cell disease "
            "(reduced MCHC in sickle cells, but primary endpoint not met) and studied as a "
            "potential treatment for Gardos channelopathy DHS — reduced MCHC and haemolysis "
            "in case reports. Future KCNN4-specific trials underway. Genetic diagnosis "
            "of KCNN4 DHS vs PIEZO1 DHS: requires panel testing; ektacytometry shows "
            "identical DHS pattern in both; MCHC alone cannot distinguish."
        ),
        "aa": "427 aa",
        "kDa": "~48 kDa",
        "locus": "19q13.31",
        "omim_gene": 602754,
        "omim_disease": 194380,
        "inheritance": "AD — gain-of-function (lower Ca2+ threshold for Gardos channel activation); dehydrated stomatocytes",
        "gene_class": (
            "KCNN4 is a 427-amino acid intermediate-conductance calcium-activated potassium channel "
            "that belongs to the SK/IK channel family (KCNN1-KCNN4). Channel structure: "
            "6 transmembrane helices (S1-S6) per subunit; functional channel is a "
            "homotetramer (4 × KCNN4 subunits, arranged around a central K+ conducting pore). "
            "Key structural features: (1) S1-S4 voltage sensor domain (non-voltage-gated in "
            "IK channels — gating is entirely Ca2+/CaM-dependent); (2) S5-S6 pore domain — "
            "includes the K+ selectivity filter (GYG motif, Gly275-Tyr276-Gly277) which "
            "confers K+ selectivity via a multi-ion mechanism; (3) cytoplasmic C-terminal "
            "calmodulin-binding domain (CaMBD, residues 300-350) — constitutively bound "
            "calmodulin (CaM; one CaM per KCNN4 subunit) acts as the Ca2+ sensor. "
            "Activation gating: Ca2+-free CaM holds the channel closed; Ca2+ binding to "
            "CaM EF-hands (particularly C-lobe EF-hands) induces a conformational change "
            "in CaMBD → mechanically opens the S6 gate → K+ efflux (conductance ~10-14 pS). "
            "GOF mutations in KCNN4 (DHS): cluster in the CaMBD and pore domain: "
            "Arg352His, Arg352Cys, Val282Met, Ala176Thr — reduce Ca2+ threshold for "
            "activation (some variants activate at basal Ca2+ concentrations in resting red "
            "cells without any elevation in Ca2+). Senicapoc (ICA-17043) blocker mechanism: "
            "channel open-state blocker, physically occludes the K+ pore — reduces "
            "constitutive K+ efflux in GOF-KCNN4 red cells, decreasing MCHC and improving "
            "survival. In sickle cell disease, KCNN4 activation (by the elevated Ca2+ in "
            "sickle cells) contributes to hyperdense sickle cell formation — senicapoc "
            "Phase III (Gardos Inhibitor for Sickle Cell Disease) reduced MCHC but did "
            "not reduce sickle pain crises (primary endpoint not met)."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 7,
        "etiologies": [
            ("KCNN4 Arg352His AD GOF — DHS Gardos channelopathy, high MCHC, stomatocytes", 0.40),
            ("KCNN4 other CaMBD missense AD GOF — DHS, pseudohyperkalaemia, splenomegaly", 0.35),
            ("KCNN4 pore domain missense AD GOF — dehydrated stomatocytes, moderate haemolysis", 0.15),
            ("KCNN4 GOF severe — transfusion-dependent, senicapoc trial candidate", 0.10),
        ],
        "key_alerts": [
            "KCNN4-SPLENECTOMY-CONTRAINDICATED-DVT: Splenectomy ABSOLUTELY CONTRAINDICATED in Gardos channelopathy DHS — HIGH postoperative DVT/PE/portal vein thrombosis risk (same as PIEZO1 DHS); clinicians may confuse DHS with HS and recommend splenectomy — this is FATAL ERROR",
            "KCNN4-SENICAPOC-GARDOS-BLOCKER: Senicapoc (ICA-17043) Gardos channel blocker — clinical evidence in Gardos channelopathy DHS; discuss compassionate use / clinical trial enrolment for severe cases; reduces MCHC and haemolysis in responders",
            "KCNN4-PSEUDOHYPERKALAEMIA-DHS: Pseudohyperkalaemia (spuriously high serum K+ due to K+ leakage from dehydrated DHS red cells at room temperature) — same artifact as PIEZO1 DHS; separate plasma at 37°C immediately; avoid inappropriate treatment",
            "KCNN4-VS-PIEZO1-PANEL-TESTING: KCNN4 DHS and PIEZO1 DHS are CLINICALLY IDENTICAL — cannot be distinguished by clinical features, blood film, or ektacytometry alone; requires genetic panel testing to differentiate; this matters for future KCNN4-specific therapies (senicapoc)",
            "KCNN4-HIGH-MCHC-DIAGNOSTIC: High MCHC (>36 g/dL) + mild haemolytic anaemia + splenomegaly + pseudohyperkalaemia = DHS phenotype → genetic panel including PIEZO1 and KCNN4 mandatory; EMA flow cytometry is NORMAL in DHS (unlike HS)",
            "KCNN4-EKTACYTOMETRY-DHS-PATTERN: Laser diffraction ektacytometry shows characteristic DHS pattern (left-shifted dehydration curve, reduced osmotic fragility) distinguishing DHS from HS and HE; request ektacytometry in any unexplained haemolytic anaemia with high MCHC",
        ],
    },
]


def _make_cohort(gd):
    r = random.Random(gd["seed"])
    gene = gd["gene"]
    pts = []
    etiols = gd["etiologies"]
    weights = [e[1] for e in etiols]
    labels = [e[0] for e in etiols]

    for i in range(gd["n_patients"]):
        roll = r.random()
        cumul = 0.0
        etiol = labels[-1]
        for lbl, wt in zip(labels, weights):
            cumul += wt
            if roll < cumul:
                etiol = lbl
                break

        sex = r.choice(["M", "F"])
        # G6PD: X-linked — mostly males
        if gene == "G6PD":
            sex = "M" if r.random() < 0.75 else "F"

        if gene in ("ANK1", "SPTB"):
            age_onset = r.gauss(14, 10)
        elif gene == "SLC4A1":
            age_onset = r.gauss(18, 12)
        elif gene == "SPTA1":
            age_onset = r.gauss(2, 5) if "HPP" in etiol else r.gauss(20, 12)
        elif gene == "PKLR":
            age_onset = r.gauss(5, 6)
        elif gene == "G6PD":
            age_onset = r.gauss(25, 12)  # episodic
        elif gene == "PIEZO1":
            age_onset = r.gauss(20, 12)
        elif gene == "KCNN4":
            age_onset = r.gauss(18, 10)
        else:
            age_onset = r.gauss(20, 12)
        age_onset = max(0.0, round(age_onset, 1))

        dx_delay = r.gauss(24, 18) if gene not in ("G6PD",) else r.gauss(36, 24)
        dx_delay = max(0.0, round(dx_delay, 1))

        # Gene-specific booleans
        splenectomy = r.random() < (0.45 if gene in ("ANK1", "SPTB", "SLC4A1", "PKLR", "SPTA1") else 0.05)
        transfusion_required = r.random() < (0.35 if gene in ("PKLR", "SPTA1") else 0.15)
        gallstones = r.random() < (0.55 if gene in ("ANK1", "SPTB", "SLC4A1") else 0.20)
        folate_prescribed = r.random() < 0.72
        aplastic_crisis = r.random() < (0.22 if gene in ("ANK1", "SPTB", "SPTA1", "PKLR") else 0.08)
        splenomegaly = r.random() < 0.78

        # ANK1/SPTB/SLC4A1 specific
        ema_flow = r.random() < 0.82 if gene in ("ANK1", "SPTB", "SLC4A1") else False
        post_spl_vaccination = r.random() < 0.75 if splenectomy else False
        penicillin_prophylaxis = r.random() < 0.68 if splenectomy else False

        # SPTA1 specific
        rbc_thermolability = r.random() < 0.80 if (gene == "SPTA1" and "HPP" in etiol) else False

        # PKLR specific
        mitapivat_prescribed = r.random() < 0.28 if gene == "PKLR" else False
        iron_chelation = r.random() < 0.30 if gene == "PKLR" else False
        paradoxical_retic = splenectomy and gene == "PKLR"
        diphosphoglycerate_elevated = r.random() < 0.85 if gene == "PKLR" else False

        # G6PD specific
        rasburicase_given = r.random() < 0.06 if gene == "G6PD" else False  # should be 0
        oxidant_trigger = r.choice(["fava beans", "primaquine", "dapsone", "nitrofurantoin", "none", "none"]) if gene == "G6PD" else "N/A"
        g6pd_assay_done = r.random() < 0.78 if gene == "G6PD" else False
        neonatal_jaundice = r.random() < 0.35 if gene == "G6PD" else False

        # PIEZO1/KCNN4 specific
        high_mchc = r.random() < 0.88 if gene in ("PIEZO1", "KCNN4") else False
        pseudohyperkalaemia = r.random() < 0.72 if gene in ("PIEZO1", "KCNN4") else False
        splenectomy_dhs_attempted = r.random() < 0.12 if gene in ("PIEZO1", "KCNN4") else False  # should be near 0
        dhs_dvt_pe = r.random() < (0.55 if splenectomy_dhs_attempted else 0.04) if gene in ("PIEZO1", "KCNN4") else False
        senicapoc_trial = r.random() < 0.10 if gene == "KCNN4" else False
        ektacytometry_done = r.random() < 0.48 if gene in ("PIEZO1", "KCNN4") else r.random() < 0.22

        pts.append({
            "id": f"{gene}-{i+1:03d}",
            "gene": gene,
            "sex": sex,
            "age_onset_years": age_onset,
            "dx_delay_months": dx_delay,
            "etiology": etiol,
            "splenectomy": splenectomy,
            "transfusion_required": transfusion_required,
            "gallstones": gallstones,
            "folate_prescribed": folate_prescribed,
            "aplastic_crisis": aplastic_crisis,
            "splenomegaly": splenomegaly,
            # ANK1/SPTB/SLC4A1
            "ema_flow_done": ema_flow,
            "post_splenectomy_vaccination": post_spl_vaccination,
            "penicillin_prophylaxis": penicillin_prophylaxis,
            # SPTA1
            "rbc_thermolability_test": rbc_thermolability,
            # PKLR
            "mitapivat_prescribed": mitapivat_prescribed,
            "iron_chelation": iron_chelation,
            "paradoxical_reticulocytosis": paradoxical_retic,
            "diphosphoglycerate_elevated": diphosphoglycerate_elevated,
            # G6PD
            "rasburicase_given": rasburicase_given,
            "oxidant_trigger": oxidant_trigger,
            "g6pd_assay_done": g6pd_assay_done,
            "neonatal_jaundice": neonatal_jaundice,
            # PIEZO1/KCNN4
            "high_mchc": high_mchc,
            "pseudohyperkalaemia": pseudohyperkalaemia,
            "splenectomy_dhs_attempted": splenectomy_dhs_attempted,
            "dhs_dvt_pe_post_splenectomy": dhs_dvt_pe,
            "senicapoc_trial": senicapoc_trial,
            "ektacytometry_done": ektacytometry_done,
        })
    return pts


def _pct(pts, key):
    if not pts:
        return 0.0
    return round(100 * sum(1 for p in pts if p.get(key)) / len(pts), 1)


def get_overview():
    all_pts = []
    gene_summaries = []
    all_alerts = []

    for gd in HAEMOLYTIC_ANAEMIA_GENES:
        pts = _make_cohort(gd)
        all_pts.extend(pts)
        gene_summaries.append({
            "gene": gd["gene"],
            "protein": gd["protein"][:80],
            "aa": gd["aa"],
            "kDa": gd["kDa"],
            "locus": gd["locus"],
            "omim_gene": gd["omim_gene"],
            "omim_disease": gd["omim_disease"],
            "inheritance": gd["inheritance"],
            "n_patients": gd["n_patients"],
            "seed": gd["seed"],
            "mean_onset_years": round(sum(p["age_onset_years"] for p in pts) / len(pts), 1),
            "mean_dx_delay_months": round(sum(p["dx_delay_months"] for p in pts) / len(pts), 1),
        })
        all_alerts.extend(gd["key_alerts"])

    # Gene-cohort subsets
    ank1 = [p for p in all_pts if p["gene"] == "ANK1"]
    sptb = [p for p in all_pts if p["gene"] == "SPTB"]
    slc4a1 = [p for p in all_pts if p["gene"] == "SLC4A1"]
    spta1 = [p for p in all_pts if p["gene"] == "SPTA1"]
    pklr = [p for p in all_pts if p["gene"] == "PKLR"]
    g6pd = [p for p in all_pts if p["gene"] == "G6PD"]
    piezo1 = [p for p in all_pts if p["gene"] == "PIEZO1"]
    kcnn4 = [p for p in all_pts if p["gene"] == "KCNN4"]

    agg = {
        "total_patients": len(all_pts),
        "mean_dx_delay_months": round(sum(p["dx_delay_months"] for p in all_pts) / len(all_pts), 1),
        "splenectomy_pct": _pct(all_pts, "splenectomy"),
        "transfusion_required_pct": _pct(all_pts, "transfusion_required"),
        "gallstones_pct": _pct(all_pts, "gallstones"),
        "folate_prescribed_pct": _pct(all_pts, "folate_prescribed"),
        "aplastic_crisis_pct": _pct(all_pts, "aplastic_crisis"),
        "splenomegaly_pct": _pct(all_pts, "splenomegaly"),
        # ANK1
        "ank1_splenectomy_pct": _pct(ank1, "splenectomy"),
        "ank1_ema_flow_pct": _pct(ank1, "ema_flow_done"),
        "ank1_post_spl_vaccination_pct": _pct(ank1, "post_splenectomy_vaccination"),
        "ank1_penicillin_pct": _pct(ank1, "penicillin_prophylaxis"),
        "ank1_aplastic_crisis_pct": _pct(ank1, "aplastic_crisis"),
        "ank1_gallstones_pct": _pct(ank1, "gallstones"),
        # SPTB
        "sptb_splenectomy_pct": _pct(sptb, "splenectomy"),
        "sptb_aplastic_crisis_pct": _pct(sptb, "aplastic_crisis"),
        "sptb_transfusion_pct": _pct(sptb, "transfusion_required"),
        "sptb_gallstones_pct": _pct(sptb, "gallstones"),
        # SLC4A1
        "slc4a1_splenectomy_pct": _pct(slc4a1, "splenectomy"),
        "slc4a1_ema_flow_pct": _pct(slc4a1, "ema_flow_done"),
        "slc4a1_aplastic_crisis_pct": _pct(slc4a1, "aplastic_crisis"),
        # SPTA1
        "spta1_transfusion_pct": _pct(spta1, "transfusion_required"),
        "spta1_rbc_thermolability_pct": _pct(spta1, "rbc_thermolability_test"),
        "spta1_splenectomy_pct": _pct(spta1, "splenectomy"),
        "spta1_aplastic_crisis_pct": _pct(spta1, "aplastic_crisis"),
        # PKLR
        "pklr_mitapivat_pct": _pct(pklr, "mitapivat_prescribed"),
        "pklr_splenectomy_pct": _pct(pklr, "splenectomy"),
        "pklr_iron_chelation_pct": _pct(pklr, "iron_chelation"),
        "pklr_paradoxical_retic_pct": _pct(pklr, "paradoxical_reticulocytosis"),
        "pklr_transfusion_pct": _pct(pklr, "transfusion_required"),
        "pklr_diphosphoglycerate_pct": _pct(pklr, "diphosphoglycerate_elevated"),
        # G6PD
        "g6pd_rasburicase_pct": _pct(g6pd, "rasburicase_given"),
        "g6pd_assay_done_pct": _pct(g6pd, "g6pd_assay_done"),
        "g6pd_neonatal_jaundice_pct": _pct(g6pd, "neonatal_jaundice"),
        "g6pd_aplastic_crisis_pct": _pct(g6pd, "aplastic_crisis"),
        # PIEZO1
        "piezo1_high_mchc_pct": _pct(piezo1, "high_mchc"),
        "piezo1_pseudohyperkalaemia_pct": _pct(piezo1, "pseudohyperkalaemia"),
        "piezo1_splenectomy_attempted_pct": _pct(piezo1, "splenectomy_dhs_attempted"),
        "piezo1_dvt_pe_pct": _pct(piezo1, "dhs_dvt_pe_post_splenectomy"),
        "piezo1_ektacytometry_pct": _pct(piezo1, "ektacytometry_done"),
        # KCNN4
        "kcnn4_high_mchc_pct": _pct(kcnn4, "high_mchc"),
        "kcnn4_pseudohyperkalaemia_pct": _pct(kcnn4, "pseudohyperkalaemia"),
        "kcnn4_splenectomy_attempted_pct": _pct(kcnn4, "splenectomy_dhs_attempted"),
        "kcnn4_dvt_pe_pct": _pct(kcnn4, "dhs_dvt_pe_post_splenectomy"),
        "kcnn4_senicapoc_pct": _pct(kcnn4, "senicapoc_trial"),
        "kcnn4_ektacytometry_pct": _pct(kcnn4, "ektacytometry_done"),
    }

    return {
        "title": "Hereditary-Haemolytic-Anaemia-Atlas — Complete 8-Gene Hereditary Haemolytic Anaemia Reference",
        "subtitle": (
            "ANK1 · SPTB · SLC4A1 · SPTA1 · PKLR · G6PD · PIEZO1 · KCNN4 — "
            "320 patients (8×40, seeds 1526–1533) — HS Splenectomy Pre-Vaccination, "
            "G6PD Rasburicase ABSOLUTE CI, DHS Splenectomy CONTRAINDICATED DVT/PE, "
            "Mitapivat FDA 2022 PK Deficiency, HPP Neonatal Severe HA"
        ),
        "genes": gene_summaries,
        "aggregate_stats": agg,
        "top_alerts": all_alerts[:12],
    }


def get_breakdown():
    breakdown = []
    for gd in HAEMOLYTIC_ANAEMIA_GENES:
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
        "atlas": "Hereditary-Haemolytic-Anaemia-Atlas — Complete 8-Gene Hereditary Haemolytic Anaemia Reference",
        "genes": [gd["gene"] for gd in HAEMOLYTIC_ANAEMIA_GENES],
        "clinical_definitions": [
            {
                "term": "ANK1/SPTB/SLC4A1 Hereditary Spherocytosis — Splenectomy Protocol and Pre-Vaccination Mandate",
                "definition": (
                    "Hereditary spherocytosis (HS) is the most common hereditary haemolytic "
                    "anaemia in Northern Europeans (prevalence 1/2,000). The three commonest "
                    "genes — ANK1 (Ankyrin-1, 30-40%), SPTB (beta-spectrin, 20-30%), and "
                    "SLC4A1 (Band 3, 15-25%) — all cause membrane skeleton vertical-linkage "
                    "defects → lipid bilayer vesiculation → spherocytes → extravascular "
                    "haemolysis in the spleen. Diagnosis: EMA binding flow cytometry "
                    "(sensitivity 93%) + blood film (spherocytes) + osmotic fragility. "
                    "Splenectomy reduces haemolysis by 90-95% but CRITICAL PROTOCOL: "
                    "(1) Vaccinate against Streptococcus pneumoniae (pneumococcal conjugate + "
                    "polysaccharide), Neisseria meningitidis (ACWY + B), and Haemophilus "
                    "influenzae type b (Hib) AT LEAST 2 weeks before surgery; "
                    "(2) Penicillin V prophylaxis post-splenectomy for minimum 2 years in "
                    "adults, lifelong in children; (3) Defer splenectomy until age ≥6 years "
                    "to preserve immunological development; (4) Combined cholecystectomy if "
                    "gallstones present. Aplastic crisis (parvovirus B19) is an acute "
                    "emergency requiring urgent transfusion in any chronic HA. Folate "
                    "supplementation mandatory."
                ),
            },
            {
                "term": "SLC4A1 SAO — Southeast Asian Ovalocytosis and Malaria Protection Without Haemolysis",
                "definition": (
                    "Southeast Asian Ovalocytosis (SAO) is caused by an in-frame 9-amino acid "
                    "deletion (Ala400-Ala408, 27-bp deletion) in SLC4A1 (Band 3/AE1). SAO "
                    "is the only Band 3 variant that rigidifies the RBC membrane rather than "
                    "causing spherocytosis: the deletion locks Band 3 at the TM1/cdAE1 junction "
                    "→ red cells are rigid, oval (ovalocytes), and highly resistant to "
                    "Plasmodium falciparum invasion — providing ~90% protection against cerebral "
                    "malaria. Critically: SAO HETEROZYGOTES HAVE NEAR-ZERO HAEMOLYSIS in "
                    "normal conditions — do not diagnose haemolytic anaemia on SAO alone. "
                    "Homozygous SAO is LETHAL in utero — only heterozygotes survive. Genetic "
                    "counselling: SAO × SAO parents have 25% risk of lethal homozygous foetus. "
                    "Biallelic SLC4A1 loss-of-function (compound het AR dRTA): completely "
                    "different phenotype — HA + distal renal tubular acidosis + nephrocalcinosis "
                    "requiring lifelong alkali therapy."
                ),
            },
            {
                "term": "SPTA1 Hereditary Pyropoikilocytosis — Most Severe Neonatal Hereditary HA, aLELY Trap",
                "definition": (
                    "Hereditary pyropoikilocytosis (HPP) is the most severe hereditary haemolytic "
                    "anaemia presenting in the neonatal period, caused by compound heterozygosity "
                    "of a pathogenic SPTA1 HE1 allele (Arg28His or other tetramerisation domain "
                    "missense) in trans with the low-expression alpha-spectrin allele (aLELY, "
                    "IVS46 variant). The aLELY allele is clinically silent alone (alpha-spectrin "
                    "is synthesised in 3-4x excess) but DANGEROUS in trans with a pathogenic "
                    "allele. HPP features: Hb 4-8 g/dL, MCV 50-60 fL, extreme poikilocytosis "
                    "(microspherocytes + elliptocytes + fragments), severe splenomegaly. "
                    "DIAGNOSTIC: RBC thermolability at 45-46°C (HPP RBCs fragment; normal RBCs "
                    "stable until >49°C). Treatment: splenectomy is dramatically effective in "
                    "HPP (transforms transfusion-dependent HA into mild/compensated) — highest "
                    "priority splenectomy indication in haematology. Clinical trap: aLELY is "
                    "present in ~20-30% of African chromosomes — always test BOTH parents when "
                    "a child has severe neonatal HA and an HE1 variant is found in one parent."
                ),
            },
            {
                "term": "PKLR Pyruvate Kinase Deficiency — Mitapivat FDA 2022 and 2,3-DPG Right-Shift Tolerance",
                "definition": (
                    "Pyruvate kinase deficiency (PKD) is the most common hereditary non-spherocytic "
                    "haemolytic anaemia, caused by biallelic PKLR variants reducing ATP production "
                    "in red cells. A striking clinical feature: PKD patients TOLERATE low Hb levels "
                    "better than expected because elevated 2,3-DPG (accumulated upstream of the PK "
                    "block) shifts the oxygen dissociation curve rightward → better O2 unloading "
                    "at tissues → symptoms may be surprisingly mild at Hb 6-7 g/dL. Do not "
                    "transfuse based on Hb threshold alone — use symptoms and exercise tolerance. "
                    "Post-splenectomy: PARADOXICAL RETICULOCYTE RISE is a sign of successful "
                    "splenectomy (reticulocytes re-enter circulation from the 'reticulocyte pool') — "
                    "reassure patient. Mitapivat (Pyrukynd; FDA Feb 2022) is the first disease-"
                    "modifying treatment: allosteric PKR activator that bypasses the deficit in "
                    "non-haem pathogenic variants; increases Hb by >1 g/dL in ~40% of adults. "
                    "Iron overload from transfusions + ineffective erythropoiesis → monitor "
                    "ferritin 6-monthly; chelation if >1000 μg/L."
                ),
            },
            {
                "term": "G6PD Deficiency — Rasburicase ABSOLUTE Contraindication and Oxidant Trigger Management",
                "definition": (
                    "Glucose-6-phosphate dehydrogenase (G6PD) deficiency is the most common human "
                    "enzymopathy (400 million affected worldwide), X-linked, predominantly affecting "
                    "males. G6PD provides the only source of NADPH in mature erythrocytes — "
                    "deficiency leaves red cells defenceless against oxidative stress → Heinz body "
                    "haemolysis. RASBURICASE (recombinant uricase for hyperuricaemia/tumour lysis) "
                    "is ABSOLUTELY CONTRAINDICATED — produces H2O2 as a byproduct, causing "
                    "catastrophic haemolysis in G6PD-deficient patients; always check G6PD BEFORE "
                    "prescribing. Other CI/triggers: primaquine, tafenoquine (anti-malarials), "
                    "dapsone, nitrofurantoin, high-dose vitamin C, methylene blue (for "
                    "methaemoglobinaemia — use O2 + ascorbic acid instead), fava beans (favism). "
                    "Diagnostic trap: G6PD assay gives FALSE NORMAL during acute crisis because "
                    "deficient old red cells are lysed leaving only younger reticulocytes — "
                    "recheck 3 months after crisis resolution. Neonatal G6PD males require "
                    "phototherapy ± exchange transfusion for severe jaundice."
                ),
            },
            {
                "term": "PIEZO1/KCNN4 Dehydrated Hereditary Stomatocytosis — Splenectomy CONTRAINDICATED, DVT/PE Risk",
                "definition": (
                    "Dehydrated hereditary stomatocytosis (DHS) is caused by gain-of-function "
                    "mutations in PIEZO1 (mechanosensitive cation channel) or KCNN4 (Gardos "
                    "channel). Both channels, when constitutively activated, cause K+ efflux "
                    "→ RBC dehydration → high MCHC (>36 g/dL), reduced cell volume, mild-to-"
                    "moderate haemolytic anaemia. MOST IMPORTANT CLINICAL RULE: "
                    "SPLENECTOMY IS CONTRAINDICATED IN DHS — multiple series document HIGH "
                    "incidence of portal vein thrombosis, pulmonary embolism, and fatal DVT "
                    "post-splenectomy in DHS patients. This is the OPPOSITE of hereditary "
                    "spherocytosis where splenectomy is first-line treatment. Pseudohyperkalaemia "
                    "(spuriously high serum K+ from K+ leakage in stored tubes at room temperature) "
                    "is a diagnostic clue — confirm by plasma separation at 37°C immediately. "
                    "Treatment options: folate, supportive; senicapoc (Gardos channel blocker) "
                    "for KCNN4-DHS; both PIEZO1 and KCNN4 DHS are clinically identical and "
                    "require genetic panel testing to differentiate — critical for future "
                    "targeted therapies."
                ),
            },
            {
                "term": "Aplastic Crisis — Parvovirus B19 Emergency in ALL Hereditary Haemolytic Anaemia",
                "definition": (
                    "Parvovirus B19 (B19V) infects and destroys erythroid precursor cells (BFU-E "
                    "and CFU-E progenitors) by binding P antigen (globoside) — this dramatically "
                    "and acutely reduces erythropoiesis. In immunocompetent individuals, the "
                    "resulting 7-10 day aplasia is tolerated because normal RBC lifespan (120 days) "
                    "buffers the gap. In any CHRONIC HAEMOLYTIC ANAEMIA (HS, HE, HPP, PKD, G6PD, "
                    "DHS), RBC lifespan is already dramatically shortened (10-30 days) — even "
                    "7-10 days of aplasia causes an ACUTE, SEVERE DROP IN HAEMOGLOBIN requiring "
                    "emergency transfusion. Key features: sudden Hb drop, ABSENT RETICULOCYTES "
                    "(reticulocyte count <0.1%), mild fever, typical B19 rash (slapped cheek) "
                    "may precede haematological crisis. Management: urgent transfusion; B19-PCR "
                    "confirmation; IVIg for immunocompromised; isolate from immunocompromised "
                    "contacts (pregnant women, immunosuppressed patients). Ensure all chronic HA "
                    "patients have an emergency haematology contact for acute B19 crisis."
                ),
            },
            {
                "term": "Hereditary Haemolytic Anaemia Diagnostic Ladder — EMA, Osmotic Fragility, Ektacytometry",
                "definition": (
                    "A systematic diagnostic approach to hereditary haemolytic anaemia: "
                    "(1) Blood film — spherocytes (HS: ANK1/SPTB/SLC4A1/SPTA1 severe), "
                    "elliptocytes (HE: SPTA1 mild), microspherocytes + poikilocytes (HPP), "
                    "stomatocytes (DHS: PIEZO1/KCNN4), crenated/echinocytes (PKLR); "
                    "(2) EMA binding flow cytometry — highly sensitive for HS (ANK1, SPTB, "
                    "SLC4A1, EPB42); NORMAL in HE, HPP, PKLR, G6PD, DHS — critical "
                    "negative discriminator; (3) Osmotic fragility — increased in HS; "
                    "DECREASED in DHS (dehydrated cells); DECREASED in SAO (rigid cells); "
                    "requires fresh blood (use EMA instead for delayed testing); "
                    "(4) Ektacytometry (laser diffraction) — gold-standard: gives characteristic "
                    "curves distinguishing HS, HE, HPP, DHS; available at reference centres; "
                    "(5) Enzyme assays — PKLR (PK), G6PD quantitation; G6PD recheck 3 months "
                    "post-crisis; (6) Genetic panel — ANK1, SPTB, SLC4A1, SPTA1, PKLR, G6PD, "
                    "PIEZO1, KCNN4, EPB41, EPB42 + others; mandatory for ambiguous cases, "
                    "pre-splenectomy genetic counselling, and family cascade testing."
                ),
            },
        ],
    }
