#!/usr/bin/env python3
"""Red-Cell-Disorders-Atlas — Complete 8-Gene Hereditary Haemolytic Anaemia Atlas
G6PD    (Glucose-6-Phosphate Dehydrogenase; 515 aa; Xq28; X-Linked;
          G6PD deficiency — most common red cell enzymopathy worldwide, >400 million affected;
          Neonatal jaundice, episodic haemolytic anaemia triggered by oxidants/fava beans/infection;
          Class I (chronic CNSHA) to Class V (elevated enzyme); Class II/III most common worldwide;
          G6PD A- (Asp202Asn + Val68Met): sub-Saharan African; G6PD Mediterranean (Ser188Phe): Mediterranean/Middle East) ·
PKLR    (Pyruvate Kinase Liver/Red Blood Cell Isoform; 574 aa; 1q22; AR;
          PK deficiency — most common AR non-spherocytic haemolytic anaemia in Northern European populations;
          Chronic compensated haemolysis; 2,3-BPG markedly elevated (right-shifted O2 dissociation curve);
          Iron overload due to ineffective erythropoiesis; mitapivat (Pyrukynd) FDA approved 2022;
          Splenomegaly; gallstones; aplastic crisis risk from parvovirus B19) ·
ANK1    (Ankyrin-1; 1881 aa; 8p11.21; AD;
          Hereditary spherocytosis type 1 (HS1) — most common hereditary spherocytosis cause (40–65%);
          Ankyrin links spectrin-4.1 network to band 3 — ANK1 haploinsufficiency → spherocyte formation;
          Osmotic fragility test positive; EMA flow cytometry diagnostic; splenectomy curative;
          Severity ranges from mild compensated to severe transfusion-dependent haemolysis) ·
SPTA1   (Spectrin Alpha Chain / Alpha-Spectrin; 2429 aa; 1q23.1; AR;
          Hereditary elliptocytosis (HE) and hereditary pyropoikilocytosis (HPP) — alpha-spectrin;
          Common α-LELY allele (αLELY) modifies clinical expression in trans;
          HPP: severe neonatal haemolysis → fragments, spherocytes, poikilocytes; heat-labile elliptocytes 45°C;
          HE usually mild AD in heterozygotes; HPP when compound heterozygous with αLELY or in trans) ·
SLC4A1  (Band 3 / Anion Exchanger 1; 911 aa; 17q21.31; AD / AR;
          AD: Hereditary spherocytosis type 4 (HS4) — haploinsufficiency or dominant-negative;
          AR: Southeast Asian ovalocytosis (SAO) — band 3 Δ400–408 deletion — malaria resistance;
          Distal renal tubular acidosis (dRTA): SLC4A1 variants in collecting duct → acid-loading test;
          Southeast Asian ovalocytosis rarely causes haemolysis but protects against cerebral malaria) ·
EPB42   (Erythrocyte Membrane Protein Band 4.2 / Protein 4.2; 691 aa; 15q15.2; AR;
          Hereditary spherocytosis type 5 (HS5) — protein 4.2 deficiency;
          Stabilises band 3–ankyrin interaction; LOF → spherocyte formation, reduced membrane stability;
          Japanese founder: p.Ala142Thr (c.424G>A) — most common HS variant in East Asian populations;
          Mild to moderate haemolysis; EMA flow cytometry diagnostic; responds to splenectomy) ·
HK1     (Hexokinase-1; 917 aa; 10q22.1; AR;
          Hexokinase deficiency — rare severe non-spherocytic haemolytic anaemia;
          Hexokinase first step of glycolysis: glucose → glucose-6-phosphate; HK1 LOF → RBC energy failure;
          Neonatal haemolytic anaemia; severe aplastic crisis risk; no specific treatment — supportive;
          Hereditary spherocytosis must be excluded (different pathway — EMA flow cytometry normal in HK1)) ·
PIEZO1  (Piezo-Type Mechanosensitive Ion Channel Component 1; 2521 aa; 16q24.3; AD;
          Dehydrated hereditary stomatocytosis (DHS) / Xerocytosis — gain-of-function PIEZO1;
          PIEZO1 GOF → constitutive Ca2+ entry → K+ efflux → water loss → dehydrated RBCs → stomatocytes;
          MCHC elevated; reticulocytes markedly elevated out of proportion to mild haemolysis;
          African-variant PIEZO1 p.Glu756del enriched in malaria-endemic regions — erythrocytosis phenotype;
          SPLENECTOMY CONTRAINDICATED — risk of life-threatening thromboembolism post-splenectomy)
320-patient aggregate cohort (8 × 40, seeds 1422–1429)
"""

import random

SEED_BASE = 1422

RCD_GENES = [
    # ── G6PD — X-Linked G6PD Deficiency ──
    {
        "gene": "G6PD",
        "protein": "Glucose-6-Phosphate Dehydrogenase (Pentose Phosphate Pathway Enzyme)",
        "alias": (
            "G6PD; OMIM gene 305900; G6PD deficiency #300908; Xq28; 515 aa; ~59 kDa homodimer; "
            "Most common human enzymopathy — >400 million affected worldwide; X-linked (males hemizygous); "
            "Class I: chronic non-spherocytic haemolytic anaemia (CNSHA) — rare, severe; "
            "Class II: G6PD Mediterranean Ser188Phe <10% residual activity; episodic severe haemolysis; "
            "Class III: G6PD A- (Asp202Asn+Val68Met) ~20% residual; commonest African variant; "
            "Triggers: fava beans (vicine/convicine), primaquine, dapsone, rasburicase, infection; "
            "Neonatal jaundice in 1st week; aplastic crisis from parvovirus B19 possible; "
            "G6PD protects Plasmodium falciparum from RBC oxidative stress — balanced polymorphism"
        ),
        "aa": "515 aa",
        "kDa": "~59 kDa homodimer",
        "locus": "Xq28",
        "omim_gene": 305900,
        "omim_disease": 300908,
        "inheritance": "X-linked recessive — males hemizygous (fully affected); females heterozygous (variable expression due to X-inactivation; homozygous females = fully affected)",
        "gene_class": (
            "G6PD is the rate-limiting enzyme of the pentose phosphate pathway (PPP), "
            "generating NADPH essential for reducing glutathione (GSH) in red blood cells. "
            "RBCs lack mitochondria and depend exclusively on PPP for NADPH; G6PD deficiency "
            "leaves RBCs vulnerable to oxidative haemolysis under physiological oxidant challenges. "
            "The G6PD A- variant (Sub-Saharan Africa) retains ~20% activity and typically causes "
            "only episodic haemolysis during severe oxidant exposure. G6PD Mediterranean (Ser188Phe) "
            "retains <10% activity, causing more severe episodic haemolysis and significant neonatal "
            "jaundice. Class I CNSHA variants abolish enzyme activity and cause chronic haemolysis. "
            "Females heterozygous for Class II/III typically show intermediate activity due to "
            "random X-inactivation; heterozygous females for Class I often show clinical disease."
        ),
        "n_patients": 40,
        "seed": SEED_BASE,
        "etiologies": [
            ("G6PD A- (Asp202Asn + Val68Met) — Sub-Saharan African variant, Class III, ~20% activity", 0.40),
            ("G6PD Mediterranean (Ser188Phe) — Mediterranean/Middle East, Class II, <10% activity", 0.35),
            ("G6PD Mahidol (Gly163Ser) — Southeast Asian, Class III", 0.10),
            ("Class I CNSHA variant — rare, severe, chronic haemolysis", 0.10),
            ("Novel/other Class II–III variant", 0.05),
        ],
        "age_onset_years_range": (0, 30),
        "sex_ratio_M": 0.80,  # predominantly male (X-linked)
        "rates": {
            "neonatal_jaundice":                    0.55,
            "episodic_haemolytic_crisis":           0.75,
            "fava_bean_trigger_identified":         0.40,
            "drug_trigger_primaquine_dapsone":      0.30,
            "infection_trigger":                    0.55,
            "chronic_cnsha_class_i":                0.10,
            "splenomegaly":                         0.30,
            "gallstones":                           0.25,
            "back_pain_at_crisis":                  0.50,
            "haemoglobinuria_cola_urine":           0.45,
            "heinz_bodies_on_smear":                0.60,
            "bite_cells_blister_cells":             0.55,
            "normal_between_episodes":              0.75,
            "female_heterozygote":                  0.20,
            "g6pd_activity_low":                    0.95,
        },
        "hallmarks": [
            "Most common enzymopathy worldwide — always consider in unexplained haemolysis in males",
            "G6PD activity assay TIMED correctly — false normal during acute crisis (reticulocytes have high activity)",
            "G6PD Mediterranean: fava beans cause severe acute crisis — dietary avoidance essential",
            "Class I CNSHA: chronic haemolysis without triggers — different from classic episodic form",
            "Heinz bodies + bite cells on blood film during crisis: pathognomonic for oxidative haemolysis",
            "Haemoglobinuria (cola-coloured urine) at crisis — renal impairment risk in severe cases",
            "Rasburicase ABSOLUTELY CONTRAINDICATED — causes catastrophic haemolysis in G6PD deficiency",
        ],
        "treatment_alerts": [
            "RASBURICASE ABSOLUTELY CONTRAINDICATED — fatal haemolysis; check G6PD before administration",
            "PRIMAQUINE / DAPSONE CONTRAINDICATED — anti-malarial oxidants; use chloroquine instead",
            "Fava beans / broad beans: avoid in G6PD Mediterranean (Class II) and all symptomatic patients",
            "Neonatal jaundice: phototherapy early; exchange transfusion if bilirubin critically elevated",
            "Acute crisis: stop offending drug/food; hydration; transfusion if Hb <7 g/dL or symptomatic",
            "G6PD testing: WAIT 3 months after acute haemolytic episode — reticulocytes give false-normal",
            "Aplastic crisis from parvovirus B19: transfusion support; usually self-limiting",
            "Folate supplementation in chronic haemolysis (CNSHA Class I)",
        ],
        "primary_treatment": (
            "Avoidance of triggers (fava beans, primaquine, dapsone, rasburicase). "
            "Acute crisis: remove precipitant; IV hydration; transfuse if Hb <7 g/dL or cardiovascular compromise. "
            "Neonatal jaundice: phototherapy; exchange transfusion threshold per bilirubin level. "
            "Class I CNSHA: folate supplementation; splenectomy in selected severe cases (Class I only). "
            "Patient/family education on trigger avoidance. Medical alert bracelet. "
            "Pre-operative G6PD screening — avoid methylene blue (CI), rasburicase (CI)."
        ),
    },

    # ── PKLR — Pyruvate Kinase Deficiency ──
    {
        "gene": "PKLR",
        "protein": "Pyruvate Kinase Liver/Red Blood Cell Isoform (Glycolytic Enzyme)",
        "alias": (
            "PKLR; OMIM gene 609712; PK deficiency #266200; 1q22; 574 aa; ~62 kDa; "
            "Most common AR non-spherocytic haemolytic anaemia in Northern European populations; "
            "PK catalyses phosphoenolpyruvate → pyruvate (last glycolytic step, net ATP generation); "
            "PKLR LOF → ATP depletion → RBC rigidity → haemolysis; 2,3-BPG markedly elevated; "
            "2,3-BPG elevation shifts O2 dissociation curve RIGHT — clinical tolerance better than Hb predicts; "
            "Mitapivat (Pyrukynd) — PK activator; FDA approved August 2022; addresses all variants; "
            "Iron overload (even without transfusions) from ineffective erythropoiesis — monitor ferritin annually; "
            "Compound heterozygous most common; common founder variant in Northern Europe: c.1529G>A (Arg510Gln)"
        ),
        "aa": "574 aa",
        "kDa": "~62 kDa",
        "locus": "1q22",
        "omim_gene": 609712,
        "omim_disease": 266200,
        "inheritance": "AR — compound heterozygous (most common) or homozygous; parents obligate carriers",
        "gene_class": (
            "PKLR encodes the pyruvate kinase isoform expressed in red blood cells (R-type PK) "
            "and liver (L-type PK). PK catalyses the final ATP-generating step of glycolysis: "
            "phosphoenolpyruvate + ADP → pyruvate + ATP. RBCs rely entirely on glycolysis for ATP; "
            "PKLR deficiency severely impairs ATP production, reducing RBC deformability and "
            "causing extravascular haemolysis predominantly in the spleen. Paradoxically, "
            "2,3-bisphosphoglycerate (2,3-BPG) accumulates massively (substrate blockade above PK), "
            "shifting the oxygen-haemoglobin dissociation curve rightward and allowing patients to "
            "tolerate surprisingly low haemoglobin levels without severe symptoms. Mitapivat "
            "(Pyrukynd) is a first-in-class allosteric PK activator approved by FDA (2022) for "
            "adult patients with PK deficiency, representing the first disease-modifying therapy."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 1,
        "etiologies": [
            ("c.1529G>A (p.Arg510Gln) — common Northern European compound heterozygote or homozygote", 0.35),
            ("Other missense compound heterozygous (one or both alleles)", 0.35),
            ("Splice site variant + missense (compound heterozygous)", 0.20),
            ("Homozygous missense — consanguineous family", 0.10),
        ],
        "age_onset_years_range": (0, 5),
        "sex_ratio_M": 0.50,
        "rates": {
            "neonatal_haemolysis":                  0.65,
            "neonatal_jaundice_requiring_treatment": 0.55,
            "exchange_transfusion_neonatal":        0.25,
            "chronic_haemolytic_anaemia":           0.95,
            "splenomegaly":                         0.80,
            "splenectomy_performed":                0.50,
            "gallstones":                           0.55,
            "elevated_2_3_bpg":                     0.92,
            "elevated_reticulocytes":               0.95,
            "iron_overload_elevated_ferritin":      0.55,
            "transfusion_dependent":                0.30,
            "aplastic_crisis_parvovirus_b19":       0.15,
            "mitapivat_eligible":                   0.40,
            "folate_supplementation":               0.90,
        },
        "hallmarks": [
            "2,3-BPG paradox: markedly elevated 2,3-BPG → right-shifted O2 curve → patient tolerates Hb much lower than expected",
            "Iron overload WITHOUT transfusion — ineffective erythropoiesis drives hepcidin suppression → monitor ferritin",
            "Splenectomy improves but does NOT cure — reticulocyte count rises post-splenectomy",
            "Mitapivat (Pyrukynd): FDA 2022 — first disease-modifying therapy; allosteric PK activator",
            "Aplastic crisis from parvovirus B19 — most dangerous acute complication; transfusion support essential",
            "Common Northern European founder: c.1529G>A (Arg510Gln) — request targeted testing first in population",
        ],
        "treatment_alerts": [
            "Iron overload: monitor ferritin annually — chelation if transferrin saturation >45% + rising ferritin",
            "Folate supplementation: 5 mg/day — all patients with chronic haemolysis",
            "Mitapivat (Pyrukynd) 50 mg twice daily — FDA 2022; assess response at 12 weeks",
            "Splenectomy: reduces transfusion burden but does not cure; post-splenectomy vaccines mandatory",
            "Post-splenectomy: lifelong penicillin V prophylaxis (or azithromycin if allergic)",
            "Parvovirus B19 aplastic crisis: transfusion support; monitor annually — especially in children",
            "Gene therapy trials ongoing — refer to specialist centre",
        ],
        "primary_treatment": (
            "Folate 5 mg/day. Transfusion support when Hb symptomatic (threshold guided by 2,3-BPG tolerance). "
            "Mitapivat (Pyrukynd) for eligible adults — 50 mg BD. "
            "Iron monitoring: ferritin annually; chelation (deferasirox) if iron-overloaded. "
            "Splenectomy in severe transfusion-dependent cases (post-splenectomy vaccines: MenACWY, PneumoVax, HiB, annual flu). "
            "Penicillin prophylaxis post-splenectomy. Gene therapy clinical trials for eligible patients."
        ),
    },

    # ── ANK1 — Hereditary Spherocytosis Type 1 ──
    {
        "gene": "ANK1",
        "protein": "Ankyrin-1 (Membrane Skeleton Scaffold Protein — Links Spectrin to Band 3)",
        "alias": (
            "ANK1; OMIM gene 612641; Hereditary spherocytosis type 1 #182900; 8p11.21; 1881 aa; ~206 kDa; "
            "Most common hereditary spherocytosis gene — 40–65% of all HS cases; "
            "ANK1 bridges the spectrin-based cytoskeletal network to the integral band 3 protein; "
            "Haploinsufficiency → reduced membrane stability → spherocyte formation → extravascular haemolysis; "
            "EMA (eosin-5'-maleimide) binding test: reduced binding PATHOGNOMONIC — sensitivity/specificity >90%; "
            "Severity: mild to severe; splenectomy curative but deferred until age >5 years if possible; "
            "Dominant inheritance: de novo mutations ~25%; family history often incomplete due to variable expressivity; "
            "Complication: aplastic crisis (parvovirus B19), gallstones, extramedullary haematopoiesis in severe"
        ),
        "aa": "1881 aa",
        "kDa": "~206 kDa",
        "locus": "8p11.21",
        "omim_gene": 612641,
        "omim_disease": 182900,
        "inheritance": "AD — haploinsufficiency most common; de novo ~25%; some compound heterozygous AR forms",
        "gene_class": (
            "ANK1 encodes ankyrin-1, the principal scaffolding protein of the erythrocyte membrane "
            "skeleton. Ankyrin-1 forms the core of the junctional complex, binding the cytoplasmic "
            "tail of band 3 (SLC4A1) on the lipid bilayer face and the beta-spectrin subunit of the "
            "spectrin-actin network on the cytoplasmic face. This bridging function is essential for "
            "membrane mechanical stability and deformability. ANK1 haploinsufficiency reduces the "
            "density of spectrin-actin attachment points, causing membrane instability and vesicle "
            "loss during repeated splenic passage. The resultant spherocytes have reduced surface "
            "area-to-volume ratio, rendering them susceptible to osmotic lysis and splenic trapping. "
            "EMA flow cytometry (reduced band 3/Rh-complex binding) is the most sensitive non-"
            "invasive diagnostic test. Splenectomy removes the site of destruction and is curative "
            "in symptomatic patients."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 2,
        "etiologies": [
            ("ANK1 haploinsufficiency — frameshift / nonsense / splice — inherited from parent", 0.50),
            ("De novo ANK1 pathogenic variant — no family history", 0.25),
            ("Missense variant — dominant-negative or haploinsufficiency", 0.20),
            ("Large intragenic deletion (MLPA-detectable)", 0.05),
        ],
        "age_onset_years_range": (0, 10),
        "sex_ratio_M": 0.50,
        "rates": {
            "neonatal_jaundice":                    0.70,
            "neonatal_exchange_transfusion":        0.20,
            "chronic_haemolytic_anaemia":           0.90,
            "splenomegaly":                         0.85,
            "gallstones":                           0.45,
            "positive_ema_flow_cytometry":          0.93,
            "positive_osmotic_fragility":           0.85,
            "splenectomy_performed":                0.55,
            "aplastic_crisis_parvovirus":           0.15,
            "family_history_spherocytosis":         0.70,
            "elevated_mchc":                        0.75,
            "elevated_reticulocytes":               0.85,
            "transfusion_dependent_severe":         0.10,
            "extramedullary_haematopoiesis":        0.05,
        },
        "hallmarks": [
            "EMA flow cytometry: reduced binding — PATHOGNOMONIC; sensitivity >90%; replace osmotic fragility test",
            "MCHC elevated: red cells dense → dehydrated → spherical; MCHC >36 g/dL in most HS",
            "Neonatal: 70% present with neonatal jaundice (1st week); phototherapy/exchange transfusion",
            "Family history: AD — but de novo ~25%; examine parents' blood smear and reticulocyte count",
            "Aplastic crisis: parvovirus B19 most dangerous acute event — can be life-threatening; transfuse",
            "Splenectomy: curative for haemolysis but deferred to age >5 due to encapsulated organism risk",
        ],
        "treatment_alerts": [
            "Splenectomy: highly effective; defer to age >5; post-splenectomy vaccines MANDATORY before surgery",
            "Post-splenectomy: lifelong penicillin V prophylaxis; patient education on fever protocol",
            "Parvovirus B19 aplastic crisis: severe acute anaemia; transfusion; usually self-limiting 10–14 days",
            "Folate 5 mg/day in all patients with ongoing haemolysis (increased erythropoietic demand)",
            "Gallstones: annual USS from age 5; cholecystectomy concurrent with splenectomy if present",
            "MCHC: monitor — significantly elevated (>38 g/dL) suggests dehydration component",
        ],
        "primary_treatment": (
            "Folate 5 mg/day. EMA flow cytometry for diagnosis (replace osmotic fragility). "
            "Mild–moderate HS: no intervention except folate and parvovirus watch. "
            "Moderate–severe: splenectomy (after age 5; concurrent cholecystectomy if gallstones). "
            "Pre-splenectomy vaccines: Pneumococcal (PCV + PPSV23), MenACWY, HiB, annual flu ≥2 weeks prior. "
            "Post-splenectomy: penicillin V 250 mg BD (lifelong or minimum 5 years). "
            "Aplastic crisis: transfusion support, parvovirus B19 serology."
        ),
    },

    # ── SPTA1 — Hereditary Elliptocytosis / Pyropoikilocytosis ──
    {
        "gene": "SPTA1",
        "protein": "Spectrin Alpha Chain / Alpha-Spectrin I (Membrane Skeleton Structural Protein)",
        "alias": (
            "SPTA1; OMIM gene 182860; Hereditary elliptocytosis 2 #130600; HPP #266140; 1q23.1; 2429 aa; ~280 kDa; "
            "Alpha-spectrin most abundant RBC cytoskeletal protein; heterotetramer with beta-spectrin; "
            "Hereditary elliptocytosis (HE): typically mild AD (heterozygous); elliptocytes on smear; "
            "Hereditary pyropoikilocytosis (HPP): severe AR; compound heterozygous SPTA1 + αLELY; "
            "αLELY allele (alpha-LELY, low expression variant): common silent modifier; in TRANS with HE allele → HPP; "
            "HPP: severe neonatal haemolysis; heat-labile RBCs (lysis at 45°C vs 49°C normal); poikilocytes; "
            "HPP splenomegaly; splenectomy required in severe; often mistaken for hereditary ovalocytosis"
        ),
        "aa": "2429 aa",
        "kDa": "~280 kDa",
        "locus": "1q23.1",
        "omim_gene": 182860,
        "omim_disease": 130600,
        "inheritance": "HE: AD — heterozygous mild; HPP: AR-like — SPTA1 allele in trans with αLELY; de novo or compound",
        "gene_class": (
            "SPTA1 encodes alpha-spectrin, the most abundant protein in the erythrocyte membrane "
            "cytoskeleton. Alpha- and beta-spectrin dimerize head-to-head and then associate "
            "laterally to form tetramers, which cross-link into a hexagonal lattice beneath the "
            "lipid bilayer via junctional complexes containing actin, protein 4.1, and adducin. "
            "SPTA1 mutations in the alpha-spectrin repeat domains weaken horizontal spectrin "
            "tetramer self-association. In the heterozygous state most patients have enough "
            "remaining normal alpha-spectrin to produce mild asymptomatic elliptocytosis. However, "
            "when a pathogenic SPTA1 allele is inherited in trans with the common αLELY "
            "(alpha-Low Expression Lisbon Yokohama) allele — a non-pathogenic splice variant that "
            "reduces alpha-spectrin expression ~50% — the effective amount of normal alpha-spectrin "
            "falls to a level that produces severe fragmentation and pyropoikilocytosis (HPP). "
            "Thermal sensitivity (RBC lysis at 45°C rather than normal 49°C) is a classic "
            "diagnostic finding in HPP."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 3,
        "etiologies": [
            ("SPTA1 missense in alpha-II spectrin repeat domain — heterozygous HE (mild)", 0.45),
            ("SPTA1 + αLELY in trans — compound heterozygous HPP (severe neonatal)", 0.30),
            ("Homozygous SPTA1 (consanguineous) — severe HE/HPP phenotype", 0.15),
            ("SPTA1 splice variant + αLELY — HPP in trans", 0.10),
        ],
        "age_onset_years_range": (0, 5),
        "sex_ratio_M": 0.50,
        "rates": {
            "elliptocytes_on_blood_smear":          0.92,
            "poikilocytes_fragments_hpe":           0.40,
            "severe_neonatal_haemolysis_hpp":       0.30,
            "heat_labile_rbc_45_degrees":           0.35,
            "splenomegaly":                         0.55,
            "splenectomy_performed":                0.30,
            "gallstones":                           0.30,
            "aplastic_crisis_parvovirus":           0.12,
            "mild_asymptomatic_he":                 0.45,
            "alely_modifier_present":               0.35,
            "elevated_reticulocytes":               0.75,
            "transfusion_dependent_neonatal":       0.25,
            "folate_supplementation":               0.80,
            "normal_osmotic_fragility_mild_he":     0.55,
        },
        "hallmarks": [
            "αLELY modifier (in trans): common allele that converts mild HE into severe HPP — always test for αLELY",
            "HPP: heat-labile RBCs lyse at 45°C (normal 49°C) — diagnostic thermal sensitivity test",
            "Blood smear: elliptocytes ± fragmented poikilocytes ± microspherocytes depending on severity",
            "Neonatal HPP: severe haemolytic anaemia, fragmented smear, often transfusion-dependent in 1st year",
            "Mild HE heterozygotes: >25% elliptocytes on smear, minimal or no anaemia — often asymptomatic",
            "HPP and mild HE: same gene, different allele combinations — αLELY genotyping changes management",
        ],
        "treatment_alerts": [
            "αLELY GENOTYPING MANDATORY when SPTA1 variant identified — determines whether HPP or mild HE",
            "HPP neonatal: transfusion support in 1st 6–12 months; often improves after infancy",
            "Splenectomy: beneficial in symptomatic HPP (after age 2–3 and vaccines); reduces but not abolish haemolysis",
            "Folate 5 mg/day in all symptomatic patients",
            "Mild HE (heterozygous): often NO treatment needed — important to avoid unnecessary interventions",
            "Parvovirus B19 aplastic crisis: monitor; transfusion support",
        ],
        "primary_treatment": (
            "Mild HE (heterozygous): observation + folate 5 mg/day. No specific therapy needed for most. "
            "HPP (SPTA1 + αLELY compound): folate; transfusion neonatal period; splenectomy after age 2–3 "
            "if transfusion-dependent (post-splenectomy vaccines + penicillin prophylaxis). "
            "αLELY genotyping to stratify severity. Thermal stability test (45°C) confirms HPP. "
            "Aplastic crisis: parvovirus B19 serology; transfusion support."
        ),
    },

    # ── SLC4A1 — Hereditary Spherocytosis Type 4 + Southeast Asian Ovalocytosis ──
    {
        "gene": "SLC4A1",
        "protein": "Solute Carrier Family 4 Member 1 / Band 3 / Anion Exchanger 1 (Membrane Structural + Transport Protein)",
        "alias": (
            "SLC4A1 (Band 3 / AE1); OMIM gene 109270; HS4 #612653; SAO #166900; dRTA #179800; 17q21.31; 911 aa; ~102 kDa; "
            "Multifunctional: structural scaffold for RBC membrane + Cl-/HCO3- anion exchanger + CO2 transport; "
            "AD: Hereditary Spherocytosis type 4 (HS4) — haploinsufficiency or dominant-negative missense; "
            "AR: Southeast Asian Ovalocytosis (SAO) — homozygous Δ400–408 LETHAL; heterozygous asymptomatic malaria protection; "
            "Distal renal tubular acidosis (dRTA): SLC4A1 AD/AR variants in collecting duct alpha-intercalated cells; "
            "Band 3 also mediates CO2 transport from tissues to lung — Cl-/HCO3- exchange at red cell membrane; "
            "EMA flow cytometry: reduced fluorescence in HS4"
        ),
        "aa": "911 aa",
        "kDa": "~102 kDa",
        "locus": "17q21.31",
        "omim_gene": 109270,
        "omim_disease": 612653,
        "inheritance": "HS4: AD (haploinsufficiency or dominant-negative); SAO: heterozygous AD (malaria protection); dRTA: AD or AR",
        "gene_class": (
            "SLC4A1 (band 3 / AE1) is the most abundant integral membrane protein in red blood cells, "
            "serving dual structural and transport roles. Structurally, band 3 is the principal "
            "attachment point for the membrane skeleton (via ankyrin-1 and protein 4.2), anchoring "
            "the spectrin-actin cytoskeletal network to the lipid bilayer. Functionally, band 3 "
            "mediates Cl⁻/HCO₃⁻ exchange across the red cell membrane at high throughput, enabling "
            "efficient CO₂ transport from peripheral tissues to the lung. In HS4, heterozygous "
            "haploinsufficiency or dominant-negative band 3 variants reduce membrane skeleton "
            "anchoring, causing spherocyte formation and extravascular haemolysis. Southeast Asian "
            "ovalocytosis (SAO) arises from the Δ400–408 deletion, which rigidifies the band 3 "
            "transmembrane domain, creating ovalocyte-shaped RBCs that resist Plasmodium falciparum "
            "invasion — homozygous SAO is embryonic lethal. The same gene harbours variants causing "
            "distal RTA in the kidney collecting duct."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 4,
        "etiologies": [
            ("SLC4A1 haploinsufficiency — AD HS4 spherocytosis", 0.45),
            ("SLC4A1 Δ400–408 heterozygous — Southeast Asian ovalocytosis (SAO, malaria-endemic region)", 0.30),
            ("SLC4A1 missense — dominant-negative HS4 or dRTA", 0.20),
            ("SLC4A1 AR — dRTA with haemolytic anaemia (compound heterozygous)", 0.05),
        ],
        "age_onset_years_range": (0, 20),
        "sex_ratio_M": 0.50,
        "rates": {
            "ovalocytes_on_blood_smear_sao":        0.30,
            "spherocytes_on_blood_smear_hs4":       0.45,
            "haemolytic_anaemia":                   0.55,
            "splenomegaly":                         0.45,
            "reduced_ema_flow_cytometry_hs4":       0.75,
            "normal_ema_sao":                       0.30,
            "distal_rta":                           0.15,
            "hypokalaemia_drt":                     0.12,
            "nephrocalcinosis_drt":                 0.10,
            "malaria_endemic_region":               0.35,
            "gallstones":                           0.30,
            "southeast_asian_ancestry":             0.30,
            "neonatal_jaundice":                    0.40,
            "positive_osmotic_fragility_hs4":       0.70,
        },
        "hallmarks": [
            "SAO (Δ400–408): ovalocytes, NOT spherocytes; EMA NORMAL (structurally different from HS); no significant haemolysis in heterozygotes",
            "HS4 EMA flow cytometry: reduced — same pattern as ANK1/SPTB; cannot distinguish by EMA alone",
            "dRTA: hyperchloraemic non-anion gap metabolic acidosis + urinary pH >5.5 unable to acidify → nephrocalcinosis",
            "SAO homozygous: LETHAL — if consanguineous + SAO allele, always check partner in reproductive counselling",
            "Malaria resistance: SAO enriched in P. falciparum-endemic regions (PNG, Malaysia, Thailand) — balanced polymorphism",
            "SLC4A1 is both structural (membrane skeleton) AND transport (Cl-/HCO3-) — dRTA and haemolysis from same gene",
        ],
        "treatment_alerts": [
            "dRTA: oral sodium bicarbonate or citrate supplementation — prevent nephrocalcinosis and renal stones",
            "SAO: NO splenectomy benefit (no significant haemolysis); avoid unnecessary intervention",
            "HS4: same splenectomy approach as ANK1-HS — post-splenectomy vaccines + penicillin",
            "dRTA with hypokalaemia: correct potassium — if hypokaemic, alkalinising therapy may worsen K+ first",
            "Folate supplementation in HS4 with ongoing haemolysis",
            "Reproductive counselling: SAO + SAO = 25% homozygous (lethal) — chorionic villus sampling / PGT",
        ],
        "primary_treatment": (
            "HS4: folate 5 mg/day; splenectomy if symptomatic (vaccines + penicillin prophylaxis). "
            "SAO: no haematological treatment; patient education on malaria protection balanced polymorphism. "
            "dRTA: oral sodium bicarbonate or potassium citrate to maintain serum bicarbonate >20 mmol/L; "
            "monitor renal function + USS for nephrocalcinosis annually. "
            "Reproductive: SAO + SAO partner → PGT/CVS (homozygous lethal). "
            "EMA flow cytometry for HS4; plain film/USS for SAO ovalocytes."
        ),
    },

    # ── EPB42 — Hereditary Spherocytosis Type 5 / Protein 4.2 Deficiency ──
    {
        "gene": "EPB42",
        "protein": "Erythrocyte Membrane Protein Band 4.2 / Protein 4.2 (Membrane Skeleton Stabiliser)",
        "alias": (
            "EPB42; OMIM gene 177070; Hereditary spherocytosis type 5 #612690; 15q15.2; 691 aa; ~72 kDa; "
            "Protein 4.2 stabilises the band 3–ankyrin interaction at the junctional complex; "
            "EPB42 LOF → reduced protein 4.2 → band 3 clustering instability → spherocyte formation; "
            "Japanese founder allele: p.Ala142Thr (c.424G>A) — most common HS variant in Japanese populations; "
            "AR inheritance (unlike most HS which is AD) — both parents carriers; 25% recurrence; "
            "Mild to moderate haemolytic anaemia; EMA flow cytometry: reduced fluorescence; "
            "Responds to splenectomy; gallstones common; aplastic crisis from parvovirus B19"
        ),
        "aa": "691 aa",
        "kDa": "~72 kDa",
        "locus": "15q15.2",
        "omim_gene": 177070,
        "omim_disease": 612690,
        "inheritance": "AR — homozygous or compound heterozygous; Japanese founder p.Ala142Thr common in East Asian",
        "gene_class": (
            "EPB42 encodes protein 4.2, a peripheral membrane protein that stabilises the ankyrin–"
            "band 3 junction within the erythrocyte membrane skeleton. Protein 4.2 wraps around "
            "the cytoplasmic domain of band 3, acting as a molecular chaperone that prevents "
            "premature band 3 clustering and membrane vesiculation. EPB42 loss-of-function "
            "disrupts the stability of the principal membrane skeletal attachment site, reducing "
            "membrane deformability and promoting spherocyte formation through the same pathway "
            "as ANK1 and SLC4A1 variants. Unlike most other HS genes (which are AD), EPB42-HS "
            "is autosomal recessive, making it clinically important that both parents are identified "
            "as carriers for recurrence counselling. The Japanese founder p.Ala142Thr allele is "
            "particularly common in East Asia, and targeted testing of this variant should be "
            "performed first in patients of East Asian ancestry before full gene sequencing."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 5,
        "etiologies": [
            ("p.Ala142Thr (c.424G>A) Japanese founder — homozygous (East Asian ancestry)", 0.45),
            ("p.Ala142Thr compound heterozygous with other EPB42 variant (East Asian)", 0.25),
            ("Novel compound heterozygous — non-East Asian", 0.20),
            ("Homozygous novel missense — consanguineous family", 0.10),
        ],
        "age_onset_years_range": (0, 10),
        "sex_ratio_M": 0.50,
        "rates": {
            "spherocytes_on_blood_smear":           0.88,
            "chronic_haemolytic_anaemia":           0.90,
            "splenomegaly":                         0.80,
            "gallstones":                           0.50,
            "neonatal_jaundice":                    0.60,
            "reduced_ema_flow_cytometry":           0.90,
            "splenectomy_performed":                0.45,
            "aplastic_crisis_parvovirus":           0.12,
            "east_asian_ancestry":                  0.55,
            "consanguinity_non_east_asian":         0.20,
            "elevated_reticulocytes":               0.88,
            "mild_moderate_haemolysis":             0.70,
            "folate_supplementation":               0.85,
            "parents_obligate_carriers_ar":         0.90,
        },
        "hallmarks": [
            "AR INHERITANCE: both parents are obligate carriers — always test parents and siblings; 25% recurrence",
            "Japanese founder p.Ala142Thr: East Asian ancestry → targeted Sanger FIRST before full EPB42 sequencing",
            "EMA flow cytometry: reduced — same pattern as other HS; cannot distinguish EPB42 from ANK1/SLC4A1 by EMA alone",
            "Phenotype: mild to moderate haemolysis — splenomegaly, gallstones, elevated reticulocytes",
            "Splenectomy: effective and curative for haemolysis; defer to age >5 if possible",
            "Protein 4.2 is absent on gel electrophoresis — band 4.2 loss on SDS-PAGE of RBC ghost is diagnostic",
        ],
        "treatment_alerts": [
            "Folate 5 mg/day in all symptomatic patients (elevated erythropoietic demand)",
            "Splenectomy: effective; post-splenectomy vaccines (Pneumococcal, MenACWY, HiB, flu) ≥2 weeks prior",
            "Post-splenectomy: penicillin V 250 mg BD (lifelong)",
            "Parvovirus B19 aplastic crisis: transfusion support; usually 10–14 days duration",
            "Gallstones: USS annually from age 5; cholecystectomy concurrent with splenectomy",
            "AR genetics: sibling testing; prenatal diagnosis / PGT available",
        ],
        "primary_treatment": (
            "Folate 5 mg/day. EMA flow cytometry for diagnosis. SDS-PAGE RBC ghost: absent band 4.2 band confirms. "
            "Splenectomy in symptomatic moderate–severe HS5 (vaccines + penicillin prophylaxis). "
            "East Asian ancestry: targeted p.Ala142Thr testing first. "
            "AR genetics: both parents obligate carriers — family cascade testing. "
            "Gallstones: concurrent cholecystectomy at splenectomy if present. "
            "Parvovirus B19: annual monitoring; transfusion during aplastic crisis."
        ),
    },

    # ── HK1 — Hexokinase Deficiency ──
    {
        "gene": "HK1",
        "protein": "Hexokinase-1 (First Committed Glycolytic Enzyme — Glucose Phosphorylation)",
        "alias": (
            "HK1; OMIM gene 142600; Hexokinase deficiency #235700; 10q22.1; 917 aa; ~102 kDa; "
            "Rare AR non-spherocytic haemolytic anaemia — HK1 is sole hexokinase isoform in RBCs; "
            "Hexokinase catalyses first committed step of glycolysis: glucose + ATP → glucose-6-phosphate; "
            "HK1 LOF → glucose-6-phosphate depletion → ATP deficit → glycolysis failure → haemolysis; "
            "EMA flow cytometry NORMAL — distinguishes from spherocytosis (no membrane defect); "
            "Osmotic fragility NORMAL — reinforces non-spherocytic mechanism; "
            "Hexokinase activity assay in RBC lysate required for diagnosis (markedly reduced); "
            "Severe neonatal haemolytic anaemia with high transfusion requirement; "
            "No approved disease-modifying therapy — splenectomy beneficial but not curative"
        ),
        "aa": "917 aa",
        "kDa": "~102 kDa",
        "locus": "10q22.1",
        "omim_gene": 142600,
        "omim_disease": 235700,
        "inheritance": "AR — compound heterozygous (most common) or homozygous; both parents obligate carriers",
        "gene_class": (
            "HK1 encodes hexokinase-1, the sole hexokinase isoform expressed in human erythrocytes. "
            "Hexokinase catalyses the ATP-dependent phosphorylation of glucose to glucose-6-phosphate "
            "— the first and rate-limiting committed step of glycolysis. Because RBCs lack "
            "mitochondria and depend exclusively on glycolysis for ATP production, HK1 deficiency "
            "creates a profound cellular ATP deficit that impairs Na+/K+-ATPase activity, reduces "
            "membrane deformability, and ultimately leads to extravascular haemolytic destruction. "
            "The membrane skeleton is structurally intact (EMA and osmotic fragility are normal), "
            "distinguishing hexokinase deficiency from spherocytosis at the diagnostic level. "
            "Enzyme assay in red cell lysate — corrected for reticulocyte count — is required "
            "to establish the diagnosis. Reticulocytes have higher HK activity than mature red "
            "cells, necessitating correction in the assay interpretation."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 6,
        "etiologies": [
            ("Compound heterozygous HK1 missense — most common presentation", 0.55),
            ("Homozygous HK1 missense — consanguineous family", 0.25),
            ("Splice site + missense compound heterozygous", 0.15),
            ("HK1 frameshift + missense compound heterozygous", 0.05),
        ],
        "age_onset_years_range": (0, 3),
        "sex_ratio_M": 0.50,
        "rates": {
            "neonatal_haemolysis":                  0.80,
            "severe_transfusion_dependent_neonatal": 0.65,
            "chronic_haemolytic_anaemia":           0.90,
            "splenomegaly":                         0.75,
            "normal_ema_flow_cytometry":            0.92,
            "normal_osmotic_fragility":             0.90,
            "reduced_hk_enzyme_activity":           0.97,
            "gallstones":                           0.40,
            "aplastic_crisis_parvovirus":           0.15,
            "splenectomy_performed":                0.40,
            "iron_overload_elevated_ferritin":      0.45,
            "folate_supplementation":               0.88,
            "elevated_reticulocytes_marked":        0.92,
            "consanguinity":                        0.30,
        },
        "hallmarks": [
            "EMA NORMAL: no membrane defect — key DDx from spherocytosis; EMA is normal in HK1 deficiency",
            "Osmotic fragility NORMAL: reinforces glycolytic (not membrane) aetiology",
            "HK1 enzyme assay in RBC lysate: markedly reduced activity — must correct for reticulocyte count",
            "Reticulocytes have HIGH HK1 activity: acute haemolysis → high reticulocyte count → false-normal assay risk",
            "Severe neonatal haemolysis: transfusion-dependent from birth; high clinical vigilance",
            "Splenectomy: beneficial but NOT curative — removes main destruction site but haemolysis persists",
        ],
        "treatment_alerts": [
            "Folate 5 mg/day — all patients (increased erythropoietic demand)",
            "Transfusion: aggressive support in neonatal period; Hb threshold guided by clinical tolerance",
            "Iron overload: monitor ferritin annually; chelation if transferrin saturation >45%",
            "Splenectomy: partial or total; reduces but does not eliminate haemolysis; post-splenectomy vaccines",
            "HK1 enzyme assay: correct for reticulocyte count — do NOT interpret raw activity without correction",
            "Parvovirus B19: most severe acute complication in non-immune patients; transfusion + IVIG consideration",
        ],
        "primary_treatment": (
            "Folate 5 mg/day. Transfusion support (Hb-guided; frequent in neonatal period). "
            "Iron monitoring: ferritin + transferrin saturation annually; deferasirox chelation if overloaded. "
            "Splenectomy in moderate–severe cases (reduces transfusion burden; post-splenectomy vaccines + penicillin). "
            "HK1 enzyme assay for diagnosis (reticulocyte-corrected). "
            "Aplastic crisis: transfusion; parvovirus B19 serology + IVIG if needed. "
            "Gene therapy trials emerging — refer eligible patients to specialist centre."
        ),
    },

    # ── PIEZO1 — Dehydrated Hereditary Stomatocytosis (Xerocytosis) ──
    {
        "gene": "PIEZO1",
        "protein": "Piezo-Type Mechanosensitive Ion Channel Component 1 (Mechanosensory Channel — RBC Volume Regulation)",
        "alias": (
            "PIEZO1; OMIM gene 611184; Dehydrated hereditary stomatocytosis / Xerocytosis #194380; 16q24.3; 2521 aa; ~286 kDa; "
            "PIEZO1 gain-of-function → constitutive Ca2+ influx → K+ efflux → cellular dehydration → RBC stomatocytes; "
            "MCHC markedly elevated; pseudohyperkalaemia (in vitro K+ release from dehydrated cells) — K+ NORMAL in vivo; "
            "African variant: p.Glu756del (common in sub-Saharan Africa) — erythrocytosis phenotype; "
            "SPLENECTOMY ABSOLUTELY CONTRAINDICATED — life-threatening thromboembolic events post-splenectomy; "
            "Perinatal oedema (lymphoedema, ascites, hydrops fetalis) at birth — RESOLVES spontaneously; "
            "Serum ferritin elevated in many patients — iron absorption increased from relative haemolysis"
        ),
        "aa": "2521 aa",
        "kDa": "~286 kDa",
        "locus": "16q24.3",
        "omim_gene": 611184,
        "omim_disease": 194380,
        "inheritance": "AD — gain-of-function; de novo mutations frequent; haploinsufficiency does NOT cause disease (GOF mechanism only)",
        "gene_class": (
            "PIEZO1 encodes a mechanosensitive cation channel that senses membrane tension and "
            "responds by gating Ca²⁺ and other cations. In erythrocytes, PIEZO1 is activated by "
            "mechanical deformation during capillary transit and regulates cell volume through "
            "calcium-activated potassium efflux (the Gardos pathway). Gain-of-function mutations "
            "extend the channel's open probability, leading to excessive and constitutive Ca²⁺ "
            "influx, disproportionate K⁺ and water efflux, and erythrocyte dehydration — producing "
            "the dehydrated stomatocyte (xerocyte). The resulting cells are dense (markedly elevated "
            "MCHC), rigid, and prone to haemolysis. A critical clinical feature is that splenectomy "
            "— which is effective in most hereditary haemolytic anaemias — is absolutely "
            "contraindicated in DHS/xerocytosis because it paradoxically precipitates life-threatening "
            "venous thromboembolic events, including portal vein thrombosis. The mechanism of "
            "this paradoxical thrombosis is not fully established but may relate to activation of "
            "dense, procoagulant microparticles. The common African variant p.Glu756del causes "
            "a milder erythrocytosis rather than haemolytic anaemia."
        ),
        "n_patients": 40,
        "seed": SEED_BASE + 7,
        "etiologies": [
            ("PIEZO1 gain-of-function missense — inherited from parent (AD)", 0.55),
            ("De novo PIEZO1 GOF — no family history", 0.25),
            ("p.Glu756del African variant — erythrocytosis phenotype, sub-Saharan African ancestry", 0.15),
            ("Novel GOF variant (in silico gain-of-function predicted)", 0.05),
        ],
        "age_onset_years_range": (0, 20),
        "sex_ratio_M": 0.50,
        "rates": {
            "stomatocytes_on_blood_smear":          0.75,
            "elevated_mchc_dense_rbc":              0.88,
            "pseudohyperkalaemia_in_vitro":         0.65,
            "perinatal_oedema_resolving":           0.35,
            "mild_to_moderate_haemolysis":          0.80,
            "splenomegaly":                         0.55,
            "elevated_ferritin_iron_absorption":    0.50,
            "family_history_stomatocytosis":        0.65,
            "thromboembolism_post_splenectomy":     0.30,
            "african_ancestry_erythrocytosis":      0.15,
            "neonatal_jaundice":                    0.45,
            "elevated_reticulocytes_mild":          0.75,
            "normal_ema_flow_cytometry":            0.70,
            "normal_osmotic_fragility":             0.65,
        },
        "hallmarks": [
            "SPLENECTOMY ABSOLUTELY CONTRAINDICATED — thromboembolic events (portal vein thrombosis) post-splenectomy",
            "Pseudohyperkalaemia: in vitro K+ release from dense cells at room temperature — K+ NORMAL in vivo (37°C EDTA)",
            "MCHC markedly elevated — dehydrated dense RBCs; high MCHC in the absence of spherocytosis is the clue",
            "Perinatal oedema (hydrops, lymphoedema, ascites): self-resolving by 6–12 months; do NOT confuse with haemolytic hydrops",
            "Stomatocytes on blood smear: central slit instead of round pale area; fragile preparation artefact — use fresh wet prep",
            "African variant p.Glu756del: common in malaria-endemic Africa; erythrocytosis (high RBC count, Hb) not haemolysis",
        ],
        "treatment_alerts": [
            "SPLENECTOMY ABSOLUTELY CONTRAINDICATED — fatal thromboembolic risk; document explicitly in medical record",
            "Pseudohyperkalaemia: always draw K+ in heparinised sample at 37°C; never base treatment on EDTA K+",
            "Anticoagulation: therapeutic LMWH or warfarin during pregnancy; consider thromboprophylaxis perioperatively",
            "Iron overload: ferritin annually; hydroxycarbamide (hydroxyurea) reduces RBC rigidity in some (off-label)",
            "Folate supplementation: 5 mg/day in all patients with ongoing haemolysis",
            "Perinatal oedema: reassure parents — self-resolving; no neonatal intervention beyond supportive care",
        ],
        "primary_treatment": (
            "Folate 5 mg/day. Avoid splenectomy absolutely — document contraindication prominently. "
            "Pseudohyperkalaemia: confirm K+ in heparinised sample at 37°C before any treatment. "
            "Perinatal oedema: supportive care; spontaneous resolution expected by 6–12 months. "
            "Iron monitoring: ferritin annually; chelation if overloaded. "
            "Anticoagulation: thromboprophylaxis during high-risk periods (pregnancy, surgery, immobility). "
            "African p.Glu756del: monitor Hb/haematocrit; erythrocytosis — phlebotomy if symptomatic polycythaemia. "
            "Transfusion only for severe acute haemolysis or aplastic crisis."
        ),
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Patient Simulation
# ─────────────────────────────────────────────────────────────────────────────
def _simulate_patients(gene_def: dict) -> list:
    rng = random.Random(gene_def["seed"])
    patients = []
    ages = list(range(gene_def["age_onset_years_range"][0], gene_def["age_onset_years_range"][1] + 1))
    n = gene_def["n_patients"]

    for i in range(n):
        age_onset = rng.choice(ages)

        r = rng.random()
        cum = 0.0
        etiology = gene_def["etiologies"][-1][0]
        for label, frac in gene_def["etiologies"]:
            cum += frac
            if r < cum:
                etiology = label
                break

        features = {}
        for feat, rate in gene_def["rates"].items():
            features[feat] = rng.random() < rate

        sex = "M" if rng.random() < gene_def["sex_ratio_M"] else "F"

        patients.append({
            "id": i + 1,
            "gene": gene_def["gene"],
            "age_onset": age_onset,
            "sex": sex,
            "etiology": etiology,
            "features": features,
        })
    return patients


def _aggregate_stats(patients: list, rates: dict) -> dict:
    if not patients:
        return {}
    n = len(patients)
    return {k: round(sum(p["features"].get(k, False) for p in patients) / n * 100, 1) for k in rates}


# ─────────────────────────────────────────────────────────────────────────────
# Build all cohorts once
# ─────────────────────────────────────────────────────────────────────────────
_ALL_PATIENTS: dict = {}
_ALL_STATS: dict = {}

for _gd in RCD_GENES:
    _pts = _simulate_patients(_gd)
    _ALL_PATIENTS[_gd["gene"]] = _pts
    _ALL_STATS[_gd["gene"]] = _aggregate_stats(_pts, _gd["rates"])


# ─────────────────────────────────────────────────────────────────────────────
# API Data Functions
# ─────────────────────────────────────────────────────────────────────────────
def get_overview() -> dict:
    """Overview — aggregate stats across all 320 patients."""
    all_pts = [p for pts in _ALL_PATIENTS.values() for p in pts]
    n = len(all_pts)

    def _pct(key: str) -> float:
        return round(sum(p["features"].get(key, False) for p in all_pts) / n * 100, 1)

    genes = [g["gene"] for g in RCD_GENES]

    top_alerts = [
        "G6PD-RASBURICASE-ABSOLUTE-CI: fatal oxidative haemolysis — check G6PD before administering rasburicase",
        "G6PD-PRIMAQUINE-DAPSONE-CI: avoid oxidant anti-malarials in G6PD deficiency — use chloroquine",
        "G6PD-TEST-TIMING: assay 3 months AFTER acute episode — reticulocytes give false-normal during crisis",
        "PKLR-MITAPIVAT: FDA 2022 (Pyrukynd) — first disease-modifying therapy for PK deficiency adults",
        "PKLR-2,3-BPG-PARADOX: markedly elevated 2,3-BPG → right-shifted O2 curve → tolerate lower Hb than expected",
        "PKLR-IRON-OVERLOAD-WITHOUT-TRANSFUSION: ineffective erythropoiesis → monitor ferritin annually",
        "ANK1-EMA-FLOW-CYTOMETRY: replace osmotic fragility — higher sensitivity/specificity for HS",
        "PIEZO1-SPLENECTOMY-ABSOLUTE-CI: life-threatening portal vein thrombosis post-splenectomy",
        "PIEZO1-PSEUDOHYPERKALAEMIA: K+ NORMAL in vivo — confirm in heparinised 37°C sample before treating",
        "SPTA1-ALELY-MANDATORY: αLELY allele in trans converts mild HE → severe HPP — always genotype αLELY",
        "EPB42-AR-BOTH-PARENTS: unlike most HS (AD), EPB42-HS is AR — test parents and siblings; 25% recurrence",
        "HK1-EMA-NORMAL: non-spherocytic HA — EMA flow cytometry is NORMAL (no membrane defect in HK1 deficiency)",
    ]

    diseases = {}
    for g in RCD_GENES:
        alias_parts = g["alias"].split(";")
        diseases[g["gene"]] = alias_parts[3].strip() + " — " + alias_parts[4].strip() if len(alias_parts) > 4 else g["alias"][:120]

    return {
        "total_patients": n,
        "genes": genes,
        "seed_range": "1422–1429",
        "aggregate_stats": {
            "haemolytic_anaemia_any":       round(sum(
                any(p["features"].get(k, False) for k in [
                    "chronic_haemolytic_anaemia", "episodic_haemolytic_crisis",
                    "neonatal_haemolysis", "mild_to_moderate_haemolysis",
                ]) for p in all_pts) / n * 100, 1),
            "splenomegaly_any":             _pct("splenomegaly"),
            "gallstones":                   _pct("gallstones"),
            "neonatal_jaundice_any":        round(sum(
                any(p["features"].get(k, False) for k in [
                    "neonatal_jaundice", "neonatal_haemolysis",
                ]) for p in all_pts) / n * 100, 1),
            "splenectomy_performed":        _pct("splenectomy_performed"),
            "elevated_reticulocytes":       round(sum(
                any(p["features"].get(k, False) for k in [
                    "elevated_reticulocytes", "elevated_reticulocytes_marked", "elevated_reticulocytes_mild",
                ]) for p in all_pts) / n * 100, 1),
            "aplastic_crisis_parvovirus":   round(sum(
                any(p["features"].get(k, False) for k in [
                    "aplastic_crisis_parvovirus", "aplastic_crisis_parvovirus_b19",
                ]) for p in all_pts) / n * 100, 1),
            "iron_overload_elevated_ferritin": _pct("iron_overload_elevated_ferritin"),
            "folate_supplementation":       _pct("folate_supplementation"),
            "g6pd_haemoglobinuria":         _pct("haemoglobinuria_cola_urine"),
            "piezo1_pseudohyperkalaemia":   _pct("pseudohyperkalaemia_in_vitro"),
            "piezo1_elevated_mchc":         _pct("elevated_mchc_dense_rbc"),
        },
        "top_alerts": top_alerts,
        "diseases": diseases,
    }


def get_breakdown() -> dict:
    """Per-gene breakdown for Gene Table and Clinical Atlas tabs."""
    result = {}
    for gd in RCD_GENES:
        gene = gd["gene"]
        pts = _ALL_PATIENTS[gene]
        stats = _ALL_STATS[gene]

        etiology_distribution = [
            {"etiology": label, "fraction": round(frac, 3)}
            for label, frac in gd["etiologies"]
        ]

        result[gene] = {
            "gene":                 gene,
            "protein":              gd["protein"],
            "aa":                   gd["aa"],
            "locus":                gd["locus"],
            "omim_gene":            gd["omim_gene"],
            "omim_disease":         gd["omim_disease"],
            "inheritance":          gd["inheritance"],
            "organ_system":         "Red blood cells / Haematology / Glycolysis / Membrane skeleton",
            "n_patients":           gd["n_patients"],
            "seed":                 gd["seed"],
            "gene_class":           gd["gene_class"],
            "hallmarks":            gd["hallmarks"],
            "treatment_alerts":     gd["treatment_alerts"],
            "primary_treatment":    gd["primary_treatment"],
            "stats":                stats,
            "etiology_distribution": etiology_distribution,
        }
    return result


def get_definitions() -> dict:
    """Disease classification, diagnostic rules, and treatment hierarchies."""
    return {
        "classification": {
            "glycolytic_enzyme_deficiencies": {
                "G6PD_deficiency": "X-linked — pentose phosphate pathway; episodic oxidative haemolysis; >400M affected worldwide",
                "PKLR_PK_deficiency": "AR — final glycolytic step; chronic non-spherocytic HA; 2,3-BPG paradox; mitapivat FDA 2022",
                "HK1_hexokinase_deficiency": "AR — first glycolytic step; severe neonatal non-spherocytic HA; EMA/osmotic fragility normal",
            },
            "red_cell_membrane_disorders": {
                "ANK1_HS1": "AD — ankyrin haploinsufficiency; most common HS (40–65%); EMA reduced; splenectomy curative",
                "SPTA1_HE_HPP": "AD/AR — alpha-spectrin; HE mild heterozygous; HPP severe with αLELY allele in trans",
                "SLC4A1_HS4_SAO_dRTA": "AD/AR — band 3; HS4 (AD) + SAO (Δ400–408 malaria protection) + distal RTA",
                "EPB42_HS5": "AR — protein 4.2; spherocytosis; Japanese founder p.Ala142Thr; unlike most HS (AR not AD)",
            },
            "ion_channel_rbc_disorders": {
                "PIEZO1_xerocytosis": "AD GOF — mechanosensory channel; dehydrated RBCs; splenectomy CONTRAINDICATED; thrombosis risk",
            },
        },
        "key_diagnostic_rules": {
            "G6PD_TEST_TIMING": (
                "G6PD enzyme activity must be measured 3 months AFTER an acute haemolytic episode. "
                "During acute crisis, reticulocytes (which have higher G6PD activity than mature RBCs) "
                "are markedly elevated and can produce a false-normal G6PD activity result. "
                "Test at steady state only."
            ),
            "G6PD_RASBURICASE_CI": (
                "Rasburicase generates hydrogen peroxide as a reaction product, which overwhelms the "
                "antioxidant capacity of G6PD-deficient red cells, causing catastrophic acute intravascular "
                "haemolysis. Check G6PD status before EVERY rasburicase administration. "
                "Primaquine and dapsone are similarly contraindicated in G6PD Class II/III."
            ),
            "PKLR_23_BPG_PARADOX": (
                "In PK deficiency, 2,3-BPG accumulates upstream of the deficient PK step (phosphoenolpyruvate "
                "accumulates → 2,3-BPG rises markedly). Elevated 2,3-BPG shifts the O2-haemoglobin "
                "dissociation curve rightward, facilitating oxygen unloading at tissues. Consequently, "
                "patients tolerate lower haemoglobin concentrations than predicted — do NOT transfuse "
                "based on Hb alone; assess symptoms and cardiovascular tolerance."
            ),
            "PIEZO1_SPLENECTOMY_CI": (
                "Dehydrated hereditary stomatocytosis (DHS/xerocytosis) caused by PIEZO1 GOF is unique "
                "among haemolytic anaemias in that splenectomy is ABSOLUTELY CONTRAINDICATED. "
                "Post-splenectomy patients with DHS develop life-threatening thromboembolic events including "
                "portal vein thrombosis and pulmonary embolism. DOCUMENT THIS CONTRAINDICATION PROMINENTLY "
                "in every patient's medical record. Never confuse with spherocytosis — always check PIEZO1 "
                "before recommending splenectomy for a stomatocytic haemolytic anaemia."
            ),
            "SPTA1_ALELY_MODIFIER": (
                "The αLELY allele (alpha-Low Expression variant) is a common non-pathogenic SPTA1 splice "
                "variant that reduces alpha-spectrin synthesis by ~50%. When αLELY is inherited in TRANS "
                "with a pathogenic SPTA1 HE allele, the effective normal alpha-spectrin level falls "
                "sufficiently to produce severe hereditary pyropoikilocytosis (HPP) rather than mild HE. "
                "Always genotype for αLELY when SPTA1 pathogenic variant is identified — it changes both "
                "prognosis (mild vs severe) and management (observation vs splenectomy)."
            ),
            "HK1_EMA_NORMAL": (
                "Hexokinase deficiency is a NON-SPHEROCYTIC haemolytic anaemia — the red cell membrane "
                "skeleton is structurally intact. EMA flow cytometry is NORMAL (distinguishes from all "
                "hereditary spherocytosis genes). Osmotic fragility is also normal. The diagnostic test "
                "is hexokinase enzyme activity in red cell lysate, corrected for reticulocyte count. "
                "Never exclude HK1 deficiency based on a normal EMA."
            ),
            "EPB42_AR_GENETICS": (
                "Unlike most hereditary spherocytosis (ANK1, SPTB — AD), EPB42-spherocytosis is AUTOSOMAL "
                "RECESSIVE. Both parents are obligate carriers and are clinically unaffected (heterozygous). "
                "The 25% sibling recurrence risk must be communicated. The Japanese founder p.Ala142Thr "
                "should be tested first in East Asian patients before full EPB42 sequencing."
            ),
            "SLC4A1_SAO_HOMOZYGOUS_LETHAL": (
                "Southeast Asian ovalocytosis (Δ400–408 SLC4A1) is LETHAL in the homozygous state — "
                "no liveborn SAO/SAO homozygotes survive. When a patient in a malaria-endemic region "
                "carries the SAO allele and their partner also carries SAO, reproductive counselling "
                "about 25% lethal homozygosity risk is MANDATORY. PGT or CVS is available."
            ),
        },
        "treatment_hierarchy": {
            "G6PD_deficiency": [
                "1. ABSOLUTE: never give rasburicase; check G6PD before primaquine/dapsone",
                "2. Avoid fava beans (Mediterranean G6PD), infections, oxidant drugs",
                "3. Acute crisis: remove precipitant; IV hydration; transfuse if Hb <7 g/dL",
                "4. Neonatal: phototherapy ± exchange transfusion (bilirubin-guided)",
                "5. Class I CNSHA: folate 5 mg/day; splenectomy in selected cases",
                "6. Medical alert bracelet; patient education on trigger avoidance",
            ],
            "PKLR_PK_deficiency": [
                "1. Folate 5 mg/day",
                "2. Mitapivat (Pyrukynd) 50 mg BD — FDA 2022; eligible adults with documented PK deficiency",
                "3. Transfusion: Hb-guided (account for 2,3-BPG tolerance — do not over-transfuse)",
                "4. Iron monitoring: ferritin annually; chelation (deferasirox) if Fe overloaded",
                "5. Splenectomy: selected severe transfusion-dependent; vaccines + penicillin",
                "6. Parvovirus B19 aplastic crisis: transfusion support",
            ],
            "ANK1_HS1_splenectomy": [
                "1. Folate 5 mg/day",
                "2. EMA flow cytometry for diagnosis (replace osmotic fragility)",
                "3. Splenectomy after age 5 in moderate–severe HS (vaccines ≥2 weeks prior)",
                "4. Post-splenectomy: penicillin V 250 mg BD lifelong",
                "5. Concurrent cholecystectomy if gallstones at splenectomy",
                "6. Parvovirus B19: transfusion support if aplastic crisis",
            ],
            "PIEZO1_xerocytosis": [
                "1. ABSOLUTE: NEVER perform splenectomy — document in record",
                "2. Confirm K+ in heparinised 37°C sample (pseudohyperkalaemia is in vitro artefact)",
                "3. Folate 5 mg/day",
                "4. Anticoagulation: thromboprophylaxis during pregnancy, surgery, immobility",
                "5. Iron monitoring: ferritin annually",
                "6. Perinatal oedema: reassure — self-resolves by 6–12 months",
            ],
            "SPTA1_HPP": [
                "1. αLELY genotyping FIRST (determines HE vs HPP)",
                "2. HPP: transfusion support neonatal period",
                "3. Splenectomy in HPP after age 2–3 if transfusion-dependent",
                "4. Folate 5 mg/day",
                "5. Mild HE (heterozygous only): no intervention needed in most",
            ],
            "EPB42_HS5": [
                "1. AR genetics: test parents + siblings; 25% recurrence counselling",
                "2. East Asian: targeted p.Ala142Thr testing first",
                "3. SDS-PAGE RBC ghost: absent band 4.2 band",
                "4. Folate 5 mg/day",
                "5. Splenectomy in moderate–severe (vaccines + penicillin)",
            ],
            "HK1_hexokinase": [
                "1. Folate 5 mg/day",
                "2. Transfusion: aggressive neonatal support; ongoing guided by symptoms",
                "3. Iron monitoring: ferritin + transferrin saturation annually; chelation PRN",
                "4. Splenectomy: reduces but does not eliminate haemolysis",
                "5. Enzyme assay for diagnosis — reticulocyte-corrected",
                "6. Gene therapy trial referral for eligible patients",
            ],
            "SLC4A1_HS4_SAO_dRTA": [
                "1. HS4: splenectomy as per other HS (vaccines + penicillin)",
                "2. SAO: no haematological treatment — malaria protection awareness",
                "3. dRTA: sodium bicarbonate or potassium citrate to maintain HCO3 >20 mmol/L",
                "4. dRTA: monitor renal function + renal USS for nephrocalcinosis",
                "5. SAO + SAO partner: reproductive counselling — PGT/CVS (homozygous lethal)",
            ],
        },
    }
