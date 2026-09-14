"""Hereditary Hemolytic Anemia Atlas — 8-Gene Reference
HBB-HBA1-G6PD-PKLR-ANK1-SLC4A1-SPTA1-KCNN4
320 patients (8 x 40), seeds 2582-2589.
Endpoints: /api/hereditary-hemolytic-anemia-atlas/overview|breakdown|definitions
"""
import random

ATLAS_GENES = [
    {
        "gene": "HBB",
        "alt_name": (
            "HBB-147aa-11p15.4-AR-CODOMINANT-SICKLE-CELL-DISEASE-BETA-THALASSEMIA-MAJOR-"
            "HbS-pGlu6Val-HYDROXYUREA-FIRST-LINE-CASGEVY-CRISPR-FDA-2023-NBS-MANDATORY"
        ),
        "protein": (
            "HBB -- 11p15.4 AR/codominant -- 147aa -- Beta-Globin-16kDa-Haemoglobin-Tetramer-"
            "Alpha2Beta2-Oxygen-Transport-RBC -- OMIM-Gene-141900-Disease-SCD-603903-BetaThal-613985"
        ),
        "locus": "11p15.4",
        "protein_size": "147 aa / 16 kDa",
        "inheritance": (
            "AR/codominant — biallelic or compound heterozygous; "
            "HbSS (homozygous HbS p.Glu6Val) = sickle cell disease (SCD) — severe; "
            "HbSC (HbS + p.Glu6Lys) = milder phenotype with prominent eye/splenic disease; "
            "HbS/beta-0-thal = equivalent severity to HbSS; HbS/beta+-thal = milder; "
            "beta-thal major (Cooley anaemia) = biallelic null/severe beta-thal alleles — transfusion-dependent; "
            "HPLC/IEF distinguishes all Hb variants; NBS MANDATED in high-incidence countries"
        ),
        "disease_category": (
            "Sickle cell disease (SCD) + beta-thalassemia spectrum; "
            "SCD: HbS polymerisation → RBC sickling → vaso-occlusion → pain crisis, organ damage; "
            "beta-thal major: absent/severe reduction in beta-globin → alpha-globin excess → ineffective "
            "erythropoiesis + haemolysis → severe transfusion-dependent anaemia; "
            "HPLC/IEF — PATHOGNOMONIC; NBS mandated for early prophylaxis"
        ),
        "disease_pathway": (
            "HbS (p.Glu6Val) replaces negatively charged glutamate with neutral valine on the "
            "beta-globin surface, creating a hydrophobic patch that enables deoxygenated HbS "
            "polymerisation into long rigid fibres. Polymer fibre formation distorts RBCs into "
            "sickle shape → increased RBC rigidity + abnormal adhesion to vascular endothelium → "
            "vaso-occlusion → ischaemia-reperfusion injury → pain crisis, stroke, acute chest syndrome. "
            "HbF (foetal haemoglobin, gamma-globin) inhibits HbS polymerisation — hydroxyurea increases "
            "HbF by gamma-globin reactivation, reducing VOC by ~45%. "
            "Beta-thal: absent beta-globin → excess unpaired alpha-globin chains precipitate → "
            "erythroid precursor destruction (ineffective erythropoiesis) + peripheral haemolysis."
        ),
        "pathognomonic": (
            "SICKLED CELLS ON PERIPHERAL BLOOD SMEAR (PBS) — elongated crescent-shaped RBCs = pathognomonic of SCD; "
            "TARGET CELLS on PBS (HbC, HbSC, HbCC, beta-thal) — thin RBCs with central dense area; "
            "HAEMOGLOBIN HPLC/IEF — separates HbS, HbC, HbF, HbA peaks — PATHOGNOMONIC + quantitative; "
            "SOLUBILITY TEST (Sickledex) — HbS positive; does not distinguish SS from trait; "
            "NBS: IEF at 24-48 hours detects FS (HbF+HbS = SCD) pattern before HbA appears"
        ),
        "treatment": (
            "HYDROXYUREA — first-line SCD (FDA 1998); reactivates gamma-globin → HbF increase → "
            "reduces VOC 45%, ACS 50%, transfusions, mortality; recommended all SCD from 9 months; "
            "VOXELOTOR — HbS polymerisation blocker (allosteric Hb modifier); increases Hb ~1 g/dL; "
            "CRIZANLIZUMAB — anti-P-selectin mAb; reduces VOC by 45%; IV monthly; "
            "BLOOD TRANSFUSION — simple or exchange; for stroke prevention (monthly transfusion), "
            "ACS, aplastic crisis; iron chelation (deferasirox) for transfusion iron overload; "
            "HSCT — curative; HLA-matched sibling preferred; 95% event-free survival; "
            "GENE THERAPY Casgevy (exagamglogene autotemcel, CRISPR-Cas9 HbF induction) — FDA 2023; "
            "Lyfgenia (betibeglogene autotemcel, lentiviral) — FDA 2023; curative; "
            "PENICILLIN V prophylaxis from birth to 5 yr (functional asplenia); "
            "VACCINATION: pneumococcal, meningococcal, Hib mandatory; "
            "L-GLUTAMINE — reduces oxidative stress; "
            "Beta-thal major: lifelong transfusion + iron chelation; luspatercept reduces transfusion burden; "
            "HSCT and gene therapy (betibeglogene) curative for thal major"
        ),
        "key_features": [
            "NBS MANDATORY — IEF at birth detects FS pattern (SCD) before HbA expressed; enables early penicillin V",
            "Hydroxyurea underprescribed globally — start from 9 months in all SCD; reduces VOC 45%, improves survival",
            "HbS polymerisation — deoxygenation-triggered; voxelotor locks Hb in oxy state to prevent polymer",
            "Splenic sequestration crisis — acute massive splenomegaly in infants; can be fatal; urgent transfusion",
            "Avascular necrosis (AVN) — femoral/humeral head ischaemia; hip/shoulder pain; MRI diagnoses early",
            "Priapism — abnormal sustained erection from sickling in penile vasculature; urological emergency",
            "HPLC/IEF distinguishes HbSS vs HbSC vs HbS/beta-thal — critical for prognosis and management",
            "Casgevy (CRISPR) and Lyfgenia (gene therapy) FDA 2023 — curative options for SCD and thal major",
        ],
        "key_ddx": [
            "HbSC disease (HbS + HbC) — milder haemolysis but prominent proliferative retinopathy + AVN; HPLC shows S+C",
            "HbS/beta-thal — HbA present on HPLC (beta+-thal) or absent (beta-0-thal); similar to HbSS severity",
            "G6PD deficiency + trigger — episodic haemolysis; PBS shows bite/blister cells; G6PD assay low",
            "Hereditary spherocytosis — spherocytes; osmotic fragility increased; EMA reduced; NBS usually normal",
        ],
        "hgb_g_dl_median": 7.5,
        "reticulocyte_pct_median": 15,
        "crisis_per_year_median": 3,
        "transfusion_pct": 40,
        "splenomegaly_pct": 30,
        "hydroxyurea_pct": 55,
        "nbs_detected": True,
    },
    {
        "gene": "HBA1",
        "alt_name": (
            "HBA1-142aa-16p13.3-AR-DELETIONAL-ALPHA-THALASSEMIA-HbBARTS-HYDROPS-FETALIS-FATAL-"
            "HbH-DISEASE-MLPA-MANDATORY-IUT-SURVIVAL-SOUTHEAST-ASIAN-SEA-DELETION"
        ),
        "protein": (
            "HBA1 -- 16p13.3 AR (deletional) -- 142aa -- Alpha-Globin-15kDa-Haemoglobin-Tetramer-"
            "Component-Alpha2Beta2-Oxygen-Transport-RBC -- OMIM-Gene-141800-Disease-AlphaThal-604131"
        ),
        "locus": "16p13.3",
        "protein_size": "142 aa / 15 kDa",
        "inheritance": (
            "AR deletional (gene deletion most common mechanism — standard sequencing MISSES deletions); "
            "4 alpha-globin genes per diploid genome (2 per chromosome 16); "
            "--/-- (4-gene deletion) = Hb Bart's hydrops fetalis — FATAL in utero without IUT; "
            "--/-alpha (3-gene deletion) = HbH disease — moderate to severe haemolytic anaemia; "
            "--SEA (Southeast Asian double deletion) most common in SE Asia; "
            "-alpha/-alpha (2 deletions, trans) = alpha-thal trait — mild microcytosis; "
            "MLPA or GAP-PCR required — standard PCR misses deletions; NBS detects Hb Bart's"
        ),
        "disease_category": (
            "Alpha-thalassaemia spectrum — from trait to fatal hydrops fetalis; "
            "Hb Bart's hydrops (--/--): gamma4 tetramers cannot deliver O2 → fatal foetal anaemia + hydrops; "
            "HbH disease (--/-alpha): beta4 tetramers (HbH) → haemolysis + splenomegaly + anaemia; "
            "HbH inclusion bodies on crystal violet stain — diagnostic; "
            "MLPA/GAP-PCR mandatory (standard PCR misses deletions); NBS detects Hb Bart's peaks"
        ),
        "disease_pathway": (
            "Normal haemoglobin requires two alpha-globin chains + two beta-globin chains (alpha2beta2). "
            "Deleted alpha-globin genes → excess beta-globin chains form unstable beta4 tetramers (HbH). "
            "HbH has very high O2 affinity (no cooperative release) → tissue hypoxia despite apparent Hb level. "
            "HbH is unstable — precipitates within RBCs as inclusion bodies → "
            "splenic RBC trapping and destruction (extravascular haemolysis). "
            "In Hb Bart's (4-gene deletion): gamma4 tetramers form in foetal life — extremely high O2 affinity, "
            "cannot release O2 to tissues → foetal hydrops. IUT is the only survival pathway. "
            "Deletions require MLPA or gap-PCR — point mutations are rare (unlike beta-thal)."
        ),
        "pathognomonic": (
            "Hb BART'S (gamma4 TETRAMERS) ON NBS IEF — pathognomonic of severe alpha-thal; "
            "Hb Bart's peak on neonatal IEF screen = 4-gene deletion (fatal unless IUT given in utero); "
            "HbH (beta4) INCLUSION BODIES ON CRYSTAL VIOLET STAIN — golf-ball appearance inside RBCs; "
            "MLPA/GAP-PCR showing deletion pattern — required for accurate gene count; "
            "MICROCYTOSIS + LOW MCV (< 70 fL) without iron deficiency — key screening clue; "
            "NORMAL SERUM IRON / FERRITIN — distinguishes from iron-deficiency anaemia"
        ),
        "treatment": (
            "Hb BART'S HYDROPS: INTRAUTERINE TRANSFUSION (IUT) from 18-20 weeks — survival possible; "
            "ex utero IUT followed by postnatal chronic transfusion programme + iron chelation; "
            "gene therapy in clinical trials (curative IUT + gene correction); "
            "HbH DISEASE: intermittent transfusion for haemolytic/aplastic crises; "
            "FOLIC ACID 5 mg/day — all alpha-thal; supports erythropoiesis; "
            "SPLENECTOMY — for hypersplenism refractory to medical management; vaccination first; "
            "IRON SUPPLEMENTATION — AVOID unless iron deficiency confirmed (ferritin low); "
            "iron overload common in HbH from transfusions + increased gut absorption; "
            "LUSPATERCEPT — investigational in HbH (reduces ineffective erythropoiesis); "
            "MLPA/GAP-PCR for family cascade testing; "
            "COUNSELLING — both parents carrier of --SEA → 25% Hb Bart's risk per pregnancy"
        ),
        "key_features": [
            "MLPA MANDATORY — standard PCR misses alpha-globin deletions; GAP-PCR for common deletion types",
            "NBS IEF detects Hb Bart's peak — identifies 4-gene deletion before foetal compromise at next pregnancy",
            "IUT for Hb Bart's — survival rate >80% with timely IUT; postnatal chronic transfusion needed lifelong",
            "AVOID IRON unless deficiency confirmed — alpha-thal trait patients often prescribed iron incorrectly for microcytosis",
            "HbH inclusion bodies (crystal violet) — golf-ball inclusions inside RBCs; bedside diagnostic test",
            "Southeast Asian SEA deletion (--SEA) most common in SE Asia; Mediterranean deletions differ",
            "Splenomegaly in HbH — common; splenectomy increases thrombosis risk; delay unless severe hypersplenism",
            "Carrier screening in partners — if both --SEA carriers, 25% Hb Bart's risk; offer prenatal diagnosis",
        ],
        "key_ddx": [
            "Iron deficiency anaemia — microcytosis + low ferritin + low serum iron; responds to iron (unlike alpha-thal trait)",
            "Beta-thalassaemia trait — HbA2 elevated (>3.5%); HbBarts absent; beta-globin sequencing confirms",
            "HbE/HbC — different HPLC peaks; haemoglobin variant points vs deletion pattern; MLPA differentiates",
            "Anaemia of chronic disease — normochromic/normocytic; ferritin elevated; no HbH inclusions",
        ],
        "hgb_g_dl_median": 8.5,
        "reticulocyte_pct_median": 10,
        "crisis_per_year_median": 0.5,
        "transfusion_pct": 25,
        "splenomegaly_pct": 45,
        "hydroxyurea_pct": 5,
        "nbs_detected": True,
    },
    {
        "gene": "G6PD",
        "alt_name": (
            "G6PD-515aa-Xq28-XLR-MOST-COMMON-RBC-ENZYMOPATHY-400-MILLION-AFFECTED-"
            "TRIGGER-INDUCED-HAEMOLYSIS-BITE-CELLS-HEINZ-BODIES-TEST-3-MONTHS-AFTER-CRISIS"
        ),
        "protein": (
            "G6PD -- Xq28 XLR -- 515aa -- Glucose-6-Phosphate-Dehydrogenase-59kDa-Pentose-Phosphate-"
            "Pathway-NADPH-Production-Oxidative-Stress-Protection -- OMIM-Gene-305900-Disease-G6PD-Def-300908"
        ),
        "locus": "Xq28",
        "protein_size": "515 aa / 59 kDa",
        "inheritance": (
            "X-linked recessive (XLR); "
            "males (XY) hemizygous → fully affected; "
            "females (XX) heterozygous — Lyon X-inactivation → variable expression; "
            "can have intermediate or full disease severity depending on X-inactivation skewing; "
            "G202A (G6PD A-) = most common African variant — class III (mild-moderate); "
            "Mediterranean (p.Ser188Phe) + Canton = class II (severe); "
            "NBS not universal; "
            "~400 million affected globally — most common RBC enzymopathy"
        ),
        "disease_category": (
            "G6PD deficiency — most common RBC enzymopathy globally (~400 million affected); "
            "trigger-induced acute haemolytic anaemia; normally ASYMPTOMATIC between episodes; "
            "oxidant stress (drugs, fava beans, infections) depletes NADPH → oxidised Hb precipitates → "
            "HEINZ BODIES → bite cells/blister cells on PBS → acute intravascular haemolysis; "
            "neonatal jaundice risk; favism in Mediterranean/class II variants"
        ),
        "disease_pathway": (
            "G6PD catalyses the rate-limiting step of the pentose phosphate pathway (PPP): "
            "glucose-6-phosphate + NADP+ → 6-phosphogluconate + NADPH. "
            "NADPH maintains glutathione in reduced form (GSH), protecting RBCs from oxidative damage. "
            "RBCs have NO mitochondria and no alternative NADPH source — entirely dependent on G6PD. "
            "Oxidant stress (primaquine, dapsone, fava beans, infections) overwhelms reduced GSH reserve → "
            "Hb oxidation → haemichrome precipitation → HEINZ BODIES (denatured Hb clusters) → "
            "spleen removes Heinz body-containing portion → BITE CELLS / BLISTER CELLS on PBS → "
            "acute intravascular + extravascular haemolysis. "
            "Reticulocytes have higher G6PD activity — testing during crisis gives falsely normal result."
        ),
        "pathognomonic": (
            "BITE CELLS (semicircular 'bitten-out' RBCs) on PBS during acute crisis = pathognomonic of G6PD deficiency; "
            "BLISTER CELLS — pale blister within RBC membrane during acute haemolysis; "
            "HEINZ BODIES — denatured Hb precipitates on crystal violet stain; "
            "G6PD ENZYME ASSAY (spectrophotometric) — low/absent enzyme activity; "
            "CRITICALLY: TEST 3 MONTHS AFTER CRISIS — reticulocytes have higher G6PD; false-normal if tested acutely; "
            "MOLECULAR TESTING — confirms variant class (G202A, p.Ser188Phe, Canton)"
        ),
        "treatment": (
            "REMOVE TRIGGER — most important intervention; identify and stop offending drug/food; "
            "FOLIC ACID 5 mg/day — supports haematinic demand during haemolysis; "
            "BLOOD TRANSFUSION — if severe acute haemolysis (Hb < 6-7 g/dL or haemodynamic compromise); "
            "ADEQUATE HYDRATION — prevents haemoglobin precipitation in renal tubules (acute tubular necrosis); "
            "EXCHANGE TRANSFUSION for neonatal jaundice if Hb falling rapidly + bilirubin rising; "
            "PHOTOTHERAPY for neonatal jaundice; "
            "AVOID: primaquine, dapsone, nitrofurantoin, rasburicase, high-dose aspirin, "
            "sulphonamides, chloroquine (high dose), methylene blue (paradoxically worsens), fava beans; "
            "NO SPECIFIC MEDICAL THERAPY between episodes — no enzyme replacement available; "
            "GENE THERAPY — clinical trial pipeline (lentiviral G6PD); "
            "PATIENT EDUCATION — carry drug avoidance card; "
            "FAMILY TESTING — X-linked; maternal brothers/sons of carrier at risk"
        ),
        "key_features": [
            "Test 3 MONTHS after crisis — reticulocytes have higher G6PD enzyme activity → false-normal during acute haemolysis",
            "G202A (G6PD A-) — most common African variant; class III (mild-moderate); activity 10-60% residual",
            "Mediterranean (p.Ser188Phe) and Canton — class II (severe); residual activity <10%; favism risk high",
            "Trigger list: primaquine, dapsone, nitrofurantoin, rasburicase, high-dose aspirin, sulphonamides, fava beans",
            "Males fully affected; females variable (Lyon inactivation) — can be severely affected with skewed inactivation",
            "Bite cells and blister cells — transient during acute crisis; may disappear within 24-48 hours",
            "Antimalarial primaquine — major trigger; screen for G6PD BEFORE prescribing primaquine or rasburicase",
            "NBS not universal — consider targeted G6PD screening in high-incidence ethnic populations",
        ],
        "key_ddx": [
            "Autoimmune haemolytic anaemia (AIHA) — DAT positive; spherocytes; no trigger; chronic vs episodic",
            "Hereditary spherocytosis — spherocytes; EMA test reduced; osmotic fragility increased; no trigger",
            "PKLR deficiency — chronic haemolysis; echinocytes; PK enzyme low; no episodic trigger pattern",
            "HBB sickle cell — sickled cells on PBS; HbS on HPLC; vaso-occlusion episodes not haemolytic triggers",
        ],
        "hgb_g_dl_median": 11.5,
        "reticulocyte_pct_median": 3,
        "crisis_per_year_median": 1,
        "transfusion_pct": 15,
        "splenomegaly_pct": 5,
        "hydroxyurea_pct": 0,
        "nbs_detected": False,
    },
    {
        "gene": "PKLR",
        "alt_name": (
            "PKLR-574aa-1q22-AR-PYRUVATE-KINASE-DEFICIENCY-MOST-COMMON-NON-G6PD-ENZYMOPATHY-"
            "ECHINOCYTES-PATHOGNOMONIC-ELEVATED-2-3-DPG-MITAPIVAT-FDA-2022-FIRST-ORAL-THERAPY"
        ),
        "protein": (
            "PKLR -- 1q22 AR -- 574aa -- Pyruvate-Kinase-LR-Liver-RBC-Isoform-62kDa-Glycolysis-"
            "ATP-Generation-RBC-Energy-Metabolism -- OMIM-Gene-609712-Disease-PKD-266200"
        ),
        "locus": "1q22",
        "protein_size": "574 aa / 62 kDa",
        "inheritance": (
            "AR (autosomal recessive — biallelic LOF or compound heterozygous); "
            "most common hereditary RBC enzymopathy after G6PD deficiency; "
            "p.Arg486Trp = most common Northern European pathogenic variant; "
            "compound heterozygous (one null + one missense) = intermediate severity; "
            "biallelic null/frameshift = severe neonatal haemolytic anaemia; "
            "NBS not standard; PK enzyme assay required"
        ),
        "disease_category": (
            "Pyruvate kinase (PK) deficiency — most common non-G6PD hereditary RBC enzymopathy; "
            "chronic extravascular haemolytic anaemia; "
            "PK catalyses final step of glycolysis (PEP → pyruvate + ATP); "
            "deficient ATP → RBC energy failure → membrane deformation → haemolysis; "
            "ECHINOCYTES (spiculated RBCs) on PBS — PATHOGNOMONIC; "
            "ELEVATED 2,3-DPG (right-shifts O2 curve) — partially compensates for anaemia; "
            "MITAPIVAT FDA 2022 — first oral disease-modifying therapy"
        ),
        "disease_pathway": (
            "Pyruvate kinase (PK-LR, liver-RBC isoform) catalyses the final ATP-generating step of glycolysis: "
            "phosphoenolpyruvate (PEP) + ADP → pyruvate + ATP. "
            "RBCs have no mitochondria — entirely dependent on glycolysis for ATP. "
            "Deficient PK → ATP depletion → impaired RBC membrane pump (Na+/K+-ATPase) → "
            "RBC dehydration and membrane rigidity → extravascular haemolysis (splenic trapping). "
            "Compensatory: PEP accumulates → 2,3-DPG rises (right-shifts O2-Hb dissociation curve → "
            "better O2 delivery to tissues at same Hb level — explains why PK patients tolerate anaemia "
            "better than expected). "
            "Mitapivat (allosteric PK activator) restores PK activity → increases ATP → reduces haemolysis."
        ),
        "pathognomonic": (
            "ECHINOCYTES (spiculated/crenated RBCs) on PBS — pathognomonic of PK deficiency; "
            "best seen on fresh smear (storage artefact if delayed); "
            "ELEVATED 2,3-DPG — compensatory; reduces symptoms at given Hb level; "
            "PK ENZYME ACTIVITY ASSAY — low/absent; confirm after RBC transfusion wash-out period; "
            "MOLECULAR GENOTYPING — confirms pathogenic PKLR variants; "
            "NBS not standard — diagnosis triggered by unexplained chronic haemolytic anaemia"
        ),
        "treatment": (
            "MITAPIVAT (Pyrukynd) 5-50 mg BD oral — FDA 2022 first disease-modifying oral therapy; "
            "allosteric PK activator; increases ATP production; reduces haemolysis; "
            "mean Hb increase 1.5 g/dL; reduces transfusion dependence; "
            "BLOOD TRANSFUSION — for haemolytic/aplastic crises; iron chelation for overload; "
            "SPLENECTOMY — reduces transfusion burden but increases post-splenectomy thrombosis risk; "
            "mandatory vaccination (pneumococcal, meningococcal, Hib) before splenectomy; "
            "delay until >5-6 years (post-splenectomy sepsis OPSI risk in young children); "
            "FOLIC ACID 5 mg/day — supports erythropoiesis; "
            "IRON CHELATION (deferasirox/deferoxamine) — transfusion iron overload; "
            "monitor ferritin + LIC (liver iron concentration by MRI T2*); "
            "LUSPATERCEPT — investigational (reduces ineffective erythropoiesis component); "
            "HSCT — curative; reserved for severe paediatric cases with HLA-matched donor; "
            "GENE THERAPY (lentiviral PKLR) — clinical trials ongoing"
        ),
        "key_features": [
            "Mitapivat (Pyrukynd) FDA 2022 — first oral disease-modifying therapy for PK deficiency; game-changing",
            "Echinocytes (spiculated RBCs) — pathognomonic of PK deficiency; examine fresh smear (storage artefact)",
            "2,3-DPG elevated — right-shifts O2 curve; patients tolerate Hb of 6-7 g/dL better than expected",
            "PK enzyme assay — test after clearing transfused donor RBCs (which have normal PK); wait 3 months",
            "Splenectomy reduces transfusion need — but post-splenectomy thrombosis risk increased; defer until >5-6 yr",
            "OPSI risk post-splenectomy — lifelong penicillin V prophylaxis + annual vaccination",
            "Iron overload — both transfusion-dependent and some non-transfusion-dependent patients; monitor ferritin/MRI",
            "Biallelic null = severe neonatal haemolytic anaemia + hydrops fetalis in most severe cases",
        ],
        "key_ddx": [
            "G6PD deficiency — episodic trigger-based haemolysis (not chronic); bite cells; G6PD assay low",
            "Hereditary stomatocytosis — overhydrated stomatocytes; different PBS morphology; KCNN4 or PIEZO1",
            "Autoimmune haemolytic anaemia — DAT positive; spherocytes; responds to steroids; no enzymopathy",
            "Hereditary spherocytosis (ANK1) — spherocytes (not echinocytes); EMA test reduced; osmotic fragility high",
        ],
        "hgb_g_dl_median": 8.0,
        "reticulocyte_pct_median": 18,
        "crisis_per_year_median": 1,
        "transfusion_pct": 35,
        "splenomegaly_pct": 60,
        "hydroxyurea_pct": 0,
        "nbs_detected": False,
    },
    {
        "gene": "ANK1",
        "alt_name": (
            "ANK1-1881aa-8p11.21-AD-HEREDITARY-SPHEROCYTOSIS-TYPE1-MOST-COMMON-HS-40-65PCT-"
            "EMA-BINDING-TEST-PATHOGNOMONIC-SPHEROCYTES-SPLENECTOMY-EFFECTIVE-PARVOVIRUS-APLASTIC-CRISIS"
        ),
        "protein": (
            "ANK1 -- 8p11.21 AD -- 1881aa -- Ankyrin-1-206kDa-Band3-AE1-Spectrin-Linkage-"
            "RBC-Membrane-Skeleton-Vertical-Connection -- OMIM-Gene-612641-Disease-HS1-182900"
        ),
        "locus": "8p11.21",
        "protein_size": "1881 aa / 206 kDa",
        "inheritance": (
            "AD (autosomal dominant — haploinsufficiency); "
            "MOST COMMON hereditary spherocytosis gene — 40-65% of all HS; "
            "de novo mutations ~25%; "
            "haploinsufficiency → ankyrin-1 reduced → spectrin deficiency (ankyrin anchors spectrin network); "
            "vertical connection loss between lipid bilayer (band 3/AE1) and horizontal spectrin skeleton; "
            "membrane vesiculation → spherocyte formation; "
            "NBS not standard"
        ),
        "disease_category": (
            "Hereditary spherocytosis type 1 (HS1) — ANK1 haploinsufficiency; "
            "most common cause of HS (~40-65% of all HS cases); "
            "ANK1 anchors band 3 (AE1) and protein 4.2 to beta-spectrin → loss disrupts vertical connections; "
            "membrane vesiculation → spherocyte formation → splenic trapping → extravascular haemolysis; "
            "EMA-BINDING TEST (flow cytometry) reduced — PATHOGNOMONIC of HS; "
            "OSMOTIC FRAGILITY INCREASED; SDS-PAGE shows spectrin/ankyrin deficiency"
        ),
        "disease_pathway": (
            "Ankyrin-1 forms the central hub of the RBC membrane skeleton, linking the horizontal "
            "spectrin-actin lattice to the vertical plasma membrane proteins (band 3/AE1, protein 4.2). "
            "Haploinsufficiency reduces ankyrin → spectrin deficiency (spectrin requires ankyrin for membrane "
            "anchorage) → loss of vertical connections between lipid bilayer and skeleton → "
            "lipid bilayer microdomains pinch off as vesicles → RBC surface area loss relative to volume → "
            "spherocyte shape (minimum surface area per volume). "
            "Spherocytes cannot deform through 3-micron splenic sinusoids → trapped and destroyed → "
            "extravascular haemolysis. Degree of spherocytosis correlates with haemolysis severity."
        ),
        "pathognomonic": (
            "SPHEROCYTES ON PBS — dense RBCs with no central pallor (vs. normal RBCs 1/3 central pale area); "
            "OSMOTIC FRAGILITY TEST INCREASED — spherocytes lyse at lower NaCl concentrations; "
            "EMA (eosin-5-maleimide) BINDING TEST BY FLOW CYTOMETRY — reduced EMA fluorescence = PATHOGNOMONIC of HS; "
            "sensitivity 93%, specificity 99% for HS; "
            "SDS-PAGE: spectrin + ankyrin bands reduced quantitatively; "
            "MCHC elevated (>36 g/dL) — dehydrated spherocytes"
        ),
        "treatment": (
            "FOLIC ACID 5 mg/day — ALL HS patients; compensates for high erythroid turnover; "
            "SPLENECTOMY — reduces haemolysis by ~90%; reserved for moderate-severe HS; "
            "defer until >5-6 years (OPSI risk from post-splenectomy sepsis in young children); "
            "laparoscopic partial or total splenectomy; "
            "VACCINATION MANDATORY BEFORE SPLENECTOMY: pneumococcal (PCV13 + PPSV23), "
            "meningococcal (MenACWY + MenB), Hib — 2-4 weeks before surgery; "
            "PENICILLIN V PROPHYLAXIS lifelong post-splenectomy; "
            "BLOOD TRANSFUSION — for aplastic crisis (Parvovirus B19); "
            "CHOLECYSTECTOMY — for pigment gallstones (symptomatic); "
            "combination laparoscopic splenectomy + cholecystectomy common; "
            "LUSPATERCEPT — investigational pipeline"
        ),
        "key_features": [
            "EMA binding test (flow cytometry) — pathognomonic of HS; sensitivity 93%, specificity 99%; preferred over osmotic fragility",
            "Splenectomy most effective treatment — reduces haemolysis 90%; defer until >5-6 yr (OPSI sepsis risk)",
            "Parvovirus B19 aplastic crisis — virus infects erythroid precursors → reticulocyte count drops → acute Hb fall",
            "Penicillin V prophylaxis post-splenectomy — lifelong; fatal OPSI from encapsulated organisms risk",
            "Gallstones — pigment gallstones in 50% of HS adults (chronic haemolysis → bilirubin excess); RUQ USS",
            "MCHC elevated (>36 g/dL) — dehydrated spherocytes; along with MCV normal = HS pattern on CBC",
            "SDS-PAGE — quantifies spectrin/ankyrin reduction; correlates with haemolysis severity",
            "De novo ANK1 mutations 25% — family history may be absent; check parents' PBS + EMA test",
        ],
        "key_ddx": [
            "AIHA (warm) — DAT positive; spherocytes (immune-mediated); steroids effective; no EMA reduction",
            "SLC4A1 HS type 2 — similar spherocytosis; AE1/band 3 mutations; SDS-PAGE shows AE1 reduction",
            "ABO haemolytic disease of newborn — maternal anti-A/B; DAT positive; resolves spontaneously",
            "Microangiopathic haemolytic anaemia (TTP, HUS) — schistocytes; thrombocytopenia; no spherocyte predominance",
        ],
        "hgb_g_dl_median": 9.5,
        "reticulocyte_pct_median": 12,
        "crisis_per_year_median": 0.5,
        "transfusion_pct": 20,
        "splenomegaly_pct": 70,
        "hydroxyurea_pct": 0,
        "nbs_detected": False,
    },
    {
        "gene": "SLC4A1",
        "alt_name": (
            "SLC4A1-911aa-17q21.31-AD-AR-AE1-BAND3-HS-TYPE2-SOUTHEAST-ASIAN-OVALOCYTOSIS-SAO-"
            "PROTECTS-CEREBRAL-MALARIA-STOMATOCYTOSIS-AR-SPLENECTOMY-CONTRAINDICATED-DISTAL-RTA"
        ),
        "protein": (
            "SLC4A1 -- 17q21.31 AD/AR -- 911aa -- AE1-Band3-Anion-Exchanger-1-95kDa-Cl-HCO3-Exchange-"
            "RBC-Membrane-Structural-Protein -- OMIM-Gene-109270-Disease-HS2-182900-SAO-166900"
        ),
        "locus": "17q21.31",
        "protein_size": "911 aa / 95 kDa",
        "inheritance": (
            "AD — hereditary spherocytosis type 2; AR — hereditary stomatocytosis / ovalostomatocytosis; "
            "SAO (Southeast Asian ovalocytosis) — heterozygous 27 bp in-frame deletion (Ala400-Ala408del); "
            "SAO heterozygotes — virtually NO haemolysis; PROTECTS against cerebral malaria; "
            "SAO homozygotes — LETHAL in utero (very rare); "
            "AD HS2 — EMA binding reduced; similar to ANK1-HS; "
            "AR stomatocytosis — SPLENECTOMY CONTRAINDICATED (high thrombosis risk); "
            "SLC4A1 also expressed in kidney distal tubule — biallelic LOF → distal renal tubular acidosis (RTA)"
        ),
        "disease_category": (
            "SLC4A1/AE1 defects — three distinct phenotypes: "
            "1. HS type 2 (AD) — spherocytic haemolytic anaemia; similar to ANK1-HS; "
            "2. SAO (heterozygous Ala400-Ala408del) — rigid ovalocytes; protective against cerebral malaria; minimal haemolysis; "
            "3. AR stomatocytosis — overhydrated stomatocytes; SPLENECTOMY CONTRAINDICATED; "
            "AE1 also expressed in distal renal tubules — biallelic LOF → distal RTA type 1; "
            "EMA test reduced (HS2); osmotic fragility NORMAL/REDUCED in SAO (opposite of classic HS)"
        ),
        "disease_pathway": (
            "AE1 (band 3) performs two roles: (1) structural — ankyrin-binding cytoplasmic domain anchors "
            "membrane skeleton; (2) functional — transmembrane domain exchanges Cl-/HCO3- for CO2 transport. "
            "AD HS2 mutations: disrupted ankyrin binding → vertical connection failure → spherocytosis (as in ANK1). "
            "SAO deletion (Ala400-Ala408del): affects the transmembrane domain → rigid band 3 protein → "
            "oval-shaped RBCs that are rigid and RESIST Plasmodium falciparum invasion → "
            "protective against cerebral malaria; heterozygous SAO has minimal haemolysis but RBC rigidity. "
            "AR stomatocytosis: gain-of-function permeability → Na+ leaks in, K+ leaks out → "
            "RBC overhydration → stomatocytes; splenectomy paradoxically increases thrombosis."
        ),
        "pathognomonic": (
            "OVAL STOMATOCYTES on PBS (SAO) — elongated oval RBCs with transverse ridge or slit; "
            "EMA BINDING TEST REDUCED (HS type 2) — same pathognomonic pattern as ANK1-HS; "
            "OSMOTIC FRAGILITY NORMAL OR REDUCED in SAO (OPPOSITE of classic HS — key DDx); "
            "BAND 3/AE1 ABSENT OR REDUCED ON SDS-PAGE — protein quantification; "
            "DISTAL RTA (hyperchloraemic normal-anion-gap metabolic acidosis) in biallelic AR — clue to SLC4A1; "
            "MOLECULAR TESTING — 27bp deletion for SAO; sequencing for HS2 and AR stomatocytosis"
        ),
        "treatment": (
            "HS TYPE 2 (AD): same as HS1 — folic acid; splenectomy if severe; vaccination first; "
            "PENICILLIN V prophylaxis post-splenectomy; "
            "SAO — NO treatment required; asymptomatic; "
            "malaria-endemic regions: SAO is protective against cerebral malaria — do not medicalise; "
            "AR STOMATOCYTOSIS — AVOID SPLENECTOMY (post-splenectomy thrombosis risk is HIGH and FATAL); "
            "FOLATE supplementation; transfusion for severe haemolytic crises; "
            "ANTICOAGULATION if thromboembolism occurs (high baseline thrombosis risk even without splenectomy); "
            "DISTAL RTA (biallelic SLC4A1): sodium bicarbonate/citrate supplementation; monitor growth and bone density; "
            "nephrocalcinosis risk — renal ultrasound monitoring"
        ),
        "key_features": [
            "SAO PROTECTS against cerebral malaria — P. falciparum cannot invade rigid SAO ovalocytes; SE Asian distribution",
            "Osmotic fragility NORMAL/REDUCED in SAO — opposite of classic HS; important DDx pitfall",
            "Splenectomy CONTRAINDICATED in AR stomatocytosis — high-risk fatal post-splenectomy thromboembolism",
            "Distal RTA (biallelic LOF) — AE1 in distal tubule; hyperchloraemic acidosis; nephrocalcinosis",
            "EMA test REDUCED in HS type 2 — same as ANK1-HS; SDS-PAGE shows AE1 band reduction (vs. spectrin in ANK1)",
            "SAO homozygotes lethal in utero — very rare; heterozygotes have remarkable malaria protection",
            "AR stomatocytosis — Na+/K+ leak → overhydrated RBCs; folate; NO splenectomy; anticoag if thrombosis",
            "Three distinct phenotypes from one gene — HS2 (AD), SAO (heterozygous deletion), AR stomatocytosis",
        ],
        "key_ddx": [
            "ANK1 HS type 1 — most common HS; SDS-PAGE shows spectrin/ankyrin loss vs. AE1 loss in SLC4A1-HS2",
            "KCNN4 DHS — dehydrated stomatocytes; MCHC elevated; splenectomy ALSO contraindicated; GOF mutations",
            "Spur cell haemolytic anaemia (liver disease) — acanthocytes on PBS; abnormal liver function; acquired",
            "SAO vs. hereditary elliptocytosis (SPTA1) — oval cells differ; EMA test normal in HE; thermal lability in HPP",
        ],
        "hgb_g_dl_median": 10.5,
        "reticulocyte_pct_median": 8,
        "crisis_per_year_median": 0.3,
        "transfusion_pct": 15,
        "splenomegaly_pct": 40,
        "hydroxyurea_pct": 0,
        "nbs_detected": False,
    },
    {
        "gene": "SPTA1",
        "alt_name": (
            "SPTA1-2429aa-1q23.1-AR-HEREDITARY-ELLIPTOCYTOSIS-HPP-PYROPOIKILOCYTOSIS-"
            "ALPHA-LELY-ALLELE-THERMAL-LABILITY-45C-EMA-NORMAL-DISTINGUISHES-FROM-HS-SPLENECTOMY-EFFECTIVE"
        ),
        "protein": (
            "SPTA1 -- 1q23.1 AR (HPP/compound HE) -- 2429aa -- Alpha-Spectrin-I-280kDa-RBC-Membrane-"
            "Horizontal-Skeleton-Antiparallel-Tetramer -- OMIM-Gene-182860-Disease-HE-130600-HPP-266140"
        ),
        "locus": "1q23.1",
        "protein_size": "2429 aa / 280 kDa",
        "inheritance": (
            "AR for HPP / severe compound HE; heterozygous = hereditary elliptocytosis (HE) usually mild; "
            "SPTA1 L260P (common pathogenic mutation in alpha-spectrin) + alphaLELY allele in trans = severe HPP; "
            "alphaLELY (Low Expression LYon) — common silent polymorphism; "
            "reduces SPTA1 expression ~75% from that allele → unmasks mutant allele effect when in trans; "
            "HE in heterozygotes: elliptocytes >25% on PBS; usually mild or asymptomatic; "
            "HPP: homozygous or compound heterozygous — severe neonatal haemolytic anaemia; "
            "EMA binding test NORMAL — distinguishes from HS"
        ),
        "disease_category": (
            "Hereditary elliptocytosis (HE) — heterozygous SPTA1 mutations; "
            "hereditary pyropoikilocytosis (HPP) — homozygous or compound heterozygous + alphaLELY; "
            "SPTA1 forms horizontal spectrin tetramer; mutations disrupt alpha-beta spectrin dimer self-association → "
            "RBC membrane mechanical failure → elliptocytes (HE) or fragmented poikilocytes (HPP); "
            "THERMAL LABILITY TEST — HPP RBCs fragment at 45°C (normal RBCs fragment at 49°C); "
            "EMA TEST NORMAL — critical DDx from HS"
        ),
        "disease_pathway": (
            "Alpha-spectrin (SPTA1) and beta-spectrin form antiparallel dimers that self-associate "
            "head-to-head to form spectrin tetramers — the horizontal scaffold of the RBC membrane skeleton. "
            "SPTA1 mutations disrupt the alpha-beta spectrin self-association site → "
            "spectrin dimers cannot form tetramers → membrane mechanical instability → "
            "RBC elongates under shear stress and cannot return to biconcave shape → ELLIPTOCYTES. "
            "In HPP (biallelic or SPTA1+alphaLELY compound): severe spectrin deficiency → "
            "RBC fragments (microspherocytes, poikilocytes) on PBS → neonatal severe haemolytic anaemia. "
            "AlphaLELY allele: common low-expression polymorphism — when in trans with SPTA1 mutation, "
            "the only functional SPTA1 produced is from the mutant allele → severe phenotype unmasked."
        ),
        "pathognomonic": (
            "ELLIPTOCYTES >25% ON PBS (HE) — elongated oval/cigar-shaped RBCs = pathognomonic of HE; "
            "POIKILOCYTES + MICROSPHEROCYTES + FRAGMENTS on PBS (HPP) — bizarre fragmented cells = HPP; "
            "THERMAL LABILITY TEST: HPP RBCs fragment at 45°C (vs. normal 49°C) — pathognomonic of HPP; "
            "EMA BINDING TEST NORMAL — CRITICAL DDx from HS (EMA is REDUCED in HS, NORMAL in HE/HPP); "
            "EKTACYTOMETRY: reduced deformability; "
            "alphaLELY ALLELE TESTING — essential if compound heterozygous suspected"
        ),
        "treatment": (
            "HE HETEROZYGOTES — NO treatment usually required; mild/asymptomatic; folic acid; "
            "monitor for haemolysis during infections (Parvovirus B19 aplastic crisis risk); "
            "HPP NEONATAL: BLOOD TRANSFUSION — often required in newborn/infancy period; "
            "PHOTOTHERAPY for neonatal jaundice; "
            "SPLENECTOMY — effective for HPP; reduces haemolysis significantly; "
            "mandatory vaccination (pneumococcal, meningococcal, Hib) before splenectomy; "
            "defer until >5-6 years; penicillin V lifelong post-splenectomy; "
            "FOLIC ACID 5 mg/day; "
            "HAEMOLYSIS IMPROVES WITH AGE — foetal gamma-globin replaced by adult spectrin as infant grows; "
            "severity may diminish by 2-3 years of age even without splenectomy; "
            "check alphaLELY in family members for accurate counselling"
        ),
        "key_features": [
            "EMA test NORMAL — most critical distinguishing feature from HS; EMA reduced in HS but NORMAL in HE/HPP",
            "alphaLELY allele — common silent polymorphism; when in trans with SPTA1 mutation → severe HPP unmasked",
            "Thermal fragmentation test at 45°C — HPP RBCs lyse at 45°C, normal at 49°C; bedside differentiation",
            "Haemolysis often improves in HPP by 2-3 years of age — foetal Hb switch affects spectrin kinetics",
            "Splenectomy effective for HPP — reduces haemolysis; delay to >5-6 yr for OPSI protection",
            "HE heterozygotes usually asymptomatic — elliptocytes >25% on PBS; no treatment usually needed",
            "Poikilocytes + fragments on neonatal PBS — HPP; dangerous if missed; check thermal lability",
            "Family testing for alphaLELY — determines true compound heterozygous status; changes risk counselling",
        ],
        "key_ddx": [
            "Hereditary spherocytosis (ANK1/SLC4A1) — spherocytes; EMA REDUCED (key); osmotic fragility high",
            "Microangiopathic haemolytic anaemia (TTP/HUS) — schistocytes; thrombocytopenia; acquired cause",
            "Iron deficiency anaemia — elliptical pencil cells (not elliptocytes); low ferritin; responds to iron",
            "SPTB (beta-spectrin) HE — same PBS morphology; SDS-PAGE shows SPTB reduction vs SPTA1",
        ],
        "hgb_g_dl_median": 7.5,
        "reticulocyte_pct_median": 20,
        "crisis_per_year_median": 2,
        "transfusion_pct": 45,
        "splenomegaly_pct": 55,
        "hydroxyurea_pct": 0,
        "nbs_detected": False,
    },
    {
        "gene": "KCNN4",
        "alt_name": (
            "KCNN4-427aa-19q13.31-AD-GOF-DEHYDRATED-HEREDITARY-STOMATOCYTOSIS-DHS-HEREDITARY-XEROCYTOSIS-"
            "SPLENECTOMY-ABSOLUTELY-CONTRAINDICATED-FATAL-THROMBOSIS-MCHC-ELEVATED-SENICAPOC-INVESTIGATIONAL"
        ),
        "protein": (
            "KCNN4 -- 19q13.31 AD GOF -- 427aa -- SK4-Gardos-Channel-IKCa1-KCa3.1-47kDa-"
            "Ca2+-Activated-K+-Channel-RBC-Volume-Regulation -- OMIM-Gene-602201-Disease-DHS-194380"
        ),
        "locus": "19q13.31",
        "protein_size": "427 aa / 47 kDa",
        "inheritance": (
            "AD gain-of-function (GOF); "
            "constitutive/increased activation of Gardos channel (SK4/KCNN4); "
            "DEHYDRATED HEREDITARY STOMATOCYTOSIS (DHS) = hereditary xerocytosis; "
            "K+ efflux constitutively activated → water follows → RBC dehydration; "
            "MCHC ELEVATED (>36 g/dL) — pathognomonic marker of dehydrated RBCs; "
            "SPLENECTOMY ABSOLUTELY CONTRAINDICATED — multiple fatal post-splenectomy thromboembolism cases; "
            "NBS not standard"
        ),
        "disease_category": (
            "Dehydrated hereditary stomatocytosis (DHS) / hereditary xerocytosis; "
            "GOF mutations → constitutive K+ efflux → water follows → dehydrated/desiccated RBCs; "
            "MCHC ELEVATED (>36 g/dL) + MCV elevated = dehydrated macrocytic stomatocytes; "
            "osmotic fragility DECREASED (dehydrated RBCs more resistant to hypotonic lysis); "
            "SPLENECTOMY ABSOLUTELY CONTRAINDICATED — fatal thromboembolism (portal vein, DVT, PE); "
            "iron overload without transfusion dependence — unusual feature"
        ),
        "disease_pathway": (
            "KCNN4 (Gardos channel / SK4 / KCa3.1) is a Ca2+-activated K+ channel in the RBC membrane. "
            "Normally activated transiently by elevated intracellular Ca2+ to regulate volume. "
            "GOF mutations → constitutive K+ efflux (independent of Ca2+ activation) → "
            "electrical gradient drives Cl- out (cotransport) → osmotic loss of KCl → "
            "water follows osmotically → RBC dehydration → MCHC rises. "
            "Dehydrated RBCs are stiff and have stomatocytic morphology. "
            "Iron overload WITHOUT transfusion: haemolysis → hepcidin suppression → increased gut iron absorption + "
            "ineffective erythropoiesis component → iron loading despite no transfusions. "
            "Post-splenectomy thrombosis: mechanism incompletely understood; may involve procoagulant phosphatidylserine "
            "on outer membrane leaflet of dehydrated RBCs triggering coagulation cascade."
        ),
        "pathognomonic": (
            "STOMATOCYTES ON PBS — RBCs with central slit (mouth-shaped) rather than round central pallor; "
            "NOTE: stomatocytes can be ARTIFACTUAL — examine multiple smear areas; "
            "MCHC ELEVATED (>36 g/dL) — pathognomonic marker; combined with MCV elevated = DHS pattern; "
            "OSMOTIC FRAGILITY DECREASED — dehydrated RBCs more resistant to hypotonic lysis; "
            "EKTACYTOMETRY — dehydration signature (rightward shift on osmotic gradient curve); "
            "MOLECULAR TESTING — confirms KCNN4 GOF variant; essential to prevent catastrophic splenectomy"
        ),
        "treatment": (
            "FOLATE 5 mg/day — supports chronic haemolysis; "
            "BLOOD TRANSFUSION — if severe anaemia (avoid if possible; accelerates iron overload); "
            "IRON CHELATION (deferasirox/deferoxamine) — iron overload WITHOUT transfusion is COMMON in DHS; "
            "monitor ferritin + LIC-MRI; start chelation if ferritin persistently >500 mcg/L; "
            "SENICAPOC (ICA-17043) — Gardos channel blocker; phase 2 trials showing efficacy; "
            "reduces K+ efflux → reduces dehydration → improves Hb; investigational; "
            "ANTICOAGULATION (warfarin, NOAC) — if thromboembolism occurs; "
            "SPLENECTOMY — ABSOLUTELY CONTRAINDICATED; "
            "fatal portal vein thrombosis, splenic vein thrombosis, DVT/PE reported post-splenectomy; "
            "multiple fatal cases in literature; NEVER perform splenectomy in DHS; "
            "GENETIC TESTING MANDATORY before any splenectomy decision in haemolytic anaemia — "
            "exclude DHS/PIEZO1 stomatocytosis first"
        ),
        "key_features": [
            "SPLENECTOMY ABSOLUTELY CONTRAINDICATED — single most critical management point in DHS; fatal thrombosis risk",
            "MCHC elevated (>36 g/dL) with elevated MCV + stomatocytes = DHS pattern on CBC + PBS",
            "Iron overload WITHOUT transfusion — unusual; haemolysis suppresses hepcidin; gut iron absorption increases",
            "Senicapoc (Gardos channel blocker) — phase 2 results promising; first targeted therapy for DHS",
            "Osmotic fragility DECREASED — dehydrated RBCs resistant to hypotonic lysis; opposite of spherocytosis",
            "Stomatocytes can be artifactual on PBS — examine multiple fields; confirm with ektacytometry",
            "GOF mechanism — channel constitutively open; K+ efflux → KCl + water loss → RBC desiccation",
            "Also known as hereditary xerocytosis — xerocytes = desiccated RBCs; MCHC highest among haemolytic anemias",
        ],
        "key_ddx": [
            "SLC4A1 AR stomatocytosis — overhydrated stomatocytes (OPPOSITE to DHS); MCHC LOW/NORMAL; different mechanism",
            "PIEZO1 (dehydrated stomatocytosis type 2) — clinically identical to KCNN4 DHS; splenectomy also CI; PIEZO1 GOF",
            "Overhydrated stomatocytosis (RHAG/RHCE) — MCV elevated but MCHC LOW; different ektacytometry signature",
            "Liver disease spur-cell haemolysis — acanthocytes; abnormal LFTs; acquired; no stomatocytes",
        ],
        "hgb_g_dl_median": 10.0,
        "reticulocyte_pct_median": 9,
        "crisis_per_year_median": 0.3,
        "transfusion_pct": 10,
        "splenomegaly_pct": 50,
        "hydroxyurea_pct": 0,
        "nbs_detected": False,
    },
]

SEEDS = list(range(2582, 2590))  # 8 seeds for 8 genes


def _rng(seed):
    rng = random.Random(seed)
    return rng


def _simulate_cohort(gene_data, seed):
    rng = _rng(seed)
    n = 40
    patients = []
    hgb_med = gene_data["hgb_g_dl_median"]
    retic_med = gene_data["reticulocyte_pct_median"]
    crisis_med = gene_data["crisis_per_year_median"]
    for i in range(n):
        hgb = max(3.5, round(rng.gauss(hgb_med, 1.5), 1))
        reticulocyte_pct = max(0.5, round(rng.gauss(retic_med, retic_med * 0.35 + 1.0), 1))
        has_splenomegaly = rng.random() < (gene_data["splenomegaly_pct"] / 100)
        splenectomy_done = has_splenomegaly and rng.random() < 0.3
        transfusion_dependent = rng.random() < (gene_data["transfusion_pct"] / 100)
        hydroxyurea_therapy = rng.random() < (gene_data["hydroxyurea_pct"] / 100)
        has_gallstones = rng.random() < 0.35 if (hgb_med < 10 or retic_med > 10) else rng.random() < 0.12
        has_iron_overload = transfusion_dependent and rng.random() < 0.4
        # KCNN4 DHS — iron overload even without transfusion
        if gene_data["gene"] == "KCNN4" and not transfusion_dependent:
            has_iron_overload = rng.random() < 0.25
        crisis_events = max(0, round(rng.gauss(crisis_med, crisis_med * 0.6 + 0.2), 1))
        nbs_detected = gene_data["nbs_detected"] and rng.random() < 0.88
        patients.append({
            "hgb_g_dl": hgb,
            "reticulocyte_pct": reticulocyte_pct,
            "has_splenomegaly": has_splenomegaly,
            "splenectomy_done": splenectomy_done,
            "transfusion_dependent": transfusion_dependent,
            "hydroxyurea_therapy": hydroxyurea_therapy,
            "has_gallstones": has_gallstones,
            "has_iron_overload": has_iron_overload,
            "crisis_events_per_year": crisis_events,
            "nbs_detected": nbs_detected,
        })
    return patients


def generate_overview():
    gene_summaries = []
    for gene, seed in zip(ATLAS_GENES, SEEDS):
        pts = _simulate_cohort(gene, seed)
        avg_hgb = round(sum(p["hgb_g_dl"] for p in pts) / len(pts), 1)
        avg_retic = round(sum(p["reticulocyte_pct"] for p in pts) / len(pts), 1)
        splenomegaly_pct = round(sum(1 for p in pts if p["has_splenomegaly"]) / len(pts) * 100, 1)
        splenectomy_pct = round(sum(1 for p in pts if p["splenectomy_done"]) / len(pts) * 100, 1)
        transfusion_pct = round(sum(1 for p in pts if p["transfusion_dependent"]) / len(pts) * 100, 1)
        hyu_pct = round(sum(1 for p in pts if p["hydroxyurea_therapy"]) / len(pts) * 100, 1)
        gallstone_pct = round(sum(1 for p in pts if p["has_gallstones"]) / len(pts) * 100, 1)
        iron_ol_pct = round(sum(1 for p in pts if p["has_iron_overload"]) / len(pts) * 100, 1)
        nbs_pct = round(sum(1 for p in pts if p["nbs_detected"]) / len(pts) * 100, 1)
        gene_summaries.append({
            "gene": gene["gene"],
            "locus": gene["locus"],
            "protein_size": gene["protein_size"],
            "inheritance": gene["inheritance"].split(";")[0].strip(),
            "disease_name": gene["disease_category"].split(";")[0].strip()[:80],
            "pathognomonic_short": gene["pathognomonic"].split(";")[0].strip()[:100],
            "hgb_g_dl_median": gene["hgb_g_dl_median"],
            "reticulocyte_pct_median": gene["reticulocyte_pct_median"],
            "avg_hgb_g_dl": avg_hgb,
            "avg_reticulocyte_pct": avg_retic,
            "splenomegaly_pct": splenomegaly_pct,
            "splenectomy_pct": splenectomy_pct,
            "transfusion_pct": transfusion_pct,
            "hydroxyurea_pct": hyu_pct,
            "gallstone_pct": gallstone_pct,
            "iron_overload_pct": iron_ol_pct,
            "nbs_detected_pct": nbs_pct,
        })

    all_pts = []
    for gene, seed in zip(ATLAS_GENES, SEEDS):
        all_pts.extend(_simulate_cohort(gene, seed))

    return {
        "title": "Hereditary Hemolytic Anemia Atlas",
        "subtitle": (
            "Complete 8-Gene Hereditary Haemolytic Anaemia Reference — "
            "HBB-HBA1-G6PD-PKLR-ANK1-SLC4A1-SPTA1-KCNN4"
        ),
        "genes": [g["gene"] for g in ATLAS_GENES],
        "n_genes": 8,
        "total_patients": 320,
        "seeds": "2582–2589",
        "disease_classes": [
            "HBB — sickle cell disease + beta-thalassemia; HbS p.Glu6Val; HPLC pathognomonic; NBS MANDATORY; hydroxyurea first-line",
            "HBA1 — alpha-thalassemia; Hb Bart's hydrops fetalis (FATAL); HbH disease; MLPA mandatory (PCR misses deletions)",
            "G6PD — most common RBC enzymopathy globally (~400M); trigger-induced haemolysis; bite cells; test 3 months after crisis",
            "PKLR — PK deficiency; echinocytes pathognomonic; 2,3-DPG elevated; mitapivat FDA 2022 first oral therapy",
            "ANK1 — hereditary spherocytosis type 1 (most common HS 40-65%); EMA test pathognomonic; splenectomy effective",
            "SLC4A1 — HS type 2 + SAO (protects cerebral malaria) + AR stomatocytosis (splenectomy CONTRAINDICATED)",
            "SPTA1 — hereditary elliptocytosis (HE) + HPP; EMA NORMAL (DDx from HS); thermal lability 45°C; alphaLELY allele",
            "KCNN4 — dehydrated hereditary stomatocytosis (DHS/xerocytosis); SPLENECTOMY ABSOLUTELY CONTRAINDICATED; MCHC elevated",
        ],
        "gene_summary": gene_summaries,
        "aggregate_metrics": {
            "avg_hgb_g_dl": round(sum(p["hgb_g_dl"] for p in all_pts) / len(all_pts), 1),
            "avg_reticulocyte_pct": round(sum(p["reticulocyte_pct"] for p in all_pts) / len(all_pts), 1),
            "splenomegaly_pct": round(sum(1 for p in all_pts if p["has_splenomegaly"]) / len(all_pts) * 100, 1),
            "splenectomy_pct": round(sum(1 for p in all_pts if p["splenectomy_done"]) / len(all_pts) * 100, 1),
            "transfusion_pct": round(sum(1 for p in all_pts if p["transfusion_dependent"]) / len(all_pts) * 100, 1),
            "hydroxyurea_pct": round(sum(1 for p in all_pts if p["hydroxyurea_therapy"]) / len(all_pts) * 100, 1),
            "gallstone_pct": round(sum(1 for p in all_pts if p["has_gallstones"]) / len(all_pts) * 100, 1),
            "iron_overload_pct": round(sum(1 for p in all_pts if p["has_iron_overload"]) / len(all_pts) * 100, 1),
            "nbs_detected_pct": round(sum(1 for p in all_pts if p["nbs_detected"]) / len(all_pts) * 100, 1),
        },
        "clinical_pearls": [
            "HBB/SCD: Hydroxyurea underprescribed — recommend from 9 months in ALL SCD patients; "
            "reduces VOC 45%, ACS 50%, stroke risk; Casgevy (CRISPR) + Lyfgenia (gene therapy) FDA 2023 curative options",
            "HBB/NBS MANDATORY: IEF at 24-48 hours detects FS pattern (HbF+HbS = SCD) before HbA appears; "
            "enables early penicillin V prophylaxis from birth; NBS saves lives",
            "HBA1/MLPA MANDATORY: standard PCR/sequencing misses alpha-globin DELETIONS — request MLPA or GAP-PCR; "
            "HbH patients misdiagnosed as iron deficiency (microcytosis); AVOID iron unless ferritin confirms deficiency",
            "G6PD/TEST TIMING: test enzyme activity 3 MONTHS after acute haemolytic crisis — reticulocytes have higher G6PD; "
            "false-normal result during acute crisis has led to missed diagnoses; screen BEFORE prescribing primaquine/rasburicase",
            "PKLR/MITAPIVAT FDA 2022: first oral disease-modifying therapy for PK deficiency; allosteric PK activator; "
            "echinocytes on fresh PBS are pathognomonic; 2,3-DPG elevation means patients tolerate lower Hb than expected",
            "ANK1/EMA TEST: EMA binding test (flow cytometry) is the GOLD STANDARD for HS diagnosis — "
            "sensitivity 93%, specificity 99%; preferred over osmotic fragility; Parvovirus B19 → aplastic crisis in ALL HS",
            "SLC4A1/SAO PROTECTS AGAINST CEREBRAL MALARIA: Southeast Asian ovalocytosis (27bp deletion) "
            "rigid ovalocytes resist P. falciparum invasion; osmotic fragility NORMAL/REDUCED (opposite of classic HS); "
            "AR stomatocytosis — SPLENECTOMY CONTRAINDICATED; distal RTA in biallelic SLC4A1 LOF",
            "SPTA1/EMA NORMAL = KEY DDx: EMA binding test is NORMAL in HE and HPP — "
            "distinguishes from HS where EMA is REDUCED; thermal fragmentation at 45°C confirms HPP; "
            "alphaLELY allele in trans unmasks severe HPP phenotype",
            "KCNN4/SPLENECTOMY ABSOLUTELY CONTRAINDICATED: Gardos channel GOF → dehydrated RBCs; "
            "multiple FATAL post-splenectomy thromboembolism cases reported; "
            "MCHC >36 g/dL + elevated MCV + stomatocytes = DHS; genetic test ALL haemolytic anaemia BEFORE splenectomy",
            "SPLENECTOMY RULE: ALWAYS perform genetic diagnosis before splenectomy in unexplained haemolytic anaemia; "
            "KCNN4 DHS and PIEZO1 DHS are absolute contraindications; SLC4A1 AR stomatocytosis also contraindicated; "
            "VACCINATION (PCV13, PPSV23, MenACWY, MenB, Hib) 2-4 weeks before splenectomy in all HS",
        ],
    }


def generate_breakdown():
    breakdowns = []
    for gene, seed in zip(ATLAS_GENES, SEEDS):
        pts = _simulate_cohort(gene, seed)
        avg_hgb = round(sum(p["hgb_g_dl"] for p in pts) / len(pts), 1)
        avg_retic = round(sum(p["reticulocyte_pct"] for p in pts) / len(pts), 1)
        splenomegaly_pct = round(sum(1 for p in pts if p["has_splenomegaly"]) / len(pts) * 100, 1)
        splenectomy_pct = round(sum(1 for p in pts if p["splenectomy_done"]) / len(pts) * 100, 1)
        transfusion_pct = round(sum(1 for p in pts if p["transfusion_dependent"]) / len(pts) * 100, 1)
        hyu_pct = round(sum(1 for p in pts if p["hydroxyurea_therapy"]) / len(pts) * 100, 1)
        gallstone_pct = round(sum(1 for p in pts if p["has_gallstones"]) / len(pts) * 100, 1)
        iron_ol_pct = round(sum(1 for p in all_pts if p["has_iron_overload"]) / len(pts) * 100, 1) if False else round(
            sum(1 for p in pts if p["has_iron_overload"]) / len(pts) * 100, 1
        )
        nbs_pct = round(sum(1 for p in pts if p["nbs_detected"]) / len(pts) * 100, 1)
        avg_crisis = round(sum(p["crisis_events_per_year"] for p in pts) / len(pts), 2)
        breakdowns.append({
            "gene": gene["gene"],
            "locus": gene["locus"],
            "protein_size": gene["protein_size"],
            "inheritance": gene["inheritance"],
            "disease_category": gene["disease_category"],
            "disease_pathway": gene["disease_pathway"],
            "pathognomonic": gene["pathognomonic"],
            "treatment": gene["treatment"],
            "key_features": gene["key_features"],
            "key_ddx": gene["key_ddx"],
            "n_patients": 40,
            "avg_hgb_g_dl": avg_hgb,
            "avg_reticulocyte_pct": avg_retic,
            "splenomegaly_pct": splenomegaly_pct,
            "splenectomy_pct": splenectomy_pct,
            "transfusion_pct": transfusion_pct,
            "hydroxyurea_pct": hyu_pct,
            "gallstone_pct": gallstone_pct,
            "iron_overload_pct": iron_ol_pct,
            "nbs_detected_pct": nbs_pct,
            "avg_crisis_events_per_year": avg_crisis,
        })
    return {"gene_breakdowns": breakdowns}


def generate_definitions():
    gene_entries = {}
    for gene in ATLAS_GENES:
        gene_entries[gene["gene"]] = {
            "gene": gene["gene"],
            "full_name": gene["gene"],
            "locus": gene["locus"],
            "protein_size": gene["protein_size"],
            "inheritance": gene["inheritance"].split(";")[0].strip(),
            "disease_name": gene["disease_category"],
            "disease_pathway": gene["disease_pathway"],
            "pathognomonic": gene["pathognomonic"],
            "treatment": gene["treatment"][:600],
            "key_features": gene["key_features"],
            "key_ddx": gene["key_ddx"],
            "hgb_g_dl_median": gene["hgb_g_dl_median"],
            "reticulocyte_pct_median": gene["reticulocyte_pct_median"],
            "transfusion_pct": gene["transfusion_pct"],
            "splenomegaly_pct": gene["splenomegaly_pct"],
            "hydroxyurea_pct": gene["hydroxyurea_pct"],
            "nbs_detected": gene["nbs_detected"],
        }
    return {
        "gene_entries": gene_entries,
        "hemolytic_anemia_glossary": {
            "Haemolytic Anaemia — Classification (Intravascular vs Extravascular)": (
                "INTRAVASCULAR haemolysis: RBC destruction within blood vessels → haemoglobinaemia, "
                "haemoglobinuria (dark urine), low/absent serum haptoglobin, elevated LDH, elevated plasma Hb; "
                "causes: G6PD (acute oxidant crisis), AIHA (complement-mediated), PNH, TTP/HUS. "
                "EXTRAVASCULAR haemolysis: RBC destruction in spleen/liver macrophages → splenomegaly, "
                "indirect hyperbilirubinaemia, pigment gallstones, elevated LDH; "
                "causes: HS (ANK1, SLC4A1), HE/HPP (SPTA1), PK deficiency (PKLR), sickle cell (HBB). "
                "Both types: reticulocytosis, elevated indirect bilirubin, low haptoglobin, elevated LDH."
            ),
            "Osmotic Fragility Test vs EMA Binding Test": (
                "OSMOTIC FRAGILITY TEST: RBCs suspended in decreasing NaCl concentrations; "
                "spherocytes lyse at higher NaCl (increased fragility); "
                "DHS/SAO RBCs lyse at LOWER NaCl (decreased fragility — dehydrated/rigid cells); "
                "sensitivity ~68% for HS — false negatives; affected by storage; not preferred. "
                "EMA (eosin-5-maleimide) BINDING TEST (flow cytometry): "
                "EMA binds band 3 (AE1), Rh, Rh50, CD47; reduced fluorescence in HS (ANK1, SLC4A1, SPTB, EPB42); "
                "sensitivity 93%, specificity 99% for HS; "
                "NORMAL EMA in HE/HPP (SPTA1) — critical DDx point; "
                "preferred over osmotic fragility; single-tube flow cytometry test."
            ),
            "Peripheral Blood Smear Morphology Keys": (
                "SICKLE CELLS (drepanocytes) — elongated crescent HBB SCD (HbSS, HbS/beta-thal). "
                "TARGET CELLS (codocytes) — bull's-eye appearance; HBB (HbC, HbSC, beta-thal), liver disease, iron def. "
                "SPHEROCYTES — dense RBCs no central pallor; HS (ANK1, SLC4A1), AIHA. "
                "ELLIPTOCYTES — oval/cigar-shaped; HE heterozygotes (SPTA1, SPTB, EPB41). "
                "POIKILOCYTES + MICROSPHEROCYTES + FRAGMENTS — HPP (SPTA1 biallelic); neonatal severe. "
                "STOMATOCYTES — central slit/mouth; DHS (KCNN4), overhydrated stomatocytosis (RHAG). "
                "ECHINOCYTES (burr cells/crenated) — uniform spicules; PK deficiency (PKLR), uraemia, storage artefact. "
                "BITE CELLS + BLISTER CELLS — G6PD deficiency during acute oxidant crisis. "
                "HEINZ BODIES (crystal violet stain) — denatured Hb precipitates; G6PD, unstable Hb. "
                "HbH INCLUSIONS (crystal violet stain) — golf-ball pattern inside RBCs; HBA1 deletion (HbH disease). "
                "OVALOCYTES (SAO) — rigid elongated ovals with transverse ridge; SLC4A1 SAO deletion."
            ),
            "MCHC in Dehydrated vs Overhydrated Stomatocytosis": (
                "DEHYDRATED HEREDITARY STOMATOCYTOSIS (DHS — KCNN4 GOF): "
                "MCHC ELEVATED (>36 g/dL); MCV elevated; cells are desiccated; "
                "osmotic fragility DECREASED (more resistant to hypotonic lysis); "
                "ektacytometry: right-shifted Omin. "
                "OVERHYDRATED HEREDITARY STOMATOCYTOSIS (OHS — RHAG/RHCE): "
                "MCHC LOW (<28 g/dL); MCV markedly elevated; cells are swollen with water; "
                "osmotic fragility INCREASED; ektacytometry: left-shifted Omin. "
                "CLINICAL RELEVANCE: BOTH types have stomatocytes on PBS but OPPOSITE MCHC — "
                "MCHC distinguishes them immediately from the CBC; "
                "BOTH have splenectomy contraindicated (thrombosis risk)."
            ),
            "Splenectomy — When INDICATED vs CONTRAINDICATED": (
                "INDICATED (with caution + vaccination + prophylaxis): "
                "Hereditary spherocytosis (ANK1-HS1, SLC4A1-HS2) — moderate-severe; reduces haemolysis 90%; "
                "defer until >5-6 years (OPSI risk); "
                "HPP (SPTA1 biallelic) — splenectomy effective; "
                "HbH disease (HBA1) — for hypersplenism refractory to medical management. "
                "ABSOLUTELY CONTRAINDICATED: "
                "KCNN4 DHS (hereditary xerocytosis) — multiple FATAL post-splenectomy thromboembolism cases; "
                "SLC4A1 AR stomatocytosis — high thrombosis risk; "
                "PIEZO1 DHS (dehydrated stomatocytosis type 2) — same contraindication as KCNN4. "
                "PRE-SPLENECTOMY MANDATORY: "
                "Molecular diagnosis to exclude contraindicated conditions; "
                "Vaccination: PCV13 + PPSV23 + MenACWY + MenB + Hib (2-4 weeks before surgery); "
                "Penicillin V prophylaxis lifelong post-splenectomy."
            ),
            "NBS (Newborn Screening) — Haemolytic Anaemias Included": (
                "INCLUDED in most high-incidence national NBS programmes: "
                "HBB (SCD): IEF at 24-48 hours; FS pattern = SCD; enables early penicillin V + hydroxyurea; "
                "HBA1 (alpha-thal): Hb Bart's peak on IEF at birth; identifies 4-gene deletion. "
                "NOT STANDARD in most NBS programmes: "
                "G6PD: targeted NBS in some high-incidence countries (Africa, SE Asia, Mediterranean); "
                "PKLR: PK deficiency NBS not available in most countries; "
                "ANK1, SLC4A1, SPTA1, KCNN4: not included in NBS; "
                "diagnose from unexplained haemolytic anaemia + PBS + EMA test + molecular testing. "
                "G6PD: WHO recommends targeted G6PD screening before antimalarial therapy."
            ),
            "Iron Overload in Haemolytic Anaemia — Transfusion vs Non-Transfusion Dependent": (
                "TRANSFUSION-DEPENDENT iron overload: HBB beta-thal major, severe SCD, severe PKLR, HBA1 HbH; "
                "each unit of packed RBCs = 200-250 mg iron; no physiological excretion mechanism; "
                "iron accumulates in heart, liver, endocrine glands; "
                "monitor: serum ferritin + MRI T2* (liver iron concentration + cardiac T2*); "
                "chelation: deferasirox (oral, preferred), deferoxamine (SC infusion), deferiprone (oral). "
                "NON-TRANSFUSION-DEPENDENT iron overload (important exception): "
                "KCNN4 DHS — iron overload WITHOUT transfusion; haemolysis → hepcidin suppression → "
                "increased gut iron absorption; monitor ferritin in all DHS patients; "
                "PKLR (even mild) — increased gut iron absorption from ineffective erythropoiesis component; "
                "check ferritin annually in all haemolytic anaemia patients regardless of transfusion history."
            ),
            "Parvovirus B19 Aplastic Crisis — Risk in All Haemolytic Anaemias": (
                "Parvovirus B19 (B19V) infects erythroid progenitor cells (via P antigen receptor) → "
                "temporary erythroid aplasia (10-14 days); reticulocyte count drops to near zero; "
                "in healthy individuals: subclinical or mild anaemia (Hb maintained); "
                "in HAEMOLYTIC ANAEMIA (any cause): shortened RBC survival (5-30 days vs. normal 120 days) → "
                "erythroid aplasia causes rapid catastrophic Hb fall; "
                "ALL hereditary haemolytic anaemia patients at risk: HBB (SCD), HBA1 (HbH), "
                "PKLR (PK def), ANK1 (HS), SLC4A1 (HS2), SPTA1 (HPP), G6PD. "
                "DIAGNOSIS: Parvovirus B19 IgM/PCR; reticulocyte count ZERO; PBS shows no reticulocytes; "
                "TREATMENT: supportive transfusion; IVIG if immunocompromised; self-limiting; "
                "lifetime immunity after one episode."
            ),
        },
    }
