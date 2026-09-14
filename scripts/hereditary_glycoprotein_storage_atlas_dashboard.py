"""Hereditary Glycoprotein Storage Atlas — 8-Gene Reference
MAN2B1-MANBA-FUCA1-NEU1-AGA-NAGA-GNPTAB-MCOLN1
320 patients (8 x 40), seeds 2646-2653.
Endpoints: /api/hereditary-glycoprotein-storage-atlas/overview|breakdown|definitions
"""
import random

ATLAS_GENES = [
    {
        "gene": "MAN2B1",
        "protein": (
            "MAN2B1 -- 19p13.13 AR -- 867aa -- Lysosomal-Alpha-D-Mannosidase-114kDa-"
            "Glycoprotein-Catabolism-Alpha-Mannosidosis-AR -- OMIM-Gene-609458-Disease-AlphaMannosidosis-248500"
        ),
        "locus": "19p13.13",
        "protein_size": "867 aa / 114 kDa",
        "inheritance": (
            "AR (biallelic MAN2B1 loss of function); Alpha-mannosidosis; prevalence ~1:500,000; "
            "Lysosomal alpha-D-mannosidase deficiency → accumulation of mannosyl oligosaccharides; "
            "Clinical triad: intellectual disability (mild-moderate) + recurrent infections + facial coarsening; "
            "Onset: infancy to childhood (varies); "
            "Progressive course: hearing loss (SNHL), ataxia, psychiatric symptoms in adolescence; "
            "Vacuolated lymphocytes on blood smear; "
            "Urine oligosaccharides: mannose-rich oligosaccharides diagnostic; "
            "Treatment: Velmanase alfa (ERT, EMA 2018) — first ERT for alpha-mannosidosis; "
            "HSCT: corrects CNS in young patients; mainstay for severe cases; "
            "p.Arg750Trp: common European pathogenic variant"
        ),
        "disease_category": (
            "Alpha-mannosidosis; lysosomal glycoprotein storage disorder; inborn error of glycoprotein catabolism; "
            "MAN2B1 encodes lysosomal acid alpha-D-mannosidase — removes terminal alpha-linked mannose residues "
            "from N-glycan chains during glycoprotein catabolism; "
            "LOF → mannosyl oligosaccharides (Man2-Man6) accumulate in lysosomes and are excreted in urine; "
            "Vacuoles visible in lymphocytes, fibroblasts, hepatocytes (cytoplasmic swelling on bone marrow biopsy); "
            "Enzyme diagnosis: leukocyte or fibroblast MAN2B1 activity <10% normal; "
            "Urine OLIGOsaccharide pattern: mannose-rich (Man2-Man6GlcNAc2) distinguishes from beta-mannosidosis and fucosidosis; "
            "IMMUNE SYSTEM: mannose-rich glycoproteins in immune cells → recurrent bacterial infections (especially Klebsiella, Streptococcus)"
        ),
        "disease_pathway": (
            "MAN2B1 encodes lysosomal alpha-D-mannosidase, a 114 kDa glycoprotein activated by processing to the "
            "A (70 kDa) + B (42 kDa) disulfide-linked form inside lysosomes. "
            "Normal function: sequential removal of alpha-1,2; alpha-1,3; alpha-1,6 mannose residues from "
            "Man8-9GlcNAc2 high-mannose N-glycan cores generated during glycoprotein turnover. "
            "Without MAN2B1: Man2-Man6GlcNAc accumulates in lysosomes → "
            "INTELLECTUAL DISABILITY: lysosomal engorgement in neurons → progressive cognitive impairment (IQ 40-80); "
            "RECURRENT INFECTIONS: abnormal glycoprotein coating of phagocytes impairs killing; "
            "MUSCULOSKELETAL: joint stiffness, kyphosis, osteoporosis; "
            "HEARING LOSS: SNHL in >75%; "
            "PSYCHIATRIC: depression, anxiety, psychosis in adolescence (~25%); "
            "ATAXIA: cerebellar involvement — prominent gait disturbance by adolescence; "
            "FACIAL: mild coarsening with macroglossia, protrusion of jaw; "
            "VELMANASE ALFA (rhLAMAN, EMA 2018): IV every 2 weeks; "
            "reduces storage, improves pulmonary function and motor scores; "
            "most effective when started early; limited CNS penetration."
        ),
        "pathognomonic": (
            "MAN2B1 DIAGNOSTIC CLUSTER: "
            "1) VACUOLATED LYMPHOCYTES on peripheral blood smear — cytoplasmic vacuoles in lymphocytes/monocytes; "
            "seen on routine CBC slide review; not specific but strongly suggestive; "
            "2) URINE OLIGOSACCHARIDES: mannose-rich Man2-Man6GlcNAc2 pattern on TLC/HPLC — "
            "distinguishes from beta-mannosidosis (mannose-GlcNAc disaccharide) and fucosidosis (fucose-containing); "
            "3) CLINICAL TRIAD: intellectual disability + recurrent bacterial infections (especially pulmonary Klebsiella) + "
            "facial coarsening with macroglossia and broad nasal bridge; "
            "4) SNHL: present in >75% by adolescence; often the presenting complaint in mild cases; "
            "5) PSYCHIATRIC SYMPTOMS: schizophrenia-like illness in ~25% adolescents — "
            "alpha-mannosidosis in differential for NEW ONSET PSYCHOSIS with mild ID + infections; "
            "6) LEUKOCYTE/FIBROBLAST ENZYME: MAN2B1 activity <10% normal; "
            "7) MRI: cerebral atrophy + white matter signal on T2-FLAIR (non-specific but present in majority); "
            "DDAVP/DEMOPRESSIN: not indicated; "
            "ERT RESPONSE: motor improvement in walking speed; no reversal of established cognitive deficits"
        ),
        "treatment": (
            "MAN2B1 / ALPHA-MANNOSIDOSIS TREATMENT: "
            "VELMANASE ALFA (Lamzede, EMA 2018): 1 mg/kg IV every 2 weeks; "
            "reduces urinary oligosaccharides; improves pulmonary function (FVC) and 6-minute walk distance; "
            "limited blood-brain barrier penetration — start before significant CNS damage; "
            "HSCT: considered in young (<6yr) severely affected patients; "
            "corrects systemic storage + some CNS effect via engraftment; "
            "most benefit in cognitive/behavioral domain; "
            "SUPPORTIVE: IVIG for recurrent infections (hypogammaglobulinaemia); "
            "hearing aids + cochlear implants for SNHL; "
            "physiotherapy for ataxia and joint disease; "
            "psychiatric management (antipsychotics for psychosis — dose carefully with CNS disease); "
            "MONITORING: urine oligosaccharides; 6MWT; pulmonary function; audiometry; neuropsychology; "
            "GENE THERAPY: AAV-based trials in preclinical stage; "
            "PROGNOSIS: most patients survive to adulthood with variable disability; "
            "respiratory failure from recurrent infections is primary mortality cause"
        ),
        "key_features": [
            "MAN2B1 (Alpha-mannosidosis): AR; lysosomal alpha-D-mannosidase deficiency; 1:500,000; progressive",
            "Clinical triad: intellectual disability (mild-moderate) + recurrent bacterial infections + facial coarsening",
            "VACUOLATED LYMPHOCYTES on blood smear — key screening finding; ask for lymphocyte morphology",
            "URINE OLIGOSACCHARIDES: mannose-rich Man2-Man6GlcNAc2 pattern on TLC — diagnostic",
            "SNHL >75% by adolescence; psychiatric symptoms (psychosis-like) in ~25% adolescents",
            "Velmanase alfa (Lamzede, EMA 2018): IV ERT every 2 weeks; improves motor + pulmonary; limited CNS",
            "HSCT: option in young severely affected patients; corrects systemic + partial CNS storage",
            "p.Arg750Trp: common European pathogenic variant",
        ],
        "key_ddx": [
            "MANBA (beta-mannosidosis): milder/rarer; mannose-GlcNAc disaccharide in urine NOT Man2-6; enzyme assay distinguishes",
            "FUCA1 (fucosidosis): fucose-containing oligosaccharides in urine; angiokeratoma + characteristic MRI changes",
            "Hurler/MPS I (IDUA): dermatan+heparan sulfate on urine GAGs NOT oligosaccharides; corneal clouding; dysostosis multiplex",
            "Mucolipidosis II (GNPTAB / I-cell): no urine oligosaccharides; PLASMA lysosomal enzymes very HIGH; severe neonatal presentation",
            "Sanfilippo (MPS III): heparan sulfate on urine GAGs; no vacuolated lymphocytes; behavioral/sleep dominant",
        ],
    },
    {
        "gene": "MANBA",
        "protein": (
            "MANBA -- 4q22-4q25 AR -- 879aa -- Lysosomal-Beta-D-Mannosidase-100kDa-"
            "Glycoprotein-Catabolism-Beta-Mannosidosis-AR -- OMIM-Gene-609489-Disease-BetaMannosidosis-248510"
        ),
        "locus": "4q22-4q25",
        "protein_size": "879 aa / 100 kDa",
        "inheritance": (
            "AR (biallelic MANBA loss of function); Beta-mannosidosis; very rare — <100 families reported; "
            "Lysosomal beta-D-mannosidase deficiency → accumulation of Man-GlcNAc disaccharide; "
            "Highly variable phenotype: severe neonatal (respiratory failure, hypotonia) to mild adult (angiokeratoma, hearing loss); "
            "Most severe: neonatal respiratory distress + dysmorphic features + profound ID; "
            "Milder: isolated intellectual disability, seizures, angiokeratoma; "
            "Urine oligosaccharides: Man-beta-1,4-GlcNAc disaccharide — diagnostic; "
            "No approved specific therapy; "
            "Management: supportive; "
            "Caprine (goat) beta-mannosidosis: well-characterized animal model"
        ),
        "disease_category": (
            "Beta-mannosidosis; lysosomal glycoprotein storage disorder; terminal step of N-glycan catabolism; "
            "MANBA encodes lysosomal acid beta-D-mannosidase — removes the last beta-1,4-linked mannose from "
            "Man-GlcNAc disaccharide (the final product of alpha-mannosidase action), yielding free mannose + GlcNAc; "
            "Beta-mannosidosis is STEP AFTER alpha-mannosidosis in the N-glycan degradation pathway; "
            "LOF → Man-beta-1,4-GlcNAc accumulates and is excreted in urine; "
            "Lysosomal vacuolization in neurons, macrophages; "
            "Enzyme assay: beta-mannosidase activity in leukocytes or fibroblasts; "
            "Urine: beta-mannobiosyl-di-N-acetylchitobiose — specific but small amount; TLC shows characteristic disaccharide band; "
            "ANGIOKERATOMA: present in milder/adult forms — similar to Fabry/Schindler/fucosidosis"
        ),
        "disease_pathway": (
            "MANBA encodes lysosomal beta-D-mannosidase, responsible for the terminal cleavage of "
            "the core disaccharide Man-beta-1,4-GlcNAc generated after sequential alpha-mannosidase action "
            "on high-mannose N-glycans. "
            "Without beta-mannosidase: Man-beta-1,4-GlcNAc accumulates — "
            "SEVERE NEONATAL FORM: profound hypotonia, respiratory failure from birth, marked dysmorphic features "
            "(hypertelorism, protruding ears, coarse facies), profound intellectual disability; "
            "MODERATE FORM: childhood-onset ID, seizures, recurrent infections, hearing loss; "
            "MILD ADULT FORM: angiokeratoma + peripheral neuropathy + hearing loss + psychiatric symptoms; "
            "The CAPRINE MODEL (goats): severe motor neuron disease, vacuolated cells in CNS; "
            "Molecular mechanism: accumulation of Man-GlcNAc in lysosomes disrupts glycoprotein recycling and "
            "lysosomal membrane integrity in neurons and macrophages; "
            "NO SPECIFIC ERT: unlike alpha-mannosidosis (velmanase alfa), no approved ERT exists; "
            "Very rare: limited natural history data; case series dominate the literature."
        ),
        "pathognomonic": (
            "MANBA DIAGNOSTIC CLUSTER: "
            "1) URINE OLIGOSACCHARIDES: Man-beta-1,4-GlcNAc DISACCHARIDE on TLC/HPLC — "
            "pathognomonic; only two sugars (mannose-GlcNAc dimer), NOT the longer mannose chains of alpha-mannosidosis; "
            "2) BETA-MANNOSIDASE ENZYME ACTIVITY: severely reduced in leukocytes/fibroblasts; "
            "substrate: 4-methylumbelliferyl-beta-D-mannopyranoside; "
            "3) PHENOTYPIC SPECTRUM: "
            "SEVERE END: neonatal respiratory failure + profound hypotonia + coarse facies — "
            "easily mistaken for other lysosomal storage disorders; "
            "MILD END: angiokeratoma (skin lesions, scrotal) + hearing loss + intellectual disability in adults — "
            "resembles Fabry/Schindler/fucosidosis; "
            "4) ANGIOKERATOMA: in milder cases — dark red/purple skin lesions on trunk, scrotum, buttocks; "
            "biopsied: dilated dermal capillaries; similar lesions in Fabry (GLA), fucosidosis (FUCA1), Schindler (NAGA); "
            "5) VACUOLATED LYMPHOCYTES: less prominent than alpha-mannosidosis but present; "
            "DISTINGUISHING FROM ALPHA-MANNOSIDOSIS: disaccharide (beta-mannosidosis) vs oligosaccharide chains (alpha-mannosidosis); "
            "enzyme assay is definitive"
        ),
        "treatment": (
            "MANBA / BETA-MANNOSIDOSIS TREATMENT: "
            "NO APPROVED ERT: unlike alpha-mannosidosis, no enzyme replacement available; "
            "SUPPORTIVE CARE PRIMARY: "
            "Physiotherapy and occupational therapy for motor + developmental delay; "
            "Speech-language therapy; "
            "Antiepileptic drugs for seizures; "
            "Hearing aids for SNHL; "
            "Angiokeratoma: laser therapy (cosmetic); "
            "HSCT: limited data; not standard of care given rarity; "
            "Substrate reduction: no approved drugs; "
            "Gene therapy: preclinical; "
            "MONITORING: urine oligosaccharides; neuropsychological assessment; audiometry; "
            "PROGNOSIS: severe neonatal form — poor; "
            "milder forms: survival to adulthood with intellectual disability; "
            "CAPRINE MODEL: important for therapeutic development"
        ),
        "key_features": [
            "MANBA (Beta-mannosidosis): AR; very rare (<100 families); highly variable phenotype from neonatal lethal to mild adult",
            "Urine DISACCHARIDE Man-beta-1,4-GlcNAc — pathognomonic; TLC shows small band distinct from alpha-mannosidosis",
            "Severe neonatal: respiratory failure + profound hypotonia + coarse facies from birth",
            "Mild adult form: angiokeratoma (scrotal/truncal) + SNHL + intellectual disability — resembles Fabry/Schindler",
            "ANGIOKERATOMA in milder cases: skin lesion DDx includes Fabry (GLA), fucosidosis (FUCA1), Schindler (NAGA)",
            "NO approved ERT unlike alpha-mannosidosis; supportive care only",
            "Beta-mannosidase enzyme assay in leukocytes/fibroblasts: definitive diagnosis",
            "CAPRINE MODEL: well-characterized goat model used for ERT/gene therapy research",
        ],
        "key_ddx": [
            "MAN2B1 (alpha-mannosidosis): oligosaccharide CHAINS in urine not disaccharide; enzyme assay distinguishes; ERT available",
            "GLA (Fabry): angiokeratoma + neuropathic pain; NO urine oligosaccharides; alpha-galactosidase A low; Gb3 in urine",
            "NAGA (Schindler/Kanzaki): angiokeratoma; N-acetylgalactosaminyl oligosaccharides in urine NOT Man-GlcNAc",
            "FUCA1 (fucosidosis): fucose-containing oligosaccharides in urine; facial coarsening; specific TLC pattern",
            "NEU1 (sialidosis): sialyloligosaccharides in urine; cherry-red spot; no angiokeratoma",
        ],
    },
    {
        "gene": "FUCA1",
        "protein": (
            "FUCA1 -- 1p36.11 AR -- 466aa -- Lysosomal-Alpha-L-Fucosidase-53kDa-"
            "Glycoprotein-Catabolism-Fucosidosis-AR -- OMIM-Gene-612280-Disease-Fucosidosis-230000"
        ),
        "locus": "1p36.11",
        "protein_size": "466 aa / 53 kDa",
        "inheritance": (
            "AR (biallelic FUCA1 loss of function); Fucosidosis; prevalence ~1:200,000 (higher in Italy, Cuba, South America); "
            "Lysosomal alpha-L-fucosidase deficiency → fucose-containing glycolipids + glycoproteins accumulate; "
            "Two clinical forms: severe (Type 1) early onset rapid progression vs milder (Type 2) slower course; "
            "Common features: intellectual disability, coarse facies, growth retardation, recurrent infections; "
            "ANGIOKERATOMA CORPORIS DIFFUSUM (Type 2): dark purple skin lesions; pathognomonic in milder forms; "
            "MRI: characteristic T2-signal changes in basal ganglia (globus pallidus) and white matter — prominent; "
            "Urine oligosaccharides: fucose-containing compounds (H-antigen related); "
            "HSCT: corrects neurological progression in young patients; "
            "No approved ERT"
        ),
        "disease_category": (
            "Fucosidosis; lysosomal glycoprotein and glycolipid storage disorder; "
            "FUCA1 encodes lysosomal alpha-L-fucosidase — removes terminal alpha-L-fucose residues from "
            "N-glycan cores, glycolipids, and blood group H-antigen containing molecules; "
            "LOF → fucosylated glycoproteins + glycolipids + blood group H/Ley substances accumulate in lysosomes; "
            "Multiple cell types affected: neurons, hepatocytes, macrophages, skin fibroblasts; "
            "GEOGRAPHIC CLUSTERS: Calabria (Italy), Cuba, central Spain — founder effects; "
            "Urine: fucose-containing oligosaccharides on TLC (H-antigen fragments: Fuc-alpha-1,2-Gal-containing); "
            "Enzyme assay: leukocyte/fibroblast alpha-fucosidase activity; pseudo-deficiency alleles exist; "
            "Radiologic: T2 hyperintensity in globus pallidus — characteristic; seen in other LSDs too"
        ),
        "disease_pathway": (
            "FUCA1 encodes lysosomal alpha-L-fucosidase (53 kDa dimer; active as homodimer), which cleaves "
            "alpha-1,2; alpha-1,3; alpha-1,4; alpha-1,6 fucosyl linkages from: "
            "1) N-glycan chains (terminal fucose on antennae and core fucose on GlcNAc); "
            "2) Glycosphingolipids (H-antigen: Fuc-alpha-1,2-Gal-beta-1,4-GlcNAc-ceramide); "
            "3) Blood group antigens (H, Lewis a/b/x/y-related); "
            "4) Glycopeptides from glycoprotein catabolism. "
            "Without FUCA1: all fucosyl residues accumulate → "
            "NEURONS: progressive lysosomal engorgement → spasticity, intellectual regression, ataxia; "
            "SKIN: fucosylated glycolipids in dermal blood vessels → ANGIOKERATOMA; "
            "LIVER/SPLEEN: mild hepatosplenomegaly from Kupffer cell storage; "
            "SKELETAL: mild dysostosis multiplex; "
            "IMMUNE: recurrent infections; "
            "MRI SIGNAL: globus pallidus T2 hyperintensity ('eye of pallidus') + progressive white matter changes; "
            "SWEAT: increased sweat NaCl concentration (elevated chloride) — diagnostic clue in some patients."
        ),
        "pathognomonic": (
            "FUCA1 DIAGNOSTIC CLUSTER: "
            "1) ANGIOKERATOMA CORPORIS DIFFUSUM (Type 2/milder): dark red-purple skin lesions on trunk, scrotum, "
            "buttocks, thighs — identical distribution to Fabry disease; biopsy shows dilated dermal vessels with storage; "
            "DDx: Fabry (GLA), beta-mannosidosis (MANBA), Schindler (NAGA) — all have angiokeratoma; "
            "2) URINE OLIGOSACCHARIDES: fucose-containing fragments on TLC; H-antigen-related materials "
            "(Fuc-alpha-1,2-Gal-GlcNAc fragments) — TLC pattern distinct from alpha/beta-mannosidosis; "
            "3) MRI BRAIN: T2 HYPERINTENSITY IN GLOBUS PALLIDUS + progressive white matter changes; "
            "may resemble Wilson disease or other metabolic conditions; "
            "4) ELEVATED SWEAT CHLORIDE: increased NaCl in sweat — occasional diagnostic clue; "
            "CF should be excluded first if presenting symptom; "
            "5) COARSE FACIAL FEATURES + ID: present in all forms; "
            "6) GEOGRAPHIC CLUSTERS: Italian (Calabria), Cuban, Spanish — cluster in specific regions; "
            "7) ALPHA-FUCOSIDASE ENZYME ASSAY: severely reduced in leukocytes/fibroblasts; "
            "PSEUDO-DEFICIENCY: FUCA1 pseudo-deficiency variants (e.g., Ala287Thr) cause low enzyme but NO DISEASE — "
            "confirm with urine oligosaccharides + molecular analysis before labelling disease"
        ),
        "treatment": (
            "FUCA1 / FUCOSIDOSIS TREATMENT: "
            "HSCT: current standard of care for early-onset severe cases; "
            "most evidence suggests cognitive stabilisation if performed before significant neurological damage; "
            "young age (<5yr) at transplant = best outcomes; "
            "systemic disease (hepatosplenomegaly) corrected; "
            "neurological benefit variable; "
            "NO APPROVED ERT: unlike some other lysosomal diseases; "
            "SUPPORTIVE: antiepileptic drugs; physiotherapy; speech therapy; special education; "
            "ANGIOKERATOMA: laser therapy (cosmetic); "
            "Recurrent infections: prophylactic antibiotics if hypogammaglobulinaemia; "
            "MONITORING: urine oligosaccharides; neuroimaging (MRI) annually; neuropsychological battery; audiometry; "
            "PROGNOSIS: severe Type 1: death in childhood/early adolescence without HSCT; "
            "milder Type 2: survival to adulthood with intellectual disability; "
            "GENE THERAPY: preclinical animal model studies ongoing"
        ),
        "key_features": [
            "FUCA1 (Fucosidosis): AR; 1:200,000; higher in Calabria/Italy, Cuba; two clinical forms (severe Type 1 / milder Type 2)",
            "ANGIOKERATOMA CORPORIS DIFFUSUM: pathognomonic in Type 2 milder forms; identical to Fabry distribution",
            "Urine oligosaccharides: fucose-containing (H-antigen fragments) on TLC — specific pattern",
            "MRI: T2 HYPERINTENSITY IN GLOBUS PALLIDUS + progressive white matter changes — characteristic",
            "ELEVATED SWEAT CHLORIDE: occasional clue; CF should be excluded first",
            "PSEUDO-DEFICIENCY ALLELES (e.g., Ala287Thr): low enzyme but NO disease — confirm with urine oligosaccharides",
            "HSCT: standard of care for severe early cases; stabilises cognitive decline if performed early",
            "NO approved ERT available (unlike alpha-mannosidosis)",
        ],
        "key_ddx": [
            "GLA (Fabry): angiokeratoma + neuropathic pain; NO urine oligosaccharides; alpha-galactosidase A low; FEMALES also affected",
            "NAGA (Schindler/Kanzaki): angiokeratoma; N-acetylgalactosaminyl compounds in urine NOT fucose-containing",
            "MANBA (beta-mannosidosis): angiokeratoma; Man-GlcNAc disaccharide in urine NOT fucose-containing",
            "MPS I (IDUA): coarse facies + ID; urine GAGs (dermatan+heparan sulfate) NOT oligosaccharides; corneal clouding",
            "Wilson disease (ATP7B): globus pallidus signal on MRI; copper studies + Kayser-Fleischer rings distinguish",
        ],
    },
    {
        "gene": "NEU1",
        "protein": (
            "NEU1 -- 6p21.33 AR -- 415aa -- Lysosomal-Sialidase-Neuraminidase-1-45kDa-"
            "Glycoprotein-Glycolipid-Sialic-Acid-Release-Sialidosis-Galactosialidosis-AR -- OMIM-Gene-608272-Disease-Sialidosis-256550"
        ),
        "locus": "6p21.33",
        "protein_size": "415 aa / 45 kDa",
        "inheritance": (
            "AR (biallelic NEU1 loss of function); Sialidosis (mucolipidosis type I); "
            "prevalence ~1:4,000,000 (rare); "
            "NEU1 encodes sialidase/neuraminidase-1 — removes sialic acid from glycoproteins + glycolipids; "
            "REQUIRES CTSA (protective protein/cathepsin A) for activation; "
            "Type 1 (normosophoric): adult onset; cherry-red spot + myoclonus + normal IQ; "
            "Type 2 (dysmorphic): childhood onset; coarse facies + intellectual disability + cherry-red spot; "
            "Galactosialidosis (CTSA deficiency): combined NEU1+CTSA dysfunction — phenotypically similar to Type 2 sialidosis; "
            "Urine: sialyloligosaccharides on TLC; "
            "Treatment: symptomatic; clonazepam/valproate for myoclonus"
        ),
        "disease_category": (
            "Sialidosis (mucolipidosis I); lysosomal glycoprotein storage disorder; sialic acid metabolism; "
            "NEU1 encodes lysosomal sialidase (neuraminidase-1) — cleaves terminal alpha-2,3 and alpha-2,6 sialyl linkages "
            "from N-glycoproteins, O-glycoproteins, and glycolipids (except GM1 which requires NEU4); "
            "NEU1 exists in a multienzyme complex with CTSA (protective protein/cathepsin A) and beta-galactosidase (GLB1); "
            "CTSA is required for NEU1 activation and stability in lysosomes; "
            "GALACTOSIALIDOSIS = CTSA deficiency → secondary deficiency of BOTH NEU1 (sialidase) AND GLB1 (beta-gal) → "
            "combined sialidosis + GM1-gangliosidosis phenotype; "
            "Urine: sialyloligosaccharides (NeuAc-containing) on TLC — alpha-2,3/2,6-sialyl conjugates; "
            "Lysosomal vacuolization in lymphocytes, liver, spleen, CNS"
        ),
        "disease_pathway": (
            "NEU1 encodes lysosomal neuraminidase-1, which removes terminal sialic acid (N-acetylneuraminic acid, NeuAc) "
            "from N-glycan and O-glycan chains of glycoproteins and glycolipids during lysosomal catabolism. "
            "NEU1 requires CTSA (protective protein/cathepsin A) to form an active complex: "
            "CTSA protects NEU1 from premature degradation and enables its lysosomal targeting. "
            "Without NEU1: sialyloligosaccharides accumulate in lysosomes → "
            "TYPE 1 (normosophoric): adult onset (10-30yr); "
            "CHERRY-RED SPOT: sialic acid storage in retinal ganglion cells → macular degeneration of surrounding cells; "
            "MYOCLONUS: action myoclonus (progressive) — disabling; main clinical problem in Type 1; "
            "NORMAL INTELLECT: distinguishes from Type 2; "
            "TYPE 2 (dysmorphic): childhood onset; "
            "coarse facies + intellectual disability + cherry-red spot + organomegaly; "
            "more severe lysosomal storage in systemic organs; "
            "GALACTOSIALIDOSIS (CTSA): NEU1 + GLB1 both secondarily deficient; "
            "similar to Type 2 sialidosis but also has GLB1 phenotype elements."
        ),
        "pathognomonic": (
            "NEU1 DIAGNOSTIC CLUSTER: "
            "1) CHERRY-RED SPOT + MYOCLONUS (Type 1 — adult): "
            "cherry-red spot on fundoscopy (retinal ganglion cell ring with central red fovea) + "
            "progressive action myoclonus — the cardinal TYPE 1 SIALIDOSIS dyad; "
            "distinguished from Tay-Sachs (HEXA) + GM1 (GLB1) by NORMAL IQ in Type 1 sialidosis; "
            "2) URINE SIALYLOLIGOSACCHARIDES: NeuAc-containing oligosaccharides on TLC — "
            "diagnostic; distinguish from MPS (GAGs), mannosidoses (mannose-rich oligos), fucosidosis (fucose-containing); "
            "3) TYPE 2 DYSMORPHIC FEATURES: childhood coarse facies + corneal clouding (some) + "
            "intellectual disability + cherry-red spot + hepatosplenomegaly; "
            "4) NEU1 ENZYME ASSAY: sialidase activity <10% in leukocytes/fibroblasts; "
            "substrate: 4-MU-NANA (4-methylumbelliferyl-N-acetylneuraminic acid); "
            "5) CTSA EXCLUSION: if both NEU1 + GLB1 activities are low → CTSA deficiency (galactosialidosis); "
            "send CTSA enzyme assay or molecular CTSA/PPCA gene analysis; "
            "CLUE: Type 1 — cherry-red spot in adult with myoclonus and NORMAL intellect → sialidosis first DDx"
        ),
        "treatment": (
            "NEU1 / SIALIDOSIS TREATMENT: "
            "NO SPECIFIC THERAPY APPROVED: no ERT, no SRT approved; "
            "MYOCLONUS MANAGEMENT (primary problem in Type 1): "
            "Clonazepam: first-line for action myoclonus; doses up to 4-6 mg/day; "
            "Valproate: adjunct antiepileptic; "
            "Piracetam: used in cortical myoclonus (off-label); "
            "Levetiracetam: SV2A modulator; effective in cortical myoclonus; "
            "N-acetylcysteine: antioxidant; some benefit in myoclonus; "
            "AVOID zonisamide/topiramate in severe myoclonus; "
            "CHERRY-RED SPOT: no specific treatment; ophthalmology monitoring; "
            "SUPPORTIVE: speech therapy; physiotherapy; "
            "MONITORING: urine oligosaccharides; ophthalmology; EEG (myoclonus monitoring); "
            "neuropsychological assessment; "
            "GENE THERAPY: AAV9 studies in preclinical models; "
            "CTSA (galactosialidosis): same supportive management; "
            "PROGNOSIS: Type 1 — survival to adulthood; progressive disability from myoclonus; "
            "Type 2 — earlier mortality"
        ),
        "key_features": [
            "NEU1 (Sialidosis / Mucolipidosis I): AR; lysosomal sialidase deficiency; requires CTSA for activation",
            "TYPE 1 (normosophoric/adult): CHERRY-RED SPOT + ACTION MYOCLONUS + NORMAL INTELLECT — cardinal triad",
            "TYPE 2 (dysmorphic/childhood): coarse facies + intellectual disability + cherry-red spot + organomegaly",
            "Cherry-red spot + myoclonus + NORMAL IQ in adult → sialidosis first DDx (not Tay-Sachs which has ID)",
            "Urine SIALYLOLIGOSACCHARIDES on TLC: NeuAc-containing oligosaccharides — diagnostic",
            "GALACTOSIALIDOSIS (CTSA deficiency): secondary combined NEU1+GLB1 deficiency; check if both enzymes low",
            "Myoclonus treatment: clonazepam + levetiracetam + piracetam; NO specific ERT available",
            "NEU1 enzyme assay in leukocytes/fibroblasts: <10% activity; substrate 4-MU-NANA",
        ],
        "key_ddx": [
            "HEXA (Tay-Sachs): cherry-red spot + intellectual DECLINE (not normal IQ); hyperacusis; no urine oligosaccharides",
            "GLB1 (GM1-gangliosidosis): cherry-red spot + ID + hepatosplenomegaly; urine has galactosyl oligosaccharides NOT sialyl",
            "CTSA (galactosialidosis): BOTH NEU1+GLB1 low — CTSA assay distinguishes from isolated sialidosis",
            "MPS I (IDUA): coarse facies + ID; urine GAGs (dermatan+heparan) NOT sialyloligosaccharides; corneal clouding",
            "Unverricht-Lundborg disease (CSTB): progressive myoclonus epilepsy; NO cherry-red spot; different gene",
        ],
    },
    {
        "gene": "AGA",
        "protein": (
            "AGA -- 4q34.3 AR -- 346aa -- Lysosomal-Aspartylglucosaminidase-24kDa-"
            "Glycoprotein-Catabolism-Aspartylglucosaminuria-AGU-AR -- OMIM-Gene-608309-Disease-Aspartylglucosaminuria-208400"
        ),
        "locus": "4q34.3",
        "protein_size": "346 aa / 24 kDa (tetrameric complex alpha2beta2)",
        "inheritance": (
            "AR (biallelic AGA loss of function); Aspartylglucosaminuria (AGU); "
            "ENRICHED IN FINLAND: Finnish founder variant p.Cys163Ser (c.488C>G) — prevalence 1:18,000 in Finland; "
            "Worldwide: very rare outside Finland; "
            "Lysosomal aspartylglucosaminidase deficiency → N-linked aspartylglucosamine accumulates; "
            "Distinctive biphasic course: apparently normal infancy + early childhood, then progressive intellectual decline; "
            "Onset: intellectual regression typically 5-10yr; "
            "Progressive ID, speech regression, motor deterioration, psychiatric symptoms; "
            "Coarse facial features appear with age; "
            "No specific therapy; HSCT: limited data; supportive care"
        ),
        "disease_category": (
            "Aspartylglucosaminuria (AGU); lysosomal glycoprotein storage disorder; final step of N-glycan catabolism; "
            "AGA encodes aspartylglucosaminidase (amidase) — cleaves the amide bond between aspartate and "
            "N-acetylglucosamine in the aspartylglucosamine (GlcNAc-Asn) dipeptide, the FINAL step of N-glycan degradation; "
            "This is the last step: after all sugars are removed, GlcNAc remains linked to asparagine; "
            "AGA cleaves this bond yielding free aspartate + free GlcNAc; "
            "LOF → 2-acetamido-1-(β-L-aspartamido)-1,2-dideoxy-β-D-glucose (aspartylglucosamine) accumulates; "
            "FINNISH FOUNDER: p.Cys163Ser allele in >98% Finnish patients; "
            "Non-Finnish: diverse private mutations; "
            "Lysosomal vacuolization in neurons, liver, spleen, lymphocytes"
        ),
        "disease_pathway": (
            "AGA encodes aspartylglucosaminidase (AGA; glycosylasparaginase), a lysosomal amidase that processes "
            "the N-glycoprotein remnant after sequential mannosidase/hexosaminidase/fucosidase action. "
            "The enzyme cleaves the N-glycosyl-asparaginyl amide bond in GlcNAc-Asn to yield "
            "free aspartate + GlcNAc — the FINAL step in N-glycan catabolism. "
            "The active enzyme is an alpha2beta2 heterotetramer: the proenzyme (71 kDa) undergoes autocatalytic "
            "processing to alpha (25 kDa) + beta (17 kDa) subunits within lysosomes. "
            "Without AGA: GlcNAc-Asn (aspartylglucosamine) accumulates in lysosomes and CSF/urine → "
            "NEURAL DEVELOPMENT: initially appears NORMAL (infants develop normally to ~3-5 years); "
            "LATE INFANCY: slight speech delay noticed; "
            "CHILDHOOD (5-10yr): PROGRESSIVE INTELLECTUAL REGRESSION — IQ declines; regression from prior milestones; "
            "ADOLESCENCE/ADULTHOOD: profound intellectual disability; behavioural problems; "
            "coarsening of facial features (NOT prominent in infancy); "
            "SKELETAL: mild dysostosis multiplex; "
            "VACUOLATED LYMPHOCYTES present but not as prominent as alpha-mannosidosis."
        ),
        "pathognomonic": (
            "AGA DIAGNOSTIC CLUSTER: "
            "1) URINE 2-ACETAMIDO-1-(β-L-ASPARTAMIDO)-1,2-DIDEOXY-β-D-GLUCOSE (aspartylglucosamine): "
            "specific metabolite in urine; quantifiable by GC-MS or LC-MS/MS; "
            "plasma aspartylglucosamine also elevated; "
            "TLC of urine: characteristic band; "
            "2) BIPHASIC COURSE — KEY CLINICAL CLUE: "
            "apparently NORMAL EARLY CHILDHOOD (0-5yr) → progressive regression from ~5-10yr; "
            "parents notice REGRESSION FROM PREVIOUSLY NORMAL MILESTONES — not failure to acquire; "
            "3) COARSE FACIAL FEATURES: develop PROGRESSIVELY — NOT prominent in infancy (unlike MPS I/II); "
            "4) VACUOLATED LYMPHOCYTES: present on blood smear; "
            "5) AGA ENZYME ASSAY: aspartylglucosaminidase activity <10% in leukocytes/fibroblasts; "
            "substrate: aspartylglucosamine (GlcNAc-Asn) or fluorogenic substrate; "
            "6) FINNISH CONTEXT: if patient has Finnish ancestry + progressive ID + regression from normal early childhood → "
            "AGU HIGH on differential; "
            "DISTINGUISH: facial coarsening DEVELOPS over years (vs MPS I at birth); no severe organomegaly"
        ),
        "treatment": (
            "AGA / ASPARTYLGLUCOSAMINURIA TREATMENT: "
            "NO SPECIFIC APPROVED THERAPY: no ERT, no approved SRT; "
            "HSCT: limited data in AGU; "
            "animal model data suggests benefit; "
            "human data: case reports of stabilisation; "
            "most benefit if performed before significant neurological damage; "
            "GENE THERAPY: AAV-based approaches in preclinical mouse models; "
            "SUPPORTIVE: special education and behavioural therapy; "
            "antiepileptic drugs if seizures develop; "
            "physiotherapy; "
            "psychosocial support for family (progressive regression is distressing); "
            "MONITORING: urine aspartylglucosamine; neuropsychological assessment annually; MRI; "
            "PROGNOSIS: progressive intellectual decline to profound ID; "
            "most patients survive to 50s-60s with severe disability; "
            "Finnish national registry: life expectancy ~50-60yr; "
            "FINLAND: AGU is one of the Finnish heritage diseases with robust national surveillance"
        ),
        "key_features": [
            "AGA (Aspartylglucosaminuria / AGU): AR; 1:18,000 in Finland (Finnish founder p.Cys163Ser); very rare outside Finland",
            "BIPHASIC COURSE: apparently normal infancy/early childhood → progressive intellectual REGRESSION from 5-10yr",
            "Regression from PREVIOUSLY NORMAL milestones (not failure to acquire) — key clinical clue",
            "Urine ASPARTYLGLUCOSAMINE (GlcNAc-Asn dipeptide): specific metabolite by GC-MS/LC-MS — pathognomonic",
            "Coarse facial features develop PROGRESSIVELY over years (not prominent in infancy unlike MPS I)",
            "Vacuolated lymphocytes on blood smear; AGA enzyme assay in leukocytes/fibroblasts: definitive",
            "Finnish heritage diseases context: p.Cys163Ser c.488C>G allele in >98% Finnish patients",
            "NO approved ERT; HSCT limited data; supportive care and special education primary",
        ],
        "key_ddx": [
            "MAN2B1 (alpha-mannosidosis): similar ID + infections; urine shows mannose-rich oligosaccharides NOT aspartylglucosamine",
            "MPS II/III (IDS/SGSH): regression from normal; urine GAGs (heparan sulfate) NOT aspartylglucosamine",
            "Niemann-Pick C (NPC1): regression; vertical gaze palsy + cataplexy; oxysterols + filipin staining",
            "Mucopolysaccharidosis VII (GUSB): regression; urine GAGs dermatan+heparan+chondroitin; hydrops fetalis history",
            "Wilson disease (ATP7B): psychiatric + cognitive; copper metabolism; Kayser-Fleischer rings; no urine oligosaccharides",
        ],
    },
    {
        "gene": "NAGA",
        "protein": (
            "NAGA -- 22q13.2 AR -- 411aa -- Lysosomal-Alpha-N-Acetylgalactosaminidase-48kDa-"
            "Glycoprotein-Glycolipid-Catabolism-Schindler-Kanzaki-Disease-AR -- OMIM-Gene-104170-Disease-SchindlerDisease-609241"
        ),
        "locus": "22q13.2",
        "protein_size": "411 aa / 48 kDa",
        "inheritance": (
            "AR (biallelic NAGA loss of function); Schindler disease (Type I severe) / Kanzaki disease (Type II milder); "
            "Very rare; "
            "NAGA encodes lysosomal alpha-N-acetylgalactosaminidase — removes terminal alpha-N-acetylgalactosamine from "
            "glycoproteins and glycolipids; "
            "Type I (Schindler): infantile neurodegeneration; severe; presents 1-3yr; "
            "Type II (Kanzaki): adult onset; angiokeratoma + mild intellectual disability + SNHL; "
            "Type III: intermediate severity with intellectual disability; "
            "Urine: N-acetylgalactosaminyl-containing glycopeptides/oligosaccharides; "
            "No approved specific therapy"
        ),
        "disease_category": (
            "Schindler disease / Kanzaki disease; lysosomal glycoprotein storage disorder; "
            "NAGA encodes lysosomal alpha-N-acetylgalactosaminidase — removes alpha-1,3 and alpha-1,6 linked "
            "N-acetylgalactosamine residues from O-glycan chains of glycoproteins (notably from GalNAc-O-Ser/Thr linkages), "
            "glycolipids, and blood group A/B antigens; "
            "LOF → GalNAc-containing glycopeptides and glycolipids accumulate in lysosomes; "
            "PHENOTYPIC SPECTRUM: "
            "TYPE I (Schindler): p.Glu325Lys Dutch allele; rapid neurodegeneration in infancy; "
            "TYPE II (Kanzaki): p.Arg329Gln Japanese allele; adult onset; angiokeratoma; "
            "TYPE III: heterozygous compound; intermediate; "
            "ALPHA-GALACTOSIDASE A (GLA) also has activity on GalNAc substrates → important DDx"
        ),
        "disease_pathway": (
            "NAGA encodes lysosomal alpha-N-acetylgalactosaminidase (NAGA; alpha-galactosidase B), "
            "a 48 kDa lysosomal hydrolase forming homodimers. "
            "NAGA cleaves terminal alpha-linked N-acetylgalactosaminyl residues from: "
            "1) O-glycan chains of glycoproteins (GalNAc-O-Ser/Thr — core 1 and Tn antigen processing); "
            "2) Glycolipids with terminal GalNAc (Forssman antigen, globoside); "
            "3) Blood group A antigen (terminal GalNAc on H antigen) — unlike GLA which cleaves Gal; "
            "Without NAGA: GalNAc-containing glycopeptides accumulate → "
            "TYPE I (SEVERE): infantile-onset neurodegeneration (cortical neurodegenerative disease); "
            "progressive loss of development from 1-3yr; severe intellectual regression; myoclonus, seizures; "
            "autistic features reported; "
            "TYPE II (KANZAKI — MILD): "
            "ANGIOKERATOMA CORPORIS DIFFUSUM: cutaneous vascular lesions; "
            "PERIPHERAL NEUROPATHY: autonomic + sensory; "
            "HEARING LOSS: SNHL; "
            "LYMPHEDEMA: distinctive feature; "
            "mild intellectual disability in some; "
            "FACIAL FEATURES: coarse facies with advancing age."
        ),
        "pathognomonic": (
            "NAGA DIAGNOSTIC CLUSTER: "
            "1) TYPE I (SCHINDLER) — INFANTILE NEURODEGENERATION: "
            "apparently normal development → REGRESSION at 1-3yr; "
            "myoclonus, seizures, cortical blindness, spastic quadriplegia; "
            "autistic features, loss of language; "
            "EEG: multifocal epileptiform activity; "
            "2) TYPE II (KANZAKI) — ADULT ANGIOKERATOMA: "
            "ANGIOKERATOMA CORPORIS DIFFUSUM + SNHL + PERIPHERAL NEUROPATHY + LYMPHEDEMA "
            "— this tetrad in adults should prompt NAGA testing; "
            "angiokeratoma distribution: scrotum, trunk, buttocks (similar to Fabry/fucosidosis); "
            "3) URINE OLIGOSACCHARIDES: N-acetylgalactosaminyl-containing sialyl-glycopeptides — "
            "distinct from other glycoprotein storage disorders; "
            "LC-MS/MS quantitative urine glycopeptide analysis; "
            "4) NAGA ENZYME ASSAY: alpha-N-acetylgalactosaminidase activity <10% in leukocytes/fibroblasts; "
            "BEWARE: GLA (alpha-galactosidase A) has overlapping activity on some substrates — "
            "use specific substrate for NAGA (GalNAc-substrate) vs GLA (Gal-substrate); "
            "5) FOUNDER ALLELES: p.Glu325Lys (Dutch families, severe) / p.Arg329Gln (Japanese families, Kanzaki)"
        ),
        "treatment": (
            "NAGA / SCHINDLER-KANZAKI DISEASE TREATMENT: "
            "NO APPROVED SPECIFIC THERAPY: no ERT approved; "
            "SUPPORTIVE: "
            "Type I (Schindler): antiepileptic drugs for myoclonus + seizures; "
            "nutrition + feeding support (PEG if swallowing impaired); "
            "physiotherapy; "
            "Type II (Kanzaki): angiokeratoma — laser treatment; "
            "hearing aids for SNHL; "
            "support stockings for lymphedema; "
            "pain management for neuropathy; "
            "GENE THERAPY: preclinical; AAV approach in development; "
            "MONITORING: urine GalNAc-glycopeptides; neuropsychological assessment; audiometry; ophthalmology; "
            "PROGNOSIS: "
            "Type I: death in childhood; severe disability; "
            "Type II: survival to adulthood with progressive disability; "
            "RARE DISEASE REGISTRIES: essential given very small patient numbers worldwide"
        ),
        "key_features": [
            "NAGA (Schindler/Kanzaki disease): AR; very rare; lysosomal alpha-N-acetylgalactosaminidase deficiency",
            "TYPE I (Schindler): infantile neurodegeneration; regression at 1-3yr; myoclonus + seizures + cortical blindness",
            "TYPE II (Kanzaki): adult onset; ANGIOKERATOMA + SNHL + peripheral neuropathy + lymphedema",
            "Angiokeratoma + SNHL + neuropathy + lymphedema tetrad in adults → NAGA testing",
            "Urine: N-acetylgalactosaminyl sialyl-glycopeptides; LC-MS/MS quantitation — specific",
            "NAGA enzyme assay: use GalNAc-specific substrate (not GLA/Gal-substrate) to distinguish from Fabry",
            "Founder alleles: p.Glu325Lys (Dutch, severe Type I) / p.Arg329Gln (Japanese, Type II Kanzaki)",
            "NO approved ERT; supportive care; gene therapy preclinical",
        ],
        "key_ddx": [
            "GLA (Fabry disease): angiokeratoma + neuropathic pain; alpha-galactosidase A low; Gb3 urine/urine; X-linked; ERT available",
            "FUCA1 (fucosidosis): angiokeratoma; fucose-containing urine oligosaccharides NOT GalNAc; HSCT option",
            "MANBA (beta-mannosidosis): angiokeratoma; Man-GlcNAc disaccharide in urine NOT GalNAc-peptides",
            "Unverricht-Lundborg/Lafora (progressive myoclonus epilepsies): Type I DDx; no glycoprotein storage; no organomegaly",
            "MPS type II (IDS): infantile regression + coarse features; heparan+dermatan sulfate on urine GAGs NOT glycopeptides",
        ],
    },
    {
        "gene": "GNPTAB",
        "protein": (
            "GNPTAB -- 12q23.2 AR -- 1256aa -- GlcNAc-1-Phosphotransferase-AlphaBeta-Subunit-277kDa-"
            "Lysosomal-Enzyme-Targeting-Mucolipidosis-II-III-ICell-Pseudo-Hurler-AR -- OMIM-Gene-607840-Disease-MucolipidosisII-252500-MLIII-252600"
        ),
        "locus": "12q23.2",
        "protein_size": "1256 aa / 277 kDa (alpha/beta subunits from single gene; GNPTG encodes gamma)",
        "inheritance": (
            "AR (biallelic GNPTAB loss of function); Mucolipidosis II (ML II / I-cell disease) and "
            "Mucolipidosis III alpha-beta (ML IIIab / pseudo-Hurler polydystrophy); "
            "GNPTAB encodes the alpha and beta subunits of GlcNAc-1-phosphotransferase (GNPT); "
            "GNPTG encodes the gamma subunit — mutations cause milder ML III gamma; "
            "GNPT is the enzyme that attaches mannose-6-phosphate (M6P) tags to lysosomal enzymes — required for lysosomal targeting; "
            "ML II: severe; neonatal/infantile onset; very rapid course; "
            "ML III: milder; childhood onset; slower progression; "
            "PATHOGNOMONIC: VERY HIGH plasma lysosomal enzymes (multiple) — lysosomal enzymes secreted into plasma instead of lysosomes"
        ),
        "disease_category": (
            "Mucolipidosis II (I-cell disease) and III; lysosomal enzyme targeting disorder; "
            "GNPTAB encodes GlcNAc-1-phosphotransferase alpha/beta precursor — the enzyme responsible for "
            "attaching mannose-6-phosphate (M6P) recognition signals to lysosomal enzyme precursors in the Golgi; "
            "The M6P signal is REQUIRED for MPR (mannose-6-phosphate receptor) binding and lysosomal targeting of "
            "~60 soluble lysosomal enzymes; "
            "Without M6P tagging: lysosomal enzymes are secreted EXTRACELLULARLY rather than delivered to lysosomes → "
            "INTRACELLULAR: all 60+ lysosomal enzymes absent from cells → MPS + sphingolipid + glycoprotein storage SIMULTANEOUSLY; "
            "EXTRACELLULAR: plasma lysosomal enzyme levels VERY HIGH (10-40× normal) — PATHOGNOMONIC; "
            "I-CELL: cytoplasmic inclusions (dense bodies) in fibroblasts — 'inclusion cells' (I-cells) on fibroblast microscopy"
        ),
        "disease_pathway": (
            "GNPTAB encodes GlcNAc-1-phosphotransferase (GNPT) alpha/beta subunits. "
            "GNPT is a hexameric complex in the trans-Golgi: (alpha/beta)2-(gamma)2. "
            "GNPT catalyses the first step of M6P biosynthesis: "
            "transfer of GlcNAc-1-phosphate to high-mannose N-glycans of >60 lysosomal enzyme precursors → "
            "GlcNAc-1-P-6-Man (uncovered phosphodiester) → processed by phosphodiesterase to Man-6-P; "
            "Man-6-P binds MPR (cation-dependent and cation-independent) → lysosomal delivery. "
            "WITHOUT GNPTAB: no M6P on lysosomal enzymes → "
            "MPR cannot bind → lysosomal enzymes escape via default secretory pathway → "
            "PLASMA: ALL 60+ lysosomal enzymes dramatically elevated (10-40× normal: beta-hexosaminidase, arylsulfatase, "
            "iduronate sulfatase, beta-galactosidase, alpha-fucosidase etc) — PATHOGNOMONIC DIAGNOSTIC FINDING; "
            "CELLS: lysosomal enzymes absent → ALL substrates accumulate simultaneously (GAGs + sphingolipids + glycoproteins); "
            "FIBROBLAST INCLUSIONS: 'I-cells' — phase contrast shows phase-dense cytoplasmic inclusions; "
            "CLINICAL: SEVERE multi-system disease; rapidly fatal in ML II."
        ),
        "pathognomonic": (
            "GNPTAB / ML II-III DIAGNOSTIC CLUSTER: "
            "1) DRAMATICALLY ELEVATED PLASMA LYSOSOMAL ENZYMES (10-40× NORMAL) — PATHOGNOMONIC: "
            "multiple lysosomal enzymes simultaneously very high in plasma "
            "(beta-hexosaminidase, arylsulfatase A, beta-galactosidase, alpha-fucosidase, "
            "iduronate-2-sulfatase, etc) — this pattern of MULTIPLE ELEVATED plasma enzymes is "
            "PATHOGNOMONIC for ML II/III; "
            "CONTRAST: in other LSDs, SPECIFIC enzyme is LOW intracellularly; in ML II/III ALL enzymes are HIGH in plasma + LOW in cells; "
            "2) I-CELLS (INCLUSION CELLS): fibroblasts show phase-dense cytoplasmic inclusions (dense bodies) on "
            "phase contrast microscopy — hallmark giving ML II its alternate name 'I-cell disease'; "
            "3) ML II CLINICAL: severe neonatal Hurler-like features; "
            "NEONATAL onset: coarse facies at birth (distinct from Hurler which appears normal at birth); "
            "severe skeletal dysplasia (dysostosis multiplex); "
            "respiratory failure common in infancy; "
            "NO corneal clouding (CONTRAST WITH HURLER/MPS I); "
            "4) ML III CLINICAL: childhood onset; joint stiffness + carpal tunnel syndrome + skeletal dysostosis; "
            "milder intellectual disability; survival to adulthood; "
            "5) URINE: mixed GAG/oligosaccharide storage; "
            "NOT as clean as specific LSDs"
        ),
        "treatment": (
            "GNPTAB / ML II-III TREATMENT: "
            "ML II (severe): NO effective disease-modifying therapy; "
            "HSCT: not curative and associated with high mortality given disease severity; generally not recommended; "
            "SUPPORTIVE (ML II): "
            "respiratory support (NIV, tracheostomy in late stages); "
            "feeding support (PEG); "
            "physiotherapy; "
            "fracture prevention; "
            "analgesia; "
            "surgical interventions for hydrocephalus, cardiac valve disease; "
            "ML III (milder): "
            "pain management for joint disease; "
            "physiotherapy; "
            "carpal tunnel release; "
            "ERT ATTEMPT: limited — individual lysosomal ERTs have been tried empirically but "
            "cannot correct the underlying targeting defect (cells cannot take up M6P-tagged enzyme without MPR function); "
            "GENE THERAPY: potential but very complex given hexameric complex; "
            "PROGNOSIS: ML II — death typically before 5-10yr from respiratory failure/cardiac disease; "
            "ML III — survival to adulthood (40-50yr) with musculoskeletal disability"
        ),
        "key_features": [
            "GNPTAB (Mucolipidosis II/I-cell disease and ML III): AR; GlcNAc-1-phosphotransferase alpha/beta; lysosomal targeting defect",
            "PATHOGNOMONIC: MULTIPLE PLASMA LYSOSOMAL ENZYMES 10-40× ELEVATED (all simultaneously) — no other LSD does this",
            "I-CELLS: phase-dense cytoplasmic inclusions in fibroblasts on phase contrast — 'inclusion cells' hallmark",
            "ML II (severe): Hurler-like coarse facies AT BIRTH (not normal at birth unlike Hurler); severe skeletal dysplasia; NO corneal clouding",
            "ML III (milder): childhood onset; joint stiffness + carpal tunnel + skeletal disease; milder ID; adult survival",
            "NO specific ERT effective: cells cannot uptake M6P-tagged ERT without functional MPR targeting mechanism",
            "GNPTG (gamma subunit): mutations cause milder ML III gamma — separate gene, similar mechanism",
            "ML II: no HSCT benefit; supportive only; death typically before 5-10yr from respiratory/cardiac disease",
        ],
        "key_ddx": [
            "MPS I (IDUA / Hurler): similar coarse features but NORMAL at birth (not neonatal coarse facies); plasma enzyme LOW not high; GAG urine",
            "GM1-gangliosidosis (GLB1): coarse facies + cherry-red spot; beta-galactosidase LOW in cells (not high in plasma)",
            "Mucolipidosis IV (MCOLN1): different gene; NORMAL plasma enzymes; corneal clouding prominent; different mechanism",
            "Sialidosis (NEU1): NEU1 LOW in cells; plasma NEU1 not dramatically elevated; cherry-red spot + myoclonus",
            "I-cell finding in fibroblast vs mucolipidosis IV: ML IV fibroblasts do NOT show I-cells; ML II fibroblasts DO",
        ],
    },
    {
        "gene": "MCOLN1",
        "protein": (
            "MCOLN1 -- 19p13.2 AR -- 580aa -- Mucolipin-1-TRPML1-65kDa-"
            "Lysosomal-Endosomal-TRP-Channel-Mucolipidosis-IV-AR -- OMIM-Gene-605248-Disease-MucolipidosisIV-252650"
        ),
        "locus": "19p13.2",
        "protein_size": "580 aa / 65 kDa",
        "inheritance": (
            "AR (biallelic MCOLN1 loss of function); Mucolipidosis IV (ML IV); "
            "ENRICHED IN ASHKENAZI JEWISH: two founder variants (p.Arg403Cys and del6.4kb) account for ~95% Ashkenazi alleles; "
            "Ashkenazi carrier frequency ~1:100 → disease prevalence ~1:40,000 in Ashkenazi; "
            "Psychomotor delay from infancy; "
            "CORNEAL CLOUDING: present from birth or early infancy — characteristic + early sign; "
            "ACHLORHYDRIA: gastric acid absent → elevated serum gastrin (>1000 pg/mL) PATHOGNOMONIC BIOMARKER; "
            "RETINAL DEGENERATION: progressive from mid-childhood; "
            "NO MPS/oligosacchariduria: NORMAL urine metabolites; "
            "NORMAL plasma lysosomal enzymes (vs ML II)"
        ),
        "disease_category": (
            "Mucolipidosis IV; lysosomal TRP channel defect; NOT a true mucolipidosis (name is historical misnomer); "
            "MCOLN1 encodes mucolipin-1 (TRPML1) — a lysosomal transient receptor potential (TRP) cation channel; "
            "TRPML1 is responsible for calcium release from lysosomes required for: "
            "lysosomal exocytosis, membrane trafficking/fusion, autophagy regulation, pH homeostasis; "
            "LOF → impaired lysosomal exocytosis + autophagy → storage of phospholipids, sphingolipids, gangliosides; "
            "DISTINCT FROM OTHER MUCOLIPIDOSES (ML I/II/III): "
            "ML I (NEU1) = sialidase deficiency; ML II/III (GNPTAB) = lysosomal targeting defect; "
            "ML IV (MCOLN1) = lysosomal TRP channel — COMPLETELY DIFFERENT MECHANISM; "
            "Urine: NORMAL (no oligosaccharides, no GAGs) — important differentiator; "
            "PLASMA LYSOSOMAL ENZYMES: NORMAL (vs dramatically elevated in ML II)"
        ),
        "disease_pathway": (
            "MCOLN1 encodes mucolipin-1 (TRPML1), a member of the transient receptor potential (TRP) superfamily, "
            "localised to late endosomal/lysosomal membranes (not plasma membrane). "
            "TRPML1 is a Ca2+-permeable non-selective cation channel activated by "
            "PI(3,5)P2 (a lysosomal-specific phosphoinositide) and stimulated by membrane tension/volume. "
            "FUNCTIONS: "
            "1) Ca2+ release from lysosome → triggers lysosomal exocytosis (membrane fusion with plasma membrane); "
            "2) Regulates endo-lysosomal trafficking (retrograde transport); "
            "3) Required for efficient autophagy (autophagosome-lysosome fusion); "
            "4) Lysosomal pH homeostasis and membrane integrity. "
            "WITHOUT MCOLN1: "
            "lysosomal exocytosis impaired → storage of lipids (phospholipids, sphingolipids, gangliosides) builds up; "
            "autophagy defective → damaged organelles accumulate; "
            "CORNEAL CELLS: lipid storage in corneal epithelia → CORNEAL CLOUDING (early, prominent); "
            "NEURONS: progressive lipid accumulation → psychomotor delay + regression; "
            "GASTRIC PARIETAL CELLS: impaired secretion machinery → ACHLORHYDRIA + HIGH GASTRIN."
        ),
        "pathognomonic": (
            "MCOLN1 / ML IV DIAGNOSTIC CLUSTER: "
            "1) CORNEAL CLOUDING: present from birth or early infancy in >90%; "
            "appears as bilateral corneal haziness/opacity; slit-lamp confirms stromal lipid deposits; "
            "DISTINCT FROM MPS I (Hurler): ML IV corneal clouding appears EARLIER (infancy) vs MPS I (after 6-12 months); "
            "also distinct from cystinosis (crystal deposits), Wilson (Kayser-Fleischer); "
            "2) ACHLORHYDRIA WITH ELEVATED SERUM GASTRIN: "
            "serum gastrin >1000 pg/mL (often >2000-3000) in fasting state — PATHOGNOMONIC BIOMARKER for ML IV; "
            "TRPML1 required for parietal cell acid secretion; loss → achlorhydria → compensatory hypergastrinaemia; "
            "CHECK SERUM GASTRIN IN ANY UNDIAGNOSED INTELLECTUAL DISABILITY + CORNEAL CLOUDING; "
            "3) PSYCHOMOTOR DELAY: global developmental delay from infancy; "
            "4) RETINAL DEGENERATION: rod+cone dystrophy progressing from ~3-5yr; "
            "5) ASHKENAZI JEWISH ANCESTRY: carrier testing available for p.Arg403Cys and del6.4kb; "
            "6) URINE/PLASMA: NORMAL metabolites (no GAGs, no oligosaccharides, NORMAL plasma lysosomal enzymes) — "
            "KEY DIFFERENTIATOR from ML II and MPS; "
            "7) MRI: corpus callosum hypoplasia; T2 signal in posterior white matter"
        ),
        "treatment": (
            "MCOLN1 / ML IV TREATMENT: "
            "NO APPROVED SPECIFIC THERAPY: no ERT, no SRT; "
            "GENE THERAPY: most promising approach given channel protein defect; "
            "AAV9 and intrathecal delivery approaches in preclinical development; "
            "N-ACETYLCYSTEINE: antioxidant trial showed modest benefit in mouse model; limited human data; "
            "SUPPORTIVE: "
            "physiotherapy + occupational therapy for psychomotor delay; "
            "ophthalmologic care: corneal transplant in some severe cases; low-vision aids; "
            "visual aids + Braille + adaptive technology for retinal degeneration; "
            "gastroenterology: nutritional support for achlorhydria (iron absorption may be impaired); "
            "B12 supplementation (impaired intrinsic factor-independent absorption due to achlorhydria); "
            "antiepileptic drugs if seizures develop; "
            "MONITORING: ophthalmology (cornea + retina) every 6 months; "
            "serum gastrin + iron + B12; "
            "neuropsychological assessment; "
            "PROGNOSIS: non-progressive CNS course in most (stable, not declining after initial delay); "
            "significant disability but survival to adulthood; "
            "ASHKENAZI CARRIER SCREENING: recommended in Ashkenazi communities"
        ),
        "key_features": [
            "MCOLN1 (Mucolipidosis IV): AR; Ashkenazi Jewish enriched (1:40,000); TRPML1 lysosomal TRP cation channel",
            "CORNEAL CLOUDING: present from birth/early infancy >90% — EARLY prominent feature; slit-lamp stromal opacity",
            "ACHLORHYDRIA + SERUM GASTRIN >1000 pg/mL: PATHOGNOMONIC BIOMARKER — check gastrin in any undiagnosed ID + corneal clouding",
            "Psychomotor delay from infancy + progressive retinal degeneration from ~3-5yr",
            "NORMAL urine metabolites (no GAGs, no oligosaccharides) + NORMAL plasma lysosomal enzymes — key differentiators",
            "DISTINCT MECHANISM from ML I/II/III: TRP channel defect not enzyme deficiency or targeting defect",
            "Ashkenazi founder alleles: p.Arg403Cys + del6.4kb — 95% of Ashkenazi MCOLN1 alleles; carrier frequency ~1:100",
            "Non-progressive CNS after initial delay (stable, not regressive); no approved specific therapy; gene therapy preclinical",
        ],
        "key_ddx": [
            "GNPTAB (ML II / I-cell): plasma lysosomal enzymes 10-40× elevated (vs NORMAL in ML IV); I-cells in fibroblasts; no corneal clouding neonatal",
            "MPS I (IDUA / Hurler): corneal clouding but urine GAGs (dermatan+heparan sulfate) + plasma lysosomal enzyme LOW; ERT available",
            "Cystinosis (CTNS): corneal crystals (not clouding); photophobia; Fanconi syndrome; urine amino acids; cysteamine treatment",
            "Congenital cataract syndromes: lens opacity NOT corneal stromal clouding; different pathology on slit-lamp",
            "NEU1 (sialidosis Type 2): corneal clouding reported; ALSO cherry-red spot + urine sialyloligosaccharides; enzyme assay distinguishes",
        ],
    },
]


def _generate_patients(gene_idx: int, n: int = 40, seed: int = 0):
    rng = random.Random(seed)
    gene = ATLAS_GENES[gene_idx]
    g = gene["gene"]
    patients = []
    for i in range(n):
        onset_base = {
            "MAN2B1": (2.0, 1.5), "MANBA": (1.5, 2.0), "FUCA1": (2.5, 2.0),
            "NEU1": (5.0, 4.0), "AGA": (6.0, 2.0), "NAGA": (4.0, 3.5),
            "GNPTAB": (0.3, 0.3), "MCOLN1": (0.5, 0.5),
        }
        mu, sd = onset_base.get(g, (3.0, 2.0))
        onset = max(0.0, round(rng.gauss(mu, sd), 1))

        hearing_probs = {
            "MAN2B1": 0.80, "MANBA": 0.45, "FUCA1": 0.55, "NEU1": 0.40,
            "AGA": 0.30, "NAGA": 0.50, "GNPTAB": 0.65, "MCOLN1": 0.15,
        }
        corneal_probs = {
            "MAN2B1": 0.15, "MANBA": 0.10, "FUCA1": 0.20, "NEU1": 0.30,
            "AGA": 0.10, "NAGA": 0.10, "GNPTAB": 0.05, "MCOLN1": 0.95,
        }
        angiokeratoma_probs = {
            "MAN2B1": 0.05, "MANBA": 0.45, "FUCA1": 0.50, "NEU1": 0.05,
            "AGA": 0.05, "NAGA": 0.55, "GNPTAB": 0.05, "MCOLN1": 0.05,
        }
        id_probs = {
            "MAN2B1": 0.90, "MANBA": 0.75, "FUCA1": 0.85, "NEU1": 0.60,
            "AGA": 0.92, "NAGA": 0.70, "GNPTAB": 0.95, "MCOLN1": 0.85,
        }
        organomegaly_probs = {
            "MAN2B1": 0.50, "MANBA": 0.35, "FUCA1": 0.60, "NEU1": 0.55,
            "AGA": 0.40, "NAGA": 0.30, "GNPTAB": 0.80, "MCOLN1": 0.20,
        }
        vacuolated_probs = {
            "MAN2B1": 0.90, "MANBA": 0.60, "FUCA1": 0.75, "NEU1": 0.65,
            "AGA": 0.70, "NAGA": 0.55, "GNPTAB": 0.85, "MCOLN1": 0.20,
        }

        patients.append({
            "patient_id": f"{g}-{2646 + gene_idx:04d}-{i+1:03d}",
            "gene": g,
            "seed": 2646 + gene_idx,
            "onset_age": onset,
            "hearing_loss": rng.random() < hearing_probs.get(g, 0.40),
            "corneal_clouding": rng.random() < corneal_probs.get(g, 0.15),
            "angiokeratoma": rng.random() < angiokeratoma_probs.get(g, 0.10),
            "intellectual_disability": rng.random() < id_probs.get(g, 0.80),
            "hepatosplenomegaly": rng.random() < organomegaly_probs.get(g, 0.45),
            "vacuolated_lymphocytes": rng.random() < vacuolated_probs.get(g, 0.70),
        })
    return patients


def generate_overview():
    all_patients = []
    for idx in range(len(ATLAS_GENES)):
        all_patients.extend(_generate_patients(idx, n=40, seed=2646 + idx))

    summaries = []
    for idx, gene_def in enumerate(ATLAS_GENES):
        pts = [p for p in all_patients if p["gene"] == gene_def["gene"]]
        n = len(pts)
        summaries.append({
            "gene": gene_def["gene"],
            "locus": gene_def["locus"],
            "n_patients": n,
            "avg_onset_age": round(sum(p["onset_age"] for p in pts) / n, 1),
            "hearing_loss_pct": round(100 * sum(p["hearing_loss"] for p in pts) / n),
            "corneal_clouding_pct": round(100 * sum(p["corneal_clouding"] for p in pts) / n),
            "angiokeratoma_pct": round(100 * sum(p["angiokeratoma"] for p in pts) / n),
            "intellectual_disability_pct": round(100 * sum(p["intellectual_disability"] for p in pts) / n),
        })

    n_total = len(all_patients)
    return {
        "atlas": "Hereditary Glycoprotein Storage Atlas",
        "genes": [g["gene"] for g in ATLAS_GENES],
        "n_genes": len(ATLAS_GENES),
        "total_patients": n_total,
        "seeds": "2646-2653",
        "aggregate_stats": {
            "overall_hearing_loss_pct": round(100 * sum(p["hearing_loss"] for p in all_patients) / n_total),
            "overall_corneal_clouding_pct": round(100 * sum(p["corneal_clouding"] for p in all_patients) / n_total),
            "overall_angiokeratoma_pct": round(100 * sum(p["angiokeratoma"] for p in all_patients) / n_total),
            "overall_intellectual_disability_pct": round(100 * sum(p["intellectual_disability"] for p in all_patients) / n_total),
            "overall_hepatosplenomegaly_pct": round(100 * sum(p["hepatosplenomegaly"] for p in all_patients) / n_total),
            "overall_vacuolated_lymphocytes_pct": round(100 * sum(p["vacuolated_lymphocytes"] for p in all_patients) / n_total),
        },
        "gene_summaries": summaries,
        "disease_classes": [
            "MAN2B1 — Alpha-mannosidosis — AR 19p13.13 — Lysosomal alpha-mannosidase — mannose-rich oligosacchariduria — vacuolated lymphocytes — recurrent infections — velmanase alfa ERT (EMA 2018)",
            "MANBA — Beta-mannosidosis — AR 4q22-4q25 — Lysosomal beta-mannosidase — Man-GlcNAc disaccharide in urine — severe neonatal to mild adult angiokeratoma — no ERT",
            "FUCA1 — Fucosidosis — AR 1p36.11 — Lysosomal alpha-fucosidase — fucose-containing oligosacchariduria — angiokeratoma + globus pallidus T2 — pseudo-deficiency alleles — HSCT",
            "NEU1 — Sialidosis (ML I) — AR 6p21.33 — Lysosomal sialidase (requires CTSA) — sialyloligosacchariduria — Type 1 (adult: cherry-red + myoclonus + NORMAL IQ) / Type 2 (childhood dysmorphic)",
            "AGA — Aspartylglucosaminuria — AR 4q34.3 — Aspartylglucosaminidase — GlcNAc-Asn dipeptide in urine — Finnish founder — biphasic course (normal infancy → regression 5-10yr)",
            "NAGA — Schindler/Kanzaki disease — AR 22q13.2 — Lysosomal alpha-N-acetylgalactosaminidase — GalNAc-glycopeptides in urine — Type I (infantile neurodegeneration) / Type II (adult angiokeratoma)",
            "GNPTAB — Mucolipidosis II (I-cell) / ML III — AR 12q23.2 — GlcNAc-1-phosphotransferase — lysosomal targeting defect — MULTIPLE PLASMA LYSOSOMAL ENZYMES 10-40× ELEVATED — I-cells in fibroblasts",
            "MCOLN1 — Mucolipidosis IV — AR 19p13.2 — TRPML1 lysosomal TRP channel — NORMAL urine/plasma enzymes — corneal clouding from birth — gastrin >1000 PATHOGNOMONIC — Ashkenazi founder",
        ],
        "key_clinical_distinctions": [
            "URINE OLIGOSACCHARIDES PATTERN: mannose-rich chains → alpha-mannosidosis (MAN2B1); Man-GlcNAc disaccharide → beta-mannosidosis (MANBA); fucose-containing → fucosidosis (FUCA1); sialyl-containing → sialidosis (NEU1); aspartylglucosamine dipeptide → AGU (AGA); GalNAc-glycopeptides → Schindler/Kanzaki (NAGA); NORMAL → ML II (GNPTAB) or ML IV (MCOLN1)",
            "ANGIOKERATOMA DIFFERENTIAL: MANBA + FUCA1 + NAGA all cause angiokeratoma; also GLA (Fabry); enzyme assays + urine pattern distinguish; no angiokeratoma in MAN2B1, NEU1, AGA, GNPTAB, MCOLN1",
            "PLASMA LYSOSOMAL ENZYMES: DRAMATICALLY ELEVATED (10-40×) only in GNPTAB (ML II/III — I-cell disease); all others have NORMAL plasma levels; I-cell disease is the sole exception in this group",
            "CORNEAL CLOUDING: MCOLN1 (ML IV) has prominent early corneal clouding from infancy; FUCA1 has mild corneal change; GNPTAB (ML II) has NO corneal clouding — CONTRAST with Hurler (MPS I) which has clouding",
            "CHERRY-RED SPOT: NEU1 (sialidosis) — Type 1 adult onset; HEXA/HEXB/GLB1/SMPD1 also have cherry-red spot; not seen in alpha/beta-mannosidosis, fucosidosis, AGU, Schindler, ML II-IV",
            "SERUM GASTRIN >1000 pg/mL PATHOGNOMONIC for MCOLN1 (ML IV): achlorhydria from TRPML1 loss in parietal cells; SCREEN SERUM GASTRIN in any infant with corneal clouding + psychomotor delay",
            "ASHKENAZI JEWISH FOUNDER EFFECTS: MCOLN1 (ML IV) p.Arg403Cys + del6.4kb; AGA p.Cys163Ser is Finnish founder (NOT Ashkenazi); carrier screening programmes differ",
            "VACUOLATED LYMPHOCYTES on blood smear: most prominent in MAN2B1; present in MANBA/FUCA1/NEU1/AGA/NAGA/GNPTAB; absent or minimal in MCOLN1",
        ],
    }


def generate_breakdown():
    all_patients = []
    for idx in range(len(ATLAS_GENES)):
        all_patients.extend(_generate_patients(idx, n=40, seed=2646 + idx))

    breakdowns = []
    for gene_def in ATLAS_GENES:
        g = gene_def["gene"]
        pts = [p for p in all_patients if p["gene"] == g]
        n = len(pts)
        feature_rates = {
            "hearing_loss": round(100 * sum(p["hearing_loss"] for p in pts) / n),
            "corneal_clouding": round(100 * sum(p["corneal_clouding"] for p in pts) / n),
            "angiokeratoma": round(100 * sum(p["angiokeratoma"] for p in pts) / n),
            "intellectual_disability": round(100 * sum(p["intellectual_disability"] for p in pts) / n),
            "hepatosplenomegaly": round(100 * sum(p["hepatosplenomegaly"] for p in pts) / n),
            "vacuolated_lymphocytes": round(100 * sum(p["vacuolated_lymphocytes"] for p in pts) / n),
        }
        breakdowns.append({
            "gene": g,
            "locus": gene_def["locus"],
            "protein_size": gene_def["protein_size"],
            "n_patients": n,
            "avg_onset_age": round(sum(p["onset_age"] for p in pts) / n, 1),
            "inheritance": gene_def["inheritance"],
            "disease_category": gene_def["disease_category"],
            "disease_pathway": gene_def["disease_pathway"],
            "pathognomonic": gene_def["pathognomonic"],
            "treatment": gene_def["treatment"],
            "key_features": gene_def["key_features"],
            "key_ddx": gene_def["key_ddx"],
            "feature_rates": feature_rates,
        })
    return {"gene_breakdowns": breakdowns}


def generate_definitions():
    gene_entries = {}
    for gene_def in ATLAS_GENES:
        g = gene_def["gene"]
        gene_entries[g] = {
            "disease_name": gene_def["protein"].split("--")[0].strip() + " — " + gene_def["disease_category"].split(";")[0].strip(),
            "locus": gene_def["locus"],
            "protein_size": gene_def["protein_size"],
            "inheritance": gene_def["inheritance"].split(";")[0].strip(),
            "disease_pathway": gene_def["disease_pathway"],
            "pathognomonic": gene_def["pathognomonic"],
        }

    glossary = {
        "Lysosomal Glycoprotein Storage Disorders (LGPSDs)": (
            "A subgroup of lysosomal storage diseases caused by deficiency of enzymes required for the stepwise "
            "degradation of N-glycan and O-glycan chains of glycoproteins and glycolipids. "
            "Unlike mucopolysaccharidoses (which store glycosaminoglycans) or sphingolipidoses (which store complex lipids), "
            "LGPSDs store oligosaccharide fragments of glycoproteins. "
            "DIAGNOSTIC CORNERSTONE: urine oligosaccharide pattern on TLC or LC-MS/MS. "
            "Key diseases: alpha-mannosidosis (MAN2B1), beta-mannosidosis (MANBA), fucosidosis (FUCA1), "
            "sialidosis (NEU1), aspartylglucosaminuria (AGA), Schindler/Kanzaki (NAGA). "
            "PLUS two disorders with abnormal lysosomal function but normal/altered urine metabolites: "
            "ML II/III (GNPTAB — lysosomal targeting defect) and ML IV (MCOLN1 — lysosomal TRP channel)."
        ),
        "N-Glycan Catabolism Pathway": (
            "Sequential lysosomal degradation of N-linked glycans released during glycoprotein turnover: "
            "1. Endoglycosidases release N-glycan chains from glycoproteins; "
            "2. NEU1 (sialidase) removes sialic acid from NeuAc-Gal; "
            "3. GLB1 (beta-galactosidase) removes galactose; "
            "4. HEXA/HEXB (hexosaminidase) removes GlcNAc from antennae; "
            "5. MAN2B1 (alpha-mannosidase) removes alpha-mannose residues (Man2-Man9 → Man); "
            "6. MANBA (beta-mannosidase) removes the final beta-1,4 mannose from Man-GlcNAc; "
            "7. FUCA1 (fucosidase) removes core and antenna fucose at various steps; "
            "8. AGA (aspartylglucosaminidase) cleaves GlcNAc-Asn dipeptide — FINAL STEP. "
            "Deficiency at any step leads to accumulation of the substrate for that specific enzyme."
        ),
        "Urine Oligosaccharide Analysis (TLC/LC-MS)": (
            "Gold standard screening test for lysosomal glycoprotein storage disorders. "
            "COLLECTION: random or first-morning urine; no special preparation. "
            "TLC METHOD: urine concentrated, applied to silica gel plate; orcinol stain visualises oligosaccharides. "
            "LC-MS/MS METHOD: quantitative; identifies specific oligosaccharide structures. "
            "PATTERNS: "
            "Alpha-mannosidosis (MAN2B1): Man2-Man6GlcNAc2 chains; "
            "Beta-mannosidosis (MANBA): Man-beta-1,4-GlcNAc disaccharide (small, distinct band); "
            "Fucosidosis (FUCA1): fucose-containing fragments (H-antigen derived); "
            "Sialidosis (NEU1): NeuAc-containing sialyloligosaccharides; "
            "AGU (AGA): GlcNAc-Asn dipeptide (aspartylglucosamine); "
            "Schindler (NAGA): GalNAc-sialyl-glycopeptides; "
            "ML II (GNPTAB): may show mixed storage but not clean oligosaccharide pattern; "
            "ML IV (MCOLN1): NORMAL urine — no oligosaccharides."
        ),
        "Mannose-6-Phosphate (M6P) Lysosomal Targeting": (
            "The M6P recognition system routes ~60 soluble lysosomal hydrolases to lysosomes. "
            "MECHANISM: "
            "1. Newly synthesised lysosomal enzyme precursors carry high-mannose N-glycans; "
            "2. GNPTAB (GlcNAc-1-phosphotransferase) adds GlcNAc-1-phosphate to mannose residues in trans-Golgi; "
            "3. Phosphodiesterase removes GlcNAc, exposing mannose-6-phosphate (M6P); "
            "4. M6P receptors (MPR46 and MPR300) in trans-Golgi bind M6P-tagged enzymes; "
            "5. MPR-enzyme complexes delivered to late endosomes/lysosomes; M6P hydrolysed at low pH. "
            "GNPTAB LOF → no M6P tags → lysosomal enzymes secreted extracellularly → "
            "DRAMATICALLY ELEVATED PLASMA LYSOSOMAL ENZYMES (pathognomonic for ML II/III) + "
            "cellular storage of all lysosomal substrates simultaneously."
        ),
        "TRPML1 Channel (MCOLN1)": (
            "Transient receptor potential mucolipin 1 (TRPML1) is the lysosomal Ca2+-channel encoded by MCOLN1. "
            "LOCALISATION: predominantly late endosomal/lysosomal membranes. "
            "ACTIVATION: PI(3,5)P2 (lysosome-specific phosphoinositide); membrane tension. "
            "FUNCTIONS: "
            "1. Ca2+ release from lysosomal lumen → triggers lysosomal exocytosis; "
            "2. Endo-lysosomal retrograde trafficking; "
            "3. Autophagosome-lysosome fusion (autophagy completion); "
            "4. Lysosomal membrane repair. "
            "MCOLN1 LOF → impaired lysosomal exocytosis → lipid accumulation (phospholipids, gangliosides); "
            "UNIQUE BIOMARKER: achlorhydria (TRPML1 required for gastric parietal cell proton pump secretion) → "
            "ELEVATED SERUM GASTRIN >1000 pg/mL — specific diagnostic marker for ML IV."
        ),
        "Angiokeratoma Corporis Diffusum — Differential Diagnosis": (
            "Angiokeratoma = dilated dermal blood vessels with overlying acanthotic/hyperkeratotic epidermis. "
            "CORPORIS DIFFUSUM = widespread distribution on trunk/scrotum/buttocks/thighs. "
            "LYSOSOMAL STORAGE DISORDERS CAUSING ANGIOKERATOMA: "
            "GLA (Fabry disease): XLR; alpha-galactosidase A low; Gb3 elevated in urine/plasma; neuropathic pain; "
            "FUCA1 (Fucosidosis): fucose-containing oligosaccharides in urine; globus pallidus T2; "
            "MANBA (Beta-mannosidosis): Man-GlcNAc disaccharide in urine; milder form; "
            "NAGA (Schindler/Kanzaki): GalNAc-glycopeptides in urine; Type II adult onset; "
            "MAN2B1 (Alpha-mannosidosis): RARE angiokeratoma; prominent infections + vacuolated lymphocytes; "
            "SMPD1 (Niemann-Pick B): mild foam cells; pulmonary disease; "
            "SCREENING APPROACH: enzyme assays for GLA (first, most common) + urine oligosaccharides for others."
        ),
        "Vacuolated Lymphocytes": (
            "Cytoplasmic vacuoles in peripheral blood lymphocytes and monocytes visible on routine Giemsa/Wright stain. "
            "MECHANISM: lysosomal engorgement with undegraded substrate expands cytoplasm; "
            "PAS-positive vacuoles in many LSDs. "
            "LSDs WITH VACUOLATED LYMPHOCYTES: MAN2B1 (most prominent), MANBA, FUCA1, NEU1, AGA, NAGA, GNPTAB, "
            "NPC1, pompe (GSD II), mucopolysaccharidoses. "
            "ABSENT or MINIMAL in: MCOLN1 (ML IV), GLA (Fabry), HEXA (Tay-Sachs). "
            "CLINICAL UTILITY: SCREENING TOOL — ask for lymphocyte morphology on CBC if LSD suspected; "
            "NOT diagnostic alone; proceed to enzyme assays + urine metabolites if vacuoles found."
        ),
        "Galactosialidosis vs Sialidosis": (
            "SIALIDOSIS (NEU1 deficiency): isolated NEU1 loss of function → sialyloligosacchariduria; "
            "Type 1 (adult, cherry-red + myoclonus, normal IQ) / Type 2 (childhood, dysmorphic). "
            "GALACTOSIALIDOSIS (CTSA deficiency): CTSA (protective protein/cathepsin A) deficiency → "
            "secondary deficiency of BOTH NEU1 (sialidase) AND GLB1 (beta-galactosidase) because "
            "CTSA forms a complex with both enzymes protecting them from intralysosomal degradation. "
            "GALACTOSIALIDOSIS phenotype: combined sialidosis + GM1 features; "
            "urine: BOTH sialyloligosaccharides + galactosyl oligosaccharides; "
            "DIAGNOSTIC: if BOTH NEU1 + GLB1 are low → check CTSA (PPCA) gene/enzyme; "
            "if only NEU1 low → isolated sialidosis (NEU1 mutation)."
        ),
    }

    return {
        "atlas": "Hereditary Glycoprotein Storage Atlas",
        "gene_entries": gene_entries,
        "glycoprotein_storage_glossary": glossary,
    }
