"""Hereditary Mucopolysaccharidosis (MPS) Atlas — 8-Gene Reference
IDUA-IDS-SGSH-GALNS-ARSB-GUSB-NAGLU-HGSNAT
320 patients (8 x 40), seeds 2630-2637.
Endpoints: /api/hereditary-mps-atlas/overview|breakdown|definitions
"""
import random

ATLAS_GENES = [
    {
        "gene": "IDUA",
        "protein": (
            "IDUA -- 4p16.3 AR -- 653aa -- Alpha-L-Iduronidase-82kDa-"
            "Lysosomal-Dermatan+Heparan-Sulfate-Hydrolase-MPS-I-AR -- OMIM-Gene-252800-Disease-Hurler-Scheie-607015"
        ),
        "locus": "4p16.3",
        "protein_size": "653 aa / 82 kDa",
        "inheritance": (
            "AR (biallelic IDUA loss of function); MPS I (Hurler/Hurler-Scheie/Scheie); "
            "Alpha-L-iduronidase deficiency; prevalence 1:100,000; "
            "Hurler (MPS IH): severe; null/null mutations; no residual enzyme; "
            "corneal clouding, coarse facies, hepatosplenomegaly, dysostosis multiplex, cardiomyopathy, severe ID; "
            "death by age 10 without treatment; "
            "Hurler-Scheie (MPS IH/S): intermediate; corneal clouding + skeletal but slower progression; "
            "Scheie (MPS IS): attenuated; corneal clouding + joint stiffness; near-normal intellect; "
            "W402X and Q70X: most common severe alleles (Hurler); "
            "No genotype-phenotype correlation for intermediate alleles — phenotypic prediction unreliable"
        ),
        "disease_category": (
            "MPS I (Hurler/Hurler-Scheie/Scheie); lysosomal storage disorder; mucopolysaccharidosis; "
            "IDUA encodes alpha-L-iduronidase — lysosomal hydrolase cleaving alpha-L-iduronic acid "
            "terminal residues from dermatan sulfate (DS) and heparan sulfate (HS); "
            "Loss of IDUA → DS + HS accumulate in lysosomes of connective tissue, brain, heart valves, "
            "liver, spleen, bones, cornea; "
            "LARONIDASE (Aldurazyme) — ERT FDA 2003; first approved ERT for MPS; "
            "Treats visceral disease; does NOT cross blood-brain barrier; "
            "HSCT: curative for CNS disease if performed before age 2-3 and before cognitive decline; "
            "Preferred for Hurler: HSCT + ERT bridge; Hurler-Scheie/Scheie: ERT alone"
        ),
        "disease_pathway": (
            "IDUA encodes alpha-L-iduronidase, a lysosomal acid hydrolase that cleaves the terminal "
            "alpha-L-iduronic acid residue from the non-reducing end of dermatan sulfate and heparan sulfate. "
            "Dermatan sulfate: structural component of connective tissue, skin, cardiac valves. "
            "Heparan sulfate: component of cell surface proteoglycans; critical for brain development. "
            "Loss of IDUA → DS and HS fragment catabolism blocked → progressive lysosomal accumulation: "
            "CONNECTIVE TISSUE: coarse facial features, stiff joints, thickened skin; "
            "SKELETON: dysostosis multiplex — J-shaped sella, anterior vertebral beaking (L1/L2), "
            "oar-shaped ribs, spatulate clavicles, oblique acetabula, metacarpal pointing; "
            "CARDIAC VALVES: DS accumulation → thickening/regurgitation (mitral>aortic); "
            "CORNEA: DS accumulation → progressive clouding — all MPS I; "
            "BRAIN: HS accumulation → neurodegeneration in severe forms (Hurler); "
            "AIRWAY: thickened soft tissue → obstructive sleep apnoea + difficult intubation. "
            "LARONIDASE (ERT): replaces lysosomal enzyme — improves visceral, respiratory, cardiac; "
            "DOES NOT cross BBB → HSCT required for CNS preservation in Hurler."
        ),
        "pathognomonic": (
            "MPS I DIAGNOSTIC CLUSTER: "
            "1) COARSE FACIES: macrocephaly, prominent supraorbital ridges, depressed nasal bridge, "
            "full lips, gingival hyperplasia — progressive from birth; "
            "2) CORNEAL CLOUDING: bilateral, progressive, all MPS I phenotypes (mild in Scheie, dense in Hurler); "
            "slit-lamp mandatory — PATHOGNOMONIC for MPS I; "
            "3) DYSOSTOSIS MULTIPLEX on X-ray: J-shaped sella, anterior vertebral beaking (L1/L2), "
            "oar-shaped ribs, short wide clavicles, metacarpal pointing (proximal), oblique acetabula; "
            "4) HEPATOSPLENOMEGALY: DS/HS storage in Kupffer cells + reticuloendothelial system; "
            "5) URINE GAG SCREENING: elevated urinary dermatan + heparan sulfate — electrophoresis or spot test; "
            "6) IDUA ENZYME ACTIVITY: markedly reduced in leukocytes or dried blood spot; "
            "7) ECHOCARDIOGRAM: cardiac valve thickening/regurgitation — present in virtually all MPS I; "
            "HURLER CLINICAL HALLMARKS: developmentally normal first 6-12 months → plateau → regression; "
            "severe ID + motor regression + frequent respiratory infections; "
            "SCHEIE: joints stiff + corneal clouding + NO cognitive decline — diagnosed later (late childhood/adult)"
        ),
        "treatment": (
            "MPS I HURLER (Severe): "
            "ALLOGENEIC HSCT: treatment of choice; best outcomes if age <2yr + DQ (developmental quotient) >70; "
            "cord blood preferred (higher cell dose, less graft-versus-host); "
            "ERT BRIDGE: laronidase pre- and post-HSCT to reduce disease burden; "
            "HSCT PRESERVES CNS: stabilises cognitive function if performed before severe decline; "
            "MPS I ATTENUATED (Hurler-Scheie / Scheie): "
            "LARONIDASE (Aldurazyme): 0.58 mg/kg IV weekly; FDA 2003; "
            "improves 6-min walk, FVC, liver size; does NOT improve corneal clouding or joint stiffness; "
            "CARDIAC SURVEILLANCE: annual echocardiogram; valve replacement if severe; "
            "SURGICAL MANAGEMENT: corneal transplant (Scheie); carpal tunnel release; "
            "cervical fusion (odontoid instability — C1/C2 subluxation screening mandatory); "
            "AIRWAY MANAGEMENT: CPAP/BiPAP for OSA; anaesthetic risk — awake fibreoptic mandatory; "
            "INTRATHECAL ERT: investigational for CNS delivery; clinical trials ongoing; "
            "GENE THERAPY: LV/AAV9-IDUA trials in progress"
        ),
        "key_features": [
            "IDUA (MPS I): AR; alpha-L-iduronidase deficiency; DS+HS accumulation; 1:100,000",
            "Hurler (IH): severe; coarse facies + corneal clouding + hepatosplenomegaly + severe ID; death <10yr without HSCT",
            "Corneal clouding PATHOGNOMONIC in all MPS I (slit-lamp mandatory at diagnosis)",
            "Dysostosis multiplex on X-ray: J-shaped sella + anterior vertebral beaking + oar-shaped ribs + metacarpal pointing",
            "Laronidase (Aldurazyme) ERT FDA 2003 — first MPS ERT; treats visceral; does NOT cross BBB",
            "HSCT: curative for CNS in Hurler if age <2yr and DQ>70 — ERT bridge pre+post-HSCT",
            "Echocardiogram mandatory: cardiac valve thickening virtually universal in MPS I",
            "C1/C2 subluxation (odontoid hypoplasia): cervical spine MRI before any anaesthesia",
        ],
        "key_ddx": [
            "MPS II Hunter (IDS): XLR males; NO corneal clouding (KEY); similar skeletal/visceral to Hurler",
            "MPS VI Maroteaux-Lamy (ARSB): similar skeletal/visceral; preserved intellect; dermatan only",
            "MPS IV Morquio (GALNS): predominantly skeletal; no ID; keratan sulfate; cornea late mild",
            "GM1 gangliosidosis (GLB1): coarse facies from birth; cherry-red spot; no urine DS/HS elevation",
        ],
        "onset_age": 1.0,
        "coarse_facies_pct": 98,
        "corneal_clouding_pct": 95,
        "hepatosplenomegaly_pct": 90,
        "cognitive_decline_pct": 75,
        "cardiac_valve_pct": 85,
        "seed": 2630,
    },
    {
        "gene": "IDS",
        "protein": (
            "IDS -- Xq28 XLR -- 550aa -- Iduronate-2-Sulfatase-76kDa-"
            "Lysosomal-Dermatan+Heparan-Sulfate-Desulfation-MPS-II-XLR -- OMIM-Gene-309900-Disease-Hunter-309900"
        ),
        "locus": "Xq28",
        "protein_size": "550 aa / 76 kDa",
        "inheritance": (
            "XLR (X-linked recessive); MPS II (Hunter syndrome); "
            "Iduronate-2-sulfatase (I2S) deficiency; prevalence 1:100,000-170,000 males; "
            "Affected males only (XLR); carrier females usually unaffected (rarely symptomatic); "
            "Severe (neuronopathic) MPS II: ~2/3 of patients; progressive ID + severe behavioural; "
            "Attenuated MPS II: ~1/3; near-normal intellect; primarily somatic disease; "
            "KEY DISTINCTION from Hurler (MPS I): NO CORNEAL CLOUDING in MPS II — CRITICAL distinguishing feature; "
            "pebbly ivory-white skin lesions on upper arms/scapular region: PATHOGNOMONIC for MPS II; "
            "hearing loss (conductive + sensorineural) universal; "
            "IDS deletions/rearrangements: 20% alleles — array CGH required if sequencing negative"
        ),
        "disease_category": (
            "MPS II (Hunter Syndrome); lysosomal storage disorder; mucopolysaccharidosis; "
            "IDS encodes iduronate-2-sulfatase — lysosomal sulfatase cleaving 2-O-sulfate groups "
            "from iduronic acid residues in dermatan sulfate (DS) and heparan sulfate (HS); "
            "Loss of IDS → DS + HS accumulate — same substrates as IDUA (upstream step); "
            "IDS ACTS BEFORE IDUA in the degradation pathway of DS and HS; "
            "IDURSULFASE (Elaprase) — ERT FDA 2006; 0.5 mg/kg IV weekly; "
            "Treats visceral/somatic disease; does NOT cross BBB; "
            "PABINAFUSP ALFA (Izcargo) — BBB-crossing ERT (Japan approved 2021); anti-hTfR1 fusion; "
            "INTRATHECAL idursulfase (Hunterase IT): investigational; "
            "HSCT: controversial in MPS II — less clear benefit than MPS I Hurler"
        ),
        "disease_pathway": (
            "IDS encodes iduronate-2-sulfatase (I2S), a lysosomal sulfatase that removes the 2-O-sulfate "
            "from iduronic acid residues within dermatan sulfate and heparan sulfate chains. "
            "This step occurs BEFORE alpha-L-iduronidase (IDUA) in the catabolism pathway. "
            "Loss of I2S → sulfated iduronate residues persist → DS and HS catabolism blocked → "
            "progressive lysosomal accumulation identical in anatomical distribution to MPS I "
            "EXCEPT: brain HS accumulation follows X-inactivation pattern (XLR) — males all affected; "
            "PEBBLY SKIN LESIONS (peau d'orange): ivory-white pebbly papules on scapular region/upper arms "
            "— HS/DS deposition in skin dermis; seen in 50-60% MPS II males; PATHOGNOMONIC when present; "
            "NO CORNEAL CLOUDING: I2S and IDUA are both needed for corneal DS/HS catabolism, "
            "but the corneal I2S substrate (dermatan sulfate) is processed differently — "
            "exact reason cornea spared in Hunter remains mechanistically debated but clinically reliable; "
            "OBSTRUCTIVE SLEEP APNOEA: virtually universal in MPS II — glycosaminoglycan in upper airway. "
            "I2S PSEUDOGENE (IDSP1): 20kb upstream on Xq28; large deletions/rearrangements between "
            "IDS and IDSP1 account for 20% alleles — standard NGS may miss; MLPA/CGH required."
        ),
        "pathognomonic": (
            "MPS II (HUNTER) DIAGNOSTIC FEATURES: "
            "1) NO CORNEAL CLOUDING — KEY DDx from Hurler (MPS I); slit-lamp examination distinguishes; "
            "2) IVORY PEBBLY SKIN LESIONS: ivory-white pebbly papules over scapular region, upper arms; "
            "DS/HS deposition in dermis; seen in ~50-60%; PATHOGNOMONIC when present; "
            "3) XLR pattern: males affected; female carriers usually asymptomatic; "
            "4) COARSE FACIES: macrocephaly, depressed nasal bridge — similar to Hurler but usually milder; "
            "5) HEARING LOSS: mixed (conductive from ossicular DS storage + sensorineural); "
            "nearly universal; audiogram mandatory at diagnosis + annually; "
            "6) DYSOSTOSIS MULTIPLEX: identical to MPS I on X-ray — J-shaped sella, vertebral beaking; "
            "7) URINE GAG: elevated DS + HS (same pattern as MPS I); "
            "IDS ENZYME ACTIVITY: markedly reduced in plasma or leukocytes; "
            "8) BEHAVIOURAL PHENOTYPE (severe): marked hyperactivity + aggression + destructive behaviour; "
            "9) CARDIAC: valve thickening + cardiomyopathy + PAH; echo mandatory"
        ),
        "treatment": (
            "MPS II SOMATIC TREATMENT: "
            "IDURSULFASE (Elaprase): 0.5 mg/kg IV weekly; FDA 2006; "
            "improves 6-min walk, FVC, liver/spleen size, joint mobility; does NOT cross BBB; "
            "premedication with antihistamine ± corticosteroid for infusion reactions; "
            "MPS II CNS DISEASE: "
            "PABINAFUSP ALFA (Izcargo): Japan 2021; anti-hTfR1 antibody fused to idursulfase; "
            "crosses BBB via transferrin receptor 1-mediated transcytosis; "
            "INTRATHECAL IDURSULFASE: investigational; stabilises CNS biomarkers; "
            "HSCT: less evidence than MPS I — recommended only if severe, age <6yr, MQ>70; "
            "SUPPORTIVE: "
            "AIRWAY: adenotonsillectomy + CPAP; anaesthetic risk — difficult intubation; "
            "HEARING AIDS: mixed hearing loss; cochlear implant if severe sensorineural; "
            "CARDIAC: annual echo; antihypertensives; valve surgery if severe; "
            "ORTHOPAEDIC: carpal tunnel release; hip/knee arthroplasty (attenuated); "
            "CERVICAL SPINE: C1/C2 MRI before anaesthesia; fusion if unstable; "
            "BEHAVIOURAL: methylphenidate; clonidine; antipsychotics (limited evidence)"
        ),
        "key_features": [
            "IDS (MPS II Hunter): XLR males only; iduronate-2-sulfatase deficiency; DS+HS storage; 1:100,000-170,000",
            "NO CORNEAL CLOUDING — KEY distinguishing feature from Hurler (MPS I); slit-lamp spared",
            "Ivory pebbly skin lesions (scapular/upper arm) PATHOGNOMONIC when present (~50-60%)",
            "Severe (2/3): progressive ID + hyperactivity + aggression; attenuated (1/3): somatic only, normal intellect",
            "Idursulfase (Elaprase) ERT FDA 2006 — somatic disease; does NOT cross BBB",
            "Pabinafusp alfa (Izcargo) Japan 2021: anti-hTfR1 BBB-crossing ERT for CNS Hunter",
            "IDS pseudogene (IDSP1) on Xq28: large deletions missed by NGS — MLPA/CGH required",
            "Hearing loss (mixed) nearly universal — audiogram mandatory at diagnosis and annually",
        ],
        "key_ddx": [
            "MPS I Hurler (IDUA): AR; CORNEAL CLOUDING present; both sexes; dermatan + heparan; upstream step",
            "MPS VI Maroteaux-Lamy (ARSB): AR both sexes; no CNS; dermatan only; NO heparan; normal intellect",
            "X-linked adrenoleukodystrophy (ABCD1): XLR; no GAG; VLCFA elevated; MRI white matter; no skin papules",
            "Pseudo-Hurler polydystrophy (GNPTAB) ML3: coarse facies + joints; inclusions on fibroblasts; normal urinary GAG",
        ],
        "onset_age": 1.5,
        "coarse_facies_pct": 95,
        "corneal_clouding_pct": 2,
        "hepatosplenomegaly_pct": 88,
        "cognitive_decline_pct": 65,
        "skin_papules_pct": 55,
        "hearing_loss_pct": 92,
        "seed": 2631,
    },
    {
        "gene": "SGSH",
        "protein": (
            "SGSH -- 17q25.3 AR -- 502aa -- Heparan-N-Sulfatase-56kDa-"
            "Lysosomal-Heparan-Sulfate-Desulfation-MPS-III-A-AR -- OMIM-Gene-252900-Disease-Sanfilippo-A-252900"
        ),
        "locus": "17q25.3",
        "protein_size": "502 aa / 56 kDa",
        "inheritance": (
            "AR (biallelic SGSH loss of function); MPS III A (Sanfilippo Syndrome Type A); "
            "Heparan-N-sulfatase (sulfamidase) deficiency; prevalence 1:60,000-80,000 (MPS III A most common subtype in Europe); "
            "Most severe and earliest onset of the four Sanfilippo subtypes (A>B>C>D for severity); "
            "SEVERE PROGRESSIVE NEURODEGENERATION with relatively MILD somatic features; "
            "Behavioural phenotype: relentless hyperactivity + aggression + sleep disturbance — most severe in MPS III; "
            "Normal development to age 2-4yr → behavioural deterioration → plateau → regression → death teens/20s; "
            "EXCLUSIVELY HEPARAN SULFATE: only HS accumulates (no DS component unlike MPS I, II, VI); "
            "R245H: most common pathogenic allele in European MPS IIIA; "
            "No clear genotype-phenotype correlation for attenuated alleles"
        ),
        "disease_category": (
            "MPS III A (Sanfilippo Syndrome Type A); lysosomal storage disorder; mucopolysaccharidosis; "
            "SGSH encodes heparan-N-sulfatase (N-sulfoglucosamine sulfohydrolase, sulfamidase) — "
            "lysosomal sulfatase removing N-sulfate from the non-reducing end glucosamine residues of heparan sulfate; "
            "This is the FIRST step in HS catabolism; loss → HS accumulation in CNS neurons predominantly; "
            "SOMATIC FEATURES MILD relative to MPS I/II/VI: mild hepatosplenomegaly; joint stiffness mild; "
            "BEHAVIOURAL PHENOTYPE DOMINATES: hyperactivity, aggression, sleep disruption — cortical neuron HS storage; "
            "NO APPROVED ERT (heparan sulfate does not respond to IV-delivered enzyme — poor CNS penetration); "
            "IV ENZYME TRIALS: negative for CNS outcomes; "
            "INTRATHECAL/GENE THERAPY: phase I/II trials ongoing — most promising approach"
        ),
        "disease_pathway": (
            "SGSH encodes heparan-N-sulfatase, the enzyme catalyzing the first step in lysosomal "
            "catabolism of heparan sulfate — removing the N-sulfate group from the non-reducing terminal "
            "glucosamine residue of HS. Loss blocks HS catabolism at this first step. "
            "Heparan sulfate is the predominant proteoglycan on the surface of neurons and in the "
            "pericellular matrix of the CNS. Unlike MPS I and II (which also accumulate DS), MPS IIIA "
            "accumulates HS ONLY — explaining the predominantly CNS phenotype with mild somatic involvement. "
            "NEURONAL HS STORAGE: "
            "Cortical neurons and hippocampal neurons accumulate HS → lysosomal swelling → "
            "autophagic pathway disruption → secondary accumulation of GM2/GM3 gangliosides → "
            "neuroinflammation (microglial activation) → synaptic loss → neurodegeneration. "
            "The profound BEHAVIOURAL phenotype (hyperactivity, aggression, sleep disruption) "
            "precedes cognitive decline by years — thalamic and frontal cortex HS storage likely. "
            "SLEEP DISRUPTION: HS storage in hypothalamic nuclei (suprachiasmatic) → circadian rhythm disruption; "
            "melatonin synthesis impaired — melatonin supplementation often partially effective. "
            "SOMATIC: hepatosplenomegaly mild; coarse facies subtle; joints mildly stiff; hearing mild."
        ),
        "pathognomonic": (
            "MPS IIIA (SANFILIPPO A) PRESENTATION: "
            "1) BEHAVIOURAL PHENOTYPE ONSET (2-4yr): PATHOGNOMONIC PATTERN — "
            "severe hyperactivity, impulsive aggression, destructive behaviour, poor sleep; "
            "often misdiagnosed as ADHD or autism spectrum disorder; "
            "2) SPEECH DELAY: prominent expressive language delay precedes motor regression; "
            "3) HEPARAN SULFATE: urine GAG = HEPARAN SULFATE ONLY (no dermatan) — critical distinguisher; "
            "thin-layer chromatography or tandem mass spectrometry of urine GAG; "
            "4) MILD SOMATIC FEATURES: subtle coarse facies; mild hepatosplenomegaly; mild joint stiffness; "
            "skeletal changes less prominent than other MPS types; NO corneal clouding; "
            "5) NEUROIMAGING: progressive cortical atrophy; periventricular white matter signal changes late; "
            "6) SGSH ENZYME ACTIVITY: markedly reduced in leukocytes, plasma, or dried blood spot; "
            "7) SLEEP DISTURBANCE: severe sleep fragmentation + reversal of day-night cycle; melatonin used; "
            "8) REGRESSION: developmental regression follows behavioural phase; swallowing difficulties late; "
            "death in second or third decade from respiratory failure or aspiration"
        ),
        "treatment": (
            "MPS IIIA — NO APPROVED DISEASE-MODIFYING THERAPY: "
            "GENE THERAPY: "
            "OAV-101 (scAAV9-SGSH): intrathecal or intraparenchymal delivery; phase I/II trials; "
            "early treatment (before severe neurological decline) may stabilise; "
            "AAV9-SGSH: systemic administration in neonatal mouse models — encouraging; "
            "INTRATHECAL ERP TRIALS: negative for meaningful CNS outcomes; "
            "IV ERT TRIALS (bimosiamose, tralesinidase alfa): failed to show CNS benefit; "
            "SUBSTRATE REDUCTION THERAPY: "
            "Genistein (isoflavone): partial HS reduction — European studies; not proven clinical benefit; "
            "SYMPTOMATIC MANAGEMENT: "
            "BEHAVIOURAL: risperidone (limited); methylphenidate often paradoxically worsens; "
            "clonidine, buspirone, melatonin (sleep); "
            "SLEEP: melatonin 5-20mg; chloral hydrate; zopiclone; "
            "ANTICONVULSANTS: seizures in late stage; levetiracetam, valproate; "
            "NUTRITION: PEG gastrostomy for dysphagia; "
            "PALLIATIVE: anticipatory care planning; respiratory support"
        ),
        "key_features": [
            "SGSH (MPS IIIA Sanfilippo A): AR; heparan-N-sulfatase deficiency; HS-only storage; most severe MPS III",
            "Behavioural phenotype PATHOGNOMONIC: severe hyperactivity + aggression + sleep disruption onset age 2-4yr",
            "Frequently misdiagnosed as ADHD or autism before coarse facies and regression noted",
            "Urine GAG: HEPARAN SULFATE ONLY (no dermatan) — critical distinguisher from MPS I/II/VI",
            "NO APPROVED ERT — IV enzyme fails to cross BBB; gene therapy trials ongoing (AAV9-SGSH)",
            "Somatic features MILD: subtle coarse facies, mild hepatosplenomegaly — easily missed early",
            "Sleep disturbance severe and early: melatonin supplementation partially effective",
            "R245H most common European allele; death in 2nd-3rd decade from aspiration/respiratory failure",
        ],
        "key_ddx": [
            "MPS IIIB (NAGLU): identical phenotype; HS only; alpha-N-acetylglucosaminidase activity low; enzyme/molecular testing",
            "MPS IIIC (HGSNAT): same phenotype; acetyl-CoA:alpha-glucosaminide N-acetyltransferase; transmembrane enzyme",
            "Autism spectrum disorder: no urinary HS; no lysosomal enzyme deficiency; normal MRI",
            "ADHD: no urinary GAG elevation; no developmental regression; normal enzyme activities",
        ],
        "onset_age": 3.0,
        "coarse_facies_pct": 55,
        "corneal_clouding_pct": 0,
        "hepatosplenomegaly_pct": 55,
        "cognitive_decline_pct": 98,
        "behavioural_pct": 97,
        "sleep_disturbance_pct": 90,
        "seed": 2632,
    },
    {
        "gene": "GALNS",
        "protein": (
            "GALNS -- 16q24.3 AR -- 552aa -- Galactosamine-6-Sulfatase-GALNS-120kDa-"
            "Lysosomal-Keratan+Chondroitin-6-Sulfate-Desulfation-MPS-IV-A-AR -- OMIM-Gene-253000-Disease-Morquio-A-253000"
        ),
        "locus": "16q24.3",
        "protein_size": "552 aa / 120 kDa",
        "inheritance": (
            "AR (biallelic GALNS loss of function); MPS IVA (Morquio Syndrome Type A); "
            "Galactosamine-6-sulfatase (GALNS) deficiency; prevalence 1:200,000-300,000; "
            "MPS IVA: EXCLUSIVELY SKELETAL PHENOTYPE — normal intellect in all patients; "
            "No CNS involvement (keratan sulfate not stored in brain to same degree); "
            "Short stature, severe skeletal dysplasia, ODONTOID HYPOPLASIA (C1/C2 subluxation), joint laxity; "
            "Heterogeneous severity: classic (severe) to attenuated (mild); "
            "p.R386C: common attenuated allele; p.G301C: common severe allele; "
            "KERATAN SULFATE + CHONDROITIN-6-SULFATE: substrates (distinct from MPS I/II DS/HS); "
            "GALNS is ALSO required for KERATAN SULFATE catabolism in cartilage → skeletal phenotype dominant"
        ),
        "disease_category": (
            "MPS IVA (Morquio A Syndrome); lysosomal storage disorder; mucopolysaccharidosis; "
            "GALNS encodes galactosamine-6-sulfatase (also called galactose-6-sulfatase, chondroitin-6-sulfatase) — "
            "lysosomal sulfatase cleaving 6-O-sulfate from N-acetylgalactosamine in keratan sulfate (KS) "
            "and from galactose in chondroitin-6-sulfate (C6S); "
            "Loss of GALNS → KS + C6S accumulate predominantly in cartilage, cornea, bone; "
            "Brain relatively spared → preserved intellect in all MPS IVA; "
            "ELOSULFASE ALFA (Vimizim): ERT FDA 2014; 2 mg/kg IV weekly; "
            "improves 6-min walk, stair-climb, respiratory function; does NOT improve skeletal deformity; "
            "Must be given BEFORE significant skeletal deformity occurs"
        ),
        "disease_pathway": (
            "GALNS encodes galactosamine-6-sulfatase, which removes the 6-O-sulfate group from "
            "N-acetylgalactosamine residues in keratan sulfate and from galactose-6-sulfate in chondroitin-6-sulfate. "
            "Keratan sulfate is the dominant proteoglycan of cartilage (aggrecan), the cornea (lumican, keratocan), "
            "and the intervertebral disc. "
            "Loss of GALNS → KS accumulation in: "
            "CARTILAGE: progressive cartilage destruction → joint laxity paradoxically (lax ligaments) "
            "combined with cartilage loss → characteristic hypermobility + progressive deformity; "
            "Spinal: platyspondyly (flat vertebrae), odontoid hypoplasia (ATLANTOAXIAL INSTABILITY), "
            "thoracic kyphoscoliosis, gibbus deformity; "
            "Hip: hip dysplasia, coxa valga; "
            "CORNEA: progressive corneal clouding (late, mild — unlike MPS I); "
            "GROWTH: short stature — disproportionate with short trunk; "
            "CARDIAC: aortic regurgitation (less common than MPS I); "
            "RESPIRATORY: restricted thorax → restrictive lung disease + sleep apnoea; "
            "INTELLIGENCE: completely PRESERVED — critical for prognosis and patient expectations. "
            "ODONTOID HYPOPLASIA: under-developed C2 dens → atlantoaxial instability → "
            "risk of cervical cord compression → tetraplegia/death — MOST CRITICAL COMPLICATION."
        ),
        "pathognomonic": (
            "MPS IVA (MORQUIO A) CLINICAL HALLMARKS: "
            "1) SEVERE SKELETAL DYSPLASIA + NORMAL INTELLECT — PATHOGNOMONIC COMBINATION: "
            "short stature, platyspondyly, kyphoscoliosis, genu valgum, joint laxity; "
            "intelligence and language normal throughout life; "
            "2) ODONTOID HYPOPLASIA (C1/C2 SUBLUXATION): "
            "MOST FEARED COMPLICATION — cervical MRI mandatory; instability can cause acute tetraplegia/death; "
            "cervical spine stability must be evaluated before ANY surgical procedure or general anaesthesia; "
            "prophylactic C1/C2 fusion recommended if instability present; "
            "3) URINE KERATAN SULFATE: elevated urinary KS — MPS IVA generates KS (not DS/HS); "
            "urine GAG screening + specific KS assay required; "
            "4) CORNEAL CLOUDING: late and mild (vs dense and early in MPS I); slit-lamp shows diffuse haze; "
            "5) GALNS ENZYME: markedly reduced in leukocytes; "
            "6) PLATYSPONDYLY on X-ray: flattened vertebrae + anterior central beak; "
            "7) HIP: coxa valga + hip dysplasia on pelvic X-ray; "
            "8) WRIST X-RAY: hypoplastic carpal bones"
        ),
        "treatment": (
            "MPS IVA TREATMENT: "
            "ELOSULFASE ALFA (Vimizim): 2 mg/kg IV weekly; FDA February 2014; "
            "improves 6-min walk test, stair climb, respiratory (FVC, FEV1); "
            "does NOT reverse established skeletal deformity; "
            "must start before severe skeletal deformity; "
            "infusion reactions: antihistamine/paracetamol premedication mandatory; "
            "SURGICAL MANAGEMENT: "
            "C1/C2 CERVICAL FUSION: most critical — reduces risk of tetraplegia; "
            "timing: before spinal cord signal change on MRI; awake fibreoptic if unstable; "
            "THORACOLUMBAR: kyphosis bracing; spinal fusion if severe; "
            "HIP ARTHROPLASTY: attenuated/adult patients; "
            "LOWER LIMB: guided growth, tibial/femoral osteotomy for genu valgum; "
            "CORNEA: keratoplasty for severe clouding (rare in IVA); "
            "CARDIAC: aortic valve surveillance + replacement if severe; "
            "RESPIRATORY: annual pulmonary function; CPAP; chest physiotherapy; "
            "ANAESTHETIC RISK: extreme — atlantoaxial instability + difficult airway + restrictive lung; "
            "neuromonitoring + awake fibreoptic intubation protocol mandatory; "
            "GENE THERAPY: AAV-GALNS preclinical; clinical trials planned"
        ),
        "key_features": [
            "GALNS (MPS IVA Morquio A): AR; galactosamine-6-sulfatase deficiency; KS+C6S storage; 1:200,000-300,000",
            "Severe skeletal dysplasia + COMPLETELY PRESERVED INTELLECT — pathognomonic combination",
            "Odontoid hypoplasia (atlantoaxial instability): most critical complication — tetraplegia/death risk",
            "Cervical spine MRI mandatory before any surgery or anaesthesia; C1/C2 fusion if unstable",
            "Elosulfase alfa (Vimizim) ERT FDA 2014 — improves function; does NOT reverse skeletal deformity",
            "Urine keratan sulfate elevated (not DS or HS) — distinguishes from MPS I/II/VI",
            "Corneal clouding late and mild (vs early and dense in Hurler)",
            "Anaesthetic risk extreme: atlantoaxial instability + difficult airway + restrictive lung disease",
        ],
        "key_ddx": [
            "MPS IVB Morquio B (GLB1): same skeletal phenotype; beta-galactosidase deficient; keratan sulfate only; same spinal risk",
            "Spondyloepiphyseal dysplasia (SED): similar skeletal; NEGATIVE urine KS; COL2A1 usually; normal enzyme",
            "Kniest dysplasia: platyspondyly + joint problems; COL2A1; negative urine KS; normal GALNS",
            "MPS VI (ARSB): skeletal dysplasia + no CNS but dermatan sulfate NOT keratan; both sexes equally affected",
        ],
        "onset_age": 1.5,
        "coarse_facies_pct": 40,
        "corneal_clouding_pct": 45,
        "hepatosplenomegaly_pct": 30,
        "cognitive_decline_pct": 0,
        "skeletal_dysplasia_pct": 99,
        "odontoid_hypoplasia_pct": 80,
        "seed": 2633,
    },
    {
        "gene": "ARSB",
        "protein": (
            "ARSB -- 5q14.1 AR -- 533aa -- Arylsulfatase-B-ARSB-60kDa-"
            "Lysosomal-Dermatan-Sulfate-Desulfation-MPS-VI-AR -- OMIM-Gene-253200-Disease-Maroteaux-Lamy-253200"
        ),
        "locus": "5q14.1",
        "protein_size": "533 aa / 60 kDa",
        "inheritance": (
            "AR (biallelic ARSB loss of function); MPS VI (Maroteaux-Lamy Syndrome); "
            "Arylsulfatase B (N-acetylgalactosamine-4-sulfatase) deficiency; prevalence 1:250,000-600,000; "
            "EXCLUSIVELY DERMATAN SULFATE: only DS accumulates (no HS component); "
            "Key difference from MPS I/II: INTELLIGENCE PRESERVED in virtually all MPS VI; "
            "Severe phenotype: coarse facies + hepatosplenomegaly + corneal clouding + cardiac + skeletal; "
            "Attenuated phenotype: later onset, milder features, normal stature; "
            "Phenotypic severity ranges from severe early childhood presentation to mild adult presentation; "
            "p.R152W: common European severe allele; "
            "Residual ARSB enzyme activity correlates with phenotype severity"
        ),
        "disease_category": (
            "MPS VI (Maroteaux-Lamy Syndrome); lysosomal storage disorder; mucopolysaccharidosis; "
            "ARSB encodes arylsulfatase B (N-acetylgalactosamine-4-sulfatase) — "
            "lysosomal sulfatase cleaving 4-O-sulfate from N-acetylgalactosamine in dermatan sulfate; "
            "Loss of ARSB → DERMATAN SULFATE ONLY accumulates (no HS); "
            "No brain HS storage → intelligence preserved (unlike MPS I Hurler, MPS II severe, MPS III); "
            "Skeletal, cardiac, corneal, visceral DS storage → somatic disease; "
            "GALSULFASE (Naglazyme): ERT FDA 2005; 1 mg/kg IV weekly; "
            "improves 6-min walk, stair-climb, FVC, FEV1, urinary DS; "
            "does not improve corneal clouding significantly; "
            "HSCT: effective when performed early — CNS sparing not needed but somatic benefit clear"
        ),
        "disease_pathway": (
            "ARSB encodes arylsulfatase B (ARSB; N-acetylgalactosamine-4-sulfatase), "
            "a lysosomal sulfatase that cleaves the 4-O-sulfate from N-acetylgalactosamine residues "
            "within dermatan sulfate chains. "
            "Dermatan sulfate: structural glycosaminoglycan of connective tissue (skin, heart valves, blood vessels, cornea). "
            "Loss of ARSB → DS catabolism blocked → DS accumulates in lysosomes of: "
            "CONNECTIVE TISSUE: coarse facies, gingival hyperplasia, thickened skin, stiff joints; "
            "CARDIAC: mitral and aortic valve thickening + regurgitation — most common cause of death; "
            "CORNEA: progressive DS accumulation → bilateral corneal clouding; "
            "SKELETON: dysostosis multiplex — same skeletal X-ray findings as MPS I; "
            "LIVER/SPLEEN: DS storage in macrophages → hepatosplenomegaly; "
            "AIRWAY: DS in soft tissue → obstructive sleep apnoea, difficult intubation; "
            "SPINAL CORD: odontoid hypoplasia/atlantoaxial instability → cord compression (less severe than MPS IVA); "
            "INTELLIGENCE PRESERVED: DS does not accumulate in neurons to cause neurodegeneration; "
            "all MPS VI patients maintain normal-range intellect even in severe phenotype. "
            "ARSB is the SAME ENZYME deficient in multiple sulfatase deficiency (MSD) — "
            "all sulfatases deficient in MSD due to SUMF1 deficiency (formylglycine-generating enzyme)."
        ),
        "pathognomonic": (
            "MPS VI (MAROTEAUX-LAMY) DIAGNOSTIC FEATURES: "
            "1) SOMATIC FEATURES WITHOUT COGNITIVE DECLINE: "
            "coarse facies + hepatosplenomegaly + skeletal dysplasia + corneal clouding in a child with NORMAL IQ — "
            "this combination is PATHOGNOMONIC (MPS I Hurler has similar soma but with ID); "
            "2) CORNEAL CLOUDING: bilateral progressive; slit-lamp; present in severe phenotype from childhood; "
            "3) DYSOSTOSIS MULTIPLEX: same X-ray pattern as MPS I — metacarpal pointing, vertebral beaking; "
            "4) CARDIAC VALVES: mitral regurgitation most common; annual echo mandatory; "
            "5) URINE GAG: DERMATAN SULFATE ONLY (no heparan sulfate) — critical GAG characterisation; "
            "thin-layer chromatography: DS band only; "
            "6) ARSB ENZYME: markedly low in leukocytes (dried blood spot for NBS); "
            "7) ATLANTOAXIAL INSTABILITY: cervical spine MRI required; "
            "8) HEARING LOSS: mixed (conductive + sensorineural); audiogram annually"
        ),
        "treatment": (
            "MPS VI TREATMENT: "
            "GALSULFASE (Naglazyme): 1 mg/kg IV weekly; FDA May 2005; "
            "significantly improves 6-min walk, stair-climb, respiratory, liver/spleen size; "
            "starts as early as possible — greater benefit in younger patients; "
            "INFUSION REACTIONS: antihistamine + antipyretic premedication; "
            "HSCT: evidence for somatic benefit; consider in severe young patients; "
            "not needed for CNS protection (intelligence preserved) but improves somatic outcomes; "
            "CARDIAC: annual echocardiogram; antihypertensives; "
            "valve replacement in severe regurgitation; "
            "CORNEAL TRANSPLANT: keratoplasty for vision if severe clouding; "
            "SKELETAL/SURGICAL: "
            "C1/C2 fusion if atlantoaxial instability; "
            "adenotonsillectomy + CPAP for OSA; "
            "hip arthroplasty/osteotomy (attenuated adults); "
            "AIRWAY: anaesthetic risk — difficult airway + atlantoaxial instability; "
            "awake fibreoptic + neuromonitoring protocol; "
            "HEARING AIDS: mixed hearing loss from birth"
        ),
        "key_features": [
            "ARSB (MPS VI Maroteaux-Lamy): AR; arylsulfatase B deficiency; DS-only storage; 1:250,000-600,000",
            "Intelligence PRESERVED in virtually all patients — key difference from MPS I Hurler and MPS II severe",
            "Urine GAG: DERMATAN SULFATE ONLY (no heparan) — critical distinguisher from MPS I, II, III",
            "Galsulfase (Naglazyme) ERT FDA 2005 — significant somatic improvement; start early",
            "Cardiac valve disease (mitral) — most common cause of death; annual echocardiogram mandatory",
            "Corneal clouding progressive — slit-lamp at diagnosis; keratoplasty if severe",
            "Atlantoaxial instability: cervical MRI mandatory before any anaesthesia",
            "Anaesthetic risk: difficult airway + atlantoaxial instability — awake fibreoptic protocol essential",
        ],
        "key_ddx": [
            "MPS I Hurler (IDUA): AR; DS+HS; corneal clouding + coarse facies but WITH severe ID in Hurler",
            "MPS II Hunter (IDS): XLR males; DS+HS; behavioural + ID (severe form); no corneal clouding",
            "MPS IVA Morquio A (GALNS): skeletal + normal intellect but KS not DS; milder facies; joint laxity dominant",
            "Fucosidosis (FUCA1): similar somatic features; angiokeratoma; fucose storage; normal urinary DS",
        ],
        "onset_age": 2.0,
        "coarse_facies_pct": 92,
        "corneal_clouding_pct": 85,
        "hepatosplenomegaly_pct": 85,
        "cognitive_decline_pct": 3,
        "cardiac_valve_pct": 80,
        "seed": 2634,
    },
    {
        "gene": "GUSB",
        "protein": (
            "GUSB -- 7q11.21 AR -- 651aa -- Beta-Glucuronidase-GUSB-75kDa-"
            "Lysosomal-Dermatan+Heparan+Chondroitin-Sulfate-Hydrolysis-MPS-VII-AR -- OMIM-Gene-253220-Disease-Sly-253220"
        ),
        "locus": "7q11.21",
        "protein_size": "651 aa / 75 kDa",
        "inheritance": (
            "AR (biallelic GUSB loss of function); MPS VII (Sly Syndrome); "
            "Beta-glucuronidase deficiency; extreme rarity: <1:1,000,000; "
            "BROADEST SUBSTRATE SPECIFICITY: dermatan + heparan + chondroitin sulfates all accumulate; "
            "Most variable phenotype of all MPS types — from hydrops fetalis (lethal) to attenuated adult; "
            "NON-IMMUNE HYDROPS FETALIS: MPS VII most common lysosomal cause — "
            "fetal ascites, pleural effusions, skin oedema; survival to term rare; "
            "Surviving neonates: severe Hurler-like phenotype; "
            "Attenuated phenotype: later onset, near-normal intellect; "
            "L176F: most common attenuated allele; "
            "MPS VII: FIRST LYSOSOMAL STORAGE DISORDER with approved NBS (newborn screening) in USA (2024)"
        ),
        "disease_category": (
            "MPS VII (Sly Syndrome); lysosomal storage disorder; mucopolysaccharidosis; "
            "GUSB encodes beta-glucuronidase — lysosomal hydrolase cleaving beta-glucuronide linkages "
            "from the non-reducing end of dermatan sulfate, heparan sulfate, and chondroitin sulfates; "
            "Broadest substrate of any MPS enzyme — all three major GAG types accumulate; "
            "VESTRONIDASE ALFA (Mepsevii): ERT FDA 2017; 4 mg/kg IV every 4 weeks; "
            "first approved ERT for MPS VII; approved through accelerated review; "
            "improves 6-min walk, spirometry, hepatosplenomegaly; "
            "HYDROPS NEONATES: vestronidase alfa trial in hydrops survivors — ongoing; "
            "MPS VII ALSO used as preclinical model for gene therapy"
        ),
        "disease_pathway": (
            "GUSB encodes beta-glucuronidase (betaG), which cleaves non-reducing terminal "
            "glucuronic acid residues (beta-glucuronide bonds) from dermatan sulfate, heparan sulfate, "
            "and chondroitin sulfate chains. This represents a LATE step in the degradation of all "
            "three major sulfated GAGs. "
            "Loss of GUSB → DS + HS + CS accumulate simultaneously: "
            "The triple GAG accumulation explains the BROAD somatic phenotype: "
            "FETAL: DS/HS/CS storage causes placental dysfunction → non-immune hydrops fetalis; "
            "CONNECTIVE TISSUE: coarse facies, stiff joints, thickened skin; "
            "SKELETON: dysostosis multiplex; "
            "BRAIN: HS storage → cognitive impairment in severe phenotype; "
            "LIVER/SPLEEN: GAG in Kupffer cells → hepatosplenomegaly; "
            "CARDIAC: DS in valves → regurgitation; "
            "CORNEA: DS/CS storage → clouding; "
            "LUNG: reduced compliance; restrictive pattern. "
            "ATTENUATED PHENOTYPE: residual GUSB activity → partial GAG catabolism → "
            "milder somatic features ± preserved cognition. "
            "L176F allele: most common attenuated variant — mild/moderate phenotype. "
            "MPS VII AS A MODEL: mouse GUSB-knockout model was first LSD mouse model used for "
            "proof-of-concept gene therapy (retroviral, 1990s) — historically important."
        ),
        "pathognomonic": (
            "MPS VII (SLY SYNDROME) FEATURES: "
            "1) NON-IMMUNE HYDROPS FETALIS: PATHOGNOMONIC presentation for MPS VII — "
            "fetal ultrasound: ascites + pleural effusions + skin oedema + placentomegaly; "
            "most common lysosomal storage cause of hydrops; "
            "hydrops + GAG elevation on amniocentesis → GUSB enzyme assay on chorionic villi/amniocytes; "
            "2) POST-NATAL SEVERE: Hurler-like — coarse facies, hepatosplenomegaly, skeletal, corneal clouding; "
            "cognitive decline variable; "
            "3) TRIPLE GAG ELEVATION: urine DS + HS + chondroitin sulfate all elevated; "
            "most MPS types show one or two substrates — triple substrate is characteristic of MPS VII; "
            "4) GUSB ENZYME ACTIVITY: virtually absent (<1% normal) in severe; reduced in attenuated; "
            "leukocytes or dried blood spot; "
            "5) USA NBS (2024): first MPS with approved NBS — dried blood spot beta-glucuronidase activity; "
            "6) ATTENUATED: late-onset mild somatic ± normal IQ; L176F allele; minimal GAG; "
            "7) RECURRENT PULMONARY INFECTIONS: DS/HS storage in airways → susceptibility"
        ),
        "treatment": (
            "MPS VII TREATMENT: "
            "VESTRONIDASE ALFA (Mepsevii): 4 mg/kg IV every 4 weeks; FDA August 2017; "
            "first approved therapy for MPS VII; accelerated approval; "
            "improves 6-min walk, spirometry, hepatosplenomegaly, corneal clouding; "
            "neonatal hydrops survivors: compassionate use trials ongoing; "
            "GENE THERAPY: AAV8/AAV9-GUSB — extensive preclinical data (mouse + dog + cat); "
            "LV-GUSB for ex vivo HSC gene therapy; clinical trials planned; "
            "HSCT: evidence in severe phenotype — cognition preservation if early; "
            "HYDROPS MANAGEMENT: intrauterine ERT delivery investigational; "
            "supported delivery if viable; intensive NICU; "
            "SUPPORTIVE: "
            "CARDIAC: echo annually; valve surgery if severe; "
            "AIRWAY: OSA management; CPAP; adenotonsillectomy; "
            "CORNEAL: slit-lamp; keratoplasty if severe; "
            "SKELETAL: C1/C2 evaluation; physiotherapy; "
            "COGNITIVE: educational support in severe phenotype"
        ),
        "key_features": [
            "GUSB (MPS VII Sly): AR; beta-glucuronidase deficiency; DS+HS+CS triple storage; <1:1,000,000",
            "Non-immune hydrops fetalis: most common lysosomal cause — PATHOGNOMONIC presentation",
            "Broadest MPS substrate: triple GAG accumulation (DS + HS + chondroitin sulfate)",
            "Vestronidase alfa (Mepsevii) ERT FDA 2017 — first approved MPS VII therapy",
            "USA NBS 2024: first MPS added to newborn screening panel (GUSB enzyme DBS)",
            "Most phenotypically variable MPS: hydrops fetalis to mild adult attenuated",
            "L176F allele: most common attenuated variant — mild/moderate adult phenotype",
            "Historically important: first LSD mouse model used for gene therapy proof-of-concept",
        ],
        "key_ddx": [
            "Other causes of hydrops fetalis: immune (Rh/ABO) — DAT positive; cardiac — echo; lysosomal — GAG/enzyme",
            "MPS I Hurler (IDUA): DS+HS only (no chondroitin); alpha-L-iduronidase deficient; commoner",
            "GM1 gangliosidosis (GLB1): hydrops rare; cherry-red spot; ganglioside not GAG; normal GUSB",
            "Farber (ASAH1): hepatosplenomegaly + hydrops rare; ceramide not GAG; different enzyme",
        ],
        "onset_age": 0.0,
        "coarse_facies_pct": 80,
        "corneal_clouding_pct": 75,
        "hepatosplenomegaly_pct": 85,
        "cognitive_decline_pct": 60,
        "hydrops_pct": 40,
        "seed": 2635,
    },
    {
        "gene": "NAGLU",
        "protein": (
            "NAGLU -- 17q21.2 AR -- 743aa -- Alpha-N-Acetylglucosaminidase-NAGLU-82kDa-"
            "Lysosomal-Heparan-Sulfate-Hydrolysis-MPS-III-B-AR -- OMIM-Gene-252920-Disease-Sanfilippo-B-252920"
        ),
        "locus": "17q21.2",
        "protein_size": "743 aa / 82 kDa",
        "inheritance": (
            "AR (biallelic NAGLU loss of function); MPS IIIB (Sanfilippo Syndrome Type B); "
            "Alpha-N-acetylglucosaminidase deficiency; prevalence 1:200,000-500,000 (geographic variation); "
            "MPS IIIB: second most common Sanfilippo subtype overall; most common in southern Europe and Australia; "
            "EXCLUSIVELY HEPARAN SULFATE: only HS accumulates (no DS); "
            "Phenotype essentially IDENTICAL to MPS IIIA: severe neurodegeneration + mild somatic; "
            "Behavioural phase: hyperactivity + aggression + sleep disruption onset 2-4yr; "
            "Cognitive regression follows behavioural phase; "
            "p.R297X: most common pathogenic allele (European); "
            "NAGLU is the SECOND enzyme in HS catabolism — acts after SGSH (step 2 of the HS degradation pathway)"
        ),
        "disease_category": (
            "MPS IIIB (Sanfilippo Syndrome Type B); lysosomal storage disorder; mucopolysaccharidosis; "
            "NAGLU encodes alpha-N-acetylglucosaminidase — lysosomal hydrolase cleaving non-reducing terminal "
            "N-acetylglucosamine residues from heparan sulfate (step 2 in HS catabolism, after SGSH step 1); "
            "Loss of NAGLU → HS accumulates in CNS neurons predominantly (same as SGSH/MPS IIIA); "
            "SOMATIC FEATURES MILD: similar to MPS IIIA — mild hepatosplenomegaly, subtle coarse facies; "
            "CNS DISEASE PREDOMINATES: behavioural, then cognitive, then motor deterioration; "
            "NO APPROVED ERT: IV enzyme fails for CNS delivery; "
            "BMN-250 (NAGLU-IGF2 fusion): biotinylated NAGLU fused to IGF2 peptide for M6P receptor targeting; "
            "phase 2 trial; promising signal for brain delivery; "
            "GENE THERAPY: scAAV9-NAGLU intraparenchymal/intrathecal — phase I/II trials"
        ),
        "disease_pathway": (
            "NAGLU encodes alpha-N-acetylglucosaminidase, which cleaves the alpha-1,4 glycosidic bond "
            "between N-acetylglucosamine residues within heparan sulfate chains. "
            "This is STEP 2 in lysosomal HS catabolism: "
            "Step 1 (SGSH/MPS IIIA): removes N-sulfate from N-sulfoglucosamine; "
            "Step 2 (NAGLU/MPS IIIB): cleaves the now-unsulfated N-acetylglucosamine residue; "
            "Steps 3-5: further modifications by HGSNAT (MPS IIIC), GNS (MPS IIID), HYAL-related enzymes. "
            "Loss of NAGLU → HS degradation blocked at step 2 → same HS-fragment accumulation "
            "as in MPS IIIA with identical tissue distribution (predominantly CNS neurons). "
            "NEURONAL PATHOLOGY: lysosomal HS storage → autophagic dysfunction → "
            "secondary GM2/GM3 ganglioside accumulation → microglial neuroinflammation → "
            "progressive neuronal loss in cortex, hippocampus, cerebellum. "
            "NAGLU-IGF2 FUSION STRATEGY: attaching an IGF2 peptide to NAGLU targets the "
            "mannose-6-phosphate (M6P) / IGF2R receptor on neurons → enhanced neuronal uptake → "
            "potential for CNS ERT delivery (clinical trials ongoing). "
            "SOMATIC: mild — hepatosplenomegaly; coarse facies subtle; hearing mild. "
            "SLEEP DISRUPTION: hypothalamic HS storage; melatonin supplementation often used."
        ),
        "pathognomonic": (
            "MPS IIIB (SANFILIPPO B) PRESENTATION: "
            "1) CLINICALLY IDENTICAL TO MPS IIIA: same behavioural phenotype, regression, timeline; "
            "ENZYME/MOLECULAR TESTING required to distinguish IIIA from IIIB (identical clinical phenotype); "
            "2) BEHAVIOURAL PHASE (age 2-4yr): PATHOGNOMONIC PATTERN — "
            "severe hyperactivity, impulsive aggression, night-time waking; misdiagnosed as ADHD/autism; "
            "3) HEPARAN SULFATE ONLY in urine: same as MPS IIIA; tandem MS or electrophoresis; "
            "4) NAGLU ENZYME ACTIVITY: markedly low in leukocytes or plasma; "
            "SGSH (IIIA) activity: NORMAL in MPS IIIB — differentiates IIIA from IIIB; "
            "enzyme panel of all four MPS III enzymes required; "
            "5) MILD SOMATIC FEATURES: subtle coarse facies; mild hepatosplenomegaly; "
            "6) SLEEP DISRUPTION: severe sleep fragmentation; melatonin used; "
            "7) SPEECH DELAY: expressive > receptive language delay; "
            "8) REGRESSION: cognitive + motor; dysphagia; seizures late; "
            "death in 2nd-3rd decade"
        ),
        "treatment": (
            "MPS IIIB — NO APPROVED DISEASE-MODIFYING THERAPY: "
            "INVESTIGATIONAL: "
            "BMN-250 (NAGLU-IGF2-M6P FUSION): phase 2 trial; IV delivery; enhanced neuronal uptake; "
            "biomarker response in CSF HS; functional outcomes pending; "
            "INTRATHECAL NAGLU-IGF2: direct CSF delivery — clinical trial ongoing; "
            "GENE THERAPY (scAAV9-NAGLU): intraparenchymal delivery — phase I/II; "
            "early treatment required before substantial neuronal loss; "
            "SUBSTRATE REDUCTION: "
            "Genistein: partial clinical evidence (European); not approved; "
            "SYMPTOMATIC: "
            "BEHAVIOURAL: risperidone; melatonin (sleep); clonidine; buspirone; "
            "ANTICONVULSANTS: levetiracetam, valproate (myoclonic epilepsy late stage); "
            "NUTRITION: PEG gastrostomy; "
            "PALLIATIVE: anticipatory care; respiratory support"
        ),
        "key_features": [
            "NAGLU (MPS IIIB Sanfilippo B): AR; alpha-N-acetylglucosaminidase deficiency; HS-only storage",
            "Clinically IDENTICAL to MPS IIIA — only enzyme/molecular testing distinguishes them",
            "Behavioural phase PATHOGNOMONIC: hyperactivity + aggression + sleep disruption age 2-4yr",
            "Urine HS only (no DS); NAGLU activity low; SGSH activity NORMAL (distinguishes from IIIA)",
            "NO approved ERT; BMN-250 (NAGLU-IGF2 fusion) in phase 2 trials — promising CNS delivery",
            "Gene therapy (AAV9-NAGLU) in clinical trials — must treat before severe neuronal loss",
            "Somatic features mild — easily missed; diagnose from behavioural phenotype + urine GAG screen",
            "Step 2 of HS catabolism — SGSH (IIIA) does step 1, NAGLU (IIIB) does step 2",
        ],
        "key_ddx": [
            "MPS IIIA (SGSH): clinically identical; SGSH activity LOW; NAGLU NORMAL — enzyme panel mandatory",
            "MPS IIIC (HGSNAT): same phenotype; acetyl-CoA acetyltransferase (transmembrane); heparan only",
            "MPS IIID (GNS): same phenotype; GNS glucosamine-6-sulfatase; rarest Sanfilippo",
            "ADHD/Autism: no urinary HS; normal lysosomal enzymes; no regression",
        ],
        "onset_age": 3.0,
        "coarse_facies_pct": 50,
        "corneal_clouding_pct": 0,
        "hepatosplenomegaly_pct": 50,
        "cognitive_decline_pct": 98,
        "behavioural_pct": 97,
        "sleep_disturbance_pct": 88,
        "seed": 2636,
    },
    {
        "gene": "HGSNAT",
        "protein": (
            "HGSNAT -- 8p11.21 AR -- 635aa -- Heparan-Alpha-Glucosaminide-N-Acetyltransferase-HGSNAT-79kDa-"
            "Lysosomal-Transmembrane-Heparan-Sulfate-Acetylation-MPS-III-C-AR -- OMIM-Gene-252930-Disease-Sanfilippo-C-252930"
        ),
        "locus": "8p11.21",
        "protein_size": "635 aa / 79 kDa",
        "inheritance": (
            "AR (biallelic HGSNAT loss of function); MPS IIIC (Sanfilippo Syndrome Type C); "
            "Heparan-alpha-glucosaminide N-acetyltransferase (HGSNAT) deficiency; prevalence 1:1,500,000; "
            "TRANSMEMBRANE ENZYME: unique among MPS III enzymes — HGSNAT is a lysosomal membrane-bound "
            "acetyltransferase (not a soluble lysosomal hydrolase); "
            "This makes ERT delivery CONCEPTUALLY DIFFERENT — membrane insertion required; "
            "EXCLUSIVELY HEPARAN SULFATE: same substrate as MPS IIIA and IIIB; "
            "Phenotype identical to other Sanfilippo types — mild somatic + severe neurodegeneration; "
            "MPS IIIC may follow SLIGHTLY MILDER/LATER course than IIIA in some patients; "
            "P.S518F: common Sanfilippo C allele; "
            "NO approved therapy — ERT conceptually challenging due to transmembrane nature"
        ),
        "disease_category": (
            "MPS IIIC (Sanfilippo Syndrome Type C); lysosomal storage disorder; mucopolysaccharidosis; "
            "HGSNAT encodes acetyl-CoA:alpha-glucosaminide N-acetyltransferase — "
            "a LYSOSOMAL TRANSMEMBRANE ENZYME that transfers an acetyl group from cytoplasmic acetyl-CoA "
            "across the lysosomal membrane to the alpha-glucosamine terminus of heparan sulfate (step 3 of 5); "
            "Unique biochemistry: requires acetyl-CoA (cytoplasmic substrate) transferred to HS (luminal substrate) "
            "across the membrane — the only such reaction in lysosomal biology; "
            "Loss → HS catabolism blocked at step 3 → HS accumulation in CNS neurons; "
            "ERT FUNDAMENTALLY CHALLENGING: membrane insertion of recombinant HGSNAT would be required; "
            "GENE THERAPY: scAAV9-HGSNAT — CNS-directed gene therapy in phase I/II trials; "
            "most promising therapeutic approach given transmembrane biology"
        ),
        "disease_pathway": (
            "HGSNAT encodes heparan-alpha-glucosaminide N-acetyltransferase — a TRANSMEMBRANE PROTEIN "
            "unique among all lysosomal enzymes in that it requires substrate on both sides of the "
            "lysosomal membrane: acetyl-CoA from the CYTOPLASM and the N-acetylglucosamine terminus "
            "of partially degraded heparan sulfate in the LYSOSOMAL LUMEN. "
            "This is STEP 3 in the HS catabolism pathway: "
            "Step 1 SGSH (MPS IIIA): desulfates N-sulfoglucosamine → glucosamine; "
            "Step 2 NAGLU (MPS IIIB): cleaves GlcN from HS chain; "
            "Step 3 HGSNAT (MPS IIIC): acetylates free non-reducing terminal glucosamine "
            "(re-acetylation required for subsequent removal by HEXA); "
            "Step 4 GNS (MPS IIID): desulfates glucosamine-6-sulfate; "
            "Step 5 HEXA: cleaves N-acetylglucosamine. "
            "Blockade at step 3 → same HS-fragment accumulation in CNS neurons as MPS IIIA/B → "
            "identical behavioural and neurodegenerative phenotype. "
            "TRANSMEMBRANE ENZYME BIOLOGY: HGSNAT cannot be internalized by mannose-6-phosphate receptor "
            "pathway (used for all soluble lysosomal enzymes) → conventional ERT infeasible; "
            "requires gene therapy for enzyme delivery to lysosomes. "
            "Mouse model: HGSNAT-KO mouse (thorough behavioural + neuropathological model)."
        ),
        "pathognomonic": (
            "MPS IIIC (SANFILIPPO C) PRESENTATION: "
            "1) IDENTICAL CLINICAL PHENOTYPE TO MPS IIIA AND IIIB: "
            "behavioural phase (hyperactivity + aggression + sleep disruption, age 2-4yr) → "
            "cognitive regression → motor decline → death 2nd-3rd decade; "
            "2) ENZYME PANEL REQUIRED: "
            "HGSNAT activity LOW; SGSH (IIIA) NORMAL; NAGLU (IIIB) NORMAL — "
            "all four MPS III enzyme activities must be tested to determine subtype; "
            "3) HEPARAN SULFATE ONLY in urine: same as IIIA, IIIB; "
            "4) TRANSMEMBRANE ENZYME — unique feature: "
            "HGSNAT assay requires fresh leucocytes (not stable in dried blood spot as well as SGSH/NAGLU); "
            "some labs use fibroblasts for HGSNAT assay; "
            "5) SLIGHTLY MILDER COURSE: some series suggest MPS IIIC has somewhat later regression onset "
            "vs MPS IIIA (most severe) — not reliable for individual prediction; "
            "6) SOMATIC FEATURES MILD: subtle coarse facies; mild hepatosplenomegaly; no corneal clouding; "
            "7) SLEEP DISTURBANCE: severe; melatonin supplementation standard"
        ),
        "treatment": (
            "MPS IIIC — NO APPROVED DISEASE-MODIFYING THERAPY: "
            "ERT CONCEPTUALLY UNFEASIBLE: HGSNAT is a transmembrane enzyme — "
            "soluble recombinant enzyme cannot insert into lysosomal membrane; "
            "ERT delivery via M6P receptor pathway not applicable; "
            "GENE THERAPY (primary therapeutic strategy): "
            "scAAV9-HGSNAT: intraparenchymal or intrathecal delivery; phase I/II trials; "
            "HGSNAT cDNA large — rAAV packaging requires optimised constructs; "
            "HSC GENE THERAPY + EX VIVO: lentiviral HGSNAT-transduced HSCs — investigational; "
            "SUBSTRATE REDUCTION: "
            "Genistein: partial evidence; not approved; reduces HS production upstream; "
            "SYMPTOMATIC: identical to MPS IIIA/B: "
            "BEHAVIOURAL: risperidone; melatonin; clonidine; "
            "SEIZURES: levetiracetam, valproate; "
            "NUTRITION: PEG gastrostomy; "
            "PALLIATIVE CARE: respiratory support; anticipatory planning"
        ),
        "key_features": [
            "HGSNAT (MPS IIIC Sanfilippo C): AR; unique TRANSMEMBRANE lysosomal enzyme — only such in MPS; HS-only storage",
            "Clinically identical to MPS IIIA/B — enzyme panel of all four MPS III enzymes required to subtype",
            "HGSNAT transmembrane biology: conventional ERT infeasible — gene therapy is primary strategy",
            "Enzyme assay: HGSNAT best performed on fresh leukocytes or fibroblasts (less stable in DBS than SGSH/NAGLU)",
            "Behavioural phase PATHOGNOMONIC: hyperactivity + aggression + sleep disruption onset 2-4yr",
            "Step 3 of HS catabolism: re-acetylation step; requires acetyl-CoA from cytoplasm (unique biochemistry)",
            "Gene therapy (AAV9-HGSNAT) in phase I/II trials — must treat early before neuronal loss",
            "Slightly milder/later course vs MPS IIIA in some cohorts — not reliable for individual prognosis",
        ],
        "key_ddx": [
            "MPS IIIA (SGSH): identical phenotype; SGSH activity LOW; HGSNAT NORMAL — enzyme panel distinguishes",
            "MPS IIIB (NAGLU): identical phenotype; NAGLU activity LOW; HGSNAT NORMAL — enzyme panel",
            "MPS IIID (GNS): rarest Sanfilippo; GNS glucosamine-6-sulfatase low; same HS; enzyme panel",
            "Neuronal ceroid lipofuscinosis (NCL): neurodegeneration + behaviour; ceroid not GAG; different enzymes",
        ],
        "onset_age": 3.5,
        "coarse_facies_pct": 48,
        "corneal_clouding_pct": 0,
        "hepatosplenomegaly_pct": 48,
        "cognitive_decline_pct": 97,
        "behavioural_pct": 96,
        "sleep_disturbance_pct": 87,
        "seed": 2637,
    },
]

SEEDS = [g["seed"] for g in ATLAS_GENES]


def _simulate_cohort(gene: dict, seed: int) -> list:
    rng = random.Random(seed)
    pts = []
    n = 40
    for i in range(n):
        age_onset = gene.get("onset_age", 2.0) + rng.gauss(0, 1.5)
        age_onset = max(0.0, age_onset)
        coarse_facies = int(rng.random() < gene.get("coarse_facies_pct", 50) / 100)
        corneal_clouding = int(rng.random() < gene.get("corneal_clouding_pct", 30) / 100)
        hepatosplenomegaly = int(rng.random() < gene.get("hepatosplenomegaly_pct", 60) / 100)
        cognitive_decline = int(rng.random() < gene.get("cognitive_decline_pct", 40) / 100)
        behavioural = int(rng.random() < gene.get("behavioural_pct", 20) / 100)
        pts.append({
            "gene": gene["gene"],
            "patient_id": f"{gene['gene']}-{seed}-{i+1:03d}",
            "age_onset": round(age_onset, 1),
            "coarse_facies": coarse_facies,
            "corneal_clouding": corneal_clouding,
            "hepatosplenomegaly": hepatosplenomegaly,
            "cognitive_decline": cognitive_decline,
            "behavioural": behavioural,
            "skeletal_dysplasia": int(gene["gene"] in ("IDUA", "IDS", "GALNS", "ARSB", "GUSB") and rng.random() < 0.85),
            "odontoid_risk": int(gene["gene"] in ("IDUA", "IDS", "GALNS", "ARSB") and rng.random() < 0.60),
            "cardiac_valve": int(gene["gene"] in ("IDUA", "IDS", "ARSB", "GUSB") and rng.random() < 0.80),
            "hydrops": int(gene["gene"] == "GUSB" and rng.random() < 0.40),
            "skin_papules": int(gene["gene"] == "IDS" and rng.random() < 0.55),
            "sleep_disturbance": int(gene["gene"] in ("SGSH", "NAGLU", "HGSNAT") and rng.random() < 0.88),
            "seed": seed,
        })
    return pts


def generate_overview() -> dict:
    summary_by_gene = []
    all_pts = []
    for gene in ATLAS_GENES:
        pts = _simulate_cohort(gene, gene["seed"])
        all_pts.extend(pts)
        n = len(pts)
        summary_by_gene.append({
            "gene": gene["gene"],
            "locus": gene["locus"],
            "n_patients": n,
            "avg_onset_age": round(sum(p["age_onset"] for p in pts) / n, 1),
            "coarse_facies_pct": round(sum(p["coarse_facies"] for p in pts) / n * 100, 1),
            "corneal_clouding_pct": round(sum(p["corneal_clouding"] for p in pts) / n * 100, 1),
            "hepatosplenomegaly_pct": round(sum(p["hepatosplenomegaly"] for p in pts) / n * 100, 1),
            "cognitive_decline_pct": round(sum(p["cognitive_decline"] for p in pts) / n * 100, 1),
        })

    total = len(all_pts)
    return {
        "atlas": "Hereditary-MPS-Atlas",
        "genes": [g["gene"] for g in ATLAS_GENES],
        "n_genes": len(ATLAS_GENES),
        "total_patients": total,
        "seeds": f"{SEEDS[0]}-{SEEDS[-1]}",
        "gene_summaries": summary_by_gene,
        "aggregate_stats": {
            "overall_coarse_facies_pct": round(sum(p["coarse_facies"] for p in all_pts) / total * 100, 1),
            "overall_corneal_clouding_pct": round(sum(p["corneal_clouding"] for p in all_pts) / total * 100, 1),
            "overall_hepatosplenomegaly_pct": round(sum(p["hepatosplenomegaly"] for p in all_pts) / total * 100, 1),
            "overall_cognitive_decline_pct": round(sum(p["cognitive_decline"] for p in all_pts) / total * 100, 1),
        },
        "disease_classes": [
            f"{g['gene']} — {g['disease_category'].split(';')[0].strip()}"
            for g in ATLAS_GENES
        ],
        "key_clinical_distinctions": [
            "IDUA MPS I Hurler: AR; DS+HS; corneal clouding + coarse facies + ID; laronidase ERT FDA2003; HSCT curative if age<2yr DQ>70",
            "IDS MPS II Hunter: XLR males; DS+HS; NO CORNEAL CLOUDING (KEY DDx); pebbly skin lesions PATHOGNOMONIC; idursulfase ERT FDA2006",
            "SGSH MPS IIIA: AR; HS only; severe behaviour + regression; lagest Sanfilippo; no approved ERT; gene therapy phase I/II",
            "GALNS MPS IVA Morquio A: AR; KS+C6S; skeletal only + NORMAL INTELLECT; odontoid hypoplasia → tetraplegia risk; elosulfase alfa ERT FDA2014",
            "ARSB MPS VI: AR; DS only; somatic disease + NORMAL INTELLECT; galsulfase ERT FDA2005; corneal clouding present",
            "GUSB MPS VII Sly: AR; DS+HS+CS triple; non-immune hydrops PATHOGNOMONIC; vestronidase alfa ERT FDA2017; USA NBS 2024",
            "NAGLU MPS IIIB: AR; HS only; identical to IIIA phenotype; SGSH normal (differentiates from IIIA); BMN-250 NAGLU-IGF2 trials",
            "HGSNAT MPS IIIC: AR; HS only; TRANSMEMBRANE enzyme (unique) — conventional ERT infeasible; gene therapy primary strategy",
        ],
    }


def generate_breakdown() -> dict:
    gene_breakdowns = []
    for gene in ATLAS_GENES:
        pts = _simulate_cohort(gene, gene["seed"])
        n = len(pts)
        entry = {
            "gene": gene["gene"],
            "locus": gene["locus"],
            "protein_size": gene["protein_size"],
            "inheritance": gene["inheritance"],
            "disease_category": gene["disease_category"],
            "pathognomonic": gene["pathognomonic"],
            "treatment": gene["treatment"],
            "key_features": gene["key_features"],
            "key_ddx": gene["key_ddx"],
            "n_patients": n,
            "avg_onset_age": round(sum(p["age_onset"] for p in pts) / n, 1),
            "coarse_facies_pct": round(sum(p["coarse_facies"] for p in pts) / n * 100, 1),
            "corneal_clouding_pct": round(sum(p["corneal_clouding"] for p in pts) / n * 100, 1),
            "hepatosplenomegaly_pct": round(sum(p["hepatosplenomegaly"] for p in pts) / n * 100, 1),
            "cognitive_decline_pct": round(sum(p["cognitive_decline"] for p in pts) / n * 100, 1),
        }
        gene_breakdowns.append(entry)
    return {"gene_breakdowns": gene_breakdowns}


def generate_definitions() -> dict:
    gene_entries = {}
    for gene in ATLAS_GENES:
        gene_entries[gene["gene"]] = {
            "locus": gene["locus"],
            "protein_size": gene["protein_size"],
            "inheritance": gene["inheritance"].split(";")[0].strip(),
            "disease_name": gene["disease_category"].split(";")[0].strip(),
            "disease_pathway": gene["disease_pathway"],
            "pathognomonic": gene["pathognomonic"],
            "treatment_summary": gene["treatment"].split(";")[0].strip() + "...",
            "key_features": gene["key_features"],
            "key_ddx": gene["key_ddx"],
        }
    return {
        "gene_entries": gene_entries,
        "mps_glossary": {
            "MPS Classification and GAG Substrates": (
                "Mucopolysaccharidoses (MPS) are lysosomal storage disorders caused by defective "
                "catabolism of glycosaminoglycans (GAGs — also called mucopolysaccharides): "
                "1) MPS I (IDUA): dermatan sulfate + heparan sulfate → Hurler (severe) / Scheie (mild); "
                "2) MPS II (IDS): dermatan sulfate + heparan sulfate → Hunter; XLR males; NO corneal clouding; "
                "3) MPS III A-D (SGSH/NAGLU/HGSNAT/GNS): HEPARAN SULFATE ONLY → Sanfilippo; severe CNS; "
                "4) MPS IVA (GALNS): keratan sulfate + chondroitin-6-sulfate → Morquio A; skeletal only; "
                "5) MPS VI (ARSB): DERMATAN SULFATE ONLY → Maroteaux-Lamy; somatic; normal intellect; "
                "6) MPS VII (GUSB): DS + HS + chondroitin sulfate → Sly; broadest substrate; hydrops; "
                "KEY DISTINGUISHERS: "
                "Corneal clouding: MPS I YES, MPS II NO (PATHOGNOMONIC DIFFERENCE); "
                "Intellect: MPS I severe/intermediate = ID; MPS IVA, VI = NORMAL; "
                "XLR: only MPS II (IDS) — all others AR; "
                "Heparan sulfate: MPS I, II, III, VII; "
                "Dermatan sulfate: MPS I, II, VI, VII; "
                "Keratan sulfate: MPS IVA ONLY; "
                "Hydrops fetalis: MPS VII most common lysosomal cause."
            ),
            "Dysostosis Multiplex — Radiological Pattern": (
                "Dysostosis multiplex (DM): skeletal X-ray findings shared by MPS I, II, VI, VII (not MPS IVA which has different pattern): "
                "SKULL: J-shaped sella turcica (elongated); macrocephaly; "
                "SPINE: anterior inferior vertebral beaking (especially L1/L2); ovoid/biconcave vertebrae; "
                "RIBS: spatulate/oar-shaped ribs (wide posterior, narrow anterior); 'canoe paddle' appearance; "
                "CLAVICLES: short, wide, irregular; "
                "PELVIS: widened iliac wings (wing-like); acetabular obliquity; "
                "LONG BONES: shortened, widened diaphysis; "
                "HANDS/WRISTS: hypoplastic carpal bones; metacarpal pointing (proximal tapering); "
                "FEMORAL HEADS: flattened, dysplastic; "
                "MPS IVA (Morquio): PLATYSPONDYLY (flat vertebrae) + anterior vertebral central beak; "
                "odontoid hypoplasia (DIFFERENT from other MPS vertebral beaking); "
                "generalized joint laxity rather than stiffness; "
                "KEY CLINICAL POINT: dysostosis multiplex on X-ray + elevated urine GAG = MPS until proven otherwise."
            ),
            "Approved ERT for MPS — Clinical Comparison": (
                "MPS I: LARONIDASE (Aldurazyme) — FDA 2003; 0.58 mg/kg/week IV; "
                "improves visceral, respiratory, liver/spleen; does NOT cross BBB; "
                "Hurler requires HSCT for CNS (ERT as bridge). "
                "MPS II: IDURSULFASE (Elaprase) — FDA 2006; 0.5 mg/kg/week IV; "
                "somatic only; pabinafusp alfa (Izcargo, Japan 2021) for CNS Hunter. "
                "MPS IVA: ELOSULFASE ALFA (Vimizim) — FDA 2014; 2 mg/kg/week IV; "
                "functional improvement; does NOT reverse skeletal deformity. "
                "MPS VI: GALSULFASE (Naglazyme) — FDA 2005; 1 mg/kg/week IV; "
                "excellent somatic benefit; start as early as possible. "
                "MPS VII: VESTRONIDASE ALFA (Mepsevii) — FDA 2017; 4 mg/kg/q4w IV; "
                "accelerated approval; somatic benefit. "
                "NO ERT APPROVED: MPS IIIA (SGSH), MPS IIIB (NAGLU), MPS IIIC (HGSNAT) — "
                "IV enzyme cannot reach CNS adequately; gene therapy is primary strategy for Sanfilippo types. "
                "All ERT: premedication mandatory (antihistamine ± antipyretic); "
                "infusion reaction rate 20-30%; dose escalation protocols if reactions."
            ),
            "Odontoid Hypoplasia — MPS Anaesthetic and Surgical Risk": (
                "Odontoid (dens) hypoplasia: under-development of the C2 dens → "
                "atlantoaxial instability → risk of acute cervical cord compression → tetraplegia/death. "
                "AFFECTS: MPS IVA (most severe — ~80% of patients); also MPS I, II, VI. "
                "PREOPERATIVE PROTOCOL (ALL MPS with skeletal involvement): "
                "1) MRI cervical spine: assess odontoid size, cord signal, instability; "
                "2) Flexion-extension cervical X-rays: atlantodental interval (ADI) >4.5mm = unstable; "
                "3) Awake fibreoptic intubation if instability present; "
                "4) Neuromonitoring (SSEP, MEP) during any procedure; "
                "5) Prophylactic C1/C2 fusion if: instability + MRI cord signal change + planned major surgery. "
                "MPS IVA ADDITIONAL RISK: restrictive lung disease + difficult airway from soft tissue thickening; "
                "even routine dental procedures under GA carry significant risk in MPS IVA. "
                "ANAESTHETIC TEAM must be experienced in MPS — ear, nose, throat, neurosurgery, anaesthesia on standby."
            ),
        },
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(generate_overview(), indent=2)[:2000])
    print("\n=== BREAKDOWN (first gene) ===")
    bd = generate_breakdown()
    print(json.dumps(bd["gene_breakdowns"][0], indent=2)[:2000])
    print("\n=== DEFINITIONS (first entry) ===")
    defs = generate_definitions()
    first_gene = list(defs["gene_entries"].keys())[0]
    print(json.dumps(defs["gene_entries"][first_gene], indent=2)[:1000])
