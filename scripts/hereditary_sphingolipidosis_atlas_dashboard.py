"""Hereditary Sphingolipidosis Atlas — 8-Gene Reference
GBA1-GLA-HEXA-HEXB-GLB1-SMPD1-NPC1-ASAH1
320 patients (8 x 40), seeds 2622-2629.
Endpoints: /api/hereditary-sphingolipidosis-atlas/overview|breakdown|definitions
"""
import random

ATLAS_GENES = [
    {
        "gene": "GBA1",
        "protein": (
            "GBA1 -- 1q22 AR -- 497aa -- Glucocerebrosidase-GCase-62kDa-"
            "Lysosomal-Glucosylceramide-Hydrolase-Gaucher-Disease-AR -- OMIM-Gene-606463-Disease-Gaucher-230800"
        ),
        "locus": "1q22",
        "protein_size": "497 aa / 62 kDa",
        "inheritance": (
            "AR (biallelic loss of function); Gaucher Disease (GD); most common lysosomal storage disorder; "
            "Prevalence: 1:40,000 general population; 1:450-800 Ashkenazi Jewish (carrier frequency 1:14); "
            "Type 1 (non-neuronopathic): 95% all Gaucher — hepatosplenomegaly, bone disease, cytopenias; NO CNS; "
            "Type 2 (acute neuronopathic): rare, infantile, rapidly fatal; minimal enzyme activity; "
            "Type 3 (chronic neuronopathic): rare, juvenile; slowly progressive CNS; "
            "N370S mutation: ~75% Ashkenazi alleles; ONLY in Type 1 (never Type 2/3); "
            "L444P mutation: associated with Type 3; homozygous L444P often Type 2/3; "
            "GBA1 heterozygous carriers: 5× increased risk for Parkinson disease (PD)"
        ),
        "disease_category": (
            "Gaucher Disease (GD); lysosomal storage disorder; glucosphingolipidosis; "
            "GBA1 encodes glucocerebrosidase (GCase) — lysosomal hydrolase cleaving glucose from glucosylceramide; "
            "requires saposin C (PSAP gene) co-factor for lysosomal substrate presentation; "
            "Loss of GCase → glucosylceramide accumulates in lysosomes of macrophages/monocytes → "
            "GAUCHER CELLS (lipid-laden macrophages with crinkled-tissue-paper cytoplasm) accumulate in "
            "liver, spleen, bone marrow, lungs (Type 1); and brain (Type 2/3); "
            "IMIGLUCERASE (Cerezyme): ERT FDA 1994 — first approved ERT for any LSD; "
            "ELIGLUSTAT (Cerdelga): substrate reduction therapy (SRT) — small molecule; oral; Type 1 adults only; "
            "MIGLUSTAT: CNS-penetrant SRT; Type 3 CNS manifestations; not first-line Type 1"
        ),
        "disease_pathway": (
            "GBA1 encodes glucocerebrosidase (GCase, glucosylceramide beta-glucosidase), a lysosomal "
            "acid hydrolase that cleaves glucose from glucosylceramide (glucocerebroside). "
            "Saposin C (encoded by PSAP) presents the lipid substrate to GCase in the lysosomal lumen. "
            "Loss of GCase → glucosylceramide accumulates predominantly in macrophages/monocytes "
            "(highest turnover of glycosphingolipids from RBC and WBC membrane recycling). "
            "Gaucher cells (Kupffer cells in liver, Kupffer cell equivalents in spleen/marrow/lung): "
            "crinkled-tissue-paper cytoplasm on H&E — distended lysosomes packed with glucosylceramide. "
            "Bone: Gaucher cell infiltration of marrow → Erlenmeyer flask deformity (distal femur/proximal tibia); "
            "avascular necrosis (hip/shoulder); pathological fractures; bone crises (acute severe bone pain). "
            "GBA1 heterozygosity → mildly impaired lysosomal GCase → alpha-synuclein clearance impaired → "
            "PD risk 5× baseline (N370S/L444P heterozygous carriers; most common genetic PD risk factor)."
        ),
        "pathognomonic": (
            "GAUCHER CELLS = PATHOGNOMONIC: crinkled-tissue-paper (wrinkled silk) cytoplasm on bone marrow biopsy; "
            "PAS-positive, CD68+, tartrate-resistant acid phosphatase (TRAP) positive; "
            "HEPATOSPLENOMEGALY: massive splenomegaly (often 5-70× normal) — most common presenting sign; "
            "BONE DISEASE: Erlenmeyer flask deformity (loss of normal tubulation, distal femur), "
            "avascular necrosis (osteonecrosis), bone crises (acute severe bone pain ± fever); "
            "CYTOPENIAS: thrombocytopenia (splenomegaly + marrow infiltration), anaemia; "
            "PLASMA CHITOTRIOSIDASE: markedly elevated (>10× normal) — useful monitoring biomarker; "
            "PLASMA GLUCOSYLSPHINGOSINE (lyso-Gb1): highly sensitive/specific Gaucher biomarker; "
            "GBA1 enzyme activity: markedly low in leukocytes (dry blood spot for NBS); "
            "N370S MUTATION: Type 1 only — if homozygous or compound het with N370S, NEVER Type 2/3"
        ),
        "treatment": (
            "TYPE 1 GAUCHER: "
            "ENZYME REPLACEMENT THERAPY (ERT) — IV infusions every 2 weeks: "
            "Imiglucerase (Cerezyme) — FDA 1994, first LSD ERT; velaglucerase alfa (VPRIV); taliglucerase alfa (Elelyso); "
            "SUBSTRATE REDUCTION THERAPY (SRT) — oral, adults Type 1: "
            "Eliglustat (Cerdelga) — FDA 2014; CYP2D6 metaboliser testing MANDATORY (poor metabolisers need dose adjustment); "
            "do NOT use eliglustat with strong CYP2D6/3A4 inhibitors; "
            "Miglustat (Zavesca) — CNS-penetrant SRT; used for Type 3 CNS; not first-line Type 1 (GI side effects); "
            "TYPE 2 GAUCHER: no disease-modifying therapy; supportive only; "
            "TYPE 3 GAUCHER: ERT for visceral disease; miglustat for CNS component (limited evidence); "
            "GBA1 HETEROZYGOUS PD: GCase activator (ambroxol) investigational for neuroprotection; "
            "MONITORING: chitotriosidase, lyso-Gb1, LFTs, CBC, imaging; "
            "GENETIC COUNSELLING: AR; 25% recurrence; carrier frequency 1:14 Ashkenazi; PD risk counselling for carriers"
        ),
        "key_features": [
            "GBA1 (Gaucher): AR; most common LSD; 1:40,000 general; 1:450-800 Ashkenazi Jewish",
            "Gaucher cells (crinkled-tissue-paper cytoplasm, PAS+, TRAP+) PATHOGNOMONIC on bone marrow biopsy",
            "Massive splenomegaly, hepatomegaly, bone disease (Erlenmeyer flask + avascular necrosis), cytopenias",
            "Imiglucerase ERT FDA 1994 — first approved ERT for any LSD; highly effective Type 1",
            "Eliglustat SRT oral FDA 2014 — CYP2D6 metaboliser testing MANDATORY before prescribing",
            "N370S mutation: ONLY Type 1 (never neuronopathic); Ashkenazi founder; ~75% of Ashkenazi alleles",
            "GBA1 heterozygous carriers: 5× Parkinson disease risk — most common genetic PD risk factor",
            "Plasma lyso-Gb1 (glucosylsphingosine): sensitive/specific biomarker for diagnosis and monitoring",
        ],
        "key_ddx": [
            "Niemann-Pick A/B (SMPD1): foam cells + sphingomyelin; cherry-red spot NPD-A; no crinkled cells",
            "Niemann-Pick C (NPC1): vertical supranuclear gaze palsy; cholesterol; filipin test; not enzyme deficiency",
            "Leishmaniasis (visceral): pseudo-Gaucher cells; serology positive; splenic aspirate; not lysosomal storage",
            "Saposin C deficiency (PSAP): identical Gaucher phenotype; GBA enzyme NORMAL; saposin sequencing required",
        ],
        "onset_age": 25.0,
        "hepatosplenomegaly_pct": 95,
        "bone_disease_pct": 70,
        "cytopenias_pct": 80,
        "seizure_pct": 5,
        "seed": 2622,
    },
    {
        "gene": "GLA",
        "protein": (
            "GLA -- Xq22.1 XLR -- 429aa -- Alpha-Galactosidase-A-50kDa-"
            "Lysosomal-Globotriaosylceramide-Gb3-Hydrolase-Fabry-Disease-XLR -- OMIM-Gene-300644-Disease-Fabry-301500"
        ),
        "locus": "Xq22.1",
        "protein_size": "429 aa / 50 kDa",
        "inheritance": (
            "XLR (X-linked recessive); Fabry Disease (FD); "
            "Males: classic (severe, all features) or late-onset (cardiac/renal only); "
            "Females (heterozygous): 70% symptomatic — X-inactivation skewing determines severity; "
            "Prevalence: 1:40,000-50,000 males classic; up to 1:3,000 in cardiac/renal screening cohorts (late-onset); "
            "Classic males: neuropathic pain crises (Fabry crises) onset childhood; "
            "Angiokeratoma, hypohidrosis, cornea verticillata (slit-lamp) PATHOGNOMONIC in classic; "
            "Late-onset (missense with residual activity): cardiac (LVH, arrhythmia) and/or renal (proteinuria, CKD)"
        ),
        "disease_category": (
            "Fabry Disease (FD); lysosomal storage disorder; sphingolipidosis; "
            "GLA encodes alpha-galactosidase A (alpha-Gal A) — lysosomal hydrolase cleaving terminal galactose "
            "from globotriaosylceramide (Gb3 / ceramide trihexoside); "
            "Loss of alpha-Gal A → Gb3 accumulates in vascular endothelium, cardiomyocytes, podocytes, dorsal root ganglion neurons; "
            "Vascular Gb3 → endothelial dysfunction → stroke (posterior circulation predominant); "
            "Cardiac Gb3 → cardiomyocyte lysosomal storage → concentric LVH (HCM-like); "
            "Renal Gb3 → podocyte injury → proteinuria → FSGS → CKD → ESRD (males 40-50yr without treatment); "
            "Peripheral nerve Gb3 → small fibre neuropathy → neuropathic pain crises (BURNING pain, fever); "
            "AGALSIDASE BETA (Fabrazyme): ERT FDA 2003; AGALSIDASE ALFA (Replagal): EMA approved"
        ),
        "disease_pathway": (
            "GLA encodes alpha-galactosidase A (alpha-Gal A), a lysosomal exoglycosidase that removes "
            "terminal alpha-galactose residues from: "
            "1) Globotriaosylceramide (Gb3): primary accumulating substrate; "
            "2) Globotriaosylsphingosine (lyso-Gb3): secondary substrate; BIOMARKER for treatment response; "
            "3) Blood group B antigen (why Fabry males are often group O+ phenotype). "
            "Loss of alpha-Gal A → Gb3 accumulates in multiple cell types: "
            "Vascular endothelium: Gb3 deposits → endothelial activation → thrombus → stroke/TIA "
            "(posterior circulation: basilar, PICA; MRI white matter lesions); "
            "Cardiomyocytes: Gb3 storage → massive concentric LVH (Fabry cardiomyopathy); "
            "T-wave inversion inferolateral leads; short PR interval; later AF/flutter/complete heart block; "
            "Podocytes: Gb3 → lipid vacuolation → 'zebra bodies' on EM (PATHOGNOMONIC); "
            "DRG neurons: Gb3 in small-unmyelinated C-fibres → BURNING neuropathic pain."
        ),
        "pathognomonic": (
            "ANGIOKERATOMA CORPORIS DIFFUSUM = PATHOGNOMONIC (classic males): "
            "Dark red/purple punctate skin lesions, bathing-trunk distribution (periumbilical, buttocks, genitalia, thighs); "
            "CORNEA VERTICILLATA = PATHOGNOMONIC (slit-lamp examination): "
            "Whorled/vortex-pattern corneal opacities (Gb3 in corneal epithelium); "
            "present in >90% males and >70% carrier females; ASYMPTOMATIC (no visual loss); "
            "HYPOHIDROSIS/ANHIDROSIS: reduced sweating → heat intolerance; CLINICAL CLUE in children; "
            "NEUROPATHIC PAIN CRISES (Fabry crises): EPISODIC BURNING pain in extremities (hands/feet) "
            "triggered by fever, exercise, temperature changes; onset childhood; can mimic appendicitis/rheumatic fever; "
            "RENAL BIOPSY EM: 'ZEBRA BODIES' (myeloid bodies / concentric lamellar inclusions in podocytes) = PATHOGNOMONIC; "
            "ECG: short PR interval (<120ms) in classic males; "
            "LYSO-GB3 (plasma globotriaosylsphingosine): elevated in males AND symptomatic females; "
            "GLA enzyme activity: markedly low in males; UNRELIABLE in females (X-inactivation skewing); "
            "ALWAYS sequence GLA in females — enzyme activity alone insufficient"
        ),
        "treatment": (
            "ENZYME REPLACEMENT THERAPY (ERT): "
            "AGALSIDASE BETA (Fabrazyme, 1 mg/kg IV q2w): FDA 2003; first-line classic males, symptomatic females; "
            "AGALSIDASE ALFA (Replagal, 0.2 mg/kg IV q2w): EMA approved; dose-independent of body weight; "
            "ERT slows progression but does NOT reverse established fibrosis/sclerosis; "
            "START BEFORE irreversible organ damage (renal GFR >60, pre-LVH fibrosis); "
            "PHARMACOLOGICAL CHAPERONE: "
            "MIGALASTAT (Galafold) — FDA/EMA 2018: oral chaperone; only for amenable GLA variants; "
            "amenability database MANDATORY (ivacaftor analogy — not all mutations respond); "
            "CARDIAC: ICD for malignant arrhythmias; pacemaker complete heart block; "
            "antithrombotic: antiplatelet (aspirin) for cerebrovascular disease; "
            "RENAL: ACEi/ARB for proteinuria; renal transplant curative for ESRD (no recurrence — Gb3 clears); "
            "PAIN: carbamazepine/phenytoin/gabapentin for neuropathic pain crises (carbamazepine most evidence); "
            "GENETIC COUNSELLING: XLR; all daughters of affected male are obligate carriers; female sequencing; "
            "FAMILY SCREENING: lyso-Gb3 + GLA sequencing (not enzyme in females)"
        ),
        "key_features": [
            "GLA (Fabry): XLR; males severe classic; females heterozygous 70% symptomatic (X-inactivation)",
            "Angiokeratoma corporis diffusum PATHOGNOMONIC (bathing-trunk distribution, dark red punctate)",
            "Cornea verticillata PATHOGNOMONIC (slit-lamp whorled opacities; all classic males + 70% females)",
            "Zebra bodies (concentric lamellar inclusions) in podocytes EM PATHOGNOMONIC",
            "Episodic burning neuropathic pain crises in extremities (hands/feet) from childhood",
            "Agalsidase beta ERT FDA 2003; migalastat oral chaperone FDA 2018 (amenable variants only)",
            "GLA enzyme activity UNRELIABLE in females — always sequence GLA gene directly",
            "Renal transplant curative for ESRD — no post-transplant Gb3 recurrence",
        ],
        "key_ddx": [
            "HCM (sarcomere mutations): Fabry cardiomyopathy mimic; lyso-Gb3 + GLA sequencing discriminates",
            "Anderson-Fabry variant: late-onset cardiac-only; GLA missense with residual activity; lyso-Gb3 low-normal",
            "Small-fibre neuropathy (other causes): Fabry crises episodic + heat trigger; enzyme + lyso-Gb3 key",
            "Cryptogenic stroke in young: Fabry screening MANDATORY (especially posterior circulation TIA/stroke)",
        ],
        "onset_age": 8.0,
        "angiokeratoma_pct": 85,
        "renal_disease_pct": 75,
        "cardiac_pct": 65,
        "stroke_pct": 30,
        "neuropathy_pct": 90,
        "seed": 2623,
    },
    {
        "gene": "HEXA",
        "protein": (
            "HEXA -- 15q23 AR -- 529aa -- Hexosaminidase-A-Alpha-60kDa-"
            "Lysosomal-GM2-Ganglioside-Hydrolase-Tay-Sachs-GM2-Type-I-AR -- OMIM-Gene-606869-Disease-TaySachs-272800"
        ),
        "locus": "15q23",
        "protein_size": "529 aa / 60 kDa",
        "inheritance": (
            "AR (biallelic HEXA loss of function); Tay-Sachs Disease (TSD) / GM2 Gangliosidosis Type I; "
            "Prevalence: 1:320,000 general population; 1:3,500 Ashkenazi Jewish (carrier 1:30); "
            "Irish-American, French-Canadian, Louisiana Cajun: carrier 1:50; "
            "Classic infantile TSD: onset 3-6m, rapidly fatal (death by 4yr); minimal HexA activity; "
            "Juvenile TSD: onset 2-10yr; slower progression; "
            "Adult/late-onset TSD: onset >30yr; motor neuron disease phenotype + spinocerebellar + psychosis; "
            "THREE FOUNDER MUTATIONS cover >90% Ashkenazi alleles: "
            "1278insTATC (exon 11, frameshift) + IVS12+1G>C (splice) + G269S (missense, late-onset); "
            "HexA = heterodimer of alpha (HEXA) + beta (HEXB) subunits; "
            "HexA cleaves GM2 ganglioside (requires GM2AP co-activator protein)"
        ),
        "disease_category": (
            "Tay-Sachs Disease (GM2 Gangliosidosis Type I); lysosomal storage disorder; gangliosidosis; "
            "HEXA encodes the alpha subunit of hexosaminidase A (HexA); "
            "HexA (alpha-beta dimer) specifically cleaves GM2 ganglioside (requires GM2 activator protein, GM2AP); "
            "HexB (beta-beta dimer, HEXB gene) cleaves GA2 but NOT GM2 (no activator independent); "
            "Loss of HexA → GM2 ganglioside accumulates in neurons of CNS; "
            "Progressive neuronal destruction → cherry-red spot (macula), hypotonia, motor regression, blindness; "
            "CHERRY-RED SPOT (fundoscopy): retinal ganglion cell ring of GM2 storage appears white; "
            "central fovea (no ganglion cells) appears red against white = cherry-red spot; "
            "NO organomegaly (in contrast to Gaucher, NPC, NPD); neurons only; "
            "NO DISEASE-MODIFYING THERAPY — NBS + carrier screening central to prevention"
        ),
        "disease_pathway": (
            "HEXA encodes the alpha (alpha) subunit of lysosomal hexosaminidase A (HexA). "
            "HexA is an alpha-beta heterodimer (alpha from HEXA, beta from HEXB). "
            "HexA + GM2 activator protein (GM2AP, encoded by GM2A) form a functional complex that "
            "cleaves the terminal N-acetylgalactosamine from GM2 ganglioside → producing GM3. "
            "Loss of HEXA → no functional alpha subunit → HexA cannot form → GM2 accumulates in lysosomes "
            "of neurons (highest GM2 concentration in CNS). "
            "Progressive neuronal lysosomal distension → neuron swelling → meganeurites → axonal spheroids → "
            "death of motor neurons, cerebellar Purkinje cells, and cortical neurons. "
            "HexB (beta-beta dimer, Sandhoff disease) cleaves GA2 + asialo-GM2 but NOT GM2 (no activator); "
            "hence both HEXA and HEXB loss lead to GM2 accumulation — distinct diseases. "
            "Residual HexA activity >10% prevents infantile disease; G269S gives 4-10% → adult-onset."
        ),
        "pathognomonic": (
            "CHERRY-RED SPOT ON FUNDOSCOPY = PATHOGNOMONIC (infantile TSD): "
            "Ring of lipid-laden retinal ganglion cells appears white/grey; "
            "Central fovea (absent ganglion cells) remains red = CHERRY-RED SPOT; "
            "Present in >90% classic infantile; NOT in late-onset (ganglionic cells less affected); "
            "HYPERACUSIS (exaggerated startle response to sound): EARLY PATHOGNOMONIC feature in infantile; "
            "begins at 3-6m as first sign; auditory startle → whole-body myoclonic jerk; "
            "HYPOTONIA (floppy infant): progressive muscle weakness from neuronal degeneration; "
            "MRI BRAIN: early = normal → T2 thalamic hyperintensity (pulvinar) → progressive brain atrophy; "
            "HexA ENZYME ACTIVITY: markedly low (leukocytes or DBS); "
            "HexA/HexB RATIO: confirms HexA deficiency selectively; "
            "CARRIER SCREENING: serum HexA activity (most accurate for Ashkenazi); "
            "DNA mutation panel + enzyme = gold standard; "
            "No organomegaly — distinguishes from Gaucher, Niemann-Pick, GM1"
        ),
        "treatment": (
            "NO APPROVED DISEASE-MODIFYING THERAPY FOR CLASSIC TSD: "
            "SUBSTRATE REDUCTION (miglustat): NOT effective in classic infantile; limited trial data; "
            "GENE THERAPY (AAV9-HEXA): preclinical + early Phase I trials; "
            "PHARMACOLOGICAL CHAPERONE: pyrimethamine increases residual HexA in late-onset TSD; "
            "SUPPORTIVE MANAGEMENT: "
            "Anti-epileptic for seizures (very common infantile/juvenile); LEV, CLB, VPA; "
            "PEG tube feeding (dysphagia); chest physiotherapy; anti-spastic; "
            "LATE-ONSET TSD: riluzole if ALS/MND phenotype; physio + occupational therapy; psychiatric (psychosis); "
            "PREVENTION: PRIMARY PREVENTION IS THE ONLY EFFECTIVE STRATEGY; "
            "CARRIER SCREENING (Ashkenazi Jews): serum HexA + DNA MANDATORY pre-conception; "
            "NBS: not yet universally implemented; enables early diagnosis; "
            "PGT/prenatal: available for at-risk couples (1:900 Ashkenazi couple risk = 1:3,500 live birth); "
            "GENETIC COUNSELLING: AR; 25% recurrence; Ashkenazi Jewish carrier screening programs"
        ),
        "key_features": [
            "HEXA (Tay-Sachs): AR; 1:3,500 Ashkenazi Jewish; carrier 1:30; most common in Ashkenazi Jewish",
            "Cherry-red spot on fundoscopy PATHOGNOMONIC (infantile) — ring of grey ganglion cells + red fovea",
            "Hyperacusis (exaggerated startle to sound) earliest clinical feature 3-6 months PATHOGNOMONIC",
            "HexA deficiency: alpha subunit loss (vs Sandhoff = HexB beta subunit loss; same GM2 phenotype)",
            "NO organomegaly — pure neuronal disease; distinguishes from Gaucher/NPC/NPD",
            "NO approved disease-modifying therapy; carrier screening + PGT is primary prevention",
            "Three founder mutations cover >90% Ashkenazi alleles — targeted panel highly effective",
            "Adult/late-onset TSD: G269S missense; motor neuron disease + spinocerebellar degeneration + psychosis",
        ],
        "key_ddx": [
            "Sandhoff (HEXB): identical clinical to TSD; beta subunit; no HexA OR HexB (GM2+GA2 accumulate); no ethnic predilection",
            "GM2 activator deficiency (GM2A): same TSD phenotype; normal HexA AND HexB enzyme; GM2AP absent; ultra-rare",
            "GM1 gangliosidosis (GLB1): cherry-red spot; ALSO facial coarsening + hepatosplenomegaly (galactosyl substrates)",
            "Adult TSD vs ALS: TSD: HexA enzyme + lower motor neuron signs; cerebellar; psychiatric; UMN less prominent",
        ],
        "onset_age": 0.4,
        "cherry_red_spot_pct": 92,
        "seizure_pct": 75,
        "hyperacusis_pct": 90,
        "hypotonia_pct": 85,
        "seed": 2624,
    },
    {
        "gene": "HEXB",
        "protein": (
            "HEXB -- 5q13.3 AR -- 556aa -- Hexosaminidase-B-Beta-63kDa-"
            "Lysosomal-GM2+GA2-Ganglioside-Hydrolase-Sandhoff-Disease-GM2-Type-II-AR -- OMIM-Gene-606873-Disease-Sandhoff-268800"
        ),
        "locus": "5q13.3",
        "protein_size": "556 aa / 63 kDa",
        "inheritance": (
            "AR (biallelic HEXB loss of function); Sandhoff Disease (SD) / GM2 Gangliosidosis Type II; "
            "Prevalence: 1:310,000 general population; NO ethnic founder effect (unlike Tay-Sachs); "
            "Classic infantile SD: onset 3-6m; identical clinical to TSD; rapidly fatal; "
            "Juvenile SD: onset 2-10yr; slower progression; "
            "Adult SD: rare; motor neuron disease, spinocerebellar, psychosis; "
            "HexB = beta-beta homodimer encoded by HEXB; "
            "Loss of HEXB → BOTH HexA (alpha-beta) AND HexB (beta-beta) lose function → "
            "GM2 + GA2 + GA2 glycolipids ALL accumulate (broader substrate range than Tay-Sachs); "
            "KEY DISTINCTION: Sandhoff patients may show MILD VISCERAL STORAGE "
            "(liver/spleen slight enlargement in some) — TSD does NOT (purely neuronal)"
        ),
        "disease_category": (
            "Sandhoff Disease (GM2 Gangliosidosis Type II); lysosomal storage disorder; gangliosidosis; "
            "HEXB encodes the beta subunit common to both HexA (alpha-beta) and HexB (beta-beta); "
            "Loss of HEXB → HexA AND HexB both absent → "
            "accumulation of: GM2 ganglioside (same as TSD) + GA2 asialo-ganglioside + GA2 glycolipid "
            "in neurons AND (mildly) visceral organs; "
            "CHERRY-RED SPOT (fundoscopy) PATHOGNOMONIC: same mechanism as TSD; "
            "Clinically nearly indistinguishable from TSD: hyperacusis, hypotonia, motor regression; "
            "KEY DIFFERENCES from TSD: no ethnic predilection; HexA AND HexB both absent; GA2 also accumulates; "
            "slight hepatosplenomegaly in some Sandhoff patients; "
            "DIAGNOSIS: HexA + HexB enzyme activity BOTH low (vs TSD: HexA low, HexB NORMAL/elevated)"
        ),
        "disease_pathway": (
            "HEXB encodes the beta (beta) subunit, which is shared between: "
            "HexA (alpha-beta heterodimer): GM2 ganglioside substrate; requires GM2AP co-activator; "
            "HexB (beta-beta homodimer): GA2 asialo-GM2 substrate (no activator needed). "
            "Loss of HEXB → BOTH dimers absent → accumulation of: "
            "GM2 ganglioside (neurons) — same as TSD; "
            "GA2 (asialo-GM2) (neurons + visceral organs mildly) — additional compared to TSD; "
            "Hepatic GA2 storage explains mild hepatosplenomegaly (rare in TSD). "
            "Neuronal pathology: identical to TSD — ganglion cell lysosomal distension → meganeurites → "
            "cerebellar + cortical + motor neuron degeneration. "
            "CRITICAL DIAGNOSTIC DISTINCTION: "
            "In TSD: HexA activity LOW, HexB (beta-beta) activity NORMAL or ELEVATED; "
            "In Sandhoff: BOTH HexA AND HexB activities LOW — enzyme assay pattern distinguishes. "
            "Residual HexB activity level correlates with severity: null → infantile; hypomorphic → juvenile/adult."
        ),
        "pathognomonic": (
            "CHERRY-RED SPOT = PATHOGNOMONIC (same as TSD): "
            "Macula normal red; perifoveal ring of white ganglion cells filled with GM2 storage → contrast; "
            "HYPERACUSIS (exaggerated startle): earliest feature infantile Sandhoff, onset 3-6m, same as TSD; "
            "MILD HEPATOSPLENOMEGALY in some Sandhoff (rare in Tay-Sachs): "
            "GA2 storage in liver/spleen (visceral cells, not just neurons); "
            "clinical clue to Sandhoff vs TSD when present; "
            "MRI BRAIN: thalamic T2/FLAIR hyperintensity (pulvinar + caudate) — same as TSD; "
            "ENZYME ASSAY PATTERN PATHOGNOMONIC FOR SANDHOFF VS TSD: "
            "Sandhoff: BOTH HexA AND HexB low (using synthetic substrates: 4-MU-GlcNAc total; + heat denaturation); "
            "Tay-Sachs: HexA low, HexB NORMAL or elevated; "
            "HEXB SEQUENCING: biallelic variants confirm Sandhoff"
        ),
        "treatment": (
            "NO APPROVED DISEASE-MODIFYING THERAPY: "
            "Same management principles as Tay-Sachs: "
            "Anti-epileptic (seizures universal in infantile); "
            "PEG feeding (dysphagia progressive); chest physio; antispastic; "
            "SUBSTRATE REDUCTION: miglustat (crosses BBB, inhibits GluCer synthase) — no proven efficacy infantile; "
            "N-butyldeoxynojirimycin (NB-DNJ/miglustat): limited data adult Sandhoff MND phenotype; "
            "GENE THERAPY (AAV-HEXB): preclinical studies in mouse models; "
            "PYRIMETHAMINE: increases residual HexA activity for adult/late-onset — limited benefit; "
            "ADULT SANDHOFF: "
            "same as late-onset TSD management (motor neuron + spinocerebellar + psychiatric); "
            "riluzole if MND phenotype; physio + occupational therapy; "
            "GENETIC COUNSELLING: AR; 25% recurrence; no ethnic predilection — universal screening impractical; "
            "CARRIER TESTING: enzyme (HexA + HexB pattern) + sequencing for at-risk families"
        ),
        "key_features": [
            "HEXB (Sandhoff): AR; identical to Tay-Sachs clinically; NO ethnic founder effect; any ancestry",
            "Cherry-red spot PATHOGNOMONIC; hyperacusis early feature (onset 3-6m) same as TSD",
            "BOTH HexA AND HexB activities LOW (vs TSD: HexA low, HexB normal) — enzyme pattern distinguishes",
            "GA2 also accumulates (vs TSD GM2 only) → mild hepatosplenomegaly in some — TSD clue if absent",
            "Beta subunit shared between HexA and HexB — loss of HEXB abolishes both dimers",
            "NO approved disease-modifying therapy; AAV-HEXB gene therapy in preclinical development",
            "Adult Sandhoff: motor neuron disease + spinocerebellar + psychiatric (same as adult TSD)",
            "Enzyme pattern key: HexA+HexB both low = Sandhoff; HexA low only = Tay-Sachs; both normal = consider GM2AP",
        ],
        "key_ddx": [
            "Tay-Sachs (HEXA): clinically identical; HexB normal in TSD vs low in Sandhoff; Ashkenazi founder TSD",
            "GM2 activator deficiency (GM2AP): both HexA+HexB NORMAL; rare; enzyme-substrate complex absent",
            "GM1 gangliosidosis (GLB1): cherry-red spot + facial coarsening + hepatosplenomegaly + skeletal; beta-gal deficient",
            "Infantile NCL (CLN1): cherry-red-like macule; storage + seizures; ERG/EEG/electron microscopy granular osmiophilic deposits",
        ],
        "onset_age": 0.4,
        "cherry_red_spot_pct": 90,
        "seizure_pct": 78,
        "hyperacusis_pct": 88,
        "hepatosplenomegaly_pct": 35,
        "seed": 2625,
    },
    {
        "gene": "GLB1",
        "protein": (
            "GLB1 -- 3p22.3 AR -- 677aa -- Beta-Galactosidase-76kDa-"
            "Lysosomal-GM1+Keratan-Sulfate-Hydrolase-GM1-Gangliosidosis-MPS-IVB-AR -- OMIM-Gene-611458-Disease-GM1-230500"
        ),
        "locus": "3p22.3",
        "protein_size": "677 aa / 76 kDa",
        "inheritance": (
            "AR (biallelic GLB1 loss of function); "
            "GM1 Gangliosidosis (GLB1-deficiency) OR Morquio B Syndrome (MPS IVB); "
            "SAME GENE, DIFFERENT PHENOTYPES: "
            "Severe GLB1 loss → GM1 Gangliosidosis (neurological predominant); "
            "Mild GLB1 loss → MPS IVB (Morquio B, skeletal predominant — enzyme has some residual activity); "
            "GM1 Gangliosidosis: "
            "Type 1 (infantile): onset at birth; most severe; facial coarsening at birth; "
            "Type 2 (late-infantile/juvenile): onset 7m-3yr; slower; "
            "Type 3 (adult/chronic): onset adolescent-adult; dystonia predominant; "
            "MPS IVB (Morquio B): short stature, skeletal dysplasia, corneal clouding; minimal CNS involvement; "
            "Prevalence GM1 gangliosidosis: 1:100,000-200,000"
        ),
        "disease_category": (
            "GM1 Gangliosidosis / MPS IVB (Morquio B); lysosomal storage disorder; "
            "GLB1 encodes lysosomal beta-galactosidase — cleaves terminal galactose from: "
            "1) GM1 ganglioside: ganglion cells CNS — neurological disease; "
            "2) Keratan sulfate (KS): connective tissue — skeletal disease (MPS IVB); "
            "3) Galactooligosaccharides; "
            "requires protective protein/cathepsin A (PPCA, CTSA gene) for lysosomal stability; "
            "Loss of beta-Gal → GM1 + KS accumulate; "
            "FACIAL COARSENING (COARSE FACIES) AT BIRTH = PATHOGNOMONIC for infantile GM1; "
            "cherry-red spot in 50% infantile GM1 (vs 90%+ in TSD/Sandhoff); "
            "HEPATOSPLENOMEGALY: present (KS accumulates in liver/spleen) — distinguishes from TSD/Sandhoff; "
            "MACULAR CHERRY-RED SPOT present in ~50% infantile type"
        ),
        "disease_pathway": (
            "GLB1 encodes lysosomal acid beta-galactosidase, which requires cathepsin A (CTSA) as "
            "protective protein for lysosomal stability. "
            "GM1 Gangliosidosis mechanism: "
            "Beta-Gal cleaves terminal galactose from GM1 ganglioside (major brain ganglioside) → "
            "producing GM2 + galactose. Without beta-Gal → GM1 accumulates in neurons. "
            "Additionally cleaves keratan sulfate, galactose-containing oligosaccharides, and GA1. "
            "Infantile GM1: GM1 neuronal accumulation → pyramidal + extrapyramidal + cerebellar degeneration; "
            "Visceral KS storage → hepatosplenomegaly, facial coarsening (GAG accumulation in face). "
            "MPS IVB mechanism: "
            "Residual beta-Gal retains some GM1-cleaving ability but loses KS-cleaving ability selectively; "
            "KS accumulates in connective tissue (cartilage, bone) → skeletal dysplasia, short stature; "
            "neurological involvement minimal. "
            "PPCA deficiency (CTSA mutations) causes galactosialidosis (combined beta-Gal + neuraminidase deficiency)."
        ),
        "pathognomonic": (
            "FACIAL COARSENING AT BIRTH = PATHOGNOMONIC for INFANTILE GM1 type 1: "
            "Coarse facial features from birth (vs. Hurler which coarsens over months); "
            "frontal bossing, depressed nasal bridge, macroglossia, gingival hyperplasia; "
            "CHERRY-RED SPOT in ~50% infantile GM1 (less prominent than TSD — only some neurons affected); "
            "HEPATOSPLENOMEGALY from birth (KS visceral storage — unlike pure neuronal TSD/Sandhoff); "
            "SKELETAL DYSPLASIA: vertebral beaking (anterior inferior beaking on X-ray), kyphoscoliosis; "
            "CARDIAC: valvular disease (mitral/aortic thickening) from GAG storage; "
            "ADULT/CHRONIC TYPE 3: dystonia DOMINANT (dystonia musculorum deformans-like); "
            "mild cognitive decline; normal or mildly abnormal MRI; "
            "MPS IVB (Morquio B): "
            "Short stature, pectus carinatum, odontoid hypoplasia (cervical cord compression risk MANDATORY screening), "
            "corneal clouding, NO neurological involvement (unless cord compression); "
            "BETA-GALACTOSIDASE ENZYME ACTIVITY: low in leukocytes confirms both phenotypes"
        ),
        "treatment": (
            "GM1 GANGLIOSIDOSIS: "
            "NO APPROVED DISEASE-MODIFYING THERAPY; "
            "GENE THERAPY (AAV9-GLB1): Phase I/II trials (intrathecal/intracisternal) — promising preclinical; "
            "SUBSTRATE REDUCTION: miglustat (GM2-directed; limited data for GM1); N-acetylcysteine; "
            "BONE MARROW TRANSPLANT: limited benefit even pre-symptomatic; not standard; "
            "SUPPORTIVE: anti-epileptic (seizures); PEG feeding; spasm/spasticity (baclofen/tizanidine); "
            "CARDIAC surveillance; ophthalmology; "
            "MPS IVB (MORQUIO B): "
            "ERT INVESTIGATIONAL: vosoritide (bone-directed) trials; no approved enzyme replacement yet; "
            "ODONTOID HYPOPLASIA SURVEILLANCE: cervical MRI MANDATORY (cord compression → quadriplegia/death); "
            "C1-C2 fusion surgery if odontoid hypoplasia severe or instability; "
            "Orthopaedic management: bracing, corrective surgery; "
            "No neurological disease modifying treatment needed (minimal CNS in Morquio B); "
            "GENETIC COUNSELLING: AR; 25% recurrence; prenatal/PGT; PPCA/CTSA excluded by galactosialidosis panel"
        ),
        "key_features": [
            "GLB1 (GM1 / MPS IVB): AR; same gene — severe = GM1 gangliosidosis; mild residual = Morquio B skeletal",
            "Facial coarsening AT BIRTH PATHOGNOMONIC for infantile GM1 (distinguishes from coarsening that develops later)",
            "Cherry-red spot ~50% infantile (less than TSD/Sandhoff 90%+); hepatosplenomegaly present",
            "Hepatosplenomegaly distinguishes GM1 from pure neuronal TSD/Sandhoff (no organomegaly in those)",
            "Adult/chronic type 3: dystonia DOMINANT feature; cognitive decline mild",
            "MPS IVB (Morquio B): odontoid hypoplasia — cervical cord compression MUST screen; no CNS storage",
            "NO approved therapy for GM1 gangliosidosis; AAV9-GLB1 gene therapy Phase I/II in progress",
            "Galactosialidosis (CTSA/PPCA): combined beta-Gal + neuraminidase deficiency — same biopsy; exclude by panel",
        ],
        "key_ddx": [
            "Tay-Sachs/Sandhoff: cherry-red spot without facial coarsening or hepatosplenomegaly; GLB1 normal",
            "MPS I (Hurler): coarse facies develops over months (not at birth); IDUA enzyme; dermatan+heparan sulfate",
            "Galactosialidosis (CTSA): combined beta-Gal + neuraminidase deficiency; CTSA mutations; cherry-red spot",
            "Morquio A (MPS IVA, GALNS): keratan sulfate + chondroitin sulfate; GalNAc-6S-sulfatase; no beta-Gal",
        ],
        "onset_age": 0.2,
        "cherry_red_spot_pct": 50,
        "hepatosplenomegaly_pct": 85,
        "seizure_pct": 65,
        "facial_coarsening_pct": 90,
        "seed": 2626,
    },
    {
        "gene": "SMPD1",
        "protein": (
            "SMPD1 -- 11p15.4 AR -- 629aa -- Acid-Sphingomyelinase-ASMase-70kDa-"
            "Lysosomal-Sphingomyelin-Phosphocholine-Hydrolase-Niemann-Pick-A-B-AR -- OMIM-Gene-607608-Disease-NPD-257200"
        ),
        "locus": "11p15.4",
        "protein_size": "629 aa / 70 kDa",
        "inheritance": (
            "AR (biallelic SMPD1 loss of function); "
            "Niemann-Pick Disease Types A and B (NPD-A/B); acid sphingomyelinase deficiency (ASMD); "
            "SAME GENE: "
            "NPD-A: null mutations → <1% residual ASMase → severe neuronopathic + visceral disease; "
            "NPD-B: hypomorphic → 1-10% residual → visceral only (non-neuronopathic); "
            "Prevalence NPD-A: 1:40,000 Ashkenazi Jewish (carrier 1:100 Ashkenazi); "
            "NPD-A FOUNDER MUTATION: p.Arg496Leu (R496L) in Ashkenazi; also p.Leu302Pro, p.Phe333del; "
            "NPD-A: onset 3-6m, death by 4yr; hepatosplenomegaly + cherry-red spot + neurodegeneration; "
            "NPD-B: onset childhood, survival to adulthood; hepatosplenomegaly + lung disease; minimal CNS; "
            "INTERMEDIATE (A/B): mild neurological + severe visceral"
        ),
        "disease_category": (
            "Niemann-Pick Disease Type A and B (ASMD); lysosomal storage disorder; sphingomyelinase deficiency; "
            "SMPD1 encodes acid sphingomyelinase (ASMase) — lysosomal enzyme cleaving "
            "sphingomyelin → ceramide + phosphocholine; "
            "Loss of ASMase → sphingomyelin accumulates in lysosomes of macrophages/monocytes → "
            "FOAM CELLS (Niemann-Pick cells): lipid-laden macrophages with foamy cytoplasm; "
            "Foam cells in liver (Kupffer cells), spleen, bone marrow, lung (alveolar macrophages); "
            "In NPD-A: also in neurons (sphingomyelin in Purkinje cells and CNS neurons); "
            "OLIPUDASE ALFA (Xenpozyme): ERT FDA/EMA 2022 — first approved ERT for ASMD; "
            "NOTE: NPD-C (NPC1/NPC2) is a SEPARATE DISEASE — cholesterol trafficking, NOT sphingomyelinase"
        ),
        "disease_pathway": (
            "SMPD1 encodes lysosomal acid sphingomyelinase (ASMase), which cleaves sphingomyelin "
            "(sphingosine + fatty acid + phosphocholine head group) into ceramide + phosphocholine. "
            "Sphingomyelin is a major component of cell membranes, particularly in macrophages "
            "processing apoptotic cell debris (RBC, WBC turnover). "
            "Loss of ASMase → sphingomyelin accumulates in lysosomes of macrophages: "
            "FOAM CELLS (Niemann-Pick cells): distended lysosomes with sphingomyelin → foamy vacuolated cytoplasm; "
            "H&E: large pale foamy cells; PAS weakly positive; "
            "Lipid storage in: liver (hepatomegaly + cirrhosis risk), spleen (massive splenomegaly), "
            "bone marrow (marrow replacement), lungs (ground-glass opacities — NPC-B pulmonary disease severe), "
            "neurons (NPD-A: Purkinje cell loss → ataxia; cortical → dementia). "
            "Cherry-red spot (NPD-A): same mechanism as TSD — retinal ganglion cell sphingomyelin storage. "
            "SPHINGOMYELIN ≠ CHOLESTEROL: NPD-A/B is sphingomyelin storage; NPC1/2 is cholesterol trafficking."
        ),
        "pathognomonic": (
            "FOAM CELLS (NIEMANN-PICK CELLS) = PATHOGNOMONIC: "
            "Large foamy macrophages in bone marrow, spleen, liver biopsy; "
            "lipid vacuoles distend cytoplasm → foamy appearance (H&E); "
            "CHERRY-RED SPOT IN NPD-A (50-60%): same mechanism as TSD — retinal ganglion cell sphingomyelin; "
            "HEPATOSPLENOMEGALY: massive splenomegaly + hepatomegaly from birth/infancy; "
            "NPD-B PULMONARY DISEASE: progressive interstitial lung disease (ground-glass opacities on HRCT); "
            "pulmonary infiltration by foam cells → DLCO reduction → respiratory failure (major cause of death); "
            "CHERRY-RED SPOT ABSENT IN NPD-B (visceral only — no neuronal involvement); "
            "ASMase ENZYME ACTIVITY: markedly low in leukocytes/DBS (both NPD-A and NPD-B); "
            "SPHINGOMYELIN ACCUMULATION: plasma LysoSM-509 (lysosphingomyelin-509): sensitive/specific biomarker; "
            "CRITICAL: DO NOT confuse with NPC (Niemann-Pick C): "
            "NPD-A/B = sphingomyelinase deficiency (SMPD1); NPC = cholesterol trafficking (NPC1/NPC2)"
        ),
        "treatment": (
            "OLIPUDASE ALFA (Xenpozyme, recombinant ASMase) — FDA/EMA 2022: "
            "ERT for non-neuronopathic ASMD (NPD-B and intermediate) in adults and children; "
            "IV infusions every 4 weeks (q4w); "
            "CAUTION: FIRST DOSE MUST BE LOW (0.03 mg/kg); titrate up gradually over months; "
            "PULMONARY EXACERBATION risk at dose initiation (foam cell lysis releases stored lipids → inflammation); "
            "Pre-treat: dexamethasone + H1+H2 antagonist + acetaminophen before first several doses; "
            "NPD-A: NO approved therapy; supportive only; death by age 4yr; "
            "LUNG TRANSPLANT: occasionally for end-stage NPD-B lung disease; "
            "HSCT: limited evidence; does not correct CNS disease in NPD-A; "
            "SUPPORTIVE NPD-B: pulmonary function monitoring; HRCT; supplemental O2 if needed; "
            "lipid-lowering (secondary dyslipidaemia common); splenectomy if hypersplenism severe; "
            "GENETIC COUNSELLING: AR; 25% recurrence; Ashkenazi Jewish NPD-A carrier screening; "
            "R496L/L302P/F333del panel for Ashkenazi; full SMPD1 sequencing for non-Ashkenazi"
        ),
        "key_features": [
            "SMPD1 (NPD-A/B): AR; ASMase deficiency; sphingomyelin storage; Ashkenazi Jewish NPD-A founder mutations",
            "Foam cells (Niemann-Pick cells, foamy lipid-laden macrophages) PATHOGNOMONIC on bone marrow/biopsy",
            "NPD-A: severe neuronopathic + visceral; cherry-red spot; death by 4yr; NO approved therapy",
            "NPD-B: non-neuronopathic; hepatosplenomegaly + pulmonary disease; Olipudase alfa ERT FDA/EMA 2022",
            "Olipudase alfa: FIRST DOSE LOW (0.03 mg/kg); titrate slowly; pulmonary exacerbation risk at initiation",
            "CRITICAL DDx: NPD-A/B = sphingomyelinase (SMPD1); NPC1/2 = cholesterol trafficking — ENTIRELY DIFFERENT",
            "Plasma lysoSM-509: highly sensitive/specific biomarker for ASMD monitoring",
            "NPD-B pulmonary disease: HRCT ground-glass; DLCO reduction; major cause of mortality in NPD-B",
        ],
        "key_ddx": [
            "NPC1/2 (Niemann-Pick C): CHOLESTEROL trafficking (not sphingomyelin); filipin test positive; VSGP; miglustat",
            "Gaucher (GBA1): Gaucher cells (crinkled, not foamy); glucosylceramide; no cherry-red spot; ERT different",
            "NPD-B vs other pulmonary LSD: foam cell alveolar pattern on BAL; lysoSM-509 elevated; SMPD1 enzyme",
            "Wolman disease (LIPA): foam cells + adrenal calcification; lipase deficiency; sebelipase alfa ERT",
        ],
        "onset_age": 0.4,
        "hepatosplenomegaly_pct": 98,
        "cherry_red_spot_pct": 55,
        "pulmonary_pct": 60,
        "seizure_pct": 40,
        "seed": 2627,
    },
    {
        "gene": "NPC1",
        "protein": (
            "NPC1 -- 18q11.2 AR -- 1278aa -- NPC1-Cholesterol-Trafficking-Protein-145kDa-"
            "Late-Endosomal-Membrane-Sterol-Transporter-Niemann-Pick-C-AR -- OMIM-Gene-607623-Disease-NPC-257220"
        ),
        "locus": "18q11.2",
        "protein_size": "1278 aa / 145 kDa",
        "inheritance": (
            "AR (biallelic NPC1 loss of function — 95% of NPC cases; "
            "NPC2 (14q24.3) accounts for ~5% — same phenotype); "
            "Niemann-Pick Disease Type C (NPC); "
            "Prevalence: 1:120,000-150,000; "
            "Age of onset: neonatal-adult; WIDE SPECTRUM; "
            "Neonatal: hydrops fetalis, cholestatic jaundice (transient), acute liver failure; "
            "Childhood (2-10yr): most common — ataxia, clumsiness, learning difficulties; "
            "Adolescent/adult: psychiatric first (psychosis, bipolar, schizophrenia-like), then neurological; "
            "VESP (vertical supranuclear gaze palsy) PATHOGNOMONIC when present (~75%); "
            "I1061T mutation: ~20% NPC1 alleles worldwide; p.P1007A in Hispanic; "
            "KEY: NPC is NOT a sphingomyelinase deficiency — it is a CHOLESTEROL TRAFFICKING defect"
        ),
        "disease_category": (
            "Niemann-Pick Disease Type C (NPC); lysosomal cholesterol trafficking disorder; "
            "NPC1 encodes NPC1 protein — large late-endosomal membrane protein with sterol-sensing domain; "
            "NPC2 is a small soluble late-endosomal protein that binds cholesterol and transfers to NPC1; "
            "NPC1/NPC2 system: late-endosomal unesterified cholesterol → esterification → export to ER/plasma membrane; "
            "Loss of NPC1 → cholesterol (+ sphingolipids) TRAP in late endosomes/lysosomes; "
            "FILIPIN TEST (fluorescent staining of unesterified cholesterol in skin fibroblasts) = GOLD STANDARD DIAGNOSIS; "
            "PLASMA OXYSTEROLS (24S-OHC, 25-OHC, 3β,5α,6β-triol): biomarker panel; "
            "MIGLUSTAT: EU/Canada approved for progressive neurological manifestations (EMEA 2009); "
            "ARIMOCLOMOL: heat shock protein amplifier; Phase III trial completed"
        ),
        "disease_pathway": (
            "NPC1 and NPC2 form a cholesterol export system in late endosomes/lysosomes: "
            "LDL → endocytosis → late endosome; NPC2 binds unesterified cholesterol in lumen; "
            "NPC2 transfers cholesterol to NPC1 luminal loop → NPC1 transports cholesterol to inner leaflet; "
            "cholesterol exits to ER (for esterification/membrane synthesis) and plasma membrane. "
            "Loss of NPC1 (or NPC2): cholesterol + glycosphingolipids (GM2, GM3, glucosylceramide) "
            "TRAP in late endosomes/lysosomes → secondary sphingolipid accumulation. "
            "CNS: cholesterol trafficking failure in neurons → axonal degeneration → neurofibrillary tangles "
            "(same tau pathology as Alzheimer disease); Purkinje cell loss (ataxia); "
            "periaxonal cholesterol accumulation → demyelination. "
            "VSGP mechanism: impaired saccade pathway (frontal eye fields → superior colliculus) — "
            "vertical gaze most sensitive; horizontal preserved until late. "
            "FILIPIN test: skin fibroblasts cultured + filipin staining → bright perinuclear cholesterol fluorescence "
            "in NPC fibroblasts (unesterified cholesterol trapped in late endosomes)."
        ),
        "pathognomonic": (
            "VERTICAL SUPRANUCLEAR GAZE PALSY (VSGP) = PATHOGNOMONIC (~75% NPC): "
            "Impaired voluntary vertical saccades (up > down initially) WITH preserved oculocephalic reflex; "
            "PATHOGNOMONIC for NPC when combined with neurological regression; "
            "absent/delayed in young children — clue: slow saccade velocity on horizontal + vertical; "
            "GELASTIC CATAPLEXY: cataplexy triggered by laughter — PATHOGNOMONIC when present (40-50%); "
            "sudden loss of muscle tone, maintained consciousness, triggered by surprise/laughter; "
            "NPC cataplexy different from narcolepsy (no hypocretin deficiency, HLA-DQ specific); "
            "FILIPIN TEST (skin fibroblasts): perinuclear fluorescence of trapped unesterified cholesterol = "
            "PATHOGNOMONIC for NPC; some NPC1 variants give 'variant' pattern — requires NPC1/NPC2 sequencing; "
            "PLASMA OXYSTEROLS (7-ketocholesterol, 25-hydroxycholesterol, 3beta,5alpha,6beta-triolcholesterol): "
            "elevated — sensitive biomarker; now preferred screening test over filipin; "
            "AUDITORY DYSFUNCTION: sensorineural hearing loss from cochlear nerve involvement; "
            "HEPATIC NPC: cholestatic jaundice neonatal → may self-resolve → reappears with neurological"
        ),
        "treatment": (
            "MIGLUSTAT (Zavesca, N-butyldeoxynojirimycin) — EU/Canada/other approved 2009 for NPC neurological disease: "
            "Substrate reduction therapy (SRT): inhibits glucosylceramide synthase → reduces sphingolipid substrate; "
            "slows neurological progression (ataxia, cognitive decline); does NOT reverse established damage; "
            "GI side effects (diarrhoea): improves with low-carbohydrate diet; treatable; "
            "TREMOR: miglustat may worsen tremor; monitor; "
            "ARIMOCLOMOL (investigational): HSP70 co-inducer; Phase III completed; "
            "INTRATHECAL CYCLODEXTRIN (HPbetaCD): Phase II/III; bypasses BBB; dissolves cholesterol complexes; "
            "most promising CNS-directed therapy; hydroxypropyl-beta-cyclodextrin; compassionate use ongoing; "
            "SUPPORTIVE: "
            "CATAPLEXY: clomipramine, venlafaxine, sodium oxybate; "
            "SEIZURES: antiepileptic (very common infantile/juvenile NPC); "
            "DYSPHAGIA: PEG tube; swallowing therapy; "
            "PSYCHIATRIC: antipsychotics if needed (CAUTION: neuroleptic malignant syndrome risk in NPC); "
            "GENETIC COUNSELLING: AR; 25%; NPC1 sequencing (95%) + filipin + oxysterols; NPC2 (5%) if NPC1 normal"
        ),
        "key_features": [
            "NPC1 (Niemann-Pick C): AR; CHOLESTEROL TRAFFICKING — NOT sphingomyelinase deficiency (distinct from NPD-A/B)",
            "Vertical supranuclear gaze palsy (VSGP) PATHOGNOMONIC — vertical saccades impaired; oculocephalic reflex intact",
            "Gelastic cataplexy (laughter-triggered tone loss) PATHOGNOMONIC when present (40-50%)",
            "Filipin test (skin fibroblasts): perinuclear cholesterol fluorescence PATHOGNOMONIC — gold standard",
            "Plasma oxysterols (7-ketocholesterol): preferred screening biomarker (easier than filipin)",
            "Miglustat SRT EU/Canada approved for NPC neurological progression; GI side effects manageable",
            "Intrathecal cyclodextrin (HPbetaCD): most promising investigational therapy; Phase II/III",
            "I1061T: most common NPC1 allele worldwide (~20% alleles); amenable to HPbetaCD",
        ],
        "key_ddx": [
            "NPD-A/B (SMPD1): sphingomyelinase deficiency; foam cells; NO VSGP; NO filipin pattern; no gaze palsy",
            "Niemann-Pick C type 2 (NPC2 gene, 5%): same NPC phenotype; soluble cholesterol binding protein",
            "Juvenile onset ataxia/psychiatric: VSGP + oxysterols distinguish NPC from Wilson, spinocerebellar ataxias",
            "Progressive supranuclear palsy (PSP): tau; late onset; different demographics; oxysterols normal",
        ],
        "onset_age": 6.0,
        "vsgp_pct": 75,
        "cataplexy_pct": 45,
        "ataxia_pct": 85,
        "seizure_pct": 50,
        "hepatosplenomegaly_pct": 70,
        "seed": 2628,
    },
    {
        "gene": "ASAH1",
        "protein": (
            "ASAH1 -- 8p22 AR -- 395aa -- Acid-Ceramidase-53kDa-"
            "Lysosomal-Ceramide-Hydrolase-Farber-Lipogranulomatosis-AR -- OMIM-Gene-613468-Disease-Farber-228000"
        ),
        "locus": "8p22",
        "protein_size": "395 aa / 53 kDa",
        "inheritance": (
            "AR (biallelic ASAH1 loss of function); Farber Disease (Farber Lipogranulomatosis, FLD); "
            "Most rare sphingolipid storage disorder; prevalence <1:1,000,000; "
            "SAME GENE: ASAH1 hypomorphic → SMA-PME (spinal muscular atrophy with progressive myoclonic epilepsy); "
            "Classic Farber: onset 2-4 months; TRIAD = PATHOGNOMONIC: "
            "1) Periarticular subcutaneous NODULES; 2) JOINT DEFORMITY/CONTRACTURES; 3) HOARSE VOICE; "
            "5 clinical subtypes (1-5) by severity + presence of neurological involvement; "
            "Type 1 (classic): triad + neurological decline; death by 2-3yr; "
            "Type 5 (neurological): without triad; cerebral and spinal storage predominant; "
            "SMA-PME variant: childhood onset; lower motor neuron weakness + myoclonic epilepsy; minimal arthritis"
        ),
        "disease_category": (
            "Farber Disease (Farber Lipogranulomatosis); lysosomal ceramide storage disorder; "
            "ASAH1 encodes acid ceramidase (N-acylsphingosine amidohydrolase 1) — "
            "lysosomal enzyme cleaving ceramide into sphingosine + fatty acid; "
            "Loss of acid ceramidase → CERAMIDE accumulates in lysosomes of connective tissue, joints, "
            "macrophages/monocytes, neurons, liver, lung, heart; "
            "LIPOGRANULOMAS: periarticular ceramide storage triggers macrophage/foam cell accumulation → "
            "granulomatous nodules under skin + around joints; "
            "TRIAD (PATHOGNOMONIC): subcutaneous nodules + joint deformity + hoarse voice (laryngeal involvement); "
            "Hoarse voice from ceramide storage in laryngeal cartilage/mucosa → progressive hoarseness/stridor; "
            "SPINAL CORD/BRAIN: ceramide storage in neurons + Schwann cells (infantile neurological subtypes); "
            "SMA-PME: ASAH1 partial loss → ceramide in anterior horn cells + neurons"
        ),
        "disease_pathway": (
            "ASAH1 encodes acid ceramidase (ACDase), a lysosomal amidase that cleaves the amide bond "
            "between the sphingosine backbone and the fatty acid chain of ceramide → "
            "producing sphingosine + free fatty acid. "
            "Ceramide is generated in lysosomes from: "
            "1) Sphingomyelin hydrolysis by acid sphingomyelinase (ASMase) — produces ceramide + phosphocholine; "
            "2) Glucocerebrosidase activity on glucosylceramide; "
            "3) Galactocerebrosidase (GALC) on galactosylceramide. "
            "ACDase is the terminal hydrolase for ceramide clearance. "
            "Loss of ACDase → ceramide accumulates: "
            "CONNECTIVE TISSUE: ceramide attracts macrophages → lipogranuloma formation → periarticular nodules; "
            "JOINTS: ceramide in synovial cells + cartilage → destruction + contractures; "
            "LARYNX: ceramide in epithelium + cartilage → hoarse voice + stridor (EARLIEST SYMPTOM); "
            "CNS: ceramide in neurons → neurodegeneration (Type 1 severe, SMA-PME); "
            "LIVER: ceramide storage in Kupffer cells → hepatomegaly. "
            "SMA-PME: partial loss → ceramide in anterior horn cells → lower motor neuron disease."
        ),
        "pathognomonic": (
            "FARBER TRIAD = PATHOGNOMONIC: "
            "1) PERIARTICULAR SUBCUTANEOUS NODULES: firm, tender, lipogranulomas at pressure points "
            "(knuckles, wrists, ankles, elbows, spine); first sign at 2-4 months; "
            "2) JOINT CONTRACTURES / DEFORMITY: progressive joint swelling → contractures → articular destruction; "
            "often initially diagnosed as juvenile idiopathic arthritis; "
            "3) HOARSE VOICE (EARLIEST): hoarseness → stridor → respiratory insufficiency from laryngeal ceramide storage; "
            "cry hoarse from first months of life — EARLIEST CLINICAL CLUE; "
            "ALL THREE = PATHOGNOMONIC for Farber disease; any two should trigger ceramide testing; "
            "LIPOGRANULOMA BIOPSY: 'banana bodies' and 'Farber bodies' on EM — curvilinear tubular inclusions; "
            "CERAMIDE IN TISSUE: elevated in urine + tissue; "
            "ACID CERAMIDASE ENZYME: markedly reduced in leukocytes, skin fibroblasts; "
            "CERAMIDE:SPHINGOSINE RATIO: ceramide elevated, sphingosine reduced; "
            "SMA-PME: lower motor neuron wasting + myoclonic epilepsy; NO nodules/arthritis/hoarseness"
        ),
        "treatment": (
            "NO APPROVED DISEASE-MODIFYING THERAPY FOR FARBER: "
            "ALLOGENEIC HSCT: substantially improves VISCERAL disease and nodules/arthritis; "
            "DOES NOT improve neurological disease (CNS ceramide persists); "
            "best for Type 3 and Type 4 (visceral ± minimal neurological); "
            "GENE THERAPY (AAV-ASAH1): highly effective in mouse models; preclinical; "
            "SUPPORTIVE MANAGEMENT: "
            "AIRWAY: intubation/tracheostomy for laryngeal involvement; "
            "JOINTS: physiotherapy, bracing, analgesia (NSAIDs, opioids); surgical release if contractures severe; "
            "ANTI-EPILEPTIC: SMA-PME variant (myoclonic epilepsy): LEV, VPA (POLG check), TPM; "
            "VPA USE: myoclonic epilepsy is common indication but POLG overlap must exclude first; "
            "RILUZOLE: SMA-PME MND component (limited evidence); "
            "GENETIC COUNSELLING: AR; 25% recurrence; extremely rare — no ethnic founder; "
            "PRENATAL/PGT: enzyme assay on chorionic villi; ASAH1 sequencing"
        ),
        "key_features": [
            "ASAH1 (Farber): AR; rarest sphingolipidosis; acid ceramidase deficiency; ceramide storage",
            "Farber Triad PATHOGNOMONIC: periarticular nodules + joint deformity + hoarse voice from infancy",
            "Hoarse cry from first months = EARLIEST clinical clue — laryngeal ceramide storage",
            "Lipogranuloma biopsy: 'Farber bodies' (curvilinear tubular inclusions) on EM PATHOGNOMONIC",
            "HSCT substantially improves visceral/joint disease but does NOT reverse neurological disease",
            "Same gene (ASAH1) hypomorphic → SMA-PME: lower motor neuron + myoclonic epilepsy, NO triad",
            "NO approved disease-modifying therapy; AAV-ASAH1 gene therapy preclinical — highly effective in mice",
            "Frequently misdiagnosed as JIA (juvenile idiopathic arthritis) before nodules/hoarseness recognized",
        ],
        "key_ddx": [
            "JIA (juvenile idiopathic arthritis): joint swelling + deformity — NO subcutaneous nodules; normal ceramidase",
            "Multicentric reticulohistiocytosis: periarticular nodules + arthritis; NO hoarseness; lipid granuloma; no ceramide",
            "Gaucher (GBA1): hepatosplenomegaly + Gaucher cells; NO nodules or hoarseness; different enzyme",
            "SMA-PME vs SMA-classic: same ASAH1 gene; SMA-PME has myoclonic epilepsy + ceramide elevation; SMA-classic is SMN1",
        ],
        "onset_age": 0.3,
        "nodules_pct": 95,
        "joint_contracture_pct": 92,
        "hoarse_voice_pct": 97,
        "hepatosplenomegaly_pct": 70,
        "seizure_pct": 35,
        "seed": 2629,
    },
]

SEEDS = [g["seed"] for g in ATLAS_GENES]


def _simulate_cohort(gene: dict, seed: int) -> list:
    rng = random.Random(seed)
    pts = []
    n = 40
    for i in range(n):
        age_onset = gene.get("onset_age", 2.0) + rng.gauss(0, 2.0)
        age_onset = max(0.1, age_onset)
        hepatosplenomegaly = int(rng.random() < gene.get("hepatosplenomegaly_pct", 30) / 100)
        seizure = int(rng.random() < gene.get("seizure_pct", 20) / 100)
        pts.append({
            "gene": gene["gene"],
            "patient_id": f"{gene['gene']}-{seed}-{i+1:03d}",
            "age_onset": round(age_onset, 1),
            "hepatosplenomegaly": hepatosplenomegaly,
            "seizure": seizure,
            "cherry_red_spot": int(rng.random() < gene.get("cherry_red_spot_pct", 10) / 100),
            "neuropathy": int(rng.random() < gene.get("neuropathy_pct", 10) / 100),
            "bone_disease": int(gene["gene"] == "GBA1" and rng.random() < 0.70),
            "angiokeratoma": int(gene["gene"] == "GLA" and rng.random() < 0.85),
            "vsgp": int(gene["gene"] == "NPC1" and rng.random() < 0.75),
            "cataplexy": int(gene["gene"] == "NPC1" and rng.random() < 0.45),
            "nodules": int(gene["gene"] == "ASAH1" and rng.random() < 0.95),
            "hoarse_voice": int(gene["gene"] == "ASAH1" and rng.random() < 0.97),
            "foam_cells": int(gene["gene"] == "SMPD1" and rng.random() < 0.90),
            "facial_coarsening": int(gene["gene"] == "GLB1" and rng.random() < 0.90),
            "hyperacusis": int(gene["gene"] in ("HEXA", "HEXB") and rng.random() < 0.88),
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
            "hepatosplenomegaly_pct": round(sum(p["hepatosplenomegaly"] for p in pts) / n * 100, 1),
            "seizure_pct": round(sum(p["seizure"] for p in pts) / n * 100, 1),
            "cherry_red_spot_pct": round(sum(p["cherry_red_spot"] for p in pts) / n * 100, 1),
        })

    total = len(all_pts)
    return {
        "atlas": "Hereditary-Sphingolipidosis-Atlas",
        "genes": [g["gene"] for g in ATLAS_GENES],
        "n_genes": len(ATLAS_GENES),
        "total_patients": total,
        "seeds": f"{SEEDS[0]}-{SEEDS[-1]}",
        "gene_summaries": summary_by_gene,
        "aggregate_stats": {
            "overall_hepatosplenomegaly_pct": round(sum(p["hepatosplenomegaly"] for p in all_pts) / total * 100, 1),
            "overall_seizure_pct": round(sum(p["seizure"] for p in all_pts) / total * 100, 1),
            "overall_cherry_red_spot_pct": round(sum(p["cherry_red_spot"] for p in all_pts) / total * 100, 1),
        },
        "disease_classes": [
            f"{g['gene']} — {g['disease_category'].split(';')[0].strip()}"
            for g in ATLAS_GENES
        ],
        "key_clinical_distinctions": [
            "GBA1 Gaucher: glucosylceramide; Gaucher cells (crinkled-tissue-paper) PATHOGNOMONIC; Imiglucerase ERT FDA1994; Eliglustat SRT FDA2014; GBA1 heterozygous = 5× PD risk",
            "GLA Fabry: Gb3; XLR; angiokeratoma + cornea verticillata + zebra bodies PATHOGNOMONIC; Agalsidase beta ERT FDA2003; Migalastat chaperone FDA2018 (amenable variants only)",
            "HEXA Tay-Sachs: GM2; cherry-red spot + hyperacusis PATHOGNOMONIC; Ashkenazi founder; NO ERT; carrier screening prevents disease",
            "HEXB Sandhoff: GM2+GA2; BOTH HexA AND HexB low (vs TSD HexA only); cherry-red spot; mild hepatosplenomegaly; NO ERT",
            "GLB1 GM1: GM1+KS; facial coarsening AT BIRTH PATHOGNOMONIC; cherry-red ~50%; hepatosplenomegaly; Morquio B (same gene)",
            "SMPD1 NPD-A/B: sphingomyelin; foam cells PATHOGNOMONIC; Olipudase alfa ERT FDA/EMA2022 for NPD-B; NPD-A no therapy; NOT NPC",
            "NPC1 Niemann-Pick C: CHOLESTEROL TRAFFICKING (not enzyme); VSGP + gelastic cataplexy PATHOGNOMONIC; Filipin test; Miglustat EU approved",
            "ASAH1 Farber: ceramide; Farber TRIAD PATHOGNOMONIC (nodules+joint+hoarse voice); rarest; HSCT helps visceral; NO ERT",
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
            "hepatosplenomegaly_pct": round(sum(p["hepatosplenomegaly"] for p in pts) / n * 100, 1),
            "seizure_pct": round(sum(p["seizure"] for p in pts) / n * 100, 1),
            "cherry_red_spot_pct": round(sum(p["cherry_red_spot"] for p in pts) / n * 100, 1),
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
        "sphingolipidosis_glossary": {
            "Sphingolipidosis Classification": (
                "Sphingolipidoses are lysosomal storage disorders caused by defective catabolism "
                "of sphingolipids — complex lipids with sphingosine backbone: "
                "1) Glucosylceramide storage (GBA1-Gaucher): macrophages; visceral + bone; ERT available; "
                "2) Gb3 storage (GLA-Fabry): endothelium + neurons; X-linked; ERT + chaperone available; "
                "3) GM2 gangliosidosis (HEXA-TSD, HEXB-Sandhoff): neurons; cherry-red spot; no ERT; "
                "4) GM1 gangliosidosis (GLB1): neurons + connective tissue; facial coarsening; "
                "5) Sphingomyelin storage (SMPD1-NPD-A/B): macrophages + neurons; Olipudase ERT NPD-B; "
                "6) Cholesterol trafficking (NPC1-NPC): NOT a sphingolipidosis enzyme defect — "
                "  secondary sphingolipid accumulation due to cholesterol transport block; "
                "7) Ceramide storage (ASAH1-Farber): connective tissue + joints; rarest; no ERT. "
                "NPC is the KEY EXCEPTION: it is not a sphingolipid hydrolase deficiency. "
                "Cherry-red spot: retinal ganglion GM1/GM2/sphingomyelin storage — TSD>Sandhoff>NPD-A>GM1."
            ),
            "Enzyme Replacement Therapy Landscape in Sphingolipidoses": (
                "ERT approved for: "
                "Gaucher Type 1/3 (imiglucerase/velaglucerase/taliglucerase alfa, 1 mg/kg q2w IV); "
                "Fabry (agalsidase beta 1 mg/kg q2w IV; agalsidase alfa 0.2 mg/kg q2w IV); "
                "NPD-B/intermediate ASMD (olipudase alfa q4w IV; LOW initial dose 0.03 mg/kg — CRITICAL safety rule); "
                "ERT NOT available for: Tay-Sachs, Sandhoff, GM1 gangliosidosis, Farber, NPC (non-enzyme); "
                "ORAL THERAPY available: eliglustat SRT (Gaucher Type 1 adults; CYP2D6 metaboliser test mandatory); "
                "  migalastat (Fabry amenable variants; amenability database essential); "
                "  miglustat SRT (NPC neurological EU/Canada approved; Gaucher Type 3 CNS limited); "
                "Gene therapy (AAV-based) in trials: GLB1, HEXA, HEXB, ASAH1, GBA1-PD."
            ),
            "Cherry-Red Spot: Pathophysiology and Differential": (
                "CHERRY-RED SPOT: macular sign caused by lysosomal storage in retinal ganglion cells. "
                "Retinal ganglion cells form a ring around the macula; fovea has no ganglion cells. "
                "When ganglion cells accumulate sphingolipid (GM1, GM2, sphingomyelin): "
                "storage-distended cells appear pale/grey-white on fundoscopy; "
                "foveal area (no storage, normal blood) appears CHERRY-RED against white ring. "
                "Sphingolipidosis frequency: TSD (>90%) > Sandhoff (90%) > NPD-A (55%) > GM1 (50%) > NPC (rare). "
                "KEY: NPC cherry-red spot is UNCOMMON; Gaucher and Fabry do NOT cause cherry-red spot. "
                "Non-sphingolipidosis cherry-red spot: "
                "Central retinal artery occlusion (CRAO): acute unilateral; no storage disease; "
                "NCL (infantile): ceroid storage not sphingolipid. "
                "CHERRY-RED SPOT + HYPERACUSIS = Tay-Sachs / Sandhoff until proven otherwise."
            ),
            "Filipin Test vs Oxysterols in NPC Diagnosis": (
                "Niemann-Pick C diagnosis: TWO complementary tests. "
                "FILIPIN TEST: skin fibroblasts cultured in cholesterol-rich medium; filipin (polyene antibiotic) "
                "binds unesterified cholesterol → fluorescence under UV; "
                "NPC fibroblasts: perinuclear ring of bright filipin fluorescence (trapped cholesterol). "
                "Classic pattern (>80% NPC): strongly positive; "
                "Variant pattern (15-20%): weaker staining — must confirm with NPC1/NPC2 sequencing. "
                "PLASMA OXYSTEROLS (7-ketocholesterol, 25-hydroxycholesterol, 3β,5α,6β-triolcholesterol): "
                "markedly elevated in NPC (cholesterol oxidation products from improperly trafficked cholesterol); "
                "7-ketocholesterol: sensitivity 99%; specificity 95%; easier to perform than filipin; "
                "NOW FIRST-LINE SCREENING TEST (oxysterols) → confirmed by NPC1/NPC2 sequencing. "
                "Filipin: historically gold standard; still used when oxysterols borderline."
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
