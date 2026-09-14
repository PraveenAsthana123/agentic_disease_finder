"""Hereditary Surfactant Dysfunction / chILD Atlas — 8-Gene Reference
SFTPB-SFTPC-ABCA3-NKX2_1-MARS1-SLC34A2-CSF2RA-CSF2RB
320 patients (8 x 40), seeds 2574-2581.
Endpoints: /api/hereditary-surfactant-dysfunction-atlas/overview|breakdown|definitions
"""
import random

ATLAS_GENES = [
    {
        "gene": "SFTPB",
        "alt_name": "SFTPB-279aa-2p11.2-AR-FATAL-NEONATAL-RDS-c.121ins2-FRAMESHIFT->70pct-NO-LAMELLAR-BODIES-EM-PATHOGNOMONIC-LUNG-TX-ONLY-CURATIVE",
        "protein": "SFTPB -- 2p11.2 AR -- 279aa -- Surfactant-Protein-B-8kDa-Mature-Alveolar-Surfactant-Film-Stability -- OMIM-Gene-178640-Disease-SFTPB-Deficiency-265120",
        "locus": "2p11.2",
        "protein_size": "279 aa / 8 kDa (mature)",
        "inheritance": (
            "AR (autosomal recessive — biallelic LOF); de novo mutations RARE; "
            "c.121ins2 net +2 frameshift (p.Pro41Hisfs) = MOST COMMON >70% of North American SFTPB alleles; "
            "biallelic null = fatal neonatal RDS without lung transplantation; "
            "compound heterozygous with hypomorphic missense = subacute/milder course"
        ),
        "disease_category": (
            "Congenital surfactant deficiency, type I (SFTPB) — fatal neonatal RDS; "
            "SP-B protein absent from BAL and alveolar type II cells; "
            "desaturated phosphatidylcholine (DSPC) critically LOW; "
            "lamellar body absence on electron microscopy = PATHOGNOMONIC of SFTPB null"
        ),
        "disease_pathway": (
            "SP-B is a critical hydrophobic protein enabling surfactant film adsorption and respreading "
            "at the air-liquid interface. Absent SP-B → catastrophic surface tension failure → "
            "diffuse alveolar damage from first breath. SP-C processing also fails secondarily "
            "(SFTPC requires SP-B for post-translational cleavage). "
            "Lamellar bodies (LB) form but are morphologically abnormal — small, dysmorphic, "
            "electron-dense without the characteristic concentric lamellae. "
            "Exogenous surfactant administration is transiently effective but cannot substitute for endogenous SP-B."
        ),
        "pathognomonic": (
            "ABSENT LAMELLAR BODIES ON ELECTRON MICROSCOPY OF LUNG BIOPSY — pathognomonic of biallelic null SFTPB; "
            "SP-B ABSENT ON IMMUNOSTAINING of type II pneumocytes (vs. reduced in hypomorphic); "
            "SP-B ABSENT IN BAL FLUID — diagnostic screening test; "
            "DSPC/total phospholipid ratio critically low; "
            "FATAL NEONATAL RDS in term infant (not premature) — term onset critical DDx clue"
        ),
        "treatment": (
            "LUNG TRANSPLANTATION — only curative option; bilateral sequential preferred; "
            "timing: bridge to transplant with mechanical ventilation/ECMO; "
            "EXOGENOUS SURFACTANT — transiently effective, buys bridge-to-transplant time; "
            "NO medical cure — gene therapy in development; "
            "SP-B gene therapy (lipid nanoparticle) — preclinical phase; "
            "Supportive: CPAP, HFV, NO, ECMO; "
            "Post-transplant: tacrolimus + MMF + steroids; "
            "Sirolimus AVOID post-transplant (ILD risk)"
        ),
        "key_features": [
            "Fatal neonatal RDS in TERM infant — onset day 0-3 of life; premature infant DDx is RDS from prematurity",
            "c.121ins2 frameshift — test FIRST in all neonatal chILD; >70% of North American alleles",
            "Absent lamellar bodies on EM — pathognomonic for biallelic null; small dysmorphic LBs in hypomorphic",
            "SP-B absent in BAL fluid — rapid non-invasive diagnostic test; immunostaining on BAL cell pellet",
            "Exogenous surfactant transiently improves compliance but cannot restore normal function",
            "Lung transplantation — ONLY curative option; evaluate urgently in any surviving neonate",
            "DSPC/phospholipid ratio LOW — alveolar lavage phospholipid analysis supports diagnosis",
            "Hypomorphic compound heterozygous variants — subacute ILD, may survive to childhood with support",
        ],
        "key_ddx": [
            "SFTPC (AD) — milder, later onset, mothers unaffected; SP-B present; BRICHOS domain misfolding",
            "ABCA3 (AR) — small DENSE lamellar bodies on EM (not absent); ABCA3 immunostaining reduced",
            "RDS of prematurity — gestational age <34 weeks; SP-B present; responds to exogenous surfactant durably",
            "NKX2-1 haploinsufficiency — brain + thyroid + lung triad; SP-B usually present; CNS involvement key",
        ],
        "onset_age_years_median": 0,
        "respiratory_failure_pct": 100,
        "lung_tx_pct": 55,
        "surfactant_response_pct": 30,
        "ild_pct": 100,
        "pap_pct": 5,
        "microlithiasis_pct": 0,
        "nbs_indicated": False,
    },
    {
        "gene": "SFTPC",
        "alt_name": "SFTPC-197aa-8p21.3-AD-DE-NOVO-ILD-CHILDHOOD-ADULT-pIle73Thr-BRICHOS-ER-STRESS-UPR-HCQ-FIRST-LINE-NINTEDANIB-ADULTS",
        "protein": "SFTPC -- 8p21.3 AD -- 197aa -- Surfactant-Protein-C-4kDa-Mature-Single-Span-TM-Palmitoylated-BRICHOS-Propeptide -- OMIM-Gene-178620-Disease-SFTPC-Dysfunction-610913",
        "locus": "8p21.3",
        "protein_size": "197 aa / 4 kDa (mature)",
        "inheritance": (
            "AD (autosomal dominant — dominant-negative misfolding or haploinsufficiency); "
            "DE NOVO mutations common (~40%); family history variable due to de novo rate; "
            "p.Ile73Thr = most common pathogenic variant — BRICHOS domain misfolding, ER stress, UPR activation; "
            "gain-of-toxic-function via protein misfolding + ER accumulation; "
            "full penetrance but highly variable expressivity (infant severe ILD to adult UIP)"
        ),
        "disease_category": (
            "SFTPC dysfunction interstitial lung disease — childhood ILD (chILD) to adult IPF-like; "
            "misfolded SP-C precursor accumulates in ER → UPR → type II pneumocyte injury; "
            "BAL: increased neutrophils + lymphocytes + foam cells; "
            "CT: ground glass (children), honeycombing (adults), subpleural reticulation; "
            "Biopsy: NSIP, DIP, UIP depending on variant and age"
        ),
        "disease_pathway": (
            "SP-C is a hydrophobic protein that stabilises the surfactant film at the air-liquid interface. "
            "SP-C propeptide contains a BRICHOS chaperone domain that directs correct folding. "
            "Pathogenic BRICHOS domain mutations (especially p.Ile73Thr) cause misfolding → "
            "ER retention → UPR activation → type II pneumocyte injury and death → ILD. "
            "The dominant-negative effect means one bad allele is sufficient for disease. "
            "N-terminal domain mutations cause different pathology (SP-C mistrafficking to lysosomes)."
        ),
        "pathognomonic": (
            "ILD IN INFANT WITH AFFECTED PARENT OR DE NOVO AD MUTATION — key clinical clue; "
            "p.Ile73Thr BRICHOS domain — most common; test early in all paediatric ILD; "
            "ER STRESS MARKER ELEVATION in lung biopsy (GRP78/BiP immunostaining) — research tool; "
            "GROUND GLASS + SUBPLEURAL RETICULATION on CT in children; HONEYCOMBING + UIP in adults; "
            "TYPE II PNEUMOCYTE HYPERPLASIA on biopsy with DIP/NSIP pattern"
        ),
        "treatment": (
            "HYDROXYCHLOROQUINE (HCQ) 5-10 mg/kg/day — first-line in children; reduces ER stress; "
            "SYSTEMIC CORTICOSTEROIDS — moderate benefit in acute exacerbations; taper slowly; "
            "AZATHIOPRINE + MMF — steroid-sparing in children and adults; "
            "NINTEDANIB (150 mg BD) — adults with progressive ILD/UIP pattern; "
            "PIRFENIDONE — adults with fibrosis; "
            "LUNG TRANSPLANT — for progressive refractory disease; "
            "AVOID tobacco, secondhand smoke — accelerates fibrosis; "
            "PULMONARY REHAB — essential for adults"
        ),
        "key_features": [
            "De novo mutations common (~40%) — family history may be ABSENT; test in all neonatal/infant ILD",
            "p.Ile73Thr BRICHOS domain — most common variant; causes ER stress and type II cell injury",
            "Variable expressivity — same variant: severe infant RDS in one family member, mild adult ILD in another",
            "Hydroxychloroquine — established first-line in children; reduces ER stress pathway activation",
            "CT: ground glass with subpleural reticulation in children; honeycombing/UIP pattern in adults",
            "Biopsy: DIP/NSIP in children; UIP in older adults — variant and age determine histological pattern",
            "Nintedanib/pirfenidone — adults with progressive fibrosis; same agents as sporadic IPF",
            "Family screening mandatory — AD inheritance with variable expressivity, some undiagnosed",
        ],
        "key_ddx": [
            "SFTPB (AR) — biallelic, no family history typically; SP-B absent; lamellar bodies absent on EM",
            "ABCA3 (AR) — biallelic; ABCA3 protein low; small dense LBs on EM; SFTPC protein present",
            "Hypersensitivity pneumonitis — antigen exposure history; lymphocytic BAL; resolves with avoidance",
            "NKX2-1 (AD) — brain-thyroid-lung triad; hypothyroidism + chorea; NKX2-1 IHC absent in lung",
        ],
        "onset_age_years_median": 5,
        "respiratory_failure_pct": 35,
        "lung_tx_pct": 20,
        "surfactant_response_pct": 15,
        "ild_pct": 100,
        "pap_pct": 5,
        "microlithiasis_pct": 0,
        "nbs_indicated": False,
    },
    {
        "gene": "ABCA3",
        "alt_name": "ABCA3-1704aa-16p13.3-AR-MOST-COMMON-GENETIC-chILD-SMALL-DENSE-LAMELLAR-BODIES-EM-PATHOGNOMONIC-HCQ-LUNG-TX",
        "protein": "ABCA3 -- 16p13.3 AR -- 1704aa -- ATP-Binding-Cassette-Transporter-A3-191kDa-Type-II-Pneumocyte-Lamellar-Body-Limiting-Membrane -- OMIM-Gene-601615-Disease-SFTPB-Deficiency-265120",
        "locus": "16p13.3",
        "protein_size": "1704 aa / 191 kDa",
        "inheritance": (
            "AR (autosomal recessive — biallelic LOF or compound heterozygous); "
            "MOST COMMON genetic cause of neonatal/childhood chILD; "
            "biallelic null = neonatal fatal (similar severity to SFTPB null); "
            "one null + one missense = variable childhood ILD; "
            "two missense = milder childhood to adult ILD; "
            "genotype-phenotype correlation: null allele number predicts severity"
        ),
        "disease_category": (
            "ABCA3 deficiency — neonatal RDS to chronic ILD; "
            "ABCA3 transports phospholipids (DPPC, PG) into lamellar bodies; "
            "dysfunction → abnormal lamellar body morphology → surfactant composition defect → ILD; "
            "SMALL DENSE ELECTRON-OPAQUE LAMELLAR BODIES on EM = PATHOGNOMONIC; "
            "ABCA3 protein low or absent on immunostaining of type II pneumocytes"
        ),
        "disease_pathway": (
            "ABCA3 is an ABC transporter located on the limiting membrane of lamellar bodies in alveolar type II cells. "
            "It actively transports phosphatidylcholine (DPPC) and phosphatidylglycerol (PG) into lamellar bodies, "
            "forming the lipid core of surfactant. Deficient ABCA3 → inadequate phospholipid loading → "
            "morphologically abnormal small dense LBs (characteristic EM finding) → surfactant composition defect → "
            "ILD. Unlike SFTPB, lamellar bodies ARE present but abnormally electron-dense and small. "
            "The degree of ABCA3 dysfunction correlates with disease severity."
        ),
        "pathognomonic": (
            "SMALL DENSE ELECTRON-OPAQUE LAMELLAR BODIES ON EM — pathognomonic for ABCA3 deficiency; "
            "lamellar bodies present but small, round, electron-dense without normal lamellae; "
            "ABCA3 PROTEIN LOW/ABSENT ON IMMUNOSTAINING — type II pneumocyte IHC; "
            "CT: ground glass opacity in neonate; mosaic attenuation + air trapping in older children; "
            "BAL: foamy alveolar macrophages + increased lipid"
        ),
        "treatment": (
            "HYDROXYCHLOROQUINE 5-10 mg/kg/day — first-line; reduces disease progression; "
            "SYSTEMIC CORTICOSTEROIDS — for acute exacerbations; "
            "AZATHIOPRINE/MMF — steroid-sparing agents; "
            "LUNG TRANSPLANTATION — for neonatal severe and progressive childhood disease; "
            "SUPPORTIVE: oxygen supplementation, CPAP/BiPAP for chronic hypoxaemia; "
            "NINTEDANIB — adults with progressive fibrosis; limited paediatric data; "
            "NUTRITIONAL SUPPORT — growth failure common; high-calorie feeds"
        ),
        "key_features": [
            "Most common genetic cause of neonatal/childhood chILD — sequence ABCA3 first in unexplained neonatal ILD",
            "Small dense electron-opaque lamellar bodies on EM — pathognomonic; distinguishes from SFTPB (absent LBs)",
            "Genotype predicts severity: two null alleles = neonatal fatal; null+missense = childhood ILD; two missense = milder",
            "p.Leu101Pro + p.Glu292Val = common European pathogenic variants (founder effects)",
            "ABCA3 immunostaining LOW/ABSENT — rapid tissue confirmation; Western blot on BAL cell pellet",
            "CT ground glass in neonate → mosaic attenuation + air trapping in chronic childhood disease",
            "Hydroxychloroquine — established first-line; reduces hospitalisation frequency",
            "Lung transplant — bilateral sequential preferred; post-transplant disease does not recur",
        ],
        "key_ddx": [
            "SFTPB (AR) — ABSENT lamellar bodies on EM (vs. SMALL DENSE in ABCA3); SP-B absent vs. present",
            "SFTPC (AD) — de novo/family history; ER stress; lamellar bodies near-normal; SFTPC protein absent",
            "NKX2-1 (AD) — brain-thyroid-lung triad; ABCA3 protein present; NKX2-1 absent on IHC",
            "RDS of prematurity — gestational age, exogenous surfactant durable response, ABCA3 protein normal",
        ],
        "onset_age_years_median": 0,
        "respiratory_failure_pct": 75,
        "lung_tx_pct": 45,
        "surfactant_response_pct": 25,
        "ild_pct": 100,
        "pap_pct": 10,
        "microlithiasis_pct": 0,
        "nbs_indicated": False,
    },
    {
        "gene": "NKX2-1",
        "alt_name": "NKX2-1-371aa-14q13.3-AD-BRAIN-THYROID-LUNG-TRIAD-PATHOGNOMONIC-HAPLOINSUFFICIENCY-CHOREA-HYPOTHYROIDISM-ILD-TTF-1",
        "protein": "NKX2-1 -- 14q13.3 AD -- 371aa -- NK2-Homeobox-1-TTF-1-Thyroid-Transcription-Factor-1-42kDa-Homeobox-NK2-Domain -- OMIM-Gene-600635-Disease-Brain-Thyroid-Lung-Syndrome-610978",
        "locus": "14q13.3",
        "protein_size": "371 aa / 42 kDa",
        "inheritance": (
            "AD (autosomal dominant — haploinsufficiency LOF); "
            "DE NOVO mutations in ~60%; familial cases follow AD inheritance; "
            "deletions of 14q13.3 including NKX2-1 = Brain-Thyroid-Lung syndrome; "
            "point mutations, frameshift, missense in homeobox or NK2 domain; "
            "full penetrance but extremely variable expressivity across three organ systems"
        ),
        "disease_category": (
            "Brain-Thyroid-Lung syndrome — NKX2-1 haploinsufficiency triad; "
            "NKX2-1 is master transcription factor for lung surfactant genes (SFTPB, SFTPC, ABCA3), "
            "thyroid differentiation (TG, TPO, SLC5A5/NIS), and brain (striatal interneurons); "
            "50% have all 3 features; 30% have 2 features; 20% have 1 feature; "
            "pulmonary component = worst prognostic factor"
        ),
        "disease_pathway": (
            "NKX2-1 (also called TTF-1) is a homeodomain transcription factor essential for: "
            "(1) Lung: transcribes SFTPB, SFTPC, ABCA3 → surfactant deficiency when haploinsufficient; "
            "(2) Thyroid: transcribes TG, TPO, SLC5A5 → congenital hypothyroidism or thyroid agenesis/ectopy; "
            "(3) Brain: development of striatal interneurons, basal ganglia → choreoathetosis. "
            "Haploinsufficiency means 50% of normal NKX2-1 is insufficient for full organogenesis. "
            "The lung defect secondary to surfactant gene transcription failure explains ILD."
        ),
        "pathognomonic": (
            "BRAIN-THYROID-LUNG TRIAD — chorea + hypothyroidism + ILD in same patient = PATHOGNOMONIC; "
            "NKX2-1/TTF-1 ABSENT ON LUNG IHC — definitive tissue diagnosis; "
            "CHOREOATHETOSIS IN INFANCY/CHILDHOOD — hyperkinetic movement disorder; "
            "CONGENITAL HYPOTHYROIDISM or thyroid dysgenesis on thyroid scan; "
            "SFTPB/SFTPC/ABCA3 SECONDARILY REDUCED on lung IHC — downstream transcription failure"
        ),
        "treatment": (
            "THYROID HORMONE REPLACEMENT — levothyroxine, titrate to TSH/fT4; mandatory for hypothyroidism; "
            "CHOREA MANAGEMENT — tetrabenazine, trihexyphenidyl, clonazepam; "
            "PULMONARY — hydroxychloroquine (limited evidence); systemic steroids for exacerbations; "
            "OXYGEN supplementation for chronic hypoxaemia; "
            "LUNG TRANSPLANT — last resort for severe progressive ILD; "
            "MULTIDISCIPLINARY — paeds neurology + endocrinology + pulmonology essential; "
            "PHYSIOTHERAPY — hypotonia and motor delay common"
        ),
        "key_features": [
            "Brain-Thyroid-Lung triad — only 50% have all 3; check ALL 3 systems in any suspected case",
            "Chorea (basal ganglia) — often the presenting feature; brain MRI may show caudate/putamen signal",
            "Congenital hypothyroidism — thyroid dysgenesis/ectopy; absent goitre unlike Pendred/dyshormonogenesis",
            "ILD — worst prognostic feature; CT ground glass + reticular; secondary surfactant deficiency",
            "NKX2-1 IHC absent on lung biopsy — diagnostic; SFTPB/SFTPC also low (downstream effects)",
            "De novo mutations 60% — family history often absent; genetic testing essential in infant ILD + chorea",
            "14q13 deletion — chromosomal microarray can detect large deletions encompassing NKX2-1",
            "Thyroid hormone replacement mandatory — regardless of pulmonary or neurological status",
        ],
        "key_ddx": [
            "Benign hereditary chorea (BHC) — isolated chorea without lung/thyroid; some BHC is NKX2-1",
            "SFTPB/ABCA3 — no chorea or thyroid disease; no NKX2-1 IHC reduction",
            "SFTPC — de novo AD; chorea and hypothyroidism absent; no NKX2-1 IHC loss",
            "DYT/CHRM — isolated chorea without lung; check thyroid to exclude NKX2-1",
        ],
        "onset_age_years_median": 1,
        "respiratory_failure_pct": 45,
        "lung_tx_pct": 15,
        "surfactant_response_pct": 10,
        "ild_pct": 80,
        "pap_pct": 15,
        "microlithiasis_pct": 0,
        "nbs_indicated": True,  # Congenital hypothyroidism NBS catches some cases
    },
    {
        "gene": "MARS1",
        "alt_name": "MARS1-900aa-12q13.3-AR-INTERSTITIAL-LUNG-AND-LIVER-DISEASE-ILLD-COMBINED-PULMONARY-HEPATIC-UNIQUE-DDx-pArg792His-MOST-COMMON",
        "protein": "MARS1 -- 12q13.3 AR -- 900aa -- Methionyl-tRNA-Synthetase-1-101kDa-Cytoplasmic-Mitochondrial-Met-tRNA-Aminoacylation -- OMIM-Gene-156560-Disease-ILLD-616241",
        "locus": "12q13.3",
        "protein_size": "900 aa / 101 kDa",
        "inheritance": (
            "AR (autosomal recessive — biallelic LOF or hypomorphic compound heterozygous); "
            "p.Arg792His = most common pathogenic variant (~60% of reported alleles); "
            "aminoacyl-tRNA synthetase defect → translation fidelity failure → mitochondrial dysfunction; "
            "combined lung + liver disease = hallmark of MARS1 (UNIQUE among chILD genes); "
            "founder variants in Jewish, Iranian, Turkish populations"
        ),
        "disease_category": (
            "Interstitial Lung and Liver Disease (ILLD) — combined pulmonary + hepatic; "
            "the combined lung + liver phenotype is UNIQUE among hereditary chILD genes; "
            "lung: ground glass on CT, alveolar opacification, diffuse ILD; "
            "liver: hepatomegaly, elevated transaminases, progressive fibrosis → cirrhosis; "
            "MARS1 catalyses methionyl-tRNA formation → deficiency impairs protein synthesis in all cells"
        ),
        "disease_pathway": (
            "MARS1 (formerly MARS) charges tRNA-Met with methionine — the first amino acid of all proteins. "
            "Reduced aminoacylation efficiency in hypomorphic variants → "
            "translational stress in high-demand tissues (lung type II cells, hepatocytes) → "
            "ER stress + mitochondrial dysfunction → tissue injury. "
            "Mitochondrial protein synthesis also depends on mitochondrial MARS1 activity. "
            "Both cytoplasmic and mitochondrial protein synthesis impaired → combined organ failure "
            "explains why lung AND liver are both affected (unlike pure surfactant gene defects)."
        ),
        "pathognomonic": (
            "COMBINED ILD + HEPATOMEGALY/ELEVATED LFTs IN INFANCY — unique among chILD; think MARS1; "
            "p.Arg792His compound with second missense/null — sequence MARS1 in all combined lung+liver ILD; "
            "LIVER BIOPSY: steatohepatitis + portal fibrosis; no specific histological marker; "
            "CT: bilateral ground glass + consolidation in infant/child + hepatomegaly; "
            "NO lamellar body abnormality on EM (distinguishes from ABCA3/SFTPB)"
        ),
        "treatment": (
            "N-ACETYLCYSTEINE (NAC) — antioxidant; may reduce ER/oxidative stress; limited evidence; "
            "SYSTEMIC CORTICOSTEROIDS — for ILD acute exacerbations; "
            "URSODEOXYCHOLIC ACID (UDCA) — hepatoprotective; for liver disease; "
            "LIVER TRANSPLANT — for end-stage hepatic cirrhosis; lung disease may improve post-liver Tx; "
            "LUNG TRANSPLANT — last resort for refractory ILD; "
            "NUTRITIONAL SUPPORT — fat-soluble vitamins ADEK; "
            "HYDROXYCHLOROQUINE — tried empirically; variable response; "
            "AVOID hepatotoxic medications"
        ),
        "key_features": [
            "Combined ILD + liver disease — UNIQUE among chILD genes; always check liver if infant has ILD",
            "p.Arg792His — most common pathogenic variant; test early in combined lung+liver phenotype",
            "Hepatomegaly + elevated transaminases — may precede pulmonary manifestations in infancy",
            "CT ground glass + consolidation + hepatomegaly — distinguishing combination from pure surfactant defects",
            "Liver fibrosis progressive — cirrhosis by adolescence in severe cases; monitor with liver biopsy/Fibroscan",
            "N-acetylcysteine — empirical antioxidant therapy; some case reports suggest benefit",
            "Liver transplant — curative for hepatic disease; may stabilise lung disease in combined phenotype",
            "No lamellar body abnormality on EM — key negative finding distinguishing from ABCA3/SFTPB",
        ],
        "key_ddx": [
            "ABCA3 — lung only typically; no liver disease; small dense LBs on EM; ABCA3 protein absent",
            "Alagille syndrome (JAG1/NOTCH2) — cholestasis + cardiac + butterfly vertebrae; no ILD",
            "Wilson disease (ATP7B) — Kayser-Fleischer rings; onset older child; urinary Cu elevated",
            "Alpha-1-antitrypsin deficiency — PiZZ; liver + lung; serum A1AT low; liver biopsy PAS+ granules",
        ],
        "onset_age_years_median": 1,
        "respiratory_failure_pct": 50,
        "lung_tx_pct": 20,
        "surfactant_response_pct": 5,
        "ild_pct": 100,
        "pap_pct": 5,
        "microlithiasis_pct": 0,
        "nbs_indicated": False,
    },
    {
        "gene": "SLC34A2",
        "alt_name": "SLC34A2-689aa-4p15.31-AR-PULMONARY-ALVEOLAR-MICROLITHIASIS-PAM-SNOWSTORM-CXR-PATHOGNOMONIC-CALCIUM-PHOSPHATE-DEPOSITS-NO-EFFECTIVE-TREATMENT",
        "protein": "SLC34A2 -- 4p15.31 AR -- 689aa -- Sodium-Phosphate-Cotransporter-IIb-NaPi-IIb-74kDa-Intestinal-Alveolar-Type-II-Phosphate-Transport -- OMIM-Gene-604217-Disease-PAM-265100",
        "locus": "4p15.31",
        "protein_size": "689 aa / 74 kDa",
        "inheritance": (
            "AR (autosomal recessive — biallelic LOF); "
            "PULMONARY ALVEOLAR MICROLITHIASIS (PAM) — distinct from PAP; "
            "SLC34A2 failure → phosphate accumulates in alveolar lumen → calcium phosphate microliths form; "
            "Turkish, Japanese, Italian founder variants; "
            "c.479+1G>A splice variant = most common Turkish allele; "
            "very slow progression over decades — asymptomatic in childhood, symptomatic in adulthood"
        ),
        "disease_category": (
            "Pulmonary Alveolar Microlithiasis (PAM) — calcium phosphate microcrystal deposition in alveoli; "
            "SLC34A2 = NaPi-IIb cotransporter; normally clears phosphate from alveolar space; "
            "LOF → phosphate accumulation → hydroxyapatite (calcium phosphate) microliths fill alveoli; "
            "SNOWSTORM APPEARANCE on CXR/CT = pathognomonic; "
            "extremely slow progression — patients may live >40 years before needing transplant"
        ),
        "disease_pathway": (
            "SLC34A2 encodes NaPi-IIb, a sodium-phosphate co-transporter expressed on the apical surface "
            "of alveolar type II cells. Normally, NaPi-IIb actively reabsorbs inorganic phosphate from "
            "alveolar fluid back into the cell. Biallelic LOF → phosphate accumulates in alveolar lumen → "
            "reacts with calcium from surfactant metabolism → calcium phosphate (hydroxyapatite) microliths "
            "precipitate and fill the alveoli over years-decades. "
            "The microliths are pathognomonic: lamellar calcium phosphate deposits 0.5-3 mm on CT. "
            "SLC34A2 is also expressed in intestine (phosphate absorption) but gut phenotype is mild/absent."
        ),
        "pathognomonic": (
            "SNOWSTORM APPEARANCE ON CXR — bilateral dense micronodular opacities giving 'sandstorm' appearance = PATHOGNOMONIC; "
            "CT: bilateral ground glass + innumerable micronodules 1-3mm, basal predominant, calcified; "
            "BAL: calcium phosphate microliths (lamellar calcospherites); "
            "LUNG BIOPSY: alveoli filled with concentric lamellar calcite microliths — pathognomonic; "
            "SERUM CALCIUM/PHOSPHATE NORMAL — systemic mineral metabolism unaffected"
        ),
        "treatment": (
            "NO EFFECTIVE MEDICAL TREATMENT — bisphosphonates INEFFECTIVE; "
            "LUNG TRANSPLANTATION — only option for end-stage disease; bilateral sequential preferred; "
            "SUPPORTIVE: oxygen supplementation for chronic hypoxaemia; pulmonary rehab; "
            "MONITORING: PFTs annually; CT every 2-3 years; "
            "AVOID: dust, smoking — accelerate progression; "
            "GENETIC COUNSELLING — family screening; siblings may be asymptomatic; "
            "CLINICAL TRIALS — inhaled etidronate (experimental); gene therapy not yet available"
        ),
        "key_features": [
            "Snowstorm appearance on CXR — bilateral sand-like calcific micronodular opacities; pathognomonic",
            "Extremely slow progression — decades before symptomatic; CXR classic but patient may feel well",
            "Serum calcium/phosphate NORMAL — systemic mineral metabolism intact; distinguishes from hyperparathyroidism",
            "BAL: lamellar calcospherites (calcium phosphate microliths) — diagnostic on BAL",
            "Turkish/Japanese/Italian founder variants — sequence SLC34A2 in ethnic background + typical CXR",
            "Bisphosphonates INEFFECTIVE — microliths are not osteoclast-mediated resorption process",
            "Lung transplant — only definitive option; disease does not recur in transplanted lung",
            "Family history — siblings should have CXR/CT; asymptomatic affected relatives common (slow disease)",
        ],
        "key_ddx": [
            "Sarcoidosis — lymphadenopathy, uveoparotitis; CXR bilateral hilar lymphadenopathy; BAL lymphocytosis",
            "Miliary TB — fever, night sweats; Mantoux/IGRA positive; BAL cultures; responds to antibiotics",
            "Pulmonary alveolar proteinosis (PAP) — PAS+ lipoproteinaceous material; no calcification; responds to WLL",
            "Hypercalcaemia — metastatic calcification in hypercalcaemic states; serum Ca elevated; systemic features",
        ],
        "onset_age_years_median": 30,
        "respiratory_failure_pct": 25,
        "lung_tx_pct": 15,
        "surfactant_response_pct": 0,
        "ild_pct": 100,
        "pap_pct": 0,
        "microlithiasis_pct": 100,
        "nbs_indicated": False,
    },
    {
        "gene": "CSF2RA",
        "alt_name": "CSF2RA-400aa-Xp22.33-PAR1-HEREDITARY-PAP-GM-CSF-RECEPTOR-ALPHA-WHOLE-LUNG-LAVAGE-INHALED-GM-CSF-EFFECTIVE-NO-ANTI-GMCSF-Ab",
        "protein": "CSF2RA -- Xp22.33-Yp11.2-PAR1 PAR/XL -- 400aa -- GM-CSF-Receptor-Alpha-Chain-CD116-45kDa-Cytokine-Binding-Alpha-Subunit -- OMIM-Gene-306250-Disease-PAP-Hereditary-300770",
        "locus": "Xp22.33 (PAR1)",
        "protein_size": "400 aa / 45 kDa",
        "inheritance": (
            "XL/PAR (X-linked but pseudoautosomal region 1 — BOTH sexes equally affected at PAR1); "
            "biallelic LOF in XX females; hemizygous in XY males = same phenotype; "
            "HEREDITARY PAP from GM-CSF signalling failure; "
            "anti-GM-CSF antibodies ABSENT — distinguishes from autoimmune PAP (the most common form); "
            "onset childhood to young adult"
        ),
        "disease_category": (
            "Hereditary Pulmonary Alveolar Proteinosis (PAP) type 4 — GM-CSF receptor alpha deficiency; "
            "GM-CSF drives alveolar macrophage maturation and surfactant clearance; "
            "absent GM-CSF signalling → alveolar macrophage immaturity → surfactant lipoprotein accumulation; "
            "PAS-POSITIVE BAL FLUID = pathognomonic of PAP (all types); "
            "WHOLE LUNG LAVAGE (WLL) = therapeutic and can be curative for episodes; "
            "INHALED GM-CSF (sargramostim) = effective — bypasses receptor α deficiency? No: "
            "CSF2RA LOF means inhaled GM-CSF still cannot signal → INEFFECTIVE — correct answer: "
            "see key_features for nuance"
        ),
        "disease_pathway": (
            "GM-CSF (CSF2) binds the alpha chain (CSF2RA/CD116) which then recruits the beta chain (CSF2RB/CD131) "
            "to form the active signalling complex. GM-CSF drives alveolar macrophage (AM) maturation, "
            "PU.1 expression, and surfactant catabolism. Absent CSF2RA → no GM-CSF signalling → "
            "AMs remain immature (macrophage activation syndrome) → surfactant lipoprotein (SP-A, SP-D, "
            "phospholipids) accumulates → PAP. Whole lung lavage (WLL) physically removes accumulated "
            "material. Inhaled recombinant GM-CSF may partially bypass signalling defect in heterozygous/partial "
            "deficiency — less effective in biallelic null."
        ),
        "pathognomonic": (
            "PAS-POSITIVE MILKY BAL FLUID — lipoproteinaceous material, pathognomonic of PAP (all types); "
            "ANTI-GM-CSF ANTIBODIES ABSENT — critical DDx from autoimmune PAP (anti-GM-CSF Ab positive); "
            "GM-CSF SIGNALLING TEST: STAT5 phosphorylation ABSENT in monocytes after GM-CSF stimulation; "
            "CT: CRAZY PAVING PATTERN — ground glass + septal thickening = PAP hallmark; "
            "SP-A/SP-D ELEVATED IN SERUM — surrogate of alveolar surfactant accumulation"
        ),
        "treatment": (
            "WHOLE LUNG LAVAGE (WLL) — 1-2 L saline instilled then lavaged; bilateral sequential; "
            "definitive treatment for acute exacerbations; may require repeat every 1-5 years; "
            "INHALED RECOMBINANT GM-CSF (sargramostim 250 mcg/day) — variable efficacy in CSF2RA; "
            "partially effective in hypomorphic/partial deficiency; less so in biallelic null; "
            "HAEMATOPOIETIC STEM CELL TRANSPLANT (HSCT) — curative in severe/refractory cases; "
            "corrects macrophage progenitor defect; "
            "LUNG TRANSPLANT — for end-stage; high recurrence risk without HSCT; "
            "AVOID: smoking, dust, respiratory infections (accelerate PAP episodes)"
        ),
        "key_features": [
            "Hereditary PAP — both sexes equally affected (PAR1 locus on both X and Y chromosomes)",
            "Anti-GM-CSF antibodies ABSENT — distinguishes from autoimmune PAP (most common type, Ab-positive)",
            "PAS-positive milky BAL fluid — pathognomonic of PAP; confirm with CT crazy-paving pattern",
            "STAT5 phosphorylation absent in monocytes — functional test for CSF2R deficiency",
            "Whole lung lavage (WLL) — effective treatment; repeat as needed; bilateral sequential preferred",
            "Inhaled GM-CSF — variable efficacy; less effective in biallelic null CSF2RA vs. autoimmune PAP",
            "HSCT — curative option in severe cases; corrects macrophage progenitor defect permanently",
            "SP-A/SP-D elevated in serum — surrogate marker of alveolar accumulation; monitor disease activity",
        ],
        "key_ddx": [
            "Autoimmune PAP — anti-GM-CSF antibodies POSITIVE; adults primarily; no family history typically",
            "CSF2RB (AR) — identical PAP phenotype; biallelic CSF2RB; GM-CSF INEFFECTIVE (beta-chain missing)",
            "Nocardia/Cryptococcus infection — secondary PAP in immunocompromised; cultures positive",
            "SFTPB null — neonatal; lamellar bodies absent on EM; no PAS-positive milky fluid",
        ],
        "onset_age_years_median": 8,
        "respiratory_failure_pct": 40,
        "lung_tx_pct": 10,
        "surfactant_response_pct": 0,
        "ild_pct": 100,
        "pap_pct": 100,
        "microlithiasis_pct": 0,
        "nbs_indicated": False,
    },
    {
        "gene": "CSF2RB",
        "alt_name": "CSF2RB-897aa-22q12.3-AR-HEREDITARY-PAP-GM-CSF-RECEPTOR-BETA-BIALLELIC-INHALED-GM-CSF-INEFFECTIVE-HSCT-CURATIVE-STAT5-ABSENT",
        "protein": "CSF2RB -- 22q12.3 AR -- 897aa -- GM-CSF-Receptor-Beta-Chain-CD131-IL-3-IL-5-Shared-Beta-c-Chain-97kDa-Signalling-Subunit -- OMIM-Gene-138981-Disease-PAP-Hereditary-614370",
        "locus": "22q12.3",
        "protein_size": "897 aa / 97 kDa",
        "inheritance": (
            "AR (autosomal recessive — biallelic LOF); "
            "HEREDITARY PAP type 5 — GM-CSF receptor BETA chain deficiency; "
            "beta chain is SHARED receptor for GM-CSF, IL-3, AND IL-5; "
            "INHALED GM-CSF INEFFECTIVE — beta chain missing means GM-CSF cannot signal even with alpha chain present; "
            "anti-GM-CSF antibodies ABSENT; "
            "broader immune signalling defect than CSF2RA (IL-3, IL-5 also blocked)"
        ),
        "disease_category": (
            "Hereditary PAP type 5 — biallelic CSF2RB LOF; identical macroscopic PAP phenotype to CSF2RA; "
            "INHALED GM-CSF IS INEFFECTIVE — critical treatment difference from autoimmune PAP and CSF2RA partial; "
            "HSCT = curative; "
            "CSF2RB also shared by IL-3 and IL-5 receptors → broader immunological defect; "
            "eosinophil count LOW (IL-5 signals through CSF2RB — eosinophil differentiation impaired)"
        ),
        "disease_pathway": (
            "CSF2RB (CD131) is the common beta chain shared by GM-CSF receptor (CSF2RA+CSF2RB), "
            "IL-3 receptor (IL3RA+CSF2RB), and IL-5 receptor (IL5RA+CSF2RB). "
            "The beta chain is the signalling subunit — all three cytokine pathways converge on JAK2/STAT5. "
            "Biallelic CSF2RB LOF → (1) GM-CSF signalling absent → PAP (same as CSF2RA); "
            "(2) IL-5 signalling absent → eosinophil differentiation/survival impaired → LOW eosinophil count; "
            "(3) IL-3 signalling absent → myeloid progenitor effect. "
            "Inhaled GM-CSF cannot work WITHOUT the beta chain — distinguishes treatment approach from autoimmune PAP."
        ),
        "pathognomonic": (
            "PAS-POSITIVE MILKY BAL FLUID + ANTI-GM-CSF Ab ABSENT — hereditary PAP; "
            "INHALED GM-CSF INEFFECTIVE — treatment failure distinguishes CSF2RB from autoimmune PAP + partial CSF2RA; "
            "STAT5 PHOSPHORYLATION ABSENT after GM-CSF AND IL-3 stimulation (vs. CSF2RA: absent only with GM-CSF); "
            "LOW PERIPHERAL EOSINOPHIL COUNT — IL-5 signalling also blocked; "
            "CT CRAZY PAVING PATTERN — bilateral GGO + septal thickening = PAP pattern"
        ),
        "treatment": (
            "WHOLE LUNG LAVAGE (WLL) — same as PAP management; saline lavage; bilateral sequential; "
            "INHALED GM-CSF IS INEFFECTIVE — critical distinction; do NOT treat as autoimmune PAP; "
            "HAEMATOPOIETIC STEM CELL TRANSPLANT (HSCT) — CURATIVE; corrects beta chain defect in macrophage progenitors; "
            "preferred over repeated WLL in children with severe disease; "
            "CONDITIONING: reduced-intensity preferred to avoid additive lung toxicity; "
            "LUNG TRANSPLANT — last resort; may recur if recipient macrophages not replaced; "
            "AVOID: infections, smoke; pneumocystis prophylaxis post-HSCT"
        ),
        "key_features": [
            "Hereditary PAP type 5 — biallelic CSF2RB; both sexes equally affected; AR inheritance",
            "Inhaled GM-CSF INEFFECTIVE — beta chain absent means ligand cannot signal; do not treat as autoimmune PAP",
            "STAT5 phosphorylation absent after GM-CSF AND IL-3 stimulation — functional test for CSF2RB (vs. CSF2RA: only GM-CSF absent)",
            "Low eosinophil count — IL-5 signalling via CSF2RB also blocked; clue to CSF2RB vs. CSF2RA",
            "HSCT is curative — corrects macrophage progenitor defect; preferred in childhood severe disease",
            "Anti-GM-CSF antibodies ABSENT — critical DDx from autoimmune PAP (most common PAP, antibody positive)",
            "WLL effective acutely — but not curative for underlying defect; bridge to HSCT",
            "Broader immune signalling defect than CSF2RA — IL-3, IL-5, GM-CSF all lost",
        ],
        "key_ddx": [
            "CSF2RA — X-linked/PAR; STAT5 absent only with GM-CSF not IL-3; inhaled GM-CSF may have partial benefit",
            "Autoimmune PAP — anti-GM-CSF Ab POSITIVE; adults; inhaled GM-CSF EFFECTIVE; no family history",
            "Secondary PAP (lysinuric protein intolerance, haem malignancy) — look for underlying cause",
            "SFTPB null — neonatal; lamellar bodies absent on EM; SP-B absent; no PAS-positive milky fluid",
        ],
        "onset_age_years_median": 5,
        "respiratory_failure_pct": 45,
        "lung_tx_pct": 8,
        "surfactant_response_pct": 0,
        "ild_pct": 100,
        "pap_pct": 100,
        "microlithiasis_pct": 0,
        "nbs_indicated": False,
    },
]

SEEDS = list(range(2574, 2582))  # 2574-2581, 8 seeds for 8 genes


def _rng(seed):
    rng = random.Random(seed)
    return rng


def _simulate_cohort(gene_data, seed):
    rng = _rng(seed)
    n = 40
    patients = []
    for i in range(n):
        onset_months = max(0, int(rng.gauss(gene_data["onset_age_years_median"] * 12, 18)))
        dx_delay_months = max(1, int(rng.gauss(8, 5)))
        had_rds = rng.random() < (gene_data["respiratory_failure_pct"] / 100)
        received_lung_tx = rng.random() < (gene_data["lung_tx_pct"] / 100)
        surfactant_given = rng.random() < 0.6
        surfactant_response = rng.random() < (gene_data["surfactant_response_pct"] / 100) if surfactant_given else False
        has_ild = rng.random() < (gene_data["ild_pct"] / 100)
        has_pap = rng.random() < (gene_data["pap_pct"] / 100)
        has_microlithiasis = rng.random() < (gene_data["microlithiasis_pct"] / 100)
        spo2_baseline = max(78, min(99, int(rng.gauss(93 - gene_data["respiratory_failure_pct"] / 20, 5))))
        fev1_pct = max(30, min(105, int(rng.gauss(82 - gene_data["respiratory_failure_pct"] / 5, 15))))
        dlco_pct = max(20, min(100, int(rng.gauss(68 - gene_data["ild_pct"] / 8, 12))))
        wll_lavages = rng.randint(0, 4) if has_pap else 0
        alive = not (had_rds and not received_lung_tx and gene_data["respiratory_failure_pct"] > 90)
        patients.append({
            "onset_months": onset_months,
            "dx_delay_months": dx_delay_months,
            "had_rds": had_rds,
            "received_lung_tx": received_lung_tx,
            "surfactant_given": surfactant_given,
            "surfactant_response": surfactant_response,
            "has_ild": has_ild,
            "has_pap": has_pap,
            "has_microlithiasis": has_microlithiasis,
            "spo2_baseline": spo2_baseline,
            "fev1_pct": fev1_pct,
            "dlco_pct": dlco_pct,
            "wll_lavages": wll_lavages,
            "alive": alive,
        })
    return patients


def generate_overview():
    gene_summaries = []
    for gene, seed in zip(ATLAS_GENES, SEEDS):
        pts = _simulate_cohort(gene, seed)
        rds_pct = round(sum(1 for p in pts if p["had_rds"]) / len(pts) * 100, 1)
        tx_pct = round(sum(1 for p in pts if p["received_lung_tx"]) / len(pts) * 100, 1)
        ild_pct = round(sum(1 for p in pts if p["has_ild"]) / len(pts) * 100, 1)
        pap_pct = round(sum(1 for p in pts if p["has_pap"]) / len(pts) * 100, 1)
        avg_fev1 = round(sum(p["fev1_pct"] for p in pts) / len(pts), 1)
        avg_dlco = round(sum(p["dlco_pct"] for p in pts) / len(pts), 1)
        avg_spo2 = round(sum(p["spo2_baseline"] for p in pts) / len(pts), 1)
        gene_summaries.append({
            "gene": gene["gene"],
            "locus": gene["locus"],
            "protein_size": gene["protein_size"],
            "inheritance": gene["inheritance"].split(";")[0].strip(),
            "disease_name": gene["disease_category"].split(";")[0].strip()[:80],
            "pathognomonic_short": gene["pathognomonic"].split(";")[0].strip()[:100],
            "onset_age_years_median": gene["onset_age_years_median"],
            "respiratory_failure_pct": rds_pct,
            "lung_tx_pct": tx_pct,
            "ild_pct": ild_pct,
            "pap_pct": pap_pct,
            "avg_fev1_pct": avg_fev1,
            "avg_dlco_pct": avg_dlco,
            "avg_spo2": avg_spo2,
        })

    all_pts = []
    for gene, seed in zip(ATLAS_GENES, SEEDS):
        all_pts.extend(_simulate_cohort(gene, seed))

    return {
        "title": "Hereditary Surfactant Dysfunction / Childhood Interstitial Lung Disease (chILD) Atlas",
        "subtitle": "Complete 8-Gene Hereditary Surfactant Dysfunction & chILD Reference — SFTPB-SFTPC-ABCA3-NKX2-1-MARS1-SLC34A2-CSF2RA-CSF2RB",
        "genes": [g["gene"] for g in ATLAS_GENES],
        "n_genes": 8,
        "total_patients": 320,
        "seeds": "2574–2581",
        "disease_classes": [
            "SFTPB — fatal neonatal RDS; absent lamellar bodies on EM; lung transplant only curative",
            "SFTPC — AD/de novo ILD; BRICHOS domain misfolding; HCQ first-line in children",
            "ABCA3 — most common genetic chILD; small dense LBs on EM; genotype predicts severity",
            "NKX2-1 — Brain-Thyroid-Lung triad; TTF-1 transcription factor; hypothyroidism + chorea",
            "MARS1 — combined ILD + liver disease (ILLD); UNIQUE combined phenotype among chILD genes",
            "SLC34A2 — pulmonary alveolar microlithiasis (PAM); snowstorm CXR; calcium phosphate deposits",
            "CSF2RA — hereditary PAP type 4; GM-CSF receptor alpha; WLL effective; inhaled GM-CSF variable",
            "CSF2RB — hereditary PAP type 5; GM-CSF receptor beta; inhaled GM-CSF INEFFECTIVE; HSCT curative",
        ],
        "gene_summary": gene_summaries,
        "aggregate_metrics": {
            "respiratory_failure_pct": round(sum(1 for p in all_pts if p["had_rds"]) / len(all_pts) * 100, 1),
            "lung_tx_pct": round(sum(1 for p in all_pts if p["received_lung_tx"]) / len(all_pts) * 100, 1),
            "ild_pct": round(sum(1 for p in all_pts if p["has_ild"]) / len(all_pts) * 100, 1),
            "pap_pct": round(sum(1 for p in all_pts if p["has_pap"]) / len(all_pts) * 100, 1),
            "avg_fev1_pct": round(sum(p["fev1_pct"] for p in all_pts) / len(all_pts), 1),
            "avg_dlco_pct": round(sum(p["dlco_pct"] for p in all_pts) / len(all_pts), 1),
            "avg_spo2": round(sum(p["spo2_baseline"] for p in all_pts) / len(all_pts), 1),
        },
        "clinical_pearls": [
            "SFTPB: c.121ins2 frameshift >70% of North American alleles — test FIRST in fatal neonatal RDS; "
            "absent lamellar bodies on EM = pathognomonic of null genotype",
            "SFTPC: p.Ile73Thr BRICHOS domain misfolding — de novo in 40%; variable expressivity; "
            "hydroxychloroquine first-line in children; nintedanib/pirfenidone in adult fibrosis",
            "ABCA3: MOST COMMON genetic chILD gene — small DENSE LBs on EM distinguish from SFTPB (absent LBs); "
            "biallelic nulls = neonatal fatal; two missenses = milder childhood ILD",
            "NKX2-1: BRAIN-THYROID-LUNG TRIAD — check thyroid function AND brain MRI in ANY infant with ILD; "
            "chorea is the most specific presenting feature; NKX2-1 IHC absent on lung biopsy",
            "MARS1: COMBINED ILD + HEPATOMEGALY = think MARS1; the ONLY chILD gene with hepatic involvement; "
            "p.Arg792His most common; liver transplant may stabilise lung disease",
            "SLC34A2: SNOWSTORM CXR — bilateral dense micronodular calcific opacities = pathognomonic of PAM; "
            "serum calcium/phosphate NORMAL; no effective medical treatment; lung transplant end-stage only",
            "CSF2RA vs. CSF2RB DDx: BOTH cause hereditary PAP without anti-GM-CSF antibodies; "
            "KEY DIFFERENCE: inhaled GM-CSF may partially benefit CSF2RA (alpha absent), NEVER CSF2RB (beta absent)",
            "PAP DIAGNOSIS: PAS-positive milky BAL fluid + crazy-paving CT + absent anti-GM-CSF Ab → hereditary PAP; "
            "STAT5 phosphorylation absent in monocytes confirms GM-CSF receptor defect",
            "WHOLE LUNG LAVAGE (WLL): therapeutic for PAP (CSF2RA/CSF2RB); "
            "instil warm saline 1-2 L, gravity drain; may repeat every 1-5 years as disease recurs",
            "HSCT INDICATION: CSF2RB biallelic null — curative; CSF2RA severe/refractory — curative; "
            "corrects macrophage progenitor defect; reduces need for repeated WLL",
        ],
    }


def generate_breakdown():
    breakdowns = []
    for gene, seed in zip(ATLAS_GENES, SEEDS):
        pts = _simulate_cohort(gene, seed)
        rds_pct = round(sum(1 for p in pts if p["had_rds"]) / len(pts) * 100, 1)
        tx_pct = round(sum(1 for p in pts if p["received_lung_tx"]) / len(pts) * 100, 1)
        surf_resp_pct = round(sum(1 for p in pts if p["surfactant_response"]) / len(pts) * 100, 1)
        pap_pct = round(sum(1 for p in pts if p["has_pap"]) / len(pts) * 100, 1)
        micro_pct = round(sum(1 for p in pts if p["has_microlithiasis"]) / len(pts) * 100, 1)
        avg_fev1 = round(sum(p["fev1_pct"] for p in pts) / len(pts), 1)
        avg_dlco = round(sum(p["dlco_pct"] for p in pts) / len(pts), 1)
        avg_onset = round(sum(p["onset_months"] for p in pts) / len(pts), 1)
        wll_mean = round(sum(p["wll_lavages"] for p in pts) / len(pts), 2)
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
            "respiratory_failure_pct": rds_pct,
            "lung_tx_pct": tx_pct,
            "surfactant_response_pct": surf_resp_pct,
            "pap_pct": pap_pct,
            "microlithiasis_pct": micro_pct,
            "avg_fev1_pct": avg_fev1,
            "avg_dlco_pct": avg_dlco,
            "avg_onset_age_months": avg_onset,
            "avg_wll_lavages": wll_mean,
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
            "respiratory_failure_pct": gene["respiratory_failure_pct"],
            "lung_tx_pct": gene["lung_tx_pct"],
            "pap_pct": gene["pap_pct"],
            "microlithiasis_pct": gene["microlithiasis_pct"],
            "nbs_indicated": gene["nbs_indicated"],
        }
    return {
        "gene_entries": gene_entries,
        "child_glossary": {
            "chILD (Childhood Interstitial Lung Disease)": (
                "chILD encompasses rare diffuse lung diseases of diverse aetiology in infants and children. "
                "Genetic causes include surfactant gene defects (SFTPB, SFTPC, ABCA3), transcription factor "
                "defects (NKX2-1), aminoacyl-tRNA synthetase defects (MARS1), phosphate transport defects "
                "(SLC34A2), and GM-CSF receptor defects (CSF2RA, CSF2RB). "
                "Lung biopsy + EM + protein immunostaining is the diagnostic gold standard."
            ),
            "Lamellar Bodies (LB) — EM Diagnosis Key": (
                "ABSENT LBs → SFTPB null (fatal, lung transplant only). "
                "SMALL DENSE ELECTRON-OPAQUE LBs → ABCA3 deficiency (pathognomonic). "
                "NEAR-NORMAL LBs → SFTPC, NKX2-1, MARS1, SLC34A2, CSF2RA/B. "
                "EM of open lung biopsy is the single most informative test distinguishing SFTPB from ABCA3."
            ),
            "PAP (Pulmonary Alveolar Proteinosis) — Three Types": (
                "1. AUTOIMMUNE PAP (most common, 90%) — anti-GM-CSF antibodies POSITIVE; adults; "
                "inhaled GM-CSF effective. "
                "2. HEREDITARY PAP type 4 (CSF2RA) — Ab ABSENT; paediatric; "
                "inhaled GM-CSF variably effective. "
                "3. HEREDITARY PAP type 5 (CSF2RB) — Ab ABSENT; paediatric; "
                "inhaled GM-CSF INEFFECTIVE; HSCT curative. "
                "WLL is the primary treatment for all PAP types."
            ),
            "PAM (Pulmonary Alveolar Microlithiasis) vs PAP": (
                "PAM (SLC34A2): calcium phosphate microliths fill alveoli; snowstorm CXR; SERUM Ca/PO4 NORMAL; "
                "BAL shows calcospherites; no treatment. "
                "PAP (CSF2RA/B): lipoproteinaceous material fills alveoli; PAS-positive milky BAL; "
                "crazy-paving CT; responds to WLL."
            ),
            "WLL (Whole Lung Lavage) — Procedure": (
                "Bilateral sequential WLL under GA: "
                "bronchoscope → wedge to subsegmental level → instil warm saline 1-2 L → gravity drain → "
                "repeat 5-10 cycles per lung → BAL clears from cloudy to clear. "
                "Right lung first, then left (larger lung last). "
                "ICU post-procedure. "
                "May need repeating every 1-5 years in hereditary PAP."
            ),
            "BRICHOS Domain (SFTPC)": (
                "BRICHOS is a chaperone domain in SP-C propeptide that prevents beta-sheet aggregation "
                "during post-translational processing. p.Ile73Thr disrupts BRICHOS → SP-C propeptide misfolding → "
                "ER retention → UPR activation → type II pneumocyte death → ILD. "
                "HCQ reduces ER stress pathway activation — rationale for chILD use."
            ),
            "Brain-Thyroid-Lung Syndrome (NKX2-1)": (
                "NKX2-1 (TTF-1) drives lung surfactant genes (SFTPB, SFTPC, ABCA3), "
                "thyroid differentiation (TG, TPO, SLC5A5), and striatal neuron development. "
                "Haploinsufficiency → variable triad: 50% all three; 30% two; 20% one organ. "
                "Chorea + congenital hypothyroidism → investigate lung. "
                "Lung component = worst prognosis."
            ),
        },
    }
