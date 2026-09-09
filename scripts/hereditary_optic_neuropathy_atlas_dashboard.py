#!/usr/bin/env python3
"""Hereditary-Optic-Neuropathy-Atlas — Complete 8-Gene Atlas
(OPA1 · OPA3 · WFS1 · CISD2 · TIMM8A · ACO2 · RTN4IP1 · SLC25A46).

OPA1     (OPA1 dynamin-like GTPase; 960 aa; ~112 kDa; 3q29; AD;
           Autosomal Dominant Optic Atrophy (ADOA/Kjer disease) — MOST COMMON HEREDITARY OPTIC NEUROPATHY;
           TRITANOPIA (BLUE-YELLOW colour vision defect) PATHOGNOMONIC — DDx LHON (red-green);
           OPA1-PLUS in 20% GTPase missense: add adPEO + mtDNA multiple deletions;
           Ethambutol/Linezolid/Amiodarone ABSOLUTE CI;
           seed SEED_BASE+0).
OPA3     (OPA3 outer IMM protein; 179 aa; ~20 kDa; 19q13.32; AD (Costeff) / AR (Behr);
           Costeff syndrome (3-methylglutaconic aciduria type III) — AD; Iraqi Jewish pGly93Ser founder;
           Behr syndrome — AR; optic atrophy + cerebellar ataxia + spastic paraplegia;
           3-METHYLGLUTACONIC ACIDURIA IN URINE PATHOGNOMONIC FOR COSTEFF;
           seed SEED_BASE+1).
WFS1     (Wolframin ER transmembrane glycoprotein; 890 aa; ~100 kDa; 4p16.1; AR (Wolfram) / AD (DFNA6);
           Wolfram syndrome (DIDMOAD) — Diabetes Insipidus, Diabetes Mellitus, Optic Atrophy, Deafness;
           OPTIC ATROPHY IS THE EARLIEST SIGN (mean age 5-8 y) — BEFORE diabetes mellitus onset;
           AR biallelic = Wolfram; AD heterozygous = DFNA6 low-frequency sensorineural deafness;
           seed SEED_BASE+2).
CISD2    (CDGSH iron-sulfur domain protein 2; 135 aa; ~15 kDa; 4q24; AR;
           Wolfram syndrome type 2 (WFS2) — GASTROINTESTINAL BLEEDING PATHOGNOMONIC (distinguishes from WFS1);
           NO diabetes insipidus — KEY DDx from WFS1 (DIDMOAD);
           Jordanian and Israeli Arab Bedouin founder variants; optic atrophy + DM + SNHL + GI bleeding;
           seed SEED_BASE+3).
TIMM8A   (Translocase of inner mitochondrial membrane 8A; 70 aa; ~8 kDa; Xq22.1; XLR;
           Mohr-Tranebjaerg syndrome (MTS) / Deafness-Dystonia-Optic Neuronopathy (DDON);
           PROGRESSIVE SENSORINEURAL DEAFNESS FIRST (childhood) → DYSTONIA (adolescence) → OPTIC NEURONOPATHY (adult);
           SEQUENTIAL ORDER IS PATHOGNOMONIC; DDP1 (deafness/dystonia protein 1); TIM8a-TIM13 complex;
           seed SEED_BASE+4).
ACO2     (Aconitase 2 / mitochondrial aconitase; 780 aa; ~83 kDa; 22q11.21; AR;
           Optic atrophy with cerebellar dysplasia and intellectual disability (AOCD);
           [4Fe-4S] IRON-SULFUR CLUSTER ESSENTIAL FOR CATALYTIC ACTIVITY — TCA cycle step 2;
           optic atrophy + cerebellar hypoplasia/atrophy + intellectual disability TRIAD;
           seed SEED_BASE+5).
RTN4IP1  (Reticulon 4 interacting protein 1 / OPA10; 435 aa; ~50 kDa; 6q21; AR;
           Autosomal recessive optic atrophy type 10 (OPA10);
           BOURBON ISLAND (French Reunion) FOUNDER c.296G>A / p.Arg99His;
           complex phenotype: optic atrophy +/- cerebellar atrophy +/- intellectual disability +/- peripheral neuropathy;
           seed SEED_BASE+6).
SLC25A46 (Solute carrier family 25 member 46; 418 aa; ~47 kDa; 5q22.1; AR;
           Hereditary motor neuropathy type VIB (HMN-VIB) + optic atrophy + cerebellar atrophy;
           OPTIC ATROPHY + CMT-LIKE PERIPHERAL NEUROPATHY + CEREBELLAR ATROPHY TRIAD;
           outer mitochondrial membrane lipid transfer, mitochondrial fission/fusion;
           seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 2414-2421).
"""

import random

SEED_BASE = 2414

ON_GENES = [
    # -- OPA1 -- Autosomal Dominant Optic Atrophy (Kjer disease) ---------------------------------
    {
        "gene": "OPA1",
        "alt_name": (
            "OPA1 (OPA1-960aa-3q29 / AD -- "
            "AUTOSOMAL-DOMINANT-OPTIC-ATROPHY-ADOA-KJER-DISEASE-MOST-COMMON-HEREDITARY-OPTIC-NEUROPATHY -- "
            "TRITANOPIA-BLUE-YELLOW-COLOUR-VISION-DEFECT-PATHOGNOMONIC-DDx-LHON-RED-GREEN -- "
            "OPA1-PLUS-20pct-GTPase-MISSENSE-adPEO-mtDNA-MULTIPLE-DELETIONS -- "
            "ETHAMBUTOL-LINEZOLID-AMIODARONE-ABSOLUTE-CI)"
        ),
        "protein": (
            "OPA1 -- 3q29 AD -- OPA1-960aa -- "
            "Dynamin-Like-GTPase-112kDa-Inner-Mitochondrial-Membrane-Cristae-Remodelling -- "
            "IMM-Fusion-Protein-Controls-Cristae-Morphology-mtDNA-Maintenance -- "
            "8-Isoforms-Alternative-Splicing-Exons-4-4b-5b -- "
            "GTPase-Domain-Middle-Domain-GED-PHD-TM-Anchor -- "
            "OMIM-Gene-605290-Disease-ADOA-165500-OPA1-Plus-125250"
        ),
        "locus": "3q29",
        "protein_size": "960 aa / 112 kDa",
        "inheritance": (
            "AD (autosomal dominant); OPA1 is a dynamin-like GTPase anchored to the inner mitochondrial membrane (IMM); "
            "controls IMM fusion, cristae morphology, cytochrome c retention, and mtDNA maintenance; "
            "haploinsufficiency mechanism: loss of 50% OPA1 -> reduced IMM fusion -> fragmented mitochondria -> "
            "retinal ganglion cell (RGC) apoptosis -> optic atrophy; "
            "most common variants: frameshift/nonsense (haploinsufficiency); GTPase domain missense (OPA1-Plus); "
            "PENETRANCE: ~80-90% for optic atrophy; variable expressivity within families; "
            "OPA1-PLUS (20% of cases): GTPase domain missense variants -> dominant-negative -> additional phenotypes: "
            "adPEO (progressive external ophthalmoplegia), proximal myopathy, ataxia, peripheral neuropathy, SNHL; "
            "mtDNA MULTIPLE DELETIONS: OPA1-Plus -> accumulate mtDNA deletions in muscle (Southern blot / long-range PCR); "
            "Prevalence: 1:25,000-1:50,000 (most common hereditary optic neuropathy)"
        ),
        "disease_category": "Autosomal dominant optic atrophy (ADOA/Kjer disease) — most common hereditary optic neuropathy; AD; tritanopia (blue-yellow) pathognomonic; OPA1-Plus in 20%",
        "disease_pathway": (
            "OPA1 (Optic Atrophy 1) encodes a dynamin-like GTPase anchored at the inner mitochondrial membrane (IMM) via an N-terminal transmembrane domain. "
            "IMM FUSION ROLE: OPA1 oligomers (L-OPA1 long-form) mediate IMM fusion together with MFN1/MFN2 (outer membrane); "
            "proteolytic cleavage by OMA1/YME1L generates S-OPA1 (short form); L:S ratio determines cristae morphology. "
            "CRISTAE MORPHOLOGY: OPA1 remodels cristae junctions; tubular cristae (high L-OPA1) -> efficient OXPHOS, cytochrome c sequestration (anti-apoptotic); "
            "ADOA MECHANISM: OPA1 haploinsufficiency -> reduced IMM fusion -> small, fragmented mitochondria in RGCs -> "
            "impaired OXPHOS -> ATP deficit -> selective RGC degeneration (RGCs have longest unmyelinated axons, highest energy demand); "
            "SELECTIVE RGC VULNERABILITY: papillomacular bundle (small fibres serving central vision) most energy-demanding -> "
            "centrocecal scotoma + temporal disc pallor pattern; "
            "OPA1-PLUS MECHANISM: GTPase-domain missense (dominant-negative) -> loss of GTPase activity -> mtDNA instability "
            "-> secondary mtDNA multiple deletions in post-mitotic tissues (muscle, neurons) -> additional OXPHOS deficiency -> "
            "adPEO, myopathy, ataxia, neuropathy, SNHL added to optic atrophy baseline. "
            "COLOUR VISION: tritanopia (blue-yellow defect) -- more severe than LHON (red-green); "
            "central scotoma enlarges over decades; some patients plateau."
        ),
        "pathognomonic": (
            "BILATERAL TEMPORAL OPTIC DISC PALLOR (temporal pallor > diffuse pallor) -- reflecting loss of "
            "papillomacular bundle fibres subserving central vision. "
            "TRITANOPIA (BLUE-YELLOW COLOUR VISION DEFECT) -- the key DDx from LHON (red-green protan/deutan defect): "
            "Farnsworth-Munsell 100-hue or D-15 test -> tritan axis errors; "
            "CENTROCECAL SCOTOMA: Amsler grid / Humphrey 30-2 visual field -- dense central scotoma extending to blind spot; "
            "OCT RNFL: temporal RNFL thinning (predominantly temporal sector); GCC/macular ganglion cell layer loss; "
            "ONSET: insidious onset in 1st-2nd decade; VA typically 6/12-6/60 at plateau; "
            "OPA1-PLUS (20%): add PEO (progressive external ophthalmoplegia) on ptosis examination + proximal myopathy; "
            "MUSCLE BIOPSY (OPA1-Plus): ragged-red fibres + multiple mtDNA deletions on Southern blot / long-range PCR; "
            "COLOUR VISION CONTRAST: OPA1 ADOA = TRITANOPIA (blue-yellow, S-cone axis); LHON = RED-GREEN (M/L-cone axis); "
            "Genetic: OPA1 pathogenic variant (NGS panel / WGS; check for large deletions by MLPA if NGS negative)."
        ),
        "treatment": (
            "NO PROVEN DISEASE-MODIFYING THERAPY (2026 approved): "
            "ETHAMBUTOL ABSOLUTE CI: ethambutol inhibits OPA1 function -> precipitates acute visual loss in ADOA; "
            "LINEZOLID ABSOLUTE CI: mitochondrial ribosome inhibitor -> ADOA exacerbation; "
            "AMIODARONE ABSOLUTE CI: mitochondrial toxin -> optic neuropathy; "
            "TOBACCO ABSOLUTE CI; "
            "OPHTHALMOLOGICAL REVIEW: annual OCT + HVF 30-2 + colour vision; "
            "LOW VISION AIDS: magnification, contrast enhancement, eccentric viewing; "
            "OPA1-PLUS: cardiac monitoring; "
            "IDEBENONE: Level C (not approved for ADOA); "
            "AVOID: valproate (mitochondrial toxin), aminoglycosides, nitrous oxide; "
            "FAMILY SCREENING: first-degree relatives (80-90% penetrance); ophthalmoscopy + OCT; "
            "GENETIC COUNSELLING: 50% transmission risk per child."
        ),
        "key_features": [
            "Bilateral temporal optic disc pallor — papillomacular bundle loss",
            "Tritanopia (blue-yellow) — KEY DDx from LHON (red-green)",
            "Centrocecal scotoma on visual field (Humphrey 30-2)",
            "Temporal RNFL thinning on OCT",
            "Insidious onset 1st-2nd decade; progressive but slow",
            "OPA1-Plus (20%): add adPEO + muscle mtDNA deletions",
            "Ethambutol / Linezolid / Amiodarone ABSOLUTE CI",
            "Most common hereditary optic neuropathy (~1:25,000)",
        ],
        "key_ddx": (
            "LHON (MT-ND4): acute/subacute onset; red-green dyschromatopsia; males>>; peripapillary telangiectasia; "
            "WFS1 (Wolfram): add DM + DI + SNHL (DIDMOAD); AR; "
            "Nutritional optic neuropathy: B12/folate; bilateral; dietary history; "
            "Ethambutol/Linezolid toxic: drug history; bilateral centrocecal; "
            "Glaucoma: RNFL nasal loss; IOP; cup/disc asymmetry; "
            "Compressive lesion: MRI orbit/brain mandatory if unilateral/asymmetric."
        ),
        "systemic_involvement": False,
        "onset_age": "1st-2nd decade (insidious, often detected at school vision screening); OPA1-Plus features emerge 3rd-4th decade",
        "surgical_urgency": "No acute surgical urgency; urgent ophthalmology if acute visual loss (consider OPA1-Plus crisis)",
        "gene_family": "Dynamin GTPase family — IMM fusion/cristae remodelling",
        "morphology": "Bilateral temporal optic disc pallor; centrocecal scotoma; temporal RNFL thinning",
    },
    # -- OPA3 -- Costeff Syndrome / Behr Syndrome ------------------------------------------------
    {
        "gene": "OPA3",
        "alt_name": (
            "OPA3 (OPA3-179aa-19q13.32 / AD-Costeff-AR-Behr -- "
            "COSTEFF-SYNDROME-3-METHYLGLUTACONIC-ACIDURIA-TYPE-III-MGA3-AD -- "
            "IRAQI-JEWISH-pGLY93SER-FOUNDER-PREVALENCE-1IN10000 -- "
            "BEHR-SYNDROME-AR-OPTIC-ATROPHY-CEREBELLAR-ATAXIA-SPASTIC-PARAPLEGIA-INTELLECTUAL-DISABILITY -- "
            "URINE-3-MGA-ELEVATED-3-10X-NORMAL-GC-MS-ORGANIC-ACIDS)"
        ),
        "protein": (
            "OPA3 -- 19q13.32 AD-AR -- OPA3-179aa -- "
            "Outer-IMM-Protein-20kDa-Mitochondrial-Dynamics-Regulator -- "
            "Localises-Outer-Mitochondrial-Membrane -- "
            "Regulates-Mitochondrial-Fission-Apoptosis-mtDNA-Maintenance -- "
            "OMIM-Gene-606580-Disease-Costeff-258501-ADOA3-165300"
        ),
        "locus": "19q13.32",
        "protein_size": "179 aa / 20 kDa",
        "inheritance": (
            "Bimodal inheritance (unique among optic atrophy genes): "
            "AD (Costeff syndrome / MGA3): p.Gly93Ser (c.277G>A) Iraqi Jewish founder; prevalence 1:10,000 Iraqi Jews; "
            "optic atrophy + spastic paraplegia + cerebellar ataxia + elevated urine 3-methylglutaconic acid; "
            "AR (Behr syndrome): biallelic OPA3 truncating/missense variants; "
            "optic atrophy + cerebellar ataxia + spastic paraplegia + intellectual disability; "
            "3-MGA-URIA: urine organic acids GC-MS: 3-methylglutaconic acid 3-10x normal; "
            "non-specific (also Barth/TAZ, DNAJC19, ATP synthase def); "
            "PENETRANCE: near-complete for p.Gly93Ser; variable expressivity in range of optic atrophy severity"
        ),
        "disease_category": "Costeff syndrome (AD, 3-MGA type III, Iraqi Jewish p.Gly93Ser) / Behr syndrome (AR); optic atrophy + cerebellar ataxia + spastic paraplegia; urine 3-MGA elevated",
        "disease_pathway": (
            "OPA3 encodes a 179 aa outer mitochondrial membrane (OMM) protein with unclear but essential function in "
            "mitochondrial dynamics, fission regulation, and metabolic coupling. "
            "3-MGA PATHWAY: leucine catabolism -> 3-methylglutaconyl-CoA -> 3-methylglutaconate; "
            "OPA3 disruption -> 3-MGA accumulation (mechanism uncertain; OPA3 may act as auxiliary hydratase or "
            "regulate HMG-CoA pathway flux); 3-MGA-URIA: detectable by urine organic acid GC-MS; "
            "NEUROLOGICAL PHENOTYPE: OPA3 loss -> mitochondrial fragmentation in retinal ganglion cells and "
            "cerebellar Purkinje cells -> RGC degeneration (optic atrophy) + cerebellar degeneration (ataxia) "
            "+ corticospinal tract degeneration (spastic paraplegia); "
            "BEHR (AR): biallelic OPA3 LOF -> more severe mitochondrial dynamics failure -> adds intellectual disability."
        ),
        "pathognomonic": (
            "COSTEFF SYNDROME (AD): "
            "URINE 3-METHYLGLUTACONIC ACID ELEVATED (3-10x normal): GC-MS urine organic acids; "
            "optic atrophy onset 1st decade; progressive spastic paraplegia; cerebellar ataxia; "
            "IRAQI JEWISH ANCESTRY: p.Gly93Ser founder; "
            "BEHR SYNDROME (AR): optic atrophy + cerebellar ataxia + spastic paraplegia + intellectual disability; "
            "MRI brain: cerebellar atrophy, white matter changes; "
            "CONTRAST WITH OPA1: OPA1 = pure optic atrophy; OPA3 = optic atrophy ALWAYS with neurological involvement."
        ),
        "treatment": (
            "NO DISEASE-MODIFYING THERAPY: "
            "Spastic paraplegia: physiotherapy, baclofen, intrathecal baclofen, orthotics; "
            "Cerebellar ataxia: OT, speech therapy, weighted utensils; "
            "Optic atrophy: low vision aids; annual OCT + HVF; "
            "CI: Ethambutol ABSOLUTE CI; Linezolid ABSOLUTE CI; Valproate HIGH RISK; "
            "MONITORING: annual ophthalmology + neurology + urine 3-MGA (progression tracking); "
            "GENETIC COUNSELLING: AD Costeff 50% risk; AR Behr 25% risk; founder testing in Iraqi Jewish families."
        ),
        "key_features": [
            "Urine 3-methylglutaconic acid elevated (GC-MS organic acids)",
            "Optic atrophy onset 1st decade (bilateral progressive)",
            "Spastic paraplegia — lower limb UMN signs",
            "Cerebellar ataxia — gait, intention tremor",
            "Iraqi Jewish founder p.Gly93Ser (AD/Costeff)",
            "AR Behr: adds intellectual disability",
            "OPA3 bimodal: AD Costeff / AR Behr",
            "Ethambutol / Linezolid ABSOLUTE CI",
        ],
        "key_ddx": (
            "OPA1 ADOA: optic atrophy only; NO 3-MGA; no spasticity; "
            "WFS1 Wolfram: DM + DI + OA + SNHL; no 3-MGA; no spasticity; "
            "Barth syndrome (TAZ): 3-MGA + cardiomyopathy; no optic atrophy; XLR males; "
            "DCMA (DNAJC19): 3-MGA + DCM + cerebellar; males; no optic atrophy; "
            "SPG7 (paraplegin): AR spastic paraplegia + cerebellar; no optic atrophy; no 3-MGA."
        ),
        "systemic_involvement": True,
        "onset_age": "1st decade (optic atrophy); spasticity/ataxia 1st-3rd decade",
        "surgical_urgency": "No acute surgical urgency; orthopaedic review for spasticity complications",
        "gene_family": "OPA3 outer mitochondrial membrane protein — mitochondrial dynamics",
        "morphology": "Bilateral optic disc pallor (temporal predominant); cerebellar atrophy on MRI",
    },
    # -- WFS1 -- Wolfram Syndrome (DIDMOAD) -------------------------------------------------------
    {
        "gene": "WFS1",
        "alt_name": (
            "WFS1 (WFS1-890aa-4p16.1 / AR-Wolfram-AD-DFNA6 -- "
            "WOLFRAM-SYNDROME-DIDMOAD-DIABETES-INSIPIDUS-DIABETES-MELLITUS-OPTIC-ATROPHY-DEAFNESS -- "
            "OPTIC-ATROPHY-EARLIEST-SIGN-MEAN-5-8Y-BEFORE-DIABETES-MELLITUS-ONSET -- "
            "WOLFRAMIN-ER-TRANSMEMBRANE-CALCIUM-UPR-REGULATOR -- "
            "AD-DFNA6-LOW-FREQUENCY-SNHL-DOMINANT-NEGATIVE-HETEROZYGOUS)"
        ),
        "protein": (
            "WFS1 -- 4p16.1 AR-AD -- WFS1-890aa -- "
            "Wolframin-100kDa-ER-Transmembrane-Glycoprotein-9-TM-Helices -- "
            "ER-Calcium-Homeostasis-UPR-ATF6alpha-Degradation-Regulator -- "
            "Expressed-Pancreatic-Beta-Cells-Neurons-Inner-Ear -- "
            "OMIM-Gene-606201-Disease-Wolfram-222300-DFNA6-600965"
        ),
        "locus": "4p16.1",
        "protein_size": "890 aa / 100 kDa",
        "inheritance": (
            "AR (autosomal recessive) for Wolfram syndrome (WFS); biallelic WFS1 pathogenic variants; "
            "wolframin is a 9-transmembrane-domain ER glycoprotein; "
            "FUNCTION: ER Ca2+ homeostasis, UPR modulation (ATF6alpha degradation); "
            "WFS1 deficiency -> ER Ca2+ dysregulation -> excess ATF6alpha -> ER stress -> apoptosis in: "
            "pancreatic beta-cells (DM), neurons (optic atrophy, DI), inner ear hair cells (SNHL); "
            "WOLFRAM COMPONENTS (DIDMOAD): "
            "Diabetes Mellitus (DM): insulin-dependent; mean age 6y; NOT autoimmune (islet antibody negative); "
            "Optic Atrophy: mean age 5-8y; EARLIEST SIGN; centrocecal scotoma; "
            "Diabetes Insipidus (DI): central; ADH deficiency; "
            "Deafness: sensorineural; mid-frequency initially; progressive; "
            "AD heterozygous: DFNA6 -- low-frequency SNHL only (dominant-negative missense)"
        ),
        "disease_category": "Wolfram syndrome / DIDMOAD (AR) — DI + DM + Optic Atrophy + Deafness; optic atrophy EARLIEST manifestation (age 5-8y); AD DFNA6 = low-frequency SNHL only",
        "disease_pathway": (
            "WFS1 (wolframin) is a 9-transmembrane ER-resident glycoprotein expressed at highest levels in pancreatic beta-cells, "
            "neurons, and inner ear spiral ganglion cells. "
            "ER CALCIUM FUNCTION: wolframin regulates ER Ca2+ efflux via IP3R/SERCA modulation; "
            "maintains ER Ca2+ stores essential for protein folding chaperone function. "
            "UPR LINK: WFS1 promotes degradation of ATF6alpha (ER stress sensor) by recruiting to ubiquitin E3 ligase complex; "
            "WFS1 deficiency -> ATF6alpha accumulates -> excess UPR activation -> "
            "CHOP-mediated apoptosis in: beta-cells (DM), RGCs (optic atrophy), "
            "paraventricular hypothalamic neurons (DI), spiral ganglion (SNHL). "
            "OPTIC ATROPHY EARLIEST SIGN: WFS1 expression in optic nerve oligodendrocytes -> "
            "demyelination precedes RGC loss; optic atrophy mean age 5-8y (predating DM in ~50%). "
            "DFNA6 (AD): missense -> dominant-negative -> ER Ca2+ in spiral ganglion -> "
            "low-frequency SNHL only (insufficient ER stress in beta-cells with haploinsufficiency)."
        ),
        "pathognomonic": (
            "WOLFRAM SYNDROME (AR) -- DIDMOAD: "
            "OPTIC ATROPHY EARLIEST SIGN (mean age 5-8y): bilateral optic disc pallor; centrocecal scotoma; "
            "OCT temporal RNFL thinning before VA loss; "
            "INSULIN-DEPENDENT DM WITHOUT ISLET ANTIBODIES: GAD65/IA2/ZnT8 NEGATIVE -- "
            "distinguishes from T1DM autoimmune; mean DM onset 6y; "
            "CENTRAL DI: MRI hypothalamus absent posterior pituitary bright spot; "
            "SENSORINEURAL DEAFNESS: mid-frequency initially; progressive; "
            "COMPARE WFS2 (CISD2): GI BLEEDING ADDED; NO DI; same DM + OA + SNHL but different mechanism."
        ),
        "treatment": (
            "DIABETES MELLITUS: insulin therapy; CGMS mandatory; "
            "DIABETES INSIPIDUS: desmopressin (DDAVP) nasal/oral/SC; careful fluid balance; avoid dehydration; "
            "OPTIC ATROPHY: low vision rehabilitation; idebenone/CoQ10 Level C; annual OCT + HVF; "
            "SNHL: hearing aids; annual audiometry; cochlear implant consideration; "
            "AVOID: Ethambutol ABSOLUTE CI; Linezolid ABSOLUTE CI; aminoglycosides AVOID (cochlear); "
            "NO DESMOPRESSIN NEEDED IN WFS2 (no DI); "
            "UROLOGICAL: urodynamics; anticholinergics; CIC if neurogenic bladder; "
            "MULTIDISCIPLINARY: endocrinology + ophthalmology + audiology + neurology + urology + psychiatry."
        ),
        "key_features": [
            "Optic atrophy EARLIEST sign (mean age 5-8y) — before DM onset",
            "DIDMOAD: DI + DM (antibody-negative) + Optic Atrophy + Deafness",
            "Insulin-dependent DM without islet antibodies (GAD/IA2 negative)",
            "Central DI — absent posterior pituitary bright spot on MRI",
            "DFNA6 (AD): isolated low-frequency SNHL only",
            "Ethambutol ABSOLUTE CI; aminoglycosides AVOID",
            "Multidisciplinary: endo + ophtho + audiology + neurology + urology",
            "Neurogenic bladder + psychiatric features in subset",
        ],
        "key_ddx": (
            "CISD2 (WFS2): ADDS GI bleeding; NO DI; same DM + OA + SNHL; "
            "Type 1 DM: islet antibodies POSITIVE; NO optic atrophy; "
            "OPA1 ADOA: optic atrophy ONLY; NO DM/DI/SNHL; "
            "LHON: acute/subacute; males; red-green; NO DM/DI; "
            "MELAS: maternal; stroke-like episodes; NO DI."
        ),
        "systemic_involvement": True,
        "onset_age": "Optic atrophy mean 5-8y; DM mean 6y; DI usually teens; SNHL variable",
        "surgical_urgency": "No acute surgical urgency; DI requires urgent DDAVP if hypernatraemia",
        "gene_family": "ER transmembrane glycoprotein — ER Ca2+/UPR regulation (wolframin family)",
        "morphology": "Bilateral optic disc pallor; centrocecal scotoma; absent posterior pituitary bright spot on MRI",
    },
    # -- CISD2 -- Wolfram Syndrome Type 2 --------------------------------------------------------
    {
        "gene": "CISD2",
        "alt_name": (
            "CISD2 (CISD2-135aa-4q24 / AR -- "
            "WOLFRAM-SYNDROME-TYPE-2-WFS2-GASTROINTESTINAL-BLEEDING-PATHOGNOMONIC-DISTINGUISHES-FROM-WFS1 -- "
            "NO-DIABETES-INSIPIDUS-KEY-DDx-WFS1-DIDMOAD -- "
            "JORDANIAN-ISRAELI-ARAB-BEDOUIN-FOUNDER-pGLU132LYS -- "
            "CDGSH-IRON-SULFUR-DOMAIN-NEET-PROTEIN-MITOCHONDRIAL-IRON-ROS)"
        ),
        "protein": (
            "CISD2 -- 4q24 AR -- CISD2-135aa -- "
            "CDGSH-Iron-Sulfur-Domain-Protein-2-15kDa-NEET-Protein -- "
            "Outer-Mitochondrial-Membrane-[2Fe2S]-Cluster -- "
            "Regulates-Mitochondrial-Iron-ROS-Homeostasis-Autophagy -- "
            "OMIM-Gene-611507-Disease-WFS2-604928"
        ),
        "locus": "4q24",
        "protein_size": "135 aa / 15 kDa",
        "inheritance": (
            "AR (autosomal recessive); CISD2 encodes a 135 aa NEET-family outer mitochondrial membrane protein; "
            "contains a [2Fe-2S] cluster in CDGSH zinc-finger domain; "
            "CISD2/NAF-1: redox-active protein involved in mitochondrial iron-sulfur cluster biogenesis and transfer; "
            "CISD2 deficiency -> excess mitochondrial iron accumulation -> Fenton chemistry -> oxidative damage; "
            "Jordanian and Israeli Arab Bedouin founder: p.Glu132Lys (c.394G>A); "
            "DISTINCTION from WFS1: WFS2 does NOT cause diabetes insipidus (no hypothalamic involvement); "
            "GASTROINTESTINAL BLEEDING: upper GI peptic-like ulcers in WFS2 (absent in WFS1)"
        ),
        "disease_category": "Wolfram syndrome type 2 (WFS2) — DM + optic atrophy + SNHL + GI bleeding (pathognomonic); NO diabetes insipidus; Bedouin founder; AR",
        "disease_pathway": (
            "CISD2 encodes NAF-1 (nutrient-deprivation autophagy factor 1), a NEET-family outer mitochondrial membrane protein. "
            "[2Fe-2S] CLUSTER: CISD2 contains a CDGSH domain coordinating a labile [2Fe-2S] cluster; "
            "transferable to recipient proteins -> iron-sulfur cluster maturation in cytosol and ER; "
            "IRON REGULATION: CISD2 modulates mitochondrial iron import and export; "
            "CISD2 LOF -> excess iron retention in mitochondria -> Fenton reaction (Fe2+ + H2O2 -> OH radical) -> "
            "mitochondrial oxidative damage -> cell death in: beta-cells (DM), RGCs (optic atrophy), "
            "cochlear cells (SNHL), GI mucosa (bleeding); "
            "GI BLEEDING MECHANISM: CISD2 expressed in GI epithelium; iron dysregulation -> "
            "mucosal oxidative damage -> peptic-like ulceration -> upper GI bleeding; "
            "KEY WFS1 vs WFS2: WFS1 -> ER Ca2+/UPR pathway -> DI involvement; "
            "WFS2 -> mitochondrial iron-ROS pathway -> GI bleeding; NO DI."
        ),
        "pathognomonic": (
            "WOLFRAM SYNDROME TYPE 2 (WFS2) -- DISTINGUISHING FROM WFS1: "
            "GASTROINTESTINAL BLEEDING PATHOGNOMONIC: recurrent upper GI bleeding (haematemesis/melaena); "
            "endoscopy: peptic ulcer-like lesions without H. pylori; "
            "NO DIABETES INSIPIDUS: WFS2 does NOT cause central DI; "
            "no posterior pituitary signal loss on MRI; no polyuria/polydipsia; "
            "KEY DDx from WFS1 (DIDMOAD includes DI); "
            "DM + OPTIC ATROPHY + SNHL: present in both WFS1 and WFS2 (overlap); "
            "JORDANIAN/BEDOUIN ANCESTRY: p.Glu132Lys; "
            "ALGORITHM: Wolfram + GI BLEEDING + NO DI -> WFS2/CISD2; Wolfram + DI + NO GI bleeding -> WFS1."
        ),
        "treatment": (
            "DIABETES MELLITUS: insulin therapy; CGMS; "
            "GASTROINTESTINAL BLEEDING: PPI prophylaxis ALL WFS2; upper GI endoscopy annual; "
            "H. pylori screen; iron supplementation if anaemia; AVOID NSAIDs/aspirin; "
            "OPTIC ATROPHY: low vision aids; annual OCT + HVF; idebenone/CoQ10 Level C; "
            "SNHL: hearing aids; annual audiometry; "
            "AVOID: Ethambutol ABSOLUTE CI; Linezolid ABSOLUTE CI; aminoglycosides AVOID; "
            "NO DESMOPRESSIN (no DI in WFS2); "
            "MONITORING: annual GI endoscopy + FBC + ophthalmology + audiology + endocrinology."
        ),
        "key_features": [
            "GI bleeding (haematemesis/melaena) PATHOGNOMONIC — absent in WFS1",
            "NO diabetes insipidus — KEY DDx from WFS1 (DIDMOAD)",
            "DM + optic atrophy + SNHL (overlap with WFS1)",
            "CISD2/NAF-1: mitochondrial iron-sulfur NEET protein [2Fe-2S]",
            "Jordanian/Israeli Arab Bedouin founder p.Glu132Lys",
            "PPI prophylaxis mandatory; NSAIDs / aspirin AVOID",
            "Ethambutol ABSOLUTE CI; aminoglycosides AVOID",
            "Annual GI endoscopy + FBC at every review",
        ],
        "key_ddx": (
            "WFS1 (Wolfram): DI PRESENT (posterior pituitary absent on MRI); NO GI bleeding; ER Ca2+ pathway; "
            "OPA1 ADOA: optic atrophy alone; NO DM/SNHL/GI bleeding; "
            "Peptic ulcer disease: GI bleeding without OA/DM; H. pylori positive; "
            "MODY: DM non-insulin; no optic atrophy; GCK/HNF genes; "
            "Haemochromatosis: iron overload; GI features; NO optic atrophy; HFE."
        ),
        "systemic_involvement": True,
        "onset_age": "DM mean childhood; optic atrophy childhood; GI bleeding 1st-2nd decade; SNHL progressive",
        "surgical_urgency": "Acute GI haemorrhage requires urgent endoscopy + PPI; no ophthalmic surgical urgency",
        "gene_family": "NEET protein family (CDGSH iron-sulfur domain) — outer mitochondrial membrane iron-ROS regulation",
        "morphology": "Bilateral optic disc pallor; centrocecal scotoma; no DI on MRI",
    },
    # -- TIMM8A -- Mohr-Tranebjaerg Syndrome / DDON -----------------------------------------------
    {
        "gene": "TIMM8A",
        "alt_name": (
            "TIMM8A (TIMM8A-70aa-Xq22.1 / XLR -- "
            "MOHR-TRANEBJAERG-SYNDROME-MTS-DEAFNESS-DYSTONIA-OPTIC-NEURONOPATHY-DDON -- "
            "PROGRESSIVE-SNHL-CHILDHOOD-DYSTONIA-ADOLESCENCE-OPTIC-NEURONOPATHY-ADULT-SEQUENTIAL-PATHOGNOMONIC -- "
            "DDP1-TIM8A-TIM13-HEXAMERIC-CHAPERONE-ANT-PRECURSOR-CHAPERONING -- "
            "X-LINKED-RECESSIVE-HEMIZYGOUS-MALES-FULLY-AFFECTED)"
        ),
        "protein": (
            "TIMM8A -- Xq22.1 XLR -- TIMM8A-70aa -- "
            "Translocase-Inner-Mitochondrial-Membrane-8A-8kDa-DDP1 -- "
            "TIM8a-TIM13-Hexameric-Chaperone-IMS-Intermembrane-Space -- "
            "Twin-CX3C-Motif-Guides-Hydrophobic-TM-Precursors-to-TIM22 -- "
            "OMIM-Gene-300356-Disease-MTS-304700"
        ),
        "locus": "Xq22.1",
        "protein_size": "70 aa / 8 kDa",
        "inheritance": (
            "XLR (X-linked recessive); hemizygous males affected; "
            "carrier females: usually asymptomatic (some mild hearing loss or focal dystonia); "
            "TIMM8A (DDP1) encodes a 70 aa small TIM protein in the mitochondrial intermembrane space (IMS); "
            "TWIN CX3C MOTIF: two conserved Cys-X3-Cys motifs; "
            "TIM8a FORMS HEXAMERIC COMPLEX with TIM13: chaperones hydrophobic transmembrane precursors across IMS; "
            "KEY SUBSTRATES: adenine nucleotide translocators ANT1/ANT2 (SLC25A4/A5), TIM23; "
            "TIMM8A LOF -> ANT precursors misfolded in IMS -> reduced ANT1/2 -> impaired ATP/ADP exchange -> "
            "bioenergetic failure in cochlear cells (SNHL), striatum (dystonia), RGCs (optic neuronopathy)"
        ),
        "disease_category": "Mohr-Tranebjaerg syndrome (MTS) / DDON — X-linked recessive; SNHL (childhood) -> dystonia (adolescence) -> optic neuronopathy (adult); sequential order pathognomonic",
        "disease_pathway": (
            "TIMM8A encodes DDP1 (deafness/dystonia protein 1), a small 70 aa protein of the TIM8/13 family in the mitochondrial IMS. "
            "TIM8A-TIM13 COMPLEX: forms a heterohexameric chaperone (3xTIM8a + 3xTIM13) in the IMS; "
            "binds hydrophobic transmembrane precursors co-translationally after OMM import (through TOM complex); "
            "guides them to the TIM22 insertion complex for IMM integration; "
            "KEY SUBSTRATES: ANT1/SLC25A4, ANT2/SLC25A5, TIM23, and other multi-spanning IMM proteins; "
            "PATHOMECHANISM: TIMM8A/DDP1 LOF -> ANT1/2 precursors misfolded in IMS -> reduced ANT -> "
            "impaired ATP/ADP exchange across IMM -> ATP deficit in: cochlear spiral ganglion (SNHL, first), "
            "striatum/basal ganglia (dystonia, second), retinal ganglion cells (optic neuronopathy, third). "
            "SEQUENTIAL ONSET: cochlear cells most vulnerable (highest TIMM8A expression + ANT dependence) first."
        ),
        "pathognomonic": (
            "MOHR-TRANEBJAERG SYNDROME -- SEQUENTIAL NEUROLOGICAL PROGRESSION IS PATHOGNOMONIC: "
            "1. PROGRESSIVE SENSORINEURAL DEAFNESS (childhood, age 2-10y): FIRST; severe-profound high-frequency SNHL; "
            "2. DYSTONIA (adolescence/early adulthood): focal -> generalised; appendicular then orofacial; "
            "3. OPTIC NEURONOPATHY (adult): gradual central visual loss; optic disc pallor; OCT RNFL thinning; "
            "X-LINKED MALE PREDOMINANCE: hemizygous males fully affected; carrier females usually normal; "
            "COCHLEAR IMPLANT CAVEAT: implant BEFORE dystonia develops (programming impaired by involuntary movements); "
            "LEVODOPA TRIAL MANDATORY: ~10-15% DRD-like response; "
            "MRI brain: striatal atrophy + generalised cerebral atrophy advanced disease."
        ),
        "treatment": (
            "SNHL: hearing aids early; cochlear implant before dystonia develops (timing critical); "
            "sign language education; "
            "DYSTONIA: Levodopa trial mandatory (10-15% DRD-like response); "
            "Trihexyphenidyl; Baclofen (systemic/intrathecal); Botulinum toxin; DBS GPi (refractory); "
            "OPTIC NEURONOPATHY: low vision aids; annual OCT + HVF; "
            "AVOID: Ethambutol ABSOLUTE CI; Linezolid ABSOLUTE CI; "
            "SWALLOWING: SALT assessment; PEG if severe dysphagia; "
            "CARRIER FEMALES: screening SNHL + neurological assessment."
        ),
        "key_features": [
            "Sequential: SNHL (childhood) -> Dystonia (adolescence) -> Optic neuronopathy (adult)",
            "X-linked recessive — hemizygous males affected; carrier females usually unaffected",
            "DDP1 / TIM8a-TIM13 chaperone; ANT precursor chaperoning",
            "Cochlear implant BEFORE dystonia (critical timing)",
            "Levodopa trial mandatory (DRD-like response ~10-15%)",
            "DBS GPi for refractory generalised dystonia",
            "Ethambutol / Linezolid ABSOLUTE CI",
            "Carrier females: screen SNHL baseline",
        ],
        "key_ddx": (
            "OPA1 ADOA: optic atrophy only; no SNHL/dystonia; AD; autosomal; "
            "LHON: acute/subacute visual loss; males; maternal; no dystonia/deafness sequential; "
            "Wilson disease: KF ring; dystonia; liver; AR; copper; "
            "PKAN (PANK2): eye-of-tiger sign; dystonia; no SNHL; AR; "
            "MELAS: maternal; stroke-like; lactic acidosis; no sequential deafness-dystonia; "
            "ARTS syndrome (TIMM8A deletion): lethal in males; distinct from MTS."
        ),
        "systemic_involvement": True,
        "onset_age": "SNHL: childhood (2-10y); dystonia: adolescence-early adulthood; optic neuronopathy: adulthood",
        "surgical_urgency": "No acute surgical urgency; cochlear implant timing critical (early childhood)",
        "gene_family": "Small TIM protein family (TIM8/13) — IMS chaperone complex",
        "morphology": "Optic disc pallor (late); striatal atrophy + cerebral atrophy on MRI (advanced)",
    },
    # -- ACO2 -- Optic Atrophy with Cerebellar Dysplasia (AOCD) -----------------------------------
    {
        "gene": "ACO2",
        "alt_name": (
            "ACO2 (ACO2-780aa-22q11.21 / AR -- "
            "OPTIC-ATROPHY-CEREBELLAR-DYSPLASIA-INTELLECTUAL-DISABILITY-TRIAD-AOCD -- "
            "MITOCHONDRIAL-ACONITASE-[4Fe-4S]-IRON-SULFUR-CLUSTER-TCA-CYCLE-STEP-2-CITRATE-TO-ISOCITRATE -- "
            "PALE-OPTIC-DISC-SMALL-CUP-PATHOGNOMONIC-VERSUS-GLAUCOMA-LARGE-CUP -- "
            "BRAZIL-POPULATION-FOUNDER-VARIANTS)"
        ),
        "protein": (
            "ACO2 -- 22q11.21 AR -- ACO2-780aa -- "
            "Aconitase-2-Mitochondrial-Aconitase-83kDa-[4Fe-4S]-Cluster -- "
            "TCA-Cycle-Enzyme-Citrate-to-Isocitrate-via-cis-Aconitate -- "
            "Matrix-Enzyme-ROS-Sensor -- "
            "OMIM-Gene-100850-Disease-AOCD-616289-SCAR4"
        ),
        "locus": "22q11.21",
        "protein_size": "780 aa / 83 kDa",
        "inheritance": (
            "AR (autosomal recessive); ACO2 encodes mitochondrial aconitase-2 (mAcon), a 780 aa TCA cycle enzyme; "
            "FUNCTION: TCA cycle step 2: citrate <-> isocitrate (via cis-aconitate); "
            "[4Fe-4S] CLUSTER: essential for catalytic activity; sensitive to superoxide/NO -> ROS -> cluster degradation; "
            "ACO2 LOF -> TCA cycle block at citrate/isocitrate -> "
            "citrate accumulates -> downstream intermediates (alpha-KG, succinate, malate) reduced -> "
            "OXPHOS/NADH impaired -> energy deficit in high-demand neurons; "
            "Brazilian population founder; Israeli families; SCAR4 allelic"
        ),
        "disease_category": "Optic atrophy with cerebellar dysplasia and intellectual disability (AOCD); AR; TCA cycle aconitase deficiency; optic atrophy + cerebellar atrophy + intellectual disability triad",
        "disease_pathway": (
            "ACO2 (mitochondrial aconitase) catalyses TCA cycle step 2: citrate -> cis-aconitate -> isocitrate; "
            "[4Fe-4S] cluster mediates substrate binding. "
            "TCA CYCLE BLOCK: ACO2 pathogenic variants -> reduced aconitase activity -> citrate accumulates -> "
            "isocitrate, alpha-KG, succinate, fumarate, malate, oxaloacetate all reduced -> "
            "NADH and FADH2 impaired -> reduced ETC input -> decreased ATP -> bioenergetic failure. "
            "NEUROTOXICITY: citrate chelates Ca2+ -> neuronal Ca2+ signalling disrupted; "
            "citrate efflux inhibits PFK1 -> glycolysis blocked -> dual bioenergetic crisis. "
            "ROS VULNERABILITY: [4Fe-4S] cluster sensitive to superoxide -> "
            "retina/photoreceptors and cerebellar Purkinje cells selectively vulnerable."
        ),
        "pathognomonic": (
            "AOCD CLINICAL TRIAD: "
            "1. OPTIC ATROPHY: bilateral disc pallor; SMALL PHYSIOLOGICAL CUP (distinguishes from glaucoma large cup); "
            "centrocecal scotoma; OCT temporal RNFL thinning; onset infancy-early childhood; "
            "2. CEREBELLAR ATROPHY/HYPOPLASIA: MRI vermal > hemispheric; truncal ataxia; dysmetria; hypotonia; "
            "3. INTELLECTUAL DISABILITY: global developmental delay; speech delay prominent; "
            "ADDITIONAL: spastic paraplegia (SCAR4 allelic), peripheral neuropathy, RP-like pigmentation subset; "
            "METABOLIC: urine organic acids may show elevated citrate/cis-aconitate (non-specific); "
            "ENZYME ASSAY: lymphocyte/fibroblast aconitase activity reduced (confirmatory)."
        ),
        "treatment": (
            "OPTIC ATROPHY: low vision aids; annual OCT + HVF; Ethambutol ABSOLUTE CI; Linezolid ABSOLUTE CI; "
            "CEREBELLAR: physiotherapy; OT; weighted utensils; coordination exercises; "
            "ID/DD: early intervention; speech-language therapy; special education; "
            "SPASTICITY (SCAR4): baclofen; physiotherapy; orthotics; "
            "EMPIRIC: riboflavin (B2) for [4Fe-4S] biogenesis; CoQ10 antioxidant; NAC; "
            "AVOID: valproate HIGH RISK; aminoglycosides AVOID; linezolid ABSOLUTE CI; "
            "MONITORING: annual ophthalmology + neurology; MRI brain 2-3 yearly."
        ),
        "key_features": [
            "Optic atrophy + cerebellar atrophy + intellectual disability TRIAD (AOCD)",
            "Pale disc with small cup (distinguishes from glaucoma large cup)",
            "ACO2: mitochondrial aconitase TCA cycle step 2 [4Fe-4S] cluster",
            "TCA cycle block: citrate accumulates -> dual bioenergetic crisis",
            "Brazilian founder variants; consanguineous families",
            "MRI: cerebellar vermal > hemispheric atrophy/hypoplasia",
            "Riboflavin + CoQ10 empiric; valproate HIGH RISK",
            "Ethambutol ABSOLUTE CI; linezolid ABSOLUTE CI",
        ],
        "key_ddx": (
            "OPA1 ADOA: optic atrophy only; no cerebellar/ID; AD; "
            "RTN4IP1 (OPA10): AR; same OA + cerebellar + ID; different gene; "
            "Joubert (JBTS): cerebellar molar tooth sign; no optic atrophy; ciliopathy; "
            "ARSACS: severe cerebellar + spasticity; Quebec founder; no optic atrophy; SACS; "
            "CDG: cerebellar + ID; transferrin isoforms; CDG genes."
        ),
        "systemic_involvement": True,
        "onset_age": "Infancy-early childhood (optic atrophy and developmental delay from birth/early infancy)",
        "surgical_urgency": "No acute surgical urgency",
        "gene_family": "Aconitase family — [4Fe-4S] iron-sulfur cluster TCA cycle enzymes",
        "morphology": "Bilateral pale optic disc with small cup; cerebellar atrophy on MRI",
    },
    # -- RTN4IP1 -- Autosomal Recessive Optic Atrophy Type 10 (OPA10) ----------------------------
    {
        "gene": "RTN4IP1",
        "alt_name": (
            "RTN4IP1 (RTN4IP1-435aa-6q21 / AR-OPA10 -- "
            "AUTOSOMAL-RECESSIVE-OPTIC-ATROPHY-TYPE-10-OPA10 -- "
            "FRENCH-REUNION-BOURBON-ISLAND-FOUNDER-c296GA-pARG99HIS -- "
            "COMPLEX-PHENOTYPE-OPTIC-ATROPHY-CEREBELLAR-INTELLECTUAL-DISABILITY-PERIPHERAL-NEUROPATHY -- "
            "MITOCHONDRIAL-MATRIX-GTPASE-UBL-DOMAIN)"
        ),
        "protein": (
            "RTN4IP1 -- 6q21 AR -- RTN4IP1-435aa -- "
            "Reticulon-4-Interacting-Protein-1-50kDa-OPA10 -- "
            "Mitochondrial-Matrix-GTPase-UBL-Domain -- "
            "Interacts-Reticulon-4-Nogo-ER-Mitochondria-Contact-Sites -- "
            "OMIM-Gene-610502-Disease-OPA10-616732"
        ),
        "locus": "6q21",
        "protein_size": "435 aa / 50 kDa",
        "inheritance": (
            "AR (autosomal recessive); RTN4IP1 encodes a mitochondrial matrix protein with "
            "N-terminal UBL (ubiquitin-like) domain and C-terminal GTPase-like domain; "
            "interacts with reticulon-4 (Nogo-A) at ER-mitochondria contact sites; "
            "FUNCTION: controls mitochondrial morphology, fusion/fission, mtDNA maintenance; "
            "BOURBON ISLAND (FRENCH REUNION) FOUNDER: c.296G>A / p.Arg99His; "
            "Phenotypic spectrum: isolated optic atrophy (mild) to full complex OPA10; "
            "Severity correlates with residual GTPase activity"
        ),
        "disease_category": "Autosomal recessive optic atrophy type 10 (OPA10); AR; complex phenotype: optic atrophy +/- cerebellar +/- ID +/- peripheral neuropathy; Bourbon Island founder",
        "disease_pathway": (
            "RTN4IP1 (Reticulon-4 interacting protein 1) is a mitochondrial matrix protein: "
            "UBL domain at N-terminus (protein interaction scaffold); GTPase domain at C-terminus. "
            "ER-MITOCHONDRIA CONTACT: RTN4IP1 interacts with reticulon-4 (Nogo-A) at MAM (ER-mitochondria associated membranes); "
            "MITOCHONDRIAL DYNAMICS: participates in fission/fusion balance regulation; "
            "RTN4IP1 LOF -> mitochondrial fragmentation -> impaired energy coupling -> "
            "RGC bioenergetic failure (optic atrophy), cerebellar Purkinje cell loss (ataxia), "
            "peripheral neuron degeneration (neuropathy), cortical neuron dysfunction (ID); "
            "GTPase ACTIVITY CORRELATES WITH SEVERITY: p.Arg99His (founder) -> partial residual GTPase -> "
            "variable phenotype from isolated OA to complex OPA10."
        ),
        "pathognomonic": (
            "OPA10 (RTN4IP1-AR) PHENOTYPIC SPECTRUM: "
            "CORE: BILATERAL OPTIC ATROPHY -- disc pallor; centrocecal scotoma; temporal RNFL thinning; "
            "onset infancy to early childhood; "
            "COMPLEX OPA10: cerebellar ataxia (MRI cerebellar hypoplasia/atrophy); "
            "intellectual disability (global DD); peripheral neuropathy (axonal, EMG/NCS confirmation); "
            "BOURBON ISLAND FOUNDER (Reunion, French Indian Ocean): c.296G>A p.Arg99His; "
            "targeted sequencing first in patients of Reunion origin; "
            "VARIABLE PENETRANCE: same variant -> isolated OA in some, full complex in others; "
            "Include RTN4IP1 in all AR optic atrophy gene panels."
        ),
        "treatment": (
            "OPTIC ATROPHY: low vision aids; annual OCT + HVF; Ethambutol ABSOLUTE CI; Linezolid ABSOLUTE CI; "
            "idebenone/CoQ10 Level C; "
            "CEREBELLAR: physiotherapy; riluzole Level C; OT; "
            "ID: early intervention; speech-language therapy; educational support; "
            "NEUROPATHY: pain management (gabapentin); orthotics (AFO); physiotherapy; "
            "AVOID: valproate HIGH RISK; aminoglycosides AVOID; "
            "GENETIC COUNSELLING: AR 25% risk; Reunion Island population founder effect."
        ),
        "key_features": [
            "AR optic atrophy type 10 (OPA10) — biallelic RTN4IP1",
            "Bourbon Island (Reunion) founder c.296G>A / p.Arg99His",
            "Complex: optic atrophy +/- cerebellar +/- ID +/- peripheral neuropathy",
            "RTN4IP1: mitochondrial matrix GTPase; ER-mitochondria contact site protein",
            "Variable severity: isolated OA (mild) to full complex OPA10",
            "Include RTN4IP1 in all AR optic atrophy NGS panels",
            "Riluzole Level C for cerebellar component",
            "Ethambutol / Linezolid ABSOLUTE CI",
        ],
        "key_ddx": (
            "OPA1 ADOA: AD; optic atrophy only (or OPA1-Plus); no cerebellar/ID; "
            "ACO2 AOCD: AR; same OA + cerebellar + ID; TCA cycle; Brazilian founder; "
            "SLC25A46: AR; OA + CMT-like neuropathy + cerebellar; neuropathy predominant; "
            "WFS1 Wolfram: AR; DM + DI + OA + SNHL; ER protein; "
            "Childhood glaucoma: large cup; high IOP; no cerebellar/ID."
        ),
        "systemic_involvement": True,
        "onset_age": "Infancy-early childhood (optic atrophy often first noted at fundus screening); variable",
        "surgical_urgency": "No acute surgical urgency",
        "gene_family": "Reticulon-interacting protein (mitochondrial matrix GTPase, UBL domain) — mitochondrial dynamics",
        "morphology": "Bilateral optic disc pallor; cerebellar atrophy on MRI (complex cases)",
    },
    # -- SLC25A46 -- Hereditary Motor Neuropathy VIB + Optic Atrophy + Cerebellar ----------------
    {
        "gene": "SLC25A46",
        "alt_name": (
            "SLC25A46 (SLC25A46-418aa-5q22.1 / AR -- "
            "HEREDITARY-MOTOR-NEUROPATHY-VIB-HMN-VIB-OPTIC-ATROPHY-CEREBELLAR-ATROPHY-TRIAD -- "
            "NEUROPATHY-PREDOMINANT-CMT-LIKE-MOTOR-GREATER-SENSORY-AXONAL -- "
            "OUTER-MITOCHONDRIAL-MEMBRANE-LIPID-TRANSFER-CARDIOLIPIN-PRECURSORS-DRP1-MFN -- "
            "PONTOCEREBELLAR-HYPOPLASIA-PCH1D-SEVERE-BIALLELIC-NULL-NEONATAL)"
        ),
        "protein": (
            "SLC25A46 -- 5q22.1 AR -- SLC25A46-418aa -- "
            "Solute-Carrier-Family-25-Member-46-47kDa-OMM-Lipid-Transfer -- "
            "Outer-Mitochondrial-Membrane-Interacts-DRP1-MFN1-MFN2-Cardiolipin-Precursor-Transfer -- "
            "OMIM-Gene-610826-Disease-HMN-VIB-616505-PCH1D-617015"
        ),
        "locus": "5q22.1",
        "protein_size": "418 aa / 47 kDa",
        "inheritance": (
            "AR (autosomal recessive); SLC25A46 encodes an outer mitochondrial membrane (OMM) protein; "
            "FUNCTION: phospholipid transfer from ER to OMM (cardiolipin precursors); "
            "interacts with DRP1 (dynamin-related protein 1) and mitofusins MFN1/MFN2 -> regulates mitochondrial fission/fusion; "
            "SLC25A46 LOF -> impaired OMM lipid composition -> abnormal mitochondrial dynamics -> "
            "fragmented mitochondria in peripheral neurons, optic nerve, cerebellum; "
            "PHENOTYPIC SPECTRUM: "
            "HMN-VIB: axonal peripheral motor > sensory neuropathy + optic atrophy + cerebellar atrophy; "
            "PCH1D: severe neonatal form; biallelic null -> pontocerebellar hypoplasia -> early death"
        ),
        "disease_category": "Hereditary motor neuropathy type VIB (HMN-VIB) + optic atrophy + cerebellar atrophy; AR; OMM lipid transfer + mitochondrial fission/fusion; neuropathy predominant phenotype",
        "disease_pathway": (
            "SLC25A46 is an outer mitochondrial membrane protein functioning as a lipid transfer factor. "
            "LIPID TRANSFER: SLC25A46 at ER-OMM contact sites transfers phospholipids "
            "(CDP-diacylglycerol -> PGP -> PG -> cardiolipin precursors) from ER to OMM; "
            "cardiolipin essential for IMM curvature, OXPHOS complex stability, cytochrome c binding, cristae; "
            "FISSION/FUSION: SLC25A46 interacts with DRP1 and MFN1/MFN2; "
            "SLC25A46 LOF -> impaired DRP1 recruitment and MFN function -> mitochondrial hyperfusion -> "
            "dysfunctional elongated mitochondria with impaired ATP production; "
            "SELECTIVE VULNERABILITY: peripheral motor > sensory axons -> CMT-like neuropathy; "
            "RGCs -> optic atrophy; cerebellar Purkinje cells -> cerebellar atrophy; "
            "PCH1D (severe): complete SLC25A46 loss -> failure of pontocerebellar development in utero."
        ),
        "pathognomonic": (
            "HMN-VIB / SLC25A46 COMPLEX TRIAD: "
            "1. PERIPHERAL NEUROPATHY (CMT-like): PREDOMINANT motor > sensory; axonal; "
            "EMG/NCS: reduced CMAP amplitudes; mild velocity slowing; denervation needle EMG; "
            "foot deformity (pes cavus), distal weakness, areflexia; "
            "2. OPTIC ATROPHY: bilateral disc pallor; temporal RNFL thinning; centrocecal scotoma; "
            "COMBINATION neuropathy + optic atrophy distinguishes from CMT alone; "
            "3. CEREBELLAR ATROPHY: MRI cerebellar hemispheric > vermal atrophy; gait ataxia; dysmetria; "
            "PCH1D (SEVERE NEONATAL): pontocerebellar hypoplasia + pontine hypoplasia + early death; "
            "DIAGNOSTIC: biallelic SLC25A46 + EMG + MRI + ophthalmology; "
            "Include SLC25A46 in hereditary neuropathy + optic atrophy panels (not standard CMT panels)."
        ),
        "treatment": (
            "PERIPHERAL NEUROPATHY: physiotherapy; AFO orthotics; pain management (gabapentin, duloxetine); "
            "powered wheelchair if severe; "
            "OPTIC ATROPHY: low vision aids; annual OCT + HVF; Ethambutol ABSOLUTE CI; Linezolid ABSOLUTE CI; "
            "CEREBELLAR: balance physio; OT; riluzole Level C; "
            "AVOID: Vincristine ABSOLUTE CI (axonal neuropathy crisis); Valproate HIGH RISK; "
            "aminoglycosides AVOID (neuropathy additive); excess vitamin B6 >500 mg/day (neuropathy risk); "
            "SUPPLEMENTS: CoQ10, riboflavin, carnitine Level C empiric; "
            "PCH1D: neonatal intensive care; palliative approach for severe neonatal form; "
            "GENETIC COUNSELLING: AR 25% risk; PCH1D biallelic null -> prenatal diagnosis."
        ),
        "key_features": [
            "Optic atrophy + CMT-like neuropathy + cerebellar atrophy TRIAD",
            "Neuropathy predominant: motor > sensory, axonal; pes cavus, foot drop",
            "SLC25A46: OMM lipid transfer (cardiolipin precursors) + DRP1/MFN regulation",
            "PCH1D: severe neonatal pontocerebellar hypoplasia (biallelic null)",
            "Vincristine ABSOLUTE CI (axonal neuropathy crisis)",
            "Ethambutol / Linezolid ABSOLUTE CI",
            "Include in hereditary neuropathy + optic atrophy gene panels",
            "Distinguishes from CMT: CMT lacks optic atrophy component",
        ],
        "key_ddx": (
            "RTN4IP1 (OPA10): AR; optic atrophy +/- cerebellar +/- ID; less neuropathy-predominant; "
            "ACO2: AR; OA + cerebellar + ID; TCA cycle; no predominant neuropathy; "
            "CMT2 (MFN2, GDAP1): neuropathy dominant; NO optic atrophy; check SLC25A46 if OA present; "
            "SANDO/POLG: mitochondrial neuropathy; mtDNA depletion; PEO; liver; "
            "ARSACS: cerebellar + neuropathy; Quebec founder; no optic atrophy."
        ),
        "systemic_involvement": True,
        "onset_age": "Childhood-early adulthood (HMN-VIB); neonatal (PCH1D severe form)",
        "surgical_urgency": "No acute surgical urgency; PCH1D may require neonatal intensive care",
        "gene_family": "SLC25 family (OMM lipid transfer) — mitochondrial fission/fusion regulation",
        "morphology": "Bilateral optic disc pallor; cerebellar atrophy on MRI; pes cavus/foot drop",
    },
]


def _make_cohort(entry: dict, seed: int) -> list:
    """Generate 40 synthetic patients for one gene cohort."""
    rng = random.Random(seed)
    patients = []
    gene = entry["gene"]
    for i in range(40):
        # Visual acuity loss (worse than 6/12)
        if gene == "OPA1":
            va_poor = rng.random() < 0.55
        elif gene == "OPA3":
            va_poor = rng.random() < 0.65
        elif gene == "WFS1":
            va_poor = rng.random() < 0.72
        elif gene == "CISD2":
            va_poor = rng.random() < 0.68
        elif gene == "TIMM8A":
            va_poor = rng.random() < 0.58
        elif gene == "ACO2":
            va_poor = rng.random() < 0.75
        elif gene == "RTN4IP1":
            va_poor = rng.random() < 0.65
        else:  # SLC25A46
            va_poor = rng.random() < 0.60

        # Colour vision defect
        if gene == "OPA1":
            colour_defect = "tritanopia"
        elif gene == "TIMM8A":
            colour_defect = "dyschromatopsia"
        elif gene in ("WFS1", "CISD2"):
            colour_defect = "dyschromatopsia" if rng.random() < 0.60 else "none"
        elif gene in ("OPA3", "ACO2", "RTN4IP1", "SLC25A46"):
            colour_defect = "dyschromatopsia" if rng.random() < 0.55 else "none"
        else:
            colour_defect = "none"

        # Sensorineural hearing loss (SNHL)
        if gene == "TIMM8A":
            snhl = True
        elif gene == "WFS1":
            snhl = rng.random() < 0.80
        elif gene == "CISD2":
            snhl = rng.random() < 0.75
        elif gene == "OPA1":
            snhl = rng.random() < 0.20
        elif gene in ("OPA3", "ACO2", "RTN4IP1", "SLC25A46"):
            snhl = rng.random() < 0.15
        else:
            snhl = False

        # Diabetes mellitus
        if gene in ("WFS1", "CISD2"):
            dm = rng.random() < 0.95
        else:
            dm = rng.random() < 0.03

        # Diabetes insipidus
        if gene == "WFS1":
            di = rng.random() < 0.75
        elif gene == "CISD2":
            di = False  # NO DI in WFS2 -- KEY DDx
        else:
            di = rng.random() < 0.03

        # Gastrointestinal bleeding
        if gene == "CISD2":
            gi_bleed = rng.random() < 0.85
        else:
            gi_bleed = rng.random() < 0.03

        # Cerebellar ataxia
        if gene in ("OPA3", "ACO2", "RTN4IP1", "SLC25A46"):
            cerebellar = rng.random() < 0.80
        elif gene == "TIMM8A":
            cerebellar = rng.random() < 0.30
        elif gene in ("WFS1", "CISD2"):
            cerebellar = rng.random() < 0.25
        else:
            cerebellar = rng.random() < 0.10

        # Peripheral neuropathy
        if gene == "SLC25A46":
            neuropathy = rng.random() < 0.90
        elif gene == "TIMM8A":
            neuropathy = rng.random() < 0.30
        elif gene == "RTN4IP1":
            neuropathy = rng.random() < 0.45
        elif gene in ("WFS1", "CISD2"):
            neuropathy = rng.random() < 0.30
        else:
            neuropathy = rng.random() < 0.10

        # Dystonia (Mohr-Tranebjaerg specific)
        if gene == "TIMM8A":
            dystonia = rng.random() < 0.85
        else:
            dystonia = rng.random() < 0.04

        # 3-methylglutaconic aciduria (OPA3 Costeff specific)
        mga3_uria = gene == "OPA3" and rng.random() < 0.90

        patients.append({
            "id": f"{gene}-{seed}-{i+1:03d}",
            "gene": gene,
            "va_poor": va_poor,
            "colour_defect": colour_defect,
            "snhl": snhl,
            "dm": dm,
            "di": di,
            "gi_bleed": gi_bleed,
            "cerebellar": cerebellar,
            "neuropathy": neuropathy,
            "dystonia": dystonia,
            "mga3_uria": mga3_uria,
            "drug_ci": "ETB/LZD absolute CI",
            "inheritance": entry["inheritance"].split(";")[0].strip(),
        })
    return patients


def generate_overview():
    all_patients = []
    for idx, entry in enumerate(ON_GENES):
        all_patients.extend(_make_cohort(entry, SEED_BASE + idx))

    total = len(all_patients)
    va_poor_count   = sum(1 for p in all_patients if p["va_poor"])
    snhl_count      = sum(1 for p in all_patients if p["snhl"])
    dm_count        = sum(1 for p in all_patients if p["dm"])
    di_count        = sum(1 for p in all_patients if p["di"])
    gi_bleed_count  = sum(1 for p in all_patients if p["gi_bleed"])
    cerebellar_count = sum(1 for p in all_patients if p["cerebellar"])
    neuropathy_count = sum(1 for p in all_patients if p["neuropathy"])
    dystonia_count  = sum(1 for p in all_patients if p["dystonia"])

    gene_summary = {}
    for idx, entry in enumerate(ON_GENES):
        gene = entry["gene"]
        cohort = _make_cohort(entry, SEED_BASE + idx)
        gene_summary[gene] = {
            "gene":              gene,
            "alt_name":          entry["alt_name"],
            "locus":             entry["locus"],
            "protein_size":      entry["protein_size"],
            "inheritance":       entry["inheritance"].split(";")[0].strip(),
            "disease_category":  entry["disease_category"],
            "pathognomonic":     entry["pathognomonic"][:300],
            "morphology":        entry["morphology"],
            "systemic_involvement": entry["systemic_involvement"],
            "onset_age":         entry["onset_age"],
            "surgical_urgency":  entry["surgical_urgency"],
            "gene_family":       entry["gene_family"],
            "n_patients":        len(cohort),
            "va_poor_pct":       round(100 * sum(1 for p in cohort if p["va_poor"])      / len(cohort), 1),
            "snhl_pct":          round(100 * sum(1 for p in cohort if p["snhl"])         / len(cohort), 1),
            "dm_pct":            round(100 * sum(1 for p in cohort if p["dm"])           / len(cohort), 1),
            "di_pct":            round(100 * sum(1 for p in cohort if p["di"])           / len(cohort), 1),
            "gi_bleed_pct":      round(100 * sum(1 for p in cohort if p["gi_bleed"])     / len(cohort), 1),
            "cerebellar_pct":    round(100 * sum(1 for p in cohort if p["cerebellar"])   / len(cohort), 1),
            "neuropathy_pct":    round(100 * sum(1 for p in cohort if p["neuropathy"])   / len(cohort), 1),
            "dystonia_pct":      round(100 * sum(1 for p in cohort if p["dystonia"])     / len(cohort), 1),
        }

    return {
        "atlas":          "Hereditary-Optic-Neuropathy-Atlas",
        "subtitle":       "Complete 8-Gene Hereditary Optic Neuropathy Reference -- OPA1/OPA3/WFS1/CISD2/TIMM8A/ACO2/RTN4IP1/SLC25A46",
        "genes_covered":  [e["gene"] for e in ON_GENES],
        "total_patients": total,
        "seeds":          f"{SEED_BASE}-{SEED_BASE + 7}",
        "aggregate_metrics": {
            "va_worse_than_6_12_pct": round(100 * va_poor_count    / total, 1),
            "snhl_pct":               round(100 * snhl_count        / total, 1),
            "dm_pct":                 round(100 * dm_count          / total, 1),
            "di_pct":                 round(100 * di_count          / total, 1),
            "gi_bleed_pct":           round(100 * gi_bleed_count    / total, 1),
            "cerebellar_pct":         round(100 * cerebellar_count  / total, 1),
            "neuropathy_pct":         round(100 * neuropathy_count  / total, 1),
            "dystonia_pct":           round(100 * dystonia_count    / total, 1),
        },
        "gene_summary": gene_summary,
    }


def generate_breakdown():
    breakdown = []
    for idx, entry in enumerate(ON_GENES):
        cohort = _make_cohort(entry, SEED_BASE + idx)
        breakdown.append({
            "gene":              entry["gene"],
            "alt_name":          entry["alt_name"],
            "locus":             entry["locus"],
            "protein_size":      entry["protein_size"],
            "inheritance":       entry["inheritance"].split(";")[0].strip(),
            "disease_category":  entry["disease_category"],
            "pathognomonic":     entry["pathognomonic"],
            "treatment":         entry["treatment"],
            "key_features":      entry["key_features"],
            "key_ddx":           entry["key_ddx"],
            "morphology":        entry["morphology"],
            "systemic_involvement": entry["systemic_involvement"],
            "onset_age":         entry["onset_age"],
            "surgical_urgency":  entry["surgical_urgency"],
            "gene_family":       entry["gene_family"],
            "n_patients":        len(cohort),
            "va_poor_pct":       round(100 * sum(1 for p in cohort if p["va_poor"])      / len(cohort), 1),
            "snhl_pct":          round(100 * sum(1 for p in cohort if p["snhl"])         / len(cohort), 1),
            "dm_pct":            round(100 * sum(1 for p in cohort if p["dm"])           / len(cohort), 1),
            "di_pct":            round(100 * sum(1 for p in cohort if p["di"])           / len(cohort), 1),
            "gi_bleed_pct":      round(100 * sum(1 for p in cohort if p["gi_bleed"])     / len(cohort), 1),
            "cerebellar_pct":    round(100 * sum(1 for p in cohort if p["cerebellar"])   / len(cohort), 1),
            "neuropathy_pct":    round(100 * sum(1 for p in cohort if p["neuropathy"])   / len(cohort), 1),
            "dystonia_pct":      round(100 * sum(1 for p in cohort if p["dystonia"])     / len(cohort), 1),
            "sample_patients":   cohort[:3],
        })
    return {"gene_breakdowns": breakdown}


def generate_definitions():
    return {
        "gene_entries": {
            entry["gene"]: {
                "gene":          entry["gene"],
                "full_name":     entry["protein"].split(" --")[0].strip(),
                "locus":         entry["locus"],
                "protein_size":  entry["protein_size"],
                "inheritance":   entry["inheritance"].split(";")[0].strip(),
                "disease_name":  entry["disease_category"],
                "disease_pathway": entry["disease_pathway"],
                "pathognomonic": entry["pathognomonic"],
                "treatment":     entry["treatment"][:600],
                "key_features":  entry["key_features"],
                "key_ddx":       entry["key_ddx"],
                "morphology":    entry["morphology"],
                "systemic_involvement": entry["systemic_involvement"],
                "onset_age":     entry["onset_age"],
                "surgical_urgency": entry["surgical_urgency"],
                "gene_family":   entry["gene_family"],
            }
            for entry in ON_GENES
        },
        "on_glossary": {
            "OPA1 vs LHON — Colour Vision as the Key DDx": (
                "COLOUR VISION IS THE FASTEST BEDSIDE DDx BETWEEN THE TWO MOST COMMON HEREDITARY OPTIC NEUROPATHIES: "
                "OPA1 (ADOA): TRITANOPIA -- blue-yellow (S-cone / tritan axis) colour vision defect; "
                "D-15 desaturated panel: tritan axis errors; Farnsworth-Munsell 100-hue: tritan cap; "
                "LHON (MT-ND4/ND1/ND6): RED-GREEN dyschromatopsia -- protan or deutan; "
                "Ishihara plates fail on red-green plates; "
                "OPA1: insidious onset; family history AD; tritanopia; temporal disc pallor; "
                "LHON: subacute painless; males 80-90%; peripapillary telangiectasia (acute phase); "
                "maternal inheritance; no family history in 60% (incomplete penetrance); "
                "ETHAMBUTOL ABSOLUTE CI IN BOTH: precipitates acute visual loss in both OPA1 and LHON."
            ),
            "WFS1 DIDMOAD — Optic Atrophy as First Sign": (
                "WOLFRAM SYNDROME (DIDMOAD) CLINICAL RULE: optic atrophy is the FIRST CLINICAL SIGN in most patients, "
                "appearing at mean age 5-8 years -- BEFORE diabetes mellitus (DM) becomes evident. "
                "IMPLICATION: a child with bilateral optic atrophy + DM without islet antibodies -> "
                "WOLFRAM SYNDROME UNTIL PROVEN OTHERWISE; WFS1 sequencing mandatory. "
                "ANTIBODY-NEGATIVE DM: GAD65, IA2, ZnT8 antibodies NEGATIVE in WFS1 -- "
                "WFS1 DM is NOT autoimmune; insulin-dependent; "
                "DIDMOAD SEQUENCE: OA (mean 6y) -> DM (mean 6-10y) -> DI (variable) -> Deafness (variable); "
                "COMPARE WFS2 (CISD2): GI BLEEDING ADDED; DI ABSENT; same DM + OA + SNHL but iron-ROS pathway."
            ),
            "CISD2 WFS2 — GI Bleeding as the Distinguishing Feature": (
                "WOLFRAM TYPE 2 (WFS2 / CISD2): GASTROINTESTINAL BLEEDING is the pathognomonic distinguisher from WFS1. "
                "WFS1 (DIDMOAD): Diabetes Insipidus PRESENT; NO GI bleeding; ER Ca2+/UPR pathway; "
                "WFS2 (CISD2): GI BLEEDING PRESENT; Diabetes Insipidus ABSENT; mitochondrial iron/ROS pathway; "
                "ALGORITHM: Wolfram + GI BLEEDING + NO DI -> WFS2/CISD2; Wolfram + DI + NO GI bleeding -> WFS1; "
                "GI ENDOSCOPY IN WFS2: peptic ulcer-like without H. pylori; PPI prophylaxis mandatory; "
                "AVOID NSAIDs (ulcerogenic); BEDOUIN ANCESTRY -> targeted p.Glu132Lys testing first."
            ),
            "TIMM8A MTS — Sequential Progression and Cochlear Implant Timing": (
                "MOHR-TRANEBJAERG SYNDROME (MTS) -- THREE KEY MANAGEMENT PRINCIPLES: "
                "1. SEQUENTIAL DIAGNOSIS: SNHL (childhood) -> Dystonia (adolescence) -> Optic neuronopathy (adulthood); "
                "NEVER diagnose MTS without SNHL as the FIRST feature; "
                "2. COCHLEAR IMPLANT TIMING -- CRITICAL: implant BEFORE dystonia develops; "
                "dystonia -> involuntary movements -> device programming impaired; "
                "establish communication (CI + signing) before speech motor pathways degrade; "
                "3. LEVODOPA TRIAL MANDATORY: ~10-15% DRD-like response; "
                "low-dose trial (2-3 mg/kg/day carbidopa-levodopa) 3 months; "
                "X-LINKED: only males fully affected; carrier female audiogram + neuro as baseline."
            ),
            "Ethambutol Absolute CI in All Hereditary Optic Neuropathies": (
                "ETHAMBUTOL IS ABSOLUTELY CONTRAINDICATED in ALL hereditary optic neuropathies -- "
                "regardless of gene (OPA1, OPA3, WFS1, CISD2, TIMM8A, ACO2, RTN4IP1, SLC25A46). "
                "MECHANISM: ethambutol chelates copper -> inhibits mitochondrial copper-dependent enzymes -> "
                "additional optic nerve bioenergetic failure on top of pre-existing genetic compromise -> "
                "ACUTE VISUAL CRISIS with irreversible loss; "
                "CLINICAL SCENARIO: patient with hereditary optic neuropathy admitted for TB -> "
                "ethambutol MUST be substituted; discuss with ID + ophthalmology + geneticist; "
                "LINEZOLID ABSOLUTE CI: mitochondrial ribosome inhibitor -> bioenergetic collapse; "
                "VINCRISTINE ABSOLUTE CI in SLC25A46 (axonal neuropathy crisis); "
                "AMIODARONE AVOID in OPA1; TOBACCO ABSOLUTE CI (CO/cyanide optic nerve); "
                "DOCUMENT in allergy/alert section of all electronic health records."
            ),
        },
    }


# API-compatible aliases (backend calls get_* variants)
get_overview    = generate_overview
get_breakdown   = generate_breakdown
get_definitions = generate_definitions
