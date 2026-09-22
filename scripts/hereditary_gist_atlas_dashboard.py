#!/usr/bin/env python3
"""Hereditary-GIST-Predisposition-Atlas — Complete 8-Gene Reference
KIT    (KIT Proto-Oncogene; 976aa; 4q12; AD GOF;
         Hereditary GIST type 1; Gain-of-function hotspot exons 8/9/11/13/17;
         Imatinib 1L; exon 11 BEST imatinib response; exon 9 needs dose 800mg;
         KIT-positive IHC 95%+ GIST PATHOGNOMONIC-surrogate;
         seed SEED_BASE+0) ·
PDGFRA (Platelet-Derived Growth Factor Receptor Alpha; 1089aa; 4q12; AD GOF;
         Hereditary GIST type 2; D842V mutation PATHOGNOMONIC = imatinib-resistant;
         Avapritinib FDA-approved D842V mGIST 2020;
         Gastric GIST predominant; PDGFRA-D842V = EPITHELIOID morphology PATHOGNOMONIC;
         seed SEED_BASE+1) ·
SDHA   (Succinate Dehydrogenase Complex Subunit A; 664aa; 5p15.33; AR/AD LOF;
         SDH-deficient GIST; Carney Triad (non-hereditary);
         SDHB IHC loss PATHOGNOMONIC for ALL SDH-deficient GIST;
         Multifocal gastric GIST; multinodular; young females; lymph node metastasis;
         seed SEED_BASE+2) ·
SDHB   (Succinate Dehydrogenase Complex Subunit B; 280aa; 1p36.13; AD LOF;
         Carney-Stratakis Syndrome (CSS); SDH-deficient GIST + Paraganglioma;
         Malignant GIST 30-50% HIGHEST SDH subunit;
         Methoxytyramine (plasma) + DOTATATE PET for PGL surveillance;
         seed SEED_BASE+3) ·
SDHC   (Succinate Dehydrogenase Complex Subunit C; 169aa; 1q23.3; AD LOF;
         PGL3/Carney-Stratakis; SDH-deficient GIST + Head-Neck PGL;
         Low malignancy rate 1-3%; NOT imprinted (both alleles active);
         HNPGL predominant; gastric GIST component in Carney-Stratakis;
         seed SEED_BASE+4) ·
SDHD   (Succinate Dehydrogenase Complex Subunit D; 160aa; 11q23.1; AD LOF;
         PGL1; PATERNAL IMPRINTING — maternal carriers NOT at risk;
         Head-Neck PGL dominant; GIST component rare;
         Multilocal HNPGL; young onset; tinnitus/pulsatile mass FIRST symptom;
         seed SEED_BASE+5) ·
NF1    (Neurofibromin; 2839aa; 17q11.2; AD LOF;
         Neurofibromatosis Type 1; NF1-GIST = MULTIFOCAL SMALL BOWEL PATHOGNOMONIC;
         NF1-GIST: KIT-neg/PDGFRA-neg/SDH-intact; imatinib POOR response;
         MEK inhibitor (binimetinib/selumetinib) active in NF1;
         seed SEED_BASE+6) ·
MAX    (MYC Associated Factor X; 160aa; 14q23.3; AD LOF;
         PGL5; PATERNAL IMPRINTING — maternal carriers NOT at risk;
         Bilateral adrenal pheochromocytoma; adrenaline-secreting;
         GIST component rare; metanephrine/normetanephrine surveillance;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 x 40, seeds 3198-3205)
"""
import random

SEED_BASE = 3198

ATLAS_GENES = [
    {
        "gene": "KIT",
        "protein": (
            "KIT -- 4q12 Autosomal-Dominant-GOF -- 976aa -- "
            "KIT-110kDa-Type-III-RTK-HGIST1-Exon11-Best-Imatinib-Exon9-800mg-"
            "KIT-IHC-95pct-GIST-PATHOGNOMONIC-Surrogate-OMIM-164920"
        ),
        "locus": "4q12",
        "protein_size": (
            "976 aa / 110 kDa / 4q12 KIT encodes KIT Proto-Oncogene Receptor Tyrosine Kinase: "
            "STRUCTURE: "
            "  976 aa / 110 kDa; Type III receptor tyrosine kinase (RTK); "
            "  Signal peptide (aa 1-21); extracellular domain: 5 Ig-like domains (aa 22-519); "
            "  Transmembrane domain (aa 521-543); Juxtamembrane (JM) domain (aa 544-580); "
            "  Split kinase domain (KD1: aa 581-665; KD2: aa 713-934) with kinase insert (aa 666-712); "
            "  KIT activating hotspots: "
            "    Exon 11 (JM): most common (~70% sporadic GIST); best imatinib response; "
            "    Exon 9 (EC-D5): ~10% sporadic GIST; needs imatinib 800mg/day (vs 400mg exon 11); "
            "    Exon 13 (KD1): ~1%; intermediate imatinib response; "
            "    Exon 17 (KD2 activation loop): ~1%; imatinib-resistant; ripretinib active; "
            "  Hereditary GIST: germline KIT exon 11/13/17 GOF → multiple GIST + ICC hyperplasia; "
            "KIT IHC — PATHOGNOMONIC SURROGATE: "
            "  KIT (CD117) IHC positive in >95% GIST = PATHOGNOMONIC surrogate for GIST diagnosis; "
            "  KIT-negative GIST (5%): PDGFRA-mutant (esp. D842V) or SDH-deficient; "
            "  DOG1 (ANO1/TMEM16A) IHC: 95%+ GIST sensitivity, MORE specific than KIT; "
            "  KIT + DOG1 dual IHC: GIST diagnostic standard; "
            "IMATINIB (1L ADVANCED GIST): "
            "  Imatinib 400mg/day: 1L metastatic/unresectable KIT-exon11-GIST; "
            "  Imatinib 800mg/day: KIT-exon9-GIST (EORTC trial confirmed); "
            "  PDGFRA-D842V: IMATINIB-RESISTANT → avapritinib; "
            "  KIT-exon17/18: IMATINIB-RESISTANT → ripretinib or avapritinib; "
            "  Adjuvant imatinib: 3yr for high-risk resected GIST (PERSIST-5); "
            "SUNITINIB (2L): "
            "  Sunitinib: 2L after imatinib failure/intolerance; "
            "  Active in KIT exon 13/14 secondary mutations; "
            "REGORAFENIB (3L) + RIPRETINIB (4L+): "
            "  Regorafenib: 3L; ripretinib: 4L+ (switch control mechanism); "
            "HEREDITARY GIST TYPE 1 (KIT): "
            "  Germline KIT mutation → multiple GIST (gastric predominant) + diffuse ICC hyperplasia; "
            "  Urticaria pigmentosa (mastocytosis component in some kindreds); "
            "  Autosomal dominant; high penetrance for GIST; "
            "  Surveillance: annual abdominal MRI/CT from age 20yr; "
            "MUTATIONAL TESTING MANDATORY: "
            "  KIT + PDGFRA sequencing MANDATORY for all primary resected GIST > 2cm; "
            "  Guides adjuvant and palliative therapy selection"
        ),
        "inheritance": "Autosomal Dominant (AD); germline GOF; de novo rare; KIT exon 11/13/17 most common; autosomal dominant high penetrance GIST; family cascade mandatory",
        "cancer_risk": "GIST (gastric/small bowel): near-universal in hereditary KIT; malignant risk proportional to mitotic index; multiple primary GISTs; ICC hyperplasia",
        "pathognomonic": "KIT (CD117) IHC positive 95%+ GIST = PATHOGNOMONIC SURROGATE; DOG1 IHC: more specific; KIT-exon-9-GIST needs imatinib 800mg (not 400mg) — critical dosing pitfall",
        "surveillance_key": "Annual abdominal MRI/CT from age 20yr; KIT+PDGFRA sequencing mandatory all primary GIST; imatinib 400mg exon 11 / 800mg exon 9; ripretinib for exon 17/18 secondary resistance",
        "key_distinctions": [
            "KIT-IHC-95PCT-GIST-PATHOGNOMONIC-SURROGATE",
            "DOG1-IHC-MORE-SPECIFIC-THAN-KIT",
            "EXON-9-IMATINIB-800MG-NOT-400MG",
            "EXON-11-BEST-IMATINIB-RESPONSE",
            "EXON-17-IMATINIB-RESISTANT-RIPRETINIB",
            "HEREDITARY-GIST-MULTIPLE-GASTRIC-ICC-HYPERPLASIA",
        ],
    },
    {
        "gene": "PDGFRA",
        "protein": (
            "PDGFRA -- 4q12 Autosomal-Dominant-GOF -- 1089aa -- "
            "PDGFRA-122kDa-Type-III-RTK-HGIST2-D842V-IMATINIB-RESISTANT-PATHOGNOMONIC-"
            "Avapritinib-FDA2020-Gastric-GIST-Epithelioid-OMIM-173490"
        ),
        "locus": "4q12",
        "protein_size": (
            "1089 aa / 122 kDa / 4q12 PDGFRA encodes Platelet-Derived Growth Factor Receptor Alpha: "
            "STRUCTURE: "
            "  1089 aa / 122 kDa; Type III RTK; same family as KIT; "
            "  Extracellular 5 Ig-like domains; transmembrane; JM + split kinase domains; "
            "  KIT and PDGFRA are neighbours on chromosome 4q12 (common amplification); "
            "  PDGFRA GOF hotspots: "
            "    Exon 18 (activation loop): D842V (~5-6% sporadic GIST) — PATHOGNOMONIC; "
            "    Exon 12 (JM): ~1% sporadic GIST; imatinib-sensitive; "
            "    Exon 14 (KD1): rare; imatinib-resistant; "
            "PDGFRA D842V — PATHOGNOMONIC + IMATINIB-RESISTANT: "
            "  D842V (p.Asp842Val) = PATHOGNOMONIC for PDGFRA-GIST; "
            "  Imatinib INTRINSICALLY RESISTANT (D842V steric block of imatinib binding); "
            "  Avapritinib (BLU-285): type I kinase inhibitor, overcomes D842V steric resistance; "
            "  NAVIGATOR trial: avapritinib PDGFRA-D842V mGIST → ORR 91%, DOR 27.6mo (BEST ever GIST trial); "
            "  FDA approved avapritinib Feb 2020 for PDGFRA-D842V mGIST; "
            "PDGFRA-D842V GIST MORPHOLOGY — PATHOGNOMONIC: "
            "  Epithelioid or mixed epithelioid/spindle morphology: PATHOGNOMONIC for PDGFRA-D842V; "
            "  Location: gastric predominant (vs small bowel KIT); "
            "  Myxoid stroma common in PDGFRA-GIST; "
            "  KIT IHC: often NEGATIVE or weak (unlike KIT-mutant GIST which is strongly positive); "
            "HEREDITARY GIST TYPE 2 (PDGFRA): "
            "  Germline PDGFRA GOF → multiple gastric GIST + ICC hyperplasia + fibrous plaques; "
            "  Rare hereditary syndrome; autosomal dominant; "
            "  Lipoma component in some kindreds; "
            "PDGFRA IHC: "
            "  PDGFRA IHC positive in PDGFRA-mutant GIST (confirms receptor expression); "
            "  KIT-negative + DOG1-positive + epithelioid gastric GIST → test PDGFRA exon 18 D842V; "
            "  PDGFRA D842V mutation can be confirmed by rapid allele-specific PCR or NGS"
        ),
        "inheritance": "Autosomal Dominant (AD); germline GOF; de novo; PDGFRA exon 18 D842V most common; hereditary GIST type 2; family cascade mandatory",
        "cancer_risk": "GIST (gastric predominant): near-universal hereditary PDGFRA; epithelioid morphology; lower malignancy rate than KIT-exon9/11; ICC hyperplasia; fibrous GI plaques",
        "pathognomonic": "PDGFRA D842V = PATHOGNOMONIC IMATINIB RESISTANCE; epithelioid/myxoid gastric GIST + KIT-weak/negative = PDGFRA D842V until proven otherwise; NAVIGATOR 91% ORR avapritinib",
        "surveillance_key": "Annual abdominal MRI from age 20yr; avapritinib FDA-approved D842V (imatinib-resistant); PDGFRA exon 12 = imatinib-sensitive (unlike D842V); DOG1+KIT-weak+gastric+epithelioid → PDGFRA D842V test",
        "key_distinctions": [
            "D842V-IMATINIB-RESISTANT-PATHOGNOMONIC",
            "AVAPRITINIB-FDA2020-91PCT-ORR-NAVIGATOR",
            "EPITHELIOID-MYXOID-GASTRIC-PATHOGNOMONIC",
            "KIT-IHC-WEAK-NEGATIVE-PDGFRA-GIST",
            "HEREDITARY-GIST-TYPE2-FIBROUS-PLAQUES",
        ],
    },
    {
        "gene": "SDHA",
        "protein": (
            "SDHA -- 5p15.33 Autosomal-Recessive-and-Dominant-LOF -- 664aa -- "
            "SDHA-70kDa-Flavoprotein-CII-Catalytic-SDH-Deficient-GIST-"
            "SDHB-IHC-Loss-PATHOGNOMONIC-Carney-Triad-Multifocal-Gastric-OMIM-600857"
        ),
        "locus": "5p15.33",
        "protein_size": (
            "664 aa / 70 kDa / 5p15.33 SDHA encodes Succinate Dehydrogenase Flavoprotein Subunit A: "
            "STRUCTURE: "
            "  664 aa / 70 kDa; largest SDH subunit; 70kDa flavoprotein; "
            "  FAD-binding domain catalyzes succinate → fumarate; "
            "  Forms heterotetramer with SDHB (Fe-S subunit) + SDHC + SDHD (membrane anchors); "
            "  SDH = Complex II (succinate:ubiquinone oxidoreductase); "
            "  TCA cycle + respiratory chain intersection; "
            "  SDHA LOF: loss of FAD-binding → entire SDH complex destabilises → SDHB degraded; "
            "  SDHB IHC: surrogate for ALL SDH-deficient GIST (SDHB degrades regardless of subunit lost); "
            "SDHB IHC LOSS — PATHOGNOMONIC FOR ALL SDH-DEFICIENT GIST: "
            "  SDHB IHC loss = PATHOGNOMONIC for SDH-deficient GIST regardless of which subunit is mutated; "
            "  SDHB is the 'canary' — degrades when any SDH subunit (A/B/C/D) is lost; "
            "  SDHA-specific IHC: SDHA-null GIST = SDHA-mutant (vs SDHB/C/D null = SDHA intact); "
            "  SDH-deficient GIST + SDHA-positive IHC = SDHB/C/D mutation; SDH-deficient + SDHA-negative = SDHA mutation; "
            "CARNEY TRIAD (SDHA-ASSOCIATED, NON-HERITABLE): "
            "  Carney Triad: GIST + Pulmonary chondroma + Extra-adrenal PGL (NOT hereditary); "
            "  Predominantly SDHA-deficient GIST; young females; no germline mutation; "
            "  Somatic SDH-deficiency via epigenetic SDHC methylation (Carney Triad); "
            "SDH-DEFICIENT GIST PHENOTYPE: "
            "  Multifocal gastric GIST: PATHOGNOMONIC for SDH-deficient GIST; "
            "  Multinodular plexiform growth pattern; lymph node metastasis common; "
            "  Young women (age <30yr); indolent despite metastasis; "
            "  KIT/PDGFRA wild-type; imatinib POOR response; "
            "  Temsirolimus (mTOR inhibitor) or cabozantinib: off-label active; "
            "SDHA GERMLINE — PARAGANGLIOMA LINK: "
            "  Germline SDHA LOF: GIST + PGL (Leigh syndrome biallelic — lethal neonatal); "
            "  SDHA PGL risk much lower than SDHB (SDHA PGL <5% penetrance); "
            "  SDHA is also linked to Leigh syndrome (mitochondrial respiratory chain)"
        ),
        "inheritance": "Autosomal Recessive (biallelic → Leigh syndrome); Autosomal Dominant monoallelic (GIST+PGL); Carney Triad is NOT hereditary (epigenetic somatic)",
        "cancer_risk": "GIST (gastric, multifocal): SDHA-deficient; PGL penetrance low (<5%); Carney Triad non-hereditary SDHA-deficient GIST in young women; indolent despite metastasis",
        "pathognomonic": "SDHB IHC loss = PATHOGNOMONIC for ALL SDH-deficient GIST (any subunit); SDHA IHC loss specifically identifies SDHA-mutant GIST; multifocal gastric GIST young female = SDH-deficient pattern",
        "surveillance_key": "SDHB IHC on all GIST (not just KIT/PDGFRA-WT) — detect SDH-deficient; SDHA vs SDHB/C/D IHC to identify specific subunit; annual abdominal MRI; mTOR/cabozantinib for SDH-deficient imatinib-resistant GIST",
        "key_distinctions": [
            "SDHB-IHC-LOSS-PATHOGNOMONIC-ALL-SDH-DEFICIENT-GIST",
            "SDHA-IHC-IDENTIFIES-SDHA-SPECIFIC-LOSS",
            "CARNEY-TRIAD-NOT-HEREDITARY-EPIGENETIC-SDHC",
            "MULTIFOCAL-GASTRIC-GIST-YOUNG-FEMALE-SDH",
            "IMATINIB-POOR-RESPONSE-SDH-DEFICIENT",
            "LEIGH-SYNDROME-BIALLELIC-SDHA-LETHAL",
        ],
    },
    {
        "gene": "SDHB",
        "protein": (
            "SDHB -- 1p36.13 Autosomal-Dominant-LOF -- 280aa -- "
            "SDHB-30kDa-Iron-Sulfur-Subunit-CSS-Carney-Stratakis-GIST-PGL-"
            "Malignant-30-50pct-HIGHEST-Methoxytyramine-DOTATATE-PET-OMIM-185470"
        ),
        "locus": "1p36.13",
        "protein_size": (
            "280 aa / 30 kDa / 1p36.13 SDHB encodes Succinate Dehydrogenase Iron-Sulfur Subunit B: "
            "STRUCTURE: "
            "  280 aa / 30 kDa; iron-sulfur (Fe-S) protein; "
            "  Three Fe-S clusters: [2Fe-2S], [4Fe-4S], [3Fe-4S]; "
            "  Transfers electrons from SDHA flavoprotein to SDHC/D membrane anchor → ubiquinone; "
            "  SDHB LOF: Fe-S cluster loss → entire SDH complex degradation (SDHB IHC loss) → SDHB IHC = null; "
            "  'Canary subunit' — SDHB degrades first when ANY SDH subunit is lost; "
            "CARNEY-STRATAKIS SYNDROME (CSS): "
            "  CSS = HEREDITARY GIST + Paraganglioma (AD germline SDHB/C/D); "
            "  SDHB: most common SDH gene in CSS; "
            "  GIST component: gastric/multifocal, SDH-deficient; "
            "  PGL component: extra-adrenal (retroperitoneal, bladder, thoracic) dominant; "
            "SDHB MALIGNANCY RISK — HIGHEST: "
            "  Malignant PGL: 30-50% in SDHB carriers = HIGHEST of all SDH genes; "
            "  vs SDHD: 0-5% malignancy; SDHC: 1-3% malignancy; "
            "  SDHB-PGL malignancy: bone, liver, lung metastases; "
            "METHOXYTYRAMINE — SDHB BIOMARKER: "
            "  Plasma methoxytyramine (MN): elevated in SDHB-PGL (dopaminergic secretion) = SDHB signature; "
            "  Most SDHB-PGL are biochemically non-secreting or dopamine-secreting (no classic hypertension); "
            "  24hr urine catecholamines often normal in SDHB — METHOXYTYRAMINE is the key biomarker; "
            "DOTATATE PET — SDHB PREFERRED: "
            "  68Ga-DOTATATE PET/CT: preferred functional imaging for SDH-deficient PGL/GIST; "
            "  DOTATATE superior to MIBG for SDHB-PGL (somatostatin receptor overexpression); "
            "  Annual imaging: abdominal MRI/CT + DOTATATE PET from age 15yr; "
            "SDHB FOUNDER MUTATIONS: "
            "  c.423+1G>A: common European SDHB splice donor mutation; "
            "  p.Pro197Arg: South African SDHB founder; "
            "  PGL1 (SDHD) vs PGL4 (SDHB) distinction: imprinting vs no-imprinting respectively"
        ),
        "inheritance": "Autosomal Dominant (AD); single-hit LOF; PGL4 nomenclature; no imprinting — both parental alleles contribute cancer risk; founder: c.423+1G>A European",
        "cancer_risk": "PGL (extra-adrenal PGL4): 30-50% malignancy HIGHEST; GIST (Carney-Stratakis SDH-deficient); Renal cell carcinoma (clear cell-like); pituitary adenoma (rare)",
        "pathognomonic": "SDHB IHC loss PATHOGNOMONIC for SDH-deficient GIST/PGL; plasma methoxytyramine elevated = SDHB-PGL signature (dopaminergic); malignancy 30-50% = HIGHEST SDH gene risk",
        "surveillance_key": "Annual DOTATATE PET/CT from age 15yr; plasma methoxytyramine (not just metanephrines); annual abdominal MRI; SDHB = highest malignant PGL risk; 3-monthly cathechol/methoxytyramine if metastatic suspected",
        "key_distinctions": [
            "MALIGNANT-PGL-30-50PCT-HIGHEST-SDH-GENE",
            "METHOXYTYRAMINE-ELEVATED-SDHB-SIGNATURE",
            "DOTATATE-PET-PREFERRED-SDHB",
            "PGL4-NO-IMPRINTING-BOTH-ALLELES-RISK",
            "CARNEY-STRATAKIS-CSS-GIST-PLUS-PGL",
            "SDHB-CANARY-SUBUNIT-SDHB-IHC-LOSS",
        ],
    },
    {
        "gene": "SDHC",
        "protein": (
            "SDHC -- 1q23.3 Autosomal-Dominant-LOF -- 169aa -- "
            "SDHC-15kDa-CII-Cytochrome-b-Large-Subunit-PGL3-HNPGL-Low-Malignancy-1-3pct-"
            "NOT-Imprinted-Carney-Stratakis-GIST-10pct-OMIM-602413"
        ),
        "locus": "1q23.3",
        "protein_size": (
            "169 aa / 15 kDa / 1q23.3 SDHC encodes Succinate Dehydrogenase Cytochrome b Large Subunit: "
            "STRUCTURE: "
            "  169 aa / 15 kDa; membrane anchor subunit (large); 3 transmembrane helices; "
            "  Forms heterodimer with SDHD (small membrane anchor) to anchor SDHAB to inner mitochondrial membrane; "
            "  Ubiquinone (coenzyme Q) binding site is at SDHC-SDHD interface; "
            "  SDHC LOF → SDH complex detaches from membrane → SDHB degradation → SDHB IHC loss; "
            "PGL3 SYNDROME: "
            "  SDHC germline LOF = PGL3; "
            "  Head-and-Neck PGL (HNPGL) dominant phenotype; "
            "  Carotid body most common; jugulotympanic, vagal PGL; "
            "CARNEY-STRATAKIS GIST COMPONENT: "
            "  ~10% SDHC carriers develop GIST component; "
            "  SDH-deficient GIST (SDHB IHC loss); gastric; multifocal; "
            "  Lower GIST penetrance than SDHB; "
            "LOW MALIGNANCY — DISTINCTIVE: "
            "  SDHC PGL malignancy: 1-3% = LOW (vs SDHB 30-50%); "
            "  SDHC PGL are predominantly benign chromaffin-cell tumours; "
            "  Annual MRI surveillance from age 15yr (HNPGL growth monitoring); "
            "NOT IMPRINTED: "
            "  SDHC is NOT imprinted — both parental alleles contribute cancer risk; "
            "  vs SDHD (paternal imprinting) and MAX (paternal imprinting); "
            "  Family history from EITHER parent is relevant for SDHC risk assessment; "
            "BIOCHEMISTRY (SDHC): "
            "  Mostly non-secreting HNPGL; normetanephrine may be mildly elevated; "
            "  Chromogranin A: may be elevated in paraganglioma; "
            "  DOTATATE PET: preferred for HNPGL localisation"
        ),
        "inheritance": "Autosomal Dominant (AD); single-hit LOF; NOT imprinted (both parental alleles active); PGL3 nomenclature; GIST component ~10% CSS penetrance",
        "cancer_risk": "HNPGL (carotid body/jugulotympanic/vagal): malignancy only 1-3% = LOW; GIST SDH-deficient ~10% CSS penetrance; mostly benign paraganglioma",
        "pathognomonic": "SDHB IHC loss PATHOGNOMONIC; SDHC LOF: malignancy 1-3% (lowest SDH malignancy risk); NOT imprinted (unlike SDHD/MAX); HNPGL = dominant phenotype",
        "surveillance_key": "Annual HNPGL surveillance: neck MRI from age 15yr; DOTATATE PET HNPGL; low malignancy risk but GIST surveillance abdominal MRI annual; NOT imprinting so maternal FHx also relevant",
        "key_distinctions": [
            "MALIGNANCY-1-3PCT-LOWEST-SDH-GENE",
            "HNPGL-DOMINANT-NOT-RETROPERITONEAL",
            "NOT-IMPRINTED-BOTH-ALLELES-RISK",
            "CARNEY-STRATAKIS-GIST-10PCT-PENETRANCE",
            "SDHB-IHC-LOSS-PATHOGNOMONIC",
        ],
    },
    {
        "gene": "SDHD",
        "protein": (
            "SDHD -- 11q23.1 Autosomal-Dominant-LOF -- 160aa -- "
            "SDHD-12kDa-CII-Cytochrome-b-Small-Subunit-PGL1-PATERNAL-IMPRINTING-"
            "MATERNAL-CARRIERS-NOT-AT-RISK-HNPGL-Multilocal-OMIM-602690"
        ),
        "locus": "11q23.1",
        "protein_size": (
            "160 aa / 12 kDa / 11q23.1 SDHD encodes Succinate Dehydrogenase Cytochrome b Small Subunit: "
            "STRUCTURE: "
            "  160 aa / 12 kDa; membrane anchor (small subunit); 3 transmembrane helices; "
            "  Partners with SDHC to anchor SDHAB complex; "
            "  Ubiquinone proton channel involvement; "
            "  SDHD LOF → SDH destabilisation → SDHB degradation → SDHB IHC loss; "
            "PATERNAL IMPRINTING — MOST CRITICAL CLINICAL POINT: "
            "  SDHD is paternally imprinted (maternal allele silenced in relevant tissues); "
            "  PATERNAL carriers: FULL PGL risk (maternally-transmitted silenced allele LOH → unifunctional); "
            "  MATERNAL carriers: NOT at risk for PGL (paternal-expressed allele intact); "
            "  Clinical rule: 'From father only does SDHD PGL come'; "
            "  Maternal SDHD carrier: reassure (very low risk); PATERNAL SDHD: full surveillance; "
            "  VHL and SDHD share this imprinting rule: paternal allele expressed in relevant tissue; "
            "PGL1 SYNDROME: "
            "  Multiple synchronous HNPGL (carotid body bilateral, jugulotympanic, vagal): PATHOGNOMONIC; "
            "  Young onset (mean ~35yr); "
            "  Tinnitus, pulsatile mass, pulsatile tinnitus: FIRST symptoms HNPGL; "
            "  Multilocal HNPGL = SDHD most common cause; "
            "MALIGNANCY (SDHD): "
            "  PGL malignancy: 0-5% = LOW (vs SDHB 30-50%); "
            "  SDHD-PGL are predominantly benign head-neck chromaffin tumours; "
            "BIOCHEMISTRY (SDHD): "
            "  Mostly non-secreting HNPGL; "
            "  Dopamine metabolites (HVA, VMA) elevated in some SDHD; "
            "  DOTATATE PET superior to MIBG for SDHD-HNPGL"
        ),
        "inheritance": "Autosomal Dominant with PATERNAL IMPRINTING; maternal carriers NOT at risk; paternal carriers FULL RISK; PGL1 nomenclature; multilocal HNPGL in paternal carriers",
        "cancer_risk": "HNPGL multilocal (carotid body bilateral/jugulotympanic/vagal): paternal carriers only; malignancy 0-5% LOW; GIST component rare (vs SDHB/SDHA)",
        "pathognomonic": "PATERNAL IMPRINTING — maternal carriers NOT at risk PATHOGNOMONIC concept; multilocal bilateral carotid body PGL in young patient = SDHD paternal inheritance; SDHB IHC loss on any SDH-deficient tissue",
        "surveillance_key": "PATERNAL carriers: full HNPGL surveillance from age 15yr; MATERNAL carriers: low-risk reassure; DOTATATE PET HNPGL; neck MRI annual; pulsatile tinnitus/neck mass = HNPGL until proven otherwise",
        "key_distinctions": [
            "PATERNAL-IMPRINTING-MATERNAL-NOT-AT-RISK",
            "MULTILOCAL-HNPGL-BILATERAL-CAROTID-BODY-PATHOGNOMONIC",
            "MALIGNANCY-0-5PCT-LOW",
            "PULSATILE-TINNITUS-FIRST-SYMPTOM-HNPGL",
            "PGL1-PATERNAL-ALLELE-EXPRESSED",
        ],
    },
    {
        "gene": "NF1",
        "protein": (
            "NF1 -- 17q11.2 Autosomal-Dominant-LOF -- 2839aa -- "
            "Neurofibromin-319kDa-RAS-GAP-NF1-GIST-MULTIFOCAL-SMALL-BOWEL-PATHOGNOMONIC-"
            "KIT-PDGFRA-WT-Imatinib-POOR-MEK-Binimetinib-Selumetinib-OMIM-162200"
        ),
        "locus": "17q11.2",
        "protein_size": (
            "2839 aa / 319 kDa / 17q11.2 NF1 encodes Neurofibromin: "
            "STRUCTURE: "
            "  2839 aa / 319 kDa; largest tumour suppressor protein; "
            "  RAS-GAP (GTPase Activating Protein) domain (GRD: aa 1198-1530); "
            "  GRD accelerates RAS GTPase activity (RAS-GTP → RAS-GDP → off); "
            "  NF1 LOF: RAS-GTP accumulation → uncontrolled MAPK/PI3K signalling; "
            "  Key features of neurofibromin: GRD, Sec14 domain, PH domain; "
            "NF1-GIST — MULTIFOCAL SMALL BOWEL PATHOGNOMONIC: "
            "  NF1-GIST location: small bowel (vs gastric for KIT/PDGFRA/SDH); "
            "  NF1-GIST is MULTIFOCAL in small bowel = PATHOGNOMONIC for NF1; "
            "  NF1-GIST: KIT-negative / PDGFRA-negative / SDH-intact = KIT/PDGFRA/SDH-WT GIST; "
            "  Typically low mitotic rate despite multiplicity; often discovered incidentally; "
            "  Incidence: ~5-7% NF1 patients develop GIST; "
            "IMATINIB POOR RESPONSE IN NF1-GIST: "
            "  NF1-GIST: KIT/PDGFRA-WT → imatinib POOR response (no targetable kinase); "
            "  NF1 drives GIST via MAPK not KIT/PDGFRA; "
            "  MEK inhibitor (binimetinib, selumetinib): active in NF1-GIST (RAS→MEK pathway); "
            "  SELUMETINIB: FDA-approved for NF1 PLEXIFORM NEUROFIBROMAS (paediatric); "
            "  Binimetinib: used off-label for NF1-GIST and NF1-MPNSTs; "
            "NF1 CAFÉ-AU-LAIT MACULES — PATHOGNOMONIC: "
            "  ≥6 CALMs >5mm prepubertal / >15mm postpubertal = NF1 NIH criteria; "
            "  Lisch nodules (iris hamartomas): NF1-PATHOGNOMONIC by slit lamp; "
            "  Axillary/inguinal freckling (Crowe's sign) PATHOGNOMONIC; "
            "  Plexiform neurofibromas + optic gliomas + bony dysplasia; "
            "MPNST — HIGHEST NF1 CANCER RISK: "
            "  Malignant peripheral nerve sheath tumour (MPNST): 10-15% lifetime NF1; "
            "  MPNST diagnosis: new pain in plexiform neurofibroma + FDG-PET avid; "
            "  MPNST: surgical resection; poor prognosis if unresectable"
        ),
        "inheritance": "Autosomal Dominant (AD); ~50% de novo; LOF; 17q11.2 microdeletion syndrome (segmental NF1); highly variable expressivity; two-hit model (somatic NF1 LOH in GIST)",
        "cancer_risk": "NF1-GIST (multifocal small bowel) 5-7% NF1; MPNST 10-15% HIGHEST NF1 cancer risk; OPG children <7yr; JMML/leukemia 500x RR children; learning disability 50-80%",
        "pathognomonic": "Multifocal small bowel GIST + NF1 stigmata = NF1-GIST PATHOGNOMONIC; KIT/PDGFRA/SDH-WT GIST + NF1 = NF1-GIST; CALMs ≥6 + Lisch nodules = NF1 PATHOGNOMONIC NIH criteria",
        "surveillance_key": "Annual full skin exam + ophtho (Lisch) from childhood; FDG-PET for MPNST in painful plexiform; selumetinib FDA for plexiform NF; annual abdominal MRI NF1-GIST; MEK inhibitor NF1-GIST imatinib-unresponsive",
        "key_distinctions": [
            "NF1-GIST-MULTIFOCAL-SMALL-BOWEL-PATHOGNOMONIC",
            "KIT-PDGFRA-SDH-WT-NF1-GIST",
            "IMATINIB-POOR-RESPONSE-NF1-GIST",
            "MEK-INHIBITOR-BINIMETINIB-SELUMETINIB-ACTIVE",
            "CALMS-LISCH-NODULES-NF1-PATHOGNOMONIC-NIH",
            "MPNST-10-15PCT-HIGHEST-NF1-CANCER",
        ],
    },
    {
        "gene": "MAX",
        "protein": (
            "MAX -- 14q23.3 Autosomal-Dominant-LOF -- 160aa -- "
            "MAX-160aa-MYC-Associated-Factor-X-bHLH-Leucine-Zipper-PGL5-"
            "PATERNAL-IMPRINTING-BILATERAL-ADRENAL-PHEO-Adrenaline-Secreting-OMIM-154950"
        ),
        "locus": "14q23.3",
        "protein_size": (
            "160 aa / 17 kDa / 14q23.3 MAX encodes MYC Associated Factor X: "
            "STRUCTURE: "
            "  160 aa / 17 kDa; bHLH-Zip (basic helix-loop-helix leucine zipper) transcription factor; "
            "  MAX is the central hub of the MYC network; "
            "  MAX heterodimerises with: MYC/MYCN (transcriptional activators), MXD family (repressors), MLXIP (MONDO); "
            "  MAX:MYC complex → proliferative gene transcription (E-box elements); "
            "  MAX:MXD complex → transcriptional repression → growth arrest; "
            "  MAX LOF: disrupts MYC:MXD balance → altered growth regulation; "
            "  Mechanism: MAX acts as obligate partner for MYC — MAX LOF may paradoxically reduce MYC target transcription; "
            "PATERNAL IMPRINTING (MAX): "
            "  MAX is PATERNALLY IMPRINTED (maternal allele silenced): "
            "  Same pattern as SDHD: paternal-only carriers express disease risk; "
            "  MATERNAL MAX carriers: NOT at risk (or very low risk); "
            "  PATERNAL MAX carriers: bilateral adrenal pheochromocytoma risk; "
            "  Clinical rule (same as SDHD): 'Only paternal MAX inheritance drives risk'; "
            "PGL5 / BILATERAL ADRENAL PHEOCHROMOCYTOMA: "
            "  MAX mutations: bilateral adrenal pheochromocytoma dominant (vs SDHD HNPGL); "
            "  Young onset (~30-35yr); bilateral synchronous or metachronous PHEO; "
            "  ADRENALINE-secreting (vs dopamine-secreting SDHB; vs noradrenaline-secreting most others); "
            "  Biochemistry: elevated METANEPHRINE (adrenaline metabolite) = MAX signature; "
            "  GIST component: rare but reported in MAX-CSS; "
            "SURVEILLANCE (MAX): "
            "  PATERNAL carriers: annual plasma metanephrines (fractionated); "
            "  Adrenal MRI annually from age 15yr; "
            "  DOTATATE PET if suspect PGL beyond adrenals; "
            "DIFFERENTIAL — SDHD vs MAX: "
            "  Both PATERNALLY imprinted; both low malignancy; "
            "  SDHD: HNPGL dominant; dopamine-secreting; "
            "  MAX: bilateral adrenal PHEO dominant; ADRENALINE-secreting = key distinguisher"
        ),
        "inheritance": "Autosomal Dominant with PATERNAL IMPRINTING; maternal carriers NOT at risk; paternal carriers: bilateral adrenal PHEO risk; PGL5 nomenclature; rare",
        "cancer_risk": "Bilateral adrenal pheochromocytoma: paternal carriers; adrenaline-secreting; malignancy low 5-10%; GIST component rare; pituitary adenoma rare",
        "pathognomonic": "PATERNAL IMPRINTING — maternal carriers NOT at risk (same as SDHD); bilateral adrenal PHEO + adrenaline-secreting + young = MAX PATHOGNOMONIC pattern; elevated metanephrine (vs normetanephrine in SDHB)",
        "surveillance_key": "PATERNAL carriers: annual plasma metanephrines (METANEPHRINE elevated = MAX signature); adrenal MRI annually from age 15yr; maternal carriers: reassure; DDx from SDHD: HNPGL vs bilateral adrenal PHEO",
        "key_distinctions": [
            "PATERNAL-IMPRINTING-MATERNAL-NOT-AT-RISK",
            "BILATERAL-ADRENAL-PHEO-DOMINANT-MAX",
            "ADRENALINE-SECRETING-METANEPHRINE-ELEVATED",
            "DDX-SDHD-HNPGL-VS-MAX-BILATERAL-ADRENAL",
            "PGL5-YOUNG-BILATERAL-SYNCHRONOUS-PHEO",
        ],
    },
]


def _make_patients(gene_entry):
    """Deterministic synthetic cohort: 40 patients per gene."""
    seed = SEED_BASE + ATLAS_GENES.index(gene_entry)
    rng  = random.Random(seed)

    gene = gene_entry["gene"]
    # Age-of-onset distributions per gene
    age_params = {
        "KIT":    (45, 12),
        "PDGFRA": (50, 12),
        "SDHA":   (28, 10),   # young females, Carney Triad
        "SDHB":   (35, 12),
        "SDHC":   (42, 12),
        "SDHD":   (35, 10),
        "NF1":    (40, 12),
        "MAX":    (32, 10),   # young bilateral PHEO
    }
    mu, sigma = age_params.get(gene, (40, 12))

    # Malignant or severe event rates
    severe_rates = {
        "KIT":    0.68,  # GIST in hereditary KIT: high penetrance
        "PDGFRA": 0.55,  # D842V GIST: high penetrance
        "SDHA":   0.62,  # SDH-deficient GIST: Carney Triad + CSS
        "SDHB":   0.72,  # HIGHEST malignancy CSS; GIST+PGL
        "SDHC":   0.42,  # HNPGL mostly benign; GIST 10%
        "SDHD":   0.48,  # HNPGL paternal carriers; multi-focal
        "NF1":    0.58,  # GIST + MPNST
        "MAX":    0.52,  # Bilateral PHEO; adrenaline-secreting
    }
    sev_rate = severe_rates.get(gene, 0.5)

    patients = []
    for i in range(40):
        age       = max(5, round(rng.gauss(mu, sigma), 1))
        sev_event = rng.random() < sev_rate
        patients.append({
            "id":        f"{gene}-{i+1:02d}",
            "age_onset": age,
            "severe":    sev_event,
            "seed":      seed,
        })
    return patients


def generate_overview():
    rows = []
    for g in ATLAS_GENES:
        pts   = _make_patients(g)
        sev_n = sum(1 for p in pts if p["severe"])
        rows.append({
            "gene":             g["gene"],
            "locus":            g["locus"],
            "n":                len(pts),
            "severe_n":         sev_n,
            "severe_pct":       round(sev_n / len(pts) * 100, 1),
            "mean_age_onset":   round(sum(p["age_onset"] for p in pts) / len(pts), 1),
            "pathognomonic":    g["pathognomonic"],
            "key_distinctions": g["key_distinctions"],
            "surveillance_key": g["surveillance_key"],
            "inheritance":      g["inheritance"],
            "cancer_risk":      g["cancer_risk"],
            "protein":          g["protein"],
        })

    total_pts  = sum(r["n"]       for r in rows)
    total_sev  = sum(r["severe_n"] for r in rows)
    highest    = max(rows, key=lambda r: r["severe_pct"])

    return {
        "atlas":              "Hereditary-GIST-Predisposition-Atlas",
        "seed_range":         f"{SEED_BASE}-{SEED_BASE + 7}",
        "genes_n":            len(ATLAS_GENES),
        "total_patients":     total_pts,
        "severe_total_n":     total_sev,
        "severe_total_pct":   round(total_sev / total_pts * 100, 1),
        "highest_risk_gene":  highest["gene"],
        "highest_risk_pct":   highest["severe_pct"],
        "gene_summary":       rows,
        "genes_detail": [
            {
                "gene":            g["gene"],
                "inheritance":     g["inheritance"],
                "cancer_risk":     g["cancer_risk"],
                "pathognomonic":   g["pathognomonic"],
                "surveillance_key": g["surveillance_key"],
            }
            for g in ATLAS_GENES
        ],
    }


def generate_breakdown():
    breakdown = []
    for g in ATLAS_GENES:
        pts      = _make_patients(g)
        seed_idx = ATLAS_GENES.index(g)
        sev_n    = sum(1 for p in pts if p["severe"])
        breakdown.append({
            "gene":             g["gene"],
            "locus":            g["locus"],
            "n":                len(pts),
            "seed":             SEED_BASE + seed_idx,
            "mean_age_onset":   round(sum(p["age_onset"] for p in pts) / len(pts), 1),
            "severe_n":         sev_n,
            "severe_pct":       round(sev_n / len(pts) * 100, 1),
            "pathognomonic":    g["pathognomonic"],
            "key_distinctions": g["key_distinctions"],
            "surveillance_key": g["surveillance_key"],
        })
    return {"atlas": "Hereditary-GIST-Predisposition-Atlas", "breakdown": breakdown}


def generate_definitions():
    defs = [
        {
            "term": "KIT / HGIST1 / KIT-IHC-95PCT-PATHOGNOMONIC / EXON9-800MG / EXON11-BEST-RESPONSE",
            "definition": (
                "KIT — 976aa / 110 kDa / 4q12 / AD GOF\n"
                "Hereditary GIST type 1; KIT IHC 95% PATHOGNOMONIC; exon 9 needs 800mg imatinib.\n\n"
                "KIT IHC — PATHOGNOMONIC SURROGATE:\n"
                "  KIT (CD117) IHC positive >95% GIST = PATHOGNOMONIC surrogate for GIST diagnosis.\n"
                "  DOG1 (ANO1) IHC: >95% sensitivity, MORE specific than KIT — use both.\n"
                "  KIT-negative GIST (5%): test PDGFRA (D842V) + SDHB IHC (SDH-deficient) + NF1 clinical.\n\n"
                "EXON 9 vs EXON 11 — CRITICAL DOSING:\n"
                "  Exon 11 (~70% sporadic GIST): imatinib 400mg/day; BEST response.\n"
                "  Exon 9 (~10% sporadic GIST, small bowel): imatinib 800mg/day MANDATORY.\n"
                "    → 400mg exon-9 GIST = undertreated; ALWAYS check mutation before dosing.\n"
                "  Exon 17/18: IMATINIB RESISTANT → ripretinib or avapritinib.\n\n"
                "THERAPEUTIC SEQUENCE (KIT-GIST):\n"
                "  1L: Imatinib (400mg exon11; 800mg exon9; avoid if exon17/18).\n"
                "  2L: Sunitinib.\n"
                "  3L: Regorafenib.\n"
                "  4L+: Ripretinib.\n"
                "  Adjuvant: 3yr imatinib high-risk resected GIST (PERSIST-5)."
            ),
        },
        {
            "term": "PDGFRA / D842V-IMATINIB-RESISTANT-PATHOGNOMONIC / AVAPRITINIB-FDA2020 / EPITHELIOID-GASTRIC",
            "definition": (
                "PDGFRA — 1089aa / 122 kDa / 4q12 / AD GOF\n"
                "PDGFRA D842V = PATHOGNOMONIC imatinib resistance; avapritinib FDA 2020; 91% ORR NAVIGATOR.\n\n"
                "D842V — PATHOGNOMONIC IMATINIB RESISTANCE:\n"
                "  PDGFRA p.Asp842Val (D842V) = PATHOGNOMONIC for imatinib-resistant GIST.\n"
                "  D842V steric clash prevents imatinib binding → intrinsic resistance.\n"
                "  Avapritinib (BLU-285): type I inhibitor, active in D842V.\n\n"
                "AVAPRITINIB (NAVIGATOR TRIAL):\n"
                "  PDGFRA-D842V mGIST: avapritinib ORR 91%, DOR 27.6mo = BEST ever GIST ORR.\n"
                "  FDA approved Feb 2020 for PDGFRA exon 18 mutant (including D842V) mGIST.\n\n"
                "PDGFRA D842V MORPHOLOGY:\n"
                "  Epithelioid or mixed epithelioid/spindle morphology → test D842V.\n"
                "  Myxoid stroma common.\n"
                "  Gastric location predominant (vs small bowel KIT-exon9).\n"
                "  KIT IHC weak or NEGATIVE in PDGFRA D842V — DO NOT exclude GIST on KIT alone."
            ),
        },
        {
            "term": "SDHA / SDHB-IHC-LOSS-PATHOGNOMONIC-ALL-SDH / CARNEY-TRIAD-NOT-HEREDITARY / SDH-DEFICIENT-MULTIFOCAL",
            "definition": (
                "SDHA — 664aa / 70 kDa / 5p15.33 / AR biallelic + AD monoallelic LOF\n"
                "SDHB IHC loss PATHOGNOMONIC ALL SDH-deficient GIST; Carney Triad NOT hereditary.\n\n"
                "SDHB IHC — UNIVERSAL SURROGATE:\n"
                "  SDHB IHC loss = PATHOGNOMONIC for ALL SDH-deficient GIST regardless of subunit (A/B/C/D).\n"
                "  Reason: SDHB protein degrades when ANY SDH subunit is lost (canary subunit).\n"
                "  SDHA IHC: SDHA-specific — only lost if SDHA mutated; SDHB/C/D mutant = SDHA intact.\n\n"
                "CARNEY TRIAD (NOT HEREDITARY):\n"
                "  Carney Triad = GIST + Pulmonary chondroma + Extra-adrenal PGL.\n"
                "  Young females; NOT hereditary; somatic SDH-deficiency via SDHC epigenetic silencing.\n"
                "  vs Carney-Stratakis Syndrome (CSS): HEREDITARY GIST + PGL (germline SDHB/C/D).\n\n"
                "SDH-DEFICIENT GIST PHENOTYPE:\n"
                "  Multifocal gastric GIST; multinodular plexiform; lymph node metastasis common.\n"
                "  Young women <30yr; KIT/PDGFRA-WT; imatinib POOR response.\n"
                "  SDH-deficient GIST despite metastasis: often indolent — watch-and-wait appropriate.\n"
                "  Off-label active agents: temsirolimus (mTOR), cabozantinib."
            ),
        },
        {
            "term": "SDHB / CSS / MALIGNANCY-30-50PCT-HIGHEST / METHOXYTYRAMINE / DOTATATE-PET",
            "definition": (
                "SDHB — 280aa / 30 kDa / 1p36.13 / AD LOF\n"
                "Carney-Stratakis; malignancy 30-50% HIGHEST; plasma methoxytyramine; DOTATATE PET.\n\n"
                "MALIGNANCY 30-50% — HIGHEST SDH GENE:\n"
                "  SDHB PGL malignancy: 30-50% = HIGHEST of all SDH genes.\n"
                "  vs SDHD: 0-5%; SDHC: 1-3%.\n"
                "  Metastatic SDHB-PGL: bone, liver, lung.\n\n"
                "PLASMA METHOXYTYRAMINE:\n"
                "  Most SDHB-PGL are biochemically non-secreting or dopaminergic.\n"
                "  Plasma methoxytyramine (dopamine metabolite): elevated = SDHB-PGL signature.\n"
                "  Standard metanephrines/normetanephrines: often NORMAL in SDHB — insufficient alone.\n"
                "  Always add plasma methoxytyramine in known/suspected SDHB-PGL workup.\n\n"
                "DOTATATE PET — SDHB PREFERRED:\n"
                "  68Ga-DOTATATE PET/CT: best functional imaging for SDHB-PGL (somatostatin receptor).\n"
                "  Superior to MIBG for SDHB-PGL detection.\n"
                "  Annual: abdominal MRI + DOTATATE PET from age 15yr in SDHB germline carriers."
            ),
        },
        {
            "term": "SDHC / PGL3 / MALIGNANCY-1-3PCT-LOWEST / NOT-IMPRINTED / HNPGL-DOMINANT",
            "definition": (
                "SDHC — 169aa / 15 kDa / 1q23.3 / AD LOF\n"
                "PGL3; malignancy 1-3% LOWEST SDH gene; NOT imprinted; HNPGL dominant.\n\n"
                "LOW MALIGNANCY (1-3%):\n"
                "  SDHC PGL: malignancy only 1-3% = lowest among SDH genes.\n"
                "  HNPGL are predominantly benign in SDHC.\n"
                "  GIST component: ~10% Carney-Stratakis penetrance.\n\n"
                "NOT IMPRINTED:\n"
                "  SDHC: not imprinted (unlike SDHD/MAX which are paternally imprinted).\n"
                "  Maternal OR paternal SDHC = equal cancer risk.\n"
                "  Contrast with SDHD: only paternal SDHD → disease risk.\n\n"
                "HNPGL DOMINANT:\n"
                "  Head-and-neck PGL: carotid body, jugulotympanic, vagal.\n"
                "  Annual neck MRI from age 15yr; DOTATATE PET for HNPGL localisation."
            ),
        },
        {
            "term": "SDHD / PGL1 / PATERNAL-IMPRINTING-MATERNAL-NOT-AT-RISK / MULTILOCAL-HNPGL / PULSATILE-TINNITUS",
            "definition": (
                "SDHD — 160aa / 12 kDa / 11q23.1 / AD LOF with PATERNAL IMPRINTING\n"
                "PGL1; paternal imprinting — maternal carriers NOT at risk; multilocal HNPGL.\n\n"
                "PATERNAL IMPRINTING — CRITICAL CLINICAL RULE:\n"
                "  SDHD maternally imprinted (maternal allele silenced in relevant chromaffin tissue).\n"
                "  PATERNAL SDHD carriers: FULL PGL risk — surveillance MANDATORY.\n"
                "  MATERNAL SDHD carriers: NOT at risk for PGL — reassure.\n"
                "  'From father only does SDHD PGL come' — memorise this rule.\n"
                "  Same rule applies to MAX: paternal imprinting → only paternal carriers at risk.\n\n"
                "MULTILOCAL HNPGL PATHOGNOMONIC:\n"
                "  Bilateral carotid body + jugulotympanic + vagal PGL in young patient = SDHD.\n"
                "  Multiple synchronous HNPGL: SDHD most common cause.\n"
                "  Young onset ~35yr; pulsatile tinnitus/mass = FIRST symptoms.\n\n"
                "MALIGNANCY 0-5% (LOW):\n"
                "  vs SDHB 30-50%; SDHD mainly benign HNPGL.\n"
                "  Annual neck MRI from age 15yr in PATERNAL carriers."
            ),
        },
        {
            "term": "NF1 / NF1-GIST-MULTIFOCAL-SMALL-BOWEL-PATHOGNOMONIC / KIT-PDGFRA-SDH-WT / MEK-INHIBITOR",
            "definition": (
                "NF1 — 2839aa / 319 kDa / 17q11.2 / AD LOF\n"
                "NF1-GIST multifocal small bowel PATHOGNOMONIC; KIT/PDGFRA/SDH-WT; MEK inhibitor active.\n\n"
                "NF1-GIST PATHOGNOMONIC PHENOTYPE:\n"
                "  Multifocal GIST in small bowel (NOT gastric) = PATHOGNOMONIC for NF1.\n"
                "  vs KIT/PDGFRA-GIST: gastric or small bowel solitary.\n"
                "  vs SDH-deficient GIST: multifocal GASTRIC in young females.\n"
                "  NF1-GIST: KIT-negative + PDGFRA-negative + SDHB-positive (SDH-intact).\n\n"
                "IMATINIB POOR RESPONSE:\n"
                "  NF1-GIST: no targetable KIT/PDGFRA — imatinib INEFFECTIVE.\n"
                "  NF1 drives GIST via RAS→MAPK (not KIT/PDGFRA).\n"
                "  MEK inhibitor (binimetinib, selumetinib): active in NF1-driven GIST and MPNST.\n"
                "  Selumetinib FDA-approved for NF1 paediatric plexiform neurofibromas (not GIST yet).\n\n"
                "NF1 CAFÉ-AU-LAIT MACULES PATHOGNOMONIC:\n"
                "  ≥6 CALMs (>5mm prepubertal, >15mm postpubertal) = NF1 NIH diagnostic criterion.\n"
                "  Lisch nodules (iris hamartomas) PATHOGNOMONIC on slit-lamp.\n"
                "  Axillary/inguinal freckling (Crowe's sign) PATHOGNOMONIC.\n\n"
                "CASCADE TESTING — HEREDITARY GIST:\n"
                "  All hereditary GIST: KIT + PDGFRA sequencing + SDHB IHC → if WT/SDH-intact = NF1 workup.\n"
                "  Germline testing: KIT / PDGFRA / SDHA / SDHB / SDHC / SDHD / MAX / NF1.\n"
                "  KIT/PDGFRA IHC: KIT-positive + PDGFRA IHC + DOG1 IHC: standard GIST workup.\n"
                "  SDHB IHC on ALL GIST (not just WT) — detect SDH-deficient cases.\n\n"
                "TIER 1 — MOST ACTIONABLE (TARGETED THERAPY):\n"
                "  KIT exon 11: imatinib 400mg — BEST response.\n"
                "  KIT exon 9: imatinib 800mg — DOSE CRITICAL.\n"
                "  PDGFRA D842V: avapritinib (imatinib resistant).\n"
                "  PDGFRA exon 12/13 WT D842V: imatinib sensitive.\n\n"
                "TIER 2 — SDH-DEFICIENT (INDOLENT BUT TRACK):\n"
                "  SDHA/SDHB/SDHC/SDHD: SDH-deficient GIST (SDHB IHC null).\n"
                "  Imatinib POOR; temsirolimus/cabozantinib off-label.\n"
                "  Annual abdominal MRI + DOTATATE PET (SDHB/D) + methoxytyramine (SDHB).\n\n"
                "TIER 3 — NF1/MAX (INDIRECT):\n"
                "  NF1: MEK inhibitor; MPNST surveillance; FDG-PET for MPNST in painful plexiform.\n"
                "  MAX: annual metanephrines (METANEPHRINE for adrenaline-secreting); adrenal MRI."
            ),
        },
    ]
    # Add MAX definition
    defs.append(
        {
            "term": "MAX / PGL5 / PATERNAL-IMPRINTING / BILATERAL-ADRENAL-PHEO / ADRENALINE-SECRETING",
            "definition": (
                "MAX — 160aa / 17 kDa / 14q23.3 / AD LOF with PATERNAL IMPRINTING\n"
                "PGL5; paternal imprinting — maternal carriers NOT at risk; bilateral adrenal PHEO adrenaline-secreting.\n\n"
                "PATERNAL IMPRINTING (MAX):\n"
                "  MAX is paternally imprinted (same mechanism as SDHD).\n"
                "  PATERNAL MAX carriers: bilateral adrenal PHEO risk.\n"
                "  MATERNAL MAX carriers: NOT at risk (or negligible).\n"
                "  'Same rule as SDHD: only paternal inheritance matters.'\n\n"
                "BILATERAL ADRENAL PHEOCHROMOCYTOMA:\n"
                "  MAX: bilateral adrenal PHEO dominant (vs SDHD HNPGL).\n"
                "  Young onset ~30-35yr; bilateral synchronous or metachronous.\n"
                "  ADRENALINE-secreting → elevated PLASMA METANEPHRINE (not normetanephrine).\n\n"
                "DDx SDHD vs MAX:\n"
                "  SDHD: HNPGL multilocal; dopaminergic; methoxytyramine elevated.\n"
                "  MAX: bilateral adrenal PHEO; adrenaline-secreting; metanephrine elevated.\n"
                "  Both: paternally imprinted; low malignancy."
            ),
        }
    )

    return {
        "atlas":       "Hereditary-GIST-Predisposition-Atlas",
        "seed_range":  f"{SEED_BASE}-{SEED_BASE + 7}",
        "definitions": defs,
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    ov = generate_overview()
    print(json.dumps({k: v for k, v in ov.items() if k not in ("genes_detail",)}, indent=2))
    print("\n=== BREAKDOWN summary ===")
    br = generate_breakdown()
    for row in br["breakdown"]:
        print(f"  {row['gene']:10s} n={row['n']} mean_age={row['mean_age_onset']} "
              f"severe_n={row['severe_n']} ({row['severe_pct']}%)")
    print("\n=== DEFINITIONS (terms only) ===")
    df = generate_definitions()
    for d in df["definitions"]:
        print(f"  {d['term'][:80]}")
