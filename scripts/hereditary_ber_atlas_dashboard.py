"""Hereditary Base Excision Repair Atlas — 8-Gene BER Reference
MUTYH-OGG1-NTHL1-NEIL1-NEIL2-NEIL3-UNG-MPG
(BER deficiency spectrum: MAP / NAP / HIGM5 / oxidative base damage / alkylated base removal)
320 patients (8 x 40), seeds 2742-2749.
Endpoints: /api/hereditary-ber-atlas/overview|breakdown|definitions
"""
import random

ATLAS_GENES = [
    {
        "gene": "MUTYH",
        "seed_base": 2742,
        "protein": (
            "MUTYH -- 1p34.1 AR -- 546aa -- MutY-DNA-Glycosylase-"
            "60kDa-Adenine-Opposite-8-oxoG-Remover-Y-family-BER-Glycosylase-"
            "OMIM-Gene-604933-Disease-MAP-MUTYH-Associated-Polyposis-608456"
        ),
        "locus": "1p34.1",
        "protein_size": "546 aa / 60 kDa (Y-family DNA glycosylase; removes adenine mis-incorporated opposite 8-oxoguanine; OG:A → OG:C correction)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE (biallelic LOF → MAP, MUTYH-Associated Polyposis); "
            "BIALLELIC DISEASE (MAP): "
            "  Colorectal adenomas: 40-100 (attenuated to moderate polyposis); "
            "  CRC lifetime risk: 50-80% by age 60 if unmonitored; "
            "  Extracolonic tumours: duodenal adenomas/cancer (5-17%), gastric polyps, sebaceous gland tumours, ovarian cancer; "
            "  Onset: typically 4th-5th decade (later than FAP); "
            "  NO biallelic = NO MAP cancer syndrome; "
            "FOUNDER/COMMON ALLELES: "
            "  c.536A>G (p.Tyr179Cys, formerly Y165C): most common in Northwestern European; "
            "  c.1187G>A (p.Gly396Asp, formerly G382D): second most common Northwestern European; "
            "  Compound heterozygous Y179C/G396D = classic biallelic MAP presentation; "
            "  Mediterranean: c.1227_1228dup (p.Pro410Alafs); "
            "  South Asian: c.1145G>A (p.Cys382Tyr); "
            "MONOALLELIC RISK: "
            "  Modest CRC risk elevation: ~1.5-2x relative risk (vs general population); "
            "  NOT equivalent to biallelic MAP — monoallelic alone does NOT cause polyposis; "
            "  Clinical management controversial: colonoscopy from 40-50 generally recommended; "
            "PREVALENCE: "
            "  MAP: 0.5-1% of all CRC; ~1% of all colorectal polyposis; "
            "  Carrier frequency: 1 in 100-200 in European ancestry"
        ),
        "disease_category": (
            "MUTYH-ASSOCIATED POLYPOSIS (MAP, OMIM 608456): "
            "COLORECTAL PHENOTYPE: "
            "  Polyp number: 40-100 adenomas (range 10 to hundreds); "
            "  Polyp histology: predominantly tubular adenomas; serrated adenomas reported; "
            "  CRC risk: 50-80% by age 60 without surveillance/colectomy; "
            "  CRC location: right-sided (proximal) predominance (unlike Lynch — left-sided); "
            "  Hypermethylation profile: MUTYH LOF → transversion mutations (G:C → T:A) due to 8-oxoG; "
            "  Somatic KRAS mutation: c.34G>T (p.Gly12Cys) signature of MUTYH BER deficiency; "
            "EXTRACOLONIC: "
            "  Duodenal adenomas: 5-17% (polyposis duodeni; periampullary cancer risk); "
            "  Gastric: hyperplastic polyps + fundic gland polyps; "
            "  Sebaceous gland tumours (Muir-Torre-like): reported in some MAP families; "
            "  Ovarian cancer: slight elevation; "
            "  Breast cancer: not firmly established (no guideline-level surveillance); "
            "ASPIRIN CHEMOPREVENTION: "
            "  Aspirin reduces polyp burden in MAP (CaPP3 trial data); "
            "  Mechanism: COX-2 inhibition reduces adenoma recurrence; "
            "COLECTOMY INDICATIONS: "
            "  Biallelic MAP with unmanageable polyp burden (>~100 adenomas, high-grade dysplasia, CRC); "
            "  Subtotal colectomy with IRA (ileorectal anastomosis) preferred if rectal sparing; "
            "  Proctocolectomy if rectum heavily involved"
        ),
        "disease_pathway": (
            "MUTYH BER MECHANISM — ADENINE MISMATCH REPAIR: "
            "BACKGROUND — 8-OXOGUANINE (8-oxoG) PROBLEM: "
            "  Reactive oxygen species (ROS) oxidise guanine → 8-oxoguanine (8-oxoG); "
            "  8-oxoG is mutagenic: DNA polymerase δ frequently inserts adenine opposite 8-oxoG (OG:A mispair); "
            "  If OG:A not corrected → replication → C:G → T:A or G:C → T:A transversions; "
            "STEP 1 — MUTYH RECOGNISES OG:A MISPAIR: "
            "  MUTYH scans double-stranded DNA; MutT-like nucleotide sanitiser (MTH1) handles free 8-oxodGTP; "
            "  MUTYH binds OG:A with NUDIX hydrolase + MutM/OGG1 interaction; "
            "  Adenine is the substrate: MUTYH removes the misincorporated adenine (NOT the 8-oxoG); "
            "STEP 2 — AP SITE GENERATION + SHORT-PATCH BER: "
            "  MUTYH cleaves N-glycosidic bond → abasic (AP) site; "
            "  APE1 cleaves AP site → nick; "
            "  DNA polymerase β inserts correct C; DNA ligase seals; "
            "STEP 3 — OGG1 REMOVES 8-oxoG: "
            "  OGG1 (bifunctional glycosylase/AP-lyase) then removes 8-oxoG from OG:C pair (after MUTYH corrects OG:A); "
            "  Together MUTYH + OGG1 clear 8-oxoG from both strands; "
            "MUTYH LOF → "
            "  OG:A mispairs not corrected → DNA replication → G:C → T:A transversions → "
            "  Characteristic somatic mutation spectrum in MAP tumours → adenomatous polyposis → CRC; "
            "  KRAS Gly12Cys (c.34G>T) = MUTYH BER-deficiency signature transversion"
        ),
        "pathognomonic": (
            "MAP DIAGNOSTIC APPROACH: "
            "CLINICAL SUSPICION: "
            "  40-100 colorectal adenomas (10+ in younger patient) + NO dominant family history of polyposis; "
            "  Biallelic MAP phenotype: recessive (unaffected parents in most cases); "
            "  APC gene sequencing NEGATIVE in attenuated polyposis + adenomas 10-99 → MUTYH next; "
            "SOMATIC MUTATION SIGNATURE: "
            "  KRAS c.34G>T (p.Gly12Cys) in MAP CRC: sensitive but not 100% specific; "
            "  Tumour mutational burden: G:C → T:A transversion enrichment; "
            "  CRC MSS (microsatellite stable) in MAP — unlike Lynch; "
            "GERMLINE MUTYH TESTING: "
            "  BIALLELIC required for MAP diagnosis: heterozygous alone = NOT MAP; "
            "  Two-tier: screen c.Tyr179Cys + Gly396Asp first (covers >85% alleles in NW Europeans); "
            "  Full gene sequencing + MLPA if common alleles negative + strong phenotype; "
            "IMMUNOHISTOCHEMISTRY: "
            "  No established IHC panel for MUTYH (unlike Lynch — no MMR IHC); "
            "  OGG1 IHC: investigational only; "
            "ENDOSCOPIC FINDINGS: "
            "  Colonoscopy: 40-100 adenomas (typically tubular, sessile); right-colon predominance; "
            "  Upper GI (UGIE): duodenal adenoma surveillance every 1-3 years from age 25-30 in MAP; "
            "ASPIRIN POLYP REDUCTION: "
            "  CaPP3: aspirin 100 mg daily shows polyp burden reduction in MAP"
        ),
        "treatment": (
            "MAP MANAGEMENT: "
            "COLORECTAL SURVEILLANCE (biallelic MAP): "
            "  Annual colonoscopy from age 18-25 (or when diagnosis confirmed); "
            "  Polypectomy of all adenomas (electrocautery/EMR); "
            "  If polyp burden unmanageable: colectomy; "
            "COLECTOMY: "
            "  Indication: CRC, high-grade dysplasia, polyp burden >100-150 adenomas; "
            "  Subtotal colectomy + IRA: preferred if rectum manageable (<20 rectal adenomas); "
            "  Total proctocolectomy + IPAA (J-pouch): if rectum heavily involved; "
            "  Timing: elective before CRC develops (age 30-40 in most MAP patients); "
            "CHEMOPREVENTION: "
            "  Aspirin 100-600 mg/day: reduces polyp burden (CaPP3 data); "
            "  Sulindac: NSAIDs second-line (lower evidence in MAP vs FAP); "
            "  Celecoxib: COX-2 inhibitor option (as per MAP-specific trial data); "
            "UPPER GI SURVEILLANCE: "
            "  Upper GI endoscopy every 1-3 years from age 25-30 (duodenal adenomas); "
            "  Spigelman classification guides interval and surgical referral; "
            "EXTRACOLONIC: "
            "  Thyroid US: consider if duodenal disease; "
            "  Breast/ovarian: no formal guideline recommendation for MAP; "
            "MONOALLELIC MANAGEMENT: "
            "  Colonoscopy from age 40-50 (5-yearly if normal); "
            "  No enhanced extracolonic surveillance; "
            "CASCADE TESTING: "
            "  AR gene: siblings 25% risk of biallelic; parents carriers; "
            "  Test all siblings and children in MAP proband family (children at 50% carrier risk)"
        ),
    },
    {
        "gene": "OGG1",
        "seed_base": 2743,
        "protein": (
            "OGG1 -- 3p25.3 AR/biallelic-low-penetrance -- 345aa -- "
            "8-Oxoguanine-DNA-Glycosylase-1-"
            "39kDa-Bifunctional-Glycosylase-AP-Lyase-"
            "OMIM-Gene-601982-Disease-Biallelic-Low-Penetrance"
        ),
        "locus": "3p25.3",
        "protein_size": "345 aa / 39 kDa (bifunctional glycosylase + AP-lyase; removes 8-oxoguanine opposite C; beta-lyase activity creates SSB)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE (biallelic compound heterozygous LOF — rare case reports; low penetrance uncertain); "
            "VARIANT OF NOTE — Lys326Gln (rs1052133): "
            "  Common polymorphism (allele frequency 15-30% European); "
            "  NOT established as pathogenic: does not meet ACMG pathogenic criteria; "
            "  Controversy: some association studies link Lys326Gln with lung/kidney/head-neck cancer; "
            "  Confounded by linkage disequilibrium; meta-analyses show modest effect sizes (OR 1.1-1.3); "
            "  Current consensus: Lys326Gln = low-effect variant; NOT actionable clinically; "
            "BIALLELIC COMPOUND HETEROZYGOUS OGG1 LOF: "
            "  Rare published case reports; phenotype incompletely characterised; "
            "  Association: oxidative stress-related cancers (lung, prostate) — suggestive; "
            "  No established Mendelian disease syndrome confirmed for OGG1 biallelic LOF; "
            "  Substrate: 8-oxoguanine (8-oxoG) in OG:C pairs (not OG:A — that is MUTYH substrate); "
            "OXIDATIVE STRESS LINK: "
            "  OGG1 Knock-out mice: accumulate 8-oxoG in nuclear + mitochondrial DNA; "
            "  Increased spontaneous mutagenesis + age-related cancer predisposition in mice; "
            "PREVALENCE: "
            "  True biallelic OGG1 disease: not well-established in clinical databases; "
            "  Lys326Gln heterozygous: common variant (not a disease allele)"
        ),
        "disease_category": (
            "OGG1 BIALLELIC LOF — PUTATIVE CANCER PREDISPOSITION (poorly characterised): "
            "CURRENT EVIDENCE: "
            "  No established human Mendelian polyposis or cancer syndrome; "
            "  Published case series: OGG1 biallelic compound heterozygous → lung cancer + renal cell carcinoma (single families); "
            "  Insufficient evidence for formal syndrome designation; "
            "MURINE MODEL: "
            "  OGG1 −/− mice: accumulate 8-oxoG; spontaneous lung adenoma/carcinoma ↑; "
            "  No polyposis (contrast MUTYH-knockout); primarily lung + liver neoplasia; "
            "OGG1 SUBSTRATE — 8-oxoG: "
            "  8-oxoG in OG:C pair: OGG1 removes 8-oxoG (leaving abasic site); "
            "  AP-lyase activity: β-elimination creates SSB; APE1 required for complete repair; "
            "  FapyGua (formamidopyrimidine-guanine): also OGG1 substrate; "
            "LYS326GLN CONTROVERSY: "
            "  Biochemical: Lys326Gln reduces OGG1 activity in vitro (reduced 8-oxoG removal); "
            "  Clinical: population association with lung/kidney cancer in some studies; "
            "  Interpretation: low-penetrance susceptibility factor, NOT a dominant or recessive actionable gene currently; "
            "DISTINCTION FROM MUTYH: "
            "  OGG1 removes 8-oxoG from OG:C (the DAMAGED strand); "
            "  MUTYH removes adenine from OG:A (the UNDAMAGED strand — preventing mutation fixation); "
            "  Conceptually: OGG1 = primary repair; MUTYH = post-replication correction of OGG1-escaped lesions"
        ),
        "disease_pathway": (
            "OGG1 BIFUNCTIONAL GLYCOSYLASE/AP-LYASE — 8-oxoG REMOVAL FROM OG:C: "
            "STEP 1 — LESION RECOGNITION: "
            "  OGG1 slides along double-stranded DNA; base-flipping mechanism; "
            "  8-oxoG extruded from helix into OGG1 active site; cytosine opposite read for specificity; "
            "  OGG1 DOES NOT act on OG:A (that is MUTYH); OGG1 requires cytosine opposite 8-oxoG; "
            "STEP 2 — GLYCOSYLASE ACTIVITY: "
            "  Lys249 (nucleophilic attack) cleaves N-glycosidic bond between 8-oxoG and deoxyribose; "
            "  Product: abasic (AP) site; 8-oxoG expelled; "
            "STEP 3 — AP-LYASE ACTIVITY (β-elimination): "
            "  OGG1 Lys249 Schiff base intermediate → β-elimination → 3'-phospho-α,β-unsaturated aldehyde (3'-PUA); "
            "  Net: nick with 3'-blocked terminus; "
            "STEP 4 — APE1 PROCESSING: "
            "  APE1 (AP endonuclease 1) cleaves 3'-PUA → 3'-OH; "
            "  DNA polymerase β fills gap; DNA ligase seals; "
            "STEP 5 — LONG-PATCH BER (alternative): "
            "  For clustered damage or inefficient short-patch → PCNA/FEN1/Pol δ/Pol ε long-patch; "
            "REDUNDANCY WITH NEIL1: "
            "  NEIL1 can also remove FapyGua (overlapping substrate with OGG1); "
            "  NEIL1 prefers single-stranded/bubble substrates; OGG1 prefers dsDNA; "
            "OGG1 LOF → 8-oxoG accumulates in OG:C → if replication occurs before repair → "
            "  DNA polymerase δ inserts A opposite 8-oxoG → OG:A mispair; "
            "  MUTYH then acts on OG:A; if both OGG1 + MUTYH defective → double BER failure → "
            "  G:C → T:A transversions at high rate → cancer-driving mutations"
        ),
        "pathognomonic": (
            "OGG1 DIAGNOSTIC APPROACH (investigational — no established clinical standard): "
            "WHEN TO SUSPECT: "
            "  Biallelic compound heterozygous OGG1 variants in patient with lung/renal/oxidative-stress cancer; "
            "  Early-onset oxidative-stress cancer with no MUTYH/NTHL1 biallelic variants; "
            "GENETIC TESTING: "
            "  OGG1 gene sequencing (NGS panel — included in some hereditary cancer panels); "
            "  Common Lys326Gln (rs1052133): DO NOT report as pathogenic — it is a benign polymorphism; "
            "  Pathogenic variants: rare LOF (splice, frameshift, nonsense); "
            "  Functional assay: OGG1 8-oxoG glycosylase activity in lymphocytes (research use); "
            "TUMOUR SIGNATURE: "
            "  8-oxoG accumulation detectable in tumour DNA (research techniques); "
            "  Somatic mutation profile: G:C → T:A transversions (overlap with MUTYH spectrum); "
            "  OGG1 LOF in tumour: detected by somatic panel or WGS; "
            "IMPORTANT CAVEAT: "
            "  Lys326Gln genotyping reports: DO NOT alter clinical management; "
            "  Biallelic OGG1 LOF: insufficient evidence for formal enhanced surveillance beyond standard cancer screening; "
            "  Register in hereditary cancer research registry; genetic counselling mandatory"
        ),
        "treatment": (
            "OGG1 BIALLELIC — MANAGEMENT (emerging/investigational): "
            "NO ESTABLISHED GUIDELINE (insufficient evidence for formal management protocol): "
            "GENERAL: "
            "  Enhanced colorectal surveillance: colonoscopy from age 35-40 (extrapolated from BER gene logic); "
            "  Lung surveillance: low-dose CT chest annually from age 40 (especially if smoker); "
            "  Renal surveillance: renal US or CT KUB from age 35-40; "
            "CANCER PREVENTION: "
            "  Antioxidant measures: reduce exogenous ROS exposure (avoid tobacco, radiation); "
            "  No proven pharmacological chemoprevention for OGG1 LOF; "
            "  Aspirin: no specific MAP-like trial data; "
            "LYS326GLN CARRIERS: "
            "  NO enhanced surveillance; standard cancer screening per age/family history; "
            "RESEARCH: "
            "  OGG1 inhibitors as chemotherapy sensitisers (reverse context — OGG1 inhibition in cancer cells); "
            "  TH10785 (OGG1 inhibitor): clinical trial in solid tumours (sensitises to oxidative stress); "
            "CASCADE TESTING: "
            "  If biallelic OGG1 LOF confirmed: sibling 25% risk; parents carriers; "
            "  No established surveillance protocol for monoallelic carriers"
        ),
    },
    {
        "gene": "NTHL1",
        "seed_base": 2744,
        "protein": (
            "NTHL1 -- 16p13.3 AR -- 312aa -- "
            "Endonuclease-III-Like-Protein-1-"
            "35kDa-Bifunctional-Formamidopyrimidine-Ring-Opened-Purine-Glycosylase-"
            "OMIM-Gene-602656-Disease-NTHL1-Associated-Polyposis-NAP-616415"
        ),
        "locus": "16p13.3",
        "protein_size": "312 aa / 35 kDa (bifunctional BER glycosylase/AP-lyase; removes oxidised pyrimidines + formamidopyrimidines; HhH-GPD superfamily)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE (biallelic LOF → NTHL1-Associated Polyposis, NAP); "
            "BIALLELIC DISEASE (NAP): "
            "  Colorectal adenomas: 5-50+ adenomas (fewer than FAP, more than Lynch); "
            "  CRC risk: substantially elevated (60-70% lifetime); "
            "  Extracolonic cancers: breast + endometrial + urothelial — critical distinction from MAP; "
            "  Onset: typically 5th-6th decade; "
            "KEY ALLELE: "
            "  Gln90Ter (c.268C>T, p.Gln90*): truncating — most common pathogenic allele; "
            "  Founder in Dutch + British populations; "
            "  Biallelic Gln90Ter = classic NAP; "
            "  Other alleles: missense variants (less characterised); "
            "PREVALENCE: "
            "  Rare: estimated 1 in 100,000-200,000; "
            "  Prevalence likely underdiagnosed (no IHC marker, relatively recent discovery 2015); "
            "MONOALLELIC: "
            "  No established disease risk for monoallelic NTHL1; "
            "  NOT equivalent to monoallelic MUTYH (~1.5-2x CRC); "
            "SOMATIC SIGNATURE: "
            "  NTHL1 LOF → Signature 30 (C>T transitions at CpG; also C>T at non-CpG in some studies); "
            "  Distinct from MUTYH transversion signature (G>T at C:G pairs)"
        ),
        "disease_category": (
            "NTHL1-ASSOCIATED POLYPOSIS (NAP, OMIM 616415): "
            "COLORECTAL: "
            "  Polyp number: 5-50+ adenomas (attenuated-moderate; fewer than MAP in most cases); "
            "  CRC: lifetime risk ~60-70%; "
            "  MSS (microsatellite stable) CRC: unlike Lynch; "
            "  IHC: no NTHL1 IHC test routinely available — blind spot vs. MMR-IHC in Lynch; "
            "EXTRACOLONIC CANCERS (critical distinction from MAP): "
            "  Breast cancer: elevated risk (ER-positive; 30-40% lifetime in some Dutch cohorts); "
            "  Endometrial cancer: elevated risk (30-40% lifetime — endometrial ENDOMETRIAL-LIKE Lynch-DDx); "
            "  Urothelial cancer (bladder + upper tract): elevated risk; "
            "  Other: duodenal adenomas, lung cancer; "
            "NTHL1 vs MAP vs Lynch DIFFERENTIAL: "
            "  MUTYH (MAP): CRC + duodenal; NO major endometrial/urothelial/breast; "
            "  NTHL1 (NAP): CRC + breast + endometrial + urothelial — much broader extracolonic; "
            "  Lynch (MMR): MSI-H tumours; CRC + endometrial; IHC MMR absent; "
            "  NAP: MSS tumours; no MMR IHC loss; requires germline NTHL1 biallelic testing; "
            "SOMATIC SIGNATURE 30: "
            "  C>T transitions at CpG and TCA context; overlaps with ageing/APOBEC but distinct context; "
            "  Useful for identifying NTHL1-deficient tumours in WGS studies"
        ),
        "disease_pathway": (
            "NTHL1 BER MECHANISM — OXIDISED PYRIMIDINE + FORMAMIDOPYRIMIDINE REMOVAL: "
            "SUBSTRATES: "
            "  Oxidised pyrimidines: thymine glycol (Tg), 5-hydroxycytosine (5-OHC), 5-hydroxyuracil (5-OHU); "
            "  Ring-opened purines: formamidopyrimidine-adenine (FapyAde), FapyGua (overlap with NEIL1); "
            "  NTHL1 = HhH-GPD family glycosylase (helix-hairpin-helix); "
            "STEP 1 — BASE RECOGNITION + EXTRUSION: "
            "  NTHL1 flips damaged base out of helix into active site cavity; "
            "  HhH motif reads DNA backbone (sequence-independent minor groove contacts); "
            "STEP 2 — GLYCOSYLASE CLEAVAGE: "
            "  Lys212 nucleophilic attack on C1' of damaged nucleotide → N-glycosidic bond cleaved → AP site; "
            "STEP 3 — β-ELIMINATION (AP-lyase): "
            "  Lys212 Schiff base → β-elimination → 3'-unsaturated aldehyde terminus; "
            "  NTHL1 has weak β-lyase; APE1 required for efficient processing; "
            "STEP 4 — SHORT-PATCH BER: "
            "  APE1 → Pol β → ligase III/XRCC1 (nuclear); "
            "  Mitochondria: NTHL1 also active in mitochondrial BER; "
            "NTHL1 LOF → "
            "  Oxidised pyrimidines + FapyAde/FapyGua accumulate → replication errors → "
            "  C>T transitions (deamination signature at damaged cytosine) → "
            "  Signature 30 somatic landscape → NAP tumourigenesis"
        ),
        "pathognomonic": (
            "NAP (NTHL1-ASSOCIATED POLYPOSIS) DIAGNOSTIC APPROACH: "
            "WHEN TO SUSPECT: "
            "  10-50+ colorectal adenomas + negative APC + negative MUTYH biallelic; "
            "  Polyposis + breast + endometrial + urothelial cancer in same patient or close relatives; "
            "  Autosomal recessive pattern (affected siblings, unaffected parents); "
            "  MSS CRC in polyposis patient without MMR deficiency (no IHC loss, MSI-H negative); "
            "GENETIC TESTING: "
            "  Gln90Ter (c.268C>T) first-tier in Dutch/British ancestry; "
            "  Full gene sequencing + MLPA (large deletions); "
            "  NGS hereditary cancer panel (NTHL1 included in most current multi-gene panels); "
            "  BIALLELIC required for NAP diagnosis; monoallelic alone = insufficient; "
            "SOMATIC SIGNATURE: "
            "  Tumour WGS: Signature 30 (C>T at CpG/TCA); "
            "  Useful to support germline NTHL1 testing in sporadic CRC patients; "
            "IHC LIMITATION: "
            "  NO NTHL1 IHC available in clinical practice (blind spot vs Lynch MMR-IHC); "
            "  Cannot screen NAP by tumour immunohistochemistry — must rely on germline testing; "
            "TUMOUR MOLECULAR CHARACTERISATION: "
            "  MSS (not MSI-H): unlike Lynch; "
            "  Confirm MSS/MSI status to differentiate from Lynch before NAP-specific management"
        ),
        "treatment": (
            "NAP (NTHL1-ASSOCIATED POLYPOSIS) MANAGEMENT: "
            "COLORECTAL SURVEILLANCE: "
            "  Annual colonoscopy from age 20-25 (biallelic NAP confirmed); "
            "  Polypectomy of all adenomas; "
            "  Colectomy if polyp burden unmanageable or CRC develops; "
            "  Subtotal colectomy with IRA preferred if rectal sparing; "
            "EXTRACOLONIC SURVEILLANCE (critical — broader than MAP): "
            "  Breast: annual breast MRI + mammography from age 30-35 (biallelic NAP women); "
            "  Endometrial: annual transvaginal US + endometrial biopsy from age 35-40; "
            "  Urothelial: annual urine cytology + USS renal tract from age 35; "
            "  Upper GI (duodenum): upper GI endoscopy every 2-3 years; "
            "CHEMOPREVENTION: "
            "  Aspirin: extrapolated from MAP (no NAP-specific trial); "
            "  No established pharmacological prevention for NAP; "
            "GYNAECOLOGICAL: "
            "  Discuss risk-reducing hysterectomy (for biallelic NAP women after family completion); "
            "  No firm guideline recommendation yet but endometrial risk justifies discussion; "
            "CASCADE TESTING: "
            "  AR gene: all siblings 25% risk; parents obligate carriers; "
            "  Carrier testing: no clinical intervention for monoallelic carriers; "
            "REGISTRY ENROLMENT: "
            "  All NAP patients should be enrolled in hereditary polyposis registry; "
            "  NTHL1 disease characterisation ongoing — clinical data needed"
        ),
    },
    {
        "gene": "NEIL1",
        "seed_base": 2745,
        "protein": (
            "NEIL1 -- 15q24.2 AR -- 390aa -- "
            "Nei-Endonuclease-VIII-Like-1-"
            "44kDa-Bifunctional-Formamidopyrimidine-Ring-Opened-Purine-Glycosylase-"
            "OMIM-Gene-608844-Disease-Putative-Metabolic-Syndrome"
        ),
        "locus": "15q24.2",
        "protein_size": "390 aa / 44 kDa (NEIL family bifunctional glycosylase/lyase; removes formamidopyrimidines, ring-opened purines, and oxidised pyrimidines; β-δ elimination AP-lyase)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE (biallelic LOF — very rare; human clinical disease poorly characterised); "
            "MICE DATA: "
            "  NEIL1 −/− mice: morbid obesity, fatty liver, dyslipidaemia, insulin resistance → metabolic syndrome; "
            "  Mechanism: mtDNA damage in metabolically active tissues (liver, adipose) → mitochondrial dysfunction; "
            "  Increased genomic instability in actively replicating cells; "
            "HUMAN DISEASE: "
            "  Gly83Asp variant: suggested founder in some populations (limited data); "
            "  Biallelic LOF in humans: extremely rare; no established Mendelian syndrome confirmed; "
            "  Association studies: NEIL1 variants linked with type 2 diabetes, obesity — population level; "
            "TRANSCRIPTION-COUPLED BER: "
            "  NEIL1 participates in transcription-coupled BER (TC-BER): "
            "  Recruited to stalled RNA Pol II at oxidative base lesions; "
            "  Removes damage from transcribed template strand; "
            "SUBSTRATES: "
            "  FapyAde (formamidopyrimidine-adenine): principal substrate; "
            "  FapyGua: shared with OGG1; "
            "  Thymine glycol (Tg), 5-hydroxyuracil, 5-hydroxycytosine (oxidised pyrimidines); "
            "  NEIL1 unique: processes single-stranded DNA, bubble structures, G-quadruplex; "
            "PREVALENCE: established human Mendelian disease: not confirmed in clinical databases"
        ),
        "disease_category": (
            "NEIL1 DEFICIENCY — PUTATIVE METABOLIC SYNDROME / POORLY CHARACTERISED HUMAN DISEASE: "
            "CURRENT CLINICAL STATUS: "
            "  No established human Mendelian polyposis or cancer syndrome for NEIL1 biallelic LOF; "
            "  Evidence primarily from mouse knockouts + population genetics association studies; "
            "NEIL1 KNOCKOUT MICE PHENOTYPE: "
            "  Obesity: spontaneous obesity on standard chow (40-50% body weight increase); "
            "  Fatty liver (hepatic steatosis/NAFLD): histologically confirmed; "
            "  Dyslipidaemia: elevated triglycerides, LDL; "
            "  Insulin resistance + type 2 diabetes features; "
            "  Mitochondrial dysfunction: 8-oxoG accumulation in mtDNA; "
            "  Hepatocellular carcinoma: elevated spontaneous HCC rate at advanced age; "
            "HUMAN ASSOCIATION STUDIES: "
            "  NEIL1 variants (including Gly83Asp): associated with obesity, T2DM in some cohorts; "
            "  Pancreatic beta-cell NEIL1: BER critical for beta-cell survival under oxidative stress; "
            "NEIL1 IN TRANSCRIPTION-COUPLED REPAIR: "
            "  NEIL1 recruited to actively transcribed genes under oxidative stress; "
            "  TC-BER deficiency: transcriptional blockade → premature ageing-like phenotype; "
            "DDx FROM NEIL2: "
            "  NEIL1: processes both strands; prefers ssDNA/bubble; meiotic function; "
            "  NEIL2: strand-specific for transcribed strand (TC-BER); different substrate overlap"
        ),
        "disease_pathway": (
            "NEIL1 GLYCOSYLASE/AP-LYASE MECHANISM — RING-OPENED PURINE + OXIDISED PYRIMIDINE REPAIR: "
            "SUBSTRATES DETAIL: "
            "  FapyAde: adenine ring opened at N7-C8 bond by ROS; highly mutagenic; NEIL1 primary remover; "
            "  FapyGua: guanine ring-opened form; NEIL1 + OGG1 overlap; "
            "  Thymine glycol (Tg): thymine oxidation product; replication-blocking; "
            "  5-Hydroxyuracil (5-OHU): cytosine deamination product under oxidation; "
            "STRUCTURAL BASIS: "
            "  NEIL1 contains Fpg/Nei superfamily fold; "
            "  Zinc finger + zinc-less finger: DNA binding without Zn in active site (unlike OGG1); "
            "  Pro2 (N-terminal proline) Schiff base: replaces Lys in OGG1/NTHL1 mechanism; "
            "  Pro2 attacks C1' → Schiff base → N-glycosidic bond hydrolysis → AP site; "
            "AP-LYASE — β-δ ELIMINATION: "
            "  NEIL1 performs β-δ elimination (not just β like OGG1/NTHL1); "
            "  β-δ elimination leaves 3'-phosphate terminus (cleaner end); "
            "  PNKP (polynucleotide kinase-phosphatase) further processes; "
            "  More efficient strand break generation than NTHL1/OGG1 AP-lyase; "
            "SINGLE-STRAND PREFERENCE: "
            "  NEIL1 processes ssDNA, bubble structures, replication fork structures; "
            "  Positioned to repair damage AHEAD of replication fork (pre-emptive repair); "
            "NEIL1 LOF → "
            "  FapyAde/FapyGua/Tg accumulate → replication errors → mutagenesis → "
            "  Metabolic tissue (liver/adipose) mitochondrial dysfunction → metabolic syndrome (mice model)"
        ),
        "pathognomonic": (
            "NEIL1 DIAGNOSTIC APPROACH (investigational — no established clinical syndrome): "
            "WHEN TO SUSPECT: "
            "  Severe early-onset obesity + metabolic syndrome + family history + no monogenic obesity gene found; "
            "  Hepatocellular carcinoma in young non-cirrhotic patient; "
            "  Cancer predisposition with oxidative base damage accumulation (research setting); "
            "GENETIC TESTING: "
            "  NGS panel (if NEIL1 included): biallelic LOF variants required; "
            "  Gly83Asp: check; interpret with caution (limited pathogenicity data); "
            "  Functional assay: NEIL1 glycosylase activity on FapyAde substrate (research); "
            "  8-oxoG accumulation in lymphocytes or fibroblasts: comet assay research tool; "
            "METABOLIC WORKUP: "
            "  Fasting glucose, HbA1c, fasting insulin, HOMA-IR; "
            "  Lipid profile (TGs, LDL, HDL); "
            "  Liver USS + LFTs; "
            "  If fatty liver: MRI-PDFF quantification; "
            "TUMOUR SIGNATURE: "
            "  FapyAde → Adenine transversion signature (A:T → T:A in some model systems); "
            "  Overlap with other BER-deficient signatures; "
            "IMPORTANT: "
            "  NEIL1 biallelic LOF = research diagnosis currently; "
            "  No actionable germline guideline; enrolment in research registry recommended"
        ),
        "treatment": (
            "NEIL1 BIALLELIC — MANAGEMENT (investigational/empirical): "
            "NO ESTABLISHED GUIDELINE: "
            "METABOLIC MANAGEMENT: "
            "  Obesity/metabolic syndrome: standard lifestyle (diet, exercise) + pharmacological; "
            "  Metformin: first-line T2DM (also has antioxidant properties in some studies); "
            "  Liver disease: NASH management (lifestyle, weight loss, vitamin E in NASH); "
            "  Regular metabolic monitoring: annual HbA1c, lipids, LFTs, liver USS; "
            "CANCER SURVEILLANCE (empirical, no guideline): "
            "  Liver (HCC): 6-monthly AFP + USS in fatty liver disease; "
            "  Colorectal: 5-yearly colonoscopy from age 40 (standard CRC screening); "
            "ANTIOXIDANT APPROACHES: "
            "  Reduce exogenous oxidative stress: no tobacco, no alcohol (hepatotoxic in fatty liver); "
            "  Dietary antioxidants: vitamin E, carotenoids (plausible; no RCT in NEIL1); "
            "RESEARCH OPTIONS: "
            "  Clinical trial eligibility: oxidative DNA damage repair trials; "
            "  Registry: enrol in BER gene registry for longitudinal phenotyping; "
            "CASCADE TESTING: "
            "  If biallelic LOF confirmed: siblings 25% risk; parents carriers; "
            "  Carriers: standard health maintenance"
        ),
    },
    {
        "gene": "NEIL2",
        "seed_base": 2746,
        "protein": (
            "NEIL2 -- 8p21.3 AR -- 342aa -- "
            "Nei-Endonuclease-VIII-Like-2-"
            "38kDa-Transcribed-Strand-Preferring-5-OHU-Oxidised-Cytosine-BER-Glycosylase-"
            "OMIM-Gene-608933-Disease-Not-Established"
        ),
        "locus": "8p21.3",
        "protein_size": "342 aa / 38 kDa (NEIL family bifunctional glycosylase; transcribed-strand-preferring BER; removes 5-hydroxyuracil, 5-hydroxycytosine, oxidised cytosines)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE (biallelic LOF — very rare; no established human Mendelian syndrome); "
            "STRAND SPECIFICITY: "
            "  NEIL2 = TRANSCRIBED STRAND PREFERRING glycosylase; "
            "  Removes oxidative base damage specifically from transcribed (template) strand; "
            "  Participates in transcription-coupled BER (TC-BER); "
            "SUBSTRATES: "
            "  5-hydroxyuracil (5-OHU): principal substrate; "
            "  5-hydroxycytosine (5-OHC): cytosine oxidation product; "
            "  Dihydrouracil (DHU); FapyAde (minor activity vs NEIL1); "
            "  Limited activity on 8-oxoG (contrast OGG1 primary); "
            "HUMAN CLINICAL DATA: "
            "  Very few pathogenic NEIL2 biallelic case reports; "
            "  Association studies: NEIL2 variants with lung cancer, gastric cancer (modest ORs); "
            "  Inflammatory bowel disease: NEIL2 variants associated with colitis severity (some studies); "
            "MICE: "
            "  NEIL2 −/− mice: elevated accumulation of oxidised cytosines in transcribed genes; "
            "  Increased gastrointestinal inflammation; "
            "  Modest cancer predisposition (less dramatic than NEIL1 −/− metabolic phenotype); "
            "PREVALENCE: no established Mendelian disease; rare variants under study"
        ),
        "disease_category": (
            "NEIL2 DEFICIENCY — TRANSCRIBED STRAND OXIDATIVE BASE DAMAGE / POORLY CHARACTERISED: "
            "HUMAN DISEASE STATUS: "
            "  No established Mendelian syndrome; "
            "  Research: biallelic NEIL2 variants → transcribed-strand 5-OHU/5-OHC accumulation; "
            "POTENTIAL PHENOTYPES (from model systems): "
            "  Gastrointestinal inflammation: gut epithelium with high cell turnover vulnerable; "
            "  Pulmonary: lung cancer association in some GWAS/candidate gene studies; "
            "  Mutagenesis in transcribed genes: transcription blockade → premature ageing? (speculative); "
            "NEIL2 TRANSCRIBED STRAND BIOLOGY: "
            "  NEIL2 co-purifies with RNA Pol II elongation complex; "
            "  Positioned to scan transcribed strand BEHIND RNA Pol II; "
            "  Rapid repair of oxidative damage in actively transcribed regions = prevents transcriptional mutagenesis; "
            "  Priority repair of transcribed sequences → evolutionarily conserved protection of gene expression; "
            "DISTINCTION FROM NEIL1: "
            "  NEIL1: non-strand-specific; bubble/ssDNA preference; wider substrate range; "
            "  NEIL2: transcribed strand preferring; TC-BER specialist; narrower substrate range; "
            "  Substrates: NEIL2 prefers oxidised cytosines (5-OHU, 5-OHC); NEIL1 prefers FapyAde/FapyGua; "
            "DISTINCTION FROM NTHL1: "
            "  Both remove oxidised pyrimidines (Tg, 5-OHU); "
            "  NEIL2 strand-selective TC-BER; NTHL1 GGR (global genome repair) + TC-BER both"
        ),
        "disease_pathway": (
            "NEIL2 GLYCOSYLASE — TRANSCRIPTION-COUPLED OXIDATIVE CYTOSINE REPAIR: "
            "STEP 1 — COUPLING TO TRANSCRIPTION: "
            "  RNA Pol II stalls at oxidised cytosine (5-OHU, 5-OHC) on template strand; "
            "  CSB (Cockayne Syndrome B, ERCC6) recruits NEIL2 to stalled complex; "
            "  NEIL2 associates with RNAP-II, PCNA, replication protein A, XRCC1; "
            "STEP 2 — SUBSTRATE RECOGNITION: "
            "  NEIL2 pro2 Schiff-base: extracts damaged base from transcription bubble context; "
            "  5-OHU: cytosine deaminated + oxidised → miscoding (reads as T) → C>T transitions; "
            "  5-OHC: cytosine oxidised → mutagenic if unrepaired; "
            "STEP 3 — β-δ ELIMINATION: "
            "  NEIL2 performs β-δ elimination (same as NEIL1); "
            "  3'-phosphate terminus; PNKP processes; "
            "STEP 4 — GAP FILLING + LIGATION: "
            "  Pol β (short-patch) or Pol δ/ε (long-patch) fill gap; "
            "  Ligase III/XRCC1 (short-patch) or Ligase I (long-patch) seals; "
            "NEIL2 LOF → "
            "  5-OHU/5-OHC accumulate in transcribed regions → C>T transitions in active genes → "
            "  Transcriptional mutagenesis + blockade → inflammation + cancer predisposition (postulated)"
        ),
        "pathognomonic": (
            "NEIL2 DIAGNOSTIC (investigational — no clinical syndrome established): "
            "GENETIC TESTING: "
            "  NGS hereditary cancer panel (NEIL2 if included); biallelic LOF variants; "
            "  Functional assay: 5-OHU glycosylase activity (research); "
            "  Comet assay: oxidative damage burden in lymphocytes; "
            "WHEN TO CONSIDER: "
            "  Young patient with GI cancer/inflammation + family history + no Lynch/MAP/NAP diagnosis; "
            "  Research protocol only currently; "
            "TUMOUR SIGNATURE: "
            "  C>T transitions (5-OHU → T:A if unrepaired); overlap with ageing signature 1; "
            "  Transcribed-strand bias in somatic mutations (research WGS analysis); "
            "IMPORTANT CAVEATS: "
            "  NEIL2 germline pathogenic variants = rare + uncertain clinical significance; "
            "  Standard cancer screening per age/FH unless confirmed biallelic LOF in research context; "
            "  Register in BER research registry"
        ),
        "treatment": (
            "NEIL2 BIALLELIC — MANAGEMENT (empirical/investigational): "
            "NO ESTABLISHED GUIDELINE: "
            "GENERAL: "
            "  Standard cancer screening per age guidelines; "
            "  GI surveillance: colonoscopy from age 40 if GI symptoms or family history of GI cancer; "
            "  Reduce exogenous mutagenic exposures (smoking cessation paramount — lung cancer link); "
            "ANTI-INFLAMMATORY: "
            "  If GI inflammation: standard IBD management (5-ASA, immunosuppression); "
            "  No specific NEIL2-targeted therapy; "
            "RESEARCH: "
            "  Research registry enrolment; "
            "  WGS somatic signature profiling of any tumour; "
            "CASCADE TESTING: "
            "  AR gene: siblings 25% risk; parents carriers; "
            "  No actionable intervention for monoallelic NEIL2 carriers currently"
        ),
    },
    {
        "gene": "NEIL3",
        "seed_base": 2747,
        "protein": (
            "NEIL3 -- 4q24 AR -- 605aa -- "
            "Nei-Endonuclease-VIII-Like-3-"
            "68kDa-G-Quadruplex-Abasic-Site-Meiotic-SSB-FA-Backup-Pathway-Glycosylase-"
            "OMIM-Gene-608770-Disease-Not-Established"
        ),
        "locus": "4q24",
        "protein_size": "605 aa / 68 kDa (NEIL family glycosylase; unique G-quadruplex/abasic-site unhooking; meiotic SSB repair; FA-backup ICL function; ZRANB3-interaction)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE (biallelic LOF — rare; no established human Mendelian syndrome); "
            "UNIQUE FUNCTIONS (distinguish from NEIL1/NEIL2): "
            "  G-QUADRUPLEX (G4) UNHOOKING: NEIL3 processes abasic sites within G4 structures; "
            "  ICL REPAIR BACKUP: NEIL3 unhooks ICLs (interstrand crosslinks) at replication forks "
            "    as backup to FA pathway; NEIL3 processes abasic ICL-type linkages; "
            "  MEIOTIC SSB REPAIR: NEIL3 expressed during meiosis; processes meiotic single-strand breaks; "
            "  Spermatogenesis: NEIL3 −/− mice: elevated base damage in germ cells; "
            "SUBSTRATES: "
            "  Hydantoins (guanine oxidation products — Sp, Gh): principal substrates; "
            "  Abasic sites within G-quadruplex secondary structures; "
            "  ICL abasic-type lesions (psoralen ICLs in some studies); "
            "HUMAN DISEASE: "
            "  No established Mendelian syndrome; "
            "  Rare biallelic LOF reported in cancer patients; incomplete penetrance characterisation; "
            "MICE: "
            "  NEIL3 −/− mice: meiotic defects + elevated germ cell DNA damage; "
            "  No dramatic somatic cancer phenotype reported; "
            "PREVALENCE: no confirmed Mendelian disease"
        ),
        "disease_category": (
            "NEIL3 DEFICIENCY — G-QUADRUPLEX REPAIR / MEIOTIC SSB / ICL-BACKUP (poorly characterised): "
            "UNIQUE BIOLOGY: "
            "  G-quadruplex (G4) DNA: guanine-rich sequences (telomeres, promoters, gene bodies) form G4 structures; "
            "  G4 lesions: oxidative damage within G4 → hydantoin products (Sp, Gh); "
            "  NEIL3 = primary glycosylase for hydantoin products in G4 context; "
            "  G4 biology = cancer biology: G4 stabilisers (pyridostatin) being developed as anticancer agents; "
            "ICL BACKUP PATHWAY: "
            "  Canonical ICL repair: FA pathway (FANCD2-I clamp + XPF-ERCC1 incision); "
            "  NEIL3 backup: directly unhooks psoralen ICL-abasic lesions at replication fork; "
            "  NEIL3 + FA = redundant ICL repair pathways; "
            "  NEIL3 LOF alone may be tolerated; NEIL3 + FANCD2 double LOF = catastrophic (speculative); "
            "MEIOTIC FUNCTION: "
            "  NEIL3 processes meiotic programmed SSBs (SPO11-generated DSB intermediates?); "
            "  Male infertility possible in biallelic NEIL3 LOF (germ cell DNA damage); "
            "HUMAN PHENOTYPE: "
            "  Insufficient data; possibly male infertility + cancer predisposition; "
            "  Research only currently"
        ),
        "disease_pathway": (
            "NEIL3 GLYCOSYLASE — G-QUADRUPLEX ABASIC-SITE AND ICL UNHOOKING: "
            "G-QUADRUPLEX (G4) LESION PROCESSING: "
            "STEP 1 — G4 DAMAGE FORMATION: "
            "  ROS oxidise guanine in G4 structures → spiroiminodihydantoin (Sp) or guanidinohydantoin (Gh); "
            "  Sp/Gh in G4: highly mutagenic; miscodes as adenine or cytosine; "
            "STEP 2 — NEIL3 RECOGNITION: "
            "  NEIL3 is unique: can access damaged bases within G4 secondary structure (other glycosylases fail); "
            "  NEIL3 melts/resolves G4 conformationally → extracts Sp/Gh; "
            "STEP 3 — GLYCOSYLASE + β-δ ELIMINATION: "
            "  N-terminal Pro residue Schiff base; β-δ elimination; 3'-phosphate; "
            "  Same mechanism as NEIL1/NEIL2 but on G4 substrate; "
            "ICL UNHOOKING PATHWAY: "
            "STEP 4 — REPLICATION FORK STALLING: "
            "  Replication fork encounters ICL → stalls; CMG helicase bypasses but leading strand blocked; "
            "STEP 5 — NEIL3 UNHOOKING: "
            "  NEIL3 cleaves N-glycosidic bond of abasic-ICL adducts → generates AP site on one strand; "
            "  Effectively unhooks one strand of ICL without dual incision (unlike FA + XPF-ERCC1 mechanism); "
            "  ZRANB3 (fork regression helicase) collaborates with NEIL3 at ICL replication forks; "
            "NEIL3 LOF → "
            "  G4 hydantoin lesions persist → G4 region mutagenesis → promoter/telomere instability → "
            "  ICL repair partially compromised (FA pathway compensates normally)"
        ),
        "pathognomonic": (
            "NEIL3 DIAGNOSTIC (investigational — no clinical syndrome established): "
            "GENETIC TESTING: "
            "  NGS panel (NEIL3 if included); biallelic LOF variants required; "
            "  Functional assay: Sp/Gh glycosylase activity on G4 substrates (research only); "
            "  ICL sensitivity assay (DEB/MMC chromosomal breakage): may be normal (NEIL3 alone); "
            "WHEN TO CONSIDER: "
            "  Male infertility + cancer predisposition + no established diagnosis; "
            "  Research protocol — biallelic NEIL3 LOF in young cancer patient; "
            "TUMOUR SIGNATURE: "
            "  G:C → T:A transversions at G4 motifs (predicted; not established); "
            "  Telomere instability; "
            "IMPORTANT: "
            "  NEIL3 = research gene; not actionable clinically in current guidelines; "
            "  Standard cancer screening per age/FH; "
            "  Consider fertility assessment in young biallelic males"
        ),
        "treatment": (
            "NEIL3 BIALLELIC — MANAGEMENT (empirical/investigational): "
            "NO ESTABLISHED GUIDELINE: "
            "FERTILITY: "
            "  Male fertility assessment (semen analysis, sperm DNA fragmentation) if biallelic confirmed; "
            "  Cryopreservation: consider before any gonadotoxic therapy; "
            "CANCER SURVEILLANCE (empirical): "
            "  Standard screening per age/sex guidelines; "
            "  If G4-rich cancer concern: lung + colorectal + brain screening from age 40; "
            "RESEARCH: "
            "  Registry enrolment + WGS of any tumour; "
            "  NEIL3 −/− animal model data; "
            "G4-TARGETED THERAPEUTICS (translational context): "
            "  G4 stabilisers (PDS, pyridostatin): sensitise NEIL3-deficient cancer cells (preclinical); "
            "  Clinical trial eligibility if cancer diagnosis; "
            "CASCADE TESTING: "
            "  AR gene: siblings 25% risk; parents carriers; "
            "  No actionable intervention for carriers currently"
        ),
    },
    {
        "gene": "UNG",
        "seed_base": 2748,
        "protein": (
            "UNG -- 12q24.11 AR -- 304aa -- "
            "Uracil-DNA-Glycosylase-"
            "35kDa-Nuclear-UNG2-Mitochondrial-UNG1-Uracil-Remover-AID-Substrate-HIGM5-CSR-"
            "OMIM-Gene-191525-Disease-Hyper-IgM-Syndrome-Type-5-608106"
        ),
        "locus": "12q24.11",
        "protein_size": "304 aa / 35 kDa (UNG isoforms: UNG2 nuclear, UNG1 mitochondrial; removes uracil from DNA; required for AID-mediated class-switch recombination and somatic hypermutation)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE (biallelic LOF → Hyper-IgM syndrome type 5, HIGM5); "
            "BIALLELIC DISEASE (HIGM5): "
            "  Recurrent infections: sinopulmonary, opportunistic; "
            "  Low IgG, low IgA, low IgE — but NORMAL IgM (class-switch recombination fails); "
            "  Normal or elevated IgM: CSR requires UNG to remove uracil introduced by AID; "
            "  Absent memory B cells; impaired somatic hypermutation (SHM) fidelity; "
            "  Phenotype onset: infancy/early childhood (recurrent infections); "
            "MECHANISM — CSR REQUIRES UNG: "
            "  AID (Activation-Induced Cytidine Deaminase) converts cytosine → uracil in switch regions; "
            "  UNG removes uracil → abasic site → strand break → CSR (IgM → IgG/IgA/IgE switching); "
            "  UNG LOF → uracil not removed → no strand break → CSR fails → only IgM produced; "
            "DISTINCTION FROM OTHER HIGM SYNDROMES: "
            "  HIGM1 (CD40L deficiency, XLR): CD40L absent — affects both CSR + other T-cell functions; "
            "  HIGM2 (AID/AICDA deficiency): AID absent → no CSR + abnormal SHM; "
            "  HIGM5 (UNG): AID NORMAL, CD40L NORMAL — post-AID defect; "
            "  Absent memory B cells: distinguishes HIGM5 from hypogammaglobulinaemia alone; "
            "ISOFORMS: "
            "  UNG1 (mitochondrial): repairs uracil in mitochondrial DNA (from cytosine deamination); "
            "  UNG2 (nuclear): replicative BER + CSR; AID-mediated uracil removal; "
            "PREVALENCE: rare; fewer than 50 families worldwide reported"
        ),
        "disease_category": (
            "HYPER-IgM SYNDROME TYPE 5 (HIGM5, OMIM 608106): "
            "IMMUNOLOGICAL PHENOTYPE: "
            "  Recurrent bacterial infections: sinusitis, pneumonia, otitis media; "
            "  Cryptosporidium parvum: GI + biliary tract (biliary cryptosporidiosis) — opportunistic; "
            "  Pneumocystis jirovecii pneumonia (PJP): increased risk; "
            "  LOW IgG + LOW IgA + LOW IgE; "
            "  NORMAL or ELEVATED IgM (class-switch failure → IgM accumulates); "
            "  Absent switched memory B cells (CD27+IgD-): hallmark of CSR failure; "
            "  IgM-secreting plasma cells: intact (HIGM5 produces IgM normally); "
            "  Autoimmune: neutropenia, haemolytic anaemia (less common than in CD40L HIGM1); "
            "LIVER/GI: "
            "  Sclerosing cholangitis + biliary cryptosporidiosis: serious GI complication (as in HIGM1); "
            "  Liver disease: cholestasis → cirrhosis if Cryptosporidium untreated; "
            "NEUROLOGICAL: "
            "  CNS Cryptosporidium: rare; "
            "  Generally better neurological outcome than HIGM1 (which has T-cell co-morbidity); "
            "CANCER RISK: "
            "  Lymphoma risk: elevated (as in most primary immunodeficiencies); "
            "  SHM impairment may affect antibody diversity and autoimmune protection; "
            "SOMATIC HYPERMUTATION (SHM): "
            "  SHM quality altered: UNG removes uracils in variable regions; "
            "  Reduced G:C mutations in SHM (reflects less UNG-dependent uracil processing)"
        ),
        "disease_pathway": (
            "UNG IN CLASS-SWITCH RECOMBINATION (CSR) AND BASE EXCISION REPAIR: "
            "NORMAL CSR PATHWAY: "
            "STEP 1 — AID DEAMINATES SWITCH REGIONS: "
            "  Activation-Induced Cytidine Deaminase (AID/AICDA) deaminates cytosine → uracil "
            "    in single-stranded DNA of switch regions (Sμ, Sγ, Sα, Sε); "
            "  Multiple uracils introduced in switch regions on both strands; "
            "STEP 2 — UNG REMOVES URACIL → ABASIC SITE: "
            "  UNG2 (nuclear isoform) excises uracil → AP site; "
            "  AP site → APE1 cleaves → single-strand nick; "
            "  Multiple nicks on both strands → double-strand break in switch region; "
            "STEP 3 — DSB + NHEJ → CLASS SWITCH: "
            "  DSBs in Sμ + downstream switch region (e.g., Sγ1) → NHEJ joins → "
            "  Ig heavy chain constant region changed: IgM Cμ replaced by IgG Cγ1 → "
            "  B cell now secretes IgG1 (class switched); "
            "UNG LOF (HIGM5): "
            "  AID still deaminates cytosine → uracil (AID intact); "
            "  Uracil NOT removed (no UNG) → no AP site → no nick → no DSB → "
            "  Switch regions intact → NHEJ cannot join Sμ to downstream region → "
            "  IgM stays (no CSR) → HIGM5 phenotype; "
            "UNG IN REPLICATIVE BER: "
            "  During DNA replication: dUMP occasionally incorporated opposite dA; "
            "  UNG removes uracil → AP site → Pol β fills with C → G:C restored; "
            "  Also removes uracil from C deamination in non-replicating DNA; "
            "UNG IN MITOCHONDRIA (UNG1): "
            "  mtDNA high cytosine deamination rate (mitochondria lack nucleotide excision repair); "
            "  UNG1 = principal mitochondrial uracil glycosylase"
        ),
        "pathognomonic": (
            "HIGM5 (UNG DEFICIENCY) DIAGNOSTIC APPROACH: "
            "IMMUNOLOGICAL WORKUP (critical): "
            "  SERUM IMMUNOGLOBULINS: Low IgG + Low IgA + Low IgE + Normal/Elevated IgM; "
            "  This pattern = class-switch defect; "
            "  MEMORY B CELLS (flow cytometry): CD27+IgD- (switched memory) ABSENT or markedly reduced; "
            "  CD27+IgD+ (marginal zone / IgM memory): preserved or elevated; "
            "  NAIVE B CELLS (CD27-IgD+): present (B cell development intact); "
            "DDx FROM OTHER HIGM SYNDROMES: "
            "  CD40L (HIGM1, XLR): absent CD40L on activated T cells; "
            "  AID/AICDA (HIGM2): absent AID on activated B cells; germinal centres absent; "
            "  CD40 (HIGM3, AR): CD40 absent on B cells; "
            "  UNG (HIGM5, AR): AID NORMAL + CD40L NORMAL + no germinal centre defect; "
            "  Test order: CD40L first (most common) → AID → CD40 → UNG; "
            "GERMLINE TESTING: "
            "  UNG biallelic sequencing + MLPA; "
            "  Two pathogenic LOF alleles required for HIGM5; "
            "FUNCTIONAL ASSAY: "
            "  CSR assay: in-vitro CD40L/IL-4 stimulated B cells → flow for IgG1/IgE isotype switching; "
            "  HIGM5 B cells: fail to switch in CSR assay (UNG required); "
            "  UNG enzymatic activity in nuclear extracts: absent in HIGM5 (research confirmation); "
            "CLINICAL DIAGNOSIS TRIGGER: "
            "  Infant/child with recurrent bacterial infections + normal IgM + low IgG/IgA/IgE"
        ),
        "treatment": (
            "HIGM5 (UNG BIALLELIC) MANAGEMENT: "
            "IMMUNOGLOBULIN REPLACEMENT (CORNERSTONE): "
            "  Intravenous IgG (IVIg) 400-600 mg/kg every 3-4 weeks; "
            "  OR Subcutaneous IgG (SubQ) weekly maintenance; "
            "  Goal: IgG trough > 500-800 mg/dL (adjust to clinical response); "
            "  IVIG does NOT correct CSR failure — it replaces missing IgG; "
            "  Continue lifelong (no spontaneous CSR recovery); "
            "INFECTION PROPHYLAXIS: "
            "  Co-trimoxazole (TMP-SMX): PJP prophylaxis from diagnosis; "
            "  Azithromycin prophylaxis: against MAC and respiratory organisms; "
            "  Cryptosporidium prevention: avoid untreated water; bottled/boiled water; "
            "  Cryptosporidium treatment: nitazoxanide (modest efficacy); no fully curative agent; "
            "HSCT (HAEMATOPOIETIC STEM CELL TRANSPLANT): "
            "  Potentially curative for HIGM5 (corrects CSR permanently); "
            "  Indications: severe/progressive disease, failure of IVIg management; "
            "  Outcomes: improving with reduced-intensity conditioning; "
            "LIVE VACCINES: ABSOLUTELY CONTRAINDICATED; "
            "  Inactivated vaccines: permitted + recommended (pneumococcal, meningococcal, influenza); "
            "MONITORING: "
            "  6-monthly: Ig levels, CBC, LFTs; "
            "  Annual: liver USS (biliary disease); LFTs; "
            "  Periodic: lymphocyte phenotyping; immunoglobulin production capacity assessment; "
            "DISTINCTION FROM HIGM1: "
            "  HIGM5 does NOT require T-cell targeted interventions (CD40L pathway intact); "
            "  B-cell biology: directed at B-cell replacement (IVIg/HSCT), not T-cell therapy"
        ),
    },
    {
        "gene": "MPG",
        "seed_base": 2749,
        "protein": (
            "MPG -- 16p13.3 AR/AD-uncertain -- 298aa -- "
            "Methylpurine-DNA-Glycosylase-"
            "33kDa-Monofunctional-3-Methyladenine-7-Methylguanine-1N6-Ethenoadenine-"
            "OMIM-Gene-156565-Disease-No-Established-Syndrome-Pharmacogenomic"
        ),
        "locus": "16p13.3",
        "protein_size": "298 aa / 33 kDa (monofunctional glycosylase — glycosylase only, NO AP-lyase; removes 3-methyladenine, 7-methylguanine, 1,N6-ethenoadenine, hypoxanthine)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE/DOMINANT UNCERTAIN (no established Mendelian disease syndrome); "
            "MPG = PHARMACOGENOMIC RELEVANCE primarily: "
            "  TMZ (temozolomide) methylates guanine → O6-methylguanine (MGMT target) AND "
            "  N7-methylguanine + N3-methyladenine (MPG targets); "
            "  MPG OVEREXPRESSION in cancer cells → removes 3-MeA faster → less cytotoxic SSBs → "
            "    TMZ resistance; "
            "  Conversely, MPG low expression + TMZ → 3-MeA accumulates → APE1 creates SSBs → "
            "    replication fork collapse → cell death (cytotoxicity); "
            "  MPG inhibition as cancer sensitisation strategy (research); "
            "HUMAN GERMLINE LOF: "
            "  No established cancer syndrome attributable to germline MPG LOF; "
            "  Alkylating agents (ENU, MNU) hypersensitivity expected in biallelic MPG LOF; "
            "SUBSTRATES: "
            "  3-methyladenine (3-MeA): replication-blocking; most cytotoxic alkylation product; "
            "  7-methylguanine (7-MeG): abundant but less cytotoxic; MPG removes it anyway; "
            "  1,N6-ethenoadenine (εA): formed by lipid peroxidation products (vinyl chloride, ethanol); "
            "  Hypoxanthine: deamination product of adenine; "
            "MICE: "
            "  MPG −/− mice: hypersensitive to alkylating agents (ENU, MMS); "
            "  Elevated 3-MeA after alkylation → more replication blocks + SSBs (paradoxically); "
            "  MPG OVEREXPRESSION mice: also impaired (AP site accumulation if APE1 insufficient); "
            "PREVALENCE: no confirmed Mendelian human disease"
        ),
        "disease_category": (
            "MPG DEFICIENCY — ALKYLATED BASE REPAIR / PHARMACOGENOMICS (no established syndrome): "
            "ALKYLATED BASE SUBSTRATES: "
            "  3-methyladenine (3-MeA): blocks DNA polymerase → replication arrest → cytotoxicity; "
            "  7-methylguanine (7-MeG): most abundant methylation product; MPG processes efficiently; "
            "  O2-methylthymine: minor substrate; "
            "  1,N6-ethenoadenine (εA): vinyl chloride/lipid peroxidation product; "
            "  Hypoxanthine: deamination of adenine; miscoding → A:T → G:C transitions; "
            "MPG IN ALKYLATING AGENT CYTOTOXICITY: "
            "  Alkylating chemotherapy (TMZ, BCNU, dacarbazine, cyclophosphamide): methylates/alkylates DNA; "
            "  MPG removes alkylated bases → AP site; "
            "  IF APE1 OVERWHELMED → AP sites accumulate → SSBs → DSBs → cell death; "
            "  MPG overexpression in glioma cells → more AP site generation → paradoxical sensitisation; "
            "  MPG overexpression + PARP inhibition → synthetic lethality with TMZ (cancer-specific); "
            "MPGI (1,N6-ETHENOADENINE): "
            "  Ethanol → acetaldehyde → lipid peroxidation → 4-hydroxynonenal → εA in DNA; "
            "  Hepatocellular carcinoma risk from ethanol: partly εA-mediated + MPG repair; "
            "  Vinyl chloride (industrial): also produces εA → hepatic angiosarcoma historically; "
            "GERMLINE MPG LOF PHENOTYPE: "
            "  Hypersensitivity to alkylating agents (predicted; not well-documented in humans); "
            "  Enhanced TMZ cytotoxicity in tumour + normal tissue (dual effect); "
            "  No polyposis, no established cancer predisposition syndrome"
        ),
        "disease_pathway": (
            "MPG MONOFUNCTIONAL GLYCOSYLASE — ALKYLATED BASE REMOVAL: "
            "STEP 1 — ALKYLATION DAMAGE RECOGNITION: "
            "  MPG recognises bulged/destabilised bases: 3-MeA, 7-MeG, εA by minor groove contacts; "
            "  Base-flipping: rotates damaged base into active-site pocket; "
            "  Intercalating residues stabilise abasic DNA; "
            "STEP 2 — N-GLYCOSIDIC BOND HYDROLYSIS (MONOFUNCTIONAL): "
            "  Glu125 activates water molecule as nucleophile; "
            "  Acid-base catalysis → hydrolysis of N-glycosidic bond → base released; "
            "  Product: abasic (AP) site + free alkylated base; "
            "  NO AP-lyase activity: MPG is monofunctional (unlike OGG1/NTHL1/NEILs); "
            "  AP site left for APE1 to process; "
            "STEP 3 — APE1 PROCESSING: "
            "  APE1 cleaves 5' of AP site → 5'-dRP flap + 3'-OH; "
            "  Pol β β-lyase removes 5'-dRP; "
            "  Pol β gap fill + ligation (short-patch BER); "
            "STEP 4 — LONG-PATCH ALTERNATIVE: "
            "  If 3-MeA generates clustered damage → long-patch BER with Pol δ/ε + FEN1 + PCNA; "
            "TMZ PHARMACOLOGY OVERLAP: "
            "  TMZ methylates N7-G (70%), N3-A (9%), O6-G (5-6%); "
            "  O6-MeG: MGMT repairs (predicts TMZ response in glioma); "
            "  N7-MeG + N3-MeA: MPG repairs; "
            "  If MPG fast + APE1 slow → AP site accumulates → SSB → toxicity in normal tissue too; "
            "  MPG INHIBITORS (PF-04523566, TRC102): block MPG → 3-MeA persists → enhanced TMZ cytotoxicity"
        ),
        "pathognomonic": (
            "MPG — PHARMACOGENOMIC ASSESSMENT (no established germline syndrome diagnostic): "
            "PHARMACOGENOMIC CONTEXT: "
            "  Patient receiving TMZ-based chemotherapy (glioblastoma, melanoma, lymphoma): "
            "  MPG EXPRESSION LEVEL (tumour IHC or RNA): "
            "    High MPG → faster 3-MeA/N7-MeG removal → may reduce TMZ efficacy; "
            "    Low MPG → 3-MeA persists → more AP sites → more SSBs → TMZ more toxic; "
            "  MPG + MGMT dual assessment: "
            "    MGMT methylated (low MGMT) + low MPG → maximum TMZ cytotoxicity (ideal); "
            "    MGMT unmethylated (high MGMT) + high MPG → TMZ resistance (worst case); "
            "GERMLINE TESTING (research): "
            "  Biallelic MPG LOF: germline sequencing; no clinical guideline indication; "
            "  Functional assay: alkylated base glycosylase activity on 3-MeA substrate; "
            "  Comet assay post-alkylating agent exposure (cell hypersensitivity); "
            "CLINICAL PHENOTYPE OF MPG LOF (expected): "
            "  Enhanced alkylating agent sensitivity (chemo-sensitivity); "
            "  Occupational: vinyl chloride exposure → εA accumulation → hepatic risk; "
            "  Alcohol-related liver disease: εA accumulation in hepatocytes; "
            "WHEN TO SUSPECT (empirical): "
            "  Unusually severe toxicity to standard alkylating agent chemotherapy; "
            "  Young patient with hepatic angiosarcoma + vinyl chloride occupational history; "
            "  Research protocol only; not clinically actionable currently"
        ),
        "treatment": (
            "MPG — MANAGEMENT (empirical/pharmacogenomic): "
            "NO ESTABLISHED CLINICAL GUIDELINE FOR GERMLINE MPG LOF: "
            "PHARMACOGENOMIC CONTEXT — TMZ THERAPY: "
            "  If MPG overexpressed in tumour (high MPG): "
            "    Consider MPG inhibitor clinical trial if available; "
            "    Assess MGMT methylation status (concurrent); "
            "  If MPG LOF (germline): "
            "    Alkylating agent chemotherapy: may be more toxic at standard doses; "
            "    Consider dose adjustment if severe myelosuppression or organ toxicity; "
            "  MPG inhibitors (TRC102): in clinical trials with alkylating agents; "
            "    Mechanism: block MPG → 3-MeA persists → APE1 creates more SSBs → sensitisation; "
            "VINYL CHLORIDE / OCCUPATIONAL: "
            "  MPG LOF + vinyl chloride exposure: HIGH RISK for εA → hepatic angiosarcoma; "
            "  Workplace: eliminate vinyl chloride exposure immediately; "
            "  Liver surveillance: annual AFP + USS if prior vinyl chloride exposure; "
            "ALCOHOL-RELATED: "
            "  MPG LOF + heavy alcohol: εA accumulation in liver; "
            "  Counsel complete alcohol abstinence; "
            "  Liver monitoring: annual LFTs + USS; "
            "GENERAL CANCER SCREENING: "
            "  Standard per age/sex guidelines (no MPG-specific protocol); "
            "  Enhance if occupational alkylating exposure history; "
            "CASCADE TESTING: "
            "  If biallelic confirmed (AR): siblings 25% risk; "
            "  Pharmacogenomic counselling if family member requires alkylating chemotherapy"
        ),
    },
]


def _make_patients(gene_data: dict) -> list:
    rng = random.Random(gene_data["seed_base"])
    gene = gene_data["gene"]
    patients = []
    for i in range(40):
        age = rng.randint(18, 82)
        sex = rng.choice(["F", "F", "M", "M"])  # balanced cohort (polyposis + immunodeficiency)

        # BER-specific phenotype probabilities per gene
        if gene == "MUTYH":
            polyp_burden = rng.randint(40, 110) if rng.random() < 0.80 else rng.randint(5, 40)
            crc = rng.random() < 0.62        # biallelic MAP CRC risk 50-80%
            breast = rng.random() < 0.08     # slight increase
            endometrial = rng.random() < 0.05
            urothelial = rng.random() < 0.05
            higm = False
            immunodef = False
            alkylated_sensitivity = False
            colectomy = rng.random() < 0.45
            aspirin_use = rng.random() < 0.58
        elif gene == "OGG1":
            polyp_burden = rng.randint(0, 5)
            crc = rng.random() < 0.12        # modest elevated (no established MAP)
            breast = rng.random() < 0.12
            endometrial = rng.random() < 0.05
            urothelial = rng.random() < 0.05
            higm = False
            immunodef = False
            alkylated_sensitivity = False
            colectomy = False
            aspirin_use = rng.random() < 0.10
        elif gene == "NTHL1":
            polyp_burden = rng.randint(10, 60) if rng.random() < 0.75 else rng.randint(1, 10)
            crc = rng.random() < 0.55        # NAP CRC risk ~60-70%
            breast = rng.random() < 0.35     # NAP breast risk 30-40%
            endometrial = rng.random() < 0.32
            urothelial = rng.random() < 0.20
            higm = False
            immunodef = False
            alkylated_sensitivity = False
            colectomy = rng.random() < 0.38
            aspirin_use = rng.random() < 0.40
        elif gene == "NEIL1":
            polyp_burden = rng.randint(0, 8)
            crc = rng.random() < 0.10
            breast = rng.random() < 0.08
            endometrial = rng.random() < 0.05
            urothelial = rng.random() < 0.04
            higm = False
            immunodef = False
            alkylated_sensitivity = rng.random() < 0.20
            colectomy = False
            aspirin_use = rng.random() < 0.08
        elif gene == "NEIL2":
            polyp_burden = rng.randint(0, 5)
            crc = rng.random() < 0.10
            breast = rng.random() < 0.06
            endometrial = rng.random() < 0.04
            urothelial = rng.random() < 0.05
            higm = False
            immunodef = False
            alkylated_sensitivity = rng.random() < 0.15
            colectomy = False
            aspirin_use = rng.random() < 0.06
        elif gene == "NEIL3":
            polyp_burden = rng.randint(0, 5)
            crc = rng.random() < 0.08
            breast = rng.random() < 0.05
            endometrial = rng.random() < 0.04
            urothelial = rng.random() < 0.04
            higm = False
            immunodef = False
            alkylated_sensitivity = rng.random() < 0.12
            colectomy = False
            aspirin_use = rng.random() < 0.05
        elif gene == "UNG":
            polyp_burden = 0
            crc = rng.random() < 0.04
            breast = rng.random() < 0.04
            endometrial = rng.random() < 0.03
            urothelial = rng.random() < 0.03
            higm = rng.random() < 0.85       # biallelic UNG → HIGM5
            immunodef = higm
            alkylated_sensitivity = False
            colectomy = False
            aspirin_use = False
        else:  # MPG
            polyp_burden = rng.randint(0, 4)
            crc = rng.random() < 0.05
            breast = rng.random() < 0.05
            endometrial = rng.random() < 0.04
            urothelial = rng.random() < 0.04
            higm = False
            immunodef = False
            alkylated_sensitivity = rng.random() < 0.55    # TMZ/alkylator sensitivity
            colectomy = False
            aspirin_use = rng.random() < 0.05

        patients.append({
            "id": f"{gene}-{i+1:03d}",
            "age": age,
            "sex": sex,
            "gene": gene,
            "polyp_burden": int(polyp_burden),
            "colorectal_cancer": crc,
            "breast_cancer": breast,
            "endometrial_cancer": endometrial,
            "urothelial_cancer": urothelial,
            "hyper_igm_syndrome": higm,
            "immunodeficiency": immunodef,
            "alkylated_base_sensitivity": alkylated_sensitivity,
            "colectomy_performed": colectomy,
            "aspirin_use": aspirin_use,
            "genetic_counselling_received": rng.random() < 0.88,
            "cascade_testing_completed": rng.random() < 0.65,
        })
    return patients


def generate_overview():
    summary = []
    for gene_data in ATLAS_GENES:
        patients = _make_patients(gene_data)
        n = len(patients)
        summary.append({
            "gene": gene_data["gene"],
            "locus": gene_data["locus"],
            "protein": gene_data["protein"],
            "protein_size": gene_data["protein_size"],
            "inheritance": gene_data["inheritance"][:300] + "...",
            "n_patients": n,
            "pct_colorectal_cancer": round(100 * sum(p["colorectal_cancer"] for p in patients) / n),
            "pct_breast_cancer": round(100 * sum(p["breast_cancer"] for p in patients) / n),
            "pct_endometrial_cancer": round(100 * sum(p["endometrial_cancer"] for p in patients) / n),
            "pct_urothelial_cancer": round(100 * sum(p["urothelial_cancer"] for p in patients) / n),
            "pct_hyper_igm": round(100 * sum(p["hyper_igm_syndrome"] for p in patients) / n),
            "pct_immunodeficiency": round(100 * sum(p["immunodeficiency"] for p in patients) / n),
            "pct_alkylated_sensitivity": round(100 * sum(p["alkylated_base_sensitivity"] for p in patients) / n),
            "pct_colectomy": round(100 * sum(p["colectomy_performed"] for p in patients) / n),
            "pct_aspirin": round(100 * sum(p["aspirin_use"] for p in patients) / n),
            "avg_polyp_burden": round(sum(p["polyp_burden"] for p in patients) / n, 1),
        })
    return {
        "atlas": "Hereditary-BER-Atlas",
        "subtitle": "Base Excision Repair — 8-Gene Clinical Reference",
        "genes": ["MUTYH", "OGG1", "NTHL1", "NEIL1", "NEIL2", "NEIL3", "UNG", "MPG"],
        "seeds": "2742-2749",
        "total_patients": 320,
        "summary": summary,
        "pathway_categories": [
            {
                "pathway": "Adenine Opposite 8-oxoG / MAP Polyposis (MUTYH + OGG1)",
                "genes": ["MUTYH", "OGG1"],
                "note": (
                    "MUTYH: removes adenine mismatch opposite 8-oxoG (OG:A → OG:C); biallelic = MAP "
                    "(40-100 adenomas, CRC 50-80%, MSS); monoallelic = modest 1.5-2x CRC risk only; "
                    "OGG1: removes 8-oxoG from OG:C pair (complementary to MUTYH); Lys326Gln common "
                    "polymorphism NOT pathogenic; biallelic OGG1 LOF: no established syndrome"
                ),
            },
            {
                "pathway": "NTHL1-NAP / Oxidised Pyrimidine / Formamidopyrimidine (NTHL1 + NEIL1 + NEIL2 + NEIL3)",
                "genes": ["NTHL1", "NEIL1", "NEIL2", "NEIL3"],
                "note": (
                    "NTHL1: biallelic Gln90Ter → NAP (polyposis + breast + endometrial + urothelial — "
                    "distinct from MAP); Signature 30 C>T; no IHC marker; "
                    "NEIL1: formamidopyrimidines + oxidised pyrimidines; metabolic syndrome in mice; TC-BER; "
                    "NEIL2: transcribed-strand-specific BER; oxidised cytosines (5-OHU); few human reports; "
                    "NEIL3: G-quadruplex unhooking; ICL backup to FA pathway; meiotic SSB repair"
                ),
            },
            {
                "pathway": "UNG / Hyper-IgM Type 5 / Class-Switch Recombination Defect",
                "genes": ["UNG"],
                "note": (
                    "UNG biallelic LOF → HIGM5: AID removes uracil → strand break → CSR; "
                    "UNG LOF → no uracil removal → no DSB → no CSR → IgM stays, IgG/IgA/IgE absent; "
                    "CD40L NORMAL + AID NORMAL (DDx from HIGM1/HIGM2); absent switched memory B cells; "
                    "IVIG + PJP prophylaxis; HSCT curative; Cryptosporidium major risk"
                ),
            },
            {
                "pathway": "MPG / Alkylated Base Removal / TMZ Pharmacogenomics",
                "genes": ["MPG"],
                "note": (
                    "MPG: monofunctional glycosylase; removes 3-methyladenine (replication-blocking), "
                    "7-methylguanine, 1,N6-ethenoadenine (vinyl chloride/ethanol); no AP-lyase activity; "
                    "TMZ pharmacogenomics: MPG overexpression → TMZ resistance; MPG inhibitors in clinical trials; "
                    "No established Mendelian cancer syndrome — pharmacogenomics and occupational medicine focus"
                ),
            },
        ],
        "critical_distinctions": [
            "MUTYH BIALLELIC ≠ MONOALLELIC: biallelic MUTYH = MAP syndrome (40-100 adenomas, CRC 50-80%); monoallelic MUTYH alone = modest 1.5-2x CRC risk, NOT MAP; always report zygosity; compound heterozygous Tyr179Cys/Gly396Asp = classic biallelic MAP",
            "NTHL1-NAP ≠ MAP (MUTYH): NTHL1 biallelic → breast + endometrial + urothelial cancers IN ADDITION to CRC polyposis; MUTYH-MAP primarily CRC + duodenal; NTHL1-NAP has much broader extracolonic spectrum — manage breast/endometrial/urothelial in NAP",
            "NTHL1-NAP ≠ LYNCH: NTHL1-NAP tumours are MSS (NOT MSI-H); no MMR IHC loss; molecular testing essential — Lynch excluded by MSI/MMR-IHC but NAP remains; no NTHL1 IHC available clinically (blind spot)",
            "UNG (HIGM5) ≠ CD40L (HIGM1): HIGM5 UNG LOF → CD40L NORMAL; CD40L HIGM1 XLR; distinguish: test CD40L first (more common); absent switched memory B-cells (CD27+IgD-) diagnostic of CSR defect; UNG + AID both present in HIGM5 (defect is downstream)",
            "OGG1 LYS326GLN (rs1052133) IS NOT PATHOGENIC: common polymorphism (15-30% allele frequency); DO NOT report as pathogenic variant; modest cancer associations at population level do not warrant clinical intervention; standard management only",
            "MPG HAS NO CANCER SYNDROME: no established Mendelian cancer predisposition for germline MPG LOF; MPG relevance is pharmacogenomic (TMZ response) and occupational (vinyl chloride → 1,N6-ethenoadenine); do not conflate with MAP/NAP polyposis genes",
            "NEIL1/NEIL2/NEIL3 — POORLY CHARACTERISED HUMAN DISEASE: no established Mendelian syndromes; evidence from mouse models + association studies; treat as research genes; biallelic LOF = register in research registry; standard cancer screening unless formal syndrome characterised",
            "MUTYH KRAS SIGNATURE: biallelic MAP CRC frequently shows KRAS c.34G>T (Gly12Cys) — transversion signature of 8-oxoG opposite adenine (OG:A mismatch); this somatic signature can raise suspicion for MAP in CRC patient before germline testing",
            "ASPIRIN IN MAP (NOT NAP/HIGM5/MPG): aspirin reduces polyp burden specifically in MUTYH-MAP (CaPP3); no equivalent evidence in NTHL1-NAP; aspirin in HIGM5 = standard GI protection only; MPG = not applicable",
            "COLECTOMY IN MAP NOT FAP: MAP polyposis is attenuated (40-100 adenomas typically); FAP (APC) has thousands; MAP colectomy timing guided by polyp burden + dysplasia; more conservative than FAP — subtotal colectomy with IRA preferred if rectal sparing possible",
        ],
    }


def generate_breakdown():
    genes_out = []
    for gene_data in ATLAS_GENES:
        patients = _make_patients(gene_data)
        n = len(patients)
        pct = lambda k: round(100 * sum(p[k] for p in patients) / n)
        avg_polyp = round(sum(p["polyp_burden"] for p in patients) / n, 1)
        genes_out.append({
            "gene": gene_data["gene"],
            "locus": gene_data["locus"],
            "protein_size": gene_data["protein_size"],
            "n_patients": n,
            "inheritance": gene_data["inheritance"],
            "disease_category": gene_data["disease_category"],
            "disease_pathway": gene_data["disease_pathway"],
            "pathognomonic": gene_data["pathognomonic"],
            "treatment": gene_data["treatment"],
            "pct_colorectal_cancer": pct("colorectal_cancer"),
            "pct_breast_cancer": pct("breast_cancer"),
            "pct_endometrial_cancer": pct("endometrial_cancer"),
            "pct_urothelial_cancer": pct("urothelial_cancer"),
            "pct_hyper_igm": pct("hyper_igm_syndrome"),
            "pct_immunodeficiency": pct("immunodeficiency"),
            "pct_alkylated_sensitivity": pct("alkylated_base_sensitivity"),
            "pct_colectomy": pct("colectomy_performed"),
            "pct_aspirin": pct("aspirin_use"),
            "avg_polyp_burden": avg_polyp,
            "patients": patients[:40],
        })
    return {"genes": genes_out}


def generate_definitions():
    return {
        "glossary": {
            "Base Excision Repair (BER)": (
                "Primary repair pathway for small base modifications: oxidised bases, alkylated bases, "
                "deaminated bases, and abasic sites; "
                "Steps: (1) glycosylase removes damaged base → AP site; "
                "(2) APE1 or AP-lyase cleaves backbone; "
                "(3) DNA polymerase β fills gap (short-patch) or Pol δ/ε fill (long-patch); "
                "(4) Ligase III/XRCC1 (short) or Ligase I (long) seals nick; "
                "Key genes: MUTYH, OGG1, NTHL1, NEIL1/2/3, UNG, MPG, APE1, XRCC1, LIG3"
            ),
            "8-oxoguanine (8-oxoG)": (
                "Major oxidative DNA lesion; guanine oxidised at C8 position by reactive oxygen species (ROS); "
                "Mutagenic: DNA polymerase δ inserts adenine opposite 8-oxoG (OG:A mispair) → G:C → T:A transversion; "
                "OGG1 removes 8-oxoG from OG:C (primary repair); "
                "MUTYH removes adenine from OG:A (post-replication correction); "
                "Accumulates in aged/oxidatively stressed cells; linked to cancer and ageing"
            ),
            "MUTYH-Associated Polyposis (MAP)": (
                "Autosomal recessive polyposis syndrome caused by biallelic MUTYH LOF; "
                "Phenotype: 40-100 colorectal adenomas, CRC 50-80% lifetime; extracolonic: duodenal adenomas; "
                "MSS CRC (not MSI-H — distinguish from Lynch); "
                "Key alleles: Tyr179Cys + Gly396Asp (compound heterozygous most common in NW Europeans); "
                "Monoallelic MUTYH: modest 1.5-2x CRC risk only — NOT MAP; "
                "Aspirin reduces polyp burden (CaPP3); colectomy if unmanageable burden/CRC"
            ),
            "NTHL1-Associated Polyposis (NAP)": (
                "Autosomal recessive polyposis syndrome caused by biallelic NTHL1 LOF; "
                "Phenotype: 5-50+ colorectal adenomas + CRC + BREAST + ENDOMETRIAL + UROTHELIAL cancer; "
                "Extracolonic spectrum broader than MAP — critical management distinction; "
                "Key allele: Gln90Ter (c.268C>T) founder Dutch/British; "
                "MSS CRC; NO NTHL1 IHC available; diagnosis requires germline testing; "
                "Signature 30 (C>T CpG/TCA context) in tumours; discovered 2015 — evolving guidelines"
            ),
            "Hyper-IgM Syndrome Type 5 (HIGM5)": (
                "AR immunodeficiency caused by biallelic UNG LOF; "
                "Phenotype: low IgG/IgA/IgE + normal/elevated IgM + absent switched memory B cells; "
                "Mechanism: AID deaminates C → U in switch regions; UNG must remove U → AP site → DSB → CSR; "
                "UNG LOF → no AP site → no DSB → CSR fails → only IgM produced; "
                "CD40L normal + AID normal (distinguish from HIGM1/HIGM2); "
                "IVIG replacement + PJP prophylaxis + Cryptosporidium prevention; HSCT curative"
            ),
            "Class-Switch Recombination (CSR)": (
                "B-cell recombination event changing Ig heavy chain constant region (IgM → IgG/IgA/IgE); "
                "Requires: AID (deaminates C→U in switch regions) + UNG (removes U → AP site) + APE1 (nick) → "
                "  DSBs in switch regions + NHEJ (joins Sμ to downstream switch region); "
                "UNG is ESSENTIAL for CSR: UNG LOF → CSR fails → HIGM5; "
                "AID LOF → also CSR fails + SHM impaired (HIGM2); "
                "Result of CSR: B cells produce IgG, IgA, or IgE instead of IgM"
            ),
            "Formamidopyrimidines (FapyAde, FapyGua)": (
                "Ring-opened purine oxidation products: adenine or guanine N7 alkylated → ring opens → formamidopyrimidine; "
                "FapyAde: formamidopyrimidine-adenine — principal NEIL1 substrate (also NTHL1); "
                "FapyGua: formamidopyrimidine-guanine — NEIL1 + OGG1 + NTHL1 substrates; "
                "Highly mutagenic: miscoding → A:T → T:A or G:C → T:A transversions; "
                "Accumulate under oxidative/alkylating conditions; "
                "β-δ AP-lyase (NEIL1/NEIL2/NEIL3) efficient at processing these lesions"
            ),
            "3-Methyladenine (3-MeA)": (
                "Cytotoxic alkylated base produced by alkylating agents (TMZ, ENU, MMS); "
                "Mechanism: N3 of adenine alkylated → bulky minor groove adduct → DNA polymerase blocked; "
                "Replication-blocking: most cytotoxic methylation product (unlike 7-MeG which is mutagenic); "
                "MPG (methylpurine-DNA glycosylase) removes 3-MeA → AP site → BER; "
                "MPG overexpression → faster 3-MeA removal → fewer SSBs → TMZ resistance; "
                "MPG inhibition → 3-MeA persists → more SSBs → more TMZ cytotoxicity (therapeutic strategy)"
            ),
            "Temozolomide (TMZ) Pharmacogenomics": (
                "Alkylating chemotherapy used in glioblastoma, melanoma, lymphoma; "
                "Methylation products: O6-MeG (MGMT substrate, 5-6%) + N7-MeG (70%) + N3-MeA (9%); "
                "MGMT promoter methylation status: predicts TMZ response (standard glioblastoma biomarker); "
                "MPG removes N7-MeG + N3-MeA → AP sites; "
                "MPG high expression in tumour → TMZ resistance (faster repair); "
                "MPG inhibitors (TRC102, PF-04523566) in clinical trials to sensitise tumours to TMZ; "
                "MGMT + MPG dual biomarker assessment under investigation"
            ),
            "1,N6-Ethenoadenine (εA)": (
                "Exocyclic DNA adduct formed from lipid peroxidation products (4-HNE) or vinyl chloride; "
                "Precursors: vinyl chloride (occupational carcinogen), ethanol → acetaldehyde → lipid peroxidation; "
                "Historical significance: vinyl chloride → hepatic angiosarcoma in PVC factory workers → εA role; "
                "Miscoding lesion: εA mispairs with Cyt or Gua → A:T → G:C transitions; "
                "MPG principal glycosylase for εA removal; "
                "AlkB dioxygenase (alkylation repair) also processes εA as alternate pathway"
            ),
            "G-Quadruplex (G4) DNA": (
                "Non-canonical secondary DNA structure formed in guanine-rich sequences (telomeres, promoters, rDNA); "
                "G4 = four guanines coordinated by monovalent cation (K+, Na+) → Hoogsteen H-bonds; "
                "Prevalent in oncogene promoters (MYC, VEGF, KRAS) + telomeres; "
                "Oxidative damage within G4: guanine → spiroiminodihydantoin (Sp) or guanidinohydantoin (Gh); "
                "NEIL3: only known glycosylase that accesses and repairs Sp/Gh within G4 structures; "
                "G4 stabilisers (pyridostatin, PDS): trap G4 → replication fork stalling → cancer cells sensitive; "
                "NEIL3-deficient cells: hypersensitive to G4 stabilisers (potential therapeutic window)"
            ),
            "Somatic Mutational Signature 30": (
                "COSMIC Mutational Signature 30: predominantly C>T transitions; "
                "Associated with NTHL1 biallelic LOF (NAP); "
                "Contextual enrichment: C>T at CpG (overlaps Signature 1/ageing) + TCA context; "
                "Mechanism: NTHL1 fails to remove oxidised cytosines (5-OHU, 5-OHC) → deamination → C>T; "
                "Distinct from MUTYH signature (G>T transversions) and APOBEC signature (C>T at TCA/TCT); "
                "Clinical utility: WGS tumour profiling → Signature 30 → suspect NTHL1 biallelic → germline testing"
            ),
            "Transcription-Coupled BER (TC-BER)": (
                "Sub-pathway of BER preferentially repairing oxidative damage on actively transcribed strand; "
                "Mechanism: RNA Pol II stalls at damaged base → CSB (ERCC6/Cockayne B) recruits BER glycosylase; "
                "Key enzymes: NEIL2 (transcribed-strand preferring) + NEIL1 (TC-BER + GGR); "
                "Evolutionary rationale: prioritise repair of coding sequences to maintain transcriptional fidelity; "
                "TC-BER deficiency → accumulation of transcriptional mutagenesis → premature ageing; "
                "Differs from TC-NER (which handles bulky helix-distorting lesions): TC-BER = small oxidative damage"
            ),
            "Cascade Testing in BER Polyposis Genes": (
                "AR genes (MUTYH, NTHL1): both alleles must be pathogenic for disease (biallelic); "
                "Proband's siblings: 25% risk of biallelic (carrier × carrier parents); "
                "Parents: obligate heterozygous carriers (phenotypically normal); "
                "Children of MAP/NAP proband: obligate monoallelic carrier (50% risk if partner is carrier); "
                "Carrier couple (two MUTYH carriers): 25% biallelic child risk per pregnancy; "
                "HIGM5 (UNG): AR — same 25% sibling risk; identify carriers for reproductive counselling; "
                "Monoallelic MUTYH: colonoscopy from age 40; monoallelic NTHL1: no established management"
            ),
        },
        "standards": [
            "Sieber OM et al. NEJM 2003: MUTYH biallelic mutations in multiple colorectal adenomas — first MAP paper",
            "Weren RD et al. Nature Genetics 2015: NTHL1 biallelic mutations cause NAP — first NAP discovery",
            "Iyama T & Wilson DM. DNA Repair 2013: Elements of BER in eukaryotes — comprehensive review",
            "Kavli B et al. EMBO J 2002: UNG2 in Ig gene diversification — CSR mechanism",
            "Rada C et al. Current Biology 2002: UNG-deficient mice impaired in class switch recombination",
            "Robertson AB et al. Cellular and Molecular Life Sciences 2009: NEIL1/NEIL2/NEIL3 — review",
            "Dou H et al. EMBO J 2003: Structural basis of NEIL1 bifunctional glycosylase/β-δ lyase",
            "Krokan HE & Bjoras M. Cold Spring Harbor Perspectives in Biology 2013: BER mechanisms review",
            "NCCN Guidelines: Hereditary Colorectal Cancer Syndromes v2.2025 — MAP/NAP management",
            "Church JM et al. Dis Colon Rectum 2022: MAP management consensus — colectomy and surveillance",
            "CaPP3 study: Lynch/polyposis aspirin RCT data relevant to MAP chemoprevention",
            "Schottker B et al. Int J Cancer 2020: OGG1 Lys326Gln meta-analysis — modest lung cancer association",
            "Tornaletti S & Hanawalt PC. Biochimie 1999: Transcription-coupled repair mechanisms",
            "Hu J et al. Molecular Cell 2015: NEIL3 role in G-quadruplex repair — key mechanistic paper",
            "Semlow DR et al. Molecular Cell 2016: Replication-coupled repair of ICLs by NEIL3 and FANCI-FANCD2",
        ],
    }
