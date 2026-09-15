"""Hereditary NER / XP Atlas — 8-Gene Nucleotide Excision Repair Reference
XPA-ERCC3-XPC-ERCC2-DDB2-ERCC4-ERCC5-POLH (XP-A through XP-G plus XP-Variant)
320 patients (8 x 40), seeds 2710-2717.
Endpoints: /api/hereditary-ner-xp-atlas/overview|breakdown|definitions
"""
import random

ATLAS_GENES = [
    {
        "gene": "XPA",
        "seed_base": 2710,
        "protein": (
            "XPA -- 9q22.33 AR -- 273aa -- XPA-Zinc-Finger-DNA-Damage-Verification-Protein-"
            "31kDa-Monomer-Core-NER-Assembly-Factor-GGR+TCR-Both-Subpathways-"
            "OMIM-Gene-611153-Disease-XP-A-278700"
        ),
        "locus": "9q22.33",
        "protein_size": "273 aa / 31 kDa (monomer; central NER scaffold zinc-finger protein)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF; "
            "XPA encodes a central scaffold protein of nucleotide excision repair (NER); "
            "XPA binds damaged DNA directly via a zinc-finger domain (C4 type, Cys-105/108/126/129); "
            "XPA simultaneously contacts ERCC1-XPF heterodimer, TFIIH, and RPA, "
            "positioning the dual-incision complex around the lesion; "
            "XPA LOF → BOTH GGR (global genome NER) and TCR (transcription-coupled NER) abolished; "
            "complete NER failure → CPD and 6-4PP lesions accumulate in ALL dividing cells; "
            "DISEASE SPECTRUM: "
            "  XP-A = most severe NER-defective XP; near-zero unscheduled DNA synthesis (UDS); "
            "  Japanese founder mutation: Arg228Ter (c.682C>T) — most common worldwide in XP-A; "
            "  De Sanctis-Cacchione syndrome: XP-A (severe) + severe neurodegeneration (historic term, now rarely used); "
            "PREVALENCE: ~1 in 1,000,000; highest in Japan (~1:22,000) and North Africa due to founder effects; "
            "XP as a whole: ~1 in 250,000 in USA/Europe; Japan 1:22,000"
        ),
        "disease_category": (
            "XERODERMA PIGMENTOSUM GROUP A (OMIM 278700); "
            "TRIAD OF XP (all groups): "
            "  1. UV HYPERSENSITIVITY: acute sunburn at 1-2 years from minimal UV; "
            "     freckling, lentigines, xerosis on UV-exposed skin; photophobia; "
            "     conjunctivitis, keratitis, corneal opacification; "
            "  2. SKIN CANCER: BCC + SCC + melanoma at median age 9 (vs 60 in general population); "
            "     10,000-fold increased cancer risk; >50% die of cancer before age 40 without UV protection; "
            "  3. NEURODEGENERATION (XP-A most severe): "
            "     progressive sensorineural hearing loss (SNHL); "
            "     microcephaly; cerebellar ataxia; spastic paraplegia; "
            "     areflexia (loss of deep tendon reflexes); "
            "     intellectual decline; dementia; choreoathetosis; "
            "MECHANISM OF NEURODEGENERATION: "
            "  Endogenous oxidative DNA lesions in post-mitotic neurons NOT repaired → neuronal apoptosis; "
            "  Transcription-blocking lesions → TCR failure → neuron-selective death; "
            "XP-A NEUROLOGICAL TIMELINE: "
            "  Loss of DTRs at 4-6 years; SNHL at 5-10 years; ataxia at 8-12 years; "
            "  wheelchair by 15-20 years; death 2nd-3rd decade without treatment"
        ),
        "disease_pathway": (
            "NUCLEOTIDE EXCISION REPAIR (NER) PATHWAY — GGR and TCR subpathways: "
            "STEP 1 — DAMAGE RECOGNITION: "
            "  GGR: XPC-RAD23B-CETN2 complex recognises helix distortion (indirect sensing); "
            "  TCR: stalled RNA Pol II triggers CSA-CSB recruitment; "
            "STEP 2 — XPA VERIFICATION + TFIIH RECRUITMENT: "
            "  XPA (GGR+TCR) verifies actual lesion chemistry → binds RPA, ERCC1-XPF, TFIIH; "
            "  XPA LOF: TFIIH recruited but cannot position dual-incision complex → NER fails; "
            "STEP 3 — TFIIH UNWINDING: "
            "  XPD (3'→5' helicase) + XPB (3'→5' ATPase) unwind ~30nt bubble; "
            "STEP 4 — DUAL INCISION: "
            "  ERCC4/XPF-ERCC1: cut 5' side (-25 to -15 nt from lesion); "
            "  ERCC5/XPG: cut 3' side (+2 to +8 nt from lesion); "
            "  Result: ~30nt oligonucleotide containing lesion is excised; "
            "STEP 5 — GAP FILL + LIGATION: "
            "  PCNA-Pol δ/ε/κ + RFC + RPA fill gap; ligase seals nick; "
            "XPA LOF: step 2 failure → steps 3-5 impossible → NO lesion excision"
        ),
        "pathognomonic": (
            "ACUTE SUNBURN AGE 1-2 YEARS FROM MINIMAL UV EXPOSURE (even cloudy day): "
            "  no other condition causes this degree of UV sensitivity in toddlers — "
            "  CARDINAL PRESENTATION in XP-A; "
            "UNSCHEDULED DNA SYNTHESIS (UDS) NEAR ZERO (0-4% of normal): "
            "  gold-standard assay — fibroblasts irradiated with UV, then [³H]thymidine incorporation "
            "  measured by autoradiography; XP-A: UDS 0-4% normal; "
            "COMPLEMENTATION GROUP ASSIGNMENT: "
            "  cell fusion restores UDS if different complementation groups; "
            "  XP-A specific: XPA cDNA correction; XPA gene sequencing; "
            "NEUROLOGICAL EXAM: "
            "  SNHL + absent DTRs + ataxia in UV-sensitive child = XP-A until proven otherwise; "
            "Japanese founder Arg228Ter: detectable by single-site PCR in at-risk populations; "
            "BRAIN MRI: diffuse cerebral + cerebellar atrophy, posterior fossa predominant; "
            "cortical thinning, white matter change in late-stage; "
            "SKIN BIOPSY: actinic keratoses, BCC, SCC pathology with UV signature mutation spectrum; "
            "C→T transitions at dipyrimidine sites (UV signature); CC→TT tandem mutations"
        ),
        "treatment": (
            "1. STRICT UV AVOIDANCE (MANDATORY, LIFE-LONG): "
            "   UV-protective clothing (UPF50+, long sleeves, gloves, hat with 360° brim); "
            "   UV-blocking window film (UV400); UV-blocking contact lenses; "
            "   UV-absorbing face shield; broadspectrum SPF50+ sunscreen (mineral preferred); "
            "   AVOID INDOOR UV sources (fluorescent lamps can emit UV — use LED); "
            "   Car windows filter UVB but NOT UVA — UV-film mandatory on all car windows; "
            "2. ANNUAL DERMATOLOGY SURVEILLANCE: "
            "   full-body skin examination every 3 months; "
            "   early excision of actinic keratoses; "
            "   aggressive excision of any BCC/SCC/melanoma; "
            "3. OPHTHALMOLOGY: "
            "   UV-blocking eyewear mandatory outdoors; "
            "   artificial tears; corneal protection; pterygium surveillance; "
            "4. NEUROLOGY: "
            "   hearing aids for SNHL (cochlear implants in advanced loss); "
            "   speech therapy; physiotherapy for ataxia/spasticity; "
            "   intrathecal baclofen for severe spasticity; "
            "5. CAPECITABINE (topical fluorouracil 5-FU): field cancerisation treatment; "
            "6. VISMODEGIB/SONIDEGIB: for multiple BCCs (hedgehog pathway inhibitor); "
            "7. XPA mRNA THERAPY (experimental 2024-2025): "
            "   mRNA replacement for hepatic delivery — preclinical; "
            "8. PHOTOSENSITISING DRUGS — ABSOLUTELY CONTRAINDICATED (see definitions)"
        ),
    },
    {
        "gene": "ERCC3",
        "seed_base": 2711,
        "protein": (
            "ERCC3 -- 2q21.3 AR -- 782aa -- XPB-TFIIH-3prime-5prime-ATPase-Helicase-Subunit-"
            "89kDa-TFIIH-Core-Component-NER+Transcription-Dual-Function-"
            "OMIM-Gene-133510-Disease-XP-B-610651-TTD-601675-CS-216400"
        ),
        "locus": "2q21.3",
        "protein_size": "782 aa / 89 kDa (TFIIH helicase 3'→5'; NER + basal transcription)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF; "
            "ERCC3 encodes XPB, the 3'→5' ATP-dependent helicase subunit of TFIIH; "
            "TFIIH is a 10-subunit complex with DUAL FUNCTIONS: "
            "  NER: uses XPD (5'→3') + XPB (3'→5') to unwind DNA around lesion for dual incision; "
            "  TRANSCRIPTION INITIATION: TFIIH phosphorylates RNA Pol II CTD (Ser5) at promoters; "
            "ERCC3 LOF → BOTH NER and RNA Pol II initiation impaired; "
            "PHENOTYPE SPECTRUM (allele-dependent): "
            "  Complete LOF: XP+CS (combined xeroderma pigmentosum + Cockayne syndrome) — embryonic lethal in animals; "
            "  Partial LOF (ATP-binding domain): TTD (trichothiodystrophy) — brittle hair, ichthyosis, no cancer; "
            "  Moderate LOF: XP-B alone (extremely rare; ~5 families worldwide 2026); "
            "DISEASE: ULTRAORPHAN — fewer than 30 XP-B patients reported globally; "
            "PREVALENCE: estimated 1 in 50,000,000+ (most cases may be embryonic lethal)"
        ),
        "disease_category": (
            "XERODERMA PIGMENTOSUM GROUP B / TRICHOTHIODYSTROPHY (OMIM 610651 / 601675); "
            "XP-B PHENOTYPIC SPECTRUM: "
            "  XP-B (rare): UV sensitivity + skin cancers + mild-moderate neurodegeneration; "
            "  XP/CS OVERLAP (most common ERCC3 phenotype): "
            "    XP features (UV sensitivity, skin cancer) PLUS "
            "    Cockayne syndrome features: growth failure, premature ageing, SNHL, "
            "    retinal degeneration, intracranial calcifications, 'cachectic dwarf' appearance; "
            "    intellectual disability; "
            "  TRICHOTHIODYSTROPHY (TTD): "
            "    brittle hair (PATHOGNOMONIC tiger-tail pattern on polarised light microscopy); "
            "    ichthyosis; intellectual disability; short stature; "
            "    NO skin cancer (reduced transcription → altered keratin/hair protein synthesis); "
            "MOLECULAR BASIS OF PHENOTYPE SPLIT: "
            "  Mutations affecting helicase domain → XP+cancer phenotype; "
            "  Mutations affecting protein stability/TFIIH incorporation → TTD phenotype; "
            "TFIIH DUAL ROLE EXPLAINS TTD: "
            "  TTD mutations reduce TFIIH level → impaired Pol II transcription of selenoproteins "
            "  + cysteine-rich proteins (hair/nail structure) → brittle hair + ichthyosis; "
            "  NER partially preserved → reduced cancer risk vs XP"
        ),
        "disease_pathway": (
            "TFIIH COMPLEX — NER AND TRANSCRIPTION DUAL FUNCTION: "
            "TFIIH COMPOSITION (10 subunits): "
            "  Core: XPB (ERCC3), XPD (ERCC2), p52 (GTF2H4), p44 (GTF2H2), p34 (GTF2H3), "
            "        p8/TTDA (GTF2H5), p62 (GTF2H1); "
            "  CAK kinase module: CDK7, cyclin H, MAT1 (separable); "
            "NER FUNCTION OF TFIIH: "
            "  XPB 3'→5' helicase: required for opening DNA (ATPase activity essential — "
            "    helicase activity may be dispensable for NER but NOT transcription); "
            "  XPD 5'→3' helicase: required for lesion verification and unwinding; "
            "  p44: activates XPD helicase via direct contact; "
            "  TTDA: stimulates XPB ATPase; stabilises TFIIH; "
            "TRANSCRIPTION FUNCTION: "
            "  TFIIH binds promoter at TATA-box recognition; "
            "  XPB ATPase opens promoter for Pol II initiation; "
            "  CDK7 phosphorylates Pol II CTD Ser5 → productive elongation; "
            "TTD MECHANISM: "
            "  Reduced TFIIH levels → impaired Pol II transcription of selenium-containing proteins "
            "  (selenoproteins: GPx, TXNRD) and cysteine-rich matrix proteins → brittle hair + nails"
        ),
        "pathognomonic": (
            "TIGER-TAIL BANDING OF HAIR ON POLARISED LIGHT MICROSCOPY (TRICHOTHIODYSTROPHY): "
            "  alternating light and dark bands in hair shaft = PATHOGNOMONIC for TTD; "
            "  bright and dark bands = alternating sulphur-rich/sulphur-poor segments; "
            "  due to impaired TFIIH-dependent transcription of sulphur-rich matrix proteins; "
            "UDS ASSAY: typically 10-30% of normal in XP-B (partial helicase function); "
            "TFIIH PROTEIN WESTERN BLOT: reduced levels of all TFIIH subunits (TTD alleles); "
            "HAIR AMINO ACID ANALYSIS: low cysteine content in hair shaft (< 5% vs normal 16%); "
            "XP-B ULTRAORPHAN CONFIRMATION: "
            "  ERCC3 cDNA complementation of UDS deficiency; "
            "  ERCC3 gene sequencing; "
            "CLINICAL TRIAD (when XP+CS): "
            "  UV sensitivity + 'cachectic bird-like facies' + premature ageing = XP/CS overlap; "
            "ERCC3 ALLELE NOTE: "
            "  Pro131Thr (c.391C>A) — helicase motif II — XP-B (only known XP-B mutation 2026); "
            "  Arg112His — stabilisation mutation → TTD phenotype (most common TTD-ERCC3)"
        ),
        "treatment": (
            "1. UV AVOIDANCE (same rigorous protocol as XP-A above); "
            "2. DERMATOLOGY: aggressive surveillance and early treatment of skin cancers; "
            "3. COCKAYNE SYNDROME FEATURES (if XP/CS overlap): "
            "   photosensitivity; feeding support (nasogastric/PEG for growth failure); "
            "   physiotherapy + orthoses for motor deficits; "
            "   hearing aids for SNHL; "
            "   annual ophthalmology (retinal degeneration); "
            "4. TTD FEATURES: "
            "   fragile hair — gentle handling; protective head covering; "
            "   ichthyosis management: emollients, keratolytics; "
            "   genetic counselling (fertility often preserved in mild TTD); "
            "5. TFIIH STABILISATION (experimental): "
            "   ERCC3 null → no effective upregulation; research stage only; "
            "6. GENE THERAPY: no clinical trials 2026 for ERCC3 (ultraorphan, <30 patients); "
            "NOTE: ERCC3/XP-B has NO approved specific therapy; management entirely supportive"
        ),
    },
    {
        "gene": "XPC",
        "seed_base": 2712,
        "protein": (
            "XPC -- 3p25.1 AR -- 940aa -- XPC-RAD23B-CETN2-GGR-Damage-Sensor-"
            "106kDa-Monomer-GGR-Initiation-Helix-Distortion-Sensing-"
            "OMIM-Gene-613208-Disease-XP-C-278720"
        ),
        "locus": "3p25.1",
        "protein_size": "940 aa / 106 kDa (GGR initiator; complexed with RAD23B and CETN2)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF; "
            "XPC encodes the initiating damage-recognition factor for GLOBAL GENOME NER (GGR) ONLY; "
            "XPC-RAD23B-CETN2 complex: "
            "  XPC binds undamaged DNA strand opposite the lesion (senses helix distortion); "
            "  RAD23B (RAD23 homolog B): ubiquitin-like domain, stabilises XPC; "
            "  CETN2 (centrin-2): stimulates XPC-DNA binding; "
            "XPC LOF → GGR abolished BUT TCR (transcription-coupled NER) is INTACT; "
            "  → only UV lesions in actively transcribed genes are repaired; "
            "  → non-transcribed genome accumulates CPD/6-4PP → high cancer risk; "
            "XPC IS THE MOST COMMON XP GROUP in the USA/Europe (~50% of non-Japanese XP); "
            "Japan: XPA most common; North Africa: XPC + XPA both common; "
            "PREVALENCE: 1 in 250,000 (XP all groups USA/Europe); XPC ~1 in 500,000"
        ),
        "disease_category": (
            "XERODERMA PIGMENTOSUM GROUP C (OMIM 278720); "
            "XP-C DISTINCTIVE FEATURES (vs other XP): "
            "  CUTANEOUS ONLY in most cases: severe UV sensitivity + skin cancer; "
            "  NEURODEGENERATION ABSENT or very mild (TCR intact → neuronal repair preserved); "
            "  SKIN CANCER risk: 10,000x elevated; "
            "  EARLIEST AGE AT SKIN CANCER: median 9 years (vs 60 in general population); "
            "  CANCER TYPES: BCC (most common), SCC, melanoma, ocular surface tumours; "
            "  PHOTOPHOBIA + CONJUNCTIVITIS + CORNEAL OPACIFICATION: common; "
            "  INTERNAL CANCERS: increased brain tumours (glioblastoma); leukaemia risk elevated; "
            "XP-C vs XP-A COMPARISON: "
            "  XP-C: GGR defect only, TCR intact → NO neurodegeneration → better survival; "
            "  XP-A: BOTH GGR+TCR defective → severe neurodegeneration → poor prognosis; "
            "FOUNDER MUTATIONS: "
            "  North African (Maghreb): c.1643_1644del (p.Trp548Ter) — most common globally; "
            "  USA/European: Lys761Arg; Japanese: various"
        ),
        "disease_pathway": (
            "GLOBAL GENOME NER (GGR) INITIATION — XPC ROLE: "
            "GGR vs TCR — WHY TWO SUBPATHWAYS? "
            "  GGR: repairs ANYWHERE in genome (including silent/non-transcribed regions); "
            "    initiated by XPC-RAD23B-CETN2 damage sensing; "
            "  TCR: repairs ONLY actively transcribed strands; "
            "    initiated by stalled RNA Pol II → CSB/CSA recruitment; "
            "  Both pathways converge at STEP 2 (XPA + TFIIH + RPA): "
            "XPC MECHANISM — INDIRECT DAMAGE SENSING: "
            "  XPC does NOT bind the damaged base directly; "
            "  XPC inserts beta-hairpin domain into the minor groove on the UNDAMAGED STRAND; "
            "  XPC binds 2 unpaired thymine nucleotides on undamaged strand (opposite CPD); "
            "  Helix distortion → enhanced XPC affinity; "
            "  UV-DDB (DDB1-DDB2/XPE) complex: ubiquitinates XPC to enhance binding at CPD lesions "
            "    (CPD is poorly sensed by XPC alone — DDB2 important for CPD repair); "
            "  6-4PP: directly sensed by XPC (greater distortion → better binding); "
            "XPC LOF CONSEQUENCE: "
            "  GGR abolished → CPD/6-4PP accumulate in non-transcribed strands; "
            "  TCR intact → actively transcribed genes repaired → neurons protected; "
            "  Net: skin/ocular cancers but usually spared neurodegeneration"
        ),
        "pathognomonic": (
            "UV HYPERSENSITIVITY FROM INFANCY WITH SKIN CANCERS IN FIRST DECADE: "
            "  combined with ABSENT NEURODEGENERATION = XP-C signature; "
            "  (presence of neurodegeneration in UV-sensitive child suggests XP-A not XP-C); "
            "UDS ASSAY: 5-25% of normal (partial GGR residual, some TCR activity detected); "
            "COMPLEMENTATION TESTING: XP-C cell line restores UDS when fused with XP-A line; "
            "XPC GENE SEQUENCING: "
            "  c.1643_1644del (p.Trp548Ter) — North African founder — PCR diagnosis; "
            "  c.2278C>T (p.Gln760Ter) — common European; "
            "UV-DOSE-RESPONSE CURVE: "
            "  XP-C fibroblasts show normal proliferation but REDUCED viability post-UV (CPD accumulation); "
            "  CPD photolyase complementation specifically rescues XP-C cells (confirms GGR defect); "
            "OCULAR SLIT-LAMP: "
            "  corneal vascularisation; pterygium; squamous cell carcinoma of conjunctiva; "
            "  band keratopathy; corneal opacification (all from UV exposure)"
        ),
        "treatment": (
            "1. UV AVOIDANCE (same rigorous protocol as XP-A — see definitions); "
            "2. DERMATOLOGY: "
            "   full-body skin exam every 3 months; "
            "   early excision of all suspicious lesions; "
            "   field treatment: 5-FU cream, imiquimod, ingenol for actinic keratoses; "
            "   vismodegib/sonidegib for multiple BCCs; "
            "3. OPHTHALMOLOGY: "
            "   UV-blocking lenses mandatory; "
            "   pterygium excision; conjunctival tumour surveillance; "
            "   corneal grafting for severe opacification; "
            "4. ONCOLOGY: "
            "   brain MRI annually (increased CNS tumour risk); "
            "   low threshold for CNS investigation; "
            "5. ORAL ISOTRETINOIN: chemoprevention for multiple BCCs (off-label); "
            "6. CAPECITABINE (oral 5-FU prodrug): skin field treatment for widespread AKs; "
            "7. GENE THERAPY (CLINICAL): "
            "   XPC mRNA-lipid nanoparticle topical delivery (Phase I/II, Sanofi/XPA-clinique 2024); "
            "   intradermal XPC gene correction — early clinical trials; "
            "8. XP CLINIC FOLLOW-UP: multidisciplinary every 6 months; "
            "NOTE: XP-C DOES NOT typically require neurological monitoring (TCR intact)"
        ),
    },
    {
        "gene": "ERCC2",
        "seed_base": 2713,
        "protein": (
            "ERCC2 -- 19q13.32 AR -- 760aa -- XPD-TFIIH-5prime-3prime-Helicase-Subunit-"
            "87kDa-TFIIH-Core-NER-Lesion-Verification-p44-Regulated-"
            "OMIM-Gene-126340-Disease-XP-D-278730-TTD-601675-COFS-214150"
        ),
        "locus": "19q13.32",
        "protein_size": "760 aa / 87 kDa (TFIIH 5'→3' helicase; lesion verification factor)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF; "
            "ERCC2 encodes XPD, the 5'→3' ATP-dependent helicase of TFIIH; "
            "XPD UNIQUE ROLE IN NER: "
            "  XPD is the LESION VERIFICATION helicase — scans DNA while unwinding; "
            "  XPD STOPS at DNA lesion → stalls → recruits XPA → positions dual-incision complex; "
            "  CONTRAST: XPB (ERCC3) opens promoter but does NOT verify lesion; "
            "  p44 (GTF2H2): activates XPD helicase activity; ARCH domain: bulky lesion sensor; "
            "ERCC2 PHENOTYPIC SPECTRUM (widest of all XP groups): "
            "  XP-D alone: UV sensitivity + skin cancers + variable neurodegeneration; "
            "  XP/CS OVERLAP: XP features + Cockayne syndrome (premature ageing, intracranial Ca²⁺); "
            "  TRICHOTHIODYSTROPHY (TTD): brittle hair + ichthyosis + photosensitivity + NO cancer; "
            "  COFS (cerebro-oculo-facio-skeletal): severe combined XP/CS + arthrogryposis; "
            "  Allele determines phenotype: helicase mutations → XP; stability mutations → TTD; "
            "XP-D MOST ALLELE-DIVERSE XP GROUP"
        ),
        "disease_category": (
            "XERODERMA PIGMENTOSUM GROUP D / TRICHOTHIODYSTROPHY (OMIM 278730 / 601675); "
            "XP-D PHENOTYPES: "
            "  MILD XP-D (Arg683Trp / Lys751Gln alleles): "
            "    UV sensitivity + skin cancers + MILD to ABSENT neurodegeneration; "
            "  SEVERE XP-D (Arg156His, Arg112His / combined with other alleles): "
            "    XP + CS overlap (cachectic dwarfism + premature ageing + intracranial calcifications); "
            "    De Sanctis-Cacchione if severe neurodegeneration (XP-A or severe XP-D); "
            "  TRICHOTHIODYSTROPHY (Asp681Asn + Cys259Tyr ERCC2 alleles): "
            "    tiger-tail banding; brittle hair; ichthyosis; NO skin cancer; photosensitivity present; "
            "    intellectual disability + short stature; no UDS defect detectable (minimal NER effect); "
            "    MECHANISM: ERCC2 TTD alleles destabilise TFIIH → reduced transcription of sulphur-rich proteins; "
            "INTRACRANIAL CALCIFICATIONS (XP-D severe): "
            "  basal ganglia calcifications; cerebellar + cerebral atrophy; "
            "  NOT CT/MRI calcium in young child = XP/CS overlap until proven otherwise; "
            "ERCC2 ACCOUNTS FOR ~50% OF ALL TTD (most common TTD gene)"
        ),
        "disease_pathway": (
            "TFIIH-XPD HELICASE FUNCTION — LESION VERIFICATION: "
            "XPD 5'→3' HELICASE IN NER: "
            "  After XPC/CSB recruits TFIIH to damaged site: "
            "  XPD helicase starts unwinding 5'→3' from TFIIH loading point; "
            "  XPD ARCH domain: Fe-S cluster in helicase core senses bulky lesion (CPD/6-4PP/cisplatin-adduct); "
            "  XPD physically STOPS when encountering bulky lesion → stalls TFIIH; "
            "  XPD stall → XPA binds lesion → ERCC1-XPF recruited 5' side; "
            "  XPG recruited 3' side → dual incision complex assembled; "
            "XPD IN TRANSCRIPTION (CAK subcomplex): "
            "  XPD helicase activity NOT required for transcription initiation (conformational role); "
            "  XPD anchors CDK7-CycH-MAT1 (CAK module) to core TFIIH; "
            "TTD ALLELES MECHANISM: "
            "  TTD mutations destabilise XPD protein folding → "
            "  reduced TFIIH complex stability → "
            "  lower nuclear TFIIH concentration → "
            "  reduced Pol II CTD phosphorylation → "
            "  impaired transcription of selenium-binding proteins, hair keratin, PARP → "
            "  brittle hair (low-sulphur) + ichthyosis; NER partially retained"
        ),
        "pathognomonic": (
            "ERCC2 ARG683TRP/LYS751GLN — MILD XP-D MOST COMMON: "
            "  compound heterozygosity Arg683Trp/Lys751Gln → moderate XP-D (skin cancer, mild neuro); "
            "  SINGLE MOST COMMON XP-D GENOTYPE in Europe/North America; "
            "TIGER-TAIL BANDING ON POLARISED MICROSCOPY = TTD PATHOGNOMONIC (see ERCC3); "
            "TFIIH PROTEIN QUANTIFICATION: "
            "  TTD alleles: TFIIH level 20-50% of normal (whole complex destabilised); "
            "  XP-D alleles: TFIIH level NORMAL but helicase activity reduced; "
            "UDS ASSAY: "
            "  XP-D (helicase): 10-50% normal; "
            "  TTD-ERCC2: normal or near-normal UDS (NER not severely impaired); "
            "INTRACRANIAL CALCIFICATION ON CT/MRI: "
            "  basal ganglia, dentate nucleus — in XP-D severe + XP/CS overlap; "
            "  young child with UV sensitivity + intracranial Ca²⁺ → ERCC2 severe allele vs ERCC3; "
            "ERCC2 SEQUENCING + FUNCTIONAL ASSAY: "
            "  helicase ATPase activity (ATP hydrolysis) in recombinant protein → "
            "  XP alleles: reduced helicase; TTD alleles: reduced TFIIH stability"
        ),
        "treatment": (
            "1. UV AVOIDANCE (same rigorous protocol — see definitions); "
            "2. SKIN CANCER MANAGEMENT: same as XP-C (dermatology every 3 months); "
            "3. COCKAYNE SYNDROME FEATURES (if XP/CS ERCC2 overlap): "
            "   feeding support; growth hormone assessment; SNHL hearing aids; "
            "   intracranial calcification monitoring (annual CT or MRI without contrast); "
            "   ophthalmology: retinal degeneration surveillance; "
            "4. TTD-SPECIFIC MANAGEMENT: "
            "   gentle hair and skin care; emollients for ichthyosis; "
            "   metabolic support: selenium supplementation (theoretical; empirical evidence limited); "
            "5. NEUROLOGICAL: physiotherapy; baclofen for spasticity; "
            "6. ERCC2/XPD GENE THERAPY (experimental): "
            "   mRNA lipid nanoparticle delivery — preclinical 2025; "
            "NOTE: TTD-ERCC2 patients do NOT need aggressive skin cancer surveillance "
            "(NER partially preserved + reduced proliferation → lower cancer risk than XP)"
        ),
    },
    {
        "gene": "DDB2",
        "seed_base": 2714,
        "protein": (
            "DDB2 -- 11p11.2 AR -- 428aa -- XPE-DDB1-DDB2-CRL4-E3-Ubiquitin-Ligase-"
            "48kDa-WD40-β-Propeller-CPD-Sensor-UV-DDB-Complex-"
            "OMIM-Gene-600811-Disease-XP-E-278740"
        ),
        "locus": "11p11.2",
        "protein_size": "428 aa / 48 kDa (WD40-repeat β-propeller; DDB1-DDB2 heterodimer / CRL4 E3)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF; "
            "DDB2 encodes the DNA damage-binding protein 2, the lesion-recognition subunit of UV-DDB; "
            "UV-DDB COMPLEX: DDB1 (127 kDa, scaffold) + DDB2 (XPE, 48 kDa, damage sensor); "
            "DDB1-DDB2 FUNCTION: "
            "  DDB2 WD40 domain: directly contacts CPD lesion in double-stranded DNA; "
            "  flips damaged bases out of helix (base-flipping) → presents lesion to XPC; "
            "  UV-DDB-CRL4 E3 ubiquitin ligase activity: ubiquitinates XPC (Ub-XPC has HIGHER NER affinity); "
            "  Also ubiquitinates H2A (local chromatin opening) and DDB2 itself (self-destruct after XPC loaded); "
            "DDB2 LOF CONSEQUENCE: "
            "  CPD lesions NOT efficiently recognised by XPC alone (XPC binds 6-4PP well but poor CPD sensor); "
            "  → CPD lesions under-repaired in GGR → increased cancer risk; "
            "  6-4PP lesions less affected (XPC senses these directly); "
            "MILDEST XP GROUP: DDB2 LOF = XP-E (mildest of all NER-defective XP groups); "
            "PREVALENCE: rarest XP group; ~3-5% of all XP; very few families worldwide"
        ),
        "disease_category": (
            "XERODERMA PIGMENTOSUM GROUP E (OMIM 278740); "
            "XP-E DISTINGUISHING FEATURES: "
            "  MILDEST CLASSIC XP GROUP: UV sensitivity + skin cancers present but less severe; "
            "  LATER CANCER ONSET: compared to XP-A, XP-C (CPD repair partially preserved via TCR); "
            "  NEURODEGENERATION: absent or very mild in most XP-E patients; "
            "  UDS: 30-50% of normal (highest among NER-defective XP); "
            "XP-E CLINICAL MANIFESTATIONS: "
            "  UV hypersensitivity: moderate (less severe than XP-A/C); "
            "  Freckles + lentigines on UV-exposed skin from early childhood; "
            "  Skin cancers: present but later onset than other XP groups; "
            "  Ocular: photophobia, conjunctivitis; corneal changes milder than XP-A; "
            "  Neurological: usually absent; rare reports of mild intellectual disability; "
            "DDB2 IN CANCER SUPPRESSION: "
            "  DDB2 is a p53 target gene (upregulated by p53 after UV) → "
            "  DDB2 LOF impairs p53-mediated UV damage repair → p53 pathway partially uncoupled from NER; "
            "  XP-E patients develop UV-associated BCC/SCC (similar spectrum to other XP groups)"
        ),
        "disease_pathway": (
            "UV-DDB-CRL4 COMPLEX — CHROMATIN-COUPLED CPD RECOGNITION: "
            "CPD vs 6-4PP RECOGNITION DIFFERENCES: "
            "  6-4PP: large helical distortion → XPC directly recognises → efficient GGR; "
            "  CPD: small helical distortion → XPC POOR SENSOR → requires UV-DDB assistance; "
            "UV-DDB-CRL4 RECRUITMENT SEQUENCE: "
            "  UV irradiation → CULLIN4A-RBX1-DDB1-DDB2 (CRL4DDB2) recruited to chromatin; "
            "  DDB2 directly contacts CPD → stacks Trp341/Phe334 residues on damaged bases; "
            "  CRL4DDB2 E3 activity: ubiquitinates "
            "    i. XPC → Ub-XPC: 3-10x higher NER affinity; "
            "    ii. H2A → opens chromatin at damage site; "
            "    iii. DDB2 itself → DDB2 degraded after XPC loading (auto-destruct mechanism); "
            "DDB2 LOF: "
            "  CPD not efficiently presented to XPC → XPC poor CPD binding → "
            "  CPD repair slow → accumulate in non-transcribed DNA → mutagenesis → cancer; "
            "  6-4PP repair: less affected (direct XPC sensing) → explains milder phenotype vs XP-C; "
            "ROLE OF p53 IN DDB2 EXPRESSION: "
            "  p53 transcriptionally activates DDB2 after UV → DDB2 is part of p53 UV-response; "
            "  DDB2 also interacts with p53 to facilitate chromatin remodelling"
        ),
        "pathognomonic": (
            "XP-E = MILDEST XP GROUP WITH RESIDUAL UDS 30-50%: "
            "  any UV-sensitive patient with UDS >25% → consider XP-E first; "
            "UDS ASSAY: 30-50% of normal (HIGHEST among NER-defective XP groups); "
            "  complement with intact CS cell lines (XP-E restores CS UDS) → "
            "  complement only with XP-E cell line → confirms group E; "
            "DDB ACTIVITY ASSAY: "
            "  electrophoretic mobility shift assay (EMSA) with UV-damaged DNA probe; "
            "  XP-E nuclear extracts: NO DDB band (DDB1-DDB2 fails to bind damaged DNA); "
            "  normal cells: DDB band present; "
            "DDB2 GENE SEQUENCING: "
            "  Lys244Glu — original XP-E family (Hwang et al. 1998); "
            "  common: Arg273His, Lys244Asn (interface with DDB1); "
            "CANCER HISTOLOGY: BCC and SCC at sun-exposed sites (similar to other XP); "
            "p53 PATHWAY: "
            "  DDB2 LOF may unmask specific p53 pathway deficiencies in affected tissue; "
            "  elevated risk despite milder photosensitivity (long-term accumulated UV lesions)"
        ),
        "treatment": (
            "1. UV AVOIDANCE: same rigorous protocol (see definitions) — though milder disease; "
            "   even milder XP-E patients develop cancer → strict protection mandatory; "
            "2. DERMATOLOGY: full-body skin exam every 3-6 months; "
            "   early intervention for AKs and early BCCs; "
            "3. OPHTHALMOLOGY: UV-protective lenses; corneal care; "
            "4. NEUROLOGICAL: not required in most XP-E patients; "
            "5. CHEMOPREVENTION: "
            "   low-dose oral isotretinoin: reduces new BCC formation; "
            "   5-FU field treatment for widespread actinic keratoses; "
            "6. XP-E PROGNOSIS: "
            "   relatively better than XP-A/XP-D; longer survival possible with strict UV protection; "
            "   cancer remains leading cause of death without protection; "
            "NOTE: DDB2 HAS NO APPROVED SPECIFIC THERAPY; "
            "mRNA/gene replacement: preclinical interest but no trials 2026"
        ),
    },
    {
        "gene": "ERCC4",
        "seed_base": 2715,
        "protein": (
            "ERCC4 -- 16p13.12 AR -- 916aa -- XPF-ERCC1-Heterodimer-5prime-Endonuclease-"
            "104kDa-Structure-Specific-Flap-Endonuclease-FANCQ-Allelic-"
            "OMIM-Gene-133520-Disease-XP-F-278760-FANCQ-615272-ERCC4-Progeroid-610965"
        ),
        "locus": "16p13.12",
        "protein_size": "916 aa / 104 kDa (XPF catalytic subunit; heterodimer with ERCC1 79 kDa)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF; "
            "ERCC4 encodes XPF, the catalytic endonuclease subunit of the XPF-ERCC1 heterodimer; "
            "XPF-ERCC1 HETERODIMER FUNCTION: "
            "  Makes the 5' incision in NER (~25 nt 5' of lesion); "
            "  structure-specific endonuclease: cuts at ssDNA-dsDNA junction; "
            "  ALSO functions in interstrand crosslink (ICL) repair (Fanconi anemia pathway); "
            "ERCC4 ALLELIC COMPLEXITY: "
            "  Mild alleles (partial function): XP-F (mild UV sensitivity + skin cancers); "
            "  Severe alleles: XP-F + CS/progeroid syndrome; "
            "  ICL-specific alleles: FANCQ (Fanconi anemia complementation group Q) — "
            "    ICL sensitivity but variable UV sensitivity; "
            "FANCQ ALLELES: biallelic ERCC4 mutations with ICL repair defect → "
            "  bone marrow failure (pancytopenia), aplastic anaemia, AML; "
            "  FA clinical features + variable XP/CS overlap; "
            "ERCC4 PROGEROID: very severe alleles → segmental progeroid syndrome + XP + CS"
        ),
        "disease_category": (
            "XERODERMA PIGMENTOSUM GROUP F / FANCONI ANEMIA Q (OMIM 278760 / 615272); "
            "XP-F MILD PHENOTYPE (most common ERCC4 presentation): "
            "  mild-moderate UV hypersensitivity; skin cancers (BCC, SCC, melanoma); "
            "  NO or mild neurodegeneration; relative preservation of cognition; "
            "  onset of cancer: 2nd-3rd decade (later than XP-A/XP-C); "
            "ERCC4-PROGEROID (severe alleles): "
            "  XP features PLUS: "
            "    premature ageing (lipodystrophy, aged facies, alopecia); "
            "    Cockayne-like features (cachexia, growth failure, hearing loss); "
            "    normal intellect may be preserved in some; "
            "    cardiovascular: accelerated atherosclerosis; "
            "FANCONI ANEMIA Q (FANCQ — biallelic ERCC4 ICL alleles): "
            "  BONE MARROW FAILURE (hallmark): pancytopenia, aplastic anaemia, AML; "
            "  CHROMOSOME FRAGILITY to crosslinking agents (diepoxybutane — FA diagnostic test); "
            "  variable UV sensitivity + skin features (may lack classic XP); "
            "  Fanconi physical stigmata: short stature, radial ray anomalies, café-au-lait spots; "
            "  Biallelic ERCC4 ICL-specific alleles → FA phenotype; "
            "TREATMENT CRITICAL DISTINCTION: FA patients → haematopoietic stem cell transplant (HSCT)"
        ),
        "disease_pathway": (
            "XPF-ERCC1 — 5' INCISION IN NER AND INTERSTRAND CROSSLINK REPAIR: "
            "NER 5' INCISION MECHANISM: "
            "  XPA-TFIIH-RPA assembly positions XPF-ERCC1 on ssDNA-dsDNA junction 5' of lesion; "
            "  ERCC4/XPF catalytic domain: PD-(D/E)XK motif → cleaves phosphodiester bond; "
            "  ERCC1 binds XPA via C-terminal HhH domain → positions XPF at correct site; "
            "  ERCC1 also binds RPA → stabilises repair complex; "
            "  CUT POSITION: 15-25 nt 5' of DNA lesion; "
            "ICL REPAIR (Fanconi anemia pathway): "
            "  Stalled replication fork at ICL → FA core complex monoubiquitinates FANCD2-FANCI; "
            "  Ub-FANCD2 recruits XPF-ERCC1 to unhook one strand of ICL (endonuclease activity); "
            "  After unhooking: translesion synthesis past unhooked ICL strand (TLS pols); "
            "  Then XPF-ERCC1 removes residual adduct; then HR (BRCA2/RAD51) repairs DSB; "
            "ALLELE-FUNCTION CORRELATION: "
            "  Mild ERCC4 alleles (R788W, Arg799Trp): retain NER but partial ICL defect → XP-F mild; "
            "  Severe ERCC4 alleles (truncating/severe missense): lose BOTH NER and ICL → XP+FA; "
            "  ICL-selective alleles: ICL defect > NER defect → FA > XP phenotype"
        ),
        "pathognomonic": (
            "XP-F MILD + VARIABLE CANCER RISK — LATER ONSET: "
            "  UV-sensitive patient with DELAYED cancer onset (2nd-3rd decade) → consider XP-F; "
            "  mild UDS (20-40% normal) + complementation restores with XP-F cell line; "
            "FANCONI ANEMIA Q DIAGNOSIS: "
            "  DEB (diepoxybutane) CHROMOSOMAL FRAGILITY TEST: "
            "    elevated chromosomal breakage — PATHOGNOMONIC for Fanconi anemia (any group); "
            "    FANCQ = ERCC4 biallelic ICL alleles; "
            "  FA FLOW CYTOMETRY: FANCD2 ubiquitination blot (reduced monoUb-FANCD2); "
            "  ICL REPAIR ASSAY: reduced survival of ERCC4 mutant cells to MMC/cisplatin; "
            "ERCC4 PROGEROID FEATURES: "
            "  lipodystrophy + aged facies + premature greying in UV-sensitive patient → ERCC4 severe alleles; "
            "ERCC4 GENE SEQUENCING: "
            "  Arg788Trp (Arg799Trp) — common mild XP-F allele (Japan + Europe); "
            "  Gln158Ter, Leu686Pro — severe ICL alleles → FANCQ; "
            "  Arg589Trp — intermediate; "
            "XPF-ERCC1 IMMUNOFLUORESCENCE: "
            "  XPF-ERCC1 nuclear foci after UV irradiation — absent in XP-F cells"
        ),
        "treatment": (
            "1. UV AVOIDANCE (standard XP protocol — see definitions); "
            "2. DERMATOLOGY: skin surveillance every 3-6 months; "
            "3. FANCONI ANEMIA Q: "
            "   HSCT (haematopoietic stem cell transplant) for bone marrow failure — CURATIVE for haematological disease; "
            "   Androgen therapy (danazol) as bridge to HSCT; "
            "   G-CSF for severe neutropenia; "
            "   AVOID alkylating chemotherapy (CONTRAINDICATED in FA); "
            "   Reduced-intensity conditioning for HSCT (Fanconi sensitivity to DNA-damaging agents); "
            "4. ERCC4 PROGEROID: "
            "   cardiovascular risk factor management (aggressive); "
            "   endocrine assessment (lipodystrophy → metabolic syndrome); "
            "   physiotherapy; multidisciplinary; "
            "5. CANCER MANAGEMENT: "
            "   low-dose radiation MUST BE USED CAUTIOUSLY in FANCQ (DNA repair defect); "
            "   platinum-based chemotherapy: CAUTION in XP-F (ICL repair impaired — cisplatin toxicity); "
            "6. GENE THERAPY: "
            "   ERCC4 gene correction — preclinical; FA gene therapy research active"
        ),
    },
    {
        "gene": "ERCC5",
        "seed_base": 2716,
        "protein": (
            "ERCC5 -- 13q33.1 AR -- 1186aa -- XPG-3prime-Endonuclease-Subunit-"
            "133kDa-FEN1-Superfamily-3prime-Incision-TFIIH-Scaffold-"
            "OMIM-Gene-133530-Disease-XP-G-278780-CS-216400"
        ),
        "locus": "13q33.1",
        "protein_size": "1186 aa / 133 kDa (FEN1 superfamily 3' endonuclease; TFIIH structural factor)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF; "
            "ERCC5 encodes XPG, the 3' incision endonuclease in NER; "
            "XPG DUAL FUNCTIONAL ROLES: "
            "  NER: makes the 3' incision (+2 to +8 nt 3' of lesion); structure-specific endonuclease; "
            "  STRUCTURAL SCAFFOLD: XPG is required for stable TFIIH retention at damaged sites; "
            "    XPG LOF → TFIIH NOT stably retained → TCR fails (despite CSB intact); "
            "ERCC5 PHENOTYPIC SPECTRUM: "
            "  Mild/partial LOF: XP-G alone (UV sensitivity + skin cancers); "
            "  Severe LOF (null alleles): XP-G + CS overlap (most severe NER-defective phenotype); "
            "    profound neurodegeneration; cachectic dwarfism; death in 1st-2nd decade; "
            "XPG NULL ALLELES: complete loss → "
            "  NO 3' incision → NER repair intermediate cannot be completed → "
            "  toxic repair intermediate (open NER bubble) may trigger apoptosis; "
            "  TFIIH not anchored → TCR also abolished → neurodegeneration severe; "
            "PREVALENCE: ~5-10% of all XP (rare); XP-G+CS is lethal in childhood"
        ),
        "disease_category": (
            "XERODERMA PIGMENTOSUM GROUP G / XPG+CS OVERLAP (OMIM 278780); "
            "XP-G ALONE (partial ERCC5 function): "
            "  UV hypersensitivity; skin cancers; variable mild neurodegeneration; "
            "  UDS: 2-25% (range); "
            "XP-G + CS OVERLAP (ERCC5 null): "
            "  XP FEATURES: UV sensitivity + skin cancers; "
            "  COCKAYNE SYNDROME FEATURES (severe): "
            "    profound growth failure ('cachectic dwarfism'); "
            "    severe intellectual disability; "
            "    progressive neurodegeneration (SNHL, cerebellar atrophy); "
            "    retinal degeneration (pigmentary retinopathy); "
            "    intracranial calcifications (basal ganglia); "
            "    premature ageing ('bird-like facies'); "
            "    photosensitivity; "
            "WORST PROGNOSIS OF ANY COMBINED XP/CS: "
            "  death typically 1st-2nd decade from XP-G null; "
            "  feeding tube usually required; "
            "  seizures common; "
            "MOLECULAR EXPLANATION FOR SEVERE PHENOTYPE: "
            "  XPG null → NO 3' incision + TFIIH not anchored → "
            "  BOTH GGR and TCR abolished + transcription partially impaired → "
            "  additive failure (NER + TFIIH-dependent transcription)"
        ),
        "disease_pathway": (
            "XPG 3' INCISION AND TFIIH SCAFFOLDING: "
            "XPG 3' INCISION MECHANISM: "
            "  TFIIH unwinding establishes NER bubble (~30 nt); "
            "  XPA + RPA position XPF-ERCC1 (5' cut) and XPG (3' cut); "
            "  XPG FEN1-superfamily catalytic domain: cuts ssDNA at 3'→dsDNA junction; "
            "  CUT POSITION: +2 to +8 nt 3' of lesion; "
            "  TIMING: XPG 3' cut precedes XPF 5' cut (incision order: 3' first, then 5'); "
            "  After dual incision: 25-30 nt oligonucleotide + lesion released; "
            "XPG AS TFIIH ANCHOR: "
            "  XPG C-terminal region contacts XPB/XPD of TFIIH; "
            "  XPG physically anchors TFIIH at the NER site; "
            "  XPG LOF → TFIIH falls off → incomplete repair complex → NER fails even if XPD/XPB intact; "
            "TCR DEPENDENCE ON XPG-TFIIH ANCHORAGE: "
            "  CSB recruits TFIIH to TCR; XPG anchors TFIIH; "
            "  XPG null → TFIIH not anchored in TCR → TCR also fails; "
            "  → neurons (reliant on TCR) lose repair → severe neurodegeneration; "
            "XPG NULL TOXIC INTERMEDIATE HYPOTHESIS: "
            "  NER bubble opened by TFIIH but XPG absent → bubble not incised → "
            "  persistent single-strand region → replication fork stall → DSB → apoptosis; "
            "  Toxic intermediates may contribute to severe CS features"
        ),
        "pathognomonic": (
            "XP-G SEVERE NULL = MOST SEVERE COMBINED XP+CS PHENOTYPE: "
            "  UV-sensitive child with cachectic dwarfism + profound ID + "
            "  cerebellar atrophy + SNHL → ERCC5 null allele until proven otherwise; "
            "  contrast: ERCC2 (XP/CS) usually has intracranial Ca²⁺ but milder neuro; "
            "  contrast: CSB/CSA (classical CS): NO skin cancer (NER intact); "
            "UDS: 2-25% of normal (range depends on residual endonuclease function); "
            "COMPLEMENTATION: XP-G cells complemented by ERCC5 cDNA; "
            "XPG 3' ENDONUCLEASE ASSAY: "
            "  recombinant XPG + ERCC5 mutant extract → no 3' incision activity; "
            "ERCC5 SEQUENCING: "
            "  Arg536Stop, Arg668Stop → null → severe XP+CS; "
            "  Cys571Arg, Asp812Asn → partial function → mild XP-G; "
            "BRAIN MRI: "
            "  cerebellar atrophy; "
            "  delayed myelination; periventricular white matter changes; "
            "  basal ganglia calcification on CT (later stage); "
            "OPHTHALMOLOGY: "
            "  pigmentary retinopathy on fundoscopy (salt-and-pepper) — rare in milder XP groups; "
            "  combined with UV-related corneal changes → ERCC5 overlap phenotype"
        ),
        "treatment": (
            "1. UV AVOIDANCE (rigorous UV protection protocol — see definitions); "
            "2. SKIN CANCER SURVEILLANCE: every 3 months; early excision; "
            "3. COCKAYNE SYNDROME / XP+CS OVERLAP: "
            "   FEEDING SUPPORT: nasogastric tube → PEG for progressive dysphagia; "
            "   high-calorie formula (200% of RDA for growth failure); "
            "   PHYSIOTHERAPY: tone management; contracture prevention; "
            "   OPHTHALMOLOGY: retinal degeneration — low vision aids; "
            "   SNHL: hearing aids; cochlear implants for profound loss; "
            "   SEIZURES: LEV or VPA (avoid enzyme inducers that accelerate DNA damage response); "
            "   INTRACRANIAL HYPERTENSION: monitoring; acetazolamide if needed; "
            "4. PALLIATIVE CARE: "
            "   XPG null: prognosis poor; palliative focus by early adolescence; "
            "   family and carer support; pain management; "
            "5. ERCC5 GENE THERAPY: preclinical interest — no clinical trials 2026 (rarity); "
            "6. PHOTOSENSITISING DRUGS: ABSOLUTELY CONTRAINDICATED in ALL XP (see definitions)"
        ),
    },
    {
        "gene": "POLH",
        "seed_base": 2717,
        "protein": (
            "POLH -- 6p21.1 AR -- 713aa -- Pol-eta-Y-Family-TLS-Polymerase-"
            "78kDa-Monomer-CPD-Translesion-Synthesis-Error-Free-XPV-"
            "OMIM-Gene-603968-Disease-XP-V-278750"
        ),
        "locus": "6p21.1",
        "protein_size": "713 aa / 78 kDa (Y-family TLS polymerase; PCNA-interacting PIP box C-terminal)",
        "inheritance": (
            "AUTOSOMAL RECESSIVE — biallelic LOF; "
            "POLH encodes DNA polymerase eta (Pol η), a Y-family translesion synthesis (TLS) polymerase; "
            "UNIQUE FEATURE — XP-V IS NOT A NER DEFECT: "
            "  Pol η functions AFTER CPD lesions bypass NER → replicates past CPD in error-FREE manner; "
            "  Pol η uniquely inserts AA opposite CPD (cis-syn TT dimer) → faithful bypass; "
            "  POLH LOF → CPD replication requires other error-PRONE TLS pols (Pol κ, Pol ι, Rev1) → "
            "  increased mutation rate at CPD sites → C→T transitions at dipyrimidines → UV signature mutations; "
            "CLINICAL DISTINCTION: XP-V has NORMAL NER (UDS normal!) but TLS defect; "
            "DELAYED ONSET compared to NER-defective XP: "
            "  UV sensitivity milder; cancer onset later (2nd-3rd decade vs 1st decade); "
            "  MISDIAGNOSED as mild sunburn susceptibility or no XP; "
            "PREVALENCE: XP-V accounts for ~20% of all XP in Europe; second most common after XP-C; "
            "    Japan: XP-V ~30-40% of all XP (most common XP group in Japan)"
        ),
        "disease_category": (
            "XERODERMA PIGMENTOSUM VARIANT (XP-V) (OMIM 278750); "
            "XP-V DISTINCTIVE FEATURES: "
            "  NORMAL NER (UDS normal) — KEY DIAGNOSTIC DISTINCTION from all other XP groups; "
            "  LATER CANCER ONSET: skin cancers typically 2nd-3rd decade (not 1st decade); "
            "  UV HYPERSENSITIVITY: milder than NER-defective XP; acute sunburn present but less extreme; "
            "  CAFE-AU-LAIT SPOTS: reported in some XP-V patients; "
            "  FRECKLING/LENTIGINES: on UV-exposed skin; "
            "  NEURODEGENERATION: absent or very mild; "
            "  INTERNAL CANCERS: brain tumours reported (as in other XP groups); "
            "XP-V AND CHECKPOINT: "
            "  Pol η loss → prolonged S-phase checkpoint activation after UV (stalled forks at CPD); "
            "  CHECKPOINT PROTEIN RAD18: ubiquitinates PCNA → recruits Pol η; "
            "  In XP-V: RAD18-Ub-PCNA recruits error-prone TLS pols instead → mutation spectrum: "
            "    C→T, CC→TT at TT dipyrimidines (UV signature in p53, PTCH1 → BCC; CDKN2A → melanoma); "
            "CANCER TYPES IN XP-V: BCC, SCC, melanoma — similar spectrum to other XP"
        ),
        "disease_pathway": (
            "TRANSLESION SYNTHESIS (TLS) — POL ETA MECHANISM: "
            "NORMAL S-PHASE ENCOUNTER WITH CPD: "
            "  DNA replication fork encounters CPD → replicative polymerase Pol δ/ε STALLS; "
            "  Two bypass mechanisms: "
            "    i. NER removes CPD before fork arrives (pre-replicative removal); "
            "    ii. TLS polymerases bypass stalled fork (post-replicative gap fill); "
            "POL ETA TLS MECHANISM: "
            "  RAD6-RAD18 E2-E3 ubiquitin ligase: monoubiquitinates PCNA (Lys164); "
            "  Ub-PCNA recruits Y-family TLS pols via UBZ or UBM domains; "
            "  Pol η binds Ub-PCNA via PIP box + UBZ domain; "
            "  Pol η active site (UNIQUELY SPACIOUS): accommodates bulky cis-syn CPD; "
            "  Pol η inserts AA opposite TT (CPD) → error-FREE bypass → correct sequence; "
            "POLH LOF CONSEQUENCES: "
            "  Stalled fork at CPD → REV1, Pol κ, Pol ι recruited (error-PRONE TLS pols); "
            "  Error-prone TLS: inserts wrong nucleotides at CPD → "
            "  C→T, CC→TT mutations at dipyrimidines → UV signature carcinogenesis; "
            "  Net: NORMAL NER but MUTAGENIC replication past CPD → delayed but certain skin cancer; "
            "OTHER POL ETA SUBSTRATES: "
            "  Cisplatin GG-crosslinks (also bypassed accurately by Pol η); "
            "  8-oxoG lesions; BPDE-dG adducts (less efficiently)"
        ),
        "pathognomonic": (
            "XP-V PATHOGNOMONIC COMBINATION: UV SENSITIVITY + NORMAL UDS: "
            "  ALL other XP groups have REDUCED UDS; "
            "  XP-V: UDS NORMAL (100%) — NER intact; "
            "  Patient with classic XP features but NORMAL UDS → XP-V until proven otherwise; "
            "COMPLEMENTATION ASSAY (modified for XP-V): "
            "  Post-replication repair assay (not UDS): "
            "  XP-V cells: elevated single-stranded gaps post-UV during S-phase "
            "  (Pol η absent → gaps not filled accurately → delayed gap filling); "
            "POLH GENE SEQUENCING: "
            "  most common allele: Arg412Stop (Japan: c.1234C>T — most prevalent worldwide); "
            "  European: Asp473Asn (splice), Leu649Ter; "
            "  Trp279Ter: USA/Europe; "
            "POL ETA TLS ASSAY IN VITRO: "
            "  recombinant Pol η: inserting AA opposite CPD in primer extension assay; "
            "  XP-V extracts: NO Pol η activity; "
            "LATE-ONSET SKIN CANCER + MILD SUNBURN HISTORY: "
            "  patient presenting in 20s-30s with multiple BCCs + history of 'mild sunburn' → "
            "  POLH sequencing mandatory (often missed earlier); "
            "JAPAN-SPECIFIC NOTE: XP-V more common in Japan than other groups; "
            "Japanese XP-V: Arg412Stop highly prevalent (founder)"
        ),
        "treatment": (
            "1. UV AVOIDANCE: same rigorous protocol (see definitions); "
            "   milder UV sensitivity does NOT reduce cancer risk — full protection mandatory; "
            "2. DERMATOLOGY: skin surveillance every 3-6 months; "
            "3. CANCER MANAGEMENT: "
            "   BCC/SCC: surgical excision; Mohs micrographic surgery; "
            "   melanoma: wide local excision + sentinel node biopsy; "
            "   systemic: vismodegib (Hedgehog inhibitor) for multiple BCCs; "
            "   nivolumab/pembrolizumab: for advanced melanoma (UV-mutated — high TMB → immunotherapy response); "
            "4. CISPLATIN CAUTION: "
            "   Pol η normally bypasses cisplatin GG-crosslinks → POLH LOF may reduce cisplatin tolerance; "
            "   consider dose reduction/alternative in XP-V receiving cisplatin; "
            "5. NEUROLOGICAL: "
            "   generally NOT required in XP-V; monitor for occult neurological changes; "
            "6. GENOTYPING: "
            "   cascade testing of siblings — XP-V has good prognosis with early UV protection; "
            "   if diagnosed early (before cancer), strict UV avoidance can prevent BCC/SCC entirely; "
            "7. GENE THERAPY: POLH mRNA delivery — early preclinical 2025; "
            "   topical mRNA-LNP delivery to skin — active area"
        ),
    },
]


# ─────────────────────────── patient simulation ────────────────────────────

def _generate_patients(gene_idx: int, n: int = 40, seed: int = 2710):
    rng = random.Random(seed)
    g = ATLAS_GENES[gene_idx]["gene"]

    # Per-gene clinical feature prevalence rates
    RATES = {
        "XPA": {
            "uv_sensitivity": 0.99, "skin_cancer": 0.80, "neurodegeneration": 0.90,
            "snhl": 0.85, "ataxia": 0.80, "id_severe": 0.85, "seizures": 0.40,
            "photosensitising_med_exposure": 0.15, "eye_involvement": 0.90,
            "brittle_hair": 0.05, "bone_marrow_failure": 0.0, "progeroid": 0.05,
            "mean_onset": 1.5, "sd_onset": 0.8,
        },
        "ERCC3": {
            "uv_sensitivity": 0.95, "skin_cancer": 0.55, "neurodegeneration": 0.70,
            "snhl": 0.70, "ataxia": 0.50, "id_severe": 0.65, "seizures": 0.30,
            "photosensitising_med_exposure": 0.15, "eye_involvement": 0.70,
            "brittle_hair": 0.45, "bone_marrow_failure": 0.0, "progeroid": 0.30,
            "mean_onset": 2.0, "sd_onset": 1.0,
        },
        "XPC": {
            "uv_sensitivity": 0.98, "skin_cancer": 0.90, "neurodegeneration": 0.10,
            "snhl": 0.10, "ataxia": 0.05, "id_severe": 0.10, "seizures": 0.05,
            "photosensitising_med_exposure": 0.15, "eye_involvement": 0.85,
            "brittle_hair": 0.0, "bone_marrow_failure": 0.0, "progeroid": 0.0,
            "mean_onset": 2.0, "sd_onset": 1.0,
        },
        "ERCC2": {
            "uv_sensitivity": 0.97, "skin_cancer": 0.75, "neurodegeneration": 0.55,
            "snhl": 0.60, "ataxia": 0.45, "id_severe": 0.50, "seizures": 0.25,
            "photosensitising_med_exposure": 0.15, "eye_involvement": 0.75,
            "brittle_hair": 0.35, "bone_marrow_failure": 0.0, "progeroid": 0.15,
            "mean_onset": 2.5, "sd_onset": 1.2,
        },
        "DDB2": {
            "uv_sensitivity": 0.88, "skin_cancer": 0.65, "neurodegeneration": 0.08,
            "snhl": 0.05, "ataxia": 0.05, "id_severe": 0.05, "seizures": 0.05,
            "photosensitising_med_exposure": 0.15, "eye_involvement": 0.60,
            "brittle_hair": 0.0, "bone_marrow_failure": 0.0, "progeroid": 0.0,
            "mean_onset": 3.5, "sd_onset": 2.0,
        },
        "ERCC4": {
            "uv_sensitivity": 0.85, "skin_cancer": 0.70, "neurodegeneration": 0.20,
            "snhl": 0.25, "ataxia": 0.15, "id_severe": 0.20, "seizures": 0.10,
            "photosensitising_med_exposure": 0.15, "eye_involvement": 0.65,
            "brittle_hair": 0.0, "bone_marrow_failure": 0.30, "progeroid": 0.20,
            "mean_onset": 3.0, "sd_onset": 1.5,
        },
        "ERCC5": {
            "uv_sensitivity": 0.97, "skin_cancer": 0.78, "neurodegeneration": 0.80,
            "snhl": 0.75, "ataxia": 0.70, "id_severe": 0.80, "seizures": 0.45,
            "photosensitising_med_exposure": 0.15, "eye_involvement": 0.85,
            "brittle_hair": 0.0, "bone_marrow_failure": 0.0, "progeroid": 0.35,
            "mean_onset": 1.8, "sd_onset": 0.9,
        },
        "POLH": {
            "uv_sensitivity": 0.82, "skin_cancer": 0.80, "neurodegeneration": 0.05,
            "snhl": 0.05, "ataxia": 0.05, "id_severe": 0.05, "seizures": 0.02,
            "photosensitising_med_exposure": 0.15, "eye_involvement": 0.55,
            "brittle_hair": 0.0, "bone_marrow_failure": 0.0, "progeroid": 0.0,
            "mean_onset": 5.0, "sd_onset": 3.0,
        },
    }

    patients = []
    rates = RATES.get(g, {})

    for i in range(n):
        sex = rng.choice(["M", "F"])
        onset = max(0.5, rng.gauss(rates.get("mean_onset", 3.0), rates.get("sd_onset", 1.5)))
        age_now = max(onset + 1.0, rng.uniform(onset + 2, onset + 25))
        age_now = round(min(age_now, 50.0), 1)
        onset = round(onset, 2)

        patients.append({
            "patient_id": f"{g}-{seed:04d}-{i+1:02d}",
            "gene": g,
            "sex": sex,
            "age_onset_years": onset,
            "age_current_years": age_now,
            "uv_sensitivity": rng.random() < rates.get("uv_sensitivity", 0.95),
            "skin_cancer": rng.random() < rates.get("skin_cancer", 0.5),
            "neurodegeneration": rng.random() < rates.get("neurodegeneration", 0.3),
            "snhl": rng.random() < rates.get("snhl", 0.2),
            "ataxia": rng.random() < rates.get("ataxia", 0.2),
            "id_severe": rng.random() < rates.get("id_severe", 0.2),
            "seizures": rng.random() < rates.get("seizures", 0.1),
            "photosensitising_med_exposure": rng.random() < rates.get("photosensitising_med_exposure", 0.15),
            "eye_involvement": rng.random() < rates.get("eye_involvement", 0.7),
            "brittle_hair": rng.random() < rates.get("brittle_hair", 0.0),
            "bone_marrow_failure": rng.random() < rates.get("bone_marrow_failure", 0.0),
            "progeroid": rng.random() < rates.get("progeroid", 0.0),
        })

    return patients


def generate_overview():
    all_patients = []
    gene_summaries = []
    seeds = list(range(2710, 2718))

    for idx, gene_data in enumerate(ATLAS_GENES):
        patients = _generate_patients(idx, n=40, seed=seeds[idx])
        all_patients.extend(patients)
        n = len(patients)

        uv_n = sum(1 for p in patients if p["uv_sensitivity"])
        cancer_n = sum(1 for p in patients if p["skin_cancer"])
        neuro_n = sum(1 for p in patients if p["neurodegeneration"])
        snhl_n = sum(1 for p in patients if p["snhl"])
        atax_n = sum(1 for p in patients if p["ataxia"])
        id_n = sum(1 for p in patients if p["id_severe"])
        seiz_n = sum(1 for p in patients if p["seizures"])
        eye_n = sum(1 for p in patients if p["eye_involvement"])
        brittle_n = sum(1 for p in patients if p["brittle_hair"])
        bmf_n = sum(1 for p in patients if p["bone_marrow_failure"])
        prog_n = sum(1 for p in patients if p["progeroid"])
        mean_onset = round(sum(p["age_onset_years"] for p in patients) / n, 1)

        gene_summaries.append({
            "gene": gene_data["gene"],
            "locus": gene_data["locus"],
            "protein_size": gene_data["protein_size"],
            "n_patients": n,
            "mean_onset_years": mean_onset,
            "pct_uv_sensitivity": round(100 * uv_n / n),
            "pct_skin_cancer": round(100 * cancer_n / n),
            "pct_neurodegeneration": round(100 * neuro_n / n),
            "pct_snhl": round(100 * snhl_n / n),
            "pct_ataxia": round(100 * atax_n / n),
            "pct_id_severe": round(100 * id_n / n),
            "pct_seizures": round(100 * seiz_n / n),
            "pct_eye_involvement": round(100 * eye_n / n),
            "pct_brittle_hair": round(100 * brittle_n / n),
            "pct_bone_marrow_failure": round(100 * bmf_n / n),
            "pct_progeroid": round(100 * prog_n / n),
            "protein": gene_data["protein"],
        })

    pathway_categories = [
        {
            "pathway": "GGR + TCR (both subpathways)",
            "genes": ["XPA"],
            "note": "XPA required for BOTH GGR and TCR; most severe neurodegeneration",
        },
        {
            "pathway": "TFIIH component (NER + transcription dual function)",
            "genes": ["ERCC3", "ERCC2"],
            "note": "XPB+XPD are TFIIH helicase subunits; allele determines XP vs TTD vs CS phenotype",
        },
        {
            "pathway": "GGR initiation (damage recognition)",
            "genes": ["XPC", "DDB2"],
            "note": "XPC: GGR initiator (helix-distortion sensing); DDB2: CPD sensor, enhances XPC at CPD",
        },
        {
            "pathway": "Dual incision endonucleases (5' and 3' cuts)",
            "genes": ["ERCC4", "ERCC5"],
            "note": "ERCC4/XPF: 5' cut (also FANCQ ICL repair); ERCC5/XPG: 3' cut + TFIIH anchor",
        },
        {
            "pathway": "Translesion synthesis past CPD (NOT NER)",
            "genes": ["POLH"],
            "note": "Pol η bypasses CPD error-free; POLH LOF → normal NER but mutagenic TLS → delayed cancer",
        },
    ]

    critical_distinctions = [
        "XP-V (POLH) = NORMAL UDS — only XP group without NER defect; misdiagnosed as mild sun sensitivity",
        "XP-C (XPC): GGR defect, TCR INTACT → NO neurodegeneration; contrast XP-A (both pathways abolished)",
        "XP-E (DDB2): MILDEST classic XP; UDS 30-50% normal; CPD recognition defect; later cancer onset",
        "ERCC2 vs ERCC3 (TFIIH): allele determines XP vs TTD vs CS overlap — same gene, opposite cancer risk",
        "FANCQ (ERCC4 ICL alleles): bone marrow failure; DEB chromosomal fragility PATHOGNOMONIC; cisplatin CI",
        "TIGER-TAIL BANDING on polarised hair microscopy = TTD PATHOGNOMONIC (ERCC3 or ERCC2 TTD alleles)",
        "PHOTOSENSITISING DRUGS ABSOLUTELY CI in ALL XP: tetracyclines, fluoroquinolones, HCT, NSAIDs, amiodarone",
        "XPG NULL: worst combined XP+CS phenotype; TFIIH not anchored → TCR also fails → severe neurodegeneration",
        "XPA Japanese founder Arg228Ter: most common XP-A allele globally; PCR detectable in at-risk populations",
        "TCR (transcription-coupled NER) protects neurons → XP groups with intact TCR spare neurodegeneration",
    ]

    return {
        "atlas": "Hereditary NER/XP Atlas",
        "genes": [g["gene"] for g in ATLAS_GENES],
        "total_patients": len(all_patients),
        "seeds": seeds,
        "pathway_categories": pathway_categories,
        "critical_distinctions": critical_distinctions,
        "gene_summaries": gene_summaries,
    }


def generate_breakdown():
    breakdown = []
    for idx, gene_data in enumerate(ATLAS_GENES):
        patients = _generate_patients(idx, n=40, seed=2710 + idx)
        n = len(patients)

        breakdown.append({
            "gene": gene_data["gene"],
            "locus": gene_data["locus"],
            "n_patients": n,
            "protein": gene_data["protein"],
            "protein_size": gene_data["protein_size"],
            "inheritance": gene_data["inheritance"],
            "disease_category": gene_data["disease_category"],
            "disease_pathway": gene_data["disease_pathway"],
            "pathognomonic": gene_data["pathognomonic"],
            "treatment": gene_data["treatment"],
            "patients": patients,
        })

    return {"atlas": "Hereditary NER/XP Atlas", "genes": breakdown}


def generate_definitions():
    glossary = {
        "NER (Nucleotide Excision Repair)": (
            "DNA repair pathway that removes bulky helix-distorting lesions (UV-induced CPD, 6-4PP; "
            "cisplatin-GG crosslinks; BPDE-dG adducts); "
            "TWO subpathways: "
            "  GGR (Global Genome NER): repairs anywhere in genome (XPC-initiated); "
            "  TCR (Transcription-Coupled NER): repairs only actively transcribed strands (CSB/CSA-initiated); "
            "Both converge at XPA → TFIIH → RPA → XPF-ERCC1 (5' cut) → XPG (3' cut) → gap fill + ligation"
        ),
        "CPD vs 6-4PP (UV Lesions)": (
            "Two major UV-induced DNA lesions: "
            "  CPD (cyclobutane pyrimidine dimer): covalent bond between adjacent pyrimidines (TT most common); "
            "    strong helix distortion but XPC is POOR sensor → requires DDB2 assistance; "
            "    Pol η (POLH) bypasses CPD accurately (AA insertion) — error-free TLS; "
            "  6-4PP (6-4 photoproduct): between 3'C-5' positions of adjacent pyrimidines; "
            "    MORE helix-distorting than CPD → XPC directly recognises; "
            "    repaired 2-5x faster than CPD; "
            "    not a Pol η substrate → repaired by other TLS pols (more error-prone)"
        ),
        "TFIIH (Transcription Factor IIH)": (
            "10-subunit complex with NER + transcription dual functions; "
            "  NER: XPB (3'→5' helicase) + XPD (5'→3' helicase) unwind DNA bubble; "
            "  Transcription: CDK7-CycH-MAT1 (CAK module) phosphorylates RNA Pol II CTD Ser5; "
            "  ERCC2 (XPD) mutations → TTD (transcription impaired, NER partial) vs XP (NER impaired only); "
            "  ERCC3 (XPB) mutations → similar spectrum: XP-B / TTD / XP+CS; "
            "TFIIH protein level reduction → TTD (sulphur-rich protein transcription impaired → brittle hair)"
        ),
        "Trichothiodystrophy (TTD)": (
            "Neurodevelopmental disorder with brittle hair (low-sulphur) + ichthyosis + intellectual disability; "
            "  tiger-tail banding on polarised light microscopy = PATHOGNOMONIC; "
            "  caused by mutations in ERCC2, ERCC3, GTF2H5, MPLKIP (TTDN1); "
            "  ERCC2 most common TTD gene (~50% of TTD); "
            "  MECHANISM: TFIIH level reduced → impaired Pol II transcription of cysteine-rich proteins "
            "    (hair structural proteins: KRTAP family); "
            "  NO SKIN CANCER (NER partially preserved or TCR compensates); "
            "  PHOTOSENSITIVITY present (reduced GGR at non-transcribed loci)"
        ),
        "UV Avoidance Protocol (All XP Groups — MANDATORY)": (
            "LIFE-LONG STRICT UV AVOIDANCE REQUIRED IN ALL XP GROUPS: "
            "  1. UV-protective clothing: UPF50+ long sleeves, pants, gloves, socks, full-brim hat 360°; "
            "  2. UV-absorbing face visor/shield outdoors; "
            "  3. Broadspectrum SPF50+ sunscreen: mineral (zinc oxide/titanium dioxide preferred); "
            "     reapply every 2 hours; "
            "  4. UV-absorbing UV400 eyewear (wraparound) outdoors; "
            "  5. ALL car/house/school windows: apply UV-blocking film (blocks UVA + UVB); "
            "     car glass = blocks UVB but NOT UVA — film MANDATORY; "
            "  6. Indoor lighting: replace fluorescent lamps with LED (fluorescent can emit UV); "
            "  7. UV-blocking contact lenses (or prescription UV-blocking lenses); "
            "  8. XP patient passport: inform all healthcare providers of diagnosis"
        ),
        "PHOTOSENSITISING DRUGS — ABSOLUTELY CONTRAINDICATED IN ALL XP": (
            "The following drug classes cause photosensitisation → dramatically increase UV-induced lesions → "
            "in XP (impaired NER): accumulation → accelerated carcinogenesis and skin damage: "
            "  TETRACYCLINES: doxycycline, minocycline, tetracycline (phototoxic — ABSOLUTELY CI); "
            "  FLUOROQUINOLONES: ciprofloxacin, levofloxacin (phototoxic — ABSOLUTELY CI); "
            "  HYDROCHLOROTHIAZIDE: thiazide diuretic (photoallergic — ABSOLUTELY CI); "
            "  NSAIDs: ibuprofen, naproxen, piroxicam (photoallergic — AVOID); "
            "  AMIODARONE: antiarrhythmic (strong phototoxic — ABSOLUTELY CI); "
            "  SULFONAMIDES: sulfamethoxazole/TMP (photosensitising — AVOID); "
            "  PHENOTHIAZINES: chlorpromazine (phototoxic — ABSOLUTELY CI); "
            "  PSORALENS (PUVA therapy): ABSOLUTELY CI (designed to damage DNA); "
            "  VORICONAZOLE: antifungal — photosensitising + SCC risk elevated; "
            "  ALTERNATIVE ANTIBIOTICS: use amoxicillin, cephalosporins, azithromycin; "
            "  ALTERNATIVE ANTIFUNGALS: fluconazole, itraconazole; "
            "  ALWAYS check photosensitisation profile before ANY new medication in XP patient"
        ),
        "UDS (Unscheduled DNA Synthesis) Assay": (
            "Gold-standard functional NER assay: "
            "  Fibroblasts UV-irradiated → incubated with [³H]thymidine (outside S-phase); "
            "  UDS = thymidine incorporation = NER gap-fill activity; "
            "  autoradiography (silver grains per nucleus) or scintillation counting; "
            "NORMAL: ~100% UDS; "
            "XP-A: 0-4% (most severe); XP-B: 10-30%; XP-C: 5-25%; XP-D: 10-50%; "
            "XP-E: 30-50% (mildest NER-defective); XP-F: 20-40%; XP-G: 2-25%; "
            "XP-V (POLH): NORMAL UDS ~100% (NER intact — TLS defect only); "
            "COMPLEMENTATION TESTING: fuse two XP cell lines → UDS restored if different groups"
        ),
        "DEB (Diepoxybutane) Test — Fanconi Anemia Diagnosis": (
            "Gold-standard chromosomal fragility test for Fanconi anemia (all groups including FANCQ/ERCC4): "
            "  Lymphocytes exposed to DEB (interstrand crosslinking agent); "
            "  FA cells: markedly elevated chromosomal breaks, gaps, radials, exchanges; "
            "  Normal cells: minimal DEB-induced breaks; "
            "FA threshold: radial formation rate > 0.05 per cell (vs < 0.01 normal); "
            "FANCQ (biallelic ERCC4 ICL alleles): DEB-POSITIVE despite some NER activity; "
            "ICL repair defect → DEB hypersensitivity; "
            "ERCC4 allele distinction: "
            "  ICL-selective alleles → FA(Q) phenotype + DEB positive; "
            "  NER-selective alleles → XP-F phenotype + DEB variable"
        ),
        "Pol η (POLH) — Y-family TLS Polymerase": (
            "DNA polymerase eta (Pol η): "
            "  encoded by POLH; expressed in all dividing tissues; localises to replication foci after UV; "
            "  Y-family: low fidelity on undamaged DNA (1 in 100-1000); "
            "  UNIQUELY ACCURATE on CPD: inserts AA opposite TT (CPD) → matches correct sequence; "
            "  MECHANISM: oversized active site accommodates bulky CPD without conformational stress; "
            "  POLH LOF: "
            "    error-prone TLS pols (Pol κ, Pol ι, Rev1) bypass CPD with increased error rate; "
            "    accumulation of C→T, CC→TT mutations at UV dipyrimidines; "
            "    p53 Arg248Trp (CpC→TpT from UV at TCC→TCC site) → p53 LOF; "
            "    PTCH1 mutations → basal cell carcinoma; CDKN2A mutations → melanoma; "
            "  POLH LOSS IN CANCER: some cancers silence POLH → increased UV mutagenesis"
        ),
        "Cockayne Syndrome (CS) vs XP Overlap": (
            "Cockayne syndrome = TCR-specific disorder (CSA/CSB mutations, rarely XPG/XPB/XPD): "
            "  CS features: cachectic dwarfism; premature ageing; SNHL; retinal degeneration; "
            "    intracranial calcifications; intellectual disability; NO skin cancer (GGR intact); "
            "  XP features: UV hypersensitivity; skin cancer; variable neurodegeneration; "
            "OVERLAP SYNDROMES: "
            "  XPB-CS: ERCC3 mutations affecting both NER + transcription → XP+CS; "
            "  XPD-CS: ERCC2 severe alleles → XP+CS; "
            "  XPG-CS: ERCC5 null → XP+CS (most severe — worst prognosis of combined); "
            "MECHANISTIC BASIS: "
            "  CS = TCR defect; XP = GGR+TCR defect or GGR defect; "
            "  XPG stabilises TFIIH in TCR → XPG null → TCR fails despite CSB intact → CS features"
        ),
    }

    standards = [
        "ACMG/AMP Variant Classification: XP genes classified using ClinVar + LOVD databases",
        "ESPCG (European Study Group on Chromosomal Instability): XP/FA diagnostic criteria 2023",
        "EADV Guidelines: Photoprotection in XP — graded UV avoidance standard; SPF50+ + UPF50+ clothing",
        "XP Society Medical Advisory Board: UV-blocking protocol; annual dermatology; photosensitising drug CI list",
        "Fanconi Anemia Research Fund (FARF): FANCQ (ERCC4) diagnostic and management guidelines 2024",
        "OMIM: XP-A 278700; XP-B 610651; XP-C 278720; XP-D 278730; XP-E 278740; XP-F 278760; XP-G 278780; XP-V 278750",
        "Cleaver JE, DNA Repair 2016: XP genotype-phenotype correlation; UV-induced mutagenesis spectrum",
        "Lehmann AR, Nat Rev Mol Cell Biol 2011: TFIIH and XPD in NER and transcription",
        "Masutani C et al., Nature 1999: POLH cloning and XP-V causal gene — original paper",
        "Nance MA, Berry SA 1992: Cockayne syndrome — Clinical classification — ORIGINAL description",
    ]

    return {"atlas": "Hereditary NER/XP Atlas", "glossary": glossary, "standards": standards}
