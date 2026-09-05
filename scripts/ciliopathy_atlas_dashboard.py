#!/usr/bin/env python3
"""Ciliopathy Atlas — Complete 8-Gene Hereditary Ciliopathy Atlas
BBS1    (Bardet-Biedl Syndrome 1 — 593 aa; 11q13.2; BBSome core subunit BBS1;
         AR; RP + obesity + polydactyly + renal anomalies + hypogonadism;
         NO specific drug — weight management + retinal surveillance) ·
BBS10   (Bardet-Biedl Syndrome 10 — 723 aa; 12q21.2; Chaperonin-like BBS chaperonin complex;
         AR; most severe BBS phenotype; earliest retinal onset; MKKS-related fold) ·
CEP290  (Joubert/Meckel/NPHP6/LCA — 2479 aa; 12q21.32; centrosomal protein 290 kDa;
         AR; pan-ciliopathy gene: LCA (retina only) → JBTS → NPHP → MKS lethal;
         molar tooth sign PATHOGNOMONIC Joubert; ataluren splice modulation trials) ·
NPHP1   (Nephronophthisis type 1 — 732 aa; 2q13; nephrocystin-1 coiled-coil + SH3;
         AR; most common genetic cause renal-only NPH; large deletion 2q13 (MLPA);
         medullary cystic fibrosis kidney — ESRD teen years; ACE-I protective) ·
DNAI1   (Primary Ciliary Dyskinesia type 1 — 699 aa; 9p21.2; ODA intermediate chain IC1;
         AR; situs inversus 50% KARTAGENER triad; bronchiectasis; male infertility;
         nasal NO <77 nL/min DIAGNOSTIC; azithromycin prophylaxis) ·
DYNC2H1 (Short-rib thoracic dystrophy 3 / Jeune asphyxiating — 4307 aa; 11q22.3;
         IFT dynein heavy chain 2; AR; narrow thorax → respiratory failure neonatal;
         rib-elongation surgery + ventilator; renal+retinal surveillance long-term) ·
TMEM67  (Joubert/Meckel syndrome 6 — 995 aa; 8q22.1; Meckelin transmembrane protein;
         AR; JBTS6 molar tooth sign; hepatic fibrosis DISTINCTIVE; renal cysts;
         ursodeoxycholic acid hepatic; portal hypertension management) ·
RPGRIP1L (NPHP/Joubert/Meckel overlap — 1315 aa; 16q12.2; RPGRIP1-Like transition zone;
          AR; JBTS7/NPHP8/MKS5; coloboma + Leber amaurosis + cerebellar vermis hypoplasia;
          ACE-I for ESRD; retinal dystrophy surveillance)
320-patient aggregate cohort (8 × 40, seeds 1206–1213)
"""

import random

SEED_BASE = 1206

CILIOPATHY_GENES = [
    # ── BBS1 — Bardet-Biedl Syndrome 1 ─────────────────────────────────────────
    {
        "gene": "BBS1",
        "protein": "Bardet-Biedl syndrome 1 protein (BBSome core)",
        "alias": (
            "BBS1; OMIM gene 209901; 11q13.2; 593 aa; BBS1 OMIM #209900; "
            "AR biallelic; BBSome core component (BBS1/2/4/5/7/8/9/18 octameric complex); "
            "p.Met390Arg — most common BBS variant worldwide (~20% BBS1 alleles)"
        ),
        "aa": "593 aa",
        "kDa": "~69 kDa",
        "gene_class": (
            "BBSome structural core — BBS1 is the adaptor/hub of the octameric BBSome complex; "
            "BBSome coats cargo vesicles within the ciliary membrane for IFT-B-dependent anterograde transport; "
            "BBS1 directly binds RAB8A·GTP and ARL6 (BBS3) to recruit BBSome to cilia; "
            "BBS1 LOF → BBSome disassembled → IFT-B stalls → GPCR/signalling receptor missorting in cilia; "
            "Consequences: photoreceptors lose rhodopsin transport → rod degeneration → RP; "
            "hypothalamic GPCR (MC4R/LEPR) missorted → leptin resistance → hyperphagia obesity; "
            "renal tubular polarity defect → nephronophthisis-like cysts; "
            "p.Met390Arg (c.1169T>G): founder variant — 20% BBS1 alleles worldwide; "
            "large deletions (exon 14-17) 8%: MLPA essential in p.Met390Arg negative patients; "
            "chaperonin complex (BBS6/BBS10/BBS12) required for BBSome assembly — BBS10 same complex"
        ),
        "locus": "11q13.2",
        "omim_gene": 209901,
        "omim_disease": 209900,
        "phenotype": (
            "Rod-cone dystrophy (RP): onset 5-10 years; night blindness first; "
            "photoreceptor degeneration → legal blindness 20s-30s; "
            "Obesity: hyperphagia onset infancy; typically BMI >35; metabolic syndrome; "
            "Polydactyly: postaxial (little-finger/toe side) — 70% hands, 80% feet; "
            "Renal anomalies: nephronophthisis-like cysts (50%); structural defects; "
            "CKD 20-30%; ESRD uncommon before age 40 in BBS1; "
            "Hypogonadism: male — cryptorchidism, small testes, infertility; "
            "female — irregular cycles, structural uterine anomalies (30%); "
            "Cognitive: learning difficulties 50%; developmental delay mild-moderate; "
            "Anosmia 30%; dental crowding common; ataxia/poor coordination"
        ),
        "hallmark": (
            "CLASSIC PENTAD: RP + obesity + polydactyly + renal anomalies + hypogonadism; "
            "p.Met390Arg: MOST COMMON BBS VARIANT WORLDWIDE — test first in suspected BBS1; "
            "POSTAXIAL POLYDACTYLY: removed at birth in most patients — document digit count; "
            "RETINAL SURVEILLANCE: annual ERG + OCT from diagnosis — retinal onset precedes visual loss; "
            "LEPTIN RESISTANCE: weight management does NOT respond to leptin therapy — MC4R pathway disrupted; "
            "RENAL MONITORING: annual renal ultrasound + eGFR from diagnosis — cysts predispose to UTIs and CKD; "
            "MLPA for large deletions: 8% of BBS1 alleles missed by sequencing"
        ),
        "treatment_alert": (
            "NO APPROVED GENE THERAPY for BBS1 yet (trials ongoing); "
            "RETINAL: avoid bright light exposure (UV400 lenses); low-vision aids; dark-adaptation support; "
            "vitamin A supplementation evidence weak — consult specialist before prescribing; "
            "OBESITY: structured dietary programme; bariatric surgery has been used (response moderate — MC4R pathway); "
            "RENAL: annual ultrasound + eGFR; ACE inhibitor for proteinuria; avoid nephrotoxic drugs; "
            "HYPOGONADISM: testosterone replacement in males (cryptorchidism corrected early to preserve fertility potential); "
            "GROWTH HORMONE: not routinely recommended — short stature management dietary; "
            "ANOSMIA: safety alert — gas + smoke detectors mandatory"
        ),
        "key_ddx": (
            "BBS10 (same BBSome chaperonin complex — more severe, earlier onset); "
            "ALSTROM syndrome (ALMS1 — no polydactyly, cardiomyopathy prominent, AR); "
            "Cohen syndrome (VPS13B — microcephaly, prominent incisors, no RP early); "
            "McKusick-Kaufman (MKKS — hydrometrocolpos, heart defects, no RP); "
            "Joubert syndrome (CEP290/TMEM67 — molar tooth sign, no polydactyly typical)"
        ),
        "gfr_pattern": "Progressive decline from 3rd-4th decade; medullary cysts; proteinuria <1g/d typical",
        "proteinuria_pattern": "Mild tubular proteinuria; glomerular proteinuria if advanced CKD",
        "primary_complication": "Rod-cone dystrophy → legal blindness 20s-30s; obesity comorbidity",
        "disease_detail": (
            "BBS1: the most common BBS gene (~20-23% of BBS families). "
            "The BBSome is an octameric protein complex that coats vesicles within the ciliary membrane. "
            "BBS1 LOF leads to BBSome disassembly, failure of ciliary GPCR transport, "
            "and downstream leptin/melanocortin pathway dysfunction causing refractory obesity. "
            "Retinal dystrophy is the defining feature — management focuses on surveillance and low-vision support."
        ),
        "inheritance": "Autosomal recessive (AR)",
        "variants": [
            {"name": "p.Met390Arg (worldwide founder)", "frequency": "~20% BBS1 alleles"},
            {"name": "Exon 14-17 deletion (MLPA)", "frequency": "~8% alleles"},
            {"name": "p.Arg160Gln (European)", "frequency": "~4% alleles"},
        ],
        "drug_ci": [
            "LEPTIN THERAPY: does NOT work in BBS obesity — MC4R/LEPR ciliary transport disrupted; do not prescribe",
            "AMINOGLYCOSIDES: renal monitoring mandatory — BBS kidneys more susceptible to nephrotoxic injury",
            "VITAMIN A HIGH-DOSE: avoid without specialist supervision — limited evidence, toxicity risk",
        ],
    },

    # ── BBS10 — Bardet-Biedl Syndrome 10 ───────────────────────────────────────
    {
        "gene": "BBS10",
        "protein": "Bardet-Biedl syndrome 10 protein (chaperonin-like BBS complex)",
        "alias": (
            "BBS10; OMIM gene 610148; 12q21.2; 723 aa; BBS10 OMIM #209900; "
            "AR biallelic; BBS chaperonin complex (BBS6/BBS10/BBS12 — Group II chaperonins); "
            "most severe BBS phenotype; p.Cys91LeufsX5 most common — European founder"
        ),
        "aa": "723 aa",
        "kDa": "~80 kDa",
        "gene_class": (
            "BBS10 is a vertebrate-specific Group II chaperonin-like protein (CCT/TRiC fold) "
            "in the trimeric BBS6/BBS10/BBS12 chaperonin complex; "
            "this complex is required for BBSome octamer assembly — BBS10 acts as an ATP-binding scaffold "
            "that folds BBS7 and BBS8 so they can join the BBSome; "
            "BBS10 LOF → BBSome cannot assemble → IFT-B cargo transport failure; "
            "BBS10 is the SECOND most common BBS gene (~20% families) after BBS1; "
            "BBS10 mutations cause MORE SEVERE phenotype than BBS1: "
            "earlier rod-cone dystrophy onset, more severe obesity, higher renal involvement, "
            "more frequent cognitive impairment; "
            "p.Cys91LeufsX5: common European founder frameshift — truncates chaperonin apical domain; "
            "p.Thr525TrpfsX43: Middle Eastern founder; "
            "Group II chaperonins require ATP hydrolysis — BBS10 APTase motif mutations severe"
        ),
        "locus": "12q21.2",
        "omim_gene": 610148,
        "omim_disease": 209900,
        "phenotype": (
            "Rod-cone dystrophy: EARLIER onset than BBS1 — fundus changes age 3-5; "
            "electroretinographic extinction by teens in severe alleles; "
            "Obesity: severe (BMI 40-50); earliest infantile hyperphagia; "
            "Polydactyly: postaxial 75%; often bilateral hands + feet; "
            "Renal: cystic nephropathy 55%; higher ESRD rate than BBS1 — ESRD 20-30% by 30s; "
            "Cognitive: moderate learning difficulties 60%; IQ 50-70 range common; "
            "Hypogonadism: males — micropenis, cryptorchidism; females — primary amenorrhoea; "
            "Diabetes type 2: 30-40% by 4th decade (obesity + insulin resistance); "
            "Hepatic steatosis: 25% (metabolic syndrome related); "
            "Dental: hypodontia; peg-shaped teeth"
        ),
        "hallmark": (
            "MORE SEVERE than BBS1: earlier retinal dystrophy + higher renal + more cognitive impairment; "
            "p.Cys91LeufsX5: MOST COMMON European BBS10 allele — frameshift truncating chaperonin; "
            "INFANTILE HYPERPHAGIA: weight gain first year of life — document feeding history; "
            "ERG EXTINCT BY TEENS in severe BBS10: annual ERG essential to track progression; "
            "RENAL CYSTS: medullary + cortical — higher rate ESRD vs BBS1 — annual ultrasound + GFR; "
            "DIABETES SCREENING: HbA1c from age 10 — obesity-related insulin resistance; "
            "COGNITIVE: formal neuropsychological testing — SEN support early"
        ),
        "treatment_alert": (
            "EARLY OPHTHALMOLOGICAL INPUT: fundoscopy + ERG from birth (or diagnosis); retinal specialist; "
            "BARIATRIC SURGERY: considered earlier than general population due to severe early obesity; "
            "RENAL: ACE inhibitor for proteinuria; avoid NSAIDs (nephrotoxic + renal cysts = double risk); "
            "RENAL TRANSPLANT: indicated earlier due to higher ESRD rate — living donor planning; "
            "DIABETES: GLP-1 agonists (semaglutide) showing promise in BBS obesity — off-label use; "
            "HYPOGONADISM: testosterone replacement early — improve bone density and metabolic risk; "
            "ENDOCRINE SURVEILLANCE: thyroid (hypothyroidism 15%), growth hormone axis from childhood"
        ),
        "key_ddx": (
            "BBS1 (same disease — BBS10 more severe; p.Met390Arg for BBS1 vs p.Cys91LeufsX5 BBS10); "
            "Alstrom syndrome (ALMS1 — dilated cardiomyopathy EARLY, no polydactyly, sensorineural HL); "
            "BBS12 (same chaperonin complex — similar severity); "
            "Prader-Willi syndrome (15q11.2 imprinting — hypotonia infantile, no polydactyly, no RP)"
        ),
        "gfr_pattern": "Earlier ESRD than BBS1; medullary + cortical cysts; renal structural anomalies",
        "proteinuria_pattern": "Tubular + glomerular proteinuria with progressive CKD",
        "primary_complication": "Early retinal extinction + severe obesity + CKD/ESRD by 4th decade",
        "disease_detail": (
            "BBS10: the chaperonin-like BBS gene responsible for the most severe BBS phenotype. "
            "The BBS10/BBS6/BBS12 complex is required for BBSome assembly. "
            "BBS10 LOF prevents BBSome formation, leading to comprehensive ciliary signalling failure. "
            "Key distinction from BBS1: earlier retinal onset, more severe obesity, higher renal failure rate."
        ),
        "inheritance": "Autosomal recessive (AR)",
        "variants": [
            {"name": "p.Cys91LeufsX5 (European founder)", "frequency": "Most common BBS10 allele"},
            {"name": "p.Thr525TrpfsX43 (Middle Eastern)", "frequency": "Common in consanguineous"},
            {"name": "p.Arg267Ter (truncation)", "frequency": "Severe phenotype"},
        ],
        "drug_ci": [
            "NSAIDs: ABSOLUTELY CONTRAINDICATED with renal cystic disease — nephrotoxicity + cyst inflammation",
            "AMINOGLYCOSIDES: avoid or minimise — renal monitoring mandatory if essential",
            "LEPTIN THERAPY: not effective — same BBSome pathway disruption as BBS1",
        ],
    },

    # ── CEP290 — Joubert/Meckel/NPHP6/LCA ─────────────────────────────────────
    {
        "gene": "CEP290",
        "protein": "Centrosomal protein of 290 kDa (nephrocystin-6)",
        "alias": (
            "CEP290 / NPHP6 / MKS4 / BBS14 / JBTS5 / LCA10; OMIM gene 610142; 12q21.32; 2479 aa; "
            "AR biallelic (AD for isolated LCA10 deep intronic c.2991+1655A>G); "
            "pan-ciliopathy gene: LCA → JBTS → NPHP → MKS (lethal); "
            "IVS26 deep intronic variant c.2991+1655A>G creates cryptic exon — CEP290-LCA10"
        ),
        "aa": "2479 aa",
        "kDa": "~290 kDa",
        "gene_class": (
            "CEP290 is a giant scaffolding protein of the ciliary transition zone (TZ) and centrosome; "
            "localises to: (1) ciliary TZ Y-links — gates ciliary entry/exit; "
            "(2) centrosome pericentriolar material; (3) mitotic spindle poles; "
            "CEP290 forms the TZ complex with NPHP1/4/8, RPGRIP1L, MKS1/3/6 — "
            "LOF → TZ gate failure → unrestricted passage of non-ciliary proteins → ciliary dysfunction; "
            "CEP290 is the MOST MUTATED gene in Joubert syndrome (JBTS5 — ~15% JBTS families); "
            "also most common LCA gene (LCA10 — ~15% all LCA); "
            "Phenotypic spectrum: SEVERITY DEPENDS ON ALLELE CLASS: "
            "homozygous c.2991+1655A>G → ISOLATED retinal (LCA10 mildest); "
            "compound heterozygous severe + mild → JBTS (brain + retina + renal); "
            "two severe truncating alleles → MKS (lethal — exencephaly + polydactyly + renal agenesis); "
            "ALLELE-SPECIFIC ANTISENSE OLIGONUCLEOTIDE (sepofarsen) targets IVS26 c.2991+1655A>G — FDA breakthrough"
        ),
        "locus": "12q21.32",
        "omim_gene": 610142,
        "omim_disease": 216360,
        "phenotype": (
            "LCA10 (isolated retinal): severe visual impairment at birth; nystagmus; "
            "fundus normal initially → progressive photoreceptor degeneration; "
            "JBTS5 (Joubert syndrome): molar tooth sign on MRI (vermis hypoplasia + PCB superior cerebellar) "
            "PATHOGNOMONIC; ataxia; intellectual disability 60%; oculomotor apraxia; "
            "renal cysts 30% (NPHP → ESRD); retinal dystrophy 50%; coloboma; "
            "MKS4 (Meckel-Gruber, lethal): exencephaly/anencephaly; renal agenesis; "
            "postaxial polydactyly; pancreatic fibrosis; lethal perinatally; "
            "NPHP6: isolated nephronophthisis phenotype (rare); "
            "BBS14: overlapping BBS features (rare)"
        ),
        "hallmark": (
            "MOLAR TOOTH SIGN (MRI axial): cerebellar vermis aplasia + thickened superior cerebellar peduncles "
            "= PATHOGNOMONIC Joubert syndrome — present in all JBTS regardless of gene; "
            "c.2991+1655A>G: MOST COMMON LCA VARIANT WORLDWIDE — deep intronic, missed by exome if not targeted; "
            "PAN-CILIOPATHY GENE: same gene, allele class determines severity: LCA → JBTS → MKS; "
            "SEPOFARSEN (ASO): antisense oligonucleotide targets IVS26 variant — breakthrough therapy trials; "
            "MKS PRENATAL: if previous MKS4 child, offer early prenatal diagnosis — lethal condition; "
            "APNOEA + HYPERPNOEA: neonatal episodic — JBTS CNS breathing pattern, resolves with age"
        ),
        "treatment_alert": (
            "SEPOFARSEN (QR-110): intravitreal ASO targeting IVS26 c.2991+1655A>G — Phase 2/3 trials; "
            "patient must be HOMOZYGOUS for c.2991+1655A>G to qualify; deep intronic variant only; "
            "GENE THERAPY (AAV-CEP290): Phase 2 trials for LCA10 — CEP290 too large for standard AAV; "
            "split-intein AAV or mini-CEP290 approaches under study; "
            "RENAL: ACE inhibitor from proteinuria onset; annual eGFR + renal ultrasound; "
            "NEONATAL JBTS: APNOEA MONITOR mandatory — episodic apnoea can be life-threatening; "
            "RESPIRATORY: no specific pharmacotherapy for hyperpnoea; supportive; "
            "OPHTHALMOLOGY: retinal specialist from birth; low-vision aids; school support; "
            "MKS: perinatal palliative care team involvement; genetic counselling for future pregnancies"
        ),
        "key_ddx": (
            "TMEM67/MKS3 (Joubert with hepatic fibrosis DISTINCTIVE — overlap JBTS); "
            "RPGRIP1L/NPHP8 (Joubert/NPHP overlap — coloboma prominent); "
            "INPP5E/JBTS1 (Joubert — insulin resistance, retinal dystrophy, early-onset); "
            "AHI1/JBTS3 (Joubert — retinal dystrophy common, good brain outcomes); "
            "RPGR (X-linked RP — no molar tooth sign, no renal)"
        ),
        "gfr_pattern": "NPHP pattern: medullary cysts, inability to concentrate urine, ESRD teens-20s (JBTS with renal)",
        "proteinuria_pattern": "Tubular proteinuria (NPHP) + glomerular if advanced; coloboma renal association",
        "primary_complication": "Molar tooth sign + LCA; ESRD in JBTS with renal; lethal MKS alleles",
        "disease_detail": (
            "CEP290: the archetypal pan-ciliopathy gene. "
            "Phenotypic outcome is entirely allele-class dependent. "
            "The IVS26 deep intronic c.2991+1655A>G variant is uniquely important: "
            "it creates a cryptic exon causing frameshift in retinal cells only (tissue-specific splicing). "
            "Sepofarsen antisense oligonucleotide corrects this splicing defect — first rational genetic therapy for CEP290-LCA10."
        ),
        "inheritance": "Autosomal recessive (AR); IVS26 variant may appear dominant in LCA10 pedigrees",
        "variants": [
            {"name": "c.2991+1655A>G IVS26 (deep intronic — LCA10)", "frequency": "Most common LCA variant worldwide"},
            {"name": "p.Arg151Ter (truncating — JBTS/MKS)", "frequency": "Severe phenotype"},
            {"name": "p.Cys998Ter (truncating — JBTS)", "frequency": "Common in consanguineous"},
        ],
        "drug_ci": [
            "SEPOFARSEN: ONLY for homozygous c.2991+1655A>G — do not use for other CEP290 variants",
            "STANDARD AAV: vector too small for full-length CEP290 (8.4 kb CDS) — use mini-CEP290 or split-intein approaches only",
        ],
    },

    # ── NPHP1 — Nephronophthisis type 1 ────────────────────────────────────────
    {
        "gene": "NPHP1",
        "protein": "Nephrocystin-1 (coiled-coil + SH3 domain)",
        "alias": (
            "NPHP1 / JBTS4; OMIM gene 607100; 2q13; 732 aa; NPH1 OMIM #256100; "
            "AR biallelic; large deletion 2q13 (homozygous) accounts for ~80% NPHP1 alleles; "
            "most common genetic cause of hereditary nephronophthisis (25-30% all NPHP)"
        ),
        "aa": "732 aa",
        "kDa": "~83 kDa",
        "gene_class": (
            "NPHP1 (nephrocystin-1) localises to the ciliary transition zone (TZ) and adherens junctions; "
            "domain structure: coiled-coil (protein interaction) + SH3 domain (signalling) + nuclear export signal; "
            "forms NPHP1/4/8 complex at the TZ — acts as scaffold for CEP290/RPGRIP1L/IQCB1; "
            "NPHP1 LOF → TZ destabilisation → failure of tubular epithelial polarity; "
            "distal tubule and collecting duct: lose ability to concentrate urine → polyuria; "
            "corticomedullary cysts at tubulo-interstitial junctions; "
            "tubulo-interstitial nephritis → fibrosis → ESRD teens-20s; "
            "LARGE GENOMIC DELETION 2q13 (~290 kb homozygous deletion): "
            "accounts for ~80% NPHP1 alleles; standard sequencing MISSES this deletion; "
            "MLPA for 2q13 deletion is MANDATORY when NPHP1 sequence is normal; "
            "NPHP1 deletion also causes Joubert syndrome (JBTS4) in ~2% — cerebellar vermis hypoplasia"
        ),
        "locus": "2q13",
        "omim_gene": 607100,
        "omim_disease": 256100,
        "phenotype": (
            "Nephronophthisis: polyuria + polydipsia (tubular concentrating defect) — FIRST SYMPTOM; "
            "normotensive anaemia of CKD (haemoglobin falls before creatinine rises significantly); "
            "ESRD: median age 13 years; corticomedullary cysts (ultrasound: normal to small kidneys + cysts); "
            "NO oedema/hypertension early — unlike glomerular disease; "
            "Retinal: Senior-Loken syndrome (SLS): NPHP1 + rod-cone dystrophy in ~15%; "
            "Joubert (JBTS4): NPHP1 deletion in ~2% JBTS — cerebellar vermis hypoplasia + molar tooth; "
            "Situs inversus: rare NPHP1-associated PCD overlap; "
            "Growth failure: CKD-related from pre-ESRD phase"
        ),
        "hallmark": (
            "2q13 LARGE DELETION: MOST COMMON NPHP1 ALLELE — MLPA MANDATORY if sequencing negative; "
            "NORMOTENSIVE CKD: NPHP causes ESRD WITHOUT hypertension or oedema early — "
            "creatinine rise a late sign; eGFR decline tracks polyuria progression; "
            "POLYURIA FIRST SYMPTOM: enuresis nocturna + polydipsia in school-age child = renal workup; "
            "SMALL KIDNEYS + CYSTS: opposite of PKD1 (large kidneys); "
            "ANAEMIA EARLY: haemoglobin falls disproportionately — EPO deficiency from fibrosis; "
            "SENIOR-LOKEN: retinal dystrophy + NPHP = SLS until proven otherwise"
        ),
        "treatment_alert": (
            "RENAL TRANSPLANT: indicated at ESRD (median age 13) — outcomes excellent; "
            "living related donor: affected siblings 25% risk — screen before donation; "
            "ACE INHIBITOR / ARB: antiproteinuric + nephroprotective from diagnosis; "
            "SALT SUPPLEMENTATION: tubular salt-wasting in early NPHP — avoid severe fluid restriction; "
            "EPO THERAPY: anaemia from EPO deficiency — start pre-dialysis when Hb <90 g/L; "
            "GROWTH HORMONE: CKD-related growth failure — recombinant GH indicated pre-transplant; "
            "NO DISEASE-MODIFYING THERAPY available for NPHP1 — mTOR inhibitors no benefit in NPHP (unlike PKD); "
            "MLPA TESTING: request explicitly — standard panels miss the 2q13 deletion"
        ),
        "key_ddx": (
            "PKD1/PKD2 (ADPKD): large kidneys, hypertension early, dominant inheritance; "
            "UMOD (FJHN): hyperuricaemia + gout + CKD, AD, no cysts on US; "
            "CEP290 (JBTS/NPHP6): pan-ciliopathy, molar tooth sign, more severe; "
            "MUC1 (MCKD1): mucin 1 frameshift, AD, limited urine concentrating, no cysts on US; "
            "Senior-Loken without NPHP1 deletion: check IQCB1/SDCCAG8/WDR19"
        ),
        "gfr_pattern": "Tubulointerstitial nephritis; ESRD median age 13; no hypertension or oedema early",
        "proteinuria_pattern": "Tubular proteinuria (low-molecular-weight); <1 g/day typically",
        "primary_complication": "ESRD teens; anaemia; growth failure; Senior-Loken (retinal) in 15%",
        "disease_detail": (
            "NPHP1: the most common hereditary nephronophthisis gene. "
            "The key diagnostic trap: 80% of NPHP1 alleles are a large 2q13 genomic deletion "
            "that standard Sanger and NGS sequencing completely miss. "
            "MLPA is mandatory. NPHP causes normotensive ESRD in early teens — "
            "the clinical presentation is deceptive because hypertension and oedema are absent."
        ),
        "inheritance": "Autosomal recessive (AR)",
        "variants": [
            {"name": "Homozygous 2q13 deletion (~290 kb) — MLPA", "frequency": "~80% NPHP1 alleles"},
            {"name": "Heterozygous 2q13 deletion + frameshift (compound het)", "frequency": "~10%"},
            {"name": "Point mutations p.Arg384Gln / p.Arg637Ter", "frequency": "~10%"},
        ],
        "drug_ci": [
            "MTOR INHIBITORS (everolimus/sirolimus): NO benefit in NPHP nephronophthisis (unlike ADPKD); do not use",
            "AMINOGLYCOSIDES: avoid in NPHP renal disease — tubular toxicity",
            "NEPHROTOXIC CONTRAST: eGFR-guided contrast protocol mandatory",
        ],
    },

    # ── DNAI1 — Primary Ciliary Dyskinesia type 1 ───────────────────────────────
    {
        "gene": "DNAI1",
        "protein": "Dynein axonemal intermediate chain 1 (ODA IC1)",
        "alias": (
            "DNAI1 / CILD1; OMIM gene 604366; 9p21.2; 699 aa; PCD1/KD OMIM #244400; "
            "AR biallelic; outer dynein arm intermediate chain (ODA IC); "
            "IVS1+2_7del (c.1-9_IVS1+9del21bp) founder deletion + p.Trp568Ter common"
        ),
        "aa": "699 aa",
        "kDa": "~78 kDa",
        "gene_class": (
            "DNAI1 encodes the cytoplasmic intermediate chain 1 (IC1) of the outer dynein arm (ODA); "
            "ODA is the main force-generator for ciliary/flagellar beat — 9+2 axoneme has 3 ODAs per 96 nm repeat; "
            "DNAI1 assembles with DNAI2 (IC2) + DNAL4 (LC4) to form the intermediate chain complex "
            "that connects ODA head to the A-tubule via DNAH5/DNAH11 heavy chains; "
            "DNAI1 LOF → ODA absent or dysmorphic → ciliary immotility or dyskinesia; "
            "Effect on ciliary organs: airway cilia → bronchiectasis + sinusitis; "
            "embryonic nodal cilia → situs determination random (situs inversus 50%); "
            "sperm flagella → male infertility (primary ciliary dyskinesia of sperm); "
            "DNAI1 is required for CYTOPLASMIC PRE-ASSEMBLY of ODA (DNAAF proteins): "
            "DNAI1 joins ODA complex in cytoplasm BEFORE transport to axoneme; "
            "IVS1+2_7del: splice site deletion — 17% all PCD alleles worldwide; "
            "p.Trp568Ter: truncating — 10% alleles"
        ),
        "locus": "9p21.2",
        "omim_gene": 604366,
        "omim_disease": 244400,
        "phenotype": (
            "Recurrent respiratory infections from birth: neonatal respiratory distress; "
            "otitis media with effusion (glue ear) — universal; "
            "sinusitis (chronic maxillary/frontal); "
            "bronchiectasis: lower lobe predominant — develops by 2nd decade if undertreated; "
            "Situs inversus totalis: 50% (random situs determination at node); "
            "KARTAGENER TRIAD: situs inversus + bronchiectasis + sinusitis (50% PCD); "
            "Male infertility: immotile sperm (asthenospermia); "
            "Female subfertility: ectopic pregnancy risk (tubal cilia); "
            "Hydrocephalus: rare, CSF flow cilia; "
            "HEARING LOSS: conductive from chronic OME — not sensorineural"
        ),
        "hallmark": (
            "NASAL NITRIC OXIDE <77 nL/min: DIAGNOSTIC screening test — PCD nasal NO dramatically LOW; "
            "KARTAGENER TRIAD: situs inversus + bronchiectasis + chronic sinusitis = PCD until proven otherwise; "
            "NEONATAL RESPIRATORY DISTRESS in term baby: investigate for PCD (no viral cause); "
            "EM: ODA absent on transmission electron microscopy of cilia cross-section; "
            "VIDEO MICROSCOPY: ciliary beat frequency + pattern — immotile or dyskinetic; "
            "MALE INFERTILITY + CHRONIC SINUSITIS: always test for PCD; "
            "SITUS INVERSUS + RESP SYMPTOMS: PCD genetic panel"
        ),
        "treatment_alert": (
            "AIRWAY CLEARANCE: twice-daily chest physiotherapy + airway clearance devices (Acapella, Flutter) — LIFELONG; "
            "HYPERTONIC SALINE (3-7%): nebulised — improves mucociliary clearance; evidence Level B; "
            "AZITHROMYCIN PROPHYLAXIS: long-term 3x/week in bronchiectasis — reduces exacerbations; "
            "PROMPT ANTIBIOTIC TREATMENT: at first signs of respiratory exacerbation; culture-guided; "
            "HEARING: grommets/ventilation tubes for OME with hearing loss — preserves speech/language; "
            "MALE INFERTILITY: ICSI/IVF with sperm retrieval — motility absent but viable; "
            "FEMALE ECTOPIC PREGNANCY: educate + early USS in pregnancy to exclude ectopic; "
            "IMMUNISATIONS: annual influenza + pneumococcal vaccines mandatory; "
            "NO INHALED TOBRAMYCIN without culture guidance (P. aeruginosa colonisation)"
        ),
        "key_ddx": (
            "DNAH5 (most common PCD gene, 28% — same ODA heavy chain; EM ODA absent same); "
            "DNAH11 (PCD — normal EM; subtle ciliary beat pattern abnormality only); "
            "RSPH4A/RSPH9 (PCD — central pair defect; situs solitus more common); "
            "CF (CFTR): bronchiectasis + chronic infection; sweat chloride; nasal NO normal; "
            "Immotile cilia syndrome without gene variant: DNAAF assembly factors"
        ),
        "gfr_pattern": "Normal kidney function in DNAI1 PCD (no renal ciliopathy component typically)",
        "proteinuria_pattern": "Not applicable — renal unaffected in pure DNAI1 PCD",
        "primary_complication": "Progressive bronchiectasis; chronic suppurative lung disease; male infertility",
        "disease_detail": (
            "DNAI1: encodes ODA intermediate chain 1 — critical structural component of respiratory cilia. "
            "Primary Ciliary Dyskinesia from DNAI1 LOF causes immotile cilia, "
            "leading to neonatal respiratory distress, chronic airway infection, and eventual bronchiectasis. "
            "The Kartagener triad (situs inversus + bronchiectasis + sinusitis) in 50% is pathognomonic when present."
        ),
        "inheritance": "Autosomal recessive (AR)",
        "variants": [
            {"name": "IVS1+2_7del (c.1-9_IVS1+9del21bp splice)", "frequency": "~17% PCD alleles worldwide"},
            {"name": "p.Trp568Ter (truncating)", "frequency": "~10%"},
            {"name": "p.Arg571Ter (truncating)", "frequency": "~5%"},
        ],
        "drug_ci": [
            "SYSTEMIC AMINOGLYCOSIDES: avoid repeated courses — ototoxic; PCD hearing already at risk from OME",
            "SEDATING ANTIHISTAMINES: impair mucociliary defence — avoid in PCD respiratory exacerbations",
        ],
    },

    # ── DYNC2H1 — Short-rib thoracic dystrophy / Jeune ─────────────────────────
    {
        "gene": "DYNC2H1",
        "protein": "Cytoplasmic dynein 2 heavy chain 1 (IFT dynein)",
        "alias": (
            "DYNC2H1 / DHC1b / DHC2; OMIM gene 603297; 11q22.3; 4307 aa; SRTD3 OMIM #613091; "
            "AR biallelic; IFT dynein heavy chain — retrograde intraflagellar transport motor; "
            "also causes SRPS type III (lethal) and JBTS34; largest IFT dynein protein"
        ),
        "aa": "4307 aa",
        "kDa": "~500 kDa",
        "gene_class": (
            "DYNC2H1 is the heavy chain of cytoplasmic dynein-2 (IFT dynein), "
            "which powers RETROGRADE IFT (ciliary tip → base); "
            "IFT is the bidirectional transport system essential for ciliary assembly/maintenance: "
            "IFT-B (anterograde, kinesin-2 driven) carries tubulin/axonemal precursors to tip; "
            "IFT-A (retrograde, dynein-2 driven) removes turnover products from ciliary tip; "
            "DYNC2H1 LOF → retrograde IFT failure → IFT particle accumulation at ciliary tip; "
            "anterograde IFT continues → bulging distended cilia with IFT cargo trapped; "
            "hedgehog signalling collapses (GLI processing requires ciliary trafficking); "
            "Skeletal consequence: HH signal loss in growth plates → narrow thorax + short limbs; "
            "Thoracic restriction → RESPIRATORY FAILURE — the life-limiting feature; "
            "DYNC2H1 is the MOST COMMON gene in Jeune asphyxiating thoracic dystrophy (30-40% SRTD cases); "
            "also causes SHORT-RIB POLYDACTYLY SYNDROME type III (lethal) with more severe alleles"
        ),
        "locus": "11q22.3",
        "omim_gene": 603297,
        "omim_disease": 613091,
        "phenotype": (
            "Short-rib thoracic dystrophy 3 (Jeune syndrome): "
            "NARROW CHEST (bell-shaped thorax) on CXR — horizontal short ribs PATHOGNOMONIC; "
            "neonatal/infantile respiratory failure in severe: ventilator dependence; "
            "limb shortening: rhizomelia (proximal); short stature; "
            "Polydactyly: postaxial 10-15% (unlike BBS not typical); "
            "Renal: nephronophthisis-like cysts 50%; ESRD 2nd-3rd decade; "
            "Retinal dystrophy: 30% (RP onset teens); "
            "Hepatic fibrosis: 15-20%; "
            "Dental: dental dysplasia; "
            "Intelligence: usually normal (unlike BBS); "
            "SRPS type III (lethal variant): all above + exencephaly + visceral anomalies"
        ),
        "hallmark": (
            "HORIZONTAL SHORT RIBS + NARROW THORAX: CXR diagnostic — horizontal ribs vs normal oblique; "
            "NEONATAL RESPIRATORY FAILURE: primary cause of death — thoracic cage restriction; "
            "RIB ELONGATION SURGERY: lateral thoracic expansion (VEPTR device) — may save life; "
            "MOST COMMON JATD GENE: 30-40% of Jeune syndrome — test DYNC2H1 first; "
            "HEDGEHOG SIGNALLING LOSS: explains skeletal phenotype — GLI processing ciliary-dependent; "
            "RENAL + RETINAL SURVEILLANCE: long-term survivors develop NPHP + RP; "
            "SRPS vs JATD: severity spectrum — two severe truncating alleles → lethal SRPS"
        ),
        "treatment_alert": (
            "RESPIRATORY: NICU + ventilator support in neonatal period — primary emergency; "
            "VEPTR (Vertical Expandable Prosthetic Titanium Rib): rib-elongation device; "
            "multiple surgeries (every 6 months) to expand chest — improves lung volumes; "
            "initiated from 6 months age; significant surgical risk each expansion; "
            "LUNG TRANSPLANT: considered in non-responding thoracic restriction; rare; "
            "RENAL: ACE inhibitor from proteinuria; annual eGFR + ultrasound; "
            "RETINAL: annual ERG + OCT from age 5 — RP surveillance; "
            "HEPATIC: UDCA if hepatic fibrosis; portal hypertension — variceal screening; "
            "NO DISEASE-MODIFYING THERAPY for DYNC2H1; management is supportive/surgical"
        ),
        "key_ddx": (
            "IFT80/WDR19/IFT140 (other JATD genes — same clinical, different IFT-A complex component); "
            "Short-rib polydactyly type II (Majewski) — lethal, DYNC2H1 severe alleles; "
            "Ellis-van Creveld (EVC/EVC2) — HH pathway, similar thorax but ectodermal features + CHD; "
            "Asphyxiating thoracic dystrophy vs Schwartz-Jampel: myotonia + narrow chest"
        ),
        "gfr_pattern": "NPHP-like nephronophthisis; ESRD 2nd-3rd decade; medullary cysts",
        "proteinuria_pattern": "Tubular proteinuria; glomerular involvement with CKD progression",
        "primary_complication": "Neonatal respiratory failure; ESRD; retinal dystrophy in survivors",
        "disease_detail": (
            "DYNC2H1: encodes the IFT dynein heavy chain responsible for retrograde intraflagellar transport. "
            "Jeune asphyxiating thoracic dystrophy results from HH signalling failure in skeletal growth plates. "
            "The narrow thorax is the primary life-threatening feature — VEPTR rib-expansion surgery is the main intervention. "
            "Long-term survivors develop renal and retinal ciliopathy complications."
        ),
        "inheritance": "Autosomal recessive (AR)",
        "variants": [
            {"name": "p.Arg3004Trp (European)", "frequency": "Most common DYNC2H1 missense"},
            {"name": "p.Leu1110Pro (truncating)", "frequency": "Severe neonatal"},
            {"name": "Exonic deletions (MLPA)", "frequency": "~10% alleles"},
        ],
        "drug_ci": [
            "SEDATION without airway support: hazardous — thoracic restriction worsens under sedation",
            "NEPHROTOXIC AGENTS: avoid with concurrent NPHP renal disease",
        ],
    },

    # ── TMEM67 — Joubert / Meckel syndrome 6 ────────────────────────────────────
    {
        "gene": "TMEM67",
        "protein": "Meckelin (transmembrane protein 67) — transition zone receptor",
        "alias": (
            "TMEM67 / MKS3 / JBTS6 / NPHP11; OMIM gene 609884; 8q22.1; 995 aa; "
            "AR biallelic; transition zone (TZ) membrane protein; "
            "JBTS6 OMIM #610688; MKS3 OMIM #607361; "
            "HEPATIC FIBROSIS + molar tooth sign PATHOGNOMONIC for TMEM67 Joubert"
        ),
        "aa": "995 aa",
        "kDa": "~111 kDa",
        "gene_class": (
            "TMEM67 (Meckelin) is a multi-pass transmembrane protein localised exclusively to the ciliary transition zone; "
            "binds RPGRIP1L, MKS1, CC2D2A forming the B9/TMEM-module of the TZ ciliary gate; "
            "acts as a receptor — extracellular domain interacts with WNT ligands (non-canonical WNT); "
            "TMEM67 LOF → TZ B9-module disintegration → ciliary gate failure + WNT non-canonical disruption; "
            "non-canonical WNT (planar cell polarity) defect → hepatic and renal tubular polarity loss; "
            "UNIQUE FEATURE: TMEM67 is the TZ gene most strongly associated with hepatic involvement; "
            "hepatic fibrosis (congenital hepatic fibrosis CHF) in >60% TMEM67 JBTS; "
            "portal hypertension, varices, splenomegaly, hepatosplenomegaly: earlier than renal ESRD; "
            "TMEM67-JBTS = JBTS subtype with LIVER as dominant extracerebral feature; "
            "allele class: severe truncating → MKS3 (lethal); moderate missense → JBTS6; "
            "p.Cys615Arg: hot-spot missense — disrupts WNT-binding extracellular domain"
        ),
        "locus": "8q22.1",
        "omim_gene": 609884,
        "omim_disease": 610688,
        "phenotype": (
            "Joubert syndrome type 6 (JBTS6): molar tooth sign PATHOGNOMONIC on MRI; "
            "cerebellar vermis aplasia; ataxia; intellectual disability 55%; oculomotor apraxia; "
            "Hepatic: CONGENITAL HEPATIC FIBROSIS (CHF) 60-70% — DISTINCTIVE for TMEM67; "
            "portal hypertension → oesophageal/gastric varices; splenomegaly; "
            "Renal: NPHP-like nephronophthisis 50%; ESRD 2nd-3rd decade; "
            "Retinal dystrophy: rod-cone dystrophy 40%; "
            "MKS3 (Meckel, lethal): occipital encephalocele + renal agenesis + polydactyly; "
            "Coloboma: iris/optic disc 15% JBTS6; "
            "Situs anomalies: rare TMEM67 situs"
        ),
        "hallmark": (
            "MOLAR TOOTH SIGN + HEPATIC FIBROSIS: TMEM67 is THE Joubert gene with liver involvement; "
            "ANY Joubert patient with hepatomegaly/splenomegaly or elevated liver enzymes → test TMEM67 FIRST; "
            "CONGENITAL HEPATIC FIBROSIS: liver biopsy shows periportal fibrosis + bile duct proliferation; "
            "PORTAL HYPERTENSION: oesophageal varices → GI bleed risk — variceal screening mandatory; "
            "URSODEOXYCHOLIC ACID: standard hepatic treatment in CHF — reduces cholestasis; "
            "NEONATAL APNOEA: JBTS breathing pattern (hyperpnoea + apnoea) resolves with age; "
            "p.Cys615Arg: hot-spot missense — JBTS6 phenotype (not MKS)"
        ),
        "treatment_alert": (
            "HEPATIC: URSODEOXYCHOLIC ACID (UDCA 10-15 mg/kg/day) — first-line for CHF/cholestasis; "
            "VARICEAL SURVEILLANCE: upper GI endoscopy every 1-3 years from diagnosis; "
            "PROPRANOLOL: portal hypertension prophylaxis for medium/large varices; "
            "BETA-BLOCKER mandatory in significant portal hypertension — reduces variceal bleed risk; "
            "LIVER TRANSPLANT: considered if end-stage hepatic failure or uncontrollable portal hypertension; "
            "RENAL: ACE inhibitor from proteinuria; annual eGFR; renal transplant at ESRD; "
            "CEREBELLAR: occupational + physiotherapy for ataxia; "
            "OPHTHALMOLOGY: annual ERG + OCT — retinal dystrophy surveillance; "
            "NEONATAL: APNOEA MONITOR + CAFFEINE if needed for apnoea of Joubert"
        ),
        "key_ddx": (
            "CEP290/JBTS5 (Joubert — molar tooth but NO hepatic fibrosis; retinal more severe); "
            "RPGRIP1L/JBTS7 (Joubert — coloboma prominent; hepatic rare); "
            "ARPKD (PKHD1): hepatic fibrosis + renal cysts; NO molar tooth sign; CHF similar; "
            "Caroli disease: biliary dilatation; no molar tooth sign; PKHD1 overlap; "
            "CF liver disease: portal hypertension; CFTR mutations; no cerebellar"
        ),
        "gfr_pattern": "NPHP-like; medullary cysts; tubular proteinuria; ESRD 2nd-3rd decade",
        "proteinuria_pattern": "Tubular (NPHP) + glomerular with progression; hepatorenal combined in late disease",
        "primary_complication": "Portal hypertension + variceal bleeding; ESRD; molar tooth neurological",
        "disease_detail": (
            "TMEM67: the Joubert gene uniquely associated with congenital hepatic fibrosis. "
            "The hepatic involvement precedes and often dominates the clinical picture in childhood. "
            "UDCA + portal hypertension management + variceal surveillance are mandatory in all TMEM67-JBTS patients. "
            "Liver transplant may be needed before renal transplant in some patients."
        ),
        "inheritance": "Autosomal recessive (AR)",
        "variants": [
            {"name": "p.Cys615Arg (TZ WNT-binding domain hot-spot)", "frequency": "Most common JBTS6 missense"},
            {"name": "p.Asn857Ser (moderate — JBTS6)", "frequency": "Recurrent"},
            {"name": "Truncating variants (MKS3 lethal)", "frequency": "Severe alleles"},
        ],
        "drug_ci": [
            "HEPATOTOXIC DRUGS: ALL hepatotoxic medications require dose reduction/avoidance with CHF + portal HTN",
            "NSAIDs: CONTRAINDICATED with portal hypertension (GI bleed risk + nephrotoxicity)",
            "ORAL CONTRACEPTIVES: caution — hepatic metabolism impaired with CHF",
        ],
    },

    # ── RPGRIP1L — NPHP/Joubert/Meckel overlap ─────────────────────────────────
    {
        "gene": "RPGRIP1L",
        "protein": "RPGRIP1-Like protein (NPHP8 / ciliary transition zone scaffold)",
        "alias": (
            "RPGRIP1L / NPHP8 / MKS5 / JBTS7; OMIM gene 610937; 16q12.2; 1315 aa; "
            "AR biallelic; ciliary transition zone C2/coiled-coil scaffold; "
            "JBTS7 OMIM #611560; MKS5 OMIM #611561; NPHP8 OMIM #613937; "
            "coloboma + Leber amaurosis DISTINCTIVE in RPGRIP1L JBTS"
        ),
        "aa": "1315 aa",
        "kDa": "~148 kDa",
        "gene_class": (
            "RPGRIP1L (RPGRIP1-Like) is a scaffolding protein of the ciliary transition zone (TZ); "
            "domain structure: C2 domains (membrane interaction) + coiled-coil (protein-protein) + RPGRIP homology (RH); "
            "forms TZ complex with NPHP1/4 + CEP290 + IQCB1 + TMEM67; "
            "RPGRIP1L anchors NPHP1 and CEP290 to the TZ Y-links — critical for TZ gate assembly; "
            "RPGRIP1L LOF → TZ Y-link disassembly → protein sorting failure → ciliary dysfunction; "
            "OCULAR PHENOTYPE DISTINCTIVE: RPGRIP1L is the TZ gene most associated with "
            "coloboma (uveal, optic disc, iris) AND Leber congenital amaurosis (LCA-type early retinal); "
            "coloboma 30-40% RPGRIP1L JBTS vs 5-10% other JBTS genes; "
            "cerebellar vermis hypoplasia → molar tooth sign; "
            "allele class gradient: severe truncating → MKS5 lethal; moderate → JBTS7; mild → NPHP8 isolated renal; "
            "RPGRIP1L interacts with RPGR (retinitis pigmentosa GTPase regulator) — explains retinal severity"
        ),
        "locus": "16q12.2",
        "omim_gene": 610937,
        "omim_disease": 611560,
        "phenotype": (
            "Joubert syndrome type 7 (JBTS7): molar tooth sign + cerebellar vermis hypoplasia; "
            "Ocular: COLOBOMA (uveal/optic disc/iris) 30-40% — MOST DISTINCTIVE feature vs other JBTS; "
            "Leber amaurosis / rod-cone dystrophy: severe early visual impairment; "
            "Nystagmus: horizontal + vertical; photophobia; "
            "Renal: NPHP-like 40%; ESRD teens-20s; "
            "Intellectual disability: moderate in 50%; ataxia; "
            "MKS5 (lethal): occipital encephalocele + renal agenesis + polydactyly + CVH; "
            "NPHP8: isolated nephronophthisis without cerebellar (mild alleles); "
            "Coloboma: uveal coloboma associated with inferior visual field defect"
        ),
        "hallmark": (
            "MOLAR TOOTH SIGN + COLOBOMA: RPGRIP1L is THE Joubert gene with coloboma — "
            "any Joubert patient with coloboma → test RPGRIP1L first; "
            "LEBER AMAUROSIS PHENOTYPE: severe early retinal dystrophy — ERG extinguished infancy; "
            "COLOBOMA-ASSOCIATED VF DEFECT: inferior altitudinal field loss + optic disc coloboma; "
            "NPHP8 ISOLATED RENAL: milder RPGRIP1L alleles → adult-onset ESRD without cerebellar; "
            "ALLELE-SEVERITY GRADIENT: truncating → MKS5 lethal; missense moderate → JBTS7; hypomorphic → NPHP8; "
            "TZ ANCHOR FOR NPHP1/CEP290: RPGRIP1L LOF destabilises entire TZ complex"
        ),
        "treatment_alert": (
            "COLOBOMA: cannot be repaired — low-vision rehabilitation; tinted lenses for photophobia; "
            "ophthalmological monitoring for coloboma-associated complications (retinal detachment 5%); "
            "RETINAL DYSTROPHY: retinal specialist from infancy — Braille + tactile learning tools early; "
            "RENAL: ACE inhibitor from proteinuria; annual eGFR + renal ultrasound; "
            "RENAL TRANSPLANT: RPGRIP1L NPHP — outcomes good; MHC typing for related donor; "
            "CEREBELLAR: physiotherapy + occupational therapy for ataxia from infancy; "
            "NEONATAL APNOEA: JBTS breathing pattern — apnoea monitor + caffeine if required; "
            "GENE THERAPY: no approved therapy; AAV retinal trials for RPGR/CEP290 ongoing; "
            "COLOBOMA-ASSOCIATED RETINAL DETACHMENT: urgent referral if floaters/shadow in visual field"
        ),
        "key_ddx": (
            "CEP290/JBTS5 (Joubert — retinal severe, NO coloboma typical, IVS26 specific therapy); "
            "TMEM67/JBTS6 (Joubert — hepatic fibrosis DISTINCTIVE, coloboma rare); "
            "PCDH15/USH1F (Usher syndrome — RP + deafness; no molar tooth sign); "
            "Isolated coloboma (PAX6/CHD7): unilateral, non-syndromic; no cerebellar; "
            "CHARGE syndrome (CHD7): coloboma + choanal atresia + ear anomalies + heart; AD"
        ),
        "gfr_pattern": "NPHP-like tubulointerstitial nephritis; ESRD teens-20s (renal involvement cases)",
        "proteinuria_pattern": "Tubular proteinuria (LMW); glomerular involvement late CKD",
        "primary_complication": "Coloboma + severe early retinal dystrophy; ESRD in NPHP8 alleles; molar tooth neurological",
        "disease_detail": (
            "RPGRIP1L: the transition zone scaffold most strongly associated with coloboma in Joubert syndrome. "
            "Coloboma occurs when optic fissure fails to close — RPGRIP1L's role in ciliary signalling during eye development. "
            "The combination of molar tooth sign + coloboma on MRI/slit-lamp exam is virtually diagnostic. "
            "Allele severity determines outcome: mild alleles → isolated NPHP; severe → lethal MKS."
        ),
        "inheritance": "Autosomal recessive (AR)",
        "variants": [
            {"name": "p.Ala229Val (TZ interface, JBTS7)", "frequency": "Recurrent missense"},
            {"name": "p.Arg765Ter (truncating — MKS5)", "frequency": "Severe lethal"},
            {"name": "p.Leu1162Arg (coiled-coil domain)", "frequency": "Moderate JBTS7"},
        ],
        "drug_ci": [
            "NEPHROTOXIC AGENTS: avoid with concurrent NPHP renal disease",
            "LIVE VACCINES: no contraindication in RPGRIP1L; unlike CEP290 immune function preserved",
        ],
    },
]


def _make_cohort(gene_dict: dict, seed: int, n: int = 40) -> list:
    rng = random.Random(seed)
    gene_id = gene_dict["gene"]
    patients = []
    for i in range(n):
        age = rng.randint(1, 60)
        sex = rng.choice(["M", "F"])

        if gene_id == "BBS1":
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[20, 48, 32])[0]
            retinal_dx = rng.random() < 0.92
            obesity = rng.random() < 0.85
            polydactyly = rng.random() < 0.70
            renal = rng.random() < 0.50
            drug_error = rng.random() < 0.14  # leptin given without effect
            dx_delayed = rng.random() < 0.35
            esrd = renal and rng.random() < 0.15
            htn = esrd or rng.random() < 0.12
            transplant = esrd and rng.random() < 0.40

        elif gene_id == "BBS10":
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[12, 42, 46])[0]
            retinal_dx = rng.random() < 0.95
            obesity = rng.random() < 0.92
            polydactyly = rng.random() < 0.75
            renal = rng.random() < 0.55
            drug_error = rng.random() < 0.18  # NSAIDs given with renal cysts
            dx_delayed = rng.random() < 0.30
            esrd = renal and rng.random() < 0.28
            htn = esrd or rng.random() < 0.20
            transplant = esrd and rng.random() < 0.45

        elif gene_id == "CEP290":
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[15, 40, 45])[0]
            lca = rng.random() < 0.50  # isolated LCA or JBTS
            joubert = rng.random() < 0.40
            mks = rng.random() < 0.10
            drug_error = rng.random() < 0.10  # sepofarsen given to wrong allele class
            dx_delayed = rng.random() < 0.45  # deep intronic IVS26 missed by exome
            esrd = joubert and rng.random() < 0.30
            htn = esrd or rng.random() < 0.10
            transplant = esrd and rng.random() < 0.35

        elif gene_id == "NPHP1":
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[18, 42, 40])[0]
            polyuria = rng.random() < 0.95
            esrd = rng.random() < 0.75  # high ESRD rate in NPHP1
            senior_loken = rng.random() < 0.15
            drug_error = rng.random() < 0.22  # mTOR inhibitor given (no benefit)
            dx_delayed = rng.random() < 0.50  # 2q13 deletion missed without MLPA
            htn = esrd or rng.random() < 0.15  # normotensive until late
            transplant = esrd and rng.random() < 0.70

        elif gene_id == "DNAI1":
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[22, 45, 33])[0]
            situs_inversus = rng.random() < 0.50
            bronchiectasis = rng.random() < 0.70
            male_infertile = sex == "M" and rng.random() < 0.90
            drug_error = rng.random() < 0.12  # aminoglycosides repeated without monitoring
            dx_delayed = rng.random() < 0.55  # PCD underdiagnosed
            esrd = rng.random() < 0.02  # renal not typical
            htn = rng.random() < 0.15
            transplant = False

        elif gene_id == "DYNC2H1":
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[8, 30, 62])[0]
            narrow_chest = rng.random() < 0.95
            resp_failure = narrow_chest and rng.random() < 0.45
            renal = rng.random() < 0.50
            retinal = rng.random() < 0.30
            drug_error = rng.random() < 0.08
            dx_delayed = rng.random() < 0.20
            esrd = renal and rng.random() < 0.35
            htn = esrd or rng.random() < 0.15
            transplant = esrd and rng.random() < 0.30
            age = rng.randint(0, 30)  # SRTD affects infancy

        elif gene_id == "TMEM67":
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[15, 40, 45])[0]
            joubert = rng.random() < 0.85
            hepatic_fibrosis = rng.random() < 0.65
            portal_htn = hepatic_fibrosis and rng.random() < 0.45
            renal = rng.random() < 0.50
            drug_error = rng.random() < 0.20  # NSAIDs + portal hypertension = GI bleed
            dx_delayed = rng.random() < 0.38
            esrd = renal and rng.random() < 0.30
            htn = portal_htn or esrd or rng.random() < 0.12
            transplant = esrd and rng.random() < 0.35

        elif gene_id == "RPGRIP1L":
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[15, 42, 43])[0]
            coloboma = rng.random() < 0.35
            retinal_dx = rng.random() < 0.80
            joubert = rng.random() < 0.75
            renal = rng.random() < 0.40
            drug_error = rng.random() < 0.10
            dx_delayed = rng.random() < 0.42
            esrd = renal and rng.random() < 0.30
            htn = esrd or rng.random() < 0.12
            transplant = esrd and rng.random() < 0.38

        else:
            severity = rng.choices(["Mild", "Moderate", "Severe"], weights=[20, 45, 35])[0]
            drug_error = rng.random() < 0.15
            dx_delayed = rng.random() < 0.35
            esrd = rng.random() < 0.25
            htn = esrd or rng.random() < 0.15
            transplant = esrd and rng.random() < 0.35

        surveillance_adherent = rng.random() < 0.58

        p = {
            "id": f"{gene_id}-{i+1:03d}",
            "gene": gene_id,
            "age": age,
            "sex": sex,
            "severity": severity,
            "esrd": esrd,
            "hypertension": htn,
            "transplant": transplant,
            "drug_error": drug_error,
            "dx_delayed": dx_delayed,
            "surveillance_adherent": surveillance_adherent,
        }
        patients.append(p)

    return patients


def _cohort_stats(patients: list) -> dict:
    n = len(patients)
    if n == 0:
        return {}
    def pct(key):
        return round(sum(1 for p in patients if p.get(key)) / n * 100, 1)
    sev = {s: round(sum(1 for p in patients if p["severity"] == s) / n * 100, 1)
           for s in ["Mild", "Moderate", "Severe"]}
    return {
        "n": n,
        "esrd_pct": pct("esrd"),
        "htn_pct": pct("hypertension"),
        "transplant_pct": pct("transplant"),
        "drug_error_pct": pct("drug_error"),
        "dx_delayed_pct": pct("dx_delayed"),
        "surveillance_adherent_pct": pct("surveillance_adherent"),
        "severity": sev,
    }


def _build_all_patients():
    all_patients = []
    for idx, gene in enumerate(CILIOPATHY_GENES):
        seed = SEED_BASE + idx
        cohort = _make_cohort(gene, seed=seed, n=40)
        all_patients.extend(cohort)
    return all_patients


ALL_PATIENTS = _build_all_patients()


# ── API response functions ────────────────────────────────────────────────────

def get_overview() -> dict:
    n = len(ALL_PATIENTS)
    agg = _cohort_stats(ALL_PATIENTS)
    return {
        "atlas_name": "Ciliopathy Atlas",
        "atlas_subtitle": (
            "Complete 8-Gene Hereditary Ciliopathy Atlas — "
            "BBS1 · BBS10 · CEP290 · NPHP1 · DNAI1 · DYNC2H1 · TMEM67 · RPGRIP1L"
        ),
        "n_genes": 8,
        "n_patients": n,
        "seeds": "1206–1213",
        "description": (
            "Comprehensive hereditary ciliopathy reference covering the 8 most clinically significant "
            "monogenic ciliary dysfunction syndromes: "
            "Bardet-Biedl syndrome BBS1 (AR — BBSome core; p.Met390Arg founder; RP+obesity+polydactyly+renal; "
            "leptin therapy DOES NOT WORK — BBSome MC4R pathway); "
            "Bardet-Biedl syndrome BBS10 (AR — chaperonin complex; MORE SEVERE than BBS1; "
            "earliest retinal onset; NSAIDs ABSOLUTELY CI with renal cysts); "
            "CEP290/NPHP6 (AR pan-ciliopathy — LCA to Joubert to Meckel; "
            "IVS26 c.2991+1655A>G MOST COMMON LCA variant; sepofarsen ASO for IVS26 only; "
            "molar tooth sign PATHOGNOMONIC Joubert); "
            "NPHP1 (AR nephronophthisis — 2q13 deletion 80% alleles MISSED by sequencing; "
            "MLPA MANDATORY; normotensive ESRD age 13; mTOR inhibitors NO BENEFIT unlike PKD); "
            "DNAI1 PCD (AR primary ciliary dyskinesia — ODA IC1; situs inversus 50%; "
            "Kartagener triad; nasal NO <77 nL/min DIAGNOSTIC; azithromycin prophylaxis); "
            "DYNC2H1 JATD (AR Jeune asphyxiating thoracic dystrophy — IFT dynein; "
            "horizontal short ribs PATHOGNOMONIC; VEPTR rib surgery; neonatal respiratory failure); "
            "TMEM67 Joubert (AR — molar tooth + HEPATIC FIBROSIS DISTINCTIVE; "
            "NSAIDs CI + portal hypertension; UDCA + variceal surveillance mandatory); "
            "RPGRIP1L Joubert (AR — molar tooth + COLOBOMA 30-40% DISTINCTIVE; "
            "Leber amaurosis + coloboma = RPGRIP1L until excluded; NPHP8 isolated renal mild alleles)"
        ),
        "aggregate_clinical": {
            "esrd_pct": agg.get("esrd_pct", 0),
            "hypertension_pct": agg.get("htn_pct", 0),
            "transplant_rate_pct": agg.get("transplant_pct", 0),
            "drug_error_pct": agg.get("drug_error_pct", 0),
            "diagnosis_delayed_pct": agg.get("dx_delayed_pct", 0),
            "surveillance_adherent_pct": agg.get("surveillance_adherent_pct", 0),
            "severity_mild_pct": agg.get("severity", {}).get("Mild", 0),
            "severity_moderate_pct": agg.get("severity", {}).get("Moderate", 0),
            "severity_severe_pct": agg.get("severity", {}).get("Severe", 0),
        },
        "drug_alerts": [
            {
                "type": "danger",
                "title": "BBS1/BBS10: Leptin Therapy DOES NOT WORK — BBSome MC4R/LEPR Ciliary Pathway Disrupted",
                "body": (
                    "Obesity in BBS is caused by failure of leptin receptor (LEPR) and melanocortin-4 receptor (MC4R) "
                    "to traffic correctly to neuronal cilia via the disrupted BBSome. "
                    "Leptin levels are paradoxically high (leptin resistance). "
                    "Exogenous leptin therapy has no effect and should NOT be prescribed. "
                    "Management is dietary + behavioural + bariatric surgery. "
                    "GLP-1 agonists (semaglutide) are under investigation."
                ),
            },
            {
                "type": "danger",
                "title": "BBS10 + TMEM67: NSAIDs ABSOLUTELY CONTRAINDICATED with Renal Cystic Disease + Portal Hypertension",
                "body": (
                    "BBS10 patients with nephronophthisis-like renal cysts face double jeopardy from NSAIDs: "
                    "nephrotoxicity + cyst inflammation. TMEM67 patients with portal hypertension risk "
                    "NSAID-induced GI haemorrhage from oesophageal/gastric varices. "
                    "Both groups: NSAIDs contraindicated. Use paracetamol/tramadol for analgesia. "
                    "Prescribers must document NSAID contraindication in patient records."
                ),
            },
            {
                "type": "danger",
                "title": "NPHP1: mTOR Inhibitors Have NO Benefit in Nephronophthisis — 2q13 Deletion MISSED by Sequencing",
                "body": (
                    "mTOR inhibitors (everolimus, sirolimus) benefit ADPKD (PKD1/PKD2) but have NO demonstrated "
                    "benefit in nephronophthisis (NPHP1). The pathology is tubulointerstitial fibrosis, not mTOR-driven cystogenesis. "
                    "Do NOT extrapolate PKD benefit to NPHP. Separately: 80% of NPHP1 alleles are a large 2q13 "
                    "genomic deletion completely invisible to standard sequencing. MLPA for 2q13 is mandatory "
                    "whenever NPHP1 sequence is negative but clinical suspicion high."
                ),
            },
            {
                "type": "danger",
                "title": "CEP290-LCA10: Sepofarsen ONLY for Homozygous IVS26 c.2991+1655A>G — Do Not Use for Other Alleles",
                "body": (
                    "Sepofarsen is an antisense oligonucleotide that specifically corrects the IVS26 "
                    "c.2991+1655A>G deep intronic splice defect in retinal cells. It has no therapeutic "
                    "mechanism for truncating, missense, or other CEP290 variants. "
                    "Administering sepofarsen to patients with different CEP290 alleles confers no benefit. "
                    "Genotype confirmation is mandatory before treatment. "
                    "Additionally, standard exome sequencing misses this deep intronic variant — "
                    "IVS26 region must be specifically targeted in the sequencing panel."
                ),
            },
            {
                "type": "warning",
                "title": "DYNC2H1 JATD: Sedation Without Airway Support Hazardous — Thoracic Restriction Worsens Under Sedation",
                "body": (
                    "Children with Jeune asphyxiating thoracic dystrophy have a bell-shaped thorax with "
                    "severely restricted respiratory reserve. Sedation for procedures rapidly reduces "
                    "respiratory drive and can precipitate respiratory failure. "
                    "All sedation and anaesthesia must be planned with a paediatric anaesthesiologist "
                    "with ICU backup. Avoid sedation without secured airway."
                ),
            },
        ],
        "clinical_pearls": [
            "BBS1 p.Met390Arg: most common BBS variant worldwide — test first in suspected BBS",
            "BBS10: more severe than BBS1 — earlier ERG extinction; higher ESRD rate",
            "CEP290 IVS26: deep intronic — standard exome misses it; must specifically target",
            "NPHP1 2q13 deletion: 80% alleles — MLPA mandatory if sequencing negative",
            "DNAI1 nasal NO <77 nL/min: diagnostic screen for PCD; situs inversus 50%",
            "DYNC2H1 horizontal ribs on CXR: Jeune syndrome PATHOGNOMONIC — horizontal not oblique",
            "TMEM67 = Joubert + hepatic fibrosis: ANY Joubert with liver disease → TMEM67 first",
            "RPGRIP1L = Joubert + coloboma 35%: uveal/optic disc coloboma + molar tooth = RPGRIP1L",
        ],
    }


def get_breakdown() -> dict:
    result = {}
    for idx, gene in enumerate(CILIOPATHY_GENES):
        seed = SEED_BASE + idx
        cohort = _make_cohort(gene, seed=seed, n=40)
        stats = _cohort_stats(cohort)
        result[gene["gene"]] = {
            "gene": gene["gene"],
            "protein": gene["protein"],
            "alias": gene["alias"],
            "aa": gene["aa"],
            "kDa": gene["kDa"],
            "locus": gene["locus"],
            "omim_gene": gene["omim_gene"],
            "omim_disease": gene["omim_disease"],
            "gene_class": gene["gene_class"],
            "phenotype": gene["phenotype"],
            "hallmark": gene["hallmark"],
            "treatment_alert": gene["treatment_alert"],
            "key_ddx": gene["key_ddx"],
            "gfr_pattern": gene["gfr_pattern"],
            "proteinuria_pattern": gene["proteinuria_pattern"],
            "primary_complication": gene["primary_complication"],
            "disease_detail": gene["disease_detail"],
            "inheritance": gene["inheritance"],
            "variants": gene["variants"],
            "drug_ci": gene["drug_ci"],
            "stats": {
                "n": stats.get("n", 40),
                "esrd_pct": stats.get("esrd_pct", 0),
                "hypertension_pct": stats.get("htn_pct", 0),
                "transplant_pct": stats.get("transplant_pct", 0),
                "drug_error_pct": stats.get("drug_error_pct", 0),
                "dx_delayed_pct": stats.get("dx_delayed_pct", 0),
                "surveillance_adherent_pct": stats.get("surveillance_adherent_pct", 0),
                "severity_mild_pct": stats.get("severity", {}).get("Mild", 0),
                "severity_moderate_pct": stats.get("severity", {}).get("Moderate", 0),
                "severity_severe_pct": stats.get("severity", {}).get("Severe", 0),
            },
        }
    return result


def get_definitions() -> dict:
    return {
        "atlas": "Ciliopathy Atlas — 8-gene hereditary ciliary dysfunction reference",
        "terms": {
            "molar_tooth_sign": (
                "Pathognomonic MRI finding of Joubert syndrome: "
                "on axial brain MRI, superior cerebellar peduncles appear thickened and horizontal, "
                "combined with cerebellar vermis aplasia/hypoplasia — creates a 'molar tooth' appearance; "
                "present in all JBTS regardless of gene (CEP290, TMEM67, RPGRIP1L, AHI1, etc.); "
                "look for it on T1/T2 axial through the pons-midbrain junction"
            ),
            "bbsome": (
                "Bardet-Biedl syndrome octameric complex (BBS1/2/4/5/7/8/9/18); "
                "coats cargo vesicles within the ciliary membrane for IFT-B anterograde transport; "
                "BBS6/BBS10/BBS12 chaperonin complex required for BBSome assembly; "
                "LOF → GPCR receptor missorting → leptin resistance + retinal degeneration"
            ),
            "ift": (
                "Intraflagellar transport (IFT): bidirectional transport system in cilia; "
                "IFT-B (anterograde, kinesin-2): tip-directed transport of axonemal precursors; "
                "IFT-A (retrograde, dynein-2/DYNC2H1): base-directed removal of turnover products; "
                "failure of either arm disrupts ciliary assembly and hedgehog signalling"
            ),
            "transition_zone": (
                "Ciliary transition zone (TZ): proximal region of cilium at base (connecting cilium in photoreceptors); "
                "Y-links connect axoneme to ciliary membrane — gate controls entry/exit of proteins; "
                "TZ complex proteins: CEP290, NPHP1/4/8 (RPGRIP1L), TMEM67, IQCB1, MKS1/3/6; "
                "LOF of any TZ component → ciliary protein sorting failure → ciliopathy"
            ),
            "nasal_no_pcd": (
                "Nasal nitric oxide (nNO): primary screening test for Primary Ciliary Dyskinesia (PCD); "
                "normal nNO: 77-370 nL/min; PCD nNO: <77 nL/min (often <30 nL/min); "
                "mechanism: NO produced in paranasal sinuses; immotile cilia fail to wash out NO; "
                "sensitivity >95% for classic PCD (some PCD with normal nNO: DNAH11 variant); "
                "requires standardised technique; not reliable in CF or nasal polyps"
            ),
            "kartagener_triad": (
                "Kartagener syndrome: situs inversus + bronchiectasis + chronic sinusitis; "
                "present in 50% of PCD patients (situs determination is random at node — 50% inversus); "
                "caused by immotile nodal cilia at embryonic node failing to create leftward fluid flow; "
                "DNAI1 (ODA IC1) accounts for ~15% of Kartagener PCD cases"
            ),
            "sepofarsen": (
                "Sepofarsen (QR-110): antisense oligonucleotide (ASO) for CEP290-LCA10; "
                "targets IVS26 deep intronic variant c.2991+1655A>G specifically; "
                "blocks cryptic exon inclusion caused by the variant in retinal cells; "
                "ONLY effective for patients HOMOZYGOUS for c.2991+1655A>G; "
                "intravitreal injection; Phase 2/3 trials showing visual improvement"
            ),
            "veptr": (
                "VEPTR (Vertical Expandable Prosthetic Titanium Rib): surgical device for rib elongation; "
                "used in Jeune asphyxiating thoracic dystrophy (DYNC2H1) to expand narrow chest; "
                "titanium rods anchored to ribs + spine; expanded every 6 months under anaesthesia; "
                "improves lung volumes and respiratory reserve; initiated from age 6 months"
            ),
            "congenital_hepatic_fibrosis": (
                "CHF: periportal fibrosis + bile duct proliferation in liver — Ductal plate malformation; "
                "distinctive for TMEM67-Joubert (60-70%); occurs in ARPKD (PKHD1) also; "
                "leads to portal hypertension → oesophageal/gastric varices; "
                "UDCA reduces cholestasis; propranolol for variceal prophylaxis; "
                "liver biopsy: periportal fibrosis + ductal plate remnants + ductular proliferation"
            ),
            "coloboma": (
                "Coloboma: failure of optic fissure closure during eye development (week 5-7 gestation); "
                "types: uveal (iris/ciliary body/choroid), optic disc, retinal; "
                "causes inferior visual field defect (optic disc coloboma) or keyhole pupil (iris); "
                "RPGRIP1L-Joubert has highest coloboma rate among JBTS genes (30-40%); "
                "coloboma cannot be surgically repaired — management is low-vision rehabilitation"
            ),
            "nphp_medullary_cysts": (
                "Nephronophthisis (NPHP) renal pattern: "
                "small-to-normal kidney size (opposite of ADPKD large kidneys); "
                "corticomedullary cysts at tubulo-interstitial junction; "
                "tubulo-interstitial nephritis with fibrosis; "
                "inability to concentrate urine (polyuria) — FIRST clinical sign; "
                "normotensive until ESRD; ESRD median age 13 (NPHP1)"
            ),
            "oda_ifa": (
                "Outer dynein arm (ODA): main force-generator of ciliary axoneme; "
                "structure: 3 heavy chains (DNAH5/DNAH11/DNAH9) + intermediate chains (DNAI1/DNAI2) + light chains; "
                "assembles in cytoplasm (DNAAF pre-assembly factors) before transport to axoneme; "
                "DNAI1 LOF → ODA absent or truncated → ciliary immotility; "
                "EM cross-section: missing ODA structures on electron microscopy"
            ),
        },
    }
