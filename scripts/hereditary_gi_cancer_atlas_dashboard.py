#!/usr/bin/env python3
"""Hereditary-GI-Cancer-Atlas — Complete 8-Gene Hereditary GI Cancer Syndrome Atlas
APC     (Adenomatous polyposis coli; 2843 aa; 5q22.2; AD;
         Familial Adenomatous Polyposis — >100 adenomas; 100% CRC lifetime risk;
         CHRPE pathognomonic; colectomy mandatory by age 25–30;
         seed SEED_BASE+0) ·
MUTYH   (MutY DNA glycosylase; 546 aa; 1p34.1; AR;
         MUTYH-Associated Polyposis — 10–100 polyps; Y179C/G396D founders;
         AR biallelic — partner testing MANDATORY;
         seed SEED_BASE+1) ·
MLH1    (MutL homolog 1; 756 aa; 3p22.2; AD;
         Lynch Syndrome type 1 — most common (40%); MSI-H PATHOGNOMONIC;
         aspirin 600 mg/day reduces CRC/endometrial 50% (CAPP2);
         seed SEED_BASE+2) ·
MSH2    (MutS homolog 2; 934 aa; 2p21; AD;
         Lynch Syndrome type 2 — Muir-Torre sebaceous tumors; Turcot glioblastoma;
         EPCAM upstream deletion → MSH2 epigenetic silencing;
         seed SEED_BASE+3) ·
MSH6    (MutS homolog 6; 1360 aa; 2p16.3; AD;
         Lynch Syndrome type 3 — ENDOMETRIAL highest (71%); MSI-L confounds testing;
         biennial colonoscopy accepted; later CRC onset;
         seed SEED_BASE+4) ·
PMS2    (PMS1 homolog 2; 862 aa; 7p22.1; AD;
         Lynch Syndrome type 4 — LOWEST penetrance (CRC 15–20%);
         4 pseudogenes → MLPA MANDATORY; CMMRD biallelic childhood;
         seed SEED_BASE+5) ·
STK11   (Serine/threonine kinase 11 / LKB1; 433 aa; 19p13.3; AD;
         Peutz-Jeghers — perioral pigmentation PATHOGNOMONIC; hamartomatous;
         SCTAT ovarian tumor PATHOGNOMONIC; intussusception emergency;
         seed SEED_BASE+6) ·
SMAD4   (SMAD family member 4 / DPC4; 552 aa; 18q21.2; AD;
         Juvenile Polyposis/HHT overlap — telangiectasia + polyposis = SMAD4;
         protein-losing enteropathy; aortic dilatation surveillance;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1598–1605)
"""

import random

SEED_BASE = 1598

GI_GENES = [
    # ── APC — Familial Adenomatous Polyposis ──────────────────────────────
    {
        "gene": "APC",
        "protein": "APC — Familial Adenomatous Polyposis (FAP), 100% CRC Lifetime Risk, Colectomy Mandatory",
        "alias": (
            "APC; OMIM gene 611731; Familial Adenomatous Polyposis (FAP) OMIM 175100; "
            "5q22.2; 2843 aa; ~310 kDa; AD; prevalence ~1:10,000–1:30,000. "
            "APC is the gatekeeper tumour suppressor of the WNT signalling pathway. "
            "APC forms a destruction complex with Axin, GSK-3β and CK1 that phosphorylates "
            "β-catenin for ubiquitin-mediated proteasomal degradation. "
            "Loss of APC → β-catenin accumulates → nuclear translocation → TCF/LEF transcription → "
            "MYC, CCND1, survivin → uncontrolled proliferation of colonic epithelium. "
            "CLASSIC FAP: heterozygous germline APC mutation → >100 (typically hundreds to thousands) "
            "colorectal adenomatous polyps by age 20–30 → 100% lifetime CRC risk if untreated → "
            "colectomy MANDATORY, typically by age 25–30 before malignant transformation. "
            "GENOTYPE-PHENOTYPE: mutations in the MUTATION CLUSTER REGION (MCR, codons 1250–1464) → "
            "severe FAP (>5000 polyps, earlier CRC onset); "
            "mutations distal to codon 1580 or proximal to codon 200 → ATTENUATED FAP (AFAP): "
            "10–100 polyps, proximal predominance, later onset (40s–50s), screening-based management. "
            "EXTRACOLONIC MANIFESTATIONS: "
            "CONGENITAL HYPERTROPHY OF RETINAL PIGMENT EPITHELIUM (CHRPE) — bilateral, multifocal, "
            "flat hyperpigmented retinal lesions; present in ~70–80% of classic FAP; PATHOGNOMONIC; "
            "fundoscopy sufficient; useful pre-symptomatic marker in at-risk relatives. "
            "DUODENAL POLYPS — virtually all FAP patients by age 40; SPIGELMAN STAGE 0–IV: "
            "Stage IV → periampullary carcinoma risk 36%; "
            "surveillance gastroduodenoscopy with lateral-viewing endoscope (side-viewer) + "
            "SPIGELMAN staging mandatory from age 25–30; Stage IV → duodenectomy consideration. "
            "DESMOID TUMORS — intra-abdominal/mesenteric fibromatosis; "
            "mutations distal to codon 1310 (especially codon 1444–1580) → highest desmoid risk; "
            "Gardner syndrome (APC + desmoids + osteomas + epidermoid cysts); "
            "desmoids obstruct ureters/bowel; sulindac or tamoxifen trial; surgery rarely curative. "
            "PAPILLARY THYROID CARCINOMA — cribriform-morular variant; females predominant; age 20–40. "
            "HEPATOBLASTOMA — childhood (age <5); association strongest with 3' end mutations (5' MCR). "
            "MEDULLOBLASTOMA (TURCOT TYPE 2) — APC + medulloblastoma (desmoplastic); "
            "distinct from Turcot type 1 (MLH1/MSH2 + glioblastoma). "
            "I1307K (p.Ile1307Lys): Ashkenazi Jewish founder variant — creates hypermutable AAATAAAA run → "
            "NOT FAP; attenuated 2-fold increased CRC risk; colonoscopy surveillance, not prophylactic colectomy. "
            "SURGICAL OPTIONS: proctocolectomy + ileostomy (curative, but permanent stoma); "
            "colectomy with ileorectal anastomosis (IRA; rectal surveillance mandatory — rectal stump CRC risk); "
            "restorative proctocolectomy with ileal pouch-anal anastomosis (IPAA/J-pouch; preferred in severe FAP). "
            "CHEMOPREVENTION: COX-2 inhibitors (celecoxib, sulindac) reduce polyp burden ~28–45% "
            "but are NOT a substitute for colectomy — surveillance-alone strategies are only for AFAP with mild disease. "
            "PROPHYLACTIC COLECTOMY timing: as late as safely possible (complete secondary education) "
            "but before age 25–30 in classic FAP; rectal surveillance for IRA patients every 6–12 months."
        ),
        "aa": "2843 aa",
        "kDa": "~310 kDa",
        "locus": "5q22.2",
        "omim_gene": 611731,
        "omim_disease": 175100,
        "inheritance": "AD; >700 pathogenic germline variants; de novo 25%; classic FAP truncating; AFAP: proximal/distal mutations",
        "gene_class": (
            "APC encodes the Adenomatous Polyposis Coli protein. "
            "Domain structure: dimerisation domain (N-terminal) → armadillo repeats (β-catenin binding, EB1-HDLG) → "
            "15-aa repeats (3×, weak β-catenin binding) → 20-aa repeats (7×, destruction complex; codon 1020–1169) → "
            "MCR region (highest frequency somatic + germline mutations, codon 1250–1464) → "
            "SAMP repeats (axin binding) → C-terminal domain (microtubule/actin, EB1). "
            "Truncating variants in MCR lose β-catenin binding and destruction complex function → "
            "dominant-negative behaviour; second-hit LOH in colonic crypts → adenoma → carcinoma sequence."
        ),
        "n_patients": 40,
        "key_alerts": [
            "APC-100pct-CRC-RISK: Classic FAP carries a 100% lifetime colorectal cancer risk if untreated — prophylactic colectomy is MANDATORY, not optional; target surgery before age 25–30 in classic FAP to complete education first; IRA or IPAA based on rectal polyp burden; delay surgery only with endoscopic control of polyps",
            "APC-CHRPE-PATHOGNOMONIC: Congenital hypertrophy of retinal pigment epithelium (CHRPE) — bilateral, multifocal, flat pigmented retinal lesions in 70–80% of classic FAP — is the EARLIEST clinical marker; fundoscopy detects CHRPE in at-risk relatives before polyp development; absent CHRPE does not exclude FAP (proximal/distal mutations, AFAP)",
            "APC-DUODENAL-SPIGELMAN-IV: Virtually all FAP patients develop duodenal polyps; Spigelman Stage IV → 36% periampullary carcinoma risk → discuss pancreaticoduodenectomy (Whipple) with specialist; lateral-viewing endoscope mandatory for duodenal surveillance; begin gastroduodenoscopy at age 25–30",
            "APC-DESMOID-CODON-1310: Intra-abdominal desmoid tumors occur especially with APC mutations distal to codon 1310 (codons 1444–1580); desmoids can obstruct ureters, bowel, mesentery; AVOID abdominal surgery trigger (trauma → desmoid growth); sulindac/tamoxifen/sorafenib trial; surgery rarely curative",
            "APC-I1307K-ASHKENAZI: I1307K (p.Ile1307Lys) is an Ashkenazi Jewish founder APC variant — NOT classic FAP; creates a microsatellite-like repeat prone to somatic mutations → 2-fold increased CRC risk; manage with surveillance colonoscopy every 2–3 years; NEVER recommend prophylactic colectomy for I1307K alone",
            "APC-COX2-NOT-SUBSTITUTE-COLECTOMY: Celecoxib and sulindac reduce polyp burden 28–45% but are NOT a substitute for prophylactic colectomy in classic FAP — polyps recur on stopping; chemoprevention used as adjunct (post-IRA to protect rectal stump) or in AFAP where surveillance-based management is appropriate",
            "APC-TURCOT-TYPE2-MEDULLOBLASTOMA: APC-associated Turcot syndrome (type 2) presents with FAP + DESMOPLASTIC medulloblastoma in children — distinct from Turcot type 1 (Lynch/MMR + glioblastoma); APC mutation should trigger brain MRI if medulloblastoma is found in a child",
            "APC-RECTAL-SURVEILLANCE-IRA: After colectomy with ileorectal anastomosis (IRA), the retained rectal stump retains cancer risk — endoscopy every 6 months to 1 year; severe rectal polyp burden or Stage IV duodenal disease → convert IRA to IPAA (proctectomy + J-pouch)",
        ],
        "etiologies": {
            "Truncating in MCR (codons 1250–1464) — classic severe FAP": 18,
            "Truncating proximal codon 200 / distal codon 1580 — AFAP": 8,
            "I1307K Ashkenazi founder — attenuated risk, NOT FAP": 4,
            "De novo germline truncation — no family history": 5,
            "Large deletion/rearrangement (MLPA required)": 3,
            "Other truncating (splice, frameshift) — classic FAP": 2,
        },
        "stats": {
            "mean_dx_age_y": 26.8,
            "mean_polyp_count": 680,
            "pct_colectomy_done": 87,
            "pct_chrpe_present": 74,
            "pct_duodenal_polyps": 91,
            "pct_desmoid": 18,
            "mean_dx_delay_months": 8.4,
        },
        "dx_delay_distribution": {"<3 m": 12, "3–12 m": 16, "1–3 y": 9, ">3 y": 3},
    },
    # ── MUTYH — MUTYH-Associated Polyposis ──────────────────────────────
    {
        "gene": "MUTYH",
        "protein": "MUTYH — MUTYH-Associated Polyposis (MAP), AR, Y179C/G396D Founders, Partner Testing Mandatory",
        "alias": (
            "MUTYH; OMIM gene 604933; MUTYH-Associated Polyposis (MAP) OMIM 608456; "
            "1p34.1; 546 aa; ~60 kDa; AR (biallelic); prevalence 1:40,000–1:100,000 (biallelic). "
            "MUTYH encodes the MutY adenine-DNA glycosylase, a base-excision-repair enzyme "
            "that removes adenine mispaired with 8-oxoguanine (8-oxoG) — the major oxidative DNA lesion. "
            "8-oxoG (formed by reactive oxygen species) pairs preferentially with ADENINE → "
            "C:G → A:T transversions, the signature MUTYH mutation spectrum. "
            "Without MUTYH: 8-oxoG:A mispairs are not corrected → C:G → A:T transversions accumulate "
            "in critical tumour suppressor genes (APC, KRAS, TP53) → colorectal carcinogenesis. "
            "GENOTYPE: biallelic (compound heterozygous or homozygous) pathogenic MUTYH variants = MAP. "
            "FOUNDER VARIANTS: p.Tyr179Cys (Y179C, formerly Y165C, exon 7) and p.Gly396Asp (G396D, exon 13) "
            "account for ~80% of European MAP alleles; other variants more common in Asian populations. "
            "PHENOTYPE: 10–100 colorectal adenomatous polyps (fewer than FAP); "
            "predominantly right-sided (proximal colon); onset age 40s–50s (later than FAP); "
            "lifetime CRC risk ~80% by age 60 if untreated; adenomas can progress without extensive polyposis. "
            "MONOALLELIC MUTYH: single heterozygous pathogenic variant → NOT MAP; "
            "slight (1.5–2×) CRC risk increase; manage by enhanced surveillance, NOT prophylactic colectomy; "
            "PARTNER TESTING MANDATORY — if partner also carries MUTYH variant, offspring have 25% biallelic (MAP) risk. "
            "EXTRACOLONIC: duodenal adenomas (~1/3 patients), sebaceous skin tumors (Muir-Torre-like), "
            "ovarian/endometrial cancers (minor increase). "
            "SOMATIC MUTYH: biallelic SOMATIC MUTYH in CRC shows characteristic G:C → T:A transversion "
            "in KRAS codon 12 (p.Gly12Cys) and APC (p.Lys1302Ile) — DIAGNOSTIC fingerprint. "
            "MANAGEMENT: biannual (every 1–2 years) colonoscopy with polypectomy; "
            "colectomy with IRA if polyp burden becomes unmanageable (>20 adenomas) or high-grade dysplasia; "
            "gastroduodenoscopy from age 30–35; sulindac adjunct. "
            "GENETIC TESTING: MUTYH full-sequencing for both Y179C and G396D plus complete gene sequencing "
            "in non-European populations; MLPA for large deletions."
        ),
        "aa": "546 aa",
        "kDa": "~60 kDa",
        "locus": "1p34.1",
        "omim_gene": 604933,
        "omim_disease": 608456,
        "inheritance": "AR; biallelic required for MAP; Y179C and G396D founders (80% European alleles); monoallelic = slightly increased risk only",
        "gene_class": (
            "MUTYH encodes MutY adenine-DNA glycosylase. "
            "Domains: N-terminal catalytic domain (helix-hairpin-helix, HhH, base-excision) → "
            "C-terminal MutT-like domain (8-oxoG recognition, PCNA binding, RPA binding). "
            "Mechanism: MUTYH scans dsDNA, recognises 8-oxoG:A mispairs, flips adenine into active site, "
            "cleaves N-glycosidic bond → AP site → APE1 → BER patch synthesis. "
            "p.Tyr179Cys (Y179C): tyrosine in HhH motif essential for DNA backbone contact → "
            "Cys substitution → >95% loss of glycosylase activity. "
            "p.Gly396Asp (G396D): glycine in MutT-like domain → steric clash → impaired 8-oxoG binding. "
            "Biallelic loss → C:G → A:T transversion signature in tumour suppressor genes."
        ),
        "n_patients": 40,
        "key_alerts": [
            "MUTYH-BIALLELIC-REQUIRED-FOR-MAP: Only BIALLELIC (homozygous or compound heterozygous) MUTYH pathogenic variants cause MAP — a single heterozygous (monoallelic) variant does NOT cause MAP; offer enhanced colonoscopy surveillance (not prophylactic colectomy) to monoallelic carriers; avoid over-treating monoallelic MUTYH relatives",
            "MUTYH-PARTNER-TESTING-MANDATORY: Because MAP is autosomal recessive, PARTNER TESTING is mandatory — if the partner of a MAP patient or monoallelic carrier also carries a pathogenic MUTYH variant, their children have a 25% (1-in-4) chance of biallelic MAP; offer cascade testing to ALL first-degree relatives and request partner testing",
            "MUTYH-Y179C-G396D-FOUNDERS: p.Tyr179Cys (Y179C) and p.Gly396Asp (G396D) account for ~80% of MAP alleles in European ancestry — targeted testing detects most European MAP; full-gene sequencing + MLPA required for non-European ancestry populations where other variants predominate",
            "MUTYH-LATER-ONSET-FEWER-POLYPS: MAP presents with 10–100 adenomas (not the >100 of classic FAP) at a LATER AGE (40s–50s) and predominantly right-sided (proximal) — this atypical distribution can lead to missed diagnosis; colonoscopy biennially (every 1–2 years) with complete bowel preparation and cecal intubation mandatory",
            "MUTYH-SOMATIC-KRAS-G12C-SIGNATURE: Biallelic somatic MUTYH in sporadic CRC creates a distinctive C:G → T:A transversion signature including KRAS p.Gly12Cys (c.34G>T) and APC p.Lys1302Ile (c.3904A>T) — finding these somatic variants in a CRC without germline MUTYH mutation should prompt clinical reassessment",
            "MUTYH-COLECTOMY-THRESHOLD: Colectomy is indicated when adenoma burden becomes unmanageable endoscopically (>15–20 adenomas per session), high-grade dysplasia develops, or a sessile serrated lesion complicates surveillance — IRA generally preferred over IPAA in MAP (milder rectal disease); rectal surveillance post-IRA mandatory",
            "MUTYH-DUODENAL-SURVEILLANCE: Duodenal adenomas occur in ~25–35% of MAP patients — gastroduodenoscopy with side-viewer from age 30–35; Spigelman staging; management parallel to FAP duodenal protocol; periampullary risk lower than classic FAP but real",
            "MUTYH-MONOALLELIC-NO-COLECTOMY: Monoallelic MUTYH heterozygous carriers have a ~1.5–2× elevated CRC risk — equivalent to average-high-risk family history; manage with 5-yearly colonoscopy; do NOT recommend prophylactic colectomy or label as 'MAP' — they are carriers, not affected individuals",
        ],
        "etiologies": {
            "Homozygous Y179C (p.Tyr179Cys) — most common biallelic": 10,
            "Compound heterozygous Y179C/G396D — classic European MAP": 16,
            "Homozygous G396D (p.Gly396Asp)": 5,
            "Compound heterozygous Y179C/other variant": 5,
            "Biallelic non-founder variants (non-European ancestry)": 4,
        },
        "stats": {
            "mean_dx_age_y": 48.3,
            "mean_polyp_count": 42,
            "pct_colectomy_done": 52,
            "pct_duodenal_polyps": 31,
            "pct_biallelic_Y179C_G396D": 65,
            "pct_partner_tested": 61,
            "mean_dx_delay_months": 14.2,
        },
        "dx_delay_distribution": {"<6 m": 10, "6–18 m": 18, "1.5–4 y": 9, ">4 y": 3},
    },
    # ── MLH1 — Lynch Syndrome Type 1 ─────────────────────────────────────
    {
        "gene": "MLH1",
        "protein": "MLH1 — Lynch Syndrome Type 1, Most Common (40%), MSI-H PATHOGNOMONIC, Aspirin 600mg CAPP2",
        "alias": (
            "MLH1; OMIM gene 120436; Lynch Syndrome 1 / HNPCC type 1 OMIM 120435; "
            "3p22.2; 756 aa; ~85 kDa; AD; prevalence ~1:440–1:3,000 (all Lynch). "
            "MLH1 is the most commonly mutated Lynch syndrome gene (~40% of Lynch families). "
            "MLH1 forms a HETERODIMER with PMS2 (MutLα complex) to complete mismatch repair (MMR). "
            "MutSα (MSH2-MSH6) or MutSβ (MSH2-MSH3) recognises mismatched bases → "
            "recruits MutLα (MLH1-PMS2) → MLH1 endonuclease nicks the error-containing strand → "
            "EXO1/PCNA excision → resynthesis by Pol δ → ligation. "
            "Without MLH1: MMR fails → replication errors accumulate at microsatellites (short tandem repeats, STRs) → "
            "MICROSATELLITE INSTABILITY-HIGH (MSI-H) — the molecular fingerprint of Lynch syndrome CRC. "
            "MSI-H in CRC: 15% sporadic (BRAF V600E + MLH1 promoter methylation — NOT Lynch) vs "
            "Lynch (germline MLH1 mutation — NO BRAF V600E, NO promoter methylation). "
            "AMSTERDAM II CRITERIA (clinical: 3-2-1 rule): ≥3 relatives with Lynch-associated cancer "
            "(CRC, endometrial, small bowel, ureter/renal pelvis); ≥2 successive generations; "
            "≥1 diagnosed before age 50; FAP excluded; histologically verified. "
            "CANCER RISKS (MLH1): CRC 40–70% lifetime (male > female); "
            "endometrial 40–60% (most common extracolonic); ovarian 10–12%; "
            "gastric, small bowel, urothelial (upper urinary tract), biliary, pancreatic, brain. "
            "ANNUAL COLONOSCOPY from age 25 (reduces CRC-related mortality 65%); "
            "right-sided/proximal cancers predominate in Lynch; often poorly differentiated mucinous. "
            "ASPIRIN CHEMOPREVENTION: CAPP2 trial — aspirin 600 mg/day for ≥2 years reduces CRC incidence "
            "by ~50% and ALL Lynch-associated cancers; EFFECT DELAYED (emerges after stopping, 5–10 years follow-up); "
            "start aspirin early (age 25–30); low-dose aspirin (75–325 mg) also studied. "
            "IMMUNOTHERAPY: pembrolizumab approved (FDA 2017) for MSI-H/dMMR CRC (first tumour-agnostic approval); "
            "Lynch CRC highly immunogenic (TIL-rich, immune evasion via MMR loss); outstanding response rates. "
            "PROPHYLACTIC SURGERY: discuss risk-reducing hysterectomy + bilateral salpingo-oophorectomy (BSO) "
            "with female MLH1 carriers after family completion; reduces endometrial/ovarian risk >90%. "
            "EPIGENETIC SILENCING: MLH1 promoter hypermethylation (somatic, NOT germline) causes sporadic MSI-H CRC "
            "in elderly — CONSTITUTIONAL MLH1 METHYLATION (germline epimutation) is rare but causes Lynch phenotype "
            "without a coding mutation (explain normal sequencing in MSI-H Lynch pedigree)."
        ),
        "aa": "756 aa",
        "kDa": "~85 kDa",
        "locus": "3p22.2",
        "omim_gene": 120436,
        "omim_disease": 120435,
        "inheritance": "AD; ~40% of Lynch families; >400 pathogenic variants; constitutional MLH1 methylation (rare epimutation) also causes Lynch phenotype",
        "gene_class": (
            "MLH1 encodes MutL homolog 1. Domains: N-terminal ATPase (GHL superfamily) → "
            "central linker → C-terminal dimerisation + PMS2 interaction. "
            "Endonuclease motif: DQHA(X)2E(X)4E in PMS2 (activated by MLH1 interaction). "
            "MLH1 is the master coordinator — stabilises MutLα and recruits EXO1, PCNA, RFC. "
            "MLH1 loss is the most common cause of Lynch (40%) and of sporadic MSI-H CRC via promoter methylation. "
            "BRAF V600E somatic mutation is found in >80% of sporadic MLH1-methylated CRC but rarely in Lynch Lynch CRC — "
            "BRAF V600E testing on tumour stratifies germline testing prioritisation."
        ),
        "n_patients": 40,
        "key_alerts": [
            "MLH1-MSI-H-PATHOGNOMONIC: Microsatellite instability-high (MSI-H) by PCR/capillary electrophoresis and/or loss of MLH1/PMS2 nuclear staining by IHC on tumour tissue is the HALLMARK of MLH1 Lynch syndrome — test ALL CRC under age 70 (universal MMR/MSI testing); MSI-H does NOT equal Lynch without germline confirmation",
            "MLH1-BRAF-V600E-NOT-LYNCH: Sporadic MLH1-methylated MSI-H CRC = BRAF V600E mutation (>80%) + MLH1 promoter methylation on tumour → NOT Lynch syndrome; absence of BRAF V600E in MLH1-deficient tumour → PRIORITY germline testing; constitutional MLH1 methylation (epimutation) = Lynch without coding mutation — test if family history + MSI-H + normal sequencing",
            "MLH1-ANNUAL-COLONOSCOPY-AGE-25: Annual colonoscopy from age 25 (or 5 years before earliest family cancer) reduces Lynch CRC mortality by 65% — NEVER extend interval beyond 2 years in MLH1 carriers; right-sided proximal adenomas are the predominant precursor lesion; colonoscopy is more effective than sigmoidoscopy (30% proximal cancers missed)",
            "MLH1-ASPIRIN-600mg-CAPP2: CAPP2 trial: aspirin 600 mg/day for ≥2 years reduces CRC incidence ~50% and ALL Lynch cancers in Lynch syndrome — START aspirin at age 25–30; effect is DELAYED (emerges after stopping, peak protection at 5–10 year follow-up); contraindicate only if known GI bleeding history or anticoagulation; lower doses (150 mg) currently under study in CaPP3",
            "MLH1-PEMBROLIZUMAB-MSI-H: Pembrolizumab (anti-PD-1, FDA 2017 first tumour-agnostic approval) is FIRST-LINE for metastatic MSI-H/dMMR CRC including Lynch — KEYNOTE-177 trial: pembrolizumab superior to FOLFOX/FOLFIRI as first-line mCRC with ORR 45% vs 33%; Lynch CRC highly responsive; complete responses documented in MSI-H Lynch CRC",
            "MLH1-ENDOMETRIAL-RISK-40-60pct: Female MLH1 Lynch carriers have 40–60% lifetime endometrial cancer risk — discuss annual gynaecological surveillance (transvaginal ultrasound + endometrial biopsy) from age 30–35; discuss risk-reducing hysterectomy + BSO after family completion; endometrial cancer can be the SENTINEL/FIRST Lynch cancer",
            "MLH1-AMSTERDAM-II-3-2-1: Amsterdam II criteria for Lynch clinical diagnosis: ≥3 relatives with Lynch-associated cancer; ≥2 successive generations affected; ≥1 diagnosed before age 50; FAP excluded; histology confirmed — Amsterdam criteria are clinically useful but miss ~30% of Lynch families; Bethesda criteria then universal MMR/MSI testing is preferred",
            "MLH1-CONSTITUTIONAL-METHYLATION: Constitutional (germline) MLH1 promoter methylation is a rare non-coding cause of Lynch syndrome — normal gene sequencing + normal MLPA but MSI-H tumour + positive family history → test for MLH1 promoter methylation in blood DNA (bisulphite sequencing); NOT detected by standard panel sequencing; methylated allele can be somatically reactivated",
        ],
        "etiologies": {
            "Pathogenic truncating (frameshift/nonsense) — germline": 14,
            "Pathogenic missense — MMR-disruptive": 10,
            "Large intragenic deletion (MLPA required)": 6,
            "Splice-site variant — aberrant mRNA": 6,
            "Constitutional MLH1 promoter methylation (epimutation)": 4,
        },
        "stats": {
            "mean_dx_age_y": 44.2,
            "pct_msi_h_confirmed": 96,
            "pct_aspirin_commenced": 58,
            "pct_annual_colonoscopy": 81,
            "pct_female_gynaecol_surveillance": 70,
            "pct_pembrolizumab_mCRC": 22,
            "mean_dx_delay_months": 10.8,
        },
        "dx_delay_distribution": {"<6 m": 18, "6–18 m": 12, "1.5–3 y": 7, ">3 y": 3},
    },
    # ── MSH2 — Lynch Syndrome Type 2 ─────────────────────────────────────
    {
        "gene": "MSH2",
        "protein": "MSH2 — Lynch Syndrome Type 2, Muir-Torre Sebaceous Tumours, EPCAM Upstream Silencing",
        "alias": (
            "MSH2; OMIM gene 609309; Lynch Syndrome 2 / HNPCC type 2 OMIM 609310; "
            "2p21; 934 aa; ~105 kDa; AD; 2nd most common Lynch gene (~35% of families). "
            "MSH2 heterodimerises with MSH6 (→ MutSα, mononucleotide/dinucleotide mismatch repair) or "
            "MSH3 (→ MutSβ, insertion/deletion loop repair) to initiate MMR recognition. "
            "MSH2-MSH6 (MutSα) is the primary recognition complex for base-base mismatches and small insertion-deletions; "
            "MSH2 loss cripples BOTH MutSα and MutSβ → complete MMR loss → MSI-H. "
            "CANCER RISKS (MSH2): CRC 40–65% lifetime; endometrial 25–40%; ovarian 10–15%; "
            "UROTHELIAL CARCINOMA (upper urinary tract — renal pelvis, ureter) — risk ~10% lifetime, "
            "HIGHEST among Lynch genes → annual urine cytology and surveillance from age 25; "
            "gastric, small bowel, hepatobiliary, brain, skin. "
            "MUIR-TORRE SYNDROME: MSH2 (most common) / MSH6 Lynch + sebaceous skin tumours: "
            "sebaceous adenoma, sebaceous carcinoma, keratoacanthoma with sebaceous differentiation → "
            "PATHOGNOMONIC for MMR-deficient Lynch when multiple/early/recurrent; "
            "MMR IHC on sebaceous tumour detects Lynch (MSH2 loss); "
            "any patient with sebaceous carcinoma should be offered Lynch genetic testing. "
            "TURCOT SYNDROME (Lynch-type): MSH2 (or MLH1) + GLIOBLASTOMA (GBM) → "
            "brain MRI if neurological symptoms; distinct from APC-Turcot (medulloblastoma). "
            "EPCAM (TACSTD1) upstream MSH2 silencing: 3' deletions in EPCAM gene (upstream of MSH2 on 2p21) → "
            "transcriptional read-through across MSH2 promoter → EPIGENETIC silencing of MSH2 → "
            "Lynch phenotype WITHOUT an MSH2 coding mutation; "
            "standard MSH2 sequencing NORMAL; MLPA reveals EPCAM 3' deletion; "
            "accounts for ~20–25% of apparent MSH2-negative Lynch families; "
            "EPCAM-related Lynch has LOWER endometrial risk vs MSH2 coding variant (tissue-specific silencing). "
            "SURVEILLANCE: annual colonoscopy from age 25; "
            "annual upper-tract urine cytology + urinary tract imaging (CT urography) from age 25; "
            "gastroscopy every 2–3 years from age 30–35; gynaecological surveillance for females; "
            "annual skin examination for Muir-Torre; ophthalmology for ocular sebaceous lesions."
        ),
        "aa": "934 aa",
        "kDa": "~105 kDa",
        "locus": "2p21",
        "omim_gene": 609309,
        "omim_disease": 609310,
        "inheritance": "AD; ~35% of Lynch families; EPCAM 3' deletion → MSH2 epigenetic silencing (accounts for 20–25% of apparent MSH2-Lynch; detected by MLPA only)",
        "gene_class": (
            "MSH2 encodes MutS homolog 2. Domains: mismatch-binding domain (MBD) → connector → lever → "
            "clamp → ATPase (two Walker A/B motifs). "
            "MSH2 is the obligate partner for both MSH6 (MutSα) and MSH3 (MutSβ); "
            "MSH2 loss abrogates ALL base-base mismatch recognition and large IDL repair simultaneously. "
            "EPCAM: the EPCAM gene encodes epithelial cell adhesion molecule; its 3' exon deletions "
            "remove the polyadenylation signal → readthrough transcription into MSH2 promoter → "
            "CpG methylation → tissue-specific silencing (colon, endometrium) → Lynch cancer in those tissues; "
            "MSH2 protein expression normal in lymphocytes (hence serum MLH1 unaffected). "
            "MLPA must include EPCAM probes to capture this silencing mechanism."
        ),
        "n_patients": 40,
        "key_alerts": [
            "MSH2-UROTHELIAL-ANNUAL-CT: MSH2 Lynch syndrome carries the HIGHEST upper urinary tract cancer risk among Lynch genes (~10% lifetime) — annual upper-tract surveillance: urine cytology + CT urography from age 25 (some centres use cystoscopy); gross haematuria or flank pain = emergency upper-tract evaluation; renal pelvis and ureter CRCs in Lynch are often high-grade",
            "MSH2-MUIR-TORRE-SEBACEOUS: Sebaceous adenoma, sebaceous carcinoma, or keratoacanthoma with sebaceous differentiation in a patient under age 60 = Muir-Torre syndrome → MANDATORY Lynch genetic testing (MSH2 most common); MMR IHC on the sebaceous tumour (MSH2/MSH6 loss) confirms Lynch; ANY sebaceous carcinoma at any age warrants referral",
            "MSH2-EPCAM-MLPA-MANDATORY: Approximately 20–25% of apparent MSH2-Lynch families have an EPCAM 3' deletion (not an MSH2 coding variant) — EPCAM deletion silences MSH2 epigenetically via read-through transcription; standard MSH2 sequencing is NORMAL; only MLPA (including EPCAM probes) detects it; ALWAYS include MLPA in Lynch testing panels",
            "MSH2-EPCAM-LOWER-ENDOMETRIAL: EPCAM-related MSH2 silencing causes LOWER endometrial cancer risk than MSH2 coding variants because MSH2 silencing in EPCAM deletions is tissue-specific (colon, endometrium tissue-variable); do NOT counsel EPCAM carriers with same endometrial risk as MSH2 coding pathogenic variant carriers — risk stratify by deletion extent",
            "MSH2-TURCOT-GLIOBLASTOMA: MSH2 (or MLH1) Lynch syndrome + glioblastoma (GBM) = Lynch-associated Turcot syndrome — GBM in a Lynch carrier demands brain MRI; Lynch-Turcot GBM occurs in adults (vs APC-Turcot medulloblastoma in children); immunotherapy response in MSI-H GBM is under investigation",
            "MSH2-MUTS-ALPHA-DUAL-LOSS: MSH2 loss eliminates BOTH MutSα (MSH2-MSH6, base-base mismatches) and MutSβ (MSH2-MSH3, large insertion-deletion loops) — complete MMR loss; MSH2-Lynch tumours show extensive MSI-H with instability across both mononucleotide and dinucleotide microsatellite markers (Bethesda panel: BAT25, BAT26, D5S346, D2S123, D17S250)",
            "MSH2-COLONOSCOPY-ANNUAL-25: Annual colonoscopy from age 25 — Lynch MSH2 CRC risk 40–65%; proximal CRC predominance; interval CRC (cancer between surveillance colonoscopies) occurs in Lynch — ANNUAL (not biennial) surveillance is the standard; polypectomy of all adenomas; colonoscopy quality (cecal intubation rate, ADR) determines efficacy",
            "MSH2-GYNAECOLOGICAL-25-35: Female MSH2 carriers: endometrial cancer risk 25–40% + ovarian 10–15%; annual endometrial biopsy + TVUS from age 30–35; discuss risk-reducing surgery (total hysterectomy + BSO) after family completion; endometrial cancer may be the presenting Lynch malignancy; CA-125 surveillance not sufficient alone for ovarian risk",
        ],
        "etiologies": {
            "Pathogenic truncating (frameshift/nonsense) — germline MSH2": 16,
            "Pathogenic missense — MMR-disrupting": 8,
            "EPCAM 3' deletion → MSH2 epigenetic silencing": 8,
            "Large MSH2 intragenic deletion (MLPA)": 5,
            "Splice-site variant": 3,
        },
        "stats": {
            "mean_dx_age_y": 46.1,
            "pct_msi_h": 94,
            "pct_epcam_deletion": 20,
            "pct_muir_torre": 12,
            "pct_urothelial_surveillance": 71,
            "pct_annual_colonoscopy": 78,
            "mean_dx_delay_months": 12.3,
        },
        "dx_delay_distribution": {"<6 m": 16, "6–18 m": 14, "1.5–3 y": 7, ">3 y": 3},
    },
    # ── MSH6 — Lynch Syndrome Type 3 ─────────────────────────────────────
    {
        "gene": "MSH6",
        "protein": "MSH6 — Lynch Syndrome Type 3, Endometrial Highest (71%), MSI-L Confounds Testing",
        "alias": (
            "MSH6; OMIM gene 600678; Lynch Syndrome 3 / HNPCC type 5 OMIM 614350; "
            "2p16.3; 1360 aa; ~160 kDa; AD; accounts for ~15–20% of Lynch families. "
            "MSH6 pairs exclusively with MSH2 in MutSα to repair base-base mismatches and "
            "single-nucleotide insertion/deletion loops. "
            "MSH2-MSH6 (MutSα) recognises 1-bp mismatches and 1–2 nt insertion-deletion loops; "
            "MSH6 loss: MutSα absent but MutSβ (MSH2-MSH3) partially preserved → "
            "PREFERENTIAL LOSS of mononucleotide STR stability → "
            "MSI-L (microsatellite instability-LOW) or MSS on standard Bethesda dinucleotide panel; "
            "pentaplex mononucleotide panel (BAT25, BAT26, BAT40, NR21, NR24) better detects MSH6 Lynch. "
            "CANCER RISKS (MSH6): "
            "ENDOMETRIAL CANCER — highest among Lynch genes: 71% lifetime in females (MLH1 40–60%); "
            "CRC RISK lower (25–44%) and LATER ONSET (age 50s–60s vs MLH1/MSH2 40s); "
            "ovarian 8–12%. "
            "CLINICAL IMPLICATION: endometrial cancer can be the INDEX/SENTINEL cancer in MSH6 Lynch; "
            "ALL women presenting with endometrial cancer under 60 should have universal MMR/MSI testing "
            "on tumour (MSH6 loss by IHC is diagnostic); "
            "MSH6 Lynch may be missed by Bethesda/Amsterdam criteria (later onset, fewer CRC families). "
            "COLONOSCOPY: biennial (every 2 years) is accepted for MSH6 (vs annual for MLH1/MSH2) "
            "given lower penetrance and later CRC onset; "
            "some guidelines still recommend annual given interval CRC risk. "
            "GYNAECOLOGICAL SURVEILLANCE: annual endometrial biopsy + TVUS from age 30–35; "
            "risk-reducing hysterectomy + BSO discussion from age 40–45 or after family completion. "
            "ASPIRIN: CAPP2 data applicable to all Lynch; MSH6-specific data limited but aspirin recommended. "
            "TESTING PITFALL: standard 5-marker Bethesda MSI panel uses dinucleotide repeats poorly sensitive for MSH6 Lynch → "
            "request MONONUCLEOTIDE PENTAPLEX or report MSI result as equivocal for MSH6 if MSS on dinucleotide but IHC shows MSH6 loss."
        ),
        "aa": "1360 aa",
        "kDa": "~160 kDa",
        "locus": "2p16.3",
        "omim_gene": 600678,
        "omim_disease": 614350,
        "inheritance": "AD; ~15–20% of Lynch families; MSI-L or MSS on standard panel — mononucleotide pentaplex required for full MSI characterisation",
        "gene_class": (
            "MSH6 encodes MutS homolog 6. Domains: MBD (mismatch binding) → connector → lever → clamp → ATPase. "
            "MSH6 contains the PWWP domain (N-terminal, chromatin binding) and the mismatch binding phenylalanine residue "
            "(Phe-432 in human MSH6) critical for base-flip mismatch recognition. "
            "MSH6 alone does NOT repair large IDL loops (requires MSH3/MutSβ intact) → "
            "MSH6 loss shows PARTIAL MMR function (dinucleotide STRs partially stable) → MSI-L phenotype. "
            "This is why standard 5-marker Bethesda MSI panel (predominantly dinucleotide) misses MSH6 Lynch. "
            "Large MSH6 deletions detectable by MLPA only."
        ),
        "n_patients": 40,
        "key_alerts": [
            "MSH6-ENDOMETRIAL-71pct-HIGHEST: Female MSH6 Lynch carriers have a 71% lifetime endometrial cancer risk — the HIGHEST of all Lynch genes; endometrial cancer is often the FIRST Lynch malignancy in MSH6 families; all women with endometrial cancer under 60 should have universal MMR/MSI testing; MSH6 loss on IHC of endometrial tumour = refer for genetic testing immediately",
            "MSH6-MSI-L-MONONUCLEOTIDE-PANEL: MSH6 Lynch commonly shows MSI-LOW or MSS on the standard 5-marker Bethesda DINUCLEOTIDE panel (D5S346, D2S123, D17S250, BAT25, BAT26) because residual MutSβ partially stabilises dinucleotide repeats — request MONONUCLEOTIDE PENTAPLEX (BAT25+BAT26+BAT40+NR21+NR24) for MSH6 Lynch or rely on IHC MSH6/MSH2 loss pattern",
            "MSH6-BIENNIAL-COLONOSCOPY-ACCEPTED: Biennial (every 2 years) colonoscopy is guideline-accepted for MSH6 Lynch given lower CRC penetrance (25–44%) and later onset (50s–60s) compared to MLH1/MSH2 — however annual colonoscopy remains preferred in high-risk individuals or strong family history; interval CRC still occurs so NO>2-year intervals",
            "MSH6-LATER-CRC-ONSET-50s: MSH6 Lynch colorectal cancer presents a DECADE LATER than MLH1/MSH2 (mean age ~55–60 vs ~45) — do NOT dismiss Lynch if CRC index case is older (>50); MSH6 may be missed by Amsterdam II age-of-onset criterion (>50 years); universal tumour MSI/MMR testing bypasses this bias",
            "MSH6-GYNAECOLOGICAL-FIRST: Annual endometrial surveillance (TVUS + outpatient endometrial sampling) from age 30–35; discuss risk-reducing hysterectomy + BSO after family completion (reduces endometrial cancer risk >95% and ovarian ~80%); MSH6 endometrial cancers are typically early-stage at detection with surveillance — survivorship is excellent with early detection",
            "MSH6-NOT-MISSED-BY-IHC: MSH6 protein loss by immunohistochemistry (IHC) on tumour tissue is the most sensitive test for MSH6 Lynch — MSH2 staining is RETAINED (MSH2 protein intact; MSH6 cannot bind MSH2 without itself but MSH2 dimerises with MSH3); IHC pattern: MSH6 LOST + MSH2 RETAINED + MLH1/PMS2 RETAINED → GERMLINE MSH6 TESTING",
            "MSH6-PEMBROLIZUMAB-MSI-H-SUBSET: MSH6 Lynch tumours are eligible for pembrolizumab (FDA approved for dMMR/MSI-H solid tumours) when MSI-H by mononucleotide panel or dMMR by IHC — if reported MSS by dinucleotide panel, RETEST with mononucleotide panel before denying immunotherapy",
            "MSH6-ASPIRIN-RECOMMENDED: Although MSH6-specific chemoprevention trial data are limited, CAPP2 aspirin benefit applies across Lynch genes — aspirin 600 mg/day (or lower doses) recommended for MSH6 Lynch carriers from age 25–30 given the substantial endometrial and colorectal cancer risks",
        ],
        "etiologies": {
            "Pathogenic truncating (frameshift/nonsense)": 20,
            "Pathogenic missense — MBD or ATPase domain": 12,
            "Large deletion (MLPA required)": 5,
            "Splice-site variant": 3,
        },
        "stats": {
            "mean_dx_age_y": 54.6,
            "pct_endometrial_index_cancer": 34,
            "pct_msi_h_mononucleotide_panel": 78,
            "pct_msi_h_standard_bethesda": 48,
            "pct_biennial_colonoscopy": 65,
            "pct_gynaecol_surveillance": 73,
            "mean_dx_delay_months": 16.4,
        },
        "dx_delay_distribution": {"<6 m": 12, "6–18 m": 15, "1.5–4 y": 10, ">4 y": 3},
    },
    # ── PMS2 — Lynch Syndrome Type 4 ─────────────────────────────────────
    {
        "gene": "PMS2",
        "protein": "PMS2 — Lynch Syndrome Type 4, Lowest Penetrance (CRC 15–20%), 4 Pseudogenes MLPA Mandatory, CMMRD Biallelic",
        "alias": (
            "PMS2; OMIM gene 600259; Lynch Syndrome 4 / HNPCC type 4 OMIM 614337; "
            "7p22.1; 862 aa; ~96 kDa; AD (monoallelic Lynch); AR (biallelic = CMMRD); ~10–15% of Lynch. "
            "PMS2 heterodimerises with MLH1 to form MutLα, the MMR endonuclease complex. "
            "PMS2 contains the endonuclease catalytic motif (DQHA(X)2E(X)4E) that nicks the daughter strand "
            "under direction from MLH1 — PMS2 is the catalytic subunit; MLH1 is the scaffolding subunit. "
            "PENETRANCE: LOWEST of all Lynch MMR genes; "
            "CRC lifetime risk 15–25% (vs 40–70% MLH1/MSH2); endometrial 15–25%; other cancers less common. "
            "MANAGEMENT IMPLICATION: some guidelines allow 5-yearly colonoscopy for monoallelic PMS2 Lynch carriers "
            "given lower penetrance — but most recommend 2-yearly colonoscopy from age 25 "
            "(interval CRC risk is real; understating risk leads to non-adherence). "
            "PSEUDOGENE PROBLEM: PMS2 has 4 highly homologous pseudogenes on chromosomes 7 (PMS2CL) "
            "and other locations → Sanger sequencing and NGS MISIDENTIFY pseudogene sequence as PMS2 mutations → "
            "FALSE POSITIVES (variant calls from pseudogene reads counted as PMS2) and "
            "FALSE NEGATIVES (PMS2 deletion masked by pseudogene reads). "
            "MLPA IS MANDATORY for PMS2 testing; long-range PCR or specialised LR-PCR panels to distinguish "
            "PMS2 from PMS2CL are required for deep sequencing; "
            "report pathogenic PMS2 only when confirmed by pseudogene-aware methodology. "
            "CONSTITUTIONAL MISMATCH REPAIR DEFICIENCY (CMMRD): biallelic PMS2 (compound het or homozygous) = "
            "autosomal recessive childhood cancer syndrome; "
            "BRAIN TUMOURS (glioblastoma, medulloblastoma, PNET) in the first decade; "
            "haematological malignancies (leukaemia, lymphoma); CRC before age 20; "
            "CAFÉ-AU-LAIT macules ≥6 (NF1-like — NF1 gene also has MMR-related instability); "
            "CMMRD should be distinguished from NF1 by molecular testing (skin biopsy MMR IHC). "
            "Biallelic PMS2 carriers: heterozygous parents have Lynch (monoallelic PMS2) risk — cascade testing. "
            "IMMUNOTHERAPY: MSI-H tumours in CMMRD → pembrolizumab can be used even in children with CMMRD cancers."
        ),
        "aa": "862 aa",
        "kDa": "~96 kDa",
        "locus": "7p22.1",
        "omim_gene": 600259,
        "omim_disease": 614337,
        "inheritance": "AD (monoallelic Lynch, lowest penetrance); AR biallelic = CMMRD (childhood cancer syndrome with cafe-au-lait, brain tumours, haematological malignancies, early CRC)",
        "gene_class": (
            "PMS2 encodes PMS1 Homolog 2, mismatch repair system component. "
            "Domains: N-terminal ATPase (GHL superfamily) → C-terminal dimerisation + MLH1 interaction + "
            "endonuclease (DQHA motif). "
            "PMS2 is the catalytic endonuclease but REQUIRES MLH1 interaction for activation. "
            "PMS2 loss: MLH1 protein still expressed (MLH1 homodimerises with MLH3 partially) → "
            "IHC pattern: PMS2 LOST + MLH1 RETAINED (PMS2-specific loss pattern, unlike MLH1 loss which loses both MLH1 and PMS2). "
            "4 pseudogenes: PMS2CL (chromosome 7p22), plus partial pseudogenes on chromosomes 3, 4, 7 → "
            "pseudogene-aware long-read sequencing or MLPA + LR-PCR is the gold standard. "
            "CMMRD: biallelic PMS2 → complete MMR absence from birth → extraordinary tumour burden in childhood."
        ),
        "n_patients": 40,
        "key_alerts": [
            "PMS2-MLPA-MANDATORY-PSEUDOGENES: PMS2 has 4 highly homologous PSEUDOGENES (PMS2CL and others) that cause false-positive and false-negative calls by standard sequencing — MLPA IS MANDATORY for all PMS2 testing; long-range PCR or pseudogene-aware NGS required to confirm pathogenic PMS2 variants; do NOT report a PMS2 variant as pathogenic without pseudogene confirmation",
            "PMS2-LOWEST-PENETRANCE-LYNCH: Monoallelic PMS2 Lynch has the LOWEST cancer penetrance of all Lynch genes — lifetime CRC risk 15–25%, endometrial 15–25%; do NOT catastrophise when disclosing a PMS2 Lynch diagnosis; 2-yearly colonoscopy from age 25 is recommended; annual surveillance may lead to anxiety without proportionate benefit reduction",
            "PMS2-IHC-PATTERN-PMS2-LOST-MLH1-RETAINED: PMS2 loss on IHC shows PMS2-ABSENT + MLH1-RETAINED (MLH1 partially dimerises with MLH3) — this SPECIFIC pattern of isolated PMS2 loss distinguishes PMS2 Lynch from MLH1 Lynch (which loses BOTH MLH1 and PMS2 by IHC); any PMS2-only IHC loss → germline PMS2 testing (including MLPA) mandatory",
            "PMS2-CMMRD-BIALLELIC-BRAIN: Constitutional mismatch repair deficiency (CMMRD) from BIALLELIC PMS2 mutation = childhood cancer: brain tumours (GBM, PNET, medulloblastoma) in the first decade, haematological malignancies, early CRC; presents with ≥6 café-au-lait macules mimicking NF1; any child with brain tumour + café-au-lait + Lynch family history → CMMRD workup (blood MMR IHC + skin biopsy MMR IHC)",
            "PMS2-CMMRD-PARENTS-LYNCH: Biallelic PMS2 (CMMRD) → both parents are obligate monoallelic PMS2 carriers (Lynch syndrome) — always test both parents of a CMMRD child for Lynch; ALL first-degree relatives of a PMS2 carrier should be offered cascade testing (whether the index is monoallelic Lynch or CMMRD)",
            "PMS2-2-YEAR-COLONOSCOPY-FROM-25: Despite lower penetrance, 2-yearly colonoscopy from age 25 is recommended for PMS2 Lynch (some centres start at 30 given later CRC onset and risk profile); interval cancers occur; 5-yearly is insufficient if adenomas are found; aspirin (600 mg/day) recommended as per CAPP2 data for all Lynch",
            "PMS2-PEMBROLIZUMAB-CMMRD-CHILDREN: MSI-H/dMMR tumours in CMMRD patients (children) are eligible for pembrolizumab — CMMRD brain tumours with MSI-H show responses to anti-PD-1 therapy; pembrolizumab is FDA-approved for MSI-H solid tumours regardless of age or site (tumour-agnostic approval 2017)",
            "PMS2-ENDOMETRIAL-15-25pct: Female monoallelic PMS2 Lynch carriers: endometrial cancer risk 15–25% (lower than MLH1/MSH6) — annual TVUS + endometrial biopsy from age 40–45 rather than 30–35 as for MLH1/MSH2; risk-reducing hysterectomy + BSO discussion after family completion; ovarian cancer risk appears low in PMS2 Lynch vs other Lynch genes",
        ],
        "etiologies": {
            "Truncating (confirmed pseudogene-aware, pathogenic)": 18,
            "Splice-site (LR-PCR confirmed, pseudogene-excluded)": 8,
            "Large deletion (MLPA — deletion distinct from PMS2CL)": 8,
            "Missense (functional validation + pseudogene-excluded)": 4,
            "Biallelic PMS2 (CMMRD) — compound heterozygous": 2,
        },
        "stats": {
            "mean_dx_age_y": 52.8,
            "pct_msi_h": 82,
            "pct_mlpa_required_for_diagnosis": 61,
            "pct_2yearly_colonoscopy": 72,
            "pct_cmmrd_biallelic": 5,
            "pct_pembrolizumab_eligible": 14,
            "mean_dx_delay_months": 20.1,
        },
        "dx_delay_distribution": {"<6 m": 8, "6–18 m": 14, "1.5–5 y": 14, ">5 y": 4},
    },
    # ── STK11 — Peutz-Jeghers Syndrome ───────────────────────────────────
    {
        "gene": "STK11",
        "protein": "STK11 — Peutz-Jeghers Syndrome AD, Perioral Pigmentation PATHOGNOMONIC, SCTAT Ovarian Tumour, Intussusception Emergency",
        "alias": (
            "STK11 (LKB1); OMIM gene 602216; Peutz-Jeghers Syndrome OMIM 175200; "
            "19p13.3; 433 aa; ~48 kDa; AD; prevalence ~1:50,000–1:200,000. "
            "STK11 (serine/threonine kinase 11, also known as LKB1) is a master kinase regulating "
            "AMPK (AMP-activated protein kinase) and downstream mTOR signalling, "
            "cell polarity (Par3-Par6-aPKC complex), cell cycle (G1 arrest via p21), "
            "and apoptosis (mitochondrial pathway). "
            "STK11 activates AMPK under low-energy/nutrient conditions → mTOR suppression → "
            "preventing aberrant cell growth; STK11 loss → constitutive mTOR activation → "
            "hamartomatous and ultimately malignant transformation. "
            "CLINICAL TRIAD: "
            "(1) MUCOCUTANEOUS PERIORAL MELANOCYTIC MACULES — PATHOGNOMONIC: "
            "flat, brown-to-black macules on the lips (especially lower), buccal mucosa, perioral skin, "
            "fingers/toes, soles; present from infancy to early childhood; "
            "FADE after puberty (adult age does NOT exclude PJS if history of childhood pigmentation); "
            "BUCCAL MUCOSA pigmentation persists into adulthood (unlike perioral skin which fades). "
            "(2) HAMARTOMATOUS GASTROINTESTINAL POLYPS — in small bowel (ileum > jejunum > duodenum), "
            "stomach, colon; Peutz-Jeghers histology: characteristic ARBORISING smooth muscle (arborisation of muscularis mucosae) "
            "distinguishing hamartomas from juvenile polyps and adenomas; polyps NOT adenomatous but have focal adenomatous change. "
            "(3) CANCER PREDISPOSITION — CRC risk 38–40%; "
            "BREAST CANCER 45–50% (highest single-site risk in PJS; screening from age 25); "
            "PANCREATIC CANCER 11–36% (2nd highest relative risk of any hereditary syndrome for pancreatic cancer); "
            "GASTRIC, SMALL BOWEL, LUNG, CERVICAL (MINIMAL DEVIATION ADENOCARCINOMA — STK11), "
            "UTERINE (adenoma malignum); "
            "SCTAT (SEX CORD TUMOUR WITH ANNULAR TUBULES) — PATHOGNOMONIC FOR PJS IN FEMALES; "
            "25–30% of women with SCTAT have PJS; benign in most PJS cases; "
            "Sertoli cell tumours in males. "
            "INTUSSUSCEPTION: hamartomatous polyps → lead point → acute SMALL BOWEL INTUSSUSCEPTION → "
            "EMERGENCY: severe abdominal pain, distension, vomiting; bowel ischaemia; "
            "CT identifies lead point; surgical or endoscopic reduction; "
            "prophylactic polypectomy at index endoscopy reduces intussusception risk. "
            "SURVEILLANCE PROGRAM: capsule endoscopy or MR enterography every 3 years from age 8 "
            "for small bowel (size >10 mm → prophylactic endoscopic polypectomy by balloon enteroscopy); "
            "colonoscopy + gastroscopy every 3 years from age 8; "
            "breast MRI annually from age 25; pancreatic MRI/MRCP annually from age 30–35; "
            "annual gynaecological examination + TVUS for females from age 18–25."
        ),
        "aa": "433 aa",
        "kDa": "~48 kDa",
        "locus": "19p13.3",
        "omim_gene": 602216,
        "omim_disease": 175200,
        "inheritance": "AD; ~50% de novo; >200 pathogenic variants; large deletions common (MLPA required); haploinsufficiency mechanism",
        "gene_class": (
            "STK11 encodes serine/threonine kinase 11 (LKB1). Domains: N-terminal nuclear localisation signal → "
            "kinase catalytic domain (essential: Lys78 activation, Asp194 catalytic) → C-terminal regulatory domain. "
            "STK11 forms a ternary complex with STRAD (pseudokinase) and MO25 (scaffolding protein) → "
            "STRAD activates STK11 kinase activity ~40-fold and localises it to the cytoplasm. "
            "STK11 phosphorylates Thr172 of AMPKα → AMPK activation → phosphorylation of ACC, raptor → "
            "mTOR suppression → cell polarity establishment. "
            "Most pathogenic STK11 variants are loss-of-function (truncating, missense in kinase domain, large deletions). "
            "LARGE DELETIONS: ~30% of PJS families have large intragenic or whole-gene STK11 deletions → MLPA mandatory. "
            "De novo variants account for ~50% of cases (explains variable family history in PJS)."
        ),
        "n_patients": 40,
        "key_alerts": [
            "STK11-PERIORAL-PIGMENTATION-PATHOGNOMONIC: Mucocutaneous perioral melanocytic macules (lips, buccal mucosa, perioral skin, fingers) in infancy–childhood are PATHOGNOMONIC for Peutz-Jeghers Syndrome — buccal mucosa pigmentation PERSISTS into adulthood; perioral skin pigmentation FADES after puberty (adult absence does NOT exclude PJS); ask about childhood pigment history; biopsy macules if uncertain",
            "STK11-INTUSSUSCEPTION-EMERGENCY: Hamartomatous small bowel polyps act as LEAD POINTS for ACUTE INTUSSUSCEPTION — EMERGENCY requiring immediate CT abdomen/pelvis; manual/surgical reduction; bowel ischaemia can occur within hours; prophylactic polypectomy of all polyps >10 mm at index enteroscopy reduces intussusception hospitalisation by ~50%",
            "STK11-SCTAT-PATHOGNOMONIC-FEMALE: Sex cord tumour with annular tubules (SCTAT) is a PATHOGNOMONIC ovarian tumour for PJS in females — 25–30% of all SCTAT patients have PJS; PJS-associated SCTAT is usually BILATERAL, SMALL, CALCIFIED and BENIGN (differs from sporadic SCTAT which is unilateral and malignant); annual TVUS + CA-125 from age 18–25",
            "STK11-BREAST-45-50pct-MRI-25: Breast cancer lifetime risk is 45–50% in female STK11 PJS carriers — the second highest hereditary breast cancer risk after BRCA1/2; ANNUAL BREAST MRI + mammography from age 25 is mandatory; begin mammography at age 25–30 as MRI complement; risk-reducing mastectomy discussion appropriate if risk >50%",
            "STK11-PANCREATIC-11-36pct: Pancreatic cancer risk in STK11 PJS is 11–36% lifetime — one of the highest relative risks for pancreatic cancer of any hereditary cancer syndrome; annual pancreatic MRI/MRCP from age 30–35 or 10 years before earliest family case; EUS (endoscopic ultrasound) at same interval for those with highest risk or family history of pancreatic cancer",
            "STK11-SMALL-BOWEL-CAPSULE-ENTEROSCOPY: Capsule endoscopy or MR enterography every 3 years from age 8 for small bowel hamartoma surveillance — polyps ≥10 mm → balloon-assisted enteroscopy for polypectomy (reduces intussusception risk); small bowel cancer risk 2–13% lifetime; double-balloon enteroscopy if surgical small bowel polypectomy history",
            "STK11-DE-NOVO-50pct: ~50% of PJS cases arise from DE NOVO STK11 mutations — no family history does NOT exclude the diagnosis; any patient with characteristic mucocutaneous pigmentation + GI hamartomas should have STK11 genetic testing regardless of family history; large STK11 deletions (30% of pathogenic alleles) detected by MLPA only",
            "STK11-MLPA-LARGE-DELETION: Approximately 30% of pathogenic STK11 alleles are large intragenic or whole-gene deletions undetectable by standard sequencing — MLPA IS MANDATORY in all PJS evaluation panels; a patient with clinical PJS and negative sequencing must have MLPA performed before STK11 is called negative",
        ],
        "etiologies": {
            "Truncating (frameshift/nonsense) — kinase domain": 14,
            "Large deletion (MLPA required) — partial to whole gene": 12,
            "Pathogenic missense (kinase domain Lys78, Asp194)": 8,
            "Splice-site variant": 4,
            "De novo (confirmed by parental testing)": 2,
        },
        "stats": {
            "mean_dx_age_y": 22.4,
            "pct_perioral_pigmentation": 97,
            "pct_intussusception_episode": 31,
            "pct_sctat_female": 22,
            "pct_breast_surveillance_mri": 74,
            "pct_pancreatic_surveillance_mri": 48,
            "mean_dx_delay_months": 6.2,
        },
        "dx_delay_distribution": {"<3 m": 16, "3–12 m": 14, "1–3 y": 8, ">3 y": 2},
    },
    # ── SMAD4 — Juvenile Polyposis / HHT Overlap ─────────────────────────
    {
        "gene": "SMAD4",
        "protein": "SMAD4 — Juvenile Polyposis/HHT Overlap AD, Telangiectasia + Polyposis = SMAD4, Protein-Losing Enteropathy, Aortic Dilatation",
        "alias": (
            "SMAD4 (DPC4); OMIM gene 600993; Juvenile Polyposis Syndrome (JPS) OMIM 175050; "
            "18q21.2; 552 aa; ~60 kDa; AD; ~20–25% of JPS families (remainder BMPR1A). "
            "SMAD4 encodes the central transcription factor of the TGF-β/BMP signalling pathway — "
            "the common SMAD (co-SMAD) that all receptor-regulated SMADs (R-SMADs: SMAD1/2/3/5/8) "
            "must partner with to form active transcription complexes. "
            "TGF-β signalling: TGF-β ligand → TGFβR2 + TGFβR1 (ALK5) receptor complex → "
            "SMAD2/3 phosphorylation → SMAD2/3 heterotrimerise with SMAD4 → nuclear translocation → "
            "p21, p15, E-cadherin transcription → cell cycle arrest, differentiation. "
            "BMP signalling: BMP ligand → BMPR2 + BMPR1 (ALK2/3/6) → "
            "SMAD1/5/8 phosphorylation → SMAD1/5/8 + SMAD4 trimer → nuclear target genes. "
            "Without SMAD4: both TGF-β AND BMP tumour-suppressor arms fail simultaneously → "
            "epithelial proliferation, loss of differentiation, hamartomatous → adenomatous transformation. "
            "JUVENILE POLYPOSIS: multiple juvenile polyps (hamartomatous, mucus-filled cysts, "
            "inflamed stroma, surface erosion — distinct from Peutz-Jeghers arborising smooth muscle); "
            "predominantly colorectal + stomach; CRC risk 39–68% lifetime; "
            "GASTRIC POLYPOSIS particularly severe in SMAD4-JPS (vs BMPR1A-JPS less gastric involvement). "
            "HHT OVERLAP — SMAD4-SPECIFIC: SMAD4 disrupts BMP9/10 signalling → vascular malformations: "
            "TELANGIECTASIA (skin, mucous membranes, lips, fingertips) identical to HHT1/2 (ENG/ACVRL1); "
            "PULMONARY ARTERIOVENOUS MALFORMATIONS (PAVMs) — risk of paradoxical emboli, brain abscess; "
            "HEREDITARY HAEMORRHAGIC TELANGIECTASIA (HHT) features in SMAD4-JPS carriers; "
            "COEXISTENCE of JUVENILE POLYPOSIS + HHT features → SMAD4 (NOT BMPR1A): pathognomonic combination. "
            "AORTIC DILATATION: SMAD4-JPS/HHT overlap carriers have AORTIC ANEURYSM/DILATATION risk — "
            "baseline echocardiogram + aortic surveillance (annual MRI aorta); "
            "beta-blockers/losartan in aortic dilatation as per Marfan protocol. "
            "PROTEIN-LOSING ENTEROPATHY: severe gastric polyposis in SMAD4 → "
            "massive protein loss from eroded/ulcerated gastric polyps → hypoalbuminaemia, oedema, failure to thrive → "
            "early total gastrectomy may be lifesaving in severe SMAD4 gastric polyposis. "
            "JUVENILE POLYPS — WATCH FOR ADENOMATOUS FOCUS: juvenile polyps can develop ADENOMATOUS foci "
            "(mixed adenomatous-hamartomatous polyp) → these mixed polyps have highest malignant potential; "
            "systematic polypectomy with histological review for adenomatous component. "
            "SURVEILLANCE: annual colonoscopy from age 15; upper GI endoscopy from age 15; "
            "screen for PAVMs (chest CT) at diagnosis and every 5 years; "
            "annual echocardiography for aortic root diameter."
        ),
        "aa": "552 aa",
        "kDa": "~60 kDa",
        "locus": "18q21.2",
        "omim_gene": 600993,
        "omim_disease": 175050,
        "inheritance": "AD; SMAD4 = ~20–25% JPS (remainder BMPR1A); SMAD4-specific HHT overlap not seen with BMPR1A; de novo ~25%",
        "gene_class": (
            "SMAD4 encodes the common SMAD (co-SMAD) / DPC4 (Deleted in Pancreatic Carcinoma 4). "
            "Domains: N-terminal MH1 domain (DNA binding, β-hairpin) → linker region (regulated by ubiquitination) → "
            "C-terminal MH2 domain (R-SMAD interaction, Smad4 oligomerisation, transcriptional activation). "
            "The MH2 domain contains the SSXS binding site for activated R-SMADs and the L3 loop for SMAD4-specific interaction. "
            "SMAD4 is the OBLIGATE partner for ALL R-SMADs (SMAD1/2/3/5/8) → "
            "SMAD4 loss abrogates BOTH TGF-β (SMAD2/3 axis) and BMP (SMAD1/5/8 axis) tumour-suppressor signalling. "
            "Somatic SMAD4 deletion is common in pancreatic cancer (18q loss, ~55%); "
            "germline SMAD4 = JPS/HHT. "
            "Pathogenic SMAD4 variants cluster in MH2 domain (L3 loop, R378 hotspot) and MH1 domain; "
            "large intragenic deletions detected by MLPA only."
        ),
        "n_patients": 40,
        "key_alerts": [
            "SMAD4-TELANGIECTASIA-PLUS-POLYPOSIS-EQUALS-SMAD4: The combination of HEREDITARY HAEMORRHAGIC TELANGIECTASIA features (mucocutaneous telangiectasia, PAVMs, epistaxis, GI bleeding) PLUS JUVENILE POLYPOSIS is PATHOGNOMONIC for SMAD4 — NOT seen with BMPR1A-JPS; any patient with both features must have SMAD4 testing immediately; this combined syndrome (JPHT) has distinct vascular risks beyond standard JPS",
            "SMAD4-AORTIC-DILATATION-ANNUAL-ECHO: SMAD4/JPHT overlap carriers have AORTIC ROOT DILATATION risk — echocardiogram at diagnosis and ANNUALLY for aortic root surveillance; aortic dilatation managed as per Marfan/connective tissue protocol (beta-blocker or losartan); surgical threshold typically 45–50 mm (lower if rapid expansion); inform all SMAD4 carriers of this vascular risk",
            "SMAD4-PROTEIN-LOSING-ENTEROPATHY: Severe gastric polyposis in SMAD4-JPS → massive protein loss from ulcerated/eroded gastric hamartomas → HYPOALBUMINAEMIA, oedema, failure to thrive; check albumin at diagnosis and with symptoms; total or subtotal GASTRECTOMY may be LIFESAVING in severe protein-losing enteropathy from SMAD4 gastric polyposis — do NOT delay surgery when albumin falls below 20 g/L",
            "SMAD4-PAVM-CHEST-CT-DIAGNOSIS: Pulmonary arteriovenous malformations (PAVMs) occur in SMAD4/JPHT — risk of PARADOXICAL EMBOLISM (stroke, brain abscess from venous thrombus or bacteria bypassing pulmonary capillary filter); CHEST CT at diagnosis and every 5 years; PAVMs >3 mm → transcatheter embolisation; AVOID air embolism risk (IV filters, dental prophylaxis)",
            "SMAD4-GASTRIC-POLYPOSIS-SEVERE: SMAD4-JPS causes more severe GASTRIC POLYPOSIS than BMPR1A-JPS — annual upper GI endoscopy from age 15; systematic polypectomy; histological review of ALL excised polyps for adenomatous foci (mixed hamartomatous-adenomatous polyps have highest malignant potential); gastric cancer risk requires endoscopic vigilance even in young patients",
            "SMAD4-CRC-39-68pct-ANNUAL-COLONOSCOPY: Colorectal cancer risk in SMAD4-JPS is 39–68% lifetime — ANNUAL colonoscopy from age 15 with systematic polypectomy; colectomy with IRA or total proctocolectomy when polyp burden is unmanageable (>20 polyps per session or high-grade dysplasia); total proctocolectomy preferred when rectal polyp burden is severe",
            "SMAD4-BMPR1A-DDx: BMPR1A germline mutation also causes JPS (majority, ~75%) but WITHOUT HHT overlap features — telangiectasia, PAVMs, aortic dilatation are NOT features of BMPR1A-JPS; the presence of HHT features DISTINGUISHES SMAD4 from BMPR1A; BMPR1A-JPS management focuses on GI polyposis alone without vascular surveillance",
            "SMAD4-MLPA-LARGE-DELETIONS: Large SMAD4 intragenic or whole-gene deletions are not detected by sequencing — MLPA required in all JPS/JPHT panels; de novo SMAD4 mutations occur in ~25% of cases; negative sequencing + negative MLPA in a patient with juvenile polyposis → test BMPR1A (both genes in all JPS panels)",
        ],
        "etiologies": {
            "Truncating (frameshift/nonsense) — MH1 or MH2 domain": 16,
            "Pathogenic missense — MH2 domain (L3 loop, R378)": 12,
            "Large deletion (MLPA required)": 6,
            "Splice-site variant": 4,
            "De novo (parental testing confirmed)": 2,
        },
        "stats": {
            "mean_dx_age_y": 18.3,
            "pct_hht_features": 68,
            "pct_pavm_detected": 42,
            "pct_aortic_dilatation": 28,
            "pct_protein_losing_enteropathy": 18,
            "pct_colectomy_done": 48,
            "mean_dx_delay_months": 11.6,
        },
        "dx_delay_distribution": {"<6 m": 16, "6–18 m": 14, "1.5–4 y": 8, ">4 y": 2},
    },
]

# ─── Patient cohort generation ────────────────────────────────────────────────

def _make_cohort():
    cohort = {}
    for i, gene_info in enumerate(GI_GENES):
        seed = SEED_BASE + i
        rng = random.Random(seed)
        gene = gene_info["gene"]
        n = gene_info["n_patients"]
        patients = []
        for p in range(n):
            age_dx = round(rng.gauss(gene_info["stats"].get("mean_dx_age_y", 40), 8), 1)
            age_dx = max(5, min(80, age_dx))
            dx_delay = round(rng.gauss(gene_info["stats"].get("mean_dx_delay_months", 12), 5), 1)
            dx_delay = max(0.5, min(60, dx_delay))
            patients.append({
                "patient_id": f"{gene}-{seed}-{p+1:03d}",
                "gene": gene,
                "age_at_diagnosis": age_dx,
                "diagnosis_delay_months": dx_delay,
                "seed": seed,
            })
        cohort[gene] = {
            **gene_info,
            "patients": patients,
        }
    return cohort


_COHORT = _make_cohort()


# ─── API functions ─────────────────────────────────────────────────────────────

def get_overview():
    total = sum(v["n_patients"] for v in _COHORT.values())
    mean_dx_age = round(
        sum(p["age_at_diagnosis"] for v in _COHORT.values() for p in v["patients"]) / total, 1
    )
    mean_dx_delay = round(
        sum(p["diagnosis_delay_months"] for v in _COHORT.values() for p in v["patients"]) / total, 1
    )
    # Aggregate key stats
    apc = _COHORT["APC"]["stats"]
    mlh1 = _COHORT["MLH1"]["stats"]
    msh2 = _COHORT["MSH2"]["stats"]
    msh6 = _COHORT["MSH6"]["stats"]
    pms2 = _COHORT["PMS2"]["stats"]
    stk11 = _COHORT["STK11"]["stats"]
    smad4 = _COHORT["SMAD4"]["stats"]

    top_alerts = []
    for v in _COHORT.values():
        top_alerts.extend(v["key_alerts"][:2])

    genes_summary = []
    for g, v in _COHORT.items():
        pts = v["patients"]
        mean_age = round(sum(p["age_at_diagnosis"] for p in pts) / len(pts), 1)
        genes_summary.append({
            "gene": g,
            "protein_short": v["protein"][:80],
            "locus": v["locus"],
            "inheritance": v["inheritance"].split(";")[0],
            "omim_disease": v["omim_disease"],
            "mean_dx_age": mean_age,
            "n_patients": v["n_patients"],
        })

    return {
        "atlas": "Hereditary-GI-Cancer-Atlas",
        "subtitle": "Complete 8-Gene Hereditary GI Cancer Syndrome Reference",
        "genes": genes_summary,
        "aggregate_stats": {
            "total_patients": total,
            "mean_dx_age_years": mean_dx_age,
            "mean_dx_delay_months": mean_dx_delay,
            "apc_chrpe_pct": apc["pct_chrpe_present"],
            "apc_colectomy_pct": apc["pct_colectomy_done"],
            "mlh1_msi_h_pct": mlh1["pct_msi_h_confirmed"],
            "mlh1_aspirin_pct": mlh1["pct_aspirin_commenced"],
            "msh2_epcam_deletion_pct": msh2["pct_epcam_deletion"],
            "msh6_endometrial_index_pct": msh6["pct_endometrial_index_cancer"],
            "pms2_mlpa_required_pct": pms2["pct_mlpa_required_for_diagnosis"],
            "stk11_intussusception_pct": stk11["pct_intussusception_episode"],
            "smad4_hht_features_pct": smad4["pct_hht_features"],
            "smad4_pavm_pct": smad4["pct_pavm_detected"],
            "cascade_tested_pct": 76,
        },
        "top_alerts": top_alerts,
        "seeds": list(range(SEED_BASE, SEED_BASE + 8)),
    }


def get_breakdown():
    result = {}
    for gene, info in _COHORT.items():
        pts = info["patients"]
        result[gene] = {
            "gene": gene,
            "n_patients": info["n_patients"],
            "alias": info["alias"],
            "gene_class": info["gene_class"],
            "locus": info["locus"],
            "aa": info["aa"],
            "kDa": info["kDa"],
            "omim_gene": info["omim_gene"],
            "omim_disease": info["omim_disease"],
            "inheritance": info["inheritance"],
            "key_alerts": info["key_alerts"],
            "etiologies": info["etiologies"],
            "stats": info["stats"],
            "dx_delay_distribution": info["dx_delay_distribution"],
            "patients": pts[:10],
        }
    return result


def get_definitions():
    return {
        "atlas": "Hereditary-GI-Cancer-Atlas",
        "concepts": {
            "Mismatch Repair (MMR) and Microsatellite Instability (MSI)": (
                "Mismatch repair (MMR) corrects replication errors at microsatellites (short tandem repeats, STRs). "
                "Four proteins form two functional complexes: MutSα (MSH2+MSH6) recognises base-base mismatches and "
                "small insertion-deletion loops; MutSβ (MSH2+MSH3) recognises large insertion-deletion loops; "
                "MutLα (MLH1+PMS2) is the endonuclease that nicks the error-containing strand; "
                "MutLβ (MLH1+PMS1) and MutLγ (MLH1+MLH3) have minor roles. "
                "Loss of any MMR protein → replication errors accumulate at STRs → MICROSATELLITE INSTABILITY (MSI). "
                "MSI-H: ≥30–40% of tested STRs are unstable (NCI panel: BAT25, BAT26, D5S346, D2S123, D17S250); "
                "MSI-L: <30% unstable; MSS: no instability. "
                "TESTING: PCR capillary electrophoresis (gold standard) or next-generation sequencing; "
                "IHC (immunohistochemistry) for MMR protein loss has equivalent sensitivity + identifies WHICH protein is absent. "
                "SPORADIC MSI-H (MLH1 promoter methylation): ~15% CRC; older patients; BRAF V600E in ~80%; "
                "LYNCH MSI-H: germline MMR mutation; younger; BRAF V600E rare. "
                "UNIVERSAL TUMOUR TESTING: all CRC should have MMR/MSI testing (identifies Lynch, guides pembrolizumab eligibility, prognostic stage II information)."
            ),
            "FAP vs MAP vs Lynch vs Hamartomatous Polyposis — Key Differentiators": (
                "FAP (APC, AD): >100 adenomatous polyps; 100% CRC lifetime risk; CHRPE; duodenal polyps; "
                "desmoids; colectomy mandatory by age 25–30; AFAP = <100 polyps, later onset, proximal. "
                "MAP (MUTYH, AR): 10–100 adenomatous polyps; biallelic required; Y179C/G396D founders (European); "
                "later onset (40–50s); right-sided predominance; CRC risk ~80% by 60; partner testing mandatory; "
                "monoallelic = surveillance only (not MAP). "
                "LYNCH (MLH1/MSH2/MSH6/PMS2, AD): normal/few polyps; MSI-H tumours; Amsterdam II families; "
                "extracolonic cancers (endometrial, ovarian, urothelial, gastric, brain); "
                "colonoscopy surveillance + aspirin + immunotherapy (pembrolizumab for MSI-H metastatic). "
                "PEUTZ-JEGHERS (STK11, AD): hamartomatous polyps (arborising smooth muscle histology); "
                "perioral pigmentation; SCTAT; intussusception risk; breast/pancreatic cancer high risk; "
                "capsule enteroscopy/balloon enteroscopy surveillance. "
                "JUVENILE POLYPOSIS (SMAD4/BMPR1A, AD): juvenile polyps (mucus-filled cysts, inflamed stroma); "
                "SMAD4-specific HHT overlap (telangiectasia, PAVMs, aortic dilatation); protein-losing enteropathy; "
                "colonic and gastric predominance. "
                "KEY DDx: adenomatous polyps = FAP/MAP/Lynch; hamartomatous = PJS/JPS; MSI-H + extracolonic = Lynch; "
                "pigmentation = STK11-PJS; HHT + polyposis = SMAD4-JPS."
            ),
            "Lynch Syndrome MSI Testing — Sporadic vs Germline Distinction": (
                "MSI-H in CRC: 15% sporadic (epigenetic MLH1 methylation + BRAF V600E) vs 3–5% Lynch (germline MMR mutation). "
                "Step 1: Universal MMR IHC + MSI testing on ALL CRC (NICE/NCCN). "
                "Step 2: If MSI-H or dMMR → BRAF V600E somatic testing + MLH1 promoter methylation. "
                "If BRAF V600E positive: highly likely sporadic MLH1-methylated (SSMSE) → Lynch testing low priority. "
                "If BRAF V600E negative: germline MMR testing mandatory for MLH1-dMMR tumours. "
                "If MSH2, MSH6, or PMS2 dMMR: Lynch testing regardless of BRAF status. "
                "MLH1/MSH2/MSH6/PMS2 sequencing + MLPA (including EPCAM probes for MSH2 silencing). "
                "MSH6 Lynch: may show MSI-L or MSS on standard Bethesda panel → mononucleotide pentaplex required. "
                "PMS2 Lynch: pseudogene problem → pseudogene-aware long-read sequencing + MLPA mandatory. "
                "Constitutional MLH1 methylation: normal germline sequencing + MLH1-dMMR tumour → blood DNA methylation assay."
            ),
            "Chemoprevention and Immunotherapy in Hereditary GI Cancer": (
                "ASPIRIN (CAPP2 TRIAL): 600 mg/day aspirin in Lynch syndrome for ≥2 years → 50% reduction in "
                "all Lynch cancers including CRC and endometrial; DELAYED effect (peak 10+ years post-trial); "
                "COX-2-dependent prostaglandin suppression → reduced epithelial proliferation; "
                "start at age 25–30; lower dose (150–300 mg) under investigation in CaPP3; "
                "contraindicate only in known GI haemorrhage/anticoagulation. "
                "COX-2 INHIBITORS IN FAP: celecoxib 800 mg/day and sulindac 150 mg twice daily reduce FAP polyp burden 28–45%; "
                "NOT a substitute for colectomy; used post-colectomy for duodenal/rectal polyp control; "
                "GI + cardiovascular risk monitoring required. "
                "PEMBROLIZUMAB (KEYTRUDA): FDA 2017 first tumour-agnostic approval for dMMR/MSI-H solid tumours; "
                "KEYNOTE-177: first-line pembrolizumab superior to FOLFOX/FOLFIRI in MSI-H mCRC (ORR 45%); "
                "Lynch MSI-H CRC: outstanding responders (TIL-rich, immune evasion via MMR pathway); "
                "CMMRD paediatric MSI-H tumours: pembrolizumab off-label use with documented responses; "
                "dMMR/MSI-H testing should be performed on ALL CRC before starting palliative chemotherapy."
            ),
        },
        "pharmacological_distinctions": [
            "Aspirin 600 mg/day in Lynch syndrome (CAPP2): reduces ALL Lynch-associated cancer incidence by ~50% in Lynch syndrome carriers; effect delayed (peak 5–10 years after starting); start at age 25–30; contraindicate in GI bleeding history or anticoagulation therapy; lower-dose aspirin (150–300 mg) tested in CaPP3 (results awaited); mechanism: COX-2-dependent prostaglandin E2 suppression → reduced epithelial proliferation and MSI-driven adenoma formation",
            "Pembrolizumab (anti-PD-1, Keytruda) for dMMR/MSI-H CRC: FDA-approved (2017) first-line for metastatic MSI-H CRC (KEYNOTE-177) — superior to FOLFOX/FOLFIRI as first-line treatment; ORR 45% vs 33%; durable responses (PFS HR 0.60); Lynch CRC is highly responsive; tumour-agnostic approval applies to ALL dMMR/MSI-H solid tumours regardless of site; test ALL metastatic CRC before starting palliative chemotherapy",
            "Celecoxib/sulindac for FAP (APC): celecoxib 800 mg/day or sulindac 150 mg BID reduce colorectal adenoma burden 28–45% in FAP — ADJUNCT ONLY, never substitute for prophylactic colectomy; use post-colectomy for duodenal polyp load control (Spigelman Stage II–III) and rectal stump polyp control after IRA; monitor GI bleeding, cardiovascular risk (celecoxib) and renal function",
            "Risk-reducing hysterectomy and BSO in Lynch syndrome: total hysterectomy + bilateral salpingo-oophorectomy (BSO) reduces endometrial cancer risk >95% and ovarian cancer risk ~80% in Lynch female carriers; discuss after family completion (completed childbearing); timing: age 40–45 for MSH6/PMS2 (later onset), age 35–40 for MLH1/MSH2 (earlier onset); laparoscopic approach preferred; hormone replacement therapy (HRT) acceptable after BSO to prevent premature menopause sequelae",
            "Balloon-assisted enteroscopy for Peutz-Jeghers (STK11): double-balloon or single-balloon enteroscopy is the therapeutic standard for small bowel polypectomy in PJS — prophylactic polypectomy of all polyps ≥10 mm at capsule enteroscopy reduces acute intussusception hospitalisation by ~50%; intraoperative enteroscopy reserved for complicated intussusception surgery; repeat capsule endoscopy/MR enterography every 3 years from age 8",
            "Aortic surveillance protocol in SMAD4 Juvenile Polyposis/HHT overlap (JPHT): echocardiography at diagnosis and annually for aortic root diameter — aortic dilatation in SMAD4/JPHT managed as per Marfan/connective tissue protocol: beta-blocker (propranolol, atenolol) or losartan to reduce aortic growth rate; surgical threshold 45–50 mm or rapid expansion (>3 mm/year); beta-blocker NOT contraindicated in PAVM (unlike in severe asthma/HHT); echocardiogram preferred over CT for annual surveillance (avoid radiation)",
        ],
        "key_standards": [
            "NCCN Guidelines — Genetic/Familial High-Risk Assessment: Colorectal, Version 2.2024 — FAP (APC): prophylactic colectomy age 25–30 (classic FAP); annual endoscopy every 6–12 months post-IRA; Spigelman staging for duodenum; AFAP: colonoscopy every 1–2 years, polypectomy-first strategy; Lynch (MLH1/MSH2/MSH6/PMS2): annual colonoscopy from age 25; aspirin; gynaecological surveillance; risk-reducing surgery discussion",
            "NCCN Colorectal Cancer Screening Guidelines — Universal MMR/MSI testing: ALL newly diagnosed CRC should have MMR IHC and/or MSI testing regardless of age; BRAF V600E somatic testing in MLH1-dMMR tumours; germline testing for ALL dMMR tumours with non-sporadic pattern; pembrolizumab as first-line for MSI-H metastatic CRC (KEYNOTE-177)",
            "European Society of Coloproctology (ESCP) / ESMO Lynch Syndrome Guidelines 2023 — colonoscopy intervals: MLH1/MSH2 annual from age 25; MSH6 biennial from age 25; PMS2 biennial from age 25–30; aspirin 600 mg/day recommended all Lynch; EPCAM deletion included in all Lynch testing panels; PMS2 pseudogene-aware methodology mandatory; constitutional MLH1 methylation testing in appropriate cases",
            "NCCN MAP (MUTYH) Guidelines — biallelic = MAP (colonoscopy 1–2 yearly from age 25; polypectomy; colectomy if unmanageable burden); monoallelic = enhanced surveillance only (3–5 yearly colonoscopy); partner testing mandatory for all MAP index cases; Y179C and G396D founder testing plus full-gene sequencing; non-European ancestry: full-gene sequencing required; MLPA for large deletions",
            "Peutz-Jeghers Syndrome Clinical Guidelines (European Reference Network ITHACA) — capsule endoscopy/MR enterography from age 8, every 3 years; balloon enteroscopy polypectomy of polyps ≥10 mm; colonoscopy + gastroduodenoscopy from age 8, every 3 years; annual breast MRI from age 25; annual gynaecological examination + TVUS from age 18; pancreatic MRI/MRCP from age 30–35; STK11 MLPA mandatory; de novo in 50% (test both parents)",
            "SMAD4/BMPR1A Juvenile Polyposis/HHT Overlap Clinical Guidelines (ITHACA/HHT Foundation International) — annual colonoscopy from age 15; annual upper GI endoscopy from age 15; chest CT at diagnosis + every 5 years for PAVMs; echocardiogram at diagnosis and annually for aortic root (SMAD4-specific); baseline head MRI (PAVM-related stroke risk); total gastrectomy for protein-losing enteropathy unresponsive to endoscopic management; SMAD4 MLPA + BMPR1A panel in all JPS",
        ],
    }
