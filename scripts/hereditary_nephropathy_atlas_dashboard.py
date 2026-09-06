#!/usr/bin/env python3
"""Hereditary-Nephropathy-Atlas — Complete 8-Gene Hereditary Nephropathy Atlas
COL4A5  (Type IV Collagen Alpha-5 Chain; 1685 aa; ~175 kDa; Xq22.3; X-linked;
          OMIM gene 303630; Alport Syndrome 1 OMIM 301050;
          most common hereditary nephritis; X-linked males: ESRD by 20–30 years;
          hematuria + SNHL + anterior lenticonus PATHOGNOMONIC;
          skin biopsy: COL4A5 absent on immunofluorescence — fast diagnosis;
          ACEi ASAP delays ESRD even pre-proteinuria; AVOID nephrotoxins;
          seed SEED_BASE+0) ·
COL4A3  (Type IV Collagen Alpha-3 Chain; 1670 aa; ~160 kDa; 2q36.3; AR/AD;
          OMIM gene 120070; Alport Syndrome 2 OMIM 203780; TBMN OMIM 141200;
          biallelic: AR Alport (severe = XL males); monoallelic: TBMN + 10–15% FSGS risk;
          ACEi mandatory if proteinuria ≥ 0.5 g/day even in TBMN;
          seed SEED_BASE+1) ·
NPHS1   (Nephrin; 1241 aa; ~135 kDa; 19q13.12; AR;
          OMIM gene 602716; Congenital Nephrotic Syndrome Finnish type OMIM 256300;
          main slit diaphragm structural component; massive proteinuria in utero;
          large placenta (>25% birth weight) PATHOGNOMONIC for CNS-F;
          Fin-major c.121delCT + Fin-minor c.3325C>T = 95% Finnish;
          bilateral nephrectomy + dialysis + transplant — recurrence RARE;
          seed SEED_BASE+2) ·
NPHS2   (Podocin; 383 aa; ~42 kDa; 1q25.2; AR;
          OMIM gene 604766; FSGS2 / Steroid-Resistant Nephrotic Syndrome OMIM 600995;
          hairpin membrane topology at slit diaphragm; steroid-RESISTANT PATHOGNOMONIC;
          R138Q most common European pathogenic variant; p.R229Q low-penetrance modifier;
          post-transplant recurrence LOW; calcineurin inhibitors partial response;
          seed SEED_BASE+3) ·
WT1     (Wilms Tumour Protein 1; 449 aa; ~52 kDa; 11p13; AD;
          OMIM gene 607102; Denys-Drash OMIM 194080; Frasier OMIM 136680;
          DDS: missense R394W hotspot → diffuse mesangial sclerosis + DSD + Wilms tumour;
          Frasier: +KTS splice variants → FSGS + gonadoblastoma — NO Wilms tumour;
          gonadectomy MANDATORY in 46XY DSD; annual renal US (DDS) until age 8;
          seed SEED_BASE+4) ·
UMOD    (Uromodulin / Tamm-Horsfall Protein; 640 aa; ~72 kDa; 16p12.3; AD;
          OMIM gene 191845; FJHN / MCKD2 OMIM 162000 / 603860;
          most common hereditary tubulointerstitial nephritis;
          hyperuricaemia + gout PRESENTING FEATURE (teenage or early adult);
          UMOD misfolds in ER → ER stress → tubular cell death → interstitial nephritis;
          urine UMOD assay < 5th centile diagnostic; avoid uricosurics;
          seed SEED_BASE+5) ·
PKD1    (Polycystin-1; 4303 aa; ~462 kDa; 16p13.3; AD;
          OMIM gene 601313; ADPKD type 1 OMIM 173900;
          most common hereditary kidney disease: 1 in 400–1000 worldwide;
          bilateral renal cysts + flank pain + haematuria + hypertension + hepatic cysts;
          tolvaptan FDA 2018 (V2R antagonist, slows kidney growth by TKV criteria);
          intracranial aneurysm MRA brain MANDATORY if family history of rupture;
          seed SEED_BASE+6) ·
TRPC6   (Transient Receptor Potential Cation Channel C6; 931 aa; ~104 kDa; 11q22.1; AD GOF;
          OMIM gene 603652; FSGS6 OMIM 614131;
          GOF → enhanced Ca2+ entry → podocyte calcium overload → podocyte loss;
          adult-onset proteinuria progressing to ESRD (30–50 years); steroid-RESISTANT;
          living-related donor EXCLUSION: exclude carrier relatives from kidney donation;
          calcineurin inhibitors partial response (reduce Ca2+ signalling);
          seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1558–1565)
"""

import random

SEED_BASE = 1558

NEPHROPATHY_GENES = [
    # ── COL4A5 — X-linked Alport Syndrome ──
    {
        "gene": "COL4A5",
        "protein": "Type IV Collagen Alpha-5 — X-linked Alport Syndrome, ACEi ASAP, Anterior Lenticonus PATHOGNOMONIC",
        "alias": (
            "COL4A5; OMIM gene 303630; Alport Syndrome OMIM 301050; Xq22.3; 1685 aa; ~175 kDa; "
            "COL4A5 encodes the alpha-5 chain of type IV collagen, the major structural component "
            "of basement membranes (glomerular BM, tubular BM, Bowman's capsule, Descemet's membrane, "
            "cochlear stria vascularis, anterior lens capsule). Type IV collagen consists of three alpha "
            "chains forming a triple helix; in the GBM the dominant network is alpha3/alpha4/alpha5 "
            "(mature GBM) — X-linked Alport males LACK this network, leaving only the foetal "
            "alpha1/alpha1/alpha2 network which is fragile and prone to splitting. "
            "X-linked (Xq22.3): hemizygous males have full disease (ESRD by 20–30 years without ACEi; "
            "by 40–50 years with ACEi); heterozygous females range from microhaematuria only (30%) "
            "to full Alport (20% by age 60, 40% by age 80 without treatment). "
            "Pathognomonic triad: (1) persistent microscopic haematuria from birth/infancy; "
            "(2) sensorineural hearing loss (bilateral, high-frequency, progressive — cochlear stria "
            "vascularis lacks alpha3/alpha4/alpha5 network); (3) anterior lenticonus (forward bulging "
            "of anterior lens on slit-lamp, PATHOGNOMONIC — present in 25–30% of males by teen years, "
            "virtually diagnostic). Electron microscopy: irregular thickening + thinning + "
            "splitting/lamellation of GBM ('basket-weave' pattern). "
            "Immunofluorescence on skin biopsy: COL4A5 ABSENT (males) or mosaic (females) — "
            "rapid diagnosis without renal biopsy in males. "
            "Treatment: ACEi (ramipril/lisinopril) START AT DIAGNOSIS regardless of proteinuria — "
            "delays ESRD by 13 years (Gross O, JASN 2012). AVOID nephrotoxins (NSAIDs, aminoglycosides, "
            "contrast media). HSCT not indicated. Cochlear implants for SNHL. "
            "Genetic counselling: daughters of affected males = obligate carriers; 50% sons of carriers "
            "affected. Molecular: 90% point mutations/small indels by WES; large deletions/duplications "
            "missed by WES → MLPA mandatory if COL4A5 negative by sequencing."
        ),
        "aa": "1685 aa",
        "kDa": "~175 kDa",
        "locus": "Xq22.3",
        "omim_gene": 303630,
        "omim_disease": 301050,
        "inheritance": "X-linked (Xq22.3); hemizygous males: ESRD 20–30y; heterozygous females: range from microhaematuria to full Alport",
        "gene_class": (
            "COL4A5 (type IV collagen alpha-5) is a 1685-amino acid extracellular matrix protein. "
            "Domain architecture: short 7S domain (N-terminal cross-linking) → long collagen triple "
            "helix (~1500 residues, Gly-X-Y repeats) → C-terminal NC1 domain (heterotrimer assembly). "
            "Triple helix partners: COL4A5 pairs exclusively with COL4A3 and COL4A4 to form the "
            "alpha3/alpha4/alpha5 heterotrimer — the dominant network in mature GBM, cochlear and "
            "ocular BMs. Pathogenic variants: missense Gly substitutions in Gly-X-Y repeats (most "
            "severe if N-terminal triple helix); truncating/splice variants (haploinsufficiency). "
            "Genotype-phenotype: truncating variants → ESRD earlier (median 25y); missense Gly → "
            "variable (median 30y); splice variants → range."
        ),
        "n_patients": 40,
        "key_alerts": [
            "COL4A5-ACEI-ASAP: Start ACEi at diagnosis regardless of proteinuria — delays ESRD by 13 years (Gross O, JASN 2012); do NOT wait for proteinuria",
            "COL4A5-ANTERIOR-LENTICONUS-PATHOGNOMONIC: Forward bulging of anterior lens on slit-lamp virtually diagnostic — present in 25-30% of males by teenage years",
            "COL4A5-SKIN-BIOPSY-FAST-DIAGNOSIS: Immunofluorescence for COL4A5 on skin punch biopsy (epidermal basement membrane) — absent in males, mosaic in female carriers; avoids renal biopsy",
            "COL4A5-MLPA-MANDATORY: Large deletions/duplications (~10%) MISSED by WES/Sanger — MLPA mandatory if sequencing negative in a clinically convincing case",
            "COL4A5-AVOID-NEPHROTOXINS: NSAIDs, aminoglycosides, IV contrast — ABSOLUTELY CONTRAINDICATED in Alport; accelerate progression to ESRD",
            "COL4A5-FEMALE-CARRIERS-AT-RISK: 20% of heterozygous females reach ESRD by age 60; 40% by age 80 — screen ALL females with haematuria, monitor annually",
            "COL4A5-COCHLEAR-IMPLANT: SNHL cochlear-origin (stria vascularis) responds well to cochlear implantation — refer early for implant evaluation",
            "COL4A5-CASCADE-TESTING: All first-degree relatives of index case require urinalysis + audiogram — haematuria in females = obligate carrier management",
        ],
    },
    # ── COL4A3 — Autosomal Alport / Thin Basement Membrane Nephropathy ──
    {
        "gene": "COL4A3",
        "protein": "Type IV Collagen Alpha-3 — AR Alport / AD TBMN, ACEi if Proteinuria, MLPA Mandatory",
        "alias": (
            "COL4A3; OMIM gene 120070; Alport Syndrome AR OMIM 203780; TBMN OMIM 141200; 2q36.3; "
            "1670 aa; ~160 kDa; COL4A3 encodes the alpha-3 chain of type IV collagen, which "
            "heterotrimer-pairs with COL4A4 and COL4A5 to form the mature GBM network. "
            "Biallelic LOF (AR Alport): identical severity to X-linked Alport in males — ESRD by "
            "20–30 years, SNHL, anterior lenticonus, basket-weave GBM ultrastructure. "
            "Monoallelic LOF (Thin Basement Membrane Nephropathy / TBMN): most common cause of "
            "familial haematuria (1 in 5,000 population); thin GBM on EM (<250 nm) without splitting; "
            "previously thought benign — 10–15% develop proteinuria and FSGS progression over decades "
            "→ significant CKD risk. COL4A3 and COL4A4 heterozygous carriers are at equivalent TBMN risk. "
            "Diagnosis: EM showing thin GBM; COL4A3 sequencing + MLPA. "
            "Treatment: ACEi mandatory if proteinuria ≥0.5 g/day (even in TBMN heterozygotes). "
            "Avoid nephrotoxins. Monitor BP, renal function, urine protein annually. "
            "Genetic counselling: biallelic affected child possible if two TBMN carrier parents — "
            'assess partner''s COL4A3/COL4A4 carrier status before family planning.'
        ),
        "aa": "1670 aa",
        "kDa": "~160 kDa",
        "locus": "2q36.3",
        "omim_gene": 120070,
        "omim_disease": 203780,
        "inheritance": "AR (biallelic → Alport, severe); AD/monoallelic → TBMN (10-15% FSGS risk over decades); locus 2q36.3",
        "gene_class": (
            "COL4A3 (type IV collagen alpha-3) 1670 aa, pairs exclusively with COL4A4 and COL4A5 "
            "in the GBM heterotrimer. Pathogenic biallelic variants: nonsense, frameshift, splice, "
            "missense Gly substitutions → absent alpha3 network from GBM → basket-weave ultrastructure. "
            "Heterozygous variants: reduced alpha3 incorporation → uniformly thin GBM (<250 nm) "
            "without splitting. Progressive TBMN risk factors: proteinuria, hypertension, male sex, "
            "concurrent APOL1 variants (in African populations). MLPA: essential to detect large "
            "deletions that span COL4A3/COL4A4 simultaneously."
        ),
        "n_patients": 40,
        "key_alerts": [
            "COL4A3-TBMN-NOT-BENIGN: 10-15% of monoallelic COL4A3 carriers develop proteinuria + FSGS + CKD over decades — lifelong annual monitoring mandatory",
            "COL4A3-ACEI-IF-PROTEINURIA: ACEi mandatory if proteinuria ≥0.5 g/day in TBMN heterozygotes — delays progression",
            "COL4A3-MLPA-COL4A4-SIMULTANEOUSLY: Large deletions can span COL4A3 AND COL4A4 — MLPA covers both genes",
            "COL4A3-BIALLELIC-AR-ALPORT: Biallelic COL4A3 = same severity as X-linked Alport males — ESRD by 20-30 years, SNHL, anterior lenticonus",
            "COL4A3-PARTNER-TESTING: If index patient is TBMN carrier, test partner for COL4A3/COL4A4 before conception — biallelic child possible",
            "COL4A3-EM-MANDATORY: Electron microscopy essential to distinguish thin BM (<250 nm, TBMN) from splitting/lamellation (Alport) — biopsy if diagnostic uncertainty",
            "COL4A3-MALE-RISK-HIGHER: Male TBMN carriers have higher CKD progression risk than females — sex-stratified monitoring intervals",
            "COL4A3-CASCADE-TESTING: All first-degree relatives require urinalysis; offer sequencing to at-risk family members",
        ],
    },
    # ── NPHS1 — Congenital Nephrotic Syndrome Finnish Type ──
    {
        "gene": "NPHS1",
        "protein": "Nephrin — Congenital Nephrotic Syndrome Finnish Type, Large Placenta PATHOGNOMONIC, Bilateral Nephrectomy + Transplant",
        "alias": (
            "NPHS1; OMIM gene 602716; CNS-F OMIM 256300; 19q13.12; 1241 aa; ~135 kDa; "
            "NPHS1 encodes nephrin, a single-pass type I transmembrane protein and the principal "
            "structural component of the slit diaphragm (SD) — the main size-selective filtration "
            "barrier of the podocyte. Nephrin belongs to the immunoglobulin superfamily; its "
            "extracellular domain forms a zipper-like structure with nephrin from the opposing podocyte "
            "foot process — each 'rung' is a disulfide-stabilised dimer. Intracellular: nephrin binds "
            "podocin (NPHS2), neph1, CD2AP, FAK — forming the SD scaffold. NPHS1 LOF: slit diaphragm "
            "absent → unrestricted passage of albumin and plasma proteins → massive proteinuria from "
            "in utero. Congenital nephrotic syndrome Finnish type: massive proteinuria at birth "
            "(urine protein/creatinine >100 g/g); oedema anasarca; large placenta (>25% birth weight "
            "PATHOGNOMONIC — oedematous placenta due to protein loss). Finnish founder mutations: "
            "Fin-major (c.121_122delCT; p.Leu41ValfsTer1539) ~80% Finnish alleles; Fin-minor "
            "(c.3325C>T; p.Arg1109Ter) ~15% Finnish alleles — combined 95% of Finnish alleles. "
            "Non-Finnish: diverse variants worldwide, often compound heterozygous. "
            "Management: bilateral nephrectomy (usually 6–12 months) + peritoneal dialysis + "
            "renal transplant (usually 2–3 years) — recurrence is RARE (no circulating factor). "
            "Albumin infusions, nutritional support, ACEi/indomethacin to reduce proteinuria before "
            "nephrectomy. Thyroid supplementation (protein-losing hypothyroidism). "
            "NO role for immunosuppression."
        ),
        "aa": "1241 aa",
        "kDa": "~135 kDa",
        "locus": "19q13.12",
        "omim_gene": 602716,
        "omim_disease": 256300,
        "inheritance": "AR (biallelic); 19q13.12; Finnish founder mutations Fin-major + Fin-minor = 95% of Finnish alleles",
        "gene_class": (
            "NPHS1 (nephrin) is a 1241-amino acid IgG superfamily transmembrane protein forming "
            "the structural backbone of the podocyte slit diaphragm. Extracellular: 8 Ig-like "
            "domains + 1 fibronectin type III domain → homodimerisation across SD. Transmembrane + "
            "intracellular: PDZ-binding motif at C-terminus → binds CD2AP, podocin, Neph1, FAK → "
            "signals PI3K/Akt, Nck → actin cytoskeleton regulation in foot processes. Pathogenic "
            "variants: Finnish frameshift (Fin-major) causes complete loss; missense in Ig domains "
            "→ variable residual SD function → milder phenotype (adolescent-onset in some). "
            "Podocyte signalling through nephrin Y-phosphorylation (Src kinase, Fyn) regulates "
            "actin dynamics — variants that preserve structure but impair signalling can cause "
            "milder SRNS."
        ),
        "n_patients": 40,
        "key_alerts": [
            "NPHS1-LARGE-PLACENTA-PATHOGNOMONIC: Placenta >25% of birth weight due to oedema from in-utero protein loss — request placenta weight at delivery",
            "NPHS1-NO-IMMUNOSUPPRESSION: NPHS1-LOF nephrotic syndrome is GENETIC not immune-mediated — steroids/cyclophosphamide are INEFFECTIVE and HARMFUL",
            "NPHS1-BILATERAL-NEPHRECTOMY: Bilateral nephrectomy (6-12 months) + dialysis + transplant is the definitive pathway — recurrence post-transplant is RARE",
            "NPHS1-THYROID-SUPPLEMENTATION: Massive proteinuria causes protein-losing hypothyroidism (loss of TBG) — check thyroid function and supplement",
            "NPHS1-TRANSPLANT-RECURRENCE-RARE: Unlike TRPC6/podocin circulating-factor FSGS, post-transplant recurrence in NPHS1-LOF is RARE — reassure families",
            "NPHS1-FIN-MAJOR-FIN-MINOR: In patients of Finnish descent test Fin-major (c.121delCT) and Fin-minor (c.3325C>T) first — covers 95% of Finnish alleles",
            "NPHS1-ANTENATAL-DIAGNOSIS: Elevated AFP in maternal serum (15-20 weeks) + prenatal ultrasound (enlarged echogenic kidneys) — genetic testing of amniotic fluid",
            "NPHS1-ALBUMIN-INFUSION: Regular albumin infusions (4-6x/week) mandatory pre-nephrectomy to sustain growth and prevent thromboembolism",
        ],
    },
    # ── NPHS2 — FSGS2 / Steroid-Resistant Nephrotic Syndrome ──
    {
        "gene": "NPHS2",
        "protein": "Podocin — FSGS2 Steroid-RESISTANT PATHOGNOMONIC, Low Recurrence Post-Transplant, R138Q Most Common European",
        "alias": (
            "NPHS2; OMIM gene 604766; FSGS2 OMIM 600995; 1q25.2; 383 aa; ~42 kDa; "
            "NPHS2 encodes podocin, a 383-amino acid stomatin-family membrane protein with a "
            "characteristic hairpin membrane topology (N- and C-terminus both intracellular, "
            "single membrane hair-pin loop). Podocin localises to the slit diaphragm where it "
            "recruits nephrin and stabilises the SD scaffold via lipid-raft membrane microdomains. "
            "Podocin forms homooligomers (multimers of 8–12 monomers) that create a protein "
            "scaffold to cluster nephrin at the SD. NPHS2 LOF: podocin absent → nephrin delocalised "
            "from SD → foot process effacement → massive proteinuria → FSGS on biopsy. "
            "Clinical: steroid-RESISTANT nephrotic syndrome (SRNS) in childhood (peak 1–6 years); "
            "pathognomonic that NPHS2-LOF SRNS is steroid-resistant from the outset. "
            "Most common pathogenic variant in European: R138Q (p.Arg138Gln) in exon 5; "
            "p.R229Q is a low-penetrance modifier (not fully pathogenic alone — acts in trans with "
            "R138Q to cause disease). Genotype-specific risk: truncating variants → earlier ESRD; "
            "missense variants → variable. Post-transplant recurrence: LOW in biallelic NPHS2 LOF "
            "(contrast: circulating-factor FSGS post-transplant recurrence is HIGH — different disease). "
            "Treatment: supportive (ACEi/ARB, diuretics); cyclosporin partial response in some "
            "missense variants; HSCT not indicated. Genetic testing prevents unnecessary "
            "immunosuppressive toxicity."
        ),
        "aa": "383 aa",
        "kDa": "~42 kDa",
        "locus": "1q25.2",
        "omim_gene": 604766,
        "omim_disease": 600995,
        "inheritance": "AR (biallelic); 1q25.2; R138Q most common European; p.R229Q low-penetrance modifier",
        "gene_class": (
            "NPHS2 (podocin) is a 383-amino acid stomatin/prohibitin homology (SPFH) domain protein "
            "with unique hairpin membrane topology. SPFH domain: forms oligomeric ring (8–12 mers) "
            "that compartmentalises nephrin into cholesterol-enriched lipid rafts at the slit "
            "diaphragm. Pathogenic variants: R138Q (exon 5, most common) disrupts oligomerisation; "
            "R138X (truncating, severe); R291W (exon 7, east European founder). p.R229Q: common "
            "low-penetrance variant (~3% European allele frequency) — disease-causing only in trans "
            "with severe pathogenic variant (e.g. R138Q). Functional classification: "
            "variants causing ER retention → no slit diaphragm targeting (most severe); "
            "variants causing SD mislocalisation (partial); variants reducing oligomerisation (mild)."
        ),
        "n_patients": 40,
        "key_alerts": [
            "NPHS2-STEROID-RESISTANT-PATHOGNOMONIC: NPHS2-LOF SRNS is steroid-resistant FROM THE OUTSET — do NOT pursue prolonged steroid trials that cause toxicity",
            "NPHS2-R138Q-MOST-COMMON-EUROPEAN: Test R138Q (exon 5) first in European SRNS — covers ~50% of European NPHS2 FSGS",
            "NPHS2-R229Q-NOT-STANDALONE-PATHOGENIC: p.R229Q alone (common variant) does NOT cause FSGS — only disease-causing when found IN TRANS with R138Q or other severe variant",
            "NPHS2-LOW-TRANSPLANT-RECURRENCE: Post-transplant recurrence of FSGS is LOW in biallelic NPHS2-LOF — distinguish from circulating-factor FSGS (high recurrence) to guide transplant decisions",
            "NPHS2-CYCLOSPORIN-PARTIAL: Cyclosporin (calcineurin inhibitor) may achieve partial remission in some missense variants via direct podocyte effect — trial warranted",
            "NPHS2-NO-IMMUNOSUPPRESSION-TOXICITY: Genetic SRNS is NOT immune-mediated — avoid prolonged cyclophosphamide, rituximab; genetic diagnosis prevents immunosuppressive toxicity",
            "NPHS2-BIALLELIC-CONFIRM: Both variants must be confirmed in trans (compound heterozygous) — parental testing or long-read sequencing mandatory",
            "NPHS2-CASCADE-TESTING: All siblings of index case require urinalysis and genetic testing — 25% recurrence risk in AR disease",
        ],
    },
    # ── WT1 — Denys-Drash / Frasier Syndrome ──
    {
        "gene": "WT1",
        "protein": "Wilms Tumour Protein 1 — DDS R394W Hotspot + Wilms Tumour, Frasier Gonadoblastoma MANDATORY Gonadectomy",
        "alias": (
            "WT1; OMIM gene 607102; Denys-Drash OMIM 194080; Frasier OMIM 136680; 11p13; 449 aa; ~52 kDa; "
            "WT1 encodes a zinc finger transcription factor with 4 C2H2 zinc fingers (ZF1–ZF4) and "
            "an N-terminal regulatory domain (proline/glutamine-rich, self-association). WT1 is "
            "essential for kidney and gonad development: regulates glomerular podocyte differentiation, "
            "nephron induction from metanephric mesenchyme, and gonadal determination. "
            "Two major isoforms generated by alternative splicing: ±KTS (lysine-threonine-serine "
            "tripeptide between ZF3 and ZF4) — +KTS: post-transcriptional RNA processing; "
            "-KTS: transcriptional regulation. Imbalance of +KTS/−KTS ratio → Frasier syndrome. "
            "Denys-Drash syndrome (DDS): heterozygous missense in ZF2/ZF3 (R394W most common hotspot, "
            "90% of DDS) → dominant negative → triad of (1) diffuse mesangial sclerosis (DMS) → "
            "nephrotic syndrome + ESRD usually <3 years; (2) disorders of sexual development (DSD) "
            "— 46XY: ambiguous/female genitalia; 46XX: occasionally streak ovaries; (3) Wilms tumour "
            "(nephroblastoma) — 10–20% (bilateral risk) — monitor until age 8 with 3-monthly renal US. "
            "Frasier syndrome: heterozygous splice variants affecting +KTS isoform ratio → "
            "FSGS (slower than DMS) + 46XY DSD (streak gonads) + gonadoblastoma 40–60% — "
            "NO Wilms tumour (important distinguishing feature). "
            "MANDATORY: 46XY patients with either syndrome → GONADECTOMY (streak gonads → "
            "gonadoblastoma risk 40–60%). Annual renal ultrasound in DDS until age 8 (Wilms). "
            "Genetic confirmation: sequencing identifies DDS/Frasier variant; R394W by targeted testing."
        ),
        "aa": "449 aa",
        "kDa": "~52 kDa",
        "locus": "11p13",
        "omim_gene": 607102,
        "omim_disease": 194080,
        "inheritance": "AD (heterozygous missense DDS; splice variant Frasier); 11p13; de novo and familial",
        "gene_class": (
            "WT1 is a 449-amino acid C2H2 zinc finger transcription factor. Domain architecture: "
            "N-terminal proline/glutamine-rich activation/repression domain (residues 1–180); "
            "4 zinc fingers (ZF1–ZF4, residues 323–449) — ZF2 and ZF3 are hotspots for DDS missense "
            "variants. ZF DNA binding: recognises EGR1-like GC-rich sequences (5'-GCGGG-3'). "
            "Alternative splicing: exon 5 insertion (+17 aa, KTS exclusion) modulates activation; "
            "intron 9 +KTS splice changes ZF3-ZF4 linker → alters RNA rather than DNA binding. "
            "Transcriptional targets in kidney: NPHS1, PODXL, SWT1, PAX2, IGF2 — all required "
            "for normal podocyte/nephron development. DDS missense R394W (ZF3): dominant negative "
            "→ mutant WT1 competes with wildtype for DNA binding at target gene promoters, "
            "reducing transcriptional activation. Frasier +KTS splice: imbalanced isoform ratio "
            "→ aberrant post-transcriptional regulation in developing gonad + glomerulus."
        ),
        "n_patients": 40,
        "key_alerts": [
            "WT1-GONADECTOMY-MANDATORY: 46XY patients with DDS or Frasier syndrome have 40-60% gonadoblastoma risk from streak gonads — gonadectomy MANDATORY, typically by age 1-2",
            "WT1-R394W-DDS-HOTSPOT: R394W accounts for ~90% of Denys-Drash syndrome — test targeted variant first in infant nephrotic syndrome + DSD",
            "WT1-WILMS-ANNUAL-US: Denys-Drash: annual renal ultrasound every 3 months until age 8 for Wilms tumour surveillance — bilateral nephrectomy after Wilms diagnosis risks ESRD",
            "WT1-FRASIER-NO-WILMS: Frasier syndrome does NOT have Wilms tumour risk — distinguishes Frasier from DDS clinically; splice variant +KTS is Frasier",
            "WT1-DMS-EARLY-ESRD: Diffuse mesangial sclerosis in DDS → nephrotic syndrome + ESRD typically by age 2-3 years — early transplant planning essential",
            "WT1-DSD-KARYOTYPE: ALL patients with WT1-related nephrotic syndrome require karyotype (46XY vs 46XX) to determine gonadoblastoma and gonadectomy need",
            "WT1-FRASIER-FSGS-SLOWER: Frasier FSGS progression slower than DDS DMS — ESRD in teens/early adulthood; earlier in males (46XY with FSGS)",
            "WT1-CASCADE-PARENT: WT1 DDS/Frasier can be de novo or AD — test parents; de novo (~50%) does not require parental gonadal surveillance",
        ],
    },
    # ── UMOD — Uromodulin Kidney Disease (FJHN / MCKD2) ──
    {
        "gene": "UMOD",
        "protein": "Uromodulin — FJHN/MCKD2 Hereditary Tubulointerstitial Nephritis, Gout PRESENTING FEATURE, Urine UMOD Assay",
        "alias": (
            "UMOD; OMIM gene 191845; FJHN OMIM 162000; MCKD2 OMIM 603860; 16p12.3; 640 aa; ~72 kDa; "
            "UMOD encodes uromodulin (Tamm-Horsfall protein), the most abundantly secreted urinary "
            "protein, produced exclusively by thick ascending limb (TAL) cells of Henle's loop. "
            "Uromodulin is a GPI-anchored glycoprotein with EGF-like domains and a unique zona "
            "pellucida (ZP) domain required for polymerisation into filaments in the tubular lumen. "
            "Functions: (1) protection against UTI (filaments entrap uropathogenic bacteria); "
            "(2) tubular water/salt transport regulation; (3) immunomodulation (TLR signalling). "
            "UMOD autosomal dominant mutations (most in EGF-like or ZP domain): misfolded protein "
            "retained in ER → ER stress (unfolded protein response) → CHOP-mediated apoptosis of "
            "TAL cells → tubulointerstitial nephritis → progressive CKD. "
            "FJHN / MCKD2 phenotype: (1) hyperuricaemia + early-onset gout (presenting feature in "
            "teens or early adulthood — before CKD apparent); (2) slowly progressive tubulointerstitial "
            "nephritis → CKD (ESRD in 3rd–7th decade depending on variant); (3) medullary cysts "
            "(not always visible on US/MRI — diagnosis is genetic + clinical, not imaging). "
            "Urine UMOD assay: < 5th centile (< 15 mg/g creatinine) — reduced secretion of "
            "misfolded UMOD into urine; diagnostic in the right clinical context. "
            "Management: allopurinol for hyperuricaemia (avoid uricosuric drugs — increase tubular "
            "uric acid load → precipitate in tubules); colchicine for acute gout (NSAIDs "
            "nephrotoxic — AVOID); ACEi/ARB to slow CKD; renal transplant at ESRD (recurrence rare)."
        ),
        "aa": "640 aa",
        "kDa": "~72 kDa",
        "locus": "16p12.3",
        "omim_gene": 191845,
        "omim_disease": 162000,
        "inheritance": "AD; 16p12.3; missense variants in EGF-like/ZP domain → ER retention; autosomal dominant with high penetrance",
        "gene_class": (
            "UMOD (uromodulin) is a 640-amino acid GPI-anchored TAL glycoprotein. Domain architecture: "
            "signal peptide → EGF-like repeat 1 (residues 29–65) → EGF-like repeat 2 → EGF-like "
            "repeat 3 → D8C domain → ZP domain (residues 444–614, required for ZP filament "
            "polymerisation in tubular lumen) → GPI attachment site. Hotspot pathogenic variants: "
            "cysteine substitutions in EGF-like domains (disrupt disulfide bonds → misfolding) and "
            "cysteine-introducing missense in ZP domain → ER retention. ER-retained mutant UMOD: "
            "activates IRE1-XBP1, ATF6, PERK pathways → chronic ER stress → CHOP-mediated TAL "
            "cell apoptosis → eventual tubular atrophy + interstitial fibrosis."
        ),
        "n_patients": 40,
        "key_alerts": [
            "UMOD-GOUT-PRESENTING-FEATURE: Early-onset gout (teenage/early adult) is the PRESENTING FEATURE of UMOD disease — always test UMOD in gout onset <40 years with family history CKD",
            "UMOD-NSAIDS-NEPHROTOXIC-AVOID: NSAIDs are ABSOLUTELY CONTRAINDICATED for gout attacks in UMOD disease — use colchicine or short-course corticosteroids instead",
            "UMOD-URINE-UMOD-ASSAY: Urine uromodulin <5th centile (ELISA, <15 mg/g creatinine) — diagnostic in correct clinical context; available in specialist labs",
            "UMOD-ALLOPURINOL-NOT-URICOSURICS: Allopurinol (xanthine oxidase inhibitor) preferred for hyperuricaemia — uricosuric agents increase tubular uric acid load and risk tubular precipitation",
            "UMOD-MEDULLARY-CYSTS-ABSENT-OK: Medullary cysts may NOT be visible on imaging — diagnosis is CLINICAL (gout + CKD + family history) + GENETIC, not imaging-dependent",
            "UMOD-CKD-SLOW-VARIABLE: ESRD in 3rd-7th decade depending on variant and lifestyle — manage CKD aggressively (BP <130/80, ACEi/ARB, low protein diet)",
            "UMOD-TRANSPLANT-NO-RECURRENCE: Renal transplant effective — UMOD disease does NOT recur in allograft (recipient's TAL cells no longer express mutant UMOD)",
            "UMOD-CASCADE-AD: AD inheritance — 50% of children at risk; offer testing to at-risk family members; gout in any family member with CKD",
        ],
    },
    # ── PKD1 — Autosomal Dominant Polycystic Kidney Disease type 1 ──
    {
        "gene": "PKD1",
        "protein": "Polycystin-1 — ADPKD Most Common Hereditary Kidney Disease, Tolvaptan FDA 2018, Intracranial Aneurysm MRA Mandatory",
        "alias": (
            "PKD1; OMIM gene 601313; ADPKD OMIM 173900; 16p13.3; 4303 aa; ~462 kDa; "
            "PKD1 encodes polycystin-1 (PC1), the largest human single-pass transmembrane protein, "
            "localised at the primary cilium (PC1/PC2 complex), epithelial cell-cell junctions, "
            "focal adhesions, and desmosome-like structures. PC1 functions as a mechanosensor: "
            "bends in response to tubular flow → PC1 cleaves its own C-terminal tail (CTT) → "
            "CTT translocates to nucleus → activates mTOR + Ca2+ signalling via PC2 (TRPP2). "
            "ADPKD type 1 (PKD1): accounts for 85% of ADPKD; 15% is PKD2 (TRPP2, 4q22). "
            "Most common hereditary kidney disease: 1 in 400–1000; total 12.5 million worldwide. "
            "Clinical: bilateral progressive renal cyst enlargement + flank pain + haematuria + "
            "hypertension (RAAS activation from stretched vessels) + hepatic cysts (60–80%) + "
            "intracranial aneurysm (ICA, 5–10%) + cardiac valve abnormalities (MVP ~25%). "
            "Tolvaptan (Jynarque): FDA 2018 — vasopressin V2R antagonist → reduces cAMP in "
            "tubular epithelium → slows kidney enlargement (TKV criteria: Mayo Class 1C-1E "
            "or annual TKV growth >5%). Monitor LFTs (rare but serious hepatotoxicity). "
            "ICA screening: MRA brain MANDATORY for all PKD1 patients with personal OR family "
            "history of ICA/rupture; or before major elective surgery; or specific occupation risk. "
            "Hypotension/hypertension: BP target <130/80 (early treatment delays ESRD). "
            "Genetic counselling: 50% risk per child; de novo in 5–10%; somatic mosaicism possible."
        ),
        "aa": "4303 aa",
        "kDa": "~462 kDa",
        "locus": "16p13.3",
        "omim_gene": 601313,
        "omim_disease": 173900,
        "inheritance": "AD; 16p13.3; ~85% of ADPKD; 1 in 400-1000 worldwide; de novo 5-10%",
        "gene_class": (
            "PKD1 (polycystin-1) is a 4303-amino acid multi-domain receptor-like protein. "
            "Extracellular: 11 immunoglobulin-like PKD repeats → cell-cell/matrix adhesion; "
            "REJ (receptor for egg jelly) domain; GPS (GPCR-proteolysis site) autocleavage site. "
            "Transmembrane: 11 TM domains (TRPL channel-like topology). Intracellular C-terminus: "
            "G-protein coupling; coiled-coil for PC2 (TRPP2/PKD2) interaction; "
            "CTT cleaved and translocates to nucleus (mTOR1/STAT3/AP1 targets). "
            "Two-hit model: heterozygous germline + somatic second hit in individual cyst-initiating "
            "cells (explains late onset and focal nature of cysts). Biallelic: embryo lethal. "
            "Hypomorphic alleles: truncating + missense → phenotype modifier (some truncating = severe, "
            "some missense = mild). PKD1 vs PKD2: PKD1 more severe (ESRD ~10y earlier)."
        ),
        "n_patients": 40,
        "key_alerts": [
            "PKD1-TOLVAPTAN-FDA-2018: Tolvaptan indicated for rapidly progressing ADPKD (Mayo Class 1C-1E or TKV growth >5%/year) — slows kidney enlargement; monitor LFTs monthly x6m then q3m",
            "PKD1-ICA-MRA-MANDATORY: Intracranial aneurysm MRA brain screening MANDATORY if personal or family history of ICA rupture; 5-10% prevalence in ADPKD",
            "PKD1-BP-TARGET-EARLY: BP <130/80 early treatment with ACEi/ARB delays ESRD — start at hypertension diagnosis even in young patients; avoid volume depletion",
            "PKD1-HEPATIC-CYSTS-60-80pct: Hepatic cysts in 60-80% — symptomatic liver involvement (massive hepatomegaly) more common in women; somatostatin analogues (lanreotide) may slow liver growth",
            "PKD1-TOLVAPTAN-HEPATOTOXICITY: Rare but serious hepatotoxicity with tolvaptan — monthly LFTs for first 6 months; discontinue if ALT/AST >3x ULN with symptoms",
            "PKD1-MVP-AORTIC-SCREEN: MVP in ~25% and aortic root dilation in ~5% — baseline cardiac echo at diagnosis; annual ECG in MVP patients",
            "PKD1-FLANK-PAIN-HAEMATURIA: Gross haematuria = cyst rupture/haemorrhage — bed rest, hydration, analgesia; anticoagulants increase cyst haemorrhage risk",
            "PKD1-CASCADE-TESTING: Ultrasound screening of first-degree relatives (>16y: ≥2 cysts each kidney for PKD1) — Pei criteria for imaging diagnosis",
        ],
    },
    # ── TRPC6 — FSGS6 (Gain-of-Function) ──
    {
        "gene": "TRPC6",
        "protein": "TRPC6 — FSGS6 AD GOF Adult-Onset, Steroid-RESISTANT, Living-Donor EXCLUSION, Calcineurin Inhibitors Partial",
        "alias": (
            "TRPC6; OMIM gene 603652; FSGS6 OMIM 614131; 11q22.1; 931 aa; ~104 kDa; "
            "TRPC6 encodes Transient Receptor Potential Cation Channel subfamily C member 6, a "
            "non-selective cation channel (Na+, Ca2+, K+) expressed in podocytes and other cell "
            "types. In podocytes, TRPC6 is concentrated at the slit diaphragm where it forms "
            "calcium-permeable channels regulated by podocin (NPHS2), nephrin, and mechanical "
            "stretch. Physiological role: TRPC6-mediated Ca2+ influx regulates podocyte actin "
            "cytoskeleton dynamics via calcineurin-NFAT signalling. "
            "FSGS6 is caused by autosomal dominant GAIN-OF-FUNCTION variants → enhanced channel "
            "activity → excess podocyte Ca2+ influx → calcineurin activation → NFAT-dependent "
            "pro-apoptotic gene expression → podocyte loss → FSGS. "
            "Most common GOF variants: P112Q, N143S (N-terminal gain-of-function); R895C (C-terminal). "
            "Clinical: adult-onset proteinuria (3rd–5th decade) → FSGS on biopsy → ESRD over 5–20 years. "
            "Steroid-resistant (typical of genetic FSGS). "
            "LIVING-RELATED DONOR EXCLUSION: carrier relatives should NOT donate a kidney — they have "
            "50% risk of eventual FSGS and donor nephrectomy could accelerate their own kidney disease. "
            "Calcineurin inhibitors (cyclosporin, tacrolimus) may achieve partial remission by "
            "reducing Ca2+/calcineurin signalling. ESRD → transplant (disease may not recur in allograft "
            "if podocyte-autonomous GOF — recipient's podocytes are affected, not donor's). "
            "Genetic testing in FSGS: TRPC6 accounts for ~1–2% of familial FSGS cases."
        ),
        "aa": "931 aa",
        "kDa": "~104 kDa",
        "locus": "11q22.1",
        "omim_gene": 603652,
        "omim_disease": 614131,
        "inheritance": "AD (gain-of-function); 11q22.1; adult-onset FSGS; P112Q/N143S/R895C most common GOF variants",
        "gene_class": (
            "TRPC6 (Transient Receptor Potential Canonical channel 6) is a 931-amino acid "
            "6-TM non-selective cation channel. Domain architecture: N-terminal ankyrin repeat "
            "domain (ARD, 4 repeats) → linker → 6 TM helices (S1–S6) with TRP box → C-terminal "
            "coiled-coil. Channel properties: non-selective cation (PCa/PNa ~5); activated by "
            "DAG (diacylglycerol), stretch, receptor-operated Ca2+ entry pathways. Tetrameric "
            "assembly: TRPC6 homotetramers; heteromers with TRPC1/TRPC3 possible. Podocyte "
            "signalling: Ca2+ entry → calcineurin (CN) → NFAT dephosphorylation → nuclear "
            "translocation → pro-apoptotic genes (TRPC6 itself, synaptopodin degradation targets). "
            "GOF mechanism: N-terminal ARD variants (P112Q, N143S) → reduced ARD-mediated "
            "autoinhibition → increased basal open probability. Pharmacology: calcineurin inhibitors "
            "(CsA, tacrolimus) reduce downstream NFAT signalling; sparsentan (ENDO antagonist + "
            "ARB) in trials for FSGS; specific TRPC6 blockers (BI 749327) in development."
        ),
        "n_patients": 40,
        "key_alerts": [
            "TRPC6-LIVING-DONOR-EXCLUSION: Carrier relatives MUST NOT donate a kidney — 50% risk of eventual FSGS; donor nephrectomy risks accelerating their own kidney disease",
            "TRPC6-ADULT-ONSET: TRPC6 FSGS presents in 3rd-5th decade — consider in adult familial FSGS with steroid-resistant proteinuria and AD family history",
            "TRPC6-STEROID-RESISTANT: TRPC6 GOF FSGS is steroid-resistant — do NOT pursue prolonged steroid trials; proceed to genetic testing in resistant cases",
            "TRPC6-CALCINEURIN-INHIBITORS: Cyclosporin or tacrolimus may achieve partial remission by reducing Ca2+/calcineurin signalling downstream of channel GOF",
            "TRPC6-GOF-NOT-LOF: TRPC6 variants causing FSGS6 are GAIN-OF-FUNCTION — gene therapy/antisense oligonucleotide approaches target reduction of channel activity",
            "TRPC6-P112Q-N143S-R895C: Three most common GOF variants — targeted sequencing; confirm GOF status by functional assay or established variant database before clinical decision",
            "TRPC6-TRANSPLANT-RECURRENCE-LOW: Disease is podocyte-intrinsic GOF — recipient's podocytes re-express mutant TRPC6 post-transplant; allograft podocytes are healthy",
            "TRPC6-CASCADE-TESTING: All first-degree relatives require urinalysis + eGFR annually; offer TRPC6 genetic testing to at-risk relatives before they become living donors",
        ],
    },
]


def _make_cohort(gene_data):
    rng = random.Random(SEED_BASE + NEPHROPATHY_GENES.index(gene_data))
    gene = gene_data["gene"]
    pts = []
    for i in range(gene_data["n_patients"]):
        if gene == "COL4A5":
            age_dx = rng.gauss(8, 6)
            delay = rng.gauss(24, 18)
            sex = rng.choice(["M", "F", "F"])
            acei = rng.random() < 0.78
            esrd = (sex == "M") and rng.random() < 0.35
            snhl = (sex == "M") and rng.random() < 0.70 or (sex == "F") and rng.random() < 0.30
            p = {
                "id": f"COL4A5-{i+1:03d}",
                "gene": "COL4A5",
                "sex": sex,
                "etiology": rng.choice(["Gly325Ser", "Gly521Val", "c.3781+2T>C", "Gly624Asp", "large_del_MLPA"]),
                "age_at_diagnosis": max(0.5, round(age_dx, 1)),
                "dx_delay_months": max(2, round(delay, 0)),
                "acei": acei,
                "esrd": esrd,
                "sensorineural_hearing_loss": snhl,
                "anterior_lenticonus": (sex == "M") and rng.random() < 0.28,
                "haematuria": rng.random() < 0.99,
                "proteinuria": rng.random() < 0.75,
                "skin_biopsy_done": rng.random() < 0.45,
                "mlpa_done": rng.random() < 0.40,
                "cascade_tested": rng.random() < 0.65,
                "nephrotoxin_exposure": rng.random() < 0.18,
            }
        elif gene == "COL4A3":
            age_dx = rng.gauss(15, 12)
            delay = rng.gauss(36, 24)
            biallelic = rng.random() < 0.25
            acei = rng.random() < 0.55
            p = {
                "id": f"COL4A3-{i+1:03d}",
                "gene": "COL4A3",
                "etiology": rng.choice(["Gly214Ser", "c.4777+1G>A", "Arg1347Ter", "Gly421Val", "large_del_MLPA"]),
                "biallelic": biallelic,
                "age_at_diagnosis": max(1, round(age_dx, 1)),
                "dx_delay_months": max(3, round(delay, 0)),
                "acei": acei,
                "haematuria": rng.random() < 0.98,
                "proteinuria": rng.random() < (0.90 if biallelic else 0.30),
                "fsgs_on_biopsy": (not biallelic) and rng.random() < 0.15,
                "esrd": biallelic and rng.random() < 0.40 or (not biallelic) and rng.random() < 0.08,
                "snhl": biallelic and rng.random() < 0.65,
                "em_thin_gbm": (not biallelic) and rng.random() < 0.90,
                "mlpa_done": rng.random() < 0.38,
                "partner_tested": rng.random() < 0.42,
                "cascade_tested": rng.random() < 0.60,
            }
        elif gene == "NPHS1":
            age_dx = rng.gauss(0.05, 0.03)
            delay = rng.gauss(1.5, 1.0)
            p = {
                "id": f"NPHS1-{i+1:03d}",
                "gene": "NPHS1",
                "etiology": rng.choice(["Fin-major/Fin-major", "Fin-major/Fin-minor", "Fin-major/missense", "compound_het_non-Finnish"]),
                "age_at_diagnosis": max(0.01, round(age_dx, 2)),
                "dx_delay_months": max(0.2, round(delay, 1)),
                "large_placenta": rng.random() < 0.92,
                "bilateral_nephrectomy": rng.random() < 0.88,
                "renal_transplant": rng.random() < 0.72,
                "transplant_recurrence": rng.random() < 0.04,
                "thyroid_supplementation": rng.random() < 0.80,
                "albumin_infusions": rng.random() < 0.95,
                "immunosuppression_attempted": rng.random() < 0.30,
                "initial_sepsis_diagnosis": rng.random() < 0.20,
                "antenatal_diagnosis": rng.random() < 0.35,
                "cascade_tested": rng.random() < 0.88,
            }
        elif gene == "NPHS2":
            age_dx = rng.gauss(4, 3)
            delay = rng.gauss(18, 12)
            steroid_resistant = rng.random() < 0.97
            p = {
                "id": f"NPHS2-{i+1:03d}",
                "gene": "NPHS2",
                "etiology": rng.choice(["R138Q/R138Q", "R138Q/R138X", "R138Q/R291W", "R138Q/R229Q", "other_compound_het"]),
                "age_at_diagnosis": max(0.5, round(age_dx, 1)),
                "dx_delay_months": max(3, round(delay, 0)),
                "steroid_resistant": steroid_resistant,
                "cyclosporin_partial_response": rng.random() < 0.22,
                "transplant_done": rng.random() < 0.40,
                "transplant_recurrence": rng.random() < 0.05,
                "fsgs_on_biopsy": rng.random() < 0.90,
                "esrd": rng.random() < 0.30,
                "prolonged_steroid_toxicity": rng.random() < 0.25,
                "biallelic_confirmed_trans": rng.random() < 0.75,
                "cascade_tested": rng.random() < 0.70,
            }
        elif gene == "WT1":
            age_dx = rng.gauss(1.5, 1.5)
            delay = rng.gauss(6, 4)
            syndrome = rng.choice(["DDS", "DDS", "DDS", "Frasier"])
            karyotype = rng.choice(["46XY", "46XY", "46XX"])
            gonadectomy = (karyotype == "46XY") and rng.random() < 0.88
            p = {
                "id": f"WT1-{i+1:03d}",
                "gene": "WT1",
                "syndrome": syndrome,
                "karyotype": karyotype,
                "etiology": rng.choice(["R394W", "R366H", "D396N", "+KTS_splice_IVS9"]) if syndrome == "DDS" else "+KTS_splice_IVS9",
                "age_at_diagnosis": max(0.1, round(age_dx, 1)),
                "dx_delay_months": max(1, round(delay, 0)),
                "dms_on_biopsy": syndrome == "DDS" and rng.random() < 0.90,
                "wilms_tumour": syndrome == "DDS" and rng.random() < 0.18,
                "gonadoblastoma": syndrome == "Frasier" and karyotype == "46XY" and rng.random() < 0.50,
                "dsd": karyotype == "46XY" and rng.random() < 0.85,
                "gonadectomy": gonadectomy,
                "esrd": rng.random() < 0.55,
                "renal_ultrasound_surveillance": syndrome == "DDS" and rng.random() < 0.80,
                "cascade_tested": rng.random() < 0.72,
            }
        elif gene == "UMOD":
            age_dx = rng.gauss(32, 12)
            delay = rng.gauss(60, 36)
            gout = rng.random() < 0.82
            p = {
                "id": f"UMOD-{i+1:03d}",
                "gene": "UMOD",
                "etiology": rng.choice(["Cys132Arg", "Cys217Ser", "Cys300Ser", "Thr62Met", "His177Pro"]),
                "age_at_diagnosis": max(15, round(age_dx, 1)),
                "dx_delay_months": max(12, round(delay, 0)),
                "gout": gout,
                "gout_onset_age": max(14, round(rng.gauss(28, 8), 0)) if gout else None,
                "hyperuricaemia": rng.random() < 0.90,
                "allopurinol": rng.random() < 0.75,
                "nsaid_used_for_gout": rng.random() < 0.30,
                "urine_umod_low": rng.random() < 0.85,
                "medullary_cysts_visible": rng.random() < 0.45,
                "ckd_stage": rng.choice(["G1", "G2", "G3a", "G3b", "G4", "G5"]),
                "esrd": rng.random() < 0.28,
                "cascade_tested": rng.random() < 0.60,
            }
        elif gene == "PKD1":
            age_dx = rng.gauss(35, 10)
            delay = rng.gauss(12, 8)
            tolvaptan = rng.random() < 0.38
            p = {
                "id": f"PKD1-{i+1:03d}",
                "gene": "PKD1",
                "etiology": rng.choice(["truncating", "missense_mild", "missense_moderate", "splice_variant", "large_del_NMD"]),
                "age_at_diagnosis": max(10, round(age_dx, 1)),
                "dx_delay_months": max(1, round(delay, 0)),
                "tolvaptan": tolvaptan,
                "tolvaptan_lft_monitoring": tolvaptan and rng.random() < 0.88,
                "ica_mra_done": rng.random() < 0.55,
                "ica_found": rng.random() < 0.08,
                "hepatic_cysts": rng.random() < 0.72,
                "hypertension": rng.random() < 0.80,
                "haematuria_episode": rng.random() < 0.55,
                "mvp": rng.random() < 0.25,
                "esrd": rng.random() < 0.30,
                "acei_arb": rng.random() < 0.82,
                "cascade_tested": rng.random() < 0.75,
            }
        else:  # TRPC6
            age_dx = rng.gauss(38, 10)
            delay = rng.gauss(24, 16)
            calcineurin_inhibitor = rng.random() < 0.45
            p = {
                "id": f"TRPC6-{i+1:03d}",
                "gene": "TRPC6",
                "etiology": rng.choice(["P112Q", "N143S", "R895C", "R895L", "S270T"]),
                "age_at_diagnosis": max(20, round(age_dx, 1)),
                "dx_delay_months": max(6, round(delay, 0)),
                "steroid_resistant": rng.random() < 0.95,
                "calcineurin_inhibitor": calcineurin_inhibitor,
                "calcineurin_partial_response": calcineurin_inhibitor and rng.random() < 0.35,
                "transplant_done": rng.random() < 0.28,
                "esrd": rng.random() < 0.35,
                "family_member_excluded_from_donation": rng.random() < 0.42,
                "fsgs_on_biopsy": rng.random() < 0.95,
                "proteinuria_g_per_day": round(rng.gauss(4.5, 2.5), 1),
                "prolonged_steroid_toxicity": rng.random() < 0.22,
                "cascade_tested": rng.random() < 0.65,
            }
        pts.append(p)
    return pts


def _pct(pts, key):
    if not pts:
        return 0.0
    return round(100.0 * sum(1 for p in pts if p.get(key)) / len(pts), 1)


def get_overview():
    all_pts = []
    gene_pts = {}
    for gd in NEPHROPATHY_GENES:
        pts = _make_cohort(gd)
        gene_pts[gd["gene"]] = pts
        all_pts.extend(pts)

    col4a5 = gene_pts["COL4A5"]
    col4a3 = gene_pts["COL4A3"]
    nphs1 = gene_pts["NPHS1"]
    nphs2 = gene_pts["NPHS2"]
    wt1 = gene_pts["WT1"]
    umod = gene_pts["UMOD"]
    pkd1 = gene_pts["PKD1"]
    trpc6 = gene_pts["TRPC6"]

    s = {
        "total_patients": len(all_pts),
        "mean_dx_age_years": round(sum(p["age_at_diagnosis"] for p in all_pts) / len(all_pts), 1),
        "mean_dx_delay_months": round(sum(p["dx_delay_months"] for p in all_pts) / len(all_pts), 0),
        "cascade_tested_pct": _pct(all_pts, "cascade_tested"),
        # COL4A5 Alport XL
        "col4a5_acei_pct": _pct(col4a5, "acei"),
        "col4a5_snhl_pct": _pct(col4a5, "sensorineural_hearing_loss"),
        "col4a5_anterior_lenticonus_pct": _pct(col4a5, "anterior_lenticonus"),
        "col4a5_esrd_pct": _pct(col4a5, "esrd"),
        "col4a5_mlpa_done_pct": _pct(col4a5, "mlpa_done"),
        # COL4A3 Alport AR / TBMN
        "col4a3_biallelic_pct": _pct(col4a3, "biallelic"),
        "col4a3_proteinuria_pct": _pct(col4a3, "proteinuria"),
        "col4a3_fsgs_on_biopsy_pct": _pct(col4a3, "fsgs_on_biopsy"),
        "col4a3_acei_pct": _pct(col4a3, "acei"),
        # NPHS1 CNS-F
        "nphs1_large_placenta_pct": _pct(nphs1, "large_placenta"),
        "nphs1_bilateral_nephrectomy_pct": _pct(nphs1, "bilateral_nephrectomy"),
        "nphs1_transplant_recurrence_pct": _pct(nphs1, "transplant_recurrence"),
        "nphs1_immunosuppression_attempted_pct": _pct(nphs1, "immunosuppression_attempted"),
        # NPHS2 FSGS2
        "nphs2_steroid_resistant_pct": _pct(nphs2, "steroid_resistant"),
        "nphs2_cyclosporin_partial_pct": _pct(nphs2, "cyclosporin_partial_response"),
        "nphs2_prolonged_steroid_toxicity_pct": _pct(nphs2, "prolonged_steroid_toxicity"),
        # WT1
        "wt1_gonadectomy_pct": _pct(wt1, "gonadectomy"),
        "wt1_wilms_pct": _pct(wt1, "wilms_tumour"),
        "wt1_dsd_pct": _pct(wt1, "dsd"),
        # UMOD
        "umod_gout_pct": _pct(umod, "gout"),
        "umod_nsaid_used_pct": _pct(umod, "nsaid_used_for_gout"),
        "umod_urine_umod_low_pct": _pct(umod, "urine_umod_low"),
        # PKD1
        "pkd1_tolvaptan_pct": _pct(pkd1, "tolvaptan"),
        "pkd1_ica_mra_done_pct": _pct(pkd1, "ica_mra_done"),
        "pkd1_ica_found_pct": _pct(pkd1, "ica_found"),
        "pkd1_hepatic_cysts_pct": _pct(pkd1, "hepatic_cysts"),
        # TRPC6
        "trpc6_steroid_resistant_pct": _pct(trpc6, "steroid_resistant"),
        "trpc6_donor_excluded_pct": _pct(trpc6, "family_member_excluded_from_donation"),
        "trpc6_calcineurin_pct": _pct(trpc6, "calcineurin_inhibitor"),
    }

    genes_out = []
    for gd in NEPHROPATHY_GENES:
        pts = gene_pts[gd["gene"]]
        genes_out.append({
            "gene": gd["gene"],
            "protein": gd["protein"],
            "aa": gd["aa"],
            "kDa": gd["kDa"],
            "locus": gd["locus"],
            "omim_gene": gd["omim_gene"],
            "omim_disease": gd["omim_disease"],
            "inheritance": gd["inheritance"],
            "n_patients": gd["n_patients"],
            "key_alerts": gd["key_alerts"],
            "mean_dx_age": round(sum(p["age_at_diagnosis"] for p in pts) / len(pts), 1),
            "mean_dx_delay_months": round(sum(p["dx_delay_months"] for p in pts) / len(pts), 0),
        })

    top_alerts = []
    for gd in NEPHROPATHY_GENES:
        for alert in gd["key_alerts"][:2]:
            top_alerts.append({"gene": gd["gene"], "alert": alert})

    return {
        "dashboard": "Hereditary Nephropathy Atlas",
        "subtitle": "Complete 8-Gene Hereditary Nephropathy Reference — COL4A5/COL4A3/NPHS1/NPHS2/WT1/UMOD/PKD1/TRPC6",
        "seeds": list(range(SEED_BASE, SEED_BASE + 8)),
        "aggregate_stats": s,
        "top_alerts": top_alerts,
        "genes": genes_out,
    }


def get_breakdown():
    out = {}
    for gd in NEPHROPATHY_GENES:
        pts = _make_cohort(gd)
        gene = gd["gene"]

        etiol_counts = {}
        for p in pts:
            etiol_counts[p["etiology"]] = etiol_counts.get(p["etiology"], 0) + 1

        age_buckets = {"<2": 0, "2–10": 0, "11–20": 0, "21–40": 0, "41–60": 0, ">60": 0}
        for p in pts:
            a = p["age_at_diagnosis"]
            if a < 2:
                age_buckets["<2"] += 1
            elif a < 11:
                age_buckets["2–10"] += 1
            elif a < 21:
                age_buckets["11–20"] += 1
            elif a < 41:
                age_buckets["21–40"] += 1
            elif a < 61:
                age_buckets["41–60"] += 1
            else:
                age_buckets[">60"] += 1

        delay_buckets = {"<6m": 0, "6–12m": 0, "1–2y": 0, "2–5y": 0, ">5y": 0}
        for p in pts:
            d = p["dx_delay_months"]
            if d < 6:
                delay_buckets["<6m"] += 1
            elif d < 12:
                delay_buckets["6–12m"] += 1
            elif d < 24:
                delay_buckets["1–2y"] += 1
            elif d < 60:
                delay_buckets["2–5y"] += 1
            else:
                delay_buckets[">5y"] += 1

        stat_keys = ["cascade_tested"]
        if gene == "COL4A5":
            stat_keys += ["acei", "esrd", "sensorineural_hearing_loss", "anterior_lenticonus",
                          "haematuria", "proteinuria", "skin_biopsy_done", "mlpa_done", "nephrotoxin_exposure"]
        elif gene == "COL4A3":
            stat_keys += ["biallelic", "acei", "haematuria", "proteinuria", "fsgs_on_biopsy",
                          "esrd", "snhl", "em_thin_gbm", "mlpa_done", "partner_tested"]
        elif gene == "NPHS1":
            stat_keys += ["large_placenta", "bilateral_nephrectomy", "renal_transplant",
                          "transplant_recurrence", "thyroid_supplementation", "albumin_infusions",
                          "immunosuppression_attempted", "initial_sepsis_diagnosis", "antenatal_diagnosis"]
        elif gene == "NPHS2":
            stat_keys += ["steroid_resistant", "cyclosporin_partial_response", "transplant_done",
                          "transplant_recurrence", "fsgs_on_biopsy", "esrd",
                          "prolonged_steroid_toxicity", "biallelic_confirmed_trans"]
        elif gene == "WT1":
            stat_keys += ["dms_on_biopsy", "wilms_tumour", "gonadoblastoma", "dsd",
                          "gonadectomy", "esrd", "renal_ultrasound_surveillance"]
        elif gene == "UMOD":
            stat_keys += ["gout", "hyperuricaemia", "allopurinol", "nsaid_used_for_gout",
                          "urine_umod_low", "medullary_cysts_visible", "esrd"]
        elif gene == "PKD1":
            stat_keys += ["tolvaptan", "tolvaptan_lft_monitoring", "ica_mra_done", "ica_found",
                          "hepatic_cysts", "hypertension", "haematuria_episode", "mvp", "esrd", "acei_arb"]
        else:  # TRPC6
            stat_keys += ["steroid_resistant", "calcineurin_inhibitor", "calcineurin_partial_response",
                          "transplant_done", "esrd", "family_member_excluded_from_donation",
                          "fsgs_on_biopsy", "prolonged_steroid_toxicity"]

        stats = {k: _pct(pts, k) for k in stat_keys}
        stats["mean_dx_age"] = round(sum(p["age_at_diagnosis"] for p in pts) / len(pts), 1)
        stats["mean_dx_delay_months"] = round(sum(p["dx_delay_months"] for p in pts) / len(pts), 0)

        out[gene] = {
            "gene": gene,
            "protein": gd["protein"],
            "alias": gd["alias"],
            "gene_class": gd["gene_class"],
            "n_patients": len(pts),
            "etiologies": etiol_counts,
            "age_at_diagnosis_distribution": age_buckets,
            "dx_delay_distribution": delay_buckets,
            "stats": stats,
            "key_alerts": gd["key_alerts"],
            "patients": pts[:10],
        }
    return out


def get_definitions():
    return {
        "atlas": "Hereditary Nephropathy Atlas — Complete 8-Gene Reference",
        "genes_covered": [gd["gene"] for gd in NEPHROPATHY_GENES],
        "concepts": {
            "Alport_Syndrome": (
                "Hereditary nephritis caused by pathogenic variants in COL4A5 (X-linked, 85%), "
                "COL4A3 or COL4A4 (autosomal, 15%). Pathognomonic triad: persistent microscopic "
                "haematuria + sensorineural hearing loss + ocular abnormalities (anterior lenticonus, "
                "macular flecks). GBM ultrastructure: irregular thickening, thinning, splitting and "
                "lamellation ('basket-weave') on EM — diagnostic gold standard. Immunofluorescence: "
                "absent COL4A5 (skin biopsy) or absent alpha3/alpha4/alpha5 network (renal biopsy). "
                "Treatment: ACEi ASAP (Gross O JASN 2012 — delays ESRD by 13 years on average). "
                "Male XL Alport: ESRD by 20–30y without treatment; 40–50y with ACEi. "
                "Females: range microhaematuria to full Alport. AR Alport (biallelic COL4A3/A4): "
                "same severity as male XL Alport. MLPA mandatory: 10% large deletions missed by WES."
            ),
            "Podocyte_Slit_Diaphragm": (
                "The glomerular slit diaphragm (SD) is a thin extracellular matrix bridge (~30 nm) "
                "spanning the filtration slits between adjacent podocyte foot processes. Principal "
                "structural components: nephrin (NPHS1) — IgG superfamily zipper-like homodimer; "
                "podocin (NPHS2) — stomatin/SPFH domain, organises nephrin in cholesterol-rich "
                "lipid rafts; neph1 (KIRREL1) — additional IgG domain protein; CD2AP — cytoplasmic "
                "scaffold linking SD to actin. The SD acts as the final size-selective filter for "
                "albumin (67 kDa, charge-selective retention). Loss of SD integrity → massive "
                "proteinuria → nephrotic syndrome. Mutations: NPHS1 → congenital NS (most severe); "
                "NPHS2 → FSGS (childhood steroid-resistant NS); TRPC6 → FSGS (adult-onset, GOF). "
                "Therapeutic implications: genetic diagnosis avoids ineffective immunosuppression."
            ),
            "Steroid_Resistant_Nephrotic_Syndrome_Genetics": (
                "Up to 30% of childhood steroid-resistant nephrotic syndrome (SRNS) has a "
                "monogenic cause. Key genes: NPHS2 (podocin, most common AR SRNS), NPHS1 (nephrin, "
                "congenital NS), WT1 (DDS/Frasier), TRPC6 (adult AD FSGS), INF2 (adult AD FSGS, "
                "CMT overlap), PLCE1 (diffuse mesangial sclerosis), LAMB2 (Pierson syndrome). "
                "Genetic SRNS is NOT immune-mediated → steroids and immunosuppression are "
                "INEFFECTIVE and HARMFUL. Genetic testing indication: all children with SRNS before "
                "prolonged immunosuppression; all adults with FSGS and family history. "
                "Low post-transplant recurrence in genetic SRNS (vs. high in circulating-factor FSGS). "
                "Calcineurin inhibitors (CsA, tacrolimus) partially effective in some missense "
                "NPHS2 and TRPC6 variants via direct podocyte/channel effect."
            ),
            "ADPKD_Management": (
                "Autosomal dominant polycystic kidney disease (PKD1 85%, PKD2 15%) is the most "
                "common hereditary kidney disease. Key management pillars: (1) BP control <130/80 "
                "(HALT-PKD trial — ACEi superior to ACEi + ARB combination); (2) Hydration: "
                "2–3 L fluid/day suppresses vasopressin-mediated cAMP cyst growth; (3) Tolvaptan "
                "(Jynarque, FDA 2018): V2R antagonist → reduces urine osmolality → slows TKV "
                "growth; indicated for Mayo Class 1C/1D/1E or TKV >5%/year; LFT monitoring "
                "essential (rare hepatotoxicity); (4) ICA screening: MRI/MRA brain if personal or "
                "family history of ICA/rupture; or occupation requiring high alertness; "
                "(5) Avoid caffeine excess, NSAIDs, smoking — accelerate cyst growth. "
                "Criteria for tolvaptan: ADPKD diagnosis + CKD G2-G3 + rapidly progressing by "
                "imaging (Mayo classification based on kidney volume/height ratio)."
            ),
            "Uromodulin_Disease": (
                "UMOD-related kidney disease (FJHN / MCKD type 2) is the most common cause of "
                "hereditary tubulointerstitial nephritis. Pathophysiology: AD UMOD missense → "
                "cysteine substitution or ZP domain disruption → protein misfolding → ER retention "
                "→ ER stress → TAL cell death → interstitial fibrosis. Clinical hallmarks: "
                "(1) Early-onset gout (teenager, 20s) — presenting feature; (2) Hyperuricaemia "
                "(reduced fractional excretion of uric acid, FE-UA < 4%); (3) Slowly progressive "
                "CKD (ESRD 3rd–7th decade); (4) Medullary cysts (inconstant, not pathognomonic). "
                "Diagnostic: urine UMOD assay < 5th centile; UMOD gene sequencing. "
                "Management: allopurinol (NOT uricosurics), colchicine for gout (NOT NSAIDs — "
                "nephrotoxic), ACEi/ARB for CKD, BP <130/80. Transplant effective, no recurrence."
            ),
            "WT1_DSD_Nephropathy": (
                "WT1 is a 449-aa zinc finger transcription factor required for kidney and gonad "
                "development. Two distinct syndromes: Denys-Drash (DDS) and Frasier. "
                "DDS: missense variants in ZF2/ZF3 (R394W, dominant negative) → triad of diffuse "
                "mesangial sclerosis (DMS) + DSD (46XY) + Wilms tumour (nephroblastoma). "
                "DMS: massive mesangial matrix expansion → nephrotic syndrome → ESRD by 2–3 years. "
                "Wilms tumour surveillance: renal ultrasound every 3 months to age 8 in DDS. "
                "Frasier: +KTS splice variants → FSGS + 46XY gonadal dysgenesis (streak gonads) "
                "+ gonadoblastoma; NO Wilms tumour. "
                "Gonadectomy: MANDATORY for 46XY streak gonads (40–60% gonadoblastoma risk). "
                "Karyotype all patients with WT1 nephropathy: determines gonadectomy need. "
                "Renal transplant: effective; pretransplant bilateral nephrectomy for residual "
                "native kidneys in DDS (Wilms risk until age 8)."
            ),
            "Thin_Basement_Membrane_Nephropathy": (
                "TBMN (formerly 'benign familial haematuria') is caused by heterozygous COL4A3 or "
                "COL4A4 variants (monoallelic Alport). Previously considered benign — now recognised "
                "that 10–15% develop proteinuria and FSGS over decades. Electron microscopy: "
                "uniformly thin GBM (<250 nm without splitting; normal 300–400 nm). Differential "
                "from IgA nephropathy: TBMN lacks mesangial IgA deposits; haematuria is persistent "
                "from early life; family history of haematuria; EM thin rather than mixed. "
                "Management: annual urinalysis + BP + eGFR; ACEi if proteinuria ≥0.5 g/day; "
                "avoid nephrotoxins; genetic counselling (biallelic Alport possible in offspring). "
                "Partner COL4A3/COL4A4 testing before family planning when index is carrier. "
                "Long-term outcome: majority maintain normal renal function lifelong; minority "
                "progress — risk factors: male sex, proteinuria, hypertension, concurrent APOL1."
            ),
            "Calcineurin_Inhibitors_in_FSGS": (
                "Calcineurin inhibitors (CNIs — cyclosporin A, tacrolimus) have two distinct "
                "mechanisms in FSGS: (1) immunosuppressive: reduce T-cell activation + cytokine "
                "production (relevant in immune-mediated minimal change disease and primary FSGS); "
                "(2) direct podocyte: stabilise synaptopodin (actin-regulating protein in podocytes) "
                "via calcineurin inhibition → protects foot process architecture. Genetic FSGS "
                "(NPHS2, TRPC6) responds to direct podocyte CNI effect (NOT immune mechanism). "
                "Partial remission (>50% proteinuria reduction) in 20–35% of NPHS2/TRPC6 FSGS. "
                "Monitoring: cyclosporin levels (trough 100–175 ng/mL for FSGS), renal function, "
                "BP; tacrolimus: trough 4–8 ng/mL. Nephrotoxicity with prolonged high-dose use "
                "— time-limited trials (6 months); taper if partial response achieved."
            ),
        },
        "key_standards": [
            "Kashtan CE et al. Kidney Int 2018 — KDIGO clinical practice guidelines for Alport syndrome",
            "Gross O et al. JASN 2012 — ACEi delays ESRD in Alport syndrome by 13 years",
            "Boute N et al. Nat Genet 2000 — NPHS2 (podocin) original description",
            "Kestila M et al. Mol Cell 1998 — NPHS1 (nephrin) original description",
            "Torres VE et al. NEJM 2017 — Tolvaptan in ADPKD (TEMPO 3:4 / REPRISE trials)",
            "Kiryluk K et al. JASN 2016 — ADPKD genotype-phenotype and progression",
            "Winn MP et al. Science 2005 — TRPC6 GOF in FSGS6 original description",
            "Zenker M et al. Nat Genet 2004 — NPHS2 R229Q low-penetrance modifier clarification",
            "Turner AN et al. Nephrol Dial Transplant 2003 — thin basement membrane nephropathy review",
            "Bleyer AJ et al. Kidney Int 2010 — UMOD uromodulin kidney disease comprehensive review",
        ],
        "pharmacological_distinctions": [
            "COL4A5/COL4A3-Alport: ACEi (ramipril 5-10mg/day) START AT DIAGNOSIS regardless of proteinuria — delays ESRD 13 years; ARB second-line; AVOID NSAIDs/aminoglycosides/contrast",
            "NPHS1-CNS-F: NO immunosuppression (genetic, not immune) — bilateral nephrectomy + dialysis + transplant pathway; albumin infusions + nutritional support pre-nephrectomy",
            "NPHS2-FSGS2: NO prolonged steroid trials (steroid-resistant by definition) — cyclosporin partial response in 20-35%; ACEi/ARB for proteinuria; transplant (low recurrence)",
            "WT1-DDS/Frasier: Gonadectomy mandatory (46XY) + renal US surveillance (DDS/Wilms) + transplant at ESRD; DDS ESRD by 2-3 years — early transplant listing",
            "UMOD-FJHN: Allopurinol (NOT uricosurics) for hyperuricaemia; colchicine (NOT NSAIDs) for gout; ACEi/ARB + BP <130/80; no disease-modifying treatment; transplant effective",
            "PKD1-ADPKD: Tolvaptan (Mayo 1C-1E or TKV >5%/year) + ACEi/ARB + BP<130/80 + 2-3L hydration daily; ICA screening MRA if family Hx rupture; avoid caffeine/NSAIDs/smoking",
            "TRPC6-FSGS6: NO steroids (GOF-genetic, not immune) — calcineurin inhibitors partial response via direct podocyte effect; EXCLUDE carrier relatives from living donation; transplant at ESRD",
        ],
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    ov = get_overview()
    print(f"Total patients: {ov['aggregate_stats']['total_patients']}")
    print(f"Genes: {[g['gene'] for g in ov['genes']]}")
    print(f"Seeds: {ov['seeds']}")
    print("\n=== BREAKDOWN (gene list) ===")
    bd = get_breakdown()
    for gene, info in bd.items():
        print(f"  {gene}: {info['n_patients']} pts, mean dx age {info['stats']['mean_dx_age']}y, delay {info['stats']['mean_dx_delay_months']}m")
    print("\n=== DEFINITIONS ===")
    df = get_definitions()
    print(f"Concepts: {len(df['concepts'])}")
