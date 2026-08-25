"""
Joubert Syndrome (JBTS — Multi-Gene Ciliopathy; Molar Tooth Sign)
================================================================================
Primary Gene : CEP290 (*610142) — most common (~25%); also BBS14
Other Genes  : AHI1 (*608894, JBTS3) · INPP5E (*613037, JBTS1) · CC2D2A (*612013) ·
               TMEM67/MKS3 (*609884, JBTS6) · RPGRIP1L (*610937) · KIF7 (*611254) ·
               TCTN1/2/3 · B9D1/D2 · MKS1 · NPHP1 · NPHP4 · SDCCAG8 · many more (>35 genes)
OMIM Dis.    : #213300 (Joubert Syndrome 1 / Classic JBTS)
               #608091 (JBTS3 / AHI1) · #614321 (JBTS17 / CPLANE1) etc.
Chromosome   : CEP290 at 12q21.32; AHI1 at 6q23.3; INPP5E at 9q34.3
Inheritance  : Autosomal Recessive (biallelic LOF); rare X-linked (OFD1 gene, males)
Prevalence   : ~1/80,000–1/100,000 live births; ~40,000–50,000 affected worldwide

Mechanism
---------
Joubert syndrome is caused by defects in the primary cilium transition zone (TZ) —
the gating compartment at the base of cilia that controls protein entry/exit.

Primary cilia are essential sensory organelles on virtually every post-mitotic cell:
  • Signal Hedgehog (SHH): developmental patterning (limb bud, brain, kidney, retina)
  • Signal PDGF-Rα: fibroblast + astrocyte migration
  • Signal Wnt/Notch: tissue patterning
  • Sense fluid flow (mechanosensation): kidney tubular cells, cholangiocytes, nodal cilia

JBTS gene products are transition zone (TZ) structural proteins or TZ complex members:
  CEP290 (12q21.32): TZ matrix protein; connects ciliary Y-links to axoneme;
    CEP290 loss → TZ collapse → unregulated protein entry into cilium →
    SHH + PDGF-Rα + Wnt signalling failure.
  AHI1 (6q23.3): TZ protein; stabilises ciliary transition fibres;
    critical for cerebellar vermis neuronal migration (explains molar tooth sign).
  INPP5E (9q34.3): phosphoinositide 5-phosphatase localised to ciliary axoneme tip;
    converts PI(4,5)P2 → PI(4)P at cilium tip; regulates SHH pathway.
    INPP5E loss → excess PI(4,5)P2 accumulates at cilium tip → impaired SHH gradient
    → cerebellar granule cell migration failure → molar tooth sign.
  CC2D2A: TZ coiled-coil protein; stabilises MKS TZ complex.
  TMEM67/MKS3: TZ transmembrane protein; critical for left-right axis determination.

JBTS pathology is entirely ciliary (TZ dysfunction) — distinct from:
  BBS (BBSome IFT cargo): retrograde IFT fail → GPCR mis-trafficking → metabolic/sensory
  Alström (ALMS1 basal body scaffold): ciliary anchoring failure → multi-organ ciliopathy
  Nephronophthisis (NPHP1-NPH): TZ/axonemal defect → tubulointerstitial nephropathy

Clinical Features — Multi-System Ciliopathy
-------------------------------------------
1. Brain / Neurological (defining features):
   • MOLAR TOOTH SIGN (MTS): pathognomonic on axial MRI:
     - Cerebellar vermis hypoplasia/aplasia (superior cerebellar peduncle elongated)
     - Deepened interpeduncular fossa (between elongated SCPs)
     - SCP + vermis + 4th ventricle → "molar tooth" appearance
   • Neonatal breathing dysrhythmia (episodic hyperpnoea alternating apnoea): unique;
     resolves in most by age 2-3 years; occasionally fatal if untreated
   • Nystagmus (horizontal pendular; acquired); ocular motor apraxia (OMA)
   • Hypotonia (neonatal + early childhood)
   • Developmental delay: gross motor > fine motor; walking age 3-5 years in most;
     never achieves independent walking (~10%); most walk with support by age 6

2. Retinal (JBTS + retinal = Leber Congenital Amaurosis-like):
   • Rod-cone dystrophy (similar to LCA) — not cone-rod as Alström;
     visual impairment ranges from mild → legal blindness by adolescence
   • CEP290 is also the most common LCA gene (LCA10/CEP290);
     milder CEP290 variants → JBTS; severe variants → isolated LCA or Meckel

3. Renal (JBTS + renal = NPHP-like):
   • Nephronophthisis (NPHP): tubulointerstitial nephropathy → ESRD;
     most common JBTS renal phenotype; non-obstructive; NPHP-genes overlap with JBTS
   • Cystic kidneys (medullary cysts; corticomedullary junction cysts): on USS
   • Renal failure: ESRD by teen years (JBTS-renal subtype)

4. Hepatic (JBTS + liver = Congenital Hepatic Fibrosis, CHF):
   • CHF: ductal plate malformation; bile duct proliferation; portal fibrosis;
     complication: portal hypertension, oesophageal varices, hypersplenism
   • Caroli disease (biliary ectasia) may coexist
   • JBTS-hepatic subtype (TMEM67/CC2D2A genes prominent)

5. Other features (gene-dependent):
   • Post-axial polydactyly (20-30%; TCTN1/KIF7/CC2D2A subtypes)
   • Orofacial: macroglossia; hamartomas; gingival frenula (TMEM67/CC2D2A subtypes)
   • Obesity (hypothalamic cilia dysfunction; less prominent than BBS/Alström)
   • Diabetes: not a primary feature of JBTS (unlike BBS/Alström/Wolfram)

JBTS Clinical Subtype Classification
--------------------------------------
1. Pure JBTS (JS): MTS + cerebellar/neurological only; no retinal/renal/hepatic
2. JS + Retinal (JSRD): MTS + retinal dystrophy (rod-cone); CEP290/AHI1 predominant
3. JS + Renal (JSRD): MTS + NPHP-like renal disease; NPHP1/NPHP4/INPP5E/CC2D2A
4. JS + Orofaciodigital (JSOFD): polydactyly + facial; TCTN/CC2D2A/TMEM67
5. JS + Hepatic (JSH): MTS + CHF; TMEM67/CC2D2A/RPGRIP1L predominant
6. Meckel-Gruber Spectrum (MKS): lethal end; exencephaly + cysts + polydactyly;
   overlapping genes (MKS1, TMEM216, CEP290 severe) — allele severity determines

CEP290 — Key Gene Details (most common JBTS gene, ~25%)
----------------------------------------------------------
• 12q21.32; 54 exons; 2480 aa; 290 kDa; centrosome + ciliary TZ
• Intronic mutation IVS26+1655A>G (c.2991+1655A>G): creates cryptic exon 26a;
  MOST COMMON JBTS/LCA10 CEP290 mutation worldwide (~20% of CEP290 alleles)
• p.Arg1933* (c.5797C>T): nonsense; exon 41; northern European
• p.Gln1745* (c.5233C>T): exon 35; truncating
• p.Arg151* (c.451C>T): exon 5; null; severe (Meckel spectrum)
• French-Canadian founder: c.1666C>T (p.Arg556*): exon 16; Eastern Quebec enriched
• p.Pro1280Leu (c.3839C>T): exon 26; missense; milder phenotype (JBTS only)

Key Mutations by Gene
-----------------------
AHI1 (JBTS3): c.2262_2265del (p.Glu754fs): European; exon 20
               p.Arg830Trp (c.2488C>T): Italian founder
               c.1635+1G>A: splice; JBTS + retinal

INPP5E (JBTS1): p.Arg563His (c.1688G>A): the most reported JBTS1 pathogenic variant
                p.Glu670Lys (c.2008G>A): moderate phenotype
                p.Leu650Pro (c.1949T>C): loss of ciliary targeting domain

CC2D2A: p.Arg1564* (c.4690C>T): European
         p.Trp1182* (c.3546G>A): Middle Eastern founder

TMEM67/MKS3: p.Cys615Arg (c.1843T>C): North African; CHF-prominent phenotype
              p.Ala334Val (c.1001C>T): European; hepato-renal JBTS

Diagnosis
----------
1. Brain MRI (axial T2/T1): Molar Tooth Sign — mandatory first-line test for JBTS
2. Gene panel (≥ 35 JBTS genes) or whole exome sequencing (WES): after MTS confirmed
3. ERG (electroretinogram): documents retinal involvement; rod-cone pattern
4. Renal USS + urine ACR + eGFR: NPHP screening; annual from diagnosis
5. LFTs + liver USS: hepatic fibrosis / CHF / Caroli screening
6. Ophthalmology: fundus photography; OCT; visual fields
7. Audiology: SNHL uncommon but screen annually
8. Polysomnography: neonatal breathing dysrhythmia (apnoea) — monitor until resolved

Management
-----------
• Brain/neuro: multidisciplinary; physiotherapy (hypotonia); early intervention;
  occupational therapy; speech therapy; special education (ID management)
• Retinal: low-vision aids; genetic counselling; gene therapy trials (CEP290 ASO antisense
  oligonucleotide trial (sepofarsen/QR-110) — partial vision restoration in LCA10/CEP290;
  CRISPR-Cas9 editing trials in CEP290-LCA10; JBTS-retinal may benefit)
• Renal: annual USS + urine ACR + eGFR; nephrology referral if eGFR < 60;
  ESRD: dialysis or transplantation (excellent outcomes — no recurrence in transplant)
• Hepatic (CHF): USS annually; gastroscopy for varices (portal hypertension);
  beta-blockers (propranolol) or endoscopic banding for varices;
  liver transplant if decompensated cirrhosis (combined liver-kidney if dual ESRD+CHF)
• Neonatal apnoea: monitoring; supplemental O2; caffeine if significant apnoea;
  rarely needs CPAP; usually self-resolves by 2-3 years
• No disease-modifying therapy for JBTS as of 2026 (CEP290 ASO is LCA-specific;
  not yet proven in full JBTS neurological phenotype)
• Genetic counselling: 25% sibling recurrence (AR); WES for gene identification;
  prenatal diagnosis by MRI (2nd trimester) + gene-directed testing
"""

import random
import statistics

_SEED = 335  # next after BBS seed=333
_N    = 40

_GENES = [
    "CEP290 (12q21.32) — c.2991+1655A>G / p.Ile984fs (IVS26 cryptic exon founder)",
    "CEP290 (12q21.32) — c.2991+1655A>G / p.Arg1933* (IVS26 + exon41 truncating)",
    "CEP290 (12q21.32) — p.Arg1933* / p.Gln1745* (biallelic truncating; severe)",
    "CEP290 (12q21.32) — c.1666C>T (p.Arg556*) French-Canadian founder / compound het",
    "CEP290 (12q21.32) — p.Pro1280Leu / c.2991+1655A>G (missense + IVS26; mild JBTS only)",
    "AHI1 (6q23.3) — c.2262_2265del (p.Glu754fs) + p.Arg830Trp (JBTS3; retinal prominent)",
    "AHI1 (6q23.3) — c.1635+1G>A splice + p.Glu754fs (JBTS3 + retinal dystrophy)",
    "INPP5E (9q34.3) — p.Arg563His / p.Glu670Lys (JBTS1; cerebellar mild; retinal late)",
    "INPP5E (9q34.3) — p.Leu650Pro / p.Arg563His (JBTS1; ciliary targeting domain loss)",
    "CC2D2A (4p15.33) — p.Arg1564* / p.Trp1182* (hepato-renal-cerebellar; TMEM67 overlap)",
    "TMEM67/MKS3 (8q22.1) — p.Cys615Arg / p.Ala334Val (hepatic fibrosis + CHF prominent)",
    "TMEM67/MKS3 (8q22.1) — p.Cys615Arg homozygous (North African founder; CHF + NPHP)",
    "RPGRIP1L (16q12.2) — c.1945C>T (p.Arg649*) / splice site (JBTS7; renal-hepatic)",
    "KIF7 (15q26.1) — p.His1387Leufs*2 / p.Arg343Pro (JBTS12; polydactyly + acrocallosal)",
    "NPHP1 (2q13) — large deletion (homozygous) / compound deletion (NPHP + JBTS overlap)",
]

_JBTS_SUBTYPES = [
    "Pure JBTS (MTS + cerebellar only; no retinal/renal/hepatic involvement)",
    "JBTS + Retinal Dystrophy (JSRD; rod-cone LCA-like; CEP290/AHI1 subtype)",
    "JBTS + Renal (JSRD-renal; NPHP-like tubulointerstitial; CC2D2A/NPHP1 subtype)",
    "JBTS + Hepatic (JSH; congenital hepatic fibrosis + CHF; TMEM67/CC2D2A subtype)",
    "JBTS + Renal + Hepatic (JSH-renal; combined ESRD + CHF; severe TZ defect)",
    "JBTS + Polydactyly (JSOFD-like; post-axial polydactyly + MTS; TCTN/KIF7 subtype)",
]

_ETHNICITIES = [
    "Northern European", "Northern European", "Northern European", "Northern European",
    "Northern European", "Northern European",
    "British", "British", "British",
    "French-Canadian", "French-Canadian", "French-Canadian",
    "North African", "North African", "North African",
    "Turkish", "Turkish",
    "Middle Eastern", "Middle Eastern",
    "South Asian", "South Asian",
    "East Asian", "East Asian",
    "Italian", "Italian",
    "Latin American", "Latin American",
    "Scandinavian", "Scandinavian",
    "Mediterranean",
    "Dutch",
    "German",
    "Polish",
    "Irish",
    "Greek",
    "Lebanese",
    "Saudi Arabian",
    "Pakistani",
    "Indian",
    "Australian-European",
]

_RETINAL_STATUSES = [
    "Normal vision (no retinal involvement — Pure JBTS subtype; under annual review)",
    "Mild retinal dystrophy (VA 0.5–0.8; rod-cone ERG reduced; no legal blindness yet)",
    "Moderate retinal dystrophy (VA 0.2–0.5; ring scotoma; significant field loss)",
    "Severe retinal dystrophy (VA < 0.1; profound field loss; legally blind)",
    "LCA-like (VA < 0.05; ERG extinguished; CEP290 IVS26 allele; early onset < 2 yr)",
    "Nystagmus + OMA only (ocular motor apraxia; retina intact; AHI1 predominant)",
]

_RENAL_STATUSES = [
    "Normal renal function (eGFR > 90; no proteinuria; USS normal — Pure JBTS/retinal subtype)",
    "Medullary cysts (USS confirmed; eGFR 80–90; sub-clinical NPHP)",
    "NPHP-like (eGFR 60–80; tubulointerstitial changes; low-grade proteinuria)",
    "CKD stage 2–3 (eGFR 30–60; NPHP-confirmed; nephrology active management)",
    "CKD stage 4–5 (eGFR < 30; ESRD approaching; dialysis planning)",
    "ESRD — transplanted (dialysis pre-transplant; functioning graft; no recurrence)",
]

_HEPATIC_STATUSES = [
    "Normal liver (no CHF; USS normal; LFTs normal — non-hepatic JBTS subtype)",
    "CHF stage 1 (periportal fibrosis only; LFTs mildly elevated; USS: echogenic liver)",
    "CHF stage 2 (established portal fibrosis; USS: dilated portal vein; splenomegaly)",
    "CHF with portal hypertension (varices on gastroscopy; propranolol started)",
    "CHF + Caroli (biliary ectasia; cholangitis risk; ursodeoxycholic acid therapy)",
    "Decompensated CHF (ascites + varices + hypersplenism; liver transplant evaluated)",
]

_BREATHING_STATUSES = [
    "Resolved (neonatal apnoea/hyperpnoea resolved by age 18 months — self-limited)",
    "Resolved by age 24 months (required caffeine + O2 monitoring in neonatal period)",
    "Resolved (mild; no intervention needed; SIDS monitor used in infancy only)",
    "Ongoing mild (age > 3 years; occasional nocturnal apnoea; home oximetry)",
    "Not applicable (JBTS diagnosed in adulthood — neonatal history not documented)",
]

_POLYDACTYLY = [
    "No polydactyly (not a KIF7/TCTN subtype; classic JBTS)",
    "Post-axial polydactyly — bilateral hands (surgically corrected in infancy; KIF7 variant)",
    "Post-axial polydactyly — hands + feet (KIF7/CC2D2A subtype; surgery planned)",
    "Pre-axial polydactyly (thumb duplication; rare; KIF7 variant)",
]

_MISDIAGNOSES = [
    "None (MTS identified on MRI; JBTS panel confirmed; direct referral)",
    "Cerebral palsy (hypotonia + motor delay labelled CP; MRI later showed MTS)",
    "Leber Congenital Amaurosis (CEP290 detected on LCA panel; MRI not done initially)",
    "Dandy-Walker malformation (MRI misread; MTS pattern not recognised; genetics review)",
    "Oculomotor apraxia syndrome (AHI1 JBTS3; OMA prominent; JBTS delayed)",
    "Non-specific intellectual disability (MTS on routine MRI at age 5; then JBTS panel)",
    "Joubert syndrome suspected clinically but confirmed late (WES at age 8; panel negative)",
    "NPHP-primary (renal biopsied; NPHP1 deletion; MRI showed MTS retrospectively)",
]


def _build_cohort() -> list:
    rng = random.Random(_SEED)

    ethnicities_pool = _ETHNICITIES[:]
    rng.shuffle(ethnicities_pool)
    ethnicities_pool = (ethnicities_pool * 2)[:_N]

    cohort = []
    for i in range(_N):
        age  = rng.randint(2, 45)
        sex  = rng.choice(["Male", "Female"])
        gene = rng.choice(_GENES)
        jbts_subtype = rng.choice(_JBTS_SUBTYPES)

        # CEP290 variants most common
        is_cep290 = "CEP290" in gene
        is_ahi1   = "AHI1" in gene
        is_tmem67 = "TMEM67" in gene
        is_cc2d2a = "CC2D2A" in gene

        # Retinal — CEP290/AHI1 → retinal; TMEM67 → hepatic mostly
        if is_cep290 or is_ahi1:
            retinal = rng.choice(_RETINAL_STATUSES[1:])  # Some retinal involvement
        elif is_tmem67:
            retinal = rng.choice(_RETINAL_STATUSES[:2])  # Usually no retinal in TMEM67
        else:
            retinal = rng.choice(_RETINAL_STATUSES)
        has_retinal_dx = "Normal" not in retinal

        # Renal — CC2D2A/TMEM67/NPHP1 → renal
        if is_cc2d2a or is_tmem67 or "NPHP1" in gene:
            renal = rng.choice(_RENAL_STATUSES[2:])  # More renal disease
        elif is_cep290:
            renal = rng.choice(_RENAL_STATUSES[1:4])
        else:
            renal = rng.choice(_RENAL_STATUSES)
        has_renal_dx = "Normal" not in renal
        has_esrd = "ESRD" in renal or "stage 4" in renal.lower() or "stage 5" in renal.lower()

        # Hepatic — TMEM67/CC2D2A → CHF
        if is_tmem67:
            hepatic = rng.choice(_HEPATIC_STATUSES[2:])  # CHF prominent
        elif is_cc2d2a:
            hepatic = rng.choice(_HEPATIC_STATUSES[1:])
        else:
            hepatic = rng.choice(_HEPATIC_STATUSES[:2])  # Usually normal
        has_hepatic_dx = "Normal" not in hepatic
        has_chf = "CHF" in hepatic

        # Breathing
        breathing = rng.choice(_BREATHING_STATUSES)

        # Polydactyly — ~20-30%
        polydactyly = rng.choices(
            _POLYDACTYLY,
            weights=[70, 15, 10, 5],
            k=1
        )[0]
        has_polydactyly = "No polydactyly" not in polydactyly

        # Diabetes — NOT a primary feature of JBTS (unlike BBS/Alström)
        # Small % may develop DM due to renal failure or obesity
        if has_esrd or (age > 30 and rng.random() < 0.15):
            hba1c = round(rng.uniform(6.2, 9.5), 1)
            has_dm = hba1c >= 6.5
        else:
            hba1c = round(rng.uniform(4.5, 6.2), 1)
            has_dm = False

        # C-peptide (if DM — insulin resistance from renal disease/obesity)
        c_peptide = round(rng.uniform(0.8, 1.8) if has_dm else rng.uniform(0.9, 1.6), 2)

        # BMI — JBTS obesity less severe than BBS
        if age < 12:
            bmi = round(rng.uniform(15.0, 24.0), 1)
        else:
            bmi = round(rng.uniform(18.0, 36.0), 1)

        # eGFR
        if has_esrd:
            egfr = rng.randint(5, 25)
        elif has_renal_dx:
            egfr = rng.randint(25, 75)
        else:
            egfr = rng.randint(70, 110)

        # IQ estimate
        iq = rng.randint(45, 85)  # Mild to moderate ID; range wide

        # Age at dx
        dx_age = rng.randint(0, min(age, 5))

        # Misdiagnosis
        misdiag = rng.choice(_MISDIAGNOSES)

        cohort.append({
            "id":               i + 1,
            "age":              age,
            "sex":              sex,
            "gene":             gene,
            "jbts_subtype":     jbts_subtype,
            "ethnicity":        ethnicities_pool[i],
            "retinal_status":   retinal,
            "renal_status":     renal,
            "hepatic_status":   hepatic,
            "breathing_status": breathing,
            "polydactyly":      polydactyly,
            "has_retinal_dx":   has_retinal_dx,
            "has_renal_dx":     has_renal_dx,
            "has_esrd":         has_esrd,
            "has_hepatic_dx":   has_hepatic_dx,
            "has_chf":          has_chf,
            "has_polydactyly":  has_polydactyly,
            "has_dm":           has_dm,
            "hba1c":            hba1c,
            "c_peptide_nmol_L": c_peptide,
            "bmi":              bmi,
            "egfr_ml_min":      egfr,
            "iq_estimate":      iq,
            "age_at_diagnosis": dx_age,
            "prior_misdiagnosis": misdiag,
        })
    return cohort


def get_overview() -> dict:
    cohort = _build_cohort()
    n = len(cohort)
    ages  = [p["age"] for p in cohort]
    bmis  = [p["bmi"] for p in cohort]
    egfrs = [p["egfr_ml_min"] for p in cohort]
    iqs   = [p["iq_estimate"] for p in cohort]

    pct_retinal    = round(sum(1 for p in cohort if p["has_retinal_dx"]) / n * 100, 1)
    pct_renal      = round(sum(1 for p in cohort if p["has_renal_dx"]) / n * 100, 1)
    pct_esrd       = round(sum(1 for p in cohort if p["has_esrd"]) / n * 100, 1)
    pct_hepatic    = round(sum(1 for p in cohort if p["has_hepatic_dx"]) / n * 100, 1)
    pct_chf        = round(sum(1 for p in cohort if p["has_chf"]) / n * 100, 1)
    pct_poly       = round(sum(1 for p in cohort if p["has_polydactyly"]) / n * 100, 1)
    pct_dm         = round(sum(1 for p in cohort if p["has_dm"]) / n * 100, 1)
    pct_cep290     = round(sum(1 for p in cohort if "CEP290" in p["gene"]) / n * 100, 1)

    kpis = {
        "cohort_n":      n,
        "gene":          "CEP290 (most common, ~25%); ≥35 JBTS genes",
        "syndrome":      "Joubert Syndrome (JBTS — TZ ciliopathy)",
        "chromosome":    "12q21.32 (CEP290); multi-locus",
        "inheritance":   "Autosomal Recessive",
        "median_age":    round(statistics.median(ages), 1),
        "mean_bmi":      round(statistics.mean(bmis), 1),
        "mean_egfr":     round(statistics.mean(egfrs), 1),
        "mean_iq":       round(statistics.mean(iqs), 0),
        "pct_retinal":   pct_retinal,
        "pct_renal":     pct_renal,
        "pct_esrd":      pct_esrd,
        "pct_hepatic":   pct_hepatic,
        "pct_chf":       pct_chf,
        "pct_polydactyly": pct_poly,
        "pct_dm":        pct_dm,
        "pct_cep290":    pct_cep290,
    }

    key_facts = [
        "MOLAR TOOTH SIGN (MTS) on axial brain MRI is pathognomonic for Joubert Syndrome — always obtain MRI first",
        "Cause: transition zone (TZ) ciliopathy — TZ structural proteins gate ciliary entry; loss → SHH/Wnt/PDGF failure",
        "CEP290 (12q21.32) is the most common gene (~25%); CEP290 IVS26 cryptic exon mutation is the most common single allele",
        "CEP290 is also BBS14 — allele severity determines phenotype: mild → JBTS, moderate → isolated LCA10, severe → Meckel",
        "JBTS subtypes: Pure cerebellar · +Retinal (CEP290/AHI1) · +Renal/NPHP (CC2D2A/NPHP1) · +Hepatic/CHF (TMEM67) · +Polydactyly (KIF7)",
        "Neonatal breathing dysrhythmia (episodic hyperpnoea ↔ apnoea) is characteristic and typically self-resolves by age 2–3",
        "Renal: NPHP-like tubulointerstitial nephropathy → ESRD in renal subtype (kidney transplant outcomes EXCELLENT — no recurrence)",
        "Hepatic: congenital hepatic fibrosis (CHF) in TMEM67/CC2D2A subtypes → portal hypertension, varices",
        "Diabetes is NOT a primary JBTS feature (unlike BBS/Alström/Wolfram) — T2D occurs only secondary to obesity/ESRD",
        "CEP290 gene therapy (antisense oligonucleotide sepofarsen) restores partial vision in LCA10 — JBTS-retinal trials ongoing",
        f"{pct_cep290}% CEP290 · {pct_retinal}% retinal · {pct_renal}% renal · {pct_hepatic}% hepatic · {pct_esrd}% ESRD",
    ]

    alerts = {
        "mts_mandatory": "Brain MRI is MANDATORY — JBTS cannot be diagnosed without MTS confirmation; panel positivity alone insufficient",
        "neonatal_apnoea": "Neonatal apnoea/hyperpnoea: monitor O2 saturation; caffeine if needed; usually self-resolves (fatal if unmanaged)",
        "renal_screen": "Annual renal USS + eGFR + urine ACR from diagnosis — NPHP-like ESRD by teens if untreated/unmonitored",
        "hepatic_screen": "Annual liver USS + LFTs — CHF/portal hypertension in TMEM67/CC2D2A subtypes; gastroscopy for varices",
    }

    return {
        "kpis":      kpis,
        "key_facts": key_facts,
        "alerts":    alerts,
        "patients":  cohort[:8],
    }


def get_breakdown() -> dict:
    cohort = _build_cohort()
    n = len(cohort)

    # Gene distribution
    gene_dist: dict = {}
    for p in cohort:
        key = p["gene"].split("(")[0].strip().split(" —")[0].strip()
        gene_dist[key] = gene_dist.get(key, 0) + 1

    # JBTS subtype
    subtype_dist: dict = {}
    for p in cohort:
        key = p["jbts_subtype"].split("(")[0].strip()
        subtype_dist[key] = subtype_dist.get(key, 0) + 1

    # Ethnicity
    eth_dist: dict = {}
    for p in cohort:
        eth_dist[p["ethnicity"]] = eth_dist.get(p["ethnicity"], 0) + 1

    # Retinal
    ret_dist: dict = {}
    for p in cohort:
        key = p["retinal_status"].split("(")[0].strip()
        ret_dist[key] = ret_dist.get(key, 0) + 1

    # Renal
    ren_dist: dict = {}
    for p in cohort:
        key = p["renal_status"].split("(")[0].strip()
        ren_dist[key] = ren_dist.get(key, 0) + 1

    # Hepatic
    hep_dist: dict = {}
    for p in cohort:
        key = p["hepatic_status"].split("(")[0].strip()
        hep_dist[key] = hep_dist.get(key, 0) + 1

    # Misdiagnosis
    mis_dist: dict = {}
    for p in cohort:
        key = p["prior_misdiagnosis"].split("(")[0].strip()
        mis_dist[key] = mis_dist.get(key, 0) + 1

    # HbA1c tiers
    hba1c_tiers = {
        "< 5.7% (Normoglycaemic)": 0,
        "5.7–6.4% (Pre-diabetic)": 0,
        "6.5–7.9% (DM controlled)": 0,
        "8.0–9.9% (DM moderate)": 0,
        "≥ 10.0% (DM poorly controlled)": 0,
    }
    for p in cohort:
        v = p["hba1c"]
        if v < 5.7:   hba1c_tiers["< 5.7% (Normoglycaemic)"] += 1
        elif v < 6.5: hba1c_tiers["5.7–6.4% (Pre-diabetic)"] += 1
        elif v < 8.0: hba1c_tiers["6.5–7.9% (DM controlled)"] += 1
        elif v < 10.: hba1c_tiers["8.0–9.9% (DM moderate)"] += 1
        else:         hba1c_tiers["≥ 10.0% (DM poorly controlled)"] += 1

    # eGFR tiers
    egfr_tiers = {
        "≥ 90 (Normal G1)": 0,
        "60–89 (Mild reduction G2)": 0,
        "30–59 (Moderate CKD G3)": 0,
        "15–29 (Severe CKD G4)": 0,
        "< 15 / ESRD (G5)": 0,
    }
    for p in cohort:
        v = p["egfr_ml_min"]
        if v >= 90:   egfr_tiers["≥ 90 (Normal G1)"] += 1
        elif v >= 60: egfr_tiers["60–89 (Mild reduction G2)"] += 1
        elif v >= 30: egfr_tiers["30–59 (Moderate CKD G3)"] += 1
        elif v >= 15: egfr_tiers["15–29 (Severe CKD G4)"] += 1
        else:         egfr_tiers["< 15 / ESRD (G5)"] += 1

    # BMI tiers
    bmi_tiers = {
        "< 18.5 (Underweight)": 0,
        "18.5–24.9 (Healthy)": 0,
        "25.0–29.9 (Overweight)": 0,
        "30.0–34.9 (Obese class I)": 0,
        "≥ 35.0 (Obese class II+)": 0,
    }
    for p in cohort:
        v = p["bmi"]
        if v < 18.5:  bmi_tiers["< 18.5 (Underweight)"] += 1
        elif v < 25:  bmi_tiers["18.5–24.9 (Healthy)"] += 1
        elif v < 30:  bmi_tiers["25.0–29.9 (Overweight)"] += 1
        elif v < 35:  bmi_tiers["30.0–34.9 (Obese class I)"] += 1
        else:         bmi_tiers["≥ 35.0 (Obese class II+)"] += 1

    summary_flags = {
        "pct_cep290":         round(sum(1 for p in cohort if "CEP290" in p["gene"]) / n * 100, 1),
        "pct_ahi1":           round(sum(1 for p in cohort if "AHI1" in p["gene"]) / n * 100, 1),
        "pct_tmem67":         round(sum(1 for p in cohort if "TMEM67" in p["gene"]) / n * 100, 1),
        "pct_retinal":        round(sum(1 for p in cohort if p["has_retinal_dx"]) / n * 100, 1),
        "pct_renal":          round(sum(1 for p in cohort if p["has_renal_dx"]) / n * 100, 1),
        "pct_esrd":           round(sum(1 for p in cohort if p["has_esrd"]) / n * 100, 1),
        "pct_chf":            round(sum(1 for p in cohort if p["has_chf"]) / n * 100, 1),
        "pct_polydactyly":    round(sum(1 for p in cohort if p["has_polydactyly"]) / n * 100, 1),
        "pct_dm":             round(sum(1 for p in cohort if p["has_dm"]) / n * 100, 1),
        "pct_antibody_neg":   100.0,
        "mts_sign_always":    100.0,
    }

    return {
        "gene_distribution":  gene_dist,
        "jbts_subtype":       subtype_dist,
        "ethnicity":          eth_dist,
        "retinal_status":     ret_dist,
        "renal_status":       ren_dist,
        "hepatic_status":     hep_dist,
        "misdiagnosis":       mis_dist,
        "hba1c_tiers":        hba1c_tiers,
        "egfr_tiers":         egfr_tiers,
        "bmi_tiers":          bmi_tiers,
        "summary_flags":      summary_flags,
    }


def get_definitions() -> dict:
    return {
        "disease": {
            "full_name":     "Joubert Syndrome (JBTS — Ciliary Transition Zone Ciliopathy)",
            "acronym":       "JBTS (Joubert Syndrome)",
            "primary_gene":  "CEP290 (*610142) most common (~25%); ≥35 JBTS genes; TZ structural/scaffold proteins",
            "disease_omim":  "#213300 (Joubert Syndrome 1 — classic); multiple OMIM allelic series by gene",
            "inheritance":   "Autosomal Recessive (biallelic LOF); rare X-linked (OFD1 in males); 25% sibling recurrence",
            "prevalence":    "~1/80,000–1/100,000 live births; ~40,000–50,000 affected worldwide; pan-ethnic; male = female",
            "mechanism": (
                "JBTS = transition zone (TZ) ciliopathy. TZ is the gating compartment at the ciliary base "
                "that controls which proteins enter and exit the primary cilium. "
                "JBTS proteins (CEP290, AHI1, INPP5E, CC2D2A, TMEM67, RPGRIP1L) are structural TZ components or complex members. "
                "Loss → TZ collapse → unregulated ciliary entry → impaired SHH, PDGF-Rα, Wnt signalling "
                "→ cerebellar granule cell migration failure (Molar Tooth Sign), retinal photoreceptor degeneration, "
                "renal tubular cilia dysfunction (NPHP-like nephropathy), biliary ductal plate malformation (CHF). "
                "Distinct from: BBS (BBSome IFT cargo mis-trafficking) · Alström (ALMS1 basal body scaffold) · "
                "NPHP-primary (axonemal/TZ defect — NPHP-only phenotype, no brain)."
            ),
            "molar_tooth_sign": (
                "MTS = pathognomonic radiological sign for JBTS. "
                "Components on axial T2-MRI at level of superior cerebellar peduncles: "
                "(1) Cerebellar vermis aplasia/hypoplasia → deepened interpeduncular fossa (IP fossa); "
                "(2) Superior cerebellar peduncles (SCPs) elongated + horizontal ('the roots of the molar'); "
                "(3) 4th ventricle bat-wing deformation. "
                "Together: 'molar tooth' appearance = SCP roots + IP fossa 'pulp chamber' + vermis absence. "
                "MTS must be present to diagnose JBTS — gene panel positivity without MTS = gene carrier + "
                "different ciliopathy (NPHP, BBS, Senior-Løken etc.)."
            ),
            "cep290_allele_spectrum": (
                "CEP290 variants determine phenotype by allele severity: "
                "Severe truncating biallelic (p.Arg151*) → Meckel-Gruber Syndrome (lethal). "
                "IVS26+1655A>G (cryptic exon) / mild missense → Joubert Syndrome (JBTS with/without retinal). "
                "IVS26+1655A>G biallelic → Leber Congenital Amaurosis 10 (LCA10 — retinal only, no MTS). "
                "Intermediate → Senior-Løken Syndrome (retinal + NPHP). "
                "CEP290 is also BBS14 → mild ciliary trafficking defects → BBS phenotype. "
                "One gene, four distinct ciliopathy phenotypes — allele type + modifier genes determine which."
            ),
            "c_peptide_note": (
                "Diabetes is NOT a primary Joubert Syndrome feature. "
                "Unlike BBS (insulin resistance, ~50%) or Alström (~80%) or Wolfram (beta-cell apoptosis), "
                "JBTS does not primarily dysregulate the pancreatic beta-cell or hypothalamic satiety circuits. "
                "T2D in JBTS occurs SECONDARY to: (1) ESRD (renal failure → insulin resistance + reduced clearance), "
                "(2) obesity if present (hypothalamic cilia less impaired than in BBS). "
                "C-peptide PRESERVED when DM occurs (insulin resistance mechanism, not beta-cell apoptosis). "
                "Autoantibodies always NEGATIVE. Management: renal-dose-adjusted metformin; GLP-1RA; dialysis patients insulin only."
            ),
            "treatment": (
                "No disease-modifying therapy for JBTS neurological/structural disease as of 2026. "
                "CEP290 ASO (sepofarsen/QR-110): partial vision restoration in CEP290-LCA10 (trials positive); "
                "JBTS-retinal CEP290 trials planned. "
                "CRISPR-Cas9 CEP290 editing (intronic IVS26 mutation): in vitro proof; clinical trial phase. "
                "Renal: ESRD management (ACE-I/ARB; dialysis; kidney transplantation — excellent outcomes, no recurrence). "
                "Hepatic (CHF): portal hypertension management (propranolol; endoscopic banding); "
                "liver transplant (decompensated CHF); combined liver-kidney transplant (dual ESRD+CHF). "
                "Neonatal apnoea: monitoring; caffeine; O2; resolves spontaneously. "
                "Neuro: physiotherapy; early intervention; speech/OT; special education. "
                "Retinal: low-vision aids; ERG + OCT monitoring; gene therapy clinical trials. "
                "Genetics: 25% sibling risk; prenatal MRI (2nd trimester) + gene-directed testing."
            ),
            "autoantibodies": "NEGATIVE — JBTS DM is secondary (renal failure / obesity); NOT autoimmune; NOT primary beta-cell disease",
        },

        "genes_and_proteins": {
            "CEP290 (*610142)": (
                "12q21.32. Centrosomal Protein 290 kDa. 2480 aa. "
                "Key structural matrix protein of the ciliary transition zone: "
                "forms Y-links between ciliary membrane and axoneme microtubule doublets. "
                "CEP290 loss → TZ gate collapse → unregulated membrane protein entry/exit → "
                "SHH-Gli signalling failure → cerebellar vermis hypoplasia (MTS). "
                "IVS26+1655A>G: most common allele worldwide (~20% of CEP290 alleles); "
                "creates cryptic exon 26a (150 bp insertion) → premature stop (partial protein retained). "
                "ASO therapy (sepofarsen): skips cryptic exon; partial CEP290 restoration → +10-15 letters BCVA in LCA10."
            ),
            "AHI1 (*608894) — JBTS3": (
                "6q23.3. Abelson helper integration site 1 protein (Jouberin). 1196 aa. "
                "TZ protein with WD40 repeat + coiled-coil domains. "
                "Stabilises ciliary transition fibres in cerebellar granule cells; "
                "AHI1 loss → impaired cerebellar neuron migration → MTS (vermis aplasia). "
                "Enriched for JBTS + retinal subtype; cognitive phenotype: mild-moderate ID. "
                "Italian founder: p.Arg830Trp (c.2488C>T); Portuguese/Brazilian enriched also."
            ),
            "INPP5E (*613037) — JBTS1": (
                "9q34.3. Inositol polyphosphate 5-phosphatase E. 644 aa. "
                "Phosphoinositide phosphatase localised to the ciliary axoneme tip. "
                "Converts PI(4,5)P2 → PI(4)P at the cilium tip, controlling SHH gradient. "
                "INPP5E loss → PI(4,5)P2 accumulation → impaired GLI3 processing → "
                "SHH gradient disruption → cerebellar granule cell migration failure → MTS. "
                "Relatively mild phenotype (compared to CEP290): pure JBTS or +retinal (late onset). "
                "p.Arg563His (c.1688G>A): most reported pathogenic variant in INPP5E."
            ),
            "TMEM67/MKS3 (*609884) — JBTS6": (
                "8q22.1. Transmembrane protein 67 (Meckelin). 995 aa. "
                "TZ transmembrane protein; structural component of TZ inner ring. "
                "TMEM67 variants produce JBTS hepatic subtype: congenital hepatic fibrosis (CHF) prominent. "
                "Ductal plate malformation (biliary duct proliferation) → portal fibrosis → "
                "portal hypertension → varices → hypersplenism → Caroli disease (biliary ectasia). "
                "North African founder: p.Cys615Arg (c.1843T>C) — high frequency in Tunisian/Moroccan. "
                "Combined renal + hepatic (JS-hepato-renal) is the most severe TMEM67 phenotype."
            ),
            "CEP290 in ciliopathy spectrum": (
                "CEP290 allele severity → phenotype spectrum (most severe to least): "
                "1. Meckel-Gruber Syndrome (MKS): biallelic severe truncating (e.g. p.Arg151*); lethal; "
                "   exencephaly + polydactyly + cystic kidneys; MTS not assessable. "
                "2. Joubert Syndrome (JBTS): IVS26+1655A>G or mild missense; MTS + multi-organ. "
                "3. Senior-Løken Syndrome (SLS): intermediate alleles; retinal + NPHP only. "
                "4. Leber Congenital Amaurosis 10 (LCA10): IVS26 biallelic; retinal only; NO MTS. "
                "5. Bardet-Biedl Syndrome type 14 (BBS14): mild alleles; BBSome cargo defect. "
                "One gene — 5 distinct ciliopathy phenotypes by allele severity."
            ),
        },

        "clinical_terms": {
            "JBTS vs BBS (key differential)": (
                "Both: AR ciliopathy; retinal dystrophy; renal disease; polydactyly (some subtypes). "
                "JBTS: MTS on MRI (PATHOGNOMONIC) — ALWAYS present; BBS: NO MTS (normal brain MRI). "
                "JBTS: neonatal apnoea/hyperpnoea — NOT in BBS. "
                "JBTS: transition zone (TZ) ciliopathy — BBS: BBSome IFT cargo mis-trafficking. "
                "JBTS: hepatic fibrosis/CHF (TMEM67 subtype) — BBS: NO CHF. "
                "JBTS: obesity NOT prominent (unlike BBS: morbid obesity; LepR mis-trafficking). "
                "JBTS: DM NOT primary feature — BBS: T2D in ~50% (insulin resistance). "
                "Gene panel: BBS1-21 (BBS panel) vs CEP290+JBTS panel (>35 genes) — must not overlap blindly."
            ),
            "JBTS vs Alström (key differential)": (
                "Both: AR ciliopathy; retinal dystrophy; renal disease; obesity (milder in JBTS). "
                "JBTS: MTS (pathognomonic) — Alström: NO MTS (ALMS1 basal body; normal brain MRI). "
                "JBTS: cerebellar vermis hypoplasia — Alström: NO cerebellar malformation. "
                "JBTS: neonatal apnoea — Alström: infantile DCM (cardiomyopathy; 60%). "
                "JBTS: CHF in TMEM67 subtype — Alström: NASH/NAFLD (different hepatic mechanism). "
                "Alström: cone-rod ERG (cones first) — JBTS: rod-cone (rods first) like BBS. "
                "Alström: C-pep PRESERVED, obesity ~80% → T2D — JBTS: DM NOT primary."
            ),
            "Molar Tooth Sign (MTS) anatomy": (
                "MTS is seen on axial T2 MRI at the level of the superior cerebellar peduncles (SCPs). "
                "3 components form the 'molar tooth': "
                "(1) SCP elongation (peduncles don't decussate normally → horizontal course → 'roots'). "
                "(2) Cerebellar vermis aplasia/hypoplasia (creates deep IP fossa → 'pulp chamber'). "
                "(3) 4th ventricle bat-wing shape (wide roof → 'crown'). "
                "The analogy: SCP roots = tooth roots; IP fossa = pulp chamber; 4th ventricle = crown. "
                "Radiologist must specifically look for MTS on paediatric brain MRI in hypotonia + nystagmus."
            ),
            "CEP290 ASO therapy (sepofarsen)": (
                "Sepofarsen (QR-110): antisense oligonucleotide (ASO) targeting the cryptic exon 26a "
                "created by the CEP290 IVS26+1655A>G intronic mutation. "
                "Mechanism: ASO binds pre-mRNA at the cryptic exon → blocks splicing → "
                "restores near-normal CEP290 transcript without the cryptic exon 26a insertion. "
                "ILLUMINATE Phase 2/3 trial (LCA10/CEP290): +10-15 letters BCVA at 12 months; "
                "+4.5 letters vs placebo; intravitreal injection every 3 months. "
                "JBTS application: CEP290 IVS26 is the same mutation; retinal component should respond; "
                "neurological (MTS/cerebellar) is structural — unlikely to reverse with ASO."
            ),
        },

        "management_pearls": {
            "mri_first_mts": (
                "Brain MRI is the FIRST diagnostic step — always before gene panel. "
                "MTS must be demonstrated to diagnose JBTS. Gene positivity without MTS = "
                "carrier or different ciliopathy (NPHP, BBS, Senior-Løken etc.). "
                "Request specific: 'axial T2 through superior cerebellar peduncles; "
                "assess for molar tooth sign, vermis hypoplasia, SCP elongation.'"
            ),
            "renal_transplant": (
                "Kidney transplantation is the definitive treatment for JBTS-ESRD. "
                "Outcomes are EXCELLENT in JBTS: no recurrence of NPHP in transplanted kidney "
                "(NPHP is a cell-autonomous ciliary disease — donor kidney has normal NPHP1/CEP290). "
                "Prepare from CKD stage 3: pre-emptive transplant listing; living related donors preferred "
                "(25% sibling risk — screen potential donor siblings first)."
            ),
            "combined_liver_kidney": (
                "Combined liver-kidney transplant for dual ESRD + decompensated CHF (TMEM67/CC2D2A subtype). "
                "Sequential (liver first, then kidney at 3-6 months) or simultaneous. "
                "CHF can be followed conservatively until decompensation (ascites/variceal bleeding). "
                "Portal hypertension management bridge: propranolol + endoscopic variceal banding. "
                "Timing: when ESRD reaches G4-5 AND CHF decompensates simultaneously."
            ),
            "neonatal_apnoea_management": (
                "Neonatal hyperpnoea ↔ apnoea in Joubert: monitor O2 saturation continuously. "
                "If apnoea > 15 sec or O2 drop < 85%: caffeine citrate (loading 20 mg/kg; maintenance 5-10 mg/kg/day). "
                "O2 supplementation if needed (nasal cannula; CPAP rarely). "
                "PSG at 3-6 months to document resolution. "
                "Typically resolves by 2-3 years — parental reassurance + monitoring plan essential."
            ),
            "gene_panel_selection": (
                "After MTS confirmed on MRI: order a comprehensive JBTS gene panel (≥35 genes). "
                "Start with: CEP290, AHI1, INPP5E, CC2D2A, TMEM67, RPGRIP1L, KIF7, TCTN1-3, B9D1/2, NPHP1/4. "
                "If panel negative: whole exome sequencing (WES) — 20-30% of JBTS remains unexplained. "
                "Clinical subtype guides gene prioritisation: JBTS+retinal → CEP290+AHI1 first; "
                "JBTS+hepatic → TMEM67+CC2D2A+RPGRIP1L first; JBTS+polydactyly → KIF7+TCTN+CC2D2A first."
            ),
        },
    }
