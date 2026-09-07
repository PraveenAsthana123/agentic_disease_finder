#!/usr/bin/env python3
"""Hereditary-Thyroid-Disorder-Atlas — Complete 8-Gene Atlas
TPO     (thyroid peroxidase; 933 aa; 2p25.3; AR;
         Dyshormonogenesis type 2A — most common dyshormonogenesis cause;
         organification defect — perchlorate discharge test POSITIVE PATHOGNOMONIC;
         sensorineural hearing loss ABSENT — distinguishes from Pendred/SLC26A4;
         seed SEED_BASE+0) .
TG      (thyroglobulin; 2767 aa; 8q24.22; AR;
         Dyshormonogenesis type 3 — serum Tg LOW/ABSENT despite elevated TSH;
         goiter + iodine-poor colloid; elevated TSH + low/absent Tg PATHOGNOMONIC;
         seed SEED_BASE+1) .
SLC26A4 (pendrin; 780 aa; 7q22.3; AR;
         Pendred syndrome — goiter + sensorineural hearing loss + EVA PATHOGNOMONIC TRIAD;
         EVA (enlarged vestibular aqueduct) on CT PATHOGNOMONIC;
         5-10% of all hereditary SNHL; EVA screen BEFORE cochlear implant;
         seed SEED_BASE+2) .
SLC5A5  (sodium-iodide symporter NIS; 643 aa; 19p13.11; AR;
         Iodide transport defect — RAI uptake ABSENT/VERY LOW PATHOGNOMONIC;
         perchlorate discharge NOT needed — RAI uptake distinguishes from organification defects;
         seed SEED_BASE+3) .
DUOX2   (dual oxidase 2; 1548 aa; 15q21.1; AR;
         Dyshormonogenesis type 6 — H2O2 generation failure;
         most common dyshormonogenesis worldwide (especially in Japan/Asia);
         perchlorate discharge POSITIVE; monoallelic = transient neonatal hypothyroidism;
         seed SEED_BASE+4) .
TSHR    (thyrotropin receptor; 764 aa; 14q31.1; AR/AD;
         AR LOF — TSH resistance; CH with HIGH TSH + NORMAL gland on ultrasound;
         AD GOF — familial non-autoimmune hyperthyroidism / neonatal hyperthyroidism;
         most common cause of TSH-unresponsive congenital hypothyroidism;
         seed SEED_BASE+5) .
PAX8    (paired box 8; 457 aa; 2q14.1; AD;
         thyroid dysgenesis — ectopic/hypoplastic gland;
         renal anomalies in 50% — kidney + hypothyroidism COMBINATION PATHOGNOMONIC;
         most common GENETIC cause of thyroid dysgenesis after TSHR;
         seed SEED_BASE+6) .
FOXE1   (forkhead box E1 / TTF2; 373 aa; 9q22.33; AR;
         Bamforth-Lazarus syndrome — thyroid agenesis + cleft palate + spiky hair TRIAD PATHOGNOMONIC;
         only hereditary cause of thyroid agenesis WITH characteristic dysmorphic features;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1886–1893)
"""

import random

SEED_BASE = 1886

THYROID_GENES = [
    # -- TPO -- Thyroid Peroxidase Deficiency (Dyshormonogenesis 2A) ------------------
    {
        "gene": "TPO",
        "protein": (
            "TPO -- 2p25.3 AR -- Thyroid-Peroxidase-933aa -- "
            "Dyshormonogenesis-Type-2A-Most-Common-Organification-Defect -- "
            "Perchlorate-Discharge-Test-POSITIVE-PATHOGNOMONIC -- "
            "SNHL-ABSENT-Distinguishes-from-Pendred-SLC26A4 -- "
            "Total-vs-Partial-Organification-Defect-Subtypes"
        ),
        "alias": (
            "TPO (thyroid peroxidase); OMIM gene 606765; "
            "Thyroid dyshormonogenesis type 2A (DH2A) OMIM 274500. "
            "2p25.3; 933 aa; ~103 kDa; type I membrane glycoprotein; haem-containing; forms homodimer; autosomal recessive. "
            "FUNCTION: TPO encodes thyroid peroxidase, the key enzyme at the apical membrane of thyroid follicular cells: "
            "(1) Iodide oxidation: I⁻ → I₀ (reactive iodine) using H₂O₂ (provided by DUOX2/DUOXA2); "
            "(2) Tyrosine iodination: tyrosine residues on thyroglobulin (TG) → MIT (monoiodotyrosine) and DIT (diiodotyrosine); "
            "(3) Coupling: MIT + DIT → T3; DIT + DIT → T4 (thyroxine). "
            "TPO is therefore THE central enzyme for thyroid hormone synthesis — "
            "without TPO: iodine cannot be incorporated into thyroglobulin; no T3/T4 produced. "
            "MECHANISM: "
            "TPO requires H₂O₂ from DUOX2/DUOXA2 as co-oxidant; "
            "TPO uses TG as scaffold for iodination; "
            "process occurs at the apical surface / thyroid follicle lumen interface. "
            "In TPO deficiency: "
            "iodide enters follicular cells via NIS (SLC5A5) normally — iodide uptake INTACT; "
            "but iodide cannot be oxidised or incorporated into TG; "
            "iodide 'pools' in follicular cell — not organified — remains as inorganic I⁻; "
            "PERCHLORATE displaces unorganified (free) iodide from thyroid → "
            "PERCHLORATE DISCHARGE TEST: inject radioiodine → wait 30 min → give perchlorate → "
            "if >10% discharge = organification defect (POSITIVE); "
            "TOTAL organification defect (TIOD): >90% discharge — severe, biallelic null mutations; "
            "PARTIAL organification defect (PIOD): 10-90% discharge — residual TPO activity. "
            "CLINICAL PRESENTATION: "
            "Congenital hypothyroidism detected on NBS (TSH elevated); "
            "thyroid gland: normally sited, ENLARGED (goiter) — develops due to chronic TSH stimulation; "
            "goiter may be present at birth or develop in childhood/adolescence; "
            "severity correlates with residual TPO activity; "
            "ABSENT sensorineural hearing loss (SNHL) — critical distinction from Pendred syndrome (SLC26A4) which also causes goiter + organification defect; "
            "intellectual disability if not treated from birth (hypothyroidism effect on brain development). "
            "DIAGNOSIS: "
            "NBS: elevated TSH; low T4; "
            "Thyroid ultrasound: enlarged, normally sited gland (goitre); "
            "Radioiodine (RAI) uptake: elevated (gland avid for iodine); "
            "PERCHLORATE DISCHARGE TEST: POSITIVE (>10% discharge) — PATHOGNOMONIC for organification defect; "
            "Audiogram: NORMAL — absent SNHL distinguishes from Pendred; "
            "Serum Tg: elevated (reactive, not diagnostic of TG gene defect); "
            "TPO gene sequencing: confirms. "
            "TREATMENT: "
            "Levothyroxine (L-thyroxine) replacement: lifelong; "
            "dose: guided by TSH suppression to low-normal range; "
            "Iodine supplementation: not curative (TPO absent — iodine cannot be organised); "
            "Goitre monitoring: thyroid ultrasound annually. "
            "KEY CLINICAL FACTS: "
            "MOST COMMON CAUSE OF DYSHORMONOGENESIS — TPO mutations account for majority of goitrous CH; "
            "PERCHLORATE DISCHARGE POSITIVE + SNHL ABSENT = TPO deficiency until proven otherwise; "
            "RAI UPTAKE ELEVATED (not absent) — distinguishes TPO deficiency from NIS/SLC5A5 defect where RAI uptake is absent; "
            "SNHL SCREEN MANDATORY — all goitrous CH patients must have audiogram to exclude Pendred; "
            "GOITRE IS PROGRESSIVE — may cause tracheal compression if untreated; thyroid ultrasound monitoring essential."
        ),
        "age_of_onset": "Neonatal (NBS detection); goitre may develop through childhood",
        "inheritance": "AR",
        "locus": "2p25.3",
        "protein_size": "933 aa",
        "key_biomarker": "Perchlorate discharge test POSITIVE (>10%); elevated RAI uptake; elevated TSH; normal/low T4; normal serum Tg",
        "pathognomonic": "Goitrous CH + perchlorate discharge POSITIVE + SNHL ABSENT (distinguishes from Pendred/SLC26A4)",
        "treatment": "Levothyroxine lifelong; goitre monitoring; iodine supplementation ineffective",
        "critical_flags": [
            "PERCHLORATE-DISCHARGE-POSITIVE-PATHOGNOMONIC — >10% discharge confirms organification defect; mandatory in goitrous CH",
            "SNHL-ABSENT-KEY-DISTINCTION — audiogram mandatory; SNHL present → SLC26A4 (Pendred); SNHL absent → TPO",
            "RAI-UPTAKE-ELEVATED-NOT-ABSENT — TPO deficiency: iodide enters but not organified; contrast NIS/SLC5A5: RAI uptake absent",
            "GOITRE-ENLARGES-UNTREATED — TSH stimulation drives goitre; L-thyroxine suppresses TSH and halts growth",
            "TOTAL-vs-PARTIAL-ORGANIFICATION-DEFECT — >90% discharge (TIOD): severe, null mutations; 10-90% (PIOD): partial residual activity",
            "MOST-COMMON-DYSHORMONOGENESIS — TPO mutations most frequent cause of goitrous congenital hypothyroidism worldwide",
            "IODINE-SUPPLEMENTATION-INEFFECTIVE — iodide enters but cannot be organised; L-thyroxine replacement is the treatment",
        ],
    },

    # -- TG -- Thyroglobulin Deficiency (Dyshormonogenesis 3) --------------------------
    {
        "gene": "TG",
        "protein": (
            "TG -- 8q24.22 AR -- Thyroglobulin-2767aa -- "
            "Dyshormonogenesis-Type-3-Serum-Tg-LOW-ABSENT-Despite-Elevated-TSH -- "
            "Goitre-Iodine-Poor-Colloid -- "
            "Elevated-TSH-PLUS-Low-Absent-Tg-PATHOGNOMONIC-Combination -- "
            "Largest-Known-Human-Secreted-Protein"
        ),
        "alias": (
            "TG (thyroglobulin); OMIM gene 188450; "
            "Thyroid dyshormonogenesis type 3 (DH3) OMIM 274700. "
            "8q24.22; 2767 aa; ~330 kDa monomer; homodimer ~660 kDa; type II integral membrane glycoprotein; secreted into follicular lumen; autosomal recessive. "
            "FUNCTION: TG encodes thyroglobulin — the LARGEST known secreted human protein: "
            "(1) Scaffold protein: provides tyrosine residues that are iodinated by TPO to form MIT and DIT; "
            "(2) Storage form: T4 and T3 stored as part of thyroglobulin in follicular colloid; "
            "(3) Prohormone: TG is the matrix from which T4 and T3 are cleaved and secreted; "
            "Synthesis of thyroid hormones: "
            "TG secreted into lumen → TPO iodination of TG-tyrosines → T4/T3 within TG → "
            "pinocytosis of TG → lysosomal proteolysis → T4 + T3 released → T4/T3 secreted into blood. "
            "In TG deficiency: "
            "TG protein absent or non-functional → no scaffold for iodination; "
            "T4/T3 production severely impaired; "
            "SERUM THYROGLOBULIN: absent or very low (important: serum Tg = released TG protein; "
            "if TG gene product absent → serum Tg LOW; "
            "this is PARADOXICAL because TSH is HIGH driving thyroid activity); "
            "TSH HIGH + serum Tg LOW/ABSENT is the PATHOGNOMONIC combination for TG deficiency; "
            "contrast: in TPO, DUOX2, or NIS defects — serum Tg may be ELEVATED (reactive); "
            "perchlorate discharge: VARIABLE (partial organification defect in some); "
            "RAI uptake: elevated (avid uptake since NIS intact). "
            "CLINICAL PRESENTATION: "
            "Goitrous congenital hypothyroidism — enlarged, normally sited thyroid; "
            "goitre may be very large — reported cases with airway compromise; "
            "hypothyroidism severity: moderate to severe; "
            "NBS detects elevated TSH; "
            "no associated SNHL (distinguishes from Pendred); "
            "no dysmorphic features (distinguishes from Bamforth-Lazarus/FOXE1). "
            "DIAGNOSIS: "
            "NBS: elevated TSH; low/undetectable T4; "
            "Thyroid ultrasound: large goitre, normally sited; "
            "Serum Tg: LOW or ABSENT despite markedly elevated TSH — PATHOGNOMONIC; "
            "(note: thyroglobulin antibodies can interfere with immunoassay — use LC-MS/MS if antibodies present); "
            "RAI uptake: elevated; "
            "Perchlorate discharge: variable — may be positive or negative; "
            "Serum Tg should rise with TSH stimulation — failure to rise is the diagnostic test; "
            "TG gene sequencing: >100 pathogenic variants reported; p.G2229R common in Afrikaner population. "
            "TREATMENT: "
            "Levothyroxine replacement: lifelong; "
            "Goitre monitoring: regular thyroid ultrasound; "
            "Surgery: rare, for tracheal compression. "
            "KEY CLINICAL FACTS: "
            "SERUM Tg LOW + TSH HIGH = TG GENE DEFECT — this combination is the biochemical fingerprint; "
            "ALL OTHER DYSHORMONOGENESES HAVE HIGH Tg (reactive) — TG deficiency uniquely has low Tg; "
            "p.G2229R IS THE AFRIKANER FOUNDER MUTATION — in South African patients with goitrous CH; "
            "PERCHLORATE DISCHARGE VARIABLE — do not use perchlorate test to rule in or out TG deficiency; use serum Tg + TSH; "
            "TG IS THE LARGEST HUMAN SECRETED PROTEIN (2767 aa) — important for understanding complex phenotype-genotype correlations."
        ),
        "age_of_onset": "Neonatal (NBS); goitre progressive through childhood",
        "inheritance": "AR",
        "locus": "8q24.22",
        "protein_size": "2767 aa",
        "key_biomarker": "Serum Tg LOW/ABSENT despite TSH markedly elevated (PATHOGNOMONIC); elevated RAI uptake; elevated TSH",
        "pathognomonic": "Goitrous CH + serum Tg LOW/ABSENT + TSH markedly HIGH — unique to TG gene defects",
        "treatment": "Levothyroxine lifelong; goitre monitoring; surgery if tracheal compression",
        "critical_flags": [
            "SERUM-Tg-LOW-PLUS-TSH-HIGH-PATHOGNOMONIC — all other dyshormonogeneses have HIGH Tg; TG deficiency uniquely low",
            "PERCHLORATE-DISCHARGE-VARIABLE — cannot use to distinguish TG deficiency; use serum Tg + TSH combination instead",
            "AFRIKANER-FOUNDER-p.G2229R — in South African patients with goitrous CH; targeted sequencing available",
            "ANTI-Tg-ANTIBODIES-INTERFERE — if Tg immunoassay falsely elevated/normal, use LC-MS/MS method",
            "LARGEST-GOITRE-RISK — TG deficiency can produce very large goitres; airway monitoring mandatory",
            "RAI-UPTAKE-ELEVATED — NIS intact; iodide enters but no TG scaffold to iodinate; contrast NIS defect (RAI absent)",
            "LIFELONG-L-THYROXINE — cannot produce endogenous T4/T3; no curative intervention beyond replacement",
        ],
    },

    # -- SLC26A4 -- Pendred Syndrome / EVA -------------------------------------------
    {
        "gene": "SLC26A4",
        "protein": (
            "SLC26A4 -- 7q22.3 AR -- Pendrin-780aa -- "
            "Pendred-Syndrome-Goitre-SNHL-EVA-PATHOGNOMONIC-TRIAD -- "
            "EVA-Enlarged-Vestibular-Aqueduct-CT-Temporal-Bone-PATHOGNOMONIC -- "
            "5-10pct-All-Hereditary-SNHL -- "
            "EVA-Screen-BEFORE-Cochlear-Implant-MANDATORY"
        ),
        "alias": (
            "SLC26A4 (solute carrier family 26 member 4, pendrin); OMIM gene 605646; "
            "Pendred syndrome OMIM 274600; DFNB4 (non-syndromic SNHL with EVA) OMIM 600791. "
            "7q22.3; 780 aa; ~86 kDa; anion transporter (chloride/iodide/formate exchanger); 11 transmembrane domains; apical membrane; autosomal recessive. "
            "FUNCTION: SLC26A4 encodes pendrin, a multifunctional anion exchanger expressed in: "
            "(1) THYROID: apical membrane of follicular cells — transports iodide (I⁻) from cell into follicular lumen (for TG iodination by TPO); "
            "(2) INNER EAR (endolymph): apical membrane of non-sensory epithelial cells (spiral ligament, utricle, cochlea, endolymphatic sac) — regulates endolymph ion composition (Cl⁻ / HCO₃⁻ homeostasis); "
            "(3) KIDNEY: intercalated cells of cortical collecting duct — HCO₃⁻ / Cl⁻ exchange. "
            "In pendrin deficiency: "
            "THYROID: iodide cannot exit follicular cell → organification defect → partial iodination of TG → partial hypothyroidism; "
            "perchlorate discharge POSITIVE (partial organification defect — iodide pools in cell); "
            "INNER EAR: endolymph composition disrupted → enlarged endolymphatic sac + duct → "
            "dilation of the vestibular aqueduct → EVA (enlarged vestibular aqueduct); "
            "EVA → sensorineural hearing loss (SNHL): fluctuating or progressive; "
            "high-frequency hearing loss earliest finding; "
            "head trauma / Valsalva manoeuvre / barotrauma can trigger acute HL deterioration. "
            "PENDRED SYNDROME = SNHL + EVA + goitre (hypothyroidism often subclinical). "
            "CLINICAL PRESENTATION: "
            "SNHL: bilateral, sensorineural; often detected in neonatal hearing screen; fluctuating with acute deteriorations; "
            "EVA: on CT/MRI temporal bone — enlarged vestibular aqueduct (>1.5 mm at midpoint); "
            "Goitre: variable — often euthyroid or subclinical hypothyroidism; overt hypothyroidism in minority; "
            "NO vestibular abnormality clinically in most (despite anatomical EVA); "
            "DFNB4: pendrin mutations without goitre — same gene, EVA + SNHL, euthyroid; spectrum overlap. "
            "DIAGNOSIS: "
            "Temporal bone CT: EVA (vestibular aqueduct >1.5 mm at midpoint) — PATHOGNOMONIC for pendrin dysfunction; "
            "Audiogram: SNHL bilateral, typically 40-70 dB range; "
            "Thyroid: TSH, T4; perchlorate discharge POSITIVE (partial organification defect); "
            "Serum Tg: elevated (reactive); "
            "SLC26A4 gene sequencing: confirms; IVS7-2A>G (c.919-2A>G) is most common European mutation; p.H723R common in Asian populations. "
            "TREATMENT: "
            "Hearing: conventional hearing aids; cochlear implant if profound; "
            "CRITICAL — EVA SCREEN BEFORE COCHLEAR IMPLANT: EVA-associated cochlear malformation may require modified surgical approach; gusher risk during implant; "
            "Avoid head trauma / contact sports: can trigger acute hearing deterioration; "
            "Thyroid: L-thyroxine if TSH elevated; goitre monitoring; "
            "Floatation devices: inner ear pressure changes (diving, Valsalva) worsen hearing. "
            "KEY CLINICAL FACTS: "
            "EVA = PATHOGNOMONIC for SLC26A4 — CT temporal bone mandatory in all SNHL children before implant; "
            "SLC26A4 ACCOUNTS FOR 5-10% OF ALL HEREDITARY SNHL — one of the most common hereditary SNHL genes; "
            "HEAD TRAUMA TRIGGERS ACUTE HL — inform school, sports staff; contact sports restriction; "
            "COCHLEAR IMPLANT OUTCOMES GOOD — but EVA must be identified pre-operatively for surgical planning; "
            "PERCHLORATE DISCHARGE POSITIVE — distinguishes from TPO only by audiogram (SNHL present in Pendred, absent in TPO)."
        ),
        "age_of_onset": "Neonatal (NBS hearing screen); thyroid may manifest in childhood/adolescence",
        "inheritance": "AR",
        "locus": "7q22.3",
        "protein_size": "780 aa",
        "key_biomarker": "CT temporal bone: EVA (vestibular aqueduct >1.5 mm) PATHOGNOMONIC; SNHL on audiogram; perchlorate discharge POSITIVE",
        "pathognomonic": "EVA on CT temporal bone + bilateral SNHL + goitre/hypothyroidism = Pendred syndrome triad",
        "treatment": "Hearing aids / cochlear implant (EVA screen pre-operatively MANDATORY); avoid head trauma; L-thyroxine if TSH elevated",
        "critical_flags": [
            "EVA-CT-TEMPORAL-BONE-PATHOGNOMONIC — enlarged vestibular aqueduct >1.5 mm; mandatory before cochlear implant",
            "EVA-SCREEN-BEFORE-COCHLEAR-IMPLANT — EVA may require modified surgical approach; gusher risk; do NOT implant without CT first",
            "HEAD-TRAUMA-TRIGGERS-ACUTE-HL — contact sports restriction mandatory; school notification; Valsalva avoidance",
            "5-10pct-ALL-HEREDITARY-SNHL — SLC26A4 is one of the most common hereditary SNHL genes worldwide",
            "PERCHLORATE-DISCHARGE-POSITIVE-PLUS-SNHL — key differentiator from TPO (perchlorate positive + SNHL absent)",
            "DFNB4-vs-PENDRED — same gene; DFNB4: EVA + SNHL without goitre; Pendred: EVA + SNHL + goitre; spectrum",
            "IVS7-2A-G-EUROPEAN-FOUNDER — most common European SLC26A4 mutation; targeted sequencing available",
        ],
    },

    # -- SLC5A5 -- Sodium-Iodide Symporter (NIS) Deficiency --------------------------
    {
        "gene": "SLC5A5",
        "protein": (
            "SLC5A5 -- 19p13.11 AR -- Sodium-Iodide-Symporter-NIS-643aa -- "
            "Iodide-Transport-Defect-ITD -- "
            "RAI-Uptake-ABSENT-VERY-LOW-PATHOGNOMONIC -- "
            "Perchlorate-Discharge-NOT-Needed-RAI-Uptake-IS-Diagnostic -- "
            "Iodine-Supplementation-MAY-Partially-Work"
        ),
        "alias": (
            "SLC5A5 (solute carrier family 5 member 5, sodium-iodide symporter, NIS); OMIM gene 601843; "
            "Iodide transport defect (ITD) OMIM 274400. "
            "19p13.11; 643 aa; ~65 kDa; 13 transmembrane domains; basolateral membrane; Na+/iodide cotransporter; autosomal recessive. "
            "FUNCTION: SLC5A5 encodes the sodium-iodide symporter (NIS), located at the BASOLATERAL membrane "
            "of thyroid follicular cells: "
            "NIS co-transports 2 Na⁺ + 1 I⁻ into the follicular cell, driven by the Na⁺ electrochemical gradient; "
            "NIS is therefore the FIRST STEP in thyroid hormone synthesis — iodide ENTRY into the follicular cell; "
            "NIS is also expressed in salivary glands, breast (lactation), stomach. "
            "In NIS/SLC5A5 deficiency: "
            "iodide CANNOT ENTER follicular cells — concentrating mechanism absent; "
            "thyroid iodide concentration approaches serum iodide (no trapping); "
            "T4/T3 production severely impaired; "
            "RAI (radioiodine, ¹²³I or ¹³¹I) UPTAKE: ABSENT or very low (thyroid-to-serum ratio approaching 1:1 vs normal >20:1); "
            "SALIVARY GLAND RAI UPTAKE: also absent (NIS expressed in salivary glands); "
            "PERCHLORATE DISCHARGE TEST: NOT useful — perchlorate displaces organified-then-trapped iodide; "
            "without iodide entry, there is nothing to discharge; "
            "therefore ITD has NEGATIVE perchlorate discharge (not positive — iodide never entered); "
            "TSH: markedly elevated; T4: low. "
            "CLINICAL PRESENTATION: "
            "Goitrous or athyreotic congenital hypothyroidism; "
            "often goitre present (chronic TSH stimulation); "
            "NBS: elevated TSH; "
            "No associated SNHL; "
            "No dysmorphic features; "
            "IMPORTANT: high dietary iodine intake can partially bypass deficiency "
            "(increases plasma iodide → passive diffusion into cell even without NIS); "
            "therefore severity may correlate inversely with dietary iodine intake. "
            "DIAGNOSIS: "
            "NBS: elevated TSH, low T4; "
            "RAI uptake (¹²³I uptake at 2h and 24h): ABSENT or very low — PATHOGNOMONIC for NIS defect; "
            "(normal: >10-15% at 24h; NIS defect: <3%); "
            "Salivary gland scintigraphy: absent salivary NIS uptake confirms; "
            "Perchlorate discharge: NEGATIVE (nothing to discharge — iodide never entered); "
            "Serum Tg: elevated (reactive); "
            "Thyroid ultrasound: goitre or normal-sized; "
            "SLC5A5 sequencing: confirms; p.T354P (T354P) is the most common reported mutation. "
            "TREATMENT: "
            "Levothyroxine replacement: standard, lifelong; "
            "High-dose iodine supplementation: MAY partially restore thyroid function by passive diffusion; "
            "not universally effective; not a substitute for L-thyroxine; "
            "Dietary iodine: avoid iodine deficiency (worsens NIS-deficient hypothyroidism). "
            "KEY CLINICAL FACTS: "
            "RAI UPTAKE ABSENT = NIS DEFECT — this is the defining test; "
            "PERCHLORATE DISCHARGE NEGATIVE in NIS defect — opposite of organification defects (TPO, DUOX2, SLC26A4); "
            "SALIVARY NIS ALSO ABSENT — salivary scintigraphy provides additional non-thyroidal confirmation; "
            "HIGH DIETARY IODINE MAY HELP — unlike organification defects where iodine is irrelevant; "
            "RAI THERAPY CANNOT BE USED for thyroid disease in NIS deficient patients — RAI will not be taken up."
        ),
        "age_of_onset": "Neonatal (NBS detection)",
        "inheritance": "AR",
        "locus": "19p13.11",
        "protein_size": "643 aa",
        "key_biomarker": "RAI (¹²³I) thyroid uptake ABSENT or very low (<3% at 24h) PATHOGNOMONIC; elevated TSH; salivary NIS uptake also absent",
        "pathognomonic": "Goitrous CH + RAI uptake ABSENT + perchlorate discharge NEGATIVE (not positive — iodide never entered)",
        "treatment": "Levothyroxine lifelong; high-dose iodine supplementation may partially help; avoid iodine deficiency",
        "critical_flags": [
            "RAI-UPTAKE-ABSENT-PATHOGNOMONIC — thyroid uptake <3% at 24h; this is the defining test for NIS/SLC5A5 defect",
            "PERCHLORATE-DISCHARGE-NEGATIVE-NOT-POSITIVE — NIS defect has no iodide to discharge; opposite of TPO/DUOX2/SLC26A4",
            "SALIVARY-NIS-ALSO-ABSENT — salivary gland scintigraphy confirms NIS deficiency outside the thyroid",
            "RAI-THERAPY-CANNOT-WORK — no iodide trapping; RAI ablation or therapy ineffective in NIS defect patients",
            "HIGH-IODINE-MAY-PARTIALLY-HELP — passive diffusion bypasses NIS at high iodide concentrations; dietary iodine matters",
            "p-T354P-MOST-COMMON-MUTATION — targeted sequencing available in confirmed clinical diagnosis",
            "DISTINGUISH-FROM-ORGANIFICATION-DEFECTS — NIS: RAI absent + perchlorate negative; organification defects: RAI elevated + perchlorate positive",
        ],
    },

    # -- DUOX2 -- Dual Oxidase 2 Deficiency (Dyshormonogenesis 6) --------------------
    {
        "gene": "DUOX2",
        "protein": (
            "DUOX2 -- 15q21.1 AR -- Dual-Oxidase-2-1548aa -- "
            "Dyshormonogenesis-Type-6-H2O2-Generation-Failure -- "
            "Most-Common-Dyshormonogenesis-Worldwide-Japan-Asia -- "
            "Perchlorate-Discharge-POSITIVE -- "
            "Monoallelic-DUOX2-Transient-Neonatal-Hypothyroidism-Most-Common-Cause"
        ),
        "alias": (
            "DUOX2 (dual oxidase 2); OMIM gene 606759; "
            "Thyroid dyshormonogenesis type 6 (DH6) OMIM 607200. "
            "15q21.1; 1548 aa; ~175 kDa; NADPH oxidase; requires DUOXA2 maturation factor; apical membrane; autosomal recessive (biallelic) or dominant-negative heterozygous. "
            "FUNCTION: DUOX2 encodes dual oxidase 2, a NADPH-oxidase-type enzyme at the APICAL membrane "
            "of thyroid follicular cells: "
            "DUOX2 generates H₂O₂ (hydrogen peroxide) which is the essential co-oxidant for TPO; "
            "TPO uses H₂O₂ (from DUOX2) to oxidise iodide → reactive iodine → TG iodination → T3/T4; "
            "DUOXA2 (dual oxidase maturation factor 2, gene DUOXA2) is essential for DUOX2 trafficking to the apical membrane; "
            "WITHOUT DUOX2: H₂O₂ is absent → TPO cannot oxidise iodide → organification fails → perchlorate discharge positive. "
            "In DUOX2 deficiency: "
            "organification defect (same functional result as TPO deficiency, different mechanism); "
            "perchlorate discharge POSITIVE; "
            "BIALLELIC LOSS-OF-FUNCTION (full DH6): permanent hypothyroidism; goitre; "
            "MONOALLELIC (single pathogenic variant): "
            "REDUCED DUOX2 activity → TRANSIENT NEONATAL HYPOTHYROIDISM (TNH) — most common identified genetic cause of TNH; "
            "TSH elevated transiently; T4 normalises; often misclassified as 'iodine deficiency' or 'idiopathic'; "
            "DUOX2 monoallelic variants are VERY COMMON (1-2% of population in some regions); "
            "may manifest as permanent hypothyroidism under iodine-deficient conditions. "
            "CLINICAL PRESENTATION: "
            "Biallelic DUOX2: "
            "permanent goitrous congenital hypothyroidism; NBS detected; "
            "SNHL absent; no dysmorphia; normal-sited enlarged thyroid; "
            "Monoallelic DUOX2 (heterozygous): "
            "transient neonatal hypothyroidism — TSH elevated on NBS → normalises by 1-3 years; "
            "may be misdiagnosed as maternal iodine deficiency; "
            "hypothyroidism may recur in iodine-deficient environments. "
            "DIAGNOSIS: "
            "NBS: elevated TSH, low/normal T4; "
            "Perchlorate discharge: POSITIVE (organification defect); "
            "RAI uptake: elevated (NIS intact); "
            "Thyroid ultrasound: goitre (biallelic) or normal size (monoallelic); "
            "DUOX2 gene sequencing: most common in Japan — p.L1067S, p.Y1553X; Korean: p.R376W; "
            "DUOXA2 gene also screened (maturation factor). "
            "TREATMENT: "
            "Biallelic: lifelong levothyroxine; "
            "Monoallelic TNH: may discontinue L-thyroxine at 2-3 years after retesting; recurrence under iodine deficiency; "
            "Iodine supplementation: NOT curative (organification defect — iodide enters but cannot be oxidised). "
            "KEY CLINICAL FACTS: "
            "MOST COMMON DYSHORMONOGENESIS WORLDWIDE — especially in Japan, Korea, and parts of Asia; "
            "MONOALLELIC = TRANSIENT NEONATAL HYPOTHYROIDISM — most common genetic cause of TNH; "
            "TRANSIENT CH SHOULD BE RECHALLENGED AT AGE 2-3 — stop L-thyroxine, retest TSH/T4; "
            "PERCHLORATE DISCHARGE POSITIVE — same as TPO deficiency; DUOX2 vs TPO distinguished by gene testing; "
            "DUOXA2 ALSO SCREENED — maturation factor; DUOXA2 variants cause same phenotype as DUOX2."
        ),
        "age_of_onset": "Neonatal (NBS); biallelic permanent; monoallelic transient",
        "inheritance": "AR (biallelic) / dominant-negative heterozygous (transient)",
        "locus": "15q21.1",
        "protein_size": "1548 aa",
        "key_biomarker": "Perchlorate discharge POSITIVE; elevated RAI uptake; TSH elevated; biallelic: permanent; monoallelic: transient",
        "pathognomonic": "Goitrous CH + perchlorate discharge POSITIVE — biallelic permanent; monoallelic transient neonatal hypothyroidism (TNH)",
        "treatment": "Biallelic: lifelong L-thyroxine; monoallelic TNH: retest at age 2-3 for possible discontinuation",
        "critical_flags": [
            "MOST-COMMON-DYSHORMONOGENESIS-WORLDWIDE — especially Japan/Korea/Asia; DUOX2 mutations most prevalent cause globally",
            "MONOALLELIC-CAUSES-TRANSIENT-NH — most common identified genetic cause of transient neonatal hypothyroidism",
            "TRANSIENT-CH-RETEST-AGE-2-3 — stop L-thyroxine at 2-3 years; recheck TSH/T4; may truly discontinue",
            "IODINE-DEFICIENCY-UNMASKS-MONOALLELIC — heterozygous DUOX2 carriers develop overt hypothyroidism under iodine deficiency",
            "DUOXA2-ALSO-SCREENED — maturation factor for DUOX2; DUOXA2 variants cause indistinguishable phenotype",
            "PERCHLORATE-POSITIVE-SAME-AS-TPO — functional organification defect; gene sequencing distinguishes DUOX2 from TPO",
            "JAPANESE-FOUNDER-L1067S-Y1553X — common variants in Japanese population; targeted sequencing available",
        ],
    },

    # -- TSHR -- Thyrotropin Receptor (TSH Resistance / Neonatal Hyperthyroidism) -----
    {
        "gene": "TSHR",
        "protein": (
            "TSHR -- 14q31.1 AR-LOF-AD-GOF -- Thyrotropin-Receptor-764aa -- "
            "AR-LOF-TSH-Resistance-CH-HIGH-TSH-NORMAL-Gland-Ultrasound -- "
            "AD-GOF-Familial-Non-Autoimmune-Hyperthyroidism-Neonatal-Hyperthyroidism -- "
            "Most-Common-Cause-TSH-Unresponsive-Congenital-Hypothyroidism"
        ),
        "alias": (
            "TSHR (thyrotropin receptor, thyroid-stimulating hormone receptor); OMIM gene 603372; "
            "CH due to TSHR inactivation OMIM 275200; familial non-autoimmune hyperthyroidism OMIM 609152. "
            "14q31.1; 764 aa; ~87 kDa; G-protein coupled receptor; serpentine transmembrane receptor; "
            "7 transmembrane domains; large extracellular leucine-rich domain for TSH binding; "
            "autosomal recessive (LOF) or autosomal dominant (GOF). "
            "FUNCTION: TSHR encodes the thyrotropin receptor, expressed on thyroid follicular cells: "
            "TSH (thyrotropin from pituitary) binds TSHR → Gs protein → adenylyl cyclase → cAMP ↑ → "
            "activates PKA → stimulates: "
            "(1) NIS (iodide uptake); "
            "(2) TPO, TG, DUOX2 gene expression; "
            "(3) T3/T4 synthesis and secretion; "
            "(4) thyrocyte proliferation (goitre). "
            "TWO DISEASE MECHANISMS: "
            "LOSS-OF-FUNCTION (AR/biallelic — TSH RESISTANCE): "
            "TSH cannot activate TSHR → downstream signalling absent → "
            "TSH effect lost despite normal TSH production; "
            "TSH markedly elevated but thyroid DOES NOT RESPOND (does not grow, does not make T4); "
            "THYROID ULTRASOUND: NORMAL-SIZED or hypoplastic gland (no TSH-driven growth); "
            "this is PATHOGNOMONIC — elevated TSH + NORMAL/SMALL gland (goitre ABSENT); "
            "contrast: TPO/DUOX2/TG deficiencies where goitre IS present; "
            "severity: proportional to residual TSHR activity; mild LOF → subclinical hypothyroidism; "
            "GAIN-OF-FUNCTION (AD/somatic — NON-AUTOIMMUNE HYPERTHYROIDISM): "
            "constitutively active TSHR → TSHR signals without TSH → autonomous thyroid activity; "
            "NEONATAL HYPERTHYROIDISM: neonatal period; tachycardia, jitteriness, feeding problems, goitre; "
            "FAMILIAL NON-AUTOIMMUNE HYPERTHYROIDISM: family history of hyperthyroidism without Graves antibodies; "
            "TSH suppressed; T4/T3 elevated; TRAb NEGATIVE (distinguishes from Graves disease); "
            "SOMATIC TSHR GOF: toxic thyroid adenoma / hot nodule in adults — unilateral hyperfunctioning nodule. "
            "CLINICAL PRESENTATION: "
            "AR LOF: CH + normal-sited gland + elevated TSH + small/normal gland; no goitre; no SNHL; "
            "AD GOF: neonatal hyperthyroidism → tachycardia, poor feeding, prematurity; "
            "family history of AD hyperthyroidism; TRAb negative = Graves excluded. "
            "DIAGNOSIS: "
            "AR LOF: TSH markedly elevated; T4 low; thyroid ultrasound NORMAL-SIZED/HYPOPLASTIC (no goitre); "
            "RAI uptake: LOW (thyroid unresponsive to TSH → NIS not upregulated); "
            "TRAb: NEGATIVE (autoimmune excluded); "
            "TSH stimulation test: failure to respond = TSHR resistance; "
            "TSHR gene sequencing: confirms LOF or GOF mutation. "
            "TREATMENT: "
            "AR LOF: lifelong L-thyroxine; dose by TSH suppression target; "
            "AD GOF neonatal hyperthyroidism: propylthiouracil / carbimazole; propranolol for cardiovascular; "
            "definitive: thyroidectomy or RAI ablation for familial GOF cases (lifelong anti-thyroid drugs otherwise). "
            "KEY CLINICAL FACTS: "
            "MOST COMMON CAUSE OF TSH-UNRESPONSIVE CH — TSHR LOF is the top genetic cause; "
            "ELEVATED TSH + NORMAL GLAND ON ULTRASOUND = TSHR DEFECT — goitre absence is the key; "
            "RAI UPTAKE LOW in TSHR LOF — contrast organification defects where RAI is elevated; "
            "TRAB NEGATIVE in GOF familial hyperthyroidism — Graves excluded; key for diagnosis; "
            "SOMATIC TSHR GOF = HOT NODULE — toxic adenoma on scan; surgical or RAI ablation."
        ),
        "age_of_onset": "Neonatal (LOF — CH on NBS; GOF — neonatal hyperthyroidism)",
        "inheritance": "AR (LOF: TSH resistance); AD (GOF: familial hyperthyroidism)",
        "locus": "14q31.1",
        "protein_size": "764 aa",
        "key_biomarker": "AR LOF: TSH elevated + T4 low + NORMAL/SMALL thyroid on ultrasound (no goitre); GOF: TSH suppressed + T3/T4 elevated + TRAb NEGATIVE",
        "pathognomonic": "Elevated TSH + normal-sized or hypoplastic thyroid on ultrasound (goitre ABSENT) = TSHR LOF; TRAb-negative hyperthyroidism = TSHR GOF",
        "treatment": "LOF: lifelong L-thyroxine; GOF: anti-thyroid drugs acutely, definitive thyroidectomy or RAI ablation",
        "critical_flags": [
            "ELEVATED-TSH-PLUS-NORMAL-GLAND-ULTRASOUND-PATHOGNOMONIC — goitre ABSENT in TSHR LOF; opposite of TPO/DUOX2/TG defects",
            "MOST-COMMON-TSH-UNRESPONSIVE-CH — TSHR inactivation is top cause of CH without gland absence",
            "RAI-UPTAKE-LOW-IN-LOF — NIS not upregulated (TSHR cannot signal); contrast organification defects (RAI elevated)",
            "TRAB-NEGATIVE-DISTINGUISHES-FROM-GRAVES — GOF familial hyperthyroidism: TRAb absent; Graves: TRAb positive",
            "SOMATIC-TSHR-GOF-HOT-NODULE — toxic adenoma in adults; single hyperfunctioning nodule on scan; surgery or RAI",
            "NEONATAL-TSHR-GOF-EMERGENCY — tachycardia, hyperthermia, poor feeding in neonate; propranolol + antithyroid drugs urgently",
            "BIALLELIC-MILD-LOF-SUBCLINICAL-HYPOTHYROIDISM — compound heterozygotes with partial TSHR function have subclinical CH only",
        ],
    },

    # -- PAX8 -- Thyroid Transcription Factor / Dysgenesis --------------------------------
    {
        "gene": "PAX8",
        "protein": (
            "PAX8 -- 2q14.1 AD -- Paired-Box-8-Thyroid-Transcription-Factor-457aa -- "
            "Thyroid-Dysgenesis-Ectopic-Hypoplastic-Gland -- "
            "Renal-Anomalies-50pct-Kidney-Plus-Hypothyroidism-PATHOGNOMONIC -- "
            "Most-Common-Genetic-Cause-Thyroid-Dysgenesis-After-TSHR -- "
            "Variable-Expressivity-Autosomal-Dominant"
        ),
        "alias": (
            "PAX8 (paired box 8); OMIM gene 167415; "
            "Thyroid dysgenesis 2 (CHNG2) OMIM 218700 (PAX8 mutations in thyroid dysgenesis). "
            "2q14.1; 457 aa; ~48 kDa; nuclear transcription factor; paired box domain; "
            "autosomal dominant with variable expressivity and incomplete penetrance. "
            "FUNCTION: PAX8 is a transcription factor critical for thyroid development and differentiation: "
            "(1) Regulates TG (thyroglobulin) gene expression; "
            "(2) Regulates NIS (SLC5A5) gene expression; "
            "(3) Regulates TPO gene expression; "
            "(4) Required for thyroid follicle formation and maintenance; "
            "(5) Also expressed in kidney — required for renal tubular development. "
            "In PAX8 deficiency: "
            "thyroid fails to develop normally (dysgenesis) — "
            "may result in: agenesis (no gland), ectopia (gland in wrong position — lingual, sublingual), "
            "or hypoplasia (small, undescended gland); "
            "RENAL ANOMALIES (in ~50% of PAX8 mutation carriers): "
            "PAX8 expressed in developing kidney; "
            "renal anomalies include: horseshoe kidney, duplex collecting system, renal hypoplasia, VUR; "
            "COMBINATION OF HYPOTHYROIDISM + RENAL ANOMALY in a child/family = PAX8 UNTIL PROVEN OTHERWISE; "
            "VARIABLE EXPRESSIVITY: same PAX8 mutation within a family may cause: "
            "severe CH in one individual, subclinical hypothyroidism in another, normal thyroid in a third; "
            "AD dominant-negative effect — one pathogenic allele impairs PAX8 function. "
            "CLINICAL PRESENTATION: "
            "Congenital hypothyroidism: variable severity; NBS detected or missed if subclinical; "
            "Thyroid: ectopic (lingual thyroid most common) or hypoplastic; "
            "Renal: incidental finding or symptomatic; UTI, hydronephrosis, VUR; "
            "Family history: may reveal multiple hypothyroid relatives with different severities; "
            "No SNHL; no dysmorphic features (distinguishes from Bamforth-Lazarus/FOXE1). "
            "DIAGNOSIS: "
            "NBS: elevated TSH (if severe) or normal (if mild expressivity); "
            "Thyroid scintigraphy or ultrasound: ectopic gland (lingual thyroid) or hypoplastic/absent; "
            "Renal ultrasound: MANDATORY in all PAX8 mutation patients (50% have renal anomaly); "
            "FAMILY HISTORY SCREENING: first-degree relatives should have TSH checked; "
            "PAX8 gene sequencing: confirms mutation; p.R31H and others (>30 pathogenic variants reported). "
            "TREATMENT: "
            "Levothyroxine replacement: lifelong for hypothyroid cases; "
            "Monitor euthyroid carriers for thyroid function changes over time; "
            "Renal anomalies: management per anomaly type (UTI prophylaxis, nephrology referral for VUR); "
            "Family cascade testing: first-degree relatives TSH + PAX8 testing. "
            "KEY CLINICAL FACTS: "
            "RENAL ANOMALY + HYPOTHYROIDISM = PAX8 UNTIL PROVEN OTHERWISE — renal ultrasound in ALL CH families; "
            "LINGUAL THYROID is the most common ectopic form — visible at tongue base; "
            "FAMILY SCREENING MANDATORY — variable expressivity means euthyroid parents may carry PAX8 variant; "
            "ECTOPIC THYROID IS STILL FUNCTIONAL — do NOT ablate lingual thyroid in PAX8 patients; may be only thyroid tissue; "
            "DO NOT ABLATE ECTOPIC THYROID — if the ectopic thyroid is the only thyroid tissue, ablation → permanent agenesis + lifelong L-T4 mandatory."
        ),
        "age_of_onset": "Neonatal (moderate/severe) or later (mild expressivity in same family)",
        "inheritance": "AD (variable expressivity, dominant-negative / haploinsufficiency)",
        "locus": "2q14.1",
        "protein_size": "457 aa",
        "key_biomarker": "Ectopic (lingual) or hypoplastic thyroid on scan; renal anomaly on ultrasound; elevated TSH; normal/low T4",
        "pathognomonic": "Congenital hypothyroidism + renal anomaly (horseshoe kidney/hypoplasia) = PAX8; lingual thyroid is most common ectopic form",
        "treatment": "Lifelong L-thyroxine; renal anomaly management; family cascade TSH screening; do NOT ablate lingual thyroid",
        "critical_flags": [
            "RENAL-ANOMALY-PLUS-CH-PATHOGNOMONIC-PAX8 — 50% of PAX8 carriers have renal anomaly; renal ultrasound mandatory in all CH",
            "DO-NOT-ABLATE-ECTOPIC-THYROID — lingual thyroid may be only thyroid tissue; ablation = permanent agenesis",
            "VARIABLE-EXPRESSIVITY-SAME-FAMILY — euthyroid parents may carry PAX8 mutation; family cascade testing mandatory",
            "FAMILY-TSH-SCREENING-MANDATORY — AD inheritance; first-degree relatives need TSH and PAX8 testing",
            "LINGUAL-THYROID-VISIBLE — tongue base thyroid mass; do NOT excise without scintigraphy confirmation of ectopia type",
            "ECTOPIC-THYROID-IS-FUNCTIONAL — provides some T4 even if ectopic; RAI ablation removes only functioning tissue",
            "PAX8-NBS-MAY-BE-MISSED — mild expressivity = subclinical hypothyroidism = TSH within NBS cutoff; clinical vigilance needed",
        ],
    },

    # -- FOXE1 -- Bamforth-Lazarus Syndrome / Thyroid Agenesis -----------------------
    {
        "gene": "FOXE1",
        "protein": (
            "FOXE1 -- 9q22.33 AR -- Forkhead-Box-E1-TTF2-373aa -- "
            "Bamforth-Lazarus-Syndrome -- "
            "Thyroid-Agenesis-PLUS-Cleft-Palate-PLUS-Spiky-Hair-PATHOGNOMONIC-TRIAD -- "
            "Only-Hereditary-Thyroid-Agenesis-WITH-Dysmorphic-Features -- "
            "Choanal-Atresia-In-Some-Bifid-Epiglottis"
        ),
        "alias": (
            "FOXE1 (forkhead box E1, thyroid transcription factor 2, TTF2); OMIM gene 602617; "
            "Bamforth-Lazarus syndrome OMIM 241850. "
            "9q22.33; 373 aa; ~42 kDa; nuclear transcription factor; forkhead/winged-helix DNA-binding domain; "
            "autosomal recessive (biallelic LOF). "
            "FUNCTION: FOXE1 (TTF2) is a forkhead transcription factor required for: "
            "(1) THYROID DEVELOPMENT: required for thyroid gland descent from foramen caecum to anterior neck; "
            "without FOXE1: thyroid primordium fails to migrate → thyroid agenesis or ectopia; "
            "FOXE1 is also required for TONGUE MUSCLE development and palatogenesis; "
            "(2) CLEFT PALATE: FOXE1 expressed in palatal epithelium — required for palate fusion; "
            "(3) HAIR SHAFT: FOXE1 expressed in hair follicles — mutations → spiky, wiry hair; "
            "(4) EPIGLOTTIS: required for epiglottis development — bifid epiglottis in some cases; "
            "(5) CHOANAE: required for choanal development — choanal atresia in some. "
            "In FOXE1 deficiency: "
            "THYROID: complete agenesis (gland absent on scintigraphy); "
            "profound neonatal hypothyroidism; "
            "CLEFT PALATE: midline cleft (secondary palate); "
            "SPIKY HAIR: characteristic stiff, upright, wiry hair — pathognomonic cosmetic sign; "
            "CHOANAL ATRESIA: bilateral in some (requires surgical correction in neonates for airway); "
            "BIFID EPIGLOTTIS: endoscopic finding. "
            "BAMFORTH-LAZARUS SYNDROME = THYROID AGENESIS + CLEFT PALATE + SPIKY HAIR ± CHOANAL ATRESIA. "
            "CLINICAL PRESENTATION: "
            "Neonatal: severe hypothyroidism + cleft palate (surgical repair needed) ± choanal atresia (airway emergency); "
            "Thyroid scintigraphy: complete thyroid ABSENCE (no uptake anywhere); "
            "Hair: spiky, stiff, kinky texture from birth — easily identified; "
            "No SNHL; "
            "This is the ONLY hereditary thyroid agenesis syndrome with characteristic dysmorphic features; "
            "PAX8 causes dysgenesis without dysmorphia; TSHR LOF causes hypoplasia without dysmorphia; "
            "FOXE1 causes agenesis WITH cleft palate + spiky hair. "
            "DIAGNOSIS: "
            "Clinical: thyroid agenesis + cleft palate + spiky hair (TRIAD PATHOGNOMONIC); "
            "NBS: very low T4, markedly elevated TSH; "
            "Thyroid scintigraphy: ABSENT thyroid (no uptake in neck, no ectopia); "
            "ENT/craniofacial assessment: choanal atresia exclusion (nasopharyngoscopy/CT); "
            "FOXE1 gene sequencing: A65V and other forkhead domain mutations most common. "
            "TREATMENT: "
            "Neonatal hypothyroidism: URGENT — L-thyroxine immediately after diagnosis; "
            "Cleft palate: surgical repair at 6-12 months; "
            "Choanal atresia: surgical correction if bilateral (neonatal airway emergency); "
            "Lifelong L-thyroxine (no thyroid tissue — permanent agenesis); "
            "Speech therapy post-palate repair; "
            "Genetic counselling: AR (1:4 recurrence risk for siblings). "
            "KEY CLINICAL FACTS: "
            "SPIKY HAIR + CLEFT PALATE + SEVERE CH = FOXE1 UNTIL PROVEN OTHERWISE — the triad is unmistakable; "
            "MOST SEVERE CONGENITAL HYPOTHYROIDISM — complete agenesis; no endogenous T4 whatsoever; "
            "CHOANAL ATRESIA IS AN AIRWAY EMERGENCY IN NEONATES — bilateral choanal atresia = obligate nasal breather cannot breathe; "
            "ONLY HEREDITARY SYNDROME WITH DYSMORPHIA + THYROID AGENESIS — PAX8 and TSHR do not cause cleft palate; "
            "FOXE1 ALSO LINKED TO THYROID CANCER SUSCEPTIBILITY — FOXE1 polymorphisms associated with papillary thyroid cancer in GWAS (different from Bamforth-Lazarus)."
        ),
        "age_of_onset": "Neonatal (birth — cleft palate + choanal atresia immediate; severe CH on NBS)",
        "inheritance": "AR (Bamforth-Lazarus); AD (thyroid cancer susceptibility polymorphisms — different mechanism)",
        "locus": "9q22.33",
        "protein_size": "373 aa",
        "key_biomarker": "Thyroid scintigraphy ABSENT (complete agenesis); markedly elevated TSH; very low T4; cleft palate; spiky hair",
        "pathognomonic": "Thyroid agenesis + cleft palate + spiky hair = Bamforth-Lazarus syndrome (FOXE1) — the only hereditary thyroid agenesis with dysmorphia",
        "treatment": "Urgent L-thyroxine; cleft palate surgery 6-12 months; choanal atresia surgery if bilateral (neonatal airway emergency); lifelong L-T4",
        "critical_flags": [
            "CLEFT-PALATE-PLUS-SPIKY-HAIR-PLUS-CH-PATHOGNOMONIC-TRIAD — Bamforth-Lazarus; FOXE1 is the only agenesis syndrome with this triad",
            "CHOANAL-ATRESIA-NEONATAL-AIRWAY-EMERGENCY — bilateral choanal atresia = cannot breathe through nose; urgent ENT surgery",
            "MOST-SEVERE-CH-COMPLETE-AGENESIS — no thyroid tissue; no endogenous T4; highest L-thyroxine requirement",
            "ONLY-HEREDITARY-AGENESIS-WITH-DYSMORPHIA — PAX8 and TSHR cause dysgenesis without cleft palate; FOXE1 unique",
            "AR-RECURRENCE-RISK-25pct — counsel parents: 1:4 recurrence; prenatal testing available",
            "SPIKY-HAIR-PATHOGNOMONIC-CLINICAL-SIGN — visible from birth; wiry, stiff, upright hair; easy bedside recognition",
            "FOXE1-CANCER-SUSCEPTIBILITY-SEPARATE — GWAS polymorphisms in FOXE1 linked to papillary thyroid cancer; distinct from Bamforth-Lazarus LOF",
        ],
    },
]


def _make_cohort(gene_entry: dict, seed: int, n: int = 40) -> list:
    rng = random.Random(seed)
    ages = [round(rng.gauss(1.5, 2.0), 1) for _ in range(n)]
    ages = [max(0.0, min(16.0, a)) for a in ages]
    sexes = [rng.choice(["M", "F"]) for _ in range(n)]
    # SLC26A4 (Pendred): slight female predominance in published series
    if gene_entry["gene"] == "SLC26A4":
        sexes = [rng.choices(["M", "F"], weights=[2, 3])[0] for _ in range(n)]
    severities = [rng.choice(["mild", "moderate", "severe"]) for _ in range(n)]
    # FOXE1 Bamforth-Lazarus: all severe (complete agenesis)
    if gene_entry["gene"] == "FOXE1":
        severities = ["severe"] * n
    # DUOX2 monoallelic often mild/transient
    if gene_entry["gene"] == "DUOX2":
        severities = [rng.choice(["mild", "moderate", "severe"], )[0] for _ in range(n)]
    return [
        {
            "patient_id": f"{gene_entry['gene']}-{i+1:03d}",
            "gene": gene_entry["gene"],
            "age_at_diagnosis_yr": ages[i],
            "sex": sexes[i],
            "severity": severities[i],
            "inheritance": gene_entry["inheritance"],
            "locus": gene_entry["locus"],
        }
        for i in range(n)
    ]


def overview() -> dict:
    all_patients = []
    for idx, g in enumerate(THYROID_GENES):
        cohort = _make_cohort(g, SEED_BASE + idx)
        all_patients.extend(cohort)

    total = len(all_patients)
    gene_counts = {}
    for p in all_patients:
        gene_counts[p["gene"]] = gene_counts.get(p["gene"], 0) + 1

    age_vals = [p["age_at_diagnosis_yr"] for p in all_patients]
    avg_age = round(sum(age_vals) / len(age_vals), 1)
    severe_count = sum(1 for p in all_patients if p["severity"] == "severe")

    gene_summary = []
    for g in THYROID_GENES:
        gene_summary.append({
            "gene": g["gene"],
            "locus": g["locus"],
            "protein_size": g["protein_size"],
            "inheritance": g["inheritance"],
            "key_biomarker": g["key_biomarker"],
            "pathognomonic": g["pathognomonic"],
            "treatment": g["treatment"],
            "n_patients": gene_counts.get(g["gene"], 0),
            "critical_flags": g["critical_flags"],
        })

    return {
        "atlas": "Hereditary-Thyroid-Disorder-Atlas",
        "subtitle": (
            "Complete 8-Gene Hereditary Thyroid Disorder Atlas — "
            "TPO-933aa-2p25.3-AR-Dyshormonogenesis-2A-Perchlorate-Discharge-POSITIVE-SNHL-ABSENT | "
            "TG-2767aa-8q24.22-AR-DH3-Serum-Tg-LOW-ABSENT-Despite-Elevated-TSH-PATHOGNOMONIC | "
            "SLC26A4-780aa-7q22.3-AR-Pendred-Goitre-SNHL-EVA-TRIAD-EVA-Screen-Before-Cochlear-Implant | "
            "SLC5A5-643aa-19p13.11-AR-NIS-Iodide-Transport-Defect-RAI-Uptake-ABSENT-PATHOGNOMONIC | "
            "DUOX2-1548aa-15q21.1-AR-DH6-Most-Common-Dyshormonogenesis-Worldwide-Transient-NH-Monoallelic | "
            "TSHR-764aa-14q31.1-AR-LOF-TSH-Resistance-CH-Normal-Gland-AD-GOF-Familial-Hyperthyroidism | "
            "PAX8-457aa-2q14.1-AD-Thyroid-Dysgenesis-Renal-Anomaly-50pct-Pathognomonic-Lingual-Thyroid | "
            "FOXE1-373aa-9q22.33-AR-Bamforth-Lazarus-Agenesis-Cleft-Palate-Spiky-Hair-PATHOGNOMONIC-TRIAD | "
            "320-Patient-Aggregate-8x40-seeds-1886-1893"
        ),
        "aggregate_stats": {
            "total_patients": total,
            "genes_covered": len(THYROID_GENES),
            "avg_age_at_diagnosis_yr": avg_age,
            "severe_cases_pct": round(100 * severe_count / total, 1),
            "seed_range": f"{SEED_BASE}–{SEED_BASE + len(THYROID_GENES) - 1}",
        },
        "gene_summary": gene_summary,
        "key_clinical_distinctions": [
            "TPO-vs-SLC26A4-Pendred: both have goitre + perchlorate discharge POSITIVE — audiogram distinguishes: SNHL absent (TPO) vs present (Pendred/SLC26A4)",
            "TPO-vs-SLC5A5-NIS: TPO has RAI uptake ELEVATED + perchlorate POSITIVE; NIS has RAI uptake ABSENT + perchlorate NEGATIVE",
            "TPO-vs-DUOX2: both have perchlorate POSITIVE + elevated RAI — gene testing distinguishes; DUOX2 monoallelic → transient NH; biallelic → permanent",
            "TG-vs-all-others: serum Tg LOW despite TSH HIGH — unique to TG defect; all other DH have HIGH serum Tg (reactive)",
            "TSHR-LOF-vs-dyshormonogeneses: TSHR LOF has ELEVATED TSH + NORMAL/SMALL gland (no goitre) — goitre ABSENT is the key; DH always have goitre",
            "PAX8-vs-FOXE1: PAX8 — dysgenesis without dysmorphia + renal anomaly; FOXE1 — agenesis WITH cleft palate + spiky hair (Bamforth-Lazarus)",
            "FOXE1-vs-TSHR-vs-PAX8: FOXE1 is the ONLY agenesis syndrome with dysmorphic features; others cause structural thyroid change without cleft/hair",
        ],
    }


def breakdown() -> dict:
    result = []
    for idx, g in enumerate(THYROID_GENES):
        cohort = _make_cohort(g, SEED_BASE + idx)
        severities = {}
        for p in cohort:
            severities[p["severity"]] = severities.get(p["severity"], 0) + 1
        result.append({
            "gene": g["gene"],
            "protein": g["protein"],
            "locus": g["locus"],
            "protein_size": g["protein_size"],
            "inheritance": g["inheritance"],
            "key_biomarker": g["key_biomarker"],
            "pathognomonic": g["pathognomonic"],
            "treatment": g["treatment"],
            "critical_flags": g["critical_flags"],
            "severity_distribution": severities,
            "n_patients": len(cohort),
            "patients": cohort[:5],
        })
    return {"genes": result, "total_genes": len(THYROID_GENES)}


def definitions() -> dict:
    return {
        "atlas": "Hereditary-Thyroid-Disorder-Atlas",
        "genes": [
            {
                "gene": g["gene"],
                "definition": g["alias"],
                "locus": g["locus"],
                "protein_size": g["protein_size"],
                "inheritance": g["inheritance"],
                "age_of_onset": g["age_of_onset"],
                "critical_flags": g["critical_flags"],
            }
            for g in THYROID_GENES
        ],
        "glossary": {
            "CH": "Congenital hypothyroidism — elevated TSH + low T4 at birth; detected by NBS; causes of dysgenesis: TSHR, PAX8, FOXE1; causes of DH: TPO, TG, SLC5A5, SLC26A4, DUOX2",
            "DH": "Dyshormonogenesis — thyroid hormone synthesis defect (not gland structure); gland present + goitre; causes: TPO, TG, SLC5A5, SLC26A4, DUOX2",
            "Dysgenesis": "Abnormal thyroid gland development — agenesis (FOXE1), ectopia (PAX8, FOXE1), hypoplasia (TSHR LOF, PAX8); NIS/iodide uptake may be normal",
            "EVA": "Enlarged vestibular aqueduct — vestibular aqueduct >1.5 mm at midpoint on CT temporal bone; PATHOGNOMONIC for SLC26A4 (pendrin) dysfunction",
            "NBS": "Newborn screening — heel prick blood TSH (and/or T4) at 24-72h; detects CH; threshold varies by country and gestational age",
            "NIS": "Sodium-iodide symporter (SLC5A5) — basolateral thyroid membrane; concentrates iodide into follicular cell; absent/low RAI uptake if defective",
            "Perchlorate discharge test": "After radioiodine administration, ClO4⁻ (perchlorate) displaces unorganified iodide; >10% discharge = organification defect (TPO, DUOX2, SLC26A4 positive; NIS/SLC5A5 negative)",
            "Pendrin": "SLC26A4 — apical anion exchanger in thyroid (iodide→lumen), inner ear (Cl⁻/HCO₃⁻ homeostasis), kidney; deficiency causes Pendred syndrome",
            "RAI uptake": "Radioiodine (¹²³I) thyroid uptake at 24h — elevated in organification defects (gland avid for iodide); absent in NIS defect; low in TSHR LOF",
            "Serum Tg": "Serum thyroglobulin — ELEVATED (reactive) in most CH; UNIQUELY LOW in TG gene defect despite TSH stimulation",
            "TNH": "Transient neonatal hypothyroidism — TSH elevated on NBS but normalises by age 2-3; DUOX2 monoallelic is most common identified genetic cause",
            "TSHR LOF": "TSH receptor loss-of-function — elevated TSH + normal/small gland (no goitre); TSH cannot signal; most common genetic TSH-unresponsive CH cause",
            "TSHR GOF": "TSH receptor gain-of-function — constitutively active; familial non-autoimmune hyperthyroidism; neonatal hyperthyroidism; TRAb NEGATIVE",
            "TTF2": "Thyroid transcription factor 2 = FOXE1 — forkhead domain TF; required for thyroid descent; FOXE1 mutations → Bamforth-Lazarus syndrome",
            "TPO": "Thyroid peroxidase — catalyses iodide oxidation + tyrosine iodination + coupling; most common dyshormonogenesis cause; perchlorate discharge positive",
            "TG": "Thyroglobulin — prohormone scaffold; largest secreted human protein (2767 aa); TG deficiency: serum Tg low despite high TSH (unique)",
            "Bamforth-Lazarus": "FOXE1 biallelic LOF — thyroid agenesis + cleft palate + spiky hair ± choanal atresia; only hereditary agenesis syndrome with dysmorphia",
            "Lingual thyroid": "Thyroid tissue at tongue base (foramen caecum) — ectopic thyroid in PAX8/FOXE1; do NOT ablate if only thyroid tissue present",
        },
    }


if __name__ == "__main__":
    import json
    print("=== HEREDITARY-THYROID-DISORDER-ATLAS — OVERVIEW ===")
    print(json.dumps(overview(), indent=2)[:3000])
    print("\n=== BREAKDOWN (first gene) ===")
    bd = breakdown()
    print(json.dumps(bd["genes"][0], indent=2)[:2000])
    print("\n=== DEFINITIONS (glossary sample) ===")
    df = definitions()
    print(json.dumps(list(df["glossary"].items())[:5], indent=2))
