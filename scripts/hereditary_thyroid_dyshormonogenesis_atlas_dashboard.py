#!/usr/bin/env python3
"""Hereditary-Thyroid-Dyshormonogenesis-Atlas — Complete 8-Gene Congenital Hypothyroidism Atlas
TPO     (thyroid peroxidase; 933 aa; 2p25.3; AR;
         most common dyshormonogenesis ~25%; goiter; organification defect;
         PERCHLORATE DISCHARGE TEST POSITIVE — confirms organification defect;
         seed SEED_BASE+0) ·
TSHR    (TSH receptor; 764 aa; 14q31.1; AR LOF — resistance;
         TSH resistance: elevated TSH + low/normal T4 + ABSENT goiter;
         NO organification defect; radionuclide scan: hypoplastic/ectopic gland;
         seed SEED_BASE+1) ·
TG      (thyroglobulin; 2768 aa; 8q24.22; AR;
         most common goitrous CH worldwide in consanguineous populations;
         discordant low thyroglobulin despite large goiter PATHOGNOMONIC;
         seed SEED_BASE+2) ·
SLC5A5  (NIS — sodium-iodide symporter; 643 aa; 19p13.11; AR;
         absent iodide UPTAKE — technetium scan shows ABSENT uptake;
         urine iodide NORMAL (dietary iodine absorbed enterally, not thyroidal);
         seed SEED_BASE+3) ·
DUOX2   (dual oxidase 2; 1548 aa; 15q15.3; AR biallelic severe / monoallelic transient;
         most common dyshormonogenesis in Europe (biallelic); H2O2 generation defect;
         organification defect — perchlorate discharge positive;
         seed SEED_BASE+4) ·
DUOXA2  (dual oxidase maturation factor 2; 320aa; 15q15.3; AR;
         essential chaperone for DUOX2 — DUOX2 cannot reach ER membrane without DUOXA2;
         identical phenotype to biallelic DUOX2; very rare but under-recognised;
         seed SEED_BASE+5) ·
SLC26A4 (pendrin; 780 aa; 7q22.3; AR;
         Pendred syndrome: goitrous CH + sensorineural hearing loss + Mondini dysplasia;
         cochlear dysplasia CT PATHOGNOMONIC; EVA enlarged vestibular aqueduct;
         seed SEED_BASE+6) ·
DEHAL1  (iodotyrosine dehalogenase 1 / DEHAL1; 289 aa; 6q25.1; AR;
         iodotyrosine recycling defect — MIT/DIT excreted in urine;
         goiter + hypothyroid + urinary iodotyrosines PATHOGNOMONIC;
         late-onset hypothyroidism possible (newborn screen may miss);
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 2918–2925)
"""
import random

SEED_BASE = 2918

ATLAS_GENES = [
    {
        "gene": "TPO",
        "protein": (
            "TPO -- 2p25.3 AR -- 933aa -- Thyroid-Peroxidase-"
            "103kDa-Haem-Enzyme-Iodination-Coupling-Organification-"
            "Most-Common-Dyshormonogenesis-25pct-Perchlorate-Discharge-Positive-"
            "OMIM-Gene-606765-Disease-CH-274500"
        ),
        "locus": "2p25.3",
        "protein_size": (
            "933 aa / 103 kDa (TPO — thyroid peroxidase; "
            "haem-containing glycoprotein; single-pass type I membrane protein; "
            "FUNCTION: two-step iodination reaction using H2O2 (from DUOX2): "
            "  Step 1 — IODINATION: iodide → iodine (I0) attached to tyrosine residues on thyroglobulin; "
            "  Step 2 — COUPLING: two iodotyrosines → T3 or T4 (di + mono = T3; di + di = T4); "
            "H2O2 PROVIDER: DUOX2 (and DUOXA2 chaperone) generates H2O2 at apical membrane; "
            "TPO LOF: iodination/coupling BLOCKED → thyroglobulin trapped with iodide but no hormone → "
            "  TSH rises → thyroid hypertrophies → GOITER; "
            "PERCHLORATE DISCHARGE TEST: "
            "  Perchlorate competes with iodide for NIS entry → "
            "    if organification defective, accumulated (unconjugated) iodide washes out → "
            "    >10% discharge = POSITIVE (confirms organification defect); "
            "    Used to distinguish organification defects (TPO, DUOX2, DUOXA2) from transport defects (NIS/SLC5A5); "
            "PREVALENCE: most common cause of dyshormonogenetic CH (~25%); "
            "encoded 2p25.3"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (AR) — LOSS-OF-FUNCTION — TPO CH: "
            "  Biallelic (compound heterozygous or homozygous) mutations; "
            "  Founder mutations: c.2268insT common in European populations; "
            "  CLINICAL FEATURES: "
            "    Congenital hypothyroidism: elevated TSH on newborn blood spot (NBS); "
            "    Goiter: present at birth or develops in infancy (TSH-driven hyperplasia); "
            "    Thyroid gland: IN SITU (normal position, enlarged — NOT ectopic); "
            "  DIAGNOSIS: "
            "    NBS TSH elevated → confirm with serum TSH, free T4, thyroglobulin; "
            "    Radionuclide scan (99mTc or 123I): gland in situ + INCREASED uptake (iodide trapped, not organified); "
            "    Perchlorate discharge test: >10% discharge POSITIVE; "
            "    Gene panel: TPO + DUOX2 + DUOXA2 + TG first (organification group); "
            "  PREVALENCE: ~25% of all dyshormonogenetic CH"
        ),
        "disease_category": (
            "ORGANIFICATION DEFECT — IODIDE TRAPPED BUT NOT ORGANIFIED: "
            "  ONSET: "
            "    NBS: TSH elevated (varies by severity); "
            "    Severe: CH detectable on day 2-5 NBS; "
            "    Mild: may be transient or partial; "
            "  CLINICAL: "
            "    Goiter: often visible/palpable at birth in severe cases; "
            "    Hypothyroid: lethargy, prolonged jaundice, large fontanelle, macroglossia, constipation; "
            "    Thyroid scintigraphy: normal/enlarged in situ gland + increased tracer uptake; "
            "  CRITICAL DISTINCTION from thyroid dysgenesis: "
            "    DYSGENESIS (ectopic/aplastic): scan shows no/ectopic gland; "
            "    DYSHORMONOGENESIS (TPO/DUOX2/TG/SLC5A5/DEHAL1): in situ gland — gene panel MANDATORY"
        ),
        "disease_pathway": (
            "TPO LOF → ORGANIFICATION BLOCKED → EXCESS TRAPPED IODIDE → GOITROUS CH: "
            "  Normal thyroid hormone synthesis: "
            "    1. NIS (SLC5A5) transports I− into follicular cell (active, 2Na+ : 1I−); "
            "    2. I− passes to apical membrane via pendrin (SLC26A4) / anoctamin-1; "
            "    3. DUOX2 (with DUOXA2) generates H2O2 at apical surface; "
            "    4. TPO uses H2O2 to iodininate tyrosyl residues on thyroglobulin (MIT, DIT); "
            "    5. TPO couples MIT + DIT → T3; DIT + DIT → T4 within thyroglobulin; "
            "    6. Thyroglobulin endocytosed → lysosomes → T3/T4 released into blood; "
            "  TPO LOF: steps 4-5 BLOCKED → iodide accumulates in follicle but unconjugated; "
            "    Thyroglobulin synthesised normally but devoid of thyroid hormones; "
            "    TSH rises (no T3/T4 feedback suppression) → gland hypertrophies → GOITER; "
            "  PERCHLORATE DISCHARGE: "
            "    Perchlorate → blocks NIS → accumulated non-organified I− washes out (>10% = defect)"
        ),
        "pathognomonic": (
            "PATTERN: Goitrous congenital hypothyroidism + in-situ thyroid on scan + "
            "PERCHLORATE DISCHARGE TEST POSITIVE (>10% iodide washout). "
            "Technetium/I-123 scan: normal/enlarged in-situ gland with INCREASED uptake. "
            "Distinguishes organification defect from: "
            "  NIS defect (SLC5A5): ABSENT scan uptake; "
            "  Thyroid dysgenesis: ectopic/absent gland. "
            "Gene panel (TPO/DUOX2/DUOXA2) confirms."
        ),
        "treatment": (
            "TREATMENT — TPO/ORGANIFICATION DEFECT CH: "
            "LEVOTHYROXINE (LT4): "
            "  Start IMMEDIATELY on NBS confirmation — target T4 in upper half of normal range; "
            "  Dose: 10-15 µg/kg/day in neonates; reduce as child grows; "
            "  TSH TARGET: 0.5–2.0 mU/L (suppression prevents goiter growth); "
            "  COMPLIANCE critical: delay >2 weeks → irreversible neurodevelopmental impairment; "
            "GOITER MANAGEMENT: "
            "  Adequate LT4 → suppresses TSH → goiter should regress; "
            "  Large goiter: surgery (subtotal thyroidectomy) if airway compromise; "
            "  Iodine supplementation: NOT helpful — more substrate without the enzyme; "
            "NEURODEVELOPMENT: "
            "  Early treatment (<2 weeks): near-normal IQ expected; "
            "  NBS universal → early detection standard in developed countries; "
            "GENETIC COUNSELLING: AR — 25% recurrence risk per pregnancy"
        ),
        "seed": 2918,
    },
    {
        "gene": "TSHR",
        "protein": (
            "TSHR -- 14q31.1 AR-LOF -- 764aa -- TSH-Receptor-"
            "84kDa-7TM-GPCR-Gs-Alpha-TSH-Resistance-"
            "Elevated-TSH-Low-T4-NO-Goiter-Hypoplastic-Gland-"
            "OMIM-Gene-603372-Disease-CH-275200"
        ),
        "locus": "14q31.1",
        "protein_size": (
            "764 aa / 84 kDa (TSHR — thyroid-stimulating hormone receptor; "
            "glycoprotein hormone receptor family; 7-transmembrane (7TM) GPCR; "
            "large N-terminal extracellular leucine-rich repeat domain (LRD) for TSH binding; "
            "SIGNALLING: TSH → TSHR → Gsα → adenylyl cyclase → cAMP ↑ → PKA → "
            "  thyroid growth, differentiation, iodide uptake, hormone synthesis; "
            "TSHR LOF (AR biallelic): TSH signalling ABSENT or reduced → "
            "  thyroid develops but does not respond to TSH → small/hypoplastic gland; "
            "  Hormone synthesis impaired (NIS not upregulated) → hypothyroid; "
            "  TSH elevated (no feedback) BUT gland DOES NOT HYPERTROPHY (no TSH signal); "
            "CRUCIAL DISTINCTION FROM DYSHORMONOGENESIS: "
            "  NO GOITER in TSHR resistance (cannot respond to TSH drive); "
            "  Organification intact (if iodide enters, TPO works); "
            "TSHR GOF (AD): familial non-autoimmune hyperthyroidism — OPPOSITE phenotype; "
            "encoded 14q31.1"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (AR) — LOSS-OF-FUNCTION — TSHR RESISTANCE CH: "
            "  Biallelic inactivating mutations → complete TSH insensitivity; "
            "  Heterozygous carriers: mild TSH elevation, usually euthyroid; "
            "  SPECTRUM: "
            "    Complete resistance (biallelic null): severe CH; "
            "    Partial resistance (compound heterozygous, one mild allele): "
            "      milder CH; TSH moderately elevated; may be missed on NBS; "
            "  PHENOTYPE: "
            "    Elevated TSH + inappropriately low/normal T4; "
            "    Thyroid gland: HYPOPLASTIC but IN SITU (not ectopic — gland formed but small); "
            "    Scan: small in-situ gland + REDUCED uptake (low NIS expression); "
            "  CRITICAL: NO ORGANIFICATION DEFECT → perchlorate discharge NEGATIVE; "
            "  CONTRAST TO TPO/DUOX2: TSHR has no goiter; TPO/DUOX2 have goiter + positive perchlorate"
        ),
        "disease_category": (
            "TSH RESISTANCE — RECEPTOR-LEVEL SIGNALLING FAILURE: "
            "  ONSET: NBS TSH elevated; severity depends on residual receptor function; "
            "  CLINICAL: "
            "    Congenital hypothyroidism: features as per other CH causes; "
            "    Thyroid on scan: SMALL, in situ — hypoplastic (not absent, not ectopic); "
            "    ABSENT GOITER — key discriminator from dyshormonogenesis; "
            "    Uptake: reduced (no TSH drive for NIS expression); "
            "  DDx TREE: "
            "    CH + goiter + positive perchlorate → TPO/DUOX2/DUOXA2; "
            "    CH + goiter + negative perchlorate + low Tg → TG; "
            "    CH + no goiter + absent scan uptake → SLC5A5 (NIS); "
            "    CH + no goiter + hypoplastic in-situ gland → TSHR; "
            "    CH + no goiter + absent/ectopic gland → dysgenesis (PAX8, NKX2-1, FOXE1); "
            "    CH + goiter + deafness → SLC26A4 (Pendred)"
        ),
        "disease_pathway": (
            "TSHR LOF → TSH SIGNALLING ABSENT → HYPOPLASTIC GLAND + HYPOTHYROID: "
            "  Normal: TSH → TSHR → Gsα-cAMP cascade → "
            "    NIS upregulation (iodide uptake ↑); "
            "    TPO upregulation; TG synthesis; thyroid growth; "
            "  TSHR LOF: TSH binds but no signal transduced → "
            "    NIS not upregulated → iodide uptake minimal; "
            "    Thyroid growth stimulus absent → gland hypoplastic; "
            "    Hormone synthesis reduced (substrate not entering); "
            "    TSH rises but no compensatory goiter (receptor absent/non-functional); "
            "  GLAND STRUCTURE: follicles present but atrophic — structural thyroid development "
            "    proceeds via PAX8/TTF1 (TSHR-independent), but FUNCTION requires TSHR"
        ),
        "pathognomonic": (
            "PATTERN: Elevated TSH + low T4 + ABSENT goiter + "
            "SMALL in-situ hypoplastic thyroid on scan + REDUCED/absent uptake. "
            "Perchlorate discharge: NEGATIVE (organification intact — it's a signalling defect). "
            "Thyroglobulin: low-normal (gland small but structurally present). "
            "KEY DISCRIMINATOR: no goiter distinguishes TSHR resistance from organification defects. "
            "TSHR gene sequencing confirms biallelic inactivating variants."
        ),
        "treatment": (
            "TREATMENT — TSHR RESISTANCE CH: "
            "LEVOTHYROXINE (LT4): identical to other CH causes; "
            "  Urgency: immediate on NBS confirmation; "
            "  Dose: 10-15 µg/kg/day; adjust by TSH/T4 levels; "
            "  NOTE: TSH suppression may not cause goiter (no functional receptor to drive growth); "
            "  TSH target: generally 0.5–2.0 mU/L; may be difficult given impaired feedback loop; "
            "MONITORING: "
            "  Free T4 monitoring crucial (TSH unreliable as sole marker when receptor dysfunctional); "
            "  Ensure T4 in upper normal range clinically; "
            "GOITER: not expected — if goiter develops, reconsider diagnosis; "
            "GENETIC COUNSELLING: AR — 25% recurrence; "
            "  Carriers (heterozygous): mild TSH elevation — monitor but no LT4 needed usually"
        ),
        "seed": 2919,
    },
    {
        "gene": "TG",
        "protein": (
            "TG -- 8q24.22 AR -- 2768aa -- Thyroglobulin-"
            "330kDa-Scaffold-Protein-Iodination-Hormone-Synthesis-"
            "Goitrous-CH-Low-Tg-Despite-Goiter-PATHOGNOMONIC-"
            "OMIM-Gene-188450-Disease-CH-274700"
        ),
        "locus": "8q24.22",
        "protein_size": (
            "2768 aa / 330 kDa (thyroglobulin — TG; "
            "homodimer (2 × 330 kDa = 660 kDa native); "
            "SCAFFOLD PROTEIN for thyroid hormone synthesis: "
            "  Contains multiple tyrosine residues that are iodinated by TPO → MIT, DIT; "
            "  TPO then couples iodotyrosines within TG → T3 (MIT+DIT) and T4 (DIT+DIT); "
            "  TG secreted into follicular lumen (colloid); "
            "  TSH-driven endocytosis → lysosomes → T3/T4 cleaved and released into blood; "
            "  TG itself circulates as tumour marker (used in thyroid cancer follow-up); "
            "TG LOF MUTATIONS: "
            "  Missense (structural), nonsense (truncation), splice-site: "
            "    Misfolded TG → retained in ER → ER stress → apoptosis (severe); "
            "    TG secreted but non-functional → no T3/T4 production; "
            "    TSH rises → GOITER; iodination proceeds (TPO working) but no hormone exported; "
            "PATHOGNOMONIC CLUE: GOITER + very LOW serum thyroglobulin level "
            "  (normally goiter → high TG from large gland; in TG LOF, the defective protein is absent/low); "
            "encoded 8q24.22"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (AR) — LOSS-OF-FUNCTION — TG CH: "
            "  Biallelic mutations — most common cause of goitrous CH in consanguineous families; "
            "  Population-specific founder mutations (Dutch, Japanese, Brazilians); "
            "  SPECTRUM: "
            "    Complete TG absence: severe CH + large goiter from birth; "
            "    Partial function: milder CH; may be missed on NBS if TSH borderline; "
            "  BIOCHEMISTRY: "
            "    TSH: markedly elevated; "
            "    Free T4: low; "
            "    Thyroglobulin: very LOW (paradoxical — large goiter but low TG); "
            "    PERCHLORATE TEST: variable — if TPO working normally, test may be negative/borderline; "
            "      (Iodination may proceed but hormones not formed — no organification DEFECT per se; "
            "       coupling step: DIT+DIT on absent/misfolded TG scaffold → fails); "
            "  RADIONUCLIDE SCAN: in-situ enlarged gland with increased uptake (TSH-driven); "
            "  CRITICAL DDx: "
            "    TG vs TPO: "
            "      TPO: perchlorate test POSITIVE; TG normal/high; "
            "      TG: perchlorate test NEGATIVE or borderline; serum TG very LOW"
        ),
        "disease_category": (
            "SCAFFOLD DEFECT — THYROGLOBULIN ABSENT/DYSFUNCTIONAL: "
            "  ONSET: NBS TSH elevated; goiter often visible at birth in severe cases; "
            "  CLINICAL: "
            "    Goitrous congenital hypothyroidism; "
            "    Thyroid gland: in situ, enlarged; "
            "    Features of CH: poor feeding, prolonged jaundice, hypotonia, constipation; "
            "  KEY DISCRIMINATOR: "
            "    GOITER + LOW SERUM THYROGLOBULIN = TG mutation until proven otherwise; "
            "    This is PARADOXICAL: large gland should secrete lots of TG → "
            "      but mutant TG is absent/retained in ER → low circulating TG; "
            "  CONSANGUINITY: strongly suggests TG or DUOX2 (both AR); "
            "  PERCHLORATE TEST: "
            "    Usually NEGATIVE in TG (organification enzyme TPO intact); "
            "    Iodination occurs but TG scaffold absent → no T3/T4 formed"
        ),
        "disease_pathway": (
            "TG LOF → NO SCAFFOLD → IODINATION PRODUCTS CANNOT BE COUPLED TO HORMONE: "
            "  Normal: TG secreted into colloid → TPO iodinates tyrosines on TG → "
            "    MIT and DIT formed within TG; TPO couples them → T3/T4 within TG; "
            "    Endocytosis → lysosomal cleavage → T3/T4 release; "
            "  TG LOF: "
            "    Misfolded TG → ER retention → UPR/ER stress → beta-cell-like apoptosis; "
            "    Insufficient TG in colloid → iodide enters follicle but no substrate for coupling; "
            "    T3/T4 cannot be formed → hypothyroid; "
            "    TSH rises → NIS/TPO upregulated → follicle fills with iodide but no hormone → "
            "      thyroglobulin production attempted → more misfolded protein → cycle of TSH elevation; "
            "    GOITER forms (TSH-driven proliferation); "
            "  SERUM TG: low because mutant TG not secreted (ER retained) → "
            "    paradox: big gland, low TG"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC: Goitrous CH + serum thyroglobulin very LOW or undetectable "
            "despite large goiter. "
            "Normal expectation: large goiter → high TG. "
            "TG mutation: large goiter → LOW TG (misfolded protein not secreted). "
            "Scan: in-situ enlarged gland + increased uptake. "
            "Perchlorate discharge: NEGATIVE (organification intact — TPO works, no TG substrate). "
            "Consanguinity: increases prior probability of TG or DUOX2 AR mutations."
        ),
        "treatment": (
            "TREATMENT — TG DEFECT CH: "
            "LEVOTHYROXINE (LT4): "
            "  Same urgency and dosing as other CH: 10-15 µg/kg/day neonates; "
            "  TSH suppression → goiter regression; "
            "  T4 in upper normal range → normal neurodevelopment if started early; "
            "GOITER: "
            "  Adequate TSH suppression → goiter should regress substantially; "
            "  Large compressive goiter at birth: may need neonatal airway management; "
            "  Surgery reserved for: airway compromise, failure to regress; "
            "MONITORING: "
            "  Serum TG: not a useful marker in TG defect (protein absent/low); "
            "  Monitor TSH, free T4; "
            "  Annual thyroid ultrasound for goiter size; "
            "RECURRENCE: AR — 25% per pregnancy; prenatal diagnosis possible via amniocentesis"
        ),
        "seed": 2920,
    },
    {
        "gene": "SLC5A5",
        "protein": (
            "SLC5A5 -- 19p13.11 AR -- 643aa -- Sodium-Iodide-Symporter-NIS-"
            "70kDa-13TM-Active-Iodide-Transport-"
            "ABSENT-Scan-Uptake-PATHOGNOMONIC-Low-Intrathyroidal-Iodide-No-Goiter-"
            "OMIM-Gene-601843-Disease-CH-274400"
        ),
        "locus": "19p13.11",
        "protein_size": (
            "643 aa / 70 kDa (NIS — sodium-iodide symporter; SLC5A5; "
            "13 transmembrane domains; co-transports 2 Na+ : 1 I- (secondary active transport); "
            "LOCATION: basolateral membrane of thyroid follicular cells; "
            "FUNCTION: concentrates iodide 20-40× against gradient into thyroid follicle; "
            "ENERGY SOURCE: Na+/K+-ATPase maintains Na+ gradient → drives iodide uptake; "
            "TSH upregulates NIS expression (TSH → cAMP → PKA → NIS transcription); "
            "NIS is also the target for radioactive iodine therapy (131I) in thyroid cancer; "
            "SLC5A5 LOF MUTATIONS: "
            "  NIS absent/non-functional → iodide NOT transported into thyroid; "
            "  Thyroid: structurally normal (TSHR signalling intact) → responds to TSH → grows; "
            "  BUT iodine substrate absent → hormone synthesis IMPOSSIBLE; "
            "  TECHNETIUM SCAN: technetium-99m (99mTc pertechnetate) is a NIS substrate → "
            "    if NIS absent, 99mTc NOT taken up → ABSENT uptake on scan; "
            "  URINE IODIDE: NORMAL (dietary iodine absorbed by gut; excreted renally; "
            "    thyroid cannot extract it); "
            "encoded 19p13.11"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (AR) — LOSS-OF-FUNCTION — NIS/SLC5A5 CH: "
            "  Biallelic inactivating mutations; "
            "  Rare — fewer than 100 reported families globally; "
            "  CLASSIC PHENOTYPE: "
            "    CH + GOITER (TSH-driven but no iodide substrate → compensatory hypertrophy); "
            "    Note: goiter can develop later (initially gland normal size, then enlarges); "
            "  SCAN: ABSENT technetium-99m or I-123 uptake — PATHOGNOMONIC; "
            "    Differentiates from TPO (scan uptake INCREASED); "
            "    Differentiates from TSHR (scan uptake reduced but not absent, gland small); "
            "  URINE IODIDE NORMAL: key — iodine absorbed enterally (dietary iodine normal); "
            "    Thyroid just cannot extract it from blood; "
            "  TREATMENT RESPONSE TO HIGH-DOSE IODIDE: in some partial LOF mutations, "
            "    pharmacological iodide bypasses NIS (passive diffusion at very high concentrations)"
        ),
        "disease_category": (
            "IODIDE TRANSPORT DEFECT — ABSENT NIS → NO IODIDE ENTRY: "
            "  ONSET: NBS TSH elevated; severity depends on residual NIS function; "
            "  CLINICAL: "
            "    Goitrous CH (goiter may develop later, not always at birth); "
            "    Features of hypothyroidism: prolonged jaundice, hypotonia, poor feeding; "
            "  SCAN: ABSENT or markedly reduced uptake — most important diagnostic clue; "
            "  DDx KEY POINT: "
            "    Absent scan uptake + NORMAL urine iodide → NIS defect (SLC5A5); "
            "    Absent scan uptake + low urine iodide → iodine deficiency (environmental); "
            "    Absent scan uptake + high urine TSH-receptor-blocking antibodies → maternal TRAB; "
            "  IODIDE LEVEL IN THYROID: very low (isotope dilution methods confirm); "
            "  SERUM TG: low-normal (gland structurally intact but inactive)"
        ),
        "disease_pathway": (
            "SLC5A5 LOF → ABSENT IODIDE TRANSPORT → SUBSTRATE STARVATION → CH: "
            "  Normal: dietary iodide → bloodstream → NIS concentrates I− into follicular cell → "
            "    I− passes to apical membrane → DUOX2-generated H2O2 + TPO → iodination of TG → "
            "    T3/T4 synthesis; "
            "  SLC5A5 LOF: NIS absent → I− CANNOT enter thyroid → "
            "    TPO present and functional but no substrate; "
            "    TG present and functional but not iodinated; "
            "    T3/T4 NOT produced → TSH rises → thyroid grows → GOITER; "
            "  TECHNETIUM-99m: "
            "    99mTc pertechnetate is transported by NIS (same charge/size as I−); "
            "    NIS absent → 99mTc not accumulated → absent signal on scan; "
            "  PARTIAL NIS MUTATIONS: some residual uptake; high-dose iodide → passive diffusion"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC: ABSENT technetium-99m or I-123 uptake on thyroid scan + "
            "NORMAL urine iodide. "
            "This combination = NIS defect (SLC5A5). "
            "Discriminators: "
            "  Absent uptake + low urine iodide → iodine deficiency; "
            "  Absent uptake + normal urine iodide → NIS (SLC5A5); "
            "  Absent uptake + maternal TSH-receptor blocking antibodies → transient; "
            "SLC5A5 gene sequencing confirms."
        ),
        "treatment": (
            "TREATMENT — SLC5A5/NIS CH: "
            "LEVOTHYROXINE (LT4): "
            "  Standard CH treatment: 10-15 µg/kg/day; immediate start on NBS; "
            "  TSH suppression: goal 0.5–2.0 mU/L → goiter regression; "
            "HIGH-DOSE IODIDE SUPPLEMENTATION (partial defects): "
            "  Pharmacological iodide (Lugol's): passive diffusion at very high [I−] → "
            "    partial NIS defect may be overcome in mild cases; "
            "  This is NOT the standard treatment — LT4 remains primary; "
            "  Trial of high-dose iodide only justified if partial NIS function confirmed by genetic testing; "
            "DIETARY IODINE: no benefit (cannot be extracted by the thyroid regardless); "
            "GOITER: TSH suppression → regression expected; "
            "GENETIC COUNSELLING: AR — 25% per pregnancy; "
            "  SLC5A5 also expressed in salivary glands, stomach — rare extra-thyroid features"
        ),
        "seed": 2921,
    },
    {
        "gene": "DUOX2",
        "protein": (
            "DUOX2 -- 15q15.3 AR-biallelic-severe / monoallelic-transient -- 1548aa -- "
            "Dual-Oxidase-2-178kDa-NADPH-Oxidase-H2O2-Generator-Apical-Membrane-"
            "Most-Common-Dyshormonogenesis-Europe-Organification-Defect-Perchlorate-Positive-"
            "OMIM-Gene-606759-Disease-CH-274900"
        ),
        "locus": "15q15.3",
        "protein_size": (
            "1548 aa / 178 kDa (DUOX2 — dual oxidase 2; "
            "NADPH oxidase family (NOX); 7 transmembrane domains + FAD/NADPH binding; "
            "also called thyroid oxidase 2 (THOX2); "
            "FUNCTION: generates H2O2 at the apical membrane of thyroid follicular cells; "
            "H2O2 is the obligate co-substrate for TPO iodination/coupling reactions; "
            "Without H2O2: TPO cannot function → organification fails; "
            "DUOXA2 (OMIM 612772): ESSENTIAL maturation chaperone for DUOX2 — "
            "  DUOX2 cannot leave ER without DUOXA2 — both encoded at 15q15.3 (adjacent genes); "
            "DUOX2 MUTATIONS: "
            "  Biallelic (AR): complete H2O2 deficiency → severe permanent CH; "
            "  Monoallelic (heterozygous): partial H2O2 reduction → TRANSIENT CH; "
            "    (NBS TSH elevated in neonatal period → normalises without LT4 in some); "
            "  MOST COMMON dyshormonogenesis gene in European populations; "
            "  ORGANIFICATION DEFECT: H2O2 absent → TPO cannot iodininate TG → iodide accumulates; "
            "  PERCHLORATE DISCHARGE POSITIVE (>10%); "
            "encoded 15q15.3"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (biallelic) — PERMANENT CH (severe): "
            "  Biallelic null/loss-of-function: complete H2O2 absence → severe goitrous CH; "
            "  PERMANENT — requires lifelong LT4; "
            "HETEROZYGOUS (monoallelic) — TRANSIENT CH (important! common trap): "
            "  Monoallelic DUOX2 mutations account for majority of transient CH on NBS; "
            "  TSH elevated on day 2-5 NBS → may normalise by 3 years (one functional allele compensates); "
            "  MANAGEMENT CONTROVERSY: treat or watch? "
            "    Treat with LT4 until 3 years → re-evaluate by free T4/TSH off LT4 for 4 weeks; "
            "    If euthyroid off LT4 at age 3: likely monoallelic DUOX2 → stop LT4; "
            "    If hypothyroid off LT4: likely biallelic → restart LT4 permanently; "
            "GENOTYPE-PHENOTYPE: "
            "  One null + one missense: may be permanent; "
            "  Two missense (partial function): may be transient; "
            "PREVALENCE: most common dyshormonogenesis gene in European and Japanese CH populations"
        ),
        "disease_category": (
            "H2O2 GENERATION DEFECT — ORGANIFICATION BLOCKED AT DUOX2 STEP: "
            "  ONSET: NBS TSH elevated; "
            "    Permanent (biallelic): TSH very elevated (>100 mU/L on NBS); "
            "    Transient (monoallelic): TSH moderately elevated (20-100 mU/L on NBS); "
            "  CLINICAL: "
            "    Goitrous CH (biallelic); "
            "    Transient CH (monoallelic): may or may not have goiter; "
            "  DIAGNOSTIC: "
            "    Perchlorate discharge test POSITIVE — same as TPO; "
            "    Thyroid scan: in-situ gland, increased uptake; "
            "  KEY CLUE TO DUOX2 vs TPO: "
            "    Both POSITIVE perchlorate; "
            "    DUOX2: monoallelic → transient CH (TPO does not cause transient CH — TPO is biallelic); "
            "    Gene panel distinguishes; "
            "  DUOXA2: identical phenotype to biallelic DUOX2 (required chaperone)"
        ),
        "disease_pathway": (
            "DUOX2 LOF → ABSENT H2O2 → TPO CANNOT FUNCTION → ORGANIFICATION DEFECT: "
            "  Normal: I− enters follicular cell via NIS; passes to apical surface; "
            "    DUOX2 (activated by Ca²+/TSHR signals) generates H2O2 at apical membrane; "
            "    TPO uses H2O2 to oxidise I− → I0 (active iodine) → iodinates TG tyrosines; "
            "    Coupling reaction also requires H2O2; "
            "  DUOX2 LOF: H2O2 absent → "
            "    TPO cannot perform iodination or coupling → "
            "    iodide accumulates in follicle (trapped, non-organified); "
            "    TSH rises → gland enlarges → GOITER; "
            "    Perchlorate displaces non-organified I− → discharge >10%; "
            "  MONOALLELIC: one DUOX2 functional → 50% H2O2 → marginal; "
            "    Sufficient in non-neonatal (lower demand) → transient correction"
        ),
        "pathognomonic": (
            "PATTERN: Goitrous CH + in-situ gland with increased uptake + "
            "PERCHLORATE DISCHARGE POSITIVE. "
            "Transient CH on NBS (normalises by age 3) = monoallelic DUOX2 most common cause. "
            "Permanent goitrous CH + positive perchlorate = biallelic DUOX2 or TPO. "
            "Gene panel distinguishes DUOX2 from TPO. "
            "CRITICAL: monoallelic DUOX2 is the most common cause of transient CH — "
            "do NOT stop LT4 in neonates without off-treatment reassessment at age 3."
        ),
        "treatment": (
            "TREATMENT — DUOX2 CH: "
            "BIALLELIC (PERMANENT): "
            "  LT4 immediately on NBS; lifelong; same dosing as TPO; "
            "  TSH suppression → goiter regression; "
            "MONOALLELIC (TRANSIENT): "
            "  LT4 started at diagnosis (do NOT withhold in neonates — brain development priority); "
            "  At age 2-3 years: trial off LT4 (4-6 weeks); "
            "    TSH/FT4 normal off LT4: discontinue (monoallelic DUOX2 confirmed); "
            "    TSH elevated off LT4: resume lifelong (biallelic or severe monoallelic); "
            "GOITER: same as TPO — TSH suppression first; surgery if compressive; "
            "NOTE: PERCHLORATE not used therapeutically (competes with NIS, not helpful for H2O2 defect); "
            "GENETIC COUNSELLING: "
            "  Biallelic: AR 25% recurrence; "
            "  Monoallelic: dominant-like; child of affected parent has 50% monoallelic inheritance"
        ),
        "seed": 2922,
    },
    {
        "gene": "DUOXA2",
        "protein": (
            "DUOXA2 -- 15q15.3 AR -- 320aa -- Dual-Oxidase-Maturation-Factor-2-"
            "36kDa-ER-Chaperone-DUOX2-Trafficking-"
            "Identical-Phenotype-Biallelic-DUOX2-Rare-Under-Recognised-"
            "OMIM-Gene-612772-Disease-CH-300523"
        ),
        "locus": "15q15.3",
        "protein_size": (
            "320 aa / 36 kDa (DUOXA2 — dual oxidase maturation factor 2; "
            "single-pass type II transmembrane protein; "
            "FUNCTION: essential ER chaperone for DUOX2 — "
            "  DUOXA2 promotes proper glycosylation and folding of DUOX2 in the ER; "
            "  Without DUOXA2: DUOX2 is retained in the ER → degraded → not expressed at apical membrane; "
            "GENOMIC LOCATION: 15q15.3 — adjacent to DUOX2 (DUOX2 at 15q15.3, DUOXA2 immediately upstream); "
            "  Also adjacent: DUOX1 and DUOXA1 (lung/oesophagus H2O2 — not thyroid-dominant); "
            "DUOXA2 MUTATIONS: "
            "  AR biallelic → DUOX2 not trafficked → effectively same as DUOX2 biallelic null; "
            "  Goitrous permanent CH; organification defect; perchlorate positive; "
            "  IMPORTANT DISTINCTION: gene panel must include DUOXA2 when DUOX2 sequencing is negative "
            "    in suspected organification defect — missing DUOXA2 = missed diagnosis; "
            "  Very rare (<50 reported cases) but under-recognised due to absent from older panels; "
            "encoded 15q15.3"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (AR) — LOSS-OF-FUNCTION — DUOXA2 CH: "
            "  Biallelic DUOXA2 mutations; "
            "  Phenotypically IDENTICAL to biallelic DUOX2: "
            "    Goitrous permanent CH; "
            "    Organification defect; "
            "    Perchlorate discharge positive; "
            "    Thyroid scan: in-situ gland with increased uptake; "
            "  RARER: fewer cases described vs DUOX2; "
            "  CLINICAL TRAP: "
            "    Organification defect CH → DUOX2 sequencing NEGATIVE → "
            "      check DUOXA2 (adjacent gene, often included on same panel now); "
            "    Without DUOXA2 on panel: diagnosis missed; "
            "  NOTE: DUOXA1 (adjacent) → DUOX1 chaperone → lung H2O2; no thyroid phenotype known; "
            "  HETEROZYGOUS DUOXA2: unclear if causes transient CH (data sparse)"
        ),
        "disease_category": (
            "H2O2 GENERATION DEFECT — CHAPERONE FAILURE — IDENTICAL TO BIALLELIC DUOX2: "
            "  ONSET: NBS TSH elevated; permanent CH; "
            "  CLINICAL: identical to DUOX2 biallelic — goitrous CH; "
            "  DIAGNOSTIC WORKUP: "
            "    Same as DUOX2: perchlorate positive; in-situ gland; increased uptake; "
            "    GENE PANEL: must include DUOXA2 alongside DUOX2; "
            "    Sequencing alone of DUOX2 negative → do NOT stop investigation → sequence DUOXA2; "
            "  KEY TEACHING: "
            "    Two adjacent genes (DUOX2 + DUOXA2) at 15q15.3; "
            "    DUOX2 is the enzyme; DUOXA2 is the chaperone; "
            "    Both AR LOF → identical clinical phenotype; "
            "    Gene panels now routinely include both — older panels missed DUOXA2"
        ),
        "disease_pathway": (
            "DUOXA2 LOF → DUOX2 RETAINED IN ER → NO APICAL MEMBRANE H2O2 → ORGANIFICATION DEFECT: "
            "  Normal: DUOXA2 in ER → binds DUOX2 → promotes glycosylation → "
            "    escorts DUOX2 through Golgi → plasma membrane trafficking → "
            "    DUOX2 at apical membrane → H2O2 generation; "
            "  DUOXA2 LOF: DUOX2 folds incorrectly in ER → ER-associated degradation (ERAD); "
            "    DUOX2 protein never reaches apical membrane; "
            "    H2O2 absent → TPO cannot function → organification blocked; "
            "    IDENTICAL downstream consequence to DUOX2 null mutation; "
            "  DISTINCTION: DUOXA2 mutation = DUOX2 protein levels low (degraded in ER); "
            "    DUOX2 mutation = DUOX2 protein absent/dysfunctional (but DUOXA2 levels normal)"
        ),
        "pathognomonic": (
            "Clinically identical to biallelic DUOX2. "
            "DIAGNOSTIC CLUE: organification defect (perchlorate positive) + goitrous CH + "
            "DUOX2 sequencing NEGATIVE → sequence DUOXA2. "
            "Both DUOX2 and DUOXA2 at 15q15.3 — adjacent genes, same metabolic step. "
            "Modern gene panels include both; older panels missed DUOXA2. "
            "Genetic diagnosis needed to distinguish DUOX2 from DUOXA2 — no clinical discriminator."
        ),
        "treatment": (
            "TREATMENT — DUOXA2 CH: "
            "IDENTICAL TO BIALLELIC DUOX2: "
            "  LT4 immediately; permanent; 10-15 µg/kg/day neonates; "
            "  TSH target: 0.5-2.0 mU/L; "
            "  Goiter: TSH suppression; surgery if compressive; "
            "  Neurodevelopment: early treatment → near-normal IQ; "
            "NOTE: NO transient CH phenotype described for biallelic DUOXA2 (unlike monoallelic DUOX2); "
            "GENETIC COUNSELLING: AR — 25% recurrence; "
            "  Cascade testing of siblings: sequence DUOXA2 (+ DUOX2) in affected families"
        ),
        "seed": 2923,
    },
    {
        "gene": "SLC26A4",
        "protein": (
            "SLC26A4 -- 7q22.3 AR -- 780aa -- Pendrin-Anion-Transporter-"
            "86kDa-11TM-Cl-HCO3-I-Exchanger-"
            "Pendred-Syndrome-CH-Goiter-SNHL-Mondini-EVA-PATHOGNOMONIC-"
            "OMIM-Gene-605646-Disease-Pendred-274600"
        ),
        "locus": "7q22.3",
        "protein_size": (
            "780 aa / 86 kDa (pendrin — SLC26A4; "
            "11 transmembrane domains; sulfate/bicarbonate/iodide anion exchanger; "
            "TISSUE EXPRESSION: thyroid (apical membrane) + inner ear (endolymphatic sac/duct) + kidney; "
            "THYROID FUNCTION: "
            "  Apical Cl−/HCO3−/I− exchanger — transports iodide from cytoplasm into follicular lumen; "
            "  REQUIRED for iodide to reach TPO/DUOX2 at apical surface; "
            "  SLC26A4 LOF → iodide trapped in cytoplasm → cannot reach apical membrane → "
            "    reduced iodination; "
            "  ORGANIFICATION PARTIALLY DEFECTIVE (iodide bottleneck); "
            "  PERCHLORATE TEST: usually POSITIVE (iodide trapped intracellularly); "
            "INNER EAR FUNCTION: "
            "  Endolymph homeostasis — maintains low Cl−, high K+ of endolymph; "
            "  SLC26A4 LOF → endolymph Cl− accumulation → endolymphatic hydrops → "
            "    SENSORINEURAL HEARING LOSS (SNHL); "
            "  MONDINI DYSPLASIA: cochlear malformation (1.5 turns instead of 2.5) on CT; "
            "  ENLARGED VESTIBULAR AQUEDUCT (EVA): most consistent radiological finding; "
            "encoded 7q22.3"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (AR) — LOSS-OF-FUNCTION — PENDRED SYNDROME: "
            "  Biallelic SLC26A4 mutations; "
            "  MOST COMMON cause of syndromic SNHL worldwide (together with connexin 26/GJB2); "
            "  PENDRED SYNDROME TRIAD: "
            "    1. Goitrous CH (or euthyroid goiter); "
            "    2. Sensorineural hearing loss (often severe-profound); "
            "    3. Cochlear dysplasia: Mondini + enlarged vestibular aqueduct (EVA); "
            "  VARIABILITY: "
            "    Not all patients have all three features; "
            "    Some: euthyroid with goiter only (EVA/SNHL may be isolated — DFNB4 allelic); "
            "    Some: CH without obvious SNHL initially; "
            "    SNHL: may be present at birth (congenital) or progressive (fluctuating/late); "
            "  NBS: TSH elevated in those with hypothyroid form; "
            "  CRITICAL DDx: any CH + SNHL → SLC26A4 until proven otherwise; "
            "    EVA on temporal bone CT is the radiological hallmark"
        ),
        "disease_category": (
            "PENDRED SYNDROME — MULTI-ORGAN (THYROID + INNER EAR): "
            "  ONSET: "
            "    SNHL: congenital (some) or progressive childhood (others); "
            "    CH: NBS positive if hypothyroid form; "
            "    Goiter: childhood/adolescence (TSH-driven enlargement); "
            "  CLINICAL: "
            "    SNHL: usually severe-profound bilateral; may fluctuate; "
            "    CH: goitrous; in-situ gland; "
            "    MONDINI DYSPLASIA: cochlea only 1.5 turns on temporal CT; "
            "    EVA: enlarged vestibular aqueduct (>1.5 mm midpoint diameter); "
            "  DIAGNOSIS: "
            "    Any CH + SNHL → temporal bone CT: look for EVA + Mondini; "
            "    Any isolated SNHL with EVA → sequence SLC26A4 ± NBS/thyroid evaluation; "
            "    Perchlorate discharge test: usually positive; "
            "  AUDIOLOGICAL EMERGENCY: "
            "    Head trauma → acute hearing deterioration in Pendred (pressure-wave to enlarged aqueduct)"
        ),
        "disease_pathway": (
            "SLC26A4 LOF → DUAL PATHOLOGY — THYROID IODIDE BOTTLENECK + INNER EAR FLUID IMBALANCE: "
            "THYROID: "
            "  NIS transports I− into follicular cell (basolateral); "
            "  SLC26A4 (pendrin) transports I− across apical membrane INTO follicular lumen; "
            "  LOF → I− accumulates in cytoplasm, cannot exit apically → "
            "    iodide unavailable to TPO/DUOX2 at apical surface → partial organification defect; "
            "  TSH rises → goiter; "
            "INNER EAR: "
            "  Endolymphatic sac/duct: SLC26A4 exchanges Cl−/HCO3−/I− → "
            "    maintains endolymph ionic composition; "
            "  LOF → Cl− accumulates in endolymph → endolymphatic hydrops; "
            "  Embryological: EVA forms (aqueduct enlarged due to aberrant fluid dynamics); "
            "  Hair cell function impaired → SNHL; "
            "  Progressive: episodes of head trauma/infection → acute endolymph pressure changes → "
            "    sudden hearing drops"
        ),
        "pathognomonic": (
            "PENDRED SYNDROME TRIAD (any 2 of 3 sufficient + SLC26A4 biallelic): "
            "1. Goitrous CH (or euthyroid goiter). "
            "2. Sensorineural hearing loss. "
            "3. Mondini dysplasia + enlarged vestibular aqueduct (EVA) on temporal bone CT. "
            "PERCHLORATE DISCHARGE: POSITIVE. "
            "KEY CLINICAL RULE: any CH + SNHL → SLC26A4 sequencing + temporal bone CT. "
            "Any child with EVA on CT → thyroid function + SLC26A4 regardless of hearing."
        ),
        "treatment": (
            "TREATMENT — SLC26A4 / PENDRED SYNDROME: "
            "THYROID: "
            "  LT4 for CH: standard dosing; early treatment mandatory for neurodevelopment; "
            "  Euthyroid goiter: LT4 to suppress TSH → goiter regression; "
            "  Surgery: rarely needed; "
            "HEARING LOSS: "
            "  Hearing aids: bilateral fitting from earliest diagnosis; "
            "  Cochlear implants: appropriate for severe-profound SNHL; "
            "    Mondini anatomy requires surgical modification — experienced implant centre mandatory; "
            "  CI OUTCOMES in Pendred: GOOD — similar to non-malformed cochleae; "
            "ACTIVITY RESTRICTIONS (CRITICAL): "
            "  Contact sports: CONTRAINDICATED in EVA — head trauma → acute hearing loss; "
            "  Swimming: avoid diving/underwater pressure changes; "
            "  Wear medical alert — head trauma protocol; "
            "MONITORING: "
            "  Annual audiometry (fluctuating SNHL); "
            "  Thyroid function 6-monthly until stable, then annually; "
            "GENETIC COUNSELLING: AR — 25% recurrence; carrier siblings screened if EVA identified"
        ),
        "seed": 2924,
    },
    {
        "gene": "DEHAL1",
        "protein": (
            "DEHAL1 -- 6q25.1 AR -- 289aa -- Iodotyrosine-Dehalogenase-1-"
            "33kDa-FMN-Dependent-Reductase-"
            "MIT-DIT-Recycling-Defect-Urine-Iodotyrosines-PATHOGNOMONIC-"
            "Late-Onset-Possible-NBS-May-Miss-"
            "OMIM-Gene-612025-Disease-CH-274800"
        ),
        "locus": "6q25.1",
        "protein_size": (
            "289 aa / 33 kDa (DEHAL1 — iodotyrosine dehalogenase 1; "
            "FMN (flavin mononucleotide)-dependent oxidoreductase; "
            "LOCATION: thyroid follicular cell microsomal membranes; "
            "FUNCTION: recycling of iodotyrosines (MIT and DIT) within the thyroid: "
            "  After TG endocytosis and lysosomal cleavage → T3 + T4 released + "
            "    MIT (monoiodotyrosine) and DIT (diiodotyrosine) also released; "
            "  MIT and DIT themselves have NO hormonal activity; "
            "  DEHAL1 deiodates MIT/DIT → free iodide + tyrosine → recycled for next synthesis cycle; "
            "  IODIDE RECYCLING: critical for iodide economy (MIT/DIT comprise ~70-80% of iodinated residues); "
            "DEHAL1 LOF: "
            "  MIT and DIT NOT deiodated → excreted in urine and blood → IODIDE WASTED; "
            "  Progressive iodine deficiency within the thyroid; "
            "  TSH rises → goiter → CH; "
            "  URINE IODOTYROSINES (MIT/DIT): elevated — PATHOGNOMONIC; "
            "  NBS MAY MISS: mild neonatal TSH elevation — worsens over time → LATE PRESENTATION; "
            "encoded 6q25.1"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (AR) — LOSS-OF-FUNCTION — DEHAL1 CH: "
            "  Biallelic inactivating mutations; "
            "  RARE — fewer cases described; "
            "  LATE-ONSET FEATURE (CRITICAL): "
            "    Newborn TSH may be NORMAL or mildly elevated; "
            "    NBS may PASS initially — CH develops progressively as iodine reserves depleted; "
            "    Goiter + hypothyroidism may appear months to years after birth; "
            "    DIAGNOSIS MISSED ON NBS — NBS is not sufficient to exclude DEHAL1; "
            "  DIETARY IODINE DEPENDENCE: "
            "    Low dietary iodine → worsens faster (recycling essential when intake marginal); "
            "    High dietary iodine → may compensate partially → later/milder presentation; "
            "  PERCHLORATE TEST: NEGATIVE (organification intact; NIS/TPO normal); "
            "    Iodide enters, is iodinated, but MIT/DIT wastes the iodine before recycling; "
            "  SCAN: in-situ gland with increased/normal uptake (NIS/TSHR functional)"
        ),
        "disease_category": (
            "IODIDE RECYCLING DEFECT — MIT/DIT WASTED IN URINE: "
            "  ONSET: variable — may be late (post-NBS, months to years); "
            "  CLINICAL: "
            "    Goitrous hypothyroidism: may present with developmental delay if late; "
            "    Goiter: develops as iodine deficiency worsens; "
            "    Features of CH: constipation, dry skin, poor growth, fatigue; "
            "    LATE: intellectual disability if untreated in childhood (missed NBS cases); "
            "  KEY DIAGNOSTICS: "
            "    Urine MIT/DIT elevated: PATHOGNOMONIC; "
            "    Quantitative urine organic acid/aminoacid profiles detect MIT/DIT; "
            "    Plasma MIT/DIT also elevated; "
            "    Serum TG: elevated (large TSH-driven gland with normal TG synthesis); "
            "    Perchlorate: NEGATIVE; "
            "  PITFALL: "
            "    NBS normal → CH presents later → diagnosis delayed → "
            "      urine MIT/DIT test requested ONLY if late goitrous CH without obvious explanation"
        ),
        "disease_pathway": (
            "DEHAL1 LOF → MIT/DIT NOT DEIODATED → IODIDE WASTED → PROGRESSIVE IODINE DEPLETION: "
            "  Normal cycle: TG endocytosis → T3/T4 released → MIT + DIT also released; "
            "    DEHAL1: MIT → I− + tyrosine; DIT → 2I− + tyrosine; "
            "    Recycled I− re-enters synthesis cycle; "
            "    WITHOUT recycling: must replace ALL iodine from diet; "
            "  DEHAL1 LOF: MIT/DIT not cleaved → exit thyroid → enter circulation → "
            "    renally filtered → urinary MIT/DIT detectable; "
            "    Net IODIDE LOSS per synthesis cycle; "
            "    Progressive intrathyroidal iodine deficiency; "
            "    TSH rises → goiter → progressive CH; "
            "    RATE: depends on dietary iodine intake — may take months-years to manifest fully; "
            "  SCAN: normal NIS/TSHR → iodine enters normally; "
            "    Deficit is in RECYCLING after hormone synthesis, not in uptake; "
            "    Perchlorate test: negative (organification TPO intact)"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC: Goitrous CH + ELEVATED URINE MIT/DIT (monoiodotyrosine + diiodotyrosine). "
            "Perchlorate discharge: NEGATIVE (organification intact). "
            "Scan: in-situ gland with increased uptake (NIS/TSHR functional). "
            "CRITICAL WARNING: NBS may be NORMAL or borderline — CH develops progressively. "
            "Any child with late-onset goitrous CH (NBS-normal or mild) → "
            "urine MIT/DIT assay MANDATORY + DEHAL1 sequencing. "
            "Dietary iodine supplementation may partially compensate (slower progression)."
        ),
        "treatment": (
            "TREATMENT — DEHAL1 CH: "
            "LEVOTHYROXINE (LT4): "
            "  Standard CH dosing regardless of age at diagnosis; "
            "  URGENCY: if late diagnosis → start immediately + assess neurodevelopment; "
            "  TSH suppression: goiter regression; "
            "HIGH-DOSE IODIDE SUPPLEMENTATION: "
            "  Logical strategy — replace wasted iodide with dietary/pharmacological iodide; "
            "  Potassium iodide supplementation reported effective in selected cases; "
            "  May reduce LT4 dose requirement (thyroid can synthesise more hormone if iodide replaced); "
            "  Dose: 100-250 µg/day potassium iodide (monitor for iodide-induced thyroid changes); "
            "GOITER: TSH suppression + iodide → regression; "
            "MONITORING: "
            "  TSH + free T4; "
            "  Urine MIT/DIT: to confirm biochemical response to iodide; "
            "  Ultrasound: goiter size annually; "
            "NEURODEVELOPMENT: if late diagnosis → formal developmental assessment; "
            "  Speech therapy, occupational therapy as needed; "
            "GENETIC COUNSELLING: AR — 25% recurrence"
        ),
        "seed": 2925,
    },
]


def _rng(seed):
    return random.Random(seed)


def _generate_patients(entry):
    rng = _rng(entry["seed"])
    gene = entry["gene"]
    n = 40
    patients = []
    for i in range(n):
        sex = rng.choice(["M", "F"])
        # Age at NBS or diagnosis
        if gene == "DEHAL1":
            # Late onset — may present after NBS
            age_dx_days = rng.randint(30, 730)  # 1 month to 2 years
            nbs_detected = rng.random() < 0.3  # often missed
        elif gene in ("TPO", "TG", "DUOX2", "DUOXA2"):
            age_dx_days = rng.randint(3, 14)  # NBS day 3-14
            nbs_detected = rng.random() < 0.9
        else:
            age_dx_days = rng.randint(3, 21)
            nbs_detected = rng.random() < 0.85
        tsh_nbs = round(rng.uniform(15, 200), 1)  # mU/L (elevated)
        ft4 = round(rng.uniform(3, 12), 1)  # pmol/L (low)
        goiter_present = gene in ("TPO", "TG", "DUOX2", "DUOXA2", "SLC5A5", "DEHAL1") and rng.random() < 0.8
        if gene == "TSHR":
            goiter_present = False  # TSHR: no goiter
        perchlorate_positive = gene in ("TPO", "DUOX2", "DUOXA2", "SLC26A4") and rng.random() < 0.85
        if gene in ("TG", "SLC5A5", "DEHAL1", "TSHR"):
            perchlorate_positive = False
        absent_scan_uptake = gene == "SLC5A5" and rng.random() < 0.95
        snhl = gene == "SLC26A4" and rng.random() < 0.85
        eva_mondini = gene == "SLC26A4" and rng.random() < 0.90
        urine_iodotyrosines = gene == "DEHAL1" and rng.random() < 0.90
        tg_paradox_low = gene == "TG" and rng.random() < 0.80  # low Tg despite goiter
        patients.append({
            "id": f"{gene}-{i+1:02d}",
            "gene": gene,
            "sex": sex,
            "age_dx_days": age_dx_days,
            "nbs_detected": nbs_detected,
            "tsh_nbs_mu_l": tsh_nbs,
            "ft4_pmol_l": ft4,
            "goiter": goiter_present,
            "perchlorate_positive": perchlorate_positive,
            "absent_scan_uptake": absent_scan_uptake,
            "snhl": snhl,
            "eva_mondini": eva_mondini,
            "urine_iodotyrosines": urine_iodotyrosines,
            "tg_paradox_low": tg_paradox_low,
        })
    return patients


def generate_overview():
    genes = [g["gene"] for g in ATLAS_GENES]
    total_patients = 0
    gene_rows = []
    for entry in ATLAS_GENES:
        pts = _generate_patients(entry)
        total_patients += len(pts)
        avg_tsh = round(sum(p["tsh_nbs_mu_l"] for p in pts) / len(pts), 1)
        goiter_n = sum(1 for p in pts if p["goiter"])
        nbs_n = sum(1 for p in pts if p["nbs_detected"])
        perchl_n = sum(1 for p in pts if p["perchlorate_positive"])
        gene_rows.append({
            "gene": entry["gene"],
            "locus": entry["locus"],
            "protein_summary": entry["protein"],
            "patients": len(pts),
            "avg_tsh_mu_l": avg_tsh,
            "goiter_pct": round(100 * goiter_n / len(pts)),
            "nbs_detected_pct": round(100 * nbs_n / len(pts)),
            "perchlorate_positive_pct": round(100 * perchl_n / len(pts)),
        })
    return {
        "atlas": "Hereditary Thyroid Dyshormonogenesis Atlas",
        "subtitle": "8-Gene Reference: TPO-TSHR-TG-SLC5A5-DUOX2-DUOXA2-SLC26A4-DEHAL1",
        "description": (
            "Comprehensive atlas of hereditary thyroid dyshormonogenesis and congenital hypothyroidism (CH), "
            "covering the eight major genetic causes of inherited thyroid hormone synthesis defects: "
            "TPO (thyroid peroxidase — organification defect, perchlorate positive, most common ~25%), "
            "TSHR (TSH receptor resistance — no goiter, hypoplastic in-situ gland, perchlorate negative), "
            "TG (thyroglobulin — scaffold defect, low TG despite goiter PATHOGNOMONIC, perchlorate negative), "
            "SLC5A5/NIS (sodium-iodide symporter — absent scan uptake PATHOGNOMONIC, normal urine iodide), "
            "DUOX2 (H2O2 generator — organification defect, biallelic permanent / monoallelic transient CH, most common in Europe), "
            "DUOXA2 (DUOX2 chaperone — identical to biallelic DUOX2, missed by older panels), "
            "SLC26A4/pendrin (Pendred syndrome — CH + SNHL + Mondini/EVA PATHOGNOMONIC), "
            "DEHAL1 (iodotyrosine recycling — MIT/DIT in urine PATHOGNOMONIC, NBS may miss late-onset CH). "
            "320 patients (8 × 40), seeds 2918-2925."
        ),
        "total_patients": total_patients,
        "total_genes": len(genes),
        "genes": genes,
        "gene_rows": gene_rows,
        "categories": {
            "Organification defect (perchlorate +ve, in-situ goiter)": ["TPO", "DUOX2", "DUOXA2"],
            "TSH signalling failure (no goiter, hypoplastic)": ["TSHR"],
            "Scaffold protein defect (low TG despite goiter)": ["TG"],
            "Iodide transport defect (absent scan uptake)": ["SLC5A5"],
            "Syndromic CH + SNHL + Mondini/EVA (Pendred)": ["SLC26A4"],
            "Iodide recycling defect (late-onset, urine MIT/DIT)": ["DEHAL1"],
        },
        "key_facts": [
            "TPO: organification defect — perchlorate discharge >10% POSITIVE + goiter + in-situ gland",
            "TSHR resistance: NO GOITER + hypoplastic in-situ gland + reduced scan uptake + perchlorate NEGATIVE",
            "TG defect: LOW serum thyroglobulin despite LARGE GOITER — paradox is pathognomonic",
            "SLC5A5/NIS: ABSENT technetium scan uptake + NORMAL urine iodide = NIS defect",
            "DUOX2 monoallelic: MOST COMMON cause of transient CH on NBS — off-treatment trial at age 3 mandatory",
            "DUOXA2: clinically identical to biallelic DUOX2 — missing from older panels; include on organification workup",
            "SLC26A4/Pendred: CH + SNHL + EVA on temporal bone CT — contact sports CONTRAINDICATED (EVA head trauma risk)",
            "DEHAL1: NBS may be NORMAL — late-onset goitrous CH — urine MIT/DIT PATHOGNOMONIC — iodide supplementation effective",
        ],
        "diagnostic_algorithm": (
            "CH WORKUP — STEP-BY-STEP: "
            "1. NBS TSH elevated → confirm: serum TSH, free T4, thyroglobulin; "
            "2. Thyroid radionuclide scan (99mTc or I-123): "
            "   a. No gland/ectopic → dysgenesis (PAX8, NKX2-1, FOXE1 — different atlas); "
            "   b. In-situ enlarged gland + INCREASED uptake → dyshormonogenesis workup; "
            "   c. ABSENT uptake + normal urine iodide → SLC5A5 (NIS); "
            "   d. Small in-situ gland + reduced uptake → TSHR resistance; "
            "3. Perchlorate discharge test (in-situ gland): "
            "   POSITIVE (>10%) → organification defect: TPO, DUOX2, DUOXA2, SLC26A4; "
            "   NEGATIVE → TG, SLC5A5, TSHR, DEHAL1; "
            "4. Serum thyroglobulin: "
            "   Low despite goiter → TG mutation; "
            "   Low with absent uptake → SLC5A5; "
            "5. Audiometry + temporal bone CT: "
            "   SNHL + EVA/Mondini → SLC26A4 (Pendred); "
            "6. Urine MIT/DIT: if late-onset goitrous CH, perchlorate negative → DEHAL1; "
            "7. Gene panel: TPO + TSHR + TG + SLC5A5 + DUOX2 + DUOXA2 + SLC26A4 + DEHAL1"
        ),
    }


def generate_breakdown():
    result = []
    for entry in ATLAS_GENES:
        pts = _generate_patients(entry)
        result.append({
            "gene": entry["gene"],
            "locus": entry["locus"],
            "protein": entry["protein"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"],
            "disease_category": entry["disease_category"],
            "disease_pathway": entry["disease_pathway"],
            "pathognomonic": entry["pathognomonic"],
            "treatment": entry["treatment"],
            "patient_count": len(pts),
            "seed": entry["seed"],
        })
    return {"genes": result, "count": len(result)}


def generate_definitions():
    return {
        "definitions": [
            {
                "term": "Perchlorate Discharge Test — Organification Defect Detector",
                "definition": (
                    "PRINCIPLE: perchlorate (ClO4−) competitively inhibits NIS → blocks further iodide uptake. "
                    "If organification is INTACT: most iodide already covalently bound to TG → cannot be displaced → "
                    "  <10% discharge = NEGATIVE = normal. "
                    "If organification DEFECTIVE: iodide trapped but not bound → perchlorate blocks further entry → "
                    "  accumulated free iodide washes out → >10% discharge = POSITIVE = organification defect. "
                    "POSITIVE: TPO, DUOX2, DUOXA2, SLC26A4 (partial). "
                    "NEGATIVE: TG (organification proceeds but no scaffold), SLC5A5 (no iodide enters anyway), "
                    "  TSHR (signalling defect), DEHAL1 (iodide recycling defect, normal organification). "
                    "METHOD: radioiodine (I-123) given → uptake measured at 2h → KClO4 600mg oral → "
                    "  uptake re-measured at 4h → % discharge calculated."
                ),
            },
            {
                "term": "Thyroid Dyshormonogenesis vs Thyroid Dysgenesis — Critical Distinction",
                "definition": (
                    "DYSGENESIS (thyroid gland structural defect — 80-85% of permanent CH): "
                    "  Ectopic (sublingual/lingual most common), hypoplastic, or absent gland. "
                    "  Scan: no uptake in normal location; uptake may be elsewhere. "
                    "  Cause: PAX8, NKX2-1/TTF1, FOXE1/TTF2, NKX2-5 mutations. "
                    "DYSHORMONOGENESIS (enzyme/transporter defect — 15-20% of permanent CH): "
                    "  Gland structurally present, IN SITU, often enlarged (goiter). "
                    "  Scan: normal/enlarged in-situ gland with increased uptake (except SLC5A5). "
                    "  Cause: TPO, DUOX2, DUOXA2, TG, SLC5A5, SLC26A4, DEHAL1, TSHR (resistance). "
                    "CLINICAL RULE: IN-SITU GLAND ON SCAN → gene panel for dyshormonogenesis MANDATORY."
                ),
            },
            {
                "term": "DUOX2 Transient vs Permanent CH — The Age-3 Re-evaluation",
                "definition": (
                    "DUOX2 monoallelic (heterozygous): partial H2O2 production → borderline organification. "
                    "NEONATAL: demand high → TSH elevated → NBS positive → treat with LT4. "
                    "BY AGE 3: demand decreases (relative), one allele may compensate. "
                    "RE-EVALUATION PROTOCOL: "
                    "  Age 3 years: hold LT4 for 4-6 weeks; "
                    "  Check TSH and free T4: "
                    "    TSH normal (<5 mU/L): euthyroid → monoallelic DUOX2 transient → STOP LT4; "
                    "    TSH elevated (>5-10 mU/L): biallelic or severe → RESTART LT4 permanently. "
                    "CRITICAL: DO NOT stop LT4 before age 3 re-evaluation — risk of hypothyroid brain injury. "
                    "Monoallelic DUOX2 = MOST COMMON cause of transient CH."
                ),
            },
            {
                "term": "Pendred Syndrome — SLC26A4 — Audiological Emergency",
                "definition": (
                    "SLC26A4 (pendrin) biallelic LOF → Pendred syndrome: "
                    "CH/goiter + SNHL + Mondini cochlear dysplasia + enlarged vestibular aqueduct (EVA). "
                    "EVA AUDIOLOGICAL EMERGENCY: enlarged vestibular aqueduct → direct pressure transmission from CSF; "
                    "  Head trauma / Valsalva / pressure changes → sudden acute SNHL. "
                    "MANAGEMENT: "
                    "  Contact sports CONTRAINDICATED; "
                    "  Swimming/diving: avoid underwater pressure changes; "
                    "  Medical alert bracelet; "
                    "  Head trauma protocol: urgent ENT assessment for acute hearing drop. "
                    "COCHLEAR IMPLANTS: indicated for severe-profound SNHL; excellent outcomes in Pendred; "
                    "  Mondini anatomy: specialist centre mandatory for implantation technique. "
                    "TEMPORAL BONE CT: mandatory in any child with SNHL to identify EVA."
                ),
            },
            {
                "term": "DEHAL1 — Late-Onset CH Missed by NBS",
                "definition": (
                    "DEHAL1 LOF → iodotyrosine recycling defect → progressive iodide depletion → late CH. "
                    "NBS (day 3-5): TSH may be NORMAL or only mildly elevated. "
                    "PROGRESSIVE: iodine stores deplete over months → TSH rises → goiter → hypothyroid. "
                    "PRESENTATION: late-onset goitrous hypothyroidism (months to years after normal NBS). "
                    "DIAGNOSIS: urine MIT/DIT (monoiodotyrosine + diiodotyrosine) ELEVATED — pathognomonic. "
                    "CLINICAL RULE: "
                    "  Any child with late goitrous CH after normal NBS → urine MIT/DIT assay + DEHAL1 sequencing. "
                    "TREATMENT: LT4 + potassium iodide supplementation (replaces wasted iodide). "
                    "IMPORTANT: iodide supplementation alone may suffice in mild partial defects."
                ),
            },
            {
                "term": "TG Defect — Paradoxical Low Thyroglobulin with Large Goiter",
                "definition": (
                    "EXPECTED: large goiter → high serum thyroglobulin (large gland secretes TG). "
                    "TG MUTATION PARADOX: large goiter + very LOW serum thyroglobulin. "
                    "MECHANISM: mutant TG misfolds in ER → ER-associated degradation (ERAD) → "
                    "  TG not secreted into colloid or bloodstream → gland large (TSH-driven) but TG absent. "
                    "PERCHLORATE TEST: NEGATIVE (organification enzyme TPO intact; "
                    "  iodination proceeds but no scaffold to couple on). "
                    "CLINICAL RULE: "
                    "  Goitrous CH + serum TG very low → TG mutation until proven otherwise. "
                    "  Especially in consanguineous families (AR). "
                    "NOTE: serum TG in TG mutations may be undetectable or <5 µg/L (normal in adults: 3-40 µg/L)."
                ),
            },
            {
                "term": "NIS Defect (SLC5A5) — Absent Technetium Uptake + Normal Urine Iodide",
                "definition": (
                    "NIS (SLC5A5) LOF: iodide cannot enter thyroid → no uptake of NIS substrates. "
                    "99mTc pertechnetate scan: ABSENT uptake (99mTc transported by NIS — absent if NIS absent). "
                    "URINE IODIDE: NORMAL — dietary iodine absorbed enterally, cannot be extracted by thyroid. "
                    "DIFFERENTIAL OF ABSENT SCAN UPTAKE: "
                    "  NIS defect: absent uptake + normal urine iodide; "
                    "  Iodine deficiency: absent uptake + LOW urine iodide (<100 µg/g creatinine); "
                    "  Maternal TSH-receptor blocking antibodies (TRAB): transient; antibody titres; "
                    "  Iatrogenic iodine excess: scan suppressed; history of iodine exposure. "
                    "GOITER IN NIS DEFECT: paradoxical — gland grows (TSH-driven) but cannot concentrate iodide. "
                    "TREATMENT: LT4; high-dose iodide may bypass NIS in partial defects."
                ),
            },
            {
                "term": "Thyroid Gene Panel — When to Order and What to Include",
                "definition": (
                    "INDICATIONS FOR DYSHORMONOGENESIS GENE PANEL: "
                    "  In-situ gland on thyroid scan with elevated uptake OR absent uptake + normal urine iodide; "
                    "  Positive family history of CH; "
                    "  Consanguinity; "
                    "  CH + SNHL (Pendred/SLC26A4); "
                    "  Transient CH considering stopping LT4 (identify DUOX2 monoallelic). "
                    "GENE PANEL MUST INCLUDE (minimum 8 genes): "
                    "  TPO, DUOX2, DUOXA2, TG, SLC5A5, TSHR, SLC26A4, DEHAL1. "
                    "  Note: DUOXA2 missing from older panels — reason for undiagnosed organification defects. "
                    "ADDITIONAL (extended): DUOX1, DUOXA1, SLC26A7, IYD (DEHAL1 alternative name). "
                    "INTERPRETATION: "
                    "  One pathogenic variant in AR gene: carrier — not causative; look for second allele; "
                    "  Two pathogenic variants in AR gene: causative. "
                    "  Variants of uncertain significance (VUS): functional studies may be required."
                ),
            },
        ]
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(generate_overview(), indent=2)[:500])
    print("\n=== BREAKDOWN (count) ===")
    bd = generate_breakdown()
    print(f"Genes: {bd['count']}")
    print("\n=== DEFINITIONS (count) ===")
    df = generate_definitions()
    print(f"Definitions: {len(df['definitions'])}")
