#!/usr/bin/env python3
"""CHI-Atlas — Complete 8-Gene Congenital Hyperinsulinism Atlas
ABCC8 · KCNJ11 · GCK · GLUD1 · HADH · HNF4A · SLC16A1 · INSR
320-patient aggregate cohort (8 × 40, seeds 904–911)

Congenital Hyperinsulinism facts:
  - Most common cause of persistent hypoglycemia in neonates and infants.
  - Unregulated insulin secretion → glucose <2.2 mmol/L (40 mg/dL) despite low/absent ketones.
  - Collective incidence ~1/30,000–1/50,000 (higher in consanguineous populations).
  - KEY TEACHING POINTS:
      ABCC8: SUR1 subunit of K-ATP channel; most common (40–45%); recessive = focal/diffuse,
             dominant = mild diffuse; 18F-DOPA PET distinguishes focal vs diffuse.
      KCNJ11: Kir6.2 pore subunit; same K-ATP channel, same focal/diffuse spectrum as ABCC8;
              neonatal diabetes if GOF (opposite direction: channel stays OPEN → no insulin).
      GCK: Glucokinase GOF; dominant; low glucose threshold → beta-cells fire at normal glucose;
           diazoxide-RESISTANT; near-total pancreatectomy may be required.
      GLUD1: GDH GOF → Hyperinsulinism-Hyperammonemia Syndrome (HHS); leucine-sensitive;
             ammonia ALWAYS elevated (asymptomatic, 100–200 µmol/L); diazoxide-responsive;
             ONLY CHI gene with hyperammonemia.
      HADH: SCHAD deficiency; HADH normally inhibits GDH → loss → GDH disinhibition;
            protein-sensitive; C4OH on acylcarnitine; diazoxide-responsive.
      HNF4A: Neonatal transient HI + macrosomia (56%); diazoxide-responsive; SAME mutation
             → neonatal CHI then adult MODY1; autosomal dominant.
      SLC16A1: MCT1 GOF → exercise-induced HI (EIHI); pyruvate fuels beta-cells during anaerobic
               exercise; diazoxide NOT effective; avoid strenuous exercise.
      INSR: Donohue/Rabson-Mendenhall; extreme insulin RESISTANCE → paradoxical high insulin;
            fasting hypoglycemia from IGF-1-like effects; mecasermin (rhIGF-1) therapy;
            diazoxide NOT effective.

COHORT: 8 × 40 = 320 patient slots (seeds 904–911; gene-specific seeds)
"""

import random

SEED_BASE = 904

# ── All 8 CHI Genes ───────────────────────────────────────────────────────────────
CHI_GENES = [
    # ── ABCC8 — SUR1 K-ATP channel (most common CHI) ─────────────────────────────
    {
        "gene": "ABCC8", "alias": "ABCC8 — SUR1 K-ATP Channel · CHI (most common, 40–45%)",
        "aa": "1581 aa", "kDa": "177 kDa",
        "gene_class": "K-ATP channel subunit (SUR1, ABC transporter superfamily)",
        "chi_subgroup": "K-ATP channel defects (ABCC8 · KCNJ11)",
        "locus": "11p15.1", "omim_gene": 600509,
        "phenotype": "Most common CHI gene (40–45%); recessive = severe focal/diffuse, dominant = mild diffuse; diazoxide variable; 18F-DOPA PET differentiates focal vs diffuse",
        "disease": (
            "ABCC8 biallelic loss → Congenital Hyperinsulinism type 1 (CHI1, OMIM #256450), "
            "the most common genetic cause of persistent neonatal/infantile hypoglycemia. "
            "ABCC8 encodes SUR1 (1581aa, 177kDa), the regulatory subunit of the pancreatic "
            "K-ATP channel (K-ATP = Kir6.2 + SUR1 octameric complex: 4×Kir6.2 + 4×SUR1). "
            "SUR1 binds ATP (closes channel) and ADP (opens channel); also the sulfonylurea "
            "receptor (glibenclamide closes; diazoxide opens). In normal physiology, rising "
            "intracellular ATP/ADP ratio after glucose metabolism closes K-ATP → membrane "
            "depolarisation → Ca2+ influx → insulin exocytosis. ABCC8 loss → K-ATP channels "
            "constitutively CLOSED → permanent depolarisation → unregulated insulin secretion "
            "regardless of plasma glucose. Clinical: severe hypoglycemia in neonates "
            "(glucose <1.5 mmol/L at birth), macrosomia (fetal hyperinsulinism drives growth), "
            "absent ketones (insulin suppresses ketogenesis), absent FFA (insulin suppresses "
            "lipolysis), elevated insulin (>2 mU/L at hypoglycemia), elevated C-peptide. "
            "Glucose requirement (GIR) often 15–20 mg/kg/min to maintain euglycemia. "
            "Two histological forms: FOCAL (somatic paternal LOH + germline paternal ABCC8 mutation "
            "→ clonal adenomatoid proliferation in a focal zone, 30–40% of recessive cases) and "
            "DIFFUSE (biallelic germline → all beta-cells affected). Dominant ABCC8 mutations → "
            "milder diffuse CHI, often responsive to diazoxide. "
            "Incidence of ABCC8 CHI: ~1/50,000; higher in Ashkenazi Jews, Finnish, Saudi populations."
        ),
        "inheritance": (
            "Autosomal recessive (severe diffuse/focal CHI) or Autosomal dominant (milder diffuse). "
            "ABCC8 11p15.1. Paternally inherited recessive allele + somatic paternal LOH = FOCAL CHI. "
            "Two recessive germline alleles = DIFFUSE CHI. "
            "Dominant heterozygous gain-of-function loss of channel gating = mild diffuse CHI."
        ),
        "hallmark": (
            "ABCC8 CHI HALLMARKS: "
            "(1) K-ATP CHANNEL CONSTITUTIVE CLOSURE: K-ATP closed regardless of glucose → "
            "permanent beta-cell membrane depolarisation → Ca2+ always influx → insulin always released; "
            "(2) 18F-DOPA PET CRITICAL FOR FOCAL vs DIFFUSE DISTINCTION: "
            "Focal CHI → localised 18F-DOPA uptake in adenomatoid zone (can be anywhere in pancreas); "
            "Diffuse CHI → uniform increased uptake throughout pancreas; "
            "PET must be interpreted by experienced centre (sensitivity 85%, specificity 96% for focal); "
            "This distinction is SURGICAL: focal → limited partial pancreatectomy → CURE; "
            "diffuse → near-total pancreatectomy → iatrogenic diabetes; "
            "(3) ABSENT KETONES AT HYPOGLYCEMIA PATHOGNOMONIC: "
            "beta-hydroxybutyrate <0.5 mmol/L, FFA <0.5 mmol/L at glucose <2.2 mmol/L; "
            "contrast: ketotic hypoglycemia (normal response), which has HIGH ketones; "
            "(4) DIAZOXIDE TRIAL MANDATORY before surgery: "
            "ABCC8 dominant → diazoxide-responsive (70–80%); "
            "ABCC8 recessive → diazoxide-resistant (K-ATP channels absent/non-functional, "
            "cannot be opened by diazoxide); exception: some partial-loss recessive may respond; "
            "(5) MACROSOMIA: fetal hyperinsulinism → increased IGF-1-like effects → LGA (>90th centile); "
            "(6) GLUCOSE INFUSION RATE (GIR) 15–20 mg/kg/min required in severe cases: "
            "central line often needed for 15% or 20% dextrose"
        ),
        "key_ddx": (
            "ABCC8 DDx: "
            "(1) KCNJ11 CHI: clinically identical; Kir6.2 pore vs SUR1 regulatory subunit; "
            "same K-ATP defect, same 18F-DOPA PET approach; distinguish only by sequencing; "
            "(2) GCK-CHI: glucokinase GOF; dominant; usually diazoxide-resistant; no focal form; "
            "glucose threshold shifted; diagnose by GCK sequencing; "
            "(3) GLUD1-HHS: hyperammonemia present (ammonia 100–200 µmol/L); leucine-sensitive; "
            "protein meal triggers; diazoxide-responsive; no focal form; "
            "(4) Insulinoma: adults, not neonates; ABCC8 somatic mutation implicated; "
            "MRI/CT/EUS localise lesion; "
            "(5) Transient neonatal hyperinsulinism (perinatal stress, maternal diabetes): "
            "resolves in days to weeks; no genetic mutation; normal glucose infusion requirement"
        ),
        "diet_treatment": (
            "Acute: IV dextrose (GIR 15–20 mg/kg/min via central line); "
            "glucagon IV/IM for acute rescue (0.03 mg/kg, max 1 mg); "
            "Diazoxide (K-ATP opener): 5–15 mg/kg/day in 3 divided doses + hydrochlorothiazide "
            "(prevent fluid retention); trial for 5 days; check response; "
            "Octreotide (somatostatin analogue): 5–35 µg/kg/day SC q6–8h; "
            "second-line; long-acting lanreotide for chronic use; "
            "Surgical (diazoxide-resistant): 18F-DOPA PET first → focal: limited "
            "pancreatectomy (CURE in 97%); diffuse: near-total (95–98%) pancreatectomy "
            "→ iatrogenic diabetes + exocrine insufficiency; "
            "Sirolimus (mTOR inhibitor): emerging for medically unresponsive diffuse CHI"
        ),
        "gene_therapy_status": (
            "No approved gene therapy for ABCC8-CHI. Focal CHI is surgical cure. "
            "Research: AAV-mediated ABCC8 delivery to pancreas — preclinical stage. "
            "Sirolimus trials ongoing for diffuse CHI refractory to diazoxide/octreotide. "
            "SUR1 (ABCC8) gain-of-function mutation → neonatal diabetes mellitus (opposite "
            "phenotype — channel stays open, no insulin; treat with sulfonylurea glibenclamide)."
        ),
        "critical_ci": (
            "CRITICAL: "
            "(1) NOT doing 18F-DOPA PET before pancreatectomy — fatal error: focal CHI cured "
            "by limited resection; diffuse requires near-total; cannot distinguish clinically; "
            "(2) Diazoxide in recessive ABCC8 — K-ATP channels absent; diazoxide cannot work "
            "without its target; do 5-day trial, do not persist if no response; "
            "(3) Missing absent ketones — key diagnostic criterion; always measure "
            "beta-hydroxybutyrate and FFA at time of hypoglycemia; "
            "(4) GIR >8 mg/kg/min = hyperinsulinism until proven otherwise — do not label as "
            "'idiopathic' without genetic workup; "
            "(5) Surgical near-total pancreatectomy for diffuse → 100% iatrogenic diabetes "
            "by adulthood — lifelong insulin therapy; inform family; "
            "(6) SUR1 GOF (neonatal diabetes) — opposite phenotype: treat with SULFONYLUREA, "
            "not diazoxide"
        ),
        "nbs_marker": (
            "No specific NBS metabolite marker for ABCC8-CHI. "
            "Screen for hypoglycemia on day 1–2 of life in all macrosomic or high-risk neonates. "
            "Diagnosis: fasting study with simultaneous glucose, insulin, C-peptide, "
            "beta-hydroxybutyrate, FFA at hypoglycemia; GIR calculation. "
            "Molecular: ABCC8 sequencing (also detect KCNJ11). "
            "18F-DOPA PET once CHI confirmed biochemically and molecular partial."
        ),
        "key_biomarker": (
            "Plasma glucose <2.2 mmol/L (40 mg/dL) + plasma insulin >2 mU/L (>14 pmol/L) "
            "= hyperinsulinism confirmed. "
            "GIR >8 mg/kg/min to maintain normoglycaemia = hyperinsulinism until proven otherwise. "
            "Beta-hydroxybutyrate <0.5 mmol/L at hypoglycemia = pathognomonic absent ketosis. "
            "FFA <0.5 mmol/L at hypoglycemia = suppressed lipolysis. "
            "C-peptide elevated = confirms endogenous insulin source. "
            "Ammonia: NORMAL in ABCC8-CHI (vs GLUD1 where ammonia elevated). "
            "18F-DOPA PET: focal uptake pattern if focal CHI."
        ),
        "severity_spectrum": (
            "Neonatal severe recessive (diffuse or focal, GIR 15–20 mg/kg/min, macrosomia) → "
            "Neonatal moderate recessive (GIR 10–15, diazoxide may partially respond) → "
            "Dominant heterozygous (mild, diazoxide-responsive, may not present until infancy/childhood) → "
            "Incidental dominant (hyperinsulinism detected only on provocation testing)"
        ),
        "founder_variant": (
            "p.Phe1388del — Ashkenazi Jewish founder (40% of Ashkenazi CHI); "
            "p.Ala1185Val — Finnish founder; "
            "p.Arg1394His — Saudi/Arab founder. "
            "Founder variant screening before full sequencing in high-risk ethnic groups."
        ),
        "key_variants": [
            "p.Phe1388del — Ashkenazi Jewish founder; recessive severe; diffuse/focal",
            "p.Ala1185Val — Finnish founder; recessive severe",
            "p.Arg1394His — Saudi/Arab founder; recessive severe",
            "p.Glu208Lys — dominant; mild; diazoxide-responsive",
            "p.Val187Asp — recessive; diazoxide-resistant",
        ],
        "seed": SEED_BASE + 0,
    },
    # ── KCNJ11 — Kir6.2 K-ATP channel ────────────────────────────────────────────
    {
        "gene": "KCNJ11", "alias": "KCNJ11 — Kir6.2 K-ATP Channel · CHI (2nd most common, focal/diffuse)",
        "aa": "390 aa", "kDa": "45 kDa",
        "gene_class": "K-ATP channel pore subunit (Kir6.2, inward-rectifier K+ channel)",
        "chi_subgroup": "K-ATP channel defects (ABCC8 · KCNJ11)",
        "locus": "11p15.1", "omim_gene": 600937,
        "phenotype": "2nd most common CHI gene; Kir6.2 pore; recessive = focal/diffuse (same as ABCC8); GOF in SAME gene → neonatal diabetes (opposite phenotype — channel stays OPEN)",
        "disease": (
            "KCNJ11 biallelic loss-of-function → CHI type 2 (same K-ATP channel as ABCC8). "
            "KCNJ11 encodes Kir6.2 (390aa, 45kDa), the pore-forming subunit of the "
            "octameric K-ATP channel (4×Kir6.2 + 4×SUR1/ABCC8). Kir6.2 forms the actual "
            "K+ pore; ATP binds to Kir6.2's N-terminus to close the pore. Loss → constitutive "
            "closure → permanent depolarisation → uncontrolled insulin secretion. "
            "Clinically IDENTICAL to ABCC8-CHI: severe neonatal hypoglycemia, macrosomia, "
            "absent ketones, elevated GIR. Same focal/diffuse histological spectrum with same "
            "paternal LOH mechanism for focal form. Same 18F-DOPA PET approach. "
            "KEY BIDIRECTIONAL PHENOTYPE: Gain-of-function KCNJ11 mutations → "
            "NEONATAL DIABETES MELLITUS (NDM, OMIM #600937) — K-ATP stays OPEN → "
            "hyperpolarisation → no Ca2+ influx → NO insulin release → neonatal "
            "insulin-dependent diabetes. NDM responds dramatically to SULFONYLUREA (glibenclamide) "
            "which CLOSES K-ATP channel. E227K, R201H, V59M are the most common NDM variants. "
            "Incidence of KCNJ11-CHI: ~15% of CHI cases."
        ),
        "inheritance": (
            "Autosomal recessive (LOF → CHI) or Autosomal dominant GOF (→ neonatal diabetes). "
            "KCNJ11 11p15.1 adjacent to ABCC8. Same paternal LOH focal mechanism as ABCC8. "
            "De novo dominant GOF mutations are common in neonatal diabetes (not inherited)."
        ),
        "hallmark": (
            "KCNJ11 HALLMARKS: "
            "(1) IDENTICAL TO ABCC8-CHI BIOCHEMICALLY: cannot distinguish KCNJ11 vs ABCC8 "
            "by clinical phenotype or biochemistry — only gene sequencing differentiates; "
            "both on chromosome 11p15.1 (adjacent genes, often sequenced together); "
            "(2) BIDIRECTIONAL PHENOTYPE — SAME GENE, OPPOSITE DISEASES: "
            "LOF (CHI) = channel closed = hyperinsulinism; "
            "GOF (NDM) = channel open = neonatal insulin-dependent diabetes; "
            "GOF NDM treated with SULFONYLUREA (not insulin) — K-ATP can be closed pharmacologically; "
            "LOF CHI treated with diazoxide (open channel) if some residual channel function; "
            "(3) 18F-DOPA PET: same principle as ABCC8 — focal vs diffuse distinction; "
            "(4) DEVELOPMENTAL DELAY IN NDM: some GOF mutations (V59M) → DEND syndrome "
            "(Developmental delay, Epilepsy, Neonatal Diabetes) — Kir6.2 GOF in neurons "
            "causes neurological features beyond diabetes; sulfonylurea also improves neuro; "
            "(5) ABSENT KETONES at hypoglycemia same as ABCC8"
        ),
        "key_ddx": (
            "KCNJ11 DDx: "
            "(1) ABCC8-CHI: clinically identical; sequencing required; "
            "(2) KCNJ11-GOF neonatal diabetes: opposite phenotype; "
            "hyperglycemia not hypoglycemia; respond to sulfonylurea; "
            "(3) GCK-NDM: permanent neonatal diabetes from glucokinase LOF; "
            "does NOT respond to sulfonylurea (not K-ATP mediated); "
            "(4) Type 1 diabetes (autoimmune): not neonatal; positive islet antibodies; "
            "(5) GLUT2 deficiency (GSD XI, Fanconi-Bickel): glucose and galactose intolerance; "
            "Fanconi syndrome; not hyperinsulinism"
        ),
        "diet_treatment": (
            "Same as ABCC8-CHI: IV dextrose (GIR 15–20 mg/kg/min), diazoxide trial, octreotide, "
            "surgical 18F-DOPA PET-guided if medically refractory. "
            "For KCNJ11-GOF (NDM): sulfonylurea (glibenclamide 0.1–0.5 mg/kg/day) — "
            "oral, not injectable; transitions patient off insulin; dramatic response; "
            "continue sulfonylurea lifelong; some DEND syndrome improves neurologically."
        ),
        "gene_therapy_status": (
            "No approved gene therapy for KCNJ11-CHI. "
            "KCNJ11-GOF (NDM) treated with sulfonylurea — NOT insulin — paradigm of pharmacogenomics. "
            "DEND syndrome (NDM + neuro): sulfonylurea also improves developmental and epilepsy features. "
            "Research: KCNJ11 AAV gene delivery in preclinical models."
        ),
        "critical_ci": (
            "CRITICAL: "
            "(1) Treating KCNJ11-GOF neonatal diabetes with INSULIN instead of SULFONYLUREA — "
            "sulfonylurea is the correct and dramatically effective treatment; "
            "insulin can be discontinued in most cases; "
            "(2) Confusing KCNJ11-LOF (CHI) with KCNJ11-GOF (NDM) — "
            "opposite phenotypes from same gene; know the allele direction; "
            "(3) Not doing 18F-DOPA PET before pancreatectomy — same as ABCC8; "
            "(4) Assuming all neonatal diabetes is autoimmune T1D — "
            "neonatal (<6 months) diabetes is almost NEVER autoimmune; "
            "always genetic; screen for KCNJ11, ABCC8, GCK, INS, others; "
            "(5) DEND syndrome: treat underlying NDM with sulfonylurea; "
            "neurological features may improve — do not just manage epilepsy symptomatically"
        ),
        "nbs_marker": (
            "No specific NBS marker for KCNJ11-CHI. "
            "Detect by hypoglycemia screen on day 1–2 of life. "
            "For KCNJ11-GOF (NDM): glucose >11.1 mmol/L in neonatal period triggers "
            "genetic testing; antibodies negative; C-peptide low but detectable. "
            "Comprehensive CHI/NDM gene panel (ABCC8, KCNJ11, GCK, INS) recommended for "
            "all neonatal diabetes <6 months."
        ),
        "key_biomarker": (
            "CHI: Same as ABCC8 — glucose <2.2 mmol/L, insulin >2 mU/L, GIR >8, "
            "absent ketones (<0.5 mmol/L BHB), absent FFA. "
            "NDM: Glucose >11.1 mmol/L in neonate; C-peptide low; islet antibodies NEGATIVE; "
            "C-peptide >0.2 nmol/L suggests K-ATP related (residual insulin). "
            "KCNJ11 sequencing is the diagnostic test."
        ),
        "severity_spectrum": (
            "CHI-LOF: neonatal severe (same as ABCC8, GIR 15–20, focal/diffuse) → "
            "NDM-GOF (mild): transient NDM resolving in weeks → "
            "NDM-GOF (permanent): lifelong diabetes → "
            "DEND syndrome: NDM + developmental delay + epilepsy (severe GOF, V59M allele)"
        ),
        "founder_variant": (
            "CHI-LOF: no common founder; diverse mutations. "
            "NDM-GOF: R201H — most common worldwide (~30% of KCNJ11-NDM); "
            "V59M — associated with DEND syndrome; "
            "E227K — intermediate DEND."
        ),
        "key_variants": [
            "p.Arg201His (R201H) — most common NDM GOF; permanent NDM; sulfonylurea-responsive",
            "p.Val59Met (V59M) — DEND syndrome; NDM + epilepsy + DD",
            "p.Glu227Lys (E227K) — intermediate DEND; NDM + mild neuro",
            "p.His175Tyr — LOF; recessive CHI; severe neonatal hypoglycemia",
            "p.Arg192Cys — LOF; recessive CHI",
        ],
        "seed": SEED_BASE + 1,
    },
    # ── GCK — Glucokinase gain-of-function ───────────────────────────────────────
    {
        "gene": "GCK", "alias": "GCK — Glucokinase GOF · CHI (diazoxide-RESISTANT, surgical)",
        "aa": "465 aa", "kDa": "52 kDa",
        "gene_class": "Hexokinase IV (glucokinase) — glucose sensor of pancreatic beta-cell",
        "chi_subgroup": "Glucose-sensing enzyme defects (GCK)",
        "locus": "7p13", "omim_gene": 138079,
        "phenotype": "Glucokinase GOF → low glucose threshold for insulin secretion; dominant; diazoxide-RESISTANT; no focal form (germline GOF); near-total pancreatectomy sometimes required",
        "disease": (
            "GCK activating (gain-of-function) mutations → CHI type 3 (OMIM #602485). "
            "GCK encodes glucokinase (hexokinase IV, 465aa, 52kDa), the primary glucose "
            "sensor in pancreatic beta-cells (low affinity, non-saturable at physiological "
            "glucose concentrations, S0.5 ~8 mmol/L normally). Glucokinase phosphorylates "
            "glucose → G6P (rate-limiting step for glycolysis in beta-cells). "
            "The rate of GCK-mediated phosphorylation determines the glucose THRESHOLD "
            "for insulin secretion: normally ~5 mmol/L. "
            "GOF mutations → increased affinity for glucose (lower Km/S0.5) → "
            "threshold SHIFTED DOWN (e.g., to 2.5 mmol/L) → beta-cells fire at concentrations "
            "that are hypoglycemic for other tissues → persistent hyperinsulinism. "
            "Autosomal dominant with variable expressivity. Severity proportional to "
            "degree of shift in glucose threshold. GOF GCK is diffuse (germline) — "
            "NO focal form (contrast with ABCC8/KCNJ11 where somatic LOH creates focal). "
            "CLINICAL: may present in neonates (severe activating mutations) or later "
            "in infancy/childhood (milder). Often diazoxide-resistant because mechanism "
            "is upstream of K-ATP channel (diazoxide opens K-ATP but if GCK drives "
            "excessive ATP regardless, K-ATP closure continues). "
            "KEY DISTINCTION from ABCC8/KCNJ11: GCK-CHI is NEVER focal; never responds "
            "to diazoxide (in most); near-total pancreatectomy sometimes needed. "
            "GCK LOF → MODY2 (mild hyperglycemia, no treatment usually)."
        ),
        "inheritance": (
            "Autosomal dominant (GOF activating mutations → CHI). "
            "GCK 7p13. De novo or inherited. Variable severity. "
            "MODY2 = GCK LOF (haploinsufficiency); opposite phenotype (mild stable hyperglycemia)."
        ),
        "hallmark": (
            "GCK-GOF CHI HALLMARKS: "
            "(1) GLUCOSE THRESHOLD SHIFTED DOWN: beta-cells perceive normal/low glucose as 'fed'; "
            "glucose threshold may be 2.0–3.0 mmol/L (normal ~5); "
            "fasting study shows hyperinsulinism at glucose where normally no insulin expected; "
            "(2) DIAZOXIDE-RESISTANT (key differentiator from ABCC8/KCNJ11 dominant): "
            "K-ATP channel is structurally normal; diazoxide cannot overcome GCK-driven "
            "excessive ATP production; 5-day diazoxide trial → NO response in most; "
            "(3) NO FOCAL FORM: GCK GOF is germline → all beta-cells affected → "
            "18F-DOPA PET shows diffuse uniform uptake (but PET not informative for GCK); "
            "(4) OCTREOTIDE MORE USEFUL: somatostatin analogue suppresses insulin directly; "
            "not targeting K-ATP → effective regardless of channel status; "
            "(5) NEAR-TOTAL PANCREATECTOMY MAY BE REQUIRED for severe GCK-GOF refractory "
            "to all medical therapy; risk: iatrogenic diabetes; "
            "(6) BIDIRECTIONAL GCK PHENOTYPE: GOF → CHI; LOF (het) → MODY2 (mild stable "
            "hyperglycemia, fasting glucose 5.4–8 mmol/L); LOF (biallelic) → neonatal "
            "diabetes (very rare)"
        ),
        "key_ddx": (
            "GCK-GOF DDx: "
            "(1) ABCC8/KCNJ11-CHI: diazoxide-RESPONSIVE (especially dominant variants); "
            "K-ATP channel defect; may be focal (18F-DOPA PET); GCK sequencing normal; "
            "(2) GLUD1-HHS: hyperammonemia (ammonia >100 µmol/L); leucine-sensitive; "
            "diazoxide-responsive; GCK sequencing normal; "
            "(3) GCK-MODY2: opposite phenotype — mild persistent hyperglycemia (not hypoglycemia); "
            "fasting glucose 5.4–8 mmol/L; stable across life; no treatment needed; "
            "(4) GCK biallelic LOF (neonatal diabetes): very rare; permanent insulin-requiring; "
            "not sulfonylurea-responsive (K-ATP not involved)"
        ),
        "diet_treatment": (
            "Diazoxide (usual first line) — UNLIKELY to work in GCK-GOF; "
            "do formal 5-day trial before declaring resistant. "
            "Octreotide (somatostatin analogue): 5–35 µg/kg/day SC; directly suppresses "
            "insulin secretion downstream of GCK; more useful in GCK-CHI than ABCC8-CHI. "
            "Lanreotide (long-acting octreotide) for chronic management. "
            "Sirolimus (rapamycin/mTOR inhibitor): investigational; reduces beta-cell "
            "proliferation and insulin secretion; used in refractory diffuse CHI. "
            "Near-total pancreatectomy (95–98%) for medically refractory severe GCK-CHI: "
            "discuss risk of iatrogenic diabetes, exocrine insufficiency. "
            "High-frequency carbohydrate feeding to maintain glucose during octreotide washout."
        ),
        "gene_therapy_status": (
            "No approved gene therapy for GCK-GOF CHI. "
            "Research: antisense oligonucleotide (ASO) to reduce GCK expression in beta-cells "
            "— preclinical. "
            "GCK activators (e.g. dorzagliatin) being developed for T2D — "
            "CONTRAINDICATED in GCK-GOF CHI (would worsen). "
            "GCK-MODY2 (LOF): no treatment needed for most."
        ),
        "critical_ci": (
            "CRITICAL: "
            "(1) Persisting with diazoxide in GCK-GOF — will not work; advance to octreotide; "
            "(2) Attempting 18F-DOPA PET to look for focal lesion in GCK-GOF — "
            "GCK is germline GOF = ALWAYS diffuse; PET shows diffuse pattern; "
            "no surgical localisation possible; "
            "(3) Using GCK activators (dorzagliatin, zydeliglustat) — these worsen GCK-GOF CHI; "
            "(4) Calling it MODY2 in a neonate — MODY2 causes mild hyperglycemia; "
            "if neonate has hypoglycemia + GCK mutation, it is GOF not LOF; "
            "(5) Overlooking associated MODY2 in parents — if one parent has mild hyperglycemia "
            "and child has GCK-CHI, check if GOF vs LOF (different mutation directions possible "
            "in family)"
        ),
        "nbs_marker": (
            "No NBS marker. Detect by hypoglycemia on day 1–2 (severe GOF) or "
            "incidentally in infancy/childhood (mild GOF). "
            "Glucokinase activating mutation panel or GCK sequencing. "
            "Fasting study: hyperinsulinism at low glucose. "
            "MODY2 (LOF): not diagnosed by NBS; incidental finding of mild fasting "
            "hyperglycemia in childhood/family screening."
        ),
        "key_biomarker": (
            "Plasma glucose <2.2–3.0 mmol/L + insulin >2 mU/L = hyperinsulinism. "
            "Glucose threshold in formal fasting study <3.0 mmol/L (normal 4.5–5 mmol/L). "
            "Ammonia: NORMAL (vs GLUD1 where ammonia elevated). "
            "Diazoxide trial: NO response (K-ATP structurally normal; GCK drives ATP regardless). "
            "18F-DOPA PET: diffuse uniform uptake (no focal form). "
            "GCK sequencing: activating mutation required for diagnosis."
        ),
        "severity_spectrum": (
            "Mild GOF (S0.5 shifted to 4 mmol/L): incidental mild HI, no symptoms → "
            "Moderate GOF (S0.5 shifted to 3 mmol/L): symptomatic fasting hypoglycemia → "
            "Severe GOF (S0.5 shifted to 2 mmol/L): neonatal severe hypoglycemia, GIR 10–15, "
            "octreotide-dependent → "
            "Refractory: requires near-total pancreatectomy"
        ),
        "founder_variant": (
            "No single founder variant. De novo mutations common. "
            "p.Tyr214Cys — activating; severe neonatal CHI; "
            "p.Val62Met — activating; moderate CHI; "
            "p.Ala456Val — activating; published in multiple families."
        ),
        "key_variants": [
            "p.Tyr214Cys — severe GOF activating; neonatal CHI",
            "p.Val62Met — moderate GOF; fasting hypoglycemia",
            "p.Ala456Val — activating; autosomal dominant familial CHI",
            "p.Gly264Ser — mild GOF; incidental hypoglycemia",
            "p.Thr255Lys — GOF; diazoxide-resistant",
        ],
        "seed": SEED_BASE + 2,
    },
    # ── GLUD1 — GDH GOF (HHS) ────────────────────────────────────────────────────
    {
        "gene": "GLUD1", "alias": "GLUD1 — GDH GOF · Hyperinsulinism-Hyperammonemia Syndrome (HHS)",
        "aa": "558 aa", "kDa": "61 kDa",
        "gene_class": "Glutamate dehydrogenase (GDH) — mitochondrial allosteric enzyme",
        "chi_subgroup": "Enzyme gain-of-function / signalling defects (GLUD1 · HADH)",
        "locus": "10q23.3", "omim_gene": 138130,
        "phenotype": "GDH GOF → HHS: ONLY CHI gene with hyperammonemia; leucine-sensitive; diazoxide-responsive; ammonia 100–200 µmol/L (ASYMPTOMATIC); no focal form",
        "disease": (
            "GLUD1 activating (gain-of-function) mutations → Hyperinsulinism-Hyperammonemia "
            "Syndrome (HHS, OMIM #606762). GLUD1 encodes glutamate dehydrogenase (GDH, 558aa, "
            "61kDa), a mitochondrial matrix enzyme converting glutamate → alpha-ketoglutarate + "
            "NH3 (or reverse). In normal beta-cells, GDH is allosterically inhibited by GTP "
            "and ATP (prevents overcatabolism of glutamate when glucose is already high). "
            "GOF mutations → loss of GTP/ATP allosteric inhibition → GDH hyperactive → "
            "excess alpha-KG enters TCA → excess ATP generated → K-ATP closes → "
            "insulin secreted. Simultaneously, excess NH3 from GDH reaction → hyperammonemia "
            "(liver also affected: hepatic GDH hyperactive → excess NH3 → mild hyperammonemia; "
            "normally buffered by urea cycle but chronically elevated). "
            "CLINICAL: Neonatal or infantile-onset hypoglycemia, particularly postprandial "
            "(protein/leucine meals are the major trigger — leucine is an allosteric activator "
            "of GDH, drives even more activity in GOF setting). Fasting hypoglycemia also occurs. "
            "Ammonia chronically 100–200 µmol/L but ASYMPTOMATIC (encephalopathy does NOT occur "
            "at these levels; children go years without recognition). "
            "Seizures occur (hypoglycemia-induced AND possibly direct ammonia neurotoxicity). "
            "Diazoxide-responsive in almost all patients (K-ATP channel structurally normal; "
            "diazoxide effectively opens it → suppresses GDH-driven insulin). "
            "Incidence: ~1/100,000; second most common genetic CHI after ABCC8/KCNJ11."
        ),
        "inheritance": (
            "Autosomal dominant (GOF). GLUD1 10q23.3. "
            "~50–70% de novo mutations; remainder familial dominant. "
            "Full penetrance but variable severity. "
            "Both sexes affected equally."
        ),
        "hallmark": (
            "GLUD1/HHS HALLMARKS: "
            "(1) HYPERAMMONEMIA IS PATHOGNOMONIC: ammonia 100–200 µmol/L in EVERY HHS patient; "
            "persistent, not episodic; ASYMPTOMATIC (no acute encephalopathy at these levels); "
            "DDx from UCD where ammonia >500 µmol/L with acute encephalopathy; "
            "CRITICAL: GLUD1 is the ONLY CHI gene with elevated ammonia; "
            "if CHI + elevated ammonia → test GLUD1 first; "
            "(2) LEUCINE-SENSITIVE (protein-sensitive): leucine activates GDH allosterically; "
            "protein meals → postprandial hypoglycemia (30–60 min after meal); "
            "fasting also triggers (less so); leucine-free/low-protein diet helps adjunctively; "
            "(3) DIAZOXIDE-RESPONSIVE: K-ATP channel structurally normal; "
            "diazoxide opens K-ATP → counter-acts GDH-driven closure; effective in >90%; "
            "(4) SEIZURES: dual mechanism — hypoglycemia AND ammonia; "
            "correct glucose first; diazoxide for long-term HI control; "
            "(5) NO FOCAL FORM: germline GOF → all beta-cells affected; "
            "18F-DOPA PET: diffuse pattern; no surgical cure; "
            "(6) WEIGHT GAIN on diazoxide therapy: monitor; "
            "add hydrochlorothiazide for fluid retention"
        ),
        "key_ddx": (
            "GLUD1/HHS DDx: "
            "(1) Urea cycle disorders (OTC, CPS1, ASS1, ASL): "
            "ammonia >500 µmol/L (much higher); acute encephalopathy; hyperammonemic crisis; "
            "low/absent citrulline; orotic acid elevated (OTC); "
            "NO hypoglycemia (unless secondary); "
            "(2) ABCC8/KCNJ11-CHI: no hyperammonemia; diazoxide-responsive (dominant) or "
            "resistant (recessive); K-ATP channel defect; "
            "(3) HADH (SCHAD-CHI): protein-sensitive HI (same); NO hyperammonemia; "
            "C4OH elevated on acylcarnitine; recessive; GDH disinhibition (same pathway); "
            "(4) Organic acidemias with secondary hyperammonemia (PA, MMA): "
            "ketoacidosis present; acylcarnitine profile abnormal; "
            "(5) Biotinidase/holocarboxylase deficiency: biotin-responsive; "
            "mixed acidosis; not purely HI"
        ),
        "diet_treatment": (
            "Diazoxide (first line): 5–15 mg/kg/day + hydrochlorothiazide; "
            "effective in >90% of GLUD1/HHS; "
            "Long-term goal: normoglycemia with controlled ammonia. "
            "Protein/leucine restriction: avoid large protein loads; "
            "leucine-free amino acid supplement NOT required; "
            "moderate protein restriction sufficient; regular carbohydrate supplementation. "
            "Octreotide: second-line if diazoxide insufficient. "
            "Low-protein diet adjunct to diazoxide. "
            "Avoid prolonged fasting. Cornstarch (uncooked) before sleep for fasting hypoglycemia. "
            "Do NOT give VPA (valproate): VPA inhibits GDH → worsens hyperammonemia "
            "(RELATIVE contraindication; LEV or LCM preferred for seizures)."
        ),
        "gene_therapy_status": (
            "No approved gene therapy for GLUD1-HHS. "
            "Diazoxide is highly effective — most patients managed lifelong on diazoxide. "
            "Research: GDH-specific allosteric inhibitors in development. "
            "Long-term: diazoxide-treated patients have normal intellectual development "
            "if hypoglycemia avoided in early life."
        ),
        "critical_ci": (
            "CRITICAL: "
            "(1) Missing hyperammonemia — test ammonia in ALL CHI patients; "
            "if elevated → GLUD1 diagnosis; "
            "(2) Confusing with urea cycle disorders: HHS ammonia 100–200 µmol/L (asymptomatic); "
            "UCD ammonia >500 (encephalopathic); different magnitude, different urgency; "
            "(3) VPA in GLUD1-HHS: RELATIVE CI — VPA inhibits GDH; "
            "can worsen hyperammonemia and trigger HI; prefer levetiracetam or lamotrigine; "
            "(4) Not restricting protein: leucine-sensitive → avoid high-protein meals "
            "without simultaneous carbohydrate; "
            "(5) Stopping diazoxide prematurely — HHS is permanent; "
            "do not expect spontaneous resolution (unlike HNF4A-CHI)"
        ),
        "nbs_marker": (
            "No NBS marker for GLUD1-HHS. "
            "Detect by hypoglycemia screen OR ammonia measurement in early infancy. "
            "Ammonia should be measured in ALL infants with unexplained hypoglycemia. "
            "Diagnosis: simultaneous glucose + insulin + ammonia at hypoglycemia; "
            "ammonia >100 µmol/L + hyperinsulinism = GLUD1 until proven otherwise. "
            "GLUD1 sequencing to confirm."
        ),
        "key_biomarker": (
            "Plasma ammonia 100–200 µmol/L (PATHOGNOMONIC for HHS among CHI types). "
            "Plasma glucose <2.2 mmol/L + insulin >2 mU/L at hypoglycemia. "
            "Postprandial hypoglycemia 30–60 min after protein meal (leucine-sensitive pattern). "
            "Beta-hydroxybutyrate <0.5 mmol/L (absent ketones). "
            "No acylcarnitine abnormality (vs HADH where C4OH elevated). "
            "Urine amino acids: normal; orotic acid: normal (vs UCD). "
            "GLUD1 sequencing: activating mutation in allosteric domain."
        ),
        "severity_spectrum": (
            "Mild GOF (moderate ammonia 100 µmol/L, mild HI): asymptomatic discovered on screening → "
            "Moderate (ammonia 150 µmol/L, fasting + postprandial HI, seizures): "
            "diazoxide-controlled → "
            "Severe (ammonia 200+ µmol/L, refractory HI, neonatal presentation): "
            "diazoxide + octreotide + protein restriction"
        ),
        "founder_variant": (
            "p.Arg269His — most common HHS variant (~25% of cases); "
            "allosteric regulatory domain (antenna region); "
            "p.Ser445Leu — common; "
            "p.Arg221Cys — moderate severity. "
            "Mutations cluster in antenna/allosteric inhibition domain of GDH."
        ),
        "key_variants": [
            "p.Arg269His — most common HHS; allosteric antenna; de novo dominant",
            "p.Ser445Leu — common; moderate HHS",
            "p.Arg221Cys — allosteric domain; moderate severity",
            "p.His454Tyr — severe HHS; neonatal",
            "p.Ser448Pro — allosteric domain; familial",
        ],
        "seed": SEED_BASE + 3,
    },
    # ── HADH — SCHAD (protein-sensitive CHI) ─────────────────────────────────────
    {
        "gene": "HADH", "alias": "HADH — SCHAD Deficiency · Protein-Sensitive CHI · GDH Disinhibition",
        "aa": "314 aa", "kDa": "35 kDa",
        "gene_class": "Short-chain L-3-hydroxyacyl-CoA dehydrogenase (SCHAD) — FAO enzyme with CHI phenotype",
        "chi_subgroup": "Enzyme gain-of-function / signalling defects (GLUD1 · HADH)",
        "locus": "4q22.1", "omim_gene": 601609,
        "phenotype": "SCHAD loss → GDH disinhibition → CHI; protein-sensitive; C4OH on NBS acylcarnitine; diazoxide-responsive; recessive; NO elevated ammonia",
        "disease": (
            "HADH biallelic loss → Congenital Hyperinsulinism due to SCHAD deficiency "
            "(OMIM #231530). HADH encodes SCHAD (short-chain 3-hydroxyacyl-CoA dehydrogenase, "
            "314aa, 35kDa), a mitochondrial enzyme of the beta-oxidation cycle. "
            "KEY MECHANISM: SCHAD directly INHIBITS GDH (glutamate dehydrogenase) "
            "through physical protein-protein interaction. When SCHAD is absent, "
            "GDH is disinhibited → hyperactive → same mechanism as GLUD1-GOF but "
            "via loss of inhibitory partner rather than direct activating mutation. "
            "Result: protein/leucine meals → GDH hyperactivation → excess ATP → "
            "K-ATP closes → insulin secreted → hypoglycemia. "
            "The ONLY CHI gene with a FAO enzyme defect causing HI (not because of "
            "FAO failure per se but because of GDH disinhibition). "
            "CLINICAL: protein-sensitive hypoglycemia (protein meals trigger), "
            "fasting hypoglycemia. Diazoxide-responsive (K-ATP structurally normal). "
            "Autosomal recessive. 3-Hydroxybutyrylcarnitine (C4OH) elevated on "
            "acylcarnitine (both NBS and plasma) — detected by newborn screening in some programmes."
        ),
        "inheritance": (
            "Autosomal recessive. HADH 4q22.1. "
            "Both parents carriers; ~1/4 risk per pregnancy. "
            "Higher prevalence in consanguineous populations."
        ),
        "hallmark": (
            "HADH/SCHAD CHI HALLMARKS: "
            "(1) GDH DISINHIBITION MECHANISM: SCHAD physically inhibits GDH; "
            "SCHAD loss → GDH hyperactive → same downstream effect as GLUD1-GOF; "
            "key difference: HADH is LOSS of inhibitor (not gain of GDH activity); "
            "(2) PROTEIN-SENSITIVE (same as GLUD1): high protein meal → triggers HI; "
            "leucine is the trigger amino acid → protein restriction helps; "
            "(3) C4OH (3-HYDROXYBUTYRYLCARNITINE) ELEVATED: "
            "acylcarnitine panel shows elevated C4OH (same as SCHAD deficiency in FAO atlases); "
            "detectable on NBS acylcarnitine in some panels; "
            "this is the biochemical clue to HADH vs GLUD1 (GLUD1 has normal acylcarnitines); "
            "(4) NO HYPERAMMONEMIA: unlike GLUD1-HHS, HADH-CHI does NOT cause hyperammonemia; "
            "GDH disinhibition via HADH is more regulated/partial than GLUD1-GOF; "
            "ammonia NORMAL — key distinguishing point; "
            "(5) DIAZOXIDE-RESPONSIVE: K-ATP channel intact; diazoxide effective; "
            "(6) RECESSIVE: both parents carriers; full sibling recurrence risk 25%"
        ),
        "key_ddx": (
            "HADH DDx: "
            "(1) GLUD1-HHS: protein-sensitive (same) but WITH hyperammonemia (100–200 µmol/L); "
            "HADH = NO hyperammonemia; GLUD1 = YES hyperammonemia — key split; "
            "(2) ABCC8/KCNJ11-CHI: no protein sensitivity pattern; no C4OH; "
            "K-ATP mutation; may be focal; "
            "(3) FAO disorders on NBS: C4OH also elevated in SCHAD deficiency in FAO context; "
            "but FAO SCHAD = hypoketotic hypoglycemia during fasting (not HI); "
            "HADH-CHI = HI with protein trigger and diazoxide-responsive; "
            "distinguish by insulin level and diazoxide response; "
            "(4) GCK-GOF: no protein sensitivity; diazoxide-resistant; GCK mutation"
        ),
        "diet_treatment": (
            "Diazoxide (first line): 5–15 mg/kg/day; effective in HADH-CHI (K-ATP intact). "
            "Protein restriction: avoid excess leucine/protein loads; "
            "regulate protein intake to prevent postprandial trigger. "
            "Leucine-free amino acid formula: sometimes used in severe cases. "
            "Octreotide: second-line if diazoxide insufficient. "
            "Avoid prolonged fasting (fasting can trigger via fat oxidation and GDH disinhibition). "
            "Frequent carbohydrate feeding / cornstarch for overnight fast."
        ),
        "gene_therapy_status": (
            "No approved gene therapy for HADH-CHI. "
            "Most patients managed on diazoxide long-term. "
            "Protein restriction + diazoxide usually sufficient. "
            "Research: understanding HADH-GDH interaction for drug targeting."
        ),
        "critical_ci": (
            "CRITICAL: "
            "(1) Confusing HADH-CHI with pure FAO-SCHAD deficiency: "
            "FAO-SCHAD is in the FAO atlas (hypoketotic fasting hypoglycemia); "
            "HADH-CHI = HI mechanism via GDH disinhibition + protein trigger; "
            "the same gene, different clinical expression depending on context; "
            "(2) Looking for hyperammonemia to confirm protein-sensitive HI — "
            "HADH does NOT cause hyperammonemia (only GLUD1 does); "
            "(3) Missing the protein-sensitive pattern — take detailed dietary history; "
            "hypoglycemia 30–60 min after high-protein meal is the clue; "
            "(4) Treating as FAO emergency (GIR without diazoxide) — "
            "diazoxide is effective; use it"
        ),
        "nbs_marker": (
            "C4OH (3-hydroxybutyrylcarnitine) elevated on acylcarnitine NBS panel "
            "(same marker as FAO-SCHAD). "
            "NBS detection depends on programme; not universal. "
            "Confirmation: CHI biochemistry (insulin + glucose + BHB at hypoglycemia) + "
            "HADH sequencing. "
            "Ammonia measurement to exclude GLUD1-HHS (should be NORMAL in HADH-CHI)."
        ),
        "key_biomarker": (
            "Plasma C4OH (3-hydroxybutyrylcarnitine) elevated (acylcarnitine panel). "
            "Plasma glucose <2.2 mmol/L + insulin >2 mU/L at hypoglycemia. "
            "Beta-hydroxybutyrate <0.5 mmol/L (absent ketones). "
            "Ammonia: NORMAL (unlike GLUD1-HHS). "
            "Protein challenge: post-protein hypoglycemia documented. "
            "HADH enzyme activity in lymphocytes or fibroblasts: reduced. "
            "HADH sequencing: biallelic pathogenic variants."
        ),
        "severity_spectrum": (
            "Mild (C4OH elevated, mild postprandial HI, no symptoms): detected on NBS → "
            "Moderate (symptomatic protein-sensitive HI, diazoxide-controlled) → "
            "Severe (neonatal presentation, fasting + postprandial HI, higher GIR needed)"
        ),
        "founder_variant": (
            "p.Gln181Glu — founder in Saudi/Arab populations; severe CHI; "
            "c.636+1G>A — splice site; founder in some European families. "
            "Diverse mutations in consanguineous populations."
        ),
        "key_variants": [
            "p.Gln181Glu — Saudi founder; recessive; severe protein-sensitive CHI",
            "c.636+1G>A — splice site; European; recessive CHI",
            "p.Glu170Ter — nonsense; recessive; severe CHI",
            "p.Arg236Ser — missense; moderate CHI",
            "p.His153Tyr — missense; mild CHI",
        ],
        "seed": SEED_BASE + 4,
    },
    # ── HNF4A — neonatal transient CHI + adult MODY1 ─────────────────────────────
    {
        "gene": "HNF4A", "alias": "HNF4A — Neonatal Transient CHI + Adult MODY1 · Macrosomia 56%",
        "aa": "474 aa", "kDa": "53 kDa",
        "gene_class": "Nuclear receptor transcription factor (HNF4α) — liver/pancreas gene regulation",
        "chi_subgroup": "Transcription factor defects (HNF4A · HNF1A)",
        "locus": "20q13.12", "omim_gene": 600281,
        "phenotype": "Neonatal CHI (transient, weeks to months) + macrosomia (56%); diazoxide-responsive; SAME dominant mutation later causes adult MODY1 (hyperglycemia); autosomal dominant",
        "disease": (
            "HNF4A heterozygous dominant loss-of-function → Neonatal CHI (OMIM #125850) that "
            "transitions to adult MODY1 (hepatocyte nuclear factor 4-alpha MODY). "
            "HNF4A encodes hepatocyte nuclear factor 4 alpha (474aa, 53kDa), a nuclear "
            "receptor transcription factor critical for pancreatic beta-cell development, "
            "liver lipid/glucose metabolism, and kidney tubular function. "
            "In neonates: HNF4A haploinsufficiency → early exaggerated insulin secretion "
            "(mechanism incompletely understood; possibly altered potassium channel regulation "
            "or beta-cell mass/activity regulation). Neonatal CHI is usually transient "
            "(resolves within weeks to months), but can occasionally persist. "
            "MACROSOMIA: fetal hyperinsulinism → LGA (>90th centile at birth in ~56% of cases) — "
            "same mechanism as ABCC8/KCNJ11 but transient. "
            "Diazoxide-responsive (K-ATP channel functionally intact). "
            "ADULT PHENOTYPE: same HNF4A LOF mutation → MODY1 in adulthood "
            "(mild-moderate beta-cell secretory failure → hyperglycemia → "
            "sulfonylurea-responsive, responds to glibenclamide). "
            "LIVER PHENOTYPE: elevated plasma apolipoproteins A2, B, C3, C4; "
            "low HDL cholesterol — HNF4A regulates liver lipid gene expression. "
            "KIDNEY: some HNF4A mutations → renal Fanconi syndrome (tubular transport genes)."
        ),
        "inheritance": (
            "Autosomal dominant (haploinsufficiency). HNF4A 20q13.12. "
            "Penetrance high but expressivity variable. "
            "Neonatal CHI phase in 54% of heterozygous carriers. "
            "All carriers eventually develop MODY1 hyperglycemia (variable age of onset). "
            "De novo or inherited; if inherited, parent may already have MODY1 diabetes."
        ),
        "hallmark": (
            "HNF4A CHI HALLMARKS: "
            "(1) TRANSIENT NEONATAL CHI: CHI resolves spontaneously in weeks to months "
            "(occasionally persists longer); contrast with ABCC8/KCNJ11 severe CHI which "
            "does not resolve without treatment; "
            "(2) MACROSOMIA IN 56%: birthweight >90th centile; fetal hyperinsulinism; "
            "key clue — neonatal hypoglycemia + macrosomia → consider HNF4A (and HNF1A); "
            "(3) DIAZOXIDE-RESPONSIVE: K-ATP channel intact; diazoxide opens channel; "
            "usually short course needed as CHI resolves; "
            "(4) SAME MUTATION → BIPHASIC PHENOTYPE OVER LIFE: "
            "birth: neonatal CHI (excess insulin) → "
            "childhood: euglycemia → "
            "adulthood: MODY1 (insufficient insulin, hyperglycemia); "
            "this paradox is HNF4A hallmark; beta-cell function overshoots in neonates, "
            "then progressively declines with haploinsufficiency; "
            "(5) LIVER LIPID ABNORMALITY: elevated Apo-A2, -B, -C3, -C4; low HDL; "
            "triglyceride-rich lipoproteins; not severe but a diagnostic clue; "
            "(6) PARENT HAS MODY1: if parent has mild-moderate diabetes + low HDL + "
            "sulfonylurea-responsive → check HNF4A in neonate with CHI"
        ),
        "key_ddx": (
            "HNF4A DDx: "
            "(1) ABCC8/KCNJ11-CHI: persistent (not transient); K-ATP channel defect; "
            "focal form possible (18F-DOPA PET); diazoxide resistant if recessive; "
            "(2) HNF1A-CHI: similar transient neonatal CHI + macrosomia + MODY3 in adults; "
            "same CHI phenotype but HNF1A locus (12q24.2); distinguishable only by sequencing; "
            "(3) Beckwith-Wiedemann syndrome (BWS): macrosomia + neonatal HI + overgrowth features "
            "(exomphalos, macroglossia, hemihypertrophy, ear anomalies); 11p15 methylation defect; "
            "(4) Maternal diabetes-induced macrosomia: transient neonatal HI; "
            "maternal T1D/T2D; resolves quickly; no genetic cause; "
            "(5) MODY1 (adults): HNF4A het LOF → mild hyperglycemia; "
            "sulfonylurea-responsive; not insulin-requiring initially"
        ),
        "diet_treatment": (
            "Acute neonatal phase: IV dextrose (GIR usually 6–10, lower than ABCC8-CHI); "
            "diazoxide 5–10 mg/kg/day + hydrochlorothiazide (short course, 1–6 months); "
            "Wean diazoxide as CHI resolves (confirm by fasting challenge off treatment). "
            "Frequent feeding during neonatal phase. "
            "Monitor glucose for several months even after apparent resolution. "
            "MODY1 phase (adulthood): sulfonylurea (glibenclamide) — first-line; "
            "low-dose sufficient in many; metformin if overweight; "
            "avoid high-dose insulin (K-ATP responsive; sulfonylurea preferred)."
        ),
        "gene_therapy_status": (
            "No gene therapy for HNF4A-CHI. "
            "Diazoxide for neonatal phase — temporary use only. "
            "MODY1 adults: sulfonylurea lifelong if needed; GLP-1 RAs being studied. "
            "Genetic counselling for dominant inheritance — each child has 50% risk."
        ),
        "critical_ci": (
            "CRITICAL: "
            "(1) NOT weaning diazoxide after CHI resolves — diazoxide side effects "
            "(hypertrichosis, weight gain, fluid retention, cardiomegaly) accumulate; "
            "do fasting challenges every 3–6 months to check for resolution; "
            "(2) Missing MODY1 diagnosis in adulthood — young adult with T2D-like "
            "diabetes + low HDL + HNF4A family history; "
            "test HNF4A sequencing before starting insulin; "
            "sulfonylurea is first-line, not insulin; "
            "(3) Treating neonatal macrosomia as gestational-diabetes related without "
            "checking parent for MODY1 — if parent has MODY1, child at 50% risk of HNF4A-CHI; "
            "(4) Missing the liver lipid phenotype — HNF4A patients have "
            "hypertriglyceridemia-pattern lipoprotein abnormality; "
            "(5) Expecting MODY1 diabetes to be severe — it is mild-moderate; "
            "sulfonylurea-responsive; does not need high-dose insulin early"
        ),
        "nbs_marker": (
            "No NBS metabolite marker. Detected by macrosomia + neonatal hypoglycemia. "
            "Fasting study confirms HI. HNF4A sequencing for diagnosis. "
            "Neonatal CHI gene panel (ABCC8, KCNJ11, GCK, GLUD1, HNF4A, HNF1A) recommended "
            "when transient CHI + macrosomia pattern present."
        ),
        "key_biomarker": (
            "Birth weight >90th centile (macrosomia, 56% of HNF4A carriers). "
            "Glucose <2.2 mmol/L + insulin >2 mU/L at neonatal hypoglycemia. "
            "Plasma Apo-A2, Apo-B, Apo-C3, Apo-C4 elevated; HDL low. "
            "Diazoxide trial: RESPONSIVE. "
            "CHI RESOLVES spontaneously in weeks to months (watch for resolution). "
            "Adults: fasting glucose 6–8 mmol/L (mild hyperglycemia); HbA1c 6–8%. "
            "HNF4A sequencing: heterozygous pathogenic LOF variant."
        ),
        "severity_spectrum": (
            "Mild carrier (CHI minimal, detected only on screening) → "
            "Moderate neonatal CHI (macrosomia, GIR 6–10, diazoxide-responsive, resolves 3–6 months) → "
            "Severe neonatal CHI (GIR 10–15, persistent 6–12 months) → "
            "Adult MODY1 (mild hyperglycemia; sulfonylurea-responsive)"
        ),
        "founder_variant": (
            "p.Arg154Trp — one of most reported; neonatal CHI phenotype; "
            "Diverse mutations across HNF4A including promoter, exon, splicing. "
            "P2 promoter mutations: pancreas-specific isoform → CHI + MODY1."
        ),
        "key_variants": [
            "p.Arg154Trp — neonatal CHI + MODY1 in adults; dominant",
            "p.Val255Met — CHI + hepatic phenotype; MODY1",
            "c.421+1G>A — splicing; pancreas-predominant; neonatal CHI",
            "P2 promoter deletion — pancreatic HNF4A isoform; CHI predominant",
            "p.Arg311His — severe neonatal CHI; persistent",
        ],
        "seed": SEED_BASE + 5,
    },
    # ── SLC16A1 — MCT1 (exercise-induced HI, EIHI) ───────────────────────────────
    {
        "gene": "SLC16A1", "alias": "SLC16A1 — MCT1 GOF · Exercise-Induced HI (EIHI) · Pyruvate Trigger",
        "aa": "494 aa", "kDa": "54 kDa",
        "gene_class": "Monocarboxylate transporter 1 (MCT1) — pyruvate/lactate plasma membrane transporter",
        "chi_subgroup": "Unique trigger defects (SLC16A1)",
        "locus": "1p13.2", "omim_gene": 600682,
        "phenotype": "MCT1 GOF → pyruvate enters beta-cells during anaerobic exercise → insulin spike → EIHI; diazoxide NOT effective; avoid strenuous exercise; autosomal dominant",
        "disease": (
            "SLC16A1 gain-of-function (promoter or coding) → Exercise-Induced Hyperinsulinism "
            "(EIHI, OMIM #606463). SLC16A1 encodes monocarboxylate transporter 1 (MCT1, 494aa, "
            "54kDa), a facilitated transporter for lactate, pyruvate, ketone bodies across "
            "plasma membranes. In normal pancreatic beta-cells, MCT1 is NOT expressed "
            "(or expressed at very low levels); pyruvate produced during anaerobic exercise "
            "cannot enter beta-cells. In EIHI: SLC16A1 GOF → MCT1 overexpressed in "
            "beta-cells → during anaerobic exercise, circulating pyruvate (rises 2–5× during "
            "intense exercise) enters beta-cells via MCT1 → pyruvate metabolised → "
            "acetyl-CoA enters TCA → ATP rises → K-ATP closes → insulin secreted → "
            "hypoglycemia during or shortly after vigorous anaerobic exercise. "
            "UNIQUELY: hypoglycemia is TRIGGERED BY EXERCISE, not by fasting or protein. "
            "Carbohydrate consumption does NOT trigger (safe to eat normally). "
            "At rest or after moderate aerobic exercise: no hypoglycemia. "
            "Clinical: affected individuals develop exercise-induced hypoglycemia "
            "during intense anaerobic exercise (sprinting, weightlifting, isometric exercise). "
            "Often not recognised until school age/adolescence when exercise participation increases. "
            "Autosomal dominant; de novo or inherited."
        ),
        "inheritance": (
            "Autosomal dominant (gain-of-function, overexpression of MCT1 in beta-cells). "
            "SLC16A1 1p13.2. De novo or familial. "
            "Promoter mutations (activate SLC16A1 expression in beta-cells) or "
            "coding mutations that increase transporter activity. "
            "Both sexes equally affected."
        ),
        "hallmark": (
            "SLC16A1/EIHI HALLMARKS: "
            "(1) EXERCISE IS THE SOLE TRIGGER: hypoglycemia ONLY during anaerobic exercise; "
            "not from fasting alone, not from protein, not from carbohydrates; "
            "this is the pathognomonic pattern; "
            "(2) ANAEROBIC-SPECIFIC: intense anaerobic exercise (sprinting, resistance training) "
            "raises circulating pyruvate/lactate → triggers HI; "
            "moderate aerobic exercise (walking, slow cycling) does NOT trigger; "
            "(3) CARBOHYDRATES SAFE: normal carbohydrate intake does NOT worsen; "
            "unlike GSD-VII (Tarui) where high carbs paradoxically worsen; "
            "glucose/carbohydrate before exercise may actually be protective; "
            "(4) DIAZOXIDE NOT EFFECTIVE: K-ATP channel is structurally normal; "
            "mechanism is pyruvate entry, not channel gating; "
            "diazoxide opens K-ATP but once pyruvate floods cell and drives ATP, "
            "channel closes again regardless; "
            "(5) TREAT WITH EXERCISE RESTRICTION: avoid intense anaerobic exercise; "
            "switch to aerobic activities; pre-exercise glucose; "
            "(6) PYRUVATE INFUSION TEST: IV pyruvate provocation causes insulin secretion "
            "in EIHI patients (diagnostic); contrast with normal individuals who do not "
            "secrete insulin in response to IV pyruvate (MCT1 absent from their beta-cells)"
        ),
        "key_ddx": (
            "SLC16A1/EIHI DDx: "
            "(1) Insulinoma: exercise-unrelated hypoglycemia; MRI/CT detects lesion; "
            "not exercise-specific; "
            "(2) ABCC8/KCNJ11-CHI: persistent fasting HI; not exercise-specific; "
            "diazoxide-responsive (dominant) or resistant (recessive); "
            "(3) Non-islet cell tumour hypoglycemia (IGF-2): exercise-unrelated; "
            "elevated pro-IGF-2; "
            "(4) Reactive (postprandial) hypoglycemia: after meals, not exercise; "
            "dumping syndrome; "
            "(5) Addison's disease: can cause hypoglycemia with exercise but via cortisol deficiency; "
            "low cortisol, electrolyte abnormalities, skin hyperpigmentation; "
            "(6) GSD-V (McArdle): exercise intolerance + elevated CK + myoglobinuria; "
            "lactate FAILS to rise with exercise (opposite pattern: no lactate = no aerobic fuel); "
            "not hypoglycemia"
        ),
        "diet_treatment": (
            "Primary: EXERCISE RESTRICTION — avoid intense anaerobic exercise; "
            "moderate aerobic exercise generally safe; "
            "Pre-exercise glucose ingestion (15–20g carbohydrate 15–30 min before exercise) "
            "may help prevent acute hypoglycemia. "
            "Emergency: glucose (oral if conscious; IV dextrose if not). "
            "Glucagon IM 0.03 mg/kg for acute severe hypoglycemia. "
            "Diazoxide: NOT effective (do not use — mechanism mismatch). "
            "Octreotide: may partially reduce insulin secretion during exercise; "
            "evidence limited; not standard. "
            "Activity modification: switch sports to aerobic activities (swimming, cycling). "
            "Continuous glucose monitoring (CGM) during exercise."
        ),
        "gene_therapy_status": (
            "No approved gene therapy for SLC16A1-EIHI. "
            "Exercise restriction is the main management. "
            "Research: MCT1 inhibitor (AZD3965) developed for oncology — "
            "could theoretically block MCT1 in beta-cells and prevent EIHI; "
            "no clinical trials in EIHI yet. "
            "Genetic counselling: autosomal dominant; 50% recurrence risk."
        ),
        "critical_ci": (
            "CRITICAL: "
            "(1) Prescribing diazoxide — will not work (mechanism is pyruvate entry, not K-ATP); "
            "do not persist with ineffective treatment; "
            "(2) Missing the exercise-trigger pattern — take detailed exercise history; "
            "ask specifically about timing of hypoglycemia relative to exercise; "
            "(3) Not recommending CGM during exercise — without CGM, dangerous episodes; "
            "(4) Allowing competitive anaerobic sport participation without warning; "
            "high-intensity exercise is dangerous; requires activity modification; "
            "(5) Confusing with McArdle syndrome (GSD-V) — McArdle has exercise intolerance "
            "and myoglobinuria (not hypoglycemia); lactate fails to rise in McArdle; "
            "in EIHI lactate rises normally (anaerobic pathway works)"
        ),
        "nbs_marker": (
            "No NBS marker. EIHI typically presents in school age/adolescence when "
            "sports participation exposes the defect. "
            "Diagnosis: careful exercise history + pyruvate infusion test (specialist centre) + "
            "exercise test with glucose monitoring. "
            "SLC16A1 sequencing (including promoter region) to confirm. "
            "CGM during exercise reveals pattern."
        ),
        "key_biomarker": (
            "Plasma glucose <2.2 mmol/L during/immediately after anaerobic exercise. "
            "Plasma insulin elevated at time of exercise-related hypoglycemia. "
            "Pyruvate infusion test: IV pyruvate → insulin secretion (EIHI specific). "
            "Lactate normal at rest; rises appropriately with exercise (anaerobic pathway intact). "
            "SLC16A1 sequencing: GOF promoter or coding variant. "
            "Fasting hypoglycemia: NOT present (or minimal) — exercise is required trigger."
        ),
        "severity_spectrum": (
            "Mild GOF (mild EIHI, only extreme exercise triggers): incidental → "
            "Moderate (competitive sports triggers, requires activity modification) → "
            "Severe GOF (moderate exercise triggers, severe hypoglycemia, injury risk)"
        ),
        "founder_variant": (
            "No single major founder. "
            "Promoter mutations activate SLC16A1 expression in beta-cells: "
            "c.-1009G>A, c.-881G>A — promoter activating variants in EIHI families. "
            "Coding GOF mutations also reported."
        ),
        "key_variants": [
            "c.-1009G>A — promoter activating; MCT1 overexpression in beta-cells; EIHI",
            "c.-881G>A — promoter activating; EIHI; familial dominant",
            "p.Arg166His — coding GOF; EIHI; moderate severity",
            "c.-958C>T — promoter variant; EIHI",
            "p.Thr54Met — coding variant; borderline GOF",
        ],
        "seed": SEED_BASE + 6,
    },
    # ── INSR — Donohue/Rabson-Mendenhall (extreme insulin resistance + HI) ───────
    {
        "gene": "INSR", "alias": "INSR — Insulin Receptor · Donohue Syndrome · Rabson-Mendenhall · Extreme Insulin Resistance",
        "aa": "1382 aa", "kDa": "155 kDa",
        "gene_class": "Receptor tyrosine kinase (insulin receptor) — cell surface receptor for insulin/IGF",
        "chi_subgroup": "Insulin receptor defects (INSR)",
        "locus": "19p13.2", "omim_gene": 147670,
        "phenotype": "Biallelic INSR null → Donohue syndrome (leprechaunism); partial biallelic → Rabson-Mendenhall; extreme insulin resistance → HIGH insulin + paradoxical fasting hypoglycemia; mecasermin (rhIGF-1) therapy; NOT true CHI",
        "disease": (
            "INSR biallelic pathogenic mutations → Donohue syndrome (leprechaunism, OMIM #246200) "
            "or Rabson-Mendenhall syndrome (OMIM #268750) depending on severity. "
            "INSR encodes the insulin receptor (1382aa, 155kDa homodimer), a receptor "
            "tyrosine kinase that mediates all cellular effects of insulin: "
            "IRS-1/PI3K/AKT pathway (glucose uptake, glycogen synthesis, lipogenesis, "
            "anti-apoptosis) and RAS/MAPK pathway (growth, gene expression). "
            "INSR biallelic null (Donohue): complete loss → extreme insulin resistance → "
            "beta-cells compensate by secreting massive insulin (hyperinsulinism is secondary, "
            "appropriate response to resistance) → plasma insulin 1000–10,000 mU/L. "
            "Paradox: fasting hypoglycemia occurs because IGF-1 receptor (structurally similar "
            "to INSR) provides some insulin-like signalling in liver at high insulin levels; "
            "postprandial hyperglycemia also occurs (tissue resistance). "
            "CLINICAL (Donohue): severe intrauterine growth restriction, elfin/leprechaun features "
            "(large eyes, low-set ears, wide mouth, prominent lips), acanthosis nigricans "
            "(diffuse), hirsutism, lipoatrophy, enlarged genitalia, phallic enlargement in males, "
            "breast hypertrophy in females, severe feeding difficulties, cardiomegaly, "
            "fatal in first year of life usually. "
            "CLINICAL (Rabson-Mendenhall): milder biallelic INSR mutations → "
            "similar features but longer survival (teens to early adulthood); "
            "dental dysplasia, acanthosis, pineal hyperplasia. "
            "Mecasermin (recombinant human IGF-1) therapy: bypasses INSR and binds IGF-1R "
            "to provide some downstream signalling."
        ),
        "inheritance": (
            "Autosomal recessive (biallelic loss for Donohue/Rabson-Mendenhall). "
            "INSR 19p13.2. Consanguinity major risk factor. "
            "Heterozygous INSR mutations → Type A insulin resistance (mild). "
            "INSR is also mutated in other insulin resistance syndromes (HAIR-AN, PCOS-severe)."
        ),
        "hallmark": (
            "INSR/Donohue-Rabson HALLMARKS: "
            "(1) EXTREME INSULIN RESISTANCE + EXTREME HYPERINSULINEMIA: "
            "fasting insulin 100–10,000 mU/L (normal <10); "
            "C-peptide massively elevated; "
            "hypoglycemia occurs despite high insulin because all tissues are resistant "
            "except liver which responds partially to high-dose insulin/IGF-1; "
            "(2) NOT TRUE CHI: the beta-cells are NOT autonomous (not a channelopathy); "
            "they secrete insulin appropriately in response to unresponsive tissues; "
            "diazoxide is NOT effective (reducing insulin from responsive beta-cells "
            "worsens hyperglycemia in fed state); "
            "(3) PARADOXICAL BIPHASIC GLUCOSE PATTERN: "
            "fasting hypoglycemia (insulin-mediated hepatic glycogen depletion + IGF-1R-mediated "
            "hepatic glucose uptake at very high insulin) AND "
            "postprandial hyperglycemia (tissue glucose uptake resistance); "
            "(4) ACANTHOSIS NIGRICANS PATHOGNOMONIC: diffuse, severe, axillary/neck/groin; "
            "marker of extreme insulin resistance; "
            "(5) LEPRECHAUN FEATURES (Donohue): dysmorphic facies, lipoatrophy, hirsutism, "
            "genital enlargement; "
            "(6) MECASERMIN (rhIGF-1) THERAPY: bypasses INSR; binds IGF-1R; "
            "reduces hyperglycemia and hyperinsulinemia; "
            "dose: 40–120 µg/kg/dose SC twice daily"
        ),
        "key_ddx": (
            "INSR DDx: "
            "(1) True CHI (ABCC8, KCNJ11, GCK, GLUD1): K-ATP or enzyme defect; "
            "LOWER plasma insulin levels; NO acanthosis or dysmorphism; "
            "diazoxide-responsive (ABCC8 dominant, GLUD1); "
            "(2) Hyperinsulinism-hyperammonemia syndrome (GLUD1): "
            "hyperammonemia; protein-sensitive; diazoxide-responsive; normal insulin receptor; "
            "(3) Type A insulin resistance (heterozygous INSR): "
            "milder; young women; HAIR-AN (hyperandrogenism, insulin resistance, acanthosis); "
            "no severe neonatal phenotype; "
            "(4) Lipodystrophy syndromes: extreme insulin resistance + lipoatrophy; "
            "normal INSR; AGPAT2, BSCL2, LMNA, PPARG mutations; "
            "(5) Neonatal diabetes: hyperglycemia not hypoglycemia; "
            "insulin deficient not resistant; normal or low insulin levels"
        ),
        "diet_treatment": (
            "Mecasermin (recombinant human IGF-1, Increlex): "
            "40–120 µg/kg SC twice daily; give with food (risk hypoglycemia); "
            "bypasses INSR; activates IGF-1R → downstream insulin-like effects; "
            "reduces HbA1c, hyperinsulinemia, growth improvement. "
            "High-carbohydrate diet to manage fasting hypoglycemia. "
            "Avoid fasting (hepatic glycogen depletes rapidly). "
            "Acarbose: may reduce postprandial hyperglycemia. "
            "Diazoxide: NOT effective (would worsen postprandial hyperglycemia). "
            "INSR-targeted antisense therapy: in early clinical trials. "
            "Supportive: manage feeding difficulties, cardiac complications, infections."
        ),
        "gene_therapy_status": (
            "No approved gene therapy for INSR-Donohue/Rabson-Mendenhall. "
            "Mecasermin (rhIGF-1) is the primary disease-modifying therapy. "
            "Research: liver-directed INSR gene replacement (AAV8-INSR); "
            "antisense RNA therapy to restore partial INSR function. "
            "Prognosis: Donohue = fatal in first year of life; Rabson-Mendenhall = teens/young adult."
        ),
        "critical_ci": (
            "CRITICAL: "
            "(1) Using diazoxide — NOT effective; will worsen postprandial hyperglycemia "
            "by further reducing insulin output from already-maximal compensating beta-cells; "
            "(2) Classifying as true CHI channelopathy — INSR is extreme insulin resistance, "
            "not channelopathy; management completely different; "
            "(3) Missing mecasermin — currently the only approved treatment; "
            "start early; refer to specialist centre; "
            "(4) Giving high-dose insulin — no response (resistance complete); "
            "contraindicated in Donohue (will worsen hypoglycemia in fasting); "
            "(5) Not recognising acanthosis as sign of insulin resistance — "
            "in neonate with fasting hypoglycemia + acanthosis: "
            "think INSR first; test fasting insulin (will be massively elevated)"
        ),
        "nbs_marker": (
            "No NBS marker. Diagnosed by severe IUGR, dysmorphic features, "
            "and severe hypoglycemia with paradoxically very high insulin. "
            "Fasting study: insulin >100 mU/L (often >1000) at any glucose. "
            "Acanthosis nigricans in neonate = warning sign. "
            "INSR sequencing: biallelic pathogenic variants."
        ),
        "key_biomarker": (
            "Fasting insulin >100 mU/L (often 1000–10,000 mU/L) at hypoglycemia — "
            "the highest insulin levels of any CHI subtype. "
            "C-peptide massively elevated (endogenous, not exogenous). "
            "Fasting glucose <2.2 mmol/L alternating with postprandial glucose >11 mmol/L. "
            "Acanthosis nigricans (clinical finding). "
            "OGTT: paradoxical — hypoglycemia fasting, hyperglycemia postprandial. "
            "IGF-1 levels: may be low or normal (IGF-1R competence preserved). "
            "INSR sequencing: biallelic pathogenic LOF variants."
        ),
        "severity_spectrum": (
            "Donohue syndrome (biallelic null INSR): fatal neonatal/infantile; "
            "severe IUGR, dysmorphism, cardiomegaly, uncontrolled glucose → "
            "Rabson-Mendenhall syndrome (partial biallelic): "
            "severe insulin resistance, teens/young adults, dental dysplasia, pineal hyperplasia → "
            "Type A insulin resistance (heterozygous): HAIR-AN, young women, PCOS-like"
        ),
        "founder_variant": (
            "No common founder. Diverse biallelic mutations. "
            "Consanguineous populations have higher rate of homozygous LOF. "
            "p.Arg1174Gln — kinase domain; Rabson-Mendenhall; partial LOF; "
            "p.Lys1068Glu — severe; Donohue; kinase domain."
        ),
        "key_variants": [
            "p.Arg1174Gln — kinase domain; Rabson-Mendenhall; severe insulin resistance",
            "p.Lys1068Glu — kinase domain; Donohue; severe LOF",
            "p.Leu1038Pro — alpha-subunit; reduced ligand binding; Donohue",
            "p.Gly1008Val — kinase domain; insulin resistance",
            "p.Arg993Trp — kinase activation loop; severe resistance",
        ],
        "seed": SEED_BASE + 7,
    },
]


# ── Simulate 40 patients per gene ────────────────────────────────────────────────
def _simulate(gene_data, n=40):
    rng = random.Random(gene_data["seed"])
    gene = gene_data["gene"]

    # Age at diagnosis distribution (months for neonatal/infantile forms)
    age_ranges = {
        "ABCC8":  (0.0, 3.0),    # neonates
        "KCNJ11": (0.0, 3.0),    # neonates
        "GCK":    (0.0, 6.0),    # neonatal to infantile
        "GLUD1":  (0.5, 18.0),   # infantile/early childhood
        "HADH":   (0.5, 12.0),   # infantile
        "HNF4A":  (0.0, 1.5),    # neonatal (transient)
        "SLC16A1":(60.0, 180.0), # school-age/adolescent
        "INSR":   (0.0, 1.0),    # neonatal severe (Donohue)
    }
    lo, hi = age_ranges.get(gene, (0, 12))

    diazoxide_resp = {
        "ABCC8":  0.35,   # dominant ~80% but recessive ~5%; mixed cohort
        "KCNJ11": 0.38,
        "GCK":    0.10,   # mostly resistant
        "GLUD1":  0.90,   # very responsive
        "HADH":   0.85,
        "HNF4A":  0.88,
        "SLC16A1":0.10,   # not effective
        "INSR":   0.05,   # not effective
    }

    focal_rate = {
        "ABCC8":  0.30,   # ~30-40% focal in recessive
        "KCNJ11": 0.25,
        "GCK":    0.00,   # never focal (germline GOF)
        "GLUD1":  0.00,
        "HADH":   0.00,
        "HNF4A":  0.00,
        "SLC16A1":0.00,
        "INSR":   0.00,
    }

    patients = []
    for i in range(n):
        age_dx_mo = round(rng.uniform(lo, hi), 1)
        age_dx_y = round(age_dx_mo / 12, 2)
        diaz_resp = rng.random() < diazoxide_resp[gene]
        focal = rng.random() < focal_rate[gene]
        gir = round(rng.uniform(8, 22) if gene in ("ABCC8", "KCNJ11") else
                    rng.uniform(6, 18) if gene == "GCK" else
                    rng.uniform(4, 14) if gene in ("GLUD1", "HADH") else
                    rng.uniform(4, 12) if gene == "HNF4A" else
                    rng.uniform(2, 8)  if gene == "SLC16A1" else
                    rng.uniform(3, 10), 1)
        ammonia_elevated = gene == "GLUD1"
        macrosomia = gene in ("HNF4A", "ABCC8", "KCNJ11") and rng.random() < (
            0.56 if gene == "HNF4A" else 0.40)
        exercise_triggered = gene == "SLC16A1"
        patients.append({
            "patient_id": f"CHI-{gene}-{i+1:03d}",
            "gene": gene,
            "age_dx_months": age_dx_mo,
            "age_dx_y": age_dx_y,
            "diazoxide_responsive": diaz_resp,
            "focal_lesion": focal,
            "gir_mgkgmin": gir,
            "ammonia_elevated": ammonia_elevated,
            "macrosomia": macrosomia,
            "exercise_triggered": exercise_triggered,
            "surgical": (not diaz_resp) and (gene in ("ABCC8", "KCNJ11", "GCK"))
                        and rng.random() < 0.60,
        })
    return patients


# ── Aggregate all patients ────────────────────────────────────────────────────────
def _all_patients():
    result = []
    for g in CHI_GENES:
        result.extend(_simulate(g))
    return result


def _gene_stats(gene_data, patients):
    g = gene_data["gene"]
    pts = [p for p in patients if p["gene"] == g]
    n = len(pts)
    mean_age_dx_y = round(sum(p["age_dx_y"] for p in pts) / n, 2)
    pct_diaz = round(100 * sum(1 for p in pts if p["diazoxide_responsive"]) / n)
    pct_focal = round(100 * sum(1 for p in pts if p["focal_lesion"]) / n)
    pct_surgical = round(100 * sum(1 for p in pts if p["surgical"]) / n)
    return {
        "gene": g,
        "alias": gene_data["alias"],
        "aa": gene_data["aa"],
        "kDa": gene_data["kDa"],
        "gene_class": gene_data["gene_class"],
        "chi_subgroup": gene_data["chi_subgroup"],
        "locus": gene_data["locus"],
        "omim_gene": gene_data["omim_gene"],
        "phenotype": gene_data["phenotype"],
        "inheritance": gene_data["inheritance"],
        "hallmark": gene_data["hallmark"],
        "key_ddx": gene_data["key_ddx"],
        "diet_treatment": gene_data["diet_treatment"],
        "gene_therapy_status": gene_data["gene_therapy_status"],
        "critical_ci": gene_data["critical_ci"],
        "nbs_marker": gene_data["nbs_marker"],
        "key_biomarker": gene_data["key_biomarker"],
        "severity_spectrum": gene_data["severity_spectrum"],
        "founder_variant": gene_data["founder_variant"],
        "key_variants": gene_data["key_variants"],
        "n_patients": n,
        "mean_age_dx_y": mean_age_dx_y,
        "pct_diazoxide_responsive": pct_diaz,
        "pct_focal_lesion": pct_focal,
        "pct_surgical": pct_surgical,
    }


# ── API endpoints ─────────────────────────────────────────────────────────────────
def get_overview():
    patients = _all_patients()
    gene_stats = [_gene_stats(g, patients) for g in CHI_GENES]
    return {
        "atlas": "CHI-Atlas",
        "full_name": "Complete 8-Gene Congenital Hyperinsulinism Atlas",
        "n_genes": len(CHI_GENES),
        "n_patients": len(patients),
        "seeds": list(range(SEED_BASE, SEED_BASE + len(CHI_GENES))),
        "gene_subgroups": {
            "K-ATP channel defects (ABCC8 · KCNJ11)":              ["ABCC8", "KCNJ11"],
            "Glucose-sensing enzyme defects (GCK)":                 ["GCK"],
            "Enzyme GOF / signalling defects (GLUD1 · HADH)":      ["GLUD1", "HADH"],
            "Transcription factor defects (HNF4A)":                 ["HNF4A"],
            "Unique trigger defects (SLC16A1 — EIHI)":             ["SLC16A1"],
            "Insulin receptor defects (INSR)":                      ["INSR"],
        },
        "gene_summary": [
            {
                "gene": g["gene"],
                "locus": g["locus"],
                "gene_class": g["gene_class"],
                "phenotype": g["phenotype"][:120] + "…",
                "chi_subgroup": g["chi_subgroup"],
                "mean_age_dx_y": gs["mean_age_dx_y"],
            }
            for g, gs in zip(CHI_GENES, gene_stats)
        ],
        "critical_clinical_rules": [
            "ABCC8/KCNJ11 CHI: 18F-DOPA PET MANDATORY before pancreatectomy — focal CHI (curative limited resection) vs diffuse (near-total, iatrogenic diabetes).",
            "GLUD1-HHS is the ONLY CHI gene with hyperammonemia — always measure ammonia in every CHI patient.",
            "GCK-GOF CHI is diazoxide-RESISTANT — do not persist with diazoxide; advance to octreotide/surgical options.",
            "KCNJ11-GOF neonatal diabetes: treat with SULFONYLUREA (not insulin) — K-ATP closes pharmacologically.",
            "SLC16A1-EIHI: diazoxide NOT effective; avoid anaerobic exercise; carbohydrate intake is SAFE.",
            "INSR-Donohue/Rabson-Mendenhall: extreme insulin resistance NOT true CHI; diazoxide CONTRAINDICATED; use mecasermin (rhIGF-1).",
            "HNF4A-CHI: transient — wean diazoxide and confirm resolution; same mutation causes adult MODY1 later.",
            "Absent ketones + insulin >2 mU/L at hypoglycemia + GIR >8 mg/kg/min = hyperinsulinism until proven otherwise.",
        ],
        "nbs_note": "HADH-CHI detected on NBS acylcarnitine (C4OH). All other CHI disorders identified by clinical hypoglycemia screen on day 1–2 of life or triggered by specific events (exercise for SLC16A1). No metabolite NBS for ABCC8/KCNJ11/GCK/GLUD1/HNF4A/INSR.",
        "total_patients": len(patients),
    }


def get_breakdown():
    patients = _all_patients()
    gene_stats = [_gene_stats(g, patients) for g in CHI_GENES]
    return {
        "atlas": "CHI-Atlas",
        "total": len(CHI_GENES),
        "total_patients": len(patients),
        "genes": gene_stats,
    }


def get_definitions():
    return {
        "chi_overview": {
            "full_name": "Congenital Hyperinsulinism — 8-Gene Atlas",
            "genes_in_atlas": len(CHI_GENES),
            "collective_incidence": "~1/30,000–1/50,000 live births (higher in consanguineous populations; Saudi 1/2,500)",
            "nbs_note": "HADH-CHI detected by C4OH on NBS acylcarnitine. All others require clinical hypoglycemia screen.",
        },
        "definitions": [
            {"term": "K-ATP Channel (SUR1 + Kir6.2)",
             "definition": "Octameric channel (4×SUR1/ABCC8 + 4×Kir6.2/KCNJ11). Closed by ATP (insulin release triggered). Opened by ADP and diazoxide (insulin release suppressed). Loss → constitutive closure → unregulated insulin."},
            {"term": "18F-DOPA PET Scan (CHI)",
             "definition": "18F-fluorodopa PET: beta-cells take up DOPA as dopamine precursor. In focal CHI: intense focal uptake at adenomatoid zone. In diffuse CHI: uniform pancreatic uptake. CRITICAL for surgical planning: focal = limited resection (curative); diffuse = near-total pancreatectomy."},
            {"term": "Diazoxide",
             "definition": "K-ATP channel opener (acts on SUR1). Opens K-ATP channel → hyperpolarisation → reduced Ca2+ influx → reduced insulin secretion. Effective when K-ATP channel is present and functional (ABCC8 dominant, KCNJ11, GLUD1, HADH, HNF4A). NOT effective if K-ATP absent (recessive ABCC8 loss) or mechanism bypasses K-ATP (GCK upstream, SLC16A1 pyruvate driven, INSR resistance)."},
            {"term": "Hyperinsulinism-Hyperammonemia Syndrome (HHS)",
             "definition": "Caused exclusively by GLUD1 GOF. GDH hyperactive → excess alpha-KG (insulin secretion) + excess NH3 (hyperammonemia). Ammonia 100–200 µmol/L, asymptomatic. Leucine/protein-sensitive. Diazoxide-responsive. ONLY CHI syndrome with elevated ammonia — pathognomonic."},
            {"term": "GIR (Glucose Infusion Rate)",
             "definition": "Milligrams of dextrose per kg per minute required to maintain blood glucose >3.3 mmol/L. Normal hepatic glucose production = 4–6 mg/kg/min. GIR >8 = hyperinsulinism (insulin suppresses hepatic glucose output). GIR 15–20 = severe CHI (ABCC8/KCNJ11 recessive)."},
            {"term": "Focal vs Diffuse CHI",
             "definition": "FOCAL: somatic paternal LOH at 11p15 + germline paternal ABCC8 or KCNJ11 mutation → clonal adenomatoid zone; surgical cure by limited resection. DIFFUSE: biallelic germline → all beta-cells affected; near-total pancreatectomy (95–98%). Distinguished by 18F-DOPA PET."},
            {"term": "Mecasermin (rhIGF-1)",
             "definition": "Recombinant human IGF-1 (Increlex). Bypasses defective insulin receptor (INSR) by activating IGF-1 receptor. Used in Donohue syndrome and Rabson-Mendenhall syndrome. Dose: 40–120 µg/kg SC twice daily with food. Reduces hyperinsulinemia and hyperglycemia; improves growth."},
            {"term": "Exercise-Induced Hyperinsulinism (EIHI)",
             "definition": "Caused by SLC16A1 GOF (MCT1 overexpressed in beta-cells). Anaerobic exercise raises circulating pyruvate → pyruvate enters beta-cells via MCT1 → ATP → insulin secretion → hypoglycemia. Trigger: anaerobic exercise only. Safe: carbohydrates, fasting. Treatment: avoid anaerobic exercise; diazoxide NOT effective."},
            {"term": "MODY Crossover (HNF4A / HNF1A)",
             "definition": "HNF4A/HNF1A dominant LOF → neonatal CHI (excess neonatal insulin, transient) then adult MODY1/MODY3 (progressive beta-cell secretory failure, hyperglycemia). SAME mutation causes opposite glucose phenotypes at different ages. Adult MODY phase: sulfonylurea-responsive."},
            {"term": "Pyruvate Infusion Test",
             "definition": "Diagnostic test for EIHI (SLC16A1). IV pyruvate infusion causes insulin secretion in EIHI patients (MCT1 expressed in beta-cells). Normal individuals do not secrete insulin to pyruvate (no MCT1 in their beta-cells). Performed only at specialist CHI centres."},
        ],
    }


if __name__ == "__main__":
    import json
    print("=== CHI-Atlas Overview ===")
    ov = get_overview()
    print(json.dumps({k: v for k, v in ov.items() if k not in ("gene_summary",)}, indent=2))
    bd = get_breakdown()
    print(f"\n=== CHI-Atlas Breakdown: {bd['total']} genes, {bd['total_patients']} patients ===")
    for g in bd["genes"]:
        print(f"  {g['gene']}: {g['n_patients']} pts, mean dx {g['mean_age_dx_y']}y, "
              f"diazoxide-resp {g['pct_diazoxide_responsive']}%, focal {g['pct_focal_lesion']}%")
