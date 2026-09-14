"""Hereditary Congenital Hyperinsulinism Atlas — 8-Gene Reference
ABCC8-KCNJ11-HADH-GLUD1-GCK-HNF4A-INSR-SLC16A1
320 patients (8 x 40), seeds 2654-2661.
Endpoints: /api/hereditary-chi-atlas/overview|breakdown|definitions
"""
import random

ATLAS_GENES = [
    {
        "gene": "ABCC8",
        "protein": (
            "ABCC8 -- 11p15.1 AR/AD -- 1581aa -- Sulphonylurea-Receptor-1-SUR1-177kDa-"
            "KATP-Channel-Regulatory-Subunit-Congenital-Hyperinsulinism-CHI-AR-AD -- OMIM-Gene-600509-Disease-PHHI-256450"
        ),
        "locus": "11p15.1",
        "protein_size": "1581 aa / 177 kDa",
        "inheritance": (
            "AR (biallelic, severe diffuse/focal CHI) or AD (monoallelic, milder); "
            "ABCC8 encodes SUR1, the regulatory subunit of the pancreatic KATP channel; "
            "Most common genetic cause of CHI: 40-45% of all cases; "
            "AR forms: medically unresponsive, require near-total pancreatectomy; "
            "AD forms: often diazoxide-responsive or spontaneously remit; "
            "Focal CHI (somatic LOH + germline ABCC8 mutation): 18F-DOPA PET-CT localises lesion → curative limited resection; "
            "Diazoxide (KATP opener): first-line; failure = cardinal feature of severe ABCC8/KCNJ11 CHI; "
            "c.3992-9G>A: founder mutation in Ashkenazi Jewish families"
        ),
        "disease_category": (
            "Congenital hyperinsulinism of infancy (CHI); persistent hyperinsulinaemic hypoglycaemia of infancy (PHHI); "
            "KATP-channel disease; SUR1 (ABCC8) + Kir6.2 (KCNJ11) form the octameric KATP channel (4+4) on beta-cell membrane; "
            "SUR1 is the ATP-binding cassette transporter regulatory subunit — senses ADP/ATP ratio; "
            "PATHOPHYSIOLOGY: LOF → KATP channels constitutively closed → persistent membrane depolarisation → "
            "continuous Ca2+ influx → continuous insulin secretion irrespective of glucose level; "
            "DIFFUSE vs FOCAL: AR biallelic = diffuse; Focal = somatic LOH at 11p15 + paternal germline mutation — "
            "FOCAL IS SURGICALLY CURABLE (limited resection vs near-total pancreatectomy for diffuse); "
            "18F-DOPA PET-CT: gold standard to distinguish diffuse vs focal and localise focal lesion; "
            "HISTOLOGY: focal = cluster of enlarged beta-cells with nuclear atypia; diffuse = normal architecture"
        ),
        "disease_pathway": (
            "ABCC8/SUR1 PATHWAY: "
            "Normal beta-cell: glucose metabolism → elevated ATP/ADP ratio → KATP channel opens → K+ efflux → "
            "hyperpolarisation → Ca2+ channels closed → no insulin release; "
            "High glucose: ATP rises → KATP closes → depolarisation → VGCC opens → Ca2+ influx → exocytosis; "
            "ABCC8 LOF: KATP constitutively closed → unregulated insulin secretion → fasting/post-prandial hypoglycaemia; "
            "DIAZOXIDE: KATP opener (acts on SUR1 NBD2) → hyperpolarises membrane → reduces insulin secretion; "
            "FAILURE of diazoxide = KATP cannot be pharmacologically opened (truncating/missense in NBD1-2 or TMD0); "
            "OCTREOTIDE: somatostatin analogue → Gi-coupled → reduces cAMP → reduces exocytosis; "
            "FOCAL: uniparental isodisomy (UPD) of 11p15 maternal allele → loss of IGF2/imprinting + "
            "homozygosity of paternal ABCC8 mutation in focal clone only — "
            "NORMAL surrounding pancreas → CURE by limited resection; "
            "POST-PANCREATECTOMY: diabetes mellitus (near-total) or exocrine insufficiency"
        ),
        "pathognomonic": (
            "ABCC8/CHI DIAGNOSTIC CLUSTER: "
            "1) NEONATAL/INFANTILE HYPOGLYCAEMIA: typically first 72 h to weeks of life; "
            "macrosomic neonates (birth weight >4 kg in 50%); "
            "2) HYPERINSULINAEMIC HYPOGLYCAEMIA: glucose <2.8 mmol/L + detectable insulin (>2 mU/L) + "
            "absent ketones + absent FFA = biochemical PATHOGNOMONIC; "
            "3) HIGH GLUCOSE INFUSION RATE: >8-10 mg/kg/min required to maintain euglycaemia = KATP-channel CHI hallmark; "
            "4) DIAZOXIDE FAILURE: no response to diazoxide 5-15 mg/kg/day in 48-72 h → "
            "STRONGLY suggests ABCC8/KCNJ11; "
            "5) 18F-DOPA PET-CT: mandatory to exclude focal lesion before surgical decision; "
            "6) c.3992-9G>A FOUNDER: Ashkenazi Jewish — screen first in this population; "
            "7) GENETIC TESTING: ABCC8 then KCNJ11 — paired sequencing + deletion analysis; "
            "MANAGEMENT: diazoxide + chlorothiazide (reduces oedema) → if fails: octreotide/nifedipine → "
            "glucagon infusion bridge → 18F-DOPA PET → surgery (focal resection or near-total pancreatectomy)"
        ),
        "treatment": (
            "ABCC8 CHI TREATMENT: "
            "DIAZOXIDE: 5-15 mg/kg/day divided 3 doses; co-prescribe chlorothiazide to prevent fluid retention; "
            "response assessed at 5 days; failure = proceed to imaging + surgery; "
            "OCTREOTIDE: 5-25 mcg/kg/day SC or CSCI; continuous more effective than bolus; risk: necrotising enterocolitis in neonates (use with caution <1 month); "
            "NIFEDIPINE (VGCC blocker): adjunct; rarely sufficient as monotherapy; "
            "SIROLIMUS: mTOR inhibitor; compassionate/case-series use in diazoxide-unresponsive; "
            "SURGERY — FOCAL: 18F-DOPA PET localised → limited pancreatectomy → cure in >90%; "
            "SURGERY — DIFFUSE: near-total (95-98%) pancreatectomy; risk post-op DM ~50% at 10-15 years + exocrine insufficiency; "
            "LONG-TERM: neurological outcomes (psychomotor delay) correlate with duration of hypoglycaemia before diagnosis; "
            "early diagnosis via NBS programmes reduces neurodevelopmental sequelae"
        ),
    },
    {
        "gene": "KCNJ11",
        "protein": (
            "KCNJ11 -- 11p15.1 AR/AD -- 390aa -- Inward-Rectifier-K-Channel-Kir6.2-43kDa-"
            "KATP-Channel-Pore-Subunit-CHI-NDM-DEND-AR-AD -- OMIM-Gene-600937-Disease-CHI-600937"
        ),
        "locus": "11p15.1",
        "protein_size": "390 aa / 43 kDa",
        "inheritance": (
            "AR (biallelic LOF → CHI, severe); AD GOF (activating mutations → neonatal diabetes mellitus NDM, DEND syndrome); "
            "KCNJ11 encodes Kir6.2, the pore-forming subunit of the KATP channel; "
            "CHI from LOF (same mechanism as ABCC8 — KATP constitutively closed); "
            "NDM from GOF (KATP constitutively open → hyperpolarised beta-cell → no insulin → neonatal diabetes); "
            "DEND syndrome (GOF severe): Developmental delay + Epilepsy + Neonatal Diabetes — "
            "KCNJ11 GOF expressed in neurons too; "
            "CRITICAL: KCNJ11-NDM can switch FROM INSULIN to SULPHONYLUREA (glibenclamide) treatment — "
            "dramatic improvement in neurological features of DEND syndrome"
        ),
        "disease_category": (
            "KCNJ11 has a dual disease spectrum — two opposite directions of KATP channel dysfunction: "
            "1) KCNJ11 LOF (CHI): same as ABCC8 LOF — biallelic recessive; constitutively closed KATP; "
            "continuous Ca2+ influx; unregulated insulin secretion; "
            "2) KCNJ11 GOF (NDM/DEND): activating mutations — KATP constitutively open; "
            "beta-cell membrane hyperpolarised → no Ca2+ entry → no insulin → neonatal diabetes onset <6 months; "
            "NEUROLOGICAL: Kir6.2 expressed in neurons — GOF in CNS → seizures + developmental delay (DEND) "
            "or intermediate DEND (IDEND: without epilepsy) or transient NDM (TNDM); "
            "p.R201H and p.V59M: most common DEND mutations; "
            "SULPHONYLUREA SWITCH: glibenclamide closes KATP regardless of ATP (direct SUR1 binding) → "
            "restores insulin secretion AND crosses BBB → improves neurological features in DEND"
        ),
        "disease_pathway": (
            "Kir6.2 forms the inner pore of the KATP channel — four Kir6.2 tetramers surrounded by four SUR1 subunits. "
            "Kir6.2 gate controlled by direct ATP binding (inhibitory) and indirect ADP via SUR1. "
            "KCNJ11 GOF mutations in pore-lining residues (TM2) or ATP-binding site → "
            "reduced ATP sensitivity → channel open even at high glucose (high ATP) → "
            "beta-cell cannot depolarise → insulin secretion absent → NEONATAL DIABETES; "
            "NEURONAL EXPRESSION: Kir6.2 in hippocampus, cortex → GOF → intrinsic hyperexcitability paradox "
            "(open KATP → hyperpolarised neuron → lower resting threshold? No: "
            "effect in excitatory synaptic terminals reduces neurotransmitter release → "
            "net effect: epilepsy through complex circuit mechanisms); "
            "SULPHONYLUREA MECHANISM: binds SUR1 NBD2 → allosterically forces Kir6.2 closed → "
            "insulin secreted; neurological improvement = CNS KATP closure → normalises neuronal circuit; "
            "higher doses needed for DEND than adult T2DM"
        ),
        "pathognomonic": (
            "KCNJ11 DIAGNOSTIC CLUSTER — TWO OPPOSITE PHENOTYPES: "
            "CHI (LOF): identical to ABCC8 CHI — diazoxide-unresponsive hyperinsulinaemic hypoglycaemia; "
            "distinguish from ABCC8 only by gene panel; same 11p15 locus; "
            "NDM/DEND (GOF): "
            "1) NEONATAL DIABETES ONSET <6 months — KCNJ11/ABCC8 GOF explains ~50% of NDM; "
            "2) SULPHONYLUREA RESPONSIVENESS — if NDM onset <6 months → ALWAYS attempt sulphonylurea trial; "
            "3) DEND TRIAD: Developmental delay + Epilepsy + Neonatal Diabetes — "
            "PATHOGNOMONIC for KCNJ11/ABCC8 GOF when all three present; "
            "4) DRAMATIC NEUROLOGICAL IMPROVEMENT on high-dose glibenclamide (0.3-0.5 mg/kg/day) — "
            "seizure frequency ↓, developmental gains; "
            "5) GENE PANEL: screen KCNJ11 + ABCC8 in ALL NDM onset <6 months before committing to insulin; "
            "KEY RULE: Never assume insulin is the only option for neonatal diabetes — "
            "KCNJ11/ABCC8 NDM responds to oral sulphonylurea in >90%"
        ),
        "treatment": (
            "KCNJ11 TREATMENT — BY PHENOTYPE: "
            "CHI (LOF): same as ABCC8 — diazoxide, octreotide, surgery; "
            "NDM (GOF): "
            "SULPHONYLUREA TRANSFER: start glibenclamide 0.1 mg/kg/day, increase to 0.5-1.0 mg/kg/day; "
            "90% transfer successfully from insulin; "
            "DEND: high-dose glibenclamide 0.5-1.0 mg/kg/day; neurological improvement over months; "
            "initiate transfer regardless of age at diagnosis — even adults improve; "
            "MONITORING: glucose + HbA1c; watch for hypoglycaemia during transfer; "
            "continue NEUROLOGICAL review — EEG, development; "
            "REGISTRY: ISPAD/SWEET registry for NDM — collect long-term outcome data"
        ),
    },
    {
        "gene": "HADH",
        "protein": (
            "HADH -- 4q25 AR -- 314aa -- L-3-Hydroxyacyl-CoA-Dehydrogenase-SCHAD-34kDa-"
            "Mitochondrial-FAO-Enzyme-Short-Chain-CHI-AR -- OMIM-Gene-601609-Disease-SCHAD-CHI-609975"
        ),
        "locus": "4q25",
        "protein_size": "314 aa / 34 kDa",
        "inheritance": (
            "AR (biallelic HADH LOF); short-chain L-3-hydroxyacyl-CoA dehydrogenase deficiency (SCHAD); "
            "Rare cause of CHI (~few dozen families described); "
            "Unique mechanism: HADH physically interacts with GDH (GLUD1) and INHIBITS it; "
            "HADH LOF → UNREGULATED GDH activity → excess glutamate oxidation → "
            "excess NADH → KATP closure → insulin secretion; "
            "PROTEIN-SENSITIVE HYPOGLYCAEMIA: high-protein meal triggers hypoglycaemia (leucine/glutamine → GDH activation); "
            "unlike GLUD1 gain-of-function, HADH CHI is NOT associated with hyperammonaemia; "
            "acylcarnitine profile: elevated C4-OH (3-hydroxybutyrylcarnitine) on NBS/plasma acylcarnitines"
        ),
        "disease_category": (
            "HADH (SCHAD) CHI is unique because the enzyme's role in CHI is NOT its FAO catalytic activity but rather "
            "its PROTEIN-PROTEIN INTERACTION with GDH (glutamate dehydrogenase). "
            "HADH binds GDH (GLUD1) → allosteric inhibition of GDH → dampens glutamate-stimulated insulin secretion; "
            "HADH LOF → GDH constitutively overactive → excessive insulin secretion in response to amino acids; "
            "ACYLCARNITINE MARKER: 3-hydroxybutyrylcarnitine (C4-OH) elevated — arises from incomplete SCHAD activity; "
            "PROTEIN-SENSITIVE: amino acid load (glutamine, leucine) → maximal stimulus; "
            "Fasting hypoglycaemia also present; "
            "DIAZOXIDE-RESPONSIVE: unlike ABCC8/KCNJ11 — KATP channel is functional; "
            "GDH-mediated mechanism → KATP can still be opened pharmacologically; "
            "MANAGEMENT: low-protein diet + diazoxide"
        ),
        "disease_pathway": (
            "SCHAD (HADH) is the mitochondrial short-chain fatty acid oxidation enzyme (3-hydroxyacyl-CoA dehydrogenase). "
            "In beta-cells, SCHAD forms a physical complex with GDH at the inner mitochondrial membrane. "
            "SCHAD:GDH INHIBITORY COMPLEX: when SCHAD is present → GDH activity reduced → glutamate oxidation dampened → "
            "lower NADH → KATP remains partially open → controlled insulin secretion; "
            "SCHAD DEFICIENCY (HADH LOF): "
            "loss of GDH inhibition → GDH constitutively active → excess alpha-ketoglutarate from glutamate → "
            "excess NADH → KATP closure → calcium influx → insulin exocytosis regardless of glucose; "
            "PROTEIN LOAD amplifies this: amino acids (esp. glutamine, leucine) → GDH substrate surge → "
            "exaggerated insulin response = post-prandial hypoglycaemia after protein meal; "
            "FAO DEFECT: mild (short-chain) — usually not clinically significant for energy; "
            "C4-OH acylcarnitine: accumulates in plasma/DBS — identified on NBS acylcarnitine panel; "
            "THERAPY: diazoxide opens KATP (intact channel) → effective; low-protein diet reduces GDH stimulus"
        ),
        "pathognomonic": (
            "HADH (SCHAD) CHI DIAGNOSTIC CLUSTER: "
            "1) PROTEIN-SENSITIVE HYPOGLYCAEMIA: post-protein-meal hypoglycaemia — "
            "PATHOGNOMONIC clue; distinguish from GLUD1 (which also has hyperammonaemia); "
            "2) NO HYPERAMMONAEMIA: key DDx from GLUD1 gain-of-function HI/HA syndrome; "
            "HADH CHI = protein-sensitive + NO hyperammonaemia; "
            "3) C4-OH ACYLCARNITINE ELEVATED: 3-hydroxybutyrylcarnitine on newborn screening or plasma; "
            "may trigger NBS flag → investigate further; "
            "4) DIAZOXIDE-RESPONSIVE: unlike KATP-defect CHI; responds to diazoxide 5-10 mg/kg/day; "
            "5) LEUCINE/GLUTAMINE TOLERANCE TEST: controlled amino acid loading → hypoglycaemia with hyperinsulinaemia; "
            "6) URINE: absent/low ketones during hypoglycaemia (hyperinsulinaemia suppresses ketogenesis); "
            "7) GENETIC: biallelic HADH mutations; functional enzyme assay in lymphocytes or fibroblasts; "
            "DDx FLOWCHART: protein-sensitive CHI → check ammonia → if HIGH = GLUD1; if NORMAL = HADH"
        ),
        "treatment": (
            "HADH/SCHAD CHI TREATMENT: "
            "DIAZOXIDE: 5-10 mg/kg/day — effective because KATP channel is normal; "
            "+ chlorothiazide to reduce fluid retention; "
            "DIET: low-protein (restrict leucine/glutamine-rich foods); "
            "protein limit reduces GDH activation → fewer hypoglycaemic episodes; "
            "MONITORING: regular glucose monitoring; protein intake tracked by dietitian; "
            "PROGNOSIS: generally good — diazoxide-responsive, no neurological complications if diagnosed promptly; "
            "NO PANCREATIC SURGERY required (unlike ABCC8/KCNJ11 diazoxide-unresponsive); "
            "LONG-TERM: many patients can be weaned off diazoxide in childhood as CHI improves with age"
        ),
    },
    {
        "gene": "GLUD1",
        "protein": (
            "GLUD1 -- 10q23.3 AD-GOF -- 558aa -- Glutamate-Dehydrogenase-1-GDH-56kDa-"
            "Mitochondrial-Matrix-Enzyme-HI-HA-Syndrome-AD-GOF -- OMIM-Gene-138130-Disease-HIHA-606762"
        ),
        "locus": "10q23.3",
        "protein_size": "558 aa / 56 kDa",
        "inheritance": (
            "AD gain-of-function (usually de novo, some familial); "
            "GLUD1 GOF → hyperinsulinism/hyperammonaemia (HI/HA) syndrome; "
            "Most common cause of non-KATP CHI (after ABCC8/KCNJ11); "
            "Mutations in allosteric inhibitory domain (GTP-binding site, antenna region) → "
            "reduced GTP/ADP inhibition of GDH → constitutive GDH overactivity; "
            "LEUCINE-TRIGGERED: 90% of cases show exaggerated leucine response — protein meals trigger hypoglycaemia; "
            "HYPERAMMONAEMIA: mild-moderate (ammonia 50-200 μmol/L); asymptomatic usually; "
            "may cause protein aversion in affected children — important diagnostic clue"
        ),
        "disease_category": (
            "HI/HA SYNDROME: the ONLY form of CHI with concomitant HYPERAMMONAEMIA — PATHOGNOMONIC DDx feature. "
            "GDH (GLUD1) catalyses reversible: glutamate + NAD+ → alpha-ketoglutarate + NH3 + NADH "
            "in mitochondrial matrix. "
            "Normal: GDH inhibited by GTP (product of oxidative phosphorylation) in liver AND beta-cells. "
            "GOF: GDH uninhibited → "
            "BETA-CELL: excess NADH from glutamate oxidation → KATP closes → Ca2+ → insulin exocytosis; "
            "LIVER: excess NH3 from glutamate deamidation → hyperammonaemia (hepatic GDH overactive); "
            "LEUCINE: positive allosteric activator of GDH → leucine-induced hypoglycaemia; "
            "PROTEIN AVERSION: affected children often avoid protein foods — early behavioural clue; "
            "FASTING HYPOGLYCAEMIA also present but typically milder than KATP forms"
        ),
        "disease_pathway": (
            "GDH is a homohexamer located in mitochondrial matrix, regulated by: "
            "INHIBITORS: GTP (competitive at nucleotide regulatory site), ADP-ribose, ATP; "
            "ACTIVATORS: ADP, leucine; "
            "GOF mutations in antenna region or GTP-binding site → reduced GTP inhibition → "
            "GDH constitutively active even when cell is energy-replete; "
            "BETA-CELL CONSEQUENCE: "
            "After protein meal: glutamine/glutamate → GDH → alpha-KG → excess NADH → KATP closes → insulin spike; "
            "LEUCINE pathway: leucine activates GDH allosterically (positive effector) → same cascade; "
            "HEPATIC CONSEQUENCE: "
            "liver GDH also constitutively active → excess NH3 from hepatic amino acid catabolism → "
            "hyperammonaemia (liver-specific: OTC is normal — no cycle defect); "
            "DIAZOXIDE RESPONSE: GDH-mediated CHI is KATP-intact → diazoxide effectively opens KATP; "
            "PROTEIN RESTRICTION + DIAZOXIDE: dual strategy targets both GDH activation and downstream KATP"
        ),
        "pathognomonic": (
            "GLUD1 HI/HA SYNDROME DIAGNOSTIC CLUSTER: "
            "1) HYPERINSULINAEMIC HYPOGLYCAEMIA + PERSISTENT HYPERAMMONAEMIA — "
            "PATHOGNOMONIC COMBINATION: no other CHI cause causes both; "
            "ammonia typically 50-200 μmol/L (usually asymptomatic — no encephalopathy); "
            "2) PROTEIN-SENSITIVE: post-protein hypoglycaemia within 30-90 min of protein-rich meal; "
            "3) LEUCINE TOLERANCE TEST: IV or oral leucine → exaggerated insulin release + hypoglycaemia; "
            "4) DIAZOXIDE-RESPONSIVE: unlike ABCC8/KCNJ11 — KATP intact; "
            "5) PROTEIN AVERSION BEHAVIOUR: children self-restrict protein foods — behavioural clue; "
            "6) EPILEPSY: ~50% develop epilepsy (focal/generalised) independent of hypoglycaemia — "
            "CNS GDH contribution; valproate CI (elevates ammonia); "
            "7) GENE PANEL: GLUD1 sequencing — exon 7 (antenna) and exons 11-13 (GTP-binding) "
            "are mutation hotspots; "
            "KEY DDx from HADH CHI: both protein-sensitive, but GLUD1 = hyperammonaemia; HADH = C4-OH acylcarnitine"
        ),
        "treatment": (
            "GLUD1 HI/HA TREATMENT: "
            "DIAZOXIDE: 5-10 mg/kg/day — effective (KATP intact); "
            "DIETARY: low-protein/leucine-restricted diet reduces GDH activation; "
            "protein restriction must be balanced with nutritional needs — dietitian essential; "
            "EPILEPSY: standard AEDs but AVOID VALPROATE (inhibits urea cycle → worsens hyperammonaemia); "
            "AMMONIA MONITORING: measure fasting ammonia; supplemental sodium benzoate rarely needed "
            "(ammonia usually clinically insignificant); "
            "PROGNOSIS: hypoglycaemia often improves with age; epilepsy may persist; "
            "some patients develop intellectual disability if hypoglycaemia prolonged before diagnosis"
        ),
    },
    {
        "gene": "GCK",
        "protein": (
            "GCK -- 7p13 AD-GOF/AR-LOF -- 465aa -- Glucokinase-52kDa-"
            "Beta-Cell-Glucose-Sensor-CHI-GOF-MODY2-LOF-AR-LOF -- OMIM-Gene-138079-Disease-CHI-GOF-602485"
        ),
        "locus": "7p13",
        "protein_size": "465 aa / 52 kDa",
        "inheritance": (
            "AD GOF (activating mutations → CHI); AD LOF (inactivating → MODY2); AR LOF (homozygous → permanent neonatal DM); "
            "GCK encodes glucokinase — the pancreatic 'glucose sensor'; "
            "GOF mutations: lower glucose threshold for insulin secretion → set point shifted left; "
            "hypoglycaemia occurs at normal glucose levels (fasting glucose 2.5-3.5 mmol/L); "
            "GOF CHI: diazoxide-responsive; typically mild; some cases spontaneously resolve; "
            "MODY2 (LOF, heterozygous): elevated fasting glucose 5.4-8.3 mmol/L; "
            "benign; no treatment needed; not progressive; PATHOGNOMONIC: stable mild HbA1c elevation from birth"
        ),
        "disease_category": (
            "Glucokinase (GCK) acts as the beta-cell glucose sensor: it phosphorylates glucose to G6P, "
            "initiating glycolysis and ATP production. "
            "GOF (CHI phenotype): mutation lowers S0.5 (substrate concentration for half-maximal activity) "
            "or increases Hill coefficient → GCK more sensitive to glucose → "
            "insulin secreted at glucose levels below normal set-point (~3.5 mmol/L vs normal ~4.5 mmol/L); "
            "Net result: fasting hypoglycaemia (set-point shifted left); "
            "DIAZOXIDE-RESPONSIVE: KATP channel is normal — GCK-driven CHI bypasses KATP; "
            "diazoxide shifts the set-point partially back toward normal; "
            "DUAL SPECTRUM: same gene — opposite mutations = opposite phenotypes "
            "(GOF = hypoglycaemia; heterozygous LOF = MODY2 mild DM; homozygous LOF = permanent NDM)"
        ),
        "disease_pathway": (
            "GCK PATHWAY: Glucose → GCK → Glucose-6-phosphate → glycolysis → ATP → KATP closure → insulin. "
            "GCK is the RATE-LIMITING STEP and GLUCOSE SENSOR. "
            "NORMAL: GCK half-maximal at ~8 mmol/L glucose (low affinity by design, no product inhibition); "
            "allows proportional insulin response to rising postprandial glucose. "
            "GOF (CHI): lowered S0.5 → GCK maximally active at 2-3 mmol/L → "
            "insulin released at fasting concentrations that normally suppress secretion; "
            "SEVERITY: GOF variants range from mild (fasting glucose 3.0-3.5) to severe (recurrent symptomatic hypoglycaemia); "
            "PROGRESSION: some GOF CHI remits in childhood (beta-cell mass adapts); "
            "others persist to adulthood — important natural history consideration; "
            "GCK in LIVER: hepatic glucokinase regulates glycogen synthesis — GOF = enhanced hepatic glucose uptake; "
            "both effects (beta-cell + liver) contribute to euglycaemia at lower glucose concentrations"
        ),
        "pathognomonic": (
            "GCK GOF CHI DIAGNOSTIC CLUSTER: "
            "1) FASTING HYPOGLYCAEMIA with DETECTABLE INSULIN at low glucose levels; "
            "2) MILD-MODERATE SEVERITY: many patients manage with dietary frequency + modest diazoxide; "
            "rare severe cases need surgery; "
            "3) DIAZOXIDE-RESPONSIVE: unlike ABCC8/KCNJ11 — responds to diazoxide (KATP intact); "
            "4) AUTOSOMAL DOMINANT FAMILY HISTORY: 50% inheritance — check parents' fasting glucose; "
            "affected parent may have asymptomatic hypoglycaemia or 'normal' low glucose; "
            "5) FUNCTIONAL TEST: intranasal glucagon stimulation — exaggerated GCK sensitivity; "
            "6) SEQUENCING: GCK exons 1-10 — activating mutations in substrate-binding site or "
            "dimer interface; distinguish from LOF (MODY2) alleles; "
            "MODY2 DDx: fasting glucose 5.4-8.3 mmol/L (MODY2) vs fasting glucose 2.5-3.5 (GOF CHI) — "
            "same gene, opposite disease spectrum"
        ),
        "treatment": (
            "GCK GOF CHI TREATMENT: "
            "DIAZOXIDE: 5-15 mg/kg/day + chlorothiazide; "
            "response typically good; titrate to lowest effective dose; "
            "DIET: frequent meals; avoid prolonged fasting; "
            "WATCHFUL WAITING: some cases remit spontaneously during childhood; "
            "SEVERE/DIAZOXIDE-UNRESPONSIVE: octreotide, then 18F-DOPA PET-CT; "
            "GCK GOF CHI is usually DIFFUSE — limited resection rarely curative; "
            "ADULT MANAGEMENT: persistent adult cases may need lifelong treatment; "
            "Pregnancy: may worsen — careful monitoring required"
        ),
    },
    {
        "gene": "HNF4A",
        "protein": (
            "HNF4A -- 20q13.12 AD -- 474aa -- Hepatocyte-Nuclear-Factor-4-Alpha-53kDa-"
            "Nuclear-Receptor-Beta-Cell-Transcription-Factor-CHI-then-MODY1-AD -- OMIM-Gene-600281-Disease-MODY1-125850"
        ),
        "locus": "20q13.12",
        "protein_size": "474 aa / 53 kDa",
        "inheritance": (
            "AD (HNF4A haploinsufficiency); DUAL PHENOTYPE — same mutation causes CHI in neonates/infants "
            "AND MODY1 (adult-onset type 2 DM-like) later in life; "
            "CHI phase: macrosomia + diazoxide-responsive neonatal/infantile hyperinsulinaemia; "
            "MODY1 phase: progressive beta-cell failure → DM onset usually 20-50 years; "
            "HNF4A encodes a nuclear receptor transcription factor — master regulator of hepatocyte and "
            "beta-cell gene expression; "
            "LOF in beta-cells: paradoxically causes excess insulin secretion in infancy "
            "(mechanism: disrupted gene networks → KATP channel component downregulation)"
        ),
        "disease_category": (
            "HNF4A DUAL PHENOTYPE: one of the most unusual presentations in genetic diabetes — "
            "the SAME haploinsufficiency causes OPPOSITE phenotypes at different life stages: "
            "NEONATAL/INFANTILE: macrosomia (birth weight +600g above normal) + "
            "hyperinsulinaemic hypoglycaemia (diazoxide-responsive); "
            "ADULT: MODY1 — progressive insulin secretory defect → diabetes onset typically 2nd-4th decade; "
            "HEPATIC: HNF4A expressed in liver → elevated transaminases, hepatomegaly, "
            "apolipoprotein abnormalities (low LDL, apo A-II, apo C-III); "
            "RENAL FANCONI: HNF4A mutations can cause renal Fanconi syndrome (aminoaciduria, glucosuria, phosphaturia); "
            "KEY CLINICAL PRINCIPLE: neonate with macrosomia + diazoxide-responsive CHI → screen HNF4A "
            "(and HNF1A for MODY3) BEFORE diagnosing 'diazoxide-responsive CHI of unknown cause'"
        ),
        "disease_pathway": (
            "HNF4A is a transcription factor (nuclear receptor superfamily) that controls expression of: "
            "KATP channel subunits (ABCC8 and KCNJ11), "
            "glycolytic enzymes, glucokinase, and MODY genes (HNF1A, HNF1B). "
            "HNF4A haploinsufficiency → reduced KATP expression → "
            "KATP density lower on beta-cell membrane → "
            "threshold for membrane depolarisation lower → "
            "insulin secretion at lower glucose (CHI phenotype in infancy); "
            "As beta-cells mature and HNF4A signalling becomes more critical for survival and secretory function → "
            "progressive loss of insulin secretory capacity → "
            "adult-onset diabetes (MODY1 phenotype); "
            "HEPATIC EFFECTS: HNF4A regulates many hepatic genes → "
            "lipid metabolism genes → low apoB, apoAII, triglycerides; "
            "RENAL FANCONI: HNF4A in proximal tubule → transporter gene expression disrupted → "
            "amino acid/glucose/phosphate leak"
        ),
        "pathognomonic": (
            "HNF4A CHI DIAGNOSTIC CLUSTER: "
            "1) MACROSOMIA: birth weight >95th centile (+500-600g on average) — "
            "PATHOGNOMONIC CLUE for HNF4A/HNF1A CHI (vs ABCC8/KCNJ11 where macrosomia is less consistent); "
            "2) DIAZOXIDE-RESPONSIVE CHI: neonatal/infantile; responds to standard doses; "
            "3) FAMILY HISTORY OF MODY: parent/grandparent with 'Type 2 diabetes' diagnosed young, "
            "low HbA1c variability, mild progressive course → may have undiagnosed MODY1; "
            "4) HEPATIC ABNORMALITIES: mildly elevated ALT/AST, fatty liver; "
            "low LDL/apolipoprotein panel → HNF4A target gene disruption; "
            "5) SPONTANEOUS RESOLUTION: CHI typically resolves in 1st year of life in most HNF4A cases; "
            "6) LONG-TERM: inform family that affected individuals WILL develop MODY1 (diabetes) — "
            "annual fasting glucose + HbA1c after age 10; "
            "early insulin secretagogue (sulphonylurea) very effective in MODY1"
        ),
        "treatment": (
            "HNF4A CHI→MODY1 TREATMENT: "
            "CHI PHASE: diazoxide 5-10 mg/kg/day — RESPONSIVE; most resolve within 1-3 years; "
            "LOW-DOSE diazoxide often sufficient; wean carefully; "
            "MODY1 PHASE (adult): sulphonylurea (glipizide, gliclazide) — very effective; "
            "stimulates residual beta-cells; "
            "AVOID insulin unless beta-cell failure complete; "
            "HEPATIC MONITORING: annual LFTs, lipid panel; "
            "FAMILY SCREENING: genetic testing of first-degree relatives; "
            "affected relatives who are asymptomatic should begin annual glucose screening; "
            "PREGNANCY: HNF4A MODY1 associated with macrosomic neonates in affected mothers"
        ),
    },
    {
        "gene": "INSR",
        "protein": (
            "INSR -- 19p13.2 AR/AD -- 1382aa -- Insulin-Receptor-155kDa-Transmembrane-"
            "RTK-Donohue-Rabson-Mendenhall-Type-A-Insulin-Resistance-AR-AD -- OMIM-Gene-147670-Disease-Donohue-246200"
        ),
        "locus": "19p13.2",
        "protein_size": "1382 aa / 155 kDa",
        "inheritance": (
            "AR biallelic LOF: Donohue syndrome (Leprechaunism) — severe; "
            "AR compound heterozygous: Rabson-Mendenhall syndrome — moderate; "
            "AD heterozygous: Type A insulin resistance — mild; "
            "INSR encodes the insulin receptor tyrosine kinase; "
            "Paradox: severe INSR LOF causes NEONATAL HYPOGLYCAEMIA (post-prandial) + FASTING HYPERGLYCAEMIA; "
            "mechanism: without insulin signalling, glucose cannot be stored → post-absorptive hyperglycaemia; "
            "PARADOXICAL HYPOGLYCAEMIA: massive compensatory hyperinsulinaemia overwhelms partial signalling → "
            "suppresses glucagon → net hypoglycaemia in fed state; "
            "Donohue: median survival <2 years; extreme growth retardation; elfin facies"
        ),
        "disease_category": (
            "INSR DISEASE SPECTRUM — severity inversely proportional to residual INSR function: "
            "DONOHUE/LEPRECHAUNISM (AR, severe LOF): "
            "Pre/postnatal growth retardation; elfin facies; large hands/feet; acanthosis nigricans; "
            "extreme insulin resistance (insulin levels 1000-100,000 mU/L); "
            "fasting hypoglycaemia (no insulin effect on glucagon suppression) + "
            "post-prandial hyperglycaemia (glucose cannot enter cells); "
            "RABSON-MENDENHALL (AR, moderate LOF): "
            "similar but milder; dental/nail abnormalities; pineal hyperplasia; survive to adolescence; "
            "TYPE A (AD, heterozygous): young females; acanthosis nigricans; ovarian hyperandrogenism (PCOS-like); "
            "no features of genetic syndrome; metabolic syndrome; "
            "COMMON THREAD: all show compensatory hyperinsulinaemia (reactive hypoglycaemia can occur); "
            "acanthosis nigricans universal marker of insulin resistance"
        ),
        "disease_pathway": (
            "INSR encodes the alpha2beta2 tetrameric insulin receptor — Type I receptor tyrosine kinase. "
            "NORMAL SIGNALLING: insulin → INSR → autophosphorylation (Tyr960, Tyr1158, Tyr1163) → "
            "IRS1/2 → PI3K → AKT → GLUT4 translocation (muscle/fat), glycogen synthesis (liver), "
            "glucagon suppression (alpha-cells). "
            "INSR LOF: no downstream signalling; "
            "COMPENSATORY HYPERINSULINAEMIA: pancreas senses peripheral insulin resistance → "
            "exponential increase in insulin secretion (100-1000x normal); "
            "PARADOXICAL POST-PRANDIAL HYPOGLYCAEMIA: massive insulin surge after meal → "
            "some residual partial signalling → glucose overshoot below normal; "
            "FASTING HYPERGLYCAEMIA: glucagon not suppressed → continuous hepatic glucose output; "
            "IGF1-RECEPTOR PARTIAL BYPASS: INSR LOF → paradoxically some insulin binds IGF1R → "
            "ovarian androgen production → hyperandrogenism in women"
        ),
        "pathognomonic": (
            "INSR CHI/INSULIN RESISTANCE DIAGNOSTIC CLUSTER: "
            "1) EXTREME INSULIN LEVELS: serum insulin >1000 mU/L (Donohue) — PATHOGNOMONIC; "
            "no other condition except INSR mutation produces these insulin levels; "
            "2) ELFIN FACIES + EXTREME GROWTH RETARDATION: "
            "DONOHUE PATHOGNOMONIC — low-set ears, hirsutism, large hands/feet, paucity of subcutaneous fat; "
            "3) PARADOXICAL POST-PRANDIAL HYPOGLYCAEMIA: fed hypoglycaemia + fasting hyperglycaemia — "
            "unique among CHI causes; glucose infusion may paradoxically worsen hypoglycaemia; "
            "4) ACANTHOSIS NIGRICANS: universal in all INSR severity spectrum — "
            "skin marker of insulin resistance (hyperpigmented velvety skin at neck/axillae/groin); "
            "5) PINEAL HYPERPLASIA (RABSON-MENDENHALL): pineal body enlargement on MRI — "
            "PATHOGNOMONIC for Rabson-Mendenhall; "
            "6) OVARIAN HYPERANDROGENISM: PCOS-like in women with Type A insulin resistance — "
            "check INSR in young females with PCOS + acanthosis; "
            "KEY MANAGEMENT PRINCIPLE: GLUCOSE INFUSION CAN CAUSE HYPOGLYCAEMIA — "
            "the hyperinsulinaemic state means adding glucose substrate amplifies insulin further"
        ),
        "treatment": (
            "INSR DISEASE TREATMENT: "
            "DONOHUE: no effective treatment; supportive; continuous feeds to prevent fasting hypoglycaemia; "
            "IGF1 THERAPY: recombinant IGF-1 (mecasermin) — bypasses INSR via IGF1-R; "
            "some glycaemic improvement; did not change survival substantially; "
            "RABSON-MENDENHALL: IGF-1; metformin (limited effect); insulin at very high doses (may partially work); "
            "TYPE A (heterozygous): metformin, SGLT2i for metabolic syndrome; "
            "anti-androgen (spironolactone/cyproterone) for hyperandrogenism; "
            "GLP-1 agonists for weight management; "
            "MONITORING: frequent glucose monitoring; avoid prolonged fasting in Donohue/RMS; "
            "nutrition support: high-carbohydrate continuous feeds (paradoxical — feeds to prevent fasting crisis)"
        ),
    },
    {
        "gene": "SLC16A1",
        "protein": (
            "SLC16A1 -- 1p13.2 AD -- 465aa -- Monocarboxylate-Transporter-1-MCT1-43kDa-"
            "Pyruvate-Lactate-Transporter-Exercise-Induced-Hyperinsulinism-EIHI-AD -- OMIM-Gene-600682-Disease-EIHI-609812"
        ),
        "locus": "1p13.2",
        "protein_size": "465 aa / 43 kDa",
        "inheritance": (
            "AD (dominant negative or haploinsufficiency — gain-of-expression in beta-cells); "
            "Exercise-induced hyperinsulinism (EIHI) — extremely rare, < 30 families worldwide; "
            "UNIQUE MECHANISM: normally SLC16A1 (MCT1) is NOT expressed in beta-cells (actively silenced); "
            "SLC16A1 promoter/regulatory region mutations → aberrant MCT1 expression IN BETA-CELLS; "
            "pyruvate enters beta-cell via MCT1 → beta-cell 'sees' anaerobic exercise metabolite → "
            "insulin secreted → hypoglycaemia triggered ONLY during/after anaerobic exercise; "
            "DIAGNOSTIC: standard CHI investigations NORMAL at rest — only triggered by exercise"
        ),
        "disease_category": (
            "EIHI (SLC16A1) is the ONLY form of CHI triggered SPECIFICALLY by anaerobic exercise — "
            "an almost PATHOGNOMONIC phenotype. "
            "SLC16A1 encodes MCT1, a monocarboxylate (lactate/pyruvate/ketone) transporter — "
            "normally expressed everywhere EXCEPT pancreatic beta-cells; "
            "Normal beta-cell: does NOT express MCT1 — hence does NOT respond to pyruvate/lactate (exercise metabolites); "
            "SLC16A1 GOF/promoter mutations: MCT1 ectopically expressed in beta-cells; "
            "during anaerobic exercise: blood pyruvate/lactate rises → enters beta-cell via MCT1 → "
            "fuels TCA → ATP → KATP closes → insulin spikes → hypoglycaemia; "
            "FASTING: normal (no pyruvate influx at rest); "
            "PROTEIN LOAD: normal (no leucine/glutamine sensitisation); "
            "DIAZOXIDE: only partially effective (KATP intact but triggered only by exercise)"
        ),
        "disease_pathway": (
            "MCT1 (SLC16A1) ECTOPIC EXPRESSION PATHWAY: "
            "Normal: beta-cell membrane lacks MCT1 → pyruvate cannot enter from circulation → "
            "beta-cell only responds to intracellular glucose-derived pyruvate; "
            "SLC16A1 GOF: MCT1 expressed on beta-cell membrane → "
            "Exercise state: anaerobic glycolysis in muscle → lactate + pyruvate released into blood → "
            "pyruvate enters beta-cell via MCT1 → pyruvate dehydrogenase → acetyl-CoA → TCA → NADH → "
            "ATP/ADP ratio rises → KATP closes → Ca2+ influx → insulin exocytosis; "
            "INSULIN SPIKE at exact time of maximal exercise when hepatic glucose production is already high → "
            "delayed hypoglycaemia 10-30 min after anaerobic exercise; "
            "GLUCOSE STIMULUS UNCHANGED: standard fasting or glucose infusion → normal insulin response; "
            "DIAZOXIDE: opens KATP → partially blunts response but exercise-induced pyruvate still enters; "
            "MCT1 is also expressed in cardiac muscle and other tissues → no other ectopic expression issues"
        ),
        "pathognomonic": (
            "SLC16A1 EIHI DIAGNOSTIC CLUSTER: "
            "1) EXERCISE-INDUCED HYPOGLYCAEMIA ONLY: "
            "PATHOGNOMONIC — hypoglycaemia triggered exclusively by ANAEROBIC exercise; "
            "no fasting hypoglycaemia, no protein-triggered hypoglycaemia; "
            "2) NORMAL STANDARD CHI INVESTIGATIONS AT REST: fasting insulin, glucose, acylcarnitines all NORMAL; "
            "standard provocative tests (fasting, glucagon) negative → "
            "DIAGNOSIS MISSED unless exercise provocation test performed; "
            "3) EXERCISE TOLERANCE TEST: 10-15 min anaerobic exercise (cycling/running) → "
            "glucose drops to <2.8 mmol/L with detectable insulin 10-30 min post-exercise; "
            "4) ANAEROBIC specificity: aerobic exercise (walking) does NOT trigger; "
            "high-intensity short-burst exercise DOES trigger; "
            "5) DIAZOXIDE: partial effect at rest; "
            "exercise still triggers via pyruvate bypass of KATP modulation; "
            "6) MANAGEMENT: avoid intense anaerobic exercise; "
            "consume carbohydrate before high-intensity sports; "
            "KEY CLINICAL TRAP: athletic child with unexplained post-exercise hypoglycaemia — "
            "ALWAYS consider EIHI and perform exercise provocation before diagnosing 'factitious' or 'psychological'"
        ),
        "treatment": (
            "SLC16A1 EIHI TREATMENT: "
            "PRIMARY: AVOID ANAEROBIC EXERCISE — restrict intense high-intensity exercise; "
            "allow low-to-moderate aerobic activity (walking, light cycling); "
            "PRE-EXERCISE CARBOHYDRATE: consume 15-30g fast-acting carbohydrate before sports; "
            "DIAZOXIDE: partially effective; prescribe if exercise restriction is impractical; "
            "5-15 mg/kg/day; "
            "GLUCOSE GEL: carry emergency glucose at all times; "
            "GLUCAGON: emergency kit for severe exercise-induced hypoglycaemia; "
            "SCHOOL/SPORTS: medical letter for PE exemption or supervised activity; "
            "PROGNOSIS: most patients manage well with exercise restriction; "
            "CHI phenotype does NOT worsen with age — EIHI is not progressive"
        ),
    },
]

# ── per-gene clinical feature probabilities ──────────────────────────────────
_MACROSOMIA = {"ABCC8": 0.50, "KCNJ11": 0.40, "HADH": 0.30, "GLUD1": 0.35,
               "GCK": 0.45, "HNF4A": 0.80, "INSR": 0.65, "SLC16A1": 0.20}
_DIAZOXIDE_RESP = {"ABCC8": 0.30, "KCNJ11": 0.30, "HADH": 0.90, "GLUD1": 0.90,
                   "GCK": 0.85, "HNF4A": 0.90, "INSR": 0.15, "SLC16A1": 0.50}
_HYPERAMMON = {"ABCC8": 0.03, "KCNJ11": 0.05, "HADH": 0.05, "GLUD1": 0.95,
               "GCK": 0.05, "HNF4A": 0.05, "INSR": 0.08, "SLC16A1": 0.03}
_EXERCISE_TRIG = {"ABCC8": 0.05, "KCNJ11": 0.05, "HADH": 0.10, "GLUD1": 0.10,
                  "GCK": 0.08, "HNF4A": 0.05, "INSR": 0.08, "SLC16A1": 0.95}
_PROTEIN_SENS = {"ABCC8": 0.20, "KCNJ11": 0.20, "HADH": 0.90, "GLUD1": 0.90,
                 "GCK": 0.30, "HNF4A": 0.30, "INSR": 0.25, "SLC16A1": 0.15}
_ACANTHOSIS = {"ABCC8": 0.15, "KCNJ11": 0.15, "HADH": 0.10, "GLUD1": 0.12,
               "GCK": 0.10, "HNF4A": 0.15, "INSR": 0.98, "SLC16A1": 0.05}
_EPILEPSY = {"ABCC8": 0.25, "KCNJ11": 0.55, "HADH": 0.15, "GLUD1": 0.50,
             "GCK": 0.15, "HNF4A": 0.12, "INSR": 0.30, "SLC16A1": 0.10}
_SURGERY = {"ABCC8": 0.60, "KCNJ11": 0.45, "HADH": 0.05, "GLUD1": 0.08,
            "GCK": 0.12, "HNF4A": 0.05, "INSR": 0.20, "SLC16A1": 0.03}


def _generate_patients(gene_idx: int, n: int = 40, seed: int = 0):
    rng = random.Random(seed)
    gene = ATLAS_GENES[gene_idx]
    g = gene["gene"]
    patients = []
    onset_params = {
        "ABCC8": (0.1, 0.2), "KCNJ11": (0.08, 0.15), "HADH": (0.3, 0.4),
        "GLUD1": (0.5, 0.6), "GCK": (0.2, 0.3), "HNF4A": (0.15, 0.25),
        "INSR": (0.05, 0.1), "SLC16A1": (5.0, 3.0),
    }
    mu, sigma = onset_params.get(g, (1.0, 0.5))
    for i in range(n):
        onset_yrs = max(0.01, rng.gauss(mu, sigma))
        fasting_glucose = rng.uniform(1.5, 2.8) if g != "INSR" else rng.uniform(1.8, 3.5)
        insulin_peak = {
            "ABCC8": rng.uniform(15, 80), "KCNJ11": rng.uniform(12, 70),
            "HADH": rng.uniform(10, 40), "GLUD1": rng.uniform(12, 50),
            "GCK": rng.uniform(8, 35), "HNF4A": rng.uniform(10, 45),
            "INSR": rng.uniform(500, 5000), "SLC16A1": rng.uniform(15, 60),
        }.get(g, rng.uniform(10, 50))
        patients.append({
            "patient_id": f"{g}-{seed}-{i+1:03d}",
            "gene": g,
            "onset_years": round(onset_yrs, 2),
            "fasting_glucose_mmol": round(fasting_glucose, 2),
            "insulin_peak_mU_L": round(insulin_peak, 1),
            "macrosomia": rng.random() < _MACROSOMIA.get(g, 0.35),
            "diazoxide_responsive": rng.random() < _DIAZOXIDE_RESP.get(g, 0.60),
            "hyperammonaemia": rng.random() < _HYPERAMMON.get(g, 0.05),
            "exercise_triggered": rng.random() < _EXERCISE_TRIG.get(g, 0.10),
            "protein_sensitive": rng.random() < _PROTEIN_SENS.get(g, 0.25),
            "acanthosis_nigricans": rng.random() < _ACANTHOSIS.get(g, 0.12),
            "epilepsy": rng.random() < _EPILEPSY.get(g, 0.20),
            "required_surgery": rng.random() < _SURGERY.get(g, 0.15),
        })
    return patients


def generate_overview():
    all_patients = []
    for idx in range(len(ATLAS_GENES)):
        all_patients.extend(_generate_patients(idx, n=40, seed=2654 + idx))

    summary = {}
    for p in all_patients:
        g = p["gene"]
        if g not in summary:
            summary[g] = {
                "gene": g,
                "n": 0,
                "macrosomia_pct": 0,
                "diazoxide_resp_pct": 0,
                "hyperammon_pct": 0,
                "exercise_trig_pct": 0,
                "protein_sens_pct": 0,
                "acanthosis_pct": 0,
                "epilepsy_pct": 0,
                "surgery_pct": 0,
                "mean_onset_yrs": 0.0,
                "mean_insulin_peak": 0.0,
            }
        s = summary[g]
        s["n"] += 1
        s["macrosomia_pct"] += int(p["macrosomia"])
        s["diazoxide_resp_pct"] += int(p["diazoxide_responsive"])
        s["hyperammon_pct"] += int(p["hyperammonaemia"])
        s["exercise_trig_pct"] += int(p["exercise_triggered"])
        s["protein_sens_pct"] += int(p["protein_sensitive"])
        s["acanthosis_pct"] += int(p["acanthosis_nigricans"])
        s["epilepsy_pct"] += int(p["epilepsy"])
        s["surgery_pct"] += int(p["required_surgery"])
        s["mean_onset_yrs"] += p["onset_years"]
        s["mean_insulin_peak"] += p["insulin_peak_mU_L"]

    gene_summaries = []
    for g, s in summary.items():
        n = s["n"]
        gene_summaries.append({
            "gene": g,
            "n": n,
            "macrosomia_pct": round(100 * s["macrosomia_pct"] / n, 1),
            "diazoxide_responsive_pct": round(100 * s["diazoxide_resp_pct"] / n, 1),
            "hyperammonaemia_pct": round(100 * s["hyperammon_pct"] / n, 1),
            "exercise_triggered_pct": round(100 * s["exercise_trig_pct"] / n, 1),
            "protein_sensitive_pct": round(100 * s["protein_sens_pct"] / n, 1),
            "acanthosis_nigricans_pct": round(100 * s["acanthosis_pct"] / n, 1),
            "epilepsy_pct": round(100 * s["epilepsy_pct"] / n, 1),
            "surgery_required_pct": round(100 * s["surgery_pct"] / n, 1),
            "mean_onset_years": round(s["mean_onset_yrs"] / n, 2),
            "mean_insulin_peak_mU_L": round(s["mean_insulin_peak"] / n, 1),
        })

    return {
        "atlas": "Hereditary Congenital Hyperinsulinism Atlas",
        "genes": [g["gene"] for g in ATLAS_GENES],
        "total_patients": len(all_patients),
        "seeds": list(range(2654, 2662)),
        "gene_summaries": gene_summaries,
        "chi_mechanisms": [
            "KATP-channel defect (ABCC8/KCNJ11) — diazoxide-UNRESPONSIVE; focal vs diffuse; 18F-DOPA PET-CT",
            "GDH-mediated (GLUD1 GOF) — diazoxide-responsive; protein-sensitive; HYPERAMMONAEMIA",
            "SCHAD-mediated (HADH LOF) — diazoxide-responsive; protein-sensitive; C4-OH acylcarnitine; NO hyperammonaemia",
            "GCK-GOF — diazoxide-responsive; set-point shifted; AD family history of CHI or MODY2",
            "HNF4A haploinsufficiency — diazoxide-responsive CHI in infancy → MODY1 in adulthood; macrosomia",
            "INSR LOF (Donohue/Rabson-Mendenhall) — extreme hyperinsulinaemia; acanthosis; paradoxical hypoglycaemia",
            "SLC16A1 GOF (EIHI) — EXERCISE-INDUCED ONLY; normal at rest; anaerobic exercise provocation test required",
        ],
        "diazoxide_responders": ["HADH", "GLUD1", "GCK", "HNF4A", "SLC16A1 (partial)"],
        "diazoxide_non_responders": ["ABCC8 (severe AR)", "KCNJ11 (severe AR)", "INSR"],
    }


def generate_breakdown():
    all_entries = []
    for idx, gene_data in enumerate(ATLAS_GENES):
        patients = _generate_patients(idx, n=40, seed=2654 + idx)
        g = gene_data["gene"]
        n = len(patients)
        macrosomia_pct = round(100 * sum(p["macrosomia"] for p in patients) / n, 1)
        diazoxide_pct = round(100 * sum(p["diazoxide_responsive"] for p in patients) / n, 1)
        hyperammon_pct = round(100 * sum(p["hyperammonaemia"] for p in patients) / n, 1)
        exercise_pct = round(100 * sum(p["exercise_triggered"] for p in patients) / n, 1)
        protein_pct = round(100 * sum(p["protein_sensitive"] for p in patients) / n, 1)
        acanthosis_pct = round(100 * sum(p["acanthosis_nigricans"] for p in patients) / n, 1)
        epilepsy_pct = round(100 * sum(p["epilepsy"] for p in patients) / n, 1)
        surgery_pct = round(100 * sum(p["required_surgery"] for p in patients) / n, 1)
        mean_onset = round(sum(p["onset_years"] for p in patients) / n, 2)
        mean_insulin = round(sum(p["insulin_peak_mU_L"] for p in patients) / n, 1)

        all_entries.append({
            "gene": g,
            "locus": gene_data["locus"],
            "protein_size": gene_data["protein_size"],
            "inheritance_summary": gene_data["inheritance"],
            "disease_category": gene_data["disease_category"],
            "disease_pathway": gene_data["disease_pathway"],
            "pathognomonic": gene_data["pathognomonic"],
            "treatment": gene_data["treatment"],
            "patients_n": n,
            "mean_onset_years": mean_onset,
            "mean_insulin_peak_mU_L": mean_insulin,
            "macrosomia_pct": macrosomia_pct,
            "diazoxide_responsive_pct": diazoxide_pct,
            "hyperammonaemia_pct": hyperammon_pct,
            "exercise_triggered_pct": exercise_pct,
            "protein_sensitive_pct": protein_pct,
            "acanthosis_nigricans_pct": acanthosis_pct,
            "epilepsy_pct": epilepsy_pct,
            "surgery_required_pct": surgery_pct,
        })
    return {"atlas": "Hereditary Congenital Hyperinsulinism Atlas", "gene_entries": all_entries}


def generate_definitions():
    gene_entries = {g["gene"]: g["pathognomonic"] for g in ATLAS_GENES}

    glossary = {
        "Congenital Hyperinsulinism (CHI)": (
            "Persistent hyperinsulinaemic hypoglycaemia of infancy (PHHI) — "
            "a group of genetic disorders causing inappropriately high insulin secretion. "
            "DEFINITION: plasma glucose <2.8 mmol/L (50 mg/dL) + detectable insulin (>2 mU/L) + "
            "absent/low ketones + absent/low fatty acids = biochemically confirmed hyperinsulinaemic hypoglycaemia. "
            "INCIDENCE: ~1:25,000-50,000 births (higher in consanguineous populations — 1:2,500 in some communities). "
            "CLASSIFICATION by mechanism: "
            "1) KATP-channel defects (ABCC8/KCNJ11) — most common; diazoxide unresponsive (AR); "
            "2) GDH excess (GLUD1 GOF) — HI/HA syndrome; protein-sensitive; diazoxide responsive; "
            "3) SCHAD deficiency (HADH) — protein-sensitive; diazoxide responsive; C4-OH acylcarnitine; "
            "4) GCK GOF — set-point CHI; diazoxide responsive; AD; "
            "5) HNF4A/HNF1A — macrosomia + CHI → MODY; "
            "6) INSR defect — extreme hyperinsulinaemia; acanthosis; paradoxical hypoglycaemia; "
            "7) SLC16A1 GOF (EIHI) — exercise-triggered ONLY. "
            "NEUROTOXICITY: hypoglycaemic brain injury threshold = glucose <2.8 mmol/L for >30 min → "
            "permanent cognitive impairment if unrecognised."
        ),
        "KATP Channel Structure and Function": (
            "The pancreatic KATP channel is an octamer: (Kir6.2)4 + (SUR1)4. "
            "Kir6.2 (KCNJ11): inward-rectifier pore subunit; ATP binds directly → channel closes; "
            "SUR1 (ABCC8): ABC transporter regulatory subunit; "
            "NBD1 + NBD2: nucleotide-binding domains that sense ADP/ATP ratio; "
            "ADP binding to NBD2 → channel opens (counter-regulatory); "
            "GTP/GDP sensitivity mediated through NBD1. "
            "PHARMACOLOGY: "
            "DIAZOXIDE: binds SUR1 NBD2 → stabilises open state → KATP opens → hyperpolarisation → "
            "reduced Ca2+ influx → reduced insulin; "
            "SULPHONYLUREAS: bind SUR1 TMD2 → stabilise closed state → KATP closes → depolarisation → "
            "increased insulin; "
            "FAILURE OF DIAZOXIDE in ABCC8/KCNJ11 CHI: mutations prevent SUR1 from responding to diazoxide; "
            "TISSUE EXPRESSION: Kir6.2/SUR1 in pancreatic beta-cells; Kir6.2/SUR2A in cardiac/skeletal muscle; "
            "Kir6.2 in neurons."
        ),
        "Focal vs Diffuse CHI": (
            "CRITICAL DISTINCTION with different surgical implications: "
            "DIFFUSE CHI: all beta-cells throughout pancreas are abnormal; "
            "requires near-total (95-98%) pancreatectomy → high risk of diabetes and exocrine insufficiency; "
            "FOCAL CHI: small cluster of abnormal beta-cells within otherwise normal pancreas; "
            "mechanism: paternal germline ABCC8 or KCNJ11 mutation + somatic maternal allele LOH at 11p15 → "
            "focal clone shows homozygous ABCC8/KCNJ11 mutation; "
            "18F-DOPA PET-CT: GOLD STANDARD — focal lesion shows increased tracer uptake (F-DOPA = L-DOPA analogue → "
            "dopamine synthesis in beta-cells); "
            "SURGICAL OUTCOME: focal resection → cure in >90%; pancreatic head/body/tail resection only; "
            "GENETICS: paternal origin of ABCC8/KCNJ11 mutation (maternal allele at 11p15 imprinted — "
            "11p15 contains IGF2/H19 imprinting region; LOH exposes paternal mutation in focal clone). "
            "18F-DOPA PET-CT must be performed at a centre with CHI experience — "
            "standardised imaging protocol essential."
        ),
        "Hyperinsulinaemia/Hyperammonaemia (HI/HA) Syndrome": (
            "CAUSED BY: GLUD1 (glutamate dehydrogenase) gain-of-function mutations. "
            "PATHOGNOMONIC: hyperinsulinaemic hypoglycaemia + fasting ammonia 50-200 μmol/L. "
            "MECHANISM: GDH overactivity → "
            "(1) BETA-CELL: excess NADH → KATP closes → insulin; "
            "(2) LIVER: excess NH3 from glutamate deamidation → hyperammonaemia. "
            "AMMONIA LEVEL: 50-200 μmol/L (NOT severe — no encephalopathy in most); "
            "rises post-protein-meal; fasting level usually 50-100; "
            "NO OTHER HI CAUSE produces hyperammonaemia — this is the KEY DDx feature. "
            "EPILEPSY: ~50% develop epilepsy (CNS GDH expressed in neurons); "
            "VALPROATE ABSOLUTELY CONTRAINDICATED: valproate inhibits CPS1 (urea cycle step 1) → "
            "worsens hyperammonaemia → risk of severe hyperammonaemic encephalopathy. "
            "LEUCINE LOAD: most HI/HA patients show exaggerated insulin response to leucine — "
            "positive leucine tolerance test supports diagnosis."
        ),
        "Diazoxide Mechanism and Dosing": (
            "DIAZOXIDE (Proglycem, Hyperstat): benzothiadiazine derivative; KATP channel opener. "
            "MECHANISM: binds SUR1 in NBD2 → stabilises KATP in open configuration → "
            "K+ efflux → beta-cell hyperpolarisation → VGCC closed → Ca2+ influx ↓ → insulin secretion ↓. "
            "DOSING: 5-15 mg/kg/day divided 3 doses; "
            "PAEDIATRIC: start 5 mg/kg/day; titrate to lowest effective dose over 5-7 days; "
            "ALWAYS co-prescribe CHLOROTHIAZIDE (7-10 mg/kg/day): "
            "prevents fluid retention (diazoxide inhibits Na/K/Cl transport in kidney); "
            "also synergistic KATP opener effect; "
            "SIDE EFFECTS: hirsutism (cosmetic, reversible); hypertrichosis; fluid retention (mitigated by chlorothiazide); "
            "rarely pulmonary hypertension in neonates (CAUTION in first weeks); "
            "FAILURE DEFINITION: blood glucose remains <3.5 mmol/L despite adequate dose × 5 days → "
            "KATP-channel defect CHI (ABCC8/KCNJ11 AR biallelic); "
            "proceed to 18F-DOPA PET-CT and surgical assessment."
        ),
        "18F-DOPA PET-CT in CHI": (
            "18F-fluorodihydroxyphenylalanine (18F-DOPA) PET-CT: most important imaging in CHI management. "
            "PRINCIPLE: 18F-DOPA is a radiolabelled L-DOPA analogue → taken up by beta-cells → "
            "converted to dopamine → stored in secretory vesicles → focal accumulation in active beta-cells. "
            "INTERPRETATION: "
            "FOCAL CHI: localised region of increased tracer uptake against background of pancreatic activity; "
            "DIFFUSE CHI: uniform increased tracer uptake throughout pancreas; "
            "SENSITIVITY: >90% for focal lesions >3 mm; "
            "PREREQUISITE: only perform in DIAZOXIDE-UNRESPONSIVE CHI after genetic testing shows ABCC8/KCNJ11; "
            "CENTRES: must be performed at paediatric CHI centres with experienced nuclear medicine + paediatric surgeons; "
            "centres in UK (GOS), France (Necker), Germany (Greifswald), USA (CHOP) — "
            "do not perform at inexperienced sites; "
            "SURGICAL GUIDANCE: PET-CT guides limited pancreatectomy → cure without post-op diabetes in focal cases."
        ),
        "Neonatal Diabetes Mellitus (NDM) vs CHI": (
            "NDM: diabetes onset <6 months of age — fundamentally different aetiology from T1DM. "
            "CAUSES: KCNJ11 GOF (30%), ABCC8 GOF (30%), 6q24 abnormality (20%), others (20%). "
            "CRITICAL RULE: ALL NDM onset <6 months must have KCNJ11 + ABCC8 sequencing. "
            "If KCNJ11 or ABCC8 GOF identified → TRANSFER TO SULPHONYLUREA: "
            "90% successfully transferred from insulin to oral glibenclamide; "
            "dramatically improves quality of life (no injections); "
            "DEND syndrome: KCNJ11 GOF + developmental delay + epilepsy → sulphonylurea improves all three; "
            "TRANSIENT NDM (TNDM): 6q24 UPD or ABCC8/KCNJ11 mild GOF → spontaneous remission <18 months; "
            "relapse risk in adulthood (30-50%) → lifelong glucose monitoring mandatory. "
            "CHI vs NDM: same genes (ABCC8/KCNJ11), OPPOSITE mutation direction (LOF → CHI; GOF → NDM)."
        ),
        "Exercise-Induced Hyperinsulinism (EIHI) Diagnosis": (
            "SLC16A1 EIHI is unique: NORMAL standard CHI workup at rest. "
            "CLUE: child/young adult with post-exercise hypoglycaemia; "
            "no fasting hypoglycaemia; no protein-sensitive hypoglycaemia; "
            "standard OGTT, fasting test, glucagon stimulation — ALL NORMAL. "
            "DIAGNOSTIC TEST: standardised ANAEROBIC EXERCISE PROVOCATION: "
            "10-15 min cycling at 50-60% VO2 max (moderate-high intensity); "
            "glucose monitored at 0, 10, 20, 30 min post-exercise; "
            "POSITIVE: glucose <2.8 mmol/L with detectable insulin (>2 mU/L) post-exercise; "
            "SAFETY: test must be done in hospital with IV access + glucose ready; "
            "DO NOT use aerobic exercise (walking) — only anaerobic triggers EIHI; "
            "GENETIC CONFIRMATION: SLC16A1 promoter or regulatory region mutations → "
            "aberrant beta-cell MCT1 expression; "
            "NOTE: SLC16A1 coding mutations are NOT the cause — promoter/regulatory mutations ectopically "
            "activate expression in beta-cells."
        ),
    }

    return {
        "atlas": "Hereditary Congenital Hyperinsulinism Atlas",
        "gene_entries": gene_entries,
        "chi_glossary": glossary,
    }
