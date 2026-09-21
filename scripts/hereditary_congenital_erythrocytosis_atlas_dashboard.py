#!/usr/bin/env python3
"""Hereditary-Congenital-Erythrocytosis-Atlas — Complete 8-Gene Hereditary Erythrocytosis Atlas
EPOR    (erythropoietin receptor; 508 aa; 19p13.2; AD; GOF;
         Primary Familial and Congenital Polycythemia (PFCP); truncation of C-terminal negative
         regulatory domain → constitutive JAK2-STAT5 signalling → autonomous erythropoiesis;
         serum EPO SUPPRESSED — key DDx from VHL/EGLN1/EPAS1 where EPO elevated; seed SEED_BASE+0) ·
VHL     (von Hippel-Lindau; 213 aa; 3p25.3; AR or AD;
         Chuvash polycythemia (biallelic p.Arg200Trp — Chuvash/Siberian founder) / Erythrocytosis 2;
         VHL–HIF axis: VHL targets HIF1α/HIF2α for proteasomal degradation; VHL LOF → HIF stable →
         EPO ↑ → erythrocytosis; THROMBOSIS prominent (Budd-Chiari, portal vein, pulmonary);
         heterozygous VHL → VHL tumour syndrome (ccRCC/haemangioblastoma/phaeochromocytoma); seed SEED_BASE+1) ·
EGLN1   (egl nine homologue 1; 426 aa; 1q42.2; AD;
         PHD2 deficiency / Erythrocytosis 3; PHD2 is the major HIF-prolyl hydroxylase;
         LOF → HIF1α/2α not hydroxylated → not VHL-targeted → HIF stable → EPO↑ → erythrocytosis;
         EPO elevated (contrast EPOR where EPO suppressed);
         paraganglioma association in some pedigrees; seed SEED_BASE+2) ·
EPAS1   (endothelial PAS domain protein 1 / HIF2α; 870 aa; 2p21; AD;
         HIF2α gain-of-function / Erythrocytosis 4; GOF → HIF2α escapes VHL/PHD degradation →
         elevated EPO, VEGF, and other HIF targets; paraganglioma/pulmonary hypertension comorbidities;
         belzutifan (HIF2α inhibitor) — FDA 2021 for VHL-related neoplasms, used off-label for EPAS1-GOF; seed SEED_BASE+3) ·
HBB     (haemoglobin beta; 147 aa; 11p15.4; AD;
         High-affinity haemoglobin variants (left-shifted ODC) / Erythrocytosis 6;
         Hb Chesapeake (α92Arg→Leu), Hb Hiroshima (β146His→Asp), Hb Malmö, Hb Rainier, etc.;
         haem-haem interaction disrupted → T-R transition favoured → oxygen not released → tissue hypoxia →
         EPO↑ → secondary erythrocytosis; NO treatment needed unless Hct >0.56; seed SEED_BASE+4) ·
HBA1    (haemoglobin alpha 1; 142 aa; 16p13.3; AD;
         High-affinity haemoglobin alpha variants / rare erythrocytosis;
         Hb Suresnes, Hb Evanston, Hb Torino (high-affinity); Hb Creteil (slightly left-shifted);
         less common than HBB high-affinity variants; 4-gene deletion analysis for both HBA1+HBA2;
         p50 O2 measurement confirms diagnosis; seed SEED_BASE+5) ·
BPGM    (bisphosphoglycerate mutase; 258 aa; 7q33; AR;
         2,3-BPG mutase deficiency / Erythrocytosis 8; 2,3-BPG is the key allosteric effector of Hb;
         BPGM deficiency → 2,3-BPG absent → Hb oxygen affinity VERY HIGH (left-shifted ODC) →
         tissue hypoxia → EPO↑ → compensatory erythrocytosis;
         same functional outcome as high-affinity Hb variants but enzymatic mechanism; seed SEED_BASE+6) ·
EPO     (erythropoietin; 193 aa; 7q22.3; AD;
         Hereditary erythrocytosis with elevated EPO / Erythrocytosis 5 (EPO gain-of-function);
         very rare; germline gain-of-function EPO → constitutively elevated EPO → erythrocytosis;
         EPO elevated (appropriate source — kidney/liver) unlike secondary where EPO is reactive;
         distinguish from polycythaemia vera (JAK2 somatic V617F) and reactive (not hereditary); seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 2886–2893)
"""
import random

# Atlas genes with full clinical data
ATLAS_GENES = [
    {
        "gene": "EPOR",
        "protein": (
            "EPOR -- 19p13.2 AD-GOF -- 508aa -- Erythropoietin-Receptor-"
            "56kDa-Type-I-Cytokine-Receptor-JAK2-STAT5-Signal-"
            "PFCP-Primary-Familial-Congenital-Polycythemia-Erythrocytosis1-"
            "OMIM-Gene-133171-Disease-PFCP-263400"
        ),
        "locus": "19p13.2",
        "protein_size": (
            "508 aa / 56 kDa (type I cytokine receptor; single transmembrane domain; "
            "homodimerises on EPO binding; C-terminal intracellular domain: negative regulatory region "
            "(aa 374-439) contains tyrosine residues required for JAK2 deactivation and STAT5 downregulation; "
            "PFCP GOF MECHANISM: truncating variants (nonsense, frameshift) in C-terminal region → "
            "  loss of negative regulatory domain → JAK2-STAT5 signalling constitutively active without EPO; "
            "  erythroid progenitors proliferate autonomously; "
            "SERUM EPO: SUPPRESSED (critical DDx from all other hereditary erythrocytoses); "
            "normal EPO feedback loop broken — receptor constitutively active even at low EPO; "
            "p50 O2 measurement: NORMAL (unlike high-affinity Hb variants); "
            "bone marrow: hypercellular erythroid hyperplasia; "
            "inheritance: AD; de novo variants possible; "
            "PFCP: ~10 described families; rare (prevalence <1:1,000,000)"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT (AD) — GAIN-OF-FUNCTION — EPOR: "
            "  MECHANISM: truncating variants remove C-terminal negative regulatory domain; "
            "    Affected domain: aa 374-508 (intracellular tail); "
            "    Result: EPO hypersensitivity + EPO-independent erythroid colony formation; "
            "    Serum EPO: LOW/SUPPRESSED (negative feedback intact but receptor always signalling); "
            "  HAPLOINSUFFICIENCY (1 normal copy): "
            "    Heterozygous truncation → dominant gain: 1 truncated receptor without brake + 1 normal; "
            "    Truncated EPOR homodimerises + heterodimerises → net excess signalling; "
            "  FULL PENETRANCE: nearly all carriers have erythrocytosis (high-penetrance AD); "
            "  FAMILY HISTORY: often multiple family members with high Hb/Hct; "
            "  BIRTH: erythrocytosis present from birth/early childhood; "
            "  KEY DDx FEATURES: "
            "    EPO: LOW (suppressed) — distinguishes from VHL/EGLN1/EPAS1 where EPO high; "
            "    JAK2 V617F: ABSENT — distinguishes from polycythaemia vera; "
            "    p50: NORMAL — distinguishes from high-affinity Hb variants"
        ),
        "disease_category": (
            "PRIMARY FAMILIAL AND CONGENITAL POLYCYTHEMIA (PFCP) — ERYTHROCYTOSIS TYPE 1 — OMIM 263400; "
            "PRIMARY AUTONOMOUS ERYTHROCYTOSIS — NO SECONDARY CAUSE: "
            "  HAEMOGLOBIN: usually 18-22 g/dL (markedly elevated from birth/childhood); "
            "  HAEMATOCRIT: 55-75% (very high); "
            "  RED CELL MASS: elevated; "
            "  EPO: LOW/SUPPRESSED (<5 IU/L; often undetectable); "
            "  SPLENOMEGALY: present in some (not universal); "
            "  THROMBOSIS RISK: elevated (hyperviscosity) but less than PV; "
            "  WHITE CELLS + PLATELETS: NORMAL (panhyperplasia NOT present — unlike PV); "
            "  BONE MARROW: erythroid hyperplasia; "
            "DIAGNOSIS: "
            "  1. Erythrocytosis (Hb >18.5 g/dL male, >16.5 g/dL female); "
            "  2. Low/undetectable serum EPO; "
            "  3. Absent JAK2 V617F (exclude PV); "
            "  4. Hereditary gene panel: EPOR sequencing; "
            "  5. In vitro erythroid colony formation in absence of EPO (research test); "
            "TREATMENT: "
            "  Venesection (phlebotomy) to target Hct <0.50-0.52; "
            "  Low-dose aspirin (thrombosis prevention); "
            "  Hydroxycarbamide (hydroxyurea): rarely needed; "
            "  JAK2 inhibitors (ruxolitinib): experimental for severe refractory cases"
        ),
        "disease_pathway": (
            "EPO-EPOR-JAK2-STAT5 SIGNALLING — NORMAL AND PFCP: "
            "NORMAL: "
            "  EPO (kidney/liver) → binds EPOR homodimer → JAK2 activation → STAT5 phosphorylation → "
            "    nuclear entry → target genes: BCL2L1 (survival), CCND3 (proliferation); "
            "  C-terminal EPOR Y residues (Y401, Y431, Y443, Y479): docking sites for SHP1/SHP2 phosphatases; "
            "    SHP1 dephosphorylates JAK2 → signal termination; "
            "PFCP (truncated EPOR): "
            "  C-terminal regulatory region deleted → SHP1 cannot dock → JAK2 phosphorylation sustained; "
            "  Erythroid progenitors survive + proliferate WITHOUT EPO stimulation; "
            "  Low EPO → intact feedback still partially suppresses EPO production → EPO LOW; "
            "POLYCYTHAEMIA VERA (JAK2 V617F — somatic, NOT hereditary): "
            "  Somatic JAK2 V617F in haematopoietic stem cells → JAK2 constitutively active; "
            "  DIFFERENT FROM PFCP: PV has ALL three cytopenias elevated (Hb+WBC+Plt); "
            "  JAK2 V617F found in neutrophils+platelets+erythrocytes (clonal); "
            "  EPOR truncation: erythrocytes only (germline); WBC+PLT normal; "
            "NATURAL HISTORY: "
            "  Lifelong erythrocytosis; generally stable; "
            "  Thrombotic events if Hct uncontrolled; "
            "  NO transformation to myelofibrosis or leukemia (unlike PV)"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC / HIGHLY CHARACTERISTIC — EPOR-PFCP: "
            "  1. ERYTHROCYTOSIS FROM BIRTH/EARLY CHILDHOOD: "
            "     Hb >18.5 g/dL male / >16.5 g/dL female since infancy; "
            "  2. SERUM EPO LOW/SUPPRESSED (<5 IU/L): "
            "     PATHOGNOMONIC for primary erythrocytosis; "
            "     Distinguishes from ALL secondary erythrocytoses (VHL/EGLN1/EPAS1/HBB/HBA1/BPGM/EPO); "
            "  3. ABSENT JAK2 V617F: "
            "     Excludes polycythaemia vera (most important clinical DDx); "
            "  4. NORMAL WBC + PLATELETS: "
            "     Excludes PV where panhyperplasia is typical; "
            "  5. FAMILY HISTORY (AD): "
            "     Multiple generations affected; "
            "  6. EPOR TRUNCATING VARIANT: "
            "     C-terminal truncation on hereditary panel"
        ),
        "treatment": (
            "TREATMENT — EPOR-PFCP: "
            "PRIMARY: "
            "  VENESECTION (PHLEBOTOMY): "
            "    Target Hct <0.50 (or <0.52) to reduce viscosity; "
            "    Frequency: typically every 4-12 weeks depending on Hct recovery; "
            "    Iron stores: will deplete with serial phlebotomy; monitor ferritin; "
            "    Iron supplementation BEFORE phlebotomy: AVOID (worsens erythrocytosis); "
            "  LOW-DOSE ASPIRIN 75-100 mg daily: "
            "    Thrombosis prevention; "
            "    Evidence from PV data extrapolated to other erythrocytoses; "
            "SECONDARY: "
            "  HYDROXYCARBAMIDE (hydroxyurea): "
            "    If venesection frequency >4/year or poorly tolerated; "
            "    Reduces erythroid production; "
            "  RUXOLITINIB (JAK1/2 inhibitor): "
            "    Off-label; for refractory disease; targets downstream JAK2 signalling; "
            "AVOID: "
            "  Iron supplementation (oral or IV) unless clear deficiency with hypochromic microcytic RBCs; "
            "  Erythropoiesis-stimulating agents (contraindicated); "
            "MONITORING: "
            "  FBC every 3-6 months; "
            "  Annual renal ultrasound + LFTs (VHL co-assessment if genetic diagnosis unclear); "
            "  Cardiovascular risk management"
        ),
        "seed": 2886,
    },
    {
        "gene": "VHL",
        "protein": (
            "VHL -- 3p25.3 AR-AD -- 213aa -- Von-Hippel-Lindau-Tumour-Suppressor-"
            "28kDa-E3-Ubiquitin-Ligase-Substrate-Recognition-Subunit-"
            "VHL-Syndrome-ccRCC-Haemangioblastoma-Phaeochromocytoma-"
            "Chuvash-Polycythemia-Biallelic-p.Arg200Trp-Erythrocytosis2-"
            "OMIM-Gene-608537-Disease-VHLS-193300-CE2-263400"
        ),
        "locus": "3p25.3",
        "protein_size": (
            "213 aa / 28 kDa (Elongin C/B/VHL E3 ubiquitin ligase complex; "
            "VHL is the substrate recognition subunit; "
            "VHL-box: recognises hydroxylated HIF-1α/2α Pro residues; "
            "beta domain: HIF binding; alpha domain: elongin C binding; "
            "TWO ENTIRELY DIFFERENT DISEASE MODES depending on whether mono- or bi-allelic: "
            "  HETEROZYGOUS (AD): VHL tumour suppressor syndrome — ccRCC, haemangioblastoma, phaeochromocytoma; "
            "  BIALLELIC p.Arg200Trp (AR): Chuvash polycythemia — erythrocytosis; "
            "    p.Arg200Trp specifically affects HIF binding without fully disrupting tumour suppressor function; "
            "    Chuvash/Krasnoyarsk/Siberia founder: ~1:1000 carrier frequency in Chuvash population; "
            "    Also reported independently in other populations (not exclusively Chuvash); "
            "    Belzutifan (Welireg): FDA 2021 for VHL-related neoplasms (not CE2 erythrocytosis)"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (biallelic p.Arg200Trp — Chuvash) or AD (VHL tumour syndrome): "
            "  CHUVASH POLYCYTHEMIA (AR — CE2): "
            "    Biallelic p.Arg200Trp (c.598C>T): VHL structure maintained but HIF1α binding pocket subtly altered; "
            "    HIF partially escapes ubiquitination → EPO/VEGF chronically elevated; "
            "    Erythrocytosis + elevated EPO; "
            "    NO VHL tumour syndrome (renal cell carcinoma/haemangioblastoma — RCC/HB do NOT occur); "
            "    Chuvash founder: p.Arg200Trp first described in Chuvashia (Russia); "
            "    Non-Chuvash biallelic: same variant found globally; also other biallelic combos; "
            "  VHL TUMOUR SYNDROME (AD — heterozygous): "
            "    NOT erythrocytosis (unless superimposed biallelic or other modifier); "
            "    ccRCC (clear cell renal cell carcinoma); CNS/retinal haemangioblastoma; phaeochromocytoma; "
            "    Managed with surveillance protocol + surgical resection + belzutifan; "
            "  CARRIER STATUS: "
            "    p.Arg200Trp heterozygotes: mildly elevated Hb (subclinical in some); "
            "    Full CE2: biallelic p.Arg200Trp or compound heterozygous"
        ),
        "disease_category": (
            "CONGENITAL ERYTHROCYTOSIS TYPE 2 (CE2) — CHUVASH POLYCYTHEMIA — OMIM 263400; "
            "SECONDARY ERYTHROCYTOSIS WITH ELEVATED EPO: "
            "  PRIMARY DEFECT: VHL LOF → HIF1α/2α not ubiquitinated → stable HIF → "
            "    EPAS1 (HIF2α) → EPO transcription in kidney → elevated serum EPO → erythrocytosis; "
            "  CLINICAL: "
            "    Haemoglobin: 18-20 g/dL typically; "
            "    Serum EPO: ELEVATED (inappropriate — unlike normal where EPO rises proportionally); "
            "    Thrombosis: PROMINENT risk — portal vein thrombosis, Budd-Chiari, pulmonary thromboembolism; "
            "      Thrombosis is LEADING CAUSE OF DEATH in CE2 (Chuvash polycythemia); "
            "      VHL erythrocytosis more thrombogenic than EPOR/HBB erythrocytosis — VEGF and other HIF targets; "
            "    Vertebral haemangiomas, varicose veins (VEGF-driven); "
            "    Pulmonary hypertension: ~30% (VEGF + EPO effects on pulmonary vasculature); "
            "  KEY DDx FROM PFCP (EPOR): "
            "    CE2: EPO elevated; EPOR: EPO suppressed; "
            "    CE2: VHL biallelic; EPOR: EPOR GOF truncation; "
            "    CE2: thrombosis more prominent; EPOR: thrombosis less prominent"
        ),
        "disease_pathway": (
            "VHL-HIF-EPAS1-EPO AXIS: "
            "NORMAL: "
            "  Normoxia → EGLN1/2/3 (PHD1/2/3) hydroxylate HIF-1α/2α Pro402/Pro564 → "
            "    VHL recognises hydroxyl-Pro → E3 ubiquitin ligase → HIF-1α/2α proteasomal degradation; "
            "  Hypoxia → PHD inactive → HIF1α/2α not hydroxylated → VHL cannot bind → "
            "    HIF stabilises → nucleus → EPO↑, VEGF↑, GLUT1↑ etc.; "
            "CE2 (biallelic p.Arg200Trp VHL): "
            "  VHL beta domain (HIF binding) subtly altered → PHD-hydroxylated HIF-1α/2α poorly recognised; "
            "  Partial HIF stabilisation at normoxia → chronic low-level HIF target upregulation; "
            "  EPAS1 (HIF2α) → EPO production elevated → Erythrocytosis + VEGF → vascular effects; "
            "  THROMBOSIS: "
            "    HIF target PAI-1 (plasminogen activator inhibitor-1) elevated → reduced fibrinolysis; "
            "    VEGF → endothelial dysfunction; hyperviscosity → stasis; "
            "    Platelet activation by hypoxia signalling; "
            "    Combined → high thrombosis risk (Budd-Chiari, portal vein, PE); "
            "BELZUTIFAN (VHL syndrome only): "
            "  HIF2α inhibitor; FDA 2021; targets EPAS1 (HIF2α)-ARNT dimerisation; "
            "  Used for VHL tumour syndrome (not CE2 erythrocytosis)"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC / HIGHLY CHARACTERISTIC — VHL-CE2: "
            "  1. ERYTHROCYTOSIS + ELEVATED SERUM EPO: "
            "     Hb >18 g/dL + EPO inappropriately elevated (>20 IU/L); "
            "     EPO raised = secondary; combined with erythrocytosis from birth = hereditary secondary; "
            "  2. CHUVASH ANCESTRY + BIALLELIC p.Arg200Trp: "
            "     Testing biallelic VHL p.Arg200Trp in Chuvash/Siberian-ancestry patient with erythrocytosis; "
            "  3. PORTAL VEIN / MESENTERIC THROMBOSIS IN YOUNG PERSON: "
            "     VHL CE2 erythrocytosis causes unusual visceral thrombosis; "
            "     Budd-Chiari syndrome in erythrocytosis + young age → test VHL; "
            "  4. PULMONARY HYPERTENSION WITH ERYTHROCYTOSIS: "
            "     HIF target upregulation → pulmonary vascular remodelling; "
            "  5. VERTEBRAL HAEMANGIOMAS (INCIDENTAL): "
            "     HIF/VEGF driven vascular lesions; not CNS haemangioblastomas (those = VHL AD syndrome)"
        ),
        "treatment": (
            "TREATMENT — VHL-CE2 (CHUVASH POLYCYTHEMIA): "
            "PRIMARY CONCERN: THROMBOSIS PREVENTION: "
            "  VENESECTION: target Hct <0.50-0.52; reduces hyperviscosity; "
            "  ANTICOAGULATION: "
            "    Lifelong anticoagulation if prior thrombosis (PE, portal vein, Budd-Chiari); "
            "    Warfarin or DOAC (avoid drugs affected by VHL-metabolised drugs); "
            "  LOW-DOSE ASPIRIN: 75-100 mg daily; "
            "ERYTHROCYTOSIS MANAGEMENT: "
            "  Venesection: cornerstone; "
            "  Hydroxycarbamide: for severe erythrocytosis/poor tolerance to phlebotomy; "
            "PULMONARY HYPERTENSION: "
            "  Refer to specialist centre; "
            "  PDE-5 inhibitors (sildenafil) / ERA (bosentan) if severe; "
            "PREGNANCY: "
            "  Very high-risk in CE2; maternal thrombosis risk; fetal anaemia (if compound het); "
            "  Low-molecular-weight heparin; haematology + obstetric high-risk co-management; "
            "VHL TUMOUR SURVEILLANCE (for AD VHL syndrome carriers — NOT same as CE2): "
            "  Annual MRI brain/spine; renal USS/MRI; phaeochromocytoma testing; retinal angiography; "
            "  Belzutifan: FDA 2021 for VHL-related neoplasms (not CE2 erythrocytosis)"
        ),
        "seed": 2887,
    },
    {
        "gene": "EGLN1",
        "protein": (
            "EGLN1 -- 1q42.2 AD -- 426aa -- Egl-Nine-Homologue-1-PHD2-"
            "Prolyl-Hydroxylase-Domain-Protein-2-46kDa-2OG-Oxygenase-"
            "HIF-Hydroxylase-Erythrocytosis3-PHD2-Deficiency-"
            "OMIM-Gene-606425-Disease-CE3-609820"
        ),
        "locus": "1q42.2",
        "protein_size": (
            "426 aa / 46 kDa (prolyl hydroxylase domain protein 2; 2-oxoglutarate (2OG) iron-dependent oxygenase; "
            "hydroxylates HIF-1α/2α at Pro402 and Pro564 → enables VHL-E3 ubiquitin ligase binding → "
            "HIF ubiquitination + proteasomal degradation; "
            "three PHD isoforms: PHD1 (EGLN2), PHD2 (EGLN1), PHD3 (EGLN3); "
            "PHD2 (EGLN1) is the dominant HIF prolyl hydroxylase under normoxia; "
            "cofactors: 2-oxoglutarate (2OG), O2, Fe2+, ascorbate; "
            "PHD2 is O2-sensitive — hypoxia directly inactivates PHD2 (the O2 sensor); "
            "CE3: LOF → PHD2 insufficient → HIF1α/2α not adequately hydroxylated → VHL cannot bind → "
            "  HIF stable at normoxia → EPO elevated → erythrocytosis; "
            "EGLN1 mutation vs EGLN2/3: EGLN1 mutations cause erythrocytosis; EGLN2/3 very rare"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT (AD) — EGLN1 — HAPLOINSUFFICIENCY: "
            "  50% PHD2 activity insufficient → mild HIF stabilisation → EPO elevated; "
            "  PENETRANCE: high but variable (modifier genes influence final Hb); "
            "  SERUM EPO: ELEVATED (secondary mechanism — EPO-driven); "
            "  ONSET: typically childhood/young adult discovery; "
            "  PARAGANGLIOMA ASSOCIATION: "
            "    EGLN1 LOF + erythrocytosis + paraganglioma (phaeo/PGL) = triad; "
            "    ~5-10% of CE3 patients have paraganglioma; "
            "    Screen for catecholamine excess in all CE3 patients; "
            "  GENOTYPE: most pathogenic variants are missense affecting catalytic function or cofactor binding; "
            "    Iron-binding residues (His374, Asp316), 2OG-binding, substrate-binding loop; "
            "  CE3 vs VHL (CE2): "
            "    Both have elevated EPO; both secondary; "
            "    CE3: EGLN1 variant; CE2: biallelic VHL; "
            "    Paraganglioma: CE3 > CE2; "
            "    Thrombosis: CE2 > CE3"
        ),
        "disease_category": (
            "CONGENITAL ERYTHROCYTOSIS TYPE 3 (CE3) — PHD2 DEFICIENCY — OMIM 609820; "
            "SECONDARY ERYTHROCYTOSIS WITH ELEVATED EPO AND PARAGANGLIOMA RISK: "
            "  Haemoglobin: typically 18-22 g/dL; "
            "  Serum EPO: ELEVATED; "
            "  Paraganglioma screening MANDATORY: "
            "    Biochemical: 24h urine metanephrines/catecholamines OR plasma free metanephrines; "
            "    Imaging: 68Ga-DOTA-SSTR PET or MRI abdomen/thorax; "
            "    Annual surveillance; "
            "  OXYGEN SENSING DEFECT: "
            "    PHD2 is the primary O2 sensor; PHD2 LOF → cells behave as if chronically hypoxic; "
            "    HIF1α + HIF2α both stabilised → pan-HIF target upregulation; "
            "  DIAGNOSIS: "
            "    Erythrocytosis + elevated EPO + EGLN1 variant; "
            "    Exclude secondary causes (hypoxia, smoking, EPO-secreting tumour, high-affinity Hb); "
            "  THROMBOSIS: "
            "    Less prominent than CE2 but still elevated risk; "
            "    Aspirin + venesection"
        ),
        "disease_pathway": (
            "EGLN1-PHD2 IN HYPOXIA SENSING: "
            "NORMAL PHD2 FUNCTION: "
            "  PHD2 uses O2 + 2OG + Fe2+ → hydroxylates HIF-1α Pro402/564 → VHL binds → ubiquitination; "
            "  At normoxia: PHD2 active → HIF rapidly degraded (t½ <5 min); "
            "  At hypoxia: PHD2 inactive (O2 limiting) → HIF stable → HIF target genes activated; "
            "CE3 PHD2 LOF: "
            "  PHD2 activity 50% → insufficient for full HIF hydroxylation at normoxia; "
            "  HIF1α/2α partially stable → "
            "    EPO↑ (HIF2α → EPO gene enhancer → kidney EPO production); "
            "    VEGF↑, GLUT1↑, LDHA↑ etc. (chronic HIF target upregulation); "
            "ERYTHROCYTOSIS PATHWAY: "
            "  EPO↑ → EPOR activation → JAK2-STAT5 → erythroid progenitor proliferation/survival; "
            "PARAGANGLIOMA PATHWAY: "
            "  HIF1α + SDH pathway sharing: PHD2 LOF creates similar HIF-stabilising milieu as SDH loss; "
            "  SDHA/B/C/D/SDHAF2 LOF → HIF-like state → paraganglioma; "
            "  EGLN1 LOF → analogous HIF1α/2α stabilisation → paraganglioma predisposition; "
            "ASCORBATE THERAPY: "
            "  PHD2 requires ascorbate (Vitamin C) to maintain Fe2+; "
            "  Ascorbate deficiency → functional PHD2 reduction → secondary polycythemia; "
            "  True CE3 not correctable with ascorbate"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC / HIGHLY CHARACTERISTIC — EGLN1-CE3: "
            "  1. ERYTHROCYTOSIS + ELEVATED EPO + EGLN1 VARIANT: "
            "     Triad confirms CE3; "
            "  2. PARAGANGLIOMA + ERYTHROCYTOSIS (TRIAD): "
            "     CE3 is the ONLY hereditary erythrocytosis with significant paraganglioma risk; "
            "     Any patient with erythrocytosis + paraganglioma/phaeochromocytoma: test EGLN1; "
            "  3. PHD2 ENZYME ACTIVITY <50%: "
            "     Research assay; specific for CE3 (not routinely available clinically); "
            "  4. OXYGEN SENSING PATHWAY (HIF AXIS): "
            "     Elevated plasma HIF2α / VEGF: indirect evidence of HIF stabilisation; "
            "  5. FAMILY HISTORY (AD) OF ERYTHROCYTOSIS: "
            "     Multiple generations; co-segregation with EGLN1 variant"
        ),
        "treatment": (
            "TREATMENT — EGLN1-CE3: "
            "ERYTHROCYTOSIS: "
            "  VENESECTION: target Hct <0.50; "
            "  LOW-DOSE ASPIRIN: thrombosis prevention; "
            "  Hydroxycarbamide: if frequent venesection needed; "
            "PARAGANGLIOMA SURVEILLANCE (MANDATORY): "
            "  ANNUAL BIOCHEMICAL SCREEN: "
            "    Plasma free metanephrines (sensitivity >95% for phaeochromocytoma); "
            "    If elevated: 24h urine metanephrines + catecholamines confirmation; "
            "  IMAGING EVERY 2-3 YEARS: "
            "    68Ga-DOTATATE PET/CT or MRI abdomen/thorax; "
            "    Head-and-neck paraganglioma: MRI head/neck; "
            "PARAGANGLIOMA TREATMENT: "
            "  Surgical resection if localised; "
            "  Alpha-blockade pre-operatively (phenoxybenzamine/doxazosin) MANDATORY before surgery; "
            "  MIBG therapy for metastatic paraganglioma; "
            "  SSA (somatostatin analogues) if SSTR-positive on PET; "
            "GENETIC COUNSELLING: "
            "  Autosomal dominant; 50% offspring risk; "
            "  Offer predictive testing to first-degree relatives; "
            "  Test: erythrocytosis + PHD/HIF pathway panel"
        ),
        "seed": 2888,
    },
    {
        "gene": "EPAS1",
        "protein": (
            "EPAS1 -- 2p21 AD-GOF -- 870aa -- Endothelial-PAS-Domain-Protein-1-"
            "HIF2alpha-Hypoxia-Inducible-Factor-2alpha-118kDa-bHLH-PAS-"
            "Erythrocytosis4-HIF2alpha-GOF-Paraganglioma-PAH-"
            "Belzutifan-Welireg-FDA2021-"
            "OMIM-Gene-603349-Disease-CE4-611783"
        ),
        "locus": "2p21",
        "protein_size": (
            "870 aa / 118 kDa (bHLH-PAS transcription factor; must heterodimerize with ARNT (HIF1β); "
            "EPAS1 = HIF2α: the dominant HIF isoform for EPO regulation in kidney; "
            "DEGRADATION DOMAIN (ODD): Pro405 and Pro531 — hydroxylated by PHD2 → VHL binding; "
            "C-TAD (C-terminal transactivation domain): interacts with p300/CBP co-activators; "
            "GOF VARIANTS: missense in ODD domain (Pro405/531 region) or N-TAD → "
            "  PHD2 cannot hydroxylate → VHL cannot bind → HIF2α escapes degradation; "
            "  Some variants: ARNT dimerisation enhanced → prolonged nuclear activity; "
            "EPAS1 DRIVES EPO IN KIDNEY: "
            "  EPAS1 (HIF2α) preferentially drives EPO gene expression (kidney/liver); "
            "  HIF1α preferentially drives metabolic targets; "
            "BELZUTIFAN: disrupts EPAS1-ARNT dimerisation (HIF2α-HIF1β) → HIF2α cannot transactivate"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT (AD) — GAIN-OF-FUNCTION — EPAS1: "
            "  GOF MECHANISM: ODD domain missense → PHD2 cannot hydroxylate → HIF2α escapes VHL; "
            "  SERUM EPO: ELEVATED (HIF2α drives EPO transcription); "
            "  SOMATIC VS GERMLINE: "
            "    SOMATIC GOF EPAS1: polycythaemia + paraganglioma syndrome (often somatic mosaic); "
            "    GERMLINE GOF EPAS1: CE4; rarer; can arise de novo; "
            "  PARAGANGLIOMA: very common with EPAS1 GOF (more than EGLN1); "
            "  PULMONARY ARTERIAL HYPERTENSION (PAH): "
            "    EPAS1 GOF → elevated VEGF → pulmonary vascular remodelling → PAH; "
            "    ~20-30% of CE4 patients develop clinically significant PAH; "
            "  POLYCYTHAEMIA-PARAGANGLIOMA-PAH TRIAD: consider EPAS1 GOF; "
            "  DE NOVO: many EPAS1 GOF are de novo (not inherited from parent); "
            "  PENETRANCE: high for erythrocytosis; variable for paraganglioma + PAH"
        ),
        "disease_category": (
            "CONGENITAL ERYTHROCYTOSIS TYPE 4 (CE4) — HIF2α GAIN-OF-FUNCTION — OMIM 611783; "
            "SECONDARY ERYTHROCYTOSIS WITH PARAGANGLIOMA + PAH TRIAD: "
            "  Haemoglobin: usually 18-23 g/dL; "
            "  Serum EPO: ELEVATED (HIF2α → kidney EPO); "
            "  Paraganglioma/phaeochromocytoma: VERY COMMON (>50% of CE4); "
            "    Head-and-neck + abdominal paraganglioma; "
            "  Pulmonary arterial hypertension: ~20-30%; "
            "  DIAGNOSIS: "
            "    Erythrocytosis + elevated EPO + EPAS1 GOF variant; "
            "    +/- paraganglioma + PAH; "
            "    Somatic mosaic: allele fraction may be low (blood biopsy may miss); "
            "      Consider NGS from paraganglioma tissue if index of suspicion high; "
            "  BELZUTIFAN (Welireg): "
            "    HIF2α inhibitor; FDA Aug 2021 for VHL-related neoplasms; "
            "    Repurposed for EPAS1 GOF (off-label but mechanistically direct target); "
            "    Also reduces erythrocytosis + may stabilise paraganglioma growth"
        ),
        "disease_pathway": (
            "EPAS1-HIF2α-EPO PATHWAY AND PATHOLOGICAL GOF: "
            "NORMAL: "
            "  PHD2 hydroxylates HIF2α (EPAS1) Pro405/531 → VHL binds → ubiquitination → proteasomal destruction; "
            "  Hypoxia → PHD2 inactive → HIF2α stable → ARNT dimerization → HIF2α-ARNT heterodimer → "
            "    binds HRE (hypoxia response element) in EPO gene → EPO transcription; "
            "CE4 GOF: "
            "  ODD variant → HIF2α cannot be hydroxylated → VHL cannot bind → HIF2α stable at normoxia; "
            "  HIF2α-ARNT constitutively active → EPO↑ → erythrocytosis; "
            "  VEGF↑ → pulmonary vascular remodelling (PAH); "
            "  SDH-like HIF pathway upregulation → paraganglioma predisposition; "
            "BELZUTIFAN MECHANISM: "
            "  Binds hydrophobic PAS-B pocket of EPAS1 (HIF2α); "
            "  Disrupts EPAS1-ARNT (HIF2α-HIF1β) protein-protein interaction; "
            "  HIF2α cannot transactivate → EPO suppressed → erythrocytosis improves; "
            "  First-in-class HIF2α inhibitor; key benchmark for EPAS1 GOF therapy"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC / HIGHLY CHARACTERISTIC — EPAS1-CE4: "
            "  1. ERYTHROCYTOSIS + ELEVATED EPO + PARAGANGLIOMA + PAH: "
            "     TRIAD: think EPAS1 GOF first when all three present; "
            "  2. DE NOVO GOF EPAS1 VARIANT IN ODD DOMAIN: "
            "     Pro405/531 region missense = classic GOF; "
            "  3. SOMATIC MOSAIC EPAS1 IN PARAGANGLIOMA TISSUE: "
            "     May not be detectable in blood — tissue sequencing required; "
            "  4. ELEVATED SERUM EPO + NORMAL JAK2 V617F: "
            "     Excludes PV; elevated EPO = secondary mechanism; "
            "  5. RESPONSE TO BELZUTIFAN: "
            "     Reduction in Hb + EPO on HIF2α inhibitor is diagnostic AND therapeutic"
        ),
        "treatment": (
            "TREATMENT — EPAS1-CE4: "
            "ERYTHROCYTOSIS: "
            "  VENESECTION: target Hct <0.50; "
            "  LOW-DOSE ASPIRIN; "
            "  BELZUTIFAN (Welireg, HIF2α inhibitor): "
            "    FDA 2021 for VHL syndrome; off-label for EPAS1 GOF CE4; "
            "    Directly targets pathogenic mechanism; reduces EPO + erythrocytosis; "
            "    40 mg daily oral; "
            "    Monitor: anaemia (overcorrection), hypertension, headache, dizziness; "
            "    Embryofoetal toxicity — contraception required; "
            "PARAGANGLIOMA: "
            "  Annual surveillance: plasma metanephrines + 68Ga-DOTATATE PET; "
            "  Surgical resection if localised; "
            "  Pre-operative alpha-blockade MANDATORY; "
            "  Belzutifan may slow paraganglioma growth (EPAS1-driven proliferation); "
            "PULMONARY ARTERIAL HYPERTENSION: "
            "  Right heart catheterisation if symptomatic; "
            "  PAH-specific therapy (ERA, PDE5i, prostacyclin) per severity; "
            "  Belzutifan: may reduce VEGF → improve PAH over time; "
            "GENETIC COUNSELLING: "
            "  Many de novo → test parents; if one parent carries → 50% offspring risk"
        ),
        "seed": 2889,
    },
    {
        "gene": "HBB",
        "protein": (
            "HBB -- 11p15.4 AD -- 147aa -- Haemoglobin-Beta-Subunit-"
            "16kDa-Globin-Alpha2Beta2-Tetramer-O2-Carrier-"
            "High-Affinity-HBB-Variants-Left-Shifted-ODC-Erythrocytosis6-"
            "Hb-Chesapeake-Hiroshima-Malmö-Rainier-"
            "OMIM-Gene-141900-Disease-CE6-617981"
        ),
        "locus": "11p15.4",
        "protein_size": (
            "147 aa / 16 kDa (haemoglobin beta subunit; globin fold; 8 alpha-helices; "
            "haem cofactor in haem pocket; "
            "T-state (deoxy, low O2 affinity) ↔ R-state (oxy, high O2 affinity): cooperative transition; "
            "2,3-BPG: binds T-state β-chain pocket (βVal1, βLys82, βHis143) → stabilises T-state → "
            "  LOWER O2 affinity; "
            "HIGH-AFFINITY VARIANTS: amino acid substitutions destabilise T-state or stabilise R-state → "
            "  LEFT-SHIFTED O2 DISSOCIATION CURVE (ODC) → O2 not released in tissues → "
            "    apparent tissue hypoxia → EPO↑ → secondary compensatory erythrocytosis; "
            "KEY HIGH-AFFINITY HBB VARIANTS: "
            "  Hb Chesapeake: α92Arg→Leu (α-chain; affects α-β2 contact — but this is HBA1); "
            "  Hb Hiroshima (HbH): β146His→Asp — Bohr effect residue; extremely high affinity; "
            "  Hb Malmö: β97Arg→Gln; very high O2 affinity; "
            "  Hb Rainier: β145Tyr→Cys; salt bridge disruption → R-state trapped; "
            "  Hb Bethesda: β145Tyr→His; very high affinity; severe erythrocytosis; "
            "  Hb Kempsey: β99Asp→Asn; alpha1-beta2 contact → R-state lock"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT (AD) — HBB HIGH-AFFINITY VARIANTS: "
            "  SINGLE HETEROZYGOUS MISSENSE SUFFICIENT: "
            "    1 abnormal beta chain → alpha2 (normal beta)(abnormal beta) AND alpha2 (abnormal beta)2 Hbs; "
            "    Mixed tetramers: high-affinity Hb dominates; "
            "  PENETRANCE: HIGH (nearly complete for erythrocytosis phenotype); "
            "  SERUM EPO: ELEVATED (tissue hypoxia signal due to O2 non-delivery → kidney EPO); "
            "  p50 O2 MEASUREMENT: LOW (left-shifted ODC); "
            "    p50: partial pressure of O2 at 50% Hb saturation; normal ~26 mmHg; "
            "    High-affinity Hb: p50 <20 mmHg (sometimes <15 mmHg); "
            "  FAMILY HISTORY: typically multigenerational; often incidentally found as polycythaemia; "
            "  HAEMOGLOBIN ELECTROPHORESIS: often normal (some variants not separable by standard methods); "
            "    HPLC: may show abnormal peak; "
            "    Best confirmation: globin gene sequencing OR direct p50 measurement; "
            "  NO SICKLING, NO HAEMOLYSIS: high-affinity variants cause erythrocytosis only"
        ),
        "disease_category": (
            "HIGH-AFFINITY HAEMOGLOBIN ERYTHROCYTOSIS — CE6 — OMIM 617981; "
            "SECONDARY COMPENSATORY ERYTHROCYTOSIS — NO TREATMENT USUALLY NEEDED: "
            "  MECHANISM: Hb cannot release O2 normally → tissues 'hypoxic' despite normal saturation; "
            "  Hb/Hct elevated: compensatory RBC mass increase to deliver adequate O2; "
            "  Serum EPO: elevated (appropriate response to reduced O2 delivery); "
            "  O2 saturation: NORMAL (Hb loaded efficiently with O2 at lungs); "
            "  SYMPTOMS: minimal (compensation maintains O2 delivery); "
            "    Headache, plethora if Hct very high (>60%); "
            "  DIAGNOSIS: "
            "    Erythrocytosis + elevated EPO + low p50 O2; "
            "    Haemoglobin variant identification: globin gene sequencing; "
            "    Exclude PV (JAK2 V617F); exclude COPD, sleep apnoea, EPO-secreting tumour; "
            "  KEY POINT: TREATMENT USUALLY NOT REQUIRED: "
            "    Erythrocytosis is COMPENSATORY — removing it (venesection) worsens O2 delivery; "
            "    Only treat if Hct >60% causing symptomatic hyperviscosity"
        ),
        "disease_pathway": (
            "HAEMOGLOBIN-OXYGEN AFFINITY AND ERYTHROCYTOSIS MECHANISM: "
            "NORMAL Hb COOPERATIVITY: "
            "  T-state (deoxy): low affinity; tense quaternary structure; αβ interfaces strained; "
            "  R-state (oxy): high affinity; relaxed structure; "
            "  COOPERATIVE: each O2 bound → facilitates next binding (sigmoid ODC); "
            "  p50 = ~26 mmHg; O2 loaded at lungs (pO2~100 mmHg); released at tissues (pO2~40 mmHg); "
            "HIGH-AFFINITY Hb VARIANT: "
            "  Key contact points disrupted: "
            "    α1-β2 interface (Chesapeake α92; Kempsey β99): contacts stabilise T-state; disrupted → R-state lock; "
            "    C-terminal β-chain salt bridges (Rainier β145Tyr, Bethesda β145Tyr, Hiroshima β146His): "
            "      Bohr proton binding sites; disrupted → reduced T-state stability; "
            "  Result: Hb stays in R-state (oxy) even at tissue pO2 → O2 NOT released; "
            "  Tissues perceive hypoxia → HIF2α activation → EPO↑; "
            "  Kidneys increase EPO → erythroid expansion → more RBCs; "
            "  More RBCs → more O2 loading → equilibrium established at higher Hb; "
            "2,3-BPG INTERACTION: "
            "  Normal 2,3-BPG stabilises T-state; "
            "  High-affinity variants: some have reduced 2,3-BPG binding pocket affinity → "
            "    further right-to-left shift; BPGM deficiency: same effect from reduced 2,3-BPG"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC / HIGHLY CHARACTERISTIC — HBB HIGH-AFFINITY VARIANTS: "
            "  1. LOW p50 O2 (<20 mmHg; normal ~26 mmHg): "
            "     LEFT-SHIFTED ODC — confirmed by oxygen dissociation curve measurement; "
            "     MOST SPECIFIC test for high-affinity Hb; "
            "  2. ERYTHROCYTOSIS + ELEVATED EPO + NORMAL JAK2 V617F: "
            "     Secondary erythrocytosis; EPO-driven; "
            "  3. HAEMOGLOBIN VARIANT IDENTIFICATION: "
            "     HPLC/capillary electrophoresis: may show abnormal peak; "
            "     Globin chain sequencing: confirms specific variant; "
            "  4. NORMAL O2 SATURATION WITH ERYTHROCYTOSIS: "
            "     SpO2 normal on pulse oximetry (Hb loaded normally at lungs); "
            "     Distinguishes from respiratory/cardiac cause of erythrocytosis; "
            "  5. MULTIGENERATIONAL POLYCYTHEMIA (AD): "
            "     Family history of polycythemia without PV or obvious secondary cause"
        ),
        "treatment": (
            "TREATMENT — HBB HIGH-AFFINITY VARIANTS: "
            "USUALLY NO TREATMENT REQUIRED: "
            "  Erythrocytosis is PHYSIOLOGICAL COMPENSATION — DO NOT SIMPLY VENESECT; "
            "  Reducing RBC mass → worsens O2 delivery → symptoms worse; "
            "  Target Hct: NOT the standard 0.50; higher Hct is the patient's homeostasis; "
            "INDICATIONS FOR VENESECTION: "
            "  Hct >0.60-0.65 causing symptomatic hyperviscosity: "
            "    Headache, visual changes, digital ischaemia; "
            "  If venesection used: target Hct 0.55-0.58 (lower than their habitual but allows O2 compensation); "
            "  Monitor for symptoms after each venesection; "
            "ASPIRIN: low-dose 75 mg daily for thrombosis prevention if Hct >55%; "
            "AVOID: "
            "  Aggressive venesection to 'normal' Hct (causes tissue hypoxia); "
            "  Hydroxycarbamide (reduces RBC production — worsens the compensatory mechanism); "
            "  Iron supplementation (worsens erythrocytosis); "
            "COUNSELLING: "
            "  Lifelong management; generally benign prognosis; "
            "  High altitude: tolerated well or better than normal (O2 affinity advantage); "
            "  Pregnancy: increased monitoring; higher Hct may be safer in high-affinity Hb pregnancy"
        ),
        "seed": 2890,
    },
    {
        "gene": "HBA1",
        "protein": (
            "HBA1 -- 16p13.3 AD -- 142aa -- Haemoglobin-Alpha-1-"
            "15kDa-Globin-HBA1-HBA2-Tandem-Alpha-Locus-"
            "High-Affinity-Alpha-Chain-Variants-Erythrocytosis-"
            "Hb-Chesapeake-Suresnes-Evanston-Torino-"
            "OMIM-Gene-141800"
        ),
        "locus": "16p13.3",
        "protein_size": (
            "142 aa / 15 kDa (haemoglobin alpha-1 subunit; globin fold; haem pocket; "
            "HBA1 and HBA2 are duplicated alpha genes at 16p13.3 (HBA2 telomeric, HBA1 centromeric); "
            "normal: 2 HBA1 + 2 HBA2 copies (4 alpha genes total); "
            "alpha chain variants causing HIGH O2 AFFINITY: "
            "  Hb Chesapeake: α92Arg→Leu (α1β2 contact region) — disrupts T-R transition; "
            "    Most important HIGH-AFFINITY alpha variant; common reported cause of erythrocytosis; "
            "  Hb Suresnes: α141Arg→His (C-terminal salt bridge) → slightly elevated affinity; "
            "  Hb Evanston: α14Trp→Arg → haem pocket variant; mild erythrocytosis; "
            "  Hb Torino: α43Phe→Val → haem pocket instability → slightly unstable + high affinity; "
            "ANALYSIS: HBA1 and HBA2 must both be sequenced (tandem duplication); "
            "MLPA/gap-PCR for deletions; "
            "p50 O2 measurement: confirms high affinity; determines clinical significance"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT (AD) — HBA1 HIGH-AFFINITY VARIANTS: "
            "  SIMILAR TO HBB HIGH-AFFINITY VARIANTS but alpha-chain specific; "
            "  PENETRANCE: HIGH for erythrocytosis phenotype; "
            "  DOSAGE: HBA1 heterozygous (1 of 4 alpha genes) → milder than HBB heterozygous; "
            "    Alpha chain variants may be less symptomatic than beta chain variants; "
            "  SERUM EPO: ELEVATED (secondary); "
            "  p50 O2: LOW (left-shifted ODC); "
            "  SPECIAL GENETIC CONSIDERATION: "
            "    HBA1 vs HBA2 distinction: clinical effect similar if same amino acid change; "
            "    Deletion causing erythrocytosis: very rare (deletions usually cause alpha-thal REDUCTION); "
            "    High-affinity alpha variants usually point mutations in HBA1 or HBA2; "
            "    SEQUENCING: must cover both HBA1 and HBA2 (cannot distinguish clinically); "
            "  DIAGNOSIS CONFIRMATION: "
            "    Globin chain reverse-phase HPLC: may show abnormal alpha-chain peak; "
            "    Mass spectrometry: precise variant identification; "
            "    p50: confirms functional significance"
        ),
        "disease_category": (
            "HIGH-AFFINITY HAEMOGLOBIN ALPHA-CHAIN ERYTHROCYTOSIS; "
            "SECONDARY COMPENSATORY ERYTHROCYTOSIS — SIMILAR TO HBB HIGH-AFFINITY: "
            "  MECHANISM: identical to HBB high-affinity variants — O2 not released → EPO↑ → erythrocytosis; "
            "  CLINICAL FEATURES: "
            "    Erythrocytosis; elevated EPO; low p50; "
            "    Less severe than HBB variants (4 alpha genes = 1 variant is 25% of alpha chains); "
            "  DISTINCTION FROM ALPHA-THALASSEMIA: "
            "    Alpha-thal: deletion/LOF → LESS alpha chains → hypochromic microcytic anaemia; "
            "    High-affinity alpha variant: point mutation → HIGH O2 affinity → erythrocytosis; "
            "    Completely different clinical direction; "
            "  MANAGEMENT: "
            "    Usually conservative (erythrocytosis is compensatory); "
            "    Venesection only if Hct >60% symptomatic; "
            "    Aspirin; "
            "  HAEMOGLOBIN VARIANT IDENTIFICATION: "
            "    HPLC/CE: abnormal alpha-chain peak; "
            "    Mass spectrometry: gold standard identification; "
            "    Gene sequencing: HBA1+HBA2 panel"
        ),
        "disease_pathway": (
            "ALPHA-CHAIN HIGH-AFFINITY VARIANTS — SAME MECHANISM AS HBB: "
            "HAEMOGLOBIN TETRAMER: "
            "  Normal: α2β2 tetramer; 2 alpha (HBA1+HBA2) + 2 beta (HBB) chains; "
            "  Alpha-chain variant: forms abnormal αmut2β2 AND α2αmut(β2) tetramers; "
            "  Cooperativity of transition depends on α1β2 interface contacts; "
            "HB CHESAPEAKE (α92Arg→Leu): "
            "  α92 is in the α1β2 contact region: stabilises T-state; "
            "  Arg→Leu: hydrogen bond disrupted → T-state less stable → R-state favoured; "
            "  p50: ~12 mmHg (very low; very high affinity); "
            "  Erythrocytosis often severe; Hb 18-22 g/dL; "
            "HB SURESNES (α141Arg→His): "
            "  C-terminal Arg141 forms salt bridge with Pro(β2)37 in T-state; "
            "  His substitution: reduced charge → weakened T-state → modest left shift; "
            "  Milder erythrocytosis; "
            "PHYSIOLOGICAL IMPLICATION: "
            "  High-affinity Hb loads well at lungs but cannot unload at tissues; "
            "  Functional hypoxia despite normal SpO2; EPO response; erythrocytosis"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC / HIGHLY CHARACTERISTIC — HBA1 HIGH-AFFINITY VARIANTS: "
            "  1. LOW p50 (<22 mmHg) + ERYTHROCYTOSIS + ELEVATED EPO: "
            "     High-affinity Hb (alpha chain); "
            "  2. ALPHA-CHAIN VARIANT ON HPLC/MASS SPECTROMETRY: "
            "     Abnormal alpha chain identifies specific variant; "
            "  3. NORMAL SpO2 + NORMAL JAK2 V617F: "
            "     Rules out respiratory erythrocytosis + PV; "
            "  4. HBA1/HBA2 SEQUENCING CONFIRMS VARIANT: "
            "     Both genes must be sequenced; "
            "  5. FAMILY HISTORY (AD): "
            "     Erythrocytosis without secondary cause across generations"
        ),
        "treatment": (
            "TREATMENT — HBA1 HIGH-AFFINITY VARIANTS: "
            "IDENTICAL APPROACH TO HBB HIGH-AFFINITY VARIANTS: "
            "  USUALLY CONSERVATIVE (erythrocytosis is compensatory): "
            "    Do NOT aggressively reduce Hct to 'normal' — worsens O2 delivery; "
            "  VENESECTION: ONLY if Hct >60-65% with symptomatic hyperviscosity; "
            "    Target Hct: the patient's own physiological level, not standard <50%; "
            "  ASPIRIN 75 mg daily: if Hct persistently >55%; "
            "  FOLLOW-UP: annual FBC; "
            "GENERALLY MILDER THAN HBB VARIANTS: "
            "  1 of 4 alpha genes affected = 25% abnormal alpha chains vs 50% beta chains in HBB het; "
            "  Clinical erythrocytosis often milder; "
            "GENETIC ANALYSIS: "
            "  Sequence both HBA1 and HBA2 + MLPA for alpha globin deletions; "
            "  Family cascade testing after index identification"
        ),
        "seed": 2891,
    },
    {
        "gene": "BPGM",
        "protein": (
            "BPGM -- 7q33 AR -- 258aa -- Bisphosphoglycerate-Mutase-"
            "30kDa-Bifunctional-2,3-BPG-Synthase-Phosphatase-"
            "2,3-BPG-Deficiency-Erythrocytosis8-Left-Shifted-ODC-"
            "OMIM-Gene-613937-Disease-CE8-222800"
        ),
        "locus": "7q33",
        "protein_size": (
            "258 aa / 30 kDa (erythrocyte-specific bifunctional enzyme; "
            "SYNTHASE ACTIVITY: 1,3-BPG → 2,3-BPG (2,3-bisphosphoglycerate); "
            "PHOSPHATASE ACTIVITY: 2,3-BPG → 3-PG; "
            "2,3-BPG FUNCTION: critical allosteric effector of haemoglobin; "
            "  Binds β-chain pocket → stabilises T-state → reduces O2 affinity → RIGHT-SHIFTS ODC; "
            "  Promotes O2 release in tissues (normal physiological O2 unloading); "
            "BPGM DEFICIENCY (CE8): no 2,3-BPG produced → Hb has NO allosteric modulation → "
            "  Hb behaves like fetal Hb (HbF which has 2,3-BPG-insensitive γ-chains) → "
            "  VERY HIGH O2 affinity → left-shifted ODC → tissue hypoxia → EPO↑ → erythrocytosis; "
            "GLYCOLYTIC CONSEQUENCE: "
            "  BPGM deficiency → 2,3-BPG absent → glycolysis not channelled through mutase shunt → "
            "  All 1,3-BPG → 3-PG via PGK → MORE ATP produced (not less); "
            "  Erythrocytes: normal survival (not haemolytic unlike PKLR deficiency); "
            "DIAGNOSIS: "
            "  2,3-BPG in RBCs: ABSENT or markedly reduced; "
            "  p50 O2: very low (sometimes <10 mmHg — extreme); "
            "  BPGM enzyme assay; BPGM gene sequencing"
        ),
        "inheritance": (
            "AUTOSOMAL RECESSIVE (AR) — BPGM — BIALLELIC LOSS-OF-FUNCTION: "
            "  BIALLELIC LOF: complete absence of 2,3-BPG in erythrocytes; "
            "  HETEROZYGOTES: 50% BPGM activity → 50% 2,3-BPG → milder left-shift; "
            "    Heterozygotes: mildly elevated Hb; usually subclinical; "
            "  HOMOZYGOUS/COMPOUND HETEROZYGOUS: complete deficiency → severe erythrocytosis; "
            "  VERY RARE: <50 cases worldwide described; "
            "  KNOWN VARIANTS: "
            "    No dominant founder; private variants; missense + frameshift; "
            "    p.Arg89Gln: first described; affects active site; "
            "  SERUM EPO: ELEVATED (left-shifted ODC → tissue hypoxia → EPO); "
            "  p50: VERY LOW (<10-15 mmHg vs normal 26 mmHg); "
            "    Most extreme left-shift of any hereditary erythrocytosis; "
            "  RBC SURVIVAL: NORMAL (not haemolytic); "
            "  NO HAEMOLYSIS: distinguishes from PKLR/G6PD deficiency (which cause haemolysis)"
        ),
        "disease_category": (
            "2,3-BPG DEFICIENCY — BPGM ERYTHROCYTOSIS — CE8 — OMIM 222800; "
            "SECONDARY ERYTHROCYTOSIS — EXTREME LEFT-SHIFTED ODC: "
            "  2,3-BPG: normally 5 mmol/L in erythrocytes (major allosteric effector); "
            "  BPGM CE8: 2,3-BPG near zero → Hb very high O2 affinity; "
            "  Haemoglobin: 18-24 g/dL (can be extremely high in biallelic); "
            "  p50: <10-15 mmHg (extremely low); "
            "  EPO: markedly elevated; "
            "  SYMPTOMS: "
            "    Plethora, headache, visual disturbance (hyperviscosity); "
            "    Splenomegaly (extramedullary haematopoiesis); "
            "  LABORATORY DDx: "
            "    RBC 2,3-BPG measurement: ZERO or near-zero (diagnostic); "
            "    p50 O2: extremely low; "
            "    Haemoglobin electrophoresis: normal (not an Hb variant); "
            "    RBC indices: normal (not anaemic, not microcytic); "
            "    Reticulocytes: mildly elevated (compensation); "
            "    No haemolysis markers (LDH normal or mildly elevated)"
        ),
        "disease_pathway": (
            "2,3-BPG PRODUCTION AND BPGM DEFICIENCY EFFECTS: "
            "NORMAL 2,3-BPG PATHWAY: "
            "  1,3-BPG (glycolysis intermediate) → BPGM synthase → 2,3-BPG → BPGM phosphatase → 3-PG; "
            "  2,3-BPG: 80-90% of RBC phosphate; most abundant organic phosphate in blood; "
            "  2,3-BPG binds β-chain central cavity (βVal1-βHis143-βLys82 pocket): "
            "    Stabilises T-state → reduces Hb-O2 affinity → RIGHT-shifts ODC → facilitates O2 release; "
            "BPGM DEFICIENCY EFFECT: "
            "  No 2,3-BPG → Hb loses T-state stabilisation → functions like HbF (fetal Hb); "
            "  HbF: lacks β-chains → 2,3-BPG cannot bind → high O2 affinity (designed for placental O2 transfer); "
            "  BPGM deficiency: Hb equivalent to HbF affinity → appropriate for fetal life, not adult tissue needs; "
            "  Tissues perceive hypoxia → EPO response → erythrocytosis; "
            "GLYCOLYTIC CHANNEL: "
            "  Mutase shunt bypasses PGK (step 7 of glycolysis); "
            "  BPGM LOF → ALL 1,3-BPG → PGK → MORE ATP/glucose; "
            "  No energy deficit in erythrocytes (unlike PK deficiency which reduces ATP); "
            "FETAL COMPARISON: "
            "  This is why neonates/fetuses have higher Hb affinity — HbF + lower 2,3-BPG; "
            "  BPGM deficiency: adults with fetal-like Hb function"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC / HIGHLY CHARACTERISTIC — BPGM-CE8: "
            "  1. RBC 2,3-BPG ABSENT OR NEAR-ZERO: "
            "     DIAGNOSTIC; normal 5 mmol/L; CE8 near 0; "
            "     Most specific test; enzymatic 2,3-BPG measurement in erythrocytes; "
            "  2. EXTREME p50 (<10-15 mmHg): "
            "     Most extreme left-shift of any hereditary erythrocytosis; "
            "     Higher Hb affinity than even severe high-affinity Hb variants; "
            "  3. ERYTHROCYTOSIS + NORMAL HAEMOGLOBIN ELECTROPHORESIS: "
            "     No Hb variant identified → excludes HBB/HBA1 high-affinity variants; "
            "     BPGM LOF = normal Hb structure, abnormal 2,3-BPG content; "
            "  4. NO HAEMOLYSIS: "
            "     Distinguishes from PKLR/G6PD (which cause haemolysis + erythrocytosis); "
            "  5. BIALLELIC BPGM VARIANT: "
            "     Gene sequencing confirms AR inheritance"
        ),
        "treatment": (
            "TREATMENT — BPGM-CE8: "
            "CONSERVATIVE IN MOST CASES: "
            "  Erythrocytosis is compensatory — essential for O2 delivery; "
            "  DO NOT aggressively reduce Hct (worsens tissue oxygenation); "
            "VENESECTION: "
            "  ONLY if Hct >65% causing symptomatic hyperviscosity; "
            "  Target Hct: 55-60% (allow compensation to be maintained); "
            "  Monitor: symptoms, p50 unchanged (structural — phlebotomy does not alter 2,3-BPG deficit); "
            "ASPIRIN: 75 mg daily if Hct >60%; "
            "NO SPECIFIC ENZYME REPLACEMENT: "
            "  No BPGM enzyme therapy available; "
            "  No gene therapy; "
            "  No dietary modification (2,3-BPG cannot be supplemented externally); "
            "FUTURE: "
            "  Myo-inositol hexaphosphate (IHP): synthetic 2,3-BPG substitute; "
            "    Stabilises T-state like 2,3-BPG; investigational; not clinically available; "
            "    If delivered into erythrocytes: could restore right-shift; "
            "  Voxelotor (GBT440): actually INCREASES O2 affinity (designed for SCD) — CONTRAINDICATED; "
            "MONITORING: "
            "  Annual FBC; 2,3-BPG monitoring (not useful — will always be 0 in biallelic); "
            "  Thrombosis risk: aspirin; watch Hct"
        ),
        "seed": 2892,
    },
    {
        "gene": "EPO",
        "protein": (
            "EPO -- 7q22.3 AD-rare -- 193aa -- Erythropoietin-"
            "34kDa-Glycoprotein-Hormone-Kidney-Primary-Source-"
            "Germline-GOF-Erythrocytosis5-Rare-Hereditary-"
            "OMIM-Gene-133170-Disease-CE5-617606"
        ),
        "locus": "7q22.3",
        "protein_size": (
            "193 aa / 34 kDa (processed from 193 aa signal + pro-peptide → mature 165 aa; "
            "4-helix bundle cytokine; 3 N-linked + 1 O-linked glycosylation sites; "
            "glycosylation: critical for stability and half-life; "
            "KIDNEY (peritubular interstitial cells): primary source (90% of circulating EPO); "
            "  Produced in response to HIFα signalling (hypoxia or HIF pathway variants); "
            "LIVER: secondary EPO source (fetal → neonatal → compensates if renal EPO insufficient); "
            "MECHANISM OF ACTION: "
            "  EPO → EPOR homodimer → JAK2-STAT5 → erythroid progenitor survival/proliferation; "
            "GERMLINE GOF EPO MUTATIONS (CE5): very rare; "
            "  Promoter/enhancer variants → constitutively elevated EPO transcription; "
            "  Structural variants disrupting negative regulatory regions → higher EPO expression; "
            "EPO HALF-LIFE: 4-12 hours (IV); 24 hours (SC); "
            "EPO ASSAY: serum or plasma EPO; normal 4-30 IU/L; CE5: >30 IU/L (often markedly elevated)"
        ),
        "inheritance": (
            "AUTOSOMAL DOMINANT (AD) — EPO GERMLINE GAIN-OF-FUNCTION (VERY RARE): "
            "  MECHANISM: "
            "    Gain-of-function variants in EPO gene regulatory regions (promoter/enhancer/3'UTR); "
            "    OR structural variants removing negative regulatory elements; "
            "    → EPO transcription constitutively elevated (without hypoxic stimulus); "
            "  VERY RARE: <20 families reported worldwide; "
            "  SERUM EPO: ELEVATED (EPO is the primary elevated mediator — unlike secondary where HIF first); "
            "  DISTINCTION FROM SECONDARY ERYTHROCYTOSIS (REACTIVE): "
            "    CE5: hereditary; germline variant; lifelong elevated EPO + erythrocytosis from childhood; "
            "    Reactive: acquired; EPO rises in response to identifiable hypoxic cause (COPD, RCC, etc.); "
            "  DISTINCTION FROM EPO DOPING: "
            "    CE5: germline; EPO gene variant; consistent elevated EPO; no cycling; "
            "    EPO doping: recombinant EPO; variable levels; no EPO gene variant; "
            "  DISTINCTION FROM VHL/EGLN1/EPAS1: "
            "    CE5: EPO gene direct GOF; "
            "    VHL/EGLN1/EPAS1: HIF pathway upstream → EPO secondarily elevated (via HIF2α); "
            "    Both produce elevated EPO but at different levels in the pathway"
        ),
        "disease_category": (
            "HEREDITARY ERYTHROCYTOSIS TYPE 5 (CE5) — EPO GERMLINE GOF — OMIM 617606; "
            "VERY RARE HEREDITARY SECONDARY ERYTHROCYTOSIS: "
            "  Primary defect: EPO gene constitutively overexpressed; "
            "  Serum EPO: markedly elevated (often >100 IU/L; sometimes >300 IU/L); "
            "  Haemoglobin: 18-22 g/dL; "
            "  Splenomegaly: may be present; "
            "  NO VHL/EGLN1/EPAS1 HIF PATHWAY VARIANTS: EPO elevated from direct gene GOF; "
            "  DIAGNOSIS: "
            "    Very elevated EPO (not proportionate to degree of erythrocytosis = inappropriately elevated); "
            "    Exclude: hypoxia, renal EPO-secreting tumour (RCC), hepatoma, haemangioblastoma; "
            "    Exclude: PV (JAK2 V617F negative, EPO suppressed in PV); "
            "    EPOR sequencing: normal; "
            "    VHL, EGLN1, EPAS1, HBB, HBA1, BPGM sequencing: all normal; "
            "    EPO gene sequencing/CNV: identifies variant; "
            "  PARAGANGLIOMA: rare but possible if HIF pathway secondarily activated by high EPO; "
            "  TREATMENT: phlebotomy; aspirin; low-dose hydroxycarbamide for high Hct"
        ),
        "disease_pathway": (
            "EPO GENE REGULATION AND CE5 MECHANISM: "
            "NORMAL EPO GENE REGULATION: "
            "  EPO gene: 7q22.3; enhancers in 3' flanking region (required for hypoxic induction); "
            "  HIF2α-ARNT heterodimer binds hypoxia response element (HRE) in EPO 3' enhancer; "
            "  Additional: HNF-4α (hepatic); retinoic acid receptor (hepatic); "
            "  Negative regulation: "
            "    GATA transcription factors: GATA4/6 suppress hepatic EPO; "
            "    VHL-independent: SIAH2 ubiquitin ligase promotes HIF degradation; "
            "  Normal EPO: 4-30 IU/L; maintains Hb 12-16 g/dL in adults; "
            "CE5 (EPO GOF): "
            "  Promoter/enhancer GOF variant → loss of negative regulation → EPO transcription unconstrained; "
            "  OR: 3'UTR deletion → mRNA stabilisation → more EPO protein; "
            "  Serum EPO chronically elevated → EPOR activation → autonomous-like erythrocytosis; "
            "DOWNSTREAM: same as other erythrocytoses: "
            "  EPOR → JAK2-STAT5 → erythroid progenitor survival/proliferation → erythrocytosis"
        ),
        "pathognomonic": (
            "PATHOGNOMONIC / HIGHLY CHARACTERISTIC — EPO-CE5: "
            "  1. MARKEDLY ELEVATED SERUM EPO (often >100 IU/L): "
            "     Disproportionately high relative to degree of erythrocytosis; "
            "     Highest EPO in any hereditary erythrocytosis (EPO gene direct overproduction); "
            "  2. ERYTHROCYTOSIS FROM CHILDHOOD/BIRTH: "
            "     Lifelong; family history (AD); "
            "  3. NEGATIVE EPO-SECRETING TUMOUR WORKUP: "
            "     Renal ultrasound/CT + brain MRI: exclude RCC/haemangioblastoma; "
            "     These conditions give very high EPO — must exclude before diagnosing CE5; "
            "  4. EPO GENE VARIANT (GOF IN REGULATORY REGION): "
            "     Sequencing of EPO promoter/enhancer/3'UTR confirms CE5; "
            "     NOTE: standard exon-only sequencing MISSES regulatory region variants; "
            "     Whole-genome sequencing or targeted regulatory sequencing needed; "
            "  5. NEGATIVE PANEL FOR OTHER HEREDITARY ERYTHROCYTOSIS GENES: "
            "     Normal EPOR, VHL, EGLN1, EPAS1, HBB, HBA1, BPGM → test EPO gene"
        ),
        "treatment": (
            "TREATMENT — EPO-CE5: "
            "ERYTHROCYTOSIS MANAGEMENT: "
            "  VENESECTION: target Hct <0.50-0.52; "
            "    Frequency depends on Hct recovery rate; typically every 4-8 weeks; "
            "  LOW-DOSE ASPIRIN 75 mg daily: thrombosis prevention; "
            "MEDICAL SUPPRESSION OF EPO PRODUCTION: "
            "  HYDROXYCARBAMIDE (hydroxyurea): "
            "    Reduces erythroid production; "
            "    Does not lower EPO production directly; "
            "    Useful if phlebotomy frequency >4-6/year or poorly tolerated; "
            "  RUXOLITINIB: "
            "    JAK1/2 inhibitor → blocks downstream EPO-EPOR-JAK2 signalling; "
            "    Off-label; for refractory disease; "
            "BELZUTIFAN: "
            "  Could reduce EPO transcription by blocking EPAS1-ARNT (HIF2α inhibitor); "
            "  CE5 EPO driven by direct gene GOF — belzutifan may have partial effect if HIF2α involved; "
            "  Not evidence-based for CE5 specifically; "
            "MONITORING: "
            "  Annual renal imaging (EPO-secreting tumour surveillance); "
            "  Annual FBC; serum EPO; "
            "  Cardiovascular risk management; "
            "  Genetic counselling: 50% offspring risk; predictive testing for at-risk relatives"
        ),
        "seed": 2893,
    },
]

SEED_BASE = 2886

def _rng(seed):
    return random.Random(seed)

def _generate_patients(gene_entry):
    rng = _rng(gene_entry["seed"])
    gene = gene_entry["gene"]
    n = 40
    patients = []
    for i in range(n):
        age_onset = rng.randint(1, 60)
        age_current = age_onset + rng.randint(1, 25)
        sex = rng.choice(["M", "F"])
        hgb = round(rng.uniform(17.5, 23.5), 1)
        hct = round(hgb / 34.0, 2)
        epo_low = gene in ("EPOR",)
        epo_val = round(rng.uniform(1.5, 4.5) if epo_low else rng.uniform(25, 120), 1)
        patients.append({
            "id": f"{gene}-{i+1:02d}",
            "gene": gene,
            "age_onset": age_onset,
            "age_current": age_current,
            "sex": sex,
            "hgb_g_dl": hgb,
            "hct": hct,
            "epo_iu_l": epo_val,
            "epo_suppressed": epo_low,
        })
    return patients

def generate_overview():
    genes = [g["gene"] for g in ATLAS_GENES]
    total_patients = 0
    gene_rows = []
    for entry in ATLAS_GENES:
        pts = _generate_patients(entry)
        total_patients += len(pts)
        avg_onset = round(sum(p["age_onset"] for p in pts) / len(pts), 1)
        avg_hgb = round(sum(p["hgb_g_dl"] for p in pts) / len(pts), 1)
        gene_rows.append({
            "gene": entry["gene"],
            "locus": entry["locus"],
            "protein_summary": entry["protein"],
            "patients": len(pts),
            "avg_onset": avg_onset,
            "avg_hgb_g_dl": avg_hgb,
            "epo_pattern": "suppressed" if entry["gene"] == "EPOR" else "elevated",
        })
    return {
        "atlas": "Hereditary Congenital Erythrocytosis Atlas",
        "subtitle": "8-Gene Reference: EPOR-VHL-EGLN1-EPAS1-HBB-HBA1-BPGM-EPO",
        "description": (
            "Comprehensive atlas of hereditary congenital erythrocytosis / polycythemia covering "
            "all major mechanistic categories: primary (EPOR GOF), HIF pathway (VHL/EGLN1/EPAS1), "
            "high-affinity haemoglobin (HBB/HBA1), and 2,3-BPG deficiency (BPGM), plus "
            "rare EPO GOF. 320 patients (8 × 40), seeds 2886-2893."
        ),
        "total_patients": total_patients,
        "total_genes": len(genes),
        "genes": genes,
        "gene_rows": gene_rows,
        "categories": {
            "Primary (autonomous erythropoiesis)": ["EPOR"],
            "HIF pathway (secondary, EPO elevated)": ["VHL", "EGLN1", "EPAS1"],
            "High-affinity haemoglobin (secondary, EPO elevated)": ["HBB", "HBA1", "BPGM"],
            "EPO gene GOF (secondary, EPO elevated)": ["EPO"],
        },
        "key_facts": [
            "EPOR (CE1): only hereditary erythrocytosis with SUPPRESSED serum EPO — all others have elevated EPO",
            "VHL (CE2/Chuvash): biallelic p.Arg200Trp; THROMBOSIS is leading cause of death; portal vein/Budd-Chiari",
            "EGLN1 (CE3): PHD2 deficiency; paraganglioma in ~5-10% — annual metanephrines mandatory",
            "EPAS1 (CE4): HIF2α GOF; polycythaemia-paraganglioma-PAH TRIAD; belzutifan (FDA 2021) direct target",
            "HBB/HBA1: HIGH-AFFINITY HB — low p50; treatment usually NOT needed (erythrocytosis compensatory)",
            "BPGM (CE8): 2,3-BPG absent; most extreme left-shifted ODC; erythrocytosis essential for O2 delivery",
            "EPO (CE5): very rare EPO GOF; markedly elevated EPO; regulatory region variant (exon sequencing misses)",
            "ALL hereditary erythrocytoses: exclude JAK2 V617F first (PV) and secondary causes",
        ],
        "diagnostic_algorithm": (
            "Erythrocytosis workup: "
            "1. Confirm erythrocytosis (Hb >18.5 g/dL M / >16.5 g/dL F OR elevated red cell mass); "
            "2. JAK2 V617F — if positive: PV diagnosis; "
            "3. Serum EPO — if suppressed (<5 IU/L): primary → test EPOR; "
            "4. If EPO elevated: secondary; "
            "   4a. Exclude reactive causes (COPD/sleep apnoea/altitude/smoking/EPO tumour); "
            "   4b. p50 O2 measurement — if low: test HBB/HBA1/BPGM; "
            "   4c. Hereditary gene panel: VHL, EGLN1, EPAS1, EPOR, HBB, HBA1, BPGM, EPO"
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
                "term": "Erythrocytosis",
                "definition": (
                    "Elevated red cell mass; Hb >18.5 g/dL (male) or >16.5 g/dL (female); "
                    "or Hct >52% (male) or >48% (female). "
                    "Primary: autonomous erythropoiesis (EPOR GOF / JAK2 V617F PV). "
                    "Secondary: elevated EPO (hereditary HIF pathway, high-affinity Hb, reactive). "
                    "First exclude JAK2 V617F (polycythaemia vera) before hereditary workup."
                ),
            },
            {
                "term": "Serum EPO in Erythrocytosis",
                "definition": (
                    "Key discriminator: "
                    "SUPPRESSED (<5 IU/L): EPOR GOF (CE1) or JAK2 V617F PV — primary erythrocytosis; "
                    "ELEVATED (>30 IU/L): secondary (VHL/EGLN1/EPAS1/HBB/HBA1/BPGM/EPO) or reactive. "
                    "Measure BEFORE phlebotomy (phlebotomy stimulates EPO temporarily)."
                ),
            },
            {
                "term": "p50 O2 (Oxygen Dissociation Curve)",
                "definition": (
                    "Partial pressure of O2 at which 50% of haemoglobin is saturated. "
                    "Normal: ~26 mmHg. Left-shifted (LOW p50) = high affinity = O2 not released. "
                    "Right-shifted (HIGH p50) = low affinity = O2 released easily. "
                    "LOW p50 (<20 mmHg): high-affinity Hb (HBB/HBA1 variants) or BPGM deficiency."
                ),
            },
            {
                "term": "Chuvash Polycythemia (VHL p.Arg200Trp biallelic)",
                "definition": (
                    "VHL biallelic p.Arg200Trp; Chuvash/Siberian founder mutation; "
                    "causes erythrocytosis (CE2) WITHOUT VHL tumour syndrome (RCC/haemangioblastoma). "
                    "THROMBOSIS is the leading cause of death — portal vein, Budd-Chiari, PE. "
                    "Distinguish from heterozygous VHL (tumour syndrome, no erythrocytosis)."
                ),
            },
            {
                "term": "Polycythaemia Vera (JAK2 V617F) vs Hereditary Erythrocytosis",
                "definition": (
                    "PV: somatic JAK2 V617F in haematopoietic stem cells; panhyperplasia (Hb+WBC+Plt); "
                    "EPO suppressed; acquired (not hereditary). "
                    "Hereditary erythrocytosis: germline variant; usually Hb only elevated (WBC/Plt normal); "
                    "EPO suppressed (EPOR) or elevated (all others); family history. "
                    "JAK2 V617F must be excluded before diagnosing any hereditary erythrocytosis."
                ),
            },
            {
                "term": "2,3-BPG (2,3-Bisphosphoglycerate)",
                "definition": (
                    "Key allosteric effector of haemoglobin; binds β-chain central cavity; "
                    "stabilises T-state → reduces O2 affinity → right-shifts ODC → promotes O2 release. "
                    "BPGM deficiency: 2,3-BPG absent → Hb functions like HbF (very high O2 affinity) → "
                    "compensatory erythrocytosis. "
                    "Measure: enzymatic 2,3-BPG assay in erythrocytes (normal ~5 mmol/L)."
                ),
            },
            {
                "term": "High-Affinity Haemoglobin Variants",
                "definition": (
                    "Missense variants in HBB or HBA1/2 destabilising T-state or stabilising R-state → "
                    "left-shifted ODC → O2 not released at tissues → compensatory erythrocytosis. "
                    "KEY POINT: erythrocytosis is compensatory — do NOT aggressively venesect to 'normal' Hb. "
                    "Key examples: Hb Chesapeake (α92R>L), Hb Hiroshima (β146H>D), Hb Rainier (β145Y>C)."
                ),
            },
            {
                "term": "Belzutifan (Welireg, HIF2α inhibitor)",
                "definition": (
                    "First-in-class HIF2α (EPAS1) inhibitor; FDA Aug 2021. "
                    "Binds PAS-B pocket of EPAS1 → disrupts HIF2α-ARNT dimerisation → "
                    "HIF2α cannot transactivate EPO/VEGF genes. "
                    "Approved for VHL-related neoplasms (RCC, CNS haemangioblastoma, pNET). "
                    "Used off-label for EPAS1 GOF erythrocytosis (CE4). "
                    "Embryofoetal toxicity: contraception mandatory."
                ),
            },
            {
                "term": "HIF Pathway in Erythrocytosis",
                "definition": (
                    "Normoxia: PHD2 (EGLN1) hydroxylates HIF1α/2α → VHL ubiquitinates → proteasomal degradation. "
                    "VHL LOF: HIF not degraded → HIF stable → EPO↑ → erythrocytosis. "
                    "EGLN1 LOF: PHD2 absent → HIF not hydroxylated → VHL cannot bind → HIF stable → EPO↑. "
                    "EPAS1 GOF: HIF2α not hydroxylatable → VHL cannot bind → constitutive EPO/VEGF↑."
                ),
            },
            {
                "term": "Paraganglioma in Hereditary Erythrocytosis",
                "definition": (
                    "EGLN1 (CE3) and EPAS1 (CE4) erythrocytoses carry paraganglioma risk. "
                    "Mechanism: HIF pathway dysregulation mimics SDH-deficiency HIF-stabilisation. "
                    "Screen: annual plasma free metanephrines + periodic 68Ga-DOTATATE PET. "
                    "EPAS1 GOF has HIGHEST paraganglioma risk (>50% in some series)."
                ),
            },
            {
                "term": "Polycythemia Vera (PV) Red Flags",
                "definition": (
                    "Features favoring PV over hereditary erythrocytosis: "
                    "Elevated WBC or platelets; JAK2 V617F positive; EPO suppressed (<5 IU/L); "
                    "acquired (no family history); BM: panmyelosis + megakaryocyte atypia; "
                    "itching after hot shower (aquagenic pruritus); splenomegaly. "
                    "PV: treat with phlebotomy + cytoreduction (hydroxycarbamide/ruxolitinib) + aspirin."
                ),
            },
        ]
    }

if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(generate_overview(), indent=2)[:2000])
    print("\n=== BREAKDOWN gene 1 ===")
    bd = generate_breakdown()
    print(json.dumps(bd["genes"][0], indent=2)[:1000])
    print("\n=== DEFINITIONS first 2 ===")
    defs = generate_definitions()
    print(json.dumps(defs["definitions"][:2], indent=2))
