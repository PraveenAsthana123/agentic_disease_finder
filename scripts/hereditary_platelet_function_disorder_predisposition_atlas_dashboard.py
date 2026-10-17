#!/usr/bin/env python3
"""Hereditary-Platelet-Function-Disorder-Predisposition-Atlas -- Complete 8-Gene Reference
Inherited platelet function / bleeding disorder germline predisposition.

ITGA2B   (GPIIb; 1039aa; 17q21.31; AR LOF;
          Glanzmann Thrombasthenia Type I (OMIM 273800);
          αIIbβ3 integrin pore subunit; ALL agonist aggregation ABSENT;
          rFVIIa/NovoSeven hemostasis; HLA-HPA alloimmunisation;
          eptifibatide/tirofiban CONTRAINDICATED ABSOLUTELY;
          seed SEED_BASE+0).
ITGB3    (GPIIIa; 788aa; 17q21.32; AR LOF;
          Glanzmann Thrombasthenia Type II / NAITP (OMIM 273800);
          HPA-1a alloantigen site; NAITP 1:1000;
          eptifibatide/tirofiban CONTRAINDICATED ABSOLUTELY;
          washed maternal platelets for NAITP;
          seed SEED_BASE+1).
GP1BA    (GPIbα; 626aa; 17p13.2; AR LOF;
          Bernard-Soulier Syndrome Type A1 (OMIM 231200);
          giant platelets 10-15fL PATHOGNOMONIC; vWF binding ABSENT;
          DDAVP INEFFECTIVE; rFVIIa first line;
          seed SEED_BASE+2).
GP1BB    (GPIbβ; 206aa; 22q11.21; AR LOF;
          Bernard-Soulier 22q / DiGeorge Overlap (OMIM 231200);
          TBX1 co-deletion; cardiac 75pct PATHOGNOMONIC;
          seed SEED_BASE+3).
NBEAL2   (NBEAL2; 2754aa; 3p21.31; AR LOF;
          Gray Platelet Syndrome (OMIM 139090);
          alpha-granule ABSENT PATHOGNOMONIC; lactoferrin absent;
          myelofibrosis 35pct progressive;
          seed SEED_BASE+4).
MYH9     (MYH9; 1960aa; 22q12.3; AD LOF;
          MYH9-Related Disease May-Hegglin/Sebastian/Fechtner/Epstein (OMIM 155100);
          giant platelets + Dohle bodies PATHOGNOMONIC; deafness 60pct;
          ACEi renal protection MANDATORY nephritis 15pct;
          seed SEED_BASE+5).
HPS1     (HPS1; 700aa; 10q24.2; AR LOF;
          Hermansky-Pudlak Syndrome Type 1 (OMIM 203300);
          delta-granule storage pool deficiency; OCA PATHOGNOMONIC;
          pulmonary fibrosis 30-50pct DOMINANT CAUSE OF DEATH;
          pirfenidone HPS-IPF;
          seed SEED_BASE+6).
P2RY12   (P2Y12; 342aa; 3q25.1; AR LOF;
          P2Y12 Deficiency (OMIM 609821);
          clopidogrel/prasugrel/ticagrelor CONTRAINDICATED ABSOLUTELY;
          ADP-induced aggregation ABSENT; aspirin still functional COX-1 intact;
          seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 3670-3677)
"""
import random

SEED_BASE = 3670

ATLAS_GENES = [
    {
        "gene": "ITGA2B",
        "protein": (
            "ITGA2B -- 17q21.31 AR LOF -- 1039aa -- "
            "GPIIb-120kDa-Integrin-Alpha-IIb-"
            "Heavy-chain-777aa-Light-chain-262aa-"
            "GT-Type-I-ALL-Agonist-Aggregation-ABSENT-"
            "ITGA2B-OMIM-273800"
        ),
        "locus": "17q21.31",
        "protein_size": (
            "1039 aa / ~120 kDa / 17q21.31 ITGA2B platelet function molecular context: "
            "STRUCTURE: "
            "  1039 aa / ~120 kDa; integrin alpha-IIb (GPIIb); αIIbβ3 (GPIIb-IIIa) pore subunit; "
            "  Heavy chain (alpha-IIb-H, aa 1-859): 7-bladed beta-propeller domain (aa 1-452) -- ligand binding; thigh domain; calf-1/calf-2; "
            "  Light chain (alpha-IIb-L, aa 860-1039): transmembrane helix; short cytoplasmic tail (Tyr984, Glu989 critical); "
            "  Furin cleavage site (Arg860-Ala861): processed in Golgi to heavy+light disulfide linked; "
            "  Propeller W3-4-5-6 blades: fibrinogen RGD/KQAGDV binding groove; "
            "  Ca2+-binding EF-hand motifs in propeller (4 sites): metal-dependent conformational activation; "
            "  ITGA2B + ITGB3 heterodimer: obligate pairing; ITGA2B LOF -> unstable complex -> both subunits absent from surface; "
            "GT TYPE I CONTEXT: "
            "  ITGA2B biallelic LOF: <5% residual αIIbβ3 surface expression -> Glanzmann Thrombasthenia Type I; "
            "  Type II: 5-20% residual surface expression (milder by flow, same bleeding phenotype in practice); "
            "  Type III (variant): normal surface expression but dysfunctional; "
            "  ALL platelet agonist aggregation ABSENT: ADP, collagen, arachidonic acid, epinephrine, thrombin -- ALL completely flat; "
            "  Clot retraction ABSENT: αIIbβ3 essential for fibrinogen binding and clot retraction; "
            "  Platelet count NORMAL; platelet morphology NORMAL (NOT giant -- distinguishes from BSS); "
            "CLINICAL RULES FOR ITGA2B-GT: "
            "  rFVIIa (NovoSeven) FIRST LINE for major bleeding -- bypasses platelet aggregation via TF-FVIIa pathway; "
            "  Platelet transfusion: last resort (HLA/HPA alloimmunisation risk -- renders future transfusions refractory); "
            "  HLA-matched platelets: use from FIRST transfusion to minimise alloimmunisation; "
            "  Eptifibatide (Integrilin) CONTRAINDICATED ABSOLUTELY: αIIbβ3 antagonist -- exacerbates GT; "
            "  Tirofiban (Aggrastat) CONTRAINDICATED ABSOLUTELY: same target; "
            "  Tranexamic acid: use routinely for mucocutaneous bleeding (antifibrinolytic adjunct); "
            "  DDAVP: LIMITED USE -- vWF released but αIIbβ3 absent so no aggregation; can give for mild epistaxis but not for surgical hemostasis; "
            "  HSCT: curative but reserved for severe refractory GT with alloimmunisation; "
            "KEY VARIANTS: "
            "  Propeller W3: p.Leu117Trp (Lebanese founder); p.Arg214Gln (Arab founder); "
            "  Exon 28: splicing mutations common; large deletions 30%; "
            "  Lebanese/Iraqi Jewish founder effect: p.Leu117Trp ITGA2B and p.Arg214Gln; "
            "SURVEILLANCE: "
            "  Annual platelet function tests (light transmission aggregometry -- LTA); "
            "  HLA/HPA typing at diagnosis (before any transfusion); "
            "  Alloantibody screen after each transfusion; "
            "  Women: plan delivery with haematology -- rFVIIa standby; "
            "TARGETED Rx: "
            "  rFVIIa 90-120 mcg/kg q2h until haemostasis; "
            "  Tranexamic acid 15-25 mg/kg TID (antifibrinolytic); "
            "  HLA-matched apheresis platelets if transfusion needed; "
            "  HSCT curative for refractory alloimmunised GT"
        ),
        "syndrome": "Glanzmann Thrombasthenia Type I/II -- ITGA2B",
        "inheritance": "AR LOF biallelic",
        "bleeding_risk": "Severe mucocutaneous bleeding; ALL agonist aggregation ABSENT; clot retraction ABSENT",
        "pathognomonic": "ALL-platelet-agonist-aggregation-ABSENT-ADP-collagen-AA-epinephrine-thrombin; clot-retraction-ABSENT; platelet-count-NORMAL",
        "key_rule": "rFVIIa-FIRST-LINE; eptifibatide-tirofiban-CONTRAINDICATED-ABSOLUTELY; HLA-matched-platelets-from-FIRST-transfusion; tranexamic-acid-routinely",
        "key_avoid": "Eptifibatide/tirofiban ABSOLUTELY CI; unmatched platelets cause alloimmunisation; NSAIDs aggravate",
        "surveillance": "Annual LTA; HLA/HPA typing at diagnosis; alloantibody screen post-transfusion; women: obstetric plan",
        "targeted_rx": "rFVIIa 90-120 mcg/kg; tranexamic acid; HLA-matched platelets; HSCT curative refractory",
    },
    {
        "gene": "ITGB3",
        "protein": (
            "ITGB3 -- 17q21.32 AR LOF -- 788aa -- "
            "GPIIIa-87kDa-Integrin-Beta-3-"
            "PSI-EGF-HybriD-domain-"
            "GT-Type-II-HPA-1a-NAITP-1in1000-"
            "ITGB3-OMIM-273800"
        ),
        "locus": "17q21.32",
        "protein_size": (
            "788 aa / ~87 kDa / 17q21.32 ITGB3 platelet function molecular context: "
            "STRUCTURE: "
            "  788 aa / ~87 kDa; integrin beta-3 (GPIIIa); αIIbβ3 and αVβ3 shared subunit; "
            "  PSI domain (aa 1-56): plexin-semaphorin-integrin fold; disulfide-rich; "
            "  Hybrid domain (aa 57-194): clasped inactive / unclasped active interface; "
            "  I-EGF domains (I-EGF 1-4, aa 452-580): stalk; long-range conformational signal; "
            "  beta-TD domain (aa 581-692): tail domain; "
            "  Transmembrane helix + cytoplasmic tail (aa 693-788): MIDAS metal ion; "
            "  RGD ligand binding site (ITGA2B β-propeller + ITGB3 PSI/hybrid interface); "
            "  HPA-1a alloantigen: Leu33 (HPA-1a) vs Pro33 (HPA-1b) polymorphism in PSI domain (rs5918); "
            "  ITGB3 LOF -> no αIIbβ3 at platelet surface; "
            "GT TYPE II / NAITP CONTEXT: "
            "  ITGB3 biallelic LOF: Glanzmann Thrombasthenia Type II (indistinguishable clinically from ITGA2B-GT); "
            "  HPA-1a (Leu33Pro in ITGB3) NAITP: maternal anti-HPA-1a IgG crosses placenta -> fetal platelet destruction; "
            "  NAITP incidence: 1:1000 live births; intracranial haemorrhage 20-25% of severe cases (most common cause of neonatal ICH from alloimmune cause); "
            "  HPA-1a negative mothers: screen in pregnancy if prior affected child; "
            "  Anti-HPA-1a: most common (85% NAITP); HPA-5b (anti-Br-a) second commonest; "
            "CLINICAL RULES FOR ITGB3-GT/NAITP: "
            "  GT management: identical to ITGA2B-GT (rFVIIa first line, eptifibatide/tirofiban CI ABSOLUTELY); "
            "  NAITP antenatal: maternal IVIG 1g/kg/week from 20-28 weeks (reduces fetal platelet destruction); "
            "  NAITP delivery: platelet count <50 -> C-section or IUT (intrauterine transfusion); washed maternal platelets for neonate; "
            "  Eptifibatide CONTRAINDICATED ABSOLUTELY: binds GPIIb-IIIa RGD site -- amplifies GT defect; "
            "  Paternal HPA-1a typing: mandatory if maternal HPA-1a negative + thrombocytopenic fetus; "
            "KEY VARIANTS: "
            "  Leu33Pro (rs5918): HPA-1a/1b polymorphism -- not LOF but alloimmunogenic; "
            "  p.Arg214Trp, p.Cys374Tyr: biallelic GT Type II ITGB3 variants; "
            "  Romiplostim/eltrombopag: ineffective for GT (TPO-pathway intact but αIIbβ3 absent); "
            "SURVEILLANCE: "
            "  GT: Annual LTA + HLA/HPA typing; "
            "  NAITP: maternal HPA typing + anti-HPA antibody in prior-affected pregnancies; "
            "  Neonatal: cranial ultrasound at birth if thrombocytopenic; "
            "TARGETED Rx: "
            "  rFVIIa first line major bleeding; tranexamic acid; "
            "  NAITP: maternal IVIG antenatal + washed maternal/HPA-compatible platelets neonatal"
        ),
        "syndrome": "Glanzmann Thrombasthenia Type II / NAITP -- ITGB3",
        "inheritance": "AR LOF biallelic",
        "bleeding_risk": "Severe GT phenotype; NAITP neonatal ICH 20-25%; HPA-1a alloantigen",
        "pathognomonic": "ALL-agonist-aggregation-ABSENT; HPA-1a-NAITP-neonatal-ICH; eptifibatide-target-CONTRAINDICATED-ABSOLUTELY",
        "key_rule": "rFVIIa-FIRST-LINE; eptifibatide-tirofiban-CONTRAINDICATED-ABSOLUTELY; NAITP-maternal-IVIG-antenatal; washed-maternal-platelets-neonatal",
        "key_avoid": "Eptifibatide/tirofiban CI absolutely; unmatched platelets; no TPO-agonists (ineffective)",
        "surveillance": "Annual LTA; NAITP: maternal HPA typing + antibodies; neonatal cranial US if thrombocytopenic",
        "targeted_rx": "rFVIIa; tranexamic acid; NAITP: maternal IVIG + HPA-compatible neonatal platelets",
    },
    {
        "gene": "GP1BA",
        "protein": (
            "GP1BA -- 17p13.2 AR LOF -- 626aa -- "
            "GPIba-73kDa-Platelet-Glycoprotein-Ib-Alpha-"
            "N-terminal-LRR-macroglycopeptide-"
            "BSS-Type-A1-Giant-Platelets-PATHOGNOMONIC-vWF-ABSENT-DDAVP-INEFFECTIVE-"
            "GP1BA-OMIM-231200"
        ),
        "locus": "17p13.2",
        "protein_size": (
            "626 aa / ~73 kDa / 17p13.2 GP1BA platelet function molecular context: "
            "STRUCTURE: "
            "  626 aa / ~73 kDa; glycoprotein Ib alpha (GPIbα); GPIb-IX-V complex core component; "
            "  N-terminal domain (aa 1-283): leucine-rich repeat (LRR) x8 -- vWF A1 domain binding; "
            "  Macroglycopeptide (aa 284-382): heavily O-glycosylated anionic sialomucin region (50% GPIbα mass); "
            "  Transmembrane helix (aa 485-506); "
            "  Cytoplasmic tail (aa 507-626): 14-3-3zeta binding (Ser587 phosphorylation); filamin A interaction; "
            "  vWF binding: GPIbα N-terminal LRR cleft binds vWF A1 domain under high shear (>1000 s-1); "
            "  Thrombin binding: LRR N-terminal also binds thrombin (secondary platelet activation); "
            "  GPIb-IX-V complex: GPIba + GPIbb (GP1BB) + GPIX (GP9) + GPV (GP5); must coexpress or complex instability -> absent; "
            "BSS TYPE A1 CONTEXT: "
            "  GP1BA biallelic LOF: Bernard-Soulier Syndrome Type A1 (most common BSS subtype); "
            "  Giant platelets 10-15fL MPV PATHOGNOMONIC (normal 7-11fL); macrothrombocytopenia; "
            "  vWF:RCo (ristocetin cofactor): ABSENT -- the DEFINING test; "
            "  RIPA (ristocetin-induced platelet aggregation): ABSENT (low dose 0.6 mg/mL AND high dose 1.2 mg/mL); "
            "  Distinguishes from vWD Type 2B: vWD2B has INCREASED RIPA at LOW DOSE (opposite); "
            "  Thrombin response: reduced (GPIbα is secondary thrombin receptor); ADP/collagen responses preserved; "
            "CLINICAL RULES FOR GP1BA-BSS: "
            "  DDAVP INEFFECTIVE in BSS: releases vWF from endothelium but GPIbα absent so vWF cannot bind platelets; "
            "  rFVIIa (NovoSeven) FIRST LINE for major bleeding -- bypasses GPIb-dependent platelet adhesion; "
            "  Platelet transfusion: for severe bleeds; alloimmunisation risk (anti-HPA-3 from GP1BA region); "
            "  Tranexamic acid: effective adjunct for mucocutaneous bleeding; "
            "  DDAVP: DO NOT USE as primary haemostatic agent in BSS -- misleads clinicians; "
            "KEY VARIANTS: "
            "  European founder: p.Ala172Val (LRR6 -- disrupts vWF binding groove); "
            "  Japanese founder: c.515delG (frameshift); "
            "  Puerto Rican: p.Leu57Phe (LRR3); "
            "  GPIb-IX complex assembly variants: GP1BB co-mutations can cause compound BSS; "
            "SURVEILLANCE: "
            "  Annual platelet count + MPV; LTA annually (check ADP/collagen preserved, ristocetin ABSENT); "
            "  vWF:RCo annually; dental/surgical planning with haematology; "
            "  Women: obstetric plan -- rFVIIa standby at delivery; "
            "TARGETED Rx: "
            "  rFVIIa 90-120 mcg/kg q2h until haemostasis; "
            "  Tranexamic acid routinely; "
            "  Avoid DDAVP as primary haemostatic agent; "
            "  HLA-matched platelets if transfusion needed; "
            "  HSCT curative for severe refractory BSS"
        ),
        "syndrome": "Bernard-Soulier Syndrome Type A1 -- GP1BA",
        "inheritance": "AR LOF biallelic",
        "bleeding_risk": "Severe mucocutaneous bleeding; giant platelets 10-15fL PATHOGNOMONIC; vWF:RCo ABSENT",
        "pathognomonic": "Giant-platelets-10-15fL-PATHOGNOMONIC; vWF-RCo-ABSENT; RIPA-ABSENT-both-doses; DDAVP-INEFFECTIVE",
        "key_rule": "DDAVP-INEFFECTIVE-NO-GPIba; rFVIIa-FIRST-LINE; ristocetin-ABSENT-distinguishes-from-vWD2B-opposite; tranexamic-acid-routinely",
        "key_avoid": "DDAVP as primary haemostatic (misleading); NSAIDs; aspirin",
        "surveillance": "Annual PLT count + MPV + LTA + vWF:RCo; obstetric plan women; dental pre-plan",
        "targeted_rx": "rFVIIa 90-120 mcg/kg; tranexamic acid; HLA-matched platelets; HSCT curative refractory",
    },
    {
        "gene": "GP1BB",
        "protein": (
            "GP1BB -- 22q11.21 AR LOF -- 206aa -- "
            "GPIbb-22kDa-Platelet-Glycoprotein-Ib-Beta-"
            "LRR-transmembrane-"
            "BSS-22q-DiGeorge-Overlap-TBX1-Cardiac-75pct-PATHOGNOMONIC-"
            "GP1BB-OMIM-231200"
        ),
        "locus": "22q11.21",
        "protein_size": (
            "206 aa / ~22 kDa / 22q11.21 GP1BB platelet function molecular context: "
            "STRUCTURE: "
            "  206 aa / ~22 kDa; glycoprotein Ib beta (GPIbβ); GPIb-IX-V complex transmembrane anchor; "
            "  Ectodomain (aa 1-122): single LRR domain; stabilises GPIbα-GPIX interactions; "
            "  Transmembrane helix (aa 150-175): Gly148-Val160 dileucine motif for complex assembly; "
            "  Cytoplasmic tail (aa 176-206): Ser166 (14-3-3zeta binding when phosphorylated); "
            "  GP1BB LOF -> destabilises entire GPIb-IX-V complex -> GPIbα, GPIX, GPV all absent from platelet surface; "
            "BSS 22q / DIGEORGE OVERLAP CONTEXT: "
            "  22q11.21 location: within the DiGeorge/22q11.2 deletion syndrome critical region (TBX1 gene nearby); "
            "  22q11.2 deletion (3Mb) co-deletes TBX1 (cardiac) + GP1BB (BSS); "
            "  Compound phenotype: BSS bleeding + DiGeorge features (cardiac 75%, palatal 70%, T-cell lymphopenia); "
            "  TBX1 co-deletion: cardiac anomalies 75% (conotruncal: truncus arteriosus, tetralogy of Fallot, IAA type B); "
            "  T-cell deficiency: mandatory CD4 count from birth; live vaccine CI until immune competence verified; "
            "  Hypocalcaemia: parathyroid hypoplasia; check calcium neonatal; "
            "  GP1BB point mutations: BSS only (no DiGeorge features) -- locus-specific vs. deletion; "
            "CLINICAL RULES FOR GP1BB-BSS: "
            "  22q11.2 deletion BSS: CARDIAC ANOMALY SCREEN MANDATORY -- cardiac surgery before platelet management planning; "
            "  T-cell assessment mandatory before any live vaccines (MMR, varicella, rotavirus CI until verified immune); "
            "  DDAVP INEFFECTIVE (same as GP1BA-BSS -- GPIbβ absent -> whole complex absent); "
            "  rFVIIa FIRST LINE same as GP1BA; "
            "  Hypocalcaemia monitoring: especially neonatal and post-cardiac surgery; "
            "KEY VARIANTS: "
            "  22q11.2 deletion (del22q11.2): detected by FISH/MLPA; standard sequencing MISSES deletion; "
            "  GP1BB point mutations: p.Ala156Val (transmembrane -- complex assembly); p.Ser166Pro (cytoplasmic); "
            "SURVEILLANCE: "
            "  22q11.2 BSS: annual cardiac follow-up + T-cell count + calcium + PLT count + MPV; "
            "  Point mutation BSS: annual PLT count + MPV + LTA + vWF:RCo; "
            "  MLPA/FISH for 22q11.2 deletion mandatory in GP1BB BSS before calling point mutation; "
            "TARGETED Rx: "
            "  Cardiac anomaly: repair before elective surgery; "
            "  IVIG for T-cell lymphopenia if IgG < 5; "
            "  BSS bleeding: rFVIIa + tranexamic acid; "
            "  Live vaccines: DEFER until T-cell competence confirmed"
        ),
        "syndrome": "Bernard-Soulier Syndrome 22q / DiGeorge Overlap -- GP1BB",
        "inheritance": "AR LOF (point mutation) or del22q11.2 (co-deletion with TBX1)",
        "bleeding_risk": "BSS severe bleeding + cardiac 75pct PATHOGNOMONIC + T-cell lymphopenia; giant platelets",
        "pathognomonic": "22q11.2-deletion-BSS-cardiac-75pct-PATHOGNOMONIC; TBX1-co-deletion; MLPA-MANDATORY-standard-sequencing-MISSES",
        "key_rule": "CARDIAC-ANOMALY-SCREEN-MANDATORY; T-cell-count-from-birth; live-vaccines-CI-until-T-cell-verified; DDAVP-INEFFECTIVE; rFVIIa-first-line",
        "key_avoid": "Live vaccines until T-cell verified; DDAVP; do not miss 22q11.2 deletion with FISH/MLPA",
        "surveillance": "Annual cardiac + T-cell + calcium + PLT + MPV + LTA; MLPA/FISH 22q11.2 at diagnosis",
        "targeted_rx": "rFVIIa + tranexamic acid bleeding; cardiac repair; IVIG T-cell lymphopenia; defer live vaccines",
    },
    {
        "gene": "NBEAL2",
        "protein": (
            "NBEAL2 -- 3p21.31 AR LOF -- 2754aa -- "
            "NBEAL2-313kDa-Neurobeachin-Like-2-"
            "BEACH-WD40-HEAT-ARM-domains-"
            "GPS-Alpha-Granule-ABSENT-PATHOGNOMONIC-Myelofibrosis-35pct-"
            "NBEAL2-OMIM-139090"
        ),
        "locus": "3p21.31",
        "protein_size": (
            "2754 aa / ~313 kDa / 3p21.31 NBEAL2 platelet function molecular context: "
            "STRUCTURE: "
            "  2754 aa / ~313 kDa; neurobeachin-like 2; BEACH (Beige and Chediak-Higashi) domain family; "
            "  N-terminal ARM repeats (aa 1-600): armadillo protein-protein interaction scaffold; "
            "  HEAT repeats (aa 601-1200): protein-protein interaction; "
            "  PH domain (aa 1201-1350): phosphoinositide binding; membrane targeting; "
            "  WD40 domain (aa 1800-2100): beta-propeller protein-protein interaction; "
            "  BEACH domain (aa 2200-2400): LYST/Beige homology -- vesicle trafficking regulation; "
            "  NBEAL2 LOF -> defective megakaryocyte alpha-granule biogenesis -> absent alpha-granules in platelets; "
            "  Alpha-granule contents absent: PF4 (CXCL4), betaTG (CXCL7), vWF, fibrinogen, P-selectin, thrombospondin, Factor V; "
            "GRAY PLATELET SYNDROME CONTEXT: "
            "  NBEAL2 biallelic LOF: Gray Platelet Syndrome (GPS) -- named for grey appearance on Wright-Giemsa stain; "
            "  Alpha-granule ABSENT PATHOGNOMONIC: flow cytometry CD63+/P-selectin- pattern; lactoferrin ABSENT; PF4 ABSENT; "
            "  Lactoferrin absent: highly specific for GPS; distinguishes from other platelet granule disorders; "
            "  Platelets appear grey/hypogranular on peripheral blood smear PATHOGNOMONIC; "
            "  Myelofibrosis 35% progressive: alpha-granule contents (PDGF, TGF-beta) released into BM stroma -> fibrosis; "
            "  Splenomegaly EXPECTED: extramedullary haematopoiesis from BM fibrosis; "
            "  Moderate thrombocytopenia (50-100 x10^9/L typical); "
            "  Autoimmune component: anti-platelet antibodies in 20-30% GPS; "
            "CLINICAL RULES FOR NBEAL2-GPS: "
            "  Grey platelet on smear + elevated MPV + moderate thrombocytopenia -> GPS diagnostic; "
            "  PF4 flow cytometry: absent (vs present in most other giant platelet disorders); "
            "  Annual BM biopsy from age 30: progressive myelofibrosis surveillance; "
            "  Splenectomy: increases platelet count transiently but accelerates BM fibrosis -- AVOID unless critical thrombocytopenia; "
            "  Ruxolitinib: JAK1/2 inhibitor for progressive myelofibrosis in GPS; "
            "  HSCT: curative for BM fibrosis; consider at fibrosis grade 2-3 before irreversible BM failure; "
            "KEY VARIANTS: "
            "  GPS originally described as autosomal dominant (some heterozygous families) but most cases AR biallelic NBEAL2; "
            "  c.1875+1G>A (splice donor exon 11): common Northern European allele; "
            "  Large deletions: MLPA recommended if point mutation not found; "
            "SURVEILLANCE: "
            "  6-monthly CBC + smear (grey platelets); annual PF4 flow; annual BM biopsy from age 30; "
            "  Splenomegaly ultrasound annually; anti-platelet antibody; "
            "TARGETED Rx: "
            "  Platelet transfusion for major bleeding; tranexamic acid; "
            "  Ruxolitinib for progressive myelofibrosis; "
            "  HSCT curative for BM fibrosis; "
            "  Avoid splenectomy (accelerates fibrosis)"
        ),
        "syndrome": "Gray Platelet Syndrome -- NBEAL2",
        "inheritance": "AR LOF biallelic",
        "bleeding_risk": "Moderate bleeding; grey platelets PATHOGNOMONIC; myelofibrosis 35pct progressive; splenomegaly",
        "pathognomonic": "Grey-hypogranular-platelets-smear-PATHOGNOMONIC; alpha-granule-ABSENT; PF4-lactoferrin-ABSENT; myelofibrosis-35pct-progressive",
        "key_rule": "Annual-BM-biopsy-from-age-30; splenectomy-AVOID-accelerates-fibrosis; ruxolitinib-myelofibrosis; HSCT-curative-BM-failure",
        "key_avoid": "Splenectomy (accelerates BM fibrosis); do not miss progressive myelofibrosis",
        "surveillance": "6-monthly CBC+smear; annual PF4 flow + BM biopsy from age 30; splenomegaly USS annually",
        "targeted_rx": "Platelet transfusion major bleeding; tranexamic acid; ruxolitinib myelofibrosis; HSCT curative",
    },
    {
        "gene": "MYH9",
        "protein": (
            "MYH9 -- 22q12.3 AD LOF -- 1960aa -- "
            "MYH9-224kDa-Non-Muscle-Myosin-IIA-Heavy-Chain-"
            "Motor-domain-Coiled-coil-Tail-"
            "MYH9-RD-May-Hegglin-Giant-Platelets-DohleBody-PATHOGNOMONIC-Deafness-60pct-Nephritis-15pct-"
            "MYH9-OMIM-155100"
        ),
        "locus": "22q12.3",
        "protein_size": (
            "1960 aa / ~224 kDa / 22q12.3 MYH9 platelet function molecular context: "
            "STRUCTURE: "
            "  1960 aa / ~224 kDa; non-muscle myosin heavy chain IIA (NMHC-IIA); cytoskeletal motor protein; "
            "  Motor domain (aa 1-780): ATP hydrolysis; actin binding; cross-bridge cycling; "
            "  Lever arm (aa 781-860): light chain binding (regulatory + essential); "
            "  Coiled-coil rod (aa 861-1930): anti-parallel dimer; filament assembly; "
            "  Non-helical tailpiece (aa 1931-1960): filament regulation; "
            "  MYH9 in megakaryocytes: cytokinesis of proplatelets; MYH9 LOF -> failed proplatelet cleavage -> giant platelets; "
            "  MYH9 in neutrophils: cytoplasmic inclusions (Dohle bodies of aggregated NMHC-IIA) PATHOGNOMONIC; "
            "  MYH9 in renal podocytes: LOF -> focal segmental glomerulosclerosis (FSGS); "
            "  MYH9 in cochlear hair cells: LOF -> sensorineural hearing loss; "
            "MYH9-RELATED DISEASE CONTEXT: "
            "  MYH9 AD LOF (heterozygous): MYH9-Related Disease (MYH9-RD); four historical eponyms (single gene): "
            "    May-Hegglin Anomaly: giant platelets + Dohle bodies, mild thrombocytopenia, mild bleeding; "
            "    Sebastian Syndrome: giant platelets + Dohle bodies + deafness; "
            "    Fechtner Syndrome: giant platelets + Dohle bodies + deafness + cataracts + nephritis; "
            "    Epstein Syndrome: giant platelets + deafness + nephritis (NO Dohle bodies); "
            "  Giant platelets (MPV 12-20fL) + thrombocytopenia (25-130 x10^9/L); "
            "  BLEEDING IS MILD despite giant platelets -- bleeding time/PFA-100 may be near-normal; "
            "  Dohle-like inclusions in neutrophils PATHOGNOMONIC: MYH9 protein aggregates on immunofluorescence (anti-NMHC-IIA staining); "
            "  Sensorineural deafness 60%: progressive bilateral high-frequency loss from age 20-40; "
            "  Cataracts 20%: anterior cortical; "
            "  Nephritis/FSGS 15%: slowly progressive; CKD risk; "
            "CLINICAL RULES FOR MYH9-RD: "
            "  ACEi/ARB MANDATORY from nephritis diagnosis: FSGS -- ACEi slows CKD progression; "
            "  Audiometry annually from age 10: early detection of progressive SNHL; "
            "  Cochlear implant: effective for profound SNHL in MYH9-RD; "
            "  BLEEDING IS MILD: platelet transfusion rarely needed; DDAVP for surgical haemostasis (vWF intact); "
            "  DDAVP EFFECTIVE in MYH9-RD (unlike BSS/GT): vWF and GPIbα both intact; "
            "  Do NOT withhold surgery for low platelet count alone in MYH9-RD (functional platelets large but effective); "
            "  Avoid nephrotoxics (NSAIDs, aminoglycosides, contrast) -- FSGS risk; "
            "KEY VARIANTS: "
            "  Motor domain hotspots: p.Arg702Cys/His/Ser (most common; SNHL + nephritis risk); "
            "  Coiled-coil hotspot: p.Glu1841Lys (May-Hegglin-like, mild phenotype); "
            "  Genotype-phenotype correlation: motor domain variants -> higher risk of deafness + nephritis; tail variants -> milder; "
            "SURVEILLANCE: "
            "  Annual CBC + MPV + urine PCR (protein:creatinine ratio); "
            "  Annual audiometry from age 10; ophthalmology every 2yr; "
            "  Renal biopsy if urine PCR > 0.1 (FSGS confirmation); "
            "TARGETED Rx: "
            "  ACEi/ARB for nephritis (MANDATORY); "
            "  DDAVP for surgery (EFFECTIVE -- vWF intact); "
            "  Cochlear implant for profound SNHL; "
            "  Avoid nephrotoxics; platelet transfusion rarely needed"
        ),
        "syndrome": "MYH9-Related Disease May-Hegglin/Sebastian/Fechtner/Epstein -- MYH9",
        "inheritance": "AD LOF heterozygous",
        "bleeding_risk": "MILD bleeding despite giant platelets; morbidity is RENAL + DEAF not bleeding; FSGS 15pct",
        "pathognomonic": "Giant-platelets-MPV-12-20fL-PATHOGNOMONIC; Dohle-bodies-neutrophils-MYH9-aggregates-PATHOGNOMONIC; SNHL-60pct; FSGS-15pct",
        "key_rule": "ACEi-ARB-MANDATORY-nephritis; audiometry-annually-from-age-10; DDAVP-EFFECTIVE-unlike-BSS; BLEEDING-MILD-do-not-cancel-surgery-for-low-PLT-alone",
        "key_avoid": "NSAIDs nephrotoxic; aminoglycosides; contrast media; do not over-treat low PLT (bleeding mild)",
        "surveillance": "Annual CBC+MPV+urine-PCR; audiometry annually from age 10; ophthalmology 2yr; renal biopsy if PCR>0.1",
        "targeted_rx": "ACEi/ARB nephritis; DDAVP surgery; cochlear implant SNHL; avoid nephrotoxics",
    },
    {
        "gene": "HPS1",
        "protein": (
            "HPS1 -- 10q24.2 AR LOF -- 700aa -- "
            "HPS1-79kDa-Hermansky-Pudlak-Protein-1-"
            "BLOC-3-VPS33B-"
            "HPS-Type1-Delta-Granule-OCA-PATHOGNOMONIC-Pulmonary-Fibrosis-30-50pct-DOMINANT-CAUSE-DEATH-"
            "HPS1-OMIM-203300"
        ),
        "locus": "10q24.2",
        "protein_size": (
            "700 aa / ~79 kDa / 10q24.2 HPS1 platelet function molecular context: "
            "STRUCTURE: "
            "  700 aa / ~79 kDa; HPS1 protein; BLOC-3 (biogenesis of lysosome-related organelles complex 3) subunit; "
            "  Coiled-coil N-terminal region (aa 1-200): BLOC-3 dimerisation (with HPS4); "
            "  Central helical bundle (aa 201-500): protein-protein interactions; Rab9 GEF activity; "
            "  C-terminal region (aa 501-700): lysosome-related organelle (LRO) targeting; "
            "  BLOC-3 (HPS1+HPS4 dimer): Rab9/Rab32/Rab38 guanine nucleotide exchange factor (GEF); regulates trafficking to LROs; "
            "  LROs: melanosomes (pigment cells), dense bodies (platelets), lamellar bodies (lung type II cells), lytic granules (CTLs); "
            "  HPS1 LOF -> BLOC-3 loss -> defective LRO biogenesis -> ALL affected: melanosomes + dense bodies + lamellar bodies; "
            "HPS TYPE 1 CONTEXT: "
            "  HPS1 biallelic LOF: Hermansky-Pudlak Syndrome Type 1 (most common HPS subtype, ~50% HPS globally); "
            "  Puerto Rican founder: 16-bp duplication exon 15 (c.1472_1487dup) -- prevalence 1:1800 PR; "
            "  Oculocutaneous albinism (OCA) PATHOGNOMONIC: absent/reduced melanosomes -> hypopigmentation skin/hair/eyes; "
            "  Nystagmus + reduced visual acuity PATHOGNOMONIC: foveal hypoplasia; misrouted optic fibres; "
            "  Delta-granule (dense body) storage pool deficiency: absent dense bodies -> absent ADP/ATP/serotonin/Ca2+; "
            "  Platelet aggregation: ADP first wave normal, second wave ABSENT (no dense body secretion for amplification); "
            "  PULMONARY FIBROSIS (HPS-IPF) DOMINANT CAUSE OF DEATH: lung type II cells (lamellar bodies absent); "
            "    HPS-IPF onset typically 30-50 yr; FVC decline 1-3% per year; "
            "    Pirfenidone slows HPS-IPF progression (Level C; Phase 2 trial data); "
            "    Nintedanib: evidence in idiopathic IPF but limited HPS-specific data; "
            "  Granulomatous colitis (IBD-like): 15% HPS1; anti-TNF (infliximab) responsive; "
            "CLINICAL RULES FOR HPS1: "
            "  PULMONARY FIBROSIS = DOMINANT MORTALITY RISK -- primary clinical priority, not bleeding; "
            "  Pirfenidone: start when FVC decline confirmed or HRCT fibrosis grade 2; "
            "  Lung transplant: consider at age >40 with FVC <50% predicted; "
            "  DDAVP: partially effective (releases vWF; dense bodies absent but thrombin pathway intact); "
            "  Tranexamic acid: effective adjunct for mucocutaneous bleeding; "
            "  Platelet transfusion: avoid (normal donor platelets sufficient; patient's own are the abnormal ones); "
            "  Ophthalmology annually: OCA -> nystagmus + photophobia + reduced VA; "
            "KEY VARIANTS: "
            "  Puerto Rican: 16-bp duplication exon 15 (c.1472_1487dup16) -- founder; 1:1800 prevalence PR; "
            "  Japanese: exon 7 deletion; "
            "  Compound heterozygotes: European background; "
            "SURVEILLANCE: "
            "  Annual PFTs (FVC, DLCO) from age 20; HRCT chest every 2yr; "
            "  Annual ophthalmology; 6-monthly CBC; "
            "  Colonoscopy if colitis symptoms; "
            "TARGETED Rx: "
            "  Pirfenidone 2403 mg/day for HPS-IPF; "
            "  Lung transplant evaluation FVC < 50%; "
            "  Anti-TNF for HPS colitis; "
            "  DDAVP + tranexamic acid for mucocutaneous bleeding"
        ),
        "syndrome": "Hermansky-Pudlak Syndrome Type 1 -- HPS1",
        "inheritance": "AR LOF biallelic",
        "bleeding_risk": "Moderate mucocutaneous; dense-granule absent; DOMINANT MORTALITY: pulmonary fibrosis 30-50pct",
        "pathognomonic": "OCA-nystagmus-PATHOGNOMONIC; dense-granule-ABSENT; pulmonary-fibrosis-DOMINANT-CAUSE-DEATH; Puerto-Rican-founder-1in1800",
        "key_rule": "Pirfenidone-HPS-IPF; PFTs-annually-from-age-20; pulmonary-fibrosis-DOMINANT-mortality-not-bleeding; lung-transplant-FVC-less-50pct",
        "key_avoid": "Do not miss pulmonary fibrosis (FVC decline before symptoms); avoid lung-toxic drugs",
        "surveillance": "Annual PFTs+HRCT every 2yr; annual ophthalmology; CBC 6-monthly; colonoscopy if colitis",
        "targeted_rx": "Pirfenidone HPS-IPF; lung transplant FVC<50%; anti-TNF HPS-colitis; DDAVP+tranexamic acid bleeding",
    },
    {
        "gene": "P2RY12",
        "protein": (
            "P2RY12 -- 3q25.1 AR LOF -- 342aa -- "
            "P2Y12-40kDa-Purinergic-Receptor-P2Y12-"
            "7TM-GPCR-Gi-coupled-"
            "P2Y12-Deficiency-ADP-Aggregation-ABSENT-Clopidogrel-Prasugrel-Ticagrelor-CONTRAINDICATED-ABSOLUTELY-"
            "P2RY12-OMIM-609821"
        ),
        "locus": "3q25.1",
        "protein_size": (
            "342 aa / ~40 kDa / 3q25.1 P2RY12 platelet function molecular context: "
            "STRUCTURE: "
            "  342 aa / ~40 kDa; P2Y12 purinergic receptor; GPCR (7 transmembrane) family; "
            "  TM1-TM7 helices: 7-pass transmembrane topology; "
            "  N-terminus (extracellular, aa 1-22): glycosylation sites; "
            "  ECL2 (extracellular loop 2, aa 163-190): ADP binding groove entrance; "
            "  ICL3 (intracellular loop 3, aa 247-278): Gi protein coupling (Galphai inhibits adenylyl cyclase); "
            "  ADP binding pocket (TM3-TM5-TM6, residues Arg256, Lys280, Tyr105): ADP recognition; "
            "  P2RY12 LOF -> no ADP-mediated Gi signalling -> cAMP NOT suppressed -> platelet NOT activated by ADP; "
            "P2Y12 DEFICIENCY CONTEXT: "
            "  P2RY12 biallelic LOF: rare AR disorder; mild-moderate bleeding; ADP-aggregation absent; "
            "  CLOPIDOGREL (Plavix) MECHANISM: P2Y12 antagonist (thienopyridine prodrug, irreversible); "
            "  PRASUGREL (Effient) MECHANISM: P2Y12 antagonist (thienopyridine prodrug, irreversible); "
            "  TICAGRELOR (Brilinta) MECHANISM: direct reversible P2Y12 antagonist; "
            "  P2RY12 LOF = PHARMACOLOGICAL P2Y12 BLOCKADE STATE -- all three antiplatelet agents target exactly this receptor; "
            "  CLOPIDOGREL/PRASUGREL/TICAGRELOR CONTRAINDICATED ABSOLUTELY in P2RY12 LOF: "
            "    No pharmacological effect (target absent); will not add any benefit; "
            "    BUT: increases bleeding risk if combined with aspirin (since ONLY aspirin effect remains); "
            "    Gives FALSE SENSE OF SECURITY for ACS/PCI antiplatelet cover -- must use alternative; "
            "  ASPIRIN STILL FUNCTIONS: COX-1 TXA2 pathway intact in P2RY12 LOF; aspirin remains effective; "
            "  ACS/PCI management: use aspirin + GPIIb-IIIa inhibitor (abciximab/tirofiban) for acute ACS; "
            "    Note: eptifibatide/tirofiban are CI in GT (ITGA2B/ITGB3 LOF) but SAFE in P2RY12 LOF; "
            "  ADP-ristocetin aggregation: ADP aggregation ABSENT (primary + secondary both flat); "
            "    Collagen/arachidonic acid/thrombin aggregation: NORMAL or reduced (ADP amplification lost); "
            "CLINICAL RULES FOR P2RY12 DEFICIENCY: "
            "  ADP agonist aggregation ABSENT: diagnoses P2RY12 deficiency (confirm with pharmacogenomics or sequencing); "
            "  NEVER prescribe clopidogrel/prasugrel/ticagrelor: no antiplatelet benefit + increased bleeding risk; "
            "  For ACS/PCI: aspirin + abciximab (GPIIb-IIIa) or cangrelor IV during procedure; "
            "  Document P2RY12 LOF prominently in medical record: prevent iatrogenic thienopyridine prescription; "
            "KEY VARIANTS: "
            "  p.Arg256Gln (ADP binding pocket): founder mutation European; "
            "  p.His187Gln: ECL2 disruption; "
            "  Compound heterozygotes common; "
            "SURVEILLANCE: "
            "  Annual LTA (ADP absent confirmed); genetic testing first-degree relatives; "
            "  Medical alert: document P2RY12 LOF + CI medications; "
            "TARGETED Rx: "
            "  AVOID clopidogrel/prasugrel/ticagrelor ABSOLUTELY; "
            "  Aspirin effective (COX-1 intact); "
            "  ACS/PCI: aspirin + abciximab; DDAVP for surgical haemostasis (mild-moderate effect); "
            "  Tranexamic acid for dental/mucosal bleeding"
        ),
        "syndrome": "P2Y12 Deficiency -- P2RY12",
        "inheritance": "AR LOF biallelic",
        "bleeding_risk": "Mild-moderate mucocutaneous; ADP aggregation ABSENT; mildest platelet function disorder in this atlas",
        "pathognomonic": "ADP-aggregation-ABSENT-primary-and-secondary; clopidogrel-prasugrel-ticagrelor-CONTRAINDICATED-ABSOLUTELY; aspirin-STILL-WORKS",
        "key_rule": "NEVER-prescribe-clopidogrel-prasugrel-ticagrelor; document-P2RY12-LOF-medical-record; aspirin-COX1-INTACT; ACS-use-abciximab-instead",
        "key_avoid": "Clopidogrel/prasugrel/ticagrelor ABSOLUTELY CI; do not rely on thienopyridines for ACS cover",
        "surveillance": "Annual LTA; genetic testing relatives; medical-alert document CI medications",
        "targeted_rx": "Aspirin effective; abciximab ACS/PCI; DDAVP surgery; tranexamic acid; NEVER thienopyridines",
    },
]

KEY_CLINICAL_DISTINCTIONS = [
    "ITGA2B-vs-ITGB3: Both GT; ITGB3 adds NAITP risk (HPA-1a); eptifibatide/tirofiban CONTRAINDICATED ABSOLUTELY in BOTH",
    "GP1BA DDAVP INEFFECTIVE: releases vWF but NO GPIba to bind it; rFVIIa is correct -- DDAVP misleads",
    "GP1BA-vs-vWD2B: Both ristocetin ABSENT but vWD2B has INCREASED RIPA at LOW DOSE (OPPOSITE); RIPA dose-response distinguishes",
    "GP1BB-22q: CARDIAC ANOMALY 75pct PATHOGNOMONIC from TBX1 co-deletion; cardiac surgery before platelet management; MLPA MANDATORY",
    "NBEAL2 SPLENECTOMY AVOID: increases platelet count transiently but ACCELERATES myelofibrosis; ruxolitinib preferred",
    "MYH9 BLEEDING IS MILD: major morbidity is RENAL (FSGS 15pct) not bleeding; ACEi MANDATORY nephritis; DDAVP EFFECTIVE (unlike BSS)",
    "HPS1 DOMINANT MORTALITY: PULMONARY FIBROSIS not bleeding; pirfenidone mandatory; PFTs annually from age 20",
    "P2RY12 ANTIPLATELET CI: clopidogrel/prasugrel/ticagrelor TARGET ABSENT -- use aspirin + abciximab for ACS; document prominently",
]


def _make_cohort(gene_index: int, seed: int) -> dict:
    """Generate a deterministic 40-patient cohort for one hereditary platelet function disorder gene."""
    rng = random.Random(seed)
    gene = ATLAS_GENES[gene_index]

    # Gene-specific parameter ranges
    params = {
        "ITGA2B":  {"age_mu": 22, "age_sd": 12, "plt_mu": 165, "plt_sd": 40, "mpv_mu": 9.0, "mpv_sd": 0.8,  "bleeding_severe": 0.72, "ddavp_effective": 0.10, "rFVIIa_use": 0.68, "alloimmun_pct": 0.35},
        "ITGB3":   {"age_mu": 20, "age_sd": 11, "plt_mu": 172, "plt_sd": 38, "mpv_mu": 8.8, "mpv_sd": 0.7,  "bleeding_severe": 0.70, "ddavp_effective": 0.10, "rFVIIa_use": 0.65, "alloimmun_pct": 0.30},
        "GP1BA":   {"age_mu": 18, "age_sd": 10, "plt_mu": 72,  "plt_sd": 28, "mpv_mu": 12.5,"mpv_sd": 1.5,  "bleeding_severe": 0.75, "ddavp_effective": 0.05, "rFVIIa_use": 0.74, "alloimmun_pct": 0.25},
        "GP1BB":   {"age_mu": 16, "age_sd": 9,  "plt_mu": 68,  "plt_sd": 25, "mpv_mu": 13.0,"mpv_sd": 1.8,  "bleeding_severe": 0.70, "ddavp_effective": 0.05, "rFVIIa_use": 0.70, "alloimmun_pct": 0.20},
        "NBEAL2":  {"age_mu": 30, "age_sd": 14, "plt_mu": 78,  "plt_sd": 30, "mpv_mu": 11.5,"mpv_sd": 1.2,  "bleeding_severe": 0.42, "ddavp_effective": 0.30, "rFVIIa_use": 0.30, "myelofib_pct": 0.35},
        "MYH9":    {"age_mu": 28, "age_sd": 15, "plt_mu": 62,  "plt_sd": 22, "mpv_mu": 13.8,"mpv_sd": 2.0,  "bleeding_severe": 0.22, "ddavp_effective": 0.75, "rFVIIa_use": 0.08, "snhl_pct": 0.60},
        "HPS1":    {"age_mu": 35, "age_sd": 12, "plt_mu": 128, "plt_sd": 35, "mpv_mu": 9.2, "mpv_sd": 0.9,  "bleeding_severe": 0.38, "ddavp_effective": 0.50, "rFVIIa_use": 0.20, "ipf_pct": 0.42},
        "P2RY12":  {"age_mu": 32, "age_sd": 14, "plt_mu": 195, "plt_sd": 42, "mpv_mu": 9.5, "mpv_sd": 0.8,  "bleeding_severe": 0.28, "ddavp_effective": 0.55, "rFVIIa_use": 0.10, "drug_error_pct": 0.25},
    }
    p = params[gene["gene"]]

    n = 40
    ages = [max(1, round(rng.gauss(p["age_mu"], p["age_sd"]))) for _ in range(n)]
    plts = [max(5, round(rng.gauss(p["plt_mu"], p["plt_sd"]))) for _ in range(n)]
    mpvs = [round(max(6.0, rng.gauss(p["mpv_mu"], p["mpv_sd"])), 1) for _ in range(n)]

    bleeding_severe = round(sum(1 for _ in range(n) if rng.random() < p["bleeding_severe"]))
    ddavp_effective = round(sum(1 for _ in range(n) if rng.random() < p["ddavp_effective"]))
    rfviia_use = round(sum(1 for _ in range(n) if rng.random() < p["rFVIIa_use"]))
    alloimmun = round(sum(1 for _ in range(n) if rng.random() < p.get("alloimmun_pct", 0.0)))
    myelofib = round(sum(1 for _ in range(n) if rng.random() < p.get("myelofib_pct", 0.0)))
    snhl = round(sum(1 for _ in range(n) if rng.random() < p.get("snhl_pct", 0.0)))
    ipf = round(sum(1 for _ in range(n) if rng.random() < p.get("ipf_pct", 0.0)))
    drug_error = round(sum(1 for _ in range(n) if rng.random() < p.get("drug_error_pct", 0.0)))

    # Aggregation pattern
    agg_agonists = {
        "ITGA2B":  {"adp": 0, "collagen": 0, "arachidonic_acid": 0, "epinephrine": 0, "ristocetin": 90, "thrombin": 10},
        "ITGB3":   {"adp": 0, "collagen": 0, "arachidonic_acid": 0, "epinephrine": 0, "ristocetin": 88, "thrombin": 12},
        "GP1BA":   {"adp": 92, "collagen": 88, "arachidonic_acid": 85, "epinephrine": 80, "ristocetin": 0, "thrombin": 45},
        "GP1BB":   {"adp": 90, "collagen": 86, "arachidonic_acid": 83, "epinephrine": 78, "ristocetin": 0, "thrombin": 42},
        "NBEAL2":  {"adp": 65, "collagen": 60, "arachidonic_acid": 58, "epinephrine": 55, "ristocetin": 82, "thrombin": 70},
        "MYH9":    {"adp": 75, "collagen": 70, "arachidonic_acid": 72, "epinephrine": 65, "ristocetin": 90, "thrombin": 80},
        "HPS1":    {"adp": 55, "collagen": 62, "arachidonic_acid": 58, "epinephrine": 50, "ristocetin": 88, "thrombin": 75},
        "P2RY12":  {"adp": 0, "collagen": 78, "arachidonic_acid": 82, "epinephrine": 25, "ristocetin": 90, "thrombin": 85},
    }

    return {
        "gene": gene["gene"],
        "locus": gene["locus"],
        "syndrome": gene["syndrome"],
        "n": n,
        "mean_age": round(sum(ages) / n, 1),
        "mean_plt": round(sum(plts) / n),
        "mean_mpv": round(sum(mpvs) / n, 1),
        "bleeding_severe_count": bleeding_severe,
        "ddavp_effective_count": ddavp_effective,
        "rfviia_use_count": rfviia_use,
        "alloimmun_count": alloimmun,
        "myelofibrosis_count": myelofib,
        "snhl_count": snhl,
        "ipf_count": ipf,
        "drug_error_count": drug_error,
        "aggregation_pct": agg_agonists[gene["gene"]],
        "bleeding_risk": gene["bleeding_risk"],
        "pathognomonic": gene["pathognomonic"],
        "key_rule": gene["key_rule"],
        "targeted_rx": gene["targeted_rx"],
    }


def generate_overview() -> dict:
    cohorts = [_make_cohort(i, SEED_BASE + i) for i in range(8)]
    total_patients = sum(c["n"] for c in cohorts)
    mean_age_overall = round(sum(c["mean_age"] * c["n"] for c in cohorts) / total_patients, 1)

    total_severe_bleed = sum(c["bleeding_severe_count"] for c in cohorts)
    total_rfviia = sum(c["rfviia_use_count"] for c in cohorts)
    total_ddavp_eff = sum(c["ddavp_effective_count"] for c in cohorts)
    total_alloimmun = sum(c["alloimmun_count"] for c in cohorts)
    total_myelofib = sum(c["myelofibrosis_count"] for c in cohorts)
    total_snhl = sum(c["snhl_count"] for c in cohorts)
    total_ipf = sum(c["ipf_count"] for c in cohorts)

    per_gene = [
        {
            "gene": c["gene"],
            "locus": c["locus"],
            "n": c["n"],
            "mean_age": c["mean_age"],
            "mean_plt": c["mean_plt"],
            "mean_mpv": c["mean_mpv"],
            "severe_bleed_pct": round(c["bleeding_severe_count"] / 40 * 100),
            "rfviia_pct": round(c["rfviia_use_count"] / 40 * 100),
            "ddavp_effective_pct": round(c["ddavp_effective_count"] / 40 * 100),
            "alloimmun_count": c["alloimmun_count"],
            "myelofibrosis_count": c["myelofibrosis_count"],
            "snhl_count": c["snhl_count"],
            "ipf_count": c["ipf_count"],
            "aggregation_pct": c["aggregation_pct"],
            "bleeding_risk": c["bleeding_risk"],
            "key_rule": c["key_rule"],
            "targeted_rx": c["targeted_rx"],
        }
        for c in cohorts
    ]

    return {
        "atlas": "Hereditary-Platelet-Function-Disorder-Predisposition-Atlas",
        "genes": [g["gene"] for g in ATLAS_GENES],
        "n_genes": 8,
        "total_patients": total_patients,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "mean_age_overall": mean_age_overall,
        "clinical_summary": {
            "severe_bleed_total": total_severe_bleed,
            "rfviia_use_total": total_rfviia,
            "ddavp_effective_total": total_ddavp_eff,
            "alloimmunisation_total": total_alloimmun,
            "myelofibrosis_total": total_myelofib,
            "snhl_total": total_snhl,
            "pulmonary_fibrosis_total": total_ipf,
        },
        "key_clinical_distinctions": KEY_CLINICAL_DISTINCTIONS,
        "per_gene": per_gene,
    }


def generate_breakdown() -> dict:
    cohorts = [_make_cohort(i, SEED_BASE + i) for i in range(8)]
    gene_data = {c["gene"]: c for c in cohorts}
    return {
        "atlas": "Hereditary-Platelet-Function-Disorder-Predisposition-Atlas",
        "gene_data": gene_data,
        "atlas_genes": [
            {
                "gene": g["gene"],
                "locus": g["locus"],
                "protein": g["protein"],
                "syndrome": g["syndrome"],
                "inheritance": g["inheritance"],
                "bleeding_risk": g["bleeding_risk"],
                "pathognomonic": g["pathognomonic"],
                "key_rule": g["key_rule"],
                "key_avoid": g["key_avoid"],
                "surveillance": g["surveillance"],
                "targeted_rx": g["targeted_rx"],
            }
            for g in ATLAS_GENES
        ],
    }


def generate_definitions() -> dict:
    defs = []
    for g in ATLAS_GENES:
        defs.append({
            "term": f"{g['gene']} -- {g['syndrome']}",
            "definition": (
                f"{g['locus']} {g['inheritance']} -- "
                f"{g['bleeding_risk']} -- "
                f"Pathognomonic: {g['pathognomonic']} -- "
                f"Key rule: {g['key_rule']} -- "
                f"Avoid: {g['key_avoid']} -- "
                f"Surveillance: {g['surveillance']} -- "
                f"Targeted Rx: {g['targeted_rx']}"
            ),
        })
        defs.append({
            "term": f"{g['gene']} molecular size",
            "definition": g["protein_size"],
        })
    defs.append({
        "term": "KEY CLINICAL DISTINCTIONS -- Hereditary Platelet Function Disorder 8-Gene Atlas",
        "definition": " | ".join(KEY_CLINICAL_DISTINCTIONS),
    })
    return {"atlas": "Hereditary-Platelet-Function-Disorder-Predisposition-Atlas", "definitions": defs}


if __name__ == "__main__":
    import json
    result = generate_overview()
    print(json.dumps(result, indent=2))
