#!/usr/bin/env python3
"""Hereditary-Pulmonary-Arterial-Hypertension-PVOD-Predisposition-Atlas -- Complete 8-Gene Reference
Inherited predispositions to pulmonary arterial hypertension (PAH) and pulmonary veno-occlusive disease (PVOD):
BMPR2 (most common; 75% familial PAH), ACVRL1/ALK1 (HHT2+PAH), ENG (HHT1+PAH), SMAD9 (BMP pathway),
KCNK3/TASK1 (potassium channel PAH), CAV1 (caveolin-1 PAH+lipodystrophy), EIF2AK4 (biallelic PVOD PATHOGNOMONIC),
TBX4 (pediatric PAH + small patella syndrome PATHOGNOMONIC).

BMPR2   (Bone morphogenetic protein receptor type-2; 1038aa; 2q33.1; AD LOF;
          Most common cause familial PAH 75%; 20% sporadic IPAH; incomplete penetrance 20% women 10% men;
          Right heart catheterisation (RHC) MANDATORY for diagnosis — echo alone insufficient PATHOGNOMONIC rule;
          seed SEED_BASE+0).
ACVRL1  (Activin A receptor-like type-1 / ALK1; 503aa; 12q13.13; AD LOF;
          Hereditary haemorrhagic telangiectasia type 2 (HHT2) + PAH;
          Telangiectases on lips/tongue/fingertips PATHOGNOMONIC HHT;
          Epistaxis first symptom 95% HHT2 by age 21;
          seed SEED_BASE+1).
ENG     (Endoglin; 658aa; 9q34.11; AD LOF;
          HHT type 1 (most common HHT); epistaxis PATHOGNOMONIC;
          Pulmonary AVM — CT chest MANDATORY in ALL HHT1; shunt → paradoxical embolism;
          seed SEED_BASE+2).
SMAD9   (SMAD family member 9 / MADH9; 530aa; 13q13.3; AD LOF;
          BMP signalling downstream of BMPR2; rare familial PAH 1-2%;
          BMPR2/SMAD pathway = canonical PAH mechanism;
          seed SEED_BASE+3).
KCNK3   (Potassium two pore domain channel subfamily K member 3 / TASK1; 354aa; 2p23.3; AD LOF;
          TASK1 potassium channel; rare familial PAH; vasoconstriction via membrane depolarisation;
          seed SEED_BASE+4).
CAV1    (Caveolin-1; 178aa; 7q31.2; AD LOF;
          Lipid raft scaffolding protein; familial PAH + partial lipodystrophy;
          Lipodystrophy in PAH = CAV1 screen PATHOGNOMONIC;
          seed SEED_BASE+5).
EIF2AK4 (Eukaryotic translation initiation factor 2 alpha kinase 4 / GCN2; 1649aa; 15q15.1; AR LOF;
          Biallelic LOF → pulmonary veno-occlusive disease (PVOD) / pulmonary capillary haemangiomatosis (PCH);
          Prostanoids and vasodilators ABSOLUTELY CONTRAINDICATED — can precipitate flash pulmonary oedema PATHOGNOMONIC rule;
          Lung transplant = only curative option;
          seed SEED_BASE+6).
TBX4    (T-box transcription factor 4; 530aa; 17q23.2; AD LOF;
          Pediatric PAH (onset < 18 yr) predominant; small patella syndrome PATHOGNOMONIC;
          Ischiocoxopodopatellar syndrome (ICPS) = hypoplastic/absent patella + coxa vara + foot anomalies;
          TBX4 PAH: younger onset, better response to therapy vs BMPR2;
          seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 3718-3725)
"""
import random

SEED_BASE = 3718

ATLAS_GENES = [
    {
        "gene": "BMPR2",
        "protein": (
            "BMPR2 -- 2q33.1 AD LOF -- 1038aa -- "
            "Bone-Morphogenetic-Protein-Receptor-Type-2-Most-Common-Familial-PAH-75pct-"
            "RHC-Mandatory-Echo-Insufficient-Penetrance-20pct-Women-10pct-Men-"
            "BMPR2-OMIM-600799"
        ),
        "locus": "2q33.1",
        "protein_size": (
            "1038 aa / ~115 kDa / 2q33.1 BMPR2 Bone Morphogenetic Protein Receptor Type 2 molecular context: "
            "STRUCTURE: "
            "  1038 aa / ~115 kDa; type II transmembrane serine/threonine kinase receptor; "
            "  Three functional domains: ligand-binding extracellular domain (ECD), TM helix, intracellular kinase domain, "
            "  and long unique cytoplasmic tail (CTD) — CTD not present in type I receptors; "
            "  BMPR2 + type I receptor (ACVRL1/ALK1 or BMPR1A/ALK3) → heterotetrameric complex on binding BMP ligands; "
            "  BMP2/BMP4/BMP9/BMP10 ligands activate SMAD1/5/9 pathway → anti-proliferative / pro-apoptotic in PASMC; "
            "  LOF → loss of BMP-mediated growth inhibition in pulmonary arterial smooth muscle cells (PASMC); "
            "GENETICS: "
            "  AD LOF; penetrance markedly incomplete: ~20% in females, ~10% in males; "
            "  75% of familial PAH (FPD/PAH); 20% of 'idiopathic' IPAH; "
            "  Mutation types: frameshift/nonsense (50%) → NMD → haploinsufficiency; missense kinase domain (25%); "
            "  Large deletions/duplications 5% — MLPA mandatory to detect; "
            "  CTD mutations associated with milder haemodynamics vs kinase domain mutations; "
            "CLINICAL: "
            "  WHO Functional Class II–III dyspnoea on exertion; syncope on exertion in severe cases; "
            "  Right heart catheterisation (RHC) MANDATORY: mPAP ≥20 mmHg + PVR ≥2 WU at rest REQUIRED for diagnosis; "
            "  Echo alone: INSUFFICIENT — echo overestimates/underestimates sPAP; never diagnose PAH without RHC; "
            "  Vasoreactivity test (VRT) with inhaled NO: positive (≥10 mmHg drop, final mPAP ≤40 mmHg) in <10% BMPR2; "
            "  VRT-positive → CCB monotherapy (nifedipine/diltiazem) — only subgroup that responds; "
            "TREATMENT: "
            "  ERA (ambrisentan, macitentan) + PDE5i (sildenafil, tadalafil) combination first-line; "
            "  Selexipag (prostacyclin IP receptor agonist) add-on for progressive disease; "
            "  IV epoprostenol (prostacyclin) for WHO FC IV; "
            "  Sotatercept (activin signalling inhibitor) FDA-approved 2024 for PAH — STELLAR trial; "
            "  Lung transplant for refractory disease; bilateral preferred over single lung; "
            "SURVEILLANCE: "
            "  Cascade genetic testing all first-degree relatives; echocardiography 1–2 yearly in carriers; "
            "  Avoid systemic vasodilators (amlodipine, ACEi) empirically — can worsen V/Q mismatch; "
            "  Avoid pregnancy — maternal mortality >30% in severe PAH; counsel ALL female BMPR2 carriers"
        ),
        "inheritance": "AD LOF / Haploinsufficiency",
        "syndrome": (
            "Familial-Pulmonary-Arterial-Hypertension / FPAH / Idiopathic-PAH-20pct / "
            "RHC-Mandatory-Echo-Insufficient / VRT-Positive-lt10pct-BMPR2 / "
            "Sotatercept-FDA2024-STELLAR / Pregnancy-Maternal-Mortality-30pct"
        ),
        "pathognomonic": (
            "RHC MANDATORY — echo alone insufficient to diagnose PAH. "
            "mPAP ≥20 mmHg + PVR ≥2 WU at rest required. "
            "Avoid pregnancy in BMPR2 carriers — maternal mortality >30% in severe PAH."
        ),
        "key_rule": (
            "BMPR2 penetrance is sex-dependent (20% F vs 10% M). "
            "Cascade test ALL first-degree relatives. "
            "Echo screening 1-2 yearly in gene-positive carriers. "
            "Sotatercept (STELLAR trial) adds to ERA+PDE5i for progressive disease."
        ),
        "bmf_risk": "PAH — mPAP ≥20 mmHg PVR ≥2 WU; BMPR2-75pct-familial; 20pct-IPAH",
        "seed": SEED_BASE,
    },
    {
        "gene": "ACVRL1",
        "protein": (
            "ACVRL1 -- 12q13.13 AD LOF -- 503aa -- "
            "Activin-Receptor-Like-Kinase-1-ALK1-HHT2-Telangiectases-Epistaxis-PATHOGNOMONIC-"
            "PAV-PAH-10pct-HHT2-Bevacizumab-Epistaxis-"
            "ACVRL1-OMIM-601284"
        ),
        "locus": "12q13.13",
        "protein_size": (
            "503 aa / ~57 kDa / 12q13.13 ACVRL1/ALK1 Activin A Receptor-Like Type 1 molecular context: "
            "STRUCTURE: "
            "  503 aa / ~57 kDa; type I serine/threonine kinase receptor (TGF-β superfamily); "
            "  Forms heterodimer with BMPR2 (type II) in endothelium; "
            "  BMP9/BMP10 preferentially signal through ACVRL1/ALK1 in vascular endothelium → Smad1/5/9; "
            "  TGF-β can signal via ACVRL1 with TGF-βR2 → Smad2/3 (pro-inflammatory, angiogenic); "
            "  Endothelial-enriched — loss → arteriovenous malformations; "
            "GENETICS: "
            "  AD LOF; HHT type 2 (HHT2) — ~40% of all HHT; Spanish/Italian founder population enriched; "
            "  ACVRL1 mutations associated with higher PAH risk than ENG (HHT1): 10% vs 2% HHT1; "
            "  Missense most common (70%); PVD loop and kinase domain clustering; "
            "CLINICAL: "
            "  Curação criteria for HHT diagnosis: 3 or 4 = definite, 2 = possible: "
            "  (1) Spontaneous epistaxis (recurrent); (2) Mucocutaneous telangiectases (lips/tongue/fingertips); "
            "  (3) Visceral AVM (GI, pulmonary, hepatic, spinal, cerebral); (4) First-degree relative with HHT; "
            "  Epistaxis: first symptom 95% by age 21; most common morbidity; iron-deficiency anaemia; "
            "  Telangiectases lips/tongue/fingertips PATHOGNOMONIC HHT — appear 2nd–3rd decade; "
            "  Pulmonary AVMs in 15-30% HHT2 → paradoxical embolism/stroke/cerebral abscess — CT chest MANDATORY; "
            "  Hepatic AVMs 30-70% ACVRL1 (higher than ENG) — hepatic encephalopathy/high-output HF; "
            "PAH + HHT2: "
            "  PAH occurs in ~10% ACVRL1 carriers — severe, biventricular dysfunction; "
            "  PAH complicating HHT2 responds poorly to CCB; ERA+PDE5i required; "
            "  Liver transplant for refractory hepatic HHT; bevacizumab (anti-VEGF) reduces epistaxis frequency; "
            "SURVEILLANCE: "
            "  CT chest at diagnosis + every 5 yr for pulmonary AVMs; "
            "  MRI brain for cerebral AVMs; embolisation prophylactic for pAVM ≥3mm; "
            "  Upper GI scope for GI AVMs if anaemia unexplained; "
            "  Echocardiography annually if ACVRL1 carrier (PAH risk higher than ENG)"
        ),
        "inheritance": "AD LOF",
        "syndrome": (
            "HHT2-Hereditary-Haemorrhagic-Telangiectasia-Type-2 / "
            "Telangiectases-Lips-Tongue-Fingertips-PATHOGNOMONIC / "
            "Pulmonary-AVM-CT-Chest-Mandatory / PAH-10pct-HHT2 / "
            "Bevacizumab-Epistaxis / Hepatic-AVM-30-70pct-ACVRL1"
        ),
        "pathognomonic": (
            "Telangiectases on lips, tongue, and fingertips = HHT PATHOGNOMONIC. "
            "Pulmonary AVM on CT → paradoxical embolism risk: antibiotic prophylaxis for dental/surgical procedures. "
            "ACVRL1 PAH risk 10% — higher than ENG/HHT1."
        ),
        "key_rule": (
            "CT chest MANDATORY at HHT2 diagnosis to screen for pAVMs. "
            "Embolise pAVMs ≥3mm to prevent paradoxical embolism and cerebral abscess. "
            "Echo annually in all ACVRL1 carriers for PAH. "
            "Bevacizumab reduces epistaxis transfusion burden."
        ),
        "bmf_risk": "HHT2 + PAH 10pct; pAVM-paradoxical-embolism-stroke; epistaxis-iron-deficiency",
        "seed": SEED_BASE + 1,
    },
    {
        "gene": "ENG",
        "protein": (
            "ENG -- 9q34.11 AD LOF -- 658aa -- "
            "Endoglin-HHT1-Most-Common-HHT-pAVM-CT-Mandatory-Cerebral-AVM-MRI-"
            "Epistaxis-PATHOGNOMONIC-PAH-2pct-HHT1-"
            "ENG-OMIM-131195"
        ),
        "locus": "9q34.11",
        "protein_size": (
            "658 aa / ~70 kDa / 9q34.11 ENG Endoglin molecular context: "
            "STRUCTURE: "
            "  658 aa / ~70 kDa; type III TGF-β receptor; homodimer on endothelial cell surface; "
            "  Binds BMP9/BMP10 and TGF-β1/β3; accessory receptor modulating ACVRL1 and TGF-βRI signalling; "
            "  Orphan domain (orphan region, ZP domain) — unique to ENG; "
            "  Highly expressed in vascular endothelium, especially angiogenic vessels; "
            "  LOF → impaired endothelial response to BMP9/10 → uncontrolled angiogenesis → AVM formation; "
            "GENETICS: "
            "  AD LOF; HHT type 1 (most common HHT overall — ~55-60% all HHT); "
            "  ENG mutations more likely to cause cerebral AVMs than ACVRL1; "
            "  Predominantly frameshift/nonsense (~60%); missense 30%; large deletion 10%; "
            "  PAH risk lower than ACVRL1: ~2% ENG carriers vs 10% ACVRL1; "
            "CLINICAL: "
            "  Curação criteria same as HHT2 (see ACVRL1); "
            "  Epistaxis: earlier onset (childhood) and more severe than HHT2; "
            "  Telangiectases PATHOGNOMONIC — same distribution (lips/tongue/fingertips); "
            "  Pulmonary AVMs in 30-50% ENG — higher prevalence than HHT2; "
            "  Cerebral AVMs in 10% ENG — higher than ACVRL1 (5-8%); "
            "  MRI brain MANDATORY in ENG at diagnosis for cerebral AVM; "
            "  Hepatic AVMs less common than ACVRL1 (10-15% vs 30-70%); "
            "SURVEILLANCE: "
            "  CT chest at diagnosis, 5-yearly; pAVM embolisation ≥3mm; "
            "  MRI brain at diagnosis (cerebral AVM more common ENG than ACVRL1); "
            "  Echo at diagnosis to screen for PAH (lower yield than ACVRL1 but still warranted); "
            "  GI endoscopy for unexplained anaemia / Hb drop; "
            "  Avoid dental/surgical procedures without antibiotic prophylaxis if pAVM present"
        ),
        "inheritance": "AD LOF",
        "syndrome": (
            "HHT1-Hereditary-Haemorrhagic-Telangiectasia-Type-1 / "
            "Most-Common-HHT-55-60pct / Cerebral-AVM-MRI-Mandatory / "
            "pAVM-30-50pct-ENG / PAH-2pct-ENG / Epistaxis-Childhood-Onset"
        ),
        "pathognomonic": (
            "Telangiectases lips/tongue/fingertips = HHT PATHOGNOMONIC. "
            "MRI brain MANDATORY at ENG diagnosis — cerebral AVM rate 10% (higher than ACVRL1). "
            "CT chest mandatory for pulmonary AVM — 30-50% prevalence."
        ),
        "key_rule": (
            "ENG cerebral AVM rate (10%) > ACVRL1 (5-8%) — brain MRI at diagnosis in ALL ENG. "
            "pAVM embolisation ≥3mm before pregnancy and before procedures with air introduction. "
            "Epistaxis starts in childhood — iron supplementation often needed from early age."
        ),
        "bmf_risk": "HHT1; pAVM-30-50pct; cerebral-AVM-10pct; epistaxis-anaemia",
        "seed": SEED_BASE + 2,
    },
    {
        "gene": "SMAD9",
        "protein": (
            "SMAD9 -- 13q13.3 AD LOF -- 530aa -- "
            "SMAD-Family-Member-9-MADH9-BMP-Signalling-Downstream-BMPR2-"
            "Rare-Familial-PAH-1-2pct-BMPR2-Pathway-"
            "SMAD9-OMIM-603295"
        ),
        "locus": "13q13.3",
        "protein_size": (
            "530 aa / ~59 kDa / 13q13.3 SMAD9/MADH9 SMAD Family Member 9 molecular context: "
            "STRUCTURE: "
            "  530 aa / ~59 kDa; receptor-regulated SMAD (R-SMAD); "
            "  Contains MH1 (DNA binding), linker, and MH2 (receptor interaction / complex formation) domains; "
            "  Phosphorylated by BMPR1 (ALK3/6) and ACVRL1 (ALK1) on SSXS motif in MH2 domain; "
            "  Forms heterotrimeric complex with SMAD4 (co-SMAD) and translocates to nucleus; "
            "  Transcriptional targets: ID1, ID2, ID3 — anti-proliferative in PASMC; "
            "  LOF → reduced BMP-SMAD signalling → loss of growth inhibition in PASMC → plexiform lesions; "
            "  Functional redundancy with SMAD1 and SMAD5 — explains lower penetrance/frequency vs BMPR2; "
            "GENETICS: "
            "  AD LOF; rare cause familial PAH — 1-2% of familial PAH cases; "
            "  First identified in Japanese PAH families (Nasim 2011); "
            "  Mutations in MH2 domain prevent SMAD4 interaction; "
            "  Clinically indistinguishable from BMPR2-associated PAH; "
            "CLINICAL/TREATMENT: "
            "  PAH phenotype mirrors BMPR2 PAH: RHC mandatory, ERA+PDE5i standard; "
            "  VRT positivity rare (< 5%); "
            "  Cascade family screening recommended once index case confirmed"
        ),
        "inheritance": "AD LOF",
        "syndrome": (
            "Familial-PAH-Rare / BMP-SMAD-Signalling / SMAD9-1-2pct-Familial-PAH / "
            "BMPR2-pathway-downstream / Clinically-Identical-to-BMPR2-PAH"
        ),
        "pathognomonic": None,
        "key_rule": (
            "SMAD9 PAH is clinically identical to BMPR2 PAH — treat identically. "
            "Panel sequencing (BMPR2+ACVRL1+ENG+SMAD9+KCNK3+CAV1+TBX4+EIF2AK4) more efficient than serial testing. "
            "Cascade screening recommended."
        ),
        "bmf_risk": "PAH-SMAD9-1-2pct-familial; BMP-pathway",
        "seed": SEED_BASE + 3,
    },
    {
        "gene": "KCNK3",
        "protein": (
            "KCNK3 -- 2p23.3 AD LOF -- 354aa -- "
            "Potassium-Channel-TASK1-Two-Pore-Domain-Membrane-Depolarisation-Vasoconstriction-"
            "Rare-Familial-PAH-GOF-Reported-"
            "KCNK3-OMIM-603220"
        ),
        "locus": "2p23.3",
        "protein_size": (
            "354 aa / ~39 kDa / 2p23.3 KCNK3/TASK1 Potassium Channel K2P3.1 molecular context: "
            "STRUCTURE: "
            "  354 aa / ~39 kDa; background/leak potassium channel (K2P family); "
            "  Two-pore domain (2P) channel with four TM helices; forms homodimers; "
            "  Expressed in pulmonary arterial smooth muscle cells (PASMC) — maintains resting membrane potential; "
            "  LOF → membrane depolarisation → Ca2+ influx via L-type VDCCs → vasoconstriction; "
            "  TASK1 (TWIK-related acid-sensitive K+ channel) — inhibited by acidosis and hypoxia; "
            "  Pharmacological activators (ML365) under investigation; "
            "GENETICS: "
            "  AD LOF; rare familial PAH — <1% of all PAH; "
            "  First described Eyries 2014 (Nature Genetics) — French and British families; "
            "  Missense mutations G203D, T8M, most reported; "
            "  Both LOF (most) and rare GOF variants reported in PAH; "
            "CLINICAL: "
            "  PASMC vasoconstriction-predominant phenotype; no extrapulmonary features; "
            "  Response to vasodilators similar to BMPR2 PAH; "
            "  KCNK3 activators (novel targeted therapy) in preclinical/early clinical trials"
        ),
        "inheritance": "AD LOF (rare GOF also reported)",
        "syndrome": (
            "Familial-PAH-KCNK3-lt1pct / TASK1-Channel / Membrane-Depolarisation-Vasoconstriction / "
            "No-Extrapulmonary-Features / KCNK3-Activators-Investigational"
        ),
        "pathognomonic": None,
        "key_rule": (
            "KCNK3 PAH has no extrapulmonary features — panel testing essential to identify. "
            "Treat identically to BMPR2 PAH (ERA+PDE5i). "
            "No KCNK3-targeted therapy is FDA-approved; activators in trials."
        ),
        "bmf_risk": "PAH-KCNK3-lt1pct; vasoconstriction-predominant",
        "seed": SEED_BASE + 4,
    },
    {
        "gene": "CAV1",
        "protein": (
            "CAV1 -- 7q31.2 AD LOF -- 178aa -- "
            "Caveolin-1-Lipid-Raft-Scaffold-Familial-PAH-Lipodystrophy-"
            "Lipodystrophy-in-PAH-PATHOGNOMONIC-CAV1-"
            "CAV1-OMIM-601047"
        ),
        "locus": "7q31.2",
        "protein_size": (
            "178 aa / ~22 kDa / 7q31.2 CAV1 Caveolin-1 molecular context: "
            "STRUCTURE: "
            "  178 aa / ~22 kDa; principal protein of caveolae; "
            "  Hairpin topology — N and C termini cytoplasmic; single intramembrane hairpin; "
            "  Scaffolding domain (CSD) binds eNOS, caveolae-associated kinases; "
            "  CAV1 oligomerises into 7S complexes → 8S caveolae on plasma membrane; "
            "  In endothelium: CAV1 inhibits eNOS via CSD → LOF → eNOS over-activation; "
            "  In PASMC: CAV1 normally inhibits proliferative signalling; LOF → hyperproliferation; "
            "  Paradox: eNOS overactivation in endothelium vs proliferative signalling in PASMC — both pro-PAH; "
            "GENETICS: "
            "  AD LOF; rare familial PAH ~1%; "
            "  First described Austin 2012 (NEJM); frameshift deletion most common; "
            "  CAV1 mutations also cause partial lipodystrophy + PAH syndrome — unique combination; "
            "CLINICAL: "
            "  Partial lipodystrophy: loss of subcutaneous fat from limbs/buttocks + central adiposity; "
            "  Lipodystrophy in PAH = CAV1 screen PATHOGNOMONIC; "
            "  Metabolic: insulin resistance, hypertriglyceridaemia, low HDL; "
            "  PAH phenotype: endothelial-dominant; plexiform lesions on biopsy; "
            "  Treatment: standard ERA+PDE5i; lipodystrophy managed with metreleptin (off-label)"
        ),
        "inheritance": "AD LOF",
        "syndrome": (
            "Familial-PAH-CAV1-lt1pct / Partial-Lipodystrophy-PATHOGNOMONIC / "
            "eNOS-Dysregulation / Lipodystrophy-in-PAH-Screen-CAV1"
        ),
        "pathognomonic": (
            "Partial lipodystrophy (loss of limb/buttock fat + central adiposity) in a PAH patient = "
            "CAV1 germline mutation screen PATHOGNOMONIC. "
            "No other hereditary PAH gene causes lipodystrophy."
        ),
        "key_rule": (
            "Assess body fat distribution in all PAH patients — lipodystrophy = CAV1 flag. "
            "Metabolic screening (glucose, TG, HDL) in CAV1 PAH. "
            "Treat PAH with standard ERA+PDE5i. "
            "Metreleptin for lipodystrophy-related metabolic complications (off-label, specialist centres)."
        ),
        "bmf_risk": "PAH-CAV1-lt1pct; partial-lipodystrophy-metabolic",
        "seed": SEED_BASE + 5,
    },
    {
        "gene": "EIF2AK4",
        "protein": (
            "EIF2AK4 -- 15q15.1 AR LOF -- 1649aa -- "
            "GCN2-Kinase-Biallelic-PVOD-PCH-PATHOGNOMONIC-"
            "Prostanoids-Vasodilators-ABSOLUTELY-CONTRAINDICATED-Flash-Oedema-Lung-Transplant-Only-Cure-"
            "EIF2AK4-OMIM-609280"
        ),
        "locus": "15q15.1",
        "protein_size": (
            "1649 aa / ~185 kDa / 15q15.1 EIF2AK4/GCN2 Eukaryotic Translation Initiation Factor 2α Kinase 4 molecular context: "
            "STRUCTURE: "
            "  1649 aa / ~185 kDa; serine/threonine kinase activated by amino acid deprivation; "
            "  Senses uncharged tRNAs as proxy for amino acid availability; "
            "  Phosphorylates eIF2α → reduced global protein synthesis + selective ATF4 upregulation; "
            "  ATF4 → integrated stress response (ISR) genes; "
            "  In pulmonary endothelium: biallelic LOF → loss of stress-response control; "
            "  Uncontrolled endothelial and haemangiopericytic proliferation → venous/capillary occlusion; "
            "GENETICS: "
            "  AR LOF; biallelic mutations REQUIRED; "
            "  ~10% of 'sporadic' IPAH harbour biallelic EIF2AK4 mutations on sequencing; "
            "  French founder mutation p.Pro1197Leu; "
            "  Consanguineous families: homozygous truncating common; "
            "CLINICAL — PVOD/PCH DISTINGUISHING FEATURES: "
            "  Haemoptysis (20-30%) — vasodilator exposure may precipitate this; "
            "  CT chest: bilateral ground-glass opacities (GGO), septal lines, mediastinal adenopathy TRIAD PATHOGNOMONIC; "
            "  BAL lavage: hemosiderin-laden macrophages; Perls stain positive; "
            "  Hypoxaemia disproportionate to haemodynamics; "
            "  Low DLCO (< 50% predicted) out of proportion to CT; "
            "  CRITICAL SAFETY RULE: "
            "    Prostanoids (epoprostenol, treprostinil) and vasodilators (sildenafil, ERA) cause "
            "    acute flash pulmonary oedema in PVOD — venoconstriction at level of involved veins; "
            "    ABSOLUTELY CONTRAINDICATED before lung transplantation;  "
            "    If PAH suspected with CT-GGO triad or haemoptysis — DEFER vasodilators; screen EIF2AK4 FIRST; "
            "  Only cure: bilateral lung transplant; "
            "  Bridge therapy: very low-dose diuresis, supplemental O2, avoid exertion; "
            "  Outcomes without transplant: median survival < 2 yr from diagnosis"
        ),
        "inheritance": "AR LOF (biallelic)",
        "syndrome": (
            "PVOD-Pulmonary-Veno-Occlusive-Disease / PCH-Pulmonary-Capillary-Haemangiomatosis / "
            "CT-GGO-Triad-PATHOGNOMONIC / Prostanoids-ABSOLUTELY-CONTRAINDICATED-Flash-Oedema / "
            "Lung-Transplant-Only-Cure / Haemoptysis-20-30pct"
        ),
        "pathognomonic": (
            "PVOD triad on CT: bilateral GGO + septal lines + mediastinal adenopathy = EIF2AK4 biallelic screen PATHOGNOMONIC. "
            "PROSTANOIDS AND VASODILATORS ABSOLUTELY CONTRAINDICATED in PVOD — flash pulmonary oedema risk is life-threatening. "
            "Screen EIF2AK4 BEFORE initiating PAH therapy if PVOD pattern present. "
            "Only bilateral lung transplant is curative."
        ),
        "key_rule": (
            "PVOD masquerades as PAH. CT chest GGO + septal lines + haemoptysis = DO NOT give vasodilators until EIF2AK4 tested. "
            "Bilateral lung transplant listing must start immediately after PVOD confirmed. "
            "Diuresis + O2 for bridging only — no vasodilator exposure."
        ),
        "bmf_risk": "PVOD-PCH-biallelic-EIF2AK4; survival-lt2yr-without-transplant",
        "seed": SEED_BASE + 6,
    },
    {
        "gene": "TBX4",
        "protein": (
            "TBX4 -- 17q23.2 AD LOF -- 530aa -- "
            "T-Box-TF-Pediatric-PAH-Small-Patella-PATHOGNOMONIC-ICPS-"
            "Younger-Onset-Better-Response-Therapy-vs-BMPR2-"
            "TBX4-OMIM-601719"
        ),
        "locus": "17q23.2",
        "protein_size": (
            "530 aa / ~58 kDa / 17q23.2 TBX4 T-Box Transcription Factor 4 molecular context: "
            "STRUCTURE: "
            "  530 aa / ~58 kDa; T-box DNA-binding transcription factor; "
            "  T-box domain (exons 2-7) — binds T-element (palindromic half-sites); "
            "  Expressed during limb and lung development; "
            "  TBX4 and TBX5 regulate lung mesenchyme and hindlimb patterning; "
            "  TBX4 activates FGF10 signalling — required for lung branching morphogenesis; "
            "  In PASMC: TBX4 is expressed; LOF → impaired BMP/FGF10 axis → PAH; "
            "  TBX4 and BMPR2 loss synergistically impair BMP10/FGF10 → PAH; "
            "GENETICS: "
            "  AD LOF; ~2-5% familial PAH and sporadic pediatric PAH; "
            "  In children (<18 yr) with PAH: TBX4 mutations more prevalent than BMPR2; "
            "  Ischiocoxopodopatellar syndrome (ICPS): hypoplastic/absent patella + coxa vara + foot/toe anomalies; "
            "  Small patella syndrome (SPS): isolated hypoplastic patella — TBX4 LOF; "
            "  PAH can occur without ICPS and vice versa — pleiotropy; "
            "CLINICAL: "
            "  PAH with TBX4: younger onset (median age 5-8 yr in children); "
            "  Better haemodynamic response to vasodilators vs BMPR2 — VRT positivity higher (~15-20%); "
            "  Small patella PATHOGNOMONIC — feel for hypoplastic patella on knee exam in any PAH patient; "
            "  Coxa vara (reduced neck-shaft angle of femur) → hip pain, abnormal gait; "
            "  Radiograph hips and knees in TBX4 carriers; "
            "TREATMENT: "
            "  ERA+PDE5i combination standard; higher CCB response rate than BMPR2 if VRT-positive; "
            "  Sotatercept data limited in pediatric TBX4; "
            "  Lung transplant if refractory (less often needed than BMPR2 due to better response)"
        ),
        "inheritance": "AD LOF",
        "syndrome": (
            "Pediatric-PAH / ICPS-Ischiocoxopodopatellar-Syndrome / "
            "Small-Patella-PATHOGNOMONIC / Coxa-Vara / TBX4-2-5pct-Familial-PAH / "
            "Better-VRT-Response-vs-BMPR2 / Younger-Onset-Age-5-8yr"
        ),
        "pathognomonic": (
            "Hypoplastic or absent patella on knee examination in a PAH patient = TBX4 mutation screen PATHOGNOMONIC. "
            "PAH in children <18yr: TBX4 more prevalent than BMPR2 — test TBX4 first in pediatric onset. "
            "Small patella syndrome (SPS) alone without PAH still warrants annual echo surveillance."
        ),
        "key_rule": (
            "Knee palpation for patella hypoplasia in all PAH patients — TBX4 flag. "
            "Radiograph pelvis and knees in TBX4 carriers for ICPS features. "
            "Better vasoreactivity and prognosis vs BMPR2 — CCB trial if VRT-positive. "
            "Pediatric PAH: panel test TBX4 and BMPR2 simultaneously."
        ),
        "bmf_risk": "PAH-TBX4-2-5pct; pediatric-predominant; small-patella-coxa-vara",
        "seed": SEED_BASE + 7,
    },
]

DEFINITIONS = [
    {
        "term": "Pulmonary Arterial Hypertension (PAH)",
        "definition": (
            "Haemodynamic definition: mPAP ≥20 mmHg + PVR ≥2 WU at rest on right heart catheterisation (RHC). "
            "WHO Group 1 PAH: idiopathic, heritable (HPAH), drug/toxin-induced, CTD-associated, CHD-associated, "
            "portal hypertension, HIV, schistosomiasis, PVOD/PCH. "
            "Echo screening is insufficient — RHC MANDATORY for diagnosis."
        ),
    },
    {
        "term": "Right Heart Catheterisation (RHC) Mandatory",
        "definition": (
            "Echocardiography overestimates or underestimates sPAP by >10 mmHg in 50% of cases. "
            "RHC is the GOLD STANDARD for PAH diagnosis. "
            "mPAP threshold lowered to ≥20 mmHg in 2022 (from 25 mmHg) — more patients diagnosed earlier. "
            "PVR ≥2 WU required in addition to mPAP ≥20 to diagnose PAH (excludes exercise-induced elevation)."
        ),
    },
    {
        "term": "Vasoreactivity Test (VRT)",
        "definition": (
            "Inhaled NO (or IV adenosine/IV epoprostenol) during RHC. "
            "POSITIVE VRT: mPAP drop ≥10 mmHg to absolute mPAP ≤40 mmHg with maintained/increased CO. "
            "VRT-positive: <10% hereditary PAH overall (higher in TBX4 ~15-20%, lower in BMPR2 ~5%). "
            "Only VRT-positive patients may trial CCB monotherapy (nifedipine/diltiazem/amlodipine). "
            "VRT-negative: ERA+PDE5i combination first-line."
        ),
    },
    {
        "term": "PVOD — Pulmonary Veno-Occlusive Disease",
        "definition": (
            "Post-capillary pulmonary hypertension from venous/capillary level obstruction. "
            "EIF2AK4 biallelic mutations are the sole identified germline cause. "
            "CT triad: bilateral GGO + thickened interlobular septa + mediastinal lymphadenopathy PATHOGNOMONIC. "
            "BAL: haemosiderin-laden macrophages. DLCO severely reduced. "
            "PROSTANOIDS AND VASODILATORS CONTRAINDICATED — can cause acute fatal pulmonary oedema. "
            "Only treatment: bilateral lung transplant listing IMMEDIATELY after diagnosis."
        ),
    },
    {
        "term": "Hereditary Haemorrhagic Telangiectasia (HHT) — Curação Criteria",
        "definition": (
            "Definite HHT: 3–4 criteria. Possible HHT: 2 criteria. Unlikely: 0–1 criterion. "
            "(1) Spontaneous, recurrent epistaxis; (2) Multiple mucocutaneous telangiectases (lips/tongue/fingertips); "
            "(3) Visceral AVMs (pulmonary, hepatic, cerebral, GI, spinal); (4) First-degree relative with HHT. "
            "Telangiectases appear in 2nd–3rd decade; epistaxis usually first symptom (childhood). "
            "ENG (HHT1) and ACVRL1 (HHT2) account for >95% of HHT."
        ),
    },
    {
        "term": "Pulmonary AVM (pAVM) Management",
        "definition": (
            "pAVMs present in 30-50% ENG and 15-30% ACVRL1 carriers. "
            "CT chest at diagnosis mandatory for ALL HHT patients. "
            "Embolise pAVMs ≥3mm feeding artery — prevents paradoxical embolism, stroke, cerebral abscess. "
            "Antibiotic prophylaxis for dental/surgical procedures in unrepaired pAVM. "
            "After embolisation: CT chest at 1 yr and then every 3-5 yr (recanalization/new pAVM risk). "
            "Paradoxical embolism risk: IV air, bacteria, clots cross into systemic circulation — NO IV air injection."
        ),
    },
    {
        "term": "Sotatercept (FDA 2024 PAH)",
        "definition": (
            "Sotatercept = recombinant fusion protein (ACVR2A-Fc) that traps activin ligands; "
            "reduces anti-BMP signalling excess — restores BMP/activin balance in PASMC. "
            "FDA-approved March 2024 for PAH (WHO Group 1). STELLAR trial: 6MWD +40.8m vs placebo. "
            "Add-on to background ERA+PDE5i (± prostanoid). "
            "Haematocrit elevation common (target Hct 55-65% at dose). No specific benefit/harm data yet for BMPR2 vs other genotypes."
        ),
    },
    {
        "term": "TBX4 Small Patella Syndrome (SPS) and ICPS",
        "definition": (
            "Small Patella Syndrome (SPS): hypoplastic or absent patella — TBX4 LOF PATHOGNOMONIC. "
            "Ischiocoxopodopatellar syndrome (ICPS): small/absent patella + ischio-pubic hypoplasia + coxa vara + "
            "foot/toe anomalies (talipes equinovarus, toe anomalies). "
            "PAH in TBX4 can occur with or without ICPS — incomplete penetrance of each feature. "
            "PAH surveillance (annual echo) in all TBX4 LOF carriers even if no ICPS. "
            "Pediatric PAH: TBX4 mutations more prevalent than BMPR2 — include in first-line panel."
        ),
    },
    {
        "term": "ERA — Endothelin Receptor Antagonists",
        "definition": (
            "ERA block ET-1 binding to ETA and/or ETB receptors on PASMC → vasodilation + anti-proliferation. "
            "Ambrisentan: ETA-selective; Bosentan: dual ETA+ETB; Macitentan: dual with longer receptor occupancy. "
            "Teratogenic (Category X) — mandatory negative pregnancy test + dual contraception in female patients. "
            "Monthly LFT monitoring for bosentan (hepatotoxicity). "
            "Do not combine dual ERA therapy — no additive benefit, increased hepatotoxicity."
        ),
    },
    {
        "term": "PDE5i — Phosphodiesterase-5 Inhibitors",
        "definition": (
            "PDE5i prolong cGMP → enhance NO-mediated vasodilation in pulmonary vasculature. "
            "Sildenafil (20mg TID) and tadalafil (40mg OD) FDA-approved for PAH. "
            "First-line combination with ERA for WHO FC II-III. "
            "Contraindicated with organic nitrates (severe hypotension). "
            "Avoid in severe liver impairment. No adjustment needed for renal impairment (sildenafil)."
        ),
    },
]


def generate_overview() -> dict:
    rng = random.Random(SEED_BASE)
    gene_summaries = {}
    for g in ATLAS_GENES:
        rng.seed(g["seed"])
        n = 40
        mean_age = rng.randint(8, 55)
        pct_f = round(rng.uniform(0.45, 0.80), 2)
        six_min = rng.randint(180, 450)
        rng2 = random.Random(g["seed"] + 1)
        pvr = round(rng2.uniform(4.0, 18.0), 1)
        gene_summaries[g["gene"]] = {
            "n": n,
            "mean_age_at_diagnosis": mean_age,
            "pct_female": pct_f,
            "mean_6mwd_m": six_min,
            "mean_pvr_wu": pvr,
            "syndrome": g["syndrome"].split("/")[0].strip(),
            "locus": g["locus"],
            "inheritance": g["inheritance"],
            "pathognomonic": g["pathognomonic"],
        }
    return {
        "atlas": "Hereditary-Pulmonary-Arterial-Hypertension-PVOD-Predisposition-Atlas",
        "total_patients": 320,
        "seeds": "3718-3725",
        "genes": list(gene_summaries.keys()),
        "gene_summaries": gene_summaries,
        "key_stats": {
            "total_pah_patients": 280,
            "total_pvod_patients": 40,
            "bilateral_lung_transplant_listed": 40,
            "rhc_mandatory_rule": "Echo alone is insufficient — RHC MANDATORY for PAH diagnosis",
            "pvod_vasodilator_ci_rule": "Prostanoids/vasodilators ABSOLUTELY CONTRAINDICATED in EIF2AK4/PVOD",
            "vrt_positive_overall_pct": 8,
            "sotatercept_fda_year": 2024,
        },
    }


def generate_breakdown() -> dict:
    rng = random.Random(SEED_BASE)
    gene_data = {}
    for g in ATLAS_GENES:
        rng.seed(g["seed"])
        n = 40
        mean_age = rng.randint(8, 55)
        pct_f = round(rng.uniform(0.45, 0.80), 2)
        rng2 = random.Random(g["seed"] + 2)
        pvr = round(rng2.uniform(4.0, 18.0), 1)
        mpap = round(rng2.uniform(28.0, 65.0), 1)
        six_min = rng2.randint(200, 460)
        vrt_pos_n = rng2.randint(0, 8) if g["gene"] == "TBX4" else rng2.randint(0, 4)
        transplant_n = n if g["gene"] == "EIF2AK4" else rng2.randint(0, 12)
        bilateral_n = rng2.randint(0, 5) if g["gene"] != "EIF2AK4" else 0
        gene_data[g["gene"]] = {
            "n": n,
            "mean_age": mean_age,
            "pct_female": pct_f,
            "mean_pvr_wu": pvr,
            "mean_mpap_mmhg": mpap,
            "mean_6mwd_m": six_min,
            "vrt_positive_n": vrt_pos_n,
            "lung_transplant_n": transplant_n,
            "bilateral_avm_n": bilateral_n,
            "syndrome": g["syndrome"].split("/")[0].strip(),
            "inheritance": g["inheritance"],
        }

    return {
        "atlas": "Hereditary-Pulmonary-Arterial-Hypertension-PVOD-Predisposition-Atlas",
        "total_patients": 320,
        "seeds": "3718-3725",
        "gene_data": gene_data,
        "atlas_genes": ATLAS_GENES,
        "summary": {
            "bmpr2_familial_pah_pct": 75,
            "acvrl1_hht2_pah_pct": 10,
            "eng_hht1_pah_pct": 2,
            "eif2ak4_pvod_pct": 100,
            "tbx4_pediatric_predominant": True,
            "rhc_mandatory": True,
            "pvod_vasodilator_ci": True,
            "sotatercept_approved_2024": True,
        },
    }


def generate_definitions() -> dict:
    return {
        "atlas": "Hereditary-Pulmonary-Arterial-Hypertension-PVOD-Predisposition-Atlas",
        "definitions": DEFINITIONS,
        "critical_clinical_rules": {
            "RHC-MANDATORY-ECHO-INSUFFICIENT": (
                "Right heart catheterisation is MANDATORY for PAH diagnosis — echocardiography alone is insufficient "
                "and misclassifies sPAP in 50% of cases. mPAP ≥20 mmHg + PVR ≥2 WU required."
            ),
            "EIF2AK4-PVOD-PROSTANOIDS-ABSOLUTELY-CI": (
                "Prostanoids (epoprostenol, treprostinil) and vasodilators (ERA, PDE5i) are ABSOLUTELY CONTRAINDICATED "
                "in PVOD/EIF2AK4 — they cause acute flash pulmonary oedema. "
                "Screen EIF2AK4 BEFORE initiating vasodilator therapy if CT shows GGO + septal lines + haemoptysis."
            ),
            "TBX4-SMALL-PATELLA-PATHOGNOMONIC": (
                "Hypoplastic or absent patella on knee palpation in a PAH patient = TBX4 mutation screen. "
                "Pediatric PAH: TBX4 more prevalent than BMPR2 — include in first-line panel."
            ),
            "CAV1-LIPODYSTROPHY-PATHOGNOMONIC": (
                "Partial lipodystrophy (loss of limb/buttock fat + central adiposity) in PAH = CAV1 germline screen. "
                "No other hereditary PAH gene causes lipodystrophy."
            ),
            "ACVRL1-ENG-PAVM-CT-MANDATORY": (
                "CT chest MANDATORY in all HHT patients to screen for pulmonary AVMs. "
                "Embolise pAVMs ≥3mm feeding artery — prevents paradoxical embolism and cerebral abscess. "
                "Antibiotic prophylaxis for all dental/surgical procedures in unrepaired pAVM."
            ),
            "BMPR2-PREGNANCY-MATERNAL-MORTALITY": (
                "Maternal mortality >30% in severe PAH during pregnancy. "
                "Counsel ALL female BMPR2 carriers — avoid pregnancy or manage with expert MDT. "
                "Vasodilators are teratogenic (ERA = Category X)."
            ),
            "HHT-CEREBRAL-AVM-ENG-GT-ACVRL1": (
                "ENG (HHT1): cerebral AVM risk 10% > ACVRL1 (HHT2): 5-8%. "
                "MRI brain MANDATORY at diagnosis in ENG carriers. "
                "ACVRL1 hepatic AVM more common (30-70%) than ENG (10-15%)."
            ),
            "VRT-POSITIVE-CCB-ONLY": (
                "Only VRT-positive patients (mPAP drop ≥10 mmHg to ≤40 mmHg with maintained CO) may trial CCB monotherapy. "
                "<10% hereditary PAH is VRT-positive (TBX4 highest ~15-20%). "
                "VRT-negative = ERA+PDE5i combination first-line, NOT CCB."
            ),
            "SOTATERCEPT-FDA2024-STELLAR": (
                "Sotatercept FDA-approved 2024 for WHO Group 1 PAH (STELLAR trial). "
                "Add-on to background ERA+PDE5i. "
                "Monitor haematocrit (target 55-65%). "
                "No genotype-specific benefit data yet."
            ),
            "BMPR2-INCOMPLETE-PENETRANCE": (
                "BMPR2 penetrance: ~20% in female carriers, ~10% in male carriers. "
                "Echocardiography surveillance 1-2 yearly in all carriers. "
                "Cascade test ALL first-degree relatives of index BMPR2 case."
            ),
        },
    }
