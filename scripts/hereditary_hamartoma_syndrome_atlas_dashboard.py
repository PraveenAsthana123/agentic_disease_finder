#!/usr/bin/env python3
"""Hereditary-Hamartoma-Syndrome-Atlas — Complete 8-Gene Atlas
NF1     (Neurofibromin 1; 2839 aa; 17q11.2; AD;
         Neurofibromatosis type 1 / Von Recklinghausen disease;
         most common single-gene tumor-predisposition syndrome (1:2,000–3,000);
         RAS-GAP loss → RAS pathway constitutive activation;
         café-au-lait macules + Lisch nodules + plexiform neurofibromas;
         selumetinib (Koselugo) MEK1/2 inhibitor FDA2020 — FIRST approved NF1 therapy;
         MPNST 8–13% lifetime; JMML risk in children;
         seed SEED_BASE+0) .
NF2     (Merlin / Schwannomin; 595 aa; 22q12.2; AD;
         Neurofibromatosis type 2;
         bilateral vestibular schwannomas PATHOGNOMONIC (95% penetrance);
         meningiomas 50–75%, ependymomas, spinal schwannomas;
         bevacizumab — reduces VS volume + preserves hearing;
         hearing loss often first symptom age 18–24 yr;
         seed SEED_BASE+1) .
TSC1    (Hamartin; 1164 aa; 9q34.13; AD;
         Tuberous Sclerosis Complex type 1;
         TSC1–TSC2 heterodimer inhibits mTORC1 → Rheb → protein synthesis/cell growth;
         cardiac rhabdomyoma PATHOGNOMONIC in fetus/neonate;
         cortical tubers, SENs, SEGA — everolimus (Afinitor) mTOR inhibitor;
         milder than TSC2; less LAM, less severe ID;
         seed SEED_BASE+2) .
TSC2    (Tuberin; 1807 aa; 16p13.3; AD;
         Tuberous Sclerosis Complex type 2 — MORE SEVERE than TSC1;
         higher de novo rate (66–80%); more cortical tubers; worse epilepsy;
         LAM (lymphangioleiomyomatosis) predominantly TSC2 — sirolimus/everolimus;
         renal AML >4 cm = embolization threshold (hemorrhage risk);
         TSC2/PKD1 contiguous deletion → early polycystic kidney disease;
         seed SEED_BASE+3) .
VHL     (VHL protein / pVHL; 213 aa; 3p25.3; AD;
         Von Hippel-Lindau syndrome;
         pVHL = E3 ubiquitin ligase; targets HIF-1α/HIF-2α for degradation;
         LOF → HIF stabilisation → VEGF/PDGF overexpression;
         clear cell RCC PATHOGNOMONIC hallmark malignancy;
         retinal + cerebellar hemangioblastomas;
         belzutifan (Welireg) HIF-2α inhibitor FDA2021 — FIRST VHL targeted therapy;
         seed SEED_BASE+4) .
PTCH1   (Patched-1; 1447 aa; 9q22.32; AD;
         Gorlin syndrome / Nevoid Basal Cell Carcinoma Syndrome (NBCCS);
         PTCH1 = Hedgehog-pathway receptor — inhibits SMO when Hh absent;
         LOF → constitutive SMO → Gli transcription;
         multiple BCCs age <20 yr PATHOGNOMONIC;
         calcified falx cerebri on plain X-ray PATHOGNOMONIC;
         RADIATION ABSOLUTELY CONTRAINDICATED → 1,000× BCC induction;
         vismodegib (Erivedge) Smoothened inhibitor FDA approved;
         seed SEED_BASE+5) .
STK11   (Serine/threonine kinase 11 / LKB1; 433 aa; 19p13.3; AD;
         Peutz-Jeghers Syndrome (PJS);
         STK11/LKB1 = serine kinase tumor suppressor; activates AMPK;
         mucocutaneous lentigines (lips/buccal/digits) PATHOGNOMONIC;
         GI hamartomatous polyps → intussusception risk (small bowel most common);
         lifetime cancer risk ~76%; breast 45%, pancreatic, cervix, GI;
         annual small-bowel MRI/capsule endoscopy mandatory;
         seed SEED_BASE+6) .
PTEN    (Phosphatase and tensin homolog; 403 aa; 10q23.31; AD;
         PTEN Hamartoma Tumor Syndrome (PHTS) / Cowden syndrome;
         PTEN = lipid/protein phosphatase; degrades PIP3 → inhibits PI3K/AKT/mTOR;
         macrocephaly (HC >97th %ile) PATHOGNOMONIC;
         trichilemmomas on histology PATHOGNOMONIC;
         breast cancer 85% lifetime risk — SAME as BRCA1;
         thyroid 35%, endometrial 28%;
         everolimus (mTOR) — trials ongoing;
         seed SEED_BASE+7)
320-patient aggregate cohort (8 × 40, seeds 1910–1917)
"""

import random

SEED_BASE = 1910

HAMARTOMA_GENES = [
    # -- NF1 — Neurofibromatosis type 1 / Von Recklinghausen -------------------------
    {
        "gene": "NF1",
        "alt_name": "Neurofibromin 1",
        "protein": (
            "NF1 -- 17q11.2 AD -- Neurofibromin1-2839aa -- "
            "NF1-Von-Recklinghausen-RAS-GAP-LOF-RAS-Constitutive -- "
            "Selumetinib-Koselugo-MEK1/2-FDA2020-FIRST-NF1-Therapy -- "
            "MPNST-8-13pct-Plexiform-Neurofibroma-Malignant-Transformation -- "
            "Cafe-au-Lait-6-or-more-NIH-Criterion-Lisch-Nodules-PATHOGNOMONIC-Adults"
        ),
        "locus": "17q11.2",
        "protein_size": "2839 aa",
        "inheritance": "AD",
        "age_of_onset": "Variable; café-au-lait macules birth–2 yr; neurofibromas puberty; MPNST adulthood",
        "key_biomarker": (
            "NIH diagnostic criteria: ≥6 café-au-lait macules (≥5 mm prepubertal; ≥15 mm postpubertal); "
            "Lisch nodules on slit-lamp; ≥2 neurofibromas or ≥1 plexiform; "
            "optic glioma 15–20% (MRI orbit/brain); axillary/inguinal freckling; osseous lesions (sphenoid dysplasia); "
            "first-degree relative with NF1 (seventh criterion); MPNST: rapid growth + pain in neurofibroma"
        ),
        "pathognomonic": (
            "Lisch nodules (melanocytic iris hamartomas) — slit-lamp PATHOGNOMONIC in adults (>6 yr); "
            "bilateral optic pathway glioma = NF1 until proven otherwise; "
            "plexiform neurofibroma on MRI/exam — ONLY in NF1 (not NF2/schwannomatosis); "
            "≥2 neurofibromas + ≥6 CALMs = NIH 1987 clinical diagnosis"
        ),
        "treatment": (
            "Selumetinib (Koselugo) — oral MEK1/2 inhibitor — FDA2020 for inoperable plexiform neurofibromas (age ≥2 yr, RECIST criteria); "
            "first disease-modifying NF1-approved therapy; "
            "MEK162 (binimetinib) — trials ongoing; "
            "optic glioma: selumetinib preferred over carboplatin/vincristine as first-line (active growth); "
            "MPNST: R0 resection if feasible; chemotherapy (doxorubicin/ifosfamide) limited response; "
            "no approved therapy for MPNST; "
            "surveillance: annual ophthalmology (till age 8), annual BP (renal artery stenosis), "
            "annual skin exam, MRI whole-body for plexiform (every 3 yr)"
        ),
        "critical_flags": [
            "NF1-SELUMETINIB-FDA2020-FIRST-APPROVED: MEK1/2 inhibitor for inoperable plexiform NF; first NF1 disease-modifying approval",
            "NF1-MPNST-8-13pct: plexiform NF → MPNST malignant transformation; rapid growth + pain = biopsy emergency",
            "NF1-OPTIC-GLIOMA-15pct: mostly asymptomatic; screen till age 8; selumetinib if growing/vision loss",
            "NF1-LISCH-NODULES-PATHOGNOMONIC: slit-lamp only; present >6 yr; do NOT use in children <6 yr for diagnosis",
            "NF1-JMML-CHILDREN: juvenile myelomonocytic leukemia NF1 children (monosomy 7); diagnose early with CBC",
            "NF1-RENAL-ARTERY-STENOSIS: annual BP from childhood; refractory HT in young NF1 → renal artery imaging",
            "NF1-RADIATION-INCREASE-MPNST: RT for optic glioma dramatically increases secondary MPNST; prefer chemo/selumetinib first",
            "NF1-CAFE-AU-LAIT-6-RULE: ≥6 CALMs ≥5 mm (prepubertal) or ≥15 mm (postpubertal) = ONE NIH criterion; alone insufficient",
        ],
        "alias": (
            "NF1 (neurofibromin 1; 2839 aa; 17q11.2) encodes neurofibromin, a RAS GTPase-activating protein (RAS-GAP). "
            "LOF → RAS-GTP accumulates → constitutive MAPK/ERK and PI3K/mTOR activation. "
            "NF1 (Von Recklinghausen disease; OMIM #162200) is the most common single-gene "
            "tumor-predisposition syndrome — prevalence 1:2,000–3,000, de novo 50%. "
            "NIH 1987 criteria (≥2 of 7): CALMs, neurofibromas, plexiform, Lisch nodules, optic glioma, "
            "axillary freckling, osseous lesion, first-degree relative. "
            "Selumetinib (Koselugo, MEK1/2 inhibitor, FDA 2020) is the FIRST approved NF1-directed therapy — "
            "targets plexiform neurofibromas (inoperable, ≥2 yr). "
            "MPNST (malignant peripheral nerve sheath tumor) is the leading cause of NF1 mortality (8–13% lifetime); "
            "arises from plexiform neurofibromas; R0 resection curative; poor chemo response. "
            "Optic pathway glioma (OPG): 15–20% NF1, mostly pilocytic astrocytoma, usually asymptomatic; "
            "treatment threshold = visual loss or tumour growth on serial MRI. "
            "JMML: NF1 children with somatic KRAS/PTPN11 second hit → monosomy 7 JMML (allogeneic HSCT). "
            "Cardiovascular: renal artery stenosis (HT in young patients), pulmonic stenosis, coronary artery aneurysms. "
            "Surveillance: ophthalmology annually to age 8; BP annually; skin annually; "
            "whole-body MRI every 3 yr for plexiform monitoring."
        ),
    },

    # -- NF2 — Neurofibromatosis type 2 / Bilateral vestibular schwannoma syndrome ---
    {
        "gene": "NF2",
        "alt_name": "Merlin / Schwannomin",
        "protein": (
            "NF2 -- 22q12.2 AD -- Merlin-595aa -- "
            "NF2-Bilateral-Vestibular-Schwannomas-PATHOGNOMONIC-95pct-Penetrance -- "
            "Bevacizumab-VEGF-Reduction-VS-Volume-Hearing-Preservation -- "
            "Meningiomas-50-75pct-Ependymomas-Spinal-Schwannomas -- "
            "Hearing-Loss-First-Symptom-Age-18-24yr"
        ),
        "locus": "22q12.2",
        "protein_size": "595 aa",
        "inheritance": "AD",
        "age_of_onset": "18–24 yr (mean first symptom: unilateral hearing loss); constitutional onset younger in mosaic",
        "key_biomarker": (
            "MRI contrast brain + IACs (internal auditory canals) — bilateral vestibular schwannomas = diagnosis; "
            "audiometry: bilateral high-frequency SNHL ± tinnitus ± disequilibrium; "
            "gadolinium MRI spine: ependymomas, meningiomas, spinal schwannomas; "
            "VEGF in plasma (bevacizumab monitoring); "
            "Manchester criteria: bilateral VS = NF2 confirmed regardless of age"
        ),
        "pathognomonic": (
            "Bilateral vestibular schwannomas — Manchester/NIH criteria = NF2 CONFIRMED (95% penetrance); "
            "MRI enhancing masses at bilateral IACs/CPA (cerebellopontine angle); "
            "first-degree relative NF2 + unilateral VS age <30 yr = probable NF2"
        ),
        "treatment": (
            "Bevacizumab (anti-VEGF) — reduces VS volume by >20% in 40–55% (hearing improvement 20–30%); "
            "off-label but FIRST effective systemic therapy for progressive VS; "
            "stereotactic radiosurgery (Gamma Knife) — VS ≤3 cm, good hearing: growth control 90%; "
            "microsurgical resection — large VS, brainstem compression, failed Gamma Knife; "
            "hearing preservation: early intervention (auditory brainstem implant if total loss); "
            "everolimus/MEK inhibitors — clinical trials; "
            "annual MRI brain + spine; audiometry 6–12 monthly; ophthalmology (posterior subcapsular cataract 80%)"
        ),
        "critical_flags": [
            "NF2-BILATERAL-VS-PATHOGNOMONIC: bilateral vestibular schwannomas = NF2 diagnosis regardless of family history",
            "NF2-BEVACIZUMAB-VEGF: off-label anti-VEGF; 40-55% shrinkage; FIRST effective systemic NF2 therapy",
            "NF2-HEARING-LOSS-FIRST: unilateral SNHL age 18-24yr = NF2 until proven; bilateral MRI IACs mandatory",
            "NF2-MENINGIOMA-50pct: multiple meningiomas ANY location; spinal ependymoma grade II most common spinal tumour",
            "NF2-POSTERIOR-SUBCAPSULAR-CATARACT-80pct: ophthalmology from diagnosis; PSC + young NF2 = diagnostic clue",
            "NF2-RADIOSURGERY-VS: Gamma Knife for VS ≤3cm; hearing preservation; preferred over open surgery (smaller tumours)",
            "NF2-NOT-NF1: no Lisch nodules, no neurofibromas, no café-au-lait criterion; DO NOT conflate",
            "NF2-SCHWANNOMATOSIS-DDx: schwannomatosis (LZTR1/SMARCB1) = multiple schwannomas WITHOUT bilateral VS",
        ],
        "alias": (
            "NF2 (merlin/schwannomin; 595 aa; 22q12.2) encodes merlin, an ERM (ezrin-radixin-moesin) "
            "family FERM-domain tumor suppressor linking actin cytoskeleton to membrane receptors. "
            "LOF → loss of contact inhibition; uncontrolled Schwann cell, meningeal, ependymal proliferation. "
            "NF2 (OMIM #101000): bilateral vestibular schwannomas — 95% penetrance by age 60 yr; "
            "Manchester criteria: bilateral VS on MRI = definitive. "
            "Additional tumors: multiple meningiomas (50–75%), spinal ependymomas (grade II, most common spinal cord tumor in NF2), "
            "peripheral/cranial nerve schwannomas. "
            "Bevacizumab (anti-VEGF, off-label): most effective systemic therapy; "
            "VEGF drives VS vascularity; response: >20% volume reduction 40–55%, hearing preservation 20–30%; "
            "maintenance dosing 7.5 mg/kg every 3 weeks. "
            "Surgical: microsurgery or stereotactic radiosurgery; hearing restoration — auditory brainstem implant when deafened. "
            "Differentiate from schwannomatosis (LZTR1/SMARCB1): multiple schwannomas WITHOUT bilateral VS; "
            "and from sporadic unilateral VS (no NF2 germline). "
            "Posterior subcapsular cataract (80% of NF2 patients): ophthalmological screening mandatory. "
            "Mosaic NF2: milder, segmental; germline testing may be negative — deep sequencing of blood + tumor."
        ),
    },

    # -- TSC1 — Tuberous Sclerosis Complex type 1 (Hamartin) -------------------------
    {
        "gene": "TSC1",
        "alt_name": "Hamartin",
        "protein": (
            "TSC1 -- 9q34.13 AD -- Hamartin-1164aa -- "
            "TSC1-mTORC1-Inhibitor-Rheb-GTPase-TSC1/TSC2-Heterodimer -- "
            "Cardiac-Rhabdomyoma-PATHOGNOMONIC-Fetus-Neonate -- "
            "Everolimus-Afinitor-mTOR-Inhibitor-SEGA-AML-LAM -- "
            "Milder-Than-TSC2-Less-LAM-Less-Severe-ID"
        ),
        "locus": "9q34.13",
        "protein_size": "1164 aa",
        "inheritance": "AD",
        "age_of_onset": "Cardiac rhabdomyoma in utero; seizures infancy (infantile spasms); skin lesions childhood",
        "key_biomarker": (
            "Cardiac echo: rhabdomyoma (pathognomonic in fetus/neonate — 90% of multiple cardiac rhabdomyomas = TSC); "
            "brain MRI: cortical tubers, SENs (subependymal nodules), SEGA (enhancing, >1 cm near foramen of Monro); "
            "renal US/MRI: AML (angiomyolipoma) — annual; "
            "skin: ash-leaf macules (Wood's lamp), facial angiofibromas, shagreen patch, periungual fibromas; "
            "EEG: infantile spasms hypsarrhythmia → vigabatrin FIRST-LINE (TSC-specific: better than ACTH)"
        ),
        "pathognomonic": (
            "Cardiac rhabdomyoma in fetus/neonate — 90% of multiple rhabdomyomas = TSC; "
            "facial angiofibromas (adenoma sebaceum) + ash-leaf macules = major criteria; "
            "SEGA: enhancing subependymal lesion ≥1 cm near foramen of Monro = SEGA until proven otherwise"
        ),
        "treatment": (
            "Everolimus (Afinitor) — mTOR inhibitor; FDA-approved for: SEGA, renal AML ≥3 cm, TSC-associated LAM; "
            "sirolimus (rapamycin) — LAM first-line (everolimus alternative); "
            "vigabatrin FIRST-LINE for TSC-related infantile spasms (superior to ACTH in TSC-IS); "
            "SEGA: everolimus ORR 35% shrinkage; surgery if acute hydrocephalus; "
            "AML: embolization if >4 cm or symptomatic (hemorrhage risk); everolimus reduces AML volume; "
            "LAM: sirolimus — MILES trial — stabilises FEV1 decline; "
            "surveillance: annual renal MRI/US, annual brain MRI (SEGA growth), annual pulmonary LFTs (women)"
        ),
        "critical_flags": [
            "TSC1-CARDIAC-RHABDOMYOMA-PATHOGNOMONIC: multiple cardiac rhabdomyomas in fetus/neonate = 90% TSC; order TSC genetic testing antenatally",
            "TSC1-VIGABATRIN-IS-FIRST-LINE: infantile spasms in TSC → vigabatrin SUPERIOR to ACTH; reverse of non-TSC IS",
            "TSC1-EVEROLIMUS-SEGA-AML-LAM: mTOR inhibitor FDA approved for all three TSC manifestations; lifelong therapy",
            "TSC1-AML-EMBOLIZE-GT4CM: renal AML >4cm = embolization threshold (hemorrhage risk); everolimus reduces size",
            "TSC1-MILDER-THAN-TSC2: TSC1 de novo 30%; TSC2 de novo 66-80%; TSC2 more severe ID and more cortical tubers",
            "TSC1-LAM-LESS-THAN-TSC2: LAM occurs in TSC2 >> TSC1; all TSC women need pulmonary surveillance",
            "TSC1-SEGA-MONITOR: serial brain MRI; SEGA >1cm near foramen of Monro = obstructive hydrocephalus risk",
            "TSC1-SKIN-WOOD-LAMP: ash-leaf macules (hypopigmented) best seen under Wood's lamp; first sign in neonates",
        ],
        "alias": (
            "TSC1 (hamartin; 1164 aa; 9q34.13) forms the TSC1–TSC2 heterodimer complex (hamartin–tuberin), "
            "which functions as a GAP for the GTPase Rheb, inhibiting mTORC1 activation. "
            "LOF → constitutive mTORC1 → excessive cell growth, protein synthesis, angiogenesis → hamartomatous lesions. "
            "Tuberous Sclerosis Complex (TSC; OMIM #191100 for TSC1): autosomal dominant; 1:6,000–10,000; "
            "de novo 30% (TSC1) vs 66–80% (TSC2). "
            "Cardiac rhabdomyoma: pathognomonic in fetus/neonate; 90% of multiple rhabdomyomas = TSC; "
            "regress spontaneously in most cases after birth — no resection unless obstructive. "
            "Cortical tubers: epileptogenic foci; worse count correlates with worse intellectual disability. "
            "SEGA (subependymal giant cell astrocytoma): near foramen of Monro; WHO grade I; "
            "everolimus shrinks 35% ORR (FDA-approved); surgery if hydrocephalus. "
            "Infantile spasms: vigabatrin is FIRST-LINE in TSC-IS (superior to ACTH — TSC-specific exception); "
            "MACS-UK trial confirmed. "
            "LAM: cystic lung destruction predominantly in TSC2; sirolimus stabilises FEV1 (MILES trial, NEJM 2011); "
            "renal AML: embolize if >4 cm (hemorrhage); everolimus reduces volume. "
            "Skin: ash-leaf macules (first sign, Wood's lamp), facial angiofibromas (adenoma sebaceum, puberty), "
            "shagreen patch (lumbar collagen), periungual fibromas (pathognomonic post-puberty)."
        ),
    },

    # -- TSC2 — Tuberous Sclerosis Complex type 2 (Tuberin) -------------------------
    {
        "gene": "TSC2",
        "alt_name": "Tuberin",
        "protein": (
            "TSC2 -- 16p13.3 AD -- Tuberin-1807aa -- "
            "TSC2-MORE-SEVERE-Than-TSC1-Higher-De-Novo-66-80pct -- "
            "LAM-Predominantly-TSC2-Sirolimus-Everolimus-MILES-Trial -- "
            "Renal-AML-GT4cm-Embolization-Threshold-Hemorrhage-Risk -- "
            "TSC2-PKD1-Contiguous-Deletion-Early-Polycystic-Kidney"
        ),
        "locus": "16p13.3",
        "protein_size": "1807 aa",
        "inheritance": "AD",
        "age_of_onset": "In utero cardiac rhabdomyoma; infantile spasms 3–12 months; more severe than TSC1",
        "key_biomarker": (
            "Higher cortical tuber count than TSC1 (correlates with worse epilepsy/ID); "
            "LAM: HRCT chest — bilateral cystic lung disease (women >40 yr predominantly TSC2); "
            "VEGF-D serum elevated in LAM — diagnostic marker; "
            "renal AML: annual renal MRI; >4 cm = hemorrhage threshold; "
            "PKD1 proximity: TSC2 at 16p13.3 adjacent to PKD1 — contiguous deletion → severe childhood PKD"
        ),
        "pathognomonic": (
            "LAM on HRCT + elevated serum VEGF-D in TSC2 = diagnostic without biopsy; "
            "TSC2/PKD1 contiguous deletion: early onset (childhood) polycystic kidney — unique to TSC2 locus; "
            "multiple cortical tubers (>5 on brain MRI) in infant = strongly TSC2"
        ),
        "treatment": (
            "Sirolimus (rapamycin) FIRST-LINE for LAM — MILES trial (NEJM 2011): stabilises FEV1 decline; lifelong; "
            "everolimus — FDA-approved alternative for LAM; also SEGA, AML; "
            "vigabatrin first-line for TSC2-related infantile spasms (same as TSC1); "
            "lung transplantation for end-stage LAM (recurrence low with sirolimus post-transplant); "
            "renal AML: embolization >4 cm; everolimus reduces size; "
            "TSC2/PKD1 syndrome: mTOR inhibition may reduce kidney cyst growth; nephrology co-management; "
            "surveillance: CT chest annually (women with TSC2), renal MRI annually, brain MRI annually"
        ),
        "critical_flags": [
            "TSC2-MORE-SEVERE-THAN-TSC1: higher de novo rate (66-80%); worse epilepsy; more cortical tubers; higher ID rate",
            "TSC2-LAM-PREDOMINANT: LAM (cystic lung disease) predominantly in TSC2; sirolimus first-line (MILES trial NEJM 2011)",
            "TSC2-VEGF-D-LAM-MARKER: serum VEGF-D >800 pg/mL in TSC2-LAM = diagnostic without lung biopsy (VEGF-D not elevated in other cystic lung diseases)",
            "TSC2-PKD1-CONTIGUOUS: TSC2 and PKD1 both on 16p13; contiguous deletion → severe early childhood PKD (must check renal US)",
            "TSC2-AML-GT4CM-BLEED: renal AML >4cm = Wunderlich syndrome hemorrhage risk; embolization or nephron-sparing surgery",
            "TSC2-SIROLIMUS-LIFELONG: stopping sirolimus → LAM rebounds; lifelong therapy required",
            "TSC2-LUNG-TRANSPLANT: end-stage LAM → bilateral lung transplant; sirolimus continuation reduces recurrence",
            "TSC2-IS-VIGABATRIN: infantile spasms → vigabatrin FIRST (same principle as TSC1); do NOT start ACTH first in TSC",
        ],
        "alias": (
            "TSC2 (tuberin; 1807 aa; 16p13.3) encodes the tuberin GTPase-activating protein, which dimerises with hamartin (TSC1) "
            "to inhibit Rheb and thereby suppress mTORC1. "
            "TSC2 (OMIM #613254) is MORE SEVERE than TSC1: higher de novo mutation rate (66–80%), "
            "higher cortical tuber burden, worse epilepsy, more severe intellectual disability. "
            "LAM (lymphangioleiomyomatosis): bilateral cystic lung destruction; almost exclusively in women; "
            "predominantly TSC2; serum VEGF-D >800 pg/mL is diagnostic (obviates biopsy); "
            "sirolimus (MILES trial, NEJM 2011) or everolimus stabilises FEV1; lifelong therapy — rebound on discontinuation. "
            "End-stage LAM: bilateral lung transplant; sirolimus reduces post-transplant recurrence. "
            "TSC2/PKD1 contiguous gene syndrome (16p13.3): severe early-onset polycystic kidney disease — "
            "unique to TSC2 locus patients; aggressive renal monitoring from childhood. "
            "Renal AML: >4 cm = hemorrhage threshold (Wunderlich syndrome — spontaneous perirenal hemorrhage); "
            "embolization preferred over nephrectomy; everolimus reduces size. "
            "SEGA and infantile spasms management: identical to TSC1 (everolimus for SEGA; vigabatrin for IS). "
            "mTOR pathway: TSC2 is the catalytic GAP subunit (tuberin's GAP domain targets Rheb); "
            "TSC1 is the scaffolding subunit — both required for full activity."
        ),
    },

    # -- VHL — Von Hippel-Lindau syndrome -------------------------------------------
    {
        "gene": "VHL",
        "alt_name": "pVHL / Von Hippel-Lindau tumour suppressor",
        "protein": (
            "VHL -- 3p25.3 AD -- pVHL-213aa -- "
            "VHL-E3-Ubiquitin-Ligase-HIF1a-HIF2a-Degradation-Oxygen-Sensing -- "
            "Clear-Cell-RCC-PATHOGNOMONIC-Hallmark-Malignancy -- "
            "Belzutifan-Welireg-HIF-2a-Inhibitor-FDA2021-FIRST-VHL-Targeted -- "
            "Retinal-Cerebellar-Spinal-Hemangioblastomas-Pheochromocytoma-Type2"
        ),
        "locus": "3p25.3",
        "protein_size": "213 aa",
        "inheritance": "AD",
        "age_of_onset": "Retinal hemangioblastoma 20s; RCC 30–40s; hemangioblastoma 30s; pheo (type 2) 20–30s",
        "key_biomarker": (
            "pVHL mutation: germline sequencing + MLPA (large deletions); "
            "annual MRI brain + spine (hemangioblastoma); annual ophthalmology (retinal haem.); "
            "annual renal MRI/US (RCC + RCC-related cysts); "
            "annual 24h urinary catecholamines/metanephrines (pheochromocytoma type 2); "
            "VEGF-A elevated (downstream of HIF-2α) in plasma"
        ),
        "pathognomonic": (
            "Clear cell RCC in young patient (<50 yr) or bilateral/multifocal RCC = VHL until proven otherwise; "
            "retinal hemangioblastoma = VHL most common cause of familial retinal hemangioma; "
            "cerebellar hemangioblastoma + retinal hemangioblastoma + RCC = VHL triad"
        ),
        "treatment": (
            "Belzutifan (Welireg) — oral HIF-2α (EPAS1) inhibitor — FDA2021 — FIRST VHL-specific targeted therapy; "
            "ORR: RCC 49%, hemangioblastoma 30%, PNET 91% in VHL-associated tumours; "
            "renal surgery: nephron-sparing for RCC when dominant lesion ≥3 cm (watch-and-wait <3 cm); "
            "laser/cryo for retinal hemangioblastoma (before vitreous haemorrhage); "
            "stereotactic radiosurgery or microsurgery for cerebellar/spinal hemangioblastoma; "
            "pheochromocytoma: alpha-blockade first (phenoxybenzamine) THEN beta-blockade, THEN adrenalectomy; "
            "NEVER beta-block first → hypertensive crisis; "
            "surveillance: annual ophthalmology, renal MRI, brain/spine MRI, catecholamines (type 2)"
        ),
        "critical_flags": [
            "VHL-BELZUTIFAN-FDA2021-FIRST: HIF-2α inhibitor; first VHL-specific targeted therapy; ORR RCC 49%, PNET 91%",
            "VHL-CLEAR-CELL-RCC-YOUNG: any young patient (<50yr) or bilateral/multifocal cRCC = VHL germline testing mandatory",
            "VHL-PHEO-ALPHA-FIRST: pheochromocytoma management: alpha-blockade (phenoxybenzamine 10mg bd) BEFORE beta-blockade — NEVER reverse order",
            "VHL-RETINAL-HEMANGIOBLASTOMA: annual ophthalmology; laser/cryo before vitreous haemorrhage; most common familial cause",
            "VHL-NEPHRON-SPARING: watch RCC <3cm; nephron-sparing surgery when ≥3cm; avoid total nephrectomy (bilateral disease)",
            "VHL-TYPE1-NO-PHEO: type 1 VHL (truncating, large deletion) = hemangioblastoma + RCC, NO pheochromocytoma",
            "VHL-TYPE2-PHEO: missense mutations; 2A (no RCC), 2B (+ RCC highest risk), 2C (pheo only); genotype-phenotype correlation",
            "VHL-PANCREATIC-CYSTS-PNET: pancreatic cysts (benign) + PNET (10-17%) in VHL; EUS + MRI surveillance",
        ],
        "alias": (
            "VHL (pVHL; 213 aa; 3p25.3) encodes the pVHL protein, the substrate-recognition subunit of an E3 ubiquitin ligase "
            "(VHL–elongin B/C–cullin 2 complex) that ubiquitinates HIF-1α and HIF-2α for proteasomal degradation in normoxia. "
            "LOF → HIF stabilisation → transcriptional activation of VEGF, PDGF, EPO, GLUT1 → tumour angiogenesis. "
            "VHL syndrome (OMIM #193300): autosomal dominant; 1:36,000; de novo 20%; penetrance >97% by age 65. "
            "Genotype–phenotype: type 1 (truncating/deletion): hemangioblastoma + clear cell RCC, NO pheochromocytoma; "
            "type 2 (missense): WITH pheochromocytoma — 2A (no RCC), 2B (+ RCC), 2C (pheo only). "
            "Clear cell RCC: pathognomonic hallmark; bilateral, multifocal; nephron-sparing if ≥3 cm; belzutifan for unresectable/progressive. "
            "Hemangioblastomas: retinal (most common familial cause) + cerebellar + spinal; "
            "SRS/microsurgery; serial MRI brain/spine annually. "
            "Belzutifan (Welireg, HIF-2α/EPAS1 inhibitor, FDA 2021): first VHL-directed systemic therapy; "
            "pivotal trial (LITESPARK-004): RCC ORR 49%, hemangioblastoma 30%, PNET 91%. "
            "Pheochromocytoma (type 2): alpha-blockade (phenoxybenzamine) BEFORE beta-blockade BEFORE surgery — "
            "reversing this order triggers lethal hypertensive crisis. "
            "Epididymal cystadenoma (men): 25–60%; rarely malignant; endolymphatic sac tumours: hearing loss. "
            "Annual surveillance protocol: ophthalmology, renal MRI, brain+spine MRI, 24h catecholamines (type 2), pancreas MRI."
        ),
    },

    # -- PTCH1 — Gorlin Syndrome / NBCCS (Hedgehog pathway) -------------------------
    {
        "gene": "PTCH1",
        "alt_name": "Patched-1",
        "protein": (
            "PTCH1 -- 9q22.32 AD -- Patched1-1447aa -- "
            "Gorlin-NBCCS-Hedgehog-Pathway-Receptor-SMO-Inhibition -- "
            "Multiple-BCCs-Age-LT20yr-PATHOGNOMONIC -- "
            "Calcified-Falx-Cerebri-Plain-XR-PATHOGNOMONIC -- "
            "RADIATION-ABSOLUTELY-CONTRAINDICATED-1000x-BCC-Induction -- "
            "Vismodegib-Erivedge-Sonidegib-SMO-Inhibitor-FDA-Approved"
        ),
        "locus": "9q22.32",
        "protein_size": "1447 aa",
        "inheritance": "AD",
        "age_of_onset": "BCCs from puberty (mean first BCC age 25 yr; multiple BCCs <20 yr pathognomonic); OKCs childhood",
        "key_biomarker": (
            "Diagnostic criteria (≥2 major or 1 major + 2 minor): "
            "multiple BCCs / BCC <20 yr; calcified falx cerebri; odontogenic keratocysts (OKCs); "
            "bifid/fused ribs (on CXR); bridged sella; first-degree relative with Gorlin; medulloblastoma (desmoplastic); "
            "palmar/plantar pits (2–3 mm); macrocephaly; frontal bossing; splayed/bifid ribs"
        ),
        "pathognomonic": (
            "Calcified falx cerebri on plain skull X-ray — PATHOGNOMONIC (present in 80% by age 20 yr); "
            "multiple basal cell carcinomas before age 20 yr — PATHOGNOMONIC; "
            "jaw OKCs: radiological multilocular radiolucency — high recurrence rate"
        ),
        "treatment": (
            "Vismodegib (Erivedge) — Smoothened inhibitor — FDA 2012 for locally advanced/metastatic BCC; "
            "first Hedgehog-pathway inhibitor approved; teratogenic — contraception mandatory; "
            "sonidegib (Odomzo) — alternative SMO inhibitor; "
            "photodynamic therapy (PDT) — multiple small superficial BCCs; "
            "sun protection: lifelong strict sunscreen + protective clothing; "
            "RADIATION ABSOLUTELY CONTRAINDICATED — dramatically increases BCC number (1,000×) in radiation field; "
            "medulloblastoma: if radiation required → PROTON THERAPY MANDATORY (minimises RT field/dose); "
            "OKC resection: enucleation with curettage; Carnoy's solution (recurrence reduction); marsupialization; "
            "annual dermatology, annual jaw OPG, annual ophthalmology, brain MRI (paediatric)"
        ),
        "critical_flags": [
            "PTCH1-RADIATION-ABSOLUTELY-CI: ANY radiation (including medulloblastoma RT) → 1000× BCC induction in radiation field; proton therapy mandatory if RT needed",
            "PTCH1-CALCIFIED-FALX-PATHOGNOMONIC: plain skull XR; calcified falx 80% by age 20; no other condition calcifies falx this early",
            "PTCH1-BCC-LT20yr-PATHOGNOMONIC: multiple BCCs before age 20 = Gorlin syndrome until proven; PTCH1 sequencing mandatory",
            "PTCH1-VISMODEGIB-TERATOGENIC: SMO inhibitor contraindicated pregnancy; strict contraception (male and female) during + 3 months post",
            "PTCH1-OKC-HIGH-RECURRENCE: odontogenic keratocysts in jaw; recur after simple enucleation; Carnoy's solution reduces recurrence",
            "PTCH1-MEDULLOBLASTOMA-DESMOPLASTIC: 5% risk; typically desmoplastic/nodular histology (different from sporadic); age <5yr",
            "PTCH1-SMO-CONSTITUTIVE: PTCH1 LOF → SMO constitutively active → Gli transcription → BCC; SMO inhibitor reverses this",
            "PTCH1-PALMAR-PITS: palmar/plantar pitting (2-3mm) = minor diagnostic criterion; helpful in equivocal cases",
        ],
        "alias": (
            "PTCH1 (Patched-1; 1447 aa; 9q22.32) encodes the transmembrane Hedgehog-pathway receptor. "
            "In the absence of Hedgehog ligand, PTCH1 inhibits Smoothened (SMO), preventing Gli transcription. "
            "Hedgehog binding → PTCH1 internalisation → SMO constitutively active → Gli nuclear → BCC proliferation. "
            "LOF PTCH1 → SMO perpetually uninhibited → basal cell carcinoma. "
            "Gorlin syndrome (NBCCS; OMIM #109400): autosomal dominant; 1:30,000–60,000; de novo 20–30%. "
            "Multiple BCCs: lifetime risk near 100% (fair-skinned Europeans); hundreds of BCCs in a lifetime without treatment; "
            "BCCs before age 20 yr is pathognomonic. "
            "Calcified falx cerebri: 80% by age 20; plain skull X-ray; bridged sella, bifid ribs. "
            "Odontogenic keratocysts (OKCs): jaw; multilocular radiolucency; recur after simple enucleation; "
            "Carnoy's solution (zinc chloride fixative) reduces recurrence; marsupialization for large cysts. "
            "Medulloblastoma (desmoplastic/nodular): 5% risk, age <5 yr — RADIATION ABSOLUTELY CONTRAINDICATED "
            "(1,000× BCC induction in radiation field); PROTON THERAPY is mandatory if radiation required. "
            "Vismodegib (Erivedge, FDA 2012): SMO inhibitor; locally advanced/metastatic BCC response rate ~48%; "
            "TERATOGENIC (FDA category X); strict contraception male + female during + 3 months after cessation. "
            "Sun avoidance/photoprotection: lifelong; UVA + UVB protection essential. "
            "Annual surveillance: dermatology (full skin exam), dental OPG (OKCs), ophthalmology (calcific cataracts 5%)."
        ),
    },

    # -- STK11 — Peutz-Jeghers Syndrome (LKB1) -------------------------------------
    {
        "gene": "STK11",
        "alt_name": "LKB1 (Serine/threonine kinase 11)",
        "protein": (
            "STK11 -- 19p13.3 AD -- LKB1-433aa -- "
            "Peutz-Jeghers-Syndrome-PJS-AMPK-Activation-Energy-Sensing -- "
            "Mucocutaneous-Lentigines-Lips-Buccal-Digits-PATHOGNOMONIC -- "
            "GI-Hamartomatous-Polyps-Small-Bowel-Intussusception-Emergency -- "
            "Lifetime-Cancer-Risk-76pct-Breast-45pct-Pancreatic-Cervix-GI"
        ),
        "locus": "19p13.3",
        "protein_size": "433 aa",
        "inheritance": "AD",
        "age_of_onset": "Lentigines birth–5 yr; GI polyps childhood; cancer risk cumulative from 3rd decade",
        "key_biomarker": (
            "Mucocutaneous lentigines: perioral (lips), buccal, fingers/toes, perinasal (birth–5 yr); "
            "fade post-puberty EXCEPT buccal; "
            "GI endoscopy: hamartomatous polyps (small bowel most common → intussusception) + gastric + colorectal; "
            "small bowel MRI/capsule endoscopy: surveillance from age 8 yr; "
            "STK11 sequencing (90% sensitive) + MLPA for large deletions; "
            "CA 19-9 + MRI pancreas: pancreatic cancer surveillance from age 30–35 yr"
        ),
        "pathognomonic": (
            "Mucocutaneous lentigines (perioral + buccal + digits) = PJS until proven otherwise; "
            "buccal lentigines persist post-puberty (pathognomonic differentiator from non-syndrome lentigines); "
            "≥2 PJS polyps histologically (hamartomatous with smooth muscle core) in a patient = PJS diagnosis"
        ),
        "treatment": (
            "No disease-modifying therapy; surveillance is the intervention; "
            "polyp management: polypectomy when ≥1 cm or symptomatic; double-balloon enteroscopy for small bowel; "
            "intussusception: emergency reduction/resection (acute abdomen); "
            "breast: annual MRI from age 25–30; prophylactic mastectomy discussion at 40–45 if not surveillance compliant; "
            "pancreas: annual EUS + MRI pancreas from age 30–35; "
            "cervix: pap smear + HPV; gynaecological surveillance minimal deviation adenocarcinoma; "
            "colorectal: colonoscopy from age 8–10; "
            "rapamycin/mTOR inhibition: preclinical benefit, no proven clinical protocol; "
            "surveillance calendar: small bowel MRI/capsule (age 8, then 2–3 yr); colonoscopy/OGD (age 8, then 2–3 yr)"
        ),
        "critical_flags": [
            "STK11-LENTIGINES-PATHOGNOMONIC: mucocutaneous lentigines on lips/buccal/digits from birth; buccal persists post-puberty (distinguish from non-syndrome)",
            "STK11-INTUSSUSCEPTION-EMERGENCY: small bowel hamartomatous polyps → intussusception; acute abdomen in PJS = emergency laparotomy/laparoscopy",
            "STK11-BREAST-45pct-LIFETIME: breast cancer risk 45%; same as BRCA2; MRI from age 25-30; discuss risk-reduction options",
            "STK11-PANCREATIC-HIGH-RISK: 11-36% lifetime pancreatic cancer; annual EUS + MRI from age 30-35; early detection crucial",
            "STK11-CERVIX-MINIMAL-DEVIATION: minimal deviation adenocarcinoma (adenoma malignum) unique to PJS; aggressive; regular gynaecological review",
            "STK11-SMALL-BOWEL-MRI-FROM-AGE8: capsule endoscopy or small bowel MRI from age 8; polyps can cause telescoping at any age",
            "STK11-76pct-LIFETIME-CANCER: cumulative cancer risk 76% by age 70; multi-organ surveillance protocol mandatory from childhood",
            "STK11-AMPK-LKB1-ENERGY: STK11/LKB1 activates AMPK (cellular energy sensor); LOF → unchecked cell growth in energy-deprived states",
        ],
        "alias": (
            "STK11 (LKB1; 433 aa; 19p13.3) encodes serine/threonine kinase 11 (LKB1), "
            "a master kinase that phosphorylates and activates AMPK and related kinases, "
            "coupling energy sensing to mTOR suppression. "
            "LOF → uncontrolled mTOR → hamartoma formation + cancer predisposition. "
            "Peutz-Jeghers Syndrome (PJS; OMIM #175200): autosomal dominant; 1:50,000–200,000; de novo 25%. "
            "Diagnostic: ≥2 histologically confirmed PJS polyps; or any PJS polyp + STK11 germline; "
            "or any number of PJS polyps + family history; or mucocutaneous lentigines + family history. "
            "Lentigines: perioral (lips), buccal mucosa, digits — PATHOGNOMONIC; "
            "buccal lentigines PERSIST post-puberty (unlike other lentigo types that fade); "
            "perioral fade post-puberty. "
            "GI polyps: hamartomatous with arborising smooth muscle (Peutz-Jeghers polyp on histology); "
            "small bowel most common site → intussusception (acute abdomen — emergency enteroscopy/laparotomy). "
            "Cancer risks (cumulative to age 70): any cancer 76%; breast 45%; colorectal 39%; pancreatic 11–36%; "
            "gastric 29%; small bowel 13%; lung 15%; cervical (minimal deviation adenocarcinoma — HPV-negative) 10%; "
            "ovarian (sex cord tumours with annular tubules, SCTAT) 21%. "
            "Surveillance protocol: colonoscopy + OGD from age 8–10, repeat every 2–3 yr; "
            "small bowel MRI/capsule endoscopy from age 8; breast MRI from age 25; "
            "EUS + MRI pancreas from age 30–35; gynaecological annually from menarche. "
            "No approved disease-modifying therapy; mTOR inhibitors in trials."
        ),
    },

    # -- PTEN — PHTS / Cowden Syndrome (PI3K/AKT/mTOR tumour suppressor) -----------
    {
        "gene": "PTEN",
        "alt_name": "Phosphatase and tensin homolog",
        "protein": (
            "PTEN -- 10q23.31 AD -- PTEN-403aa -- "
            "PHTS-Cowden-Syndrome-PI3K-AKT-mTOR-Pathway-Tumour-Suppressor -- "
            "Macrocephaly-HC-GT97th-Percentile-PATHOGNOMONIC -- "
            "Trichilemmomas-Follicular-Infundibulum-Histology-PATHOGNOMONIC -- "
            "Breast-Cancer-85pct-Lifetime-SAME-As-BRCA1-MRI-From-Age30 -- "
            "Thyroid-35pct-Endometrial-28pct-Everolimus-mTOR-Trials"
        ),
        "locus": "10q23.31",
        "protein_size": "403 aa",
        "inheritance": "AD",
        "age_of_onset": "Macrocephaly at birth; mucocutaneous lesions young adult; cancers 30–50s",
        "key_biomarker": (
            "Macrocephaly: HC ≥58 cm in adult female (97th percentile) — PATHOGNOMONIC; "
            "trichilemmoma (follicular infundibulum hamartoma) on skin biopsy — PATHOGNOMONIC; "
            "mucocutaneous lesions: papillomatous papules (cobblestone tongue/gingiva), acral keratoses, "
            "oral mucosal papillomatosis; "
            "thyroid US + biopsy (follicular/papillary carcinoma 35%); "
            "PTEN sequencing + MLPA (large deletions in 10%); "
            "brain MRI: Lhermitte-Duclos (dysplastic cerebellar gangliocytoma — virtually PATHOGNOMONIC for PTEN)"
        ),
        "pathognomonic": (
            "Trichilemmoma (histologically confirmed) — PATHOGNOMONIC for Cowden syndrome (PHTS); "
            "macrocephaly (>97th percentile HC) = strong major criterion; "
            "Lhermitte-Duclos disease (dysplastic cerebellar gangliocytoma on MRI) = PATHOGNOMONIC for PTEN in adult"
        ),
        "treatment": (
            "Breast: MRI + mammography annually from age 30–35 (MRI preferred; dense breast); "
            "prophylactic bilateral mastectomy discussion (risk reduction >90%); "
            "thyroid: annual US; total thyroidectomy if high-risk lesion; "
            "endometrium: annual transvaginal US (consider endometrial biopsy from 30–35 yr); "
            "risk-reducing hysterectomy at completion of childbearing; "
            "colorectal: colonoscopy from age 35 every 5 yr; "
            "Everolimus (mTOR inhibitor) — compassionate use for inoperable PHTS-associated tumours; "
            "MATCH trial data support; "
            "ASD/ID: early neurodevelopmental intervention; "
            "surveillance: annual thyroid US, annual breast MRI, annual endometrial US/biopsy, "
            "colonoscopy 5-yearly, annual dermatology"
        ),
        "critical_flags": [
            "PTEN-BREAST-85pct-LIFETIME: breast cancer risk 85% — identical to BRCA1; MRI from age 30-35; do NOT miss this diagnosis",
            "PTEN-MACROCEPHALY-PATHOGNOMONIC: HC >97th percentile (≥58cm adult female) = PTEN testing immediately; most underdiagnosed syndrome",
            "PTEN-LHERMITTE-DUCLOS-PATHOGNOMONIC: dysplastic cerebellar gangliocytoma on MRI = PTEN in adult (virtually 100% specific)",
            "PTEN-TRICHILEMMOMA-HISTOLOGY: must have skin biopsy (cobblestone papules); histological trichilemmoma = major diagnostic criterion",
            "PTEN-THYROID-35pct: predominantly follicular + papillary thyroid cancer; annual US; total thyroidectomy for high-risk lesion",
            "PTEN-AUTISM-ASD: PTEN germline in children with ASD + macrocephaly 17-20%; screen PTEN in macrocephalic ASD",
            "PTEN-ENDOMETRIAL-28pct: risk-reducing hysterectomy at completion of childbearing; annual TVUS + biopsy from 30-35yr",
            "PTEN-MOST-UNDERDIAGNOSED: macrocephaly often not measured in adults; PHTS remains severely underdiagnosed — estimate 1:200,000 diagnosed vs 1:8,000-200,000 true prevalence",
        ],
        "alias": (
            "PTEN (phosphatase and tensin homolog; 403 aa; 10q23.31) encodes a dual-specificity lipid/protein phosphatase "
            "that dephosphorylates PIP3 (phosphatidylinositol-3,4,5-trisphosphate) to PIP2, "
            "opposing PI3K and suppressing AKT/mTOR signalling. "
            "LOF → unrestricted AKT/mTOR → cell growth, survival, proliferation → hamartoma + cancer. "
            "PTEN Hamartoma Tumor Syndrome (PHTS; OMIM #158350 Cowden / #153480 BRRS): "
            "autosomal dominant; prevalence 1:200,000 (diagnosed) — true prevalence likely 1:8,000–200,000; "
            "severely underdiagnosed — macrocephaly often not measured in adults. "
            "Clinical entities: Cowden syndrome (adult, cancer-dominant), Bannayan-Riley-Ruvalcaba (BRR; paediatric, vascular malformations + macrocephaly + penile lentigines), Lhermitte-Duclos. "
            "Macrocephaly: HC >97th percentile (≥58 cm adult female, ≥60 cm male); present from birth; "
            "PATHOGNOMONIC major criterion. "
            "Trichilemmoma: follicular infundibulum hamartoma; PATHOGNOMONIC on histology; cobblestone papules on face/oral mucosa. "
            "Lhermitte-Duclos disease: dysplastic cerebellar gangliocytoma; 'tiger-stripe' pattern on T2 MRI; "
            "virtually 100% specific for germline PTEN in adults. "
            "Cancer risks: breast 85% (same as BRCA1 — MRI + mammography from age 30–35; discuss prophylactic mastectomy); "
            "thyroid 35% (follicular > papillary); endometrial 28%; colorectal 9%; renal 34% (papillary). "
            "ASD + macrocephaly: PTEN germline in 17–20% → neurodevelopmental referral + PTEN testing. "
            "Everolimus (mTOR inhibitor): MATCH trial — PHTS tumours show responses; no approved indication yet. "
            "Annual surveillance: breast MRI, thyroid US, endometrial TVUS/biopsy (age ≥30), colonoscopy (age 35, 5-yearly), dermatology, neurology."
        ),
    },
]


def _make_cohort(gene_entry: dict, seed: int, n: int = 40) -> list:
    rng = random.Random(seed)
    # Age at diagnosis varies by gene and typical presentation
    mean_age = {
        "NF1":   8.0,   # café-au-lait early childhood; formal diagnosis ~8 yr
        "NF2":  22.0,   # first symptom (hearing loss) typically early 20s
        "TSC1":  1.0,   # infantile spasms / cardiac rhabdomyoma in infancy
        "TSC2":  0.8,   # more severe; earlier detection
        "VHL":  28.0,   # retinal/cerebellar hemangioblastoma 20s–30s
        "PTCH1":25.0,   # first BCC late teens/early 20s
        "STK11":12.0,   # lentigines from birth; first GI event childhood/teen
        "PTEN": 35.0,   # cancer diagnosis typically 30s–40s; macrocephaly from birth
    }.get(gene_entry["gene"], 20.0)
    sd_age = {
        "NF1":   4.0,
        "NF2":   5.0,
        "TSC1":  0.6,
        "TSC2":  0.4,
        "VHL":   7.0,
        "PTCH1": 8.0,
        "STK11": 6.0,
        "PTEN": 10.0,
    }.get(gene_entry["gene"], 5.0)
    ages = [round(rng.gauss(mean_age, sd_age), 1) for _ in range(n)]
    ages = [max(0.0, min(70.0, a)) for a in ages]
    sexes = [rng.choice(["M", "F"]) for _ in range(n)]
    # Severity distribution
    if gene_entry["gene"] in ("TSC1", "TSC2"):
        # severe infantile spasms in TSC2, moderate-severe in TSC1
        if gene_entry["gene"] == "TSC2":
            severities = [rng.choices(["moderate", "severe"], weights=[2, 3])[0] for _ in range(n)]
        else:
            severities = [rng.choices(["mild", "moderate", "severe"], weights=[2, 2, 1])[0] for _ in range(n)]
    elif gene_entry["gene"] == "NF2":
        # progressive hearing loss — mostly moderate
        severities = [rng.choices(["mild", "moderate", "severe"], weights=[2, 3, 1])[0] for _ in range(n)]
    elif gene_entry["gene"] == "PTCH1":
        # BCC burden varies widely
        severities = [rng.choices(["mild", "moderate", "severe"], weights=[3, 3, 1])[0] for _ in range(n)]
    else:
        severities = [rng.choice(["mild", "moderate", "severe"]) for _ in range(n)]
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
    for idx, g in enumerate(HAMARTOMA_GENES):
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
    for g in HAMARTOMA_GENES:
        gene_summary.append({
            "gene": g["gene"],
            "alt_name": g.get("alt_name", ""),
            "locus": g["locus"],
            "protein_size": g["protein_size"],
            "inheritance": g["inheritance"],
            "age_of_onset": g["age_of_onset"],
            "key_biomarker": g["key_biomarker"],
            "pathognomonic": g["pathognomonic"],
            "treatment": g["treatment"],
            "n_patients": gene_counts.get(g["gene"], 0),
            "critical_flags": g["critical_flags"],
        })

    return {
        "atlas": "Hereditary-Hamartoma-Syndrome-Atlas",
        "subtitle": (
            "Complete 8-Gene Hereditary Hamartoma & Phakomatosis Syndrome Atlas — "
            "NF1/Neurofibromin1-2839aa-17q11.2-AD-RAS-GAP-Selumetinib-MEK-FDA2020-MPNST-8-13pct | "
            "NF2/Merlin-595aa-22q12.2-AD-Bilateral-VS-PATHOGNOMONIC-Bevacizumab-VEGF | "
            "TSC1/Hamartin-1164aa-9q34.13-AD-Cardiac-Rhabdomyoma-PATHOGNOMONIC-Everolimus-mTOR | "
            "TSC2/Tuberin-1807aa-16p13.3-AD-MORE-SEVERE-LAM-Sirolimus-MILES-Trial | "
            "VHL/pVHL-213aa-3p25.3-AD-ClearCell-RCC-PATHOGNOMONIC-Belzutifan-HIF2a-FDA2021 | "
            "PTCH1/Patched1-1447aa-9q22.32-AD-Gorlin-Multiple-BCC-Radiation-ABSOLUTELY-CI-Vismodegib | "
            "STK11/LKB1-433aa-19p13.3-AD-Peutz-Jeghers-Lentigines-PATHOGNOMONIC-Breast-45pct | "
            "PTEN-403aa-10q23.31-AD-Cowden-Macrocephaly-PATHOGNOMONIC-Breast-85pct-Trichilemmoma | "
            "320-Patient-Aggregate-8x40-seeds-1910-1917"
        ),
        "aggregate_stats": {
            "total_patients": total,
            "genes_covered": len(HAMARTOMA_GENES),
            "avg_age_at_diagnosis_yr": avg_age,
            "severe_cases_pct": round(100 * severe_count / total, 1),
            "seed_range": f"{SEED_BASE}–{SEED_BASE + len(HAMARTOMA_GENES) - 1}",
        },
        "gene_summary": gene_summary,
        "key_clinical_distinctions": [
            "NF1-SELUMETINIB-FIRST-APPROVED: MEK1/2 inhibitor (Koselugo) FDA2020 — FIRST NF1-specific therapy; plexiform neurofibromas; no approved MPNST therapy",
            "NF2-BILATERAL-VS-PATHOGNOMONIC: bilateral vestibular schwannomas = NF2 confirmed; bevacizumab FIRST effective systemic therapy; no FDA-approved targeted drug",
            "TSC-VIGABATRIN-IS-REVERSAL: infantile spasms in TSC → vigabatrin SUPERIOR to ACTH (MACS trial); reverse of non-TSC IS where ACTH is preferred",
            "TSC2-MORE-SEVERE-TSC1: TSC2 higher de novo (66-80% vs 30%), worse epilepsy, more cortical tubers, more LAM; genotype guides prognosis",
            "VHL-BELZUTIFAN-FDA2021: HIF-2α inhibitor FIRST VHL-targeted therapy; ORR RCC 49%, PNET 91%; alpha-blockade BEFORE beta-blockade for pheo",
            "PTCH1-RADIATION-ABSOLUTELY-CI: RT in Gorlin → 1,000× BCC induction in field; proton therapy MANDATORY if medulloblastoma requires RT",
            "STK11-INTUSSUSCEPTION-EMERGENCY: small bowel hamartomatous polyps → intussusception; acute abdomen = emergency reduction; capsule endoscopy from age 8",
            "PTEN-BREAST-85pct-BRCA1-EQUIVALENT: PTEN breast risk identical to BRCA1 — manage identically; macrocephaly = screen PTEN urgently; most underdiagnosed syndrome",
            "VHL-NF1-PHEO-MANAGEMENT: pheo in VHL type 2 — alpha-blockade THEN beta; NF1 pheo (1-5% NF1) — same protocol; NEVER beta-first",
            "PTEN-ASD-MACROCEPHALY: PTEN germline 17-20% of macrocephalic ASD children; macrocephaly must be measured in ALL ASD evaluations",
        ],
    }


def breakdown() -> dict:
    result = []
    for idx, g in enumerate(HAMARTOMA_GENES):
        cohort = _make_cohort(g, SEED_BASE + idx)
        severities = {}
        for p in cohort:
            severities[p["severity"]] = severities.get(p["severity"], 0) + 1
        result.append({
            "gene": g["gene"],
            "alt_name": g.get("alt_name", ""),
            "protein": g["protein"],
            "locus": g["locus"],
            "protein_size": g["protein_size"],
            "inheritance": g["inheritance"],
            "age_of_onset": g["age_of_onset"],
            "key_biomarker": g["key_biomarker"],
            "pathognomonic": g["pathognomonic"],
            "treatment": g["treatment"],
            "critical_flags": g["critical_flags"],
            "severity_distribution": severities,
            "n_patients": len(cohort),
            "patients": cohort[:5],
        })
    return {"genes": result, "total_genes": len(HAMARTOMA_GENES)}


def definitions() -> dict:
    return {
        "atlas": "Hereditary-Hamartoma-Syndrome-Atlas",
        "genes": [
            {
                "gene": g["gene"],
                "alt_name": g.get("alt_name", ""),
                "definition": g["alias"],
                "locus": g["locus"],
                "protein_size": g["protein_size"],
                "inheritance": g["inheritance"],
                "age_of_onset": g["age_of_onset"],
                "critical_flags": g["critical_flags"],
            }
            for g in HAMARTOMA_GENES
        ],
        "glossary": {
            "NF1 (Neurofibromatosis type 1)": "Von Recklinghausen disease; RAS-GAP LOF; MOST common single-gene tumor predisposition; 1:2,000–3,000; selumetinib MEK1/2 inhibitor FDA2020 first approved therapy",
            "NF2 (Neurofibromatosis type 2)": "Bilateral vestibular schwannoma syndrome; merlin FERM tumor suppressor 22q12.2; 1:25,000–40,000; bevacizumab (anti-VEGF) first effective systemic therapy",
            "TSC (Tuberous Sclerosis Complex)": "TSC1 (hamartin) + TSC2 (tuberin) → Rheb GAP → mTORC1 inhibition; LOF → mTOR constitutive; 1:6,000–10,000; cardiac rhabdomyoma PATHOGNOMONIC in neonate; everolimus (FDA-approved SEGA/AML/LAM)",
            "VHL (Von Hippel-Lindau)": "E3 ubiquitin ligase targeting HIF-1α/HIF-2α; LOF → HIF stabilisation → VEGF; clear cell RCC + hemangioblastoma; belzutifan HIF-2α inhibitor FDA2021",
            "Gorlin syndrome (NBCCS)": "PTCH1 LOF → SMO constitutive → Hedgehog → BCC; calcified falx PATHOGNOMONIC; radiation ABSOLUTELY contraindicated (1,000× BCC); vismodegib SMO inhibitor",
            "Peutz-Jeghers Syndrome (PJS)": "STK11/LKB1 LOF → AMPK deficiency; mucocutaneous lentigines PATHOGNOMONIC; GI hamartomas → intussusception; lifetime cancer risk 76%; breast 45%",
            "PHTS (PTEN Hamartoma Tumor Syndrome)": "Cowden syndrome + BRRS + Lhermitte-Duclos; PTEN lipid phosphatase LOF → PI3K/AKT/mTOR; macrocephaly + trichilemmoma PATHOGNOMONIC; breast cancer 85% (= BRCA1)",
            "Hemangioblastoma": "Highly vascular WHO grade I tumour; retinal + cerebellar + spinal cord; VHL-associated; pVHL LOF → HIF → VEGF-A → tumour angiogenesis; belzutifan for unresectable",
            "Meningioma": "Dural meningeal tumour; NF2-associated (50–75%): multiple, atypical, location-unusual (spinal, orbital); NF2 LOF → contact inhibition loss; surgery or SRS",
            "SEGA (Subependymal Giant Cell Astrocytoma)": "WHO grade I glioma near foramen of Monro in TSC; enhancing on MRI ≥1 cm; risk obstructive hydrocephalus; everolimus shrinks 35% ORR (FDA approved)",
            "LAM (Lymphangioleiomyomatosis)": "Cystic lung destruction by smooth muscle-like cells harbouring TSC2 mutations; predominantly women; TSC2 >> TSC1; sirolimus stabilises FEV1 (MILES trial, NEJM 2011); lung transplant end-stage",
            "AML (Renal Angiomyolipoma)": "Benign mesenchymal renal tumour (fat + smooth muscle + blood vessels); TSC and VHL-associated; >4 cm = haemorrhage risk (Wunderlich syndrome); embolization preferred; everolimus reduces size",
            "Selumetinib (Koselugo)": "MEK1/2 inhibitor; FDA2020 for NF1-associated inoperable plexiform neurofibromas (age ≥2 yr); FIRST approved NF1 disease-modifying therapy; targets RAS-MAPK downstream of neurofibromin LOF",
            "Belzutifan (Welireg)": "HIF-2α (EPAS1) inhibitor; FDA2021 for VHL disease-associated tumours; FIRST VHL-targeted therapy; ORR: RCC 49%, hemangioblastoma 30%, PNET 91%",
            "Vismodegib (Erivedge)": "Smoothened inhibitor; FDA2012 for locally advanced/metastatic BCC; blocks Hh signalling downstream of PTCH1 LOF; TERATOGENIC (Category X); contraception mandatory",
            "MPNST (Malignant Peripheral Nerve Sheath Tumour)": "Malignant transformation of plexiform neurofibroma; 8–13% NF1 lifetime risk; rapid growth + pain in neurofibroma = biopsy emergency; poor prognosis; R0 resection best outcome",
            "Lhermitte-Duclos disease": "Dysplastic cerebellar gangliocytoma; 'tiger stripe' T2 signal on MRI; virtually 100% specific for germline PTEN mutation in adults; major diagnostic criterion for PHTS",
            "Odontogenic keratocyst (OKC)": "Jaw cyst in Gorlin syndrome; multilocular radiolucency on OPG; high recurrence after simple enucleation; Carnoy's solution reduces recurrence; pathognomonic jaw lesion of NBCCS",
            "Pheochromocytoma alpha-first rule": "VHL type 2 (and NF1) pheochromocytoma — alpha-blockade (phenoxybenzamine) BEFORE beta-blockade BEFORE surgery; reversing this order → lethal hypertensive crisis (beta-blockade alone → unopposed alpha → crisis)",
        },
    }


if __name__ == "__main__":
    import json
    print("=== HEREDITARY-HAMARTOMA-SYNDROME-ATLAS — OVERVIEW ===")
    print(json.dumps(overview(), indent=2)[:3000])
    print("\n=== BREAKDOWN (VHL — belzutifan gene) ===")
    bd = breakdown()
    vhl = next(g for g in bd["genes"] if g["gene"] == "VHL")
    print(json.dumps(vhl, indent=2)[:2000])
    print("\n=== DEFINITIONS (glossary sample) ===")
    df = definitions()
    print(json.dumps(list(df["glossary"].items())[:5], indent=2))
