#!/usr/bin/env python3
"""Hereditary-Thyroid-Disorder-Atlas — Complete 8-Gene Thyroid Atlas
(TSHR · DUOX2 · TPO · TG · SLC5A5 · FOXE1 · PAX8 · NKX2-1).

TSHR    (TSH receptor; 764 aa; 87 kDa; 14q31.1; AD GOF / AR LOF;
         GOF: Familial non-autoimmune hyperthyroidism (FNAH) + familial gestational hyperthyroidism;
         LOF: TSH resistance type II — normal gland but blunted TSH response;
         SUPPRESSED TSH + NEGATIVE TRAb/TPO-Ab = TSHR GOF PATHOGNOMONIC;
         methimazole long-term or total thyroidectomy — NOT radioiodine;
         seed SEED_BASE+0).
DUOX2   (Dual oxidase 2; 1548 aa; 175 kDa; 15q21.1; AR;
         MOST COMMON gene for congenital hypothyroidism from dyshormonogenesis (15-20% of CH);
         POSITIVE PERCHLORATE DISCHARGE TEST (>10%) PATHOGNOMONIC for biallelic DUOX2;
         TRANSIENT CH possible with monoallelic DUOX2;
         seed SEED_BASE+1).
TPO     (Thyroid peroxidase; 933 aa; 103 kDa; 2p25.3; AR;
         TOTAL organification defect — PERCHLORATE DISCHARGE >90% PATHOGNOMONIC;
         large goitre at birth; permanent CH (no transient cases);
         Pendred DDx (SLC26A4 — EVA on imaging + SNHL);
         seed SEED_BASE+2).
TG      (Thyroglobulin; 2768 aa; 330 kDa; 8q24.22; AR;
         CH + goitre + UNDETECTABLE/LOW serum TG (paradox) = TG mutation PATHOGNOMONIC;
         most common goitrous CH in iodine-sufficient countries;
         seed SEED_BASE+3).
SLC5A5  (Sodium-iodide symporter/NIS; 643 aa; 70 kDa; 19p13.11; AR;
         Iodide transport defect (ITD) — 123I thyroid uptake <5% at 24h PATHOGNOMONIC;
         saliva/serum iodide ratio <30; pertechnetate uptake absent;
         radioiodine NOT effective (NIS absent);
         seed SEED_BASE+4).
FOXE1   (Forkhead box E1/TTF-2; 373 aa; 42 kDa; 9q22.33; AR;
         Bamforth-Lazarus syndrome — THYROID AGENESIS + CLEFT PALATE + BIFID EPIGLOTTIS +
         SPIKY HAIR PATHOGNOMONIC TETRAD;
         neonatal emergency: hypoxia from choanal atresia + CH;
         seed SEED_BASE+5).
PAX8    (Paired box 8; 450 aa; 48 kDa; 2q13; AD;
         thyroid dysgenesis (athyreosis/hypoplasia/ectopy); VARIABLE EXPRESSIVITY;
         affected parent may be euthyroid (subclinical hypoplasia);
         ectopic lingual thyroid: do NOT excise;
         seed SEED_BASE+6).
NKX2-1  (NK2 homeobox 1/TTF-1/TITF1; 371 aa; 41 kDa; 14q13.3; AD;
         Brain-Lung-Thyroid (BLT) syndrome — CH + NEONATAL RDS + BENIGN HEREDITARY CHOREA
         PATHOGNOMONIC TRIAD; chorea onset 1-5 years; no cognitive decline;
         seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 2526-2533).
"""

import random

SEED_BASE = 2526

THYROID_GENES = [
    # -- TSHR -- Familial non-autoimmune hyperthyroidism / TSH resistance ----------------------
    {
        "gene": "TSHR",
        "alt_name": (
            "TSHR (TSHR-764aa-14q31.1 / AD-GOF / AR-LOF -- "
            "FNAH-FAMILIAL-NON-AUTOIMMUNE-HYPERTHYROIDISM-SUPPRESSED-TSH-NEGATIVE-AUTOANTIBODIES -- "
            "FGH-FAMILIAL-GESTATIONAL-HYPERTHYROIDISM-hCG-SENSITIVE -- "
            "TSH-RESISTANCE-LOF-NORMAL-GLAND-BLUNTED-TSH-RESPONSE -- "
            "METHIMAZOLE-LONG-TERM-NOT-RADIOIODINE-GOF-MUTATION-PERSISTS)"
        ),
        "protein": (
            "TSHR -- 14q31.1 AD GOF / AR LOF -- TSHR-764aa -- "
            "TSH-Receptor-87kDa-7TM-GPCR-Leucine-Rich-Ectodomain-LRR-Serpentine-Domain -- "
            "Gs-Alpha-cAMP-PKA-Stimulates-Thyroid-Hormone-Synthesis-Growth -- "
            "GOF-Constitutive-Receptor-Activation-Without-TSH-Binding-Ala623Ile-Val-Most-Common -- "
            "AR-LOF-TSH-Resistance-Type-II-Normal-Gland-Size-Blunted-TSH-Response -- "
            "Somatic-TSHR-GOF-In-Toxic-Adenoma-Same-Mechanism-As-Germline-FNAH -- "
            "FGH-Exquisite-hCG-Sensitivity-TSHR-Ectodomain-Variants-Low-Threshold-Activation -- "
            "OMIM-Gene-603372-Disease-FNAH-603373-TSH-Resistance-275200"
        ),
        "locus": "14q31.1",
        "protein_size": "764 aa / 87 kDa",
        "inheritance": (
            "AD GOF (familial non-autoimmune hyperthyroidism, FNAH) / AR LOF (TSH resistance type II); "
            "FNAH: TSH suppressed + hyperthyroid biochemistry + NEGATIVE TRAb + NEGATIVE TPO-Ab = "
            "TSHR activating mutation PATHOGNOMONIC — excludes Graves disease; "
            "FGH: exquisite hCG sensitivity — hyperemesis + hyperthyroidism in pregnancy with no prior thyroid disease; "
            "TSH resistance (LOF): TSH elevated/normal-high with normal or low FT4 — normal gland size; "
            "treat FNAH with methimazole (controls symptoms) or total thyroidectomy — NOT radioiodine "
            "(RAI does not eliminate the activating mutation); "
            "Ala623Ile and Ala623Val most common GOF variants; somatic TSHR GOF identical mechanism in toxic adenoma"
        ),
        "disease_category": (
            "FNAH (Familial Non-Autoimmune Hyperthyroidism) — AD GOF: suppressed TSH + elevated FT4/FT3 + "
            "negative TRAb + negative TPO-Ab; FGH (Familial Gestational Hyperthyroidism) — hCG-sensitive TSHR; "
            "TSH Resistance Type II — AR LOF: elevated TSH with normal/low FT4"
        ),
        "disease_pathway": (
            "TSHR (TSH receptor) is a 7TM G-protein coupled receptor (GPCR) expressed on thyroid follicular cells. "
            "Normally TSH binds the large leucine-rich ectodomain to trigger Gs-alpha coupling, elevating cAMP and "
            "activating PKA, which drives thyroid hormone synthesis (NIS-mediated iodide uptake, TPO-mediated "
            "organification, TG synthesis) and thyrocyte growth. "
            "GOF mutations (Ala623Ile, Ala623Val in transmembrane domain): receptor adopts active conformation "
            "constitutively without TSH binding — persistent cAMP elevation — chronic hyperthyroidism despite "
            "suppressed TSH; NEGATIVE autoantibodies (TRAb, TPO-Ab) distinguish FNAH from Graves disease. "
            "Radioiodine therapy does not eliminate the activating allele — residual TSHR-GOF thyrocytes regrow — "
            "recurrence inevitable; methimazole or total thyroidectomy required. "
            "FGH: TSHR ectodomain mutations lower threshold for hCG activation — hCG and TSH share structural "
            "homology at receptor-binding regions — first-trimester hCG surge triggers gestational thyrotoxicosis, "
            "hyperemesis, and fetal thyrotoxicosis risk. "
            "TSH resistance (AR LOF): impaired Gs coupling — blunted cAMP response to TSH — compensatory TSH "
            "elevation — normal or mildly hypothyroid phenotype; gland size normal."
        ),
        "pathognomonic": (
            "TSH SUPPRESSED + HYPERTHYROID BIOCHEMISTRY + NEGATIVE AUTOANTIBODIES (TRAb/TPO-Ab) = "
            "ACTIVATING TSHR MUTATION PATHOGNOMONIC; "
            "FGH: hyperemesis + hyperthyroidism in first pregnancy + NO prior thyroid disease + "
            "no TRAb = TSHR FGH PATHOGNOMONIC; "
            "radioiodine NOT curative in FNAH (GOF mutation remains); "
            "TSH resistance LOF: elevated TSH + normal FT4 + normal-sized gland + negative autoantibodies"
        ),
        "hormone_profile": (
            "FNAH: TSH suppressed (<0.1 mU/L), FT4/FT3 elevated, TRAb negative, TPO-Ab negative; "
            "FGH: same during pregnancy; "
            "TSH resistance (LOF): TSH elevated or normal-high, FT4 normal or low"
        ),
        "severity_sds": "N/A — thyroid function disorder; GOF: overt hyperthyroidism; LOF: variable hypothyroidism",
        "treatment": (
            "FNAH: methimazole long-term (controls hyperthyroidism) or total thyroidectomy (curative); "
            "NOT radioiodine (GOF mutation in all thyrocytes — recurrence guaranteed); "
            "FGH: propylthiouracil first trimester if needed; methimazole second/third trimester; "
            "TSH resistance: levothyroxine if clinical hypothyroidism; cascade testing first-degree relatives"
        ),
        "key_features": [
            "FNAH: suppressed TSH + negative TRAb/TPO-Ab = PATHOGNOMONIC for TSHR GOF",
            "Radioiodine NOT curative for FNAH (GOF mutation persists in all thyrocytes)",
            "FGH: hCG-sensitive TSHR — gestational thyrotoxicosis, hyperemesis, first pregnancy",
            "Somatic TSHR GOF = toxic adenoma (same mechanism as germline FNAH)",
            "AR LOF: TSH resistance type II — elevated TSH, normal FT4, normal gland",
            "Most common GOF variants: Ala623Ile, Ala623Val (transmembrane domain)",
            "Total thyroidectomy curative for FNAH (removes all mutant thyrocytes)",
        ],
        "key_ddx": [
            "Graves disease (TRAb positive — key distinguishing test)",
            "Toxic multinodular goitre (imaging shows nodules; older patients)",
            "Autonomously functioning thyroid nodule (single hot nodule on scan; somatic TSHR GOF)",
            "Hashimoto thyroiditis with hyperthyroid phase (TPO-Ab positive)",
        ],
        "onset_age": "FNAH: childhood to adulthood; FGH: first pregnancy; TSH resistance: neonatal to childhood",
        "ch_type": "hyperthyroidism (GOF) / TSH resistance (LOF)",
        "nbs_indicated": False,
        "severity": "moderate",
    },

    # -- DUOX2 -- Congenital hypothyroidism / dyshormonogenesis (partial organification) -------
    {
        "gene": "DUOX2",
        "alt_name": (
            "DUOX2 (DUOX2-1548aa-15q21.1 / AR -- "
            "MOST-COMMON-HEREDITARY-CH-DYSHORMONOGENESIS-15-20pct-OF-ALL-CH -- "
            "PERCHLORATE-DISCHARGE-TEST-POSITIVE->10pct-PARTIAL-ORGANIFICATION-DEFECT -- "
            "TRANSIENT-CH-POSSIBLE-MONOALLELIC-DUOX2-TRIAL-LT4-CESSATION-AGE-3 -- "
            "BIALLELIC-PERMANENT-CH-DUOXA2-MATURATION-FACTOR-SAME-PHENOTYPE)"
        ),
        "protein": (
            "DUOX2 -- 15q21.1 AR -- DUOX2-1548aa -- "
            "Dual-Oxidase-2-175kDa-NADPH-Oxidase-EF-Hand-Ca2+-Sensing-Peroxidase-Homology-Domain -- "
            "Apical-Membrane-Thyrocyte-H2O2-Generator-TPO-Substrate -- "
            "DUOX2-LOF-Insufficient-H2O2-TPO-Cannot-Oxidise-Iodide-Partial-Organification-Defect -- "
            "DUOXA2-Maturation-Factor-ER-Exit-Required-For-DUOX2-Surface-Expression -- "
            "DUOXA2-LOF-Identical-Phenotype-DUOX2-LOF -- "
            "Biallelic-Permanent-CH-Monoallelic-Transient-CH-Possible -- "
            "OMIM-Gene-606759-Disease-CH-Dyshormonogenesis-274900"
        ),
        "locus": "15q21.1",
        "protein_size": "1548 aa / 175 kDa",
        "inheritance": (
            "AR (autosomal recessive); most common hereditary cause of congenital hypothyroidism (CH) "
            "from dyshormonogenesis (~15-20% of all CH cases); "
            "biallelic DUOX2 = permanent CH; monoallelic/heterozygous DUOX2 may cause transient CH; "
            "TRANSIENT CH: trial of L-T4 cessation at age 3 appropriate in monoallelic cases; "
            "DUOXA2 (maturation factor, same locus) causes identical phenotype; "
            "perchlorate discharge test >10% = partial iodide organification defect"
        ),
        "disease_category": (
            "Congenital Hypothyroidism from dyshormonogenesis (partial organification defect); "
            "normal/enlarged thyroid (goitre) on scan; NBS elevated TSH; "
            "POSITIVE PERCHLORATE DISCHARGE TEST (>10%); "
            "TRANSIENT CH possible (monoallelic); biallelic = permanent"
        ),
        "disease_pathway": (
            "DUOX2 (Dual Oxidase 2) is an NADPH oxidase expressed at the apical membrane of thyroid follicular cells. "
            "Its primary function is to generate H2O2 at the apical surface, serving as the essential electron acceptor "
            "for thyroid peroxidase (TPO) to oxidise iodide and catalyse iodination of thyroglobulin (TG). "
            "Without sufficient H2O2 (DUOX2 LOF), TPO cannot organify iodide — iodide accumulates without being "
            "incorporated into TG — partial organification defect. "
            "PERCHLORATE DISCHARGE TEST: perchlorate competitively blocks NIS (iodide transporter); "
            "in a normal gland, most iodide is rapidly organified (covalently bound to TG) — "
            "perchlorate cannot discharge it; in organification defect, unorganified iodide remains free — "
            "perchlorate discharges it — >10-15% discharge = partial organification defect. "
            "Biallelic DUOX2 — severe H2O2 deficiency — persistent partial organification defect — "
            "elevated TSH (compensatory) — thyroid enlargement (goitre) — elevated TSH on NBS. "
            "Monoallelic DUOX2: one functional allele may provide sufficient H2O2 in post-neonatal period — "
            "transient CH resolves; neonatal period may show deficiency. "
            "DUOXA2 is the endoplasmic reticulum maturation factor required for DUOX2 trafficking to the apical membrane; "
            "DUOXA2 LOF — DUOX2 retained in ER — functional absence of H2O2 — identical phenotype."
        ),
        "pathognomonic": (
            "CH + GOITRE ON ULTRASOUND + POSITIVE PERCHLORATE DISCHARGE TEST (>10%) = DYSHORMONOGENESIS; "
            "DUOX2 most common gene; "
            "TRANSIENT CH: trial cessation at age 3 if monoallelic; "
            "DUOXA2 on same locus — test if DUOX2 negative; "
            "biallelic = permanent (no trial off L-T4); monoallelic = potentially transient"
        ),
        "hormone_profile": (
            "TSH elevated on NBS (>10 mU/L); FT4 low or normal (compensated); "
            "thyroglobulin elevated (goitre); perchlorate discharge test >10% (partial organification defect)"
        ),
        "severity_sds": "Variable; biallelic: severe permanent CH; monoallelic: mild/transient; goitre common",
        "treatment": (
            "Levothyroxine from neonatal period; biallelic DUOX2: permanent CH — lifelong L-T4; "
            "monoallelic DUOX2: TRANSIENT CH possible — trial of L-T4 cessation at age 3 "
            "(recheck TFTs 3-4 weeks after stopping; if normal — transient CH confirmed); "
            "goitre: usually regresses with adequate L-T4; DUOXA2: same management"
        ),
        "key_features": [
            "Most common hereditary CH from dyshormonogenesis (~15-20% of all CH)",
            "Perchlorate discharge test >10% = partial organification defect = DUOX2/DUOXA2",
            "Goitre on ultrasound (TSH-stimulated hyperplastic gland)",
            "TRANSIENT CH: monoallelic DUOX2 — trial L-T4 cessation at age 3",
            "Biallelic = permanent CH (no cessation trial appropriate)",
            "DUOXA2 (maturation factor, same locus) = identical phenotype",
            "NBS detects elevated TSH in first 48-72 hours",
        ],
        "key_ddx": [
            "TPO mutation (total organification defect: perchlorate discharge >90%; larger goitre)",
            "TG mutation (low/undetectable TG despite goitre — paradox; perchlorate mildly positive)",
            "SLC26A4/Pendred (EVA on CT/MRI + SNHL; partial organification but EVA distinguishes)",
            "Thyroid dysgenesis PAX8/NKX2-1/FOXE1 (absent/ectopic/hypoplastic gland on Tc scan)",
        ],
        "onset_age": "Neonatal (NBS at 48-72 hours); goitre may be visible at birth or develop in infancy",
        "ch_type": "CH (dyshormonogenesis — partial organification defect)",
        "nbs_indicated": True,
        "severity": "moderate-severe",
    },

    # -- TPO -- Congenital hypothyroidism / TOTAL organification defect ------------------------
    {
        "gene": "TPO",
        "alt_name": (
            "TPO (TPO-933aa-2p25.3 / AR -- "
            "TOTAL-ORGANIFICATION-DEFECT-PERCHLORATE-DISCHARGE->90pct-PATHOGNOMONIC -- "
            "LARGE-GOITRE-AT-BIRTH-PROFOUND-CH -- "
            "PERMANENT-CH-NO-TRANSIENT-CASES-UNLIKE-DUOX2 -- "
            "PENDRED-DDx-SLC26A4-EVA-IMAGING-SNHL-Absent-in-TPO)"
        ),
        "protein": (
            "TPO -- 2p25.3 AR -- TPO-933aa -- "
            "Thyroid-Peroxidase-103kDa-Haem-Containing-Enzyme-Apical-Membrane-Thyrocyte -- "
            "Oxidises-Iodide-I-minus-To-I-Plus-HOI-For-TG-Iodination -- "
            "MIT-DIT-Formation-On-Thyroglobulin-T3-T4-Coupling-Reaction -- "
            "Requires-H2O2-From-DUOX2-As-Electron-Acceptor -- "
            "TPO-LOF-Total-Organification-Absence-All-Iodide-Remains-Unorganified -- "
            "TPO-Also-Major-Autoantigen-In-Hashimoto-Graves-TPO-Ab-Autoimmune -- "
            "OMIM-Gene-606765-Disease-CH-Dyshormonogenesis-274900"
        ),
        "locus": "2p25.3",
        "protein_size": "933 aa / 103 kDa",
        "inheritance": (
            "AR (autosomal recessive); most common cause of TOTAL iodide organification defect; "
            "profound CH with large goitre often visible at birth; "
            "perchlorate discharge test >90% PATHOGNOMONIC (distinguishes total from partial defect of DUOX2); "
            "NO transient cases — all biallelic TPO = permanent CH (unlike DUOX2 monoallelic); "
            "goitre may be compressive and require surgery; "
            "Pendred syndrome DDx (SLC26A4) also has organification defect but has EVA imaging + SNHL"
        ),
        "disease_category": (
            "Congenital Hypothyroidism from TOTAL organification defect; "
            "large goitre at birth; profound CH; "
            "PERCHLORATE DISCHARGE >90% PATHOGNOMONIC; "
            "permanent CH; Pendred DDx (EVA + SNHL)"
        ),
        "disease_pathway": (
            "TPO (thyroid peroxidase) is a haem-containing enzyme at the apical membrane of thyroid follicular cells. "
            "It performs two critical reactions in thyroid hormone synthesis: "
            "(1) iodide oxidation — converts I- to reactive iodine (I+, HOI) using H2O2 from DUOX2 as electron acceptor; "
            "(2) iodination and coupling — incorporates reactive iodine into tyrosine residues of thyroglobulin, "
            "forming monoiodotyrosine (MIT) and diiodotyrosine (DIT), then coupling T3 and T4. "
            "TOTAL TPO LOF: complete absence of organification activity — all iodide entering thyrocytes via NIS remains "
            "as free, unorganified iodide — perchlorate (NIS blocker) discharges virtually all accumulated iodide — "
            ">90% perchlorate discharge PATHOGNOMONIC for total organification defect. "
            "Persistent TSH stimulation — massive thyroid hyperplasia — large goitre at or soon after birth. "
            "Unlike DUOX2 (H2O2 generation defect, partial), TPO absence means ZERO organification — "
            "more severe phenotype with larger goitre, more profound CH, and NO transient cases. "
            "TPO is also the dominant autoantigen in Hashimoto thyroiditis (TPO-Ab) and Graves disease, "
            "but hereditary TPO LOF is a separate disorder without autoimmune activation. "
            "Pendred syndrome (SLC26A4/pendrin): SNHL + enlarged vestibular aqueduct (EVA) on CT/MRI + "
            "partial/near-total organification defect — distinguishable by EVA imaging and SNHL."
        ),
        "pathognomonic": (
            "CH + LARGE GOITRE AT BIRTH + PERCHLORATE DISCHARGE >90% = "
            "TPO TOTAL ORGANIFICATION DEFECT PATHOGNOMONIC; "
            "no transient cases — ALL TPO biallelic = permanent CH; "
            "PENDRED DDx: SLC26A4 has EVA (enlarged vestibular aqueduct) + SNHL — both absent in TPO; "
            "DUOX2 has only partial discharge (<90%) — >90% points to TPO or Pendred"
        ),
        "hormone_profile": (
            "TSH severely elevated at NBS; FT4 very low; "
            "thyroglobulin very high (hyperplastic gland synthesising scaffold but no hormone formed); "
            "perchlorate discharge >90% (total organification defect)"
        ),
        "severity_sds": "Severe permanent CH; one of the most severe dyshormonogeneses; large goitre",
        "treatment": (
            "High-dose levothyroxine from NBS; lifelong L-T4 — NO cessation trial (always permanent); "
            "goitre: L-T4 may reduce size by suppressing TSH; "
            "compressive goitre: surgical debulking/thyroidectomy if tracheal compression; "
            "monitor TFTs 3-monthly first year; genetic counselling: 25% recurrence risk (AR)"
        ),
        "key_features": [
            "TOTAL organification defect — perchlorate discharge >90% PATHOGNOMONIC",
            "Most common cause of total (not partial) organification defect",
            "Large goitre at birth — TSH-driven hyperplasia",
            "Permanent CH — no transient cases (unlike DUOX2 monoallelic)",
            "No SNHL and no EVA — key DDx from Pendred (SLC26A4)",
            "Compressive goitre may require surgery beyond L-T4 management",
            "TPO is major autoantigen in Hashimoto/Graves (separate from hereditary LOF)",
        ],
        "key_ddx": [
            "Pendred syndrome SLC26A4 (organification defect + EVA on CT/MRI + SNHL)",
            "DUOX2 partial organification (perchlorate 10-80%; not >90%; transient possible)",
            "TG mutation (low TG despite goitre paradox; discharge <90%)",
            "Thyroid dysgenesis PAX8/NKX2-1 (absent/small thyroid on Tc scan)",
        ],
        "onset_age": "Neonatal (NBS at 48-72 hours); goitre often visible at birth",
        "ch_type": "CH (dyshormonogenesis — total organification defect)",
        "nbs_indicated": True,
        "severity": "severe",
    },

    # -- TG -- Congenital hypothyroidism / thyroglobulin deficiency ----------------------------
    {
        "gene": "TG",
        "alt_name": (
            "TG (TG-2768aa-8q24.22 / AR -- "
            "CH-GOITRE-UNDETECTABLE-TG-DESPITE-GOITRE-PARADOX-PATHOGNOMONIC -- "
            "MOST-COMMON-GOITROUS-CH-IODINE-SUFFICIENT-COUNTRIES -- "
            "TG-LARGEST-SECRETED-PROTEIN-330kDa-SCAFFOLD-THYROID-HORMONE-SYNTHESIS -- "
            "PERCHLORATE-DISCHARGE-MILDLY-POSITIVE-15-30pct)"
        ),
        "protein": (
            "TG -- 8q24.22 AR -- TG-2768aa -- "
            "Thyroglobulin-330kDa-LARGEST-Secreted-Protein-Human-Homo-Dimer-660kDa -- "
            "Colloid-Stored-In-Thyroid-Follicular-Lumen-Backbone-For-T3-T4-Synthesis -- "
            "Type-1-Repeat-EGF-Like-Fibronectin-Type-III-Domains-Multiple-Tyrosine-Residues -- "
            "Iodinated-By-TPO-MIT-DIT-Formed-Coupling-T3-T4-Released-By-Cathepsins-Lysosomes -- "
            "TG-LOF-No-Scaffold-Iodide-Cannot-Be-Incorporated-Into-Hormone -- "
            "Serum-TG-Marker-Thyroid-Tissue-Presence-Thyroid-Cancer-Surveillance -- "
            "OMIM-Gene-188450-Disease-CH-Goitrous-Dyshormonogenesis-274700"
        ),
        "locus": "8q24.22",
        "protein_size": "2768 aa / 330 kDa",
        "inheritance": (
            "AR (autosomal recessive); most common cause of goitrous CH in iodine-sufficient countries; "
            "UNDETECTABLE or very low serum TG despite goitre = TG gene mutation PATHOGNOMONIC (paradox: "
            "TSH drives gland enlargement but no TG is produced as substrate — no TG in circulation); "
            "perchlorate discharge may be mildly positive (15-30%): iodide enters via NIS normally but "
            "cannot be incorporated (TG scaffold absent) — slight free iodide; "
            "permanent CH — levothyroxine lifelong; goitre regresses with adequate L-T4"
        ),
        "disease_category": (
            "Congenital Hypothyroidism from thyroglobulin absence (goitrous dyshormonogenesis); "
            "CH + GOITRE + UNDETECTABLE/LOW SERUM TG (paradox) PATHOGNOMONIC; "
            "most common goitrous CH in iodine-sufficient countries; permanent CH"
        ),
        "disease_pathway": (
            "Thyroglobulin (TG) is the largest secreted human protein (2768 aa, 330 kDa monomer; "
            "660 kDa homodimer) and the unique scaffold on which thyroid hormones T3 and T4 are synthesised. "
            "In normal synthesis: TG secreted into follicular lumen — TPO iodination of tyrosine residues "
            "(MIT, DIT formation) — T4/T3 coupling — storage as iodinated TG in colloid — "
            "endocytosis and lysosomal proteolysis — T4/T3 released into bloodstream. "
            "TG LOF: no TG scaffold produced — iodide enters thyrocyte via NIS normally — "
            "TPO is active and H2O2 from DUOX2 is available — but NO substrate for organification — "
            "iodide cannot be incorporated — PARTIAL free iodide accumulation (mild perchlorate discharge 15-30%). "
            "PARADOX: elevated TSH — massive thyroid hypertrophy (goitre) — gland enlarges responding to TSH, "
            "but serum TG remains UNDETECTABLE (no TG produced) — normally elevated TSH + goitre = high serum TG; "
            "this paradoxical combination (goitre + undetectable TG) = TG mutation PATHOGNOMONIC. "
            "Permanent CH: without TG there is no substrate for T3/T4 synthesis regardless of other pathway "
            "components — lifelong L-T4 essential. "
            "Goitre regresses with adequate L-T4 therapy (removes TSH drive for gland hypertrophy). "
            "TG is also the main thyroid cancer tumour marker — TG NOT useful for cancer surveillance "
            "in TG LOF patients (endogenous TG absent — cannot distinguish recurrence from baseline)."
        ),
        "pathognomonic": (
            "CH + GOITRE + UNDETECTABLE OR VERY LOW SERUM TG = TG MUTATION PATHOGNOMONIC (PARADOX); "
            "serum TG < 1-2 ng/mL despite TSH elevation and palpable goitre = diagnostic paradox; "
            "perchlorate discharge mildly positive (15-30%) — iodide unorganified (no TG scaffold) "
            "but much less than TPO total defect (>90%); "
            "RAI uptake NORMAL (NIS intact) — distinguishes from SLC5A5/ITD"
        ),
        "hormone_profile": (
            "TSH elevated; FT4 low; "
            "serum TG undetectable or very low despite elevated TSH and goitre (PARADOX); "
            "perchlorate discharge mildly positive (15-30%)"
        ),
        "severity_sds": "Moderate-severe permanent CH; goitre prominent; TG paradox diagnostic",
        "treatment": (
            "Levothyroxine lifelong from NBS; goitre regresses with adequate L-T4; "
            "NO indication for perchlorate therapy; NO cessation trial (always permanent CH biallelic TG LOF); "
            "TFTs 3-monthly first year; 6-monthly thereafter; "
            "serum TG monitoring NOT useful for cancer surveillance in TG LOF patients; "
            "genetic counselling: 25% recurrence risk (AR)"
        ),
        "key_features": [
            "TG is the LARGEST secreted human protein (2768 aa, 330 kDa monomer)",
            "Low/undetectable TG despite goitre = PARADOX = TG mutation PATHOGNOMONIC",
            "Most common goitrous CH in iodine-sufficient countries",
            "Perchlorate discharge mildly positive (15-30%) — partial, unlike TPO total defect",
            "RAI uptake NORMAL (NIS intact) — distinguishes from SLC5A5/ITD",
            "Goitre regresses with adequate L-T4 (TSH suppressed — no drive for hyperplasia)",
            "Serum TG NOT useful for thyroid cancer surveillance in TG LOF patients",
        ],
        "key_ddx": [
            "DUOX2/TPO organification defects (TG elevated in those; perchlorate much higher)",
            "Thyroid dysgenesis (no goitre; Tc scan shows absent/ectopic thyroid)",
            "Iodine deficiency (TG elevated, not low; dietary history, geographic context)",
            "SLC5A5/ITD (absent RAI uptake <5%; TG may be elevated due to goitre)",
        ],
        "onset_age": "Neonatal (NBS at 48-72 hours); goitre may develop in first weeks to months",
        "ch_type": "CH (dyshormonogenesis — thyroglobulin absence)",
        "nbs_indicated": True,
        "severity": "moderate-severe",
    },

    # -- SLC5A5 -- Iodide transport defect (ITD) / NIS ----------------------------------------
    {
        "gene": "SLC5A5",
        "alt_name": (
            "SLC5A5 (SLC5A5-643aa-19p13.11 / AR -- "
            "IODIDE-TRANSPORT-DEFECT-ITD -- "
            "ABSENT-RAI-UPTAKE-<5pct-AT-24H-DESPITE-ELEVATED-TSH-PATHOGNOMONIC -- "
            "SALIVA-SERUM-IODIDE-RATIO-<30-PATHOGNOMONIC-NORMAL->40 -- "
            "RADIOIODINE-NOT-EFFECTIVE-NIS-ABSENT-PERTECHNETATE-ALSO-ABSENT)"
        ),
        "protein": (
            "SLC5A5 -- 19p13.11 AR -- SLC5A5-643aa -- "
            "NIS-Sodium-Iodide-Symporter-70kDa-Na-K-ATPase-Driven-Basolateral-Membrane-Thyrocyte -- "
            "Concentrates-Iodide-Against-Electrochemical-Gradient-20-40x-Plasma -- "
            "13-TM-Segments-Electrogenic-Na+-Iodide-Co-Transport-2-Na+-Per-I-Minus -- "
            "Also-Expressed-Salivary-Glands-Stomach-Breast-Thyroid-Major-Expression-Site -- "
            "NIS-Exploited-For-Radioiodine-Thyroid-Cancer-Therapy-Imaging -- "
            "Somatic-NIS-Downregulation-In-Thyroid-Cancer-Reduces-RAI-Efficacy -- "
            "OMIM-Gene-601843-Disease-Iodide-Transport-Defect-274400"
        ),
        "locus": "19p13.11",
        "protein_size": "643 aa / 70 kDa",
        "inheritance": (
            "AR (autosomal recessive); iodide transport defect (ITD) — NIS cannot concentrate iodide into thyrocyte; "
            "24h 123I thyroid uptake <5% (normal 10-30%) despite elevated TSH = PATHOGNOMONIC; "
            "pertechnetate (TcO4-) uptake also absent (NIS transports both iodide and pertechnetate); "
            "saliva/serum iodide ratio <30 (normal >40) — NIS absent in salivary glands too; "
            "goitre forms (TSH stimulation) but thyroid cannot accumulate iodide; "
            "radioiodine NOT effective (NIS absent); iodide supplementation does NOT help; "
            "total or subtotal thyroidectomy for compressive goitre"
        ),
        "disease_category": (
            "Congenital Hypothyroidism from Iodide Transport Defect (ITD); "
            "goitre + CH + ABSENT/LOW 123I THYROID UPTAKE (<5% at 24h) + saliva/serum iodide ratio <30 PATHOGNOMONIC; "
            "NIS also expressed in salivary glands, stomach, breast"
        ),
        "disease_pathway": (
            "SLC5A5 (NIS, Sodium-Iodide Symporter) is a 13-transmembrane electrogenic cotransporter at the basolateral "
            "membrane of thyroid follicular cells. NIS cotransports 2 Na+ ions and 1 I- ion into the cell; "
            "the driving force is the Na+ gradient maintained by Na+/K+-ATPase. "
            "Normal: NIS concentrates intracellular iodide 20-40x above plasma concentration — "
            "iodide diffuses apically to follicular lumen — TPO organifies it onto TG. "
            "NIS LOF: iodide cannot enter thyrocytes — TPO and TG present and functional but have NO substrate — "
            "CH results from iodine starvation at the cellular level despite normal dietary iodine intake. "
            "TSH elevation — thyroid hypertrophy (goitre) driven by TSH signalling, not by iodide. "
            "DIAGNOSTIC FEATURE: radioiodine (123I or 131I) uptake test shows <5% thyroid uptake at 24h "
            "despite elevated TSH (normally TSH drives NIS expression — increases uptake); "
            "pertechnetate (99mTcO4-) also absent (NIS transports both). "
            "SALIVA/PLASMA IODIDE RATIO: NIS expressed in salivary glands — NIS LOF reduces salivary iodide "
            "concentration — ratio <30 (normal >40) = ITD pathognomonic non-invasive confirmatory test. "
            "CLINICAL IMPORTANCE: somatic NIS downregulation in differentiated thyroid cancers reduces RAI efficacy; "
            "NIS re-expression is a target for radioiodine re-sensitisation therapies."
        ),
        "pathognomonic": (
            "CH + GOITRE + 123I THYROID UPTAKE <5% AT 24h DESPITE ELEVATED TSH = ITD PATHOGNOMONIC; "
            "ABSENT PERTECHNETATE UPTAKE = confirms NIS absence; "
            "SALIVA/SERUM IODIDE RATIO <30 (normal >40) = NIS absent in salivary glands; "
            "RADIOIODINE NOT EFFECTIVE — NIS absent; "
            "iodide supplementation does NOT help (transport defect, not substrate deficiency)"
        ),
        "hormone_profile": (
            "TSH elevated; FT4 low; 123I thyroid uptake <5% at 24h; "
            "pertechnetate (Tc) scan: absent thyroid uptake (despite goitre); "
            "saliva/plasma iodide ratio <30 (normal >40)"
        ),
        "severity_sds": "Moderate-severe permanent CH; goitre present but incapable of concentrating iodide",
        "treatment": (
            "Levothyroxine lifelong from NBS; "
            "goitre: L-T4 suppresses TSH; compressive goitre requires total/subtotal thyroidectomy; "
            "DO NOT use iodide supplementation (transport defect); "
            "DO NOT use radioiodine for goitre (NIS absent); "
            "TFTs 3-monthly first year; genetic counselling: 25% recurrence risk (AR)"
        ),
        "key_features": [
            "Absent/low 123I thyroid uptake (<5% at 24h) = PATHOGNOMONIC for NIS/ITD",
            "Pertechnetate uptake also absent (NIS transports TcO4- as well as iodide)",
            "Saliva/serum iodide ratio <30 (normal >40) — NIS absent in salivary glands",
            "Iodide supplementation does NOT work (transport defect, not dietary deficiency)",
            "Radioiodine NOT effective — NIS absent from thyrocytes",
            "NIS expressed in salivary glands, stomach, breast (explains radioiodine whole-body scan rationale)",
            "Somatic NIS downregulation in thyroid cancer reduces RAI treatment efficacy",
        ],
        "key_ddx": [
            "Thyroid agenesis (no goitre; Tc scan shows no gland at all; FOXE1/PAX8/NKX2-1)",
            "DUOX2/TPO organification defects (RAI uptake NORMAL or HIGH — NIS intact)",
            "TG mutation (RAI uptake normal; low TG despite goitre)",
            "Iodine deficiency (RAI uptake may be high — NIS upregulated; dietary history)",
        ],
        "onset_age": "Neonatal (NBS at 48-72 hours); may present later if partial NIS function",
        "ch_type": "CH (iodide transport defect — NIS absence)",
        "nbs_indicated": True,
        "severity": "moderate-severe",
    },

    # -- FOXE1 -- Bamforth-Lazarus syndrome / thyroid agenesis ---------------------------------
    {
        "gene": "FOXE1",
        "alt_name": (
            "FOXE1 (FOXE1-373aa-9q22.33 / AR -- "
            "BAMFORTH-LAZARUS-SYNDROME-THYROID-AGENESIS+CLEFT-PALATE+BIFID-EPIGLOTTIS+SPIKY-HAIR-TETRAD-PATHOGNOMONIC -- "
            "CHOANAL-ATRESIA-NEONATAL-AIRWAY-EMERGENCY -- "
            "IMMEDIATE-LEVOTHYROXINE-IV-NEONATAL-PERIOD -- "
            "TTF-2-THYROID-TRANSCRIPTION-FACTOR-2)"
        ),
        "protein": (
            "FOXE1 -- 9q22.33 AR -- FOXE1-373aa -- "
            "Forkhead-Box-E1-TTF-2-Thyroid-Transcription-Factor-2-42kDa -- "
            "Forkhead-Domain-DNA-Binding-Transcription-Factor -- "
            "Essential-Thyroid-Anlage-Migration-Ultimobranchial-Body-Fusion-Foregut-Separation -- "
            "Also-Required-Palate-Development-Epiglottis-Morphogenesis-Hair-Follicle-Precursors -- "
            "Biallelic-LOF-Athyreosis-Arrested-Migration-Thyroid-Anlage -- "
            "Heterozygous-FOXE1-rs965513-GWAS-Thyroid-Cancer-Predisposition-Separate -- "
            "OMIM-Gene-602617-Disease-Bamforth-Lazarus-241850"
        ),
        "locus": "9q22.33",
        "protein_size": "373 aa / 42 kDa",
        "inheritance": (
            "AR (autosomal recessive, biallelic LOF); "
            "BAMFORTH-LAZARUS SYNDROME: thyroid agenesis + cleft palate + bifid epiglottis (or choanal atresia) + "
            "spiky hair = PATHOGNOMONIC TETRAD (<50 cases reported worldwide); "
            "neonatal emergency: choanal atresia — hypoxia (obligate nasal breathers); CH (athyreosis); "
            "immediate IV levothyroxine AND airway management required; "
            "FOXE1 heterozygous variants (rs965513 GWAS locus) associated with thyroid cancer risk — "
            "separate from biallelic Bamforth-Lazarus; lifelong athyreosis — no thyroid on scan"
        ),
        "disease_category": (
            "Bamforth-Lazarus syndrome — THYROID AGENESIS + CLEFT PALATE + BIFID EPIGLOTTIS/CHOANAL ATRESIA "
            "+ SPIKY HAIR PATHOGNOMONIC TETRAD; most severe thyroid developmental disorder; "
            "neonatal emergency (hypoxia + CH)"
        ),
        "disease_pathway": (
            "FOXE1 (Forkhead Box E1, also called TTF-2, Thyroid Transcription Factor 2) is a forkhead domain "
            "transcription factor essential for thyroid gland development and morphogenesis. "
            "During embryogenesis, the thyroid anlage originates at the foramen caecum (base of tongue) and "
            "migrates caudally to its final pre-tracheal position by week 7 of gestation. "
            "FOXE1 is required for: (1) migration of the thyroid anlage from the foramen caecum; "
            "(2) separation of the developing thyroid from the pharyngeal/foregut epithelium; "
            "(3) palate fusion (secondary palate development); (4) epiglottis morphogenesis. "
            "Biallelic FOXE1 LOF: thyroid anlage fails to migrate and/or survive — thyroid agenesis (athyreosis); "
            "palate development impaired — cleft palate; epiglottis morphogenesis arrested — bifid epiglottis "
            "or choanal atresia (neonatal hypoxia as neonates are obligate nasal breathers). "
            "Spiky hair: FOXE1 expressed in hair follicle precursors — hair shaft morphology abnormality. "
            "NEONATAL EMERGENCY: combination of choanal atresia (hypoxia) + profound CH (athyreosis — no thyroid) "
            "requires immediate parallel management: "
            "(1) airway intervention (McGovern nasal trumpet, oral airway, surgical choanal atresia repair); "
            "(2) IV levothyroxine (oral feeding impossible with cleft palate). "
            "FOXE1 heterozygous variants (common GWAS SNP rs965513, 9q22.33) are associated with papillary thyroid "
            "cancer risk in the general population — a separate, common, low-penetrance variant, "
            "not the same biallelic syndrome."
        ),
        "pathognomonic": (
            "THYROID AGENESIS + CLEFT PALATE + BIFID EPIGLOTTIS + SPIKY HAIR = "
            "BAMFORTH-LAZARUS SYNDROME PATHOGNOMONIC (<50 cases worldwide); "
            "CHOANAL ATRESIA: hypoxia (neonatal obligate nasal breathers) + profound CH = "
            "neonatal dual emergency — airway FIRST + immediate IV levothyroxine; "
            "thyroid absent on BOTH ultrasound AND Tc scan — no ectopic tissue (unlike PAX8 ectopic lingual thyroid)"
        ),
        "hormone_profile": (
            "TSH severely elevated at NBS; FT4 undetectable; "
            "TG undetectable (no thyroid tissue); "
            "thyroid absent on ultrasound and Tc scan; no ectopic uptake on scan"
        ),
        "severity_sds": "Most severe — complete athyreosis from birth; neonatal emergency; cleft palate; choanal atresia",
        "treatment": (
            "NEONATAL EMERGENCY: immediate IV levothyroxine (oral route impossible with cleft palate/choanal atresia); "
            "airway management: McGovern nasal trumpet/oral airway — surgical choanal atresia correction; "
            "cleft palate repair (timing: usually 6-12 months); "
            "lifelong levothyroxine (athyreosis — permanent); standard CH dosing: 10-15 mcg/kg/day IV initially"
        ),
        "key_features": [
            "Bamforth-Lazarus tetrad: thyroid agenesis + cleft palate + bifid epiglottis + spiky hair",
            "RAREST hereditary thyroid disorder (<50 cases worldwide)",
            "Neonatal emergency: choanal atresia (hypoxia) + profound CH — parallel dual management",
            "Airway first + IV levothyroxine (oral impossible with cleft palate/choanal atresia)",
            "Thyroid absent on BOTH ultrasound AND Tc scan (no ectopic tissue — vs PAX8)",
            "FOXE1 heterozygous variants (rs965513): GWAS thyroid cancer risk — separate from syndrome",
            "Lifelong athyreosis — no trial off L-T4",
        ],
        "key_ddx": [
            "PAX8 thyroid dysgenesis (AD, variable expressivity; ectopic lingual thyroid; NO cleft palate/choanal atresia)",
            "NKX2-1 BLT syndrome (CH + RDS + chorea; thyroid hypoplastic not agenetic; NO cleft palate)",
            "Isolated cleft palate (thyroid normal; no CH; no bifid epiglottis)",
            "Sporadic thyroid agenesis (no extraglandular features; de novo PAX8 mutation common)",
        ],
        "onset_age": "Birth (neonatal emergency — first hours of life)",
        "ch_type": "CH (thyroid agenesis — athyreosis)",
        "nbs_indicated": True,
        "severity": "severe",
    },

    # -- PAX8 -- Thyroid dysgenesis (hypoplasia/ectopy/agenesis) — AD --------------------------
    {
        "gene": "PAX8",
        "alt_name": (
            "PAX8 (PAX8-450aa-2q13 / AD -- "
            "THYROID-DYSGENESIS-HYPOPLASIA-ECTOPY-ATHYREOSIS -- "
            "VARIABLE-EXPRESSIVITY-AFFECTED-PARENT-MAY-BE-EUTHYROID-CLINICAL-TRAP -- "
            "ECTOPIC-LINGUAL-THYROID-DO-NOT-EXCISE-ONLY-THYROID-TISSUE -- "
            "POSITIVE-FAMILY-HISTORY-KEY-DDx-AD-INHERITANCE)"
        ),
        "protein": (
            "PAX8 -- 2q13 AD -- PAX8-450aa -- "
            "Paired-Box-8-48kDa-Paired-Domain-Homeodomain-TF-Thyroid-Gland-Development -- "
            "NKX2-1-PAX8-Synergistic-Activation-TG-TPO-SLC5A5-Thyroid-Differentiation-Genes -- "
            "PAX8-Required-Thyroid-Follicular-Cell-Survival-Differentiation-Neonatal -- "
            "PAX8-Also-Expressed-Kidney-Mullerian-Structures-Renal-Phenotype-Mild-Rare -- "
            "PAX8-PPARgamma-Fusion-Somatic-Follicular-Thyroid-Cancer-t(2-3) -- "
            "AD-LOF-Variable-Expressivity-Profound-CH-to-Subclinical-Hypothyroidism -- "
            "OMIM-Gene-167415-Disease-Thyroid-Dysgenesis-218700"
        ),
        "locus": "2q13",
        "protein_size": "450 aa / 48 kDa",
        "inheritance": (
            "AD (autosomal dominant, LOF); VARIABLE EXPRESSIVITY is the key clinical trap: "
            "same PAX8 variant in the same family can cause profound CH in one member and "
            "subclinical hypothyroidism or even a normal thyroid in another; "
            "affected parent may appear euthyroid — DO NOT assume family history negative without testing; "
            "thyroid dysgenesis: hypoplasia, ectopy (lingual thyroid most common ectopic site), or athyreosis; "
            "ectopic lingual thyroid: tongue base mass + CH — DO NOT excise (only functioning thyroid tissue); "
            "L-T4 treatment shrinks ectopic thyroid by suppressing TSH; "
            "PAX8-PPARgamma fusion: somatic translocation t(2;3)(q13;p25) in follicular thyroid cancer — not germline"
        ),
        "disease_category": (
            "Thyroid dysgenesis — hypoplasia, ectopy, or athyreosis; "
            "CH + thyroid dysgenesis + POSITIVE FAMILY HISTORY + VARIABLE EXPRESSIVITY; "
            "AD inheritance distinguishes from FOXE1 (AR, Bamforth-Lazarus)"
        ),
        "disease_pathway": (
            "PAX8 (Paired box 8) is a paired domain transcription factor essential for thyroid follicular cell "
            "differentiation and survival. PAX8 acts in concert with NKX2-1 (TTF-1) to activate thyroid-specific "
            "genes including TG, TPO, and SLC5A5 (NIS). PAX8 also drives follicular cell survival in the "
            "developing thyroid gland — LOF — follicular cell apoptosis or failure to differentiate — "
            "hypoplastic, ectopic, or absent thyroid. "
            "VARIABLE EXPRESSIVITY mechanism: PAX8 is haploinsufficient but threshold effects vary between "
            "individuals and family members; modifier genes, epigenetic factors, or stochastic variation in "
            "developmental timing affect whether the remaining functional allele is sufficient. "
            "Consequence: profound CH (near-complete dysgenesis) vs. subclinical hypothyroidism (partial "
            "hypoplasia) vs. normal exam with only mild scan finding in different family members. "
            "Ectopic lingual thyroid: thyroid anlage initiates at foramen caecum but fails to migrate — "
            "ectopic thyroid tissue at tongue base; this may be the patient's ONLY thyroid tissue — "
            "DO NOT excise (surgical excision — permanent athyreosis). "
            "L-T4 treatment suppresses TSH — ectopic lingual thyroid regresses in size. "
            "PAX8-PPARgamma somatic fusion (t(2;3)(q13;p25)): follicular thyroid carcinoma-specific somatic "
            "translocation — completely separate from germline PAX8 haploinsufficiency syndrome."
        ),
        "pathognomonic": (
            "CH + THYROID DYSGENESIS (hypoplastic/ectopic/absent on Tc scan) + POSITIVE FAMILY HISTORY = "
            "PAX8 AD mutation; "
            "VARIABLE EXPRESSIVITY CLINICAL TRAP: parent may be euthyroid on TFTs but have subclinical hypoplasia — "
            "test all first-degree relatives with TFTs AND thyroid scan; "
            "ECTOPIC LINGUAL THYROID: mass at tongue base + CH — DO NOT EXCISE (only thyroid tissue)"
        ),
        "hormone_profile": (
            "TSH elevated on NBS; FT4 low (degree varies with dysgenesis severity); "
            "TG low (hypoplastic gland); "
            "Tc scan: small/ectopic/absent thyroid"
        ),
        "severity_sds": "Variable — ranges from profound CH (athyreosis) to subclinical hypothyroidism",
        "treatment": (
            "Levothyroxine from NBS; ectopic lingual thyroid: L-T4 suppresses TSH — shrinks it — DO NOT excise; "
            "dysgenesis confirmed by Tc scan (not just ultrasound); "
            "family cascade testing: TFTs + thyroid scan for all first-degree relatives (variable expressivity trap); "
            "monitor annually; dose adjustments as child grows"
        ),
        "key_features": [
            "AD with VARIABLE EXPRESSIVITY — parent may appear euthyroid (clinical trap)",
            "Thyroid dysgenesis: hypoplasia, ectopy (lingual thyroid), or athyreosis",
            "Ectopic lingual thyroid: DO NOT excise — often only thyroid tissue; L-T4 shrinks it",
            "Tc scan distinguishes dysgenesis subtypes (not just ultrasound)",
            "Cascade test all first-degree relatives with TFTs + scan (not TFTs alone)",
            "PAX8-PPARgamma fusion in follicular thyroid cancer — somatic, not germline",
            "PAX8 + NKX2-1 synergistically activate TG/TPO/NIS in thyroid development",
        ],
        "key_ddx": [
            "FOXE1 Bamforth-Lazarus (AR; cleft palate + bifid epiglottis; no ectopic thyroid)",
            "NKX2-1 BLT syndrome (AD; chorea + RDS + CH; thyroid hypoplastic not ectopic-lingual)",
            "Sporadic thyroid dysgenesis (no family history; de novo PAX8 mutation possible)",
            "FNAH/TSHR (hyperthyroid, not hypothyroid; normal or large thyroid)",
        ],
        "onset_age": "Neonatal (NBS at 48-72 hours); variable severity across family members",
        "ch_type": "CH (thyroid dysgenesis — hypoplasia/ectopy/agenesis)",
        "nbs_indicated": True,
        "severity": "variable",
    },

    # -- NKX2-1 -- Brain-Lung-Thyroid (BLT) syndrome ------------------------------------------
    {
        "gene": "NKX2-1",
        "alt_name": (
            "NKX2-1 (NKX2-1-371aa-14q13.3 / AD -- "
            "BRAIN-LUNG-THYROID-BLT-SYNDROME -- "
            "BENIGN-HEREDITARY-CHOREA-ONSET-1-5Y+NEONATAL-RDS-SURFACTANT+CH-TRIAD-PATHOGNOMONIC -- "
            "CHOREA-NO-COGNITIVE-DECLINE-DDx-HUNTINGTON -- "
            "NKX2-1-TTF-1-TITF1-LUNG-ADENOCARCINOMA-IHC-MARKER-SOMATIC)"
        ),
        "protein": (
            "NKX2-1 -- 14q13.3 AD -- NKX2-1-371aa -- "
            "NK2-Homeobox-1-TTF-1-TITF1-41kDa-Homeodomain-TF -- "
            "Essential-Thyroid-Follicular-Cell-Lung-Type-II-Pneumocyte-Basal-Ganglia-Striatum-Development -- "
            "Activates-TG-TPO-SLC5A5-In-Thyroid-SP-B-SP-C-Surfactant-Proteins-In-Lung -- "
            "14q13.3-Deletion-Syndrome-Contiguous-Gene-Larger-Phenotype -- "
            "TTF-1-IHC-Marker-Lung-Adenocarcinoma-Immunohistochemistry-Somatic -- "
            "NKX2-1-Haploinsufficiency-BLT-Variable-Expressivity -- "
            "OMIM-Gene-600635-Disease-BLT-Syndrome-610978"
        ),
        "locus": "14q13.3",
        "protein_size": "371 aa / 41 kDa",
        "inheritance": (
            "AD (autosomal dominant, haploinsufficiency); "
            "BRAIN-LUNG-THYROID (BLT) syndrome — classic triad: "
            "(1) Benign hereditary chorea (onset 1-5 years; dance-like involuntary movements; "
            "NO cognitive decline distinguishes from Huntington); "
            "(2) Neonatal RDS (surfactant deficiency — SP-B, SP-C); "
            "(3) CH from thyroid dysgenesis/hypoplasia (50-70% of NKX2-1 patients); "
            "VARIABLE EXPRESSIVITY: not all 3 features present; "
            "chorea often prompts diagnosis AFTER CH and RDS already detected; "
            "14q13.3 deletion syndrome: larger deletion — broader neurodevelopmental phenotype; "
            "chromosomal microarray mandatory"
        ),
        "disease_category": (
            "Brain-Lung-Thyroid (BLT) syndrome — NKX2-1 haploinsufficiency: "
            "CH (50-70%) + Neonatal RDS surfactant deficiency + Benign hereditary chorea (onset 1-5y); "
            "TRIAD PATHOGNOMONIC; chorea without cognitive decline; AD"
        ),
        "disease_pathway": (
            "NKX2-1 (NK2 homeobox 1, also called TTF-1 or TITF1) is a homeodomain transcription factor "
            "with expression in three key organ systems: thyroid, lung, and basal ganglia (striatum). "
            "THYROID: NKX2-1 acts synergistically with PAX8 to activate TG, TPO, SLC5A5 (NIS) — required for "
            "follicular cell differentiation and survival; haploinsufficiency — thyroid dysgenesis (hypoplastic "
            "or ectopic) + CH. "
            "LUNG: NKX2-1 drives expression of surfactant protein genes (SP-B/SFTPB, SP-C/SFTPC) in type II "
            "pneumocytes; haploinsufficiency — SP-B and SP-C deficiency — neonatal respiratory distress syndrome "
            "at birth — insufficient surfactant to maintain alveolar patency — neonatal oxygen requirement, "
            "mechanical ventilation; pulmonary phenotype may include interstitial lung disease later. "
            "BRAIN (basal ganglia/striatum): NKX2-1 expressed during striatal interneuron development; "
            "haploinsufficiency — impaired inhibitory interneuron maturation in basal ganglia — "
            "disinhibition of motor pathways — chorea (involuntary, dance-like, non-progressive movements); "
            "onset typically 1-5 years; often improve in adulthood; "
            "NO cognitive decline (benign hereditary chorea, BHC) — key distinguishing feature from Huntington. "
            "TETRABENAZINE: AVOID in NKX2-1-associated chorea (may paradoxically worsen symptoms). "
            "TTF-1 in lung adenocarcinoma: NKX2-1/TTF-1 is a somatic expression marker used in IHC to identify "
            "lung primary vs metastasis — entirely separate from BLT germline syndrome."
        ),
        "pathognomonic": (
            "BENIGN HEREDITARY CHOREA (onset 1-5y, NO cognitive decline) + "
            "NEONATAL RDS (surfactant SP-B/SP-C deficiency) + CH = "
            "NKX2-1 BLT SYNDROME PATHOGNOMONIC TRIAD; "
            "NO cognitive decline in BHC = key DDx from Huntington disease (cognitive decline + CAG repeat); "
            "TETRABENAZINE: AVOID (may worsen NKX2-1 BHC); "
            "14q13.3 deletion: chromosomal microarray mandatory"
        ),
        "hormone_profile": (
            "TSH elevated on NBS; FT4 low (CH component, present in ~50-70%); "
            "SP-B/SP-C deficiency on BAL (lung component); "
            "Tc scan: hypoplastic thyroid (not agenetic — vs FOXE1); "
            "chorea diagnosed clinically"
        ),
        "severity_sds": "Variable — BLT triad rarely all present; chorea non-progressive; RDS usually neonatal-limited",
        "treatment": (
            "Levothyroxine for CH component (if present); "
            "neonatal RDS: oxygen, surfactant replacement, mechanical ventilation as needed; "
            "chorea: NO specific treatment needed (benign, may improve with age); "
            "AVOID tetrabenazine; levodopa trial may help; physiotherapy; "
            "14q13.3 deletion: chromosomal microarray mandatory"
        ),
        "key_features": [
            "BLT triad: Brain (chorea) + Lung (RDS) + Thyroid (CH) — not all 3 always present",
            "Benign hereditary chorea: onset 1-5y, dance-like, NO cognitive decline",
            "No cognitive decline = key DDx from Huntington disease",
            "Neonatal RDS: SP-B/SP-C deficiency; may need ventilation at birth",
            "AVOID tetrabenazine (may worsen NKX2-1-associated chorea)",
            "NKX2-1/TTF-1 = lung adenocarcinoma IHC marker (somatic, not germline syndrome)",
            "14q13.3 deletion syndrome: chromosomal microarray mandatory",
        ],
        "key_ddx": [
            "Huntington disease (CAG repeat; cognitive decline; adult onset; no thyroid/lung)",
            "Wilson disease (KF rings; liver disease; ceruloplasmin low)",
            "GNAO1 movement disorder (early infantile; encephalopathy; no thyroid/lung)",
            "PAX8 thyroid dysgenesis (AD; thyroid only; no chorea/lung)",
            "FOXE1 Bamforth-Lazarus (AR; athyreosis + cleft palate; no chorea/lung)",
        ],
        "onset_age": "Neonatal (RDS + NBS CH); chorea onset 1-5 years; variable expression",
        "ch_type": "CH (thyroid dysgenesis/hypoplasia)",
        "nbs_indicated": True,
        "severity": "variable",
    },
]


# ---------------------------------------------------------------------------
# Cohort simulation rates
# ---------------------------------------------------------------------------
_RATES = {
    "TSHR":    {"surgery": 0.25, "goitre": 0.50, "dysgenesis": 0.00, "surveillance": 0.88, "levothyroxine": 0.38, "nbs": 0.15},
    "DUOX2":   {"surgery": 0.05, "goitre": 0.70, "dysgenesis": 0.00, "surveillance": 0.92, "levothyroxine": 0.85, "nbs": 0.98},
    "TPO":     {"surgery": 0.20, "goitre": 0.78, "dysgenesis": 0.00, "surveillance": 0.96, "levothyroxine": 0.99, "nbs": 0.99},
    "TG":      {"surgery": 0.06, "goitre": 0.88, "dysgenesis": 0.00, "surveillance": 0.94, "levothyroxine": 0.99, "nbs": 0.95},
    "SLC5A5":  {"surgery": 0.18, "goitre": 0.82, "dysgenesis": 0.00, "surveillance": 0.93, "levothyroxine": 0.99, "nbs": 0.90},
    "FOXE1":   {"surgery": 0.40, "goitre": 0.00, "dysgenesis": 1.00, "surveillance": 0.98, "levothyroxine": 1.00, "nbs": 0.98},
    "PAX8":    {"surgery": 0.05, "goitre": 0.15, "dysgenesis": 0.88, "surveillance": 0.95, "levothyroxine": 0.95, "nbs": 0.92},
    "NKX2-1":  {"surgery": 0.04, "goitre": 0.12, "dysgenesis": 0.60, "surveillance": 0.90, "levothyroxine": 0.68, "nbs": 0.62},
}

_AGE_RANGES = {
    "TSHR":    (15, 65),
    "DUOX2":   (0,  4),
    "TPO":     (0,  3),
    "TG":      (0,  5),
    "SLC5A5":  (0, 10),
    "FOXE1":   (0,  0.5),
    "PAX8":    (0,  5),
    "NKX2-1":  (0,  5),
}

_SUPPRESSED_TSH_RATES = {
    "TSHR":    0.68, "DUOX2": 0.00, "TPO": 0.00, "TG": 0.00,
    "SLC5A5":  0.00, "FOXE1": 0.00, "PAX8": 0.00, "NKX2-1": 0.00,
}

_AUTOAB_NEG_RATES = {
    "TSHR":    0.90, "DUOX2": 0.92, "TPO": 0.94, "TG": 0.93,
    "SLC5A5":  0.92, "FOXE1": 0.98, "PAX8": 0.95, "NKX2-1": 0.93,
}

_PERCHLORATE_POS_RATES = {
    "TSHR":    0.00, "DUOX2": 0.78, "TPO": 0.92, "TG": 0.40,
    "SLC5A5":  0.00, "FOXE1": 0.00, "PAX8": 0.00, "NKX2-1": 0.00,
}

_NBS_DETECTED_RATES = {
    "TSHR":    0.10, "DUOX2": 0.97, "TPO": 0.99, "TG": 0.97,
    "SLC5A5":  0.90, "FOXE1": 0.98, "PAX8": 0.95, "NKX2-1": 0.62,
}

_TRANSIENT_CH_RATES = {
    "TSHR":    0.00, "DUOX2": 0.32, "TPO": 0.00, "TG": 0.00,
    "SLC5A5":  0.00, "FOXE1": 0.00, "PAX8": 0.00, "NKX2-1": 0.00,
}

_TYPE_VARIANTS = {
    "TSHR":    ["FNAH-Ala623Ile-GOF", "FNAH-Ala623Val-GOF", "FGH-hCG-sensitive", "TSH-Resistance-LOF"],
    "DUOX2":   ["Biallelic-Permanent-CH", "Biallelic-Compound-Het", "Monoallelic-Transient-CH", "DUOXA2-Equivalent"],
    "TPO":     ["Total-Organification-Defect", "Total-Organification-Compound-Het", "Total-Organification-Homozygous", "Total-Organification-Splice"],
    "TG":      ["TG-LOF-Compound-Het", "TG-LOF-Homozygous", "TG-LOF-Frameshift", "TG-LOF-Splice"],
    "SLC5A5":  ["ITD-Homozygous", "ITD-Compound-Het", "ITD-Missense-Severe", "ITD-Splice-Site"],
    "FOXE1":   ["Bamforth-Lazarus-Cleft-Palate", "Bamforth-Lazarus-Choanal-Atresia", "Bamforth-Lazarus-Full-Tetrad", "Bamforth-Lazarus-Partial"],
    "PAX8":    ["Thyroid-Hypoplasia", "Ectopic-Lingual-Thyroid", "Thyroid-Athyreosis", "Thyroid-Ectopy-Other"],
    "NKX2-1":  ["BLT-Full-Triad", "BLT-Chorea-CH-Only", "BLT-RDS-CH-Only", "BLT-Chorea-Only"],
}

_GENE_IDX = {e["gene"]: i for i, e in enumerate(THYROID_GENES)}


def _make_cohort(entry: dict, seed: int, n: int = 40) -> list:
    rng = random.Random(seed)
    gene = entry["gene"]
    r = _RATES[gene]
    age_min, age_max = _AGE_RANGES[gene]
    variants = _TYPE_VARIANTS[gene]

    cohort = []
    sexes = ["M", "F"]
    for j in range(n):
        sex = rng.choice(sexes)
        age_dx = round(rng.uniform(age_min, age_max), 2)
        type_variant = rng.choice(variants)
        on_levothyroxine = rng.random() < r["levothyroxine"]
        goitre_present = rng.random() < r["goitre"]
        had_surgery = rng.random() < r["surgery"]
        thyroid_dysgenesis = rng.random() < r["dysgenesis"]
        autoantibody_negative = rng.random() < _AUTOAB_NEG_RATES[gene]
        suppressed_tsh = rng.random() < _SUPPRESSED_TSH_RATES[gene]
        nbs_detected = rng.random() < _NBS_DETECTED_RATES[gene]
        on_surveillance = rng.random() < 0.70
        positive_perchlorate = rng.random() < _PERCHLORATE_POS_RATES[gene]
        transient_ch = rng.random() < _TRANSIENT_CH_RATES[gene]

        patient = {
            "patient_id": f"THY-{seed:04d}-{j:03d}",
            "gene": gene,
            "sex": sex,
            "age_at_diagnosis_years": age_dx,
            "type_variant": type_variant,
            "on_levothyroxine": on_levothyroxine,
            "goitre_present": goitre_present,
            "had_surgery": had_surgery,
            "thyroid_dysgenesis": thyroid_dysgenesis,
            "autoantibody_negative": autoantibody_negative,
            "suppressed_tsh": suppressed_tsh,
            "nbs_detected": nbs_detected,
            "on_surveillance": on_surveillance,
            "positive_perchlorate": positive_perchlorate,
            "transient_ch": transient_ch,
        }
        cohort.append(patient)
    return cohort


def generate_overview():
    all_patients = []
    for idx, entry in enumerate(THYROID_GENES):
        seed = SEED_BASE + idx
        all_patients.extend(_make_cohort(entry, seed))

    total = len(all_patients)
    levo_count       = sum(1 for p in all_patients if p["on_levothyroxine"])
    goitre_count     = sum(1 for p in all_patients if p["goitre_present"])
    surgery_count    = sum(1 for p in all_patients if p["had_surgery"])
    dysgenesis_count = sum(1 for p in all_patients if p["thyroid_dysgenesis"])
    surv_count       = sum(1 for p in all_patients if p["on_surveillance"])

    gene_summary = {}
    for idx, entry in enumerate(THYROID_GENES):
        gene = entry["gene"]
        cohort = _make_cohort(entry, SEED_BASE + idx)
        gene_summary[gene] = {
            "gene": gene,
            "alt_name": entry["alt_name"],
            "locus": entry["locus"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"].split(";")[0].strip(),
            "disease_category": entry["disease_category"],
            "pathognomonic": entry["pathognomonic"][:300],
            "hormone_profile": entry["hormone_profile"],
            "n_patients": len(cohort),
            "levothyroxine_pct": round(100 * sum(1 for p in cohort if p["on_levothyroxine"]) / len(cohort), 1),
            "goitre_pct": round(100 * sum(1 for p in cohort if p["goitre_present"]) / len(cohort), 1),
            "surgery_pct": round(100 * sum(1 for p in cohort if p["had_surgery"]) / len(cohort), 1),
            "dysgenesis_pct": round(100 * sum(1 for p in cohort if p["thyroid_dysgenesis"]) / len(cohort), 1),
            "surveillance_pct": round(100 * sum(1 for p in cohort if p["on_surveillance"]) / len(cohort), 1),
            "avg_age_dx_years": round(sum(p["age_at_diagnosis_years"] for p in cohort) / len(cohort), 1),
        }

    return {
        "atlas": "Hereditary-Thyroid-Disorder-Atlas",
        "subtitle": "Complete 8-Gene Reference -- TSHR/DUOX2/TPO/TG/SLC5A5/FOXE1/PAX8/NKX2-1",
        "genes_covered": [e["gene"] for e in THYROID_GENES],
        "total_patients": total,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "aggregate_metrics": {
            "levothyroxine_pct": round(100 * levo_count / total, 1),
            "goitre_pct": round(100 * goitre_count / total, 1),
            "surgery_pct": round(100 * surgery_count / total, 1),
            "dysgenesis_pct": round(100 * dysgenesis_count / total, 1),
            "surveillance_pct": round(100 * surv_count / total, 1),
        },
        "gene_summary": gene_summary,
    }


def generate_breakdown():
    breakdown = []
    for idx, entry in enumerate(THYROID_GENES):
        seed = SEED_BASE + idx
        cohort = _make_cohort(entry, seed)
        breakdown.append({
            "gene": entry["gene"],
            "alt_name": entry["alt_name"],
            "locus": entry["locus"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"],
            "disease_category": entry["disease_category"],
            "disease_pathway": entry["disease_pathway"],
            "pathognomonic": entry["pathognomonic"],
            "treatment": entry["treatment"],
            "key_features": entry["key_features"],
            "key_ddx": entry["key_ddx"],
            "hormone_profile": entry["hormone_profile"],
            "onset_age": entry["onset_age"],
            "severity_sds": entry["severity_sds"],
            "ch_type": entry["ch_type"],
            "nbs_indicated": entry["nbs_indicated"],
            "severity": entry["severity"],
            "n_patients": len(cohort),
            "levothyroxine_pct": round(100 * sum(1 for p in cohort if p["on_levothyroxine"]) / len(cohort), 1),
            "goitre_pct": round(100 * sum(1 for p in cohort if p["goitre_present"]) / len(cohort), 1),
            "surgery_pct": round(100 * sum(1 for p in cohort if p["had_surgery"]) / len(cohort), 1),
            "dysgenesis_pct": round(100 * sum(1 for p in cohort if p["thyroid_dysgenesis"]) / len(cohort), 1),
            "surveillance_pct": round(100 * sum(1 for p in cohort if p["on_surveillance"]) / len(cohort), 1),
            "avg_age_dx_years": round(sum(p["age_at_diagnosis_years"] for p in cohort) / len(cohort), 1),
            "sample_patients": cohort[:3],
        })
    return {"gene_breakdowns": breakdown}


def generate_definitions():
    return {
        "gene_entries": {
            entry["gene"]: {
                "gene": entry["gene"],
                "full_name": entry["alt_name"].split(" (")[0].strip(),
                "locus": entry["locus"],
                "protein_size": entry["protein_size"],
                "inheritance": entry["inheritance"].split(";")[0].strip(),
                "disease_name": entry["disease_category"],
                "disease_pathway": entry["disease_pathway"],
                "pathognomonic": entry["pathognomonic"],
                "treatment": entry["treatment"][:400],
                "key_features": entry["key_features"],
                "key_ddx": entry["key_ddx"],
                "hormone_profile": entry["hormone_profile"],
                "onset_age": entry["onset_age"],
                "severity_sds": entry["severity_sds"],
                "ch_type": entry["ch_type"],
                "nbs_indicated": entry["nbs_indicated"],
            }
            for entry in THYROID_GENES
        },
        "thyroid_glossary": {
            "Congenital Hypothyroidism — NBS and Classification": (
                "Congenital hypothyroidism (CH) affects approximately 1 in 2,000-4,000 newborns worldwide "
                "and is the most common preventable cause of intellectual disability. "
                "Neonatal blood spot screening (NBS) detects elevated TSH in the first 48-72 hours of life, "
                "triggering urgent levothyroxine initiation before overt neurological damage occurs. "
                "CH is classified as: (1) thyroid dysgenesis (80% of cases) — athyreosis, hypoplasia, or ectopy "
                "due to defects in thyroid development genes (PAX8, NKX2-1, FOXE1); "
                "(2) dyshormonogenesis (15-20%) — normal or enlarged thyroid (goitre) but a biochemical defect "
                "in hormone synthesis (DUOX2, TPO, TG, SLC5A5, DEHAL1); "
                "(3) central CH — pituitary or hypothalamic defect (TSH deficiency — NBS misses central CH on TSH-based screening). "
                "Levothyroxine dosing: 10-15 mcg/kg/day in neonates to normalise FT4 within 2 weeks; "
                "neurological outcome correlates with pre-treatment FT4 and speed of treatment initiation. "
                "Genetic panel should include: PAX8, NKX2-1, FOXE1, DUOX2, DUOXA2, TPO, TG, SLC5A5. "
                "Transient CH (DUOX2 monoallelic, maternal antibody-mediated, iodine excess/deficiency) requires "
                "trial of L-T4 cessation at age 3 to identify cases not requiring lifelong therapy. "
                "Permanent CH (all dysgenesis types, biallelic dyshormonogenesis): lifelong L-T4 without cessation trial."
            ),
            "Thyroid Dyshormonogenesis — Perchlorate Discharge Test": (
                "The perchlorate discharge test is the gold standard for detecting iodide organification defects "
                "in patients with congenital hypothyroidism and a goitre on ultrasound. "
                "Mechanism: 123I is administered orally — taken up by NIS into thyrocytes — in a normal gland, "
                "TPO rapidly organifies iodide (covalently bonds it to TG tyrosines) — perchlorate (given 2h after "
                "123I) blocks NIS but cannot discharge organified iodide — less than 5% discharge is normal. "
                "In organification defects: iodide accumulates as free (unorganified) iodide — perchlorate blocks "
                "NIS — unorganified iodide cannot be retained — discharged from gland — positive test (>10-15%). "
                "DUOX2 (H2O2 generator deficiency): partial organification defect — discharge typically 10-80%. "
                "TPO (thyroid peroxidase absence): TOTAL organification defect — discharge >90% PATHOGNOMONIC. "
                "TG mutation: mild positive discharge (15-30%) — iodide enters but no TG scaffold to accept it. "
                "SLC5A5/NIS defect: negative perchlorate test (no iodide uptake at all — NIS absent). "
                "SLC26A4/Pendred: partial organification defect similar to DUOX2 — but EVA on CT/MRI + SNHL distinguish. "
                "The test is most useful when goitre is present (suggests dyshormonogenesis not dysgenesis) to determine "
                "whether the defect is at transport (SLC5A5) or organification (TPO/DUOX2/TG) level."
            ),
            "TSH Receptor Mutations — GOF vs LOF": (
                "TSHR (TSH receptor) mutations cause opposite thyroid phenotypes depending on whether the mutation "
                "is gain-of-function (GOF, constitutive activation) or loss-of-function (LOF, impaired signalling). "
                "GOF mutations (FNAH — Familial Non-Autoimmune Hyperthyroidism): predominantly in the transmembrane "
                "domain (Ala623Ile/Val most common); receptor assumes active conformation without TSH binding — "
                "constitutive Gs-cAMP-PKA activation — chronic hyperthyroidism with SUPPRESSED TSH. "
                "CRITICAL DDx from Graves disease: FNAH has NEGATIVE TRAb AND NEGATIVE TPO-Ab — "
                "autoantibodies are the key distinguishing test; Graves disease has positive TRAb. "
                "Radioiodine NOT curative in FNAH: RAI destroys thyrocytes but all residual cells carry the GOF allele — "
                "hyperthyroid regrowth — methimazole or total thyroidectomy required. "
                "Familial Gestational Hyperthyroidism (FGH): TSHR ectodomain variants with exquisite hCG sensitivity — "
                "first-trimester hCG surge — gestational thyrotoxicosis — hyperemesis gravidarum; "
                "presents in first pregnancy with no prior thyroid disease and negative autoantibodies. "
                "LOF mutations (TSH resistance): Gs coupling impaired — blunted cAMP response — TSH elevated "
                "compensatorily but FT4 normal or low; gland size normal; "
                "usually AR biallelic for significant resistance. "
                "Somatic TSHR GOF: same mutations in toxic adenoma (hot nodule on scan); "
                "single somatic mutation — single autonomously functioning nodule — treated with RAI or surgery."
            ),
            "Bamforth-Lazarus Syndrome — FOXE1": (
                "Bamforth-Lazarus syndrome is caused by biallelic LOF mutations in FOXE1 (Forkhead Box E1, also called TTF-2) "
                "and represents one of the rarest hereditary thyroid disorders, with fewer than 50 cases reported worldwide. "
                "PATHOGNOMONIC TETRAD: (1) thyroid agenesis (athyreosis) — complete absence of thyroid on ultrasound "
                "and Tc scan; (2) cleft palate — failure of palate fusion during embryogenesis; "
                "(3) bifid epiglottis or choanal atresia — failure of epiglottis/choanal morphogenesis; "
                "(4) spiky (coarse) hair — FOXE1 expression in hair follicle precursors. "
                "NEONATAL EMERGENCY: combination of choanal atresia (neonates are obligate nasal breathers — "
                "choanal obstruction — hypoxia) AND profound CH (athyreosis — no thyroid hormones) "
                "requires SIMULTANEOUS airway management + IV levothyroxine in the first hours of life. "
                "Oral levothyroxine is not feasible initially due to cleft palate. "
                "Airway management: McGovern nasal trumpet, oral airway, early surgical choanal atresia repair. "
                "Levothyroxine: IV initially (10-15 mcg/kg/day), transition to oral after palate repair. "
                "FOXE1 heterozygous variants: the common GWAS SNP rs965513 at 9q22.33 (FOXE1 locus) is associated "
                "with papillary thyroid cancer risk in the general population — completely separate from the biallelic syndrome. "
                "Carrier parents of Bamforth-Lazarus children (obligate heterozygotes) may be reassured that "
                "standard thyroid cancer screening is appropriate but no specific heightened surveillance is proven."
            ),
            "Brain-Lung-Thyroid Syndrome — NKX2-1": (
                "Brain-Lung-Thyroid (BLT) syndrome is caused by NKX2-1 haploinsufficiency and is the definitive "
                "hereditary disorder linking congenital hypothyroidism with a movement disorder and respiratory phenotype. "
                "NKX2-1 (also called TTF-1 or TITF1) is a homeodomain transcription factor expressed in three tissues: "
                "thyroid follicular cells, lung type II pneumocytes, and basal ganglia (striatal interneurons). "
                "THYROID COMPONENT (50-70%): thyroid dysgenesis (hypoplastic or ectopic) — CH detected on NBS; "
                "not always complete athyreosis, distinguishing from FOXE1; some patients have only subclinical CH. "
                "LUNG COMPONENT (50-70%): NKX2-1 drives SP-B (SFTPB) and SP-C (SFTPC) surfactant protein expression "
                "in type II pneumocytes; haploinsufficiency — surfactant deficiency — neonatal RDS at birth "
                "requiring oxygen and ventilatory support; later interstitial lung disease in some adults. "
                "BRAIN COMPONENT (60-80%): NKX2-1 expressed in striatal GABAergic interneurons; "
                "haploinsufficiency — impaired inhibitory interneuron maturation — disinhibition of motor circuits — "
                "chorea (involuntary, dance-like) with onset 1-5 years; non-progressive; improves in adulthood; "
                "NO cognitive decline = BENIGN HEREDITARY CHOREA (BHC) — key DDx from Huntington. "
                "Tetrabenazine: AVOID in NKX2-1 BHC (may paradoxically worsen symptoms). "
                "14q13.3 deletion syndrome: contiguous deletion beyond NKX2-1 — broader features; "
                "chromosomal microarray mandatory in all NKX2-1 patients. "
                "TTF-1 IHC (NKX2-1 protein): somatic expression in lung adenocarcinoma used in pathological staging — "
                "completely separate from the germline BLT syndrome."
            ),
            "Iodide Transport Defect — NIS/SLC5A5": (
                "Iodide transport defect (ITD) is caused by biallelic LOF mutations in SLC5A5, encoding the Sodium-Iodide "
                "Symporter (NIS), and is the only dyshormonogenesis where radioiodine uptake is absent rather than elevated. "
                "NIS is a 13-transmembrane electrogenic cotransporter at the basolateral membrane of thyroid follicular cells; "
                "it cotransports 2 Na+ ions with each I- ion, driven by the Na+ electrochemical gradient from Na+/K+-ATPase, "
                "concentrating intracellular iodide 20-40x above plasma. "
                "DIAGNOSTIC HALLMARKS: (1) 24h 123I thyroid uptake <5% (normal 10-30%) despite elevated TSH — "
                "TSH normally upregulates NIS, so low uptake despite high TSH = NIS LOF; "
                "(2) Pertechnetate (TcO4-) uptake also absent — NIS transports both iodide and pertechnetate; "
                "(3) Saliva/plasma iodide ratio <30 (normal >40) — NIS expressed in salivary gland acinar cells "
                "concentrates iodide into saliva; NIS LOF — saliva iodide low — low ratio. "
                "TREATMENT PITFALLS: iodide supplementation does not work (transport machinery absent — oral iodide "
                "cannot enter thyrocytes regardless of dose); radioiodine not useful for goitre (no NIS to concentrate "
                "131I). Compressive goitre: total or subtotal thyroidectomy, not RAI. "
                "CLINICAL IMPORTANCE IN THYROID CANCER: NIS is expressed in well-differentiated thyroid cancers and "
                "exploited for 131I whole-body scan and ablation after thyroidectomy; somatic downregulation of NIS in "
                "poorly differentiated thyroid cancer reduces RAI efficacy — NIS re-expression strategies "
                "(BRAF inhibitors, MEK inhibitors) are active research areas."
            ),
        },
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(generate_overview(), indent=2)[:2000])
    print("\n=== BREAKDOWN (gene 0 only) ===")
    bd = generate_breakdown()
    print(json.dumps(bd["gene_breakdowns"][0], indent=2)[:2000])
    print("\n=== DEFINITIONS (first key) ===")
    defn = generate_definitions()
    first_key = next(iter(defn["gene_entries"]))
    print(json.dumps(defn["gene_entries"][first_key], indent=2)[:1500])
