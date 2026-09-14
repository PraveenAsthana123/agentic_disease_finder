#!/usr/bin/env python3
"""Hereditary-Short-Stature-Atlas — Complete 8-Gene Hereditary Short Stature Atlas
(SHOX · GH1 · GHR · IGF1 · IGFALS · STAT5B · CUL7 · NPR2).

SHOX     (Short Stature Homeobox; 758 aa; ~29 kDa; Xp22.33/Yp11.32 PAR1; AD/AR;
          AD haploinsufficiency → Léri-Weill dyschondrosteosis (LWD) — Madelung
          deformity PATHOGNOMONIC; AR biallelic null → Langer mesomelic dysplasia (LMD);
          rGH FDA-approved for SHOX deficiency; PAR1 escapes X-inactivation;
          seed SEED_BASE+0).
GH1      (Growth Hormone 1; 217 aa prepro / 22 kDa mature; 17q23.3; AR/AD;
          Isolated GHD types IA (AR null, anti-GH antibodies on therapy) / IB (AR
          splice/missense) / II (AD dominant-negative splice) / III (X-linked via BTK);
          GH stimulation <10 mcg/L on 2 tests PATHOGNOMONIC; recombinant GH curative;
          seed SEED_BASE+1).
GHR      (Growth Hormone Receptor; 638 aa; ~70 kDa; 5p13.1; AR;
          Laron syndrome — GH insensitivity; HIGH GH + LOW IGF-1 + LOW IGFBP-3
          PATHOGNOMONIC; somatomedin generation test fails; Ecuadorian founder community;
          mecasermin (recombinant IGF-1) treatment; seed SEED_BASE+2).
IGF1     (Insulin-Like Growth Factor 1; 195 aa prepro / 70 aa mature / 7.6 kDa; 12q23.2; AR;
          IGF-1 deficiency; IUGR + SNHL + MICROCEPHALY TRIAD PATHOGNOMONIC;
          GH elevated distinguishes from GHR Laron; mecasermin treatment;
          seed SEED_BASE+3).
IGFALS   (IGF Acid Labile Subunit; 605 aa; ~67 kDa; 16p13.3; AR;
          ALS deficiency — ternary IGF complex disrupted; MILD phenotype;
          VERY LOW IGFBP-3 disproportionate to IGF-1 PATHOGNOMONIC;
          pubertal growth spurt preserved; often asymptomatic; seed SEED_BASE+4).
STAT5B   (Signal Transducer and Activator of Transcription 5B; 786 aa; ~90 kDa; 17q21.2; AR;
          GH insensitivity + immune dysregulation; HIGH GH + LOW IGF-1 + RECURRENT
          SEVERE VARICELLA + T-CELL LYMPHOPENIA PATHOGNOMONIC; mecasermin;
          seed SEED_BASE+5).
CUL7     (Cullin 7; 1698 aa; ~192 kDa; 6p21.1; AR;
          3-M syndrome — proportionate severe dwarfism −8 to −10 SDS;
          TRIANGULAR FACE + PROMINENT HEEL + SLENDER TUBULAR BONES PATHOGNOMONIC;
          NORMAL GH/IGF-1 axis; NORMAL intelligence; seed SEED_BASE+6).
NPR2     (Natriuretic Peptide Receptor 2 / GC-B; 1047 aa; ~117 kDa; 9p13.3; AD/AR;
          AD haploinsufficiency → short stature + brachydactyly type E2 (short metacarpals)
          PATHOGNOMONIC; AR biallelic null → acromesomelic dysplasia Maroteaux (ADM)
          — severe mesomelic limb shortening PATHOGNOMONIC; vosoritide NOT effective for
          NPR2 LOF (receptor itself absent — contrast achondroplasia FGFR3 where NPR2
          intact); seed SEED_BASE+7).
320-patient aggregate cohort (8 x 40, seeds 2510-2517).
"""

import random

SEED_BASE = 2510

SHORT_STATURE_GENES = [
    # -- SHOX -- Léri-Weill dyschondrosteosis / Langer mesomelic dysplasia ----------------
    {
        "gene": "SHOX",
        "alt_name": (
            "SHOX (SHOX-758aa-Xp22.33/Yp11.32-PAR1 / AD/AR -- "
            "AD-HAPLOINSUFFICIENCY-LERI-WEILL-DYSCHONDROSTEOSIS -- "
            "MADELUNG-DEFORMITY-PATHOGNOMONIC-DORSAL-SUBLUXATION-DISTAL-ULNA -- "
            "AR-BIALLELIC-NULL-LANGER-MESOMELIC-DYSPLASIA-SEVERE -- "
            "rGH-FDA-APPROVED-SHOX-DEFICIENCY-ESCAPES-X-INACTIVATION-PAR1)"
        ),
        "protein": (
            "SHOX -- Xp22.33/Yp11.32-PAR1 AD/AR -- SHOX-758aa -- "
            "Short-Stature-Homeobox-Protein-29kDa-Paired-Homeodomain-Transcription-Factor -- "
            "Expressed-Chondrocytes-Limb-Buds-Pharyngeal-Arches -- "
            "OAR-Domain-C-terminal-Protein-Protein-Interaction-Enhancer-Binding -- "
            "Pseudoautosomal-Region-1-PAR1-Present-on-Both-X-and-Y-Chromosomes -- "
            "Escapes-X-Inactivation-Diploid-Dosage-Required-Males-and-Females -- "
            "SHOX-Enhancers-35kb-Downstream-PAR1-Deletions-Same-Phenotype-Without-Coding -- "
            "OMIM-Gene-312865-Disease-LWD-127300-LMD-249700"
        ),
        "locus": "Xp22.33/Yp11.32 (PAR1)",
        "protein_size": "758 aa / 29 kDa",
        "inheritance": (
            "AD (autosomal dominant equivalent — haploinsufficiency) for Léri-Weill dyschondrosteosis (LWD); "
            "AR (biallelic null) for Langer mesomelic dysplasia (LMD); "
            "PAR1 (pseudoautosomal region 1) — located on both X and Y chromosomes, escapes X-inactivation; "
            "therefore diploid dosage required; behaves as autosomal despite X/Y location; "
            "deletions of SHOX + downstream PAR1 enhancer region (35 kb downstream) are equally pathogenic; "
            "enhancer-only deletions missed by standard exome → MLPA or array CGH mandatory; "
            "females with LWD often more severely affected than males (X-linked pattern with extra X-dose effect); "
            "also seen in Turner syndrome (45,X) — one mechanism of short stature in Turner"
        ),
        "disease_category": "Léri-Weill dyschondrosteosis (LWD, AD) / Langer mesomelic dysplasia (LMD, AR) / SHOX-deficiency short stature",
        "disease_pathway": (
            "SHOX is a transcription factor expressed in growth plate chondrocytes (proliferating and prehypertrophic zones) "
            "in the limbs. It activates transcription of genes required for chondrocyte hypertrophy and endochondral ossification. "
            "SHOX directly activates FGFR3 and BMP4 to coordinate growth plate chondrocyte maturation. "
            "Haploinsufficiency (one functional copy): mesomelic segment predominantly affected — radius and ulna, tibia and fibula "
            "grow at different rates → dorsal subluxation of the distal ulna relative to the curved radius "
            "(Madelung deformity). Growth velocity reduced −2 to −3 SD; disproportionate short stature. "
            "Biallelic null (LMD): severe mesomelic dwarfism with rudimentary fibula; short-limbed dwarfism "
            "comparable in severity to achondroplasia but mesomelic pattern (middle segments of limbs). "
            "rGH increases growth velocity by augmenting paracrine SHOX-regulated pathways in intact growth plates. "
            "Spontaneous Madelung deformity correction does not occur — corrective osteotomy required if symptomatic. "
            "Downstream PAR1 enhancer deletions: same coding protein, same phenotype — "
            "critical diagnostic pitfall as these deletions lie 35 kb from SHOX coding region and are outside exome target regions."
        ),
        "pathognomonic": (
            "MADELUNG DEFORMITY PATHOGNOMONIC FOR SHOX/LWD: dorsal and ulnar subluxation of the distal ulna relative "
            "to the curved (bowed) radius — 'dinner fork' appearance of wrist on X-ray; "
            "radiolucent triangular area at distal radius metaphysis (Vickers ligament tenting). "
            "Short stature disproportionate — forearm+leg relatively shorter than trunk (mesomelic). "
            "Wrist X-ray findings appear in mid-childhood (7-12 years); earlier in females. "
            "LANGER MESOMELIC DYSPLASIA: biallelic SHOX null → severe mesomelic limb shortening at birth, "
            "rudimentary fibula, hypoplastic distal ulna — diagnosed at birth radiographically. "
            "TURNER SYNDROME TRAP: SHOX monosomy contributes ~50% of Turner short stature; "
            "testing SHOX in 45,X patients guides rGH dose response expectations. "
            "ENHANCER DELETION DIAGNOSTIC PITFALL: exome normal → must do MLPA or SNP array for PAR1 enhancer."
        ),
        "treatment": (
            "Recombinant human GH (rhGH, somatropin) — FDA-approved specifically for SHOX deficiency (2006): "
            "approved dose 0.35 mg/kg/week (higher than GHD dose of 0.18-0.3 mg/kg/week). "
            "Start before epiphyseal fusion (best results age 4-12 years). "
            "Mean height gain ~0.5-0.7 SD (~3-5 cm additional adult height) in controlled trials. "
            "Madelung deformity: conservative initially (rest, wrist splinting); "
            "Vickers ligament release (physiolysis) halts progression and allows some remodelling; "
            "corrective osteotomy (Suave-Kapandji or Darrach procedure) for established adult deformity. "
            "Leg lengthening (Ilizarov distraction osteogenesis) for LMD — requires multidisciplinary team. "
            "Monitor: IGF-1 every 6 months, bone age annually, wrist X-ray to track Madelung. "
            "Contraindications: active malignancy, Prader-Willi with severe obesity (sleep apnoea risk). "
            "Cascade genetic testing: parents + siblings — PAR1 MLPA, not just gene sequencing."
        ),
        "key_features": [
            "Madelung deformity — dorsal ulnar subluxation — pathognomonic wrist X-ray finding",
            "Mesomelic short stature — forearms and lower legs disproportionately short",
            "PAR1 enhancer deletions (35 kb downstream SHOX) cause same LWD — exome misses these",
            "rGH FDA-approved specifically for SHOX deficiency (dose higher than GHD)",
            "Langer mesomelic dysplasia — AR biallelic null — severe form at birth",
            "SHOX monosomy is key mechanism of short stature in Turner syndrome (45,X)",
            "Spontaneous Madelung correction does not occur — surgical if symptomatic adult",
        ],
        "key_ddx": [
            "Idiopathic short stature: no Madelung, no disproportion, normal wrist X-ray",
            "Achondroplasia (FGFR3 GOF): rhizomelic pattern (proximal limbs), not mesomelic",
            "Turner syndrome (45,X): also has SHOX monosomy — chromosomes confirm; same pathomechanism overlap",
            "Hypochondroplasia (FGFR3 mild GOF): mild rhizomelia, genetic testing separates",
            "Dyschondrosteosis due to BNIP3L/Nix: rare, different molecular defect, similar phenotype",
            "GHD: proportionate short stature, no Madelung, GH stimulation low (normal in SHOX deficiency)",
        ],
        "treatment_response": "rGH therapy — +0.5 to +0.7 SD additional adult height",
        "hormone_profile": "GH normal; IGF-1 low-normal; no endocrine axis defect",
        "onset_age": "Childhood growth deceleration; Madelung deformity 7-12 years",
        "severity_sds": "−2 to −3 SDS (LWD, AD); −4 to −7 SDS (LMD, AR)",
        "autosomal_recessive_risk": True,
        "gh_deficiency": False,
        "mecasermin_indicated": False,
    },

    # -- GH1 -- Isolated GH Deficiency types IA/IB/II -------------------------------------
    {
        "gene": "GH1",
        "alt_name": (
            "GH1 (GH1-217aa-prepro-17q23.3 / AR-type-IA-IB / AD-type-II / XL-type-III -- "
            "ISOLATED-GH-DEFICIENCY-IGHD -- "
            "GH-STIMULATION-<10-mcg/L-ON-2-TESTS-PATHOGNOMONIC -- "
            "TYPE-IA-ANTI-GH-ANTIBODIES-ON-THERAPY-KEY-DDx -- "
            "rGH-CURATIVE-TYPES-IB-AND-II)"
        ),
        "protein": (
            "GH1 -- 17q23.3 AR/AD -- GH1-217aa-prepro -- "
            "Growth-Hormone-1-22kDa-Major-Isoform-191aa-Mature-Peptide-Single-Chain-4-Alpha-Helix-Bundle -- "
            "20kDa-Alternatively-Spliced-Isoform-Exon-3-Deleted-15pct-Circulating-GH -- "
            "GH-Locus-5-Gene-Cluster-on-17q23.3-GH1-CSHL1-GH2-CSH1-CSH2 -- "
            "Type-IA-Biallelic-Null-No-GH-Protein-→-Anti-GH-Antibodies-on-rGH-Therapy -- "
            "Type-IB-Missense-Splice-AR-Partial-GH-Deficiency-No-Antibodies -- "
            "Type-II-IVS3+1GT→AT-Dominant-Negative-22kDa-Absent-17.5kDa-Exon3-Del-Accumulates -- "
            "OMIM-Gene-139250-Disease-IGHD-173100"
        ),
        "locus": "17q23.3",
        "protein_size": "217 aa (prepro) / 22 kDa (mature major isoform)",
        "inheritance": (
            "Multiple inheritance patterns for GH1 gene: "
            "Type IA (AR, biallelic null/deletion): most severe; GH completely absent; anti-GH antibodies form on treatment, "
            "blocking rGH therapy — IVIG or desensitisation may help; "
            "Type IB (AR, splice or missense): partial deficiency; milder; responds well to rGH; "
            "Type II (AD, IVS3+1G>T most common): dominant-negative exon-3-deleted 17.5 kDa isoform traps and degrades "
            "normal 22 kDa GH in secretory granules → pituitary somatotroph apoptosis over time; "
            "also associated with progressive hypopituitarism (ACTH, TSH loss later); "
            "Type III (X-linked): X-linked agammaglobulinaemia (BTK gene) + GHD phenotype — "
            "separate molecular mechanism via GH1 locus regulatory deletion"
        ),
        "disease_category": "Isolated GH Deficiency (IGHD) types IA, IB, II — congenital pituitary somatotroph failure",
        "disease_pathway": (
            "GH1 encodes the 22 kDa isoform of growth hormone (major circulating form). "
            "GH is secreted in pulses from anterior pituitary somatotrophs, stimulated by GHRH and ghrelin, "
            "inhibited by somatostatin. GH acts on liver (primary source of IGF-1) and peripheral tissues. "
            "TYPE IA: biallelic GH1 deletion or nonsense → no GH protein → profound GH deficiency; "
            "pituitary MRI often shows small anterior pituitary; on rhGH therapy, immune system mounts "
            "anti-GH antibodies (tolerant to 'foreign' protein it never saw) → blocks therapy. "
            "TYPE II: IVS3+1G>T splice site → exon 3 skipped → 17.5 kDa GH isoform accumulates → "
            "dominant-negative effect trapping 22 kDa GH in secretory granules → somatotroph apoptosis; "
            "progressive loss of other pituitary hormones (ACTH, TSH) over years → "
            "mandatory annual pituitary function testing. "
            "TYPE IB: missense or splice reducing GH secretion; partial production maintained; "
            "IGF-1 generation test shows partial response (unlike GHR/IGF1 where minimal/absent response). "
            "Pituitary MRI: may show small or absent GH-secreting cells; ectopic posterior pituitary in some."
        ),
        "pathognomonic": (
            "GH STIMULATION TEST <10 mcg/L ON TWO SEPARATE TESTS WITH DIFFERENT STIMULI PATHOGNOMONIC "
            "(insulin tolerance test + glucagon; or arginine + GHRH; per national guidelines). "
            "Type IA specific: anti-GH antibodies on therapy — loss of catch-up growth after initial response "
            "on recombinant GH is PATHOGNOMONIC for Type IA; antibody titre confirms. "
            "Type II specific: family history (AD) + progressive hypopituitarism over years (ACTH/TSH loss after GHD). "
            "Pituitary MRI: small anterior pituitary or hypoplastic pituitary on T1; ectopic neurohypophysis "
            "in pituitary stalk transection syndrome (overlap). "
            "Proportionate short stature — normal body proportions (unlike skeletal dysplasias); "
            "adiposity, high-pitched voice, immature facies, delayed bone age. "
            "IGF-1 and IGFBP-3 both low — if IGF-1 low with GH stimulation test normal, consider GHR/IGFALS."
        ),
        "treatment": (
            "Recombinant human GH (somatropin) injections daily subcutaneous — first-line for all GH1 types. "
            "Type IA anti-GH antibody management: continue rhGH (antibodies often reduce with continued exposure); "
            "IVIG desensitisation protocols; alternative: GH secretagogues (ghrelin agonists investigational) bypass antibodies. "
            "Dose: 0.025-0.05 mg/kg/day (25-50 µg/kg/day) in children; "
            "adjust based on IGF-1 target (mid-normal range for age/sex). "
            "Type II: consider early rhGH before somatotroph apoptosis is complete; "
            "annual pituitary function monitoring (cortisol stimulation, TFTs, prolactin); "
            "may need hydrocortisone + levothyroxine replacement as hypopituitarism progresses. "
            "Transition to adult GH therapy (lower dose 0.2-0.5 mg/day) after epiphyseal fusion "
            "for metabolic effects (lipids, bone density, quality of life). "
            "Contraindications: active malignancy, intracranial hypertension. "
            "POLG1 screen: not specifically indicated for GH1 but standard genetic workup before valproate "
            "if concurrent epilepsy (separate CPIC requirement)."
        ),
        "key_features": [
            "GH stimulation <10 mcg/L on 2 tests with different stimuli required for diagnosis",
            "Type IA — anti-GH antibodies form on rhGH therapy — critical monitoring",
            "Type II (AD) — dominant-negative — progressive hypopituitarism (ACTH/TSH loss years later)",
            "Proportionate short stature with adiposity, delayed bone age, immature facies",
            "Low IGF-1 + Low IGFBP-3 + Low GH on stimulation test triad",
            "Pituitary MRI mandatory — small anterior pituitary, ectopic posterior, or empty sella",
        ],
        "key_ddx": [
            "GHR (Laron): HIGH GH + LOW IGF-1 — not low GH; GHR genetic testing confirms",
            "IGFALS deficiency: low IGFBP-3, GH elevated/normal, milder short stature",
            "STAT5B: HIGH GH + immune dysregulation (varicella, lymphopenia) — not in GH1",
            "Constitutional delay (CDGA): GH stimulation may be falsely low; bone age + priming separates",
            "Hypothyroidism-induced GHD: TSH elevated; GH normalises after thyroxine replacement",
            "PROP1/POU1F1 mutations: multiple pituitary hormone deficiency — not isolated GHD",
        ],
        "treatment_response": "rGH therapy — near-complete height normalisation in types IB/II if started early",
        "hormone_profile": "GH low (<10 mcg/L on stimulation); IGF-1 low; IGFBP-3 low",
        "onset_age": "Neonatal (hypoglycaemia, micropenis in type IA) to childhood growth failure",
        "severity_sds": "−2 to −5 SDS (severity varies by type; IA most severe)",
        "autosomal_recessive_risk": True,
        "gh_deficiency": True,
        "mecasermin_indicated": False,
    },

    # -- GHR -- Laron syndrome (GH insensitivity) -----------------------------------------
    {
        "gene": "GHR",
        "alt_name": (
            "GHR (GHR-638aa-5p13.1 / AR -- "
            "LARON-SYNDROME-GH-INSENSITIVITY-SYNDROME -- "
            "HIGH-GH-LOW-IGF-1-LOW-IGFBP-3-PATHOGNOMONIC -- "
            "SOMATOMEDIN-GENERATION-TEST-FAILS -- "
            "MECASERMIN-RECOMBINANT-IGF-1-TREATMENT)"
        ),
        "protein": (
            "GHR -- 5p13.1 AR -- GHR-638aa -- "
            "Growth-Hormone-Receptor-70kDa-Class-I-Cytokine-Receptor-Superfamily -- "
            "Extracellular-Binding-Domain-620-aa-Single-TM-Helix-Intracellular-Box1-Box2-JAK2-Docking -- "
            "GHR-Homodimerises-on-GH-Binding-2:1-Ratio → JAK2-Phosphorylation → STAT5B-Activation -- "
            "GHBP-GH-Binding-Protein-Circulating-Shed-Extracellular-Domain-GHR -- "
            "Low-GHBP-Indicates-GHR-Extracellular-Domain-Defect-High-GHBP-Suggests-Post-Receptor-Defect -- "
            "Ecuadorian-Founder-p.E180-Splice-Most-Studied-Cohort-Valter-Igulden-1984-First-Described -- "
            "Exon-3-Deleted-d3-GHR-Polymorphism-NOT-Pathological-Common-10-15pct-Population -- "
            "OMIM-Gene-600946-Disease-Laron-262500"
        ),
        "locus": "5p13.1",
        "protein_size": "638 aa / ~70 kDa",
        "inheritance": (
            "AR (autosomal recessive, biallelic); most common genetic cause of GH insensitivity; "
            "homozygous or compound heterozygous LOF in GHR; "
            "Laron described it 1966 — first clinically recognised GH resistance syndrome; "
            "Ecuadorian founder community (E180 splice site): ~100 affected individuals, studied for cancer/IGF protection; "
            "Ashkenazi Jewish founder variants: D152H, I179M; "
            "GHBP (GH binding protein) level: low if extracellular domain mutation; "
            "normal-high GHBP indicates post-receptor defect (STAT5B); "
            "Heterozygous GHR: partial resistance possible (dominant-negative effect of some variants) "
            "but most carriers have normal/near-normal stature"
        ),
        "disease_category": "Laron syndrome — Primary GH Insensitivity / GH Receptor deficiency (Laron dwarfism)",
        "disease_pathway": (
            "GH (growth hormone) binds to GHR on hepatocytes and peripheral tissues. "
            "GH binding triggers GHR homodimerisation → JAK2 kinase activation → STAT5B phosphorylation → "
            "nuclear translocation → IGF-1 gene transcription (primarily in liver). "
            "In GHR LOF: GH cannot signal → IGF-1 production severely impaired → "
            "IGF-1 and IGFBP-3 both very low despite elevated serum GH "
            "(GH accumulates as pituitary cannot receive negative feedback from liver IGF-1). "
            "Growth plates: IGF-1 required for growth plate chondrocyte proliferation and IGF-1R signalling; "
            "without IGF-1, growth plates fail to expand → severe proportionate dwarfism. "
            "Metabolic features: GH promotes lipolysis directly → without GHR signalling → "
            "increased adiposity (especially abdominal), hypoglycaemia in infants; "
            "paradoxically, Laron syndrome cohorts show dramatically lower cancer incidence and insulin-like "
            "metabolic protection (Guevara-Aguirre 2011 Science Translational Medicine). "
            "GHBP (shed extracellular domain of GHR): low GHBP indicates extracellular domain GHR mutation; "
            "normal GHBP with same phenotype suggests post-receptor defect → sequence STAT5B."
        ),
        "pathognomonic": (
            "HIGH GH (>10 ng/mL, often >20-50 ng/mL) WITH LOW IGF-1 AND LOW IGFBP-3 PATHOGNOMONIC. "
            "SOMATOMEDIN GENERATION TEST FAILS: exogenous rhGH administration (0.1 U/kg for 4 days) "
            "does NOT raise IGF-1 >15 mcg/L — confirms GH resistance at receptor/post-receptor level. "
            "Clinical: severe proportionate dwarfism (−4 to −10 SDS); "
            "blue sclera; frontal bossing; saddle nose (hypoplastic nasal bridge); small hands/feet; "
            "delayed dentition; high-pitched voice; neonatal hypoglycaemia. "
            "GHBP low → confirms extracellular GHR domain defect. "
            "Insulin sensitivity INCREASED (paradox — GH normally promotes insulin resistance; "
            "Laron patients often hypoglycaemic, not diabetic)."
        ),
        "treatment": (
            "Recombinant IGF-1 (mecasermin, Increlex): FDA-approved for severe primary IGF-1 deficiency "
            "including GHR deficiency (Laron syndrome). "
            "Dose: 0.04-0.12 mg/kg/dose twice daily SC with meals (range 40-120 µg/kg/dose); "
            "MANDATORY: administer WITH FOOD to prevent hypoglycaemia; "
            "monitor glucose 30-60 minutes after injection in new patients. "
            "Height gain: ~+4-6 cm additional in first year; variable benefit; "
            "best results when started young (before age 3-5); "
            "growth plates must remain open — limited benefit after epiphyseal fusion. "
            "Hypoglycaemia is MOST COMMON serious adverse event — "
            "titrate slowly, always give with food, never skip meal after injection. "
            "CONTRAINDICATION: rhGH therapy is INEFFECTIVE (GHR absent/non-functional); "
            "do NOT administer rhGH in Laron syndrome — no receptor to transduce signal. "
            "NEVER confuse with GH1 deficiency treatment (rhGH works in GH1; fails in GHR). "
            "Annual monitoring: IGF-1, IGFBP-3, glucose, fasting insulin, lipids. "
            "Tonsil/adenoid hypertrophy reported on mecasermin — ENT review if symptoms."
        ),
        "key_features": [
            "HIGH GH + LOW IGF-1 + LOW IGFBP-3 triad — distinguishes from GH deficiency (low GH in GHD)",
            "Somatomedin generation test fails — confirms GH resistance",
            "Low GHBP confirms extracellular GHR domain defect",
            "Severe proportionate dwarfism; frontal bossing; saddle nose; delayed dentition",
            "Mecasermin (recombinant IGF-1) is treatment — NOT rhGH (receptor absent)",
            "Neonatal hypoglycaemia — heightened insulin sensitivity paradox",
            "Ecuadorian cohort: remarkable cancer and diabetes protection — IGF-1 pathway research",
        ],
        "key_ddx": [
            "GH1 deficiency (IGHD): LOW GH on stimulation (not high); IGF-1 rises with rhGH therapy",
            "STAT5B deficiency: same GH↑/IGF-1↓ profile but WITH immune dysregulation (varicella, lymphopenia)",
            "IGF1 gene deficiency: high GH, low IGF-1 but triad includes SNHL + microcephaly",
            "IGFALS deficiency: mild short stature; disproportionately low IGFBP-3; pubertal spurt preserved",
            "Malnutrition-related GH resistance: acquired; low IGF-1 with elevated GH; nutritional history separates",
            "Bioinactive GH: GH structurally abnormal, normal immunoassay but cannot activate GHR — functional bioassay separates",
        ],
        "treatment_response": "Mecasermin — +4-6 cm/year first year; partial, not complete, height normalisation",
        "hormone_profile": "GH elevated (>10 ng/mL); IGF-1 very low; IGFBP-3 very low; GHBP low",
        "onset_age": "Neonatal (hypoglycaemia) to infancy/childhood (growth failure)",
        "severity_sds": "−4 to −10 SDS (severe)",
        "autosomal_recessive_risk": True,
        "gh_deficiency": False,
        "mecasermin_indicated": True,
    },

    # -- IGF1 -- IGF-1 deficiency ----------------------------------------------------------
    {
        "gene": "IGF1",
        "alt_name": (
            "IGF1 (IGF1-195aa-prepro-12q23.2 / AR -- "
            "IGF-1-DEFICIENCY -- "
            "IUGR+SNHL+MICROCEPHALY-TRIAD-PATHOGNOMONIC -- "
            "HIGH-GH-DISTINGUISHES-FROM-GHR-LARON -- "
            "MECASERMIN-RECOMBINANT-IGF-1-TREATMENT)"
        ),
        "protein": (
            "IGF1 -- 12q23.2 AR -- IGF1-195aa-prepro -- "
            "Insulin-Like-Growth-Factor-1-7.6kDa-Mature-70aa-Single-Chain-Polypeptide-AB-C-D-domains -- "
            "Structurally-Similar-Proinsulin-A-B-Chain-Disulfide-Bridges -- "
            "Liver-Primary-Source-Circulating-IGF-1-GH-Dependent -- "
            "Bone-Muscle-Other-Tissues-Local-Autocrine-Paracrine-Production-GH-Independent-Component -- "
            "Circulates-Bound-IGFBPs-Especially-IGFBP-3-and-ALS-Ternary-Complex -- "
            "IGF1R-Signals-PI3K-AKT-mTOR-MAPK-Proliferation-Survival-Growth -- "
            "OMIM-Gene-147440-Disease-IGF-1-Deficiency-608747"
        ),
        "locus": "12q23.2",
        "protein_size": "195 aa (prepro) / 70 aa / 7.6 kDa (mature)",
        "inheritance": (
            "AR (autosomal recessive, biallelic); very rare — fewer than 20 patients described worldwide; "
            "first patient described 1996 (Woods et al. NEJM); "
            "consanguineous pedigrees predominantly; "
            "homozygous or compound heterozygous LOF in IGF1; "
            "brain, cochlea, and growth plate all critically dependent on local IGF-1 — "
            "explaining the triad of short stature + SNHL + microcephaly; "
            "heterozygous IGF1 variants: possible dose effect, milder short stature (complex trait contribution)"
        ),
        "disease_category": "Primary IGF-1 deficiency — severe prenatal and postnatal growth failure with sensorineural hearing loss and microcephaly",
        "disease_pathway": (
            "IGF-1 (insulin-like growth factor 1) mediates most anabolic effects of GH. "
            "Liver-derived IGF-1 (endocrine): GH binds GHR on hepatocytes → JAK2/STAT5B → IGF-1 gene transcription; "
            "IGF-1 released into circulation bound to IGFBP-3 and ALS (ternary complex, 120-150 kDa). "
            "Local (autocrine/paracrine) IGF-1: also produced by many tissues independent of GH. "
            "In IGF1 gene LOF: NO IGF-1 protein produced → absent GH-dependent and GH-independent IGF-1. "
            "BRAIN: IGF-1 critical for neural cell proliferation (fetal), myelination, cerebellar neurogenesis → "
            "microcephaly and intellectual disability (variable severity). "
            "COCHLEA: IGF-1 essential for cochlear hair cell survival and auditory nerve maturation → "
            "sensorineural hearing loss (SNHL) — dose-dependent with severity of deficiency. "
            "GROWTH PLATES: IGF-1R (receptor) signalling → chondrocyte proliferation → longitudinal bone growth; "
            "absent IGF-1 → IUGR (prenatal, because IGF-1 is produced independently of GH in fetus) "
            "and severe postnatal growth failure. "
            "GH elevated (no IGF-1 feedback to pituitary) — this distinguishes IGF1 LOF from GH1 deficiency. "
            "Insulin sensitivity: low IGF-1 → reduced IGF-1R-mediated glucose uptake → mild insulin resistance "
            "(contrast with Laron syndrome: paradoxically higher insulin sensitivity despite absent IGF-1)."
        ),
        "pathognomonic": (
            "TRIAD OF IUGR + SENSORINEURAL HEARING LOSS + MICROCEPHALY PATHOGNOMONIC FOR IGF1 DEFICIENCY. "
            "All three components: intrauterine growth restriction (birth weight and length <3rd centile); "
            "SNHL (usually bilateral, moderate to severe); microcephaly (OFC <2 SDS). "
            "HIGH GH (>10 ng/mL) WITH LOW IGF-1 (similar to Laron syndrome) — "
            "KEY DISTINGUISHING FEATURE FROM LARON: absence of microcephaly and SNHL in GHR Laron. "
            "IGF-1 generation test (rhGH administration): IGF-1 DOES rise in IGF1 gene LOF "
            "if residual GHR signalling generates other growth factors — test not diagnostic; "
            "contrast GHR Laron where generation test completely fails. "
            "Cognitive impairment: variable (mild to severe); MRI may show dysplasia/reduced volume. "
            "Insulin resistance possible (elevated fasting glucose/insulin)."
        ),
        "treatment": (
            "Recombinant IGF-1 (mecasermin, Increlex): same agent as for Laron syndrome. "
            "FDA-approved for severe primary IGF-1 deficiency. "
            "Dose: 0.04-0.12 mg/kg SC twice daily WITH MEALS (hypoglycaemia risk identical to Laron). "
            "Hearing: mecasermin may partially improve cochlear function if cochlear hair cells not destroyed; "
            "cochlear implants indicated for severe-profound SNHL; audiological assessment mandatory before therapy start. "
            "Cognitive effects: limited evidence for improvement; early IGF-1 therapy in infancy theorised to "
            "improve neurological outcomes (animal models robust; human data limited). "
            "Growth: partial improvement in height velocity; less effective than for GHR Laron — "
            "may reflect non-GHR pathways contributing to growth. "
            "rhGH NOT effective: GH signalling to IGF-1 is blocked at IGF1 gene level. "
            "Genetic counselling: AR recurrence risk 25%; consanguinity counselling. "
            "Cochlear implant multidisciplinary team: audiologist, ENT, speech therapy."
        ),
        "key_features": [
            "IUGR + SNHL + microcephaly triad — pathognomonic for IGF1 deficiency (not in Laron GHR)",
            "HIGH GH + LOW IGF-1 — same as Laron; triad distinguishes",
            "Prenatal growth failure (IUGR) — because fetal IGF-1 production is GH-independent",
            "Mecasermin (recombinant IGF-1) treatment — NOT rhGH",
            "Cochlear implants for severe SNHL — audiological assessment mandatory",
            "Very rare — fewer than 20 patients described; consanguinity in most families",
        ],
        "key_ddx": [
            "GHR (Laron): HIGH GH + LOW IGF-1 but NO SNHL, NO microcephaly, NO IUGR — key distinction",
            "STAT5B: HIGH GH + LOW IGF-1 but with immune dysregulation, not SNHL/microcephaly",
            "SGA/IUGR from placental cause: GH normal or low, IGF-1 low; no SNHL",
            "Congenital cytomegalovirus (CMV): IUGR + microcephaly + SNHL mimics — serology/PCR separates",
            "Waardenburg syndrome (PAX3/MITF): SNHL + pigmentation — no growth failure, no IUGR",
            "GH1 deficiency: LOW GH on stimulation; no SNHL; no microcephaly; no IUGR",
        ],
        "treatment_response": "Mecasermin — partial height improvement; SNHL addressed with cochlear implants",
        "hormone_profile": "GH elevated; IGF-1 very low; IGFBP-3 low; insulin resistance possible",
        "onset_age": "Prenatal (IUGR) — neonatal hearing screening may detect SNHL at birth",
        "severity_sds": "−4 to −8 SDS (severe prenatal and postnatal failure)",
        "autosomal_recessive_risk": True,
        "gh_deficiency": False,
        "mecasermin_indicated": True,
    },

    # -- IGFALS -- Acid Labile Subunit deficiency (mild phenotype) ------------------------
    {
        "gene": "IGFALS",
        "alt_name": (
            "IGFALS (IGFALS-605aa-16p13.3 / AR -- "
            "ACID-LABILE-SUBUNIT-ALS-DEFICIENCY -- "
            "VERY-LOW-IGFBP-3-DISPROPORTIONATE-TO-IGF-1-PATHOGNOMONIC -- "
            "MILD-SHORT-STATURE-PUBERTAL-SPURT-PRESERVED -- "
            "OFTEN-ASYMPTOMATIC-CHILDHOOD)"
        ),
        "protein": (
            "IGFALS -- 16p13.3 AR -- IGFALS-605aa -- "
            "Insulin-Like-Growth-Factor-Binding-Protein-Acid-Labile-Subunit-ALS-67kDa -- "
            "18-Leucine-Rich-Repeats-LRR-Scaffold-Ligand-Binding-Glycoprotein -- "
            "Forms-Ternary-Complex-IGF-1-or-IGF-2-Plus-IGFBP-3-or-IGFBP-5-Plus-ALS -- "
            "Ternary-Complex-Prolongs-IGF-1-Half-Life-From-10-20min-to-12-15hr -- "
            "ALS-Keeps-IGF-1-in-Circulation-Without-Ternary-Complex-IGF-1-Cleared-Rapidly -- "
            "ALS-Deficiency-Circulating-IGF-1-Pool-Reduced-but-Tissue-Autocrine-IGF-1-Preserved -- "
            "OMIM-Gene-601489-Disease-ALS-Deficiency-614674"
        ),
        "locus": "16p13.3",
        "protein_size": "605 aa / ~67 kDa",
        "inheritance": (
            "AR (autosomal recessive, biallelic); "
            "consanguineous families predominantly reported; "
            "heterozygous carriers: mildly low IGF-1 and IGFBP-3 but normal stature; "
            "penetrance of growth failure high but severity mild — "
            "most patients have height −1 to −3 SDS (mild-moderate, not severe); "
            "pubertal growth spurt largely preserved (distinguishes from GHR Laron, IGF1 gene defects); "
            "some patients diagnosed incidentally on routine growth assessment; "
            "global prevalence unknown — may be underdiagnosed due to mild phenotype"
        ),
        "disease_category": "Acid-Labile Subunit (ALS) deficiency — circulating IGF-1 pool reduced; mild short stature with preserved pubertal growth",
        "disease_pathway": (
            "ALS (acid-labile subunit, encoded by IGFALS) is a liver-derived glycoprotein that forms a ternary complex "
            "with IGF-1 (or IGF-2) and IGFBP-3 (or IGFBP-5). "
            "The ternary complex (150 kDa) extends IGF-1 serum half-life from ~10-20 minutes (free) to 12-15 hours "
            "(ternary complex), creating a stable circulating pool of bioactive IGF-1. "
            "In ALS LOF: no ternary complex → circulating IGF-1 and IGFBP-3 both dramatically reduced; "
            "BUT: tissue-level (autocrine/paracrine) IGF-1 production is INTACT (ALS produced only in liver; "
            "local tissue IGF-1 does not require ALS for short-range paracrine action). "
            "This explains the MILD phenotype: paracrine IGF-1 sustains most local growth plate activity; "
            "only endocrine IGF-1 pool is reduced. "
            "Bioavailable IGF-1 (free IGF-1) may be relatively more preserved than total IGF-1 assay indicates. "
            "GH elevated (loss of endocrine IGF-1 feedback to pituitary): "
            "GH normal or mildly elevated (less elevated than in GHR/IGF1 deficiency). "
            "No brain, cochlear, or prenatal growth effects — all three are GH-independent or paracrine → "
            "explains absence of SNHL, microcephaly, or IUGR (distinguishing from IGF1 gene LOF)."
        ),
        "pathognomonic": (
            "DISPROPORTIONATELY VERY LOW IGFBP-3 RELATIVE TO IGF-1 LEVEL IS PATHOGNOMONIC FOR ALS DEFICIENCY. "
            "Typical pattern: IGFBP-3 <0.5 mg/L (markedly below normal) while IGF-1 is low but "
            "less dramatically reduced (e.g., IGF-1 50-100 ng/mL when normal is 100-300 ng/mL). "
            "This disproportionate IGFBP-3 suppression occurs because IGFBP-3 requires ALS for ternary complex "
            "stability — without ALS, IGFBP-3 is rapidly degraded even if produced normally. "
            "MILD-MODERATE SHORT STATURE: height −1 to −3 SDS with PRESERVED PUBERTAL GROWTH SPURT "
            "(distinguishes from GHR Laron and IGF1 gene LOF where pubertal spurt blunted). "
            "No SNHL, no microcephaly, no IUGR — absence of these features argues strongly against IGF1 gene defect. "
            "GH mildly elevated or high-normal. "
            "IGF-1 generation test: IGF-1 may partially rise (some endocrine reserve) — "
            "less predictive than in GHR diagnosis."
        ),
        "treatment": (
            "No established specific treatment approved for ALS deficiency. "
            "Observational approach: most patients reach adult height −1 to −2 SDS (within acceptable range); "
            "rhGH may improve growth velocity (raises GH-dependent IGF-1, but without ALS, IGF-1 clears faster); "
            "small case series suggest modest benefit from rhGH — not routinely recommended without clear indication. "
            "Mecasermin (recombinant IGF-1) investigational — administered exogenously bypasses ALS requirement "
            "for circulating pool, but clearing rapid without ALS ternary stabilisation. "
            "Monitor: IGF-1, IGFBP-3, GH every 6-12 months; bone age; growth velocity. "
            "Counselling: adult height typically within normal range, some cases at lower normal; "
            "quality of life and metabolic consequences: mild insulin-like metabolic effects possible. "
            "Genetic testing: cascade testing of first-degree relatives; AR recurrence risk 25%. "
            "Puberty: normal timing, normal growth spurt — reassurance important for families."
        ),
        "key_features": [
            "Very low IGFBP-3 disproportionate to IGF-1 level — hallmark biochemistry",
            "MILD short stature (−1 to −3 SDS) — much less severe than GHR or IGF1 gene defects",
            "Pubertal growth spurt PRESERVED — distinguishes from GHR Laron and IGF1 deficiency",
            "No SNHL, no microcephaly, no IUGR — ALS is liver-only; local/paracrine IGF-1 intact",
            "Often asymptomatic in childhood — frequently diagnosed incidentally",
            "Tissue/paracrine IGF-1 intact — explains mild phenotype despite very low serum IGF-1",
        ],
        "key_ddx": [
            "GHR (Laron): SEVERE dwarfism; HIGH GH; VERY LOW IGF-1; pubertal spurt absent — not mild like ALS",
            "IGF1 gene defect: IUGR + SNHL + microcephaly triad; much more severe — absent in ALS deficiency",
            "GH deficiency (GH1): LOW GH on stimulation; IGFBP-3 low but proportionate to IGF-1",
            "Malnutrition: low IGF-1 + low IGFBP-3 but acquired; nutritional history, IGFBP-3 less dramatic",
            "Hypothyroidism: low IGF-1 and IGFBP-3; TSH elevated; responds to levothyroxine",
            "STAT5B: HIGH GH + LOW IGF-1 + immune dysregulation — not mild like ALS",
        ],
        "treatment_response": "No established treatment; most reach acceptable adult height; rhGH modest benefit",
        "hormone_profile": "GH normal to mildly elevated; IGF-1 low; IGFBP-3 very low (disproportionate)",
        "onset_age": "Childhood growth deceleration; often diagnosed 8-15 years; no neonatal features",
        "severity_sds": "−1 to −3 SDS (mild-moderate; not severe)",
        "autosomal_recessive_risk": True,
        "gh_deficiency": False,
        "mecasermin_indicated": False,
    },

    # -- STAT5B -- GH insensitivity + immune dysregulation --------------------------------
    {
        "gene": "STAT5B",
        "alt_name": (
            "STAT5B (STAT5B-786aa-17q21.2 / AR -- "
            "GH-INSENSITIVITY-PLUS-IMMUNE-DYSREGULATION -- "
            "HIGH-GH-LOW-IGF-1-PLUS-SEVERE-VARICELLA-T-CELL-LYMPHOPENIA-PATHOGNOMONIC -- "
            "ECZEMA-LIKE-SKIN-INFLAMMATORY-LUNG-DISEASE -- "
            "MECASERMIN-FOR-GROWTH-IMMUNOSUPPRESSION-FOR-IMMUNE-DEFECT)"
        ),
        "protein": (
            "STAT5B -- 17q21.2 AR -- STAT5B-786aa -- "
            "Signal-Transducer-and-Activator-of-Transcription-5B-90kDa-SH2-Domain-Transactivation-Domain -- "
            "Downstream-of-JAK2-GHR-Signaling-Also-Cytokine-Receptors-IL-2-IL-7-IL-15-IL-21-Prolactin -- "
            "GH-GHR→JAK2→STAT5B-Phospho-Y694→Homodimerize→Nuclear-IGF-1-Gene -- "
            "Immunological-Roles-IL-2Rγ-Common-Gamma-Chain-Cytokine-Signaling -- "
            "Regulatory-T-Cell-Treg-Development-Requires-STAT5B -- "
            "NK-Cell-and-CD8+-T-Cell-Survival-Requires-STAT5B -- "
            "OMIM-Gene-604260-Disease-STAT5B-Deficiency-245590"
        ),
        "locus": "17q21.2",
        "protein_size": "786 aa / ~90 kDa",
        "inheritance": (
            "AR (autosomal recessive, biallelic); rare — <50 patients described worldwide; "
            "consanguineous families predominantly; "
            "STAT5B serves dual functions: GH signalling (JAK2→STAT5B→IGF-1) AND "
            "cytokine receptor signalling (IL-2, IL-7, IL-15, IL-21, prolactin); "
            "this dual role explains why STAT5B LOF causes BOTH GH insensitivity (Laron-like phenotype) "
            "AND combined immunodeficiency; "
            "STAT5A (closely related, same chromosome 17q21.2) partially compensates for STAT5B in some tissues — "
            "explains why immune defect is variable (not X-SCID severity); "
            "GHBP typically NORMAL in STAT5B deficiency (extracellular GHR domain intact — "
            "distinguishes from GHR Laron where GHBP is low)"
        ),
        "disease_category": "STAT5B deficiency — GH insensitivity syndrome with combined immune dysregulation",
        "disease_pathway": (
            "STAT5B mediates signal transduction for GHR (via JAK2) and multiple cytokine receptors "
            "(IL-2Rγ common gamma chain family: IL-2, IL-7, IL-15, IL-21; prolactin receptor). "
            "GH signalling axis: GH → GHR → JAK2 → phospho-STAT5B (Tyr694) → dimerisation → "
            "nuclear translocation → IGF-1, IGFBP-3 gene transcription. "
            "In STAT5B LOF: GHR signalling intact up to JAK2 (GHR normal, GHBP normal) "
            "but STAT5B cannot be phosphorylated → no IGF-1 production despite elevated GH. "
            "Immune consequences: "
            "CD4+ Treg development requires STAT5B → Treg deficiency → autoimmune manifestations; "
            "NK cell survival requires IL-15/STAT5B → NK lymphopenia; "
            "T cell homeostasis requires IL-7/STAT5B → T lymphopenia (especially naive T cells); "
            "clinical result: susceptibility to viral infections (especially varicella-zoster virus), "
            "eczema-like skin disease, inflammatory lung disease. "
            "Key distinction from GHR Laron: "
            "GHBP is NORMAL in STAT5B (GHR intact, shed extracellular domain normal) "
            "vs LOW in GHR Laron (extracellular domain absent/truncated)."
        ),
        "pathognomonic": (
            "HIGH GH + LOW IGF-1 + LOW IGFBP-3 (Laron-like profile) COMBINED WITH "
            "SEVERE VARICELLA-ZOSTER INFECTIONS, ECZEMA-LIKE SKIN DISEASE, AND "
            "T-CELL LYMPHOPENIA IS PATHOGNOMONIC FOR STAT5B DEFICIENCY. "
            "GHBP NORMAL — critical distinguishing feature from GHR Laron (where GHBP is LOW). "
            "Immune features: recurrent/severe varicella (may be disseminated); "
            "autoimmune haemolytic anaemia; inflammatory lung disease (lymphocytic interstitial pneumonitis); "
            "eczema refractory to standard topical treatment; autoimmune thyroiditis possible. "
            "IGF-1 generation test: fails (similar to GHR Laron) — no rise despite rhGH administration; "
            "GHBP measurement and immune phenotyping separates STAT5B from GHR. "
            "Growth: Laron-like dwarfism (−4 to −8 SDS); frontal bossing; delayed dentition. "
            "Prolactin signalling also impaired — some patients have elevated prolactin."
        ),
        "treatment": (
            "Growth component: recombinant IGF-1 (mecasermin) — same as Laron syndrome; "
            "dose 0.04-0.12 mg/kg SC twice daily WITH FOOD; monitor for hypoglycaemia. "
            "Immune component: immunoglobulin replacement (IVIG/SCIG) for humoral support; "
            "prophylactic antivirals (aciclovir/valaciclovir) especially during VZV exposure periods; "
            "live attenuated vaccines CONTRAINDICATED (varicella vaccine live → risk of disseminated VZV); "
            "give non-live influenza, pneumococcal, meningococcal vaccines. "
            "Inflammatory lung disease: inhaled or oral corticosteroids; "
            "hydroxychloroquine for autoimmune manifestations; "
            "JAK1/2 inhibitors (ruxolitinib) investigational for immune dysregulation component "
            "— STAT5B is downstream of JAK2, but partial rescue via STAT5A possible. "
            "Haematopoietic stem cell transplantation (HSCT): curative for immune defect but does not correct "
            "GHR signalling → may still require mecasermin post-HSCT for growth. "
            "VARICELLA EXPOSURE EMERGENCY: VZV-ZIG (varicella-zoster immunoglobulin) within 72 hours; "
            "IV aciclovir if disseminated VZV."
        ),
        "key_features": [
            "HIGH GH + LOW IGF-1 + SEVERE VARICELLA + T-CELL LYMPHOPENIA — unique combined profile",
            "GHBP NORMAL — critical distinction from GHR Laron (where GHBP low)",
            "Immune dysregulation: Treg deficiency, NK lymphopenia, varicella susceptibility",
            "Inflammatory lung disease (lymphocytic interstitial pneumonitis) — important extra-growth feature",
            "Live vaccines CONTRAINDICATED — VZV vaccine is live; dissemination risk",
            "HSCT corrects immune defect but not growth failure — mecasermin still needed after HSCT",
        ],
        "key_ddx": [
            "GHR (Laron): HIGH GH + LOW IGF-1 but NO immune defect; GHBP LOW (not normal as in STAT5B)",
            "IGF1 gene defect: HIGH GH + LOW IGF-1 + SNHL + microcephaly — not immune dysregulation",
            "X-SCID (IL2RG): more severe combined immunodeficiency; short stature not primary feature",
            "DOCK8 deficiency: recurrent herpes infections + eczema; no GH insensitivity",
            "WAS (Wiskott-Aldrich): immune + eczema but not GH insensitivity",
            "IGFALS deficiency: MILD phenotype; no immune defect; GHBP normal as in STAT5B but much milder",
        ],
        "treatment_response": "Mecasermin for growth (partial); HSCT for immune defect",
        "hormone_profile": "GH elevated; IGF-1 very low; IGFBP-3 very low; GHBP NORMAL (key distinguisher)",
        "onset_age": "Infancy (growth failure + recurrent infections); VZV events often in childhood",
        "severity_sds": "−4 to −8 SDS (Laron-severity growth; with immune morbidity)",
        "autosomal_recessive_risk": True,
        "gh_deficiency": False,
        "mecasermin_indicated": True,
    },

    # -- CUL7 -- 3-M syndrome --------------------------------------------------------------
    {
        "gene": "CUL7",
        "alt_name": (
            "CUL7 (CUL7-1698aa-6p21.1 / AR -- "
            "3-M-SYNDROME -- "
            "PROPORTIONATE-SEVERE-DWARFISM-MINUS-8-TO-MINUS-10-SDS -- "
            "TRIANGULAR-FACE-PROMINENT-HEEL-SLENDER-TUBULAR-BONES-PATHOGNOMONIC -- "
            "NORMAL-GH-IGF1-AXIS-NORMAL-INTELLIGENCE)"
        ),
        "protein": (
            "CUL7 -- 6p21.1 AR -- CUL7-1698aa -- "
            "Cullin-7-192kDa-Scaffold-Subunit-CRL7-E3-Ubiquitin-Ligase-Complex -- "
            "CUL7-FBXW8-SKP1-ROC1-Forms-SCF-Like-Ubiquitin-Ligase -- "
            "Substrates-IRS-1-Insulin-Receptor-Substrate-1 -- "
            "Regulates-IGF-1-Signaling-Downstream-Cellular-Level -- "
            "Also-Targets-p57KIP2-Cell-Cycle-Inhibitor -- "
            "3-M-Syndrome-Locus-Heterogeneity-OBSL1-CCDC8-Also-Cause-Same-Syndrome -- "
            "OMIM-Gene-609577-Disease-3M-Syndrome-273750"
        ),
        "locus": "6p21.1",
        "protein_size": "1698 aa / ~192 kDa",
        "inheritance": (
            "AR (autosomal recessive, biallelic); "
            "3-M syndrome is named for three researchers: Miller, McKusick, and Malvaux; "
            "genetic heterogeneity: CUL7 (most common, ~70%), OBSL1 (~25%), CCDC8 (<5%); "
            "worldwide rare; higher incidence in certain populations (Mennonite community: CUL7 founder p.Cys1100X); "
            "consanguinity common in reported families; "
            "de novo variants rare; "
            "normal intelligence and normal GH/IGF-1 axis distinguishes 3-M from pituitary/receptor disorders; "
            "CUL7 mutations: nonsense, frameshift, splice site throughout gene; "
            "OBSL1 (2q35): obscurin-like 1, cytoskeletal; CCDC8 (19q13.32): coiled-coil domain containing 8"
        ),
        "disease_category": "3-M syndrome — proportionate severe short stature; GH-independent cellular growth defect",
        "disease_pathway": (
            "CUL7 is the scaffold subunit of the CRL7 (Cullin-RING Ligase 7) E3 ubiquitin ligase complex. "
            "CRL7 components: CUL7 + FBXW8 (F-box substrate receptor) + SKP1 + ROC1/RBX1. "
            "Key substrate: IRS-1 (Insulin Receptor Substrate 1) — polyubiquitinated by CRL7 → proteasomal degradation. "
            "Paradox: CRL7 degrades IRS-1, a positive mediator of IGF-1R/insulin signalling. "
            "In CUL7 LOF: IRS-1 not degraded → IRS-1 accumulates → but chronic IRS-1 accumulation "
            "disrupts normal feedback regulation → ultimately impairs downstream PI3K/AKT/mTOR signalling. "
            "Alternatively: CUL7 is required for normal cell size checkpoint at G1/S transition; "
            "CUL7 LOF → smaller cell size → reduced growth plate chondrocyte volume → smaller bones. "
            "Growth plates: structurally normal but hypocellular; cartilage matrix normal; "
            "endochondral ossification proceeds normally but at reduced rate. "
            "GH/IGF-1 axis: completely NORMAL — GH stimulation tests, IGF-1, IGFBP-3 all in normal range. "
            "This is critical: 3-M syndrome is a CELLULAR growth defect, not an endocrine defect. "
            "OBSL1 and CCDC8 act in same pathway — OBSL1 is CUL7-interacting partner; "
            "CCDC8 interacts with OBSL1 to recruit CUL7 complex."
        ),
        "pathognomonic": (
            "PROPORTIONATE SEVERE DWARFISM (HEIGHT −8 TO −10 SDS OR MORE) WITH NORMAL GH/IGF-1 AXIS "
            "AND NORMAL INTELLIGENCE IS PATHOGNOMONIC FOR 3-M SYNDROME. "
            "Distinctive facial features PATHOGNOMONIC: "
            "triangular face (widely spaced eyes, small pointed chin); frontal bossing; "
            "anteverted nares; prominent mouth/lips. "
            "PROMINENT HEELS (protruding calcaneum): pathognomonic orthopaedic sign, present from infancy. "
            "Slender/thin tubular bones on X-ray: reduced cortical thickness; spine may show platyspondyly. "
            "CLINODACTYLY of 5th fingers; soft tissue folds at wrists and ankles. "
            "Delayed eruption of teeth. "
            "Birth: already severely growth-restricted (birth length −4 to −6 SDS; birth weight less affected). "
            "SEVERE POSTNATAL GROWTH FAILURE: height further drops to −8 to −10 SDS. "
            "Puberty timing: variable (often delayed). "
            "REPRODUCTIVE: female 3-M patients may develop cystic ovaries; "
            "males may have hypogonadism."
        ),
        "treatment": (
            "No curative treatment available. "
            "Recombinant GH (rhGH) has been tried — modest growth velocity improvement reported "
            "(1-2 cm/year additional) but does not correct the cellular defect; "
            "because GH/IGF-1 axis is normal, the standard rationale for rhGH does not apply; "
            "use only in context of documented benefit monitoring. "
            "Limb lengthening (Ilizarov distraction osteogenesis): used in select centres; "
            "technically feasible; significant complication rate (pin site infection, nerve injury); "
            "psychological preparation and patient/family consent critical. "
            "Orthopaedic management: scoliosis surveillance (annual spine films); "
            "physiotherapy for mobility; orthotics for foot/heel anomalies. "
            "Multidisciplinary approach: endocrinologist (confirm normal GH axis), geneticist, "
            "orthopaedic surgeon, physiotherapist, psychologist. "
            "Normal intelligence: standard schooling; cognitive support not required. "
            "Genetic counselling: AR recurrence risk 25%; CUL7, OBSL1, CCDC8 panel testing; "
            "prenatal diagnosis available for known familial variants."
        ),
        "key_features": [
            "Severe proportionate dwarfism (−8 to −10 SDS or more) — more severe than GHD/Laron",
            "NORMAL GH and IGF-1 axis — this is a CELLULAR growth defect, not endocrine",
            "NORMAL intelligence — distinguishes from many severe dwarfism syndromes",
            "Triangular face + prominent heels + slender tubular bones — pathognomonic clinical triad",
            "Birth length already severely reduced (−4 to −6 SDS) — prenatal growth failure",
            "Genetic heterogeneity: CUL7 (~70%), OBSL1 (~25%), CCDC8 (<5%) — panel testing needed",
        ],
        "key_ddx": [
            "Laron syndrome (GHR): HIGH GH + LOW IGF-1 — completely different biochemistry; 3-M is NORMAL",
            "Silver-Russell syndrome (SRS): IUGR + relative macrocephaly + 5th finger clinodactyly; IGF-2 dysregulation",
            "Seckel syndrome: severe IUGR + microcephaly + prominent nose + ATR/other genes",
            "Primordial dwarfism syndromes (PCNT, CPAP/CEP152): overlapping features; gene panels separate",
            "Hypochondroplasia: rhizomelic disproportion; FGFR3 GOF; GH/IGF-1 normal in both — imaging separates",
            "Noonan syndrome: PTPN11/RAF1/SOS1 — cardiac + lymphedema + distinctive face — different phenotype",
        ],
        "treatment_response": "No curative treatment; rhGH modest (1-2 cm/year); limb lengthening possible",
        "hormone_profile": "GH NORMAL; IGF-1 NORMAL; IGFBP-3 NORMAL — all endocrine axes intact",
        "onset_age": "Prenatal (severe IUGR, −4 to −6 SDS birth length); neonatal recognition",
        "severity_sds": "−8 to −10 SDS or more (most severe in this atlas after biallelic SHOX/IGF1)",
        "autosomal_recessive_risk": True,
        "gh_deficiency": False,
        "mecasermin_indicated": False,
    },

    # -- NPR2 -- Brachydactyly-E2 / Acromesomelic Dysplasia Maroteaux --------------------
    {
        "gene": "NPR2",
        "alt_name": (
            "NPR2 (NPR2-1047aa-9p13.3 / AD/AR -- "
            "AD-HAPLOINSUFFICIENCY-SHORT-STATURE-BRACHYDACTYLY-TYPE-E2-SHORT-METACARPALS-PATHOGNOMONIC -- "
            "AR-BIALLELIC-NULL-ACROMESOMELIC-DYSPLASIA-MAROTEAUX-ADM-SEVERE-MESOMELIC-SHORTENING -- "
            "CNP-C-TYPE-NATRIURETIC-PEPTIDE-RECEPTOR-VOSORITIDE-NOT-EFFECTIVE-IN-NPR2-LOF)"
        ),
        "protein": (
            "NPR2 -- 9p13.3 AD/AR -- NPR2-1047aa -- "
            "Natriuretic-Peptide-Receptor-2-GC-B-NPR-B-117kDa-Guanylate-Cyclase-Coupled-Receptor -- "
            "Extracellular-CNP-Binding-Domain-Single-TM-Helix-Kinase-Homology-Domain-Guanylate-Cyclase-Domain -- "
            "CNP-Ligand-C-Type-Natriuretic-Peptide-22aa-Peptide-Predominant-Bone-CNP -- "
            "CNP-NPR2-→-cGMP-→-PKG-II-Phosphodiesterase-Inhibits-FGFR3-Signaling-in-Growth-Plate -- "
            "Key-Counter-Regulator-of-FGFR3-Achondroplasia-Pathway -- "
            "FGFR3-GOF-Inhibits-CNP-Effect-→-Vosoritide-Bypasses-by-Direct-CNP-Mimicry -- "
            "NPR2-LOF-→-Cannot-Respond-to-CNP-or-Vosoritide-Critical-DDx -- "
            "OMIM-Gene-108961-Disease-ACMSD-201250-BDE2-613382"
        ),
        "locus": "9p13.3",
        "protein_size": "1047 aa / ~117 kDa",
        "inheritance": (
            "Bimodal: "
            "AD (autosomal dominant haploinsufficiency) → short stature + brachydactyly type E2 (BDE2): "
            "short 4th metacarpals bilaterally; milder growth failure (−2 to −3 SDS); "
            "AR (biallelic null/LOF) → acromesomelic dysplasia Maroteaux (ADM): "
            "severe limb shortening with predominant acromesomelic pattern (hands, feet, forearms, legs); "
            "first described by Maroteaux 1971; "
            "FGFR3-achondroplasia comparison: "
            "achondroplasia = FGFR3 GOF → FGFR3 signal too active → inhibits CNP/NPR2 pathway; "
            "NPR2 LOF = NPR2 absent → cannot respond to CNP signal → "
            "same downstream effect (cGMP pathway inactive, FGFR3 growth inhibition disinhibited); "
            "VOSORITIDE TRAP: vosoritide (TransCon CNP) FDA 2021 for achondroplasia — "
            "bypasses hyperactive FGFR3 by directly activating NPR2; "
            "in NPR2 LOF, the RECEPTOR IS ABSENT/NON-FUNCTIONAL → vosoritide has no target → ineffective"
        ),
        "disease_category": "NPR2 deficiency — brachydactyly type E2 (AD) / acromesomelic dysplasia Maroteaux (AR); CNP signal pathway defect",
        "disease_pathway": (
            "The CNP (C-type natriuretic peptide) / NPR2 / cGMP pathway is a critical regulator of endochondral bone growth. "
            "CNP is produced by growth plate chondrocytes and the liver; it binds NPR2 on growth plate chondrocytes → "
            "activates intrinsic guanylyl cyclase domain → cGMP → activates PKG-II → "
            "phosphorylates CDC25A phosphatase and inhibits FGFR3-RAS-MAPK pathway → "
            "promotes chondrocyte proliferation and hypertrophy → longitudinal bone growth. "
            "Key interaction with FGFR3: FGFR3 (achondroplasia gene) activates MAPK pathway → "
            "INHIBITS growth plate chondrocyte hypertrophy. "
            "CNP/NPR2 counter-regulates FGFR3 inhibition → net result: balanced growth. "
            "FGFR3 GOF (achondroplasia): excess FGFR3 signalling overwhelms CNP/NPR2 → CNP analogue "
            "(vosoritide/TransCon CNP) restores balance by flooding NPR2 signal. "
            "NPR2 LOF (this gene): receptor absent → CNP signal cannot be transduced regardless of dose → "
            "FGFR3 pathway disinhibited → reduced longitudinal bone growth. "
            "AD NPR2 haploinsufficiency: 50% receptor → partial CNP signal → milder phenotype (BDE2 + mild short stature). "
            "AR NPR2 biallelic null: no receptor → no CNP signal → severe mesomelic/acromesomelic dwarfism (ADM). "
            "Brachydactyly E2: NPR2-mediated CNP signalling in metacarpal/metatarsal growth plates → "
            "without full NPR2 activity, metacarpal growth plates (especially 4th) fail to elongate normally."
        ),
        "pathognomonic": (
            "AD NPR2 (BDE2): SHORT 4TH METACARPALS (brachydactyly) BILATERALLY ON HAND X-RAY PATHOGNOMONIC "
            "combined with mild short stature (−2 to −3 SDS). "
            "Short 4th metacarpals visible clinically as dimple over 4th metacarpophalangeal joint "
            "('knuckle sign' when fist made — 4th knuckle sits below 3rd and 5th). "
            "Also 4th + 5th metatarsal shortening. "
            "Hand X-ray: 4th metacarpal length <3rd and 5th; 4th metatarsal affected; "
            "brachydactyly type E pattern (distal aspect of metacarpals). "
            "AR NPR2 (ADM): SEVERE ACROMESOMELIC SHORTENING — "
            "severely shortened radius/ulna + tibia/fibula + hands/feet; "
            "severe short stature (−6 to −10 SDS); normal trunk length; "
            "radiographic narrowing of spinal canal may cause neurological compression. "
            "VOSORITIDE PITFALL: vosoritide approved for FGFR3 achondroplasia, NOT for NPR2 LOF; "
            "clinician must confirm FGFR3 vs NPR2 before vosoritide prescription; "
            "in NPR2 LOF, prescribing vosoritide would be ineffective and exposes patient to unnecessary cost/risk."
        ),
        "treatment": (
            "No FDA-approved targeted therapy for NPR2 deficiency (as of 2026). "
            "AD NPR2 (BDE2): mild short stature may be managed with: "
            "rhGH if IGF-1 low (GH axis normal in most, so benefit modest); "
            "limb lengthening for short metacarpals/metatarsals if cosmetically/functionally significant. "
            "AR NPR2 (ADM): no specific medical therapy; "
            "limb lengthening considered in specialised centres; "
            "spinal surveillance: MRI/CT for canal stenosis if neurological symptoms. "
            "Experimental: exogenous CNP analogues — theoretically cannot work if NPR2 non-functional; "
            "cGMP pathway agonists downstream of NPR2 (e.g., PDE5 inhibitors) under laboratory investigation. "
            "VOSORITIDE CONTRAINDICATION: do NOT prescribe vosoritide for NPR2 LOF — "
            "vosoritide (TransCon CNP) requires functional NPR2 receptor; FGFR3 gene testing mandatory "
            "before prescribing vosoritide to any short stature + limb shortening patient. "
            "Genetic counselling: AD families (BDE2): 50% recurrence; "
            "AR families (ADM): 25% recurrence; prenatal diagnosis available."
        ),
        "key_features": [
            "Short 4th metacarpals bilaterally (brachydactyly E2) — pathognomonic AD NPR2 hand X-ray",
            "Acromesomelic dysplasia Maroteaux (ADM) — AR biallelic null — severe mesomelic shortening",
            "CNP/NPR2/cGMP pathway — counterregulator of FGFR3 achondroplasia pathway",
            "VOSORITIDE NOT EFFECTIVE for NPR2 LOF — receptor itself is absent; only works if NPR2 intact (achondroplasia)",
            "Confirm FGFR3 vs NPR2 genotype before any CNP-analogue therapy",
            "Spinal canal stenosis in ADM — neurological surveillance needed",
        ],
        "key_ddx": [
            "Achondroplasia (FGFR3 GOF): rhizomelic (proximal limbs); NPR2 LOF is acromesomelic/BDE2 — imaging + genetics",
            "Hypochondroplasia (FGFR3 mild GOF): mild rhizomelia; hand X-ray may mimic BDE2 — FGFR3 genetic testing",
            "Turner syndrome (SHOX monosomy): also has short 4th metacarpals; 45,X chromosomes + SHOX deficiency",
            "Pseudohypoparathyroidism (GNAS): short 4th metacarpals (Albright hereditary osteodystrophy) + PTH resistance",
            "Brachydactyly type E2 from other causes (PDE3A GOF): short metacarpals + hypertension — different gene",
            "SHOX deficiency: mesomelic but Madelung deformity; no 4th metacarpal-specific shortening",
        ],
        "treatment_response": "No targeted treatment; rhGH modest; limb lengthening for severe ADM",
        "hormone_profile": "GH NORMAL; IGF-1 NORMAL; IGFBP-3 NORMAL — all endocrine axes intact",
        "onset_age": "Birth (ADM — severe); childhood recognition (BDE2 — hand X-ray finding)",
        "severity_sds": "−2 to −3 SDS (AD BDE2); −6 to −10 SDS (AR ADM)",
        "autosomal_recessive_risk": True,
        "gh_deficiency": False,
        "mecasermin_indicated": False,
    },
]


def _make_cohort(entry: dict, seed: int, n: int = 40) -> list:
    rng = random.Random(seed)
    gene = entry["gene"]
    severity_sds = entry["severity_sds"]
    gh_deficiency = entry["gh_deficiency"]
    mecasermin_indicated = entry["mecasermin_indicated"]

    # gene-specific rate parameters
    rates = {
        "SHOX":   {"rgh": 0.72, "meca": 0.0,  "surgery": 0.28, "snhl": 0.0,  "micro": 0.0,  "immune": 0.0,  "normal_iq": 1.0},
        "GH1":    {"rgh": 0.92, "meca": 0.0,  "surgery": 0.05, "snhl": 0.0,  "micro": 0.0,  "immune": 0.0,  "normal_iq": 1.0},
        "GHR":    {"rgh": 0.0,  "meca": 0.78, "surgery": 0.10, "snhl": 0.0,  "micro": 0.0,  "immune": 0.05, "normal_iq": 1.0},
        "IGF1":   {"rgh": 0.0,  "meca": 0.65, "surgery": 0.08, "snhl": 0.88, "micro": 0.82, "immune": 0.0,  "normal_iq": 0.55},
        "IGFALS": {"rgh": 0.22, "meca": 0.0,  "surgery": 0.0,  "snhl": 0.0,  "micro": 0.0,  "immune": 0.0,  "normal_iq": 1.0},
        "STAT5B": {"rgh": 0.0,  "meca": 0.74, "surgery": 0.0,  "snhl": 0.05, "micro": 0.0,  "immune": 0.95, "normal_iq": 0.9},
        "CUL7":   {"rgh": 0.30, "meca": 0.0,  "surgery": 0.42, "snhl": 0.0,  "micro": 0.0,  "immune": 0.0,  "normal_iq": 1.0},
        "NPR2":   {"rgh": 0.18, "meca": 0.0,  "surgery": 0.45, "snhl": 0.0,  "micro": 0.0,  "immune": 0.0,  "normal_iq": 1.0},
    }
    r = rates[gene]

    age_ranges = {
        "SHOX":   (6, 22),
        "GH1":    (1, 18),
        "GHR":    (1, 15),
        "IGF1":   (0, 12),
        "IGFALS": (8, 20),
        "STAT5B": (1, 14),
        "CUL7":   (0, 10),
        "NPR2":   (3, 25),
    }
    age_min, age_max = age_ranges[gene]

    cohort = []
    sexes = ["M", "F"]
    for i in range(n):
        sex = rng.choice(sexes)
        age_dx = round(rng.uniform(age_min, age_max), 1)
        height_sds = round(rng.uniform(-1.5, 0.5) + float(severity_sds.split(" to ")[0].split("(")[0].strip().split("SDS")[0].strip().split("−")[1].split()[0]) * -1, 1)
        height_sds = max(-12.0, min(-1.0, height_sds))

        on_rgh = rng.random() < r["rgh"]
        on_meca = rng.random() < r["meca"]
        had_surgery = rng.random() < r["surgery"]
        has_snhl = rng.random() < r["snhl"]
        has_micro = rng.random() < r["micro"]
        has_immune = rng.random() < r["immune"]
        normal_iq = rng.random() < r["normal_iq"]

        anti_gh_ab = False
        if gene == "GH1":
            # type IA rate ~20% of GH1 patients; IA develop anti-GH antibodies
            is_type_ia = rng.random() < 0.20
            anti_gh_ab = is_type_ia and on_rgh

        type_variant = {
            "SHOX":   rng.choice(["Léri-Weill LWD (AD)", "Léri-Weill LWD (AD)", "Léri-Weill LWD (AD)", "Langer LMD (AR)"]),
            "GH1":    rng.choice(["Type IB (AR)", "Type IB (AR)", "Type II (AD)", "Type IA (AR null)"]),
            "GHR":    rng.choice(["LOF splice", "LOF missense", "LOF deletion"]),
            "IGF1":   rng.choice(["biallelic null", "splice site"]),
            "IGFALS": rng.choice(["biallelic missense", "biallelic frameshift"]),
            "STAT5B": rng.choice(["biallelic null", "biallelic LOF missense"]),
            "CUL7":   rng.choice(["CUL7 biallelic", "OBSL1 biallelic", "CUL7 biallelic"]),
            "NPR2":   rng.choice(["AD haploinsufficiency (BDE2)", "AD haploinsufficiency (BDE2)", "AR biallelic null (ADM)"]),
        }[gene]

        patient = {
            "patient_id": f"{gene}-{i+1:03d}",
            "gene": gene,
            "sex": sex,
            "age_at_diagnosis_years": age_dx,
            "type_variant": type_variant,
            "height_sds": height_sds,
            "on_rgh_therapy": on_rgh,
            "on_mecasermin": on_meca,
            "had_orthopedic_surgery": had_surgery,
            "sensorineural_hearing_loss": has_snhl,
            "microcephaly": has_micro,
            "immune_dysregulation": has_immune,
            "anti_gh_antibodies": anti_gh_ab,
            "normal_iq": normal_iq,
        }
        cohort.append(patient)
    return cohort


def generate_overview():
    all_patients = []
    for idx, entry in enumerate(SHORT_STATURE_GENES):
        seed = SEED_BASE + idx
        all_patients.extend(_make_cohort(entry, seed))

    total = len(all_patients)
    rgh_count = sum(1 for p in all_patients if p["on_rgh_therapy"])
    meca_count = sum(1 for p in all_patients if p["on_mecasermin"])
    surgery_count = sum(1 for p in all_patients if p["had_orthopedic_surgery"])
    snhl_count = sum(1 for p in all_patients if p["sensorineural_hearing_loss"])
    immune_count = sum(1 for p in all_patients if p["immune_dysregulation"])
    anti_gh_count = sum(1 for p in all_patients if p["anti_gh_antibodies"])

    gene_summary = {}
    for idx, entry in enumerate(SHORT_STATURE_GENES):
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
            "severity_sds": entry["severity_sds"],
            "hormone_profile": entry["hormone_profile"],
            "n_patients": len(cohort),
            "rgh_pct": round(100 * sum(1 for p in cohort if p["on_rgh_therapy"]) / len(cohort), 1),
            "mecasermin_pct": round(100 * sum(1 for p in cohort if p["on_mecasermin"]) / len(cohort), 1),
            "surgery_pct": round(100 * sum(1 for p in cohort if p["had_orthopedic_surgery"]) / len(cohort), 1),
            "snhl_pct": round(100 * sum(1 for p in cohort if p["sensorineural_hearing_loss"]) / len(cohort), 1),
            "immune_pct": round(100 * sum(1 for p in cohort if p["immune_dysregulation"]) / len(cohort), 1),
            "avg_age_dx_years": round(sum(p["age_at_diagnosis_years"] for p in cohort) / len(cohort), 1),
            "avg_height_sds": round(sum(p["height_sds"] for p in cohort) / len(cohort), 1),
        }

    return {
        "atlas": "Hereditary-Short-Stature-Atlas",
        "subtitle": "Complete 8-Gene Hereditary Short Stature Reference -- SHOX/GH1/GHR/IGF1/IGFALS/STAT5B/CUL7/NPR2",
        "genes_covered": [e["gene"] for e in SHORT_STATURE_GENES],
        "total_patients": total,
        "seeds": f"{SEED_BASE}-{SEED_BASE + 7}",
        "aggregate_metrics": {
            "rgh_therapy_pct": round(100 * rgh_count / total, 1),
            "mecasermin_pct": round(100 * meca_count / total, 1),
            "orthopedic_surgery_pct": round(100 * surgery_count / total, 1),
            "snhl_pct": round(100 * snhl_count / total, 1),
            "immune_dysregulation_pct": round(100 * immune_count / total, 1),
            "anti_gh_antibodies_pct": round(100 * anti_gh_count / total, 1),
        },
        "gene_summary": gene_summary,
    }


def generate_breakdown():
    breakdown = []
    for idx, entry in enumerate(SHORT_STATURE_GENES):
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
            "treatment_response": entry["treatment_response"],
            "hormone_profile": entry["hormone_profile"],
            "onset_age": entry["onset_age"],
            "severity_sds": entry["severity_sds"],
            "gh_deficiency": entry["gh_deficiency"],
            "mecasermin_indicated": entry["mecasermin_indicated"],
            "autosomal_recessive_risk": entry["autosomal_recessive_risk"],
            "n_patients": len(cohort),
            "rgh_pct": round(100 * sum(1 for p in cohort if p["on_rgh_therapy"]) / len(cohort), 1),
            "mecasermin_pct": round(100 * sum(1 for p in cohort if p["on_mecasermin"]) / len(cohort), 1),
            "surgery_pct": round(100 * sum(1 for p in cohort if p["had_orthopedic_surgery"]) / len(cohort), 1),
            "snhl_pct": round(100 * sum(1 for p in cohort if p["sensorineural_hearing_loss"]) / len(cohort), 1),
            "immune_pct": round(100 * sum(1 for p in cohort if p["immune_dysregulation"]) / len(cohort), 1),
            "anti_gh_pct": round(100 * sum(1 for p in cohort if p["anti_gh_antibodies"]) / len(cohort), 1),
            "avg_age_dx_years": round(sum(p["age_at_diagnosis_years"] for p in cohort) / len(cohort), 1),
            "avg_height_sds": round(sum(p["height_sds"] for p in cohort) / len(cohort), 1),
            "sample_patients": cohort[:3],
        })
    return {"gene_breakdowns": breakdown}


def generate_definitions():
    return {
        "gene_entries": {
            entry["gene"]: {
                "gene": entry["gene"],
                "full_name": entry["protein"].split(" --")[0].strip(),
                "locus": entry["locus"],
                "protein_size": entry["protein_size"],
                "inheritance": entry["inheritance"].split(";")[0].strip(),
                "disease_name": entry["disease_category"],
                "disease_pathway": entry["disease_pathway"],
                "pathognomonic": entry["pathognomonic"],
                "treatment": entry["treatment"][:400],
                "key_features": entry["key_features"],
                "key_ddx": entry["key_ddx"],
                "treatment_response": entry["treatment_response"],
                "hormone_profile": entry["hormone_profile"],
                "onset_age": entry["onset_age"],
                "severity_sds": entry["severity_sds"],
                "gh_deficiency": entry["gh_deficiency"],
                "mecasermin_indicated": entry["mecasermin_indicated"],
            }
            for entry in SHORT_STATURE_GENES
        },
        "short_stature_glossary": {
            "IGF-1 Generation Test — Protocol and Interpretation": (
                "The IGF-1 generation test distinguishes GH deficiency (responds to exogenous GH) from "
                "GH insensitivity (does not respond). Protocol: administer rhGH 0.1 U/kg/day (or 33 µg/kg/day) SC daily for 4 days; "
                "measure baseline IGF-1 before first dose and 12-24 hours after last dose. "
                "Normal response (excludes GH insensitivity): IGF-1 rise >15 mcg/L above baseline, or to >15th centile. "
                "Non-response (confirms GH insensitivity): IGF-1 fails to rise — indicates GHR or post-receptor defect. "
                "Limitations: not standardised across centres; cut-off values vary; nutritional status affects results. "
                "Where IGF-1 rises partially: consider STAT5A partial compensation, IGFALS deficiency, malnutrition. "
                "GH1 deficiency: responds normally to exogenous rhGH (receptor intact). "
                "GHR Laron: fails completely. STAT5B: fails completely. IGF1 gene LOF: partial/variable. IGFALS: variable."
            ),
            "GH Stimulation Tests — Methodology and Interpretation": (
                "GH stimulation tests are used to diagnose GH deficiency. At least TWO tests with different stimuli required "
                "for diagnosis (single test has high false-positive rate ~30%). "
                "Peak GH <10 mcg/L (or <7 mcg/L by some guidelines) on both tests = GH deficiency. "
                "Common agents: insulin tolerance test (ITT) — gold standard; glucagon; arginine; clonidine; GHRH+arginine. "
                "Sex-steroid priming: prepubertal children with constitutional delay may have false-low results — "
                "prime with oestrogen (2 mg/day × 2 days) or testosterone (50 mg IM × 1) before testing. "
                "Caveats: cortisol must be >450 nmol/L on ITT (safety check); "
                "BMI affects GH response (obese patients have blunted peak). "
                "Random GH: useless for diagnosis — GH is pulsatile; a random low GH is normal. "
                "IGF-1 + IGFBP-3: screen before stimulation tests — normal age-matched levels nearly exclude GHD."
            ),
            "Mecasermin (Increlex) — Administration and Hypoglycaemia Safety": (
                "Mecasermin (Increlex) is recombinant IGF-1 (mecasermin rinfabate no longer marketed). "
                "Indication: severe primary IGF-1 deficiency confirmed by GH insensitivity (GHR LOF, STAT5B LOF, IGF1 gene LOF). "
                "Administration: SC injection twice daily, ALWAYS with a meal or snack (hypoglycaemia prevention). "
                "Starting dose: 40 µg/kg/dose twice daily; titrate up by 40 µg/kg increments to max 120 µg/kg/dose. "
                "HYPOGLYCAEMIA is the most common serious adverse event: "
                "monitor glucose 30-60 min after injection in first weeks; "
                "never inject if patient cannot eat; carry glucose gel for rescue. "
                "Other adverse effects: injection site lipohypertrophy; tonsil hypertrophy (lymphoid tissue); "
                "papilloedema (raised intracranial pressure — rare). "
                "IGF-1 monitoring: target mid-normal range for age/sex; avoid high-normal/supraphysiological levels "
                "(soft tissue growth, acromegalic side effects at very high levels). "
                "CONTRAINDICATION: active/suspected malignancy; "
                "not for GH1 deficiency (use rhGH instead); not for NPR2 LOF."
            ),
            "Laron Syndrome vs STAT5B Deficiency — GHBP as the Key Distinguisher": (
                "Both Laron syndrome (GHR LOF) and STAT5B deficiency present with: "
                "HIGH GH + LOW IGF-1 + LOW IGFBP-3 + severe short stature + negative IGF-1 generation test. "
                "CRITICAL DISTINGUISHING FEATURE: GH Binding Protein (GHBP). "
                "GHBP is the shed extracellular domain of GHR (protease cleavage releases it into circulation). "
                "In GHR LOF (Laron): extracellular domain absent or truncated → LOW/ABSENT GHBP. "
                "In STAT5B LOF: GHR intact (extracellular domain normal, shed normally) → NORMAL GHBP. "
                "IMMUNE PHENOTYPE distinguishes further: "
                "Laron syndrome: no immune defect (GHR LOF only; cytokine receptors intact); "
                "STAT5B deficiency: immune dysregulation (varicella susceptibility, T-cell lymphopenia, eczema, lung disease) "
                "because STAT5B also mediates IL-2, IL-7, IL-15, IL-21 receptor signalling. "
                "Practical approach: HIGH GH + LOW IGF-1 → measure GHBP → "
                "LOW GHBP → GHR Laron; NORMAL GHBP → STAT5B or post-GHBP defect → immunophenotype."
            ),
            "PAR1 Biology — Why SHOX Behaves Like an Autosomal Gene Despite X Location": (
                "PAR1 (Pseudoautosomal Region 1) is a 2.7 Mb region at the tip of the short arm of both "
                "the X chromosome (Xp22.33) and the Y chromosome (Yp11.3). "
                "PAR1 undergoes obligatory crossover during male meiosis — chromosomes behave as autosomes in this region. "
                "Genes in PAR1 ESCAPE X-INACTIVATION — both X and Y copies remain active in females and males. "
                "DIPLOID DOSAGE of PAR1 genes (including SHOX) is required for normal function. "
                "Females (46,XX): two X chromosomes → two copies of SHOX (both active). "
                "Males (46,XY): one X copy + one Y copy → two copies of SHOX (both active). "
                "Turner syndrome (45,X): ONE COPY → haploinsufficiency → short stature (same mechanism as LWD). "
                "Klinefelter syndrome (47,XXY): THREE COPIES → overdosage → tallness (excess SHOX effect). "
                "DIAGNOSTIC IMPLICATION: SHOX variants and PAR1 deletions cause same phenotype in males and females; "
                "PAR1 enhancer deletions (35 kb downstream of SHOX) are outside exome capture regions → "
                "require MLPA (Multiplex Ligation-Dependent Probe Amplification) or array CGH for detection."
            ),
            "CNP/NPR2/cGMP Pathway vs FGFR3/RAS/MAPK Pathway — Growth Plate Counter-Regulation": (
                "Two opposing pathways regulate growth plate chondrocyte proliferation and hypertrophy: "
                "PRO-GROWTH: CNP (C-type natriuretic peptide) → NPR2 (receptor) → cGMP → PKG-II → "
                "inhibits FGFR3-RAS-MAPK pathway → promotes chondrocyte hypertrophy → longitudinal growth. "
                "ANTI-GROWTH: FGF18 → FGFR3 → RAS → MAPK → STAT1 → inhibits chondrocyte proliferation. "
                "ACHONDROPLASIA (FGFR3 GOF): excess FGFR3 activation overwhelms CNP/NPR2 pathway → "
                "RAS-MAPK hyperactive → severe rhizomelic dwarfism. "
                "VOSORITIDE (TransCon CNP): CNP analogue → activates NPR2 → raises cGMP → "
                "counteracts FGFR3 hyperactivity → FDA approved 2021 for achondroplasia. "
                "REQUIRES INTACT NPR2. "
                "NPR2 LOF (this atlas gene): receptor itself absent → cGMP signal cannot be produced → "
                "similar downstream effect to achondroplasia but at NPR2 level; "
                "FGFR3 is normal → vosoritide cannot work (no receptor). "
                "KEY CLINICAL RULE: measure FGFR3 AND NPR2 in mesomelic/acromesomelic dwarfism BEFORE "
                "prescribing vosoritide — NPR2 LOF patients must NOT receive vosoritide."
            ),
            "3-M Syndrome — Clinical Triad and Genetic Heterogeneity": (
                "3-M syndrome (OMIM 273750) is named after three clinicians who independently described it: "
                "Miller, McKusick, and Malvaux. "
                "CLINICAL TRIAD (pathognomonic): "
                "(1) Proportionate severe short stature (−8 to −10 SDS or more); "
                "(2) Triangular facies — prominent forehead, pointed chin, widely spaced eyes; "
                "(3) Skeletal features — prominent heels, slender tubular bones, clinodactyly, soft tissue folds at wrists/ankles. "
                "NORMAL GH/IGF-1 axis and NORMAL intelligence distinguish 3-M from endocrine causes of severe dwarfism. "
                "Prenatal growth failure: birth length severely reduced (−4 to −6 SDS) while birth weight less affected. "
                "GENETIC HETEROGENEITY: "
                "CUL7 (6p21.1) ~70%; OBSL1 (2q35, obscurin-like 1) ~25%; CCDC8 (19q13.32) <5%. "
                "All three proteins interact: CUL7 is scaffold; OBSL1 interacts with CUL7; CCDC8 interacts with OBSL1. "
                "Molecular pathway: E3 ubiquitin ligase complex (CUL7-FBXW8-SKP1-ROC1) targets IRS-1 for degradation. "
                "Management: no curative treatment; rhGH modest; limb lengthening considered; physiotherapy; genetics."
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
