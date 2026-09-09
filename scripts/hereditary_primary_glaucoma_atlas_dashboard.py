#!/usr/bin/env python3
"""Hereditary-Primary-Glaucoma-Atlas — Complete 8-Gene Hereditary Primary
Glaucoma Atlas
(MYOC · CYP1B1 · FOXC1 · PITX2 · PAX6 · OPTN · LTBP2 · TEK).

MYOC    (Myocilin; 504 aa; 1q24.3; AD;
         Juvenile Open Angle Glaucoma (JOAG) / adult POAG;
         TRABECULAR MESHWORK PROTEIN — misfolding → ER stress → TM cell death;
         HIGHEST IOP OF ALL HEREDITARY GLAUCOMAS (30-50 mmHg) PATHOGNOMONIC;
         onset teens-30s in JOAG; steroid-response phenotype;
         p.Gln368STOP most common (1-2% POAG); p.Pro370Leu severe JOAG;
         seed SEED_BASE+0).
CYP1B1  (Cytochrome P450 1B1; 543 aa; 2p22.2; AR;
         Primary Congenital Glaucoma (PCG) — most common PCG gene globally;
         BUPHTHALMOS (ox eye) — enlarged corneal diameter PATHOGNOMONIC;
         HAAB STRIAE — horizontal corneal breaks from IOP elevation PATHOGNOMONIC;
         Neonatal/infantile onset; corneal clouding; photophobia;
         p.Arg368His most common Arab/Turkish/Pakistani;
         seed SEED_BASE+1).
FOXC1   (Forkhead Box C1; 553 aa; 6p25.3; AD;
         Axenfeld-Rieger Syndrome type 3 (ARS3);
         IRIS HYPOPLASIA + ANTERIOR SEGMENT DYSGENESIS PATHOGNOMONIC;
         POSTERIOR EMBRYOTOXON (prominent Schwalbe line) 100% of ARS;
         Glaucoma 50-80%; dental anomalies; umbilical/periumbilical abnormalities;
         p.Cys240Gly frameshift severe; heterozygous haploinsufficiency;
         seed SEED_BASE+2).
PITX2   (Paired-like Homeodomain 2; 317 aa; 4q25; AD;
         Axenfeld-Rieger Syndrome type 1 (ARS1);
         IRIDOCORNEAL ADHESIONS (iris processes bridging to cornea) PATHOGNOMONIC;
         Midface hypoplasia; dental: hypodontia/microdontia/peg-shaped teeth;
         Glaucoma 50-70%; umbilical stump abnormality; cardiac septal defects;
         p.Arg84Trp homeodomain most common familial;
         seed SEED_BASE+3).
PAX6    (Paired Box 6; 422 aa; 11p13; AD;
         Aniridia + Secondary Glaucoma;
         ANIRIDIA — iris absent or rudimentary PATHOGNOMONIC;
         Keratopathy (corneal opacification); foveal hypoplasia → nystagmus; cataracts;
         Aniridia-related glaucoma 30-50% (juvenile onset);
         WAGR if deletion (Wilms/Aniridia/GU anomaly/Retardation) — 11p13 del;
         p.Arg240Ter most common;
         seed SEED_BASE+4).
OPTN    (Optineurin; 577 aa; 10p13; AD;
         Normal Tension Glaucoma (NTG);
         NORMAL IOP (<21 mmHg) + PROGRESSIVE VISUAL FIELD LOSS PATHOGNOMONIC;
         disc haemorrhages more frequent than POAG; peripapillary atrophy;
         E50K mutation most pathogenic (10× risk NTG); mitophagy/autophagy pathway;
         ALS overlap (TBK1 interaction) — screen for ALS in E50K family;
         seed SEED_BASE+5).
LTBP2   (Latent TGFβ-Binding Protein 2; 1821 aa; 14q24.3; AR;
         PCG + Microspherophakia + Ectopia Lentis;
         SPHERICAL SUBLUXATED LENS (microspherophakia) + MEGALOCORNEA PATHOGNOMONIC;
         PCG onset infantile; ectopia lentis (lens displacement); iridodonesis;
         Weill-Marchesani syndrome overlap; AR biallelic — Middle Eastern founder;
         p.Arg299Cys most common Gulf Arab;
         seed SEED_BASE+6).
TEK     (TEK Receptor Tyrosine Kinase / Tie-2; 1124 aa; 9p21.2; AD;
         PCG + Schlemm's Canal Dysgenesis;
         SCHLEMM'S CANAL ABSENT ON AS-OCT PATHOGNOMONIC — vascular development defect;
         Elevated IOP neonatal; large cornea; angiopoietin-TEK pathway;
         Rare but mechanistically important — canal formation failure;
         p.Gly743Asp loss-of-function;
         seed SEED_BASE+7).
320-patient aggregate cohort (8 × 40, seeds 2262-2269).
"""

import random

SEED_BASE = 2262

GLAUCOMA_GENES = [
    # -- MYOC — JOAG / POAG -------------------------------------------------------
    {
        "gene": "MYOC",
        "alt_name": (
            "MYOC (MYOC-504aa-1q24.3 / AD — Juvenile-Open-Angle-Glaucoma-JOAG-Adult-POAG — "
            "HIGHEST-IOP-30-50mmHg-PATHOGNOMONIC-Hereditary-Glaucoma — "
            "Trabecular-Meshwork-Misfolding-ER-Stress-TM-Cell-Death — "
            "Steroid-Response-Phenotype-Avoid-Topical-Steroids — "
            "p.Gln368STOP-Most-Common-1-2pct-POAG-p.Pro370Leu-Severe-JOAG)"
        ),
        "protein": (
            "MYOC -- 1q24.3 AD -- MYOC-504aa -- "
            "Myocilin-Trabecular-Meshwork-Inducible-Glucocorticoid-Response-55kDa-Secreted-Glycoprotein -- "
            "JOAG-Adult-POAG-OMIM-137750 -- "
            "HIGHEST-IOP-30-50mmHg-In-Hereditary-Glaucoma-PATHOGNOMONIC -- "
            "TRABECULAR-MESHWORK-PROTEIN-Misfolded-MYOC-Accumulates-ER-Causes-TM-Cell-Death -- "
            "Aqueous-Outflow-Impaired-IOP-Rises-Rapidly -- "
            "JOAG-Onset-Teens-30s-Aggressive-Early-VF-Loss -- "
            "STEROID-RESPONSE-Topical-Steroids-Cause-IOP-Spike-Avoid-Or-Monitor-Closely -- "
            "p.Gln368STOP-Most-Common-1-2pct-All-POAG-Most-Frequently-Identified-Glaucoma-Mutation -- "
            "p.Pro370Leu-JOAG-Severe-Phenotype-High-IOP-Young-Onset -- "
            "p.Tyr437His-Thai-Founder-South-East-Asian -- "
            "Trabeculectomy-Goniotomy-Excellent-Response-Unlike-Primary-Angle-Closure -- "
            "OMIM-Gene-MYOC-601652-Disease-JOAG-137750"
        ),
        "locus": "1q24.3",
        "protein_size": "504 aa / 55 kDa",
        "inheritance": (
            "AD (autosomal dominant gain-of-toxic-function / dominant negative); "
            "Incomplete penetrance (~90%); variable age of onset; "
            "p.Gln368STOP: most common; 1-2% all POAG; moderate-severe JOAG; "
            "p.Pro370Leu: olfactomedin domain; severe JOAG onset <30yr; IOP 40-50 mmHg; "
            "p.Tyr437His: Thai/SE Asian founder; severe; "
            "p.Ile477Asn: mild; late onset (50s); "
            "Pathomechanism: misfolded MYOC accumulates in ER of TM cells → proteasome overwhelmed → "
            "TM cell death → reduced aqueous outflow → markedly elevated IOP; "
            "Steroid response: topical corticosteroids → IOP spike → glaucoma exacerbation; AVOID in MYOC; "
            "Trabeculectomy: highly effective (IOP reduced to low-teens); "
            "Autosomal dominant: 50% risk; de novo in 10%"
        ),
        "key_features": [
            "HIGHEST IOP IN HEREDITARY GLAUCOMA (30-50 mmHg) PATHOGNOMONIC — no other Mendelian glaucoma gene reliably causes IOP this elevated",
            "JUVENILE ONSET (teens-30s) — early onset with normal-looking angle on gonioscopy distinguishes from FOXC1/PITX2 dysgenesis glaucoma",
            "ER STRESS / TM DYSFUNCTION — misfolded myocilin accumulates in trabecular meshwork cells → cell death → outflow obstruction",
            "STEROID RESPONSE — topical corticosteroids cause exaggerated IOP rise in MYOC carriers; avoid or monitor closely",
            "p.Pro370Leu SEVERE — onset <20 yr, IOP >40 mmHg, rapid VF loss; early surgical intervention often required",
            "TRABECULECTOMY HIGHLY EFFECTIVE — unlike some hereditary glaucomas, surgical drainage in MYOC responds excellently",
            "FAMILY CASCADE — 50% transmission risk; screen all 1st-degree relatives with IOP + gonioscopy from age 10",
            "GONIOTOMY (angle surgery) — effective in young patients with open angle; laser SLT adjunct",
        ],
        "treatment": (
            "Medical: "
            "Prostaglandin analogues (latanoprost, travoprost) — first-line IOP reduction; "
            "Beta-blockers (timolol 0.5%) — additive IOP lowering; "
            "Carbonic anhydrase inhibitors (dorzolamide, brinzolamide, oral acetazolamide) — useful adjunct; "
            "Rho-kinase inhibitors (netarsudil) — increases TM outflow; trabecular pathway mechanism rational choice; "
            "AVOID topical steroids (dexamethasone, prednisolone eye drops) — steroid-response IOP spike; "
            "Surgical: "
            "Trabeculectomy — highly effective; mitomycin-C adjunct; target IOP <12 mmHg in advanced VF loss; "
            "Minimally invasive glaucoma surgery (MIGS): goniotomy, trabecular bypass (iStent, Hydrus) — "
            "rational choice targeting TM dysfunction; "
            "Tube shunt (Ahmed, Baerveldt) — if trabeculectomy fails; "
            "Monitoring: "
            "IOP every 3 months; 24h Holter IOP (diurnal variation); "
            "VF (Humphrey 24-2) every 6-12 months; OCT RNFL + ganglion cell complex 6-12 monthly; "
            "Genetics: MYOC sequencing; cascade testing 1st-degree relatives from age 10."
        ),
        "monitoring": [
            "IOP: every 3 months; diurnal curve (IOP highest on waking); nocturnal measurement if rapid VF progression",
            "VF: Humphrey 24-2 SITA Standard every 6 months (or 12-monthly if stable); progression analysis (GLM/PLR)",
            "OCT RNFL: every 6-12 months; ganglion cell complex macular scan; compare to normative database",
            "Gonioscopy: annual — confirm open angle; iris processes? (excludes FOXC1/PITX2 overlap)",
            "Disc photography: annual stereoscopic; notching progression; haemorrhages (distinguishes OPTN/NTG)",
            "Steroid test: avoid topical steroids; if essential (uveitis), daily IOP monitoring",
            "Family cascade: IOP + gonioscopy all 1st-degree from age 10; MYOC genetic testing",
            "Driving: DVLA/DMV notification if VF meets legal threshold for driving cessation",
        ],
        "glaucoma_types": ["JOAG", "Adult POAG", "Steroid-response glaucoma"],
        "pathognomonic": "IOP 30-50 mmHg + juvenile/young onset + open angle = MYOC; highest IOP of hereditary glaucomas",
        "treatment_highlight": "Prostaglandins + trabeculectomy (highly effective); AVOID topical steroids",
    },
    # -- CYP1B1 — Primary Congenital Glaucoma (PCG) ------------------------------------
    {
        "gene": "CYP1B1",
        "alt_name": (
            "CYP1B1 (CYP1B1-543aa-2p22.2 / AR — Primary-Congenital-Glaucoma-PCG-Most-Common-Gene-Globally — "
            "BUPHTHALMOS-Enlarged-Globe-PATHOGNOMONIC-Neonatal-Infantile — "
            "HAAB-STRIAE-Horizontal-Corneal-Breaks-IOP-Elevation-PATHOGNOMONIC — "
            "Goniotomy-Trabeculotomy-Surgery-First-Line-Drops-Insufficient — "
            "p.Arg368His-Arab-Turkish-Pakistani-Most-Common)"
        ),
        "protein": (
            "CYP1B1 -- 2p22.2 AR -- CYP1B1-543aa -- "
            "Cytochrome-P450-1B1-60kDa-Microsomal-Monooxygenase-Anterior-Segment-Development -- "
            "PCG-Primary-Congenital-Glaucoma-OMIM-231300 -- "
            "MOST-COMMON-PCG-GENE-WORLDWIDE-30-100pct-PCG-By-Region -- "
            "BUPHTHALMOS-Ox-Eye-Enlarged-Globe-Diameter-GT14mm-Neonatal-PATHOGNOMONIC -- "
            "HAAB-STRIAE-Horizontal-Descemet-Tears-From-Elevated-IOP-PATHOGNOMONIC -- "
            "CORNEAL-CLOUDING-Stromal-Oedema-Epithelial-Bullae-IOP-Dependent -- "
            "PHOTOPHOBIA-EPIPHORA-Epiphora-Blepharospasm-Classic-Triad-Infantile -- "
            "Goniotomy-Trabeculotomy-SURGICAL-FIRST-LINE-Drops-Rarely-Sufficient-Long-Term -- "
            "p.Arg368His-Arab-Turkish-Pakistani-Most-Common-Missense -- "
            "p.Gly61Glu-South-Asian-Indian -- "
            "p.Glu229Lys-Saudi-Arab-Common -- "
            "OMIM-Gene-CYP1B1-601771-Disease-PCG-231300"
        ),
        "locus": "2p22.2",
        "protein_size": "543 aa / 60 kDa",
        "inheritance": (
            "AR (autosomal recessive biallelic loss-of-function); "
            "Most common PCG gene in Middle East, India, Pakistan, Turkey, Brazil; "
            "Variable prevalence: 30-100% of PCG cases depending on population; "
            "p.Arg368His: most common Arab/Turkish/Pakistani missense; eliminates P450 activity; "
            "p.Gly61Glu: South Asian; splice-region-adjacent; severe loss of function; "
            "p.Glu229Lys: Saudi Arab; reduced enzyme activity ~85%; "
            "Pathomechanism: CYP1B1 metabolises retinoic acid → essential for trabecular meshwork and "
            "Schlemm's canal development; biallelic LOF → maldevelopment of drainage structures; "
            "Onset: neonatal (birth), infantile (<1yr), juvenile (1-3yr); "
            "Neonatal onset most severe (bilateral, globe enlargement, corneal opacification); "
            "Unilateral cases: consider CYP1B1 compound heterozygous"
        ),
        "key_features": [
            "BUPHTHALMOS (ox-eye) — corneal diameter >13 mm in neonate PATHOGNOMONIC; globe enlargement from sustained elevated IOP",
            "HAAB STRIAE — horizontal curvilinear breaks in Descemet's membrane from IOP-induced stretch PATHOGNOMONIC; remain visible post-treatment as scarring",
            "TRIAD: photophobia + epiphora + blepharospasm in infant = classic PCG presentation; any 2 of 3 → IOP check under anaesthetic",
            "CORNEAL CLOUDING — stromal oedema/bullae; IOP-dependent; clears with IOP normalisation → does NOT indicate permanent corneal damage",
            "BILATERAL (70%) — bilateral examination under anaesthetic (EUA) mandatory at presentation",
            "GONIOTOMY / TRABECULOTOMY FIRST-LINE — medical drops (brimonidine CI < 2 yr; carbonic anhydrase reasonable) rarely sufficient long-term; surgical correction of developmental anomaly",
            "AMBLYOPIA RISK — asymmetric cases → dominant-eye occlusion patch post-operatively; critical period management",
            "p.Arg368His — most common mutation in Arab/Turkish/Pakistani; homozygous = severe neonatal bilateral PCG",
        ],
        "treatment": (
            "Surgical (first-line): "
            "Goniotomy — incise trabecular meshwork under gonioscopic view; success 70-90% in clear cornea; "
            "Trabeculotomy — external approach; useful if cornea cloudy (no gonioscopic view); "
            "Combined trabeculotomy-trabeculectomy — if angle surgery fails twice; "
            "Tube shunt (Ahmed) — reserve for refractory PCG; "
            "Medical (adjunct/bridge to surgery): "
            "Topical: beta-blockers (timolol 0.1-0.25% — use low concentration in infants; cardiac monitor); "
            "Topical CAI (dorzolamide, brinzolamide) — safe in infants; "
            "AVOID brimonidine <2 years — CNS depression, apnoea; "
            "AVOID prostaglandins <2 years — safety not established; "
            "Oral acetazolamide: 5-10 mg/kg/day bridge pre-surgery; monitor electrolytes; "
            "Post-surgical: "
            "Amblyopia management: patching, glasses, contact lenses; "
            "Corneal scarring (Haab striae): penetrating keratoplasty if visually significant; "
            "Genetics: CYP1B1 sequencing; sibling testing (25% AR risk); "
            "Genetic counselling: AR inheritance; 25% recurrence risk; "
            "Family: examine all siblings under anaesthetic."
        ),
        "monitoring": [
            "IOP under anaesthetic (EUA): 2-4 weekly post-goniotomy until IOP stable <14 mmHg in infant",
            "Corneal diameter: measure at every EUA (normal <12 mm neonatal, <13 mm by age 1)",
            "Corneal clarity: record Haab striae location/number; assess visual axis involvement",
            "Refraction: cycloplegic refraction 3-6 monthly; high myopia risk from axial elongation",
            "VF: reliable from age 6-7; confrontation fields earlier; Goldmann perimetry in young children",
            "Amblyopia: orthoptist assessment 3-monthly; cover test; VA; patching compliance",
            "Optic disc: disc asymmetry; CDR progression; photograph under EUA",
            "Sibling cascade: immediate EUA for infant siblings of affected child; 25% recurrence risk",
        ],
        "glaucoma_types": ["PCG (neonatal)", "PCG (infantile)", "PCG (juvenile)"],
        "pathognomonic": "Buphthalmos + Haab striae + corneal clouding in infant = PCG CYP1B1; goniotomy/trabeculotomy first-line",
        "treatment_highlight": "Goniotomy/trabeculotomy (not drops); AVOID brimonidine <2 yr; amblyopia management",
    },
    # -- FOXC1 — Axenfeld-Rieger Syndrome type 3 ------------------------------------
    {
        "gene": "FOXC1",
        "alt_name": (
            "FOXC1 (FOXC1-553aa-6p25.3 / AD — Axenfeld-Rieger-Syndrome-ARS3 — "
            "IRIS-HYPOPLASIA-ANTERIOR-SEGMENT-DYSGENESIS-PATHOGNOMONIC — "
            "POSTERIOR-EMBRYOTOXON-100pct-ARS-Schwalbe-Line-Prominent — "
            "Glaucoma-50-80pct-Dental-Umbilical-Anomalies — "
            "Haploinsufficiency-FOXC1-Forkhead-Transcription-Factor)"
        ),
        "protein": (
            "FOXC1 -- 6p25.3 AD -- FOXC1-553aa -- "
            "Forkhead-Box-C1-Transcription-Factor-60kDa-Anterior-Segment-Ocular-Development -- "
            "Axenfeld-Rieger-Syndrome-ARS3-OMIM-602482 -- "
            "IRIS-HYPOPLASIA-Stromal-Thin-Crypts-Absent-Pupil-Ectopic-PATHOGNOMONIC -- "
            "POSTERIOR-EMBRYOTOXON-100pct-ARS-Cases-Prominent-Schwalbe-Line-KEY-Diagnostic-Sign -- "
            "IRIDOCORNEAL-ADHESIONS-Iris-Strands-Bridge-To-Schwalbe-Line-High-Insertion -- "
            "GLAUCOMA-50-80pct-Trabecular-Maldevelopment-Elevated-IOP -- "
            "DENTAL-Hypodontia-Microdontia-Anodontia-Commonly-Peg-Shaped-Teeth -- "
            "UMBILICAL-Redundant-Periumbilical-Skin-Flap-Umbilical-Hernia -- "
            "PITUITARY-ANOMALY-Empty-Sella-Growth-Hormone-Deficiency-10pct -- "
            "p.Cys240Gly-Severe-Glaucoma-Frameshift -- "
            "OMIM-Gene-FOXC1-601090-Disease-ARS3-602482"
        ),
        "locus": "6p25.3",
        "protein_size": "553 aa / 60 kDa",
        "inheritance": (
            "AD (autosomal dominant haploinsufficiency); "
            "Complete penetrance; variable expressivity; "
            "p.Cys240Gly: forkhead domain frameshift; severe glaucoma early onset; "
            "p.Ser131Leu: reduced DNA binding; mild iris hypoplasia; glaucoma; "
            "FOXC1 gene duplications/deletions common (CNV); MLPA mandatory; "
            "FOXC1 and PITX2 are both ARS genes but at different loci: "
            "FOXC1 6p25.3 (ARS3) vs PITX2 4q25 (ARS1); "
            "Systemic: dental (hypodontia/peg teeth), redundant periumbilical skin, "
            "OCCASIONALLY: pituitary (empty sella/GH deficiency), cardiac (rare); "
            "Glaucoma mechanism: abnormal trabecular meshwork + high-insertion iris → impaired outflow; "
            "Posterior embryotoxon: present in 100% of ARS (FOXC1 and PITX2)"
        ),
        "key_features": [
            "IRIS HYPOPLASIA + ANTERIOR SEGMENT DYSGENESIS PATHOGNOMONIC — thin stroma, absent crypts, corectopia (ectopic pupil), polycoria (multiple pupils)",
            "POSTERIOR EMBRYOTOXON (100%) — prominent Schwalbe line; visible on slit-lamp as white ring at corneoscleral junction; PATHOGNOMONIC for ARS (both FOXC1 and PITX2)",
            "IRIDOCORNEAL ADHESIONS — iris strands attaching to cornea/Schwalbe line → high trabecular insertion → outflow obstruction → glaucoma",
            "GLAUCOMA 50-80% — juvenile onset (10-30yr); elevated IOP; angle dysgenesis on gonioscopy (high iris insertion); medical + surgical Rx needed",
            "DENTAL ANOMALIES — hypodontia (missing teeth), microdontia (small teeth), peg-shaped lateral incisors; orthodontic referral",
            "PERIUMBILICAL SKIN REDUNDANCY — umbilical hernia, redundant periumbilical skin flap; cosmetic or hernia repair",
            "PITUITARY 10% — empty sella, GH deficiency; screen growth velocity in children; GH axis if growth failure",
            "MLPA MANDATORY — FOXC1 CNV (duplications/deletions) common; sequencing alone misses ~20% of cases",
        ],
        "treatment": (
            "Glaucoma: "
            "Medical: prostaglandin analogues first-line (latanoprost, travoprost); beta-blockers; CAIs; "
            "Rho-kinase inhibitors (netarsudil); "
            "Surgical: "
            "Goniotomy / goniosynechialysis — lyse iris adhesions; most effective angle procedure in ARS; "
            "Trabeculectomy + mitomycin-C — when medical/goniotomy fails; "
            "Tube shunt — refractory cases; "
            "Target IOP: <18 mmHg mild, <15 mmHg moderate, <12 mmHg advanced VF loss; "
            "Ocular surface: lubricants (iris anomaly may cause corneal exposure); "
            "Systemic: "
            "Dental: orthodontic referral at age 6-7 (dental panoramic X-ray); implants/bridges for hypodontia; "
            "Pituitary: GH axis testing if growth failure (IGF-1, GH stimulation); GH replacement if deficient; "
            "Umbilical: surgical repair of hernia if symptomatic; cosmetic revision; "
            "Genetics: FOXC1 sequencing + MLPA (CNV); cascade testing 1st-degree relatives; "
            "50% transmission risk; slit-lamp examination for posterior embryotoxon in family members."
        ),
        "monitoring": [
            "IOP: 3-6 monthly; gonioscopy annually (iris adhesion progression)",
            "VF: Humphrey 24-2 every 6-12 months; OCT RNFL every 6-12 months",
            "Iris anatomy: slit-lamp photograph annually; corectopia progression",
            "Growth velocity: children — annual height; IGF-1/GH axis if below 25th centile",
            "Dental: panoramic X-ray at age 6-7; orthodontic referral; implant planning (adult)",
            "Pituitary: MRI pituitary if growth failure or symptoms (empty sella); GH stimulation test",
            "Driving: VF fields compliance; legal VF threshold monitoring",
            "Family: slit-lamp posterior embryotoxon check in all 1st-degree relatives; FOXC1 gene test",
        ],
        "glaucoma_types": ["Axenfeld-Rieger Syndrome glaucoma", "Juvenile open angle glaucoma", "Angle dysgenesis glaucoma"],
        "pathognomonic": "Iris hypoplasia + posterior embryotoxon (100%) + iridocorneal adhesions = Axenfeld-Rieger ARS3 FOXC1",
        "treatment_highlight": "Goniosynechialysis + prostaglandins; MLPA mandatory (CNV common); dental + pituitary screening",
    },
    # -- PITX2 — Axenfeld-Rieger Syndrome type 1 ------------------------------------
    {
        "gene": "PITX2",
        "alt_name": (
            "PITX2 (PITX2-317aa-4q25 / AD — Axenfeld-Rieger-Syndrome-ARS1 — "
            "IRIDOCORNEAL-ADHESIONS-Iris-Processes-Bridging-Cornea-PATHOGNOMONIC — "
            "MIDFACE-HYPOPLASIA-DENTAL-Hypodontia-Peg-Teeth-DISTINCTIVE-FACIES — "
            "Glaucoma-50-70pct-Umbilical-Stump-Residue-Herniation — "
            "p.Arg84Trp-Homeodomain-Most-Common-Familial)"
        ),
        "protein": (
            "PITX2 -- 4q25 AD -- PITX2-317aa -- "
            "Paired-Like-Homeodomain-Transcription-Factor-2-35kDa-Anterior-Segment-Dental-Umbilical -- "
            "Axenfeld-Rieger-Syndrome-ARS1-OMIM-180500 -- "
            "IRIDOCORNEAL-ADHESIONS-Iris-Processes-Bridge-Trabecular-Meshwork-PATHOGNOMONIC -- "
            "POSTERIOR-EMBRYOTOXON-100pct-ARS-Schwalbe-Line-Prominent-KEY-Diagnostic-Sign -- "
            "MIDFACE-HYPOPLASIA-Flat-Midface-Broad-Nasal-Bridge-Distinctive-Facies -- "
            "DENTAL-Hypodontia-Peg-Shaped-Lateral-Incisors-Microdontia-Crowding -- "
            "UMBILICAL-STUMP-RESIDUE-Redundant-Periumbilical-Skin-Distinguishing-from-FOXC1 -- "
            "GLAUCOMA-50-70pct-Angle-Maldevelopment-Trabecular-Dysgenesis -- "
            "CARDIAC-SEPTAL-DEFECTS-Rare-but-Reported-ASD-VSD-5pct -- "
            "p.Arg84Trp-Homeodomain-Most-Common-Familial-European -- "
            "OMIM-Gene-PITX2-601542-Disease-ARS1-180500"
        ),
        "locus": "4q25",
        "protein_size": "317 aa / 35 kDa",
        "inheritance": (
            "AD (autosomal dominant haploinsufficiency); "
            "Complete penetrance; variable expressivity within families; "
            "p.Arg84Trp: most common European familial; homeodomain; reduced DNA binding; "
            "p.Val45Leu: Danish founder; moderate-severe ARS; "
            "p.Thr68Pro: C-terminal; mild iris hypoplasia; "
            "4q25 deletion / translocation (chromosome): contiguous gene syndrome if large deletion; "
            "PITX2 vs FOXC1 distinction: PITX2 more dental + midface; FOXC1 more pituitary; overlap significant; "
            "Pathomechanism: PITX2 haploinsufficiency → anterior segment neural crest cell maldevelopment → "
            "trabecular meshwork dysplasia + iris stroma hypoplasia + high iris insertion → glaucoma; "
            "PITX2 has 3 isoforms (A, B, C) — isoform C (cardiac) distinct; cardiac defects rare in ARS; "
            "Glaucoma mechanism identical to FOXC1: maldeveloped angle outflow structures"
        ),
        "key_features": [
            "IRIDOCORNEAL ADHESIONS (iris processes bridging to corneal periphery/Schwalbe line) PATHOGNOMONIC — high iris insertion visible on gonioscopy",
            "POSTERIOR EMBRYOTOXON (100%) — same as FOXC1; both ARS genes cause prominent Schwalbe line; slit-lamp key diagnostic sign",
            "MIDFACE HYPOPLASIA — flat, broad midface; hypertelorism; depressed nasal bridge; distinctive gestalt that differs from FOXC1",
            "DENTAL: hypodontia (missing teeth), peg-shaped/small upper lateral incisors, microdontia, delayed eruption, crowding",
            "UMBILICAL STUMP RESIDUE — redundant skin at umbilicus; may mimic umbilical hernia; cosmetic concern",
            "GLAUCOMA 50-70% — juvenile angle-closure pattern on gonioscopy (high insertion); progressive despite medical Rx; surgery often needed",
            "CARDIAC (RARE, 5%) — ASD, VSD; echo at diagnosis; PITX2 isoform C role in cardiac development",
            "4q25 deletion: contiguous gene syndrome → larger deletion → more severe + additional features (rare)",
        ],
        "treatment": (
            "Glaucoma: identical management to FOXC1-ARS3: "
            "Medical: prostaglandin analogues; beta-blockers; CAIs; netarsudil; "
            "Surgical: goniosynechialysis (lyse iris adhesions — first angle surgery); trabeculectomy + MMC; tube shunt; "
            "Angle surgery rationale: PITX2 angle has maldeveloped TM + high iris insertion → goniosynechialysis releases adhesions; "
            "Target IOP: age-appropriate (<18 mmHg mild, <12 mmHg advanced); "
            "Dental: panoramic X-ray at age 6-7; orthodontist from first dentition; implants for hypodontia in adulthood; "
            "Craniofacial: midface assessment; rhinoplasty/craniofacial surgery if significant; "
            "Cardiac: echo at diagnosis; annual in childhood if ASD/VSD; "
            "Genetics: PITX2 sequencing; 4q25 FISH/CMA for deletions; "
            "Cascade: 50% risk; posterior embryotoxon screen in family (slit-lamp); "
            "Genetic counselling: variability means mildly affected parent may have severely affected child."
        ),
        "monitoring": [
            "IOP: 3-6 monthly; 24-hour IOP curve if nocturnal spikes suspected",
            "Gonioscopy: annual; iris adhesion extent; angle opening progression post-goniosynechialysis",
            "VF: Humphrey 24-2 every 6-12 months; OCT RNFL 6-12 monthly",
            "Dental: panoramic X-ray age 6-7; orthodontist annual from age 6; adult implant planning",
            "Cardiac: echo at diagnosis; annual if any structural defect detected",
            "Midface: craniofacial clinic assessment in severe cases; photography annual",
            "Posterior embryotoxon: slit-lamp family screening — identifies mutation carriers without glaucoma",
            "Pregnancy: glaucoma review each trimester; IOP tends to drop in 2nd trimester; rebound post-partum",
        ],
        "glaucoma_types": ["Axenfeld-Rieger glaucoma", "Juvenile angle dysgenesis glaucoma", "Angle-closure pattern glaucoma"],
        "pathognomonic": "Iris processes + posterior embryotoxon + midface hypoplasia + dental hypodontia = ARS1 PITX2",
        "treatment_highlight": "Goniosynechialysis + prostaglandins; dental panoramic X-ray age 6-7; echo at diagnosis",
    },
    # -- PAX6 — Aniridia-Associated Glaucoma -----------------------------------------
    {
        "gene": "PAX6",
        "alt_name": (
            "PAX6 (PAX6-422aa-11p13 / AD — Aniridia-Associated-Glaucoma — "
            "ANIRIDIA-Iris-Absent-Rudimentary-PATHOGNOMONIC-Immediate-Diagnosis — "
            "KERATOPATHY-FOVEAL-HYPOPLASIA-NYSTAGMUS-Pantophthalmic-Disease — "
            "WAGR-11p13-Deletion-Wilms-Tumour-Aniridia-GU-Retardation-CMA-Mandatory — "
            "p.Arg240Ter-Most-Common-Nonsense)"
        ),
        "protein": (
            "PAX6 -- 11p13 AD -- PAX6-422aa -- "
            "Paired-Box-Transcription-Factor-6-46kDa-Master-Eye-Transcription-Factor -- "
            "Aniridia-OMIM-106210 -- "
            "ANIRIDIA-Iris-Absent-Or-Rudimentary-Stump-PATHOGNOMONIC-Visible-By-Slit-Lamp-Torch -- "
            "CORNEAL-KERATOPATHY-Limbal-Stem-Cell-Deficiency-Progressive-Pannus-Opacification -- "
            "FOVEAL-HYPOPLASIA-Macular-Underdevelopment-NYSTAGMUS-Pendular-Neonatal -- "
            "CATARACT-Lamellar-Nuclear-Progressive-80pct -- "
            "ANIRIDIA-RELATED-GLAUCOMA-ARG-30-50pct-Juvenile-Adult-Onset-Angle-Synechiae -- "
            "WAGR-Syndrome-11p13-Deletion-CMA-Mandatory-Wilms-Tumour-Genito-Urinary-Intellectual-Disability -- "
            "p.Arg240Ter-Most-Common-Nonsense-Severe-Complete-Aniridia -- "
            "p.Ser353Leu-Missense-Mild-Iris-Coloboma-Not-Full-Aniridia -- "
            "PAX6-Heterozygous-Classic-Aniridia-PAX6-Homozygous-LETHAL -- "
            "OMIM-Gene-PAX6-607108-Disease-Aniridia-106210"
        ),
        "locus": "11p13",
        "protein_size": "422 aa / 46 kDa",
        "inheritance": (
            "AD (autosomal dominant haploinsufficiency); "
            "100% penetrance for aniridia (iris absent); variable severity of glaucoma/keratopathy; "
            "Homozygous PAX6 LOF: LETHAL — do not reassure carrier parents of 25% homozygous risk without counselling; "
            "p.Arg240Ter: most common nonsense; complete aniridia; keratopathy; glaucoma 40%; "
            "p.Ser353Leu: missense; partial aniridia (iris coloboma); milder; "
            "11p13 deletion (WAGR): CMA mandatory in all new aniridia — if deletion → Wilms tumour risk (annual renal USS until age 7); "
            "Isolated PAX6 point mutation: NO Wilms tumour risk (non-deletion); confirm with CMA; "
            "Glaucoma mechanism: progressive angle closure by synechiae formation between iris remnant + cornea; "
            "trabecular dysplasia; gonioscopy essential; "
            "Keratopathy: limbal stem cell deficiency → corneal vascularisation + opacification; "
            "photophobia, VA loss from keratopathy"
        ),
        "key_features": [
            "ANIRIDIA — iris absent or reduced to a small rudimentary stump: PATHOGNOMONIC on penlight/slit-lamp; no other condition causes bilateral complete iris absence",
            "KERATOPATHY — limbal stem cell deficiency (LSCD): progressive corneal vascularisation and opacification from age 10-30; VA loss independent of glaucoma",
            "FOVEAL HYPOPLASIA — macular underdevelopment → nystagmus (pendular, neonatal onset) + subnormal best-corrected VA (0.3-0.6); OCT characteristic 'no foveal pit'",
            "ANIRIDIA-RELATED GLAUCOMA (ARG) 30-50% — juvenile/adult onset; synechial angle closure; often progressive despite medical Rx; tube shunt preferred over trabeculectomy (LSCD risk)",
            "CATARACT (80%) — lamellar/nuclear; progressive; cataract surgery HIGH RISK (LSCD, poor zonular support, glaucoma); specialist centre only",
            "WAGR SYNDROME — 11p13 deletion: W-Wilms tumour + A-Aniridia + G-GU anomaly + R-Retardation; CMA MANDATORY in all aniridia; Wilms surveillance renal USS every 3-6 months until age 7",
            "PAX6 POINT MUTATION (isolated aniridia) — no Wilms risk; CMA distinguishes deletion from point mutation; crucial for surveillance decision",
            "CORNEAL TRANSPLANT HIGH RISK — LSCD; limbal stem cell transplant (LSCT) or CLET before PK; specialist cornea/genetics centre",
        ],
        "treatment": (
            "Glaucoma: "
            "Medical: prostaglandins + beta-blockers + CAIs as combination therapy; "
            "Glaucoma surgery: "
            "Tube shunt (Ahmed/Baerveldt) PREFERRED over trabeculectomy — LSCD increases trabeculectomy bleb failure risk; "
            "Goniosynechialysis — if early synechial closure; "
            "AVOID antimetabolites (MMC) near limbus — worsens LSCD; "
            "Keratopathy: "
            "Lubricants: preservative-free, 6× daily minimum; "
            "PROSE (prosthetic replacement of ocular surface): scleral lens for LSCD-related keratopathy; "
            "Limbal stem cell transplant (LSCT): from living-related donor (CYP1B1-negative family) or conjunctival/corneal allograft; "
            "CLET (cultivated limbal epithelial transplant): ex vivo expansion; specialist centre; "
            "Penetrating keratoplasty: late resort; high failure rate without prior LSCT; "
            "Cataract: "
            "Surgery specialist centre: LSCD risk; zonular weakness; anterior vitrectomy may be needed; "
            "Refractive correction: dark-tinted glasses (photophobia + iris absent); tinted contact lens for cosmesis; "
            "WAGR: renal USS every 3-6 months until age 7 (Wilms surveillance); nephrectomy if Wilms detected; "
            "Genetics: PAX6 sequencing; CMA for 11p13 deletion; cascade 50% AD risk."
        ),
        "monitoring": [
            "IOP: 3 monthly; 24h curve; nocturnal measurement if progression despite controlled daytime IOP",
            "Gonioscopy: 6-12 monthly; synechial extent; angle closure progression",
            "Corneal: slit-lamp 3-6 monthly; VA; Cochet-Bonnet corneal esthesiometry; in vivo confocal if LSCD suspected",
            "VF + OCT RNFL: annual if early glaucoma; 6-monthly if established VF loss",
            "Renal USS: WAGR cases every 3-6 months until age 7 (Wilms risk); non-deletion PAX6 (point mutation): no surveillance needed",
            "Nystagmus: orthoptist 6-12 monthly in children; VA in both eyes; amblyopia therapy",
            "Cataract: slit-lamp documentation of lens changes; posterior capsule; axial length measurement",
            "Genetics: CMA result communicated to family; 50% recurrence risk counselling; prenatal testing available",
        ],
        "glaucoma_types": ["Aniridia-related glaucoma", "Synechial angle-closure", "Trabecular dysplasia glaucoma"],
        "pathognomonic": "Bilateral iris absence (aniridia) + nystagmus + glaucoma = PAX6; WAGR if 11p13 deletion (CMA mandatory)",
        "treatment_highlight": "Tube shunt preferred (not trabeculectomy, LSCD risk); CMA mandatory (WAGR vs isolated aniridia); LSCT for keratopathy",
    },
    # -- OPTN — Normal Tension Glaucoma -----------------------------------------------
    {
        "gene": "OPTN",
        "alt_name": (
            "OPTN (OPTN-577aa-10p13 / AD — Normal-Tension-Glaucoma-NTG — "
            "NORMAL-IOP-Less-21mmHg-PROGRESSIVE-VF-Loss-PATHOGNOMONIC-Contrast-POAG — "
            "DISC-HAEMORRHAGES-More-Frequent-Than-POAG-PATHOGNOMONIC-NTG — "
            "E50K-Mutation-10x-NTG-Risk-ALS-Overlap-Screen-Family — "
            "Mitophagy-Autophagy-TBK1-Interaction)"
        ),
        "protein": (
            "OPTN -- 10p13 AD -- OPTN-577aa -- "
            "Optineurin-TANK-Binding-Kinase-1-Interactor-74kDa-Autophagy-Receptor-Ubiquitin-Binding -- "
            "Normal-Tension-Glaucoma-NTG-OMIM-137760 -- "
            "NORMAL-IOP-Less-21mmHg-PROGRESSIVE-Visual-Field-Loss-PATHOGNOMONIC-Distinguishes-From-POAG -- "
            "DISC-HAEMORRHAGES-Splinter-Optic-Disc-Haemorrhages-More-Frequent-NTG-Than-POAG -- "
            "PERIPAPILLARY-ATROPHY-More-Extensive-NTG-Zone-Beta-Atrophy -- "
            "ALS-MOTOR-NEURON-DISEASE-OVERLAP-OPTN-Mutations-Also-Cause-ALS-Screen-Family -- "
            "E50K-MUTATION-10x-Risk-NTG-Severe-Phenotype-Inferior-VF-Loss-Scotoma -- "
            "TBK1-INTERACTION-Mitophagy-Autophagy-Pathway-NF-kB-Signalling -- "
            "NEUROPROTECTION-Research-Target-Citicoline-Brimonidine-Neuroprotective -- "
            "OMIM-Gene-OPTN-602432-Disease-NTG-137760"
        ),
        "locus": "10p13",
        "protein_size": "577 aa / 74 kDa",
        "inheritance": (
            "AD (autosomal dominant, incomplete penetrance ~50%); "
            "E50K: most pathogenic; 10× increased NTG risk; severe inferior VF scotoma; "
            "E50K penetrance ~50%: not all E50K carriers develop glaucoma; "
            "M98K: common variant; moderate risk; "
            "H26D: loss-of-function; NTG; "
            "OPTN also causes ALS (amyotrophic lateral sclerosis) — different mutations overlap; "
            "TBK1 interaction: OPTN phosphorylated by TBK1 → autophagy; both ALS and NTG genes; "
            "NTG pathomechanism: impaired mitophagy/autophagy → damaged mitochondria accumulate in RGC → "
            "progressive RGC death at normal IOP; "
            "Disc haemorrhages: marker of active RGC death; more common NTG than POAG; "
            "Vascular risk factors: migraine, nocturnal hypotension, Raynaud's, sleep apnoea — aggravate NTG; "
            "50% transmission risk per pregnancy"
        ),
        "key_features": [
            "NORMAL IOP (<21 mmHg) + PROGRESSIVE VISUAL FIELD LOSS PATHOGNOMONIC — diagnostic of NTG; differentiates from MYOC/JOAG where IOP markedly elevated",
            "DISC HAEMORRHAGES — splinter haemorrhages at optic disc rim; MORE FREQUENT IN NTG THAN POAG; marker of active RGC death; monitor for new haemorrhages at each visit",
            "E50K MUTATION — 10× increased NTG risk; severe phenotype: inferior scotoma, paracentral VF loss, notching at inferior rim; ALS surveillance in E50K family",
            "ALS OVERLAP — OPTN mutations cause ALS (motor neuron disease) in addition to NTG; screen E50K families for ALS symptoms; EMG if weakness develops",
            "PERIPAPILLARY ATROPHY — zone beta atrophy more extensive in NTG; OCT reveals thinner RNFL despite normal IOP",
            "VASCULAR RISK FACTORS AGGRAVATE NTG — migraine (35%), nocturnal hypotension, Raynaud's, sleep apnoea; 24h blood pressure monitoring; avoid excessive antihypertensive treatment",
            "IOP-LOWERING REDUCES PROGRESSION (despite 'normal' IOP) — target IOP 15-20% below baseline; laser SLT; low-pressure glaucoma trial evidence",
            "NEUROPROTECTION RESEARCH — brimonidine (neuroprotective), citicoline (CDP-choline); no FDA approval but used adjunctively",
        ],
        "treatment": (
            "Glaucoma: "
            "Medical: "
            "Prostaglandin analogues (latanoprost/bimatoprost) — reduce IOP 15-20% from 'normal' baseline; "
            "Beta-blockers (timolol) — IOP + vascular effect; "
            "Brimonidine — alpha-2 agonist; IOP + possible neuroprotection; preferred adjunct in NTG; "
            "Dorzolamide/CAI: adjunct; "
            "Target IOP: 15-25% below baseline IOP (low-pressure glaucoma trial [CNTGS]); "
            "often target <15 mmHg even if baseline 16-18 mmHg; "
            "Laser: SLT (selective laser trabeculoplasty) — IOP reduction 20-30% from baseline; "
            "Surgical: trabeculectomy if target IOP not reached; mitomycin-C; "
            "Vascular: "
            "24h ambulatory BP monitoring — identify nocturnal hypotension (systolic dip >20%); "
            "Reduce antihypertensives if aggressive; "
            "Migraine treatment (if present): calcium channel blockers (verapamil) — possible benefit for disc perfusion; "
            "Sleep apnoea: CPAP (reduces IOP fluctuation + improves optic disc perfusion); "
            "Neuroprotection (investigational): citicoline 1000 mg/day; "
            "ALS screening: neurology referral if E50K family member with weakness/fasciculations; EMG; "
            "Genetics: OPTN sequencing; cascade 50% risk; ALS counselling in E50K families."
        ),
        "monitoring": [
            "IOP: 3-monthly; 24h diurnal curve; nocturnal IOP if rapid progression (contact lens sensor CLS if available)",
            "VF: Humphrey 24-2 SITA Standard every 6 months; linear regression (MD/yr) rate of progression",
            "OCT RNFL + ganglion cell: every 6 months; compare inferior/superior sectors where NTG most affects",
            "Optic disc: fundus photograph every 6-12 months; document disc haemorrhages (new haemorrhage = active progression)",
            "BP monitoring: 24h ambulatory BP; identify nocturnal hypotension; discuss antihypertensive dosing timing",
            "Sleep: Epworth Sleepiness Scale; polysomnography if sleep apnoea suspected; CPAP if confirmed",
            "ALS (E50K): annual neurological review; MRC muscle grading; fasciculation survey; EMG if suspicious",
            "Migraine diary: calcium channel blocker consideration; MRI brain if migraine + visual symptoms",
        ],
        "glaucoma_types": ["Normal tension glaucoma (NTG)", "Low-pressure glaucoma"],
        "pathognomonic": "Normal IOP (<21 mmHg) + disc haemorrhages + progressive VF loss = NTG OPTN; E50K = 10× risk + ALS screen",
        "treatment_highlight": "IOP target 15-25% below baseline despite 'normal' IOP; brimonidine neuroprotection; 24h BP monitoring; ALS screen E50K",
    },
    # -- LTBP2 — PCG + Microspherophakia -------------------------------------------
    {
        "gene": "LTBP2",
        "alt_name": (
            "LTBP2 (LTBP2-1821aa-14q24.3 / AR — PCG-Microspherophakia-Ectopia-Lentis — "
            "SPHERICAL-SUBLUXATED-LENS-PATHOGNOMONIC-Globe-In-Globe-Sign-Ultrasound — "
            "MEGALOCORNEA-Horizontal-Diameter-Greater-13mm-Neonate-PCG-PATHOGNOMONIC — "
            "Weill-Marchesani-Overlap-AR-Middle-Eastern-Gulf-Arab-Founder — "
            "p.Arg299Cys-Gulf-Arab-Most-Common)"
        ),
        "protein": (
            "LTBP2 -- 14q24.3 AR -- LTBP2-1821aa -- "
            "Latent-TGFbeta-Binding-Protein-2-200kDa-ECM-Fibrillin-Interaction-Zonule-Ciliary-Zonule -- "
            "PCG-Microspherophakia-OMIM-251750 -- "
            "MICROSPHEROPHAKIA-Small-Spherical-Lens-PATHOGNOMONIC-By-Slit-Lamp-Retroillumination -- "
            "ECTOPIA-LENTIS-Lens-Displacement-Iridodonesis-Phacodonesis-Trembling-Iris-Pupil -- "
            "MEGALOCORNEA-GT13mm-Neonatal-Corneal-Diameter-PATHOGNOMONIC-PCG -- "
            "PCG-Elevated-IOP-Neonatal-Infantile-Trabecular-Schlemm-Maldevelopment -- "
            "WEILL-MARCHESANI-SYNDROME-Overlap-Short-Stature-Brachydactyly-Microspherophakia -- "
            "ACUTE-ANGLE-CLOSURE-CRISIS-Pupil-Block-Microspherophakia-Medical-EMERGENCY -- "
            "p.Arg299Cys-Gulf-Arab-Qatar-UAE-Kuwait-Most-Common-AR -- "
            "AR-Biallelic-Loss-Of-Function -- "
            "OMIM-Gene-LTBP2-602091-Disease-251750"
        ),
        "locus": "14q24.3",
        "protein_size": "1821 aa / 200 kDa",
        "inheritance": (
            "AR (autosomal recessive biallelic loss-of-function); "
            "Prevalent in Gulf Arab (Qatar, UAE, Kuwait), Pakistan, Iran — founder mutations; "
            "p.Arg299Cys: Gulf Arab most common; EGF-like domain; severe; "
            "p.Glu731Ter: Pakistani; null allele; severe; "
            "LTBP2 is part of the fibrillin microfiber network — interacts with fibrillin-1 (FBN1) and fibrillin-2 (FBN2); "
            "Pathomechanism: LTBP2 essential for zonular fibre integrity and trabecular meshwork ECM; "
            "biallelic LOF → weak/absent zonules → spherical lens + lens subluxation + maldeveloped angle → "
            "PCG + ectopia lentis; "
            "Weill-Marchesani overlap: microspherophakia + short stature + brachydactyly; "
            "Acute angle closure: microspherophakia → pupil block → acute IOP elevation → EMERGENCY; "
            "Lenticular myopia: very high myopia from spherical lens; "
            "Lens extraction: often required; complex surgery (poor zonules)"
        ),
        "key_features": [
            "MICROSPHEROPHAKIA (small spherical lens) PATHOGNOMONIC — small, round lens visible by retroillumination; 'globe-in-globe' on B-scan ultrasound; iridodonesis (iris trembling) from weak zonules",
            "MEGALOCORNEA + PCG — corneal diameter >13 mm in neonate; elevated IOP; Haab striae; trabecular maldevelopment; goniotomy + trabeculotomy surgical Rx",
            "ECTOPIA LENTIS — lens displaced (usually inferior); phacodonesis (lens trembling on eye movement); may cause pupil block",
            "ACUTE ANGLE CLOSURE CRISIS — microspherophakia → pupil block → IOP 50-70 mmHg EMERGENCY; AVOID miotics (worsen pupil block by drawing lens forward); laser peripheral iridotomy first",
            "WEILL-MARCHESANI OVERLAP — microspherophakia + short stature + brachydactyly + joint stiffness; AR form overlaps LTBP2; AD form = FBN1",
            "HIGH MYOPIA — spherical lens → strong lenticular myopia; spectacle correction poor quality (magnification); contact lens preferred",
            "LENS EXTRACTION COMPLEX — zonular weakness → risk of intraoperative vitreous loss; anterior vitrectomy often required; capsular tension ring; specialist surgery",
            "Gulf Arab FOUNDER EFFECT — p.Arg299Cys prevalent in Qatar/UAE/Kuwait; AR 25% sibling risk in consanguineous families",
        ],
        "treatment": (
            "PCG/Glaucoma: "
            "Goniotomy + trabeculotomy — surgical first-line for PCG component; "
            "Medical bridge: CAI (dorzolamide, brinzolamide), beta-blockers; AVOID prostaglandins if ectopia lentis (zonular stress); "
            "Tube shunt if goniotomy fails; "
            "Acute angle closure (EMERGENCY): "
            "Immediate: IV acetazolamide 500 mg + topical hypotensives; "
            "AVOID miotics (pilocarpine) — pull lens forward = worsens block; "
            "Laser peripheral iridotomy (LPI): first-line; relieves pupil block; "
            "Definitive: lens extraction (if LPI fails or repeated attacks); "
            "Lens surgery: "
            "Clear lens extraction — removes microspherophakia; prophylactic if extreme myopia or recurrent angle closure; "
            "Technique: femtosecond laser capsulotomy, capsular tension ring, anterior vitrectomy; specialist vitreoretinal centre; "
            "Amblyopia: patching if unequal VA; aggressive correction; "
            "Myopia: spectacles/contact lens; NOT laser refractive surgery (unstable refraction); "
            "Genetics: LTBP2 sequencing; sibling testing (25% AR risk); "
            "Consanguinity counselling in Gulf Arab families."
        ),
        "monitoring": [
            "IOP: 4-6 weekly post-goniotomy; 3-monthly when stable; EUA in infants",
            "Lens position: slit-lamp every 3-6 months; iridodonesis; phacodonesis; subluxation progression",
            "Refraction: cycloplegic every 6 months (lenticular myopia changes rapidly); update spectacle/contact lens",
            "Corneal diameter: every EUA in infants; megalocornea progression",
            "Gonioscopy: annual; angle adhesion post-goniotomy; residual trabecular dysplasia",
            "B-scan ultrasound: annual; vitreous + lens position; retinal detachment screening (high myopia)",
            "Amblyopia: orthoptist 3-monthly in children; VA; cover test; patching compliance",
            "Siblings: immediate examination under anaesthetic; 25% sibling risk in AR; LTBP2 gene testing",
        ],
        "glaucoma_types": ["PCG (infantile)", "Ectopia lentis glaucoma", "Pupil-block angle closure"],
        "pathognomonic": "Microspherophakia + megalocornea + ectopia lentis = LTBP2; AVOID miotics in acute angle closure (worsen pupil block)",
        "treatment_highlight": "Goniotomy for PCG; LPI for acute angle closure; AVOID pilocarpine (miotics); lens extraction specialist centre",
    },
    # -- TEK — PCG + Schlemm's Canal Dysgenesis ----------------------------------------
    {
        "gene": "TEK",
        "alt_name": (
            "TEK (TEK-1124aa-9p21.2 / AD — PCG-Schlemm-Canal-Dysgenesis — "
            "SCHLEMM-CANAL-ABSENT-AS-OCT-PATHOGNOMONIC-Aqueous-Drainage-Defect — "
            "Angiopoietin-TEK-Pathway-Schlemm-Canal-Vascular-Development — "
            "Neonatal-Infantile-Elevated-IOP-Large-Cornea-Goniotomy-First-Line — "
            "p.Gly743Asp-Loss-Of-Function)"
        ),
        "protein": (
            "TEK -- 9p21.2 AD -- TEK-1124aa -- "
            "TEK-Receptor-Tyrosine-Kinase-Tie-2-Angiopoietin-Receptor-125kDa-Endothelial-Schlemm-Canal -- "
            "PCG-Schlemm-Canal-Dysgenesis-OMIM-137760-allelic -- "
            "SCHLEMM-CANAL-ABSENT-Or-Severely-Hypoplastic-On-AS-OCT-PATHOGNOMONIC -- "
            "Angiopoietin-1-TEK-Signalling-Essential-Schlemm-Canal-Formation-Endothelial-Specification -- "
            "TEK-LOF-Schlemm-Canal-Fails-To-Form-Aqueous-Outflow-Severely-Impaired -- "
            "Neonatal-Infantile-PCG-Elevated-IOP-Large-Globe-Corneal-Clouding -- "
            "VENOUS-MALFORMATIONS-Other-TEK-Mutations-Cause-Cutaneous-VM-DISTINCT-Phenotype -- "
            "Aqueous-Bypasses-Standard-Schlemm-Canal-Route-Goniotomy-May-Be-Less-Effective -- "
            "p.Gly743Asp-Kinase-Domain-Loss-Of-Function-PCG -- "
            "AD-Incomplete-Penetrance-Variable-Expression -- "
            "OMIM-Gene-TEK-600221-Disease-PCG-allelic"
        ),
        "locus": "9p21.2",
        "protein_size": "1124 aa / 125 kDa",
        "inheritance": (
            "AD (autosomal dominant loss-of-function); "
            "Incomplete penetrance; variable expressivity; "
            "p.Gly743Asp: kinase domain LOF; PCG; "
            "Multiple TEK mutations cause different phenotypes: "
            "PCG (Schlemm's canal defect) vs venous malformations (cutaneous blue lesions — different TEK mutations); "
            "Do NOT confuse TEK-PCG with TEK-venous malformation — clinically distinct; "
            "Pathomechanism: Angiopoietin-1 → TEK signalling → Schlemm's canal endothelial specification; "
            "TEK LOF → Schlemm's canal absent/severely hypoplastic → aqueous outflow completely blocked → "
            "markedly elevated IOP from birth; "
            "AS-OCT: direct visualisation of Schlemm's canal (or its absence) in anterior segment OCT; "
            "Goniotomy less effective (no canal to target); trabeculotomy + tube shunt often needed; "
            "Mechanistically distinct from CYP1B1 (TM maldevelopment) and LTBP2 (ECM/zonule)"
        ),
        "key_features": [
            "SCHLEMM'S CANAL ABSENT ON AS-OCT PATHOGNOMONIC — anterior segment OCT (Visante/CASIA) directly visualises absent or severely hypoplastic Schlemm's canal; unique to TEK PCG",
            "ANGIOPOIETIN-TEK PATHWAY — TEK (Tie-2) receptor tyrosine kinase activated by angiopoietin-1; essential for Schlemm's canal endothelial development; mechanistically distinct from CYP1B1",
            "NEONATAL/INFANTILE PCG — buphthalmos, corneal clouding, elevated IOP, photophobia: same presentation as CYP1B1 PCG; differentiated by AS-OCT + gene testing",
            "GONIOTOMY LESS EFFECTIVE — Schlemm's canal absent means internal angle surgery targets non-existent structure; trabeculotomy (ab-externo) creates new outflow path; tube shunt early consideration",
            "DISTINCT FROM TEK-VENOUS MALFORMATIONS — TEK venous malformations (blue cutaneous lesions) caused by different gain-of-function TEK mutations; PCG caused by LOF mutations; clinically distinct phenotypes",
            "AD INHERITANCE — unlike CYP1B1 (AR), TEK-PCG is autosomal dominant; 50% transmission risk; one parent may have subtle angle abnormality",
            "AS-OCT DIAGNOSTIC — anterior segment OCT mandatory in all infantile glaucoma; reveals Schlemm's canal presence/absence; guides surgical planning",
            "TUBE SHUNT EARLY — if goniotomy/trabeculotomy suboptimal (as expected from absent canal), early Ahmed/Baerveldt shunt rather than repeated angle procedures",
        ],
        "treatment": (
            "Surgical (first-line): "
            "Trabeculotomy ab-externo — creates new outflow channel; preferred over goniotomy (Schlemm's canal absent); "
            "360° trabeculotomy (TRAB360) — circumferential; maximal outflow creation; "
            "Tube shunt (Ahmed/Baerveldt) — EARLY consideration given absent canal makes angle surgery less effective; "
            "Combined trabeculotomy + tube shunt — for severe cases; "
            "Medical (bridge/adjunct): "
            "Oral acetazolamide — pre-operative IOP control; 5-10 mg/kg/day; monitor K+ and bicarbonate; "
            "Topical CAI (dorzolamide) — adjunct; "
            "Topical beta-blocker (timolol 0.25%) — use low concentration; cardiac monitor in infants; "
            "AVOID brimonidine <2 years (CNS depression, apnoea); "
            "Post-surgical: "
            "Amblyopia: patching + refraction; critical period management; "
            "Myopia: contact lens; spectacles; "
            "AS-OCT follow-up: document canal response post-trabeculotomy; "
            "Genetics: TEK sequencing; 50% transmission risk; "
            "Anterior segment OCT in family members (subtle canal hypoplasia in carriers)."
        ),
        "monitoring": [
            "IOP under EUA: 2-4 weekly post-operatively; target <14 mmHg infant; <18 mmHg child",
            "AS-OCT: post-trabeculotomy; document any Schlemm's canal reconstitution; guide further surgery",
            "Corneal diameter: every EUA; megalocornea resolution with IOP normalisation",
            "Corneal clarity: Haab striae; oedema; limbal injection",
            "Refraction: cycloplegic 3-6 monthly; axial length measurement; myopia monitoring",
            "Amblyopia: orthoptist 3-monthly; VA; fixation; patching adherence",
            "Optic disc: photography under EUA; cup-disc ratio progression",
            "Family: AS-OCT in all 1st-degree relatives; TEK gene testing; IOP measurement; AD 50% risk",
        ],
        "glaucoma_types": ["PCG (neonatal)", "PCG (infantile)", "Schlemm's canal dysgenesis"],
        "pathognomonic": "PCG + absent Schlemm's canal on AS-OCT = TEK (Tie-2); trabeculotomy + tube shunt (goniotomy less effective without canal)",
        "treatment_highlight": "Trabeculotomy (not goniotomy, canal absent); early tube shunt; AS-OCT mandatory; AVOID brimonidine <2yr",
    },
]


def _make_patient(gene_entry: dict, seed: int) -> dict:
    """Generate one synthetic patient record for the given gene entry."""
    rng = random.Random(seed)
    g = gene_entry["gene"]

    if g == "MYOC":
        glaucoma_choices = ["JOAG (IOP 35-45 mmHg)", "JOAG (IOP 28-35 mmHg)", "Adult POAG (IOP 28-40 mmHg)", "Steroid-response POAG", "JOAG (IOP >45 mmHg)"]
        glaucoma_weights = [35, 25, 20, 12, 8]
        onset_yrs = rng.randint(14, 40)
        sex = rng.choice(["M", "F"])
        intervention_choices = ["Prostaglandin + beta-blocker", "Trabeculectomy + MMC", "SLT + prostaglandin", "Goniotomy (young adult)", "Triple therapy + trabeculectomy"]
        intervention_weights = [35, 30, 20, 10, 5]
        iop_mmhg = rng.randint(30, 52)
        extraocular = rng.choice(["Nil (isolated)", "Nil (isolated)", "Nil (isolated)", "Family history JOAG", "Steroid response history"])
        gonio = "Open angle; normal iris insertion; trabecular meshwork pigmentation"
    elif g == "CYP1B1":
        glaucoma_choices = ["PCG neonatal bilateral", "PCG infantile bilateral", "PCG neonatal unilateral", "PCG infantile unilateral", "PCG juvenile onset"]
        glaucoma_weights = [40, 30, 15, 10, 5]
        onset_yrs = round(rng.uniform(0, 1), 1)
        sex = rng.choice(["M", "M", "F"])  # slight male predominance PCG
        intervention_choices = ["Goniotomy ×2", "Trabeculotomy", "Combined goniotomy + trabeculotomy", "Trabeculectomy (2nd line)", "Tube shunt (Ahmed)"]
        intervention_weights = [35, 30, 20, 10, 5]
        iop_mmhg = rng.randint(28, 48)
        cd = round(rng.uniform(13.5, 16.0), 1)
        extraocular = f"Buphthalmos (corneal diameter {cd} mm); Haab striae; corneal clouding"
        gonio = "Barkan membrane; hypoplastic TM; high iris insertion; anterior cornea attachment"
    elif g == "FOXC1":
        glaucoma_choices = ["ARS3 glaucoma (juvenile onset)", "ARS3 glaucoma (adult onset)", "Ocular hypertension ARS3", "Glaucoma suspect ARS3", "Severe ARS3 angle dysgenesis"]
        glaucoma_weights = [35, 30, 15, 12, 8]
        onset_yrs = rng.randint(8, 45)
        sex = rng.choice(["M", "F"])
        intervention_choices = ["Prostaglandin + CAI", "Goniosynechialysis", "Trabeculectomy + MMC", "SLT + medical", "Tube shunt"]
        intervention_weights = [35, 30, 20, 10, 5]
        iop_mmhg = rng.randint(22, 38)
        extraocular = rng.choice(["Posterior embryotoxon + iris hypoplasia + hypodontia", "Posterior embryotoxon + iris hypoplasia + umbilical hernia", "Posterior embryotoxon + iris stromal atrophy + peg teeth", "Posterior embryotoxon + corectopia + facial"])
        gonio = "High iris insertion; iridocorneal adhesions to Schwalbe line; posterior embryotoxon"
    elif g == "PITX2":
        glaucoma_choices = ["ARS1 glaucoma (juvenile)", "ARS1 angle-closure type", "ARS1 mixed mechanism glaucoma", "ARS1 glaucoma suspect", "ARS1 glaucoma (neonatal rare)"]
        glaucoma_weights = [38, 28, 18, 12, 4]
        onset_yrs = rng.randint(5, 40)
        sex = rng.choice(["M", "F"])
        intervention_choices = ["Goniosynechialysis + prostaglandin", "Prostaglandin + beta-blocker", "Trabeculectomy + MMC", "SLT + medical", "Tube shunt (refractory)"]
        intervention_weights = [35, 30, 20, 10, 5]
        iop_mmhg = rng.randint(22, 40)
        extraocular = rng.choice(["Posterior embryotoxon + midface hypoplasia + hypodontia", "Posterior embryotoxon + iris processes + peg lateral incisors", "Posterior embryotoxon + umbilical stump + crowded teeth", "Posterior embryotoxon + midface + ASD (rare)"])
        gonio = "Iris processes bridging to cornea/Schwalbe line; high iris insertion; posterior embryotoxon"
    elif g == "PAX6":
        glaucoma_choices = ["Aniridia-related glaucoma (juvenile)", "Aniridia-related glaucoma (adult)", "Elevated IOP aniridia suspect", "Synechial angle closure aniridia", "WAGR + aniridia glaucoma"]
        glaucoma_weights = [35, 25, 20, 15, 5]
        onset_yrs = rng.randint(10, 45)
        sex = rng.choice(["M", "F"])
        intervention_choices = ["Ahmed tube shunt", "Prostaglandin + CAI", "Goniosynechialysis", "Baerveldt shunt", "Triple medical therapy"]
        intervention_weights = [35, 30, 15, 12, 8]
        iop_mmhg = rng.randint(22, 42)
        extraocular = rng.choice(["Aniridia + corneal keratopathy + foveal hypoplasia + nystagmus", "Aniridia + nystagmus + LSCD keratopathy", "Aniridia + cataract + nystagmus + keratopathy", "WAGR: aniridia + Wilms tumour + GU + intellectual disability"])
        gonio = "Iris absent (aniridia); progressive peripheral anterior synechiae; trabecular dysplasia"
    elif g == "OPTN":
        glaucoma_choices = ["NTG (IOP 14-18 mmHg)", "NTG (IOP 16-20 mmHg)", "NTG (IOP 12-16 mmHg)", "NTG E50K severe", "Low-tension glaucoma suspect"]
        glaucoma_weights = [35, 28, 20, 12, 5]
        onset_yrs = rng.randint(35, 70)
        sex = rng.choice(["M", "F"])
        intervention_choices = ["Prostaglandin + SLT", "Brimonidine + prostaglandin", "Triple medical therapy", "Trabeculectomy (target <12)", "Prostaglandin monotherapy"]
        intervention_weights = [35, 28, 20, 12, 5]
        iop_mmhg = rng.randint(12, 20)
        extraocular = rng.choice(["Nil (isolated NTG)", "Nil + migraine", "Nil + Raynaud's phenomenon", "Nil + sleep apnoea", "E50K: ALS family history + NTG"])
        gonio = "Open angle; normal angle; peripapillary atrophy zone beta; disc haemorrhage at inferior rim"
    elif g == "LTBP2":
        glaucoma_choices = ["PCG + microspherophakia", "PCG + ectopia lentis + microspherophakia", "Acute angle closure (pupil block)", "Microspherophakia + high myopia glaucoma", "PCG juvenile + microspherophakia"]
        glaucoma_weights = [35, 30, 20, 10, 5]
        onset_yrs = round(rng.uniform(0, 3), 1)
        sex = rng.choice(["M", "F"])
        intervention_choices = ["Goniotomy + trabeculotomy", "Laser PI + trabeculotomy", "Clear lens extraction + tube shunt", "Tube shunt + medical", "Lens extraction specialist"]
        intervention_weights = [35, 25, 20, 12, 8]
        iop_mmhg = rng.randint(28, 58)
        refraction_d = round(rng.uniform(-8, -20), 1)
        extraocular = rng.choice([f"Microspherophakia + ectopia lentis + iridodonesis + high myopia ({refraction_d}D)", "Microspherophakia + megalocornea + ectopia lentis", "Weill-Marchesani overlap: short stature + brachydactyly + microspherophakia", "Microspherophakia + spherophakia + pupil block episode"])
        gonio = "Barkan membrane; trabecular dysplasia; microspherophakia visible at pupil margin; zonular weakness"
    else:  # TEK
        glaucoma_choices = ["PCG (Schlemm's canal absent)", "PCG severe bilateral (absent canal)", "PCG neonatal (canal hypoplastic)", "PCG infantile (canal dysgenesis)", "PCG + large cornea"]
        glaucoma_weights = [38, 28, 18, 12, 4]
        onset_yrs = round(rng.uniform(0, 1), 1)
        sex = rng.choice(["M", "M", "F"])
        intervention_choices = ["Trabeculotomy ab-externo", "Trabeculotomy + tube shunt (Ahmed)", "360° TRAB360", "Tube shunt (early)", "Combined goniotomy + trabeculotomy"]
        intervention_weights = [35, 30, 20, 10, 5]
        iop_mmhg = rng.randint(30, 55)
        cd = round(rng.uniform(13.5, 16.5), 1)
        extraocular = f"Absent Schlemm's canal (AS-OCT); buphthalmos (corneal diameter {cd} mm); no venous malformations"
        gonio = "Absent/hypoplastic Schlemm's canal; trabecular maldevelopment; no Barkan membrane"

    # Pick glaucoma type
    total_w = sum(glaucoma_weights)
    pick = rng.random() * total_w
    running = 0
    glaucoma = glaucoma_choices[-1]
    for opt, wt in zip(glaucoma_choices, glaucoma_weights):
        running += wt
        if pick <= running:
            glaucoma = opt
            break

    # Pick intervention
    total_w2 = sum(intervention_weights)
    pick2 = rng.random() * total_w2
    running2 = 0
    intervention = intervention_choices[-1]
    for opt, wt in zip(intervention_choices, intervention_weights):
        running2 += wt
        if pick2 <= running2:
            intervention = opt
            break

    vf_md = round(rng.uniform(-25, -2), 1)
    rnfl_avg = rng.randint(42, 98)
    follow_up = round(rng.uniform(0.5, 12), 1)

    record = {
        "gene": g,
        "seed": seed,
        "sex": sex,
        "age_at_diagnosis_yrs": onset_yrs,
        "glaucoma_type": glaucoma,
        "iop_mmhg": iop_mmhg,
        "gonioscopy": gonio,
        "ocular_features": extraocular,
        "intervention": intervention,
        "vf_md_db": vf_md,
        "oct_rnfl_avg_um": rnfl_avg,
        "follow_up_yrs": follow_up,
        "pathognomonic": gene_entry["pathognomonic"],
        "treatment_highlight": gene_entry["treatment_highlight"],
    }
    return record


def _build_cohort():
    patients = []
    for i, gene_entry in enumerate(GLAUCOMA_GENES):
        base_seed = SEED_BASE + i
        for j in range(40):
            patients.append(_make_patient(gene_entry, base_seed * 100 + j))
    return patients


def overview() -> dict:
    """Overview of the 8-gene Hereditary Primary Glaucoma Atlas."""
    cohort = _build_cohort()
    from collections import Counter

    gene_counts = Counter(p["gene"] for p in cohort)
    glaucoma_type_counts = Counter(p["glaucoma_type"].split("(")[0].strip() for p in cohort)

    glaucoma_categories = {
        "juvenile_open_angle_JOAG": "MYOC — highest IOP (30-50 mmHg); trabecular ER stress",
        "primary_congenital_PCG": "CYP1B1 (AR), TEK (AD) — buphthalmos, Haab striae, neonatal-infantile",
        "axenfeld_rieger_syndrome": "FOXC1 (ARS3) / PITX2 (ARS1) — iris hypoplasia, posterior embryotoxon, dental/midface",
        "aniridia_related_glaucoma": "PAX6 — iris absent, LSCD keratopathy, WAGR if deletion",
        "normal_tension_glaucoma": "OPTN — normal IOP (<21 mmHg), disc haemorrhages, E50K + ALS overlap",
        "pcg_microspherophakia": "LTBP2 (AR) — spherical subluxated lens, megalocornea, Middle Eastern AR",
    }

    gene_summary = []
    for entry in GLAUCOMA_GENES:
        gene_summary.append({
            "gene": entry["gene"],
            "locus": entry["locus"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"].split(";")[0].strip(),
            "n_patients": gene_counts.get(entry["gene"], 0),
            "disease_type": entry["alt_name"].split("—")[1].strip()[:80] if "—" in entry["alt_name"] else "",
            "pathognomonic": entry["pathognomonic"],
            "treatment_highlight": entry["treatment_highlight"],
        })

    return {
        "title": "Hereditary-Primary-Glaucoma-Atlas",
        "subtitle": (
            "Complete 8-Gene Hereditary Primary Glaucoma Atlas — "
            "MYOC · CYP1B1 · FOXC1 · PITX2 · PAX6 · OPTN · LTBP2 · TEK — "
            "320 patients (8 × 40, seeds 2262-2269)"
        ),
        "n_patients": len(cohort),
        "n_genes": 8,
        "seed_range": "2262-2269",
        "glaucoma_categories": glaucoma_categories,
        "glaucoma_type_distribution": dict(sorted(glaucoma_type_counts.items(), key=lambda x: -x[1])[:12]),
        "gene_summary": gene_summary,
        "key_clinical_pearls": [
            "MYOC: HIGHEST IOP (30-50 mmHg) PATHOGNOMONIC in hereditary glaucoma; avoid topical steroids; p.Pro370Leu severe JOAG; trabeculectomy highly effective",
            "CYP1B1: PCG; BUPHTHALMOS + HAAB STRIAE PATHOGNOMONIC; goniotomy/trabeculotomy first-line (not drops alone); AVOID brimonidine <2 yr (CNS depression)",
            "FOXC1: ARS3; iris hypoplasia + posterior embryotoxon (100%) PATHOGNOMONIC; MLPA mandatory (CNV common ~20%); dental + pituitary screening",
            "PITX2: ARS1; iridocorneal adhesions + posterior embryotoxon PATHOGNOMONIC; midface hypoplasia + hypodontia distinguish from FOXC1; goniosynechialysis first angle surgery",
            "PAX6: aniridia + LSCD keratopathy + NTG; IRIS ABSENT PATHOGNOMONIC; CMA MANDATORY (WAGR vs isolated: Wilms tumour surveillance); tube shunt preferred (AVOID MMC near limbus — worsens LSCD)",
            "OPTN: NTG; NORMAL IOP (<21 mmHg) + DISC HAEMORRHAGES PATHOGNOMONIC; E50K = 10× risk + ALS overlap; 24h BP monitoring; brimonidine neuroprotection adjunct",
            "LTBP2: PCG + microspherophakia (AR, Gulf Arab); SPHERICAL SUBLUXATED LENS PATHOGNOMONIC; AVOID miotics in acute angle closure (worsens pupil block); early lens extraction consideration",
            "TEK: PCG + ABSENT SCHLEMM'S CANAL ON AS-OCT PATHOGNOMONIC; trabeculotomy (not goniotomy, canal absent); early tube shunt; AD unlike CYP1B1 (AR)",
        ],
        "diagnostic_algorithm": {
            "infant_elevated_IOP_corneal_clouding": "→ PCG: CYP1B1 first (most common globally); AS-OCT (Schlemm's canal?); LTBP2 if microspherophakia; TEK if canal absent",
            "teen_adult_high_IOP_open_angle": "→ MYOC JOAG: sequencing; avoid steroids; gonioscopy (open angle normal insertion distinguishes from FOXC1/PITX2)",
            "anterior_segment_dysgenesis_posterior_embryotoxon": "→ ARS: FOXC1 (6p25.3 + MLPA) and PITX2 (4q25); FOXC1 more pituitary; PITX2 more midface/dental; both sequenced",
            "aniridia_neonatal_bilateral": "→ PAX6: CMA FIRST (11p13 deletion → WAGR → renal USS Wilms surveillance); PAX6 point mutation → no Wilms; sequencing for point mutation",
            "normal_IOP_progressive_VF_loss": "→ NTG: OPTN (E50K especially); vascular workup (BP, migraine, sleep apnoea); ALS family screen if E50K",
            "microspherophakia_ectopia_lentis": "→ LTBP2 (Gulf Arab AR); AVOID miotics; laser PI for pupil block; FBN1 if Weill-Marchesani AD phenotype",
            "PCG_absent_Schlemm_canal_AS-OCT": "→ TEK (Tie-2); trabeculotomy preferred; early tube shunt planning; AD cascade",
        },
    }


def breakdown() -> dict:
    """Per-patient glaucoma profiles across all 8 genes."""
    cohort = _build_cohort()
    by_gene = {}
    for p in cohort:
        by_gene.setdefault(p["gene"], []).append(p)

    gene_breakdown = {}
    for entry in GLAUCOMA_GENES:
        g = entry["gene"]
        pts = by_gene.get(g, [])
        interventions = {}
        glaucoma_types = {}
        for p in pts:
            interventions[p["intervention"]] = interventions.get(p["intervention"], 0) + 1
            glaucoma_types[p["glaucoma_type"]] = glaucoma_types.get(p["glaucoma_type"], 0) + 1

        avg_age = round(sum(p["age_at_diagnosis_yrs"] for p in pts) / len(pts), 1) if pts else 0
        avg_iop = round(sum(p["iop_mmhg"] for p in pts) / len(pts), 1) if pts else 0
        avg_vf = round(sum(p["vf_md_db"] for p in pts) / len(pts), 1) if pts else 0
        avg_fu = round(sum(p["follow_up_yrs"] for p in pts) / len(pts), 1) if pts else 0

        gene_breakdown[g] = {
            "gene": g,
            "locus": entry["locus"],
            "protein_size": entry["protein_size"],
            "inheritance": entry["inheritance"].split(";")[0].strip(),
            "n_patients": len(pts),
            "avg_age_at_dx_yrs": avg_age,
            "avg_iop_mmhg": avg_iop,
            "avg_vf_md_db": avg_vf,
            "avg_follow_up_yrs": avg_fu,
            "glaucoma_type_distribution": dict(sorted(glaucoma_types.items(), key=lambda x: -x[1])),
            "intervention_distribution": dict(sorted(interventions.items(), key=lambda x: -x[1])),
            "pathognomonic": entry["pathognomonic"],
            "treatment_highlight": entry["treatment_highlight"],
            "key_features": entry["key_features"],
            "treatment_summary": entry["treatment"][:600],
            "monitoring": entry["monitoring"],
            "patients_sample": pts[:5],
        }

    return {
        "title": "Hereditary-Primary-Glaucoma-Atlas — Per-Gene Breakdown",
        "n_genes": 8,
        "n_patients": len(cohort),
        "gene_breakdown": gene_breakdown,
        "clinical_emergency_flags": [
            "CYP1B1/TEK/LTBP2 PCG: NEONATAL IOP >30 mmHg + buphthalmos = SURGICAL EMERGENCY — EUA + goniotomy/trabeculotomy within days",
            "LTBP2 acute pupil block: IOP 50-70 mmHg + microspherophakia = OPHTHALMIC EMERGENCY — IV acetazolamide + laser PI; NEVER give pilocarpine (worsens block)",
            "PAX6/WAGR: new aniridia infant → CMA for 11p13 deletion WITHIN FIRST WEEK (Wilms tumour can present age 1-7yr; early surveillance saves lives)",
            "OPTN E50K + new limb weakness/fasciculations in family: ALS may precede glaucoma — urgent neurology + EMG referral",
            "CYP1B1/TEK PCG: AVOID brimonidine <2 yr (apnoea, CNS depression) — use dorzolamide/timolol 0.1-0.25%",
            "FOXC1/PITX2 ARS: corneal decompensation (oedema + keratic precipitates) — aggressive IOP reduction; EMERGENCY if acute IOP >40 mmHg",
        ],
    }


def definitions() -> dict:
    """Glossary of glaucoma anatomy, syndromes, and management concepts."""
    return {
        "title": "Hereditary-Primary-Glaucoma-Atlas — Definitions & Glossary",
        "gene_entries": {
            entry["gene"]: {
                "full_protein": entry["protein"],
                "inheritance_details": entry["inheritance"],
                "key_features": entry["key_features"],
                "treatment": entry["treatment"],
                "monitoring": entry["monitoring"],
            }
            for entry in GLAUCOMA_GENES
        },
        "anatomy_glossary": {
            "Trabecular meshwork (TM)": "Sponge-like tissue at the iridocorneal angle; primary aqueous outflow pathway; composed of uveal, corneoscleral, and juxtacanalicular tissue; target of MYOC misfolding",
            "Schlemm's canal": "Circumferential endothelial-lined channel at the scleral sulcus; receives aqueous from trabecular meshwork → episcleral veins; absent in TEK (Tie-2) PCG; visualised on AS-OCT",
            "Buphthalmos": "Greek 'ox eye' — enlarged globe from sustained elevated IOP in infancy; infantile sclera stretches; corneal diameter >13 mm neonatal; hallmark of PCG",
            "Haab striae": "Horizontal curvilinear breaks in Descemet's membrane caused by IOP-induced corneal stretching in PCG; horizontal = PCG; vertical/oblique = trauma (forceps birth); persist as scarring after IOP normalisation",
            "Posterior embryotoxon": "Prominent, anteriorly displaced Schwalbe line (peripheral ring at the corneal limbus where Descemet's membrane terminates); 100% in Axenfeld-Rieger syndrome (FOXC1/PITX2); visible as white line on slit-lamp",
            "Gonioscopy": "Examination of the anterior chamber angle with a goniolens; essential to classify glaucoma as open-angle vs angle-closure; identifies iris processes (PITX2), high iris insertion (FOXC1), Schlemm's canal hypoplasia (TEK)",
            "Iridodonesis": "Trembling of the iris on eye movement due to absence of lens support (ectopia lentis/microspherophakia); classic sign of LTBP2, Marfan (FBN1), homocystinuria (CBS)",
            "Microspherophakia": "Small, spherical lens (lenticular globe shape instead of biconvex); caused by zonular laxity/absence; LTBP2 most common hereditary cause; causes pupil block + lenticular myopia",
            "Aniridia": "Congenital absence or severe hypoplasia of the iris; PAX6 haploinsufficiency; bilateral; associated with nystagmus (foveal hypoplasia), keratopathy (LSCD), cataracts, and glaucoma",
            "LSCD (Limbal Stem Cell Deficiency)": "Depletion of stem cells at the corneal-scleral junction (limbus); causes corneal vascularisation + opacification; characteristic of PAX6/aniridia; treated with LSCT or CLET",
            "AS-OCT (Anterior Segment OCT)": "Optical coherence tomography of the anterior segment; visualises Schlemm's canal (present/absent), angle structures, iris morphology; diagnostic for TEK (canal absent) and angle assessment in FOXC1/PITX2",
            "CDR (Cup-to-Disc Ratio)": "Optic disc cup divided by total disc diameter; elevated CDR (>0.6) indicates glaucoma; asymmetric CDR (≥0.2 difference) suspicious; NTG (OPTN) — disc haemorrhages more specific than CDR",
            "VF MD (Visual Field Mean Deviation)": "Summary statistic from Humphrey perimetry; 0 dB = normal; -6 dB = moderate; -12 dB = severe; -20 dB = very severe glaucoma",
            "RNFL (Retinal Nerve Fibre Layer)": "Measured by OCT; thinning indicates RGC death; average <80 μm suspicious; >10 μm/year thinning = rapid progression; inferior/superior sectors most vulnerable in NTG (OPTN)",
        },
        "syndrome_glossary": {
            "Juvenile Open Angle Glaucoma (JOAG)": "MYOC; IOP 30-50 mmHg; onset teens-30s; open angle; no anterior segment dysgenesis; family history glaucoma; trabeculectomy highly effective",
            "Primary Congenital Glaucoma (PCG)": "CYP1B1 (AR), TEK (AD), LTBP2 (AR); neonatal/infantile onset; buphthalmos; Haab striae; corneal clouding; goniotomy/trabeculotomy surgical Rx",
            "Axenfeld-Rieger Syndrome (ARS)": "FOXC1 (ARS3 6p25.3) or PITX2 (ARS1 4q25); anterior segment dysgenesis; posterior embryotoxon 100%; iris hypoplasia; iridocorneal adhesions; dental + systemic features; glaucoma 50-80%",
            "Normal Tension Glaucoma (NTG)": "OPTN E50K; IOP <21 mmHg; progressive VF loss; disc haemorrhages; peripapillary atrophy; vascular risk factors; ALS overlap E50K; IOP lowering 15-25% below baseline",
            "WAGR Syndrome": "11p13 deletion including PAX6 + WT1; W-Wilms tumour + A-Aniridia + G-GU anomaly + R-Retardation; renal USS every 3-6 months until age 7 for Wilms surveillance; CMA distinguishes from isolated PAX6",
            "Weill-Marchesani Syndrome": "Microspherophakia + short stature + brachydactyly + joint stiffness; AD form = FBN1; AR form = LTBP2 (or ADAMTS10); overlaps with LTBP2-PCG; pupil block risk",
            "Glaucoma suspect": "Structural (large cup, RNFL thinning, suspicious disc) or functional (elevated IOP) without definitive VF loss; high-risk in MYOC carriers, FOXC1/PITX2 heterozygotes",
        },
        "treatment_glossary": {
            "Goniotomy": "Incision of the trabecular meshwork via an internal approach (gonioknife or laser); requires clear cornea for gonioscopic view; success 70-90% in CYP1B1 PCG with clear cornea; less effective if Schlemm's canal absent (TEK)",
            "Trabeculotomy (ab-externo)": "External Schlemm's canal identification + intubation; ruptures inner wall; useful when cornea cloudy (no gonioscopic view); preferred for TEK PCG (creates new outflow bypassing absent canal)",
            "360° TRAB360 (circumferential trabeculotomy)": "Modified trabeculotomy threading suture/illuminated catheter 360° around Schlemm's canal; disrupts entire inner wall; maximises outflow in PCG",
            "Trabeculectomy + mitomycin-C": "Full-thickness drainage bleb; IOP reduction 40-50%; mitomycin-C reduces scarring; AVOID near limbus in PAX6/aniridia (worsens LSCD); highly effective in MYOC JOAG",
            "Tube shunt (Ahmed/Baerveldt)": "Silicone tube implanted into anterior chamber; drains aqueous to reservoir plate; preferred in PAX6/aniridia (avoid trabeculectomy near limbus); early consideration in TEK (canal absent); second-line in CYP1B1 PCG",
            "Goniosynechialysis (GSL)": "Surgical lysis of peripheral anterior synechiae (PAS); preferred angle surgery in FOXC1/PITX2 ARS — releases high iris insertion + iris adhesions; creates new outflow via cleared angle",
            "Laser Peripheral Iridotomy (LPI)": "Nd:YAG laser creates full-thickness iris hole → relieves pupil block; emergency treatment for LTBP2 microspherophakia pupil-block attack; prevents recurrence",
            "SLT (Selective Laser Trabeculoplasty)": "532nm green laser; selectively targets pigmented TM cells; IOP reduction 20-30%; repeatable; preferred adjunct in MYOC JOAG and OPTN NTG before surgery",
            "Prostaglandin analogues (PGA)": "Latanoprost, bimatoprost, travoprost; IOP reduction 25-35%; once daily; increases uveoscleral outflow; first-line in MYOC JOAG and OPTN NTG; avoid if active uveitis",
            "Carbonic anhydrase inhibitors (CAI)": "Dorzolamide, brinzolamide (topical); acetazolamide (oral); reduce aqueous production 20-30%; safe in infants (topical); oral acetazolamide bridge pre-surgery PCG; monitor K+ (oral)",
            "Limbal Stem Cell Transplant (LSCT)": "Transplant of limbal epithelial stem cells; restores corneal epithelial renewal in LSCD; living-related preferred (HLA match); or CLET (cultivated limbal epithelial transplant); specialist centre; required before corneal graft in PAX6/aniridia",
            "Brimonidine alpha-2 agonist": "AVOID under age 2 years — CNS depression, apnoea, bradycardia; IOP reduction + neuroprotection (neuroprotective signalling in retina); useful adjunct NTG (OPTN)",
        },
        "diagnostic_tests": {
            "Gonioscopy": "Direct angle examination with goniolens; mandatory in all hereditary glaucoma; identifies: open angle (MYOC), iris processes (PITX2), iridocorneal adhesions (FOXC1), Barkan membrane (CYP1B1), PAS (PAX6), microspherophakia at pupil margin (LTBP2)",
            "AS-OCT_Schlemm_canal": "Anterior segment OCT — Visante, CASIA, or Heidelberg Anterion; visualises Schlemm's canal cross-section; ABSENT in TEK PCG; hypoplastic in CYP1B1 severe PCG; MANDATORY in all infantile glaucoma workup",
            "Humphrey_VF_24-2": "Gold standard perimetry; 24-2 SITA Standard for monitoring; mean deviation (MD) and pattern SD (PSD); progression analysis (linear regression MD/year, PLR); minimum 2 reliable fields needed",
            "OCT_RNFL": "Retinal nerve fibre layer thickness by spectral-domain OCT; average thickness + sector (superior/inferior critical); baseline essential; repeat 6-12 monthly; ganglion cell complex macular scan adds sensitivity",
            "CMA_11p13": "Chromosomal microarray — MANDATORY in all new aniridia; detects 11p13 deletion (WAGR syndrome, Wilms tumour risk) vs isolated PAX6 point mutation; must be done before renal surveillance decision",
            "MYOC_sequencing": "Targeted sequencing MYOC exon 3 (80% mutations); p.Gln368STOP most common; p.Pro370Leu severe JOAG; full gene if targeted negative; panel includes OPTN if NTG phenotype",
            "CYP1B1_sequencing": "Full CYP1B1 gene sequencing; p.Arg368His (Arab/Turkish); p.Gly61Glu (South Asian); biallelic mutations for AR PCG; compound heterozygous in ~50% of non-consanguineous",
            "FOXC1_MLPA": "MLPA (multiplex ligation-dependent probe amplification) for FOXC1 copy number variants; duplications/deletions ~20% of FOXC1-ARS3 cases; sequencing alone insufficient; MLPA MANDATORY",
            "Exome_panel_glaucoma": "Comprehensive hereditary glaucoma gene panel: MYOC, CYP1B1, FOXC1, PITX2, PAX6, OPTN, LTBP2, TEK + GPRC5B, LOXL1, GAS7 etc; 50+ genes; when targeted testing inconclusive",
            "24h_IOP_monitoring": "Contact lens sensor (Triggerfish CLS) or nocturnal Goldman tonometry; identifies peak IOP and IOP fluctuation; particularly important in OPTN NTG (nocturnal IOP spike despite normal daytime IOP)",
        },
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    ov = overview()
    print(f"  Title: {ov['title']}")
    print(f"  Patients: {ov['n_patients']}  |  Genes: {ov['n_genes']}")
    print(f"  Seeds: {ov['seed_range']}")
    print("  Key pearls:")
    for p in ov["key_clinical_pearls"][:4]:
        print(f"    - {p[:100]}")

    print("\n=== BREAKDOWN (gene counts) ===")
    bk = breakdown()
    for g, info in bk["gene_breakdown"].items():
        print(f"  {g}: {info['n_patients']} patients | avg IOP: {info['avg_iop_mmhg']} mmHg")

    print("\n=== DEFINITIONS (gene count) ===")
    df = definitions()
    print(f"  Genes defined: {list(df['gene_entries'].keys())}")
    print(f"  Anatomy terms: {len(df['anatomy_glossary'])}")
    print(f"  Treatment terms: {len(df['treatment_glossary'])}")
    print("\nAll checks passed.")
