'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'CEP120 Architecture & Centriole Biology', 'Definitions'];

// NPHP20 colour scheme — CEP120 / centriole elongation / JBTS31 / SRPS2B
const ACCENT  = '#1b5e20';   // deep green — CEP120 / centriole scaffold; unique upstream mechanism
const ACCENT2 = '#880e4f';   // deep pink — Joubert syndrome (JBTS31); MTS; cerebellar
const ACCENT3 = '#e65100';   // burnt orange — retinal dystrophy; rod-cone
const ACCENT4 = '#4a148c';   // deep purple — centriole elongation mechanism; upstream biology
const ACCENT5 = '#b71c1c';   // deep red — ESRD; CKD progression; SRPS2B severity
const ACCENT6 = '#37474f';   // dark slate — molecular architecture; ARM/HEAT repeats
const ACCENT7 = '#f57f17';   // amber — misdiagnosis alerts; CPAP digenic; skeletal DDx
const ACCENT8 = '#01579b';   // dark blue — SRPS2B skeletal; centriole biology

const SEED = 379;
const _COHORT_SIZE = 40;

function KPI({ label, value, color }) {
  return (
    <div className="col-6 col-md-4 col-lg-2 mb-3">
      <div className="card h-100 shadow-sm text-center">
        <div className="card-body py-2 px-1">
          <div className="fw-bold fs-5" style={{ color }}>{value}</div>
          <div className="text-muted small">{label}</div>
        </div>
      </div>
    </div>
  );
}

function Alert({ color, children }) {
  return (
    <div className="alert mb-2" style={{ background: color + '18', borderLeft: `4px solid ${color}`, borderRadius: 6 }}>
      {children}
    </div>
  );
}

function Section({ title, color, children }) {
  return (
    <div className="mb-4">
      <h6 className="fw-bold mb-2" style={{ color, borderBottom: `2px solid ${color}`, paddingBottom: 4 }}>{title}</h6>
      {children}
    </div>
  );
}

function Badge({ text, color }) {
  return <span className="badge me-1" style={{ background: color, fontSize: '0.72em' }}>{text}</span>;
}

function Bar({ label, value, max, color }) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="fw-bold">{value}</span>
      </div>
      <div className="progress" style={{ height: 8, borderRadius: 4 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, background: color, borderRadius: 4 }} />
      </div>
    </div>
  );
}

export default function NPHP20Page() {
  const [tab, setTab]   = useState(0);
  const [ov, setOv]     = useState(null);
  const [bk, setBk]     = useState(null);
  const [df, setDf]     = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/nphp20/overview`).then(r => r.json()),
      fetch(`${API}/api/nphp20/breakdown`).then(r => r.json()),
      fetch(`${API}/api/nphp20/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBk(b); setDf(d); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center text-muted">Loading NPHP20 cohort…</div>;
  if (error)   return <div className="container py-5 text-danger">Error: {error}</div>;
  if (!ov)     return null;

  return (
    <div className="container-fluid py-3" style={{ maxWidth: 1200 }}>
      {/* Header */}
      <div className="mb-3 p-3 rounded" style={{ background: ACCENT + '12', border: `2px solid ${ACCENT}` }}>
        <div className="d-flex align-items-start gap-3 flex-wrap">
          <div style={{ flex: 1, minWidth: 260 }}>
            <h4 className="fw-bold mb-1" style={{ color: ACCENT }}>
              &#x1f9ec; Nephronophthisis Type 20 / Joubert Syndrome 31 (NPHP20/JBTS31) — CEP120 · Daughter Centriole Elongation Scaffold
            </h4>
            <div className="small text-muted mb-1">
              <strong>CEP120 (CCDC100)</strong> · 5q23.2 · ~1085 aa · ARM/HEAT repeats + coiled-coil ·
              daughter centriole elongation scaffold · JBTS31 ~55% · retinal ~30% ·
              SRPS2B ~4–10% (null×null) · ultra-rare (&lt;25 families 2026)
            </div>
            <div className="small">
              <Badge text="OMIM *613446" color={ACCENT} />
              <Badge text="#617761 JBTS31/NPHP20" color={ACCENT} />
              <Badge text="AR biallelic LOF" color={ACCENT6} />
              <Badge text="5q23.2" color={ACCENT8} />
              <Badge text="Centriole elongation" color={ACCENT4} />
              <Badge text="JBTS31 ~55%" color={ACCENT2} />
              <Badge text="Retinal ~30%" color={ACCENT3} />
              <Badge text="SRPS2B ~4–10%" color={ACCENT8} />
              <Badge text="CPAP/CENPJ co-sequence" color={ACCENT7} />
              <Badge text="Ultra-rare &lt;25 families" color={ACCENT5} />
            </div>
          </div>
          <div className="d-flex gap-2 flex-wrap">
            <span className="badge px-3 py-2" style={{ background: ACCENT, fontSize: '0.8em' }}>
              Ultra-rare &lt;25 families
            </span>
            <span className="badge px-3 py-2" style={{ background: ACCENT2, fontSize: '0.8em' }}>
              JBTS31 {ov.pct_jbts31_confirmed}%
            </span>
            <span className="badge px-3 py-2" style={{ background: ACCENT3, fontSize: '0.8em' }}>
              Retinal {ov.pct_retinal_dystrophy}%
            </span>
            <span className="badge px-3 py-2" style={{ background: ACCENT8, fontSize: '0.8em' }}>
              SRPS2B {ov.pct_srps2b_skeletal}%
            </span>
            <span className="badge px-3 py-2" style={{ background: ACCENT5, fontSize: '0.8em' }}>
              ESRD/Tx {ov.pct_esrd_or_transplant}%
            </span>
          </div>
        </div>
      </div>

      {/* Critical alerts */}
      <Alert color={ACCENT4}>
        <strong>&#x1f9ec; CEP120 IS THE ONLY NPHP SUBTYPE CAUSED BY DAUGHTER CENTRIOLE ELONGATION SCAFFOLD LOSS:</strong> CEP120
        (Centrosomal Protein 120 kDa) localises to the distal tip of daughter centrioles and is required
        for their full elongation. Without CEP120, daughter centrioles are shortened (~80% normal length)
        and cannot form competent basal bodies in subsequent cell cycles. This defect is <strong>upstream
        of ALL other NPHP mechanisms</strong>: distal appendage subtypes (NPHP15/CEP164, NPHP18/CEP83)
        assume a normal-length basal body is present; IFT transport subtypes (NPHP12/TTC21B, NPHP13/WDR19,
        NPHP19/IFT81) assume cilia have already formed; TZ scaffold subtypes (NPHP1, NPHP4, NPHP8) assume
        the basal body is docked. CEP120 defect prevents competent basal body formation entirely.
      </Alert>
      <Alert color={ACCENT8}>
        <strong>&#x1f9b4; SRPS2B — SHORT-RIB POLYDACTYLY SYNDROME TYPE 2B — BIALLELIC NULL CEP120 (~4–10%) — CHEST X-RAY MANDATORY:</strong> Severe
        biallelic null CEP120 alleles cause SRPS2B: extremely narrow bell-shaped thorax, short horizontal
        ribs, postaxial polydactyly, limb shortening, ± Joubert features. Potentially lethal in neonates
        (respiratory failure). Chest X-ray + skeletal survey MANDATORY in ALL newly diagnosed CEP120
        patients regardless of apparent phenotype. CEP120 SRPS2B is MORE SEVERE than Jeune/ATD4 (TTC21B)
        or ATD5 (WDR19) from IFT-A subtypes. SRPS2B with Joubert features distinguishes CEP120 from
        pure TZ and IFT subtypes that have NO skeletal involvement.
      </Alert>
      <Alert color={ACCENT2}>
        <strong>&#x1f9e0; JOUBERT SYNDROME 31 (JBTS31) — MOLAR TOOTH SIGN IN ~{ov.pct_jbts31_confirmed}% — BRAIN MRI MANDATORY:</strong> Biallelic
        CEP120 LOF causes Joubert syndrome (Molar Tooth Sign: cerebellar vermis hypoplasia + SCP
        elongation) in ~55% — similar penetrance to NPHP18/CEP83 (~55%). Oculomotor apraxia (OMA),
        neonatal hypotonia, episodic breathing irregularity (self-resolves 2–3yr), cerebellar ataxia,
        developmental delay. Brain MRI MANDATORY at diagnosis. Hypomorphic alleles: pure renal NPHP20
        without Joubert (~22%).
      </Alert>
      <Alert color={ACCENT7}>
        <strong>&#x26a0;&#xfe0f; CPAP/CENPJ OBLIGATE CO-SEQUENCING + DIGENIC DOCUMENTED — JBTS9/MCPH6 INTERACTION PAIR:</strong> CENPJ/CPAP
        is the DIRECT binding partner of CEP120&apos;s ARM repeat domain — CENPJ recruits CEP120 to daughter
        centrioles. CENPJ mutations cause JBTS9 (Joubert syndrome 9) and MCPH6 (primary microcephaly 6).
        CENPJ MUST always be co-sequenced alongside CEP120. Digenic CEP120 + CENPJ heterozygous variants
        documented. Single heterozygous CEP120 variant: always check CENPJ for a second hit.
        Key clinical DDx: JBTS with microcephaly → CENPJ first; JBTS without microcephaly + skeletal/
        SRPS2B → CEP120 first.
      </Alert>
      <Alert color={ACCENT}>
        <strong>&#x2705; RENAL TRANSPLANT CURATIVE · RETINAL/CEREBELLAR CELL-AUTONOMOUS · 5q23.2 NO ARM CONFUSION · WES MANDATORY:</strong> Donor
        kidney with functional CEP120 → normal daughter centriole elongation → competent basal bodies →
        intact tubular cilia → no TIN recurrence. Excellent renal outcomes. Retinal dystrophy (~{ov.pct_retinal_dystrophy}%) and
        JBTS31 cerebellar features are cell-autonomous — NOT corrected by renal transplant. Annual ERG
        mandatory. CEP120 at 5q23.2 has no chromosome arm confusion with other NPHP loci. However,
        NPHP1 MLPA (290kb) does NOT detect CEP120; WES is the only reliable diagnostic approach.
      </Alert>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {/* ── Tab 0: Overview ── */}
      {tab === 0 && (
        <div>
          <div className="row g-2 mb-3">
            <KPI label="Cohort (n)" value={ov.cohort_n} color={ACCENT} />
            <KPI label="Median GFR" value={`${ov.median_gfr} ml/min`} color={ACCENT} />
            <KPI label="Median Hb" value={`${ov.median_hb} g/dL`} color={ACCENT6} />
            <KPI label="Median age Dx" value={`${ov.median_age_renal_dx}yr`} color={ACCENT} />
            <KPI label="ESRD/Transplant" value={`${ov.pct_esrd_or_transplant}%`} color={ACCENT5} />
            <KPI label="JBTS31 (MTS)" value={`${ov.pct_jbts31_confirmed}%`} color={ACCENT2} />
            <KPI label="Retinal dystrophy" value={`${ov.pct_retinal_dystrophy}%`} color={ACCENT3} />
            <KPI label="SRPS2B skeletal" value={`${ov.pct_srps2b_skeletal}%`} color={ACCENT8} />
            <KPI label="Situs inversus" value="<2%" color={ACCENT} />
            <KPI label="NPHP1 MLPA misdiag" value={`${ov.pct_misdiagnosed_nphp1}%`} color={ACCENT7} />
            <KPI label="DA subtype misdiag" value={`${ov.pct_misdiagnosed_da}%`} color={ACCENT7} />
            <KPI label="JBTS-unknown misdiag" value={`${ov.pct_misdiagnosed_jbts_unknown}%`} color={ACCENT7} />
          </div>

          <div className="row g-3">
            <div className="col-md-6">
              <Section title="&#x1f9ec; CEP120 — Daughter Centriole Elongation Scaffold (5q23.2)" color={ACCENT6}>
                <div className="small text-muted mb-2">
                  CEP120 (Centrosomal Protein 120 kDa; ~1085 aa) is a centriole-associated scaffold
                  required for full elongation of daughter centrioles during mitosis and for ciliogenesis
                  initiation. CEP120 contains ARM/HEAT repeats and a central coiled-coil; interacts
                  directly with CPAP/CENPJ at the distal tip of daughter centrioles.
                </div>
                <table className="table table-sm table-bordered small">
                  <tbody>
                    <tr><td className="fw-bold">Gene</td><td>CEP120 (also CCDC100 — Coiled-Coil Domain Containing 100)</td></tr>
                    <tr><td className="fw-bold">Location</td><td>5q23.2 — unique arm; no confusion with other NPHP loci</td></tr>
                    <tr><td className="fw-bold">Protein</td><td>~1085 aa · ARM/HEAT repeats (aa 1–500) · coiled-coil (aa 500–850) · C-terminal regulatory (aa 850–1085)</td></tr>
                    <tr><td className="fw-bold">Mechanism</td><td>Daughter centriole elongation scaffold → competent basal body formation → cilia assembly</td></tr>
                    <tr><td className="fw-bold">OMIM Gene</td><td>*613446</td></tr>
                    <tr><td className="fw-bold">OMIM Disease</td><td>#617761 (JBTS31 / NPHP20); severe alleles → SRPS2B</td></tr>
                    <tr><td className="fw-bold">Key partners</td><td>CPAP/CENPJ (ARM domain; JBTS9; co-sequence mandatory); CEP135/CNTRL; CP110/CEP97; SAS-4; centrin</td></tr>
                    <tr><td className="fw-bold">Phenotypes</td><td>NPHP20 (renal TIN) + JBTS31 (~55%) + retinal (~30%) + SRPS2B (~4–10%); no situs; no CHF</td></tr>
                    <tr><td className="fw-bold">Prevalence</td><td>~1/800,000–2,000,000; &lt;25 published families (2026); ultra-rare</td></tr>
                    <tr><td className="fw-bold">Inheritance</td><td>Autosomal Recessive · biallelic LOF · digenic (CEP120 + CENPJ) documented</td></tr>
                  </tbody>
                </table>
              </Section>
            </div>
            <div className="col-md-6">
              <Section title="&#x1f6a8; NPHP20 / JBTS31 Hallmark Features" color={ACCENT}>
                {[
                  ['CEP120 is the ONLY NPHP subtype caused by daughter centriole elongation scaffold loss — upstream of ALL other NPHP mechanisms', ACCENT4,
                   'CEP120 loss → short daughter centrioles (~80% normal length) → incompetent basal bodies in subsequent cell cycles → primary cilia assembly fails. This defect is UPSTREAM of: distal appendage subtypes (NPHP15/CEP164, NPHP18/CEP83) which require a normal-length basal body to dock DA proteins; IFT transport subtypes (NPHP12/13/19) which require cilia to be forming; TZ subtypes (NPHP1/4/8) which require basal body docking. CEP120 is the most proximal ciliogenesis checkpoint defect among all described NPHP subtypes (2026)'],
                  ['SRPS2B — Short-Rib Polydactyly Syndrome type 2B — biallelic null CEP120 (~4–10%) — chest X-ray mandatory all patients', ACCENT8,
                   'Severe biallelic null alleles (both truncating) cause SRPS2B: bell-shaped narrow thorax, extremely short horizontal ribs, postaxial polydactyly, rhizomelic/mesomelic limb shortening, ± Joubert features if patient survives. Potentially neonatal lethal (respiratory failure from narrow thorax). Chest X-ray + skeletal survey MANDATORY in ALL newly diagnosed CEP120 patients regardless of phenotype — SRPS2B may be subclinical initially. More severe than Jeune/ATD (TTC21B, WDR19) which are caused by IFT-A retrograde defects'],
                  [`JBTS31 — Molar Tooth Sign in ~${ov.pct_jbts31_confirmed}% — brain MRI mandatory at diagnosis`, ACCENT2,
                   'Cerebellar vermis hypoplasia + SCP elongation (MTS) in ~55% of biallelic CEP120 cases — similar penetrance to NPHP18/CEP83 (~55%). Oculomotor apraxia (OMA), neonatal hypotonia, episodic breathing irregularity (self-resolves 2–3yr), cerebellar ataxia, developmental delay. Brain MRI MANDATORY at diagnosis. Allele class determines Joubert penetrance: null×null → JBTS31 certain; hypomorphic×hypomorphic → pure renal NPHP20 (no MTS, ~22% of cohort)'],
                  [`Retinal dystrophy ~${ov.pct_retinal_dystrophy}% — moderate penetrance — lower than NPHP19 (~50–60%); annual ERG mandatory`, ACCENT3,
                   'Rod-cone dystrophy in ~28–32% (ERG abnormal, progressive); LCA-like severe early in ~2% (null×null). Moderate retinal penetrance — lower than NPHP19/IFT81 (~50–60%) and NPHP15/CEP164 (~35–40%) but higher than NPHP17/MAPKBP1 (0%) and NPHP16/ANKS6 (0%). Retinal does NOT improve post-renal transplant (cell-autonomous photoreceptor connecting cilium defect). Annual ERG + fundoscopy from diagnosis mandatory regardless of initial ERG'],
                  ['CPAP/CENPJ obligate co-sequencing — direct ARM-domain binding partner — JBTS9/MCPH6 interaction pair', ACCENT7,
                   'CENPJ/CPAP is the direct binding partner of CEP120\'s N-terminal ARM repeat domain and is required for recruiting CEP120 to daughter centriole distal tips. CENPJ mutations cause JBTS9 and MCPH6 (primary microcephaly 6). CENPJ MUST be co-sequenced alongside CEP120. Digenic CEP120 (het) + CENPJ (het) combinations documented. Single heterozygous CEP120 variant: always check CENPJ for second hit. Key DDx: JBTS with microcephaly → CENPJ; JBTS without microcephaly + SRPS2B → CEP120'],
                  ['5q23.2 — No chromosome arm confusion — NPHP1 MLPA misses CEP120 — WES only reliable method', ACCENT,
                   'CEP120 at 5q23.2 does not share chromosome arm confusion with other NPHP genes (NPHP1 2q13; NPHP2 9q31; NPHP6 12q21; NPHP17 2q13.3; NPHP18 12q22; NPHP19 12q23.1). However, NPHP1 MLPA (290kb) does NOT detect CEP120. Ultra-rare status means CEP120 is often absent from targeted NPHP/JBTS gene panels. Gene-unknown Joubert with skeletal features + renal: ALWAYS check CEP120 and CENPJ on WES. ~28% of cohort initially labelled Joubert gene-unknown'],
                  ['NPHP20 renal — TIN + corticomedullary cysts + ESRD adolescent/young adult; transplant curative', ACCENT,
                   'Tubulointerstitial nephritis + corticomedullary cysts + concentrating defect. ESRD onset adolescent to young adult (~14–18yr typical; later with hypomorphic alleles). Bilateral small echogenic kidneys ± cysts on USS. Renal transplant CURATIVE: donor kidney has functional CEP120 → normal centriole elongation → competent basal bodies → intact tubular cilia → no TIN recurrence. Pre-emptive transplant preferred. Multi-disciplinary team essential for JBTS31 + SRPS2B cases'],
                  ['No situs inversus; no congenital hepatic fibrosis — CEP120 not expressed in nodal cilia or biliary epithelium', ACCENT,
                   'CEP120 is not expressed at functionally significant levels in nodal cilia (left-right axis determination) or biliary cholangiocytes, thus: NO situs inversus/totalis (contrast with NPHP2/INVS ~85% and NPHP16/ANKS6 ~20–30%). NO congenital hepatic fibrosis (contrast with NPHP11/TMEM67 ~50% and NPHP9/NEK8 ~40–55%). Kartagener/PCD is excluded (no bronchiectasis, no ciliary ultrastructure defect)'],
                ].map(([title, color, sub], i) => (
                  <div key={i} className="mb-2 p-2 rounded" style={{ background: color + '12', borderLeft: `3px solid ${color}` }}>
                    <div className="fw-bold small" style={{ color }}>{title}</div>
                    <div className="text-muted" style={{ fontSize: '0.75em' }}>{sub}</div>
                  </div>
                ))}
              </Section>
            </div>
          </div>

          {/* Sample patients */}
          <Section title={`&#x1f4cb; Sample Patients (first 8 of ${_COHORT_SIZE} · seed=${SEED})`} color={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-striped small">
                <thead>
                  <tr style={{ background: ACCENT + '22' }}>
                    <th>ID</th><th>Ethnicity</th><th>CKD Stage</th>
                    <th>Age Dx (yr)</th><th>GFR</th>
                    <th>JBTS31</th><th>Retinal</th><th>Skeletal</th><th>Misdiagnosis</th>
                  </tr>
                </thead>
                <tbody>
                  {ov.patients.map(p => (
                    <tr key={p.id}>
                      <td><span className="badge" style={{ background: ACCENT }}>{p.id}</span></td>
                      <td style={{ fontSize: '0.72em' }}>{p.ethnicity.split('(')[0].trim()}</td>
                      <td style={{ fontSize: '0.72em' }}>{p.ckd_stage.split('(')[0].trim()}</td>
                      <td>{p.age_renal_dx_yr}</td>
                      <td>{p.gfr_now_ml_min}</td>
                      <td style={{ fontSize: '0.72em' }}>{p.jbts31_status.split('(')[0].trim().slice(0, 22)}</td>
                      <td style={{ fontSize: '0.72em' }}>{p.retinal_status.split('(')[0].trim().slice(0, 20)}</td>
                      <td style={{ fontSize: '0.72em' }}>{p.skeletal_status.split('(')[0].trim().slice(0, 20)}</td>
                      <td style={{ fontSize: '0.72em' }}>{p.prior_misdiagnosis.split('(')[0].trim().slice(0, 24)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        </div>
      )}

      {/* ── Tab 1: Diagnostic Breakdown ── */}
      {tab === 1 && bk && (
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="CKD Stage / Renal Status Distribution" color={ACCENT}>
              {Object.entries(bk.ckd_stage_distribution).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
              ))}
            </Section>
            <Section title="JBTS31 Status (Joubert — Molar Tooth Sign)" color={ACCENT2}>
              {Object.entries(bk.jbts31_status).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT2} />
              ))}
            </Section>
            <Section title="Retinal Involvement" color={ACCENT3}>
              {Object.entries(bk.retinal_status).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT3} />
              ))}
            </Section>
            <Section title="Skeletal / SRPS2B Status" color={ACCENT8}>
              {Object.entries(bk.skeletal_status).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT8} />
              ))}
            </Section>
          </div>
          <div className="col-md-6">
            <Section title="Prior Misdiagnosis Distribution" color={ACCENT7}>
              {Object.entries(bk.prior_misdiagnosis).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT7} />
              ))}
            </Section>
            <Section title="First Presenting Symptom" color={ACCENT}>
              {Object.entries(bk.first_symptom_distribution).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
              ))}
            </Section>
            <Section title="Kidney USS Phenotype" color={ACCENT6}>
              {Object.entries(bk.kidney_phenotype).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT6} />
              ))}
            </Section>
            <Section title="Urine Osmolality Tiers (Tubular Concentrating Defect)" color={ACCENT}>
              {Object.entries(bk.urine_osmolality_tiers).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
              ))}
            </Section>
            <Section title="Ethnicity Distribution" color={ACCENT6}>
              {Object.entries(bk.ethnicity).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT6} />
              ))}
            </Section>
          </div>
        </div>
      )}

      {/* ── Tab 2: CEP120 Architecture & Centriole Biology ── */}
      {tab === 2 && df && (
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="&#x1f9ec; CEP120 Architecture, Centriole Elongation & Ciliogenesis" color={ACCENT4}>
              {df.genetic_architecture && Object.entries(df.genetic_architecture).filter(([k]) => k !== 'key_variants').map(([k, v]) => (
                <div key={k} className="mb-2 p-2 rounded" style={{ background: ACCENT + '08', borderLeft: `3px solid ${ACCENT4}` }}>
                  <div className="fw-bold small" style={{ color: ACCENT4 }}>{k.replace(/_/g, ' ')}</div>
                  <div className="text-muted small">{typeof v === 'string' ? v : JSON.stringify(v)}</div>
                </div>
              ))}
            </Section>
          </div>
          <div className="col-md-6">
            <Section title="&#x1f9ea; Key Variants (CEP120 NPHP20/JBTS31)" color={ACCENT}>
              {df.genetic_architecture && df.genetic_architecture.key_variants && df.genetic_architecture.key_variants.map((v, i) => (
                <div key={i} className="mb-2 p-2 rounded" style={{ background: ACCENT + '08', borderLeft: `3px solid ${ACCENT}` }}>
                  <div className="small text-muted">{v}</div>
                </div>
              ))}
            </Section>
            <Section title="&#x1f4ca; NPHP Subtype Comparison" color={ACCENT6}>
              {df.nphp_comparison && Object.entries(df.nphp_comparison).map(([k, v]) => (
                <div key={k} className="mb-2 p-2 rounded"
                  style={{
                    background: k.includes('★') ? ACCENT + '18' : ACCENT6 + '08',
                    borderLeft: `3px solid ${k.includes('★') ? ACCENT : ACCENT6}`
                  }}>
                  <div className="fw-bold small" style={{ color: k.includes('★') ? ACCENT : ACCENT6 }}>{k}</div>
                  <div className="text-muted small">{v}</div>
                </div>
              ))}
            </Section>
          </div>
          <div className="col-12">
            <Section title="&#x1fa7a; Differential Diagnosis Table" color={ACCENT7}>
              <div className="table-responsive">
                <table className="table table-sm table-bordered small">
                  <thead style={{ background: ACCENT7 + '18' }}>
                    <tr><th>Condition</th><th>Key Distinguishing Features from NPHP20/JBTS31</th></tr>
                  </thead>
                  <tbody>
                    {df.ddx_table && Object.entries(df.ddx_table).map(([k, v]) => (
                      <tr key={k}>
                        <td className="fw-bold" style={{ color: ACCENT7, minWidth: 200 }}>{k}</td>
                        <td className="small">{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Section>
          </div>
        </div>
      )}

      {/* ── Tab 3: Definitions ── */}
      {tab === 3 && df && (
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="Disease Definition" color={ACCENT}>
              <table className="table table-sm table-bordered small">
                <tbody>
                  {['disease','omim_gene','omim_disease','chromosome','inheritance','prevalence'].map(k => (
                    <tr key={k}>
                      <td className="fw-bold text-capitalize" style={{ width: 140 }}>{k.replace(/_/g,' ')}</td>
                      <td>{df[k]}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="Mechanism" color={ACCENT2}>
              <div className="small text-muted p-2 rounded" style={{ background: ACCENT + '06', lineHeight: 1.7 }}>
                {df.mechanism}
              </div>
            </Section>
            <Section title="&#x1f3e5; Treatment" color={ACCENT}>
              {df.treatment && Object.entries(df.treatment).map(([k, v]) => (
                <div key={k} className="mb-2 p-2 rounded" style={{ background: ACCENT + '08', borderLeft: `3px solid ${ACCENT}` }}>
                  <div className="fw-bold small" style={{ color: ACCENT }}>{k.replace(/_/g, ' ')}</div>
                  <div className="text-muted small">{v}</div>
                </div>
              ))}
            </Section>
          </div>
          <div className="col-md-6">
            <Section title="Key Clinical Features" color={ACCENT3}>
              {df.key_clinical_features && Object.entries(df.key_clinical_features).map(([k, v]) => (
                <div key={k} className="mb-2 p-2 rounded" style={{ background: ACCENT3 + '08', borderLeft: `3px solid ${ACCENT3}` }}>
                  <div className="fw-bold small" style={{ color: ACCENT3 }}>{k.replace(/_/g, ' ')}</div>
                  <div className="text-muted small">{v}</div>
                </div>
              ))}
            </Section>
            <Section title="Diagnostic Criteria" color={ACCENT6}>
              {df.diagnostic_criteria && Object.entries(df.diagnostic_criteria).map(([k, v]) => (
                <div key={k} className="mb-2 p-2 rounded" style={{ background: ACCENT6 + '08', borderLeft: `3px solid ${ACCENT6}` }}>
                  <div className="fw-bold small" style={{ color: ACCENT6 }}>{k.replace(/_/g, ' ')}</div>
                  <div className="text-muted small">{v}</div>
                </div>
              ))}
            </Section>
            <Section title="Prognosis" color={ACCENT}>
              <div className="small text-muted p-2 rounded" style={{ background: ACCENT + '06', lineHeight: 1.7 }}>
                {df.prognosis}
              </div>
            </Section>
            <Section title="Cohort Note" color={ACCENT6}>
              <div className="small text-muted p-2 rounded" style={{ background: ACCENT6 + '06', lineHeight: 1.6 }}>
                {df.cohort_note}
              </div>
            </Section>
          </div>
        </div>
      )}

      <div className="mt-4 pt-3 border-top">
        <Link href="/" className="btn btn-sm btn-outline-secondary me-2">&#x2190; Portal Home</Link>
        <Link href="/nphp19" className="btn btn-sm btn-outline-primary">&#x2190; NPHP19 (IFT81/IFT-B Anterograde)</Link>
      </div>
    </div>
  );
}
