'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'Genetics & Allele Spectrum', 'Definitions'];

// NPHP11 colour scheme — TMEM67/MKS3; hepatic fibrosis mahogany + Joubert teal + COACH purple
const ACCENT  = '#6d4c41';   // deep mahogany — TMEM67; TZ scaffold; hepatic-renal ciliopathy identity
const ACCENT2 = '#1b5e20';   // deep green — renal transplant curative / liver Tx before renal in CHF
const ACCENT3 = '#e65100';   // deep orange — CHF (congenital hepatic fibrosis; most frequent extra-renal)
const ACCENT4 = '#004d40';   // dark teal — Joubert Syndrome 6 (molar tooth; cerebellar vermis hypoplasia)
const ACCENT5 = '#4a148c';   // deep purple — COACH syndrome (cerebellar + oligophrenia + ataxia + coloboma + hepatic)
const ACCENT6 = '#1a237e';   // deep navy — epidemiology / rare; MKS3/TZ complex interactome
const ACCENT7 = '#880e4f';   // dark magenta — Alagille misdiagnosis (liver disease dominates → JAG1 first)
const ACCENT8 = '#37474f';   // dark slate — MKS/NPHP/JBTS transition-zone complex scaffold

const SEED = 361;
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

export default function NPHP11Page() {
  const [tab, setTab]   = useState(0);
  const [ov, setOv]     = useState(null);
  const [bk, setBk]     = useState(null);
  const [df, setDf]     = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/nphp11/overview`).then(r => r.json()),
      fetch(`${API}/api/nphp11/breakdown`).then(r => r.json()),
      fetch(`${API}/api/nphp11/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBk(b); setDf(d); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center text-muted">Loading NPHP11 cohort…</div>;
  if (error)   return <div className="container py-5 text-danger">Error: {error}</div>;
  if (!ov)     return null;

  return (
    <div className="container-fluid py-3" style={{ maxWidth: 1200 }}>
      {/* Header */}
      <div className="mb-3 p-3 rounded" style={{ background: ACCENT + '12', border: `2px solid ${ACCENT}` }}>
        <div className="d-flex align-items-start gap-3 flex-wrap">
          <div style={{ flex: 1, minWidth: 260 }}>
            <h4 className="fw-bold mb-1" style={{ color: ACCENT }}>
              &#x1f9ec; Nephronophthisis Type 11 / Joubert Syndrome 6 / COACH Syndrome (NPHP11 / JBTS6 / MKS3)
            </h4>
            <div className="small text-muted mb-1">
              <strong>TMEM67</strong> · 8q22.1 · 995 aa · Meckelin · Type I transmembrane glycoprotein ·
              MKS/NPHP/JBTS transition-zone scaffold · CHF most frequent extra-renal feature ·
              COACH syndrome gene (most common, &gt;80%)
            </div>
            <div className="small">
              <Badge text="OMIM *609884" color={ACCENT} />
              <Badge text="#613550 NPHP11" color={ACCENT} />
              <Badge text="#610688 JBTS6" color={ACCENT4} />
              <Badge text="#607361 MKS3" color={ACCENT6} />
              <Badge text="AR biallelic LOF" color={ACCENT6} />
              <Badge text="8q22.1" color={ACCENT8} />
              <Badge text="TZ scaffold" color={ACCENT4} />
            </div>
          </div>
          <div className="d-flex gap-2 flex-wrap">
            <span className="badge px-3 py-2" style={{ background: ACCENT, fontSize: '0.8em' }}>
              ESRD median ~18–22yr
            </span>
            <span className="badge px-3 py-2" style={{ background: ACCENT3, fontSize: '0.8em' }}>
              CHF {ov.pct_chf}% (most frequent)
            </span>
            <span className="badge px-3 py-2" style={{ background: ACCENT4, fontSize: '0.8em' }}>
              Joubert {ov.pct_joubert}%
            </span>
            <span className="badge px-3 py-2" style={{ background: ACCENT5, fontSize: '0.8em' }}>
              COACH {ov.pct_coach}%
            </span>
          </div>
        </div>
      </div>

      {/* Critical alerts */}
      <Alert color={ACCENT7}>
        <strong>&#x26a0;&#xfe0f; MOST COMMON MISDIAGNOSIS — Alagille Syndrome / biliary liver disease:</strong> Hepatic
        fibrosis (CHF) dominates presentation → JAG1/NOTCH2 panel → TMEM67 missed → renal monitoring omitted → ESRD
        at unscheduled presentation. KEY DDx: Alagille = bile duct PAUCITY (cholestatic) + butterfly vertebrae + pulmonary
        stenosis; NPHP11 = ductal plate MALFORMATION (fibrotic/portal HTN) + TIN cysts. TMEM67 MUST be on ALL
        NPHP, JBTS, MKS, and CHF ciliopathy panels.
      </Alert>
      <Alert color={ACCENT3}>
        <strong>&#x1fac0; CHF {ov.pct_chf}% — CONGENITAL HEPATIC FIBROSIS (most frequent extra-renal feature):</strong> Ductal
        plate malformation → periportal fibrosis → portal hypertension → oesophageal varices. Annual APRI + USS liver
        mandatory. Liver transplant may precede renal transplant in ~10–15% (severe portal HTN / hepatopulmonary
        syndrome). Renal disease continues independently after liver Tx.
      </Alert>
      <Alert color={ACCENT4}>
        <strong>&#x1f9e0; JOUBERT SYNDROME 6 ({ov.pct_joubert}%) — MOLAR TOOTH SIGN on MRI:</strong> TMEM67 is JBTS6.
        Cerebellar vermis hypoplasia + superior cerebellar peduncle elongation = molar tooth sign. Brain MRI
        MANDATORY at NPHP11 diagnosis. Oculomotor apraxia; ataxia; variable intellectual disability. COACH is a
        JBTS6 allele subset: cerebellar + oligophrenia + ataxia + coloboma + hepatic fibrosis.
      </Alert>
      <Alert color={ACCENT5}>
        <strong>&#x1f9a0; COACH SYNDROME {ov.pct_coach}% — TMEM67 most common gene (&gt;80% of COACH):</strong> C = Cerebellar
        vermis hypoplasia; O = Oligophrenia (intellectual disability); A = Ataxia; C = Coloboma (retinochoroidal);
        H = Hepatic fibrosis. CC2D2A (second most common COACH gene). Coloboma ({ov.pct_coloboma}%) ≠ retinal dystrophy —
        structural eye defect; low-vision rehabilitation; does NOT improve post-transplant.
      </Alert>
      <Alert color={ACCENT2}>
        <strong>&#x2705; RENAL TRANSPLANT = CURATIVE:</strong> Cell-autonomous TZ defect. NO recurrence. Excellent
        graft outcomes. CHF does NOT improve after renal Tx (independent liver progression). Liver Tx
        addresses hepatic component. Both may be required in complex cases; coordinated multi-organ planning.
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
            <KPI label="Median Hb" value={`${ov.median_hb} g/dL`} color="#5d4037" />
            <KPI label="Median age Dx" value={`${ov.median_age_renal_dx}yr`} color={ACCENT} />
            <KPI label="ESRD/Transplant" value={`${ov.pct_esrd_or_transplant}%`} color={ACCENT} />
            <KPI label="CHF" value={`${ov.pct_chf}%`} color={ACCENT3} />
            <KPI label="Joubert (JBTS6)" value={`${ov.pct_joubert}%`} color={ACCENT4} />
            <KPI label="COACH" value={`${ov.pct_coach}%`} color={ACCENT5} />
            <KPI label="Coloboma" value={`${ov.pct_coloboma}%`} color={ACCENT5} />
            <KPI label="Polyuria first" value={`${ov.pct_polyuria_first_symptom}%`} color={ACCENT8} />
            <KPI label="Alagille misdiag" value={`${ov.pct_misdiagnosed_as_alagille}%`} color={ACCENT7} />
            <KPI label="Retinal dystrophy" value="0%" color={ACCENT2} />
          </div>

          <div className="row g-3">
            <div className="col-md-6">
              <Section title="&#x1f9ec; TMEM67 — MKS/NPHP/JBTS Transition-Zone Scaffold" color={ACCENT8}>
                <div className="small text-muted mb-2">
                  TMEM67 (Meckelin; 995 aa) is a type I transmembrane glycoprotein localised to the ciliary
                  transition zone (TZ). It is a core component of the MKS module (with CC2D2A, B9D1, B9D2,
                  TMEM216) and directly interacts with the NPHP module (NPHP1, NPHP4) and RPGRIP1L (NPHP8/JBTS7).
                  LOF causes the broadest TZ-disease spectrum: MKS3 (lethal) → JBTS6/COACH → NPHP11.
                </div>
                <table className="table table-sm table-bordered small">
                  <tbody>
                    <tr><td className="fw-bold">Gene</td><td>TMEM67 (Transmembrane Protein 67 / Meckelin / MKS3)</td></tr>
                    <tr><td className="fw-bold">Location</td><td>8q22.1</td></tr>
                    <tr><td className="fw-bold">Protein</td><td>995 aa · ~114 kDa · type I transmembrane glycoprotein</td></tr>
                    <tr><td className="fw-bold">Domains</td><td>N-terminal extracellular (EGF-like repeats) + transmembrane + C-terminal cytoplasmic (CC2D2A/RPGRIP1L binding)</td></tr>
                    <tr><td className="fw-bold">OMIM Gene</td><td>*609884</td></tr>
                    <tr><td className="fw-bold">OMIM Disease</td><td>#613550 (NPHP11) · #610688 (JBTS6) · #607361 (MKS3)</td></tr>
                    <tr><td className="fw-bold">Key interactors</td><td>CC2D2A (JBTS9/MKS6), RPGRIP1L (NPHP8/JBTS7), CEP290 (NPHP6), NPHP1, NPHP4, B9D1, B9D2</td></tr>
                    <tr><td className="fw-bold">Mechanism</td><td>TZ scaffold disruption → diffusion barrier collapse → ectopic ciliary signalling proteins → TIN + Joubert + CHF</td></tr>
                    <tr><td className="fw-bold">Prevalence</td><td>~1/300,000–700,000 (combined TMEM67 spectrum)</td></tr>
                    <tr><td className="fw-bold">Inheritance</td><td>Autosomal Recessive · biallelic LOF</td></tr>
                  </tbody>
                </table>
              </Section>
            </div>
            <div className="col-md-6">
              <Section title="&#x1f6a8; NPHP11 Hallmark Features" color={ACCENT}>
                {[
                  [`CHF ${ov.pct_chf}% (congenital hepatic fibrosis)`, ACCENT3,
                   'Ductal plate malformation → periportal fibrosis → portal HTN → varices; most frequent extra-renal feature; liver Tx may precede renal Tx in ~10–15%; ANNUAL APRI + USS'],
                  [`Joubert ${ov.pct_joubert}% (JBTS6 — molar tooth sign)`, ACCENT4,
                   'TMEM67 = JBTS6; cerebellar vermis hypoplasia + SCP elongation → molar tooth on axial MRI; oculomotor apraxia; ataxia; brain MRI MANDATORY'],
                  [`COACH ${ov.pct_coach}% (most common COACH gene >80%)`, ACCENT5,
                   'Cerebellar + Oligophrenia + Ataxia + Coloboma + Hepatic fibrosis — TMEM67 causes >80% of COACH; CC2D2A second; all five features required for COACH label'],
                  [`Coloboma ${ov.pct_coloboma}% (retinochoroidal; NOT retinal dystrophy)`, ACCENT5,
                   'Structural coloboma of iris, retina, optic nerve; NOT rod-cone degeneration; low-vision rehabilitation; does not improve post-Tx; ophthalmology from diagnosis'],
                  ['NO retinal dystrophy (0%)', ACCENT2,
                   'TMEM67 not expressed in photoreceptors; no ERG abnormality; coloboma (structural) ≠ retinal dystrophy (degenerative) — critical distinction'],
                  ['NO pancreatic ductal ectasia (0%)', ACCENT2,
                   'TMEM67 absent from pancreatic ducts — unlike NPHP9 (NEK8); no exocrine dysfunction'],
                  [`ESRD median ~${ov.median_age_renal_dx}yr (cohort, range 5–35yr)`, ACCENT,
                   'Small echogenic kidneys; corticomedullary cysts; TIN — later ESRD than most NPHP subtypes; concentrating defect precedes GFR decline'],
                  ['Renal transplant CURATIVE', ACCENT2,
                   'Cell-autonomous TZ defect; NO recurrence; CHF independent — may need liver Tx separately; coordinated hepato-renal planning in severe cases'],
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
                    <th>CHF</th><th>Joubert</th><th>COACH</th>
                    <th>First Symptom</th>
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
                      <td>
                        {p.hepatic_fibrosis
                          ? <span className="badge" style={{ background: ACCENT3 }}>CHF ✓</span>
                          : <span className="text-muted small">—</span>}
                      </td>
                      <td>
                        {p.joubert_syndrome
                          ? <span className="badge" style={{ background: ACCENT4 }}>JBTS ✓</span>
                          : <span className="text-muted small">—</span>}
                      </td>
                      <td>
                        {p.coach_syndrome
                          ? <span className="badge" style={{ background: ACCENT5 }}>COACH ✓</span>
                          : <span className="text-muted small">—</span>}
                      </td>
                      <td style={{ fontSize: '0.72em' }}>{p.first_symptom.split('(')[0].trim().slice(0, 30)}</td>
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
            <Section title="Kidney Phenotype Distribution" color={ACCENT}>
              {Object.entries(bk.kidney_phenotype_distribution).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
              ))}
            </Section>
            <Section title="CHF Status Distribution" color={ACCENT3}>
              {Object.entries(bk.chf_status_distribution).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT3} />
              ))}
            </Section>
            <Section title="Joubert Status Distribution (JBTS6)" color={ACCENT4}>
              {Object.entries(bk.joubert_status_distribution).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT4} />
              ))}
            </Section>
            <Section title="COACH Syndrome Distribution" color={ACCENT5}>
              {Object.entries(bk.coach_status_distribution).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT5} />
              ))}
            </Section>
          </div>
          <div className="col-md-6">
            <Section title="CKD Stage Distribution" color={ACCENT6}>
              {Object.entries(bk.ckd_stage_current).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT6} />
              ))}
            </Section>
            <Section title="RRT / Transplant Status" color={ACCENT2}>
              {Object.entries(bk.rrt_transplant_status).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT2} />
              ))}
            </Section>
            <Section title="Prior Misdiagnosis (most common: Alagille / biliary)" color={ACCENT7}>
              {Object.entries(bk.prior_misdiagnosis).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT7} />
              ))}
            </Section>
            <Section title="Age at Renal Diagnosis — Tiers" color={ACCENT}>
              {Object.entries(bk.age_at_renal_dx_tiers).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
              ))}
            </Section>
          </div>
          <div className="col-12">
            <div className="row g-3">
              <div className="col-md-4">
                <Section title="Ethnicity Distribution" color={ACCENT8}>
                  {Object.entries(bk.ethnicity).map(([k, v]) => (
                    <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT8} />
                  ))}
                </Section>
              </div>
              <div className="col-md-4">
                <Section title="Urine Osmolality Tiers (Tubular Defect)" color={ACCENT}>
                  {Object.entries(bk.urine_osmolality_tiers).map(([k, v]) => (
                    <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
                  ))}
                </Section>
              </div>
              <div className="col-md-4">
                <Section title="GFR Slope Tiers (Progression Rate)" color={ACCENT6}>
                  {Object.entries(bk.gfr_slope_tiers).map(([k, v]) => (
                    <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT6} />
                  ))}
                </Section>
              </div>
            </div>
          </div>
          <div className="col-12">
            <Section title="First Presenting Symptom" color={ACCENT}>
              <div className="row g-2">
                {Object.entries(bk.first_symptom_distribution).map(([k, v]) => (
                  <div key={k} className="col-md-6">
                    <Bar label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
                  </div>
                ))}
              </div>
            </Section>
          </div>
        </div>
      )}

      {/* ── Tab 2: Genetics & Allele Spectrum ── */}
      {tab === 2 && df && (
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="&#x1f9ec; Gene Structure & TZ Scaffold Architecture" color={ACCENT8}>
              {df.genetic_architecture && Object.entries(df.genetic_architecture).map(([k, v]) => (
                <div key={k} className="mb-2 p-2 rounded" style={{ background: ACCENT + '08', borderLeft: `3px solid ${ACCENT}` }}>
                  <div className="fw-bold small" style={{ color: ACCENT }}>{k.replace(/_/g, ' ')}</div>
                  <div className="text-muted small">{v}</div>
                </div>
              ))}
            </Section>
          </div>
          <div className="col-md-6">
            <Section title="&#x1f9ea; Key Variants (TMEM67 NPHP11 / JBTS6 / MKS3)" color={ACCENT3}>
              {df.key_variants && df.key_variants.map((v, i) => (
                <div key={i} className="mb-2 p-2 rounded" style={{ background: ACCENT3 + '08', borderLeft: `3px solid ${ACCENT3}` }}>
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
                    <tr><th>Condition</th><th>Key Distinguishing Features from NPHP11</th></tr>
                  </thead>
                  <tbody>
                    {df.ddx_table && Object.entries(df.ddx_table).map(([k, v]) => (
                      <tr key={k}>
                        <td className="fw-bold" style={{ color: ACCENT7, minWidth: 160 }}>{k}</td>
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
            <Section title="Mechanism" color={ACCENT8}>
              <div className="small text-muted p-2 rounded" style={{ background: ACCENT + '06', lineHeight: 1.7 }}>
                {df.mechanism}
              </div>
            </Section>
            <Section title="&#x1f3e5; Treatment" color={ACCENT2}>
              {df.treatment && Object.entries(df.treatment).map(([k, v]) => (
                <div key={k} className="mb-2 p-2 rounded" style={{ background: ACCENT2 + '08', borderLeft: `3px solid ${ACCENT2}` }}>
                  <div className="fw-bold small" style={{ color: ACCENT2 }}>{k.replace(/_/g, ' ')}</div>
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
            <Section title="Diagnostic Criteria" color={ACCENT4}>
              {df.diagnostic_criteria && Object.entries(df.diagnostic_criteria).map(([k, v]) => (
                <div key={k} className="mb-2 p-2 rounded" style={{ background: ACCENT4 + '08', borderLeft: `3px solid ${ACCENT4}` }}>
                  <div className="fw-bold small" style={{ color: ACCENT4 }}>{k.replace(/_/g, ' ')}</div>
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
        <Link href="/nphp10" className="btn btn-sm btn-outline-primary">&#x2190; NPHP10 (SDCCAG8)</Link>
      </div>
    </div>
  );
}
