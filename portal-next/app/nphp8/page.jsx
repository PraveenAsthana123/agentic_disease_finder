'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'Genetics & Allele Spectrum', 'Definitions'];

// NPHP8 colour scheme — deep-indigo-amber-teal-red (RPGRIP1L; multi-spectrum; TZ-NPHP4-module)
const ACCENT  = '#1a237e';   // deep indigo — RPGRIP1L TZ scaffold; broadest NPHP spectrum
const ACCENT2 = '#1b5e20';   // deep green — transplant curative / excellent renal outcome
const ACCENT3 = '#e65100';   // deep orange — JBTS7 allele class / Molar Tooth Sign
const ACCENT4 = '#880e4f';   // dark magenta — retinal dystrophy / JBTS7 visual loss
const ACCENT5 = '#006064';   // deep teal — NPHP8 pure renal component
const ACCENT6 = '#4a148c';   // deep purple — epidemiology / rare spectrum
const ACCENT7 = '#bf360c';   // deep burnt orange — ADPKD misdiagnosis (most common error)
const ACCENT8 = '#263238';   // dark blue-grey — digenic / NPHP4 interaction

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

export default function NPHP8Page() {
  const [tab, setTab]   = useState(0);
  const [ov, setOv]     = useState(null);
  const [bk, setBk]     = useState(null);
  const [df, setDf]     = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/nphp8/overview`).then(r => r.json()),
      fetch(`${API}/api/nphp8/breakdown`).then(r => r.json()),
      fetch(`${API}/api/nphp8/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBk(b); setDf(d); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border" style={{ color: ACCENT }} /></div>;
  if (error)   return <div className="container py-4"><div className="alert alert-danger">Error: {error}</div></div>;
  if (!ov)     return null;

  return (
    <div className="container-fluid py-3 px-3" style={{ maxWidth: 1100 }}>
      {/* Header */}
      <div className="mb-3">
        <h4 className="fw-bold mb-1" style={{ color: ACCENT }}>
          Nephronophthisis Type 8 / Joubert Syndrome Type 7
        </h4>
        <div className="text-muted small mb-1">
          <Badge text="RPGRIP1L" color={ACCENT} />
          <Badge text="16q12.2" color={ACCENT3} />
          <Badge text="OMIM Gene *610937" color={ACCENT5} />
          <Badge text="NPHP8 #613237" color={ACCENT} />
          <Badge text="JBTS7 #611560" color={ACCENT3} />
          <Badge text="MKS5 #611561" color={ACCENT4} />
          <Badge text="AR Biallelic LOF" color={ACCENT8} />
        </div>
        <div className="small text-muted">
          TZ scaffold · NPHP1-4-8 ternary module · Coiled-coil + C2 + RPGR-interacting domain ·
          Broadest allele-phenotype spectrum in NPHP after CEP290 · n={_COHORT_SIZE} synthetic cohort
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button className={`nav-link${tab === i ? ' active fw-bold' : ''}`}
              style={tab === i ? { color: ACCENT, borderBottom: `2px solid ${ACCENT}` } : {}}
              onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ── Tab 0: Overview ── */}
      {tab === 0 && (
        <div>
          <Alert color={ACCENT3}>
            <strong>JBTS7 / MKS5 / NPHP8 — Broadest Allele-Phenotype Spectrum in NPHP after CEP290.</strong>{' '}
            Brain MRI is MANDATORY in every RPGRIP1L patient — Molar Tooth Sign stratifies JBTS7 (MTS present)
            vs pure NPHP8 (MTS absent). NPHP4 genotype must ALWAYS be checked — monoallelic RPGRIP1L +
            monoallelic NPHP4 = digenic Joubert Syndrome.
          </Alert>
          <Alert color={ACCENT}>
            <strong>NPHP1-4-8 Ternary TZ Module:</strong>{' '}
            RPGRIP1L (NPHP8) directly binds NPHP4 which binds NPHP1 — together forming the
            transition zone diffusion barrier Y-link scaffold. LOF → TZ gate collapse → Hh/Wnt/PDGF
            signalling failure → tubulointerstitial nephritis → ESRD.
          </Alert>

          <div className="row g-2 mb-4">
            <KPI label="Cohort N" value={ov.cohort_n} color={ACCENT} />
            <KPI label="Gene" value={ov.gene} color={ACCENT} />
            <KPI label="Chr" value={ov.chromosome} color={ACCENT3} />
            <KPI label="Median GFR" value={`${ov.median_gfr} ml/min`} color={ACCENT5} />
            <KPI label="Median Hb" value={`${ov.median_hb} g/dL`} color={ACCENT4} />
            <KPI label="Median Age Renal Dx" value={`${ov.median_age_renal_dx} yr`} color={ACCENT3} />
            <KPI label="% ESRD/Transplant" value={`${ov.pct_esrd_or_transplant}%`} color={ACCENT4} />
            <KPI label="% Polyuria First" value={`${ov.pct_polyuria_first_symptom}%`} color={ACCENT5} />
            <KPI label="% Retinal Dystrophy" value={`${ov.pct_retinal_dystrophy}%`} color={ACCENT4} />
            <KPI label="% Molar Tooth (JBTS7)" value={`${ov.pct_molar_tooth_sign}%`} color={ACCENT3} />
            <KPI label="% Hepatic Fibrosis" value={`${ov.pct_hepatic_fibrosis}%`} color={ACCENT7} />
            <KPI label="% JBTS7 Allele Class" value={`${ov.pct_jbts7_allele_class}%`} color={ACCENT3} />
          </div>

          <Section title="Allele-Phenotype Spectrum Overview" color={ACCENT3}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead><tr style={{ background: ACCENT3 + '18' }}>
                  <th>Allele Class</th><th>Phenotype</th><th>MTS</th><th>Retinal</th><th>Hepatic</th><th>ESRD</th>
                </tr></thead>
                <tbody>
                  <tr><td><strong>Biallelic null</strong></td><td>MKS5 (lethal)</td>
                    <td><span className="badge bg-danger">Yes + encephalocele</span></td>
                    <td><span className="badge bg-danger">Severe</span></td>
                    <td><span className="badge bg-danger">Severe</span></td><td>—</td></tr>
                  <tr><td><strong>Truncating + strong missense</strong></td><td>JBTS7</td>
                    <td><span className="badge" style={{ background: ACCENT3 }}>Yes (SCP elongation)</span></td>
                    <td><span className="badge" style={{ background: ACCENT4 }}>25–35%</span></td>
                    <td><span className="badge" style={{ background: ACCENT7 }}>15–20%</span></td>
                    <td><span className="badge" style={{ background: ACCENT5 }}>~12–15yr</span></td></tr>
                  <tr><td><strong>Truncating + mild missense</strong></td><td>NPHP8 (pure renal)</td>
                    <td><span className="badge bg-success">No</span></td>
                    <td><span className="badge bg-success">No</span></td>
                    <td><span className="badge bg-success">No</span></td>
                    <td><span className="badge" style={{ background: ACCENT5 }}>~15–18yr</span></td></tr>
                  <tr><td><strong>Biallelic mild missense</strong></td><td>NPHP8 (late renal)</td>
                    <td><span className="badge bg-success">No</span></td>
                    <td><span className="badge bg-success">No</span></td>
                    <td><span className="badge bg-success">No</span></td>
                    <td><span className="badge bg-secondary">~20–30yr</span></td></tr>
                  <tr><td><strong>RPGRIP1L mono + NPHP4 mono</strong></td><td>Digenic JBTS</td>
                    <td><span className="badge" style={{ background: ACCENT8 }}>Yes (digenic)</span></td>
                    <td><span className="badge" style={{ background: ACCENT4 }}>Variable</span></td>
                    <td><span className="badge bg-secondary">Rare</span></td>
                    <td><span className="badge" style={{ background: ACCENT5 }}>Variable</span></td></tr>
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="Critical Diagnostic Rules" color={ACCENT}>
            <div className="row g-2">
              {[
                ['Brain MRI MANDATORY', 'Every RPGRIP1L patient — Molar Tooth Sign stratifies JBTS7 (MTS+) vs pure NPHP8 (MTS−); changes surveillance, prognosis, genetic counselling', ACCENT3],
                ['NPHP4 Always Check', 'RPGRIP1L monoallelic + NPHP4 monoallelic → digenic JBTS; panels MUST include NPHP4; never assume monoallelic RPGRIP1L is benign', ACCENT8],
                ['ADPKD Most Common Misdiagnosis', 'Cysts on USS + family history assumed dominant → PKD1/PKD2 tested first; NPHP8 is AR — renal genetics referral mandatory', ACCENT7],
                ['Retinal Panel Omission', 'JBTS7 retinal resembles LCA — RPGRIP1L must be included on all LCA/SLS/NPHP panels; retinal-only workup misses renal disease', ACCENT4],
              ].map(([title, text, col]) => (
                <div key={title} className="col-md-6">
                  <div className="card shadow-sm h-100" style={{ borderLeft: `3px solid ${col}` }}>
                    <div className="card-body py-2 px-3">
                      <div className="fw-bold small" style={{ color: col }}>{title}</div>
                      <div className="small text-muted">{text}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Section>

          <Section title="Sample Patients (first 8)" color={ACCENT5}>
            <div className="table-responsive">
              <table className="table table-sm table-hover small">
                <thead><tr style={{ background: ACCENT5 + '18' }}>
                  <th>ID</th><th>Allele Class</th><th>Ethnicity</th>
                  <th>CKD Stage</th><th>GFR</th><th>Age Dx</th>
                  <th>Retinal</th><th>MTS</th><th>Hepatic</th>
                </tr></thead>
                <tbody>
                  {(ov.patients || []).map(p => (
                    <tr key={p.id}>
                      <td className="fw-bold">{p.id}</td>
                      <td>{(p.allele_class || '').split('(')[0].trim()}</td>
                      <td>{(p.ethnicity || '').split('(')[0].trim()}</td>
                      <td>{(p.ckd_stage || '').split('(')[0].trim()}</td>
                      <td>{p.gfr_now_ml_min}</td>
                      <td>{p.age_renal_dx_yr} yr</td>
                      <td>{p.retinal_dystrophy ? <span className="badge" style={{ background: ACCENT4 }}>Yes</span> : <span className="badge bg-success">No</span>}</td>
                      <td>{p.molar_tooth_sign ? <span className="badge" style={{ background: ACCENT3 }}>MTS</span> : <span className="badge bg-success">No</span>}</td>
                      <td>{p.hepatic_fibrosis ? <span className="badge" style={{ background: ACCENT7 }}>CHF</span> : <span className="badge bg-success">No</span>}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <div className="d-flex justify-content-between mt-3">
            <Link href="/nphp7" style={{ color: ACCENT }}>← NPHP7 / GLIS2 (Pure Renal)</Link>
            <Link href="/joubert" style={{ color: ACCENT3 }}>Joubert Syndrome (Multi-gene) →</Link>
          </div>
        </div>
      )}

      {/* ── Tab 1: Diagnostic Breakdown ── */}
      {tab === 1 && bk && (
        <div>
          <div className="row">
            <div className="col-md-6">
              <Section title="Allele Class Distribution" color={ACCENT3}>
                {Object.entries(bk.allele_class_distribution || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT3} />
                ))}
              </Section>
              <Section title="CKD Stage (current)" color={ACCENT5}>
                {Object.entries(bk.ckd_stage_current || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT5} />
                ))}
              </Section>
              <Section title="Age at Renal Diagnosis" color={ACCENT3}>
                {Object.entries(bk.age_at_renal_dx_tiers || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT3} />
                ))}
              </Section>
              <Section title="Retinal Status Distribution" color={ACCENT4}>
                {Object.entries(bk.retinal_status_distribution || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT4} />
                ))}
              </Section>
            </div>
            <div className="col-md-6">
              <Section title="Prior Misdiagnosis" color={ACCENT7}>
                {Object.entries(bk.prior_misdiagnosis || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT7} />
                ))}
              </Section>
              <Section title="Ethnicity Distribution" color={ACCENT8}>
                {Object.entries(bk.ethnicity || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT8} />
                ))}
              </Section>
              <Section title="Hepatic Status (JBTS7 allele)" color={ACCENT7}>
                {Object.entries(bk.hepatic_status_distribution || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT7} />
                ))}
              </Section>
              <Section title="MTS Status (Brain MRI)" color={ACCENT3}>
                {Object.entries(bk.mts_status_distribution || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT3} />
                ))}
              </Section>
            </div>
          </div>
          <div className="row mt-2">
            <div className="col-md-6">
              <Section title="RRT / Transplant Status" color={ACCENT2}>
                {Object.entries(bk.rrt_transplant_status || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT2} />
                ))}
              </Section>
              <Section title="Urine Osmolality Tiers" color={ACCENT5}>
                {Object.entries(bk.urine_osmolality_tiers || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT5} />
                ))}
              </Section>
            </div>
            <div className="col-md-6">
              <Section title="First Symptom / Presentation" color={ACCENT}>
                {Object.entries(bk.first_symptom_distribution || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
                ))}
              </Section>
              <Section title="GFR Decline Slope" color={ACCENT4}>
                {Object.entries(bk.gfr_slope_tiers || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT4} />
                ))}
              </Section>
            </div>
          </div>
        </div>
      )}

      {/* ── Tab 2: Genetics & Allele Spectrum ── */}
      {tab === 2 && df && (
        <div>
          <Section title="Molecular Genetics" color={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <tbody>
                  {df.genetic_architecture && Object.entries(df.genetic_architecture).map(([k, v]) => (
                    <tr key={k}>
                      <td className="fw-bold text-nowrap" style={{ color: ACCENT, width: '22%' }}>{k.replace(/_/g, ' ')}</td>
                      <td>{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="Key Published Variants" color={ACCENT3}>
            <ul className="small">
              {(df.key_variants || []).map((v, i) => <li key={i}>{v}</li>)}
            </ul>
          </Section>

          <Section title="Digenic JBTS — RPGRIP1L + NPHP4" color={ACCENT8}>
            <Alert color={ACCENT8}>
              <strong>Critical digenic interaction:</strong> A single pathogenic allele in RPGRIP1L
              + a single pathogenic allele in NPHP4 (1p36) → Joubert Syndrome phenotype (Molar Tooth Sign).
              This triallelic / digenic model is unique to the NPHP1-4-8 TZ module. Every RPGRIP1L
              patient must have full NPHP4 sequencing — do not stop at "monoallelic" RPGRIP1L.
            </Alert>
          </Section>

          <Section title="NPHP Subtype Comparison" color={ACCENT5}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead><tr style={{ background: ACCENT5 + '18' }}>
                  <th>Subtype</th><th>Key Features</th>
                </tr></thead>
                <tbody>
                  {df.nphp_comparison && Object.entries(df.nphp_comparison).map(([k, v]) => (
                    <tr key={k} style={k.includes('★') ? { background: ACCENT + '10', fontWeight: 'bold' } : {}}>
                      <td style={{ color: ACCENT, whiteSpace: 'nowrap' }}>{k}</td>
                      <td>{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="Differential Diagnosis" color={ACCENT7}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead><tr style={{ background: ACCENT7 + '18' }}>
                  <th>Diagnosis</th><th>How to Distinguish</th>
                </tr></thead>
                <tbody>
                  {df.ddx_table && Object.entries(df.ddx_table).map(([k, v]) => (
                    <tr key={k}>
                      <td className="fw-bold" style={{ color: ACCENT7, whiteSpace: 'nowrap' }}>{k}</td>
                      <td>{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        </div>
      )}

      {/* ── Tab 3: Definitions ── */}
      {tab === 3 && df && (
        <div>
          <Section title="Disease Definition" color={ACCENT}>
            <table className="table table-sm small table-bordered">
              <tbody>
                {['disease', 'omim_gene', 'omim_disease', 'chromosome', 'inheritance', 'prevalence'].map(k => (
                  <tr key={k}>
                    <td className="fw-bold" style={{ color: ACCENT, width: '18%' }}>{k}</td>
                    <td>{df[k]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Section>

          <Section title="Mechanism" color={ACCENT}>
            <p className="small">{df.mechanism}</p>
          </Section>

          <Section title="Key Clinical Features" color={ACCENT3}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <tbody>
                  {df.key_clinical_features && Object.entries(df.key_clinical_features).map(([k, v]) => (
                    <tr key={k}>
                      <td className="fw-bold text-nowrap" style={{ color: ACCENT3, width: '28%' }}>{k.replace(/_/g, ' ')}</td>
                      <td>{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="Diagnostic Criteria" color={ACCENT5}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <tbody>
                  {df.diagnostic_criteria && Object.entries(df.diagnostic_criteria).map(([k, v]) => (
                    <tr key={k}>
                      <td className="fw-bold text-nowrap" style={{ color: ACCENT5, width: '22%' }}>{k.replace(/_/g, ' ')}</td>
                      <td>{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="Treatment" color={ACCENT2}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <tbody>
                  {df.treatment && Object.entries(df.treatment).map(([k, v]) => (
                    <tr key={k}>
                      <td className="fw-bold text-nowrap" style={{ color: ACCENT2, width: '26%' }}>{k.replace(/_/g, ' ')}</td>
                      <td>{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="Prognosis" color={ACCENT4}>
            <p className="small">{df.prognosis}</p>
          </Section>

          <div className="alert alert-warning small mt-3">
            <strong>Cohort note:</strong> {df.cohort_note}
          </div>
        </div>
      )}
    </div>
  );
}
