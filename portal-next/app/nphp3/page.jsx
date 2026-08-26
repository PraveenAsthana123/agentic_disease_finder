'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'Genetics & Subtype', 'Definitions'];

// NPHP3 colour scheme — teal-amber-violet-sage (adolescent renal; CHF; situs; TZ scaffold)
const ACCENT  = '#00695c';   // dark teal — NPHP3 / adolescent ESRD / TZ scaffold
const ACCENT2 = '#1b5e20';   // deep green — transplant / curative outcome
const ACCENT3 = '#1a237e';   // dark navy — genetics / TZ module / NPHP1-3-4 complex
const ACCENT4 = '#bf360c';   // deep burnt orange — ESRD progression / anaemia
const ACCENT5 = '#4a148c';   // deep violet — situs inversus (less penetrant than NPHP2)
const ACCENT6 = '#37474f';   // dark slate — epidemiology / AR / consanguinity
const ACCENT7 = '#e65100';   // deep orange — misdiagnosis (FSGS / ADPKD) alert
const ACCENT8 = '#33691e';   // dark olive — CHF / hepatic fibrosis / liver surveillance

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
      <div className="progress" style={{ height: 8 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  );
}

export default function Nphp3Page() {
  const [tab, setTab] = useState(0);
  const [ov, setOv]   = useState(null);
  const [bk, setBk]   = useState(null);
  const [df, setDf]   = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/nphp3/overview`).then(r => r.json()),
      fetch(`${API}/api/nphp3/breakdown`).then(r => r.json()),
      fetch(`${API}/api/nphp3/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBk(b); setDf(d); })
      .catch(e => setErr(e.message));
  }, []);

  if (err) return <div className="container py-4 text-danger">Error: {err}</div>;
  if (!ov)  return <div className="container py-4 text-muted">Loading NPHP3 dashboard…</div>;

  const kpis = ov.kpis || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3">
        <h4 className="fw-bold mb-0" style={{ color: ACCENT }}>
          🧬 Nephronophthisis Type 3 (NPHP3 — Adolescent NPHP; Nephrocystin-3)
        </h4>
        <div className="text-muted small">
          NPHP3 · 3q22.1 · 1330 aa TZ scaffold · NPHP1–NPHP3–NPHP4 module · AR Biallelic LOF · OMIM #604387
          &nbsp;|&nbsp; Cohort: {_COHORT_SIZE} patients (seed-345) · 3 endpoints verified
        </div>
        <div className="mt-1">
          <Badge text="TZ Ciliopathy (Adolescent)" color={ACCENT} />
          <Badge text="ESRD Median ~19yr" color={ACCENT4} />
          <Badge text="Situs Inversus ~15–20%" color={ACCENT5} />
          <Badge text="CHF ~45%" color={ACCENT8} />
          <Badge text="Transplant EXCELLENT" color={ACCENT2} />
          <Badge text="No Retinal (no SLS)" color={ACCENT6} />
          <Badge text="AR · ~1/200,000" color={ACCENT6} />
        </div>
      </div>

      {/* Tab Nav */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link${tab === i ? ' active fw-bold' : ''}`}
              onClick={() => setTab(i)}
              style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}}
            >{t}</button>
          </li>
        ))}
      </ul>

      {/* ── TAB 0: Overview ── */}
      {tab === 0 && (
        <div>
          {/* KPI row */}
          <div className="row mb-2">
            <KPI label="Cohort (n)" value={kpis.cohort_n ?? _COHORT_SIZE} color={ACCENT} />
            <KPI label="Median Dx Age (yr)" value={kpis.median_age_dx_yr ?? '—'} color={ACCENT4} />
            <KPI label="Median GFR at Dx" value={kpis.median_gfr_at_dx_ml_min ? `${kpis.median_gfr_at_dx_ml_min} mL/min` : '—'} color={ACCENT4} />
            <KPI label="Median U-Osm" value={kpis.median_urine_osmolality ? `${kpis.median_urine_osmolality} mosm/kg` : '—'} color={ACCENT3} />
            <KPI label="% ESRD/RRT" value={kpis.pct_esrd_or_rrt ? `${kpis.pct_esrd_or_rrt}%` : '—'} color={ACCENT4} />
            <KPI label="% Transplanted" value={kpis.pct_transplanted ? `${kpis.pct_transplanted}%` : '—'} color={ACCENT2} />
          </div>
          <div className="row mb-3">
            <KPI label="% Situs Inversus" value={kpis.pct_situs_inversus ? `${kpis.pct_situs_inversus}%` : '—'} color={ACCENT5} />
            <KPI label="% CHF (liver)" value={kpis.pct_chf ? `${kpis.pct_chf}%` : '—'} color={ACCENT8} />
            <KPI label="% Consanguineous" value={kpis.pct_consanguineous ? `${kpis.pct_consanguineous}%` : '—'} color={ACCENT6} />
            <KPI label="% Kidneys Small" value={kpis.pct_kidneys_small ? `${kpis.pct_kidneys_small}%` : '—'} color={ACCENT4} />
            <KPI label="Mean Hgb (g/dL)" value={kpis.mean_hgb_g_dl ?? '—'} color={ACCENT4} />
            <KPI label="% Prior Misdiag." value={kpis.pct_prior_misdiagnosis ? `${kpis.pct_prior_misdiagnosis}%` : '—'} color={ACCENT7} />
          </div>

          {/* Alerts */}
          <Section title="⚠️ Clinical Alerts" color={ACCENT4}>
            {ov.alerts && Object.entries(ov.alerts).map(([k, v]) => (
              <Alert key={k} color={
                k === 'fsgs_misdiagnosis' || k === 'adolescent_onset_clue' ? ACCENT7 :
                k === 'chf_surveillance' ? ACCENT8 :
                k === 'transplant_curative' ? ACCENT2 : ACCENT4
              }>
                <span className="fw-bold">{k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}:</span>{' '}{v}
              </Alert>
            ))}
          </Section>

          {/* Key Facts */}
          <Section title="🔬 Key Facts — NPHP3 Clinical & Molecular" color={ACCENT}>
            <ul className="mb-0 small">
              {(ov.key_facts || []).map((f, i) => <li key={i} className="mb-1">{f}</li>)}
            </ul>
          </Section>

          {/* Comparison table NPHP1/2/3 */}
          <Section title="🫘 NPHP1 / NPHP2 / NPHP3 — Key Differences" color={ACCENT2}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered small mb-0">
                <thead style={{ background: ACCENT + '15' }}>
                  <tr>
                    <th>Feature</th>
                    <th style={{ color: ACCENT3 }}>NPHP1 (Juvenile)</th>
                    <th style={{ color: ACCENT4 }}>NPHP2 (Infantile)</th>
                    <th style={{ color: ACCENT }}>NPHP3 — THIS DISEASE</th>
                  </tr>
                </thead>
                <tbody>
                  <tr><td className="fw-bold">Onset type</td>
                    <td>Juvenile</td><td style={{ color: ACCENT4 }}>INFANTILE (earliest)</td>
                    <td style={{ color: ACCENT }}>ADOLESCENT (latest)</td></tr>
                  <tr><td className="fw-bold">ESRD median</td>
                    <td>13 yr</td><td style={{ color: ACCENT4 }}>3 yr</td>
                    <td style={{ color: ACCENT }}>~19 yr</td></tr>
                  <tr><td className="fw-bold">Kidneys</td>
                    <td>Small/Normal</td><td style={{ color: ACCENT7 }}>ENLARGED (ARPKD mimic)</td>
                    <td style={{ color: ACCENT4 }}>SMALL (echogenic)</td></tr>
                  <tr><td className="fw-bold">Situs inversus</td>
                    <td>Absent</td><td style={{ color: ACCENT5 }}>30–50% (UNIQUE)</td>
                    <td style={{ color: ACCENT5 }}>15–20% (less penetrant)</td></tr>
                  <tr><td className="fw-bold">CHF (liver)</td>
                    <td>Absent</td><td style={{ color: ACCENT8 }}>~55%</td>
                    <td style={{ color: ACCENT8 }}>~45%</td></tr>
                  <tr><td className="fw-bold">Retinal (SLS)</td>
                    <td>~12% Senior-Løken</td><td>None</td>
                    <td>None</td></tr>
                  <tr><td className="fw-bold">Key misdiagnosis</td>
                    <td>ADPKD</td><td>ARPKD</td>
                    <td style={{ color: ACCENT7 }}>FSGS / ADPKD</td></tr>
                  <tr><td className="fw-bold">Genetics</td>
                    <td>290kb del 80%</td><td>Heterogeneous; no founder</td>
                    <td>p.Gln872Ter European enriched</td></tr>
                  <tr><td className="fw-bold">Transplant</td>
                    <td style={{ color: ACCENT2 }}>EXCELLENT</td>
                    <td style={{ color: ACCENT2 }}>EXCELLENT</td>
                    <td style={{ color: ACCENT2 }}>EXCELLENT, no recurrence</td></tr>
                </tbody>
              </table>
            </div>
          </Section>

          {/* Patient Sample */}
          <Section title={`👥 Sample Patients (first 8 of ${_COHORT_SIZE})`} color={ACCENT6}>
            <div className="table-responsive">
              <table className="table table-sm table-hover small mb-0">
                <thead style={{ background: ACCENT + '15' }}>
                  <tr>
                    <th>ID</th><th>Gene/Allele</th><th>Ethnicity</th>
                    <th>Dx Age</th><th>GFR Dx</th><th>GFR Now</th>
                    <th>U-Osm</th><th>Hgb</th><th>Situs</th><th>CHF</th><th>RRT/Tx</th>
                  </tr>
                </thead>
                <tbody>
                  {(ov.patients || []).map(p => (
                    <tr key={p.id}>
                      <td><code style={{ fontSize: '0.75em' }}>{p.id}</code></td>
                      <td style={{ maxWidth: 130, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                          title={p.gene}>{p.gene?.split('—')?.slice(-1)[0]?.trim()?.split('/')[0]?.trim()}</td>
                      <td>{p.ethnicity?.split('(')[0]?.trim()}</td>
                      <td>{p.age_at_diagnosis_yr} yr</td>
                      <td style={{ color: p.gfr_at_dx_ml_min < 30 ? ACCENT4 : ACCENT2 }}>
                        {p.gfr_at_dx_ml_min}
                      </td>
                      <td style={{ color: p.gfr_now_ml_min < 15 ? ACCENT4 : p.gfr_now_ml_min < 30 ? ACCENT7 : ACCENT2 }}>
                        {p.gfr_now_ml_min}
                      </td>
                      <td style={{ color: p.urine_osmolality_mosm < 150 ? ACCENT4 : ACCENT3 }}>
                        {p.urine_osmolality_mosm}
                      </td>
                      <td style={{ color: p.hemoglobin_g_dl < 9 ? ACCENT4 : ACCENT6 }}>
                        {p.hemoglobin_g_dl}
                      </td>
                      <td style={{ color: p.situs?.includes('inversus') ? ACCENT5 : ACCENT6 }}>
                        {p.situs?.includes('inversus') ? '↔ SI' : p.situs?.includes('ambiguus') ? '? Amb' : '→ SS'}
                      </td>
                      <td style={{ color: p.chf_grade?.includes('Absent') ? ACCENT6 : ACCENT8 }}>
                        {p.chf_grade?.split('(')[0]?.trim()?.replace('Absent', '—')}
                      </td>
                      <td style={{ maxWidth: 120, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                          title={p.rrt_or_transplant}>{p.rrt_or_transplant?.split('—')[0]?.trim()?.split('(')[0]?.trim()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        </div>
      )}

      {/* ── TAB 1: Diagnostic Breakdown ── */}
      {tab === 1 && bk && (
        <div>
          <div className="row">
            <div className="col-md-6">
              <Section title="🫘 Kidney Size Distribution" color={ACCENT4}>
                {Object.entries(bk.kidney_size_distribution || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT4} />
                ))}
              </Section>

              <Section title="↔ Situs Distribution (Laterality)" color={ACCENT5}>
                {Object.entries(bk.situs_distribution || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT5} />
                ))}
              </Section>

              <Section title="🫀 CHF Grade (Hepatic Fibrosis)" color={ACCENT8}>
                {Object.entries(bk.chf_grade_distribution || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT8} />
                ))}
              </Section>

              <Section title="📊 CKD Stage (Current Visit)" color={ACCENT4}>
                {Object.entries(bk.ckd_stage_current || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT4} />
                ))}
              </Section>

              <Section title="📉 GFR Slope (ml/min/yr decline)" color={ACCENT3}>
                {Object.entries(bk.gfr_slope_tiers || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT3} />
                ))}
              </Section>
            </div>

            <div className="col-md-6">
              <Section title="💧 Urine Osmolality Tiers (mosm/kg)" color={ACCENT3}>
                {Object.entries(bk.urine_osmolality_tiers || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT3} />
                ))}
              </Section>

              <Section title="💊 RRT / Transplant Status" color={ACCENT2}>
                {Object.entries(bk.rrt_transplant_status || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT2} />
                ))}
              </Section>

              <Section title="⏱️ Age at Diagnosis Tiers" color={ACCENT6}>
                {Object.entries(bk.age_at_diagnosis_tiers || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT6} />
                ))}
              </Section>

              <Section title="⚠️ Prior Misdiagnosis" color={ACCENT7}>
                {Object.entries(bk.prior_misdiagnosis || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT7} />
                ))}
              </Section>
            </div>
          </div>

          <div className="row">
            <div className="col-md-6">
              <Section title="🧬 NPHP3 Allele Distribution" color={ACCENT3}>
                {Object.entries(bk.gene_distribution || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT3} />
                ))}
              </Section>
            </div>
            <div className="col-md-6">
              <Section title="🌍 Ethnicity" color={ACCENT6}>
                {Object.entries(bk.ethnicity || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT6} />
                ))}
              </Section>
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 2: Genetics & Subtype ── */}
      {tab === 2 && df && (
        <div>
          <div className="row">
            <div className="col-md-6">
              <Section title="🧬 Genetic Architecture (NPHP3/3q22.1)" color={ACCENT3}>
                {Object.entries(df.genetic_architecture || {}).map(([k, v]) => (
                  <div key={k} className="mb-2 small">
                    <span className="fw-bold" style={{ color: ACCENT3 }}>{k.replace(/_/g, ' ')}:</span>{' '}{v}
                  </div>
                ))}
              </Section>

              <Section title="🔍 Founder Variants" color={ACCENT3}>
                <ul className="small mb-0">
                  {(df.founder_variants || []).map((v, i) => <li key={i} className="mb-1">{v}</li>)}
                </ul>
              </Section>

              <Section title="🦠 NPHP Subtype Comparison" color={ACCENT}>
                <table className="table table-sm table-bordered small mb-0">
                  <thead style={{ background: ACCENT + '15' }}>
                    <tr><th>Subtype / Gene</th><th>Key Distinguishing Feature</th></tr>
                  </thead>
                  <tbody>
                    {Object.entries(df.nphp_comparison || {}).map(([k, v]) => (
                      <tr key={k} style={k.includes('NPHP3') ? { background: ACCENT + '15' } : {}}>
                        <td className="fw-bold" style={{ color: k.includes('NPHP3') ? ACCENT : ACCENT6, whiteSpace: 'nowrap' }}>{k}</td>
                        <td className="small">{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Section>
            </div>

            <div className="col-md-6">
              <Section title="🔍 Differential Diagnosis" color={ACCENT7}>
                <table className="table table-sm table-bordered small mb-0">
                  <thead style={{ background: ACCENT7 + '15' }}>
                    <tr><th>Condition</th><th>Key Differentiators</th></tr>
                  </thead>
                  <tbody>
                    {Object.entries(df.ddx_table || {}).map(([k, v]) => (
                      <tr key={k}>
                        <td className="fw-bold" style={{ color: ACCENT7, whiteSpace: 'nowrap' }}>{k.replace(/_/g, ' ')}</td>
                        <td className="small">{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Section>
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 3: Definitions ── */}
      {tab === 3 && df && (
        <div>
          <div className="row">
            <div className="col-md-6">
              <Section title="📖 Disease Overview" color={ACCENT}>
                <table className="table table-sm table-bordered small mb-3">
                  <tbody>
                    <tr><td className="fw-bold">Disease</td><td>{df.disease}</td></tr>
                    <tr><td className="fw-bold">OMIM Gene</td><td>{df.omim_gene}</td></tr>
                    <tr><td className="fw-bold">OMIM Disease</td><td>{df.omim_disease}</td></tr>
                    <tr><td className="fw-bold">Chromosome</td><td>{df.chromosome}</td></tr>
                    <tr><td className="fw-bold">Inheritance</td><td>{df.inheritance}</td></tr>
                    <tr><td className="fw-bold">Prevalence</td><td>{df.prevalence}</td></tr>
                  </tbody>
                </table>
                <div className="small p-2 rounded" style={{ background: ACCENT + '08', border: `1px solid ${ACCENT}30` }}>
                  <b>Mechanism:</b> {df.mechanism}
                </div>
              </Section>

              <Section title="🏥 Diagnostic Criteria" color={ACCENT3}>
                {Object.entries(df.diagnostic_criteria || {}).map(([k, v]) => (
                  <div key={k} className="mb-2 small">
                    <span className="fw-bold" style={{ color: ACCENT3 }}>{k.replace(/_/g, ' ')}:</span>{' '}{v}
                  </div>
                ))}
              </Section>
            </div>

            <div className="col-md-6">
              <Section title="🩺 Key Clinical Features" color={ACCENT2}>
                {Object.entries(df.key_clinical_features || {}).map(([k, v]) => (
                  <div key={k} className="mb-2 small">
                    <span className="fw-bold" style={{ color: ACCENT2 }}>{k.replace(/_/g, ' ')}:</span>{' '}{v}
                  </div>
                ))}
              </Section>

              <Section title="💊 Treatment" color={ACCENT4}>
                {Object.entries(df.treatment || {}).map(([k, v]) => (
                  <div key={k} className="mb-2 small">
                    <span className="fw-bold" style={{ color: ACCENT4 }}>{k.replace(/_/g, ' ')}:</span>{' '}{v}
                  </div>
                ))}
              </Section>

              <Section title="📈 Prognosis" color={ACCENT2}>
                <div className="small p-2 rounded" style={{ background: ACCENT2 + '10', border: `1px solid ${ACCENT2}40` }}>
                  {df.prognosis}
                </div>
              </Section>

              <div className="text-muted small mt-2">{df.cohort_note}</div>
            </div>
          </div>
        </div>
      )}

      <div className="mt-3 pt-2 border-top">
        <Link href="/" className="text-muted small">← Back to Portal</Link>
      </div>
    </div>
  );
}
