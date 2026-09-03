'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const COLOR  = '#1a237e';   // deep indigo — anomaly detection / signal analytics
const LIGHT  = '#e8eaf6';

const TABS = ['Overview', 'Patient Breakdown', 'Timeline', 'Definitions'];

function KPI({ label, value, color = COLOR, sub }) {
  return (
    <div className="col-6 col-md-4 col-lg-2 mb-3">
      <div className="card h-100 shadow-sm text-center">
        <div className="card-body py-2 px-1">
          <div className="fw-bold fs-5" style={{ color }}>{value ?? '—'}</div>
          <div className="text-muted small">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.68rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function Bar({ label, value, max, color = COLOR }) {
  const pct = max > 0 ? Math.min((value / max) * 100, 100) : 0;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span>
        <span className="text-muted fw-semibold">{value}</span>
      </div>
      <div className="progress" style={{ height: 10 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function SectionCard({ title, children, icon = '' }) {
  return (
    <div className="card mb-4 shadow-sm" style={{ borderTop: `3px solid ${COLOR}` }}>
      <div className="card-body">
        <h6 className="card-title fw-bold mb-3" style={{ color: COLOR }}>{icon} {title}</h6>
        {children}
      </div>
    </div>
  );
}

function SeverityBadge({ severity, count }) {
  const bg = severity === 'Severe'   ? '#b71c1c'
           : severity === 'Moderate' ? '#e65100'
           : severity === 'Mild'     ? '#f9a825'
           : '#388e3c';
  return (
    <span className="badge me-2 mb-1" style={{ backgroundColor: bg, fontSize: '0.78rem' }}>
      {severity}: {count}
    </span>
  );
}

function Spinner() {
  return <div className="text-center py-5"><div className="spinner-border" style={{ color: COLOR }} /></div>;
}

// ── Tab 1: Overview ──────────────────────────────────────────────────────────
function OverviewTab({ ov }) {
  if (!ov) return <Spinner />;
  const kpis = ov.kpis || {};
  const cats = ov.anomaly_by_category || [];
  const sev  = ov.anomaly_severity_distribution || [];
  const topF = ov.top_anomalous_features || [];
  const maxCat = Math.max(...cats.map(c => c.anomaly_count || 0), 1);
  const maxTopF = Math.max(...topF.map(f => f.anomaly_count || 0), 1);

  return (
    <div>
      <SectionCard title="Detection Summary" icon="🔍">
        <div className="row">
          <KPI label="Total Analyses"      value={kpis.total_analyses}         sub="EEG analyses scanned" />
          <KPI label="Patients Covered"    value={kpis.total_patients}         sub="unique patients" />
          <KPI label="Features Monitored"  value={kpis.total_features_monitored} sub="EEG features" />
          <KPI label="Anomalous Analyses"  value={kpis.anomalous_analyses}     sub="at least 1 anomaly" color="#e65100" />
          <KPI label="Anomaly Rate"        value={kpis.anomaly_rate}            color="#b71c1c" />
          <KPI label="Severe Anomalies"    value={kpis.severe_anomalies}       sub="|z| ≥ 3.0" color={kpis.severe_anomalies > 0 ? '#b71c1c' : '#388e3c'} />
        </div>
        <div className="mt-2 p-2 rounded small" style={{ background: LIGHT, borderLeft: `4px solid ${COLOR}` }}>
          🎯 Most anomalous feature: <strong>{kpis.most_anomalous_feature}</strong> ·
          Mean anomalies/analysis: <strong>{kpis.mean_anomalies_per_analysis}</strong>
        </div>
      </SectionCard>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Anomalies by Category" icon="📂">
            {cats.map(c => (
              <Bar key={c.category} label={`${c.category} (${c.feature_count} features)`}
                   value={c.anomaly_count} max={maxCat} />
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Severity Distribution" icon="⚡">
            <div className="mb-3">
              {sev.map(s => <SeverityBadge key={s.severity} severity={s.severity} count={s.count} />)}
            </div>
            <table className="table table-sm table-bordered mb-0 small">
              <thead><tr><th>Severity</th><th>Count</th><th>Threshold</th></tr></thead>
              <tbody>
                {sev.map(s => (
                  <tr key={s.severity}>
                    <td>{s.severity}</td>
                    <td>{s.count}</td>
                    <td className="text-muted">
                      {s.severity === 'Normal' ? '|z| < 2.0'
                       : s.severity === 'Mild' ? '2.0 ≤ |z| < 2.5'
                       : s.severity === 'Moderate' ? '2.5 ≤ |z| < 3.0'
                       : '|z| ≥ 3.0'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </SectionCard>
        </div>
      </div>

      <SectionCard title="Top Anomalous Features" icon="📊">
        {topF.map(f => (
          <div key={f.feature} className="mb-2">
            <Bar label={`${f.feature} [${f.category}]`} value={f.anomaly_count} max={maxTopF} />
            <div className="text-muted" style={{ fontSize: '0.72rem', marginTop: -6 }}>
              Mean z-score: {f.mean_zscore}
            </div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab 2: Patient Breakdown ─────────────────────────────────────────────────
function PatientTab({ bk }) {
  const [filter, setFilter] = useState('');
  if (!bk) return <Spinner />;
  const patients = bk.per_patient_anomalies || [];
  const features = bk.per_feature_stats || [];
  const maxAnom = Math.max(...patients.map(p => p.total_anomalies || 0), 1);
  const filtered = filter
    ? patients.filter(p => p.patient_id?.toLowerCase().includes(filter.toLowerCase()))
    : patients;

  return (
    <div>
      <SectionCard title="Per-Patient Anomaly Summary" icon="🧑‍⚕️">
        <input className="form-control form-control-sm mb-3"
               placeholder="Filter by patient ID…"
               value={filter} onChange={e => setFilter(e.target.value)} />
        <div style={{ maxHeight: 360, overflowY: 'auto' }}>
          <table className="table table-sm table-hover table-bordered mb-0 small">
            <thead className="table-dark">
              <tr>
                <th>Patient ID</th>
                <th>Total Anomalies</th>
                <th>Severe</th>
                <th>Confidence</th>
                <th>Has Seizures</th>
                <th>Anomalous Features</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map(p => (
                <tr key={p.patient_id}>
                  <td className="fw-semibold">{p.patient_id}</td>
                  <td>
                    <div className="d-flex align-items-center gap-1">
                      <div className="progress flex-grow-1" style={{ height: 8 }}>
                        <div className="progress-bar" style={{
                          width: `${(p.total_anomalies / maxAnom) * 100}%`,
                          backgroundColor: p.severe > 0 ? '#b71c1c' : COLOR,
                        }} />
                      </div>
                      <span>{p.total_anomalies}</span>
                    </div>
                  </td>
                  <td className={p.severe > 0 ? 'text-danger fw-bold' : 'text-muted'}>{p.severe}</td>
                  <td>{(p.confidence * 100).toFixed(1)}%</td>
                  <td>
                    {p.has_seizures
                      ? <span className="badge bg-danger">Yes</span>
                      : <span className="badge bg-secondary">No</span>}
                  </td>
                  <td className="text-muted" style={{ fontSize: '0.7rem' }}>
                    {(p.anomalous_features || []).join(', ')}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Per-Feature Statistics" icon="📈">
        <div style={{ maxHeight: 280, overflowY: 'auto' }}>
          <table className="table table-sm table-hover table-bordered mb-0 small">
            <thead className="table-dark">
              <tr>
                <th>Feature</th>
                <th>Category</th>
                <th>Anomaly Count</th>
                <th>Mean Z-Score</th>
                <th>Max Z-Score</th>
              </tr>
            </thead>
            <tbody>
              {features.map((f, i) => (
                <tr key={i}>
                  <td className="fw-semibold">{f.feature}</td>
                  <td>{f.category}</td>
                  <td>{f.anomaly_count}</td>
                  <td>{f.mean_zscore?.toFixed ? f.mean_zscore.toFixed(3) : f.mean_zscore}</td>
                  <td>{f.max_zscore?.toFixed ? f.max_zscore.toFixed(3) : f.max_zscore}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab 3: Timeline ──────────────────────────────────────────────────────────
function TimelineTab({ bk }) {
  if (!bk) return <Spinner />;
  const timeline = bk.anomaly_timeline || [];
  const corrAnom = bk.feature_correlation_anomalies || [];
  const sigQ     = bk.signal_quality_analysis   || [];

  return (
    <div>
      <SectionCard title="Anomaly Timeline (per analysis)" icon="⏱️">
        <div style={{ maxHeight: 300, overflowY: 'auto' }}>
          <table className="table table-sm table-hover table-bordered mb-0 small">
            <thead className="table-dark">
              <tr>
                <th>Analysis ID</th>
                <th>Patient ID</th>
                <th>Date</th>
                <th>Anomaly Count</th>
                <th>Severe</th>
                <th>Confidence</th>
              </tr>
            </thead>
            <tbody>
              {timeline.map(t => (
                <tr key={t.analysis_id}>
                  <td>{t.analysis_id}</td>
                  <td className="fw-semibold">{t.patient_id}</td>
                  <td className="text-muted">{t.created_at?.split('T')[0] || '—'}</td>
                  <td className={t.anomaly_count > 5 ? 'text-danger fw-bold' : ''}>{t.anomaly_count}</td>
                  <td className={t.severe_count > 0 ? 'text-danger fw-bold' : 'text-muted'}>{t.severe_count}</td>
                  <td>{t.confidence != null ? (t.confidence * 100).toFixed(1) + '%' : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {corrAnom.length > 0 && (
        <SectionCard title="Feature Correlation Anomalies" icon="🔗">
          <div style={{ maxHeight: 200, overflowY: 'auto' }}>
            <table className="table table-sm table-bordered mb-0 small">
              <thead className="table-dark">
                <tr>{Object.keys(corrAnom[0] || {}).map(k => <th key={k}>{k}</th>)}</tr>
              </thead>
              <tbody>
                {corrAnom.map((r, i) => (
                  <tr key={i}>{Object.values(r).map((v, j) => <td key={j}>{String(v)}</td>)}</tr>
                ))}
              </tbody>
            </table>
          </div>
        </SectionCard>
      )}

      {sigQ.length > 0 && (
        <SectionCard title="Signal Quality Analysis" icon="📡">
          <div style={{ maxHeight: 200, overflowY: 'auto' }}>
            <table className="table table-sm table-bordered mb-0 small">
              <thead className="table-dark">
                <tr>{Object.keys(sigQ[0] || {}).map(k => <th key={k}>{k}</th>)}</tr>
              </thead>
              <tbody>
                {sigQ.map((r, i) => (
                  <tr key={i}>{Object.values(r).map((v, j) => <td key={j}>{String(v)}</td>)}</tr>
                ))}
              </tbody>
            </table>
          </div>
        </SectionCard>
      )}
    </div>
  );
}

// ── Tab 4: Definitions ───────────────────────────────────────────────────────
function DefsTab({ defs }) {
  if (!defs) return <Spinner />;
  return (
    <div>
      {(defs.sections || []).map((sec, i) => (
        <SectionCard key={i} title={sec.title} icon="📖">
          <dl className="mb-0">
            {(sec.items || []).map((item, j) => (
              <div key={j} className="mb-2">
                <dt className="fw-semibold small" style={{ color: COLOR }}>{item.term}</dt>
                <dd className="text-muted small mb-0">{item.definition}</dd>
              </div>
            ))}
          </dl>
        </SectionCard>
      ))}
    </div>
  );
}

// ── Main ─────────────────────────────────────────────────────────────────────
export default function AnomalyDetectionDashboard() {
  const [tab, setTab] = useState(0);
  const [ov,  setOv]  = useState(null);
  const [bk,  setBk]  = useState(null);
  const [defs,setDefs]= useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/anomaly-detection/overview`).then(r => r.json()),
      fetch(`${API}/api/anomaly-detection/breakdown`).then(r => r.json()),
      fetch(`${API}/api/anomaly-detection/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBk(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Error: {err}</div>;

  const kpis = ov?.kpis || {};

  return (
    <div>
      <div className="d-flex align-items-center gap-2 mb-1">
        <span style={{ fontSize: '1.6rem' }}>🔍</span>
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: COLOR }}>
            EEG Anomaly Detection Dashboard
          </h4>
          <div className="text-muted small">
            Z-Score · IQR dual-method unsupervised detection ·{' '}
            <strong>{kpis.total_features_monitored || '…'}</strong> features ·{' '}
            <strong>{kpis.total_patients || '…'}</strong> patients ·{' '}
            anomaly rate <strong>{kpis.anomaly_rate || '…'}</strong>
          </div>
        </div>
      </div>

      {/* Tab nav */}
      <ul className="nav nav-tabs mb-3 mt-3">
        {TABS.map((t, i) => (
          <li className="nav-item" key={t}>
            <button className={`nav-link${tab === i ? ' active' : ''}`}
                    style={tab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}
                    onClick={() => setTab(i)}>
              {t}
            </button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab ov={ov} />}
      {tab === 1 && <PatientTab  bk={bk} />}
      {tab === 2 && <TimelineTab bk={bk} />}
      {tab === 3 && <DefsTab     defs={defs} />}
    </div>
  );
}
