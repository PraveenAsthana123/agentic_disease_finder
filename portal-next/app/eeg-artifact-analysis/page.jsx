'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const sevColor = s => ({ mild: 'success', moderate: 'warning', severe: 'danger' }[s] || 'secondary');
const burdenColor = b => ({ High: 'danger', Moderate: 'warning', Low: 'success' }[b] || 'secondary');
const typeIcon = t => ({
  muscle: '💪', ECG: '❤️', electrode_pop: '⚡', movement: '🏃', eye_blink: '👁️', sweat: '💧'
}[t] || '📊');

export default function EegArtifactAnalysisDashboard() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/eeg-artifact-analysis/overview`).then(r => r.json()),
      fetch(`${API}/api/eeg-artifact-analysis/breakdown`).then(r => r.json()),
      fetch(`${API}/api/eeg-artifact-analysis/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(e.message));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Error: {err}</div>;
  if (!ov || !bd || !defs) return <div className="text-center p-5"><div className="spinner-border text-primary" /></div>;

  const k = ov.kpis;
  const total = bd.type_severity_matrix.reduce((s, r) => s + (r.mild || 0) + (r.moderate || 0) + (r.severe || 0), 0);
  const maxTypeCt = Math.max(...ov.type_distribution.map(t => t.count), 1);
  const maxChCt = Math.max(...bd.channel_distribution.map(c => c.count), 1);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center gap-2 mb-3">
        <h4 className="mb-0">📊 EEG Artifact Analysis</h4>
        <span className="badge bg-secondary">{k.total_annotations} annotations · {k.patients_affected} patients · {k.artifact_types} types</span>
      </div>

      {/* KPI row */}
      <div className="row g-3 mb-3">
        {[
          { label: 'Total Annotations', val: k.total_annotations, color: 'primary', icon: '📋' },
          { label: 'Patients Affected', val: k.patients_affected, color: 'info', icon: '🧑‍⚕️' },
          { label: 'Artifact Types', val: k.artifact_types, color: 'secondary', icon: '🔬' },
          { label: 'Avg / Patient', val: k.avg_per_patient, color: 'warning', icon: '📈' },
          { label: 'Max Burden', val: k.max_patient_burden, color: 'danger', icon: '⚠️' },
          { label: 'Mild Rate', val: `${k.mild_pct}%`, color: 'success', icon: '✅' },
        ].map(({ label, val, color, icon }) => (
          <div className="col-6 col-md-4 col-lg-2" key={label}>
            <div className={`card border-${color} border-2 h-100`}>
              <div className="card-body text-center p-2">
                <div style={{ fontSize: '1.4rem' }}>{icon}</div>
                <div className={`fw-bold fs-5 text-${color}`}>{val}</div>
                <div className="text-muted small">{label}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {['overview', 'channels', 'matrix', 'patients', 'definitions'].map(t => (
          <li className="nav-item" key={t}>
            <button className={`nav-link ${tab === t ? 'active' : ''}`} onClick={() => setTab(t)}>
              {{ overview: '📊 Overview', channels: '📡 Channels', matrix: '🔢 Type×Severity', patients: '🧑‍⚕️ Per Patient', definitions: '📚 Definitions' }[t]}
            </button>
          </li>
        ))}
      </ul>

      {/* OVERVIEW TAB */}
      {tab === 'overview' && (
        <div className="row g-3">
          <div className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold">Artifact Type Distribution</div>
              <div className="card-body">
                {ov.type_distribution.map(({ type, count, avg_duration_sec }) => (
                  <div key={type} className="mb-2">
                    <div className="d-flex justify-content-between align-items-center mb-1">
                      <span>{typeIcon(type)} <strong>{type}</strong></span>
                      <span className="badge bg-primary">{count} &nbsp; avg {avg_duration_sec}s</span>
                    </div>
                    <div className="progress" style={{ height: 10 }}>
                      <div className="progress-bar bg-primary" style={{ width: `${(count / maxTypeCt) * 100}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold">Severity Distribution</div>
              <div className="card-body">
                {ov.severity_distribution.map(({ severity, count, pct }) => (
                  <div key={severity} className="mb-3">
                    <div className="d-flex justify-content-between align-items-center mb-1">
                      <span className={`badge bg-${sevColor(severity)} fs-6 px-3`}>{severity}</span>
                      <span className="fw-bold">{count} <small className="text-muted">({pct}%)</small></span>
                    </div>
                    <div className="progress" style={{ height: 14 }}>
                      <div className={`progress-bar bg-${sevColor(severity)}`} style={{ width: `${pct}%` }}>
                        {pct}%
                      </div>
                    </div>
                  </div>
                ))}
                <hr />
                <div className="alert alert-info py-2 mb-0 small">
                  💡 {k.mild_pct}% of artifacts are mild — manageable with standard ICA filtering.
                  {k.severe_pct > 10 && ` ⚠️ ${k.severe_pct}% severe — these epochs require exclusion.`}
                </div>
              </div>
            </div>
          </div>
          {bd.monthly_trend.length > 0 && (
            <div className="col-12">
              <div className="card shadow-sm">
                <div className="card-header fw-semibold">Monthly Annotation Volume</div>
                <div className="card-body">
                  <div className="d-flex align-items-end gap-2" style={{ height: 80 }}>
                    {bd.monthly_trend.map(({ month, count }) => {
                      const maxMo = Math.max(...bd.monthly_trend.map(x => x.count), 1);
                      return (
                        <div key={month} className="d-flex flex-column align-items-center" style={{ flex: 1 }}>
                          <div className="small text-muted mb-1">{count}</div>
                          <div className="bg-primary rounded-top" style={{ width: '100%', height: `${(count / maxMo) * 60}px`, minHeight: 4 }} />
                          <div className="small text-muted mt-1" style={{ fontSize: '0.6rem' }}>{month.slice(5)}</div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* CHANNELS TAB */}
      {tab === 'channels' && (
        <div className="card shadow-sm">
          <div className="card-header fw-semibold">Top Affected EEG Channels</div>
          <div className="card-body">
            <p className="text-muted small mb-3">
              Channels with highest artifact frequency indicate electrode placement issues, proximity to muscle groups, or high-motion regions.
            </p>
            {bd.channel_distribution.map(({ channel, count }) => (
              <div key={channel} className="mb-2">
                <div className="d-flex justify-content-between align-items-center mb-1">
                  <span className="fw-semibold font-monospace">{channel}</span>
                  <span className="badge bg-info text-dark">{count} annotations</span>
                </div>
                <div className="progress" style={{ height: 12 }}>
                  <div className="progress-bar bg-info" style={{ width: `${(count / maxChCt) * 100}%` }} />
                </div>
              </div>
            ))}
            <div className="alert alert-warning small mt-3 mb-0">
              ⚠️ High artifact frequency on temporal channels (T-leads) is common — proximity to temporalis muscle. Occipital (O-leads) flag eye-movement contamination.
            </div>
          </div>
        </div>
      )}

      {/* TYPE×SEVERITY MATRIX TAB */}
      {tab === 'matrix' && (
        <div className="card shadow-sm">
          <div className="card-header fw-semibold">Artifact Type × Severity Matrix</div>
          <div className="card-body">
            <div className="table-responsive">
              <table className="table table-bordered table-sm align-middle text-center">
                <thead className="table-dark">
                  <tr>
                    <th>Type</th>
                    <th><span className="badge bg-success">Mild</span></th>
                    <th><span className="badge bg-warning text-dark">Moderate</span></th>
                    <th><span className="badge bg-danger">Severe</span></th>
                    <th>Total</th>
                  </tr>
                </thead>
                <tbody>
                  {bd.type_severity_matrix.map(row => {
                    const rowTotal = (row.mild || 0) + (row.moderate || 0) + (row.severe || 0);
                    return (
                      <tr key={row.type}>
                        <td className="text-start fw-semibold">{typeIcon(row.type)} {row.type}</td>
                        <td><span className="badge bg-success">{row.mild || 0}</span></td>
                        <td><span className="badge bg-warning text-dark">{row.moderate || 0}</span></td>
                        <td><span className="badge bg-danger">{row.severe || 0}</span></td>
                        <td className="fw-bold">{rowTotal}</td>
                      </tr>
                    );
                  })}
                  <tr className="table-secondary fw-bold">
                    <td>Total</td>
                    {['mild', 'moderate', 'severe'].map(s => (
                      <td key={s}>{bd.type_severity_matrix.reduce((sum, r) => sum + (r[s] || 0), 0)}</td>
                    ))}
                    <td>{total}</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <div className="alert alert-info small mb-0">
              💡 ECG and electrode_pop artifacts tend to produce sharp transients that can mimic epileptiform discharges — classifier pre-screening recommended.
            </div>
          </div>
        </div>
      )}

      {/* PER PATIENT TAB */}
      {tab === 'patients' && (
        <div className="card shadow-sm">
          <div className="card-header fw-semibold">Per-Patient Artifact Burden</div>
          <div className="card-body p-0">
            <div className="table-responsive">
              <table className="table table-sm table-striped align-middle mb-0">
                <thead className="table-dark">
                  <tr>
                    <th>Patient</th>
                    <th>Total Artifacts</th>
                    <th>Severe</th>
                    <th>Unique Types</th>
                    <th>Total Duration</th>
                    <th>Burden</th>
                  </tr>
                </thead>
                <tbody>
                  {bd.per_patient.map(p => (
                    <tr key={p.patient_id}>
                      <td className="fw-semibold font-monospace">{p.patient_id}</td>
                      <td>{p.total_artifacts}</td>
                      <td>{p.severe_count > 0
                        ? <span className="badge bg-danger">{p.severe_count}</span>
                        : <span className="text-muted">0</span>}
                      </td>
                      <td>{p.unique_types}</td>
                      <td>{p.total_duration_sec}s</td>
                      <td><span className={`badge bg-${burdenColor(p.burden)}`}>{p.burden}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'definitions' && (
        <div>
          <div className="alert alert-secondary mb-3">{defs.overview}</div>
          <div className="row g-3 mb-3">
            {defs.artifact_types.map(at => (
              <div className="col-md-6" key={at.type}>
                <div className="card shadow-sm h-100">
                  <div className="card-header fw-semibold">
                    {typeIcon(at.type)} {at.label}
                  </div>
                  <div className="card-body small">
                    <p>{at.description}</p>
                    <p><strong>Channels:</strong> {at.channels_affected}</p>
                    <p><strong>Clinical Impact:</strong> {at.clinical_impact}</p>
                    <p className="mb-0"><strong>Mitigation:</strong> {at.mitigation}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-semibold">Severity Levels</div>
            <div className="card-body">
              <div className="row g-2">
                {defs.severity_levels.map(sl => (
                  <div className="col-md-4" key={sl.level}>
                    <div className={`card border-${sl.badge}`}>
                      <div className="card-body p-2 small">
                        <span className={`badge bg-${sl.badge} mb-1`}>{sl.label}</span>
                        <p className="mb-0">{sl.description}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="card shadow-sm">
            <div className="card-header fw-semibold">References</div>
            <div className="card-body small">
              <ul className="mb-1">
                {defs.references.map(r => <li key={r}>{r}</li>)}
              </ul>
              <p className="text-muted mb-0">Source: {defs.data_source}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
