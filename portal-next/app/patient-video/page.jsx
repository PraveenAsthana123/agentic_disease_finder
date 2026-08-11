'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const RISK_COLOR = { low: 'success', medium: 'warning', high: 'danger' };
const PAT_COLOR = {
  tonic_extension: 'danger', clonic_rhythmic: 'warning', automatism_oral: 'info',
  automatism_manual: 'info', automatism_pedal: 'info', hypermotor_thrashing: 'danger',
  versive_head_turn: 'secondary', dystonic_limb: 'warning', myoclonic_jerk: 'warning',
  atonic_collapse: 'danger', tremor: 'secondary', normal_movement: 'success',
};

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-3 mb-2">
      <div className="card shadow-sm border-0 h-100">
        <div className="card-body text-center py-2">
          <div className={`h4 mb-0 fw-bold text-${color || 'primary'}`}>{value ?? '—'}</div>
          <div className="text-muted small">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: 10 }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function Bar({ items, labelKey, valueKey }) {
  if (!items || !items.length) return null;
  const mx = Math.max(...items.map(i => i[valueKey] || 0));
  return (
    <div>
      {items.map((item, idx) => {
        const color = PAT_COLOR[item.key] || 'primary';
        return (
          <div key={idx} className="d-flex align-items-center mb-1">
            <div className="text-end small me-2" style={{ width: 175, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {item[labelKey]}
            </div>
            <div className="flex-grow-1">
              <div className="progress" style={{ height: 20 }}>
                <div className={`progress-bar bg-${color}`}
                  style={{ width: `${mx ? (item[valueKey] / mx) * 100 : 0}%` }}>
                  <span className="small">{item[valueKey]}</span>
                </div>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function ConfHist({ bins }) {
  if (!bins || !bins.length) return null;
  const mx = Math.max(...bins.map(b => b.count));
  return (
    <div className="d-flex align-items-end gap-1" style={{ height: 80 }}>
      {bins.map((b, i) => (
        <div key={i} className="d-flex flex-column align-items-center flex-grow-1">
          <div className="bg-primary rounded-top"
            style={{ width: '100%', height: `${mx ? (b.count / mx) * 64 : 4}px`, minHeight: 4 }} />
          <div className="text-muted" style={{ fontSize: 9, marginTop: 2, textAlign: 'center' }}>{b.bin}</div>
        </div>
      ))}
    </div>
  );
}

export default function PatientVideoPage() {
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [expandedPt, setExpandedPt] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/patient-video/overview`).then(r => r.json()),
      fetch(`${API}/api/patient-video/breakdown`).then(r => r.json()),
      fetch(`${API}/api/patient-video/definitions`).then(r => r.json()),
    ]).then(([ov, bd, df]) => {
      setOverview(ov);
      setBreakdown(bd);
      setDefs(df);
    }).catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-4">{err}</div>;
  if (!overview) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const TABS = [
    { id: 'overview', label: 'Overview' },
    { id: 'patterns', label: 'Motor Patterns' },
    { id: 'patients', label: 'Per Patient' },
    { id: 'definitions', label: 'Definitions' },
  ];

  const patients = breakdown?.patients || [];

  return (
    <div className="container-fluid py-3">
      <h4 className="mb-1 fw-bold">Patient Video Seizure Analysis</h4>
      <p className="text-muted small mb-3">
        Automated video-based seizure detection — pose estimation + action recognition.
        {' '}{overview.total_patients} patients · {overview.total_video_events} events · {overview.seizure_events_detected} seizures detected.
      </p>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link${tab === t.id ? ' active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* ── Overview ── */}
      {tab === 'overview' && (
        <>
          <div className="row g-2 mb-3">
            <KPI label="Patients Monitored" value={overview.total_patients} color="primary" />
            <KPI label="Total Video Events" value={overview.total_video_events} color="info" />
            <KPI label="Seizures Detected" value={overview.seizure_events_detected} color="danger" />
            <KPI label="Fall Alerts" value={overview.fall_alerts} color="warning"
              sub={`${overview.fall_alert_pct}% of events`} />
          </div>
          <div className="row g-2 mb-3">
            <KPI label="Automatism Events" value={overview.automatism_events} color="secondary" />
            <KPI label="Avg Confidence" value={`${Math.round((overview.average_confidence || 0) * 100)}%`} color="success" />
            <KPI label="Motor Patterns" value={(overview.pattern_distribution || []).length} color="primary" />
            <KPI label="Detection Models" value="3" color="info" sub="3D-CNN · SlowFast · TimeSformer" />
          </div>

          {/* Confidence histogram */}
          <div className="card shadow-sm mb-3">
            <div className="card-header py-2 fw-semibold">Confidence Score Distribution</div>
            <div className="card-body">
              <ConfHist bins={overview.confidence_histogram || []} />
              <div className="text-muted small mt-2 text-center">
                Average: {Math.round((overview.average_confidence || 0) * 100)}% confidence across all video detections
              </div>
            </div>
          </div>

          {/* Top patterns preview */}
          <div className="card shadow-sm">
            <div className="card-header py-2 fw-semibold">Motor Pattern Overview (top 6)</div>
            <div className="card-body">
              <Bar
                items={(overview.pattern_distribution || []).slice(0, 6)}
                labelKey="pattern"
                valueKey="count"
              />
              <div className="text-muted small mt-2">Switch to Motor Patterns tab for full breakdown.</div>
            </div>
          </div>
        </>
      )}

      {/* ── Motor Patterns ── */}
      {tab === 'patterns' && (
        <>
          <div className="card shadow-sm mb-3">
            <div className="card-header py-2 fw-semibold">All Motor Patterns — Event Count</div>
            <div className="card-body">
              <Bar
                items={overview.pattern_distribution || []}
                labelKey="pattern"
                valueKey="count"
              />
            </div>
          </div>
          <div className="card shadow-sm">
            <div className="card-header py-2 fw-semibold">Pattern × Fall Risk</div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm mb-0">
                  <thead className="table-light">
                    <tr>
                      <th>Motor Pattern</th>
                      <th>Events</th>
                      <th>Fall Risk</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(defs?.motor_patterns || []).map((p, i) => {
                      const ev = (overview.pattern_distribution || []).find(d => d.key === p.key);
                      const riskPct = Math.round((p.fall_risk || 0) * 100);
                      return (
                        <tr key={i}>
                          <td className="small">{p.label}</td>
                          <td className="small">{ev?.count ?? '—'}</td>
                          <td>
                            <div className="progress" style={{ height: 14, width: 80 }}>
                              <div
                                className={`progress-bar bg-${riskPct >= 70 ? 'danger' : riskPct >= 40 ? 'warning' : 'success'}`}
                                style={{ width: `${riskPct}%` }}
                              >
                                <span style={{ fontSize: 9 }}>{riskPct}%</span>
                              </div>
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </>
      )}

      {/* ── Per Patient ── */}
      {tab === 'patients' && (
        <div className="card shadow-sm">
          <div className="card-header py-2 fw-semibold">Per-Patient Video Monitoring Summary ({patients.length} patients)</div>
          <div className="card-body p-0">
            <div className="table-responsive">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light">
                  <tr>
                    <th>Patient</th>
                    <th>Age/Sex</th>
                    <th>Events</th>
                    <th>Seizures</th>
                    <th>Automatisms</th>
                    <th>Falls</th>
                    <th>Risk</th>
                    <th>Conf</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {patients.map((pt, i) => (
                    <>
                      <tr key={i} style={{ cursor: 'pointer' }}
                        onClick={() => setExpandedPt(expandedPt === i ? null : i)}>
                        <td className="small fw-semibold">{pt.patient_id}</td>
                        <td className="small">{pt.age} / {pt.sex?.[0]}</td>
                        <td className="small">{pt.total_video_events}</td>
                        <td className="small text-danger">{pt.seizure_events}</td>
                        <td className="small text-info">{pt.automatism_events}</td>
                        <td className="small text-warning">{pt.fall_alerts}</td>
                        <td>
                          <span className={`badge bg-${RISK_COLOR[pt.fall_risk_level] || 'secondary'}`}>
                            {pt.fall_risk_level}
                          </span>
                        </td>
                        <td className="small">{pt.avg_confidence ? `${Math.round(pt.avg_confidence * 100)}%` : '—'}</td>
                        <td className="small text-muted">{expandedPt === i ? '▲' : '▼'}</td>
                      </tr>
                      {expandedPt === i && pt.events && pt.events.length > 0 && (
                        <tr key={`${i}-detail`}>
                          <td colSpan={9} className="bg-light p-2">
                            <div className="small fw-semibold mb-1">Event Log</div>
                            <table className="table table-sm table-bordered mb-0">
                              <thead>
                                <tr>
                                  <th>ID</th>
                                  <th>Motor Pattern</th>
                                  <th>Confidence</th>
                                  <th>Duration</th>
                                  <th>Fall</th>
                                </tr>
                              </thead>
                              <tbody>
                                {pt.events.slice(0, 10).map((ev, ei) => (
                                  <tr key={ei}>
                                    <td>{ev.event_id}</td>
                                    <td>
                                      <span className={`badge bg-${PAT_COLOR[ev.pattern_key] || 'secondary'}`}>
                                        {ev.motor_pattern}
                                      </span>
                                    </td>
                                    <td>{ev.confidence ? `${Math.round(ev.confidence * 100)}%` : '—'}</td>
                                    <td>{ev.duration_sec ? `${ev.duration_sec}s` : '—'}</td>
                                    <td>{ev.fall_alert ? <span className="badge bg-danger">Yes</span> : <span className="badge bg-success">No</span>}</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                            {pt.events.length > 10 && (
                              <div className="text-muted small mt-1">Showing 10 of {pt.events.length} events.</div>
                            )}
                          </td>
                        </tr>
                      )}
                    </>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ── Definitions ── */}
      {tab === 'definitions' && defs && (
        <>
          <div className="card shadow-sm mb-3">
            <div className="card-header py-2 fw-semibold">{defs.title}</div>
            <div className="card-body">
              <p className="small text-muted">{defs.description}</p>
            </div>
          </div>
          <div className="card shadow-sm mb-3">
            <div className="card-header py-2 fw-semibold">Motor Pattern Glossary</div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm mb-0">
                  <thead className="table-light">
                    <tr><th>Pattern</th><th>Description</th><th>Fall Risk</th></tr>
                  </thead>
                  <tbody>
                    {(defs.motor_patterns || []).map((p, i) => (
                      <tr key={i}>
                        <td className="small fw-semibold">
                          <span className={`badge bg-${PAT_COLOR[p.key] || 'secondary'} me-1`}>{p.label}</span>
                        </td>
                        <td className="small">{p.description}</td>
                        <td className="small">{Math.round((p.fall_risk || 0) * 100)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="card shadow-sm mb-3">
            <div className="card-header py-2 fw-semibold">Detection Models</div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm mb-0">
                  <thead className="table-light">
                    <tr><th>Model</th><th>Type</th><th>Task</th></tr>
                  </thead>
                  <tbody>
                    {(defs.models || []).map((m, i) => (
                      <tr key={i}>
                        <td className="small fw-semibold">{m.name}</td>
                        <td className="small">{m.type}</td>
                        <td className="small">{m.task}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          {defs.fall_detection_criteria && (
            <div className="card shadow-sm mb-3">
              <div className="card-header py-2 fw-semibold">Fall Detection Criteria</div>
              <div className="card-body">
                <p className="small mb-1"><strong>Definition:</strong> {defs.fall_detection_criteria.definition}</p>
                <p className="small mb-1"><strong>Triggers:</strong> {(defs.fall_detection_criteria.triggers || []).join(' · ')}</p>
                <p className="small mb-0"><strong>Alert threshold:</strong> {defs.fall_detection_criteria.alert_threshold}</p>
              </div>
            </div>
          )}
          {defs.references && defs.references.length > 0 && (
            <div className="card shadow-sm">
              <div className="card-header py-2 fw-semibold">References</div>
              <div className="card-body">
                <ul className="mb-0 small">
                  {defs.references.map((r, i) => <li key={i}>{r}</li>)}
                </ul>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
