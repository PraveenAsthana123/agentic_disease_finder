'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const sevColor = s => s === 'Mild' ? 'info' : s === 'Moderate' ? 'warning' : s === 'Severe' ? 'danger' : 'secondary';

export default function VideoEEGPage() {
  const [ov, setOv]     = useState(null);
  const [bd, setBd]     = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab]   = useState('overview');
  const [expandedPt, setExpandedPt] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/video-eeg/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/video-eeg/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/video-eeg/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const tabs = [
    { id: 'overview',    label: 'Overview' },
    { id: 'timeline',   label: 'Seizure Timeline' },
    { id: 'eeg',        label: 'EEG Features (CHB-MIT)' },
    { id: 'definitions', label: 'Definitions & Protocol' },
  ];

  const totalPts = ov.per_patient_summary?.length || 1;

  return (
    <div>
      <h3>📹 Video EEG Monitoring Dashboard</h3>
      <p className="text-muted small">{ov.description}</p>

      {/* KPI cards */}
      <div className="row mb-3">
        {[
          { label: 'Sessions',         value: ov.kpis.total_sessions,           color: 'primary' },
          { label: 'Seizures Captured', value: ov.kpis.total_seizures_captured, color: 'danger' },
          { label: 'Mean Duration',    value: `${ov.kpis.mean_duration_sec}s`,  color: 'info' },
          { label: 'Ictal Capture Rate', value: `${ov.kpis.ictal_capture_rate}%`, color: ov.kpis.ictal_capture_rate >= 80 ? 'success' : 'warning' },
          { label: 'Patients Monitored', value: ov.kpis.patients_monitored,     color: 'secondary' },
        ].map(c => (
          <div key={c.label} className="col-6 col-md-2 mb-2">
            <div className="card text-center shadow-sm border-0">
              <div className="card-body py-2">
                <div className={`h3 mb-0 text-${c.color}`}>{c.value}</div>
                <div className="text-muted small">{c.label}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {tabs.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>{t.label}</button>
          </li>
        ))}
      </ul>

      {/* ── Overview Tab ─────────────────────────────────────── */}
      {tab === 'overview' && (
        <div className="row">
          {/* Monitoring Distribution */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Monitoring Outcome Distribution</div>
              <div className="card-body">
                {(ov.monitoring_distribution || []).map((m, i) => (
                  <div key={i} className="d-flex align-items-center mb-2">
                    <span className="small" style={{minWidth: 140}}>{m.name}</span>
                    <div className="flex-grow-1 mx-2">
                      <div className="progress" style={{height: 18}}>
                        <div className={`progress-bar ${m.name === 'Seizure Captured' ? 'bg-danger' : m.name === 'Interictal Only' ? 'bg-warning' : 'bg-success'}`}
                             style={{width: `${totalPts ? (m.value / totalPts * 100) : 0}%`}}>
                          {m.value}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Severity Distribution */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Seizure Severity Distribution</div>
              <div className="card-body">
                {(ov.seizure_severity_distribution || []).map((s, i) => (
                  <div key={i} className="d-flex align-items-center mb-2">
                    <span className={`badge bg-${sevColor(s.name)} me-2`} style={{minWidth: 60}}>{s.name}</span>
                    <div className="flex-grow-1 mx-2">
                      <div className="progress" style={{height: 18}}>
                        <div className={`progress-bar bg-${sevColor(s.name)}`}
                             style={{width: `${ov.kpis.total_sessions ? (s.value / ov.kpis.total_sessions * 100) : 0}%`}}>
                          {s.value}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Per-Patient Summary */}
          <div className="col-md-4 mb-3">
            {bd && (
              <div className="card shadow-sm">
                <div className="card-header fw-bold">Aura Distribution</div>
                <div className="card-body">
                  {(bd.aura_distribution || []).map((a, i) => (
                    <div key={i} className="d-flex align-items-center mb-2">
                      <span className="small" style={{minWidth: 80}}>{a.name}</span>
                      <div className="flex-grow-1 mx-2">
                        <div className="progress" style={{height: 16}}>
                          <div className={`progress-bar ${a.name === 'None' ? 'bg-secondary' : 'bg-warning'}`}
                               style={{width: `${ov.kpis.total_sessions ? (a.value / ov.kpis.total_sessions * 100) : 0}%`}}>
                            {a.value}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Duration Histogram */}
          {bd && (
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-bold">Seizure Duration Histogram</div>
                <div className="card-body">
                  <div className="small text-muted mb-2">Status epilepticus risk: &gt;300s</div>
                  {(bd.duration_histogram || []).map((b, i) => (
                    <div key={i} className="d-flex align-items-center mb-1">
                      <span className="small" style={{minWidth: 80}}>{b.range}</span>
                      <div className="flex-grow-1 mx-2">
                        <div className="progress" style={{height: 18}}>
                          <div className={`progress-bar ${b.range === '300+s' ? 'bg-danger' : b.range === '120-300s' ? 'bg-warning' : 'bg-info'}`}
                               style={{width: `${b.count > 0 ? Math.max(8, b.count / ov.kpis.total_sessions * 100) : 0}%`}}>
                            {b.count}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Temporal Pattern */}
          {bd && (
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-bold">Temporal Pattern (Circadian)</div>
                <div className="card-body">
                  <div className="small text-muted mb-2">Seizure occurrence by 6-hour blocks</div>
                  {(bd.temporal_pattern || []).map((t, i) => (
                    <div key={i} className="d-flex align-items-center mb-1">
                      <span className="small" style={{minWidth: 120}}>{t.hour_block}</span>
                      <div className="flex-grow-1 mx-2">
                        <div className="progress" style={{height: 18}}>
                          <div className="progress-bar bg-primary"
                               style={{width: `${t.count > 0 ? Math.max(8, t.count / ov.kpis.total_sessions * 100) : 0}%`}}>
                            {t.count}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Trigger Analysis */}
          {bd && bd.trigger_analysis && bd.trigger_analysis.length > 0 && (
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-bold">Seizure Trigger Analysis</div>
                <div className="card-body">
                  {bd.trigger_analysis.map((t, i) => (
                    <div key={i} className="d-flex align-items-center mb-1">
                      <span className="small" style={{minWidth: 140}}>{t.trigger}</span>
                      <div className="flex-grow-1 mx-2">
                        <div className="progress" style={{height: 16}}>
                          <div className="progress-bar bg-warning"
                               style={{width: `${Math.max(8, t.count / ov.kpis.total_sessions * 100)}%`}}>
                            {t.count}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Per-Patient Summary Table */}
          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Per-Patient VEM Summary ({ov.per_patient_summary?.length} patients)</div>
              <div className="card-body p-0">
                <div style={{maxHeight: 320, overflowY: 'auto'}}>
                  <table className="table table-sm table-striped mb-0">
                    <thead className="table-dark sticky-top">
                      <tr>
                        <th>Patient</th><th>Sessions</th><th>Seizures</th>
                        <th>Mean Duration</th><th>Longest</th>
                        <th>Aura</th><th>Awareness</th><th>Severity</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(ov.per_patient_summary || []).map((p, i) => (
                        <tr key={i}>
                          <td className="fw-semibold small">{p.patient_id}</td>
                          <td>{p.sessions}</td>
                          <td className={p.seizures_captured > 0 ? 'text-danger fw-bold' : ''}>{p.seizures_captured}</td>
                          <td>{p.mean_duration_sec}s</td>
                          <td className={p.longest_event_sec >= 300 ? 'text-danger fw-bold' : ''}>{p.longest_event_sec}s</td>
                          <td>{p.had_aura ? <span className="badge bg-warning">Yes</span> : <span className="badge bg-secondary">No</span>}</td>
                          <td className="small">{p.awareness_level}</td>
                          <td><span className={`badge bg-${sevColor(p.severity)}`}>{p.severity}</span></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Seizure Timeline Tab ─────────────────────────────── */}
      {tab === 'timeline' && bd && (
        <div>
          {/* Concordance Summary */}
          <div className="row mb-3">
            <div className="col-12">
              <div className="card shadow-sm">
                <div className="card-header fw-bold">Clinical–EEG Concordance</div>
                <div className="card-body p-0">
                  <div style={{maxHeight: 200, overflowY: 'auto'}}>
                    <table className="table table-sm table-striped mb-0">
                      <thead className="table-dark sticky-top">
                        <tr><th>Patient</th><th>Clinical Seizures</th><th>EEG Seizures</th><th>Concordant</th></tr>
                      </thead>
                      <tbody>
                        {(bd.concordance || []).map((c, i) => (
                          <tr key={i}>
                            <td className="fw-semibold small">{c.patient_id}</td>
                            <td>{c.clinical_seizures}</td>
                            <td>{c.eeg_seizures}</td>
                            <td>
                              {c.concordant
                                ? <span className="badge bg-success">Concordant</span>
                                : <span className="badge bg-warning text-dark">Discordant</span>}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Clinical Seizure Timeline */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold">Clinical Seizure Diary — Timeline ({bd.seizure_timeline?.length} events)</div>
            <div className="card-body p-0">
              <div style={{maxHeight: 400, overflowY: 'auto'}}>
                <table className="table table-sm table-striped mb-0">
                  <thead className="table-dark sticky-top">
                    <tr>
                      <th>Patient</th><th>Date</th><th>Time</th>
                      <th>Duration</th><th>Severity</th>
                      <th>Aura</th><th>Awareness</th><th>Motor Signs</th><th>Post-ictal</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(bd.seizure_timeline || []).map((e, i) => (
                      <tr key={i}>
                        <td className="fw-semibold small">{e.patient_id}</td>
                        <td className="small">{e.event_date}</td>
                        <td className="small">{e.event_time}</td>
                        <td className={e.duration_sec >= 300 ? 'text-danger fw-bold' : ''}>{e.duration_sec ? `${e.duration_sec}s` : '—'}</td>
                        <td><span className={`badge bg-${sevColor(e.severity)}`}>{e.severity || '—'}</span></td>
                        <td className="small">{e.aura || '—'}</td>
                        <td className="small">{e.awareness || '—'}</td>
                        <td className="small">{e.motor_signs || '—'}</td>
                        <td className="small">{e.post_ictal || '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── EEG Features Tab ─────────────────────────────────── */}
      {tab === 'eeg' && bd && (
        <div>
          <div className="alert alert-info small mb-3">
            CHB-MIT Scalp EEG Database (Shoeb 2009): continuous recordings from pediatric epilepsy patients
            at Children's Hospital Boston. Expert-annotated seizure onset/offset with 10-20 electrode placement at 256 Hz.
          </div>

          {/* EEG Features Table */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold">EEG Seizure Annotations — CHB-MIT ({bd.eeg_features?.length} events)</div>
            <div className="card-body p-0">
              <div style={{maxHeight: 400, overflowY: 'auto'}}>
                <table className="table table-sm table-striped mb-0">
                  <thead className="table-dark sticky-top">
                    <tr><th>Subject</th><th>File</th><th>Onset (s)</th><th>Offset (s)</th><th>Duration</th><th>Channels Involved</th></tr>
                  </thead>
                  <tbody>
                    {(bd.eeg_features || []).map((f, i) => (
                      <tr key={i}>
                        <td className="fw-semibold small">{f.patient_id}</td>
                        <td className="small text-muted">{f.file}</td>
                        <td>{f.onset_sec}</td>
                        <td>{f.offset_sec}</td>
                        <td className={f.duration_sec >= 300 ? 'text-danger fw-bold' : ''}>{f.duration_sec}s</td>
                        <td>
                          <div style={{maxWidth: 320, fontSize: '0.7rem'}}>
                            {(f.channels_involved || []).map((ch, j) => (
                              <span key={j} className="badge bg-secondary me-1 mb-1" style={{fontSize: '0.65rem'}}>{ch}</span>
                            ))}
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Concordance by CHB subject */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold">Clinical–EEG Concordance Summary</div>
            <div className="card-body">
              {(() => {
                const concorded = (bd.concordance || []).filter(c => c.concordant).length;
                const total = (bd.concordance || []).length;
                const pct = total ? Math.round(concorded / total * 100) : 0;
                return (
                  <div>
                    <div className="d-flex align-items-center mb-2">
                      <span className="me-3">Concordance Rate:</span>
                      <div className="flex-grow-1">
                        <div className="progress" style={{height: 24}}>
                          <div className={`progress-bar ${pct >= 70 ? 'bg-success' : pct >= 50 ? 'bg-warning' : 'bg-danger'}`}
                               style={{width: `${pct}%`}}>
                            {pct}% ({concorded}/{total})
                          </div>
                        </div>
                      </div>
                    </div>
                    <p className="small text-muted">
                      Concordant = patient has seizures in both clinical diary and EEG annotations (or none in both).
                      Discordant may indicate subclinical events, diary under-reporting, or dataset subject mismatch.
                    </p>
                  </div>
                );
              })()}
            </div>
          </div>
        </div>
      )}

      {/* ── Definitions & Protocol Tab ────────────────────────── */}
      {tab === 'definitions' && defs && (
        <div className="row">
          {/* Clinical Significance */}
          <div className="col-12 mb-3">
            <div className="card shadow-sm border-primary">
              <div className="card-header fw-bold bg-primary text-white">Clinical Significance</div>
              <div className="card-body small">{defs.clinical_significance}</div>
            </div>
          </div>

          {/* Monitoring Protocol */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Monitoring Protocol</div>
              <div className="card-body small">
                {defs.monitoring_protocol && Object.entries(defs.monitoring_protocol).map(([k, v]) => (
                  k !== 'indications' ? (
                    <div key={k} className="mb-2">
                      <span className="fw-semibold text-capitalize">{k.replace(/_/g, ' ')}: </span>
                      <span className="text-muted">{v}</span>
                    </div>
                  ) : (
                    <div key={k} className="mb-2">
                      <span className="fw-semibold">Indications:</span>
                      <ul className="mb-0 mt-1">
                        {v.map((ind, i) => <li key={i}>{ind}</li>)}
                      </ul>
                    </div>
                  )
                ))}
              </div>
            </div>
          </div>

          {/* Seizure Semiology */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Seizure Semiology Glossary</div>
              <div className="card-body" style={{maxHeight: 340, overflowY: 'auto'}}>
                {(defs.seizure_semiology || []).map((s, i) => (
                  <div key={i} className="mb-2">
                    <strong className="text-primary small">{s.term}</strong>
                    <div className="small text-muted">{s.description}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Metric Definitions */}
          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Metric Definitions ({(defs.metrics || []).length})</div>
              <div className="card-body p-0">
                <table className="table table-sm table-striped mb-0">
                  <thead className="table-dark">
                    <tr><th>Metric</th><th>Description</th><th>Data Source</th></tr>
                  </thead>
                  <tbody>
                    {(defs.metrics || []).map((m, i) => (
                      <tr key={i}>
                        <td className="fw-semibold small">{m.name}</td>
                        <td className="small">{m.description}</td>
                        <td className="small text-muted">{m.source}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
