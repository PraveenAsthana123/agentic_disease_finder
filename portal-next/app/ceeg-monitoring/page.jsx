'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-4 col-lg-2 mb-2">
      <div className={`card border-${color || 'primary'} text-center h-100`}>
        <div className="card-body py-2 px-1">
          <div className={`h4 fw-bold mb-0 text-${color || 'primary'}`}>{value ?? '—'}</div>
          <div className="small text-muted">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.68rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function Bar({ items, labelKey = 'label', countKey = 'count', colorClass = 'primary' }) {
  const mx = Math.max(...(items || []).map(i => i[countKey] || 0), 1);
  return (
    <div>
      {(items || []).map((it, i) => {
        const val = it[countKey] ?? 0;
        const label = it[labelKey] || '?';
        const pct = Math.round((val / mx) * 100);
        return (
          <div key={i} className="d-flex align-items-center mb-1 gap-2">
            <div className="text-end small text-muted" style={{ width: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontSize: '0.75rem' }}>
              {label}
            </div>
            <div className="flex-grow-1">
              <div className="progress" style={{ height: 16 }}>
                <div className={`progress-bar bg-${colorClass}`} style={{ width: `${pct}%` }}>
                  <span className="small px-1">{val}</span>
                </div>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function CriticalBadge({ critical }) {
  return critical
    ? <span className="badge bg-danger">Critical</span>
    : <span className="badge bg-secondary">Normal</span>;
}

function BoolBadge({ val, trueLabel = 'Yes', falseLabel = 'No', trueColor = 'success', falseColor = 'secondary' }) {
  return <span className={`badge bg-${val ? trueColor : falseColor}`}>{val ? trueLabel : falseLabel}</span>;
}

export default function CeegMonitoring() {
  const [tab, setTab] = useState('overview');
  const [ov, setOv] = useState(null);
  const [bk, setBk] = useState(null);
  const [df, setDf] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    const load = async (path, setter) => {
      try {
        const r = await fetch(`${API}${path}`);
        if (!r.ok) throw new Error(`${r.status}`);
        setter(await r.json());
      } catch (e) { setErr(e.message); }
    };
    load('/api/ceeg-monitoring/overview', setOv);
    load('/api/ceeg-monitoring/breakdown', setBk);
    load('/api/ceeg-monitoring/definitions', setDf);
  }, []);

  const kpis = ov?.kpis || {};

  return (
    <div className="container-fluid py-3">
      <h3 className="fw-bold mb-1">📡 Continuous EEG (cEEG) ICU Monitoring</h3>
      <p className="text-muted small mb-3">
        Long-term EEG · Non-convulsive seizure detection · ICU critical patterns · ACNS 2021 nomenclature · real clinical.db data
      </p>

      {err && <div className="alert alert-danger small">API error: {err}</div>}

      {/* Tab nav */}
      <ul className="nav nav-tabs mb-3">
        {['overview', 'recordings', 'icu', 'artifacts', 'definitions'].map(t => (
          <li key={t} className="nav-item">
            <button className={`nav-link ${tab === t ? 'active' : ''}`} onClick={() => setTab(t)}>
              {t === 'overview' ? '📊 Overview'
                : t === 'recordings' ? '🎵 Recordings'
                : t === 'icu' ? '🏥 ICU Patients'
                : t === 'artifacts' ? '⚡ Artifacts'
                : '📖 Definitions'}
            </button>
          </li>
        ))}
      </ul>

      {/* ── Overview ── */}
      {tab === 'overview' && (
        <div>
          <div className="row g-2 mb-3">
            <KPI label="Total EEG Recordings" value={kpis.total_recordings} color="primary" />
            <KPI label="Long-Term (LTM)" value={kpis.continuous_ltm_recordings} color="info" sub="Continuous ≥8 h" />
            <KPI label="Ambulatory cEEG" value={kpis.ambulatory_recordings} color="primary" />
            <KPI label="ICU Admissions" value={kpis.icu_admissions} color="danger" />
            <KPI label="Avg Session" value={kpis.avg_monitoring_hours != null ? `${kpis.avg_monitoring_hours}h` : '—'} color="secondary" />
            <KPI label="LTM Avg Duration" value={kpis.ltm_avg_hours != null ? `${kpis.ltm_avg_hours}h` : '—'} color="info" />
            <KPI label="Max Session" value={kpis.max_session_hours != null ? `${kpis.max_session_hours}h` : '—'} color="warning" />
            <KPI label="Total Monitoring" value={kpis.total_monitoring_hours != null ? `${kpis.total_monitoring_hours}h` : '—'} color="dark" />
            <KPI label="Critical EEG Patterns" value={kpis.critical_eeg_patterns} color="danger" sub="PLDs/hypsarr/SW<3Hz" />
            <KPI label="PLD Cases" value={kpis.pld_cases} color="danger" sub="Periodic lateralized" />
            <KPI label="ICU Seizure-Free Dc" value={kpis.icu_seizure_free_discharge} color="success" />
            <KPI label="ICU Readmit 30d" value={kpis.icu_readmission_30d} color="warning" />
          </div>

          <div className="row g-3">
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-bold small">📻 Recording Type Distribution</div>
                <div className="card-body">
                  <Bar
                    items={(ov?.recording_type_distribution || []).map(i => ({ label: i.type, count: i.count }))}
                    colorClass="primary"
                  />
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-bold small">🔬 EEG Pattern Landscape</div>
                <div className="card-body">
                  {(ov?.eeg_pattern_landscape || []).map((p, i) => (
                    <div key={i} className="d-flex align-items-center justify-content-between mb-1">
                      <span className="small text-muted" style={{ maxWidth: '65%', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.pattern}</span>
                      <div className="d-flex align-items-center gap-1">
                        <span className={`badge bg-${p.critical ? 'danger' : 'secondary'}`}>{p.count}</span>
                        {p.critical && <span className="badge bg-danger" style={{ fontSize: '0.6rem' }}>⚠</span>}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            <div className="col-md-4">
              <div className="card h-100">
                <div className="card-header fw-bold small">📡 Sampling Rate Distribution</div>
                <div className="card-body">
                  <Bar
                    items={(ov?.sampling_rate_distribution || []).map(i => ({ label: i.rate_hz, count: i.count }))}
                    colorClass="info"
                  />
                </div>
              </div>
            </div>
            <div className="col-md-4">
              <div className="card h-100">
                <div className="card-header fw-bold small">🔗 Montage Distribution</div>
                <div className="card-body">
                  <Bar
                    items={(ov?.montage_distribution || []).map(i => ({ label: i.montage, count: i.count }))}
                    colorClass="success"
                  />
                </div>
              </div>
            </div>
            <div className="col-md-4">
              <div className="card h-100">
                <div className="card-header fw-bold small">🏥 ICU Admission Reasons</div>
                <div className="card-body">
                  <Bar
                    items={(ov?.icu_admission_breakdown || []).map(i => ({ label: i.reason, count: i.count }))}
                    colorClass="danger"
                  />
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-bold small">⚡ Artifact Burden by Type</div>
                <div className="card-body">
                  <Bar
                    items={(ov?.artifact_burden || []).map(i => ({ label: i.artifact_type, count: i.count }))}
                    colorClass="warning"
                  />
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card h-100 border-warning">
                <div className="card-header fw-bold small text-warning">⚠️ Artifact Severity Summary</div>
                <div className="card-body">
                  {ov?.severity_summary && (
                    <div className="row text-center">
                      <div className="col-4">
                        <div className="h4 fw-bold text-danger">{ov.severity_summary.severe}</div>
                        <div className="small text-muted">Severe</div>
                      </div>
                      <div className="col-4">
                        <div className="h4 fw-bold text-warning">{ov.severity_summary.moderate}</div>
                        <div className="small text-muted">Moderate</div>
                      </div>
                      <div className="col-4">
                        <div className="h4 fw-bold text-secondary">{ov.severity_summary.mild}</div>
                        <div className="small text-muted">Mild</div>
                      </div>
                      <div className="col-12 mt-2">
                        <div className="small text-muted">Total artifacts: {ov.severity_summary.total_artifacts}</div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Recordings ── */}
      {tab === 'recordings' && (
        <div>
          <div className="row g-3 mb-3">
            <div className="col-md-6">
              <div className="card">
                <div className="card-header fw-bold small">⏱️ Session Duration Histogram</div>
                <div className="card-body">
                  <Bar
                    items={(bk?.duration_histogram || []).map(i => ({ label: i.bucket, count: i.count }))}
                    colorClass="info"
                  />
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card border-info">
                <div className="card-header fw-bold small text-info">💡 ACNS 2021 Duration Guidance</div>
                <div className="card-body">
                  {(df?.monitoring_duration_guidance || []).map((g, i) => (
                    <div key={i} className="mb-2">
                      <span className="badge bg-info me-2">{g.duration}</span>
                      <span className="small text-muted">{g.recommendation}</span>
                    </div>
                  ))}
                  {!df && <div className="text-muted small">Loading definitions…</div>}
                </div>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card-header fw-bold small">📋 Per-Patient Recording Table (sorted by duration)</div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0 small">
                  <thead className="table-dark">
                    <tr>
                      <th>Patient</th>
                      <th>Type</th>
                      <th>Duration (h)</th>
                      <th>Tier</th>
                      <th>Sampling</th>
                      <th>Montage</th>
                      <th>EEG Pattern</th>
                      <th>Critical</th>
                      <th>DRE</th>
                      <th>Onset Zone</th>
                      <th>Study Date</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(bk?.per_patient_recordings || []).map((r, i) => (
                      <tr key={i}>
                        <td className="fw-bold">{r.patient_id}</td>
                        <td><span className="badge bg-primary">{r.recording_type}</span></td>
                        <td>{r.duration_h}</td>
                        <td className="text-muted" style={{ fontSize: '0.7rem' }}>{r.monitoring_tier}</td>
                        <td>{r.sampling_rate}</td>
                        <td>{r.montage}</td>
                        <td className="text-muted" style={{ fontSize: '0.7rem', maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.eeg_pattern || '—'}</td>
                        <td><CriticalBadge critical={r.critical_pattern} /></td>
                        <td><BoolBadge val={r.drug_resistant} trueLabel="DRE" falseLabel="—" trueColor="danger" falseColor="light" /></td>
                        <td>{r.onset_zone || '—'}</td>
                        <td>{r.study_date || '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── ICU Patients ── */}
      {tab === 'icu' && (
        <div>
          <div className="alert alert-danger small fw-bold mb-3">
            🏥 {bk?.icu_patients?.length ?? 0} ICU admissions — cEEG mandatory for ≥ 24 h per ACNS 2021 in altered-consciousness patients.
            {bk?.drug_resistant_in_icu && (
              <span className="ms-2">Drug-resistant epilepsy: {bk.drug_resistant_in_icu.count}/{bk.drug_resistant_in_icu.total} ICU patients.</span>
            )}
          </div>
          <div className="card">
            <div className="card-header fw-bold small">🏥 ICU Patient Detail Table (sorted by LOS)</div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0 small">
                  <thead className="table-dark">
                    <tr>
                      <th>Patient</th>
                      <th>Age</th>
                      <th>Admission Reason</th>
                      <th>LOS (d)</th>
                      <th>EEG Pattern</th>
                      <th>Critical</th>
                      <th>DRE</th>
                      <th>Seizure-Free Dc</th>
                      <th>Readmit 30d</th>
                      <th>Disposition</th>
                      <th>Cost ($)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(bk?.icu_patients || []).map((r, i) => (
                      <tr key={i}>
                        <td className="fw-bold">{r.patient_id}</td>
                        <td>{r.age ?? '—'}</td>
                        <td>{r.admission_reason}</td>
                        <td>{r.los_days}</td>
                        <td className="text-muted" style={{ fontSize: '0.7rem', maxWidth: 150, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.eeg_pattern || '—'}</td>
                        <td><CriticalBadge critical={r.critical_pattern} /></td>
                        <td><BoolBadge val={r.drug_resistant} trueLabel="DRE" falseLabel="—" trueColor="danger" falseColor="light" /></td>
                        <td><BoolBadge val={r.seizure_free} trueLabel="Yes" falseLabel="No" /></td>
                        <td><BoolBadge val={r.readmit_30d} trueLabel="Yes" falseLabel="No" trueColor="warning" falseColor="success" /></td>
                        <td>{r.discharge_disposition}</td>
                        <td>{r.cost_usd ? r.cost_usd.toLocaleString() : '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Artifacts ── */}
      {tab === 'artifacts' && (
        <div>
          <div className="alert alert-warning small mb-3">
            ⚡ ICU artifact management is critical — ventilator rhythmic artifact, ECG contamination, and electrode displacement must be distinguished from real EEG patterns.
          </div>
          <div className="row g-3">
            <div className="col-md-6">
              <div className="card">
                <div className="card-header fw-bold small">⚡ Artifact Severity by Type</div>
                <div className="card-body p-0">
                  <div className="table-responsive">
                    <table className="table table-sm mb-0 small">
                      <thead className="table-dark">
                        <tr>
                          <th>Artifact Type</th>
                          <th>Mild</th>
                          <th>Moderate</th>
                          <th>Severe</th>
                          <th>Total</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(bk?.artifact_severity_table || []).map((a, i) => (
                          <tr key={i}>
                            <td className="fw-bold">{a.type}</td>
                            <td className="text-secondary">{a.mild}</td>
                            <td className="text-warning">{a.moderate}</td>
                            <td className="text-danger">{a.severe}</td>
                            <td className="fw-bold">{a.total}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card">
                <div className="card-header fw-bold small">⚠️ ICU Artifact Challenges</div>
                <div className="card-body">
                  <ul className="small mb-0">
                    {(df?.artifact_challenges || []).map((c, i) => (
                      <li key={i} className="mb-1 text-muted">{c}</li>
                    ))}
                    {!df && <li className="text-muted">Loading…</li>}
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Definitions ── */}
      {tab === 'definitions' && df && (
        <div className="row g-3">
          <div className="col-12">
            <div className="card border-primary">
              <div className="card-header fw-bold small text-primary">📡 {df.term}</div>
              <div className="card-body">
                <p className="small mb-1">{df.definition}</p>
                <div className="text-muted small"><strong>NCSE Prevalence:</strong> {df.ncse_prevalence}</div>
              </div>
            </div>
          </div>

          <div className="col-12">
            <div className="card">
              <div className="card-header fw-bold small">🏥 Monitoring Indications (ACNS 2021)</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0 small">
                  <thead className="table-dark">
                    <tr><th>Indication</th><th>Detail</th><th>Grade</th></tr>
                  </thead>
                  <tbody>
                    {(df.monitoring_indications || []).map((ind, i) => (
                      <tr key={i}>
                        <td className="fw-bold">{ind.indication}</td>
                        <td className="text-muted">{ind.detail}</td>
                        <td><span className={`badge bg-${ind.grade?.includes('Level A') ? 'success' : 'warning'}`}>{ind.grade}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="col-12">
            <div className="card">
              <div className="card-header fw-bold small">🔬 Critical ACNS EEG Patterns</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0 small">
                  <thead className="table-dark">
                    <tr><th>Pattern</th><th>Description</th><th>Risk</th></tr>
                  </thead>
                  <tbody>
                    {(df.acns_patterns || []).map((p, i) => (
                      <tr key={i}>
                        <td className="fw-bold text-danger">{p.pattern}</td>
                        <td className="text-muted">{p.description}</td>
                        <td><span className={`badge bg-${p.risk?.toLowerCase().includes('critical') ? 'danger' : p.risk?.toLowerCase().includes('high') ? 'warning' : 'info'}`}>{p.risk}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="col-md-6">
            <div className="card h-100">
              <div className="card-header fw-bold small">💡 Clinical Pearls</div>
              <div className="card-body">
                <ul className="small mb-0">
                  {(df.clinical_pearls || []).map((p, i) => (
                    <li key={i} className="mb-1 text-muted">{p}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>

          <div className="col-md-6">
            <div className="card h-100">
              <div className="card-header fw-bold small">🔤 Abbreviations</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0 small">
                  <thead className="table-dark"><tr><th>Abbr.</th><th>Meaning</th></tr></thead>
                  <tbody>
                    {Object.entries(df.abbreviations || {}).map(([k, v], i) => (
                      <tr key={i}><td className="fw-bold">{k}</td><td className="text-muted">{v}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="col-12">
            <div className="card">
              <div className="card-header fw-bold small">📚 References</div>
              <div className="card-body">
                <ul className="small mb-0">
                  {(df.references || []).map((r, i) => (
                    <li key={i} className="text-muted">{r}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
