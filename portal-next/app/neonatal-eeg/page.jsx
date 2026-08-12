'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const RISK_COLOR = r => {
  if (!r) return 'secondary';
  const l = r.toLowerCase();
  if (l.includes('high')) return 'danger';
  if (l.includes('moderate')) return 'warning';
  if (l.includes('low')) return 'success';
  return 'secondary';
};

const PROGNOSIS_COLOR = p => {
  if (!p) return 'secondary';
  const l = p.toLowerCase();
  if (l.includes('poor') || l.includes('bad')) return 'danger';
  if (l.includes('variable') || l.includes('uncertain')) return 'warning';
  if (l.includes('good') || l.includes('favourable')) return 'success';
  return 'secondary';
};

export default function NeonatalEEGPage() {
  const [ov, setOv]     = useState(null);
  const [bd, setBd]     = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab]   = useState('overview');
  const [expandedProfile, setExpandedProfile] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/neonatal-eeg/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/neonatal-eeg/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/neonatal-eeg/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const kpi = ov.kpis || {};
  const gaDist = ov.gestational_age_distribution || [];
  const bgChars = ov.eeg_background_characteristics || [];
  const seizureTypes = ov.neonatal_seizure_types || [];
  const differences = ov.differences_from_adult_eeg || [];
  const detBench = ov.detection_performance_benchmark || [];
  const modelReady = ov.model_readiness || {};

  const gaProfiles = (bd || {}).gestational_age_pattern_profiles || [];
  const etiologies = (bd || {}).seizure_etiology_distribution || [];
  const bgClasses  = (bd || {}).eeg_background_classification || [];
  const detPerf    = (bd || {}).detection_performance_comparison || {};
  const montage    = (bd || {}).montage_comparison || {};
  const aeeg       = (bd || {}).aeeg_patterns || {};
  const defSections = (defs || {}).sections || [];

  const tabs = [
    { id: 'overview',    label: 'Overview' },
    { id: 'ga_profiles', label: 'GA Profiles' },
    { id: 'seizures',    label: 'Seizures & Etiology' },
    { id: 'detection',   label: 'AI Detection' },
    { id: 'definitions', label: 'Definitions' },
  ];

  return (
    <div>
      <h3>&#x1f476; Neonatal EEG Dashboard</h3>
      <p className="text-muted">
        Helsinki Neonatal EEG Dataset — 79 neonates (28&ndash;44 wk GA), 22-channel EEG, 256 Hz,
        460 annotated seizure events. Stevenson et al. 2019, <em>Scientific Data</em>.
      </p>

      {/* KPI cards */}
      <div className="row mb-3">
        {[
          { label: 'Total Neonates',        value: kpi.total_neonates,               color: 'primary' },
          { label: 'Seizure-Positive',       value: kpi.seizure_positive_neonates,    color: 'danger' },
          { label: 'Seizure-Negative',       value: kpi.seizure_negative_neonates,    color: 'success' },
          { label: 'Seizure Events',         value: kpi.total_seizure_events_annotated, color: 'warning' },
          { label: 'Median Duration (h)',    value: kpi.median_recording_duration_hours, color: 'info' },
          { label: 'Inter-rater \u03ba',    value: kpi.annotation_agreement_kappa,   color: 'secondary' },
        ].map(c => (
          <div key={c.label} className="col-6 col-md mb-2">
            <div className="card text-center shadow-sm border-0">
              <div className="card-body py-2">
                <div className={`h3 mb-0 text-${c.color}`}>{c.value ?? '\u2014'}</div>
                <div className="text-muted small">{c.label}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Dataset meta pill row */}
      <div className="mb-3 d-flex flex-wrap gap-2">
        <span className="badge bg-secondary fs-6">{kpi.channels} channels</span>
        <span className="badge bg-secondary fs-6">{kpi.sampling_rate_hz} Hz</span>
        <span className="badge bg-secondary fs-6">GA {kpi.gestational_age_range_weeks} wk</span>
        <span className="badge bg-secondary fs-6">{kpi.recording_site}</span>
        <span className="badge bg-info text-dark fs-6">Published {kpi.dataset_published_year}</span>
        {kpi.dataset_doi && (
          <span className="badge bg-light text-dark border fs-6">DOI: {kpi.dataset_doi}</span>
        )}
      </div>

      {/* Model readiness banner */}
      <div className={`alert alert-${modelReady.status === 'gap' ? 'warning' : 'success'} mb-3 py-2`}>
        <strong>Model Readiness:</strong>{' '}
        {modelReady.status === 'gap' ? (
          <>
            <span className="badge bg-warning text-dark me-2">GAP</span>
            {modelReady.next_step}
            {modelReady.transfer_learning_candidate && (
              <span className="ms-2 text-muted small">
                (Transfer from: <code>{modelReady.transfer_learning_candidate}</code>)
              </span>
            )}
          </>
        ) : (
          <span className="badge bg-success">Ready</span>
        )}
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {tabs.map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={`nav-link${tab === t.id ? ' active' : ''}`}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* ── Overview Tab ── */}
      {tab === 'overview' && (
        <div>
          {/* Gestational age distribution */}
          <div className="card mb-3 shadow-sm border-0">
            <div className="card-body">
              <h6 className="card-title">Gestational Age Distribution</h6>
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light">
                    <tr>
                      <th>Group</th><th>GA Range</th><th>n</th><th>%</th><th>EEG Note</th>
                    </tr>
                  </thead>
                  <tbody>
                    {gaDist.map((g, i) => (
                      <tr key={i}>
                        <td><strong>{g.group}</strong></td>
                        <td>{g.ga_range}</td>
                        <td>{g.n}</td>
                        <td>
                          <div className="d-flex align-items-center gap-2">
                            <div className="progress flex-grow-1" style={{ height: 8 }}>
                              <div
                                className="progress-bar bg-primary"
                                style={{ width: `${g.pct}%` }}
                              />
                            </div>
                            <small>{g.pct}%</small>
                          </div>
                        </td>
                        <td className="text-muted small">{g.note}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* EEG Background Characteristics */}
          <div className="card mb-3 shadow-sm border-0">
            <div className="card-body">
              <h6 className="card-title">EEG Background Patterns — Clinical Significance</h6>
              <div className="row">
                {bgChars.map((b, i) => (
                  <div key={i} className="col-12 col-md-6 mb-2">
                    <div className="card border-start border-4 border-primary h-100">
                      <div className="card-body py-2">
                        <div className="fw-semibold">{b.pattern}</div>
                        <div className="text-muted small mb-1">{b.description}</div>
                        <div className="small"><strong>Clinical:</strong> {b.clinical_significance}</div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Adult vs Neonatal EEG Differences */}
          {differences.length > 0 && (
            <div className="card mb-3 shadow-sm border-0">
              <div className="card-body">
                <h6 className="card-title">Key Differences: Neonatal vs Adult EEG</h6>
                <div className="table-responsive">
                  <table className="table table-sm table-hover mb-0">
                    <thead className="table-light">
                      <tr>
                        <th>Feature</th>
                        <th>Adult EEG</th>
                        <th>Neonatal EEG</th>
                        <th>Clinical Impact</th>
                      </tr>
                    </thead>
                    <tbody>
                      {differences.map((d, i) => (
                        <tr key={i}>
                          <td><strong>{d.feature}</strong></td>
                          <td>{d.adult}</td>
                          <td>{d.neonatal}</td>
                          <td className="text-muted small">{d.clinical_impact}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── GA Profiles Tab ── */}
      {tab === 'ga_profiles' && (
        <div>
          {gaProfiles.map((p, i) => (
            <div key={i} className="card mb-3 shadow-sm border-0">
              <div
                className="card-header d-flex justify-content-between align-items-center"
                style={{ cursor: 'pointer' }}
                onClick={() => setExpandedProfile(expandedProfile === i ? null : i)}
              >
                <span>
                  <strong>{p.group}</strong>
                  <span className="text-muted ms-2">({p.ga_range})</span>
                </span>
                <div className="d-flex align-items-center gap-2">
                  <span className={`badge bg-${RISK_COLOR(p.seizure_risk)}`}>
                    Seizure risk: {p.seizure_risk}
                  </span>
                  <span>{expandedProfile === i ? '▲' : '▼'}</span>
                </div>
              </div>
              {expandedProfile === i && (
                <div className="card-body">
                  <div className="row mb-2">
                    <div className="col-md-6">
                      <p className="mb-1"><strong>Background:</strong> {p.background}</p>
                      <p className="mb-1"><strong>Delta brushes:</strong> {p.delta_brushes}</p>
                      <p className="mb-1"><strong>Amplitude:</strong> {p.amplitude_range_uv} µV</p>
                      <p className="mb-1"><strong>Typical IBI:</strong> {p.typical_ibi_sec} s</p>
                    </div>
                    <div className="col-md-6">
                      <strong>Dominant EEG Features:</strong>
                      <ul className="mb-0 mt-1">
                        {(p.dominant_features || []).map((f, j) => (
                          <li key={j} className="small">{f}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </div>
              )}
            </div>
          ))}

          {/* EEG Background Classification */}
          <div className="card mb-3 shadow-sm border-0">
            <div className="card-body">
              <h6 className="card-title">EEG Background Classification (Helsinki Cohort)</h6>
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light">
                    <tr>
                      <th>Class</th><th>% in Helsinki</th><th>Prognosis</th><th>Description</th>
                    </tr>
                  </thead>
                  <tbody>
                    {bgClasses.map((c, i) => (
                      <tr key={i}>
                        <td><strong>{c.class}</strong></td>
                        <td>
                          <div className="d-flex align-items-center gap-2">
                            <div className="progress flex-grow-1" style={{ height: 8 }}>
                              <div
                                className="progress-bar bg-info"
                                style={{ width: `${c.pct_in_helsinki}%` }}
                              />
                            </div>
                            <small>{c.pct_in_helsinki}%</small>
                          </div>
                        </td>
                        <td>
                          <span className={`badge bg-${PROGNOSIS_COLOR(c.prognosis_category)}`}>
                            {c.prognosis_category}
                          </span>
                        </td>
                        <td className="text-muted small">{c.description}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* aEEG Patterns */}
          {aeeg.description && (
            <div className="card mb-3 shadow-sm border-0">
              <div className="card-body">
                <h6 className="card-title">aEEG (Amplitude-Integrated EEG)</h6>
                <p className="text-muted small mb-2">{aeeg.description}</p>
                <div className="row">
                  <div className="col-md-6">
                    <strong>Background Pattern Classes:</strong>
                    <ul className="mt-1">
                      {(aeeg.background_pattern_classes || []).map((c, i) => (
                        <li key={i} className="small">{c}</li>
                      ))}
                    </ul>
                  </div>
                  <div className="col-md-6">
                    <p className="mb-1 small"><strong>Seizure Appearance:</strong> {aeeg.seizure_appearance}</p>
                    <p className="mb-1 small">
                      <strong>Seizure Sensitivity:</strong>{' '}
                      <span className="badge bg-warning text-dark">{aeeg.sensitivity_for_seizure_pct}%</span>
                    </p>
                    {aeeg.note && <p className="mb-0 text-muted small">{aeeg.note}</p>}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Seizures & Etiology Tab ── */}
      {tab === 'seizures' && (
        <div>
          {/* Neonatal Seizure Types */}
          <div className="card mb-3 shadow-sm border-0">
            <div className="card-body">
              <h6 className="card-title">Neonatal Seizure Types</h6>
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light">
                    <tr>
                      <th>Type</th><th>% of Neonatal Sz</th><th>EEG Correlate</th><th>Note</th>
                    </tr>
                  </thead>
                  <tbody>
                    {seizureTypes.map((s, i) => (
                      <tr key={i}>
                        <td><strong>{s.type}</strong></td>
                        <td>
                          {s.pct_of_neonatal_seizures != null ? (
                            <div className="d-flex align-items-center gap-2">
                              <div className="progress flex-grow-1" style={{ height: 8 }}>
                                <div
                                  className="progress-bar bg-danger"
                                  style={{ width: `${s.pct_of_neonatal_seizures}%` }}
                                />
                              </div>
                              <small>{s.pct_of_neonatal_seizures}%</small>
                            </div>
                          ) : '—'}
                        </td>
                        <td className="small">{s.eeg_correlate}</td>
                        <td className="text-muted small">{s.note}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Seizure Etiology */}
          <div className="card mb-3 shadow-sm border-0">
            <div className="card-body">
              <h6 className="card-title">Seizure Etiology Distribution (Helsinki Cohort)</h6>
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light">
                    <tr>
                      <th>Etiology</th><th>%</th><th>EEG Pattern</th><th>Prognosis</th>
                    </tr>
                  </thead>
                  <tbody>
                    {etiologies.map((e, i) => (
                      <tr key={i}>
                        <td><strong>{e.etiology}</strong></td>
                        <td>
                          <div className="d-flex align-items-center gap-2">
                            <div className="progress flex-grow-1" style={{ height: 8 }}>
                              <div
                                className="progress-bar bg-warning"
                                style={{ width: `${e.pct}%` }}
                              />
                            </div>
                            <small>{e.pct}%</small>
                          </div>
                        </td>
                        <td className="small">{e.eeg_pattern}</td>
                        <td>
                          <span className={`badge bg-${PROGNOSIS_COLOR(e.prognosis)}`}>
                            {e.prognosis}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── AI Detection Tab ── */}
      {tab === 'detection' && (
        <div>
          {/* Detection Performance Benchmark */}
          {detBench.length > 0 && (
            <div className="card mb-3 shadow-sm border-0">
              <div className="card-body">
                <h6 className="card-title">AI Detection Performance Benchmark</h6>
                <div className="table-responsive">
                  <table className="table table-sm table-hover mb-0">
                    <thead className="table-light">
                      <tr>
                        <th>Model</th><th>Sensitivity</th><th>Specificity</th><th>AUC</th>
                        <th>False Alarm Rate</th><th>Note</th>
                      </tr>
                    </thead>
                    <tbody>
                      {detBench.map((b, i) => (
                        <tr key={i}>
                          <td><strong>{b.model}</strong></td>
                          <td>
                            <span className={`badge bg-${b.sensitivity_pct >= 80 ? 'success' : b.sensitivity_pct >= 60 ? 'warning' : 'danger'}`}>
                              {b.sensitivity_pct}%
                            </span>
                          </td>
                          <td>{b.specificity_pct != null ? `${b.specificity_pct}%` : '—'}</td>
                          <td>{b.auc != null ? b.auc : '—'}</td>
                          <td>{b.false_alarm_rate_per_h != null ? `${b.false_alarm_rate_per_h}/h` : '—'}</td>
                          <td className="text-muted small">{b.note}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {/* Detection Performance Comparison */}
          {detPerf.neonatal_model && (
            <div className="card mb-3 shadow-sm border-0">
              <div className="card-body">
                <h6 className="card-title">Neonatal vs Adult Model Performance Gap</h6>
                <div className="row">
                  <div className="col-md-4">
                    <div className="card border-success h-100">
                      <div className="card-header bg-success text-white py-1">Neonatal-Specific Model</div>
                      <div className="card-body small">
                        {Object.entries(detPerf.neonatal_model).map(([k, v]) => (
                          <div key={k}><strong>{k}:</strong> {String(v)}</div>
                        ))}
                      </div>
                    </div>
                  </div>
                  <div className="col-md-4">
                    <div className="card border-warning h-100">
                      <div className="card-header bg-warning py-1">Adult Model on Neonatal EEG</div>
                      <div className="card-body small">
                        {Object.entries(detPerf.adult_model || {}).map(([k, v]) => (
                          <div key={k}><strong>{k}:</strong> {String(v)}</div>
                        ))}
                      </div>
                    </div>
                  </div>
                  <div className="col-md-4">
                    <div className="card border-danger h-100">
                      <div className="card-header bg-danger text-white py-1">Performance Gap</div>
                      <div className="card-body small">
                        {Object.entries(detPerf.performance_gap || {}).map(([k, v]) => (
                          <div key={k}><strong>{k}:</strong> {String(v)}</div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Montage Comparison */}
          {montage.neonatal_modified && (
            <div className="card mb-3 shadow-sm border-0">
              <div className="card-body">
                <h6 className="card-title">Montage Comparison: Neonatal vs Adult</h6>
                <div className="row">
                  <div className="col-md-6">
                    <div className="card border-primary h-100">
                      <div className="card-header bg-primary text-white py-1">Neonatal Modified Montage</div>
                      <div className="card-body small">
                        {Object.entries(montage.neonatal_modified).map(([k, v]) => (
                          <div key={k}><strong>{k}:</strong> {String(v)}</div>
                        ))}
                      </div>
                    </div>
                  </div>
                  <div className="col-md-6">
                    <div className="card border-secondary h-100">
                      <div className="card-header bg-secondary text-white py-1">Adult Standard (10-20)</div>
                      <div className="card-body small">
                        {Object.entries(montage.adult_standard || {}).map(([k, v]) => (
                          <div key={k}><strong>{k}:</strong> {String(v)}</div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
                {montage.spatial_resolution_impact && (
                  <div className="alert alert-info mt-2 py-2 mb-0 small">
                    <strong>Spatial Resolution Impact:</strong> {montage.spatial_resolution_impact}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Definitions Tab ── */}
      {tab === 'definitions' && (
        <div>
          {defSections.map((sec, i) => (
            <div key={i} className="card mb-3 shadow-sm border-0">
              <div className="card-header bg-light"><strong>{sec.title}</strong></div>
              <div className="card-body p-0">
                <table className="table table-sm table-hover mb-0">
                  <tbody>
                    {(sec.items || []).map((item, j) => (
                      <tr key={j}>
                        <td style={{ width: '30%' }} className="fw-semibold text-primary">{item.term}</td>
                        <td className="small text-muted">{item.definition}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ))}

          {/* References */}
          <div className="card border-0 shadow-sm mt-2">
            <div className="card-body">
              <h6>References</h6>
              <ul className="mb-0 small">
                <li>Stevenson NJ et al. &ldquo;A dataset of neonatal EEG recordings with seizure annotations.&rdquo; <em>Sci Data</em> 6, 190039 (2019). DOI: 10.1038/s41597-019-0109-9</li>
                <li>Glass HC. &ldquo;Neonatal seizures: advances in mechanisms and management.&rdquo; <em>Clin Perinatol</em> 41(1): 177-190 (2014).</li>
                <li>Shellhaas RA et al. &ldquo;The American Clinical Neurophysiology Society&rsquo;s Guideline on Continuous Electroencephalography Monitoring in Neonates.&rdquo; <em>J Clin Neurophysiol</em> 28(6): 611-617 (2011).</li>
                <li>Pressler RM et al. &ldquo;Treatment of seizures in the neonate.&rdquo; <em>Epilepsia</em> 64(10): 2511-2543 (2023). ILAE guidelines.</li>
                <li>Neonatal Seizure Detection Challenge 2016 — benchmark reference for automated neonatal EEG detection algorithms.</li>
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
