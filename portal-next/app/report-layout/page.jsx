'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const COMP_ICONS = {
  acquisition: '📡',
  artifacts: '🧹',
  background: '🌊',
  epileptiform: '⚡',
  explainability: '🔍',
  video: '🎥',
};

const COMP_COLORS = {
  acquisition: 'primary',
  artifacts: 'warning',
  background: 'info',
  epileptiform: 'danger',
  explainability: 'success',
  video: 'secondary',
};

const SECTION_ICONS = {
  AI: '🤖',
  Expert: '👨‍⚕️',
  Final: '✅',
  Audit: '🔒',
};

function KPI({ label, value, color }) {
  return (
    <div className="col-6 col-md-3 mb-2">
      <div className="card shadow-sm border-0 h-100">
        <div className="card-body text-center py-3">
          <div className={`h3 mb-1 text-${color || 'primary'} fw-bold`}>{value}</div>
          <div className="text-muted small">{label}</div>
        </div>
      </div>
    </div>
  );
}

function HBar({ items, colorFn }) {
  if (!items || !items.length) return null;
  const mx = Math.max(...items.map(i => i.value));
  return (
    <div>
      {items.map((it, i) => (
        <div key={i} className="d-flex align-items-center mb-2">
          <div className="text-end small me-2" style={{ width: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {it.name}
          </div>
          <div className="flex-grow-1">
            <div className="progress" style={{ height: 20 }}>
              <div
                className={`progress-bar bg-${colorFn ? colorFn(it.name, i) : 'primary'}`}
                style={{ width: `${mx ? (it.value / mx) * 100 : 0}%` }}
              >
                <span className="small px-1">{it.value}</span>
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

const PALETTE = ['primary', 'info', 'success', 'warning', 'danger', 'secondary'];

export default function ReportLayoutPage() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [vcOv, setVcOv] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/report-layout/overview`).then(r => r.json()),
      fetch(`${API}/api/report-layout/breakdown`).then(r => r.json()),
      fetch(`${API}/api/report-layout/definitions`).then(r => r.json()),
      fetch(`${API}/api/video-correlation/overview`).then(r => r.json()).catch(() => null),
    ]).then(([o, b, d, vc]) => {
      setOv(o); setBd(b); setDefs(d); setVcOv(vc);
    }).catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Error: {err}</div>;
  if (!ov) return <div className="text-muted p-4">Loading EEG Report Layout...</div>;

  const kpis = ov.kpis || {};
  const TABS = [
    { id: 'overview', label: 'Overview' },
    { id: 'components', label: 'Report Components' },
    { id: 'video-ai', label: 'Video AI Component' },
    { id: 'sections', label: 'Sections' },
    { id: 'definitions', label: 'Definitions' },
  ];

  const statusSummary = kpis.status_summary || {};
  const aiComponentsStatus = statusSummary.ai_components || '';

  return (
    <div className="container-fluid py-3">
      <h3>📋 EEG / Video-EEG Summary Report Layout</h3>
      <p className="text-muted small mb-1">{ov.description}</p>
      <p className="text-muted small mb-3">
        Report types: <strong>{(ov.report_types || []).join(' · ')}</strong>
        &nbsp;· Updated: {ov.updated_at}
      </p>

      {/* Status badges */}
      <div className="d-flex flex-wrap gap-2 mb-3">
        {Object.entries(statusSummary).map(([k, v]) => (
          <span key={k} className={`badge bg-${v.includes('planned') ? 'warning text-dark' : 'success'} px-3 py-2`}>
            {k}: {v}
          </span>
        ))}
      </div>

      {/* KPI Row */}
      <div className="row mb-3">
        <KPI label="Report Types" value={kpis.total_report_types} color="primary" />
        <KPI label="AI Components" value={kpis.total_components} color="info" />
        <KPI label="Report Sections" value={kpis.total_sections} color="success" />
        <KPI label="Editable Sections" value={kpis.editable_sections} color="warning" />
        <KPI label="AI Sources" value={kpis.total_ai_sources} color="secondary" />
        {vcOv && <KPI label="Video Events" value={vcOv.kpis?.total_behavioral_events} color="danger" />}
        {vcOv && <KPI label="EEG-Semiology Concordant" value={vcOv.kpis?.semiology_eeg_concordant} color="success" />}
        {vcOv && <KPI label="Video Frames" value={vcOv.kpis?.video_frames_available} color="info" />}
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* OVERVIEW TAB */}
      {tab === 'overview' && (
        <div>
          <div className="row">
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm border-0 h-100">
                <div className="card-header bg-primary text-white fw-semibold">AI Component Sources</div>
                <div className="card-body">
                  <HBar
                    items={ov.components_by_source || []}
                    colorFn={(name, i) => PALETTE[i % PALETTE.length]}
                  />
                </div>
              </div>
            </div>
            <div className="col-md-3 mb-3">
              <div className="card shadow-sm border-0 h-100">
                <div className="card-header bg-success text-white fw-semibold">Section Types</div>
                <div className="card-body">
                  {(ov.section_type_distribution || []).map((s, i) => (
                    <div key={i} className="d-flex justify-content-between align-items-center mb-2">
                      <span>{SECTION_ICONS[s.name] || '•'} {s.name}</span>
                      <span className={`badge bg-${PALETTE[i % PALETTE.length]}`}>{s.value}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            <div className="col-md-3 mb-3">
              <div className="card shadow-sm border-0 h-100">
                <div className="card-header bg-info text-white fw-semibold">Editability</div>
                <div className="card-body">
                  {(ov.editability_distribution || []).map((e, i) => (
                    <div key={i} className="d-flex justify-content-between align-items-center mb-2">
                      <span>{e.name}</span>
                      <span className={`badge bg-${i === 0 ? 'success' : 'secondary'}`}>{e.value}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Report Types */}
          <div className="card shadow-sm border-0 mb-3">
            <div className="card-header bg-dark text-white fw-semibold">Report Types Supported</div>
            <div className="card-body">
              <div className="row">
                {(bd?.report_type_details || []).map((rt, i) => (
                  <div key={i} className="col-md-3 mb-3">
                    <div className={`card border-${PALETTE[i % PALETTE.length]} h-100`}>
                      <div className={`card-header bg-${PALETTE[i % PALETTE.length]} text-white small fw-semibold`}>
                        {rt.name}
                      </div>
                      <div className="card-body small text-muted">{rt.description}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Components summary */}
          <div className="card shadow-sm border-0">
            <div className="card-header bg-secondary text-white fw-semibold">AI Component Summary</div>
            <div className="table-responsive">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light">
                  <tr>
                    <th>#</th>
                    <th>Component</th>
                    <th>AI Source</th>
                    <th>AI Finding</th>
                    <th>AI Recommendation</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {(ov.components_table || []).map((c, i) => (
                    <tr key={i}>
                      <td><span className="badge bg-secondary">{i + 1}</span></td>
                      <td>
                        <span className="me-1">{COMP_ICONS[c.id] || '•'}</span>
                        <strong>{c.label}</strong>
                      </td>
                      <td><code className="small">{c.ai_source}</code></td>
                      <td className="small text-muted">{c.ai_finding}</td>
                      <td className="small text-muted">{c.ai_recommendation}</td>
                      <td>
                        <span className={`badge bg-${c.id === 'video' ? 'success' : 'success'}`}>built</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* COMPONENTS TAB */}
      {tab === 'components' && (
        <div>
          <div className="row">
            {(bd?.per_component || []).map((c, i) => (
              <div key={i} className="col-md-6 mb-3">
                <div className={`card shadow-sm border-${COMP_COLORS[c.id] || 'secondary'} h-100`}>
                  <div className={`card-header bg-${COMP_COLORS[c.id] || 'secondary'} text-white`}>
                    <span className="me-2">{COMP_ICONS[c.id] || '•'}</span>
                    <strong>{c.label}</strong>
                  </div>
                  <div className="card-body">
                    <table className="table table-sm table-borderless mb-0">
                      <tbody>
                        <tr>
                          <td className="fw-semibold small" style={{ width: 140 }}>AI Source</td>
                          <td><code className="small">{c.ai_source}</code></td>
                        </tr>
                        <tr>
                          <td className="fw-semibold small">AI Finding</td>
                          <td className="small">{c.ai_finding}</td>
                        </tr>
                        <tr>
                          <td className="fw-semibold small">AI Recommendation</td>
                          <td className="small">{c.ai_recommendation}</td>
                        </tr>
                        <tr>
                          <td className="fw-semibold small">Status</td>
                          <td><span className="badge bg-success">built</span></td>
                        </tr>
                      </tbody>
                    </table>
                    {c.id === 'video' && vcOv && (
                      <div className="alert alert-success mt-2 py-2 mb-0 small">
                        <strong>Live data:</strong> {vcOv.kpis?.total_behavioral_events} behavioral events ·{' '}
                        {vcOv.kpis?.semiology_eeg_concordant} EEG-concordant ·{' '}
                        {vcOv.kpis?.video_frames_available} frames
                        &nbsp;→ <a href="/video-correlation" className="alert-link">Video Correlation Dashboard</a>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* VIDEO AI COMPONENT TAB */}
      {tab === 'video-ai' && (
        <div>
          <div className="alert alert-success mb-3">
            <strong>🎥 Video AI Component — Built</strong>: Behavioral semiology events from video-EEG are
            synced to EEG onset zones and automatically included in the AI findings section of every
            video-EEG report. Source: <code>/api/video-correlation/overview</code>
          </div>

          {vcOv ? (
            <div>
              <div className="row mb-3">
                {[
                  { label: 'Patients Monitored', value: vcOv.kpis?.total_patients, color: 'primary' },
                  { label: 'Behavioral Events', value: vcOv.kpis?.total_behavioral_events, color: 'danger' },
                  { label: 'Semiology Categories', value: vcOv.kpis?.semiology_categories, color: 'info' },
                  { label: 'EEG-Concordant Pairs', value: vcOv.kpis?.semiology_eeg_concordant, color: 'success' },
                  { label: 'Lateralized Patients', value: vcOv.kpis?.lateralized_patients, color: 'warning' },
                  { label: 'Video Frames Available', value: vcOv.kpis?.video_frames_available, color: 'secondary' },
                ].map(k => <KPI key={k.label} {...k} />)}
              </div>

              <div className="row">
                <div className="col-md-6 mb-3">
                  <div className="card shadow-sm border-0">
                    <div className="card-header bg-danger text-white fw-semibold">Semiology Category Distribution</div>
                    <div className="card-body">
                      <HBar
                        items={vcOv.semiology_category_distribution || []}
                        colorFn={(_, i) => PALETTE[i % PALETTE.length]}
                      />
                    </div>
                  </div>
                </div>
                <div className="col-md-6 mb-3">
                  <div className="card shadow-sm border-0">
                    <div className="card-header bg-info text-white fw-semibold">Lateralization Distribution</div>
                    <div className="card-body">
                      <HBar
                        items={vcOv.lateralization_distribution || []}
                        colorFn={(_, i) => PALETTE[i % PALETTE.length]}
                      />
                    </div>
                  </div>
                </div>
              </div>

              <div className="card shadow-sm border-0 mb-3">
                <div className="card-header bg-warning text-dark fw-semibold">Top Semiology Signs (Video AI Findings)</div>
                <div className="table-responsive">
                  <table className="table table-sm table-hover mb-0">
                    <thead className="table-light">
                      <tr>
                        <th>#</th>
                        <th>Sign</th>
                        <th>Count</th>
                        <th>Proportion</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(vcOv.top_semiology_signs || []).slice(0, 10).map((s, i) => {
                        const total = vcOv.kpis?.total_behavioral_events || 1;
                        const pct = Math.round((s.count / total) * 100);
                        return (
                          <tr key={i}>
                            <td><span className="badge bg-secondary">{i + 1}</span></td>
                            <td className="small">{s.name}</td>
                            <td><strong>{s.count}</strong></td>
                            <td>
                              <div className="progress" style={{ height: 14, width: 100 }}>
                                <div
                                  className={`progress-bar bg-${PALETTE[i % PALETTE.length]}`}
                                  style={{ width: `${pct}%` }}
                                >
                                  <span className="small">{pct}%</span>
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

              {/* AI Report Integration Card */}
              <div className="card shadow-sm border-success mb-3">
                <div className="card-header bg-success text-white fw-semibold">Video AI Component — Report Integration</div>
                <div className="card-body">
                  <div className="row">
                    <div className="col-md-6">
                      <h6 className="fw-semibold">AI Finding (auto-generated)</h6>
                      <div className="bg-light rounded p-3 small font-monospace">
                        Video-EEG analysis: {vcOv.kpis?.total_behavioral_events} behavioral events
                        across {vcOv.kpis?.total_patients} patients. Semiology categories: Motor
                        ({(vcOv.semiology_category_distribution || []).find(s => s.name === 'Motor')?.value || 0}
                        ), Dialeptic (
                        {(vcOv.semiology_category_distribution || []).find(s => s.name === 'Dialeptic')?.value || 0}
                        ), Aura (
                        {(vcOv.semiology_category_distribution || []).find(s => s.name === 'Aura')?.value || 0}
                        ). EEG-semiology concordance:{' '}
                        {vcOv.kpis?.semiology_eeg_concordant}/{vcOv.kpis?.lateralized_patients} lateralized pairs.
                        {vcOv.kpis?.video_frames_available} video frames available for review.
                      </div>
                    </div>
                    <div className="col-md-6">
                      <h6 className="fw-semibold">AI Recommendation (auto-generated)</h6>
                      <div className="bg-light rounded p-3 small font-monospace">
                        Review EEG-semiology concordance ({vcOv.kpis?.semiology_eeg_concordant} concordant pairs
                        identified). Correlate lateralization (Left:{' '}
                        {(vcOv.lateralization_distribution || []).find(l => l.name === 'Left')?.value || 0},
                        Right:{' '}
                        {(vcOv.lateralization_distribution || []).find(l => l.name === 'Right')?.value || 0}
                        ) with EEG onset zone for presurgical evaluation.
                      </div>
                    </div>
                  </div>
                  <div className="mt-3">
                    <a href="/video-correlation" className="btn btn-sm btn-outline-success me-2">
                      Open Video Correlation Dashboard
                    </a>
                    <a href="/video-eeg" className="btn btn-sm btn-outline-secondary">
                      Open Video EEG Monitoring
                    </a>
                  </div>
                </div>
              </div>

              {/* References */}
              <div className="card shadow-sm border-0">
                <div className="card-header bg-secondary text-white fw-semibold">Clinical References — Video AI Component</div>
                <div className="card-body small text-muted">
                  <ul className="mb-0">
                    <li><strong>Lüders et al. (1998)</strong> — Semiological seizure classification; provides the taxonomy for video behavioral event annotation</li>
                    <li><strong>ILAE 2017</strong> — Operational classification of seizure types; used for semiology-to-EEG mapping</li>
                    <li><strong>Rosenow & Lüders (2001)</strong> — Presurgical evaluation of epilepsies; concordance methodology between semiology and EEG onset zone</li>
                    <li><strong>Noachtar & Rémi (2009)</strong> — The role of EEG in epilepsy; defines EEG-semiology correlation standards</li>
                  </ul>
                </div>
              </div>
            </div>
          ) : (
            <div className="alert alert-warning">Video correlation data not available.</div>
          )}
        </div>
      )}

      {/* SECTIONS TAB */}
      {tab === 'sections' && (
        <div>
          {(bd?.per_section_type || []).map((group, gi) => (
            <div key={gi} className="card shadow-sm border-0 mb-3">
              <div className={`card-header bg-${PALETTE[gi % PALETTE.length]} text-white fw-semibold`}>
                {SECTION_ICONS[group.type] || '•'} {group.type} Sections
              </div>
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light">
                    <tr>
                      <th>ID</th>
                      <th>Label</th>
                      <th>Source</th>
                      <th>Editable</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(group.sections || []).map((s, si) => (
                      <tr key={si}>
                        <td><code className="small">{s.id}</code></td>
                        <td className="small">{s.label}</td>
                        <td className="small text-muted">{s.source}</td>
                        <td>
                          {s.editable
                            ? <span className="badge bg-success">✏️ Editable</span>
                            : <span className="badge bg-secondary">Read-only</span>}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ))}

          {/* Sections overview */}
          <div className="card shadow-sm border-0">
            <div className="card-header bg-dark text-white fw-semibold">Full Sections List</div>
            <div className="table-responsive">
              <table className="table table-sm mb-0">
                <thead className="table-light">
                  <tr>
                    <th>#</th>
                    <th>Section</th>
                    <th>Source</th>
                    <th>Type</th>
                    <th>Editable</th>
                  </tr>
                </thead>
                <tbody>
                  {(ov.sections_table || []).map((s, i) => (
                    <tr key={i}>
                      <td>{i + 1}</td>
                      <td className="small">{s.label}</td>
                      <td className="small text-muted">{s.source}</td>
                      <td><span className={`badge bg-${PALETTE[['AI','Expert','Final','Audit'].indexOf(s.type) % PALETTE.length]}`}>{s.type}</span></td>
                      <td>
                        {s.editable
                          ? <span className="badge bg-success">✏️</span>
                          : <span className="badge bg-secondary">—</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'definitions' && defs && (
        <div>
          <div className="row mb-3">
            <div className="col-md-6">
              <div className="card shadow-sm border-0 mb-3">
                <div className="card-header bg-dark text-white fw-semibold">Status Legend</div>
                <div className="card-body">
                  {(defs.status_legend || []).map((s, i) => (
                    <div key={i} className="mb-2">
                      <span className={`badge me-2 ${s.status === 'built' ? 'bg-success' : 'bg-warning text-dark'}`}>{s.status}</span>
                      <span className="small text-muted">{s.description}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="card shadow-sm border-0">
                <div className="card-header bg-info text-white fw-semibold">Clinical Notes</div>
                <div className="card-body">
                  <ul className="small text-muted mb-0">
                    {(defs.clinical_notes || []).map((n, i) => <li key={i}>{n}</li>)}
                  </ul>
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card shadow-sm border-0">
                <div className="card-header bg-secondary text-white fw-semibold">References</div>
                <div className="card-body">
                  <ul className="small mb-0">
                    {(defs.references || []).map((r, i) => (
                      <li key={i}><strong>{r.ref}</strong> — {r.detail}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          </div>

          <div className="card shadow-sm border-0">
            <div className="card-header bg-primary text-white fw-semibold">Glossary</div>
            <div className="table-responsive">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light">
                  <tr><th>Term</th><th>Definition</th></tr>
                </thead>
                <tbody>
                  {(defs.glossary || []).map((g, i) => (
                    <tr key={i}>
                      <td><strong className="small">{g.term}</strong></td>
                      <td className="small text-muted">{g.definition}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
