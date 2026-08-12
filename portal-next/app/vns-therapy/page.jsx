'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const RESP_COLOR = r => {
  if (!r) return 'secondary';
  const l = r.toLowerCase();
  if (l.includes('non')) return 'danger';
  if (l.includes('partial')) return 'warning';
  if (l.includes('responder')) return 'success';
  return 'secondary';
};

const BATT_COLOR = pct => {
  if (pct < 20) return 'danger';
  if (pct < 40) return 'warning';
  return 'success';
};

export default function VNSTherapyPage() {
  const [ov, setOv]     = useState(null);
  const [bd, setBd]     = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab]   = useState('overview');
  const [selPt, setSelPt] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/vns-therapy/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/vns-therapy/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/vns-therapy/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  const loading = !ov && !bd;
  if (loading) return <div className="container py-5 text-center text-white">Loading VNS Therapy data…</div>;

  const k = ov?.kpis || {};
  const bt = ov?.battery_alerts || [];
  const rd = ov?.response_distribution || {};
  const ps = ov?.parameter_summary || {};
  const se = ov?.side_effects_summary || {};
  const mt = ov?.monthly_trends || {};

  return (
    <div className="container-fluid py-4" style={{ background: '#0d1117', minHeight: '100vh', color: '#c9d1d9' }}>
      <div className="d-flex align-items-center gap-3 mb-4">
        <span style={{ fontSize: 36 }}>⚡</span>
        <div>
          <h2 className="mb-0 text-white fw-bold">VNS Therapy Monitoring</h2>
          <small className="text-info">Vagus Nerve Stimulation · Drug-Resistant Epilepsy · {k.total_vns_patients} patients</small>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="row g-3 mb-4">
        {[
          { label: 'VNS Patients', val: k.total_vns_patients, sub: `${k.pct_of_cohort}% of cohort`, color: '#58a6ff' },
          { label: 'Responder Rate', val: `${k.responder_rate_pct}%`, sub: `≥50% seizure reduction`, color: '#3fb950' },
          { label: 'Mean Sz Reduction', val: `${k.mean_seizure_reduction_pct}%`, sub: 'avg vs baseline', color: '#58a6ff' },
          { label: 'AutoStim Enabled', val: k.autostim_enabled, sub: `of ${k.total_vns_patients} patients`, color: '#bc8cff' },
          { label: 'Battery Alerts', val: k.low_battery_alert, sub: '<20% remaining', color: k.low_battery_alert > 0 ? '#f85149' : '#3fb950' },
          { label: 'Avg Therapy Duration', val: `${k.mean_therapy_years} yr`, sub: 'since implant', color: '#79c0ff' },
        ].map(c => (
          <div className="col-6 col-md-4 col-lg-2" key={c.label}>
            <div className="rounded p-3 text-center h-100" style={{ background: '#161b22', border: '1px solid #30363d' }}>
              <div className="fw-bold fs-4" style={{ color: c.color }}>{c.val}</div>
              <div className="small text-white">{c.label}</div>
              <div className="text-muted" style={{ fontSize: 11 }}>{c.sub}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-4" style={{ borderColor: '#30363d' }}>
        {['overview', 'patients', 'parameters', 'side-effects', 'definitions'].map(t => (
          <li className="nav-item" key={t}>
            <button
              className={`nav-link ${tab === t ? 'active text-white bg-dark border-info' : 'text-secondary'}`}
              style={{ borderColor: '#30363d' }}
              onClick={() => setTab(t)}
            >
              {{ overview: '📊 Overview', patients: '🧑‍⚕️ Per Patient', parameters: '⚙️ Parameters',
                 'side-effects': '⚠️ Side Effects', definitions: '📖 Definitions' }[t]}
            </button>
          </li>
        ))}
      </ul>

      {/* ── OVERVIEW TAB ── */}
      {tab === 'overview' && (
        <div>
          {/* Battery Alerts */}
          {bt.length > 0 && (
            <div className="alert alert-danger mb-4" style={{ background: '#2d1a1a', borderColor: '#f85149' }}>
              <strong>🔋 Battery Replacement Required Soon:</strong>&nbsp;
              {bt.map(p => `${p.patient_id} (${p.battery_pct}% — ${p.battery_years_left} yr remaining)`).join(' · ')}
            </div>
          )}

          <div className="row g-4">
            {/* Response Distribution */}
            <div className="col-md-4">
              <div className="rounded p-3 h-100" style={{ background: '#161b22', border: '1px solid #30363d' }}>
                <h6 className="text-white mb-3">Response Distribution</h6>
                {[
                  { label: 'Responders (≥50%)', count: rd.responder, color: '#3fb950' },
                  { label: 'Partial (25–49%)', count: rd.partial, color: '#d29922' },
                  { label: 'Non-responders (<25%)', count: rd.non_responder, color: '#f85149' },
                ].map(r => (
                  <div key={r.label} className="mb-3">
                    <div className="d-flex justify-content-between mb-1">
                      <span className="small" style={{ color: r.color }}>{r.label}</span>
                      <span className="small text-white fw-bold">{r.count}</span>
                    </div>
                    <div className="progress" style={{ height: 10, background: '#21262d' }}>
                      <div className="progress-bar" style={{
                        width: `${(r.count / k.total_vns_patients) * 100}%`,
                        background: r.color
                      }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Model Distribution */}
            <div className="col-md-4">
              <div className="rounded p-3 h-100" style={{ background: '#161b22', border: '1px solid #30363d' }}>
                <h6 className="text-white mb-3">Device Model</h6>
                {[
                  { label: 'LivaNova SenTiva™', count: k.model_sentiva, color: '#58a6ff' },
                  { label: 'Demipulse®', count: k.model_demipulse, color: '#bc8cff' },
                ].map(m => (
                  <div key={m.label} className="mb-3">
                    <div className="d-flex justify-content-between mb-1">
                      <span className="small text-white">{m.label}</span>
                      <span className="small fw-bold" style={{ color: m.color }}>{m.count}</span>
                    </div>
                    <div className="progress" style={{ height: 10, background: '#21262d' }}>
                      <div className="progress-bar" style={{
                        width: `${(m.count / k.total_vns_patients) * 100}%`,
                        background: m.color
                      }} />
                    </div>
                  </div>
                ))}
                <hr style={{ borderColor: '#30363d' }} />
                <div className="small text-muted">AutoStim enabled: <strong className="text-white">{k.autostim_enabled}/{k.total_vns_patients}</strong></div>
                <div className="small text-muted mt-1">Focal epilepsy: <strong className="text-white">{k.focal_epilepsy_pct}%</strong></div>
              </div>
            </div>

            {/* Side Effects Summary */}
            <div className="col-md-4">
              <div className="rounded p-3 h-100" style={{ background: '#161b22', border: '1px solid #30363d' }}>
                <h6 className="text-white mb-3">Top Side Effects</h6>
                {Object.entries(se).sort((a, b) => b[1] - a[1]).map(([effect, count]) => (
                  <div key={effect} className="mb-2">
                    <div className="d-flex justify-content-between mb-1">
                      <span className="small text-white text-capitalize">{effect}</span>
                      <span className="small text-warning">{count} pts ({Math.round(count / k.total_vns_patients * 100)}%)</span>
                    </div>
                    <div className="progress" style={{ height: 6, background: '#21262d' }}>
                      <div className="progress-bar bg-warning" style={{ width: `${count / k.total_vns_patients * 100}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Monthly Seizure Trend */}
            <div className="col-12">
              <div className="rounded p-3" style={{ background: '#161b22', border: '1px solid #30363d' }}>
                <h6 className="text-white mb-3">📈 Monthly Seizure Frequency — Representative Patients (12-month retrospective)</h6>
                {mt.patients && Object.entries(mt.patients).map(([pid, info]) => (
                  <div key={pid} className="mb-3">
                    <div className="small text-info mb-1">{info.label} — trend since VNS start</div>
                    <div className="d-flex gap-1 flex-wrap">
                      {(mt.months || []).map((month, i) => {
                        const val = info.seizures[i] ?? 0;
                        const maxVal = Math.max(...(info.seizures || [1]));
                        const pct = Math.round((val / maxVal) * 100);
                        return (
                          <div key={month} className="text-center" style={{ minWidth: 44 }}>
                            <div style={{
                              height: 50, width: 38,
                              background: '#21262d', borderRadius: 4,
                              position: 'relative', overflow: 'hidden', display: 'inline-block'
                            }}>
                              <div style={{
                                position: 'absolute', bottom: 0, width: '100%',
                                height: `${pct}%`, background: '#58a6ff', borderRadius: '2px 2px 0 0'
                              }} />
                            </div>
                            <div style={{ fontSize: 9, color: '#8b949e', marginTop: 2 }}>{val}</div>
                            <div style={{ fontSize: 8, color: '#6e7681' }}>{month.replace("'", '\u2019')}</div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── PATIENTS TAB ── */}
      {tab === 'patients' && bd && (
        <div>
          <div className="table-responsive">
            <table className="table table-dark table-hover table-sm" style={{ fontSize: 13 }}>
              <thead style={{ background: '#161b22' }}>
                <tr>
                  {['Patient', 'Age/Sex', 'Epilepsy', 'Model', 'Current(mA)', 'Freq(Hz)',
                    'AutoStim', 'Baseline', 'Current Sz', '% Red', 'Response', 'Battery', 'Side Effects'].map(h => (
                    <th key={h} className="text-info" style={{ whiteSpace: 'nowrap' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(bd.patients || []).map(p => (
                  <tr key={p.patient_id}
                    style={{ cursor: 'pointer', background: selPt === p.patient_id ? '#1f2937' : 'transparent' }}
                    onClick={() => setSelPt(selPt === p.patient_id ? null : p.patient_id)}>
                    <td className="fw-bold text-white">{p.patient_id}</td>
                    <td>{p.age} / {p.sex}</td>
                    <td>{p.epilepsy_type}</td>
                    <td><span className="badge bg-secondary">{p.model}</span></td>
                    <td>{p.output_current_ma} mA</td>
                    <td>{p.frequency_hz} Hz</td>
                    <td>{p.autostim
                      ? <span className="badge bg-success">ON</span>
                      : <span className="badge bg-secondary">OFF</span>}
                    </td>
                    <td>{p.baseline_sz_month}/mo</td>
                    <td>{p.current_sz_month}/mo</td>
                    <td className="fw-bold">{p.pct_reduction}%</td>
                    <td><span className={`badge bg-${RESP_COLOR(p.response)}`}>{p.response}</span></td>
                    <td>
                      <div className="d-flex align-items-center gap-1">
                        <div className="progress flex-grow-1" style={{ height: 8, background: '#21262d', minWidth: 40 }}>
                          <div className={`progress-bar bg-${BATT_COLOR(p.battery_pct)}`}
                            style={{ width: `${p.battery_pct}%` }} />
                        </div>
                        <span style={{ fontSize: 11, color: p.battery_pct < 20 ? '#f85149' : '#8b949e' }}>
                          {p.battery_pct}%
                        </span>
                        {p.battery_alert && <span className="badge bg-danger" style={{ fontSize: 9 }}>⚠</span>}
                      </div>
                    </td>
                    <td>
                      {p.side_effects.length === 0
                        ? <span className="text-muted">none</span>
                        : p.side_effects.map(se => (
                          <span key={se} className="badge bg-warning text-dark me-1" style={{ fontSize: 10 }}>{se}</span>
                        ))}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {selPt && (
            <div className="rounded p-3 mt-2" style={{ background: '#1a2233', border: '1px solid #30363d' }}>
              <div className="small text-info">Expanded: {selPt} — Therapy {(bd.patients || []).find(p => p.patient_id === selPt)?.therapy_years} years · Implanted {(bd.patients || []).find(p => p.patient_id === selPt)?.implant_year}</div>
            </div>
          )}
        </div>
      )}

      {/* ── PARAMETERS TAB ── */}
      {tab === 'parameters' && bd && (
        <div className="row g-4">
          <div className="col-md-4">
            <div className="rounded p-3" style={{ background: '#161b22', border: '1px solid #30363d' }}>
              <h6 className="text-white mb-3">Output Current Distribution</h6>
              {Object.entries(bd.parameter_distributions?.current_bins || {}).map(([bin, count]) => (
                <div key={bin} className="mb-2">
                  <div className="d-flex justify-content-between mb-1">
                    <span className="small text-white">{bin}</span>
                    <span className="small text-info">{count}</span>
                  </div>
                  <div className="progress" style={{ height: 8, background: '#21262d' }}>
                    <div className="progress-bar bg-info" style={{ width: `${count / k.total_vns_patients * 100}%` }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
          <div className="col-md-4">
            <div className="rounded p-3" style={{ background: '#161b22', border: '1px solid #30363d' }}>
              <h6 className="text-white mb-3">Frequency Setting</h6>
              {Object.entries(bd.parameter_distributions?.frequency_dist || {}).map(([freq, count]) => (
                <div key={freq} className="mb-2">
                  <div className="d-flex justify-content-between mb-1">
                    <span className="small text-white">{freq}</span>
                    <span className="small text-success">{count}</span>
                  </div>
                  <div className="progress" style={{ height: 8, background: '#21262d' }}>
                    <div className="progress-bar bg-success" style={{ width: `${count / k.total_vns_patients * 100}%` }} />
                  </div>
                </div>
              ))}
              <div className="mt-3">
                <h6 className="text-white mb-2">Mean Parameters</h6>
                <div className="small text-muted">Mean current: <strong className="text-white">{ps.mean_output_current_ma} mA</strong></div>
              </div>
            </div>
          </div>
          <div className="col-md-4">
            <div className="rounded p-3" style={{ background: '#161b22', border: '1px solid #30363d' }}>
              <h6 className="text-white mb-3">Battery Status Distribution</h6>
              {Object.entries(bd.parameter_distributions?.battery_bins || {}).map(([bin, count]) => {
                const color = bin.includes('Critical') ? '#f85149' : bin.includes('Low') ? '#d29922' : '#3fb950';
                return (
                  <div key={bin} className="mb-2">
                    <div className="d-flex justify-content-between mb-1">
                      <span className="small" style={{ color }}>{bin}</span>
                      <span className="small text-white fw-bold">{count}</span>
                    </div>
                    <div className="progress" style={{ height: 8, background: '#21262d' }}>
                      <div className="progress-bar" style={{ width: `${count / k.total_vns_patients * 100}%`, background: color }} />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* AutoStim comparison */}
          <div className="col-12">
            <div className="rounded p-3" style={{ background: '#161b22', border: '1px solid #30363d' }}>
              <h6 className="text-white mb-3">AutoStim (Cardiac-Based Detection) — Efficacy Comparison</h6>
              <div className="row g-3">
                {Object.entries(bd.autostim_comparison || {}).map(([key, val]) => (
                  <div className="col-md-4" key={key}>
                    <div className="rounded p-3 text-center" style={{ background: '#21262d' }}>
                      <div className="text-info small mb-1">{key === 'autostim_on' ? 'AutoStim ON' : 'AutoStim OFF'}</div>
                      <div className="fs-4 fw-bold text-white">{val.mean_reduction}%</div>
                      <div className="text-muted small">mean seizure reduction</div>
                      <div className="text-muted small">{val.n} patients</div>
                    </div>
                  </div>
                ))}
                <div className="col-md-4">
                  <div className="rounded p-3" style={{ background: '#21262d' }}>
                    <div className="text-warning small mb-1">Therapy Duration Bands</div>
                    {Object.entries(bd.therapy_duration_bands || {}).map(([band, count]) => (
                      <div key={band} className="d-flex justify-content-between small py-1" style={{ borderBottom: '1px solid #30363d' }}>
                        <span className="text-white">{band}</span>
                        <span className="text-info">{count} pts</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── SIDE EFFECTS TAB ── */}
      {tab === 'side-effects' && bd && (
        <div className="row g-4">
          <div className="col-md-6">
            <div className="rounded p-3" style={{ background: '#161b22', border: '1px solid #30363d' }}>
              <h6 className="text-white mb-3">Side Effect Prevalence by Response Group</h6>
              <table className="table table-dark table-sm table-bordered" style={{ fontSize: 13 }}>
                <thead>
                  <tr>
                    <th className="text-info">Side Effect</th>
                    <th className="text-success">Responder</th>
                    <th className="text-warning">Partial</th>
                    <th className="text-danger">Non-Resp</th>
                    <th className="text-white">All Pts (%)</th>
                  </tr>
                </thead>
                <tbody>
                  {(bd.side_effects?.ranked || []).map(se => (
                    <tr key={se.effect}>
                      <td className="text-white text-capitalize">{se.effect}</td>
                      <td className="text-success">{bd.side_effects?.by_response_group?.responder?.[se.effect] || 0}</td>
                      <td className="text-warning">{bd.side_effects?.by_response_group?.partial?.[se.effect] || 0}</td>
                      <td className="text-danger">{bd.side_effects?.by_response_group?.non_responder?.[se.effect] || 0}</td>
                      <td className="text-white fw-bold">{se.pct}%</td>
                    </tr>
                  ))}
                  {(bd.side_effects?.ranked || []).length === 0 && (
                    <tr><td colSpan={5} className="text-muted text-center">No side effects recorded</td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
          <div className="col-md-6">
            <div className="rounded p-3" style={{ background: '#161b22', border: '1px solid #30363d' }}>
              <h6 className="text-white mb-3">Clinical Notes on Side Effect Management</h6>
              {defs && Object.entries(defs.side_effects || {}).map(([key, val]) => (
                <div key={key} className="mb-3 p-2 rounded" style={{ background: '#21262d' }}>
                  <div className="small fw-bold text-warning text-capitalize">{key.replace(/_/g, ' ')}</div>
                  {val.prevalence_pct && (
                    <div className="small text-muted">Prevalence: <span className="text-white">{val.prevalence_pct}%</span></div>
                  )}
                  {val.management && (
                    <div className="small text-muted mt-1">{val.management}</div>
                  )}
                  {val.note && (
                    <div className="small text-info mt-1">{val.note}</div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ── DEFINITIONS TAB ── */}
      {tab === 'definitions' && defs && (
        <div className="row g-4">
          <div className="col-md-6">
            <div className="rounded p-3" style={{ background: '#161b22', border: '1px solid #30363d' }}>
              <h6 className="text-white mb-3">What is VNS Therapy?</h6>
              <p className="small text-muted">{defs.what_is_vns}</p>
              <h6 className="text-white mt-3 mb-2">Indications</h6>
              <ul className="small text-muted ps-3">
                {(defs.indications || []).map((ind, i) => <li key={i}>{ind}</li>)}
              </ul>
              <h6 className="text-white mt-3 mb-2">Device Models</h6>
              {Object.entries(defs.device_models || {}).map(([model, desc]) => (
                <div key={model} className="mb-2 p-2 rounded" style={{ background: '#21262d' }}>
                  <div className="small fw-bold text-info">{model}</div>
                  <div className="small text-muted">{desc}</div>
                </div>
              ))}
            </div>
          </div>
          <div className="col-md-6">
            <div className="rounded p-3" style={{ background: '#161b22', border: '1px solid #30363d' }}>
              <h6 className="text-white mb-3">Stimulation Parameters</h6>
              {Object.entries(defs.stimulation_parameters || {}).map(([param, info]) => (
                <div key={param} className="mb-2 p-2 rounded" style={{ background: '#21262d' }}>
                  <div className="small fw-bold text-success text-capitalize">{param.replace(/_/g, ' ')}</div>
                  {info.range && <div className="small text-muted">Range: <span className="text-white">{info.range}</span></div>}
                  {info.standard && <div className="small text-muted">Standard: <span className="text-white">{info.standard}</span></div>}
                  {info.typical_target && <div className="small text-muted">Target: <span className="text-white">{info.typical_target}</span></div>}
                  {info.note && <div className="small text-muted mt-1 fst-italic">{info.note}</div>}
                  {info.description && <div className="small text-muted">{info.description}</div>}
                </div>
              ))}
            </div>
          </div>
          <div className="col-md-6">
            <div className="rounded p-3" style={{ background: '#161b22', border: '1px solid #30363d' }}>
              <h6 className="text-white mb-3">Efficacy Evidence</h6>
              {Object.entries(defs.efficacy_evidence || {}).map(([key, ev]) => (
                <div key={key} className="mb-3 p-2 rounded" style={{ background: '#21262d' }}>
                  <div className="small fw-bold text-info text-capitalize">{key.replace(/_/g, ' ')}</div>
                  {ev.responder_rate && <div className="small text-muted">Responder rate: <strong className="text-white">{ev.responder_rate}</strong></div>}
                  {ev.mean_seizure_reduction && <div className="small text-muted">Mean reduction: <strong className="text-white">{ev.mean_seizure_reduction}</strong></div>}
                  {ev.positive_predictors && (
                    <div className="small text-muted">Positive: {ev.positive_predictors.join(', ')}</div>
                  )}
                  {ev.reference && <div className="small text-success mt-1">📚 {ev.reference}</div>}
                  {ev.note && <div className="small text-muted fst-italic">{ev.note}</div>}
                </div>
              ))}
            </div>
          </div>
          <div className="col-md-6">
            <div className="rounded p-3" style={{ background: '#161b22', border: '1px solid #30363d' }}>
              <h6 className="text-white mb-3">AI Integration Opportunities</h6>
              {Object.entries(defs.ai_integration || {}).map(([key, val]) => (
                <div key={key} className="mb-2 p-2 rounded" style={{ background: '#21262d' }}>
                  <div className="small fw-bold text-bc8cff" style={{ color: '#bc8cff' }}>{key.replace(/_/g, ' ')}</div>
                  <div className="small text-muted">{val}</div>
                </div>
              ))}
              <div className="mt-3 p-2 rounded" style={{ background: '#1a2f1a' }}>
                <div className="small fw-bold text-success">NICE NG217 Guidance</div>
                <div className="small text-muted">{defs.nice_guidance?.recommendation}</div>
                <div className="small text-muted mt-1">Review: {defs.nice_guidance?.review_schedule}</div>
              </div>
              <div className="mt-3">
                <div className="small text-white fw-bold mb-2">References</div>
                {(ov?.references || []).map((ref, i) => (
                  <div key={i} className="small text-muted mb-1">📚 {ref}</div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
