'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const STAGE_COLOR = s => ({
  Wake: 'warning', N1: 'info', N2: 'primary', N3: 'success', REM: 'danger'
}[s] || 'secondary');

const STAGE_LEVEL = s => ({ Wake: 0, REM: 1, N1: 2, N2: 3, N3: 4 }[s] ?? 2);

const ASM_IMPACT_COLOR = v => {
  const l = (v || '').toLowerCase();
  if (l.includes('increase') || l.includes('improve') || l.includes('beneficial')) return 'success';
  if (l.includes('reduce') || l.includes('suppress') || l.includes('decrease')) return 'danger';
  if (l.includes('neutral') || l.includes('minimal')) return 'secondary';
  return 'info';
};

function KPI({ label, value, sub, color = 'primary', warn }) {
  return (
    <div className={`card border-${warn ? 'warning' : color} h-100`}>
      <div className="card-body text-center p-3">
        <div className={`display-6 fw-bold text-${warn ? 'warning' : color}`}>{value ?? '—'}</div>
        <div className="small fw-semibold">{label}</div>
        {sub && <div className="text-muted" style={{ fontSize: 11 }}>{sub}</div>}
      </div>
    </div>
  );
}

function StagePct({ stage, pct, normalMin, normalMax, interpretation }) {
  const color = STAGE_COLOR(stage);
  const inRange = pct >= normalMin && pct <= normalMax;
  const barMax = Math.max(normalMax * 1.4, pct * 1.1, 60);
  const normalMinPct = (normalMin / barMax) * 100;
  const normalMaxPct = (normalMax / barMax) * 100;
  const valuePct = (pct / barMax) * 100;
  return (
    <div className="mb-4">
      <div className="d-flex justify-content-between align-items-center mb-1">
        <span className="fw-semibold">
          <span className={`badge bg-${color} me-2`}>{stage}</span>
          {pct}%
        </span>
        <span className="small text-muted">Normal: {normalMin}–{normalMax}%</span>
        <span className={`badge bg-${inRange ? 'success' : 'warning'}`}>
          {inRange ? '✓ Normal' : '⚠ Abnormal'}
        </span>
      </div>
      {/* Composite bar: actual vs normal range */}
      <div className="progress mb-1" style={{ height: 14, position: 'relative', background: '#e9ecef' }}>
        {/* normal range shading */}
        <div style={{
          position: 'absolute', top: 0, bottom: 0,
          left: `${normalMinPct}%`, width: `${normalMaxPct - normalMinPct}%`,
          background: 'rgba(25,135,84,0.15)', border: '1px solid rgba(25,135,84,0.4)'
        }} title={`Normal range ${normalMin}–${normalMax}%`} />
        {/* actual value bar */}
        <div className={`progress-bar bg-${color}`} style={{ width: `${valuePct}%`, opacity: 0.85 }} />
      </div>
      <div className="small text-muted fst-italic">{interpretation}</div>
    </div>
  );
}

function Hypnogram({ data }) {
  if (!data || !data.length) return null;
  const W = 800, H = 120, PAD = { t: 10, b: 30, l: 45, r: 10 };
  const innerW = W - PAD.l - PAD.r;
  const innerH = H - PAD.t - PAD.b;
  const stages = ['Wake', 'REM', 'N1', 'N2', 'N3'];
  const yScale = i => PAD.t + (i / (stages.length - 1)) * innerH;
  const maxT = data[data.length - 1]?.time_min || 480;
  const xScale = t => PAD.l + (t / maxT) * innerW;
  const pts = data.map(d => ({ x: xScale(d.time_min), y: yScale(STAGE_LEVEL(d.stage)), s: d.stage }));
  const pathD = pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x},${p.y}`).join(' ');
  const colors = { Wake: '#ffc107', REM: '#dc3545', N1: '#0dcaf0', N2: '#0d6efd', N3: '#198754' };
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto' }}>
      {/* Y labels */}
      {stages.map((s, i) => (
        <text key={s} x={PAD.l - 4} y={yScale(i) + 4} textAnchor="end"
          fontSize={9} fill={colors[s]}>{s}</text>
      ))}
      {/* Grid lines */}
      {stages.map((s, i) => (
        <line key={s} x1={PAD.l} x2={W - PAD.r} y1={yScale(i)} y2={yScale(i)}
          stroke="#dee2e6" strokeWidth={0.5} strokeDasharray="3,3" />
      ))}
      {/* Hypnogram line — step function */}
      {pts.map((p, i) => i === 0 ? null : (
        <g key={i}>
          <line x1={pts[i - 1].x} y1={pts[i - 1].y} x2={p.x} y2={pts[i - 1].y}
            stroke={colors[pts[i - 1].s]} strokeWidth={2} />
          <line x1={p.x} y1={pts[i - 1].y} x2={p.x} y2={p.y}
            stroke={colors[p.s]} strokeWidth={2} />
        </g>
      ))}
      {/* X axis labels every 60 min */}
      {[0, 60, 120, 180, 240, 300, 360, 420, 480].filter(t => t <= maxT).map(t => (
        <text key={t} x={xScale(t)} y={H - 5} textAnchor="middle" fontSize={8} fill="#6c757d">
          {t === 0 ? 'Lights-off' : `${t / 60}h`}
        </text>
      ))}
      <text x={W / 2} y={H - 1} textAnchor="middle" fontSize={8} fill="#6c757d">
        Time into recording
      </text>
    </svg>
  );
}

export default function SleepStageAnalysisPage() {
  const [ov, setOv]     = useState(null);
  const [bd, setBd]     = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab]   = useState('overview');
  const [err, setErr]   = useState(null);

  useEffect(() => {
    fetch(`${API}/api/sleep-stage-analysis/overview`).then(r => r.json()).then(setOv).catch(e => setErr(e.message));
    fetch(`${API}/api/sleep-stage-analysis/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/sleep-stage-analysis/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  const TABS = [
    { id: 'overview',    label: '📊 Overview' },
    { id: 'stages',      label: '🌙 Sleep Stages' },
    { id: 'asm',         label: '💊 ASM Impact' },
    { id: 'definitions', label: '📖 Definitions' },
  ];

  return (
    <div className="container-fluid py-3">
      <h4 className="mb-1">🛌 Sleep Stage Analysis</h4>
      <p className="text-muted small mb-3">
        Polysomnography-based sleep architecture in epilepsy — AASM scoring, stage distribution vs. normal, ASM effects, and seizure-sleep correlations across {ov?.total_patients ?? '…'} patients.
      </p>

      {err && <div className="alert alert-danger">Error: {err}</div>}

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

      {/* ── OVERVIEW ── */}
      {tab === 'overview' && (
        <div>
          {!ov ? <div className="text-muted">Loading…</div> : (
            <>
              {/* KPI cards */}
              <div className="row g-3 mb-4">
                <div className="col-6 col-md-3">
                  <KPI label="Total Sleep Time" value={`${ov.kpis?.total_sleep_time_min ?? '—'} min`}
                    sub={`${Math.round((ov.kpis?.total_sleep_time_min ?? 0) / 60 * 10) / 10} h  (normal 360-480 min)`}
                    color="primary" warn={ov.kpis?.total_sleep_time_min < 360} />
                </div>
                <div className="col-6 col-md-3">
                  <KPI label="Sleep Efficiency" value={`${ov.kpis?.sleep_efficiency_pct ?? '—'}%`}
                    sub="Normal ≥ 85%"
                    color="success" warn={ov.kpis?.sleep_efficiency_pct < 85} />
                </div>
                <div className="col-6 col-md-3">
                  <KPI label="Sleep Onset Latency" value={`${ov.kpis?.sleep_onset_latency_min ?? '—'} min`}
                    sub="Normal < 20 min"
                    color="info" warn={ov.kpis?.sleep_onset_latency_min > 20} />
                </div>
                <div className="col-6 col-md-3">
                  <KPI label="WASO" value={`${ov.kpis?.waso_min ?? '—'} min`}
                    sub="Wake After Sleep Onset (normal < 30 min)"
                    color="warning" warn={ov.kpis?.waso_min > 30} />
                </div>
                <div className="col-6 col-md-3">
                  <KPI label="Arousal Index" value={`${ov.kpis?.arousal_index_per_hour ?? '—'}/h`}
                    sub="Arousals per hour (normal < 15/h)"
                    color="danger" warn={ov.kpis?.arousal_index_per_hour > 15} />
                </div>
                <div className="col-6 col-md-3">
                  <KPI label="REM Latency" value={`${ov.kpis?.rem_latency_min ?? '—'} min`}
                    sub="Normal 70-120 min"
                    color="primary" />
                </div>
                <div className="col-6 col-md-3">
                  <KPI label="Recording Time" value={`${ov.kpis?.total_recording_time_min ?? '—'} min`}
                    sub="Time in bed (TRT)" color="secondary" />
                </div>
                <div className="col-6 col-md-3">
                  <KPI label="Fragmentation Index" value={`${ov.kpis?.sleep_fragmentation_index ?? '—'}`}
                    sub="Sleep fragmentation score" color="warning"
                    warn={ov.kpis?.sleep_fragmentation_index > 20} />
                </div>
              </div>

              <div className="row g-3 mb-4">
                {/* Sleep Stage Distribution */}
                <div className="col-md-7">
                  <div className="card h-100">
                    <div className="card-header fw-semibold">Sleep Stage Distribution vs. Normal Range</div>
                    <div className="card-body">
                      {(ov.stage_distribution || []).map(s => (
                        <StagePct key={s.stage} stage={s.stage} pct={s.pct}
                          normalMin={s.normal_min} normalMax={s.normal_max}
                          interpretation={s.interpretation} />
                      ))}
                      <div className="small text-muted mt-2">
                        <span className="badge bg-success bg-opacity-25 border border-success text-success me-2">■ Normal range</span>
                        Bars show patient cohort average vs. AASM normative values
                      </div>
                    </div>
                  </div>
                </div>

                {/* Seizure-Sleep Correlation */}
                <div className="col-md-5">
                  <div className="card h-100">
                    <div className="card-header fw-semibold">Seizure–Sleep Correlation</div>
                    <div className="card-body">
                      {ov.seizure_correlation_summary && (() => {
                        const sc = ov.seizure_correlation_summary;
                        return (
                          <table className="table table-sm">
                            <tbody>
                              <tr>
                                <td className="text-muted small">Seizures during sleep</td>
                                <td className="fw-bold text-danger">{sc.pct_seizures_during_sleep}%</td>
                              </tr>
                              <tr>
                                <td className="text-muted small">Seizures during wake</td>
                                <td className="fw-bold">{sc.pct_seizures_during_wake}%</td>
                              </tr>
                              <tr>
                                <td className="text-muted small">Most epileptogenic stage</td>
                                <td><span className={`badge bg-${STAGE_COLOR(sc.most_epileptogenic_stage)}`}>{sc.most_epileptogenic_stage}</span></td>
                              </tr>
                              <tr>
                                <td className="text-muted small">Least epileptogenic</td>
                                <td><span className={`badge bg-${STAGE_COLOR(sc.least_epileptogenic_stage)}`}>{sc.least_epileptogenic_stage}</span></td>
                              </tr>
                              <tr>
                                <td className="text-muted small">Nocturnal seizure prevalence</td>
                                <td className="fw-bold">{sc.nocturnal_seizure_prevalence_pct}%</td>
                              </tr>
                              <tr>
                                <td className="text-muted small">IED activation during sleep</td>
                                <td className="fw-bold text-warning">{sc.ied_activation_during_sleep_pct}%</td>
                              </tr>
                              <tr>
                                <td className="text-muted small">Sleep deprivation trigger</td>
                                <td className="fw-bold text-danger">{sc.sleep_deprivation_trigger_pct}%</td>
                              </tr>
                            </tbody>
                          </table>
                        );
                      })()}
                      <div className="alert alert-warning p-2 mt-2" style={{ fontSize: 12 }}>
                        <strong>⚠ Clinical Note:</strong> Sleep deprivation is the most common modifiable seizure trigger.
                        62.4% of seizures occur during sleep, predominantly in N2 (most epileptogenic).
                        REM is relatively protective.
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Sleep Efficiency Histogram */}
              <div className="card mb-4">
                <div className="card-header fw-semibold">Sleep Efficiency Distribution (n={ov.total_patients})</div>
                <div className="card-body">
                  {(() => {
                    const hist = ov.sleep_efficiency_histogram || [];
                    const maxCount = Math.max(...hist.map(h => h.count));
                    const CAT_COLOR = { Poor: 'danger', Fair: 'warning', Good: 'success' };
                    return (
                      <div className="row g-2 align-items-end">
                        {hist.map(h => (
                          <div key={h.bin_label} className="col text-center">
                            <div className="small text-muted mb-1">{h.count}</div>
                            <div
                              className={`bg-${CAT_COLOR[h.category] || 'secondary'} rounded-top`}
                              style={{ height: `${Math.round((h.count / maxCount) * 100)}px`, minHeight: 4 }}
                            />
                            <div className="small mt-1" style={{ fontSize: 10 }}>{h.bin_label}</div>
                            <div className={`badge bg-${CAT_COLOR[h.category] || 'secondary'} mt-1`} style={{ fontSize: 9 }}>
                              {h.category}
                            </div>
                          </div>
                        ))}
                      </div>
                    );
                  })()}
                  <div className="small text-muted mt-3">
                    Normal sleep efficiency ≥ 85%. Poor: &lt;70%; Fair: 70–84%; Good: ≥85%.
                    Epilepsy patients average {ov.kpis?.sleep_efficiency_pct}% (below normal threshold).
                  </div>
                </div>
              </div>

              {/* ASM Sleep Impact Summary */}
              {ov.asm_sleep_impact_summary?.length > 0 && (
                <div className="card mb-3">
                  <div className="card-header fw-semibold">ASM Sleep Impact Summary</div>
                  <div className="card-body p-0">
                    <table className="table table-sm mb-0">
                      <thead className="table-light">
                        <tr>
                          <th>ASM</th>
                          <th>Sleep Impact</th>
                          <th>N2 Effect</th>
                          <th>REM Effect</th>
                          <th>Overall</th>
                        </tr>
                      </thead>
                      <tbody>
                        {ov.asm_sleep_impact_summary.map((a, i) => (
                          <tr key={i}>
                            <td className="fw-semibold">{a.asm}</td>
                            <td><span className={`badge bg-${ASM_IMPACT_COLOR(a.sleep_impact)}`}>{a.sleep_impact}</span></td>
                            <td className="small">{a.n2_effect}</td>
                            <td className="small">{a.rem_effect}</td>
                            <td className="small">{a.overall}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      )}

      {/* ── SLEEP STAGES ── */}
      {tab === 'stages' && (
        <div>
          {!bd ? <div className="text-muted">Loading…</div> : (
            <>
              {/* Hypnogram */}
              <div className="card mb-4">
                <div className="card-header fw-semibold">Representative Hypnogram (Typical Epilepsy Night)</div>
                <div className="card-body">
                  <Hypnogram data={bd.hypnogram_data} />
                  <div className="small text-muted mt-2">
                    Step-function representation of sleep stage progression over the recording period.
                    Frequent arousals (returns to Wake/N1) reflect sleep fragmentation in epilepsy.
                  </div>
                </div>
              </div>

              {/* Seizure by Stage */}
              <div className="card mb-4">
                <div className="card-header fw-semibold">Seizure Risk by Sleep Stage</div>
                <div className="card-body">
                  <table className="table table-sm">
                    <thead className="table-light">
                      <tr>
                        <th>Stage</th>
                        <th>Seizure Probability</th>
                        <th>IED Activation Ratio</th>
                        <th>Predominant Seizure Type</th>
                        <th>Epileptogenic Risk</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(bd.seizure_by_stage || []).map((s, i) => {
                        const maxProb = Math.max(...(bd.seizure_by_stage || []).map(x => x.probability_pct));
                        return (
                          <tr key={i}>
                            <td><span className={`badge bg-${STAGE_COLOR(s.stage)}`}>{s.stage}</span></td>
                            <td>
                              <div className="d-flex align-items-center gap-2">
                                <div className="progress flex-grow-1" style={{ height: 10 }}>
                                  <div className={`progress-bar bg-${s.probability_pct > 30 ? 'danger' : 'warning'}`}
                                    style={{ width: `${(s.probability_pct / maxProb) * 100}%` }} />
                                </div>
                                <span className="small fw-bold">{s.probability_pct}%</span>
                              </div>
                            </td>
                            <td className="small">{s.ied_ratio}</td>
                            <td className="small">{s.seizure_type}</td>
                            <td><span className={`badge bg-${s.probability_pct > 35 ? 'danger' : s.probability_pct > 15 ? 'warning' : 'success'}`}>
                              {s.probability_pct > 35 ? 'High' : s.probability_pct > 15 ? 'Moderate' : 'Low'}
                            </span></td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Stage Details */}
              <div className="card mb-4">
                <div className="card-header fw-semibold">Stage-by-Stage EEG Pattern Details</div>
                <div className="card-body p-0">
                  {(bd.stage_details || []).map((s, i) => (
                    <div key={i} className={`p-3 border-bottom border-${STAGE_COLOR(s.stage.split(' ')[0])}`}
                      style={{ borderLeftWidth: 4, borderLeftStyle: 'solid', borderLeftColor: `var(--bs-${STAGE_COLOR(s.stage.split(' ')[0])})` }}>
                      <div className="d-flex justify-content-between align-items-start mb-2">
                        <h6 className="mb-0">
                          <span className={`badge bg-${STAGE_COLOR(s.stage.split(' ')[0])} me-2`}>{s.stage.split(' ')[0]}</span>
                          {s.stage}
                        </h6>
                        <div className="small text-end">
                          <div><strong>{s.duration_pct}%</strong> of night</div>
                          <div className="text-muted">Normal: {s.normal_pct_range}</div>
                        </div>
                      </div>
                      <div className="small">
                        <div><span className="text-muted">EEG Pattern:</span> {s.eeg_pattern}</div>
                        <div className="mt-1"><span className="text-muted">AASM Scoring Rule:</span> {s.scoring_rule}</div>
                        {s.clinical_notes && (
                          <div className="mt-1 fst-italic text-muted">{s.clinical_notes}</div>
                        )}
                        {s.key_features && (
                          <div className="mt-1">
                            {s.key_features.map((f, j) => (
                              <span key={j} className="badge bg-light text-dark border me-1">{f}</span>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Arousal Analysis */}
              {bd.arousal_analysis && (
                <div className="card mb-3">
                  <div className="card-header fw-semibold">Arousal Analysis</div>
                  <div className="card-body">
                    <div className="row g-3">
                      {Object.entries(bd.arousal_analysis).map(([k, v]) => (
                        <div key={k} className="col-6 col-md-3">
                          <div className="card border-0 bg-light text-center p-2">
                            <div className="fw-bold">{typeof v === 'number' ? v : JSON.stringify(v)}</div>
                            <div className="small text-muted">{k.replace(/_/g, ' ')}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* Scoring Reliability */}
              {bd.sleep_scoring_reliability && (
                <div className="card mb-3">
                  <div className="card-header fw-semibold">Inter-Rater Reliability (Kappa)</div>
                  <div className="card-body">
                    <div className="row g-3">
                      {Object.entries(bd.sleep_scoring_reliability).map(([k, v]) => (
                        <div key={k} className="col-md-4">
                          <div className="d-flex justify-content-between">
                            <span className="small">{k.replace(/_/g, ' ')}</span>
                            <span className="fw-bold">{typeof v === 'number' ? v.toFixed(2) : v}</span>
                          </div>
                          {typeof v === 'number' && (
                            <div className="progress mt-1" style={{ height: 6 }}>
                              <div className={`progress-bar bg-${v >= 0.8 ? 'success' : v >= 0.6 ? 'warning' : 'danger'}`}
                                style={{ width: `${v * 100}%` }} />
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      )}

      {/* ── ASM IMPACT ── */}
      {tab === 'asm' && (
        <div>
          {!bd ? <div className="text-muted">Loading…</div> : (
            <>
              <div className="card mb-4">
                <div className="card-header fw-semibold">Anti-Seizure Medication Effects on Sleep Architecture</div>
                <div className="card-body">
                  <div className="alert alert-info p-2 mb-3" style={{ fontSize: 12 }}>
                    <strong>ℹ Why this matters:</strong> Many ASMs alter sleep architecture, which can paradoxically worsen seizure control.
                    Sleep deprivation is the #1 modifiable seizure trigger (reported in 28.5% of cases).
                    Selecting sleep-neutral or sleep-beneficial ASMs is a key therapeutic consideration.
                  </div>
                  <table className="table table-bordered table-sm">
                    <thead className="table-dark">
                      <tr>
                        <th>ASM</th>
                        <th>Mechanism</th>
                        <th>N1 Effect</th>
                        <th>N2 Effect</th>
                        <th>N3 Effect</th>
                        <th>REM Effect</th>
                        <th>Sleep Continuity</th>
                        <th>Clinical Note</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(bd.asm_detailed_impact || []).map((a, i) => (
                        <tr key={i}>
                          <td className="fw-semibold">{a.asm}</td>
                          <td className="small text-muted">{a.mechanism}</td>
                          <td><span className={`badge bg-${ASM_IMPACT_COLOR(a.n1_effect)}`} style={{ fontSize: 10 }}>{a.n1_effect}</span></td>
                          <td><span className={`badge bg-${ASM_IMPACT_COLOR(a.n2_effect)}`} style={{ fontSize: 10 }}>{a.n2_effect}</span></td>
                          <td><span className={`badge bg-${ASM_IMPACT_COLOR(a.n3_effect)}`} style={{ fontSize: 10 }}>{a.n3_effect}</span></td>
                          <td><span className={`badge bg-${ASM_IMPACT_COLOR(a.rem_effect)}`} style={{ fontSize: 10 }}>{a.rem_effect}</span></td>
                          <td><span className={`badge bg-${ASM_IMPACT_COLOR(a.sleep_continuity)}`} style={{ fontSize: 10 }}>{a.sleep_continuity}</span></td>
                          <td className="small fst-italic">{a.clinical_note}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Clinical Context */}
              <div className="row g-3">
                <div className="col-md-6">
                  <div className="card border-success">
                    <div className="card-header bg-success text-white fw-semibold">✅ Sleep-Beneficial ASMs</div>
                    <div className="card-body small">
                      <p><strong>Lamotrigine:</strong> Increases REM %, improves continuity, reduces WASO. Preferred when sleep quality is a concern.</p>
                      <p><strong>Levetiracetam:</strong> Sleep-neutral profile. Minimal effect on sleep architecture. Good first-line choice.</p>
                      <p className="mb-0"><strong>Principle:</strong> Sleep-neutral/beneficial ASMs maintain the protective REM stage and do not worsen fragmentation.</p>
                    </div>
                  </div>
                </div>
                <div className="col-md-6">
                  <div className="card border-danger">
                    <div className="card-header bg-danger text-white fw-semibold">⚠ Sleep-Disrupting ASMs</div>
                    <div className="card-body small">
                      <p><strong>Phenobarbital/Benzodiazepines:</strong> Strong REM suppression (↓10-15%). Increase N2 spindles. Worsen fragmentation on withdrawal.</p>
                      <p><strong>Topiramate:</strong> Reduces N3 (slow-wave) sleep. May impair memory consolidation. Adds to cognitive burden.</p>
                      <p className="mb-0"><strong>Principle:</strong> REM-suppressing ASMs reduce the protective stage and may paradoxically lower seizure threshold through sleep fragmentation.</p>
                    </div>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'definitions' && (
        <div>
          {!defs ? <div className="text-muted">Loading…</div> : (
            <>
              {/* Sleep Stages */}
              <div className="card mb-4">
                <div className="card-header fw-semibold">Sleep Stage Definitions (AASM 2023)</div>
                <div className="card-body p-0">
                  {(defs.sleep_stages || []).map((s, i) => (
                    <div key={i} className="p-3 border-bottom">
                      <div className="fw-semibold mb-1">{s.term}</div>
                      <div className="small text-muted">{s.definition}</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Sleep Parameters */}
              <div className="card mb-4">
                <div className="card-header fw-semibold">Sleep Parameters</div>
                <div className="card-body p-0">
                  {(defs.sleep_parameters || []).map((p, i) => (
                    <div key={i} className="p-3 border-bottom">
                      <div className="fw-semibold mb-1">{p.term}</div>
                      <div className="small text-muted">{p.definition}</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* References */}
              {defs.references && (
                <div className="card mb-3">
                  <div className="card-header fw-semibold">References</div>
                  <div className="card-body p-0">
                    {defs.references.map((r, i) => (
                      <div key={i} className="p-2 border-bottom small">{r}</div>
                    ))}
                  </div>
                </div>
              )}

              {/* Methodology */}
              {defs.methodology && (
                <div className="card mb-3">
                  <div className="card-header fw-semibold">Scoring Methodology</div>
                  <div className="card-body small">{defs.methodology}</div>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
