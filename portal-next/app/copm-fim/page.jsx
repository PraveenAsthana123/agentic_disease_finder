'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const CHANGE_COLOR = v => v >= 2 ? 'success' : v >= 1 ? 'warning' : 'secondary';
const FIM_LEVEL = v => {
  if (v >= 7) return { label: 'Complete Indep.', color: 'success' };
  if (v >= 6) return { label: 'Modified Indep.', color: 'primary' };
  if (v >= 5) return { label: 'Supervision', color: 'info' };
  if (v >= 4) return { label: 'Min Assist', color: 'warning' };
  if (v >= 3) return { label: 'Mod Assist', color: 'orange' };
  if (v >= 2) return { label: 'Max Assist', color: 'danger' };
  return { label: 'Total Assist', color: 'dark' };
};
const AREA_COLOR = a => ({ self_care: 'primary', productivity: 'success', leisure: 'info' }[a] || 'secondary');
const AREA_LABEL = a => ({ self_care: 'Self-Care', productivity: 'Productivity', leisure: 'Leisure' }[a] || a);

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

function BarChart({ data, labelKey, countKey, colorFn, maxLabel }) {
  if (!data?.length) return <div className="text-muted small">No data</div>;
  const max = Math.max(...data.map(d => d[countKey] || 0), 1);
  return (
    <div>
      {data.map((d, i) => {
        const pct = ((d[countKey] || 0) / max) * 100;
        const color = colorFn ? colorFn(d[labelKey]) : 'primary';
        return (
          <div key={i} className="mb-2">
            <div className="d-flex justify-content-between small mb-1">
              <span className="fw-semibold">{d[labelKey]}</span>
              <span className="text-muted">{d[countKey]}{maxLabel ? ` / ${maxLabel}` : ''}</span>
            </div>
            <div className="progress" style={{ height: 10 }}>
              <div className={`progress-bar bg-${color}`} style={{ width: `${pct}%` }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

function DeltaBar({ baseline, followup, max, label }) {
  const bPct = (baseline / max) * 100;
  const fPct = (followup / max) * 100;
  const delta = followup - baseline;
  return (
    <div className="mb-3">
      <div className="d-flex justify-content-between small mb-1">
        <span className="fw-semibold">{label}</span>
        <span>
          <span className="text-muted">{baseline}</span>
          <span className="text-muted"> → </span>
          <span className="fw-bold">{followup}</span>
          {' '}
          <span className={`badge bg-${delta > 0 ? 'success' : 'secondary'}`}>
            {delta > 0 ? '+' : ''}{delta}
          </span>
        </span>
      </div>
      <div className="progress" style={{ height: 14, background: '#e9ecef' }}>
        <div className="progress-bar bg-secondary bg-opacity-50" style={{ width: `${bPct}%` }} title={`Baseline ${baseline}/${max}`} />
        <div className="progress-bar bg-success" style={{ width: `${Math.max(0, fPct - bPct)}%` }} title={`Improvement +${delta}`} />
      </div>
      <div className="d-flex justify-content-between" style={{ fontSize: 9, color: '#999' }}>
        <span>0</span><span>{max}</span>
      </div>
    </div>
  );
}

export default function CopmFimPage() {
  const [tab, setTab] = useState('overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [search, setSearch] = useState('');
  const [sortKey, setSortKey] = useState('patient_id');
  const [sortAsc, setSortAsc] = useState(true);

  useEffect(() => {
    fetch(`${API}/api/copm-fim/overview`).then(r => r.json()).then(setOverview).catch(() => {});
    fetch(`${API}/api/copm-fim/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    fetch(`${API}/api/copm-fim/definitions`).then(r => r.json()).then(setDefinitions).catch(() => {});
  }, []);

  const ov = overview || {};
  const copm = ov.copm || {};
  const fim = ov.fim || {};

  const filteredPatients = (breakdown?.per_patient || [])
    .filter(p => !search || p.patient_id.toLowerCase().includes(search.toLowerCase()) || p.epilepsy_type?.toLowerCase().includes(search.toLowerCase()))
    .sort((a, b) => {
      const av = a[sortKey] ?? 0, bv = b[sortKey] ?? 0;
      return sortAsc ? (av > bv ? 1 : -1) : (av < bv ? 1 : -1);
    });

  const sort = key => { if (sortKey === key) setSortAsc(!sortAsc); else { setSortKey(key); setSortAsc(true); } };
  const Th = ({ k, children }) => (
    <th className="small" style={{ cursor: 'pointer', whiteSpace: 'nowrap' }} onClick={() => sort(k)}>
      {children} {sortKey === k ? (sortAsc ? '↑' : '↓') : ''}
    </th>
  );

  return (
    <div className="container-fluid py-4" style={{ maxWidth: 1400 }}>
      <h2 className="fw-bold mb-1">COPM / FIM Standardized OT Instruments</h2>
      <p className="text-muted small mb-3">
        Canadian Occupational Performance Measure + Functional Independence Measure —{' '}
        {ov.n_patients || 25} patients · {ov.total_goals || '—'} COPM goals · Baseline vs 3-Month Follow-Up
      </p>

      <ul className="nav nav-tabs mb-4">
        {['overview', 'copm', 'fim', 'patients', 'definitions'].map(t => (
          <li key={t} className="nav-item">
            <button className={`nav-link${tab === t ? ' active' : ''}`} onClick={() => setTab(t)}>
              {t === 'overview' ? 'Overview' : t === 'copm' ? 'COPM Goals' : t === 'fim' ? 'FIM Scores' : t === 'patients' ? 'Per Patient' : 'Definitions'}
            </button>
          </li>
        ))}
      </ul>

      {/* ── OVERVIEW TAB ─────────────────────────────────────── */}
      {tab === 'overview' && (
        <>
          <div className="row g-3 mb-4">
            <div className="col-6 col-md-2"><KPI label="Patients" value={ov.n_patients} color="primary" /></div>
            <div className="col-6 col-md-2"><KPI label="Total Goals" value={ov.total_goals} sub={`avg ${ov.avg_goals_per_patient}/pt`} color="info" /></div>
            <div className="col-6 col-md-2"><KPI label="COPM Perf Change" value={copm.avg_performance_change != null ? `+${copm.avg_performance_change}` : '—'} sub="avg pts (baseline→f/u)" color="success" /></div>
            <div className="col-6 col-md-2"><KPI label="MCID Achieved" value={copm.mcid_achieved_n} sub={`${copm.mcid_achieved_pct}% (≥2pt change)`} color="success" /></div>
            <div className="col-6 col-md-2"><KPI label="FIM Improvement" value={fim.avg_change != null ? `+${fim.avg_change}` : '—'} sub={`avg total (max ${fim.total_max})`} color="primary" /></div>
            <div className="col-6 col-md-2"><KPI label="Driving Restricted" value={ov.driving_restricted_n} sub={`${ov.driving_restricted_pct}% of cohort`} color="warning" warn /></div>
          </div>

          <div className="row g-3 mb-4">
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header py-2 fw-semibold small">COPM — Baseline vs Follow-Up (Population Avg)</div>
                <div className="card-body p-3">
                  <DeltaBar baseline={copm.avg_baseline_performance || 0} followup={copm.avg_followup_performance || 0} max={10} label="Performance" />
                  <DeltaBar baseline={copm.avg_baseline_satisfaction || 0} followup={copm.avg_followup_satisfaction || 0} max={10} label="Satisfaction" />
                  <div className="alert alert-info py-2 small mt-2 mb-0">
                    MCID = ≥2 point change. {copm.mcid_achieved_n}/{ov.n_patients} patients achieved MCID on performance.
                  </div>
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header py-2 fw-semibold small">FIM — Baseline vs Follow-Up (Population Avg)</div>
                <div className="card-body p-3">
                  <DeltaBar baseline={fim.avg_baseline_total || 0} followup={fim.avg_followup_total || 0} max={fim.total_max || 126} label="Total FIM" />
                  <DeltaBar baseline={fim.avg_baseline_motor || 0} followup={fim.avg_followup_motor || 0} max={fim.motor_max || 91} label="Motor" />
                  <DeltaBar baseline={fim.avg_baseline_cognitive || 0} followup={fim.avg_followup_cognitive || 0} max={fim.cognitive_max || 35} label="Cognitive" />
                </div>
              </div>
            </div>
          </div>

          <div className="row g-3">
            <div className="col-md-4">
              <div className="card">
                <div className="card-header py-2 fw-semibold small">Goal Area Distribution</div>
                <div className="card-body p-3">
                  <BarChart data={copm.area_distribution || []} labelKey="area" countKey="count"
                    colorFn={a => AREA_COLOR(a)} />
                  <div className="d-flex gap-2 flex-wrap mt-2">
                    {['self_care', 'productivity', 'leisure'].map(a => (
                      <span key={a} className={`badge bg-${AREA_COLOR(a)}`}>{AREA_LABEL(a)}</span>
                    ))}
                  </div>
                </div>
              </div>
            </div>
            <div className="col-md-4">
              <div className="card">
                <div className="card-header py-2 fw-semibold small">Top COPM Goals (frequency)</div>
                <div className="card-body p-3">
                  <BarChart data={(copm.top_goals || []).slice(0, 6)} labelKey="goal" countKey="count" />
                </div>
              </div>
            </div>
            <div className="col-md-4">
              <div className="card">
                <div className="card-header py-2 fw-semibold small">Performance Change Distribution</div>
                <div className="card-body p-3">
                  <BarChart
                    data={copm.performance_change_distribution || []}
                    labelKey="bin" countKey="count"
                    colorFn={b => b === '≥4' || b === '3–4' ? 'success' : b === '2–3' ? 'primary' : b === '1–2' ? 'warning' : 'secondary'}
                  />
                  <div className="text-muted small mt-2">≥2pt = MCID (clinically meaningful improvement)</div>
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      {/* ── COPM GOALS TAB ───────────────────────────────────── */}
      {tab === 'copm' && (
        <>
          <div className="row g-3 mb-4">
            <div className="col-md-4">
              <div className="card">
                <div className="card-header py-2 fw-semibold small">Average Scores by Area</div>
                <div className="card-body p-3">
                  {['self_care', 'productivity', 'leisure'].map(area => {
                    const pts = (breakdown?.per_patient || []).flatMap(p =>
                      p.goals.filter(g => g.area === area)
                    );
                    if (!pts.length) return null;
                    const avgBase = (pts.reduce((a, g) => a + g.baseline_performance, 0) / pts.length).toFixed(1);
                    const avgFu = (pts.reduce((a, g) => a + g.followup_performance, 0) / pts.length).toFixed(1);
                    return (
                      <div key={area} className="mb-3">
                        <div className="d-flex justify-content-between mb-1">
                          <span className={`badge bg-${AREA_COLOR(area)}`}>{AREA_LABEL(area)}</span>
                          <span className="small text-muted">{pts.length} goals</span>
                        </div>
                        <DeltaBar baseline={parseFloat(avgBase)} followup={parseFloat(avgFu)} max={10} label={`Performance`} />
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
            <div className="col-md-8">
              <div className="card">
                <div className="card-header py-2 fw-semibold small">All COPM Goals — Top 10 by Frequency</div>
                <div className="card-body p-3">
                  <div className="table-responsive">
                    <table className="table table-sm table-hover small mb-0">
                      <thead className="table-light">
                        <tr>
                          <th>Goal</th>
                          <th>Area</th>
                          <th>Count</th>
                          <th>Avg Importance</th>
                          <th>Avg Baseline Perf</th>
                          <th>Avg F/U Perf</th>
                          <th>Avg Change</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(() => {
                          const all = (breakdown?.per_patient || []).flatMap(p => p.goals);
                          const byGoal = {};
                          all.forEach(g => {
                            if (!byGoal[g.goal]) byGoal[g.goal] = { goal: g.goal, area: g.area, goals: [] };
                            byGoal[g.goal].goals.push(g);
                          });
                          return Object.values(byGoal).sort((a, b) => b.goals.length - a.goals.length).slice(0, 10).map((grp, i) => {
                            const gs = grp.goals;
                            const avgImp = (gs.reduce((a, g) => a + g.importance, 0) / gs.length).toFixed(1);
                            const avgBase = (gs.reduce((a, g) => a + g.baseline_performance, 0) / gs.length).toFixed(1);
                            const avgFu = (gs.reduce((a, g) => a + g.followup_performance, 0) / gs.length).toFixed(1);
                            const avgChg = (gs.reduce((a, g) => a + g.performance_change, 0) / gs.length).toFixed(1);
                            return (
                              <tr key={i}>
                                <td style={{ maxWidth: 220, wordBreak: 'break-word' }}>{grp.goal}</td>
                                <td><span className={`badge bg-${AREA_COLOR(grp.area)}`}>{AREA_LABEL(grp.area)}</span></td>
                                <td>{gs.length}</td>
                                <td>{avgImp}/10</td>
                                <td>{avgBase}/10</td>
                                <td>{avgFu}/10</td>
                                <td><span className={`badge bg-${CHANGE_COLOR(parseFloat(avgChg))}`}>+{avgChg}</span></td>
                              </tr>
                            );
                          });
                        })()}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      {/* ── FIM SCORES TAB ───────────────────────────────────── */}
      {tab === 'fim' && (
        <div className="row g-3">
          <div className="col-md-7">
            <div className="card">
              <div className="card-header py-2 fw-semibold small">FIM Subscale Summary — Population Avg</div>
              <div className="card-body p-3">
                {(fim.subscale_summary || []).map((sub, i) => (
                  <div key={i} className="mb-3">
                    <DeltaBar
                      baseline={sub.baseline_avg}
                      followup={sub.followup_avg}
                      max={sub.max_score}
                      label={`${sub.subscale} (${sub.n_items} items, max ${sub.max_score})`}
                    />
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="col-md-5">
            <div className="card mb-3">
              <div className="card-header py-2 fw-semibold small">FIM Score Summary</div>
              <div className="card-body p-3">
                <table className="table table-sm small mb-0">
                  <thead className="table-light">
                    <tr><th>Measure</th><th>Baseline</th><th>F/U</th><th>Change</th><th>Max</th></tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td className="fw-semibold">Total FIM</td>
                      <td>{fim.avg_baseline_total}</td>
                      <td>{fim.avg_followup_total}</td>
                      <td><span className="badge bg-success">+{fim.avg_change}</span></td>
                      <td>{fim.total_max}</td>
                    </tr>
                    <tr>
                      <td className="fw-semibold">Motor</td>
                      <td>{fim.avg_baseline_motor}</td>
                      <td>{fim.avg_followup_motor}</td>
                      <td><span className="badge bg-success">+{(fim.avg_followup_motor - fim.avg_baseline_motor).toFixed(1)}</span></td>
                      <td>{fim.motor_max}</td>
                    </tr>
                    <tr>
                      <td className="fw-semibold">Cognitive</td>
                      <td>{fim.avg_baseline_cognitive}</td>
                      <td>{fim.avg_followup_cognitive}</td>
                      <td><span className="badge bg-success">+{(fim.avg_followup_cognitive - fim.avg_baseline_cognitive).toFixed(1)}</span></td>
                      <td>{fim.cognitive_max}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
            <div className="card">
              <div className="card-header py-2 fw-semibold small">FIM Scoring Scale</div>
              <div className="card-body p-3">
                {[
                  [7, 'Complete Independence', 'success'],
                  [6, 'Modified Independence', 'primary'],
                  [5, 'Supervision/Setup', 'info'],
                  [4, 'Minimal Assistance (≥75% effort)', 'warning'],
                  [3, 'Moderate Assistance (50–74%)', 'warning'],
                  [2, 'Maximal Assistance (25–49%)', 'danger'],
                  [1, 'Total Assistance (<25%)', 'dark'],
                ].map(([score, label, color]) => (
                  <div key={score} className="d-flex align-items-center gap-2 mb-1">
                    <span className={`badge bg-${color}`} style={{ minWidth: 22 }}>{score}</span>
                    <span className="small">{label}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── PER PATIENT TAB ──────────────────────────────────── */}
      {tab === 'patients' && (
        <>
          <div className="d-flex gap-3 mb-3 align-items-center">
            <input className="form-control form-control-sm" style={{ maxWidth: 240 }}
              placeholder="Search patient / epilepsy type…" value={search}
              onChange={e => setSearch(e.target.value)} />
            <span className="text-muted small">{filteredPatients.length} patients</span>
          </div>
          {selectedPatient && (() => {
            const p = selectedPatient;
            return (
              <div className="card mb-4 border-primary">
                <div className="card-header py-2 d-flex justify-content-between align-items-center">
                  <span className="fw-bold">{p.patient_id} — Detailed Profile</span>
                  <button className="btn btn-sm btn-outline-secondary" onClick={() => setSelectedPatient(null)}>Close</button>
                </div>
                <div className="card-body p-3">
                  <div className="row g-3 mb-3">
                    <div className="col-md-6">
                      <strong className="small">COPM Goals</strong>
                      <div className="table-responsive mt-2">
                        <table className="table table-sm table-bordered small mb-0">
                          <thead className="table-light">
                            <tr><th>Goal</th><th>Area</th><th>Imp</th><th>Base P/S</th><th>F/U P/S</th><th>ΔP</th></tr>
                          </thead>
                          <tbody>
                            {(p.goals || []).map((g, i) => (
                              <tr key={i}>
                                <td style={{ maxWidth: 180, wordBreak: 'break-word' }}>{g.goal}</td>
                                <td><span className={`badge bg-${AREA_COLOR(g.area)}`} style={{ fontSize: 9 }}>{AREA_LABEL(g.area)}</span></td>
                                <td>{g.importance}</td>
                                <td>{g.baseline_performance}/{g.baseline_satisfaction}</td>
                                <td>{g.followup_performance}/{g.followup_satisfaction}</td>
                                <td><span className={`badge bg-${CHANGE_COLOR(g.performance_change)}`}>+{g.performance_change}</span></td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                    <div className="col-md-6">
                      <strong className="small">FIM Subscales</strong>
                      <div className="mt-2">
                        {Object.entries(p.fim_items || {}).map(([subscale, items]) => (
                          <div key={subscale} className="mb-3">
                            <div className="fw-semibold small text-muted mb-1">{subscale}</div>
                            {Object.entries(items).map(([item, scores]) => (
                              <div key={item} className="d-flex justify-content-between align-items-center mb-1">
                                <span className="small" style={{ fontSize: 11 }}>{item}</span>
                                <span>
                                  <span className="text-muted small">{scores.baseline}</span>
                                  <span className="text-muted small"> → </span>
                                  <span className="fw-bold small">{scores.followup}</span>
                                  {scores.followup > scores.baseline && <span className="badge bg-success ms-1" style={{ fontSize: 9 }}>+{scores.followup - scores.baseline}</span>}
                                </span>
                              </div>
                            ))}
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            );
          })()}
          <div className="card">
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-hover small mb-0">
                  <thead className="table-light">
                    <tr>
                      <Th k="patient_id">Patient</Th>
                      <Th k="age">Age</Th>
                      <th>Epilepsy Type</th>
                      <th>Sz Freq</th>
                      <Th k="n_goals">Goals</Th>
                      <Th k="copm_baseline_performance">COPM Base P</Th>
                      <Th k="copm_followup_performance">COPM F/U P</Th>
                      <Th k="copm_performance_change">COPM ΔP</Th>
                      <Th k="fim_baseline">FIM Base</Th>
                      <Th k="fim_followup">FIM F/U</Th>
                      <Th k="fim_change">FIM Δ</Th>
                      <th>Driving</th>
                      <th>Detail</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredPatients.map((p, i) => (
                      <tr key={i}>
                        <td className="fw-semibold">{p.patient_id}</td>
                        <td>{p.age}</td>
                        <td style={{ maxWidth: 140, wordBreak: 'break-word' }}>{p.epilepsy_type}</td>
                        <td><span className={`badge bg-${p.seizure_frequency === 'Seizure-free' ? 'success' : p.seizure_frequency === '1–3/month' ? 'warning' : 'danger'}`} style={{ fontSize: 9 }}>{p.seizure_frequency}</span></td>
                        <td>{p.n_goals}</td>
                        <td>{p.copm_baseline_performance}/10</td>
                        <td>{p.copm_followup_performance}/10</td>
                        <td><span className={`badge bg-${CHANGE_COLOR(p.copm_performance_change)}`}>+{p.copm_performance_change}</span></td>
                        <td>{p.fim_baseline}/{fim.total_max || 126}</td>
                        <td>{p.fim_followup}/{fim.total_max || 126}</td>
                        <td><span className={`badge bg-${p.fim_change > 0 ? 'success' : 'secondary'}`}>+{p.fim_change}</span></td>
                        <td>{p.driving_restricted ? <span className="badge bg-warning text-dark">Restricted</span> : <span className="badge bg-success">OK</span>}</td>
                        <td><button className="btn btn-outline-primary btn-sm py-0" style={{ fontSize: 10 }} onClick={() => setSelectedPatient(p)}>View</button></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </>
      )}

      {/* ── DEFINITIONS TAB ──────────────────────────────────── */}
      {tab === 'definitions' && definitions && (
        <div className="row g-3">
          {Object.entries(definitions.instruments || {}).map(([key, inst]) => (
            <div key={key} className="col-md-6">
              <div className="card h-100">
                <div className="card-header py-2 fw-bold">{inst.full_name} ({key})</div>
                <div className="card-body p-3">
                  <p className="small text-muted mb-2">{inst.developer}</p>
                  <p className="small mb-2"><strong>Purpose:</strong> {inst.purpose}</p>
                  <p className="small mb-2"><strong>Administration:</strong> {inst.administration}</p>
                  <p className="small mb-2"><strong>Scoring:</strong> {inst.scoring}</p>
                  {inst.total_range && <p className="small mb-2"><strong>Range:</strong> {inst.total_range}</p>}
                  {inst.motor_subscale && <p className="small mb-2"><strong>Motor:</strong> {inst.motor_subscale}</p>}
                  {inst.cognitive_subscale && <p className="small mb-2"><strong>Cognitive:</strong> {inst.cognitive_subscale}</p>}
                  <div className="alert alert-success py-2 small mb-2">
                    <strong>MCID:</strong> {inst.mcid}
                  </div>
                  <p className="small mb-0 text-muted"><strong>Epilepsy relevance:</strong> {inst.epilepsy_relevance}</p>
                  {inst.areas && (
                    <div className="mt-2">
                      {Object.entries(inst.areas).map(([a, desc]) => (
                        <div key={a} className="d-flex gap-2 align-items-start mb-1">
                          <span className={`badge bg-${AREA_COLOR(a)}`}>{AREA_LABEL(a)}</span>
                          <span className="small text-muted">{desc}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
          <div className="col-12">
            <div className="card">
              <div className="card-header py-2 fw-semibold small">Clinical Context — Epilepsy OT</div>
              <div className="card-body p-3">
                <div className="row g-3">
                  <div className="col-md-4">
                    <strong className="small">Driving Cessation</strong>
                    <p className="small text-muted mt-1">{definitions.clinical_context?.driving_cessation}</p>
                  </div>
                  <div className="col-md-4">
                    <strong className="small">Seizure Safety Adaptations</strong>
                    <ul className="small text-muted mt-1 ps-3 mb-0">
                      {(definitions.clinical_context?.seizure_safety_adaptations || []).map((s, i) => (
                        <li key={i}>{s}</li>
                      ))}
                    </ul>
                  </div>
                  <div className="col-md-4">
                    <strong className="small">Return to Work</strong>
                    <p className="small text-muted mt-1">{definitions.clinical_context?.return_to_work}</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div className="col-12">
            <div className="card">
              <div className="card-header py-2 fw-semibold small">References</div>
              <div className="card-body p-3">
                <ol className="small text-muted mb-0">
                  {(definitions.references || []).map((r, i) => <li key={i}>{r}</li>)}
                </ol>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
