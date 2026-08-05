'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const fmt = v => (v == null ? '--' : typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v));

const CAND_COLOR = {
  strong_candidate: '#22c55e',
  possible_candidate: '#eab308',
  not_candidate:     '#ef4444',
};
const CAND_LABEL = {
  strong_candidate:  'Strong Candidate',
  possible_candidate:'Possible Candidate',
  not_candidate:     'Not Candidate',
};

function Badge({ cls }) {
  const c = CAND_COLOR[cls] || '#94a3b8';
  return (
    <span style={{
      display:'inline-block', padding:'2px 9px', borderRadius:6,
      fontSize:11, fontWeight:600, background:c+'22', color:c,
    }}>
      {CAND_LABEL[cls] || cls || '—'}
    </span>
  );
}

function KPI({ label, value, color }) {
  return (
    <div className="col-6 col-md-3 mb-3">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center py-3">
          <div className="fw-bold" style={{ fontSize:26, color: color||'#1e293b' }}>{value}</div>
          <div className="text-muted small mt-1">{label}</div>
        </div>
      </div>
    </div>
  );
}

/* simple inline bar */
function Bar({ label, count, total, color }) {
  const pct = total ? ((count / total) * 100).toFixed(1) : 0;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span>
        <span className="fw-semibold">{count} ({pct}%)</span>
      </div>
      <div style={{ background:'#f1f5f9', borderRadius:4, height:8 }}>
        <div style={{ background: color||'#3b82f6', width:`${pct}%`, height:8, borderRadius:4 }} />
      </div>
    </div>
  );
}

export default function PreSurgicalEvaluationPage() {
  const [ov, setOv]     = useState(null);
  const [bd, setBd]     = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab]   = useState('overview');
  const [sort, setSort] = useState({ col:'score', dir:'desc' });
  const [err, setErr]   = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/presurgical-evaluation/overview`).then(r => r.json()),
      fetch(`${API}/api/presurgical-evaluation/breakdown`).then(r => r.json()),
      fetch(`${API}/api/presurgical-evaluation/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov)  return <div className="text-muted p-4">Loading pre-surgical evaluation data…</div>;

  const TABS = [
    { id: 'overview',  label: 'Overview' },
    { id: 'candidates',label: 'Candidates' },
    { id: 'workup',    label: 'Workup Gaps' },
    { id: 'lesion',    label: 'Lesion Analysis' },
    { id: 'defs',      label: 'Definitions' },
  ];

  const kpi  = ov.kpis || {};
  const total = kpi.total_evaluated || 0;

  /* sorted patient list */
  const patients = [...(bd?.patients || [])].sort((a, b) => {
    const av = a[sort.col] ?? 0, bv = b[sort.col] ?? 0;
    return sort.dir === 'asc' ? (av > bv ? 1 : -1) : (av < bv ? 1 : -1);
  });

  const toggleSort = col => setSort(s => ({
    col, dir: s.col === col && s.dir === 'desc' ? 'asc' : 'desc',
  }));
  const sortIcon = col => sort.col === col ? (sort.dir === 'desc' ? ' ▼' : ' ▲') : '';

  const gap = bd?.gap_analysis || {};

  return (
    <div className="p-3">
      <h3>Pre-Surgical Evaluation</h3>
      <p className="text-muted">
        Surgery candidacy assessment for {total} patients &mdash;{' '}
        {kpi.strong_candidates} strong candidates, {kpi.possible_candidates} possible,{' '}
        {kpi.not_candidates} not candidates &mdash; avg candidacy score {kpi.avg_candidacy_score}.
        {' '}{kpi.lesional_cases} lesional cases, {kpi.hippocampal_sclerosis} with hippocampal sclerosis.
      </p>

      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={`nav-link ${tab === t.id ? 'active' : ''}`}
              onClick={() => setTab(t.id)}
            >{t.label}</button>
          </li>
        ))}
      </ul>

      {/* ── OVERVIEW ── */}
      {tab === 'overview' && (
        <div>
          <div className="row">
            <KPI label="Evaluated"           value={total}                      color="#3b82f6" />
            <KPI label="Strong Candidates"   value={kpi.strong_candidates}      color="#22c55e" />
            <KPI label="Possible Candidates" value={kpi.possible_candidates}    color="#eab308" />
            <KPI label="Not Candidates"      value={kpi.not_candidates}         color="#ef4444" />
            <KPI label="Avg Candidacy Score" value={kpi.avg_candidacy_score}    color="#8b5cf6" />
            <KPI label="Lesional Cases"      value={kpi.lesional_cases}         color="#06b6d4" />
            <KPI label="Hippocampal Sclerosis" value={kpi.hippocampal_sclerosis} color="#f97316" />
            <KPI label="Non-Lesional"        value={(ov.lesion_type_distribution||[]).find(l=>l.type==='NL')?.count || 0} color="#64748b" />
          </div>

          <div className="row mt-2">
            <div className="col-md-4 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-body">
                  <h6 className="card-title">Candidacy Distribution</h6>
                  {(ov.candidacy_distribution || []).map(c => (
                    <Bar
                      key={c.status}
                      label={c.status}
                      count={c.count}
                      total={total}
                      color={
                        c.status === 'Strong Candidate'   ? '#22c55e' :
                        c.status === 'Possible Candidate' ? '#eab308' : '#ef4444'
                      }
                    />
                  ))}
                </div>
              </div>
            </div>

            <div className="col-md-4 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-body">
                  <h6 className="card-title">Lesion Types</h6>
                  {(ov.lesion_type_distribution || []).map((l, i) => {
                    const colors = ['#3b82f6','#22c55e','#64748b','#f97316','#8b5cf6','#06b6d4','#ec4899','#eab308'];
                    return (
                      <Bar
                        key={l.type}
                        label={`${l.type} — ${(ov.lesion_type_labels||{})[l.type] || l.type}`}
                        count={l.count}
                        total={total}
                        color={colors[i % colors.length]}
                      />
                    );
                  })}
                </div>
              </div>
            </div>

            <div className="col-md-4 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-body">
                  <h6 className="card-title">Laterality</h6>
                  {(ov.laterality_distribution || []).map((lat, i) => {
                    const colors2 = ['#3b82f6','#f97316','#64748b','#22c55e'];
                    return (
                      <Bar
                        key={lat.side}
                        label={lat.side}
                        count={lat.count}
                        total={total}
                        color={colors2[i % colors2.length]}
                      />
                    );
                  })}

                  <h6 className="card-title mt-4">Score Histogram</h6>
                  {(ov.score_histogram || []).map(h => (
                    <Bar
                      key={h.range}
                      label={`Score ${h.range}`}
                      count={h.count}
                      total={total}
                      color="#8b5cf6"
                    />
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── CANDIDATES TABLE ── */}
      {tab === 'candidates' && (
        <div className="card shadow-sm">
          <div className="card-body">
            <h6>All Evaluated Patients — sorted by {sort.col} ({sort.dir})</h6>
            <div className="table-responsive">
              <table className="table table-sm table-hover">
                <thead className="table-dark">
                  <tr>
                    {[
                      ['patient_id','Patient'],
                      ['score','Score'],
                      ['classification','Candidacy'],
                      ['lesion_info','Lesion'],
                      ['aed_count','AEDs'],
                      ['workup_completeness_pct','Workup %'],
                    ].map(([col, label]) => (
                      <th key={col} onClick={() => toggleSort(col)} style={{ cursor:'pointer', whiteSpace:'nowrap' }}>
                        {label}{sortIcon(col)}
                      </th>
                    ))}
                    <th>MRI</th>
                    <th>EEG</th>
                    <th>Diary</th>
                    <th>Meds</th>
                  </tr>
                </thead>
                <tbody>
                  {patients.map(p => (
                    <tr key={p.patient_id}>
                      <td className="fw-semibold">{p.patient_id}</td>
                      <td>
                        <span style={{ fontWeight:600, color: p.score >= 70 ? '#22c55e' : p.score >= 45 ? '#eab308' : '#ef4444' }}>
                          {p.score}
                        </span>
                      </td>
                      <td><Badge cls={p.classification} /></td>
                      <td className="small">{p.lesion_info || '—'}</td>
                      <td>{p.aed_count ?? '—'}</td>
                      <td>
                        <div className="d-flex align-items-center gap-1">
                          <div style={{ background:'#f1f5f9', borderRadius:4, height:6, width:60 }}>
                            <div style={{ background:'#3b82f6', width:`${p.workup_completeness_pct}%`, height:6, borderRadius:4 }} />
                          </div>
                          <span className="small">{p.workup_completeness_pct}%</span>
                        </div>
                      </td>
                      <td>{p.has_mri ? '✅' : '❌'}</td>
                      <td>{p.has_eeg ? '✅' : '❌'}</td>
                      <td>{p.has_seizure_diary ? '✅' : '❌'}</td>
                      <td>{p.has_medication_history ? '✅' : '❌'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ── WORKUP GAPS ── */}
      {tab === 'workup' && (
        <div className="row">
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-body">
                <h6>Workup Coverage Gaps</h6>
                <p className="text-muted small">
                  Pre-surgical workup completeness across all {total} evaluated patients.
                  Missing data elements reduce candidacy score accuracy.
                </p>
                {[
                  { label:'MRI Completed',           pct: gap.mri_coverage,        color:'#3b82f6' },
                  { label:'EEG Available',            pct: gap.eeg_coverage,        color:'#22c55e' },
                  { label:'Seizure Diary Present',    pct: gap.seizure_diary_coverage, color:'#eab308' },
                  { label:'Medication History',       pct: gap.medication_coverage, color:'#f97316' },
                ].map(g => (
                  <div key={g.label} className="mb-3">
                    <div className="d-flex justify-content-between small mb-1">
                      <span className="fw-semibold">{g.label}</span>
                      <span style={{ color: g.pct >= 80 ? '#22c55e' : g.pct >= 50 ? '#eab308' : '#ef4444', fontWeight:600 }}>
                        {g.pct}%
                      </span>
                    </div>
                    <div style={{ background:'#f1f5f9', borderRadius:6, height:12 }}>
                      <div style={{ background: g.color, width:`${g.pct}%`, height:12, borderRadius:6 }} />
                    </div>
                    <div className="text-muted small mt-1">
                      {g.pct < 100 && `${Math.round(total * (100 - g.pct) / 100)} patients missing this element`}
                      {g.pct === 100 && 'Complete — all patients have this element'}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-body">
                <h6>Patients by Workup Completeness</h6>
                {[
                  { tier:'Complete (100%)',     color:'#22c55e', patients: patients.filter(p => p.workup_completeness_pct === 100) },
                  { tier:'Partial (75%)',       color:'#eab308', patients: patients.filter(p => p.workup_completeness_pct === 75) },
                  { tier:'Incomplete (<75%)',   color:'#ef4444', patients: patients.filter(p => p.workup_completeness_pct < 75) },
                ].map(t => (
                  <div key={t.tier} className="mb-3">
                    <div className="d-flex justify-content-between small mb-1">
                      <span>{t.tier}</span>
                      <span className="fw-semibold" style={{ color: t.color }}>{t.patients.length} patients</span>
                    </div>
                    <div style={{ background:'#f1f5f9', borderRadius:4, height:8 }}>
                      <div style={{ background: t.color, width:`${(t.patients.length/total)*100}%`, height:8, borderRadius:4 }} />
                    </div>
                  </div>
                ))}

                <div className="mt-3">
                  <h6 className="small text-muted text-uppercase">Key Workup Elements</h6>
                  {[
                    ['MRI Brain + Epilepsy Protocol', 'Structural imaging to identify lesion (high-resolution T2, FLAIR, MPRAGE)'],
                    ['EEG / Video-EEG', 'Ictal + interictal recordings for localisation; SEEG if non-lesional'],
                    ['Seizure Diary', 'Frequency, severity, semiology — required for surgical outcome prediction'],
                    ['Medication History', 'Document AED trials to confirm drug resistance (ILAE definition: ≥2 AEDs)'],
                    ['Neuropsychology', 'Cognitive baseline, Wada test if dominant-hemisphere surgery planned'],
                    ['PET / SPECT', 'Ictal SPECT + FDG-PET for non-lesional cases or discordant workup'],
                  ].map(([element, desc]) => (
                    <div key={element} className="border-bottom py-2">
                      <div className="fw-semibold small">{element}</div>
                      <div className="text-muted" style={{ fontSize:12 }}>{desc}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── LESION ANALYSIS ── */}
      {tab === 'lesion' && (
        <div className="row">
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-body">
                <h6>Lesion Type Breakdown</h6>
                {(ov.lesion_type_distribution || []).map((l, i) => {
                  const colors = ['#3b82f6','#22c55e','#64748b','#f97316','#8b5cf6','#06b6d4','#ec4899','#eab308'];
                  const label = (ov.lesion_type_labels||{})[l.type] || l.type;
                  const desc = (defs?.lesion_types || {})[l.type] || '';
                  return (
                    <div key={l.type} className="border-bottom pb-2 mb-2">
                      <div className="d-flex justify-content-between align-items-start">
                        <div>
                          <span className="fw-semibold" style={{ color: colors[i % colors.length] }}>{l.type}</span>
                          {' — '}
                          <span className="small">{label}</span>
                        </div>
                        <span className="badge" style={{ background: colors[i % colors.length]+'22', color: colors[i % colors.length] }}>
                          {l.count} ({total ? ((l.count/total)*100).toFixed(1) : 0}%)
                        </span>
                      </div>
                      {desc && <div className="text-muted small mt-1">{desc}</div>}
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-body">
                <h6>Laterality Distribution</h6>
                {(ov.laterality_distribution || []).map((lat, i) => {
                  const lc = ['#3b82f6','#f97316','#64748b','#22c55e'];
                  return (
                    <div key={lat.side} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span>{lat.side}</span>
                        <span className="fw-semibold">{lat.count} ({total ? ((lat.count/total)*100).toFixed(1) : 0}%)</span>
                      </div>
                      <div style={{ background:'#f1f5f9', borderRadius:4, height:8 }}>
                        <div style={{ background: lc[i%lc.length], width:`${(lat.count/total)*100}%`, height:8, borderRadius:4 }} />
                      </div>
                    </div>
                  );
                })}

                <h6 className="mt-4">Surgical Procedures</h6>
                <ul className="list-unstyled small">
                  {(defs?.surgical_procedures || []).map(p => (
                    <li key={p} className="border-bottom py-1">⚕ {p}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'defs' && defs && (
        <div className="row">
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-body">
                <h6>Candidacy Score Criteria</h6>
                <p className="text-muted small">{defs.description}</p>
                <table className="table table-sm table-hover">
                  <thead className="table-light">
                    <tr>
                      <th>Criterion</th>
                      <th>Weight</th>
                      <th>Description</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(defs.metrics || []).map(m => (
                      <tr key={m.name}>
                        <td className="fw-semibold small">{m.name}</td>
                        <td className="small">{m.weight ? `${m.weight} pts` : m.range || '—'}</td>
                        <td className="small text-muted">{m.description}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-body">
                <h6>Engel Surgical Outcome Scale</h6>
                <table className="table table-sm table-hover">
                  <thead className="table-light">
                    <tr><th>Class</th><th>Outcome</th></tr>
                  </thead>
                  <tbody>
                    {Object.entries(defs.engel_outcomes || {}).map(([cls, desc]) => (
                      <tr key={cls}>
                        <td>
                          <span className={`badge bg-${cls==='I'?'success':cls==='II'?'info':cls==='III'?'warning':'danger'}`}>
                            {cls}
                          </span>
                        </td>
                        <td className="small">{desc}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>

                <h6 className="mt-3">Candidacy Thresholds</h6>
                {[
                  { label:'Strong Candidate', range:'Score ≥ 70', color:'#22c55e' },
                  { label:'Possible Candidate', range:'Score 45–69', color:'#eab308' },
                  { label:'Not Candidate', range:'Score < 45', color:'#ef4444' },
                ].map(t => (
                  <div key={t.label} className="d-flex justify-content-between border-bottom py-2">
                    <span className="small fw-semibold" style={{ color: t.color }}>{t.label}</span>
                    <span className="small text-muted">{t.range}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
