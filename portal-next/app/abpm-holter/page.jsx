'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const pct = (n, total) => total ? ((n / total) * 100).toFixed(1) : '0.0';

const severityColor = s => ({
  Normal: 'success', Mild: 'warning', Moderate: 'danger', Severe: 'dark',
}[s] || 'secondary');

const dippingColor = cat => ({
  'Normal Dipper': 'success',
  'Extreme Dipper': 'warning',
  'Non Dipper': 'danger',
  'Reverse Dipper': 'dark',
}[cat] || 'secondary');

const qtcColor = bucket => {
  if (bucket && bucket.startsWith('Normal')) return 'success';
  if (bucket && bucket.startsWith('Border')) return 'warning';
  return 'danger';
};

export default function ABPMHolterDashboard() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);
  const [patSort, setPatSort] = useState('cardiac_score');
  const [patDir, setPatDir] = useState(-1);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/abpm-holter/overview`).then(r => r.json()),
      fetch(`${API}/api/abpm-holter/breakdown`).then(r => r.json()),
      fetch(`${API}/api/abpm-holter/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov) return <div className="text-muted p-3">Loading ABPM/Holter data…</div>;

  const TABS = [
    { id: 'overview',   label: 'Overview' },
    { id: 'bp',         label: 'Blood Pressure' },
    { id: 'arrhythmia', label: 'Arrhythmia / ECG' },
    { id: 'patients',   label: 'Per Patient' },
    { id: 'defs',       label: 'Definitions' },
  ];

  const sortedPats = bd ? [...(bd.patients || [])].sort((a, b) => {
    const av = a[patSort] ?? 0;
    const bv = b[patSort] ?? 0;
    if (av < bv) return -patDir;
    if (av > bv) return patDir;
    return 0;
  }) : [];

  const sortBy = col => {
    if (patSort === col) setPatDir(d => -d);
    else { setPatSort(col); setPatDir(-1); }
  };
  const sortIcon = col => patSort === col ? (patDir === 1 ? ' ▲' : ' ▼') : '';

  return (
    <div className="container-fluid py-3">
      <h3 className="mb-1">❤️ ABPM/Holter Cardiac Monitoring Dashboard</h3>
      <p className="text-muted small mb-3">
        {ov.total_studies} studies · {ov.total_patients} patients · 24h ambulatory BP + Holter ECG · AED cardiac safety &amp; syncope differential
      </p>

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
        <>
          <div className="row g-3 mb-4">
            {[
              { label: 'Studies', val: ov.total_studies, sub: `${ov.total_patients} patients`, color: 'primary' },
              { label: 'Avg SBP 24h', val: `${ov.avg_sbp_24h} mmHg`, sub: `DBP ${ov.avg_dbp_24h} mmHg`, color: 'info' },
              { label: 'Avg HR 24h', val: `${ov.avg_hr_24h} bpm`, sub: 'Heart rate', color: 'success' },
              { label: 'Avg QTc', val: `${ov.avg_qtc_ms} ms`, sub: 'Normal <440 ms', color: ov.avg_qtc_ms >= 440 ? 'warning' : 'success' },
              { label: 'Adverse Dipping', val: ov.adverse_dipping_count, sub: `${ov.adverse_dipping_pct}% non/reverse`, color: 'danger' },
              { label: 'Avg Risk Score', val: ov.avg_cardiac_score, sub: '0–100 composite', color: 'warning' },
            ].map(k => (
              <div key={k.label} className="col-6 col-md-4 col-xl-2">
                <div className={`card border-${k.color} h-100`}>
                  <div className="card-body text-center p-2">
                    <div className={`fw-bold text-${k.color}`} style={{ fontSize: '1.3rem' }}>{k.val}</div>
                    <div className="small fw-semibold">{k.label}</div>
                    <div className="text-muted" style={{ fontSize: '0.72rem' }}>{k.sub}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="row g-2 mb-4">
            {[
              { label: 'AF Patients', val: ov.af_patients, icon: '⚡', color: 'danger' },
              { label: 'VT Patients', val: ov.vt_patients, icon: '📉', color: 'danger' },
              { label: 'Bradycardia', val: ov.brady_patients, icon: '🐢', color: 'warning' },
              { label: 'ST Depression', val: ov.st_depression_patients, icon: '❗', color: 'warning' },
              { label: 'Total PVCs', val: ov.total_pvc, icon: '🫀', color: 'secondary' },
            ].map(k => (
              <div key={k.label} className="col-6 col-md-4 col-lg-2">
                <div className={`card border-${k.color}`}>
                  <div className="card-body text-center py-2 px-1">
                    <div style={{ fontSize: '1.4rem' }}>{k.icon}</div>
                    <div className={`fw-bold text-${k.color}`}>{k.val}</div>
                    <div className="small text-muted">{k.label}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="row g-3">
            <div className="col-md-4">
              <div className="card h-100">
                <div className="card-header fw-semibold">Dipping Pattern</div>
                <div className="card-body">
                  {(ov.dipping_distribution || []).map(item => (
                    <div key={item.category} className="mb-3">
                      <div className="d-flex justify-content-between small mb-1">
                        <span>{item.label}</span>
                        <span className={`badge bg-${dippingColor(item.label)}`}>{item.count}</span>
                      </div>
                      <div className="progress" style={{ height: 10 }}>
                        <div className={`progress-bar bg-${dippingColor(item.label)}`}
                          style={{ width: `${pct(item.count, ov.total_studies)}%` }} />
                      </div>
                    </div>
                  ))}
                  <p className="text-muted small mb-0">Non/reverse dippers: elevated SUDEP &amp; cardiovascular risk.</p>
                </div>
              </div>
            </div>

            <div className="col-md-4">
              <div className="card h-100">
                <div className="card-header fw-semibold">BP Pattern Labels</div>
                <div className="card-body">
                  {(ov.bp_pattern_distribution || []).map(item => (
                    <div key={item.pattern} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span>{item.pattern}</span>
                        <span className="badge bg-info text-dark">{item.count}</span>
                      </div>
                      <div className="progress" style={{ height: 8 }}>
                        <div className="progress-bar bg-info"
                          style={{ width: `${pct(item.count, ov.total_studies)}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="col-md-4">
              <div className="card mb-3">
                <div className="card-header fw-semibold">Severity</div>
                <div className="card-body">
                  {(ov.severity_distribution || []).map(item => (
                    <div key={item.severity} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span>{item.severity}</span>
                        <span className={`badge bg-${severityColor(item.severity)}`}>{item.count}</span>
                      </div>
                      <div className="progress" style={{ height: 8 }}>
                        <div className={`progress-bar bg-${severityColor(item.severity)}`}
                          style={{ width: `${pct(item.count, ov.total_studies)}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              <div className="card">
                <div className="card-header fw-semibold">QTc Distribution</div>
                <div className="card-body">
                  {(ov.qtc_distribution || []).map(item => (
                    <div key={item.bucket} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span>{item.bucket}</span>
                        <span className={`badge bg-${qtcColor(item.bucket)}`}>{item.count}</span>
                      </div>
                      <div className="progress" style={{ height: 8 }}>
                        <div className={`progress-bar bg-${qtcColor(item.bucket)}`}
                          style={{ width: `${pct(item.count, ov.total_studies)}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      {/* ── BLOOD PRESSURE ── */}
      {tab === 'bp' && bd && (
        <>
          <div className="row g-3 mb-3">
            {[
              { label: 'Avg 24h SBP', val: `${ov.avg_sbp_24h} mmHg`, note: 'Normal <130 mmHg', color: ov.avg_sbp_24h >= 130 ? 'warning' : 'success' },
              { label: 'Avg 24h DBP', val: `${ov.avg_dbp_24h} mmHg`, note: 'Normal <80 mmHg', color: ov.avg_dbp_24h >= 80 ? 'warning' : 'success' },
              { label: 'Avg 24h HR', val: `${ov.avg_hr_24h} bpm`, note: 'Normal 60-100', color: 'info' },
            ].map(k => (
              <div key={k.label} className="col-md-4">
                <div className={`card border-${k.color}`}>
                  <div className="card-body text-center">
                    <div className={`display-5 fw-bold text-${k.color}`}>{k.val}</div>
                    <div className="fw-semibold">{k.label}</div>
                    <div className="text-muted small">{k.note}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="card mb-3">
            <div className="card-header fw-semibold">Day vs Night SBP by Dipping Category</div>
            <div className="card-body p-0">
              <table className="table table-sm table-bordered mb-0">
                <thead className="table-dark">
                  <tr>
                    <th>Dipping Category</th><th>Count</th>
                    <th>Avg Day SBP (mmHg)</th><th>Avg Night SBP (mmHg)</th>
                    <th>Night/Day Ratio</th><th>Clinical Risk</th>
                  </tr>
                </thead>
                <tbody>
                  {(bd.dipping_bp_comparison || []).map((row, i) => {
                    const ratio = row.avg_day_sbp ? (row.avg_night_sbp / row.avg_day_sbp * 100).toFixed(1) : '—';
                    const risk = row.category === 'Reverse Dipper' ? 'Very High' :
                                 row.category === 'Non Dipper' ? 'High' :
                                 row.category === 'Extreme Dipper' ? 'Moderate' : 'Low';
                    const rc = { 'Very High': 'danger', 'High': 'danger', 'Moderate': 'warning', 'Low': 'success' }[risk];
                    return (
                      <tr key={i}>
                        <td className="fw-semibold">{row.category}</td>
                        <td>{row.count}</td>
                        <td>{row.avg_day_sbp}</td>
                        <td className={row.category === 'Reverse Dipper' || row.category === 'Non Dipper' ? 'text-danger fw-bold' : ''}>
                          {row.avg_night_sbp}
                        </td>
                        <td>{ratio}%</td>
                        <td><span className={`badge bg-${rc}`}>{risk}</span></td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          <div className="alert alert-info">
            <strong>Clinical note:</strong> Non-dippers (&lt;10% nocturnal dip) and reverse dippers have
            significantly elevated stroke, organ damage, and SUDEP risk. {ov.adverse_dipping_count} patients
            ({ov.adverse_dipping_pct}%) warrant cardiology review and antihypertensive optimization.
          </div>
        </>
      )}

      {/* ── ARRHYTHMIA / ECG ── */}
      {tab === 'arrhythmia' && bd && (
        <>
          <div className="row g-3 mb-4">
            <div className="col-lg-7">
              <div className="card">
                <div className="card-header fw-semibold">Arrhythmia Burden Summary</div>
                <div className="card-body p-0">
                  <table className="table table-sm table-bordered mb-0">
                    <thead className="table-dark">
                      <tr>
                        <th>Arrhythmia Type</th><th>Total Events</th>
                        <th>Patients Affected</th><th>Prevalence</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(bd.arrhythmia_summary || []).map((row, i) => (
                        <tr key={i}>
                          <td className="fw-semibold small">{row.type}</td>
                          <td>
                            <span className={`badge ${row.total_events > 0 ? 'bg-danger' : 'bg-success'}`}>
                              {row.total_events}
                            </span>
                          </td>
                          <td>{row.patients_affected}</td>
                          <td>
                            <div className="d-flex align-items-center gap-2">
                              <div className="progress flex-grow-1" style={{ height: 8 }}>
                                <div className={`progress-bar ${row.patients_affected > 0 ? 'bg-danger' : 'bg-success'}`}
                                  style={{ width: `${pct(row.patients_affected, ov.total_patients)}%` }} />
                              </div>
                              <span className="small">{pct(row.patients_affected, ov.total_patients)}%</span>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            <div className="col-lg-5">
              <div className="card">
                <div className="card-header fw-semibold">QTc Monitoring (AED Safety)</div>
                <div className="card-body">
                  {(ov.qtc_distribution || []).map(item => (
                    <div key={item.bucket} className="mb-3">
                      <div className="d-flex justify-content-between small mb-1">
                        <span>{item.bucket}</span>
                        <span className={`badge bg-${qtcColor(item.bucket)}`}>{item.count}</span>
                      </div>
                      <div className="progress" style={{ height: 12 }}>
                        <div className={`progress-bar bg-${qtcColor(item.bucket)}`}
                          style={{ width: `${pct(item.count, ov.total_studies)}%` }} />
                      </div>
                    </div>
                  ))}
                  <p className="text-muted small mb-0">
                    Avg QTc: <strong>{ov.avg_qtc_ms} ms</strong> — carbamazepine, lamotrigine, and
                    phenytoin can prolong QTc. Monthly monitoring required (ESC 2022).
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div className="alert alert-warning">
            <strong>Cardiac syncope vs seizure differential:</strong> AF ({ov.af_patients} patients) and
            VT ({ov.vt_patients} patients) can cause syncope mimicking epileptic seizures.
            Holter monitoring is gold standard — 15-20% of patients referred to epilepsy
            clinics have primary cardiac disease (ILAE).
          </div>
        </>
      )}

      {/* ── PER PATIENT ── */}
      {tab === 'patients' && (
        <div className="card">
          <div className="card-header fw-semibold">
            Per-Patient Cardiac Profile ({sortedPats.length} records) — sorted by risk score ↓
          </div>
          <div className="card-body p-0">
            <div className="table-responsive">
              <table className="table table-sm table-bordered table-hover mb-0" style={{ fontSize: '0.76rem' }}>
                <thead className="table-dark">
                  <tr>
                    {[
                      ['patient_id','Patient'], ['study_date','Date'],
                      ['systolic_24h','SBP 24h'], ['diastolic_24h','DBP 24h'],
                      ['heart_rate_24h','HR 24h'], ['qtc_ms','QTc (ms)'],
                      ['dipping_category','Dipping'], ['pattern_label','BP Pattern'],
                      ['severity','Severity'], ['cardiac_score','Risk'],
                      ['af_episodes','AF'], ['vt_runs','VT'],
                      ['pvc_count','PVC'], ['bradycardia_episodes','Brady'],
                      ['st_depression_events','ST'],
                    ].map(([col, label]) => (
                      <th key={col} onClick={() => sortBy(col)}
                        style={{ cursor: 'pointer', whiteSpace: 'nowrap' }}>
                        {label}{sortIcon(col)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {sortedPats.map((p, i) => (
                    <tr key={i} className={p.is_abnormal ? 'table-danger' : ''}>
                      <td className="fw-semibold">{p.patient_id}</td>
                      <td>{p.study_date}</td>
                      <td className={p.systolic_24h >= 130 ? 'text-warning fw-bold' : ''}>{p.systolic_24h}</td>
                      <td className={p.diastolic_24h >= 80 ? 'text-warning fw-bold' : ''}>{p.diastolic_24h}</td>
                      <td>{p.heart_rate_24h}</td>
                      <td className={p.qtc_ms >= 440 ? 'text-danger fw-bold' : ''}>{p.qtc_ms}</td>
                      <td>
                        <span className={`badge bg-${dippingColor(p.dipping_category)}`} style={{ fontSize: '0.65rem' }}>
                          {p.dipping_category}
                        </span>
                      </td>
                      <td>{p.pattern_label}</td>
                      <td>
                        <span className={`badge bg-${severityColor(p.severity)}`}>{p.severity}</span>
                      </td>
                      <td className={`fw-bold ${(p.cardiac_score||0) >= 30 ? 'text-danger' : (p.cardiac_score||0) >= 20 ? 'text-warning' : 'text-success'}`}>
                        {p.cardiac_score}
                      </td>
                      <td className={p.af_episodes > 0 ? 'text-danger fw-bold' : 'text-muted'}>{p.af_episodes ?? '0'}</td>
                      <td className={p.vt_runs > 0 ? 'text-danger fw-bold' : 'text-muted'}>{p.vt_runs ?? '0'}</td>
                      <td className={p.pvc_count > 100 ? 'text-warning' : 'text-muted'}>{p.pvc_count ?? '0'}</td>
                      <td className={p.bradycardia_episodes > 0 ? 'text-warning fw-bold' : 'text-muted'}>{p.bradycardia_episodes ?? '0'}</td>
                      <td className={p.st_depression_events > 0 ? 'text-danger fw-bold' : 'text-muted'}>{p.st_depression_events ?? '0'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'defs' && defs && (
        <div className="row g-3">
          <div className="col-lg-8">
            <div className="card">
              <div className="card-header fw-semibold">Clinical Definitions</div>
              <div className="card-body p-0">
                <table className="table table-sm table-bordered mb-0">
                  <thead className="table-light">
                    <tr><th style={{ width: '30%' }}>Term</th><th>Definition</th></tr>
                  </thead>
                  <tbody>
                    {(defs.terms || []).map(t => (
                      <tr key={t.term}>
                        <td className="fw-semibold">{t.term}</td>
                        <td className="small">{t.definition}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-lg-4">
            <div className="card">
              <div className="card-header fw-semibold">Abbreviations</div>
              <div className="card-body p-0">
                <table className="table table-sm table-bordered mb-0">
                  <tbody>
                    {Object.entries(defs.abbreviations || {}).map(([abbr, full]) => (
                      <tr key={abbr}>
                        <td className="fw-bold">{abbr}</td>
                        <td className="small">{full}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
            <div className="card mt-3">
              <div className="card-header fw-semibold">Data Source</div>
              <div className="card-body small text-muted">
                <p className="mb-1"><strong>{defs.data_source}</strong></p>
                <p className="mb-0">Dashboard: {defs.dashboard}</p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
