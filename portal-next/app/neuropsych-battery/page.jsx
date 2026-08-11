'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const pct = (n, total) => total ? ((n / total) * 100).toFixed(1) : '0.0';
const impColor = flag => ({ none: 'success', mild: 'warning', moderate: 'danger' }[flag] || 'secondary');
const indexBar = val => {
  const pctVal = Math.min(100, Math.max(0, ((val - 50) / 100) * 100));
  const color = val >= 90 ? 'success' : val >= 80 ? 'info' : val >= 70 ? 'warning' : 'danger';
  return (
    <div className="d-flex align-items-center gap-2">
      <div className="progress flex-grow-1" style={{ height: 10 }}>
        <div className={`progress-bar bg-${color}`} style={{ width: `${pctVal}%` }} />
      </div>
      <span className="small fw-bold">{val}</span>
    </div>
  );
};

export default function NeuropsychBatteryDashboard() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);
  const [patSort, setPatSort] = useState('patient_id');
  const [patDir, setPatDir] = useState(1);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/neuropsych-battery/overview`).then(r => r.json()),
      fetch(`${API}/api/neuropsych-battery/breakdown`).then(r => r.json()),
      fetch(`${API}/api/neuropsych-battery/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov) return <div className="text-muted p-3">Loading neuropsychological battery data…</div>;

  const TABS = [
    { id: 'overview',  label: 'Overview' },
    { id: 'domains',   label: 'Cognitive Domains' },
    { id: 'screening', label: 'Screening Scores' },
    { id: 'patients',  label: 'Per Patient' },
    { id: 'defs',      label: 'Definitions' },
  ];

  /* sorted patient table */
  const sortedPatients = bd ? [...(bd.patients || [])].sort((a, b) => {
    const av = a[patSort] ?? '';
    const bv = b[patSort] ?? '';
    if (av < bv) return -patDir;
    if (av > bv) return patDir;
    return 0;
  }) : [];

  const sortBy = col => {
    if (patSort === col) setPatDir(d => -d);
    else { setPatSort(col); setPatDir(1); }
  };
  const sortIcon = col => patSort === col ? (patDir === 1 ? ' ▲' : ' ▼') : '';

  return (
    <div className="container-fluid py-3">
      <h3 className="mb-1">🧠 Neuropsychological Battery Dashboard</h3>
      <p className="text-muted small mb-3">
        {ov.total_assessments} assessments · {ov.total_patients} patients · cognitive index profiles, MoCA/MMSE, PHQ-9/GAD-7, impairment &amp; lateralization
      </p>

      {/* Tab nav */}
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
          {/* KPI row */}
          <div className="row g-3 mb-4">
            {[
              { label: 'Assessments', val: ov.total_assessments, sub: `${ov.total_patients} patients`, color: 'primary' },
              { label: 'Impaired', val: ov.impaired_count, sub: `${ov.impairment_rate_pct}% of assessments`, color: 'warning' },
              { label: 'MoCA Impaired (<26)', val: ov.moca_impaired, sub: `avg MoCA ${ov.avg_moca}`, color: 'danger' },
              { label: 'PHQ-9 Elevated (≥10)', val: ov.phq9_elevated, sub: `avg ${ov.avg_phq9}`, color: 'info' },
              { label: 'GAD-7 Elevated (≥10)', val: ov.gad7_elevated, sub: `avg ${ov.avg_gad7}`, color: 'info' },
            ].map(k => (
              <div key={k.label} className="col-6 col-md-4 col-xl-2">
                <div className={`card border-${k.color} h-100`}>
                  <div className="card-body text-center p-2">
                    <div className={`display-6 fw-bold text-${k.color}`}>{k.val}</div>
                    <div className="small fw-semibold">{k.label}</div>
                    <div className="text-muted" style={{ fontSize: '0.72rem' }}>{k.sub}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="row g-3">
            {/* Impairment distribution */}
            <div className="col-md-4">
              <div className="card h-100">
                <div className="card-header fw-semibold">Impairment Classification</div>
                <div className="card-body">
                  {(ov.impairment_distribution || []).map(item => (
                    <div key={item.flag} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span className="text-capitalize">{item.flag}</span>
                        <span className={`badge bg-${impColor(item.flag)}`}>{item.count}</span>
                      </div>
                      <div className="progress" style={{ height: 8 }}>
                        <div
                          className={`progress-bar bg-${impColor(item.flag)}`}
                          style={{ width: `${pct(item.count, ov.total_assessments)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Battery type */}
            <div className="col-md-4">
              <div className="card h-100">
                <div className="card-header fw-semibold">Battery Type</div>
                <div className="card-body">
                  {(ov.battery_type_distribution || []).map(item => (
                    <div key={item.type} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span>{item.type}</span>
                        <span className="badge bg-secondary">{item.count}</span>
                      </div>
                      <div className="progress" style={{ height: 8 }}>
                        <div className="progress-bar bg-primary"
                          style={{ width: `${pct(item.count, ov.total_assessments)}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Lateralization */}
            <div className="col-md-4">
              <div className="card h-100">
                <div className="card-header fw-semibold">Lateralization Hypothesis</div>
                <div className="card-body">
                  {(ov.lateralization_distribution || []).map(item => (
                    <div key={item.hypothesis} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span className="text-capitalize">{item.hypothesis}</span>
                        <span className="badge bg-info text-dark">{item.count}</span>
                      </div>
                      <div className="progress" style={{ height: 8 }}>
                        <div className="progress-bar bg-info"
                          style={{ width: `${pct(item.count, ov.total_assessments)}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Referral reason */}
            <div className="col-md-6">
              <div className="card">
                <div className="card-header fw-semibold">Referral Reason</div>
                <div className="card-body">
                  {(ov.referral_reason_distribution || []).map(item => (
                    <div key={item.reason} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span>{item.reason}</span>
                        <span className="badge bg-secondary">{item.count}</span>
                      </div>
                      <div className="progress" style={{ height: 8 }}>
                        <div className="progress-bar bg-secondary"
                          style={{ width: `${pct(item.count, ov.total_assessments)}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* MoCA distribution */}
            <div className="col-md-6">
              <div className="card">
                <div className="card-header fw-semibold">MoCA Score Distribution</div>
                <div className="card-body">
                  {(ov.moca_distribution || []).map(item => (
                    <div key={item.bucket} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span>{item.bucket}</span>
                        <span className="badge bg-primary">{item.count}</span>
                      </div>
                      <div className="progress" style={{ height: 8 }}>
                        <div className="progress-bar bg-primary"
                          style={{ width: `${pct(item.count, ov.total_assessments)}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      {/* ── COGNITIVE DOMAINS ── */}
      {tab === 'domains' && (
        <>
          <div className="row g-3 mb-4">
            <div className="col-lg-7">
              <div className="card">
                <div className="card-header fw-semibold">Mean Cognitive Domain Indices (50–150 scale)</div>
                <div className="card-body">
                  <table className="table table-sm table-bordered">
                    <thead className="table-light">
                      <tr>
                        <th>Domain</th>
                        <th>Mean Score</th>
                        <th style={{ width: '50%' }}>Profile</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(ov.index_profile || []).map(row => (
                        <tr key={row.domain}>
                          <td>{row.domain}</td>
                          <td className={`fw-bold text-${row.mean >= 90 ? 'success' : row.mean >= 80 ? 'warning' : 'danger'}`}>
                            {row.mean}
                          </td>
                          <td>{indexBar(row.mean)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <p className="text-muted small mb-0">Score &lt; 85 = below average. Normal range: 90–110.</p>
                </div>
              </div>
            </div>
            <div className="col-lg-5">
              <div className="card h-100">
                <div className="card-header fw-semibold">Domain Weakness (index &lt; 85)</div>
                <div className="card-body">
                  {(bd?.domain_weakness || []).map(row => (
                    <div key={row.domain} className="mb-3">
                      <div className="d-flex justify-content-between small mb-1">
                        <span>{row.domain}</span>
                        <span>
                          <span className="badge bg-danger me-1">{row.below_85_count}</span>
                          <span className="text-muted">{row.pct}%</span>
                        </span>
                      </div>
                      <div className="progress" style={{ height: 10 }}>
                        <div className="progress-bar bg-danger" style={{ width: `${row.pct}%` }} />
                      </div>
                    </div>
                  ))}
                  <p className="text-muted small mb-0">
                    Count of patients with domain index below 85 (≈ 1 SD below average).
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Pre-surgical callout */}
          {bd && (
            <div className="alert alert-warning d-flex align-items-center gap-3">
              <span style={{ fontSize: '1.5rem' }}>🔬</span>
              <div>
                <strong>Pre-Surgical Referrals:</strong>{' '}
                {bd.presurgical_count} patients referred pre-surgically —{' '}
                <strong>{bd.presurgical_impaired}</strong> ({pct(bd.presurgical_impaired, bd.presurgical_count)}%) have mild/moderate cognitive impairment,
                requiring neuropsychological risk review before surgery.
              </div>
            </div>
          )}
        </>
      )}

      {/* ── SCREENING SCORES ── */}
      {tab === 'screening' && (
        <>
          <div className="row g-3 mb-4">
            {[
              { label: 'Avg MoCA', val: ov.avg_moca, max: 30, note: 'Normal ≥26', color: 'primary' },
              { label: 'Avg MMSE', val: ov.avg_mmse, max: 30, note: 'Normal ≥24', color: 'info' },
              { label: 'Avg PHQ-9', val: ov.avg_phq9, max: 27, note: '≥10 = elevated', color: 'warning' },
              { label: 'Avg GAD-7', val: ov.avg_gad7, max: 21, note: '≥10 = elevated', color: 'danger' },
            ].map(k => (
              <div key={k.label} className="col-sm-6 col-md-3">
                <div className={`card border-${k.color}`}>
                  <div className="card-body text-center">
                    <div className={`display-5 fw-bold text-${k.color}`}>{k.val}</div>
                    <div className="fw-semibold">{k.label}</div>
                    <div className="text-muted small">{k.note}</div>
                    <div className="progress mt-2" style={{ height: 8 }}>
                      <div className={`progress-bar bg-${k.color}`} style={{ width: `${(k.val / k.max) * 100}%` }} />
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="row g-3">
            {/* PHQ-9 severity */}
            <div className="col-md-6">
              <div className="card">
                <div className="card-header fw-semibold">PHQ-9 Severity Distribution</div>
                <div className="card-body">
                  {(bd?.phq9_severity_distribution || []).map(item => (
                    <div key={item.severity} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span>{item.severity}</span>
                        <span className="badge bg-warning text-dark">{item.count}</span>
                      </div>
                      <div className="progress" style={{ height: 8 }}>
                        <div className="progress-bar bg-warning"
                          style={{ width: `${pct(item.count, ov.total_assessments)}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* GAD-7 severity */}
            <div className="col-md-6">
              <div className="card">
                <div className="card-header fw-semibold">GAD-7 Severity Distribution</div>
                <div className="card-body">
                  {(bd?.gad7_severity_distribution || []).map(item => (
                    <div key={item.severity} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span>{item.severity}</span>
                        <span className="badge bg-danger">{item.count}</span>
                      </div>
                      <div className="progress" style={{ height: 8 }}>
                        <div className="progress-bar bg-danger"
                          style={{ width: `${pct(item.count, ov.total_assessments)}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      {/* ── PER PATIENT ── */}
      {tab === 'patients' && (
        <div className="card">
          <div className="card-header fw-semibold">
            Per-Patient Battery Detail ({sortedPatients.length} records)
          </div>
          <div className="card-body p-0">
            <div className="table-responsive">
              <table className="table table-sm table-bordered table-hover mb-0" style={{ fontSize: '0.78rem' }}>
                <thead className="table-dark">
                  <tr>
                    {[
                      ['patient_id', 'Patient'],
                      ['battery_type', 'Battery'],
                      ['impairment_flag', 'Impairment'],
                      ['lateralization', 'Lateralization'],
                      ['moca', 'MoCA'],
                      ['mmse', 'MMSE'],
                      ['phq9', 'PHQ-9'],
                      ['gad7', 'GAD-7'],
                      ['memory_index', 'Mem'],
                      ['attention_index', 'Att'],
                      ['executive_index', 'Exec'],
                      ['language_index', 'Lang'],
                      ['processing_speed_index', 'ProcSpd'],
                      ['referral_reason', 'Referral'],
                      ['assessor', 'Assessor'],
                    ].map(([col, label]) => (
                      <th key={col} className="cursor-pointer" onClick={() => sortBy(col)}
                        style={{ cursor: 'pointer', whiteSpace: 'nowrap' }}>
                        {label}{sortIcon(col)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {sortedPatients.map((p, i) => (
                    <tr key={i}>
                      <td className="fw-semibold">{p.patient_id}</td>
                      <td>{p.battery_type}</td>
                      <td>
                        <span className={`badge bg-${impColor(p.impairment_flag)}`}>
                          {p.impairment_flag}
                        </span>
                      </td>
                      <td>{p.lateralization}</td>
                      <td className={p.moca < 26 ? 'text-danger fw-bold' : ''}>{p.moca ?? '—'}</td>
                      <td className={p.mmse < 24 ? 'text-danger fw-bold' : ''}>{p.mmse ?? '—'}</td>
                      <td className={p.phq9 >= 10 ? 'text-warning fw-bold' : ''}>{p.phq9 ?? '—'}</td>
                      <td className={p.gad7 >= 10 ? 'text-warning fw-bold' : ''}>{p.gad7 ?? '—'}</td>
                      <td className={p.memory_index < 85 ? 'text-danger' : ''}>{p.memory_index ?? '—'}</td>
                      <td className={p.attention_index < 85 ? 'text-danger' : ''}>{p.attention_index ?? '—'}</td>
                      <td className={p.executive_index < 85 ? 'text-danger' : ''}>{p.executive_index ?? '—'}</td>
                      <td className={p.language_index < 85 ? 'text-danger' : ''}>{p.language_index ?? '—'}</td>
                      <td className={p.processing_speed_index < 85 ? 'text-danger' : ''}>{p.processing_speed_index ?? '—'}</td>
                      <td>{p.referral_reason}</td>
                      <td className="text-muted">{p.assessor ?? '—'}</td>
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
