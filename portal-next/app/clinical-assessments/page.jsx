'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const levelColor = l => {
  const s = (l || '').toLowerCase();
  if (s === 'severe' || s === 'critical') return 'danger';
  if (s === 'moderate' || s === 'borderline') return 'warning';
  if (s === 'mild' || s === 'low average') return 'info';
  if (s === 'normal' || s === 'none') return 'success';
  return 'secondary';
};

export default function ClinicalAssessmentsPage() {
  const [ov, setOv]     = useState(null);
  const [bd, setBd]     = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab]   = useState('overview');
  const [selInst, setSelInst] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/clinical-assessments/overview`).then(r => r.json()).then(d => { setOv(d); }).catch(() => {});
    fetch(`${API}/api/clinical-assessments/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/clinical-assessments/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const tabs = [
    { id: 'overview',    label: 'Overview' },
    { id: 'instruments', label: 'Per-Instrument' },
    { id: 'patients',   label: 'Patient Alerts' },
    { id: 'definitions',label: 'Definitions' },
  ];

  const levelEntries = Object.entries(ov.level_distribution || {}).sort((a, b) => b[1] - a[1]);
  const alertEntries = Object.entries(ov.alert_distribution || {}).sort((a, b) => b[1] - a[1]);
  const instruments = ov.instrument_summary || [];
  const defaultInst = selInst || (instruments.length ? instruments[0].instrument : null);

  return (
    <div>
      <h3>Clinical Assessments Dashboard</h3>
      <p className="text-muted small">
        16 validated neuropsychological instruments · 424 assessments · 29 patients ·
        Source: <code>assessments</code> table · clinical.db
      </p>

      {/* KPI cards */}
      <div className="row mb-3">
        {[
          { label: 'Total Assessments',   value: ov.kpis.total_assessments,      color: 'primary' },
          { label: 'Patients',            value: ov.kpis.distinct_patients,       color: 'info' },
          { label: 'Instruments',         value: ov.kpis.distinct_instruments,    color: 'secondary' },
          { label: 'Alerts Flagged',      value: ov.kpis.total_alerts,            color: ov.kpis.total_alerts > 0 ? 'warning' : 'success' },
          { label: 'Severe / Critical',   value: ov.kpis.severe_or_critical,      color: ov.kpis.severe_or_critical > 0 ? 'danger' : 'success' },
          { label: 'Safety Escalations',  value: ov.kpis.safety_escalations,      color: ov.kpis.safety_escalations > 0 ? 'danger' : 'success' },
        ].map(c => (
          <div key={c.label} className="col-6 col-md-4 col-lg-2 mb-2" style={{ minWidth: 130 }}>
            <div className="card text-center shadow-sm border-0">
              <div className="card-body py-2">
                <div className={`h4 mb-0 text-${c.color}`}>{c.value}</div>
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
          {/* Instrument summary table */}
          <div className="col-lg-7 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Instrument Summary</div>
              <div className="card-body p-0">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light">
                    <tr>
                      <th>Instrument</th>
                      <th>N</th>
                      <th>Avg Score</th>
                      <th>Max</th>
                      <th>Alerts</th>
                      <th>Severe</th>
                    </tr>
                  </thead>
                  <tbody>
                    {instruments.map(i => (
                      <tr key={i.instrument} className={i.alerts > 0 ? 'table-warning' : ''}>
                        <td><button className="btn btn-link btn-sm p-0" onClick={() => { setSelInst(i.instrument); setTab('instruments'); }}>{i.instrument}</button></td>
                        <td>{i.n}</td>
                        <td>{i.avg_score ?? '—'}</td>
                        <td>{i.max_score ?? '—'}</td>
                        <td>{i.alerts > 0 ? <span className="badge bg-warning text-dark">{i.alerts}</span> : <span className="text-muted">0</span>}</td>
                        <td>{i.severe_count > 0 ? <span className="badge bg-danger">{i.severe_count}</span> : <span className="text-muted">0</span>}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Level distribution + Alert distribution */}
          <div className="col-lg-5 mb-3">
            <div className="card shadow-sm mb-3">
              <div className="card-header fw-bold">Severity Level Distribution</div>
              <div className="card-body">
                {levelEntries.map(([lev, cnt]) => (
                  <div key={lev} className="d-flex align-items-center mb-2">
                    <span className={`badge bg-${levelColor(lev)} me-2`} style={{ minWidth: 90, fontSize: '0.7rem' }}>{lev || 'unspecified'}</span>
                    <div className="flex-grow-1 me-2">
                      <div className="progress" style={{ height: 16 }}>
                        <div className={`progress-bar bg-${levelColor(lev)}`}
                             style={{ width: `${ov.kpis.total_assessments ? cnt / ov.kpis.total_assessments * 100 : 0}%` }}>
                          {cnt}
                        </div>
                      </div>
                    </div>
                    <span className="small text-muted">{ov.kpis.total_assessments ? Math.round(cnt / ov.kpis.total_assessments * 100) : 0}%</span>
                  </div>
                ))}
              </div>
            </div>

            {alertEntries.length > 0 && (
              <div className="card shadow-sm border-warning">
                <div className="card-header fw-bold text-warning">Alert Types</div>
                <div className="card-body p-2">
                  {alertEntries.map(([msg, cnt]) => (
                    <div key={msg} className="d-flex justify-content-between align-items-start mb-1 border-bottom pb-1">
                      <span className="small" style={{ maxWidth: '85%' }}>{msg}</span>
                      <span className="badge bg-warning text-dark ms-2">{cnt}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Per-Instrument Tab ─────────────────────────────── */}
      {tab === 'instruments' && bd && (
        <div>
          <div className="mb-3">
            {instruments.map(i => (
              <button key={i.instrument}
                className={`btn btn-sm me-1 mb-1 ${defaultInst === i.instrument ? 'btn-primary' : 'btn-outline-secondary'}`}
                onClick={() => setSelInst(i.instrument)}>
                {i.instrument}
                {i.alerts > 0 && <span className="badge bg-warning text-dark ms-1">{i.alerts}</span>}
              </button>
            ))}
          </div>
          {defaultInst && bd.score_tables?.[defaultInst] && (
            <div className="card shadow-sm">
              <div className="card-header fw-bold">{defaultInst} — Score Table</div>
              <div className="card-body p-0">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light">
                    <tr><th>Patient</th><th>Score</th><th>Max</th><th>Level</th><th>Alert</th><th>Examiner</th><th>Date</th></tr>
                  </thead>
                  <tbody>
                    {bd.score_tables[defaultInst].map((r, i) => (
                      <tr key={i} className={r.alert ? 'table-warning' : ''}>
                        <td><code>{r.patient_id}</code></td>
                        <td><strong>{r.score}</strong></td>
                        <td className="text-muted">{r.max_score ?? '—'}</td>
                        <td><span className={`badge bg-${levelColor(r.level)}`}>{r.level || '—'}</span></td>
                        <td><span className="small text-danger">{r.alert || ''}</span></td>
                        <td className="small text-muted">{r.examiner}</td>
                        <td className="small text-muted">{r.date}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Patient Alerts Tab ─────────────────────────────── */}
      {tab === 'patients' && bd && (
        <div>
          {bd.patient_cards?.filter(p => p.n_alerts > 0 || p.safety_flag).map(p => (
            <div key={p.patient_id} className={`card mb-3 shadow-sm ${p.safety_flag ? 'border-danger' : 'border-warning'}`}>
              <div className={`card-header d-flex justify-content-between align-items-center ${p.safety_flag ? 'bg-danger text-white' : 'bg-warning text-dark'}`}>
                <strong>{p.patient_id}</strong>
                <div>
                  {p.safety_flag && <span className="badge bg-white text-danger me-2">SAFETY ESCALATION</span>}
                  <span className="badge bg-dark">{p.n_assessments} assessments</span>
                </div>
              </div>
              <div className="card-body">
                <div className="row mb-2">
                  <div className="col-4 text-center"><div className="h5 text-warning">{p.n_alerts}</div><div className="small text-muted">Alerts</div></div>
                  <div className="col-4 text-center"><div className="h5 text-danger">{p.n_severe}</div><div className="small text-muted">Severe/Critical</div></div>
                  <div className="col-4 text-center"><div className="h5 text-info">{p.n_instruments}</div><div className="small text-muted">Instruments</div></div>
                </div>
                <div className="mb-2">
                  {p.instruments.map(inst => <span key={inst} className="badge bg-secondary me-1 mb-1">{inst}</span>)}
                </div>
                {p.alert_messages.map((msg, i) => (
                  <div key={i} className="alert alert-warning py-1 px-2 mb-1 small">{msg}</div>
                ))}
              </div>
            </div>
          ))}
          {bd.patient_cards?.filter(p => p.n_alerts === 0 && !p.safety_flag).length > 0 && (
            <div className="text-muted small mt-2">
              {bd.patient_cards.filter(p => p.n_alerts === 0 && !p.safety_flag).length} patients with no alerts (not shown).
            </div>
          )}
        </div>
      )}

      {/* ── Definitions Tab ───────────────────────────────── */}
      {tab === 'definitions' && defs && (
        <div className="row">
          <div className="col-lg-8 mb-3">
            <div className="card shadow-sm mb-3">
              <div className="card-header fw-bold">Instrument Glossary</div>
              <div className="card-body p-0">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light">
                    <tr><th>Code</th><th>Name</th><th>Domain</th><th>Range</th><th>Description</th></tr>
                  </thead>
                  <tbody>
                    {defs.instruments?.map(inst => (
                      <tr key={inst.code}>
                        <td><code>{inst.code}</code></td>
                        <td><strong>{inst.name}</strong></td>
                        <td><span className="badge bg-secondary">{inst.domain}</span></td>
                        <td className="small">{inst.range}</td>
                        <td className="small text-muted">{inst.description}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="col-lg-4 mb-3">
            <div className="card shadow-sm mb-3">
              <div className="card-header fw-bold">KPI Definitions</div>
              <div className="card-body">
                {defs.kpi_definitions?.map(k => (
                  <div key={k.name} className="mb-2 border-bottom pb-2">
                    <strong>{k.name}</strong>
                    <p className="small text-muted mb-0">{k.description}</p>
                  </div>
                ))}
              </div>
            </div>

            <div className="card shadow-sm border-danger">
              <div className="card-header fw-bold text-danger">Safety Escalation Policy</div>
              <div className="card-body">
                <p className="small">{defs.safety_escalation_policy}</p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
