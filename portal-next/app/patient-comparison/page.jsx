'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const LEVEL_COLOR = {
  severe: 'danger', high: 'danger', moderate: 'warning', elevated: 'warning',
  mild: 'info', low: 'info', minimal: 'success', normal: 'success', good: 'success',
};
function lvlColor(l) { return LEVEL_COLOR[(l || '').toLowerCase()] || 'secondary'; }

function KpiCard({ label, value, color = 'primary', sub }) {
  return (
    <div className="col">
      <div className={`card border-${color} h-100`}>
        <div className="card-body text-center py-2">
          <div className={`fs-4 fw-bold text-${color}`}>{value ?? '—'}</div>
          <div className="small text-muted">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function ScoreBar({ value, max, color = 'primary' }) {
  const pct = max > 0 ? Math.min(100, Math.round((value / max) * 100)) : 0;
  return (
    <div className="d-flex align-items-center gap-2">
      <div className="progress flex-grow-1" style={{ height: 10 }}>
        <div className={`progress-bar bg-${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="small text-muted" style={{ minWidth: 48 }}>{value}/{max}</span>
    </div>
  );
}

function DemoGrid({ demo }) {
  if (!demo) return <div className="text-muted small">No demographics data</div>;
  const fields = [
    ['Age', demo.age],
    ['Sex', demo.sex],
    ['Epilepsy Type', demo.epilepsy_type],
    ['Onset Age', demo.epilepsy_onset_age != null ? `${demo.epilepsy_onset_age} yrs` : '—'],
    ['Years w/ Epilepsy', demo.years_with_epilepsy != null ? `${demo.years_with_epilepsy} yrs` : '—'],
    ['BMI', demo.bmi != null ? demo.bmi.toFixed(1) : '—'],
    ['Education', demo.education_level],
    ['Insurance', demo.insurance_type],
    ['City', demo.address_city ? `${demo.address_city}, ${demo.address_state}` : '—'],
  ];
  return (
    <table className="table table-sm table-borderless mb-0">
      <tbody>
        {fields.map(([k, v]) => (
          <tr key={k}>
            <td className="text-muted small fw-semibold" style={{ width: '45%' }}>{k}</td>
            <td className="small">{v ?? '—'}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function SeizureCard({ seizure, triggers }) {
  if (!seizure) return <div className="text-muted small">No seizure data</div>;
  return (
    <div>
      <table className="table table-sm table-borderless mb-1">
        <tbody>
          <tr><td className="text-muted small fw-semibold" style={{ width: '55%' }}>Total Events</td><td className="small">{seizure.total_events ?? '—'}</td></tr>
          <tr><td className="text-muted small fw-semibold">Avg Duration</td><td className="small">{seizure.avg_duration_sec != null ? `${seizure.avg_duration_sec}s` : '—'}</td></tr>
          <tr><td className="text-muted small fw-semibold">Max Severity</td><td className="small">{seizure.max_severity ?? '—'}</td></tr>
          <tr><td className="text-muted small fw-semibold">ER Visits</td><td className="small">{seizure.er_visits ?? 0}</td></tr>
          <tr><td className="text-muted small fw-semibold">Injuries</td><td className="small">{seizure.injuries ?? 0}</td></tr>
        </tbody>
      </table>
      {triggers && triggers.length > 0 && (
        <div className="mt-1">
          <div className="text-muted small fw-semibold mb-1">Triggers:</div>
          <div className="d-flex flex-wrap gap-1">
            {triggers.map((t, i) => (
              <span key={i} className="badge bg-warning text-dark">{t.trigger} ×{t.count}</span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function AssessmentCard({ assessments }) {
  if (!assessments || !assessments.length) return <div className="text-muted small">No assessments</div>;
  return (
    <div style={{ maxHeight: 220, overflowY: 'auto' }}>
      <table className="table table-sm mb-0">
        <thead className="table-light sticky-top">
          <tr><th style={{ fontSize: '0.75rem' }}>Instrument</th><th style={{ fontSize: '0.75rem' }}>Score</th><th style={{ fontSize: '0.75rem' }}>Interpretation</th></tr>
        </thead>
        <tbody>
          {assessments.map((a, i) => (
            <tr key={i}>
              <td className="small fw-semibold">{a.instrument}</td>
              <td className="small"><ScoreBar value={a.score} max={a.max_score} color={lvlColor(a.level)} /></td>
              <td><span className={`badge bg-${lvlColor(a.level)}`} style={{ fontSize: '0.65rem' }}>{a.interpretation}</span></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function CognitiveCard({ domains }) {
  if (!domains || !domains.length) return <div className="text-muted small">No cognitive data</div>;
  return (
    <table className="table table-sm mb-0">
      <thead className="table-light">
        <tr><th style={{ fontSize: '0.75rem' }}>Domain</th><th style={{ fontSize: '0.75rem' }}>Accuracy %</th><th style={{ fontSize: '0.75rem' }}>Tests</th></tr>
      </thead>
      <tbody>
        {domains.map((d, i) => {
          const acc = d.avg_accuracy != null ? d.avg_accuracy.toFixed(1) : '—';
          const color = d.avg_accuracy >= 75 ? 'success' : d.avg_accuracy >= 50 ? 'warning' : 'danger';
          return (
            <tr key={i}>
              <td className="small">{d.domain}</td>
              <td className="small">
                <div className="d-flex align-items-center gap-1">
                  <div className="progress flex-grow-1" style={{ height: 8 }}>
                    <div className={`progress-bar bg-${color}`} style={{ width: `${Math.min(100, d.avg_accuracy || 0)}%` }} />
                  </div>
                  <span style={{ minWidth: 35 }}>{acc}%</span>
                </div>
              </td>
              <td className="small text-muted">{d.test_count}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function MedCard({ med }) {
  if (!med) return <div className="text-muted small">No medication data</div>;
  const color = med.avg_adherence >= 90 ? 'success' : med.avg_adherence >= 70 ? 'warning' : 'danger';
  return (
    <table className="table table-sm table-borderless mb-0">
      <tbody>
        <tr><td className="text-muted small fw-semibold" style={{ width: '55%' }}>Avg Adherence</td>
          <td><span className={`badge bg-${color}`}>{med.avg_adherence != null ? `${med.avg_adherence.toFixed(1)}%` : '—'}</span></td></tr>
        <tr><td className="text-muted small fw-semibold">Medications</td><td className="small">{med.medications ?? '—'}</td></tr>
        <tr><td className="text-muted small fw-semibold">Total Records</td><td className="small">{med.total_records ?? '—'}</td></tr>
        <tr><td className="text-muted small fw-semibold">Taken / Late / Missed</td>
          <td className="small">{med.doses_taken ?? 0} / {med.doses_late ?? 0} / {med.doses_missed ?? 0}</td></tr>
      </tbody>
    </table>
  );
}

function RadarBars({ radar }) {
  if (!radar || !radar.length) return null;
  return (
    <div className="card mb-3">
      <div className="card-header fw-semibold small">Radar Comparison (normalised 0–100)</div>
      <div className="card-body py-2">
        {radar.map((r, i) => (
          <div key={i} className="mb-2">
            <div className="d-flex justify-content-between small mb-1">
              <span className="fw-semibold">{r.dimension}</span>
              <span className="text-muted">A:{r.patient_a?.toFixed(0) ?? 0} · B:{r.patient_b?.toFixed(0) ?? 0}</span>
            </div>
            <div className="d-flex gap-1">
              <div className="flex-grow-1">
                <div className="progress" style={{ height: 10 }}>
                  <div className="progress-bar bg-primary" style={{ width: `${Math.min(100, r.patient_a || 0)}%` }} />
                </div>
                <div className="text-muted" style={{ fontSize: '0.65rem' }}>Patient A</div>
              </div>
              <div className="flex-grow-1">
                <div className="progress" style={{ height: 10 }}>
                  <div className="progress-bar bg-warning" style={{ width: `${Math.min(100, r.patient_b || 0)}%` }} />
                </div>
                <div className="text-muted" style={{ fontSize: '0.65rem' }}>Patient B</div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function SideBySide({ cmp }) {
  if (!cmp) return null;
  const pa = cmp.patient_a || {};
  const pb = cmp.patient_b || {};
  const da = pa.demographics || {};
  const db = pb.demographics || {};

  const sections = [
    {
      title: '📋 Demographics',
      a: <DemoGrid demo={da} />,
      b: <DemoGrid demo={db} />,
    },
    {
      title: '⚡ Seizure Profile',
      a: <SeizureCard seizure={pa.seizure_summary} triggers={pa.seizure_triggers} />,
      b: <SeizureCard seizure={pb.seizure_summary} triggers={pb.seizure_triggers} />,
    },
    {
      title: '📊 Clinical Assessments',
      a: <AssessmentCard assessments={pa.assessments} />,
      b: <AssessmentCard assessments={pb.assessments} />,
    },
    {
      title: '🧠 Cognitive Domains',
      a: <CognitiveCard domains={pa.cognitive_domains} />,
      b: <CognitiveCard domains={pb.cognitive_domains} />,
    },
    {
      title: '💊 Medication Adherence',
      a: <MedCard med={pa.medication_adherence} />,
      b: <MedCard med={pb.medication_adherence} />,
    },
  ];

  return (
    <div>
      <RadarBars radar={cmp.radar_comparison} />
      {sections.map((s, i) => (
        <div key={i} className="card mb-3">
          <div className="card-header fw-semibold small">{s.title}</div>
          <div className="card-body py-2">
            <div className="row g-3">
              <div className="col-md-6">
                <div className="mb-1">
                  <span className="badge bg-primary me-2">Patient A</span>
                  <span className="fw-semibold small">{da.full_name || pa.patient_id || '—'}</span>
                </div>
                {s.a}
              </div>
              <div className="col-md-6 border-start">
                <div className="mb-1">
                  <span className="badge bg-warning text-dark me-2">Patient B</span>
                  <span className="fw-semibold small">{db.full_name || pb.patient_id || '—'}</span>
                </div>
                {s.b}
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

export default function PatientComparisonDashboard() {
  const [ov, setOv]       = useState(null);
  const [defs, setDefs]   = useState(null);
  const [cmp, setCmp]     = useState(null);
  const [tab, setTab]     = useState('overview');
  const [err, setErr]     = useState(null);
  const [patA, setPatA]   = useState('EPAT001');
  const [patB, setPatB]   = useState('EPAT002');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/patient-comparison/overview`).then(r => r.json()),
      fetch(`${API}/api/patient-comparison/definitions`).then(r => r.json()),
    ])
      .then(([o, d]) => { setOv(o); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  function runCompare() {
    if (!patA || !patB || patA === patB) return;
    setLoading(true);
    setCmp(null);
    fetch(`${API}/api/patient-comparison/compare?a=${encodeURIComponent(patA)}&b=${encodeURIComponent(patB)}`)
      .then(r => r.json())
      .then(d => { setCmp(d); setTab('compare'); })
      .catch(e => setErr(String(e)))
      .finally(() => setLoading(false));
  }

  if (err) return <div className="alert alert-danger m-3">Error: {err}</div>;
  if (!ov) return <div className="text-muted p-4">Loading Patient Comparison…</div>;

  const patients = ov.patients || [];
  const kpis = ov.kpis || {};
  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'compare', label: 'Compare' },
    { id: 'definitions', label: 'Definitions' },
  ];

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3 gap-2">
        <span style={{ fontSize: '1.5rem' }}>🔀</span>
        <div>
          <h4 className="mb-0 fw-bold">Patient Side-by-Side Comparison</h4>
          <div className="text-muted small">Select two patients to compare demographics, assessments, cognition, seizures, and medication adherence</div>
        </div>
      </div>

      {/* Patient selector */}
      <div className="card mb-3">
        <div className="card-body py-2">
          <div className="row g-2 align-items-end">
            <div className="col-auto">
              <label className="form-label small mb-1">Patient A</label>
              <select className="form-select form-select-sm" value={patA} onChange={e => setPatA(e.target.value)} style={{ minWidth: 200 }}>
                {patients.map(p => (
                  <option key={p.patient_id} value={p.patient_id}>
                    {p.patient_id} — {p.full_name} ({p.sex}, {p.age}yr)
                  </option>
                ))}
              </select>
            </div>
            <div className="col-auto text-muted fw-bold">vs</div>
            <div className="col-auto">
              <label className="form-label small mb-1">Patient B</label>
              <select className="form-select form-select-sm" value={patB} onChange={e => setPatB(e.target.value)} style={{ minWidth: 200 }}>
                {patients.map(p => (
                  <option key={p.patient_id} value={p.patient_id}>
                    {p.patient_id} — {p.full_name} ({p.sex}, {p.age}yr)
                  </option>
                ))}
              </select>
            </div>
            <div className="col-auto">
              <button
                className="btn btn-primary btn-sm"
                onClick={runCompare}
                disabled={loading || patA === patB}
              >
                {loading ? '⏳ Comparing…' : '🔍 Compare'}
              </button>
            </div>
            {patA === patB && <div className="col-auto text-warning small">Select two different patients</div>}
          </div>
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {tabs.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link${tab === t.id ? ' active' : ''}`} onClick={() => setTab(t.id)}>{t.label}</button>
          </li>
        ))}
      </ul>

      {/* Overview */}
      {tab === 'overview' && (
        <div>
          <div className="row row-cols-2 row-cols-md-3 g-3 mb-3">
            <KpiCard label="Total Patients" value={kpis.total_patients} color="primary" />
            <KpiCard label="Seizure Events" value={kpis.total_seizure_events} color="warning" />
            <KpiCard label="Assessments" value={kpis.total_assessments} color="info" />
            <KpiCard label="Cognitive Tests" value={kpis.total_cognitive_tests} color="success" />
            <KpiCard label="Medication Records" value={kpis.total_med_records?.toLocaleString()} color="secondary" />
            <KpiCard label="EEG Analyses" value={kpis.total_analyses} color="primary" />
          </div>
          <div className="card">
            <div className="card-header fw-semibold small">Patient Roster ({patients.length} patients)</div>
            <div className="table-responsive">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-dark">
                  <tr>
                    <th>ID</th><th>Name</th><th>Age</th><th>Sex</th><th>Epilepsy Type</th>
                  </tr>
                </thead>
                <tbody>
                  {patients.map(p => (
                    <tr
                      key={p.patient_id}
                      style={{ cursor: 'pointer' }}
                      onClick={() => {
                        if (!patA || patA === p.patient_id) setPatB(p.patient_id);
                        else setPatA(p.patient_id);
                      }}
                    >
                      <td className="small fw-semibold">{p.patient_id}</td>
                      <td className="small">{p.full_name}</td>
                      <td className="small">{p.age}</td>
                      <td className="small">{p.sex}</td>
                      <td><span className="badge bg-primary" style={{ fontSize: '0.65rem' }}>{p.epilepsy_type}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Compare */}
      {tab === 'compare' && (
        <div>
          {!cmp && !loading && (
            <div className="alert alert-info">
              Select two patients above and click <strong>Compare</strong> to see side-by-side analysis.
            </div>
          )}
          {loading && <div className="text-muted p-3">⏳ Loading comparison…</div>}
          {cmp && <SideBySide cmp={cmp} />}
        </div>
      )}

      {/* Definitions */}
      {tab === 'definitions' && defs && (
        <div>
          <div className="card mb-3">
            <div className="card-header fw-semibold small">Comparison Dimensions</div>
            <div className="table-responsive">
              <table className="table table-sm mb-0">
                <thead className="table-light">
                  <tr><th>Dimension</th><th>Description</th></tr>
                </thead>
                <tbody>
                  {(defs.comparison_dimensions || []).map((d, i) => (
                    <tr key={i}>
                      <td className="fw-semibold small">{d.dimension}</td>
                      <td className="small text-muted">{d.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          <div className="card">
            <div className="card-header fw-semibold small">Glossary</div>
            <div className="table-responsive">
              <table className="table table-sm mb-0">
                <thead className="table-light">
                  <tr><th>Term</th><th>Definition</th></tr>
                </thead>
                <tbody>
                  {(defs.glossary || []).map((g, i) => (
                    <tr key={i}>
                      <td className="fw-semibold small">{g.term}</td>
                      <td className="small text-muted">{g.definition}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          {defs.clinical_notes && (
            <div className="alert alert-info mt-3 small">{defs.clinical_notes}</div>
          )}
        </div>
      )}
    </div>
  );
}
