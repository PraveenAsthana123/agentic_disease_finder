'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',    label: 'Overview' },
  { id: 'by-drug',     label: 'By Drug' },
  { id: 'per-patient', label: 'Per Patient' },
  { id: 'refills',     label: 'Refills' },
  { id: 'definitions', label: 'Definitions' },
];

function adherenceColor(pct) {
  if (pct >= 80) return 'success';
  if (pct >= 50) return 'warning';
  return 'danger';
}

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-3 mb-3">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center">
          <div className={`h4 mb-1 fw-bold text-${color || 'primary'}`}>{value ?? '—'}</div>
          <div className="text-muted small">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function StackedBar({ taken, late, missed, label }) {
  const total = taken + late + missed;
  if (!total) return null;
  const tPct = (taken / total) * 100;
  const lPct = (late / total) * 100;
  const mPct = (missed / total) * 100;
  return (
    <div className="mb-2">
      {label && <div className="small fw-semibold mb-1">{label}</div>}
      <div className="d-flex justify-content-between small text-muted mb-1">
        <span>Taken {taken.toFixed(1)}%</span>
        <span>Late {late.toFixed(1)}%</span>
        <span>Missed {missed.toFixed(1)}%</span>
      </div>
      <div className="progress" style={{ height: 14 }}>
        <div className="progress-bar bg-success" style={{ width: `${tPct}%` }} title={`Taken ${taken.toFixed(1)}%`} />
        <div className="progress-bar bg-warning" style={{ width: `${lPct}%` }} title={`Late ${late.toFixed(1)}%`} />
        <div className="progress-bar bg-danger"  style={{ width: `${mPct}%` }} title={`Missed ${missed.toFixed(1)}%`} />
      </div>
    </div>
  );
}

function OverviewPanel({ ov }) {
  if (!ov) return <div className="text-muted">Loading…</div>;
  if (ov.error) return <div className="alert alert-warning">{ov.error}</div>;

  const adh = ov.adherence_rate ?? 0;
  const drugs = ov.adherence_by_drug || [];

  return (
    <div>
      <div className="row mb-3">
        <KPI label="Total Dose Logs" value={(ov.total_logs ?? 0).toLocaleString()} color="primary" sub="12 months" />
        <KPI label="Patients Tracked" value={ov.total_patients} color="info" sub="on ASMs" />
        <KPI label="Unique ASMs" value={ov.unique_drugs} color="secondary" sub="antiseizure meds" />
        <KPI label="Avg Lateness" value={`${(ov.avg_minutes_late ?? 0).toFixed(0)} min`} color="warning" sub="when late doses" />
      </div>
      <div className="row mb-3">
        <KPI
          label="Overall Adherence"
          value={`${adh.toFixed(1)}%`}
          color={adherenceColor(adh)}
          sub="target ≥80%"
        />
        <KPI
          label="Late Rate"
          value={`${(ov.late_rate ?? 0).toFixed(1)}%`}
          color="warning"
          sub="taken outside window"
        />
        <KPI
          label="Missed Rate"
          value={`${(ov.missed_rate ?? 0).toFixed(1)}%`}
          color={ov.missed_rate > 10 ? 'danger' : 'secondary'}
          sub="omitted doses"
        />
        <KPI
          label="Avg Side Effect Severity"
          value={`${(ov.avg_side_effect_severity ?? 0).toFixed(2)} / 10`}
          color="secondary"
          sub="patient-reported"
        />
      </div>

      <div className="card mb-3">
        <div className="card-header fw-semibold small">
          Overall Dose Status Breakdown
        </div>
        <div className="card-body">
          <StackedBar
            taken={ov.adherence_rate ?? 0}
            late={ov.late_rate ?? 0}
            missed={ov.missed_rate ?? 0}
          />
          <div className="d-flex gap-3 mt-2 small">
            <span><span className="badge bg-success me-1">&nbsp;</span>Taken on time</span>
            <span><span className="badge bg-warning me-1">&nbsp;</span>Late</span>
            <span><span className="badge bg-danger me-1">&nbsp;</span>Missed</span>
          </div>
        </div>
      </div>

      {drugs.length > 0 && (
        <div className="card">
          <div className="card-header fw-semibold small">Adherence by Drug (all patients)</div>
          <div className="card-body">
            {drugs.map((d, i) => (
              <StackedBar
                key={i}
                label={d.drug}
                taken={d.taken_pct}
                late={d.late_pct}
                missed={d.missed_pct}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function ByDrugPanel({ ov }) {
  if (!ov) return <div className="text-muted">Loading…</div>;
  if (ov.error) return <div className="alert alert-warning">{ov.error}</div>;

  const drugs = ov.adherence_by_drug || [];
  const sorted = [...drugs].sort((a, b) => b.taken_pct - a.taken_pct);

  return (
    <div>
      <div className="card mb-3">
        <div className="card-header fw-semibold small">Drug Adherence Comparison ({drugs.length} ASMs)</div>
        <div className="card-body p-0">
          <div className="table-responsive">
            <table className="table table-sm table-hover mb-0">
              <thead className="table-light">
                <tr>
                  <th>Drug (ASM)</th>
                  <th>Taken On Time %</th>
                  <th>Late %</th>
                  <th>Missed %</th>
                  <th>Adherence Grade</th>
                  <th>Adherence Bar</th>
                </tr>
              </thead>
              <tbody>
                {sorted.map((d, i) => {
                  const grade = d.taken_pct >= 85 ? 'A' : d.taken_pct >= 80 ? 'B' : d.taken_pct >= 70 ? 'C' : 'D';
                  const gradeColor = { A: 'success', B: 'primary', C: 'warning', D: 'danger' }[grade];
                  return (
                    <tr key={i}>
                      <td className="fw-semibold">{d.drug}</td>
                      <td>
                        <span className={`badge bg-${adherenceColor(d.taken_pct)}`}>
                          {d.taken_pct.toFixed(1)}%
                        </span>
                      </td>
                      <td className="text-warning fw-semibold">{d.late_pct.toFixed(1)}%</td>
                      <td className="text-danger fw-semibold">{d.missed_pct.toFixed(1)}%</td>
                      <td><span className={`badge bg-${gradeColor}`}>{grade}</span></td>
                      <td style={{ minWidth: 120 }}>
                        <div className="progress" style={{ height: 12 }}>
                          <div className="progress-bar bg-success" style={{ width: `${d.taken_pct}%` }} />
                          <div className="progress-bar bg-warning" style={{ width: `${d.late_pct}%` }} />
                          <div className="progress-bar bg-danger" style={{ width: `${d.missed_pct}%` }} />
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="alert alert-info small mb-0">
        <strong>Adherence threshold:</strong> ≥80% taken on time = adequate (PDC threshold; ILAE guideline). &nbsp;
        Grade A ≥85% · B 80–85% · C 70–80% · D &lt;70%
      </div>
    </div>
  );
}

function PerPatientPanel({ bd }) {
  if (!bd) return <div className="text-muted">Loading…</div>;
  if (bd.error) return <div className="alert alert-warning">{bd.error}</div>;

  const [sortCol, setSortCol] = useState('adherence_rate');
  const [sortDir, setSortDir] = useState(-1);
  const [filter, setFilter] = useState('all');

  const patients = bd.per_patient || [];
  let filtered = patients;
  if (filter === 'adequate')   filtered = patients.filter(p => p.adherence_rate >= 80);
  if (filter === 'suboptimal') filtered = patients.filter(p => p.adherence_rate >= 50 && p.adherence_rate < 80);
  if (filter === 'critical')   filtered = patients.filter(p => p.adherence_rate < 50);

  const sorted = [...filtered].sort((a, b) => sortDir * (a[sortCol] - b[sortCol]));

  function toggleSort(col) {
    if (sortCol === col) setSortDir(d => -d);
    else { setSortCol(col); setSortDir(-1); }
  }
  function th(col, label) {
    const active = sortCol === col;
    return (
      <th
        onClick={() => toggleSort(col)}
        style={{ cursor: 'pointer', userSelect: 'none' }}
        className={active ? 'text-primary' : ''}
      >
        {label} {active ? (sortDir === -1 ? '▼' : '▲') : '↕'}
      </th>
    );
  }

  return (
    <div>
      <div className="d-flex gap-2 mb-3">
        {[
          ['all', 'All Patients'],
          ['adequate', '≥80% (Adequate)'],
          ['suboptimal', '50–80% (Suboptimal)'],
          ['critical', '<50% (Critical)'],
        ].map(([val, lbl]) => (
          <button
            key={val}
            className={`btn btn-sm ${filter === val ? 'btn-primary' : 'btn-outline-secondary'}`}
            onClick={() => setFilter(val)}
          >
            {lbl} ({val === 'all' ? patients.length : patients.filter(p =>
              val === 'adequate' ? p.adherence_rate >= 80 :
              val === 'suboptimal' ? (p.adherence_rate >= 50 && p.adherence_rate < 80) :
              p.adherence_rate < 50
            ).length})
          </button>
        ))}
      </div>

      <div className="card">
        <div className="card-header fw-semibold small">
          Per-Patient Adherence — {sorted.length} patients shown
        </div>
        <div className="card-body p-0">
          <div className="table-responsive" style={{ maxHeight: 480 }}>
            <table className="table table-sm table-hover mb-0">
              <thead className="table-light">
                <tr>
                  <th>Patient ID</th>
                  <th>Drugs</th>
                  {th('total_doses', 'Doses')}
                  {th('adherence_rate', 'Adherence %')}
                  {th('late_rate', 'Late %')}
                  {th('missed_rate', 'Missed %')}
                  {th('avg_minutes_late', 'Avg Late (min)')}
                  <th>Top Side Effects</th>
                </tr>
              </thead>
              <tbody>
                {sorted.map((p, i) => (
                  <tr key={i}>
                    <td><span className="badge bg-secondary">{p.patient_id}</span></td>
                    <td className="small">{(p.drugs || []).join(', ')}</td>
                    <td>{p.total_doses}</td>
                    <td>
                      <span className={`badge bg-${adherenceColor(p.adherence_rate)}`}>
                        {p.adherence_rate.toFixed(1)}%
                      </span>
                    </td>
                    <td className="text-warning fw-semibold small">{p.late_rate.toFixed(1)}%</td>
                    <td>
                      {p.missed_rate > 10
                        ? <span className="text-danger fw-semibold">{p.missed_rate.toFixed(1)}%</span>
                        : <span className="text-muted small">{p.missed_rate.toFixed(1)}%</span>}
                    </td>
                    <td className="small">{p.avg_minutes_late.toFixed(0)} min</td>
                    <td className="small text-muted">{(p.top_side_effects || []).slice(0, 3).join(', ')}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

function RefillsPanel({ bd }) {
  if (!bd) return <div className="text-muted">Loading…</div>;
  if (bd.error) return <div className="alert alert-warning">{bd.error}</div>;

  const refills = bd.recent_refills || [];
  if (!refills.length) return <div className="alert alert-secondary">No refill records available.</div>;

  return (
    <div className="card">
      <div className="card-header fw-semibold small">Recent Medication Refills ({refills.length} records)</div>
      <div className="card-body p-0">
        <div className="table-responsive" style={{ maxHeight: 500 }}>
          <table className="table table-sm table-hover mb-0">
            <thead className="table-light">
              <tr>
                <th>Patient</th>
                <th>Drug</th>
                <th>Refill Date</th>
                <th>Quantity</th>
                <th>Days Supply</th>
                <th>Pharmacy</th>
                <th>Auto-Refill</th>
              </tr>
            </thead>
            <tbody>
              {refills.map((r, i) => (
                <tr key={i}>
                  <td><span className="badge bg-secondary">{r.patient_id}</span></td>
                  <td className="fw-semibold small">{r.drug_name}</td>
                  <td className="small">{r.refill_date}</td>
                  <td>{r.quantity}</td>
                  <td>{r.days_supply}d</td>
                  <td className="small">{r.pharmacy}</td>
                  <td>
                    <span className={`badge bg-${r.auto_refill ? 'success' : 'secondary'}`}>
                      {r.auto_refill ? 'Yes' : 'No'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function DefinitionsPanel({ defs }) {
  if (!defs) return <div className="text-muted">Loading…</div>;
  if (defs.error) return <div className="alert alert-warning">{defs.error}</div>;

  const concepts = defs.concepts || [];

  return (
    <div className="card">
      <div className="card-header fw-semibold small">Glossary — Medication Adherence Concepts</div>
      <div className="card-body p-0">
        <table className="table table-sm mb-0">
          <thead className="table-light">
            <tr><th style={{ width: '22%' }}>Term</th><th>Definition</th></tr>
          </thead>
          <tbody>
            {concepts.map((c, i) => (
              <tr key={i}>
                <td className="fw-semibold small align-top">{c.name}</td>
                <td className="small">{c.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function MedicationAdherencePage() {
  const [tab, setTab] = useState('overview');
  const [ov, setOv]   = useState(null);
  const [bd, setBd]   = useState(null);
  const [defs, setDefs] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/medication-adherence/overview`)
      .then(r => r.json()).then(setOv)
      .catch(() => setOv({ error: 'Failed to load overview' }));
    fetch(`${API}/api/medication-adherence/breakdown`)
      .then(r => r.json()).then(setBd)
      .catch(() => setBd({ error: 'Failed to load breakdown' }));
    fetch(`${API}/api/medication-adherence/definitions`)
      .then(r => r.json()).then(setDefs)
      .catch(() => setDefs({ error: 'Failed to load definitions' }));
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center gap-2 mb-3">
        <span style={{ fontSize: '1.6rem' }}>&#x1f48a;</span>
        <div>
          <h4 className="mb-0 fw-bold">Medication Adherence Dashboard</h4>
          <div className="text-muted small">
            ASM dose tracking · adherence by drug · per-patient breakdown · refill records
            — 12,600 dose logs, 30 patients, 8 antiseizure medications
          </div>
        </div>
      </div>

      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
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

      {tab === 'overview'    && <OverviewPanel ov={ov} />}
      {tab === 'by-drug'     && <ByDrugPanel ov={ov} />}
      {tab === 'per-patient' && <PerPatientPanel bd={bd} />}
      {tab === 'refills'     && <RefillsPanel bd={bd} />}
      {tab === 'definitions' && <DefinitionsPanel defs={defs} />}
    </div>
  );
}
