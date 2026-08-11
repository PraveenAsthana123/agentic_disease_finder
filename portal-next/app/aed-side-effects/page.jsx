'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const SEV_COLOR = (s) => {
  if (s === 0) return 'secondary';
  if (s <= 2) return 'success';
  if (s <= 4) return 'warning';
  if (s <= 6) return 'orange';
  return 'danger';
};
const SEV_BADGE = (s) => {
  if (s === 0 || s === '0') return 'bg-secondary';
  if (s <= 2) return 'bg-success';
  if (s <= 4) return 'bg-warning text-dark';
  if (s <= 6) return 'bg-orange text-white';
  return 'bg-danger';
};

export default function AedSideEffectsPage() {
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [expandedDrug, setExpandedDrug] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/aed-side-effects/overview`).then(r => r.json()).then(setOverview).catch(() => {});
    fetch(`${API}/api/aed-side-effects/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    fetch(`${API}/api/aed-side-effects/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!overview) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const k = overview.kpis || {};
  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'by-drug', label: 'By Drug' },
    { id: 'by-effect', label: 'By Side Effect' },
    { id: 'per-patient', label: 'Per Patient' },
    { id: 'definitions', label: 'Definitions' },
  ];

  const maxEffect = overview.top_side_effects?.length
    ? Math.max(...overview.top_side_effects.map(e => e.count))
    : 1;
  const maxDrugSev = overview.drug_summary?.length
    ? Math.max(...overview.drug_summary.map(d => d.avg_severity))
    : 1;

  return (
    <div className="container-fluid py-3">
      <h4 className="fw-bold mb-1">AED Side Effects Dashboard</h4>
      <p className="text-muted small mb-3">
        Anti-epileptic drug adverse effect profiling — {k.total_records?.toLocaleString()} adherence records ·{' '}
        {k.total_patients} patients · {k.total_drugs} AEDs · {k.records_with_effects?.toLocaleString()} records with effects
      </p>

      {/* KPI cards */}
      <div className="row g-2 mb-3">
        {[
          { label: 'Records with Effects', value: k.records_with_effects?.toLocaleString(), sub: `${k.effect_rate_pct}% of all records`, color: 'warning' },
          { label: 'Avg Severity (when present)', value: k.avg_severity_when_present, sub: 'scale 0–8', color: 'info' },
          { label: 'Severe Events (≥6)', value: k.severe_records?.toLocaleString(), sub: 'requiring urgent review', color: 'danger' },
          { label: 'Patients with Severe Effects', value: k.patients_with_severe_effects, sub: `of ${k.total_patients} total`, color: 'dark' },
        ].map((card) => (
          <div key={card.label} className="col-6 col-md-3">
            <div className={`card border-${card.color} h-100`}>
              <div className="card-body p-2 text-center">
                <div className={`fs-4 fw-bold text-${card.color}`}>{card.value ?? '—'}</div>
                <div className="small fw-semibold">{card.label}</div>
                <div className="text-muted" style={{ fontSize: '0.7rem' }}>{card.sub}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {tabs.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link${tab === t.id ? ' active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* OVERVIEW */}
      {tab === 'overview' && (
        <div className="row g-3">
          {/* Top side effects bar chart */}
          <div className="col-md-6">
            <div className="card h-100">
              <div className="card-header fw-semibold">Top Side Effects (all AEDs)</div>
              <div className="card-body">
                {(overview.top_side_effects || []).map((e) => (
                  <div key={e.effect} className="mb-2">
                    <div className="d-flex justify-content-between small">
                      <span className="fw-semibold">{e.effect}</span>
                      <span className="text-muted">{e.count.toLocaleString()} records</span>
                    </div>
                    <div className="progress" style={{ height: 10 }}>
                      <div
                        className="progress-bar bg-warning"
                        style={{ width: `${(e.count / maxEffect) * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Drug severity ranking */}
          <div className="col-md-6">
            <div className="card h-100">
              <div className="card-header fw-semibold">AED Burden by Average Severity</div>
              <div className="card-body">
                {(overview.drug_summary || []).map((d) => (
                  <div key={d.drug} className="mb-2">
                    <div className="d-flex justify-content-between small">
                      <span className="fw-semibold">{d.drug}</span>
                      <span>
                        <span className={`badge bg-${SEV_COLOR(Math.round(d.avg_severity))} me-1`}>
                          Avg {d.avg_severity}
                        </span>
                        <span className="text-muted">{d.effect_rate_pct}% with effects</span>
                      </span>
                    </div>
                    <div className="progress" style={{ height: 10 }}>
                      <div
                        className="progress-bar bg-danger"
                        style={{ width: `${(d.avg_severity / Math.max(maxDrugSev, 1)) * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Severity distribution */}
          <div className="col-12">
            <div className="card">
              <div className="card-header fw-semibold">Severity Distribution (all records)</div>
              <div className="card-body">
                <div className="row g-1">
                  {(overview.severity_distribution || []).map((s) => (
                    <div key={s.severity} className="col text-center">
                      <div className="small fw-bold">{s.label}</div>
                      <div className="fs-5 fw-bold">{s.count.toLocaleString()}</div>
                      <div className="text-muted" style={{ fontSize: '0.65rem' }}>Severity {s.severity}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* BY DRUG */}
      {tab === 'by-drug' && breakdown && (
        <div className="row g-3">
          <div className="col-12">
            <div className="card">
              <div className="card-header fw-semibold">Per-Drug Side Effect Profiles</div>
              <div className="card-body p-0">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light">
                    <tr>
                      <th>Drug</th>
                      <th>Class</th>
                      <th>Effect Events</th>
                      <th>Top Side Effects</th>
                      <th>None</th>
                      <th>Mild</th>
                      <th>Moderate</th>
                      <th>Severe</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(breakdown.drug_effect_profiles || []).map((d) => {
                      const sbd = (breakdown.severity_by_drug || []).find(s => s.drug === d.drug) || {};
                      return (
                        <tr key={d.drug}>
                          <td className="fw-semibold">{d.drug}</td>
                          <td><small className="text-muted">{d.class}</small></td>
                          <td>{d.total_effect_events.toLocaleString()}</td>
                          <td>
                            {d.top_effects.slice(0, 3).map(e => (
                              <span key={e.effect} className="badge bg-warning text-dark me-1" style={{ fontSize: '0.65rem' }}>
                                {e.effect}
                              </span>
                            ))}
                          </td>
                          <td className="text-secondary">{sbd.none?.toLocaleString()}</td>
                          <td className="text-success">{sbd.mild?.toLocaleString()}</td>
                          <td className="text-warning">{sbd.moderate?.toLocaleString()}</td>
                          <td className="text-danger fw-semibold">{sbd.severe?.toLocaleString()}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Adherence vs severity */}
          <div className="col-md-6">
            <div className="card">
              <div className="card-header fw-semibold">Adherence vs Side Effect Severity</div>
              <div className="card-body">
                {(breakdown.adherence_vs_severity || []).map((a) => (
                  <div key={String(a.taken)} className="d-flex justify-content-between align-items-center mb-2 p-2 rounded bg-light">
                    <span className="fw-semibold">{a.taken ? '✅ Dose Taken' : '❌ Dose Missed'}</span>
                    <span>
                      <span className="badge bg-info me-1">{a.count.toLocaleString()} records</span>
                      <span className="badge bg-secondary">Avg Sev {a.avg_severity}</span>
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* BY SIDE EFFECT */}
      {tab === 'by-effect' && (
        <div className="row g-3">
          <div className="col-12">
            <div className="card">
              <div className="card-header fw-semibold">Side Effect Frequency Across All AEDs</div>
              <div className="card-body">
                {(overview.top_side_effects || []).map((e, i) => (
                  <div key={e.effect} className="d-flex align-items-center mb-2 gap-2">
                    <div style={{ width: 20 }} className="text-muted small text-end">{i + 1}</div>
                    <div style={{ width: 160 }} className="fw-semibold small">{e.effect}</div>
                    <div className="flex-grow-1">
                      <div className="progress" style={{ height: 18 }}>
                        <div
                          className="progress-bar bg-warning"
                          style={{ width: `${(e.count / maxEffect) * 100}%` }}
                        >
                          <span style={{ fontSize: '0.65rem' }}>{e.count.toLocaleString()}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {defs && (
            <div className="col-12">
              <div className="card">
                <div className="card-header fw-semibold">Side Effects — Clinical Notes</div>
                <div className="card-body p-0">
                  <table className="table table-sm table-hover mb-0">
                    <thead className="table-light">
                      <tr><th>Side Effect</th><th>Associated AEDs</th><th>Clinical Note</th></tr>
                    </thead>
                    <tbody>
                      {(defs.side_effects_glossary || []).map((g) => (
                        <tr key={g.effect}>
                          <td className="fw-semibold">{g.effect}</td>
                          <td><small className="text-muted">{g.drugs}</small></td>
                          <td><small>{g.clinical_note}</small></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* PER PATIENT */}
      {tab === 'per-patient' && breakdown && (
        <div className="card">
          <div className="card-header fw-semibold">Per-Patient Side Effect Burden</div>
          <div className="card-body p-0">
            <table className="table table-sm table-hover mb-0">
              <thead className="table-light">
                <tr>
                  <th>Patient</th>
                  <th>Records</th>
                  <th>Avg Severity</th>
                  <th>Max Severity</th>
                  <th>Drugs Tried</th>
                  <th>Top Side Effect</th>
                  <th>Tier</th>
                </tr>
              </thead>
              <tbody>
                {(breakdown.per_patient || []).map((p) => (
                  <tr key={p.patient_id}>
                    <td className="fw-semibold">{p.patient_id}</td>
                    <td>{p.records}</td>
                    <td>{p.avg_severity}</td>
                    <td>{p.max_severity}</td>
                    <td>{p.drugs_tried}</td>
                    <td><small>{p.top_side_effect}</small></td>
                    <td>
                      <span className={`badge ${SEV_BADGE(Math.round(p.avg_severity))}`}>
                        {p.severity_label}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* DEFINITIONS */}
      {tab === 'definitions' && defs && (
        <div className="row g-3">
          <div className="col-12">
            <div className="card">
              <div className="card-header fw-semibold">Dashboard Purpose</div>
              <div className="card-body"><p className="mb-0">{defs.dashboard_purpose}</p></div>
            </div>
          </div>

          <div className="col-md-6">
            <div className="card h-100">
              <div className="card-header fw-semibold">Severity Scale (0–8)</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-light"><tr><th>Score</th><th>Label</th><th>Clinical Action</th></tr></thead>
                  <tbody>
                    {(defs.severity_scale || []).map((s) => (
                      <tr key={s.score}>
                        <td><span className={`badge ${SEV_BADGE(typeof s.score === 'number' ? s.score : 5)}`}>{s.score}</span></td>
                        <td className="fw-semibold">{s.label}</td>
                        <td><small className="text-muted">{s.description}</small></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="col-md-6">
            <div className="card h-100">
              <div className="card-header fw-semibold">AED Classes</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-light"><tr><th>Class</th><th>Drugs</th><th>Key Effects</th></tr></thead>
                  <tbody>
                    {(defs.aed_classes || []).map((c) => (
                      <tr key={c.class}>
                        <td className="fw-semibold small">{c.class}</td>
                        <td><small className="text-muted">{c.drugs}</small></td>
                        <td><small>{c.key_effects}</small></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="col-12">
            <div className="card">
              <div className="card-header fw-semibold">Data Source</div>
              <div className="card-body">
                <div className="row g-2">
                  {Object.entries(defs.data_source || {}).map(([k, v]) => (
                    <div key={k} className="col-md-4">
                      <div className="p-2 bg-light rounded">
                        <div className="small text-muted text-capitalize">{k.replace(/_/g, ' ')}</div>
                        <div className="fw-bold">{typeof v === 'number' ? v.toLocaleString() : v}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          <div className="col-12">
            <div className="card">
              <div className="card-header fw-semibold">References</div>
              <div className="card-body">
                <ul className="mb-0">
                  {(defs.clinical_references || []).map((r, i) => (
                    <li key={i} className="small text-muted">{r}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
