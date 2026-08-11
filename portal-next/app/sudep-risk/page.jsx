'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TIER_COLOR = {
  Low: 'success',
  Moderate: 'warning',
  High: 'danger',
  'Very High': 'dark',
};

const TIER_BG = {
  Low: '#22c55e',
  Moderate: '#f59e0b',
  High: '#f97316',
  'Very High': '#ef4444',
};

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-4 col-lg-2 mb-2">
      <div className={`card border-${color || 'primary'} text-center h-100`}>
        <div className="card-body py-2 px-1">
          <div className={`h4 fw-bold mb-0 text-${color || 'primary'}`}>{value ?? '—'}</div>
          <div className="small text-muted">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.68rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function Bar({ items, maxVal, colorKey }) {
  const mx = maxVal || Math.max(...(items || []).map(i => i.count || i.n || 0));
  return (
    <div>
      {(items || []).map((it, i) => {
        const val = it.count ?? it.n ?? 0;
        const label = it.label || it.tier || it.factor || it.freq || it.bucket || it.gender || it.syndrome || '?';
        const pct = mx ? Math.round((val / mx) * 100) : 0;
        const color = it.color ? '' : (colorKey || 'primary');
        return (
          <div key={i} className="d-flex align-items-center mb-1 gap-2">
            <div className="text-end small text-muted" style={{ width: 180, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontSize: '0.75rem' }}>
              {label}
            </div>
            <div className="flex-grow-1">
              <div className="progress" style={{ height: 16 }}>
                <div
                  className={it.color ? '' : `progress-bar bg-${color}`}
                  style={{ width: `${pct}%`, backgroundColor: it.color || undefined }}
                >
                  <span className="small px-1">{val}</span>
                </div>
              </div>
            </div>
            {it.pct !== undefined && (
              <span className="small text-muted" style={{ width: 42 }}>{it.pct}%</span>
            )}
          </div>
        );
      })}
    </div>
  );
}

function RiskBadge({ tier }) {
  return <span className={`badge bg-${TIER_COLOR[tier] || 'secondary'}`}>{tier}</span>;
}

export default function SudepRiskPage() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [search, setSearch] = useState('');
  const [sortBy, setSortBy] = useState('risk_score');
  const [sortDir, setSortDir] = useState(-1);
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/sudep-risk/overview`).then(r => r.json()),
      fetch(`${API}/api/sudep-risk/breakdown`).then(r => r.json()),
      fetch(`${API}/api/sudep-risk/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load SUDEP Risk data: {err}</div>;
  if (!ov) return <div className="text-muted p-4">Loading SUDEP Risk Assessment Dashboard…</div>;

  const TABS = [
    { id: 'overview', label: '📊 Overview' },
    { id: 'risk-factors', label: '⚡ Risk Factors' },
    { id: 'patients', label: '🧑 Per Patient' },
    { id: 'high-risk', label: '🚨 High-Risk Alerts' },
    { id: 'definitions', label: '📖 Definitions' },
  ];

  const kpis = ov.kpis || {};

  // Sort & filter patients
  const allPatients = bd?.patients || [];
  const filtered = allPatients
    .filter(p =>
      !search ||
      p.patient_id?.toLowerCase().includes(search.toLowerCase()) ||
      p.syndrome?.toLowerCase().includes(search.toLowerCase()) ||
      p.risk_tier?.toLowerCase().includes(search.toLowerCase())
    )
    .sort((a, b) => {
      const va = a[sortBy] ?? 0;
      const vb = b[sortBy] ?? 0;
      return typeof va === 'string'
        ? sortDir * va.localeCompare(vb)
        : sortDir * (va - vb);
    });

  const toggleSort = (key) => {
    if (sortBy === key) setSortDir(d => d * -1);
    else { setSortBy(key); setSortDir(-1); }
  };

  const sortArrow = (key) => sortBy === key ? (sortDir > 0 ? ' ▲' : ' ▼') : '';

  const highRisk = bd?.high_risk_summary || {};
  const durationHist = bd?.disease_duration_hist || [];

  // GTCS × Resistance matrix
  const matrix = ov.gtcs_resistance_matrix || {};

  return (
    <div className="p-3">
      {/* Header */}
      <div className="d-flex align-items-start gap-2 mb-2">
        <div>
          <h3 className="mb-0">⚡ SUDEP Risk Assessment Dashboard</h3>
          <p className="text-muted small mb-0">
            Sudden Unexpected Death in Epilepsy — 14-point validated risk model (Hesdorffer 2011 · Surges 2012 · Ryvlin 2013 · NICE NG217 2022).
            Real data: {kpis.total_patients} patients · seizure_metadata + medication_adherence + patients tables.
          </p>
        </div>
        <span className="badge bg-danger ms-auto">P0 Safety</span>
      </div>

      {/* KPI Cards */}
      <div className="row g-2 mb-3">
        <KPI label="Total Patients" value={kpis.total_patients} color="primary" />
        <KPI label="High / Very High Risk" value={kpis.high_very_high_risk}
              color="danger" sub={`${kpis.high_risk_pct}% of cohort`} />
        <KPI label="Avg Risk Score" value={ov.avg_risk_score?.toFixed(1)}
              color={ov.avg_risk_score >= 7 ? 'danger' : ov.avg_risk_score >= 4 ? 'warning' : 'success'}
              sub="out of 14 points" />
        <KPI label="GTCS Patients" value={kpis.gtcs_patients}
              color="warning" sub="Generalised tonic-clonic" />
        <KPI label="Drug-Resistant" value={kpis.drug_resistant_patients}
              color="danger" sub="Failed ≥2 AEDs" />
        <KPI label="Non-Adherent" value={kpis.non_adherent_patients}
              color="warning" sub=">10% missed doses" />
      </div>

      {/* Risk tier badges strip */}
      <div className="d-flex flex-wrap gap-2 mb-3">
        {(ov.risk_tier_distribution || []).map(rt => (
          <div key={rt.tier} className="card text-center px-3 py-2" style={{ borderLeft: `4px solid ${rt.color}`, minWidth: 120 }}>
            <div className="h5 fw-bold mb-0" style={{ color: rt.color }}>{rt.count}</div>
            <div className="small text-muted">{rt.label} Risk</div>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={`nav-link ${tab === t.id ? 'active' : ''}`}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* ── Overview Tab ── */}
      {tab === 'overview' && (
        <div className="row g-3">
          <div className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header small fw-bold">📊 Risk Tier Distribution</div>
              <div className="card-body">
                <Bar items={ov.risk_tier_distribution} />
              </div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header small fw-bold">🔢 Score Histogram</div>
              <div className="card-body">
                <Bar items={ov.score_histogram} colorKey="warning" />
              </div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header small fw-bold">📅 Seizure Frequency Distribution</div>
              <div className="card-body">
                <Bar items={ov.frequency_distribution} colorKey="info" />
              </div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header small fw-bold">⚥ Avg Risk Score by Gender</div>
              <div className="card-body">
                {(ov.avg_score_by_gender || []).map(g => (
                  <div key={g.gender} className="mb-2">
                    <div className="d-flex justify-content-between small mb-1">
                      <span>{g.gender}</span>
                      <span className="fw-bold">{g.avg_score?.toFixed(1)}</span>
                    </div>
                    <div className="progress" style={{ height: 18 }}>
                      <div
                        className={`progress-bar ${g.avg_score >= 7 ? 'bg-danger' : g.avg_score >= 4 ? 'bg-warning' : 'bg-success'}`}
                        style={{ width: `${Math.min(100, (g.avg_score / 14) * 100)}%` }}
                      >
                        <span className="small">{g.avg_score?.toFixed(1)} / 14</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Risk Factors Tab ── */}
      {tab === 'risk-factors' && (
        <div className="row g-3">
          <div className="col-md-8">
            <div className="card shadow-sm h-100">
              <div className="card-header small fw-bold">⚡ Risk Factor Prevalence (of {kpis.total_patients} patients)</div>
              <div className="card-body">
                <Bar items={ov.factor_prevalence} colorKey="danger" />
              </div>
            </div>
          </div>
          <div className="col-md-4">
            <div className="card shadow-sm h-100">
              <div className="card-header small fw-bold">🔀 GTCS × Drug-Resistance Matrix</div>
              <div className="card-body">
                <table className="table table-sm table-bordered text-center mb-0">
                  <thead className="table-light">
                    <tr>
                      <th></th>
                      <th>Drug-Sensitive</th>
                      <th>Drug-Resistant</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <th className="table-light text-start">GTCS</th>
                      <td className="table-warning fw-bold">{matrix.gtcs_only}</td>
                      <td className="table-danger fw-bold">{matrix.gtcs_and_resistant}</td>
                    </tr>
                    <tr>
                      <th className="table-light text-start">No GTCS</th>
                      <td className="table-success fw-bold">{matrix.neither}</td>
                      <td className="table-warning fw-bold">{matrix.resistant_only}</td>
                    </tr>
                  </tbody>
                </table>
                <div className="small text-muted mt-2">
                  Highest risk = GTCS + drug-resistant ({matrix.gtcs_and_resistant} patients)
                </div>
              </div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header small fw-bold">📅 Disease Duration Distribution</div>
              <div className="card-body">
                <Bar items={durationHist} colorKey="info" />
              </div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header small fw-bold">⚠️ Risk Score Interpretation</div>
              <div className="card-body">
                {[
                  { tier: 'Very High (11+)', color: '#ef4444', action: 'Immediate clinical review; SUDEP counselling; night-time monitoring; VNS/surgery evaluation.' },
                  { tier: 'High (7–10)', color: '#f97316', action: 'Enhanced monitoring; optimise AED adherence; seizure diary; SOS alert device.' },
                  { tier: 'Moderate (4–6)', color: '#f59e0b', action: 'SUDEP counselling; review adherence; optimise seizure control.' },
                  { tier: 'Low (0–3)', color: '#22c55e', action: 'Standard monitoring; patient education; annual review.' },
                ].map(r => (
                  <div key={r.tier} className="mb-2 p-2 rounded" style={{ borderLeft: `4px solid ${r.color}`, background: '#f8f9fa' }}>
                    <div className="small fw-bold" style={{ color: r.color }}>{r.tier}</div>
                    <div className="text-muted" style={{ fontSize: '0.75rem' }}>{r.action}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Per Patient Tab ── */}
      {tab === 'patients' && (
        <div>
          <div className="d-flex gap-2 mb-2 align-items-center flex-wrap">
            <input
              className="form-control form-control-sm"
              style={{ maxWidth: 240 }}
              placeholder="Search patient / syndrome / tier…"
              value={search}
              onChange={e => setSearch(e.target.value)}
            />
            <span className="small text-muted">{filtered.length} / {allPatients.length} patients</span>
          </div>
          <div className="table-responsive">
            <table className="table table-sm table-hover">
              <thead className="table-light">
                <tr>
                  <th style={{ cursor: 'pointer' }} onClick={() => toggleSort('patient_id')}>Patient{sortArrow('patient_id')}</th>
                  <th style={{ cursor: 'pointer' }} onClick={() => toggleSort('risk_score')}>Score{sortArrow('risk_score')}</th>
                  <th>Tier</th>
                  <th style={{ cursor: 'pointer' }} onClick={() => toggleSort('age')}>Age{sortArrow('age')}</th>
                  <th>Gender</th>
                  <th style={{ cursor: 'pointer' }} onClick={() => toggleSort('disease_duration_years')}>Duration{sortArrow('disease_duration_years')}</th>
                  <th>GTCS</th>
                  <th>Resistant</th>
                  <th style={{ cursor: 'pointer' }} onClick={() => toggleSort('non_adherence_pct')}>Non-adh%{sortArrow('non_adherence_pct')}</th>
                  <th>Frequency</th>
                  <th>Top Risk Factors</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map(p => (
                  <tr key={p.patient_id}>
                    <td><code className="small">{p.patient_id}</code></td>
                    <td>
                      <span className={`badge bg-${TIER_COLOR[p.risk_tier] || 'secondary'}`}>
                        {p.risk_score} / 14
                      </span>
                    </td>
                    <td><RiskBadge tier={p.risk_tier} /></td>
                    <td>{p.age}</td>
                    <td>{p.gender}</td>
                    <td>{p.disease_duration_years?.toFixed(0)}y</td>
                    <td>
                      {p.gtcs
                        ? <span className="badge bg-danger">Yes</span>
                        : <span className="badge bg-success">No</span>}
                    </td>
                    <td>
                      {p.drug_resistant
                        ? <span className="badge bg-danger">Yes</span>
                        : <span className="badge bg-success">No</span>}
                    </td>
                    <td>
                      <span className={`badge ${p.non_adherence_pct > 10 ? 'bg-warning' : 'bg-success'}`}>
                        {p.non_adherence_pct?.toFixed(1)}%
                      </span>
                    </td>
                    <td><small>{p.seizure_freq}</small></td>
                    <td>
                      <div style={{ maxWidth: 260 }}>
                        {(p.top_factors || []).map((f, i) => (
                          <span key={i} className="badge bg-light text-dark me-1 mb-1" style={{ fontSize: '0.65rem' }}>
                            {f}
                          </span>
                        ))}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── High-Risk Alerts Tab ── */}
      {tab === 'high-risk' && (
        <div>
          <div className="alert alert-danger d-flex align-items-start gap-2 mb-3">
            <span className="fs-4">🚨</span>
            <div>
              <strong>{highRisk.count || 0} patients</strong> are in the High or Very High SUDEP risk tier (score ≥ 7).
              Immediate clinical review, SUDEP counselling, and optimised seizure management are recommended.
            </div>
          </div>
          <div className="row g-2 mb-3">
            {(highRisk.patients || []).map(pid => {
              const pt = allPatients.find(p => p.patient_id === pid);
              if (!pt) return <div key={pid} className="col-6 col-md-4 col-lg-3">
                <div className="card border-danger p-2 text-center">
                  <code className="small">{pid}</code>
                  <span className="badge bg-danger mt-1">High Risk</span>
                </div>
              </div>;
              return (
                <div key={pid} className="col-6 col-md-4 col-lg-3">
                  <div className={`card border-${TIER_COLOR[pt.risk_tier] || 'danger'} h-100`}>
                    <div className="card-body p-2">
                      <div className="d-flex justify-content-between align-items-start">
                        <code className="small">{pt.patient_id}</code>
                        <RiskBadge tier={pt.risk_tier} />
                      </div>
                      <div className={`h5 fw-bold mb-1 text-${TIER_COLOR[pt.risk_tier] || 'danger'}`}>
                        {pt.risk_score} / 14
                      </div>
                      <div className="text-muted" style={{ fontSize: '0.72rem' }}>
                        {pt.age}y · {pt.gender} · {pt.seizure_freq}
                      </div>
                      <div className="mt-1">
                        {pt.gtcs && <span className="badge bg-danger me-1" style={{ fontSize: '0.65rem' }}>GTCS</span>}
                        {pt.drug_resistant && <span className="badge bg-danger me-1" style={{ fontSize: '0.65rem' }}>Resistant</span>}
                        {pt.non_adherence_pct > 10 && <span className="badge bg-warning me-1" style={{ fontSize: '0.65rem' }}>Non-adherent</span>}
                      </div>
                      <div className="mt-1">
                        {(pt.top_factors || []).slice(0, 2).map((f, i) => (
                          <div key={i} className="text-muted" style={{ fontSize: '0.65rem' }}>{f}</div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
          <div className="card shadow-sm">
            <div className="card-header small fw-bold">📋 Clinical Action Checklist — High / Very High Risk Patients</div>
            <div className="card-body">
              <ul className="mb-0 small">
                <li>Schedule SUDEP counselling session with patient and family/caregivers</li>
                <li>Review and optimise anti-epileptic drug (AED) regimen — aim for seizure freedom</li>
                <li>Assess medication adherence — address barriers (side effects, cost, complexity)</li>
                <li>Evaluate for surgical candidacy if drug-resistant ≥2 years</li>
                <li>Consider VNS, RNS, or DBS for eligible patients</li>
                <li>Prescribe wearable seizure-detection device (Embrace, Nighwatch) for nocturnal monitoring</li>
                <li>Train caregivers in SUDEP prevention (sleep position, supervised sleeping)</li>
                <li>Document SUDEP risk discussion in clinical notes (medicolegal)</li>
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* ── Definitions Tab ── */}
      {tab === 'definitions' && defs && (
        <div>
          <div className="card shadow-sm mb-3">
            <div className="card-body">
              <h6 className="fw-bold">{defs.term}</h6>
              <p className="text-muted small mb-1">{defs.definition}</p>
              <div className="small"><strong>Incidence:</strong> {defs.incidence}</div>
            </div>
          </div>
          <div className="row g-3">
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header small fw-bold">Risk Tiers</div>
                <div className="card-body p-2">
                  {(defs.risk_tiers || []).map(t => (
                    <div key={t.tier} className="mb-2 p-2 rounded border-start border-3" style={{ background: '#f8f9fa' }}>
                      <div className="small fw-bold">{t.tier}</div>
                      <div className="text-muted" style={{ fontSize: '0.78rem' }}>{t.description}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header small fw-bold">Risk Factors &amp; Weights</div>
                <div className="card-body p-2">
                  <div className="table-responsive">
                    <table className="table table-sm mb-0">
                      <thead><tr><th>Factor</th><th>Weight</th></tr></thead>
                      <tbody>
                        {(defs.risk_factors || []).map((f, i) => (
                          <tr key={i}>
                            <td style={{ fontSize: '0.75rem' }}>{f.factor}</td>
                            <td><span className="badge bg-danger">{f.weight}</span></td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
            <div className="col-12">
              <div className="card shadow-sm">
                <div className="card-header small fw-bold">References</div>
                <div className="card-body">
                  <ul className="mb-0 small">
                    {(defs.references || []).map((r, i) => <li key={i}>{r}</li>)}
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
