'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

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

function Bar({ items, colorFn }) {
  const mx = Math.max(...(items || []).map(i => i.count || 0), 1);
  return (
    <div>
      {(items || []).map((it, i) => {
        const val = it.count ?? 0;
        const label = it.label || it.gender || '?';
        const pct = Math.round((val / mx) * 100);
        const color = colorFn ? colorFn(it) : 'primary';
        return (
          <div key={i} className="d-flex align-items-center mb-1 gap-2">
            <div className="text-end small text-muted" style={{ width: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontSize: '0.75rem' }}>
              {label}
            </div>
            <div className="flex-grow-1">
              <div className="progress" style={{ height: 16 }}>
                <div className={`progress-bar bg-${color}`} style={{ width: `${pct}%` }}>
                  <span className="small px-1">{val}</span>
                </div>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function DREBadge({ dre }) {
  return dre
    ? <span className="badge bg-danger">DRE</span>
    : <span className="badge bg-success">Responsive</span>;
}

function SurgBadge({ surgical }) {
  return surgical
    ? <span className="badge bg-warning text-dark">Surgical Candidate</span>
    : <span className="badge bg-secondary">Medical Mgmt</span>;
}

export default function DREPage() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [search, setSearch] = useState('');
  const [filterDRE, setFilterDRE] = useState('all');
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/dre/overview`).then(r => r.json()),
      fetch(`${API}/api/dre/breakdown`).then(r => r.json()),
      fetch(`${API}/api/dre/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-4">{err}</div>;
  if (!ov) return <div className="p-4 text-muted">Loading Drug-Resistant Epilepsy dashboard…</div>;

  const kp = ov.kpis || {};

  // Filter patients
  let patients = bd?.patients || [];
  if (search) {
    const s = search.toLowerCase();
    patients = patients.filter(p =>
      String(p.patient_id).toLowerCase().includes(s) ||
      (p.onset_zone || '').toLowerCase().includes(s) ||
      (p.syndrome || '').toLowerCase().includes(s)
    );
  }
  if (filterDRE === 'dre') patients = patients.filter(p => p.dre);
  if (filterDRE === 'responsive') patients = patients.filter(p => !p.dre);
  if (filterDRE === 'surgical') patients = patients.filter(p => p.surgical_candidate);

  return (
    <div className="container-fluid py-3">
      <h4 className="fw-bold mb-1">
        <span className="me-2">💊</span>Drug-Resistant Epilepsy (DRE) Dashboard
      </h4>
      <p className="text-muted small mb-3">
        ILAE 2010 classification · {kp.total_patients} patients · {kp.dre_prevalence_pct}% DRE prevalence
        · Source: clinical.db — seizure_metadata + patients + medication_adherence
      </p>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {['overview', 'breakdown', 'definitions'].map(t => (
          <li className="nav-item" key={t}>
            <button
              className={`nav-link${tab === t ? ' active' : ''}`}
              onClick={() => setTab(t)}
            >
              {t.charAt(0).toUpperCase() + t.slice(1)}
            </button>
          </li>
        ))}
      </ul>

      {/* ── OVERVIEW ── */}
      {tab === 'overview' && (
        <>
          {/* KPIs */}
          <div className="row mb-3">
            <KPI label="Total Patients" value={kp.total_patients} color="primary" />
            <KPI label="DRE Patients" value={kp.dre_patients} color="danger" sub={`${kp.dre_prevalence_pct}% prevalence`} />
            <KPI label="Responsive" value={kp.non_dre_patients} color="success" />
            <KPI label="Surgical Candidates" value={kp.surgical_candidates} color="warning" sub={`${kp.surgical_of_dre_pct}% of DRE`} />
            <KPI label="Avg AED Trials (DRE)" value={kp.avg_aed_trials_dre} color="danger" />
            <KPI label="Avg AED Trials (Non-DRE)" value={kp.avg_aed_trials_non_dre} color="success" />
          </div>

          <div className="row">
            {/* AED Burden */}
            <div className="col-md-6 mb-3">
              <div className="card h-100">
                <div className="card-header py-2 fw-semibold small">AED Trial Burden Distribution</div>
                <div className="card-body">
                  <Bar items={ov.aed_burden_chart} colorFn={it =>
                    it.label?.includes('5+') ? 'danger' :
                    it.label?.includes('ILAE') ? 'warning' :
                    it.label?.includes('mono') ? 'info' : 'secondary'
                  } />
                </div>
              </div>
            </div>

            {/* Onset Zone (DRE) */}
            <div className="col-md-6 mb-3">
              <div className="card h-100">
                <div className="card-header py-2 fw-semibold small">Onset Zone — DRE Patients</div>
                <div className="card-body">
                  <Bar items={ov.onset_zone_chart} colorFn={() => 'danger'} />
                </div>
              </div>
            </div>

            {/* Seizure Control */}
            <div className="col-md-6 mb-3">
              <div className="card h-100">
                <div className="card-header py-2 fw-semibold small">Seizure Control Distribution</div>
                <div className="card-body">
                  <Bar items={ov.seizure_control_chart} colorFn={it =>
                    it.label === 'Seizure-Free' ? 'success' :
                    it.label === 'Daily' ? 'danger' :
                    it.label === 'Weekly' ? 'warning' : 'secondary'
                  } />
                </div>
              </div>
            </div>

            {/* DRE by Gender */}
            <div className="col-md-6 mb-3">
              <div className="card h-100">
                <div className="card-header py-2 fw-semibold small">DRE Rate by Gender</div>
                <div className="card-body">
                  {(ov.gender_dre_chart || []).map((g, i) => (
                    <div key={i} className="d-flex align-items-center mb-2 gap-3">
                      <div style={{ width: 80 }} className="small text-muted">{g.gender}</div>
                      <div className="flex-grow-1">
                        <div className="progress" style={{ height: 18 }}>
                          <div className="progress-bar bg-danger" style={{ width: `${g.pct}%` }}>
                            <span className="small px-1">{g.dre}/{g.total}</span>
                          </div>
                        </div>
                      </div>
                      <div className="small text-muted" style={{ width: 44 }}>{g.pct}%</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Disease Duration (DRE) */}
            <div className="col-md-6 mb-3">
              <div className="card h-100">
                <div className="card-header py-2 fw-semibold small">Disease Duration — DRE Patients</div>
                <div className="card-body">
                  <Bar items={ov.disease_duration_chart} colorFn={() => 'danger'} />
                </div>
              </div>
            </div>

            {/* Top AEDs in DRE */}
            <div className="col-md-6 mb-3">
              <div className="card h-100">
                <div className="card-header py-2 fw-semibold small">Most-Tried AEDs in DRE Patients</div>
                <div className="card-body">
                  <Bar items={bd?.top_aeds_in_dre || []} colorFn={() => 'warning'} />
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      {/* ── BREAKDOWN ── */}
      {tab === 'breakdown' && (
        <>
          <div className="row mb-2 g-2 align-items-end">
            <div className="col-md-5">
              <input
                className="form-control form-control-sm"
                placeholder="Search patient ID / onset zone / syndrome…"
                value={search}
                onChange={e => setSearch(e.target.value)}
              />
            </div>
            <div className="col-md-4">
              <select className="form-select form-select-sm" value={filterDRE} onChange={e => setFilterDRE(e.target.value)}>
                <option value="all">All patients</option>
                <option value="dre">DRE only</option>
                <option value="responsive">Responsive only</option>
                <option value="surgical">Surgical candidates</option>
              </select>
            </div>
            <div className="col-md-3 text-muted small">
              Showing {patients.length} / {bd?.total || 0} patients
            </div>
          </div>

          {/* Surgical Candidates summary */}
          {filterDRE !== 'responsive' && (bd?.surgical_candidates || []).length > 0 && (
            <div className="alert alert-warning py-2 small mb-2">
              <strong>⚠ {bd.surgical_candidates.length} surgical candidates identified</strong> — focal-onset DRE with disease duration ≥ 2 years. Pre-surgical evaluation recommended.
            </div>
          )}

          <div className="table-responsive">
            <table className="table table-sm table-hover small align-middle">
              <thead className="table-dark">
                <tr>
                  <th>Patient ID</th>
                  <th>Status</th>
                  <th>AED Trials</th>
                  <th>Seizure Control</th>
                  <th>Surgical</th>
                  <th>Onset Zone</th>
                  <th>Syndrome</th>
                  <th>Duration (y)</th>
                  <th>Age</th>
                  <th>Gender</th>
                  <th>Non-Adh %</th>
                </tr>
              </thead>
              <tbody>
                {patients.map((p, i) => (
                  <tr key={i} className={p.dre ? 'table-danger' : ''}>
                    <td className="fw-semibold">{p.patient_id}</td>
                    <td><DREBadge dre={p.dre} /></td>
                    <td>
                      <span className={`badge bg-${p.aed_count >= 2 ? 'danger' : 'secondary'}`}>
                        {p.aed_count}
                      </span>
                      <span className="text-muted ms-1" style={{ fontSize: '0.7rem' }}>{p.aed_bucket}</span>
                    </td>
                    <td>
                      <span className={`badge bg-${
                        p.seizure_control === 'Seizure-Free' ? 'success' :
                        p.seizure_control === 'Daily' ? 'danger' :
                        p.seizure_control === 'Weekly' ? 'warning' : 'secondary'
                      }`}>{p.seizure_control}</span>
                    </td>
                    <td><SurgBadge surgical={p.surgical_candidate} /></td>
                    <td>{p.onset_zone || '—'}</td>
                    <td className="text-muted">{p.syndrome || '—'}</td>
                    <td>{p.disease_duration_years ?? '—'}</td>
                    <td>{p.age ?? '—'}</td>
                    <td>{p.gender || '—'}</td>
                    <td>
                      <span className={p.non_adherence_pct > 10 ? 'text-danger fw-bold' : ''}>
                        {p.non_adherence_pct}%
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'definitions' && defs && (
        <div className="row">
          <div className="col-lg-8">
            <div className="card mb-3">
              <div className="card-header fw-semibold">ILAE 2010 DRE Definition</div>
              <div className="card-body">
                <p className="small">{defs.ilae_definition}</p>
                <p className="small text-muted mb-0"><strong>Prevalence:</strong> {defs.prevalence}</p>
              </div>
            </div>

            <div className="card mb-3">
              <div className="card-header fw-semibold">Classification Criteria (This Dashboard)</div>
              <div className="card-body">
                <ul className="small mb-0">
                  {(defs.dre_criteria_used || []).map((c, i) => <li key={i}>{c}</li>)}
                </ul>
                <p className="small text-muted mt-2 mb-0">
                  <strong>Surgical candidacy proxy:</strong> {defs.surgical_candidacy_proxy}
                </p>
              </div>
            </div>

            <div className="card mb-3">
              <div className="card-header fw-semibold">Surgical Pathway (4 Phases)</div>
              <div className="card-body">
                <ol className="small mb-0">
                  {(defs.surgical_pathway || []).map((s, i) => <li key={i}>{s}</li>)}
                </ol>
              </div>
            </div>

            <div className="card mb-3">
              <div className="card-header fw-semibold">Risk Tiers</div>
              <div className="card-body p-0">
                <table className="table table-sm small mb-0">
                  <thead className="table-light"><tr><th>Tier</th><th>Criteria</th><th>Action</th></tr></thead>
                  <tbody>
                    {(defs.risk_tiers || []).map((t, i) => (
                      <tr key={i}>
                        <td><span className="badge" style={{ backgroundColor: t.color }}>{t.tier}</span></td>
                        <td>{t.criteria}</td>
                        <td>{t.action}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="col-lg-4">
            <div className="card mb-3">
              <div className="card-header fw-semibold">Glossary</div>
              <div className="card-body p-0">
                <table className="table table-sm small mb-0">
                  <thead className="table-light"><tr><th>Term</th><th>Definition</th></tr></thead>
                  <tbody>
                    {(defs.glossary || []).map((g, i) => (
                      <tr key={i}><td className="fw-semibold">{g.term}</td><td>{g.definition}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="card mb-3">
              <div className="card-header fw-semibold">References</div>
              <div className="card-body">
                <ul className="small mb-0 ps-3">
                  {(defs.references || []).map((r, i) => <li key={i} className="mb-1">{r}</li>)}
                </ul>
              </div>
            </div>

            <div className="card">
              <div className="card-header fw-semibold">AED Generation Note</div>
              <div className="card-body">
                <p className="small mb-0">{defs.aed_generation_note}</p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
