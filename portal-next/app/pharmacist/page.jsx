'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',    label: 'Overview' },
  { id: 'interactions',label: 'Drug Interactions' },
  { id: 'tdm',         label: 'TDM' },
  { id: 'adherence',   label: 'Adherence' },
  { id: 'pregnancy',   label: 'Pregnancy Safety' },
  { id: 'adr',         label: 'ADR / Side Effects' },
];

const ENDPOINTS = {
  overview:     '/api/pharmacist',
  interactions: '/api/pharmacist/interactions',
  tdm:          '/api/pharmacist/tdm',
  adherence:    '/api/pharmacist/adherence',
  pregnancy:    '/api/pharmacist/pregnancy-safety',
  adr:          '/api/pharmacist/adr',
};

const sevColor = s =>
  s === 'major'        ? 'danger'  :
  s === 'moderate'     ? 'warning' :
  s === 'minor'        ? 'info'    :
  s === 'contraindicated' ? 'danger' :
  s === 'high_risk'    ? 'warning' :
  s === 'caution'      ? 'info'    :
  s === 'low'          ? 'success' :
  s === 'medium'       ? 'warning' :
  s === 'high'         ? 'danger'  : 'secondary';

function KPI({ label, value, color }) {
  return (
    <div className="col-6 col-md-3 mb-2">
      <div className="card shadow-sm border-0 h-100">
        <div className="card-body text-center py-2">
          <div className={`h3 mb-0 text-${color || 'primary'}`}>{value ?? '—'}</div>
          <div className="text-muted small">{label}</div>
        </div>
      </div>
    </div>
  );
}

// ─── Overview ────────────────────────────────────────────────────────────────
function OverviewPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  if (data.detail) return <div className="alert alert-warning">{data.detail}</div>;
  const s = data.summary || {};
  const recon = data.reconciliation || {};
  return (
    <div>
      <div className="row mb-3">
        <KPI label="Patients" value={s.total_patients} color="primary" />
        <KPI label="Medication Records" value={s.total_medication_records} color="info" />
        <KPI label="Interactions Found" value={s.total_interactions_found} color="warning" />
        <KPI label="Major Interactions" value={s.major_interactions} color="danger" />
      </div>
      <div className="row mb-3">
        <KPI label="Contraindicated (Pregnancy)" value={s.contraindicated_in_pregnancy} color="danger" />
        <KPI label="Overlapping ADRs" value={s.overlapping_adr_count} color="warning" />
        <KPI label="Low Adherence Patients" value={s.low_adherence_patients} color="warning" />
        <KPI label="ASM Catalog Size" value={s.asm_catalog_size} color="success" />
      </div>

      {recon.reconciled && (
        <div className="card mb-3">
          <div className="card-header fw-semibold">Medication Reconciliation Summary</div>
          <div className="card-body p-0">
            <div className="table-responsive">
              <table className="table table-sm table-striped mb-0">
                <thead><tr>
                  <th>Patient</th><th>Medications</th><th>Unique Drugs</th><th>Duplicates</th><th>Gaps</th>
                </tr></thead>
                <tbody>
                  {recon.reconciled.map(pt => (
                    <tr key={pt.patient_id}>
                      <td className="fw-bold">{pt.patient_id}</td>
                      <td>
                        <small>{(pt.medications || []).map(m => `${m.drug_name} ${m.dose_mg}mg ${m.frequency}`).join(', ')}</small>
                      </td>
                      <td>{pt.unique_medications}</td>
                      <td>{pt.duplicates_found?.length || 0}</td>
                      <td>{pt.gaps_detected?.length || 0}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Drug Interactions ───────────────────────────────────────────────────────
function InteractionsPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const results = data.results || [];
  const withIx = results.filter(r => r.interactions_found > 0);
  return (
    <div>
      <div className="row mb-3">
        <KPI label="Patients Checked" value={data.total_patients} color="primary" />
        <KPI label="Interaction KB Size" value={data.interaction_knowledge_base_size} color="info" />
        <KPI label="Patients with Interactions" value={withIx.length} color="warning" />
        <KPI label="Total Interactions" value={results.reduce((a,r)=>a+r.interactions_found,0)} color="warning" />
      </div>

      {withIx.length === 0 && (
        <div className="alert alert-success">No clinically significant interactions found across current medication records.</div>
      )}

      {withIx.map(pt => (
        <div key={pt.patient_id} className="card mb-3">
          <div className="card-header">
            <strong>{pt.patient_id}</strong>
            <span className="ms-2 badge bg-secondary">{(pt.drugs_checked || []).join(', ')}</span>
          </div>
          <div className="card-body p-0">
            <div className="table-responsive">
              <table className="table table-sm mb-0">
                <thead><tr><th>Drug A</th><th>Drug B</th><th>Severity</th><th>Mechanism</th><th>Action</th></tr></thead>
                <tbody>
                  {(pt.interactions || []).map((ix, i) => (
                    <tr key={i}>
                      <td className="fw-bold">{ix.drug_a}</td>
                      <td className="fw-bold">{ix.drug_b}</td>
                      <td><span className={`badge bg-${sevColor(ix.severity)}`}>{ix.severity}</span></td>
                      <td><small>{ix.mechanism}</small></td>
                      <td><small>{ix.recommended_action}</small></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      ))}

      <div className="card">
        <div className="card-header fw-semibold">All Patients — Interaction Summary</div>
        <div className="card-body p-0">
          <div className="table-responsive">
            <table className="table table-sm table-striped mb-0">
              <thead><tr><th>Patient</th><th>Drugs</th><th>Pairs Checked</th><th>Found</th><th>Major</th><th>Moderate</th><th>Minor</th></tr></thead>
              <tbody>
                {results.map(pt => (
                  <tr key={pt.patient_id}>
                    <td className="fw-bold">{pt.patient_id}</td>
                    <td><small>{(pt.drugs_checked||[]).join(', ')}</small></td>
                    <td>{pt.pairs_checked}</td>
                    <td>{pt.interactions_found > 0
                      ? <span className="badge bg-warning text-dark">{pt.interactions_found}</span>
                      : <span className="badge bg-success">0</span>}
                    </td>
                    <td>{pt.severity_summary?.major || 0}</td>
                    <td>{pt.severity_summary?.moderate || 0}</td>
                    <td>{pt.severity_summary?.minor || 0}</td>
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

// ─── TDM ─────────────────────────────────────────────────────────────────────
function TDMPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const results = data.results || [];
  return (
    <div>
      <div className="row mb-3">
        <KPI label="Patients Monitored" value={data.total_patients} color="primary" />
        <KPI label="Drug-Level Pairs" value={results.reduce((a,r)=>a+r.medications_monitored,0)} color="info" />
      </div>
      <p className="text-muted small mb-3">
        Therapeutic Drug Monitoring (TDM) — target serum trough levels per published therapeutic ranges.
        Draw trough levels before next scheduled dose for accurate interpretation.
      </p>
      {results.map(pt => (
        <div key={pt.patient_id} className="card mb-3">
          <div className="card-header fw-semibold">{pt.patient_id} — {pt.medications_monitored} drug{pt.medications_monitored!==1?'s':''} monitored</div>
          <div className="card-body p-0">
            <div className="table-responsive">
              <table className="table table-sm mb-0">
                <thead><tr><th>Drug</th><th>Brand</th><th>Dose</th><th>Frequency</th><th>Target Range (mcg/mL)</th><th>Recommendation</th></tr></thead>
                <tbody>
                  {(pt.tdm || []).map((m, i) => (
                    <tr key={i}>
                      <td className="fw-bold">{m.drug}</td>
                      <td><small className="text-muted">{m.brand}</small></td>
                      <td>{m.dose_mg} mg</td>
                      <td>{m.frequency}</td>
                      <td>
                        {m.therapeutic_range_mcg_ml
                          ? <span className="badge bg-info text-dark">{m.therapeutic_range_mcg_ml[0]}–{m.therapeutic_range_mcg_ml[1]}</span>
                          : <span className="badge bg-secondary">N/A</span>}
                      </td>
                      <td><small>{m.recommendation}</small></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

// ─── Adherence ───────────────────────────────────────────────────────────────
function AdherencePanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const s = data.summary || {};
  const results = data.results || [];
  return (
    <div>
      <div className="row mb-3">
        <KPI label="Total Patients" value={data.total_patients} color="primary" />
        <KPI label="High Adherence" value={s.high_adherence} color="success" />
        <KPI label="Medium Adherence" value={s.medium_adherence} color="warning" />
        <KPI label="Low Adherence" value={s.low_adherence} color="danger" />
      </div>
      <div className="row mb-3">
        <KPI label="Avg MMAS-8 Score" value={s.avg_mmas8} color="info" />
        <KPI label="Avg MPR" value={s.avg_mpr} color="info" />
      </div>
      <div className="table-responsive">
        <table className="table table-sm table-striped">
          <thead><tr>
            <th>Patient</th><th>Medications</th><th>MMAS-8 Score</th><th>MMAS Level</th><th>MPR</th><th>Seizure Gap?</th><th>Notes</th>
          </tr></thead>
          <tbody>
            {results.map(pt => (
              <tr key={pt.patient_id}>
                <td className="fw-bold">{pt.patient_id}</td>
                <td><small>{(pt.medications||[]).join(', ')}</small></td>
                <td><strong>{pt.mmas8_proxy_score}</strong>/8</td>
                <td><span className={`badge bg-${sevColor(pt.mmas8_level)}`}>{pt.mmas8_level}</span></td>
                <td>{(pt.mpr_estimate*100).toFixed(0)}%</td>
                <td>{pt.seizure_gap_flag
                  ? <span className="badge bg-warning text-dark">Yes</span>
                  : <span className="badge bg-secondary">No</span>}
                </td>
                <td><small>{(pt.coaching_notes||[]).join('; ') || '—'}</small></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {results.some(pt => Object.values(pt.mmas8_risk_factors||{}).some(Boolean)) && (
        <div className="card mt-3">
          <div className="card-header fw-semibold">MMAS-8 Risk Factors by Patient</div>
          <div className="card-body p-0">
            <div className="table-responsive">
              <table className="table table-sm mb-0">
                <thead><tr>
                  <th>Patient</th><th>Polypharmacy</th><th>High-Freq Dosing</th><th>ADR Burden</th><th>Pregnancy Drug</th><th>Drug Interactions</th>
                </tr></thead>
                <tbody>
                  {results.map(pt => {
                    const rf = pt.mmas8_risk_factors || {};
                    return (
                      <tr key={pt.patient_id}>
                        <td className="fw-bold">{pt.patient_id}</td>
                        {['polypharmacy','high_freq_dosing','high_adr_burden','pregnancy_risk_drug','drug_interactions'].map(k => (
                          <td key={k}>{rf[k]
                            ? <span className="badge bg-warning text-dark">Yes</span>
                            : <span className="badge bg-success">No</span>}
                          </td>
                        ))}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Pregnancy Safety ────────────────────────────────────────────────────────
function PregnancyPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const results = data.results || [];
  const urgent = results.filter(pt => pt.urgent_alert);
  return (
    <div>
      <div className="row mb-3">
        <KPI label="Patients Assessed" value={data.total_patients} color="primary" />
        <KPI label="Urgent Alerts" value={urgent.length} color="danger" />
        <KPI label="Contraindicated Drugs" value={results.reduce((a,r)=>a+r.contraindicated_count,0)} color="danger" />
        <KPI label="High-Risk Drugs" value={results.reduce((a,r)=>a+r.high_risk_count,0)} color="warning" />
      </div>

      {urgent.length > 0 && (
        <div className="alert alert-danger mb-3">
          <strong>Urgent — Contraindicated AED in reproductive-age patients:</strong>{' '}
          {urgent.map(p=>p.patient_id).join(', ')}
        </div>
      )}

      {results.map(pt => (
        <div key={pt.patient_id} className={`card mb-3 border-${pt.urgent_alert?'danger':pt.high_risk_count>0?'warning':'secondary'}`}>
          <div className="card-header">
            <strong>{pt.patient_id}</strong>
            {pt.urgent_alert && <span className="badge bg-danger ms-2">URGENT</span>}
            {pt.folate_supplementation_needed && <span className="badge bg-warning text-dark ms-1">Folate needed</span>}
          </div>
          <div className="card-body p-0">
            <div className="table-responsive">
              <table className="table table-sm mb-0">
                <thead><tr><th>Drug</th><th>Brand</th><th>Pregnancy Cat.</th><th>Risk Level</th><th>Guidance</th></tr></thead>
                <tbody>
                  {(pt.medications||[]).map((m,i) => (
                    <tr key={i}>
                      <td className="fw-bold">{m.drug}</td>
                      <td><small className="text-muted">{m.brand}</small></td>
                      <td><strong>{m.pregnancy_category}</strong></td>
                      <td><span className={`badge bg-${sevColor(m.risk_level)}`}>{m.risk_level}</span></td>
                      <td><small>{m.guidance}</small></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

// ─── ADR / Side Effects ──────────────────────────────────────────────────────
function ADRPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const results = data.results || [];
  const withOverlap = results.filter(pt => Object.keys(pt.overlapping_adrs||{}).length > 0);
  return (
    <div>
      <div className="row mb-3">
        <KPI label="Patients Assessed" value={data.total_patients} color="primary" />
        <KPI label="Overlapping ADRs" value={withOverlap.length} color="warning" />
        <KPI label="Total Unique ADRs" value={results.reduce((a,r)=>a+r.total_unique_adrs,0)} color="info" />
      </div>

      {withOverlap.map(pt => (
        <div key={pt.patient_id} className="alert alert-warning mb-2">
          <strong>{pt.patient_id}:</strong> overlapping ADRs —{' '}
          {Object.entries(pt.overlapping_adrs||{}).map(([adr,cnt]) => (
            <span key={adr} className="badge bg-warning text-dark me-1">{adr} ({cnt} drugs)</span>
          ))}
        </div>
      ))}

      {results.map(pt => (
        <div key={pt.patient_id} className="card mb-3">
          <div className="card-header fw-semibold">
            {pt.patient_id}
            {pt.high_risk_flags?.length > 0 && (
              <span className="badge bg-danger ms-2">{pt.high_risk_flags.length} high-risk flag{pt.high_risk_flags.length!==1?'s':''}</span>
            )}
          </div>
          <div className="card-body p-0">
            <div className="table-responsive">
              <table className="table table-sm mb-0">
                <thead><tr><th>Drug</th><th>Brand</th><th>Known ADRs</th></tr></thead>
                <tbody>
                  {(pt.medications||[]).map((m,i) => (
                    <tr key={i}>
                      <td className="fw-bold">{m.drug}</td>
                      <td><small className="text-muted">{m.brand}</small></td>
                      <td><small>{(m.known_adrs||[]).join(', ')}</small></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          {pt.high_risk_flags?.length > 0 && (
            <div className="card-footer text-danger small">
              <strong>Flags:</strong> {pt.high_risk_flags.join(', ')}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

// ─── Main page ───────────────────────────────────────────────────────────────
export default function PharmacistPage() {
  const [tab, setTab]   = useState('overview');
  const [data, setData] = useState({});
  const [loading, setLoading] = useState(false);
  const [err, setErr]   = useState(null);
  const [pid, setPid]   = useState('');
  const [filterPid, setFilterPid] = useState('');

  function loadTab(t, pid) {
    const ep = ENDPOINTS[t];
    if (!ep) return;
    const url = pid ? `${API}${ep}?patient_id=${pid}` : `${API}${ep}`;
    setLoading(true); setErr(null);
    fetch(url)
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
      .then(d => { setData(prev => ({ ...prev, [t]: d })); setLoading(false); })
      .catch(e => { setErr(`${t}: ${e.message}`); setLoading(false); });
  }

  useEffect(() => { loadTab('overview', filterPid); }, [filterPid]);

  function switchTab(t) {
    setTab(t);
    if (!data[t]) loadTab(t, filterPid);
  }

  function handleFilter(e) {
    e.preventDefault();
    setFilterPid(pid.trim());
    setData({});
    setTab('overview');
  }

  return (
    <div className="container-fluid p-3">
      <h3 className="mb-1">&#x1f48a; Clinical Pharmacist (Epilepsy)</h3>
      <p className="text-muted mb-3">
        Medication reconciliation, drug-drug interactions (DDI), therapeutic drug monitoring (TDM),
        MMAS-8 adherence, pregnancy safety, and ADR profiling — all from real clinical.db data.
      </p>

      {/* Patient filter */}
      <form className="row g-2 mb-3 align-items-end" onSubmit={handleFilter}>
        <div className="col-auto">
          <label className="form-label small mb-0">Filter by Patient ID</label>
          <input className="form-control form-control-sm" placeholder="e.g. P0002" value={pid}
            onChange={e => setPid(e.target.value)} />
        </div>
        <div className="col-auto">
          <button className="btn btn-sm btn-primary" type="submit">Filter</button>
          {filterPid && (
            <button className="btn btn-sm btn-outline-secondary ms-1" type="button"
              onClick={() => { setPid(''); setFilterPid(''); setData({}); }}>
              Clear
            </button>
          )}
        </div>
        {filterPid && <div className="col-auto"><span className="badge bg-info">Filtered: {filterPid}</span></div>}
      </form>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link${tab === t.id ? ' active' : ''}`}
              onClick={() => switchTab(t.id)}>{t.label}</button>
          </li>
        ))}
      </ul>

      {err    && <div className="alert alert-danger small">{err}</div>}
      {loading && <div className="text-muted small">Loading…</div>}

      {!loading && tab === 'overview'     && <OverviewPanel     data={data.overview}      />}
      {!loading && tab === 'interactions' && <InteractionsPanel data={data.interactions}  />}
      {!loading && tab === 'tdm'          && <TDMPanel          data={data.tdm}           />}
      {!loading && tab === 'adherence'    && <AdherencePanel    data={data.adherence}     />}
      {!loading && tab === 'pregnancy'    && <PregnancyPanel    data={data.pregnancy}     />}
      {!loading && tab === 'adr'          && <ADRPanel          data={data.adr}           />}
    </div>
  );
}
