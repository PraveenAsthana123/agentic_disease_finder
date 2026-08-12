'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const SEVERITY_BADGE = (val) => {
  if (val == null) return <span className="badge bg-secondary">N/A</span>;
  if (val < 2) return <span className="badge bg-success">{val}</span>;
  if (val < 4) return <span className="badge bg-warning text-dark">{val}</span>;
  return <span className="badge bg-danger">{val}</span>;
};

const ADH_BADGE = (val) => {
  if (val == null) return <span className="badge bg-secondary">N/A</span>;
  if (val >= 90) return <span className="badge bg-success">{val}%</span>;
  if (val >= 75) return <span className="badge bg-warning text-dark">{val}%</span>;
  return <span className="badge bg-danger">{val}%</span>;
};

export default function AEDPolypharmacyPage() {
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [sortField, setSortField] = useState('patient_id');
  const [sortDir, setSortDir] = useState('asc');
  const [patientSearch, setPatientSearch] = useState('');

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/aed-polypharmacy/overview`).then(r => r.json()),
      fetch(`${API}/api/aed-polypharmacy/breakdown`).then(r => r.json()),
      fetch(`${API}/api/aed-polypharmacy/definitions`).then(r => r.json()),
    ])
      .then(([ov, bk, df]) => {
        setOverview(ov);
        setBreakdown(bk);
        setDefinitions(df);
        setLoading(false);
      })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  const handleSort = (field) => {
    if (sortField === field) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortField(field); setSortDir('asc'); }
  };

  const sortedPatients = () => {
    if (!breakdown?.patients) return [];
    let pts = [...breakdown.patients];
    if (patientSearch) {
      const q = patientSearch.toLowerCase();
      pts = pts.filter(p =>
        p.patient_id.toLowerCase().includes(q) ||
        p.regimen_label.toLowerCase().includes(q)
      );
    }
    pts.sort((a, b) => {
      const av = a[sortField] ?? '';
      const bv = b[sortField] ?? '';
      const cmp = typeof av === 'string' ? av.localeCompare(bv) : (av - bv);
      return sortDir === 'asc' ? cmp : -cmp;
    });
    return pts;
  };

  const SortTh = ({ field, label }) => (
    <th onClick={() => handleSort(field)} style={{ cursor: 'pointer', userSelect: 'none' }}>
      {label} {sortField === field ? (sortDir === 'asc' ? '\u25b2' : '\u25bc') : '\u2195'}
    </th>
  );

  if (loading) return (
    <div className="container-fluid py-5 text-center">
      <div className="spinner-border text-primary" role="status" />
      <p className="mt-3 text-muted">Loading AED Polypharmacy data...</p>
    </div>
  );

  if (error) return (
    <div className="container-fluid py-4">
      <div className="alert alert-danger">Error: {error}</div>
    </div>
  );

  const kpis = overview?.kpis || {};
  const pairs = overview?.top_drug_pairs || [];
  const triples = overview?.triple_regimens || [];
  const comboadh = overview?.drug_combo_adherence || [];
  const sideEffectBurden = overview?.side_effect_burden_by_count || [];
  const szControl = overview?.seizure_control_by_regimen || [];
  const distrib = overview?.drug_count_distribution || [];

  const matrix = breakdown?.drug_pair_matrix || {};
  const drugs = matrix.drugs || [];
  const matData = matrix.matrix || {};

  return (
    <div className="container-fluid py-4" style={{ fontFamily: 'system-ui, sans-serif' }}>
      {/* Header */}
      <div className="row mb-4">
        <div className="col">
          <h2 className="fw-bold" style={{ color: '#0b1f3a' }}>
            <span className="me-2">&#x1f48a;</span>AED Polypharmacy Analysis
          </h2>
          <p className="text-muted mb-0">
            Multi-drug regimen analysis &mdash; 30 patients on polytherapy (14 dual / 16 triple) &middot; 12,600 dose records &middot; 8 AEDs
          </p>
          {overview?.updated_at && (
            <small className="text-muted">Updated: {new Date(overview.updated_at).toLocaleString()}</small>
          )}
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-4">
        {[
          ['overview', 'Overview'],
          ['combinations', 'Drug Combinations'],
          ['matrix', 'Co-occurrence Matrix'],
          ['patients', 'Per Patient'],
          ['definitions', 'Definitions'],
        ].map(([id, label]) => (
          <li className="nav-item" key={id}>
            <button
              className={`nav-link ${activeTab === id ? 'active fw-semibold' : ''}`}
              onClick={() => setActiveTab(id)}
            >
              {label}
            </button>
          </li>
        ))}
      </ul>

      {/* TAB 1: Overview */}
      {activeTab === 'overview' && (
        <div>
          {/* KPI Cards */}
          <div className="row g-3 mb-4">
            {[
              { label: 'Total Patients', value: kpis.total_patients, color: '#0b1f3a', icon: '&#x1f465;' },
              { label: 'Dual Therapy', value: kpis.on_dual_therapy, sub: '2 AEDs', color: '#1a6b3c', icon: '&#x1f48a;' },
              { label: 'Triple Therapy', value: kpis.on_triple_therapy, sub: '3 AEDs', color: '#7b2d00', icon: '&#x1f48a;&#x1f48a;' },
              { label: 'Polytherapy Rate', value: `${kpis.polytherapy_rate}%`, color: '#4a0072', icon: '&#x1f4ca;' },
              { label: 'Avg Adherence', value: `${kpis.avg_adherence_pct}%`, color: '#155724', icon: '&#x2705;' },
              { label: 'Total Dose Records', value: kpis.total_dose_records?.toLocaleString(), color: '#0c5460', icon: '&#x1f4dd;' },
              { label: 'AEDs in Use', value: kpis.drugs_in_use, color: '#721c24', icon: '&#x1f9ea;' },
              { label: 'Unique Regimens', value: kpis.unique_regimen_combos, color: '#383d41', icon: '&#x1f500;' },
            ].map(({ label, value, sub, color, icon }) => (
              <div className="col-6 col-md-3 col-xl-3" key={label}>
                <div className="card border-0 shadow-sm h-100" style={{ borderLeft: `4px solid ${color}` }}>
                  <div className="card-body p-3">
                    <div className="d-flex justify-content-between align-items-center">
                      <div>
                        <div className="text-muted small">{label}</div>
                        <div className="fs-4 fw-bold" style={{ color }}>{value}</div>
                        {sub && <div className="text-muted" style={{ fontSize: '0.75rem' }}>{sub}</div>}
                      </div>
                      <span style={{ fontSize: '1.5rem' }} dangerouslySetInnerHTML={{ __html: icon }} />
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Row: Distribution + Dual vs Triple Adherence */}
          <div className="row g-3 mb-4">
            <div className="col-md-4">
              <div className="card border-0 shadow-sm h-100">
                <div className="card-body">
                  <h6 className="fw-bold mb-3">Drug Count Distribution</h6>
                  {distrib.map(d => (
                    <div key={d.label} className="mb-3">
                      <div className="d-flex justify-content-between mb-1">
                        <span className="small fw-semibold">{d.label}</span>
                        <span className="badge bg-primary">{d.count} patients</span>
                      </div>
                      <div className="progress" style={{ height: '20px' }}>
                        <div
                          className="progress-bar"
                          style={{
                            width: `${(d.count / (kpis.total_patients || 30)) * 100}%`,
                            backgroundColor: d.label.startsWith('Dual') ? '#1a6b3c' : '#7b2d00',
                          }}
                        >
                          {Math.round((d.count / (kpis.total_patients || 30)) * 100)}%
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="col-md-4">
              <div className="card border-0 shadow-sm h-100">
                <div className="card-body">
                  <h6 className="fw-bold mb-3">Adherence: Dual vs Triple</h6>
                  {comboadh.map(row => (
                    <div key={row.label} className="mb-3">
                      <div className="d-flex justify-content-between mb-1">
                        <span className="small fw-semibold">{row.label}</span>
                        <span className="small text-muted">{row.avg_adherence_pct}%</span>
                      </div>
                      <div className="progress" style={{ height: '20px' }}>
                        <div
                          className={`progress-bar ${row.avg_adherence_pct >= 90 ? 'bg-success' : 'bg-warning'}`}
                          style={{ width: `${row.avg_adherence_pct}%` }}
                        >
                          {row.avg_adherence_pct}%
                        </div>
                      </div>
                      <small className="text-muted">{row.patient_count} patients</small>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="col-md-4">
              <div className="card border-0 shadow-sm h-100">
                <div className="card-body">
                  <h6 className="fw-bold mb-3">Side Effect Burden by Regimen</h6>
                  {sideEffectBurden.map(row => (
                    <div key={row.drug_count} className="mb-3">
                      <div className="d-flex justify-content-between mb-1">
                        <span className="small fw-semibold">{row.label}</span>
                        <span className="small fw-bold">{row.avg_severity} / 10</span>
                      </div>
                      <div className="progress" style={{ height: '20px' }}>
                        <div
                          className={`progress-bar ${row.avg_severity < 3 ? 'bg-success' : row.avg_severity < 6 ? 'bg-warning' : 'bg-danger'}`}
                          style={{ width: `${(row.avg_severity / 10) * 100}%` }}
                        >
                          {row.avg_severity}
                        </div>
                      </div>
                    </div>
                  ))}
                  <p className="text-muted small mt-2 mb-0">Scale: 0 (none) to 10 (severe). Higher drug count = greater combined burden.</p>
                </div>
              </div>
            </div>
          </div>

          {/* Seizure control by regimen */}
          {szControl.length > 0 && (
            <div className="card border-0 shadow-sm mb-4">
              <div className="card-body">
                <h6 className="fw-bold mb-3">Seizure Control by Drug Combination (lower rate = better control)</h6>
                <div className="table-responsive">
                  <table className="table table-sm table-hover">
                    <thead className="table-light">
                      <tr>
                        <th>Drug Combination</th>
                        <th>Patients</th>
                        <th>Avg Seizure Rate (%)</th>
                        <th>Control</th>
                      </tr>
                    </thead>
                    <tbody>
                      {szControl.map(r => (
                        <tr key={r.combination}>
                          <td className="fw-semibold">{r.combination}</td>
                          <td>{r.patient_count}</td>
                          <td>{r.avg_seizure_rate}%</td>
                          <td>
                            <div className="progress" style={{ height: '14px', minWidth: '80px' }}>
                              <div
                                className={`progress-bar ${r.avg_seizure_rate < 30 ? 'bg-success' : r.avg_seizure_rate < 60 ? 'bg-warning' : 'bg-danger'}`}
                                style={{ width: `${r.avg_seizure_rate}%` }}
                              />
                            </div>
                          </td>
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

      {/* TAB 2: Drug Combinations */}
      {activeTab === 'combinations' && (
        <div>
          <div className="card border-0 shadow-sm mb-4">
            <div className="card-body">
              <h5 className="fw-bold mb-3">Top Drug Pairs (2-AED Co-prescriptions)</h5>
              <div className="table-responsive">
                <table className="table table-hover">
                  <thead className="table-dark">
                    <tr>
                      <th>#</th>
                      <th>Combination</th>
                      <th>Patients</th>
                      <th>Avg Adherence</th>
                      <th>Avg Side Effect Severity</th>
                      <th>Avg Seizure Rate</th>
                    </tr>
                  </thead>
                  <tbody>
                    {pairs.map((p, i) => (
                      <tr key={p.combination}>
                        <td className="text-muted">{i + 1}</td>
                        <td className="fw-semibold">{p.combination}</td>
                        <td><span className="badge bg-primary">{p.patient_count}</span></td>
                        <td>{ADH_BADGE(p.avg_adherence_pct)}</td>
                        <td>{SEVERITY_BADGE(p.avg_side_effect_severity)}</td>
                        <td>
                          {p.avg_seizure_rate != null
                            ? <span className="badge bg-info text-dark">{p.avg_seizure_rate}%</span>
                            : <span className="badge bg-secondary">N/A</span>
                          }
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="card border-0 shadow-sm">
            <div className="card-body">
              <h5 className="fw-bold mb-3">Triple Therapy Regimens (3-AED Combinations)</h5>
              <div className="table-responsive">
                <table className="table table-hover">
                  <thead className="table-dark">
                    <tr>
                      <th>#</th>
                      <th>Regimen</th>
                      <th>Patients</th>
                      <th>Avg Adherence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {triples.map((t, i) => (
                      <tr key={t.combination}>
                        <td className="text-muted">{i + 1}</td>
                        <td className="fw-semibold">{t.combination}</td>
                        <td><span className="badge bg-danger">{t.patient_count}</span></td>
                        <td>{ADH_BADGE(t.avg_adherence_pct)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="text-muted small mt-2 mb-0">
                Triple regimens represent higher-complexity polytherapy. Rational selection considers complementary mechanisms
                (sodium-channel blockers + GABAergic + SV2A modulators) per ILAE 2021 pharmacotherapy guidelines.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* TAB 3: Co-occurrence Matrix */}
      {activeTab === 'matrix' && (
        <div className="card border-0 shadow-sm">
          <div className="card-body">
            <h5 className="fw-bold mb-1">Drug Pair Co-occurrence Matrix</h5>
            <p className="text-muted small mb-3">
              Number of patients concurrently prescribed each pair of AEDs. Diagonal = N/A (same drug). Higher values indicate more common co-prescriptions.
            </p>
            {drugs.length > 0 ? (
              <div className="table-responsive">
                <table className="table table-bordered table-sm" style={{ fontSize: '0.82rem' }}>
                  <thead>
                    <tr>
                      <th className="table-dark" style={{ minWidth: '120px' }}>Drug</th>
                      {drugs.map(d => (
                        <th key={d} className="table-dark text-center" style={{ minWidth: '90px', fontSize: '0.75rem' }}>
                          {d.length > 10 ? d.substring(0, 10) + '…' : d}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {drugs.map((rowDrug, ri) => (
                      <tr key={rowDrug}>
                        <td className="fw-semibold table-light">{rowDrug}</td>
                        {drugs.map((colDrug, ci) => {
                          const val = matData[rowDrug]?.[ci] ?? (matData[rowDrug] ? matData[rowDrug][ci] : 0);
                          const v = Array.isArray(matData[rowDrug]) ? matData[rowDrug][ci] : (matData[rowDrug]?.[colDrug] ?? 0);
                          if (ri === ci) {
                            return <td key={colDrug} className="table-secondary text-center text-muted">&mdash;</td>;
                          }
                          const maxVal = 5;
                          const intensity = Math.min(v / maxVal, 1);
                          const bg = v === 0
                            ? '#f8f9fa'
                            : `rgba(11, 31, 58, ${0.1 + intensity * 0.7})`;
                          const color = intensity > 0.5 ? '#fff' : '#0b1f3a';
                          return (
                            <td
                              key={colDrug}
                              className="text-center fw-semibold"
                              style={{ background: bg, color }}
                            >
                              {v}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="alert alert-warning">No matrix data available.</div>
            )}
            <div className="mt-3 d-flex align-items-center gap-2">
              <span className="text-muted small">Co-occurrence intensity:</span>
              {[0, 1, 2, 3, 4, 5].map(v => {
                const intensity = Math.min(v / 5, 1);
                const bg = v === 0 ? '#f8f9fa' : `rgba(11, 31, 58, ${0.1 + intensity * 0.7})`;
                const color = intensity > 0.5 ? '#fff' : '#0b1f3a';
                return (
                  <span
                    key={v}
                    style={{ background: bg, color, padding: '2px 8px', borderRadius: '4px', fontSize: '0.8rem', border: '1px solid #dee2e6' }}
                  >
                    {v}
                  </span>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* TAB 4: Per Patient */}
      {activeTab === 'patients' && (
        <div className="card border-0 shadow-sm">
          <div className="card-body">
            <div className="d-flex justify-content-between align-items-center mb-3">
              <h5 className="fw-bold mb-0">Per-Patient Polypharmacy Profiles</h5>
              <input
                className="form-control form-control-sm w-auto"
                placeholder="Search patient or regimen..."
                value={patientSearch}
                onChange={e => setPatientSearch(e.target.value)}
                style={{ minWidth: '220px' }}
              />
            </div>
            <div className="table-responsive">
              <table className="table table-hover table-sm">
                <thead className="table-dark">
                  <tr>
                    <SortTh field="patient_id" label="Patient ID" />
                    <SortTh field="therapy_type" label="Therapy Type" />
                    <th>Regimen</th>
                    <SortTh field="drug_count" label="# AEDs" />
                    <SortTh field="adherence_pct" label="Adherence" />
                    <SortTh field="avg_side_effect_severity" label="Side Effect Sev." />
                    <SortTh field="seizure_rate" label="Seizure Rate" />
                  </tr>
                </thead>
                <tbody>
                  {sortedPatients().map(p => (
                    <tr key={p.patient_id}>
                      <td className="fw-semibold">{p.patient_id}</td>
                      <td>
                        <span className={`badge ${p.drug_count >= 3 ? 'bg-danger' : 'bg-primary'}`}>
                          {p.therapy_type}
                        </span>
                      </td>
                      <td className="small">{p.regimen_label}</td>
                      <td className="text-center">{p.drug_count}</td>
                      <td>{ADH_BADGE(p.adherence_pct)}</td>
                      <td className="text-center">{SEVERITY_BADGE(p.avg_side_effect_severity)}</td>
                      <td className="text-center">
                        {p.seizure_rate != null
                          ? <span className="badge bg-info text-dark">{p.seizure_rate}%</span>
                          : <span className="badge bg-secondary">N/A</span>
                        }
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="text-muted small mt-2 mb-0">
              {sortedPatients().length} patient(s) shown. Click column headers to sort. Seizure rate N/A = no seizure log overlap.
            </p>
          </div>
        </div>
      )}

      {/* TAB 5: Definitions */}
      {activeTab === 'definitions' && (
        <div>
          <div className="card border-0 shadow-sm mb-3">
            <div className="card-body">
              <h5 className="fw-bold mb-1">{definitions?.dashboard}</h5>
              <p className="text-muted mb-0">{definitions?.scope}</p>
            </div>
          </div>
          <div className="row g-3 mb-4">
            {(definitions?.definitions || []).map(d => (
              <div className="col-md-6" key={d.term}>
                <div className="card border-0 shadow-sm h-100">
                  <div className="card-body">
                    <h6 className="fw-bold text-primary mb-2">{d.term}</h6>
                    <p className="small mb-2">{d.definition}</p>
                    <p className="text-muted" style={{ fontSize: '0.75rem' }}>
                      <em>{d.reference}</em>
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
          <div className="card border-0 shadow-sm">
            <div className="card-body">
              <h6 className="fw-bold mb-2">All References</h6>
              <ol className="mb-0 small text-muted">
                {(definitions?.references || []).map((r, i) => (
                  <li key={i}>{r}</li>
                ))}
              </ol>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
