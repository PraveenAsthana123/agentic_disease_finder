'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const SYN_COLORS = {
  'CAE': '#22c55e',
  'JME': '#6366f1',
  'JAE': '#3b82f6',
  'LGS': '#ef4444',
  'TLE-lat': '#f59e0b',
  'TLE-mes': '#f97316',
  'FLE': '#8b5cf6',
  'UFE': '#6b7280',
  'GTCS-only': '#10b981',
  'UGE': '#64748b',
};

const DR_COLORS = {
  true: '#ef4444',
  false: '#22c55e',
};

function StatCard({ label, value, sub, color = '#6366f1' }) {
  return (
    <div className="col-6 col-md mb-3">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center py-2">
          <div className="h4 mb-0 fw-bold" style={{ color }}>{value}</div>
          <div className="text-muted small">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function MiniBar({ value, total, color }) {
  const pct = total ? Math.min(100, (value / total) * 100) : 0;
  return (
    <div className="d-flex align-items-center gap-2">
      <div className="progress flex-grow-1" style={{ height: 10, minWidth: 80 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, backgroundColor: color || '#6366f1' }} />
      </div>
      <span className="small fw-bold">{value}</span>
    </div>
  );
}

function AbbrBadge({ abbrev }) {
  const bg = SYN_COLORS[abbrev] || '#6b7280';
  return (
    <span style={{
      display: 'inline-block', padding: '2px 7px', borderRadius: 4,
      fontSize: '0.72rem', fontWeight: 700, color: '#fff', backgroundColor: bg,
    }}>{abbrev}</span>
  );
}

export default function PediatricEpilepsyDashboard() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);
  const [patSort, setPatSort] = useState('age_at_onset');
  const [patDir, setPatDir] = useState(1);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/pediatric-epilepsy/overview`).then(r => r.json()),
      fetch(`${API}/api/pediatric-epilepsy/breakdown`).then(r => r.json()),
      fetch(`${API}/api/pediatric-epilepsy/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov) return <div className="text-muted p-3">Loading pediatric epilepsy data…</div>;

  const kpi = ov.kpis || {};
  const n = kpi.total_pediatric || 0;

  const TABS = [
    { id: 'overview', label: 'Overview' },
    { id: 'syndromes', label: 'Syndrome Profiles' },
    { id: 'patients', label: 'Patient Table' },
    { id: 'outcomes', label: 'Outcomes' },
    { id: 'definitions', label: 'Definitions' },
  ];

  return (
    <div className="container-fluid py-3">
      <h4 className="fw-bold mb-1">🧒 Pediatric Epilepsy Syndromes Dashboard</h4>
      <p className="text-muted small mb-3">
        Childhood-onset epilepsy (onset &lt; 18 years) — CAE · JME · JAE · LGS · Focal epilepsies ·
        ILAE 2022 classification · {n} patients with pediatric onset
      </p>

      {/* KPI Row */}
      <div className="row g-2 mb-3">
        <StatCard label="Pediatric-Onset" value={kpi.total_pediatric}
          sub={`of ${kpi.total_cohort} total (${kpi.pediatric_fraction_pct}%)`} color="#6366f1" />
        <StatCard label="Mean Onset Age" value={`${kpi.mean_onset_age_years}y`} color="#3b82f6" />
        <StatCard label="Drug Resistant" value={kpi.drug_resistant_count}
          sub={`${kpi.drug_resistant_pct}% of pediatric`} color="#ef4444" />
        <StatCard label="Generalized" value={kpi.generalized_count}
          sub="(CAE/JME/JAE/LGS)" color="#10b981" />
        <StatCard label="Focal" value={kpi.focal_count}
          sub="(TLE/FLE/focal unclassified)" color="#f59e0b" />
      </div>

      {/* Nav Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={`nav-link ${tab === t.id ? 'active' : ''}`}
              onClick={() => setTab(t.id)}
            >{t.label}</button>
          </li>
        ))}
      </ul>

      {/* OVERVIEW TAB */}
      {tab === 'overview' && (
        <div className="row g-3">
          {/* Syndrome Distribution */}
          <div className="col-md-6">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Syndrome Distribution</div>
              <div className="card-body">
                {(ov.syndrome_distribution || []).map(s => (
                  <div key={s.syndrome} className="mb-2">
                    <div className="d-flex justify-content-between small mb-1">
                      <span>
                        <AbbrBadge abbrev={s.abbrev} />
                        <span className="ms-2 text-truncate" style={{ maxWidth: 220, display: 'inline-block', verticalAlign: 'middle' }}>
                          {s.syndrome}
                        </span>
                      </span>
                      <span className="fw-bold">{s.count} ({s.pct}%)</span>
                    </div>
                    <MiniBar value={s.count} total={n} color={SYN_COLORS[s.abbrev] || '#6366f1'} />
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Onset Age Distribution */}
          <div className="col-md-6">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Onset Age Distribution</div>
              <div className="card-body">
                {(ov.onset_age_buckets || []).map(b => (
                  <div key={b.bucket} className="mb-2">
                    <div className="d-flex justify-content-between small mb-1">
                      <span>{b.bucket}</span>
                      <span className="fw-bold">{b.count}</span>
                    </div>
                    <MiniBar value={b.count} total={n} color="#6366f1" />
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Etiology */}
          <div className="col-md-6">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Etiology Distribution</div>
              <div className="card-body">
                {(ov.etiology_distribution || []).slice(0, 6).map(e => (
                  <div key={e.etiology} className="mb-2">
                    <div className="d-flex justify-content-between small mb-1">
                      <span className="text-truncate" style={{ maxWidth: 250 }}>{e.etiology}</span>
                      <span className="fw-bold">{e.count} ({e.pct}%)</span>
                    </div>
                    <MiniBar value={e.count} total={n} color="#10b981" />
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Gender + Generalized vs Focal */}
          <div className="col-md-6">
            <div className="card shadow-sm mb-3">
              <div className="card-header fw-semibold">Gender Breakdown</div>
              <div className="card-body">
                {(ov.gender_distribution || []).map(g => (
                  <div key={g.gender} className="mb-2">
                    <div className="d-flex justify-content-between small mb-1">
                      <span>{g.gender}</span>
                      <span className="fw-bold">{g.count}</span>
                    </div>
                    <MiniBar value={g.count} total={n} color="#8b5cf6" />
                  </div>
                ))}
              </div>
            </div>
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Generalized vs Focal</div>
              <div className="card-body">
                <div className="d-flex justify-content-between small mb-1">
                  <span>Generalized (CAE/JME/JAE/LGS)</span>
                  <span className="fw-bold">{kpi.generalized_count}</span>
                </div>
                <MiniBar value={kpi.generalized_count} total={n} color="#10b981" />
                <div className="d-flex justify-content-between small mb-1 mt-2">
                  <span>Focal (TLE/FLE/unclassified)</span>
                  <span className="fw-bold">{kpi.focal_count}</span>
                </div>
                <MiniBar value={kpi.focal_count} total={n} color="#f59e0b" />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* SYNDROME PROFILES TAB */}
      {tab === 'syndromes' && bd && (
        <div className="row g-3">
          <div className="col-12">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Syndrome × Drug-Resistance Summary</div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm table-hover mb-0">
                    <thead className="table-light">
                      <tr>
                        <th>Syndrome</th>
                        <th>Patients</th>
                        <th style={{ color: '#ef4444' }}>Drug-Resistant</th>
                        <th style={{ color: '#ef4444' }}>DR Rate</th>
                        <th style={{ color: '#22c55e' }}>Seizure-Free</th>
                        <th style={{ color: '#22c55e' }}>SF Rate</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(bd.syndrome_summary || []).map(s => (
                        <tr key={s.syndrome}>
                          <td className="small">{s.syndrome}</td>
                          <td>{s.total}</td>
                          <td className="fw-bold" style={{ color: '#ef4444' }}>{s.drug_resistant}</td>
                          <td>
                            <span className="fw-bold" style={{
                              color: s.dr_rate_pct > 40 ? '#ef4444' : s.dr_rate_pct > 20 ? '#f59e0b' : '#22c55e'
                            }}>{s.dr_rate_pct}%</span>
                          </td>
                          <td className="fw-bold" style={{ color: '#22c55e' }}>{s.seizure_free}</td>
                          <td>
                            <span className="fw-bold" style={{
                              color: s.sf_rate_pct > 50 ? '#22c55e' : s.sf_rate_pct > 30 ? '#f59e0b' : '#ef4444'
                            }}>{s.sf_rate_pct}%</span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          {/* Top AEDs */}
          <div className="col-md-6">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Top AEDs in Pediatric Cohort</div>
              <div className="card-body">
                {(bd.top_aeds || []).map(a => (
                  <div key={a.aed} className="mb-2">
                    <div className="d-flex justify-content-between small mb-1">
                      <span className="fw-semibold">{a.aed}</span>
                      <span className="fw-bold">{a.count} patients</span>
                    </div>
                    <MiniBar value={a.count} total={n} color="#6366f1" />
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Onset × Syndrome Heatmap */}
          <div className="col-md-6">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Onset Age × Syndrome</div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm mb-0">
                    <thead className="table-light">
                      <tr>
                        <th>Onset Bucket</th>
                        <th>Syndrome</th>
                        <th>Count</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(bd.onset_syndrome_heatmap || []).slice(0, 12).map((row, i) => (
                        <tr key={i}>
                          <td><span className="badge bg-secondary">{row.onset_bucket}y</span></td>
                          <td className="small text-truncate" style={{ maxWidth: 200 }}>{row.syndrome}</td>
                          <td className="fw-bold">{row.count}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* PATIENT TABLE TAB */}
      {tab === 'patients' && bd && (() => {
        const rows = [...(bd.per_patient || [])].sort((a, b) => {
          let va = a[patSort], vb = b[patSort];
          if (typeof va === 'string') va = va.toLowerCase();
          if (typeof vb === 'string') vb = vb.toLowerCase();
          return va < vb ? -patDir : va > vb ? patDir : 0;
        });
        const th = (col, label) => (
          <th
            key={col}
            style={{ cursor: 'pointer', userSelect: 'none' }}
            onClick={() => { if (patSort === col) setPatDir(-patDir); else { setPatSort(col); setPatDir(1); } }}
          >
            {label} {patSort === col ? (patDir === 1 ? '▲' : '▼') : ''}
          </th>
        );
        return (
          <div className="card shadow-sm">
            <div className="card-header fw-semibold">
              Pediatric Patient Table — {rows.length} patients (onset &lt; 18 years)
            </div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light">
                    <tr>
                      {th('patient_id', 'Patient')}
                      {th('age', 'Age')}
                      {th('gender', 'Sex')}
                      {th('age_at_onset', 'Onset Age')}
                      {th('syndrome_abbrev', 'Syndrome')}
                      {th('drug_responsiveness', 'Drug Response')}
                      {th('current_seizure_frequency', 'Seizure Freq')}
                      <th>Seizure-Free</th>
                      <th>AEDs</th>
                      <th>MRI</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map(r => (
                      <tr key={r.patient_id}>
                        <td className="fw-semibold">{r.patient_id}</td>
                        <td>{r.age}</td>
                        <td>{r.gender}</td>
                        <td>
                          <span className="badge bg-primary">{r.age_at_onset}y</span>
                        </td>
                        <td>
                          <AbbrBadge abbrev={r.syndrome_abbrev} />
                          <span className="ms-1 small text-muted">{r.syndrome.length > 30 ? r.syndrome.slice(0, 28) + '…' : r.syndrome}</span>
                        </td>
                        <td>
                          <span className="badge" style={{
                            backgroundColor: (r.drug_responsiveness || '').toLowerCase().includes('drug-resistant') ? '#ef4444' : '#22c55e',
                            color: '#fff',
                          }}>
                            {(r.drug_responsiveness || 'Unknown').replace('Drug-resistant (failed ≥2 AEDs)', 'Drug-resistant')}
                          </span>
                        </td>
                        <td className="small">{r.current_seizure_frequency}</td>
                        <td>
                          {r.seizure_free
                            ? <span className="badge bg-success">Yes</span>
                            : <span className="badge bg-secondary">No</span>}
                        </td>
                        <td>
                          <div className="d-flex gap-1 flex-wrap">
                            {(r.aeds || []).slice(0, 2).map(a => (
                              <span key={a} className="badge bg-info text-dark" style={{ fontSize: '0.68rem' }}>{a}</span>
                            ))}
                            {(r.aeds || []).length > 2 && (
                              <span className="badge bg-secondary" style={{ fontSize: '0.68rem' }}>+{r.aeds.length - 2}</span>
                            )}
                          </div>
                        </td>
                        <td className="small text-truncate" style={{ maxWidth: 140 }}>{r.mri_finding}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        );
      })()}

      {/* OUTCOMES TAB */}
      {tab === 'outcomes' && bd && (
        <div className="row g-3">
          {/* Seizure-free summary */}
          <div className="col-12">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Seizure Freedom & Drug-Resistance by Syndrome</div>
              <div className="card-body">
                <div className="row g-2">
                  {(bd.syndrome_summary || []).map(s => {
                    const drColor = s.dr_rate_pct > 40 ? '#ef4444' : s.dr_rate_pct > 20 ? '#f59e0b' : '#22c55e';
                    const sfColor = s.sf_rate_pct > 50 ? '#22c55e' : s.sf_rate_pct > 30 ? '#f59e0b' : '#ef4444';
                    return (
                      <div key={s.syndrome} className="col-md-4">
                        <div className="p-2 border rounded" style={{ borderTop: `3px solid ${drColor}` }}>
                          <div className="fw-semibold small mb-1 text-truncate">{s.syndrome}</div>
                          <div className="d-flex justify-content-between small">
                            <span style={{ color: '#ef4444' }}>DR: {s.dr_rate_pct}%</span>
                            <span style={{ color: sfColor }}>SF: {s.sf_rate_pct}%</span>
                            <span className="text-muted">n={s.total}</span>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>

          {/* Cognitive impact */}
          <div className="col-md-6">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Cognitive Impact Risk</div>
              <div className="card-body">
                {(() => {
                  const withCog = (bd.per_patient || []).filter(p => p.cognitive_impact).length;
                  const without = n - withCog;
                  return (
                    <>
                      <div className="mb-2">
                        <div className="d-flex justify-content-between small mb-1">
                          <span>Cognitive impact risk</span>
                          <span className="fw-bold text-warning">{withCog} ({Math.round(100 * withCog / n)}%)</span>
                        </div>
                        <MiniBar value={withCog} total={n} color="#f59e0b" />
                      </div>
                      <div className="mb-2">
                        <div className="d-flex justify-content-between small mb-1">
                          <span>No cognitive impact</span>
                          <span className="fw-bold text-success">{without} ({Math.round(100 * without / n)}%)</span>
                        </div>
                        <MiniBar value={without} total={n} color="#22c55e" />
                      </div>
                      <p className="text-muted" style={{ fontSize: '0.72rem' }} className="mt-2">
                        Cognitive impact proxy: high-risk for LGS (90%), moderate for generalized (25–50%), lower for CAE/JAE (20%).
                      </p>
                    </>
                  );
                })()}
              </div>
            </div>
          </div>

          {/* Seizure freedom overall */}
          <div className="col-md-6">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Seizure Freedom Overview</div>
              <div className="card-body">
                {(() => {
                  const sf = (bd.per_patient || []).filter(p => p.seizure_free).length;
                  const notSf = n - sf;
                  return (
                    <>
                      <div className="mb-2">
                        <div className="d-flex justify-content-between small mb-1">
                          <span>Seizure-free</span>
                          <span className="fw-bold text-success">{sf} ({Math.round(100 * sf / n)}%)</span>
                        </div>
                        <MiniBar value={sf} total={n} color="#22c55e" />
                      </div>
                      <div className="mb-2">
                        <div className="d-flex justify-content-between small mb-1">
                          <span>Active seizures</span>
                          <span className="fw-bold text-danger">{notSf} ({Math.round(100 * notSf / n)}%)</span>
                        </div>
                        <MiniBar value={notSf} total={n} color="#ef4444" />
                      </div>
                    </>
                  );
                })()}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'definitions' && defs && (
        <div className="row g-3">
          <div className="col-12">
            <div className="alert alert-info small mb-0">{defs.dashboard_purpose}</div>
          </div>

          {/* Syndrome profile cards */}
          <div className="col-12">
            <h6 className="fw-bold">ILAE Pediatric Syndrome Profiles</h6>
            <div className="row g-2">
              {(defs.syndrome_profiles || []).map(sp => {
                const bg = SYN_COLORS[sp.abbrev] || '#6366f1';
                return (
                  <div key={sp.syndrome} className="col-md-4">
                    <div className="card shadow-sm h-100" style={{ borderTop: `3px solid ${bg}` }}>
                      <div className="card-body p-2">
                        <div className="d-flex align-items-center gap-2 mb-1">
                          <AbbrBadge abbrev={sp.abbrev} />
                          <span className="fw-semibold small">{sp.syndrome}</span>
                        </div>
                        <div className="small"><b>Onset:</b> {sp.onset_range} (peak {sp.peak_onset_years}y)</div>
                        <div className="small"><b>EEG:</b> {sp.eeg_signature}</div>
                        <div className="small"><b>Seizure type:</b> {sp.seizure_type}</div>
                        <div className="small"><b>1st-line AED:</b> {sp.first_line_aed}</div>
                        <div className="small"><b>Genetics:</b> {sp.genetics}</div>
                        <div className="small mt-1" style={{ color: sp.remission_rate_pct > 50 ? '#22c55e' : sp.remission_rate_pct > 30 ? '#f59e0b' : '#ef4444' }}>
                          <b>Remission:</b> {sp.remission_rate_pct}%
                        </div>
                        <div className="small text-muted mt-1">{sp.clinical_note}</div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* AED Reference */}
          <div className="col-md-6">
            <h6 className="fw-bold">Pediatric AED Reference</h6>
            <div className="table-responsive">
              <table className="table table-sm table-hover">
                <thead className="table-light">
                  <tr><th>Drug</th><th>Target</th><th>Best For</th><th>FDA Ped.</th></tr>
                </thead>
                <tbody>
                  {(defs.aed_reference || []).map(a => (
                    <tr key={a.drug}>
                      <td className="fw-semibold">{a.drug}</td>
                      <td className="small text-muted">{a.target}</td>
                      <td className="small">{a.best_for}</td>
                      <td>
                        {a.fda_pediatric
                          ? <span className="badge bg-success">Yes</span>
                          : <span className="badge bg-secondary">No</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* ILAE classification notes */}
          <div className="col-md-6">
            <h6 className="fw-bold">ILAE 2022 Classification Notes</h6>
            <ul className="list-unstyled">
              {(defs.ilae_classification_notes || []).map((note, i) => (
                <li key={i} className="small mb-2">
                  <span className="me-2 fw-bold text-primary">•</span>{note}
                </li>
              ))}
            </ul>

            <h6 className="fw-bold mt-3">Outcome Definitions</h6>
            {(defs.outcome_definitions || []).map(o => (
              <div key={o.term} className="mb-2 p-2 border rounded">
                <div className="fw-semibold small">{o.term}</div>
                <div className="text-muted small">{o.definition}</div>
              </div>
            ))}
          </div>

          {/* Data sources */}
          <div className="col-md-6">
            <h6 className="fw-bold">Data Sources</h6>
            <div className="table-responsive">
              <table className="table table-sm">
                <thead className="table-light">
                  <tr><th>Table</th><th>Rows</th><th>Use</th></tr>
                </thead>
                <tbody>
                  {(defs.data_sources || []).map(s => (
                    <tr key={s.table}>
                      <td><code>{s.table}</code></td>
                      <td>{s.rows}</td>
                      <td className="small text-muted">{s.use}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* References */}
          <div className="col-12">
            <h6 className="fw-bold">Clinical References</h6>
            <ul className="list-unstyled">
              {(defs.clinical_references || []).map((ref, i) => (
                <li key={i} className="small text-muted mb-1">
                  <span className="me-2 fw-bold text-dark">[{i + 1}]</span>{ref}
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}
