'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const RESP_COLORS = {
  'Remission': '#22c55e',
  'Partial Response': '#f59e0b',
  'Refractory': '#ef4444',
  'Relapsed': '#8b5cf6',
};

const AB_COLORS = {
  'NMDA-R': '#6366f1',
  'LGI1': '#3b82f6',
  'CASPR2': '#10b981',
  'GABA-B': '#f59e0b',
  'AMPA-R': '#ef4444',
  'VGKC': '#8b5cf6',
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

function Badge({ label, colorMap }) {
  const bg = (colorMap && colorMap[label]) || '#6b7280';
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 4,
      fontSize: '0.75rem', fontWeight: 600, color: '#fff', backgroundColor: bg,
    }}>{label}</span>
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

export default function AutoimmuneEpilepsyDashboard() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);
  const [patSort, setPatSort] = useState('patient_id');
  const [patDir, setPatDir] = useState(1);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/autoimmune-epilepsy/overview`).then(r => r.json()),
      fetch(`${API}/api/autoimmune-epilepsy/breakdown`).then(r => r.json()),
      fetch(`${API}/api/autoimmune-epilepsy/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov) return <div className="text-muted p-3">Loading autoimmune epilepsy data…</div>;

  const kpi = ov.kpis || {};
  const totalPts = kpi.total_patients || 0;

  const TABS = [
    { id: 'overview', label: 'Overview' },
    { id: 'antibodies', label: 'Antibody Profiles' },
    { id: 'patients', label: 'Patient Table' },
    { id: 'outcomes', label: 'Outcomes & Relapse' },
    { id: 'definitions', label: 'Definitions' },
  ];

  return (
    <div className="container-fluid py-3">
      <h4 className="fw-bold mb-1">🧬 Autoimmune Epilepsy Dashboard</h4>
      <p className="text-muted small mb-3">
        Antibody-mediated epilepsies (NMDA-R · LGI1 · CASPR2 · GABA-B · AMPA-R · VGKC) —
        immunotherapy monitoring, treatment response &amp; outcome tracking · {totalPts} patients
      </p>

      {/* KPI Row */}
      <div className="row g-2 mb-3">
        <StatCard label="Total Patients" value={kpi.total_patients} color="#6366f1" />
        <StatCard label="Antibody Positive" value={kpi.antibody_positive} color="#3b82f6" />
        <StatCard label="On Immunotherapy" value={kpi.receiving_immunotherapy} color="#10b981" />
        <StatCard
          label="Remission Rate"
          value={`${kpi.remission_rate_pct}%`}
          sub={`${kpi.remission_count} patients`}
          color="#22c55e"
        />
        <StatCard
          label="CSF Positivity"
          value={`${kpi.csf_positivity_rate_pct}%`}
          sub={`${kpi.csf_positive} / ${totalPts}`}
          color="#f59e0b"
        />
        <StatCard label="MRI Abnormal" value={kpi.mri_abnormal} color="#ef4444" />
        <StatCard label="Paraneoplastic" value={kpi.paraneoplastic_cases} color="#8b5cf6" />
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
          {/* Antibody Distribution */}
          <div className="col-md-6">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Antibody Distribution</div>
              <div className="card-body">
                {(ov.antibody_distribution || []).map(ab => (
                  <div key={ab.antibody} className="mb-2">
                    <div className="d-flex justify-content-between small mb-1">
                      <span>
                        <Badge label={ab.antibody} colorMap={AB_COLORS} />
                      </span>
                      <span className="fw-bold">{ab.count} ({ab.pct}%)</span>
                    </div>
                    <MiniBar value={ab.count} total={totalPts} color={AB_COLORS[ab.antibody] || '#6366f1'} />
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Treatment Response */}
          <div className="col-md-6">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Treatment Response</div>
              <div className="card-body">
                {(ov.treatment_response || []).map(r => (
                  <div key={r.response} className="mb-2">
                    <div className="d-flex justify-content-between small mb-1">
                      <span><Badge label={r.response} colorMap={RESP_COLORS} /></span>
                      <span className="fw-bold">{r.count} ({r.pct}%)</span>
                    </div>
                    <MiniBar value={r.count} total={totalPts} color={RESP_COLORS[r.response]} />
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Top Immunotherapy Drugs */}
          <div className="col-12">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Immunotherapy Agents in Use</div>
              <div className="card-body">
                <div className="row">
                  {(ov.top_immunotherapy_drugs || []).map(d => (
                    <div key={d.drug} className="col-md-4 mb-2">
                      <div className="d-flex justify-content-between align-items-center p-2 border rounded">
                        <div>
                          <div className="fw-semibold small">{d.drug}</div>
                          <div className="text-muted" style={{ fontSize: '0.72rem' }}>{d.class}</div>
                        </div>
                        <span className="badge bg-primary">{d.count} pts</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ANTIBODY PROFILES TAB */}
      {tab === 'antibodies' && bd && (
        <div className="row g-3">
          <div className="col-12">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Antibody × Treatment Response Matrix</div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm table-hover mb-0">
                    <thead className="table-light">
                      <tr>
                        <th>Antibody</th>
                        <th>Total</th>
                        <th style={{ color: RESP_COLORS['Remission'] }}>Remission</th>
                        <th style={{ color: RESP_COLORS['Partial Response'] }}>Partial</th>
                        <th style={{ color: RESP_COLORS['Refractory'] }}>Refractory</th>
                        <th style={{ color: RESP_COLORS['Relapsed'] }}>Relapsed</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(bd.antibody_response_matrix || []).map(row => {
                        const total = (row.Remission || 0) + (row['Partial Response'] || 0) +
                          (row.Refractory || 0) + (row.Relapsed || 0);
                        return (
                          <tr key={row.antibody}>
                            <td><Badge label={row.antibody} colorMap={AB_COLORS} /></td>
                            <td>{total}</td>
                            <td className="fw-bold" style={{ color: RESP_COLORS['Remission'] }}>{row.Remission || 0}</td>
                            <td className="fw-bold" style={{ color: RESP_COLORS['Partial Response'] }}>{row['Partial Response'] || 0}</td>
                            <td className="fw-bold" style={{ color: RESP_COLORS['Refractory'] }}>{row.Refractory || 0}</td>
                            <td className="fw-bold" style={{ color: RESP_COLORS['Relapsed'] }}>{row.Relapsed || 0}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          <div className="col-md-6">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Time to Diagnosis</div>
              <div className="card-body">
                {(bd.time_to_diagnosis || []).map(b => (
                  <div key={b.bucket} className="mb-2">
                    <div className="d-flex justify-content-between small mb-1">
                      <span>{b.bucket}</span>
                      <span className="fw-bold">{b.count}</span>
                    </div>
                    <MiniBar value={b.count} total={totalPts} color="#6366f1" />
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="col-md-6">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">MRI Findings Distribution</div>
              <div className="card-body">
                {(bd.mri_findings || []).slice(0, 8).map(m => (
                  <div key={m.finding} className="mb-2">
                    <div className="d-flex justify-content-between small mb-1">
                      <span className="text-truncate" style={{ maxWidth: 200 }}>{m.finding}</span>
                      <span className="fw-bold">{m.count}</span>
                    </div>
                    <MiniBar value={m.count} total={totalPts} color="#10b981" />
                  </div>
                ))}
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
            <div className="card-header fw-semibold">Per-Patient Autoimmune Profile ({rows.length} patients)</div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light">
                    <tr>
                      {th('patient_id', 'Patient')}
                      {th('age', 'Age')}
                      {th('gender', 'Sex')}
                      {th('antibody', 'Antibody')}
                      {th('response', 'Response')}
                      <th>Immunotherapy</th>
                      {th('time_to_diagnosis_weeks', 'Dx Weeks')}
                      {th('cycles_completed', 'Cycles')}
                      <th>Paraneoplastic</th>
                      {th('mri_finding', 'MRI')}
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map(r => (
                      <tr key={r.patient_id}>
                        <td className="fw-semibold">{r.patient_id}</td>
                        <td>{r.age}</td>
                        <td>{r.gender}</td>
                        <td><Badge label={r.antibody} colorMap={AB_COLORS} /></td>
                        <td><Badge label={r.response} colorMap={RESP_COLORS} /></td>
                        <td>
                          <div className="d-flex gap-1 flex-wrap">
                            {(r.immunotherapy || []).map(d => (
                              <span key={d} className="badge bg-secondary">{d}</span>
                            ))}
                          </div>
                        </td>
                        <td>{r.time_to_diagnosis_weeks}w</td>
                        <td>{r.cycles_completed}</td>
                        <td>
                          {r.paraneoplastic
                            ? <span className="badge bg-danger">Yes</span>
                            : <span className="badge bg-secondary">No</span>}
                        </td>
                        <td className="text-truncate" style={{ maxWidth: 160 }}>{r.mri_finding}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        );
      })()}

      {/* OUTCOMES & RELAPSE TAB */}
      {tab === 'outcomes' && bd && (
        <div className="row g-3">
          <div className="col-md-6">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Relapse Rate by Antibody</div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm table-hover mb-0">
                    <thead className="table-light">
                      <tr>
                        <th>Antibody</th>
                        <th>Total</th>
                        <th>Relapsed</th>
                        <th>Relapse Rate</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(bd.relapse_by_antibody || []).map(r => (
                        <tr key={r.antibody}>
                          <td><Badge label={r.antibody} colorMap={AB_COLORS} /></td>
                          <td>{r.total}</td>
                          <td className="fw-bold" style={{ color: '#ef4444' }}>{r.relapsed}</td>
                          <td>
                            <span className="fw-bold" style={{
                              color: r.relapse_rate_pct > 20 ? '#ef4444' : r.relapse_rate_pct > 10 ? '#f59e0b' : '#22c55e'
                            }}>{r.relapse_rate_pct}%</span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          <div className="col-md-6">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Gender Distribution</div>
              <div className="card-body">
                {(bd.gender_distribution || []).map(g => (
                  <div key={g.gender} className="mb-2">
                    <div className="d-flex justify-content-between small mb-1">
                      <span>{g.gender}</span>
                      <span className="fw-bold">{g.count}</span>
                    </div>
                    <MiniBar value={g.count} total={totalPts} color="#6366f1" />
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Response summary cards */}
          <div className="col-12">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Treatment Outcome Summary</div>
              <div className="card-body">
                <div className="row g-2">
                  {(ov.treatment_response || []).map(r => (
                    <div key={r.response} className="col-md-3">
                      <div className="p-3 border rounded text-center" style={{ borderLeft: `4px solid ${RESP_COLORS[r.response]}` }}>
                        <div className="h5 fw-bold mb-0" style={{ color: RESP_COLORS[r.response] }}>{r.pct}%</div>
                        <div className="fw-semibold small">{r.response}</div>
                        <div className="text-muted" style={{ fontSize: '0.72rem' }}>{r.count} of {totalPts} patients</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'definitions' && defs && (
        <div className="row g-3">
          <div className="col-12">
            <div className="alert alert-info small mb-3">{defs.dashboard_purpose}</div>
          </div>

          {/* Antibody Syndrome Cards */}
          <div className="col-12">
            <h6 className="fw-bold">Antibody Syndromes</h6>
            <div className="row g-2">
              {(defs.antibody_syndromes || []).map(ab => (
                <div key={ab.antibody} className="col-md-4">
                  <div className="card shadow-sm h-100" style={{ borderTop: `3px solid ${AB_COLORS[ab.antibody] || '#6366f1'}` }}>
                    <div className="card-body p-2">
                      <div className="fw-bold" style={{ color: AB_COLORS[ab.antibody] || '#6366f1' }}>{ab.antibody}</div>
                      <div className="text-muted small mb-1">{ab.syndrome}</div>
                      <div className="small"><b>Target:</b> {ab.target}</div>
                      <div className="small"><b>Prevalence:</b> ~{ab.prevalence_pct}%</div>
                      <div className="small"><b>Age:</b> {ab.typical_age}</div>
                      <div className="small"><b>CSF+:</b> {ab.csf_positive_pct}%</div>
                      <div className="small"><b>Response:</b> {ab.immunotherapy_response}</div>
                      <div className="small"><b>Relapse:</b> {ab.relapse_risk}</div>
                      <div className="small mt-1"><b>1st line:</b> {ab.first_line_treatment}</div>
                      <div className="small"><b>2nd line:</b> {ab.second_line_treatment}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Immunotherapy agents */}
          <div className="col-md-6">
            <h6 className="fw-bold">Immunotherapy Agents</h6>
            <div className="table-responsive">
              <table className="table table-sm table-hover">
                <thead className="table-light">
                  <tr><th>Drug</th><th>Class</th><th>Mechanism</th></tr>
                </thead>
                <tbody>
                  {(defs.immunotherapy_agents || []).map(d => (
                    <tr key={d.drug}>
                      <td className="fw-semibold">{d.drug}</td>
                      <td><span className="badge bg-info text-dark">{d.class}</span></td>
                      <td className="small text-muted">{d.mechanism}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Response definitions */}
          <div className="col-md-6">
            <h6 className="fw-bold">Response Definitions</h6>
            {(defs.response_definitions || []).map(r => (
              <div key={r.response} className="mb-2 p-2 border rounded" style={{ borderLeft: `3px solid ${RESP_COLORS[r.response]}` }}>
                <div className="fw-semibold small" style={{ color: RESP_COLORS[r.response] }}>{r.response}</div>
                <div className="text-muted small">{r.definition}</div>
              </div>
            ))}
          </div>

          {/* Diagnostic criteria */}
          <div className="col-md-6">
            <h6 className="fw-bold">Diagnostic Criteria</h6>
            <ul className="list-unstyled">
              {(defs.diagnostic_criteria || []).map((c, i) => (
                <li key={i} className="small mb-1">
                  <span className="me-2">•</span>{c}
                </li>
              ))}
            </ul>
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
