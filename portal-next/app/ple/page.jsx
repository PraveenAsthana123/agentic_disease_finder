'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizure Types & Surgery', 'Treatments', 'Definitions'];

function KPI({ label, value, color }) {
  return (
    <div className="col-6 col-md-4 col-lg-2 mb-3">
      <div className="card h-100 shadow-sm text-center">
        <div className="card-body py-2 px-1">
          <div className="fw-bold fs-5" style={{ color }}>{value}</div>
          <div className="text-muted small">{label}</div>
        </div>
      </div>
    </div>
  );
}

function PctBar({ label, pct, color = '#1d4ed8' }) {
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="text-muted">{pct}%</span>
      </div>
      <div className="progress" style={{ height: 10 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function Expander({ title, children, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border rounded mb-2">
      <button className="btn btn-light w-100 text-start fw-semibold py-2 px-3"
        onClick={() => setOpen(o => !o)}>
        {open ? '▾' : '▸'} {title}
      </button>
      {open && <div className="px-3 pb-3 small">{children}</div>}
    </div>
  );
}

// ─── Overview Tab ────────────────────────────────────────────────────────────
function OverviewTab({ overview }) {
  if (!overview) return <div className="text-center py-4 text-muted">Loading overview…</div>;
  const szDist = overview.seizure_type_distribution || {};
  const maxSz = Math.max(...Object.values(szDist), 1);

  const kpis = [
    { label: 'Total Patients', value: overview.total_patients, color: '#1d4ed8' },
    { label: 'Drug-Resistant', value: `${overview.drug_resistant_n} (${overview.drug_resistant_pct}%)`, color: '#dc2626' },
    { label: 'Seizure-Free', value: `${overview.seizure_free_n} (${overview.seizure_free_pct}%)`, color: '#16a34a' },
    { label: 'Surgical Candidates', value: overview.surgical_candidates_n, color: '#7c3aed' },
    { label: 'Somatosensory Aura', value: `${overview.somatosensory_aura_n} (${overview.somatosensory_aura_pct}%)`, color: '#d97706' },
    { label: "Todd's Paresis Post-FBTCS", value: `${overview.post_ictal_todd_n} (${overview.post_ictal_todd_pct}%)`, color: '#0284c7' },
  ];

  return (
    <div>
      <div className="alert alert-info py-2 small mb-3">
        <strong>Parietal Lobe Epilepsy (PLE):</strong> ~5% of focal epilepsies — the <strong>most under-recognised lobe epilepsy</strong>.
        Hallmark: <em>contralateral somatosensory aura (tingling/electric sensation following Penfield homunculus)</em>.
        Frequently <strong>misdiagnosed as FLE or TLE</strong> due to rapid ictal propagation — SEEG required for definitive localisation.
        Surgical outcomes: <strong>Engel I 55–70%</strong> (structural PLE).
        <span className="text-danger fw-bold"> SCN8A gain-of-function: sodium channel blockers may WORSEN seizures — gene panel before AED prescription.</span>
      </div>

      <div className="row mb-4">
        {kpis.map(k => <KPI key={k.label} {...k} />)}
      </div>

      {(overview.clinical_alerts || []).length > 0 && (
        <div className="alert alert-danger py-2 mb-3">
          <strong>⚠️ Clinical Alerts:</strong>
          <ul className="mb-0 mt-1 small">
            {overview.clinical_alerts.map((a, i) => <li key={i}>{a}</li>)}
          </ul>
        </div>
      )}

      <div className="row">
        <div className="col-md-6 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-body">
              <h6 className="card-title fw-semibold">Seizure Type Distribution</h6>
              {Object.entries(szDist).map(([type, count]) => (
                <div key={type} className="mb-2">
                  <div className="d-flex justify-content-between small mb-1">
                    <span>{type}</span><span className="text-muted">{count} pts</span>
                  </div>
                  <div className="progress" style={{ height: 10 }}>
                    <div className="progress-bar bg-warning"
                      style={{ width: `${Math.round(count / maxSz * 100)}%` }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-6 mb-3">
          <div className="card shadow-sm h-100">
            <div className="card-body">
              <h6 className="card-title fw-semibold">Key PLE Features</h6>
              <PctBar label="Somatosensory Aura (S1 onset marker)" pct={overview.somatosensory_aura_pct || 0} color="#d97706" />
              <PctBar label="Body Schema Disturbance (TPJ)" pct={overview.body_schema_pct || 0} color="#7c3aed" />
              <PctBar label="Post-ictal Todd's Paresis" pct={overview.post_ictal_todd_pct || 0} color="#0284c7" />
              <PctBar label="Drug-Resistant (≥2 AED failures)" pct={overview.drug_resistant_pct || 0} color="#dc2626" />
              <PctBar label="Seizure-Free" pct={overview.seizure_free_pct || 0} color="#16a34a" />
            </div>
          </div>
        </div>
      </div>

      {overview.surgical_targets && (
        <div className="card shadow-sm mb-3">
          <div className="card-body">
            <h6 className="card-title fw-semibold">🔪 Surgical Outcomes (Parietal Cortectomy / SEEG-Guided Resection)</h6>
            <div className="row small">
              <div className="col-md-3">
                <div className="alert alert-success py-2 text-center">
                  <div className="fw-bold">Engel I — FCD IIb</div>
                  <div className="fs-6 fw-bold">{overview.surgical_targets.engel_i_fcd_iib}</div>
                </div>
              </div>
              <div className="col-md-3">
                <div className="alert alert-primary py-2 text-center">
                  <div className="fw-bold">Engel I — Tumour (DNT/GG)</div>
                  <div className="fs-6 fw-bold">{overview.surgical_targets.engel_i_tumour}</div>
                </div>
              </div>
              <div className="col-md-3">
                <div className="alert alert-warning py-2 text-center">
                  <div className="fw-bold">Engel I — Non-Lesional SEEG</div>
                  <div className="fs-6 fw-bold">{overview.surgical_targets.engel_i_non_lesional}</div>
                </div>
              </div>
              <div className="col-md-3">
                <div className="alert alert-secondary py-2 text-center">
                  <div className="fw-bold">Post-op S1 Deficit</div>
                  <div style={{ fontSize: '0.75rem' }}>{overview.surgical_targets.postop_somatosensory_deficit}</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="card shadow-sm mb-3">
        <div className="card-body">
          <h6 className="card-title fw-semibold">📚 Key References</h6>
          <ol className="small mb-0">
            {(overview.references || []).map((r, i) => <li key={i}>{r}</li>)}
          </ol>
        </div>
      </div>
    </div>
  );
}

// ─── Patients & Etiology Tab ─────────────────────────────────────────────────
function PatientsTab({ breakdown }) {
  const [search, setSearch] = useState('');
  if (!breakdown) return <div className="text-center py-4 text-muted">Loading…</div>;

  const filtered = (breakdown.patients || []).filter(p =>
    !search || p.patient_id?.toLowerCase().includes(search.toLowerCase()) ||
    p.etiology?.toLowerCase().includes(search.toLowerCase()) ||
    p.seizure_control?.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div>
      <div className="row mb-3">
        <div className="col-md-6">
          <input className="form-control form-control-sm" placeholder="Search patient / etiology…"
            value={search} onChange={e => setSearch(e.target.value)} />
        </div>
      </div>

      {/* Etiology Catalog */}
      <h6 className="fw-semibold mb-2">🧬 Etiology Catalog ({(breakdown.etiology_catalog || []).length} classes)</h6>
      {(breakdown.etiology_catalog || []).map((e, i) => (
        <Expander key={i} title={`${e.etiology} — ${e.pct}% (${e.category})`} defaultOpen={i === 0}>
          <div className="row mt-2">
            <div className="col-md-6">
              <strong>Mechanism:</strong>
              <p className="mt-1">{e.mechanism}</p>
            </div>
            {e.eeg_correlate && (
              <div className="col-md-3">
                <strong>EEG Correlate:</strong>
                <p className="mt-1">{e.eeg_correlate}</p>
              </div>
            )}
            {e.mri_finding && (
              <div className="col-md-3">
                <strong>MRI Finding:</strong>
                <p className="mt-1">{e.mri_finding}</p>
                {e.clinical_note && (
                  <>
                    <strong>Clinical Note:</strong>
                    <p className="mt-1">{e.clinical_note}</p>
                  </>
                )}
              </div>
            )}
          </div>
        </Expander>
      ))}

      {/* Patient Table */}
      <h6 className="fw-semibold mt-4 mb-2">👥 Patient Cohort ({filtered.length} / {(breakdown.patients || []).length})</h6>
      <div className="table-responsive">
        <table className="table table-sm table-striped table-hover small">
          <thead className="table-dark">
            <tr>
              <th>ID</th><th>Age</th><th>Sex</th><th>Onset</th><th>Etiology</th>
              <th>Primary Seizure</th><th>AED</th><th>Control</th>
              <th>Som. Aura</th><th>Body Schema</th><th>Todd's</th><th>Surgical</th>
            </tr>
          </thead>
          <tbody>
            {filtered.slice(0, 50).map(p => (
              <tr key={p.patient_id}>
                <td className="fw-semibold">{p.patient_id}</td>
                <td>{p.age}</td>
                <td>{p.gender}</td>
                <td>{p.onset_age}y</td>
                <td style={{ maxWidth: 140 }}>{p.etiology}</td>
                <td>{p.primary_seizure_type}</td>
                <td><span className="badge bg-secondary">{p.current_aed}</span></td>
                <td>
                  <span className={`badge ${p.seizure_control === 'Seizure-free' ? 'bg-success' :
                    p.seizure_control === 'Drug-resistant' ? 'bg-danger' : 'bg-warning text-dark'}`}>
                    {p.seizure_control}
                  </span>
                </td>
                <td>{p.somatosensory_aura ? '✅' : '—'}</td>
                <td>{p.body_schema_disturbance ? '✅' : '—'}</td>
                <td>{p.post_ictal_todd ? '✅' : '—'}</td>
                <td>{p.surgical_candidate ? <span className="badge bg-primary">Yes</span> : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {filtered.length > 50 && <div className="text-muted small">Showing 50 of {filtered.length}</div>}
      </div>
    </div>
  );
}

// ─── Seizure Types & Surgery Tab ─────────────────────────────────────────────
function SeizureTab({ breakdown }) {
  if (!breakdown) return <div className="text-center py-4 text-muted">Loading…</div>;
  const trigColors = ['#dc2626', '#f59e0b', '#0284c7', '#16a34a', '#7c3aed', '#d97706', '#db2777', '#64748b'];

  return (
    <div>
      {/* Seizure Types */}
      <h6 className="fw-semibold mb-2">⚡ Seizure Types ({(breakdown.seizure_types || []).length})</h6>
      {(breakdown.seizure_types || []).map((s, i) => (
        <Expander key={i} title={`${s.type} — ${s.freq_pct}% of cohort`} defaultOpen={i === 0}>
          <div className="row mt-2">
            <div className="col-md-4">
              <strong>Duration:</strong> <span>{s.duration_sec}</span>
              <p className="mt-2">{s.description}</p>
            </div>
            <div className="col-md-4">
              <strong>EEG Correlate:</strong>
              <p className="mt-1">{s.eeg_correlate}</p>
            </div>
            <div className="col-md-4">
              <div className="alert alert-warning py-2 small">
                <strong>💡 Clinical Tip:</strong>
                <p className="mb-0 mt-1">{s.clinical_tip}</p>
              </div>
            </div>
          </div>
        </Expander>
      ))}

      {/* Triggers */}
      <h6 className="fw-semibold mt-4 mb-2">⚠️ Seizure Triggers ({(breakdown.triggers || []).length})</h6>
      <div className="row">
        {(breakdown.triggers || []).map((t, i) => (
          <div key={i} className="col-md-6 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-body small">
                <div className="d-flex align-items-center mb-2">
                  <div className="fw-bold me-2">{t.trigger}</div>
                  <span className="badge ms-auto" style={{ backgroundColor: trigColors[i % trigColors.length] }}>
                    {t.pct}%
                  </span>
                </div>
                <div className="mb-1">
                  <div className="progress" style={{ height: 8 }}>
                    <div className="progress-bar"
                      style={{ width: `${t.pct}%`, backgroundColor: trigColors[i % trigColors.length] }} />
                  </div>
                </div>
                <Expander title="Mechanism">
                  <p className="mb-0">{t.mechanism}</p>
                </Expander>
                <Expander title="Management">
                  <p className="mb-0">{t.management}</p>
                </Expander>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Clinical Lifecycle */}
      <h6 className="fw-semibold mt-4 mb-2">🔄 Clinical Lifecycle ({(breakdown.lifecycle || []).length} windows)</h6>
      <div className="table-responsive">
        <table className="table table-sm table-bordered small">
          <thead className="table-warning">
            <tr><th>Window</th><th>Age</th><th>Key Events</th><th>Focus</th></tr>
          </thead>
          <tbody>
            {(breakdown.lifecycle || []).map((l, i) => (
              <tr key={i}>
                <td className="fw-semibold" style={{ minWidth: 160 }}>{l.window}</td>
                <td><span className="badge bg-secondary">{l.age_range}</span></td>
                <td>{l.key_events}</td>
                <td className="text-muted">{l.focus}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ─── Treatments Tab ──────────────────────────────────────────────────────────
function TreatmentsTab({ breakdown }) {
  if (!breakdown) return <div className="text-center py-4 text-muted">Loading…</div>;

  const getEvidenceClass = (txt) => {
    if (!txt) return 'bg-secondary';
    if (txt.includes('Level A')) return 'bg-success';
    if (txt.includes('Level B')) return 'bg-primary';
    return 'bg-secondary';
  };

  return (
    <div>
      {/* Contraindication alerts */}
      <div className="alert alert-danger py-2 mb-3 small">
        <strong>🚨 CRITICAL CONTRAINDICATIONS — Parietal Lobe Epilepsy:</strong>
        <ul className="mb-0 mt-1">
          <li><strong>SCN8A gain-of-function PLE:</strong> Sodium channel blockers (CBZ/OXC/PHT/LTG) may paradoxically WORSEN seizures — obtain gene panel before AED prescription</li>
          <li><strong>Vigabatrin (VGB):</strong> Irreversible concentric visual field loss — CONTRAINDICATED in PLE with occipital spread / visual aura</li>
          <li><strong>Lacosamide (LCM):</strong> ECG mandatory before every dose increase — 2nd/3rd degree AV block is an absolute contraindication</li>
        </ul>
      </div>

      {/* Treatments */}
      {(breakdown.treatments || []).map((t, i) => (
        <Expander key={i} title={`${t.drug} — ${(t.efficacy || '').substring(0, 60)}…`} defaultOpen={i < 2}>
          <div className="row mt-2">
            <div className="col-md-3">
              <div><strong>Class:</strong> {t.class}</div>
              <div className="mt-1"><strong>Adult dose:</strong> {t.dose_adult}</div>
              <div className="mt-1"><strong>Paed dose:</strong> {t.dose_paed}</div>
              <div className="mt-2">
                <span className={`badge ${getEvidenceClass(t.efficacy)}`}>{(t.efficacy || '').substring(0, 40)}</span>
              </div>
              <div className="mt-1 text-muted">{t.evidence}</div>
            </div>
            <div className="col-md-3">
              <strong>Mechanism (MOA):</strong>
              <p className="mt-1">{t.moa}</p>
            </div>
            <div className="col-md-3">
              <strong>Safety:</strong>
              <p className="mt-1">{t.safety}</p>
            </div>
            <div className="col-md-3">
              <div className="alert alert-warning py-2 small">
                <strong>Monitoring:</strong>
                <p className="mb-1 mt-1">{t.monitoring}</p>
                {t.contraindications && (
                  <>
                    <strong>Contraindications:</strong>
                    <p className="mb-0 mt-1 text-danger">{t.contraindications}</p>
                  </>
                )}
              </div>
            </div>
          </div>
        </Expander>
      ))}

      {/* AED Monitoring Table */}
      <h6 className="fw-semibold mt-4 mb-2">🩺 AED Monitoring Protocol</h6>
      <div className="table-responsive">
        <table className="table table-sm table-bordered small">
          <thead className="table-warning">
            <tr><th>Monitoring Item</th><th>Frequency</th><th>Rationale</th></tr>
          </thead>
          <tbody>
            {(breakdown.aed_monitoring || []).map((m, i) => (
              <tr key={i}>
                <td className="fw-semibold">{m.item}</td>
                <td><span className="badge bg-info text-dark">{m.frequency}</span></td>
                <td>{m.rationale}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Standards & Thresholds */}
      <div className="row mt-3">
        <div className="col-md-6">
          <h6 className="fw-semibold">📋 Standards</h6>
          <ul className="small">
            {(breakdown.standards || []).map((s, i) => (
              <li key={i}><strong>{s.standard}:</strong> {s.relevance}</li>
            ))}
          </ul>
        </div>
        <div className="col-md-6">
          <h6 className="fw-semibold">🎯 Clinical Thresholds</h6>
          <table className="table table-sm small">
            <tbody>
              {(breakdown.thresholds || []).map((t, i) => (
                <tr key={i}>
                  <td className="fw-semibold">{t.threshold}</td>
                  <td><span className="badge bg-dark">{t.value}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

// ─── Definitions Tab ─────────────────────────────────────────────────────────
function DefinitionsTab({ definitions }) {
  if (!definitions) return <div className="text-center py-4 text-muted">Loading…</div>;
  return (
    <div>
      <h6 className="fw-semibold mb-3">📖 PLE Concept Dictionary ({(definitions.concepts || []).length} concepts)</h6>
      {(definitions.concepts || []).map((c, i) => (
        <Expander key={i} title={c.term} defaultOpen={i < 3}>
          <p className="mb-0 mt-1">{c.definition}</p>
        </Expander>
      ))}

      <h6 className="fw-semibold mt-4 mb-2">📐 Thresholds & Targets</h6>
      <table className="table table-sm table-bordered small">
        <thead className="table-secondary"><tr><th>Threshold</th><th>Value</th></tr></thead>
        <tbody>
          {(definitions.thresholds || []).map((t, i) => (
            <tr key={i}>
              <td className="fw-semibold">{t.threshold}</td>
              <td><span className="badge bg-dark">{t.value}</span></td>
            </tr>
          ))}
        </tbody>
      </table>

      <h6 className="fw-semibold mt-3 mb-2">📚 References</h6>
      <ol className="small">
        {(definitions.references || []).map((r, i) => <li key={i}>{r}</li>)}
      </ol>
    </div>
  );
}

// ─── Main Page ───────────────────────────────────────────────────────────────
export default function PLEPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/ple/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(e.message));
  }, []);

  useEffect(() => {
    if (tab === 1 || tab === 2 || tab === 3) {
      if (!breakdown) {
        fetch(`${API}/api/ple/breakdown`)
          .then(r => r.json()).then(setBreakdown).catch(e => setError(e.message));
      }
    }
    if (tab === 4 && !definitions) {
      fetch(`${API}/api/ple/definitions`)
        .then(r => r.json()).then(setDefinitions).catch(e => setError(e.message));
    }
  }, [tab]);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3 gap-3">
        <div>
          <h4 className="mb-0 fw-bold">🧠 Parietal Lobe Epilepsy (PLE)</h4>
          <div className="text-muted small">Somatosensory aura · Body schema · SEEG localisation · Awake craniotomy · Jacksonian march</div>
        </div>
        <span className="badge bg-warning text-dark ms-auto">41 Patients</span>
      </div>

      {error && (
        <div className="alert alert-danger py-2 small">
          Backend error: {error} — check <code>http://localhost:8010/api/ple/overview</code>
        </div>
      )}

      {/* Tab Nav */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button className={`nav-link${tab === i ? ' active fw-semibold' : ''}`}
              onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab overview={overview} />}
      {tab === 1 && <PatientsTab breakdown={breakdown} />}
      {tab === 2 && <SeizureTab breakdown={breakdown} />}
      {tab === 3 && <TreatmentsTab breakdown={breakdown} />}
      {tab === 4 && <DefinitionsTab definitions={definitions} />}
    </div>
  );
}
