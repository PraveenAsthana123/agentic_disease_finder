'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiologies', 'EEG & Monitoring', 'Treatments', 'Definitions'];

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

function Bar({ label, value, max, color = '#7c3aed' }) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="text-muted">{value}</span>
      </div>
      <div className="progress" style={{ height: 12 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading overview…</div>;
  const maxEtio = Math.max(...(data.etiology_distribution || []).map(e => e.count), 1);
  const maxTx = Math.max(...(data.treatment_use || []).map(t => t.n_patients), 1);
  const maxDev = Math.max(...(data.developmental_distribution || []).map(d => d.count), 1);

  return (
    <div>
      <div className="alert alert-danger py-2 small mb-3">
        <strong>Developmental Epileptic Encephalopathy:</strong> West Syndrome — 2–4 per 10,000 live births;
        peak onset <strong>5–7 months</strong>. Classic triad: infantile spasm clusters +
        hypsarrhythmia EEG + developmental arrest/regression.
        <strong> 70–80% drug-resistant</strong> (symptomatic); 30–40% evolve to Lennox-Gastaut Syndrome.
        <em> Treat within 2 weeks of onset — delay worsens outcome.</em>
      </div>

      <div className="row mb-4">
        {(data.kpis || []).map(k => <KPI key={k.label} {...k} />)}
      </div>

      {(data.clinical_alerts || []).length > 0 && (
        <div className="alert alert-warning py-2 mb-3">
          <strong>Clinical Alerts:</strong>
          <ul className="mb-0 mt-1 small">
            {data.clinical_alerts.map((a, i) => <li key={i}>{a}</li>)}
          </ul>
        </div>
      )}

      <div className="row">
        <div className="col-md-4 mb-3">
          <div className="card h-100 shadow-sm">
            <div className="card-header fw-bold">Etiology Distribution</div>
            <div className="card-body">
              {(data.etiology_distribution || []).map(e => (
                <Bar key={e.etiology} label={`${e.etiology} (${e.pct}%)`} value={e.count} max={maxEtio} />
              ))}
            </div>
          </div>
        </div>

        <div className="col-md-4 mb-3">
          <div className="card h-100 shadow-sm">
            <div className="card-header fw-bold">First-Line Treatment Use</div>
            <div className="card-body">
              {(data.treatment_use || []).map(t => (
                <Bar key={t.drug} label={t.drug} value={t.n_patients} max={maxTx} color="#0284c7" />
              ))}
            </div>
          </div>
        </div>

        <div className="col-md-4 mb-3">
          <div className="card h-100 shadow-sm">
            <div className="card-header fw-bold">Developmental Outcome</div>
            <div className="card-body">
              {(data.developmental_distribution || []).map(d => (
                <Bar key={d.level} label={d.level} value={d.count} max={maxDev} color="#f59e0b" />
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="row">
        <div className="col-md-6 mb-3">
          <div className="card shadow-sm">
            <div className="card-header fw-bold">EEG Features (West Syndrome)</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light"><tr><th>Feature</th><th>Prevalence</th><th>Significance</th></tr></thead>
                <tbody>
                  {(data.eeg_features || []).map(e => (
                    <tr key={e.feature}>
                      <td className="fw-semibold small">{e.feature}</td>
                      <td className="small">{e.prevalence}</td>
                      <td className="small text-muted">{e.clinical_significance}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
        <div className="col-md-6 mb-3">
          <div className="card shadow-sm">
            <div className="card-header fw-bold">Key Statistics</div>
            <div className="card-body">
              <table className="table table-sm mb-0">
                <tbody>
                  <tr><td className="text-muted small">Prevalence</td><td className="small">{data.prevalence}</td></tr>
                  <tr><td className="text-muted small">ICD-10</td><td className="small fw-semibold">{data.icd10}</td></tr>
                  <tr><td className="text-muted small">Drug Resistance (symptomatic)</td><td className="small text-danger fw-bold">{data.drug_resistance_rate}</td></tr>
                  <tr><td className="text-muted small">LGS Evolution Risk</td><td className="small text-danger">{data.lgs_evolution_pct}% of cohort</td></tr>
                  <tr><td className="text-muted small">ACTH Efficacy (non-TSC)</td><td className="small">76% spasm cessation at 2W (UKISS 2004)</td></tr>
                  <tr><td className="text-muted small">Vigabatrin Efficacy (TSC-IS)</td><td className="small">70–95% spasm cessation (Capal 2021)</td></tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function PatientsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  return (
    <div>
      <div className="row mb-3">
        {(data.etiologies || []).map(e => (
          <div key={e.class} className="col-md-6 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold small">{e.class} ({e.pct}%)</div>
              <div className="card-body small">
                <p className="mb-1"><strong>Mechanism:</strong> {e.mechanism}</p>
                <p className="mb-1"><strong>First-line:</strong> {e.first_line}</p>
                <p className="mb-0 text-muted"><strong>Outcome:</strong> {e.outcome}</p>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="card shadow-sm">
        <div className="card-header fw-bold">Patient Cohort — IS Overlay ({(data.patients || []).length} patients)</div>
        <div className="card-body p-0" style={{ overflowX: 'auto' }}>
          <table className="table table-sm mb-0">
            <thead className="table-light">
              <tr>
                <th>Patient</th><th>Sex</th><th>Onset (mo)</th><th>Etiology</th>
                <th>First-line</th><th>Spasm-Free</th><th>Relapse</th>
                <th>Dev. Level</th><th>→LGS</th><th>ASD</th>
              </tr>
            </thead>
            <tbody>
              {(data.patients || []).map(p => (
                <tr key={p.patient_id}>
                  <td className="small fw-semibold">{p.patient_id}</td>
                  <td className="small">{p.sex}</td>
                  <td className="small">{p.spasm_onset_months}M</td>
                  <td className="small">{p.etiology}</td>
                  <td className="small">{p.first_line_tx}</td>
                  <td className="small">{p.spasm_free ? <span className="text-success fw-bold">Yes</span> : <span className="text-danger">No</span>}</td>
                  <td className="small">{p.relapse ? <span className="text-warning">Yes</span> : '—'}</td>
                  <td className="small">{p.developmental_level}</td>
                  <td className="small">{p.lgs_evolved ? <span className="text-danger fw-bold">Yes</span> : '—'}</td>
                  <td className="small">{p.asd_diagnosis ? <span className="text-info">Yes</span> : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function EEGTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  return (
    <div>
      <div className="row mb-3">
        {(data.eeg_features || []).map(e => (
          <div key={e.feature} className="col-md-6 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold small">{e.feature}</div>
              <div className="card-body small">
                <p className="mb-1"><strong>Description:</strong> {e.description}</p>
                <p className="mb-1"><strong>Prevalence:</strong> {e.prevalence}</p>
                <p className="mb-1"><strong>EEG band:</strong> <code>{e.eeg_band}</code></p>
                <p className="mb-0 text-muted"><strong>Clinical significance:</strong> {e.clinical_significance}</p>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="card shadow-sm mb-3">
        <div className="card-header fw-bold">AED Safety Monitoring Requirements</div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0">
            <thead className="table-light">
              <tr><th>Drug</th><th>Category</th><th>Risk</th><th>Monitoring</th><th>Mitigation</th></tr>
            </thead>
            <tbody>
              {(data.aed_monitoring || []).map(a => (
                <tr key={a.drug}>
                  <td className="small fw-semibold">{a.drug}</td>
                  <td className="small">
                    <span className={`badge ${a.category.includes('REMS') ? 'bg-danger' : a.category.includes('STEROID') ? 'bg-warning text-dark' : 'bg-info text-dark'}`}>
                      {a.category}
                    </span>
                  </td>
                  <td className="small text-danger">{a.risk}</td>
                  <td className="small">{a.monitoring}</td>
                  <td className="small text-muted">{a.mitigation}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card shadow-sm">
        <div className="card-header fw-bold">Developmental Trajectory — Age Windows</div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0">
            <thead className="table-light">
              <tr><th>Age Window</th><th>IS Manifestation</th><th>Intervention</th><th>Red Flags</th></tr>
            </thead>
            <tbody>
              {(data.developmental_trajectory || []).map(d => (
                <tr key={d.age_window}>
                  <td className="small fw-semibold">{d.age_window}</td>
                  <td className="small">{d.is_manifestation}</td>
                  <td className="small">{d.intervention}</td>
                  <td className="small text-danger">{d.flags}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function TreatmentsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  return (
    <div>
      {(data.treatments || []).map(t => (
        <div key={t.drug} className="card shadow-sm mb-3">
          <div className="card-header fw-bold d-flex justify-content-between align-items-center">
            <span>{t.drug}</span>
            <span className="badge bg-primary small">{t.fda_status}</span>
          </div>
          <div className="card-body small">
            <div className="row">
              <div className="col-md-6">
                <p className="mb-1"><strong>Dose:</strong> {t.dose}</p>
                <p className="mb-1"><strong>Mechanism:</strong> {t.moa}</p>
              </div>
              <div className="col-md-6">
                <p className="mb-1"><strong>Efficacy:</strong> {t.efficacy}</p>
                <p className="mb-1"><strong>Safety:</strong> <span className="text-danger">{t.safety}</span></p>
                <p className="mb-0 text-muted"><strong>Evidence:</strong> {t.evidence_level}</p>
              </div>
            </div>
          </div>
        </div>
      ))}

      <div className="card shadow-sm">
        <div className="card-header fw-bold">Clinical Standards & Guidelines</div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0">
            <thead className="table-light"><tr><th>Standard</th><th>Body</th><th>Scope</th></tr></thead>
            <tbody>
              {(data.standards || []).map(s => (
                <tr key={s.name}>
                  <td className="small fw-semibold">{s.name}</td>
                  <td className="small">{s.body}</td>
                  <td className="small text-muted">{s.scope}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  return (
    <div>
      <div className="card shadow-sm mb-3">
        <div className="card-header fw-bold">Key Concepts ({(data.concepts || []).length})</div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0">
            <thead className="table-light"><tr><th style={{ width: '22%' }}>Term</th><th>Definition</th></tr></thead>
            <tbody>
              {(data.concepts || []).map(c => (
                <tr key={c.term}>
                  <td className="small fw-semibold">{c.term}</td>
                  <td className="small">{c.definition}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card shadow-sm mb-3">
        <div className="card-header fw-bold">Key Thresholds</div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0">
            <thead className="table-light"><tr><th>Parameter</th><th>Threshold</th><th>Unit</th><th>Significance</th></tr></thead>
            <tbody>
              {(data.thresholds || []).map(t => (
                <tr key={t.parameter}>
                  <td className="small fw-semibold">{t.parameter}</td>
                  <td className="small text-primary fw-bold">{t.threshold}</td>
                  <td className="small">{t.unit}</td>
                  <td className="small text-muted">{t.significance}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card shadow-sm">
        <div className="card-header fw-bold">References</div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0">
            <thead className="table-light"><tr><th style={{ width: '40%' }}>Citation</th><th>Key Finding</th></tr></thead>
            <tbody>
              {(data.references || []).map(r => (
                <tr key={r.citation}>
                  <td className="small fw-semibold">{r.citation}</td>
                  <td className="small text-muted">{r.key_finding}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default function WestSyndromePage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/west-syndrome/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setErr(e.message));
    fetch(`${API}/api/west-syndrome/breakdown`)
      .then(r => r.json()).then(setBreakdown).catch(() => {});
    fetch(`${API}/api/west-syndrome/definitions`)
      .then(r => r.json()).then(setDefinitions).catch(() => {});
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 className="fw-bold mb-0">&#x1F476; West Syndrome (Infantile Spasms)</h4>
        <div className="text-muted small">
          Developmental Epileptic Encephalopathy · ICD-10 G40.40 · Onset 3–12 months · ILAE 2022
        </div>
      </div>

      {err && <div className="alert alert-danger small">Error: {err}</div>}

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={t} className="nav-item">
            <button className={`nav-link${tab === i ? ' active' : ''}`} onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <PatientsTab data={breakdown} />}
      {tab === 2 && <EEGTab data={breakdown} />}
      {tab === 3 && <TreatmentsTab data={breakdown} />}
      {tab === 4 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
