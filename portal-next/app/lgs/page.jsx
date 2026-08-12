'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiologies', 'Seizure Types & Monitoring', 'Treatments', 'Definitions'];

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

function Bar({ label, value, max, color = '#3b82f6' }) {
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
  const maxSz = Math.max(...(data.seizure_type_distribution || []).map(s => s.count), 1);
  const maxTx = Math.max(...(data.treatment_use || []).map(t => t.n_patients), 1);
  const maxCog = Math.max(...(data.cognitive_distribution || []).map(c => c.count), 1);

  return (
    <div>
      <div className="alert alert-danger py-2 small mb-3">
        <strong>Catastrophic Epileptic Encephalopathy:</strong> Lennox-Gastaut Syndrome — 1–4% of all epilepsies,
        onset 1–8 years. Classic triad: multi-seizure types (tonic / atonic / atypical absence) +
        slow spike-wave EEG (1.5–2.5 Hz) + intellectual disability.
        <strong> 80–90% drug-resistant</strong>; drop attacks cause major injury risk — <em>helmet mandatory</em>.
      </div>

      <div className="row mb-4">
        {(data.kpis || []).map(k => <KPI key={k.label} {...k} />)}
      </div>

      <div className="row">
        <div className="col-md-4 mb-3">
          <div className="card h-100 shadow-sm">
            <div className="card-header fw-bold">Etiology Distribution</div>
            <div className="card-body">
              {(data.etiology_distribution || []).map(e => (
                <Bar key={e.etiology} label={`${e.etiology.split(' (')[0]} (${e.pct}%)`} value={e.count} max={maxEtio} />
              ))}
            </div>
          </div>
        </div>

        <div className="col-md-4 mb-3">
          <div className="card h-100 shadow-sm">
            <div className="card-header fw-bold">Seizure Type Prevalence</div>
            <div className="card-body">
              {(data.seizure_type_distribution || []).map(s => (
                <Bar key={s.type} label={`${s.type} (${s.prevalence_pct}%)`} value={s.count} max={maxSz} color="#ef4444" />
              ))}
              <div className="text-muted small mt-2">Multiple seizure types per patient expected</div>
            </div>
          </div>
        </div>

        <div className="col-md-4 mb-3">
          <div className="card h-100 shadow-sm">
            <div className="card-header fw-bold">Cognitive Level</div>
            <div className="card-body">
              {(data.cognitive_distribution || []).map(c => (
                <Bar key={c.level} label={c.level} value={c.count} max={maxCog} color="#8b5cf6" />
              ))}
              <div className="text-muted small mt-2">70–80% Severe/Moderate ID</div>
            </div>
          </div>
        </div>

        <div className="col-md-12 mb-3">
          <div className="card shadow-sm">
            <div className="card-header fw-bold">Current Treatment Use</div>
            <div className="card-body">
              <div className="row">
                {(data.treatment_use || []).map(t => (
                  <div key={t.drug} className="col-6 col-md-3 mb-2">
                    <Bar label={t.drug} value={t.n_patients} max={maxTx} color="#6366f1" />
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="alert alert-warning py-2 small">
        <strong>4 FDA-Approved Therapies for LGS:</strong>{' '}
        Rufinamide (2008) · Clobazam (2011) · Cannabidiol/Epidiolex (2018) · Fenfluramine/Fintepla (2020).
        Corpus callosotomy: Level A surgical option for drop attacks.
      </div>
      <div className="text-muted small mt-2"><strong>Source:</strong> {data.reference}</div>
    </div>
  );
}

function PatientsTab({ data }) {
  const [filter, setFilter] = useState('all');
  const [search, setSearch] = useState('');
  if (!data) return <div className="text-center py-4 text-muted">Loading breakdown…</div>;

  let patients = data.patient_table || [];
  if (filter === 'drop') patients = patients.filter(p => p.drop_attacks_per_month > 0);
  if (filter === 'resistant') patients = patients.filter(p => p.drug_resistant === 'Yes');
  if (filter === 'callosotomy') patients = patients.filter(p => p.corpus_callosotomy === 'Yes');
  if (filter === 'responder') patients = patients.filter(p => p.responder === 'Yes');
  if (search) patients = patients.filter(p =>
    p.patient_id.toLowerCase().includes(search.toLowerCase()) ||
    p.etiology.toLowerCase().includes(search.toLowerCase()) ||
    p.seizure_types.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div>
      <div className="row mb-3">
        <div className="col-md-8">
          <div className="btn-group flex-wrap" role="group">
            {['all', 'drop', 'resistant', 'callosotomy', 'responder'].map(f => (
              <button key={f} className={`btn btn-sm ${filter === f ? 'btn-primary' : 'btn-outline-secondary'}`}
                onClick={() => setFilter(f)}>
                {f === 'all' ? 'All Patients' : f === 'drop' ? 'Drop Attacks' :
                  f === 'resistant' ? 'Drug-Resistant' : f === 'callosotomy' ? 'Callosotomy' : 'Responders'}
              </button>
            ))}
          </div>
        </div>
        <div className="col-md-4">
          <input className="form-control form-control-sm" placeholder="Search patient / etiology…"
            value={search} onChange={e => setSearch(e.target.value)} />
        </div>
      </div>

      <div className="table-responsive mb-4" style={{ maxHeight: 420, overflowY: 'auto' }}>
        <table className="table table-sm table-striped table-hover small">
          <thead className="table-dark sticky-top">
            <tr>
              <th>Patient</th><th>Age</th><th>Etiology</th><th>Seizure Types</th>
              <th>Drop/Mo</th><th>Total Sz/Mo</th><th>Drug-R</th>
              <th>Callosotomy</th><th>Regimen</th><th>Cognitive</th><th>Responder</th>
            </tr>
          </thead>
          <tbody>
            {patients.map(p => (
              <tr key={p.patient_id}>
                <td><code>{p.patient_id}</code></td>
                <td>{p.age}</td>
                <td><span className="badge bg-secondary small" title={p.etiology}>{p.etiology.split(' (')[0]}</span></td>
                <td><small>{p.seizure_types}</small></td>
                <td>
                  {p.drop_attacks_per_month > 0
                    ? <span className={`badge ${p.drop_attacks_per_month > 15 ? 'bg-danger' : 'bg-warning text-dark'}`}>{p.drop_attacks_per_month}</span>
                    : '—'}
                </td>
                <td>{p.total_seizures_per_month}</td>
                <td>{p.drug_resistant === 'Yes' ? <span className="badge bg-danger">Yes</span> : <span className="badge bg-success">No</span>}</td>
                <td>{p.corpus_callosotomy === 'Yes' ? <span className="badge bg-primary">Yes</span> : '—'}</td>
                <td><small>{p.current_regimen}</small></td>
                <td><small className="text-muted">{p.cognitive_level}</small></td>
                <td>{p.responder === 'Yes' ? <span className="badge bg-success">Yes</span> : <span className="badge bg-secondary">No</span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="card shadow-sm">
        <div className="card-header fw-bold">Etiology Catalog</div>
        <div className="card-body p-0">
          <table className="table table-sm small mb-0">
            <thead className="table-light">
              <tr><th>Etiology Class</th><th>Example</th><th>%</th><th>Mechanism</th><th>Surgical Relevance</th></tr>
            </thead>
            <tbody>
              {(data.etiology_catalog || []).map(e => (
                <tr key={e.class}>
                  <td><strong>{e.class}</strong></td>
                  <td className="text-muted"><small>{e.example}</small></td>
                  <td><span className="badge bg-info text-dark">{e.pct}%</span></td>
                  <td><small>{e.mechanism}</small></td>
                  <td><small className="text-muted">{e.surgical_relevance}</small></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function SeizureTab({ data }) {
  const [open, setOpen] = useState(null);
  if (!data) return <div className="text-center py-4 text-muted">Loading breakdown…</div>;

  return (
    <div>
      <h6 className="fw-bold mb-3">Seizure Types in LGS</h6>
      {(data.seizure_type_catalog || []).map((s, i) => (
        <div key={s.type} className="card mb-2 shadow-sm">
          <div className="card-header d-flex justify-content-between align-items-center"
            style={{ cursor: 'pointer' }} onClick={() => setOpen(open === i ? null : i)}>
            <span>
              <strong>{s.type}</strong>
              <span className="badge bg-primary ms-2">{s.prevalence_pct}% of LGS</span>
            </span>
            <span>{open === i ? '▲' : '▼'}</span>
          </div>
          {open === i && (
            <div className="card-body small">
              <div className="row">
                <div className="col-md-3"><strong>Onset:</strong> {s.onset}</div>
                <div className="col-md-5"><strong>EEG:</strong> {s.eeg}</div>
                <div className="col-md-4"><strong>Treatment Priority:</strong> {s.treatment_priority}</div>
              </div>
              <div className="alert alert-info py-1 mt-2 small">{s.outcome_note}</div>
            </div>
          )}
        </div>
      ))}

      <h6 className="fw-bold mb-3 mt-4">AED Monitoring Requirements &amp; Contraindications</h6>
      {(data.aed_monitoring || []).map(a => (
        <div key={a.aed} className="card mb-2 shadow-sm">
          <div className="card-body py-2">
            <div className="d-flex justify-content-between mb-1">
              <strong>{a.aed}</strong>
              <span className={`badge ${a.category === 'ABSOLUTE CONTRAINDICATION' ? 'bg-danger' :
                a.category === 'CARDIAC REMS MONITORING' ? 'bg-danger' :
                a.category === 'LFT MONITORING' ? 'bg-warning text-dark' :
                a.category === 'ECG / QT MONITORING' ? 'bg-warning text-dark' :
                'bg-secondary'}`}>
                {a.category}
              </span>
            </div>
            <div className="small text-muted mb-1"><strong>Risk:</strong> {a.risk}</div>
            <div className="small mb-1"><strong>Monitoring:</strong> {a.monitoring}</div>
            <div className="small"><strong>Evidence:</strong> {a.evidence}</div>
          </div>
        </div>
      ))}
    </div>
  );
}

function TreatmentsTab({ data }) {
  const [open, setOpen] = useState(0);
  if (!data) return <div className="text-center py-4 text-muted">Loading breakdown…</div>;

  return (
    <div>
      <div className="alert alert-success py-2 small mb-3">
        <strong>4 FDA-Approved LGS Therapies (adjunct):</strong>{' '}
        Rufinamide/Banzel (EIAED 2008) ·
        Clobazam/Onfi (COALITION-I 2011) ·
        Cannabidiol/Epidiolex (GWPCARE3/4 2018) ·
        Fenfluramine/Fintepla (2020 — REMS required).
        Corpus callosotomy: Level A for drop attacks.
      </div>
      {(data.treatment_catalog || []).map((t, i) => (
        <div key={t.drug} className="card mb-2 shadow-sm">
          <div className="card-header d-flex justify-content-between align-items-center"
            style={{ cursor: 'pointer' }} onClick={() => setOpen(open === i ? null : i)}>
            <span>
              <strong>{t.drug}</strong>
              <span className={`badge ms-2 ${t.fda_status.includes('FDA-approved') ? 'bg-success' :
                t.fda_status.includes('Level A') ? 'bg-primary' : 'bg-secondary'}`}>
                {t.fda_status.split('(')[0].trim()}
              </span>
              {t.year && <span className="badge bg-light text-dark border ms-1">{t.year}</span>}
            </span>
            <span>{open === i ? '▲' : '▼'}</span>
          </div>
          {open === i && (
            <div className="card-body small">
              <div className="mb-2"><strong>Dose:</strong> {t.dose}</div>
              <div className="mb-2"><strong>Mechanism:</strong> {t.moa}</div>
              <div className="mb-2 text-success"><strong>Efficacy:</strong> {t.efficacy}</div>
              <div className="text-danger"><strong>Safety:</strong> {t.safety}</div>
            </div>
          )}
        </div>
      ))}

      <h6 className="fw-bold mt-4 mb-3">Developmental Trajectory</h6>
      <div className="table-responsive">
        <table className="table table-sm table-bordered small">
          <thead className="table-dark">
            <tr><th>Age Window</th><th>Expected Milestone</th><th>LGS Pattern</th></tr>
          </thead>
          <tbody>
            {(data.developmental_trajectory || []).map(m => (
              <tr key={m.age_window}>
                <td><strong>{m.age_window}</strong></td>
                <td className="text-muted">{m.expected}</td>
                <td>{m.lgs_pattern}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading definitions…</div>;
  return (
    <div>
      <h6 className="fw-bold mb-3">Core Concepts</h6>
      <div className="table-responsive mb-4">
        <table className="table table-sm table-striped small">
          <thead className="table-dark"><tr><th style={{ width: '220px' }}>Term</th><th>Definition</th></tr></thead>
          <tbody>
            {(data.concepts || []).map(c => (
              <tr key={c.term}>
                <td><strong>{c.term}</strong></td>
                <td>{c.definition}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h6 className="fw-bold mb-3">Standards &amp; Guidelines</h6>
      <div className="table-responsive mb-4">
        <table className="table table-sm table-striped small">
          <thead className="table-dark"><tr><th>Standard</th><th>Scope</th></tr></thead>
          <tbody>
            {(data.standards || []).map(s => (
              <tr key={s.name}><td><strong>{s.name}</strong></td><td>{s.scope}</td></tr>
            ))}
          </tbody>
        </table>
      </div>

      <h6 className="fw-bold mb-3">Key Thresholds</h6>
      <div className="table-responsive mb-4">
        <table className="table table-sm table-striped small">
          <thead className="table-dark">
            <tr><th>Metric</th><th>Target</th><th>Action Below / Issue</th><th>Action Above</th></tr>
          </thead>
          <tbody>
            {(data.thresholds || []).map(t => (
              <tr key={t.metric}>
                <td><strong>{t.metric}</strong></td>
                <td><span className="badge bg-success">{t.target}</span></td>
                <td className="text-danger small">{t.action_below}</td>
                <td className="text-muted small">{t.action_above}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h6 className="fw-bold mb-2">References</h6>
      <ul className="small text-muted">
        {(data.references || []).map((r, i) => <li key={i}>{r}</li>)}
      </ul>
    </div>
  );
}

export default function LGSDashboard() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/lgs/overview`).then(r => r.json()),
      fetch(`${API}/api/lgs/breakdown`).then(r => r.json()),
      fetch(`${API}/api/lgs/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOverview(o); setBreakdown(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger">Error: {err}</div>;

  return (
    <div>
      <div className="d-flex justify-content-between align-items-center mb-3">
        <div>
          <h3>&#x1f9e0; Lennox-Gastaut Syndrome (LGS)</h3>
          <div className="text-muted small">
            Catastrophic epileptic encephalopathy · Multi-seizure-type · Slow spike-wave EEG (1.5–2.5 Hz) ·
            Drop attacks · 4 FDA-approved therapies · Corpus callosotomy
            {overview && ` · ${overview.total_patients} patients`}
          </div>
        </div>
        <span className="badge bg-danger fs-6">Drug-Resistant</span>
      </div>

      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button className={`nav-link ${tab === t ? 'active' : ''}`} onClick={() => setTab(t)}>{t}</button>
          </li>
        ))}
      </ul>

      {tab === 'Overview' && <OverviewTab data={overview} />}
      {tab === 'Patients & Etiologies' && <PatientsTab data={breakdown} />}
      {tab === 'Seizure Types & Monitoring' && <SeizureTab data={breakdown} />}
      {tab === 'Treatments' && <TreatmentsTab data={breakdown} />}
      {tab === 'Definitions' && <DefinitionsTab data={defs} />}
    </div>
  );
}
